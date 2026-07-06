"""Stage E1 retry: speculative-step cost attribution with the GPU ngram drafter.

Stage A's E1 arms failed on an engine validator: CPU "ngram" is rejected with
--async-scheduling on vLLM v0.24.0. The working method is "ngram_gpu" (verified
against the v0.24.0 tag source, vllm/config/vllm.py:940-959 +
speculative.py:34-69,633: same lookup params, composes with async scheduling, TP,
fp8 KV, MLA, FULL_AND_PIECEWISE; compile cache force-disabled upstream -> slower
boot, no correctness issue).

The question: the EAGLE k=5 step costs 12.36ms (stage-A anchor, T1 6.63ms same
node). Is the ~5.7ms margin the drafter loop or the MoE expert-union verify width?
ngram_gpu drafting costs ~0 GPU time, so the per-drafted-step time is the pure
verify cost V(width).

Arms (one 4xB200 session):
  E1A ngram_gpu k=5  tool 0-8  -> V6
  E1B no-spec        tool 0-4  -> T1 same node (E1A boots first: cold boot is the
                                  expensive slot, spend it on the arm that matters)
  E1C ngram_gpu k=8  tool 0-8  -> V9 -> verify_slope = (V9-V6)/3
  E1D eagle3 k=5     tool 0-4  -> same-node step_k5 (draft_pass = (step-V6)/5),
                                  optional, spend-guarded
Verdict (pre-registered, same thresholds as stage A): V6 <= 8.5ms -> draft-loop
(margin is the drafter's sequential passes); V6 >= 11.0ms -> verify-bytes
(margin is verify width).

Artifact: /out/brl11_stage600e1.json on volume k26-draft-out.

Usage:
  python stage600_e1.py --selftest
  modal run --detach stage600_e1.py
  modal app stop brl11-stage600e1 --yes
"""
import json
import os
import sys

try:
    import modal
except ImportError:
    modal = None

import record_run as rr
import stage600_a as sa

HARD_CAP = 32.0           # this session's spend ceiling at $24/hr
STAGE_A_STEP_K5 = 12.36   # cross-node fallback for draft_pass if E1D is skipped
STAGE_A_T1 = 6.63         # stage-A no-spec step at 10k (cross-node T1 fallback)


def run_stage600e1(ctx):
    res = {"campaign": "stage600_e1", "arms": {}, "decisions": [],
           "hourly_usd": 24.0, "hard_cap_usd": HARD_CAP,
           "stage_a_refs": {"step_k5_ms": STAGE_A_STEP_K5, "t1_ms": STAGE_A_T1,
                            "anchor_f8": 403.2},
           "protocol": "identical to brl11_record; probe slots as stage-A",
           "prompt_sha256_10k": ctx.prompt_sha,
           "zero_init": ctx.zero_init_state,
           "p0_manifest": ctx.p0_manifest_summary}

    def note(msg):
        res["decisions"].append(msg)
        ctx.log("DECISION: " + msg)

    run_cell = sa.run_cell_a

    # ---- E1A: ngram_gpu k=5 (cold boot) ----
    ng5_cell = None
    if sa.run_arm_boot_a(ctx, res, "E1A_ngramgpu_k5", sa.cfg_a("ngram_gpu", 5), 45):
        ng5_cell = run_cell(ctx, res, "E1A_ngramgpu_k5", "tool_nothink_10k", "tool",
                            2048, False, 0, 8)
        ctx.stop()
    else:
        note("E1A ngram_gpu failed to boot; aborting session (nothing to attribute)")
        ctx.save(res)
        return res

    # ---- E1B: no-spec T1, same node ----
    t1_ms, t1_source = None, "measured same-node"
    if ctx.spent() < HARD_CAP - 8.0 and sa.run_arm_boot_a(ctx, res, "E1B_nospec", sa.cfg_a("none"), 22):
        run_cell(ctx, res, "E1B_nospec", "tool_nothink_10k", "tool", 2048, False, 0, 4)
        c = res["arms"]["E1B_nospec"]["cells"].get("tool_nothink_10k")
        if c and c.get("tok_s_agg"):
            t1_ms = round(1000.0 / c["tok_s_agg"], 2)
        ctx.stop()
    if t1_ms is None:
        t1_ms, t1_source = STAGE_A_T1, "stage-A cross-node fallback"
    note(f"T1 = {t1_ms}ms ({t1_source})")

    # ---- E1C: ngram_gpu k=8 ----
    ng8_cell = None
    if ctx.spent() < HARD_CAP - 8.0 and sa.run_arm_boot_a(ctx, res, "E1C_ngramgpu_k8", sa.cfg_a("ngram_gpu", 8), 25):
        ng8_cell = run_cell(ctx, res, "E1C_ngramgpu_k8", "tool_nothink_10k", "tool",
                            2048, False, 0, 8)
        ctx.stop()

    # ---- E1D: same-node eagle k=5 anchor (optional) ----
    step_k5, step_src = None, "measured same-node"
    if ctx.spent() < HARD_CAP - 7.0 and sa.run_arm_boot_a(
            ctx, res, "E1D_eagle_k5", sa.cfg_a("eagle3", 5, rr.EAGLE, argmax=True), 25):
        cell = run_cell(ctx, res, "E1D_eagle_k5", "tool_nothink_10k", "tool", 2048, False, 0, 4)
        step_k5 = sa.step_ms_cell(cell)
        ctx.stop()
    if step_k5 is None:
        step_k5, step_src = STAGE_A_STEP_K5, "stage-A cross-node fallback (+-10% node noise)"
    note(f"eagle step_k5 = {step_k5}ms ({step_src})")

    res["e1"] = sa.e1_block(t1_ms, t1_source, ng5_cell, ng8_cell, step_k5)
    res["e1"]["step_k5_source"] = step_src
    note(f"E1 RETRY: verdict={res['e1'].get('verdict')} v6={res['e1'].get('v6_ms')} "
         f"v9={res['e1'].get('v9_ms')} hit={res['e1'].get('v6_hit_rate')} "
         f"draft_pass={res['e1'].get('draft_pass_ms')} "
         f"verify_slope={res['e1'].get('verify_slope_ms_per_tok')}")
    res["total_spent_usd_est"] = round(ctx.spent(), 2)
    ctx.save(res)
    return res


# ---------------- Modal wiring ----------------
if modal is not None:
    app = modal.App("brl11-stage600e1")
    kimi = modal.Volume.from_name("kimi-k26", create_if_missing=False)
    hfvol = modal.Volume.from_name("hf-cache", create_if_missing=True)
    outvol = modal.Volume.from_name("k26-draft-out", create_if_missing=True)

    image = rr.vllm_image.add_local_python_source("record_run", "stage600_a")

    @app.function(image=image, gpu="B200:4",
                  volumes={"/models": kimi, "/cache": hfvol, "/out": outvol},
                  timeout=85 * 60, region="us-east")
    def session():
        import time
        import urllib.request

        env = os.environ.copy()
        env["TRTLLM_ENABLE_PDL"] = "1"

        zero_init_state = rr.apply_zero_init_patch(log=print)
        assert zero_init_state in ("present", "patched"), "NVFP4 zero-init unresolved"
        assert os.path.exists("/cache/kimi_chat_template.jinja"), "chat template missing"
        prompts, sha, real_tok = rr.canonical_prompts(strict=True)
        assert real_tok, "requires the real Kimi tokenizer"
        manifest = json.load(open("/out/brl11_p0_manifest.json"))
        assert manifest.get("prompt_sha256_10k") == sha == rr.PROMPT_SHA_10K, "prompt sha mismatch"

        class Ctx:
            base_url = "http://127.0.0.1:8000/v1"
            model = rr.NVFP4

            def __init__(self):
                self.t0 = time.time()
                self.server = None
                self.prompts = prompts
                self.prompt_sha = sha
                self.zero_init_state = zero_init_state
                self.p0_manifest_summary = {k: manifest.get(k) for k in
                                            ("vllm_version", "vllm_sha", "flashinfer")}
                count_fn = rr.load_kimi_count_fn("/root/assets")
                self.warmup_10k = rr.build_docpack("math", 90, count_fn)

            def spent(self):
                return (time.time() - self.t0) / 3600 * 24.0

            def log(self, msg):
                print(f"[stage600e1|${self.spent():.1f}] {msg}", flush=True)

            def save(self, res):
                with open("/out/brl11_stage600e1.json", "w") as f:
                    json.dump(res, f, indent=1)
                outvol.commit()

            def schat(self, prompt, max_tokens, thinking=True, seed=None, keep_text=True):
                return rr.schat(self.base_url, self.model, prompt, max_tokens, thinking,
                                seed, keep_text)

            def counters(self):
                try:
                    txt = urllib.request.urlopen("http://127.0.0.1:8000/metrics",
                                                 timeout=15).read().decode()
                    return sa.parse_metrics_a(txt)
                except Exception:
                    return None

            def start(self, tag, cfg, boot_budget_min=25):
                import subprocess
                import threading
                self.stop()
                cmd = sa.build_cmd_a(cfg)
                self.log(f"ARM {tag}: {cfg} :: {' '.join(cmd)}")
                self.server = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                               stderr=subprocess.STDOUT, text=True)
                threading.Thread(target=lambda: [print(f"[srv:{tag}]", l.rstrip(), flush=True)
                                                 for l in self.server.stdout], daemon=True).start()
                t0 = time.time()
                while True:
                    if self.server.poll() is not None:
                        raise RuntimeError(f"arm {tag} engine died rc={self.server.returncode}")
                    try:
                        urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3)
                        break
                    except Exception:
                        time.sleep(10)
                    if time.time() - t0 > boot_budget_min * 60:
                        self.stop()
                        raise TimeoutError(f"{tag} not ready in {boot_budget_min} min")
                self.log(f"ARM {tag} READY in {time.time() - t0:.0f}s")

            def stop(self):
                if self.server and self.server.poll() is None:
                    self.server.terminate()
                    try:
                        self.server.wait(timeout=60)
                    except Exception:
                        self.server.kill()
                if self.server is not None:
                    time.sleep(10)
                self.server = None

        res = run_stage600e1(Ctx())
        print("==== STAGE600-E1R DONE ====", flush=True)
        print(json.dumps({"e1": res.get("e1"), "decisions": res.get("decisions")},
                         indent=1), flush=True)
        return {"e1": res.get("e1")}

    @app.local_entrypoint()
    def main():
        print("FINAL:", json.dumps(session.remote(), indent=1))


# ---------------- selftest ----------------
def _selftest():
    os.environ.setdefault("BRL_ALLOW_EST", "1")

    print("[1] ngram_gpu cmdline:", end=" ")
    s = " ".join(sa.build_cmd_a(sa.cfg_a("ngram_gpu", 5)))
    assert '"method": "ngram_gpu"' in s and '"num_speculative_tokens": 5' in s
    assert '"prompt_lookup_max": 4' in s and '"prompt_lookup_min": 2' in s
    assert "--async-scheduling" in s and '"model"' not in s
    assert '"cudagraph_capture_sizes": [1, 2, 3, 4, 5, 6, 12, 18, 24, 48, 96, 192]' in s
    s8 = " ".join(sa.build_cmd_a(sa.cfg_a("ngram_gpu", 8)))
    assert '"num_speculative_tokens": 8' in s8 and "[1, 2, 3, 4, 5, 9, 18, 27, 36, 72, 144, 288]" in s8
    print("OK")

    print("[2] campaign dry-run:", end=" ")
    prompts, sha, _ = rr.canonical_prompts(strict=False)
    PROFILE = {"E1A_ngramgpu_k5": (210.0, 1.9, 0.45), "E1B_nospec": (150.0, None, None),
               "E1C_ngramgpu_k8": (200.0, 2.1, 0.35), "E1D_eagle_k5": (400.0, 4.8, None)}

    class MockCtx:
        def __init__(self, fail_tags=()):
            self.fail_tags = fail_tags
            self.tag = None
            self.boots = []
            self.fake_spent = 0.0
            self.saved = None
            self.prompts = prompts
            self.prompt_sha = sha
            self.zero_init_state = "present"
            self.p0_manifest_summary = {"vllm_sha": "mock"}
            self.warmup_10k = prompts["math"][0][:2000]
            self._c = {"d": 0.0, "a": 0.0, "dt": 0.0, "pp": {}}

        def spent(self):
            self.fake_spent += 0.02
            return self.fake_spent

        def log(self, msg):
            print("   [dry]", msg)

        def save(self, res):
            self.saved = json.loads(json.dumps(res))

        def schat(self, prompt, max_tokens, thinking=True, seed=None, keep_text=True):
            tok_s, tau, hit = PROFILE[self.tag]
            if max_tokens < 200:
                return {"ctok": max_tokens, "ttft_ms": 30.0, "tok_s": tok_s}
            ctok = max_tokens
            if tau is not None:
                k = 8 if "k8" in self.tag else 5
                h = 1.0 if hit is None else hit
                dr = (ctok - 1) * h / (tau * h + (1 - h))
                self._c["d"] += dr
                self._c["a"] += dr * (tau - 1)
                self._c["dt"] += dr * k
            return {"ctok": ctok, "ttft_ms": 300.0, "tok_s": tok_s}

        def counters(self):
            return {k: (dict(v) if isinstance(v, dict) else v) for k, v in self._c.items()}

        def start(self, tag, cfg, boot_budget_min=25):
            if any(t in tag for t in self.fail_tags):
                raise RuntimeError(f"simulated boot failure {tag}")
            sa.build_cmd_a(cfg)
            self.tag = tag
            self.boots.append(tag)

        def stop(self):
            pass

    ctx = MockCtx()
    res = run_stage600e1(ctx)
    assert ctx.boots == ["E1A_ngramgpu_k5", "E1B_nospec", "E1C_ngramgpu_k8", "E1D_eagle_k5"]
    assert res["e1"]["verdict"] in ("draft-loop", "mixed", "verify-bytes")
    assert res["e1"]["v6_ms"] and res["e1"]["v9_ms"] and res["e1"]["draft_pass_ms"] is not None
    assert res["e1"]["t1_source"] == "measured same-node"
    print("OK verdict:", res["e1"]["verdict"], "v6:", res["e1"]["v6_ms"])

    print("[3] E1A boot failure aborts:", end=" ")
    ctx2 = MockCtx(fail_tags=("E1A",))
    res2 = run_stage600e1(ctx2)
    assert "aborting session" in " ".join(res2["decisions"])
    print("OK")

    print("[4] fallbacks on E1B/E1D failure:", end=" ")
    ctx3 = MockCtx(fail_tags=("E1B", "E1D"))
    res3 = run_stage600e1(ctx3)
    assert "fallback" in res3["e1"]["t1_source"] and res3["e1"]["v6_ms"]
    assert "fallback" in res3["e1"]["step_k5_source"]
    print("OK")

    print("SELFTEST PASS")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    elif modal is None:
        print("modal not installed; --selftest available")
