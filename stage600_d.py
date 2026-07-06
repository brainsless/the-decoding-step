"""Stage D: SGLang draft-depth ladder for Kimi-K2.6 EAGLE3 on 4xB200.

Measures how far draft depth pays on SGLang. Prior measurements (same protocol):
depth-6 487.2 tok/s n=16 (tau 5.12, step 10.72ms), depth-7 f8 520.6. SGLang's step
slope is ~0.61 ms/position, so each deeper level costs little; the question is
where the acceptance tail dies. This session climbs depth 8 -> 9 -> 10 (-> 12
stretch) and completes the best level to n=16 on both domains.

Backend gotcha: tokenspeed_mla (grouped-Q kernel) caps verify width at 8 tokens
(MAX_Q_LEN=8), i.e. depth <= 7. Depths >= 8 try it anyway (SGLang may pad/split or
reject); on boot/runtime failure the arm retries once with the auto-selected MLA
backend (no --attention-backend flag) and the artifact records which backend ran.
Absolute tok/s is the readout either way.

Ladder rule (pre-registered): run d8; continue to d(n+1) while
f8(n+1) >= f8(n) * 0.995; stop on decline. Complete the best arm to n=16 tool
(+ math n=16 if spend allows) iff its f8 >= 500.

Artifact: /out/brl11_stage600d.json on volume k26-draft-out.

Usage:
  python stage600_d.py --selftest
  modal run --detach stage600_d.py
  modal app stop brl11-stage600d --yes
"""
import json
import os
import sys

try:
    import modal
except ImportError:
    modal = None

import record_run as rr
import stage600_sg as sg

HARD_CAP = 25.0        # this session's spend ceiling
HOURLY = 24.0          # Modal B200:4 $/hr
D7_REF_F8 = 520.6      # prior-session depth-7 first-8 (same protocol)
CONTINUE_RATIO = 0.995  # climb while f8(n+1) >= f8(n) * this
COMPLETE_FLOOR = 500.0  # complete the best arm to n=16 only above this f8
DEPTHS = (8, 9, 10, 12)


def run_stage600d(ctx):
    res = {"campaign": "stage600_d", "engine": "sglang-0.5.14",
           "arms": {}, "decisions": [], "hourly_usd": HOURLY, "hard_cap_usd": HARD_CAP,
           "refs": {"d6_n16": 487.2, "d7_f8": D7_REF_F8},
           "bars": {"crusoe_median": 438.1, "crusoe_peak": 449.0, "stretch": 500.0,
                    "dream": 600.0},
           "protocol": "brl11_record protocol; probe cells slots 0-8",
           "prompt_sha256_10k": ctx.prompt_sha}

    def note(msg):
        res["decisions"].append(msg)
        ctx.log("DECISION: " + msg)

    def boot_with_fallback(tag, depth):
        """tokenspeed first; auto backend on failure. Returns backend used or None."""
        if ctx.boot(tag, sg.sg_cmd(depth), boot_budget_min=45 if not ctx.booted_once else 25):
            res["arms"].setdefault(tag, {})["backend"] = "tokenspeed_mla"
            res["arms"][tag]["cmd"] = " ".join(sg.sg_cmd(depth))
            return "tokenspeed_mla"
        note(f"{tag}: tokenspeed_mla path failed at depth {depth}; retrying auto backend")
        if ctx.boot(tag, sg.sg_cmd(depth, backend=None), boot_budget_min=25):
            res["arms"].setdefault(tag, {})["backend"] = "auto"
            res["arms"][tag]["cmd"] = " ".join(sg.sg_cmd(depth, backend=None))
            return "auto"
        return None

    prev_f8 = D7_REF_F8
    best = {"f8": None, "tag": None, "depth": None}
    for depth in DEPTHS:
        if ctx.spent() > HARD_CAP - 6.0:
            note(f"depth {depth} skipped by spend guard")
            break
        if depth == 12 and best.get("depth") != 10:
            note("depth 12 stretch skipped: ladder not climbing at 10")
            break
        tag = f"D{depth}"
        backend = boot_with_fallback(tag, depth)
        if backend is None:
            note(f"{tag} failed both backends; stopping ladder")
            break
        cell = sg.run_sg_cell(ctx, res, tag, "tool_nothink_10k", "tool", 2048, False, 0, 8)
        f8 = (cell or {}).get("tok_s_first8_median")
        note(f"depth {depth} [{backend}]: f8={f8} tau={(cell or {}).get('tau')} "
             f"step={(cell or {}).get('step_ms')}ms (prev {prev_f8})")
        if f8 and (best["f8"] is None or f8 > best["f8"]):
            best = {"f8": f8, "tag": tag, "depth": depth}
        stop = not (f8 and prev_f8 and f8 >= prev_f8 * CONTINUE_RATIO)
        prev_f8 = f8 or prev_f8
        ctx.stop()
        if stop:
            note(f"ladder stops: depth {depth} fell below {CONTINUE_RATIO} of previous")
            break

    # complete the best arm if it clears the floor
    if best["tag"] and best["f8"] and best["f8"] >= COMPLETE_FLOOR \
            and ctx.spent() < HARD_CAP - 5.0:
        tag, depth = best["tag"], best["depth"]
        backend = res["arms"][tag].get("backend", "tokenspeed_mla")
        cmd_backend = "tokenspeed_mla" if backend == "tokenspeed_mla" else None
        if ctx.boot(tag, sg.sg_cmd(depth, backend=cmd_backend), boot_budget_min=25):
            note(f"completing best arm {tag} (f8 {best['f8']}) to n=16 on a fresh process; "
                 f"probe rows dropped (single-process cell convention)")
            res["arms"][tag].setdefault("probe_cell", res["arms"][tag]["cells"].pop("tool_nothink_10k", None))
            sg.run_sg_cell(ctx, res, tag, "tool_nothink_10k", "tool", 2048, False, 0, 8)
            sg.run_sg_cell(ctx, res, tag, "tool_nothink_10k", "tool", 2048, False, 8, 16)
            if ctx.spent() < HARD_CAP - 3.0:
                sg.run_sg_cell(ctx, res, tag, "math_nothink_10k", "math", 2048, False, 0, 8)
                sg.run_sg_cell(ctx, res, tag, "math_nothink_10k", "math", 2048, False, 8, 16)
            ctx.stop()
    elif best["tag"]:
        note(f"best arm {best['tag']} f8 {best['f8']} under completion floor {COMPLETE_FLOOR}")

    n16 = None
    for tag, arm in res["arms"].items():
        c = (arm.get("cells") or {}).get("tool_nothink_10k")
        if c and c.get("n", 0) >= 16 and c.get("tok_s_median"):
            if n16 is None or c["tok_s_median"] > n16["tok_s"]:
                n16 = {"tok_s": c["tok_s_median"], "arm": tag, "tau": c.get("tau"),
                       "step_ms": c.get("step_ms"), "backend": arm.get("backend")}
    res["ship"] = {"best_probe": best, "best_n16": n16}
    res["total_spent_usd_est"] = round(ctx.spent(), 2)
    note(f"DONE: best_probe={best} best_n16={n16} ${ctx.spent():.1f}")
    ctx.save(res)
    return res


# ---------------- Modal wiring ----------------
if modal is not None:
    app = modal.App("brl11-stage600d")
    kimi = modal.Volume.from_name("kimi-k26", create_if_missing=False)
    hfvol = modal.Volume.from_name("hf-cache", create_if_missing=True)
    outvol = modal.Volume.from_name("k26-draft-out", create_if_missing=True)

    image = sg.sg_image.add_local_python_source("stage600_d")

    @app.function(image=image, gpu="B200:4",
                  volumes={"/models": kimi, "/cache": hfvol, "/out": outvol},
                  timeout=80 * 60, region="us-east")
    def session():
        import subprocess
        import threading
        import time
        import urllib.request

        assert os.path.exists(f"{sg.MODEL}/hf_quant_config.json"), "modelopt marker missing"
        assert os.path.exists("/cache/kimi_chat_template.jinja"), "chat template missing"
        prompts, sha, real_tok = rr.canonical_prompts(strict=True)
        assert real_tok and sha == rr.PROMPT_SHA_10K, "prompt handshake failed"

        class Ctx:
            base_url = f"http://127.0.0.1:{sg.SG_PORT}/v1"
            model = sg.MODEL

            def __init__(self):
                self.t0 = time.time()
                self.server = None
                self.booted_once = False
                self.prompts = prompts
                self.prompt_sha = sha

            def spent(self):
                return (time.time() - self.t0) / 3600 * HOURLY

            def log(self, msg):
                print(f"[stage600d|${self.spent():.1f}] {msg}", flush=True)

            def save(self, res):
                with open("/out/brl11_stage600d.json", "w") as f:
                    json.dump(res, f, indent=1)
                outvol.commit()

            def schat(self, prompt, max_tokens, thinking=True, seed=None, keep_text=True):
                return rr.schat(self.base_url, self.model, prompt, max_tokens, thinking,
                                seed, keep_text)

            def counters(self):
                try:
                    txt = urllib.request.urlopen(
                        f"http://127.0.0.1:{sg.SG_PORT}/metrics", timeout=15).read().decode()
                    return sg.parse_metrics_sg(txt)
                except Exception:
                    return None

            def boot(self, tag, cmd, boot_budget_min=25):
                self.stop()
                self.log(f"ARM {tag}: {' '.join(cmd)}")
                try:
                    self.server = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                                   stderr=subprocess.STDOUT, text=True)
                    threading.Thread(target=lambda: [print(f"[srv:{tag}]", l.rstrip(), flush=True)
                                                     for l in self.server.stdout], daemon=True).start()
                    t0 = time.time()
                    while True:
                        if self.server.poll() is not None:
                            raise RuntimeError(f"{tag} engine died rc={self.server.returncode}")
                        try:
                            urllib.request.urlopen(
                                f"http://127.0.0.1:{sg.SG_PORT}/health", timeout=3)
                            break
                        except Exception:
                            time.sleep(10)
                        if time.time() - t0 > boot_budget_min * 60:
                            self.stop()
                            raise TimeoutError(f"{tag} not ready in {boot_budget_min} min")
                    self.log(f"ARM {tag} READY in {time.time() - t0:.0f}s")
                    self.schat("Reply with exactly: WARMUP OK", 48, True, seed=rr.SEED_BASE, keep_text=False)
                    self.schat("Reply with exactly: WARMUP OK", 48, False, seed=rr.SEED_BASE, keep_text=False)
                    count_fn = rr.load_kimi_count_fn("/root/assets")
                    self.schat(rr.build_docpack("math", 90, count_fn), 128, False,
                               seed=rr.SEED_BASE, keep_text=False)
                    self.booted_once = True
                    return True
                except Exception as e:  # noqa: BLE001
                    self.log(f"ARM {tag} FAILED TO BOOT: {repr(e)[:250]}")
                    self.stop()
                    return False

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

        res = run_stage600d(Ctx())
        print("==== STAGE600-D DONE ====", flush=True)
        print(json.dumps({"ship": res.get("ship"), "decisions": res.get("decisions")},
                         indent=1), flush=True)
        return {"ship": res.get("ship")}

    @app.local_entrypoint()
    def main():
        print("FINAL:", json.dumps(session.remote(), indent=1))


# ---------------- selftest ----------------
def _selftest():
    os.environ.setdefault("BRL_ALLOW_EST", "1")

    print("[1] cmdlines incl. backend fallback:", end=" ")
    s8 = " ".join(sg.sg_cmd(8))
    assert "--speculative-num-steps 8" in s8 and "--speculative-num-draft-tokens 9" in s8
    assert "--attention-backend tokenspeed_mla" in s8
    s8a = " ".join(sg.sg_cmd(8, backend=None))
    assert "--attention-backend" not in s8a and "--kv-cache-dtype fp8_e4m3" in s8a
    s12 = " ".join(sg.sg_cmd(12))
    assert "--speculative-num-draft-tokens 13" in s12
    print("OK")

    print("[2] ladder dry-runs:", end=" ")
    prompts, sha, _ = rr.canonical_prompts(strict=False)

    class MockCtx:
        def __init__(self, profile, fail_first=()):
            self.profile = profile
            self.fail_first = set(fail_first)
            self.tag = None
            self.boots = []
            self.fake_spent = 0.0
            self.saved = None
            self.booted_once = False
            self.prompts = prompts
            self.prompt_sha = sha
            self._steps = 0.0

        def spent(self):
            self.fake_spent += 0.015
            return self.fake_spent

        def log(self, msg):
            print("   [dry]", msg)

        def save(self, res):
            self.saved = json.loads(json.dumps(res))

        def schat(self, prompt, max_tokens, thinking=True, seed=None, keep_text=True):
            tok_s, tau = self.profile[self.tag]
            if max_tokens < 200:
                return {"ctok": max_tokens, "ttft_ms": 30.0, "tok_s": tok_s}
            self._steps += max_tokens / tau
            return {"ctok": max_tokens, "ttft_ms": 240.0, "tok_s": tok_s}

        def counters(self):
            return {"spec_verify_calls_total": self._steps}

        def boot(self, tag, cmd, boot_budget_min=25):
            if tag in self.fail_first and "tokenspeed_mla" in " ".join(cmd):
                return False
            self.tag = tag
            self.boots.append((tag, "ts" if "tokenspeed_mla" in " ".join(cmd) else "auto"))
            self.booted_once = True
            return True

        def stop(self):
            pass

    # climbing ladder with d8 tokenspeed failure -> auto fallback; peak at d9
    ctx = MockCtx({"D8": (528.0, 6.1), "D9": (541.0, 6.7), "D10": (531.0, 7.1)},
                  fail_first=("D8",))
    res = run_stage600d(ctx)
    assert ("D8", "auto") in ctx.boots, ctx.boots
    assert res["ship"]["best_probe"]["depth"] == 9
    assert res["arms"]["D9"]["cells"]["tool_nothink_10k"]["n"] == 16
    assert res["arms"]["D9"]["cells"]["math_nothink_10k"]["n"] == 16
    assert "ladder stops: depth 10" in " ".join(res["decisions"])
    print("OK peak d9,", res["ship"]["best_n16"]["tok_s"])

    print("[3] immediate-decline stops ladder, no completion under floor:", end=" ")
    ctx2 = MockCtx({"D8": (490.0, 6.0), "D9": (480.0, 6.5), "D10": (470.0, 7.0)})
    res2 = run_stage600d(ctx2)
    assert [t for t, _ in ctx2.boots] == ["D8"], ctx2.boots
    assert res2["ship"]["best_n16"] is None
    assert "under completion floor" in " ".join(res2["decisions"])
    print("OK")

    print("SELFTEST PASS")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    elif modal is None:
        print("modal not installed; --selftest available")
