"""SGLang bring-up and step-time comparison for Kimi-K2.6 speculative decoding.

Context: on vLLM the decode step is dominated by drafter-loop and host overhead
(draft pass 0.73-1.07 ms against a ~0.2 ms bandwidth floor; verify slope
0.397 ms/tok), and TP8 does not improve the step over TP4 (6.54 vs 6.60 ms).
SGLang's overlap scheduler, the tokenspeed_mla attention backend, and the
flashinfer_trtllm MoE backend target that overhead, and SGLang CI covers this
exact model + head combination (Kimi-K2.6-NVFP4 with
lightseekorg/kimi-k2.6-eagle3.1-mla) on B200 (PRs #26506, #28467; v0.5.14).

Arms (4x B200; prompts, seeds, and protocol identical to the record runner):
  SG0  spec triple 3/1/4 (steps/topk/draft-tokens), the CI-tested config; its
       published accept length (3.18-3.20 on GSM8K) sanity-checks the tau
       parser in-session (different domain, so ordering check only).
  SG1  5/1/6, direct step-time comparison with vLLM k=5 (12.24 ms).
  SG2  6/1/7, the vLLM k=6 configuration (407 tok/s reference).
Decision rule, fixed before the run: step(SG1) <= 11.0 ms selects SGLang for
the record session; any arm reaching f8 >= 412 tok/s is completed to n=16.

Launch constraints (source-verified against SGLang v0.5.14):
  1  draft head kept BF16: --speculative-draft-model-quantization unquant
  2  attention backend and KV dtype pinned together: tokenspeed_mla + fp8_e4m3
  3  spec triple fully specified (upstream asserts draft_tokens == steps+1)
  4  head repo and SGLang version pinned exactly
  5  mem-fraction 0.85, cuda-graph-max-bs-decode 8, max-running-requests 8
Pre-serve checks fail closed: modelopt export marker in the model dir, chat
template file present, prompt SHA equal to the pin.

Usage:
  modal run stage600_sg.py           # standalone SGLang session
  python stage600_sg.py --selftest   # offline logic check, no GPU or Modal auth

Writes /out/brl11_stage600sg.json: raw SGLang metric snapshots per cell for
offline re-derivation; tau = completion tokens / verify-step delta,
cross-checked against the SG0 anchor.
"""
import json
import os
import re
import statistics
import sys

try:
    import modal
except ImportError:
    modal = None

import record_run as rr

HARD_CAP = 35.0  # session spend ceiling, USD
HOURLY = 24.0    # 4x B200 rate, USD/hr
# vLLM reference measurements under the same protocol (ms and tok/s)
VLLM_REFS = {"t1_ms": 6.6, "step_k5_ms": 12.24, "k5_f8": 403.2, "k6_f8": 407.0,
             "verify_slope_ms": 0.397, "draft_pass_ms": "0.73-1.07"}
GATE_STEP_SG1 = 11.0     # step(5/1/6) at or under this selects SGLang for the record run
BANK_FLOOR = 412.0       # f8 at or above this completes the cell to n=16 immediately
CI_ANCHOR = (3.18, 3.20)  # PR #26506 accept length for triple 3/1/4 (GSM8K; order check)
SG_PORT = 30000
MODEL = "/models/Kimi-K2.6-NVFP4"
DRAFT = "lightseekorg/kimi-k2.6-eagle3.1-mla"


def sg_cmd(steps, tp=4, draft=DRAFT, backend="tokenspeed_mla"):
    """SGLang launch command. Flag set matches SGLang's own 4x B200 K2.6 CI test
    (PR #28467) plus the chat template and single-stream graph bounds. draft_tokens =
    steps+1 (eagle_worker_v2.py:275 assert, topk=1 chain). backend=None omits the
    attention override (auto-select); the tokenspeed_mla kernel caps verify width at
    8 tokens (grouped-Q MAX_Q_LEN=8), so depths >= 8 may need the fallback."""
    return ["python", "-m", "sglang.launch_server",
            "--model-path", MODEL,
            "--tp-size", str(tp),
            "--trust-remote-code",
            "--quantization", "modelopt_fp4",
            "--kv-cache-dtype", "fp8_e4m3",
            *(["--attention-backend", backend] if backend else []),
            "--moe-runner-backend", "flashinfer_trtllm",
            "--speculative-algorithm", "EAGLE3",
            "--speculative-draft-model-path", draft,
            "--speculative-num-steps", str(steps),
            "--speculative-eagle-topk", "1",
            "--speculative-num-draft-tokens", str(steps + 1),
            "--speculative-draft-model-quantization", "unquant",
            "--mem-fraction-static", "0.85",
            "--max-running-requests", "8",
            "--cuda-graph-max-bs-decode", "8",
            "--chat-template", "/cache/kimi_chat_template.jinja",
            "--enable-metrics",
            "--host", "0.0.0.0", "--port", str(SG_PORT)]


def parse_metrics_sg(txt):
    """Scrape every sglang:* counter. Tau is derived at cell level from stored raw
    deltas; metric naming is version-dependent, so everything is kept."""
    out = {}
    for m in re.finditer(r'^sglang:([a-z_]+)(?:{[^}]*})?\s+([0-9.eE+-]+)\s*$', txt, re.M):
        out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


def tau_from_deltas(ctok_sum, deltas):
    """tau = completion tokens / verify steps. Candidate step counters in preference
    order; SG0's CI anchor (3.18-3.20 at 3/1/4) validates the choice in-session."""
    # spec_verify_calls_total is the verify-step counter on v0.5.14 (identified from a
    # prior run's metric deltas; depth-5 tau 4.825 vs 4.815 on vLLM with the same head)
    for key in ("spec_verify_calls_total", "spec_num_steps", "spec_verify_ct", "num_spec_steps"):
        v = deltas.get(key)
        if v and v > 0:
            return round(ctok_sum / v, 4), key
    return None, None


def run_sg_cell(ctx, res, arm, name, dom, max_tokens, thinking, lo, hi):
    if ctx.spent() > HARD_CAP - 2.0:
        ctx.log(f"SPEND GUARD: skipping {arm}/{name}")
        return None
    rows = []
    consec_err = 0
    for slot in range(lo, hi):
        if ctx.spent() > HARD_CAP - 1.5:
            ctx.log(f"SPEND GUARD inside {arm}/{name} at slot {slot}")
            break
        b = ctx.counters()
        try:
            row = ctx.schat(ctx.prompts[dom][slot], max_tokens, thinking,
                            seed=rr.SEED_BASE + slot)
            consec_err = 0
        except Exception as e:  # noqa: BLE001
            row = {"error": repr(e)[:120]}
            consec_err += 1
        a = ctx.counters()
        row["slot"] = slot
        if b and a:
            row["sg_deltas"] = {k: round(a[k] - b.get(k, 0.0), 1) for k in a
                                if a[k] != b.get(k, 0.0)}
        rows.append(row)
        if consec_err >= 3:
            ctx.log(f"ABORT CELL {arm}/{name}: 3 consecutive request errors")
            break
    cells = res["arms"].setdefault(arm, {}).setdefault("cells", {})
    prev = cells.get(name, {}).get("rows", [])
    rows = prev + rows
    out = rr.cell_stats(rows, power=None)
    ctok_sum = sum(r.get("ctok") or 0 for r in rows if r.get("tok_s"))
    agg_deltas = {}
    for r in rows:
        for k, v in (r.get("sg_deltas") or {}).items():
            agg_deltas[k] = agg_deltas.get(k, 0.0) + v
    out["tau"], out["tau_counter"] = tau_from_deltas(ctok_sum, agg_deltas)
    out["sg_deltas_cell"] = agg_deltas
    out["step_ms"] = round(1000.0 * out["tau"] / out["tok_s_agg"], 2) \
        if out.get("tau") and out.get("tok_s_agg") else None
    cells[name] = out
    ctx.log(f"{arm} {name}[{lo}:{hi}]: med={out['tok_s_median']} f8={out['tok_s_first8_median']} "
            f"agg={out['tok_s_agg']} tau={out['tau']}({out['tau_counter']}) "
            f"step={out['step_ms']}ms n={out['n']} ${ctx.spent():.1f}")
    ctx.save(res)
    return out


def run_stage600sg(ctx):
    res = {"campaign": "stage600_sg", "engine": "sglang-0.5.14",
           "arms": {}, "gates": {}, "decisions": [],
           "hourly_usd": HOURLY, "hard_cap_usd": HARD_CAP,
           "vllm_refs": VLLM_REFS,
           "bars": {"crusoe_median_2026_07_05": 438.1, "crusoe_peak": 449.0,
                    "prior_record": 398.5},
           "protocol": "identical prompts/seeds/streaming stat to brl11_record",
           "prompt_sha256_10k": ctx.prompt_sha}

    def note(msg):
        res["decisions"].append(msg)
        ctx.log("DECISION: " + msg)

    steps_ladder = [(3, "SG0_s3"), (5, "SG1_s5"), (6, "SG2_s6")]
    step_sg1 = None
    for steps, tag in steps_ladder:
        if ctx.spent() > HARD_CAP - 7.0:
            note(f"{tag} skipped by spend guard")
            break
        if not ctx.boot(tag, sg_cmd(steps), boot_budget_min=45 if steps == 3 else 25):
            res["arms"].setdefault(tag, {})["boot_failed"] = True
            note(f"{tag} failed to boot")
            if steps == 3:
                note("CI-proven config failed to boot: aborting (env problem, not config)")
                break
            continue
        res["arms"].setdefault(tag, {})["cmd"] = " ".join(sg_cmd(steps))
        cell = run_sg_cell(ctx, res, tag, "tool_nothink_10k", "tool", 2048, False, 0, 8)
        if steps == 3 and cell and cell.get("tau"):
            ok = cell["tau"] > 2.0  # domain differs from GSM8K; sanity order-check only
            note(f"SG0 tau parser calibration: tau={cell['tau']} via {cell['tau_counter']} "
                 f"(CI anchor {CI_ANCHOR} on GSM8K) -> {'OK' if ok else 'SUSPECT'}")
        if steps == 5:
            step_sg1 = (cell or {}).get("step_ms")
        f8 = (cell or {}).get("tok_s_first8_median")
        if f8 and f8 >= BANK_FLOOR and ctx.spent() < HARD_CAP - 5.0:
            note(f"{tag} banks: f8 {f8} >= {BANK_FLOOR}; completing to n=16")
            run_sg_cell(ctx, res, tag, "tool_nothink_10k", "tool", 2048, False, 8, 16)
        ctx.stop()

    res["gates"]["G_sglang"] = {
        "step_sg1_ms": step_sg1, "vllm_step_k5_ms": VLLM_REFS["step_k5_ms"],
        "thresh": GATE_STEP_SG1,
        "fund_record": bool(step_sg1 and step_sg1 <= GATE_STEP_SG1)}
    best = None
    for tag, arm in res["arms"].items():
        c = (arm.get("cells") or {}).get("tool_nothink_10k")
        if c and c.get("tok_s_median") and (best is None or c["tok_s_median"] > best["tok_s"]):
            best = {"tok_s": c["tok_s_median"], "arm": tag, "n": c.get("n"),
                    "tau": c.get("tau"), "step_ms": c.get("step_ms")}
    res["ship"] = {"best_cell": best}
    res["total_spent_usd_est"] = round(ctx.spent(), 2)
    note(f"DONE: fund_record={res['gates']['G_sglang']['fund_record']} "
         f"step_sg1={step_sg1} best={best} ${ctx.spent():.1f}")
    ctx.save(res)
    return res


# ---------------- Modal wiring ----------------
if modal is not None:
    app = modal.App("brl11-stage600sg")
    kimi = modal.Volume.from_name("kimi-k26", create_if_missing=False)
    hfvol = modal.Volume.from_name("hf-cache", create_if_missing=True)
    outvol = modal.Volume.from_name("k26-draft-out", create_if_missing=True)

    sg_image = (
        modal.Image.from_registry("lmsysorg/sglang:v0.5.14", add_python=None)
        .pip_install("openai")
        .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache/hf"})
        .add_local_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "kimi_tiktoken.model"),
                        remote_path="/root/assets/kimi_tiktoken.model")
        .add_local_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "tokenization_kimi.py"),
                        remote_path="/root/assets/tokenization_kimi.py")
        .add_local_python_source("record_run", "stage600_sg")
    )

    @app.function(image=sg_image, gpu="B200:4",
                  volumes={"/models": kimi, "/cache": hfvol, "/out": outvol},
                  timeout=95 * 60, region="us-east")
    def session():
        import subprocess
        import threading
        import time
        import urllib.request

        assert os.path.exists(f"{MODEL}/hf_quant_config.json"), \
            f"modelopt export marker missing in {MODEL} (sglang modelopt_fp4 needs it); " \
            f"dir: {os.listdir(MODEL)[:20]}"
        assert os.path.exists("/cache/kimi_chat_template.jinja"), "chat template missing"
        prompts, sha, real_tok = rr.canonical_prompts(strict=True)
        assert real_tok, "requires the real Kimi tokenizer"
        assert sha == rr.PROMPT_SHA_10K, "prompt sha mismatch"

        class Ctx:
            base_url = f"http://127.0.0.1:{SG_PORT}/v1"
            model = MODEL

            def __init__(self):
                self.t0 = time.time()
                self.server = None
                self.prompts = prompts
                self.prompt_sha = sha

            def spent(self):
                return (time.time() - self.t0) / 3600 * HOURLY

            def log(self, msg):
                print(f"[stage600sg|${self.spent():.1f}] {msg}", flush=True)

            def save(self, res):
                with open("/out/brl11_stage600sg.json", "w") as f:
                    json.dump(res, f, indent=1)
                outvol.commit()

            def schat(self, prompt, max_tokens, thinking=True, seed=None, keep_text=True):
                return rr.schat(self.base_url, self.model, prompt, max_tokens, thinking,
                                seed, keep_text)

            def counters(self):
                try:
                    txt = urllib.request.urlopen(
                        f"http://127.0.0.1:{SG_PORT}/metrics", timeout=15).read().decode()
                    return parse_metrics_sg(txt)
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
                                f"http://127.0.0.1:{SG_PORT}/health", timeout=3)
                            break
                        except Exception:
                            time.sleep(10)
                        if time.time() - t0 > boot_budget_min * 60:
                            self.stop()
                            raise TimeoutError(f"{tag} not ready in {boot_budget_min} min")
                    self.log(f"ARM {tag} READY in {time.time() - t0:.0f}s")
                    # warmups: chat template in both thinking modes, then a long-context prompt
                    self.schat("Reply with exactly: WARMUP OK", 48, True, seed=rr.SEED_BASE, keep_text=False)
                    self.schat("Reply with exactly: WARMUP OK", 48, False, seed=rr.SEED_BASE, keep_text=False)
                    count_fn = rr.load_kimi_count_fn("/root/assets")
                    self.schat(rr.build_docpack("math", 90, count_fn), 128, False,
                               seed=rr.SEED_BASE, keep_text=False)
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

        res = run_stage600sg(Ctx())
        print("==== STAGE600-SG DONE ====", flush=True)
        print(json.dumps({"gates": res.get("gates"), "ship": res.get("ship"),
                          "decisions": res.get("decisions")}, indent=1), flush=True)
        return {"gates": res.get("gates"), "ship": res.get("ship")}

    @app.local_entrypoint()
    def main():
        print("FINAL:", json.dumps(session.remote(), indent=1))


# ---------------- selftest ----------------
def _selftest():
    os.environ.setdefault("BRL_ALLOW_EST", "1")

    print("[1] sglang cmdline:", end=" ")
    c = sg_cmd(5)
    s = " ".join(c)
    assert "--speculative-num-steps 5" in s and "--speculative-num-draft-tokens 6" in s
    assert "--speculative-eagle-topk 1" in s
    assert "--speculative-draft-model-quantization unquant" in s, "draft head must stay BF16"
    assert "--attention-backend tokenspeed_mla" in s and "--kv-cache-dtype fp8_e4m3" in s, \
        "attention backend and KV dtype must be pinned together"
    assert "--quantization modelopt_fp4" in s and "--enable-metrics" in s
    assert "--chat-template /cache/kimi_chat_template.jinja" in s
    s8 = " ".join(sg_cmd(6, tp=8))
    assert "--tp-size 8" in s8 and "--speculative-num-draft-tokens 7" in s8
    print("OK")

    print("[2] sg metrics parser + tau:", end=" ")
    txt = ('sglang:spec_num_steps{x="1"} 500\n'
           'sglang:spec_num_draft_tokens{x="1"} 2500\n'
           'sglang:num_requests_total 12\n')
    m = parse_metrics_sg(txt)
    assert m["spec_num_steps"] == 500 and m["spec_num_draft_tokens"] == 2500
    tau, key = tau_from_deltas(1600, {"spec_num_steps": 500})
    assert tau == 3.2 and key == "spec_num_steps"
    assert tau_from_deltas(100, {})[0] is None
    print("OK tau", tau)

    print("[3] dry-run:", end=" ")
    prompts, sha, _ = rr.canonical_prompts(strict=False)
    PROFILE = {"SG0_s3": (380.0, 3.2), "SG1_s5": (455.0, 4.8), "SG2_s6": (470.0, 5.3)}

    class MockCtx:
        def __init__(self, fail_tags=()):
            self.fail_tags = fail_tags
            self.tag = None
            self.fake_spent = 0.0
            self.saved = None
            self.prompts = prompts
            self.prompt_sha = sha
            self._steps = 0.0

        def spent(self):
            self.fake_spent += 0.03
            return self.fake_spent

        def log(self, msg):
            print("   [dry]", msg)

        def save(self, res):
            self.saved = json.loads(json.dumps(res))

        def schat(self, prompt, max_tokens, thinking=True, seed=None, keep_text=True):
            tok_s, tau = PROFILE[self.tag]
            if max_tokens < 200:
                return {"ctok": max_tokens, "ttft_ms": 30.0, "tok_s": tok_s}
            self._steps += max_tokens / tau
            return {"ctok": max_tokens, "ttft_ms": 300.0, "tok_s": tok_s}

        def counters(self):
            return {"spec_num_steps": self._steps, "spec_num_draft_tokens": self._steps * 5}

        def boot(self, tag, cmd, boot_budget_min=25):
            if any(t in tag for t in self.fail_tags):
                return False
            self.tag = tag
            return True

        def stop(self):
            pass

    ctx = MockCtx()
    res = run_stage600sg(ctx)
    g = res["gates"]["G_sglang"]
    assert g["fund_record"] and g["step_sg1_ms"] and g["step_sg1_ms"] < 11.0, g
    assert res["arms"]["SG1_s5"]["cells"]["tool_nothink_10k"]["n"] == 16, \
        "f8 455 must complete to n=16"
    assert "calibration" in " ".join(res["decisions"])
    print("OK step_sg1:", g["step_sg1_ms"], "best:", res["ship"]["best_cell"]["tok_s"])

    print("[4] SG0 boot failure aborts:", end=" ")
    ctx2 = MockCtx(fail_tags=("SG0",))
    res2 = run_stage600sg(ctx2)
    assert "aborting" in " ".join(res2["decisions"]) and "SG1_s5" not in res2["arms"]
    print("OK")

    print("SELFTEST PASS")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    elif modal is None:
        print("modal not installed; --selftest available")
