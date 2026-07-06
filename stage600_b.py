"""Stage B: 8xB200 TP8 probe for Kimi-K2.6 EAGLE3 serving on vLLM.

Run after stage600_a analysis; pass the best ladder k via --k2 (default 7).

Measures (tool-nothink 10k protocol, prompts/seeds identical to prior sessions):
  Tt(TP8)      no-spec single-token step at 10k. 4xB200 reference: ~7.2ms (A1_nospec).
  step(k5,TP8) EAGLE k=5 full step. 4xB200 reference: ~12.0-12.5ms.
  step(kX,TP8) EAGLE at the best ladder k from stage A.
  V6(TP8)      optional ngram k=5 pure-verify step (E1 cross-check on 8 GPUs).

Pre-registered gates (written into the artifact):
  fund_c:     Tt <= 4.8ms AND step(k5) <= 10.4ms -> predicted >= 460 tok/s at
              measured tau; a full 8xB200 record session is justified.
  comm_bound: Tt >= 5.6ms -> vLLM TP8 is latency/comm-bound (capped < ~480);
              higher rates need an engine/kernel change, not more GPUs.
  bank:       any spec cell with first-8 median >= 412 completes to n=16 in the
              same session as a record-eligible cell.

Cells: B0 no-spec, B1 EAGLE k=5, B2 EAGLE k=k2, B3 optional ngram k=5.

Artifact: /out/brl11_stage600b.json on volume k26-draft-out (saved after every cell).

Usage:
  python stage600_b.py --selftest
  modal run --detach stage600_b.py --k2 7
  modal app stop <app-id> --yes   # app id (ap-...) is printed at launch; name lookup does not work for detached runs
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

HARD_CAP = 80.0          # this session's spend ceiling
HOURLY = 50.0            # Modal B200:8 $/hr
BANK_FLOOR = 412.0       # first-8 median needed to complete a cell to n=16
GATE_FUND_TT = 4.8
GATE_FUND_STEP5 = 10.4
GATE_COMM_TT = 5.6
REF_4X = {"t1_ms": None, "step_k5_ms": None}  # filled at launch from stage-A artifact


def cfg_b(method, k=None, head=None, argmax=False):
    # Same config surface as stage600_a with one change: tensor-parallel-size 8.
    # Capture sizes, chat template, attention-config, async-scheduling, fp8 KV are
    # identical. TP8 sanity: 64 attention heads / 8 = 8 heads per GPU; 554GiB of
    # weights over 1536GB HBM; K2.6 EAGLE3-MLA at TP8 has community precedent
    # (vLLM issue #40608, sm120).
    c = sa.cfg_a(method, k=k, head=head, argmax=argmax)
    c["tp"] = 8
    return c


def build_cmd_b(cfg):
    """stage600_a command with tensor-parallel-size swapped to 8 (only diff, asserted)."""
    cmd = sa.build_cmd_a(cfg)  # same verified flag surface as stage A
    i = cmd.index("--tensor-parallel-size")
    assert cmd[i + 1] == "4"
    cmd[i + 1] = "8"
    for tok in cmd:
        if tok.startswith("--"):
            assert tok in rr.ALLOWED_FLAGS, f"unverified flag {tok}"
    return cmd


def gate_block(t1_ms, step_k5, step_kx, kx, tau_k5, tau_kx):
    g = {"t1_tp8_ms": t1_ms, "step_k5_tp8_ms": step_k5, f"step_k{kx}_tp8_ms": step_kx,
         "thresholds": {"fund_tt": GATE_FUND_TT, "fund_step5": GATE_FUND_STEP5,
                        "comm_bound_tt": GATE_COMM_TT}}
    g["fund_c"] = bool(t1_ms and step_k5 and t1_ms <= GATE_FUND_TT and step_k5 <= GATE_FUND_STEP5)
    g["comm_bound"] = bool(t1_ms and t1_ms >= GATE_COMM_TT)
    if step_k5 and tau_k5:
        g["pred_tok_s_k5"] = round(1000.0 * tau_k5 / step_k5, 1)
    if step_kx and tau_kx:
        g["pred_tok_s_kx"] = round(1000.0 * tau_kx / step_kx, 1)
    return g


def run_stage600b(ctx, k2=7, ngram_probe=True):
    res = {"campaign": "stage600_b", "arms": {}, "gates": {}, "decisions": [],
           "hourly_usd": HOURLY, "hard_cap_usd": HARD_CAP,
           "bars": {"crusoe_median_2026_07_05": sa.BAR_CRUSOE, "crusoe_peak": 449.0,
                    "prior_record": sa.PRIOR_RECORD},
           "ref_4x": REF_4X,
           "protocol": "identical to brl11_record; TP8; probe cells slots 0-8",
           "prompt_sha256_10k": ctx.prompt_sha,
           "zero_init": ctx.zero_init_state,
           "p0_manifest": ctx.p0_manifest_summary}

    def note(msg):
        res["decisions"].append(msg)
        ctx.log("DECISION: " + msg)

    def guard(need):
        return ctx.spent() < HARD_CAP - need

    # ---- B0: no-spec TP8 -> Tt ----
    t1_ms = None
    if not sa.run_arm_boot_a(ctx, res, "B0_nospec_tp8", cfg_b("none"), boot_budget_min=40):
        note("B0 no-spec failed to boot at TP8; session aborted (nothing downstream valid)")
        ctx.save(res)
        return res
    run_cell_b = sa.run_cell_a  # identical measurement core
    run_cell_b(ctx, res, "B0_nospec_tp8", "tool_nothink_10k", "tool", 2048, False, 0, 4)
    c0 = res["arms"]["B0_nospec_tp8"]["cells"].get("tool_nothink_10k")
    if c0 and c0.get("tok_s_agg"):
        t1_ms = round(1000.0 / c0["tok_s_agg"], 2)
    note(f"Tt(TP8) = {t1_ms}ms (4x ref {REF_4X.get('t1_ms')}ms)")
    ctx.stop()

    # ---- B1: EAGLE k=5 TP8 ----
    step_k5 = tau_k5 = None
    if guard(20.0) and sa.run_arm_boot_a(ctx, res, "B1_k5_tp8",
                                         cfg_b("eagle3", 5, rr.EAGLE, argmax=True), 25):
        cell = run_cell_b(ctx, res, "B1_k5_tp8", "tool_nothink_10k", "tool", 2048, False, 0, 8)
        step_k5, tau_k5 = sa.step_ms_cell(cell), (cell or {}).get("tau")
        f8 = (cell or {}).get("tok_s_first8_median")
        if f8 and f8 >= BANK_FLOOR and guard(8.0):
            note(f"B1 banks: f8 {f8} >= {BANK_FLOOR}; completing to n=16")
            run_cell_b(ctx, res, "B1_k5_tp8", "tool_nothink_10k", "tool", 2048, False, 8, 16)
        ctx.stop()

    # ---- B2: EAGLE k=k2 TP8 ----
    step_kx = tau_kx = None
    if guard(14.0) and k2 != 5 and sa.run_arm_boot_a(ctx, res, f"B2_k{k2}_tp8",
                                                     cfg_b("eagle3", k2, rr.EAGLE, argmax=True), 25):
        cell = run_cell_b(ctx, res, f"B2_k{k2}_tp8", "tool_nothink_10k", "tool", 2048, False, 0, 8)
        step_kx, tau_kx = sa.step_ms_cell(cell), (cell or {}).get("tau")
        f8 = (cell or {}).get("tok_s_first8_median")
        if f8 and f8 >= BANK_FLOOR and guard(8.0):
            note(f"B2 banks: f8 {f8} >= {BANK_FLOOR}; completing to n=16")
            run_cell_b(ctx, res, f"B2_k{k2}_tp8", "tool_nothink_10k", "tool", 2048, False, 8, 16)
        ctx.stop()

    # ---- B3: optional ngram k=5 TP8 (V6 on 8 GPUs; E1 cross-check) ----
    if ngram_probe and guard(12.0) and t1_ms and \
            sa.run_arm_boot_a(ctx, res, "B3_ngram_k5_tp8", cfg_b("ngram", 5), 25):
        cell = run_cell_b(ctx, res, "B3_ngram_k5_tp8", "tool_nothink_10k", "tool", 2048, False, 0, 4)
        if cell and cell.get("rows"):
            res["e1_tp8"] = {"v6_tp8_ms": sa.verify_cost_ms(cell["rows"], t1_ms),
                             "hit_rate": sa.hit_rate(cell["rows"]),
                             "t1_ms": t1_ms}
            note(f"V6(TP8) = {res['e1_tp8']['v6_tp8_ms']}ms")
        ctx.stop()

    res["gates"] = gate_block(t1_ms, step_k5, step_kx, k2, tau_k5, tau_kx)
    note(f"GATES: fund_c={res['gates'].get('fund_c')} comm_bound={res['gates'].get('comm_bound')} "
         f"pred_k5={res['gates'].get('pred_tok_s_k5')} pred_k{k2}={res['gates'].get('pred_tok_s_kx')}")

    best = None
    for tag, arm in res["arms"].items():
        c = (arm.get("cells") or {}).get("tool_nothink_10k")
        if c and c.get("n", 0) >= 16 and c.get("tok_s_median"):
            if best is None or c["tok_s_median"] > best["tok_s"]:
                best = {"tok_s": c["tok_s_median"], "arm": tag, "tau": c.get("tau"),
                        "aa_len_ok": c.get("aa_len_ok")}
    res["ship"] = {"best_n16_cell": best,
                   "vs": {"prior_record": sa.PRIOR_RECORD, "crusoe": sa.BAR_CRUSOE}}
    res["total_spent_usd_est"] = round(ctx.spent(), 2)
    note(f"DONE: gates={res['gates'].get('fund_c')}/{res['gates'].get('comm_bound')} "
         f"best_n16={best} ${ctx.spent():.1f}")
    ctx.save(res)
    return res


# ---------------- Modal wiring ----------------
if modal is not None:
    app = modal.App("brl11-stage600b")
    kimi = modal.Volume.from_name("kimi-k26", create_if_missing=False)
    hfvol = modal.Volume.from_name("hf-cache", create_if_missing=True)
    outvol = modal.Volume.from_name("k26-draft-out", create_if_missing=True)

    image = rr.vllm_image.add_local_python_source("record_run", "stage600_a")

    @app.function(image=image, gpu="B200:8",
                  volumes={"/models": kimi, "/cache": hfvol, "/out": outvol},
                  timeout=100 * 60, region="us-east")
    def session(k2: int = 7, ngram_probe: bool = True,
                ref_t1: float = 0.0, ref_step5: float = 0.0):
        import time
        import urllib.request

        env = os.environ.copy()
        env["TRTLLM_ENABLE_PDL"] = "1"

        zero_init_state = rr.apply_zero_init_patch(log=print)
        assert zero_init_state in ("present", "patched"), "NVFP4 zero-init unresolved"
        assert os.path.exists("/cache/kimi_chat_template.jinja"), "chat template missing"
        prompts, sha, real_tok = rr.canonical_prompts(strict=True)
        assert real_tok, "stage600b requires the real Kimi tokenizer"
        manifest = json.load(open("/out/brl11_p0_manifest.json"))
        assert manifest.get("prompt_sha256_10k") == sha == rr.PROMPT_SHA_10K, "prompt sha mismatch"
        if ref_t1:
            REF_4X["t1_ms"] = ref_t1
        if ref_step5:
            REF_4X["step_k5_ms"] = ref_step5

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
                return (time.time() - self.t0) / 3600 * HOURLY

            def log(self, msg):
                print(f"[stage600b|${self.spent():.1f}] {msg}", flush=True)

            def save(self, res):
                with open("/out/brl11_stage600b.json", "w") as f:
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
                cmd = build_cmd_b(cfg)
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

        res = run_stage600b(Ctx(), k2=k2, ngram_probe=ngram_probe)
        print("==== STAGE600-B DONE ====", flush=True)
        print(json.dumps({"gates": res.get("gates"), "e1_tp8": res.get("e1_tp8"),
                          "ship": res.get("ship"), "decisions": res.get("decisions")},
                         indent=1), flush=True)
        return {"gates": res.get("gates"), "ship": res.get("ship")}

    @app.local_entrypoint()
    def main(k2: int = 7, ngram_probe: bool = True, ref_t1: float = 0.0, ref_step5: float = 0.0):
        print("FINAL:", json.dumps(session.remote(k2=k2, ngram_probe=ngram_probe,
                                                  ref_t1=ref_t1, ref_step5=ref_step5), indent=1))


# ---------------- selftest ----------------
def _selftest():
    os.environ.setdefault("BRL_ALLOW_EST", "1")

    print("[1] TP8 cmdlines:", end=" ")
    c = build_cmd_b(cfg_b("eagle3", 5, rr.EAGLE, argmax=True))
    s = " ".join(c)
    assert "--tensor-parallel-size 8" in s and '"num_speculative_tokens": 5' in s
    assert '"use_local_argmax_reduction": true' in s and "--kv-cache-dtype fp8_e4m3" in s
    base4 = " ".join(sa.build_cmd_a(sa.cfg_a("eagle3", 5, rr.EAGLE, argmax=True)))
    assert s.replace("--tensor-parallel-size 8", "--tensor-parallel-size 4") == base4, \
        "TP8 cmd must differ from proven TP4 cmd ONLY in tensor-parallel-size"
    ng = " ".join(build_cmd_b(cfg_b("ngram", 5)))
    assert "--tensor-parallel-size 8" in ng and '"method": "ngram"' in ng
    ns = " ".join(build_cmd_b(cfg_b("none")))
    assert "--tensor-parallel-size 8" in ns and "--speculative-config" not in ns
    print("OK")

    print("[2] gates:", end=" ")
    g = gate_block(4.5, 9.8, 11.9, 7, 4.9, 6.1)
    assert g["fund_c"] and not g["comm_bound"]
    assert abs(g["pred_tok_s_k5"] - 500.0) < 1 and abs(g["pred_tok_s_kx"] - 512.6) < 1, g
    g2 = gate_block(5.9, 12.4, None, 7, 4.9, None)
    assert not g2["fund_c"] and g2["comm_bound"]
    g3 = gate_block(None, None, None, 7, None, None)
    assert not g3["fund_c"] and not g3["comm_bound"]
    print("OK", g["pred_tok_s_k5"], g["pred_tok_s_kx"])

    print("[3] campaign dry-run:", end=" ")
    prompts, sha, _ = rr.canonical_prompts(strict=False)

    PROFILE = {"B0_nospec_tp8": (205.0, None, None), "B1_k5_tp8": (480.0, 4.9, None),
               "B2_k7_tp8": (500.0, 6.0, None), "B3_ngram_k5_tp8": (260.0, 1.9, 0.45)}

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
            self.fake_spent += 0.03
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
                k = 5 if "k5" in self.tag else 7
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
            build_cmd_b(cfg)
            self.tag = tag
            self.boots.append(tag)

        def stop(self):
            pass

    ctx = MockCtx()
    res = run_stage600b(ctx, k2=7)
    assert ctx.boots[0] == "B0_nospec_tp8" and "B1_k5_tp8" in ctx.boots and "B2_k7_tp8" in ctx.boots
    assert res["gates"]["t1_tp8_ms"] and res["gates"]["pred_tok_s_k5"]
    assert res["arms"]["B1_k5_tp8"]["cells"]["tool_nothink_10k"]["n"] == 16, "480 f8 must bank"
    assert res.get("e1_tp8") and res["e1_tp8"]["v6_tp8_ms"]
    print("OK gates:", {k: v for k, v in res["gates"].items() if not isinstance(v, dict)})

    print("[4] dry-run B0 failure aborts:", end=" ")
    ctx2 = MockCtx(fail_tags=("B0",))
    res2 = run_stage600b(ctx2, k2=7)
    assert ctx2.boots == [] and "aborted" in " ".join(res2["decisions"])
    print("OK")

    print("SELFTEST PASS")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    elif modal is None:
        print("modal not installed; --selftest available")
