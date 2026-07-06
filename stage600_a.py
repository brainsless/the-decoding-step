"""Stage A: speculative-step cost attribution and EAGLE k-ladder for Kimi-K2.6
on 4xB200 (vLLM), one session with warm engine reloads, tool-nothink 10k protocol.

Measures:
  E1  Attribution of the EAGLE step's marginal cost (~12.0ms total, ~6.1ms over the
      bare forward): is the margin the drafter's k sequential passes, or the MoE
      expert-union verify width? Method: serve a weights-free ngram drafter at k=5
      and k=8. Drafting costs ~0 GPU time, so the measured per-drafted-step time is
      the pure verify cost V(width). Decomposition:
        draft_pass_ms   = (step_eagle_k5 - V6) / 5
        verify_slope_ms = (V9 - V6) / 3
      cross-checked against the within-session EAGLE step(k) fit from E2.
  E2  EAGLE k-ladder at 10k context: k=6/7/8 vs a same-session k=5 anchor. Fits the
      per-position step slope within one session; any arm whose first-8 median
      clears the bank floor is completed to n=16 as a record-eligible cell.

Cells: A0 anchor (EAGLE k=5, health + tool_nothink_10k), A1 no-spec T1,
A2/A3 ngram k=5/k=8, A4-A6 EAGLE k=6/7/8.

Artifact: /out/brl11_stage600a.json on volume k26-draft-out (saved after every cell).

Usage:
  python stage600_a.py --selftest
  modal run --detach stage600_a.py
  modal app stop brl11-stage600a --yes
"""
import json
import os
import statistics
import sys

try:
    import modal
except ImportError:
    modal = None

import record_run as rr

# ---------------- pre-registered constants ----------------
HARD_CAP = 60.0          # this session's spend ceiling at $24/hr
BAR_CRUSOE = 438.1       # artificialanalysis.ai 72h median, pinned 2026-07-05 (peak 449)
PRIOR_RECORD = 398.5     # brl11_record C2 tool_nothink_10k n=16
ANCHOR_F8_RECORD = 390.6  # record-day C2 first-8 (node-quality reference)
BANK_FLOOR = 400.0       # complete a ladder arm to n=16 only if f8 >= this AND beats anchor
SLOW_NODE_F8 = 372.0     # anchor f8 below this = slow node; ladder still informative,
                         # but skip n=16 completions (not record-eligible)
EST_ARM = 6.5            # warm reload + one 8-slot cell, $ estimate
T1_FALLBACK = (6.6, 7.6)  # no-spec step bounds at 10k if A1 fails (cross-session: 6.58
                          # short-ctx measured + ~1.0-1.3ms/step context tax at 10k)
E1_DRAFT_LOOP_MS = 8.5   # V6 at or under this: margin attributed to the drafter loop
E1_VERIFY_BYTES_MS = 11.0  # V6 at or over this: margin attributed to verify width
NGRAM_LOOKUP = {"prompt_lookup_max": 4, "prompt_lookup_min": 2}  # verified in the P0 manifest (cheap_proofs.py)


# ---------------- config + cmdline (extends record_run's proven surface) ----------------
def cfg_a(method, k=None, head=None, argmax=False):
    # Gotcha: CPU "ngram" fails VllmConfig validation with --async-scheduling on
    # vLLM v0.24.0 (allowed set is EAGLE/MTP/draft_model/ngram_gpu; see
    # vllm/config/vllm.py:940-959 at the v0.24.0 tag). "ngram_gpu" is the working
    # weights-free drafter: same prompt_lookup_max/min and num_speculative_tokens
    # semantics (speculative.py:633 shares one branch), composes with async
    # scheduling, TP, fp8 KV, MLA, FULL_AND_PIECEWISE. Its torch.compile cache is
    # force-disabled upstream -> slower boots, no correctness issue.
    assert method in ("eagle3", "ngram", "ngram_gpu", "none")
    return {"method": method, "k": k, "head": head, "argmax": bool(argmax),
            "kv": "fp8_e4m3", "ml": 16384}


def build_cmd_a(cfg):
    """vllm serve command. The eagle3 path delegates to record_run.build_cmd;
    ngram and none are assembled from the same verified flag set."""
    if cfg["method"] == "eagle3":
        return rr.build_cmd(rr.cfg_dict(cfg["head"], cfg["k"], kv=cfg["kv"],
                                        argmax=cfg["argmax"], ml=cfg["ml"]))
    cmd = ["vllm", "serve", rr.NVFP4,
           "--tensor-parallel-size", "4", "--gpu-memory-utilization", "0.90",
           "--quantization", "modelopt_fp4", "--max-model-len", str(cfg["ml"]),
           "--kv-cache-dtype", cfg["kv"]]
    if cfg["method"] in ("ngram", "ngram_gpu"):
        spec = {"method": cfg["method"], "num_speculative_tokens": cfg["k"], **NGRAM_LOOKUP}
        caps = rr.caps_for_k(cfg["k"])
        cmd += ["--speculative-config", json.dumps(spec)]
    else:  # none: no speculative config at all; proven capture list from k=3 sessions
        caps = rr.caps_for_k(3)
    cmd += ["--compilation-config", json.dumps({"cudagraph_mode": "FULL_AND_PIECEWISE",
                                                "cudagraph_capture_sizes": caps}),
            "--attention-config", json.dumps({"disable_flashinfer_q_quantization": True}),
            "--chat-template", "/cache/kimi_chat_template.jinja",
            "--limit-mm-per-prompt", '{"image":0,"video":0}',
            "--async-scheduling", "--trust-remote-code", "--port", "8000"]
    for tok in cmd:
        if tok.startswith("--"):
            assert tok in rr.ALLOWED_FLAGS, f"unverified flag {tok}"
    return cmd


# ---------------- metrics (adds draft-token counter to the proven parser) ----------------
def parse_metrics_a(txt):
    out = rr.parse_metrics_text(txt)
    import re
    out["dt"] = sum(float(m.group(1)) for m in re.finditer(
        r'^vllm:spec_decode_num_draft_tokens(?:_total)?(?:{[^}]*})?\s+([0-9.eE+-]+)\s*$',
        txt, re.M))
    return out


def measure_rows_a(ctx, arm, name, dom, max_tokens, thinking, lo, hi):
    """record_run.measure_rows plus the draft-token delta (dtok) per row."""
    rows = []
    consec_err = 0
    for slot in range(lo, hi):
        if ctx.spent() > HARD_CAP - 1.5:
            ctx.log(f"SPEND GUARD inside {arm}/{name} at slot {slot}")
            break
        p = ctx.prompts[dom][slot]
        b = ctx.counters()
        try:
            row = ctx.schat(p, max_tokens, thinking, seed=rr.SEED_BASE + slot)
            consec_err = 0
        except Exception as e:  # noqa: BLE001
            row = {"error": repr(e)[:120]}
            consec_err += 1
        a = ctx.counters()
        row["slot"] = slot
        row["is_10k"] = dom in ("math", "tool")
        if b and a and a["d"] > b["d"]:
            row["dr"] = a["d"] - b["d"]
            row["ac"] = a["a"] - b["a"]
            row["dtok"] = round(a.get("dt", 0.0) - b.get("dt", 0.0), 1)
            row["pp_raw"] = {str(i): a["pp"].get(i, 0.0) - b["pp"].get(i, 0.0)
                             for i in sorted(a.get("pp", {}))}
            row["tau"] = round(1 + row["ac"] / row["dr"], 4)
        rows.append(row)
        if consec_err >= 3:
            ctx.log(f"ABORT CELL {arm}/{name}: 3 consecutive request errors")
            break
    return rows


def run_cell_a(ctx, res, arm, name, dom, max_tokens, thinking, lo, hi):
    if ctx.spent() > HARD_CAP - 2.0:
        ctx.log(f"SPEND GUARD: skipping {arm}/{name}")
        return None
    new_rows = measure_rows_a(ctx, arm, name, dom, max_tokens, thinking, lo, hi)
    cells = res["arms"].setdefault(arm, {}).setdefault("cells", {})
    prev = cells.get(name, {}).get("rows", [])
    rows = prev + new_rows
    out = rr.cell_stats(rows, power=None)
    drafts = sum(r.get("dr") or 0 for r in rows)
    dtoks = sum(r.get("dtok") or 0 for r in rows)
    out["mean_proposed"] = round(dtoks / drafts, 2) if drafts and dtoks else None
    cells[name] = out
    ctx.log(f"{arm} {name}[{lo}:{hi}]: med={out['tok_s_median']} f8={out['tok_s_first8_median']} "
            f"agg={out['tok_s_agg']} tau={out['tau']} w={out['mean_proposed']} "
            f"ctok_med={out['ctok_median']} n={out['n']} ${ctx.spent():.1f}")
    ctx.save(res)
    return out


def run_arm_boot_a(ctx, res, tag, cfg, boot_budget_min):
    if ctx.spent() > HARD_CAP - 5.0:
        res["arms"][tag] = {"skipped": f"spend guard at ${ctx.spent():.1f}"}
        ctx.save(res)
        return False
    try:
        ctx.start(tag, cfg, boot_budget_min)
        res["arms"].setdefault(tag, {})["cfg"] = cfg
        res["arms"][tag]["cmd"] = " ".join(build_cmd_a(cfg))
        ctx.schat("Reply with exactly: WARMUP OK", 48, True, seed=rr.SEED_BASE, keep_text=False)
        ctx.schat("Reply with exactly: WARMUP OK", 48, False, seed=rr.SEED_BASE, keep_text=False)
        ctx.schat(ctx.warmup_10k, 128, False, seed=rr.SEED_BASE, keep_text=False)
        ctx.save(res)
        return True
    except Exception as e:  # noqa: BLE001
        res["arms"].setdefault(tag, {})["error"] = repr(e)[:300]
        ctx.log(f"ARM {tag} FAILED TO BOOT: {repr(e)[:200]}")
        ctx.save(res)
        ctx.stop()
        return False


# ---------------- E1 attribution math (pure; selftested) ----------------
def step_ms_cell(cell):
    """Mean full step time of a spec cell: tau / agg rate (time-weighted)."""
    if not cell or not cell.get("tau") or not cell.get("tok_s_agg"):
        return None
    return round(1000.0 * cell["tau"] / cell["tok_s_agg"], 2)


def verify_cost_ms(rows, t1_ms):
    """Per-drafted-step time for a weights-free drafter: strip non-drafted steps at the
    measured single-token step cost, divide the rest by drafted-step count."""
    vals = []
    for r in rows:
        if r.get("tok_s") and r.get("ctok") and r.get("dr"):
            t_ms = (r["ctok"] - 1) / r["tok_s"] * 1000.0
            nd = max(0.0, (r["ctok"] - 1) - (r.get("ac") or 0) - r["dr"])
            v = (t_ms - nd * t1_ms) / r["dr"]
            if v > 0:
                vals.append(v)
    return round(statistics.median(vals), 2) if vals else None


def hit_rate(rows):
    dr = sum(r.get("dr") or 0 for r in rows)
    nd = sum(max(0.0, (r["ctok"] - 1) - (r.get("ac") or 0) - (r.get("dr") or 0))
             for r in rows if r.get("ctok"))
    return round(dr / (dr + nd), 3) if dr + nd else None


def e1_block(t1_ms, t1_source, ng5_cell, ng8_cell, eagle_step_k5):
    """Assemble the pre-registered E1 decision block."""
    out = {"t1_ms": t1_ms, "t1_source": t1_source,
           "thresholds": {"draft_loop_max": E1_DRAFT_LOOP_MS,
                          "verify_bytes_min": E1_VERIFY_BYTES_MS},
           "eagle_step_k5_ms": eagle_step_k5}
    v6 = v9 = None
    if ng5_cell and ng5_cell.get("rows"):
        v6 = verify_cost_ms(ng5_cell["rows"], t1_ms)
        out["v6_ms"] = v6
        out["v6_hit_rate"] = hit_rate(ng5_cell["rows"])
        out["v6_mean_proposed"] = ng5_cell.get("mean_proposed")
        out["v6_sensitivity"] = {f"t1={t1_ms + d:+.1f}": verify_cost_ms(ng5_cell["rows"], t1_ms + d)
                                 for d in (-0.5, 0.5)}
    if ng8_cell and ng8_cell.get("rows"):
        v9 = verify_cost_ms(ng8_cell["rows"], t1_ms)
        out["v9_ms"] = v9
        out["v9_hit_rate"] = hit_rate(ng8_cell["rows"])
        out["v9_mean_proposed"] = ng8_cell.get("mean_proposed")
    if v6 is None:
        out["verdict"] = "blocked"
    elif v6 <= E1_DRAFT_LOOP_MS:
        out["verdict"] = "draft-loop"
    elif v6 >= E1_VERIFY_BYTES_MS:
        out["verdict"] = "verify-bytes"
    else:
        out["verdict"] = "mixed"
    if v6 and eagle_step_k5:
        out["draft_pass_ms"] = round((eagle_step_k5 - v6) / 5.0, 3)
    if v6 and v9:
        out["verify_slope_ms_per_tok"] = round((v9 - v6) / 3.0, 3)
    return out


def fit_step_curve(points):
    """OLS fit step_ms = a + b*k over the within-session EAGLE ladder."""
    pts = [(k, s) for k, s in points if s]
    if len(pts) < 2:
        return None
    n = len(pts)
    sx = sum(k for k, _ in pts)
    sy = sum(s for _, s in pts)
    sxx = sum(k * k for k, _ in pts)
    sxy = sum(k * s for k, s in pts)
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    a = (sy - b * sx) / n
    return {"intercept_ms": round(a, 3), "slope_ms_per_k": round(b, 3),
            "points": {str(k): s for k, s in pts}}


# ---------------- the session ----------------
def run_stage600a(ctx):
    res = {"campaign": "stage600_a", "arms": {}, "gates": {}, "decisions": [],
           "hourly_usd": 24.0, "hard_cap_usd": HARD_CAP,
           "bars": {"crusoe_median_2026_07_05": BAR_CRUSOE, "crusoe_peak": 449.0,
                    "prior_record": PRIOR_RECORD},
           "protocol": "identical to brl11_record (10k docpacks, temp 0.6, streaming, "
                       "interp median, per-request seeds); probe cells slots 0-8",
           "prompt_sha256_10k": ctx.prompt_sha,
           "zero_init": ctx.zero_init_state,
           "p0_manifest": ctx.p0_manifest_summary}
    gates = res["gates"]

    def note(msg):
        res["decisions"].append(msg)
        ctx.log("DECISION: " + msg)

    # ---- A0: same-session EAGLE k=5 anchor (the record config) + node health ----
    a0 = cfg_a("eagle3", 5, rr.EAGLE, argmax=True)
    if not run_arm_boot_a(ctx, res, "A0_anchor_k5", a0, boot_budget_min=35):
        note("A0 anchor failed to boot; session aborted")
        ctx.save(res)
        return res
    health = run_cell_a(ctx, res, "A0_anchor_k5", "tool_short8_health", "tool_short", 1200, False, 0, 8)
    h8 = (health or {}).get("tok_s_median")
    h_tau = (health or {}).get("tau")
    if h8 is not None and (h8 < 250 or (h_tau is not None and h_tau < 2.5)):
        note(f"HEALTH ABORT: {h8} tok/s @ tau {h_tau} (floors 250 / tau 2.5)")
        ctx.stop()
        ctx.save(res)
        return res
    anchor = run_cell_a(ctx, res, "A0_anchor_k5", "tool_nothink_10k", "tool", 2048, False, 0, 8)
    anchor_f8 = (anchor or {}).get("tok_s_first8_median")
    eagle_step_k5 = step_ms_cell(anchor)
    slow_node = bool(anchor_f8 and anchor_f8 < SLOW_NODE_F8)
    note(f"anchor k5 f8={anchor_f8} step={eagle_step_k5}ms (record-day f8 {ANCHOR_F8_RECORD}); "
         f"slow_node={slow_node}")
    ctx.stop()

    # ---- A1: no-spec T1 at 10k (4 slots; the E1 denominator) ----
    t1_ms, t1_source = None, "measured"
    if run_arm_boot_a(ctx, res, "A1_nospec", cfg_a("none"), boot_budget_min=22):
        run_cell_a(ctx, res, "A1_nospec", "tool_nothink_10k", "tool", 2048, False, 0, 4)
        c = res["arms"]["A1_nospec"]["cells"].get("tool_nothink_10k")
        if c and c.get("tok_s_agg"):
            t1_ms = round(1000.0 / c["tok_s_agg"], 2)
        ctx.stop()
    if t1_ms is None:
        t1_ms, t1_source = sum(T1_FALLBACK) / 2, f"fallback {T1_FALLBACK} (A1 failed)"
    note(f"T1 at 10k = {t1_ms}ms ({t1_source})")

    # ---- A2/A3: ngram k=5 and k=8 (E1 core; boot failure tolerated once each) ----
    ng5_cell = ng8_cell = None
    if run_arm_boot_a(ctx, res, "A2_ngram_k5", cfg_a("ngram", 5), boot_budget_min=22):
        ng5_cell = run_cell_a(ctx, res, "A2_ngram_k5", "tool_nothink_10k", "tool", 2048, False, 0, 8)
        ctx.stop()
    if ng5_cell and ctx.spent() < HARD_CAP - 3 * EST_ARM:
        if run_arm_boot_a(ctx, res, "A3_ngram_k8", cfg_a("ngram", 8), boot_budget_min=22):
            ng8_cell = run_cell_a(ctx, res, "A3_ngram_k8", "tool_nothink_10k", "tool", 2048, False, 0, 8)
            ctx.stop()
    elif not ng5_cell:
        note("A2 ngram failed/blocked: skipping A3 (same failure class); E1 rides the ladder fit only")

    res["e1"] = e1_block(t1_ms, t1_source, ng5_cell, ng8_cell, eagle_step_k5)
    note(f"E1: verdict={res['e1'].get('verdict')} v6={res['e1'].get('v6_ms')} "
         f"v9={res['e1'].get('v9_ms')} draft_pass={res['e1'].get('draft_pass_ms')} "
         f"verify_slope={res['e1'].get('verify_slope_ms_per_tok')}")
    ctx.save(res)

    # ---- A4-A6: EAGLE k ladder (E2); bank any arm that clears the floor ----
    ladder_steps = [(5, eagle_step_k5)]
    ladder_f8 = {5: anchor_f8}
    for k in (6, 7, 8):
        if ctx.spent() > HARD_CAP - EST_ARM - 2.0:
            note(f"k={k} skipped by spend guard")
            break
        if k == 8 and ladder_f8.get(7) and ladder_f8.get(6) and \
                ladder_f8[7] < ladder_f8[6] * 0.995:
            note("k=8 skipped: k=7 fell below k=6 (ladder collapsed)")
            break
        tag = f"A{k - 2}_k{k}"
        if not run_arm_boot_a(ctx, res, tag, cfg_a("eagle3", k, rr.EAGLE, argmax=True), 22):
            continue
        cell = run_cell_a(ctx, res, tag, "tool_nothink_10k", "tool", 2048, False, 0, 8)
        f8 = (cell or {}).get("tok_s_first8_median")
        ladder_f8[k] = f8
        ladder_steps.append((k, step_ms_cell(cell)))
        gates[f"G_k{k}"] = {"f8": f8, "anchor_f8": anchor_f8,
                            "bank": bool(f8 and anchor_f8 and not slow_node
                                         and f8 >= BANK_FLOOR and f8 >= anchor_f8 + 4)}
        if gates[f"G_k{k}"]["bank"] and ctx.spent() < HARD_CAP - 5.0:
            note(f"k={k} banks: f8 {f8} >= {BANK_FLOOR}; completing to n=16")
            run_cell_a(ctx, res, tag, "tool_nothink_10k", "tool", 2048, False, 8, 16)
        ctx.stop()

    res["step_fit"] = fit_step_curve(ladder_steps)
    note(f"within-session step fit: {res['step_fit']}")

    # ---- ship block ----
    best = None
    for tag, arm in res["arms"].items():
        c = (arm.get("cells") or {}).get("tool_nothink_10k")
        if c and c.get("n", 0) >= 16 and c.get("tok_s_median"):
            if best is None or c["tok_s_median"] > best["tok_s"]:
                best = {"tok_s": c["tok_s_median"], "arm": tag, "tau": c.get("tau"),
                        "aa_len_ok": c.get("aa_len_ok")}
    res["ship"] = {"best_n16_cell": best,
                   "vs": {"prior_record": PRIOR_RECORD, "crusoe": BAR_CRUSOE},
                   "note": "stage-A is an information session; records bank opportunistically"}
    res["total_spent_usd_est"] = round(ctx.spent(), 2)
    note(f"DONE: e1={res['e1'].get('verdict')} best_n16={best} ${ctx.spent():.1f}")
    ctx.save(res)
    return res


# ---------------- Modal wiring ----------------
if modal is not None:
    app = modal.App("brl11-stage600a")
    kimi = modal.Volume.from_name("kimi-k26", create_if_missing=False)
    hfvol = modal.Volume.from_name("hf-cache", create_if_missing=True)
    outvol = modal.Volume.from_name("k26-draft-out", create_if_missing=True)

    image = rr.vllm_image.add_local_python_source("record_run")

    @app.function(image=image, gpu="B200:4",
                  volumes={"/models": kimi, "/cache": hfvol, "/out": outvol},
                  timeout=150 * 60, region="us-east")
    def session():
        import subprocess
        import threading
        import time
        import urllib.request

        env = os.environ.copy()
        env["TRTLLM_ENABLE_PDL"] = "1"

        zero_init_state = rr.apply_zero_init_patch(log=print)
        assert zero_init_state in ("present", "patched"), \
            "NVFP4 zero-init unresolved; refusing to serve (PR #45739)"
        assert os.path.exists("/cache/kimi_chat_template.jinja"), "chat template missing (gotcha 1)"
        prompts, sha, real_tok = rr.canonical_prompts(strict=True)
        assert real_tok, "stage600a requires the real Kimi tokenizer"
        manifest_path = "/out/brl11_p0_manifest.json"
        assert os.path.exists(manifest_path), "P0 manifest missing: run cheap_proofs.py first"
        manifest = json.load(open(manifest_path))
        assert manifest.get("prompt_sha256_10k") == sha == rr.PROMPT_SHA_10K, "prompt sha mismatch"
        assert manifest.get("spec_ngram_fields"), "P0 did not verify ngram fields; refusing E1"

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
                                            ("vllm_version", "vllm_sha", "flashinfer",
                                             "zero_init_fix_in_source", "spec_ngram_fields")}
                count_fn = rr.load_kimi_count_fn("/root/assets")
                self.warmup_10k = rr.build_docpack("math", 90, count_fn)

            def spent(self):
                return (time.time() - self.t0) / 3600 * 24.0

            def log(self, msg):
                print(f"[stage600a|${self.spent():.1f}] {msg}", flush=True)

            def save(self, res):
                with open("/out/brl11_stage600a.json", "w") as f:
                    json.dump(res, f, indent=1)
                outvol.commit()

            def schat(self, prompt, max_tokens, thinking=True, seed=None, keep_text=True):
                return rr.schat(self.base_url, self.model, prompt, max_tokens, thinking,
                                seed, keep_text)

            def counters(self):
                try:
                    txt = urllib.request.urlopen("http://127.0.0.1:8000/metrics",
                                                 timeout=15).read().decode()
                    return parse_metrics_a(txt)
                except Exception:
                    return None

            def start(self, tag, cfg, boot_budget_min=22):
                self.stop()
                cmd = build_cmd_a(cfg)
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

        res = run_stage600a(Ctx())
        print("==== STAGE600-A DONE ====", flush=True)
        print(json.dumps({"e1": res.get("e1"), "step_fit": res.get("step_fit"),
                          "gates": res.get("gates"), "ship": res.get("ship"),
                          "decisions": res.get("decisions")}, indent=1), flush=True)
        return {"e1": res.get("e1"), "step_fit": res.get("step_fit"), "ship": res.get("ship")}

    @app.local_entrypoint()
    def main():
        print("FINAL:", json.dumps(session.remote(), indent=1))


# ---------------- selftest ----------------
def _selftest():
    os.environ.setdefault("BRL_ALLOW_EST", "1")

    print("[1] cmdlines:", end=" ")
    e6 = " ".join(build_cmd_a(cfg_a("eagle3", 6, rr.EAGLE, argmax=True)))
    assert '"num_speculative_tokens": 6' in e6 and '"use_local_argmax_reduction": true' in e6
    assert '"cudagraph_capture_sizes": [1, 2, 3, 4, 5, 7, 14, 21, 28, 56, 112, 224]' in e6
    e8 = " ".join(build_cmd_a(cfg_a("eagle3", 8, rr.EAGLE, argmax=True)))
    assert '"num_speculative_tokens": 8' in e8 and "[1, 2, 3, 4, 5, 9, 18, 27, 36, 72, 144, 288]" in e8
    ng = " ".join(build_cmd_a(cfg_a("ngram", 5)))
    assert '"method": "ngram"' in ng and '"prompt_lookup_max": 4' in ng \
        and '"prompt_lookup_min": 2' in ng and '"model"' not in ng \
        and "use_local_argmax_reduction" not in ng
    assert '"cudagraph_capture_sizes": [1, 2, 3, 4, 5, 6, 12, 18, 24, 48, 96, 192]' in ng
    ns = " ".join(build_cmd_a(cfg_a("none")))
    assert "--speculative-config" not in ns and "--kv-cache-dtype fp8_e4m3" in ns
    assert "--chat-template /cache/kimi_chat_template.jinja" in ns
    print("OK")

    print("[2] metrics parser with draft tokens:", end=" ")
    txt = ('vllm:spec_decode_num_drafts_total{engine="0"} 100\n'
           'vllm:spec_decode_num_accepted_tokens_total{engine="0"} 250\n'
           'vllm:spec_decode_num_draft_tokens_total{engine="0"} 480\n'
           'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="0"} 80\n')
    m = parse_metrics_a(txt)
    assert m["d"] == 100 and m["a"] == 250 and m["dt"] == 480 and m["pp"][0] == 80
    print("OK")

    print("[3] E1 math:", end=" ")
    # synthetic: 2048 tok, hit-rate ~0.5: 400 drafted steps (tau-ish 2.5 accepted+1 each
    # = 1400 tok) + 647 plain steps = 2047 timed tokens. verify 9.0ms, T1 7.0ms
    # -> time = 400*9.0 + 647*7.0 = 8129ms -> tok_s = 2047/8.129 = 251.8
    rows = [{"tok_s": 2047 / 8.129, "ctok": 2048, "dr": 400, "ac": 1000, "dtok": 2000}]
    v = verify_cost_ms(rows, 7.0)
    assert abs(v - 9.0) < 0.05, v
    assert abs(hit_rate(rows) - 400 / 1047) < 0.01
    blk = e1_block(7.0, "measured", {"rows": rows, "mean_proposed": 5.0}, None, 12.0)
    assert blk["verdict"] == "mixed" and abs(blk["draft_pass_ms"] - 0.6) < 0.02, blk
    rows_cheap = [{"tok_s": 2047 / (400 * 7.4 + 647 * 7.0) * 1000, "ctok": 2048,
                   "dr": 400, "ac": 1000, "dtok": 2000}]
    blk2 = e1_block(7.0, "measured", {"rows": rows_cheap, "mean_proposed": 5.0},
                    {"rows": [{"tok_s": 2047 / (300 * 8.6 + 947 * 7.0) * 1000, "ctok": 2048,
                               "dr": 300, "ac": 800, "dtok": 2400}], "mean_proposed": 8.0}, 12.0)
    assert blk2["verdict"] == "draft-loop" and abs(blk2["v6_ms"] - 7.4) < 0.1
    assert abs(blk2["verify_slope_ms_per_tok"] - 0.4) < 0.05, blk2
    exp = [{"tok_s": 2047 / (400 * 11.8 + 647 * 7.0) * 1000, "ctok": 2048,
            "dr": 400, "ac": 1000, "dtok": 2000}]
    assert e1_block(7.0, "measured", {"rows": exp}, None, 12.0)["verdict"] == "verify-bytes"
    assert e1_block(7.0, "fallback", None, None, 12.0)["verdict"] == "blocked"
    print("OK v6", v)

    print("[4] step fit:", end=" ")
    fit = fit_step_curve([(5, 12.25), (6, 13.4), (7, 14.7), (8, 15.9)])
    assert abs(fit["slope_ms_per_k"] - 1.223) < 0.02 and abs(fit["intercept_ms"] - 6.11) < 0.15, fit
    assert fit_step_curve([(5, 12.0), (6, None)]) is None
    print("OK", fit)

    print("[5] campaign dry-run:", end=" ")
    prompts, sha, _ = rr.canonical_prompts(strict=False)

    ARM_PROFILE = {  # tok_s, tau, hit (None hit = eagle: every step drafted)
        "A0_anchor_k5": (396.0, 4.95, None), "A1_nospec": (139.0, None, None),
        "A2_ngram_k5": (215.0, 1.9, 0.45), "A3_ngram_k8": (205.0, 2.1, 0.35),
        "A4_k6": (408.0, 5.5, None), "A5_k7": (413.0, 6.1, None), "A6_k8": (403.0, 6.6, None),
    }

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
            tok_s, tau, hit = ARM_PROFILE[self.tag]
            if max_tokens < 200:  # warmup / health short calls
                return {"ctok": max_tokens, "ttft_ms": 30.0, "tok_s": tok_s}
            ctok = max_tokens
            if tau is not None:
                k = {"A2_ngram_k5": 5, "A3_ngram_k8": 8}.get(self.tag) or int(self.tag[-1])
                h = 1.0 if hit is None else hit
                # drafted steps produce tau tokens, plain steps 1
                dr = (ctok - 1) * h / (tau * h + (1 - h))
                self._c["d"] += dr
                self._c["a"] += dr * (tau - 1)
                self._c["dt"] += dr * k
                for i in range(min(k, 3)):
                    self._c["pp"][i] = self._c["pp"].get(i, 0.0) + dr * (0.9 - 0.1 * i)
            return {"ctok": ctok, "ttft_ms": 300.0, "tok_s": tok_s}

        def counters(self):
            return {k: (dict(v) if isinstance(v, dict) else v) for k, v in self._c.items()}

        def start(self, tag, cfg, boot_budget_min=22):
            if any(t in tag for t in self.fail_tags):
                raise RuntimeError(f"simulated boot failure {tag}")
            build_cmd_a(cfg)
            self.tag = tag
            self.boots.append(tag)

        def stop(self):
            pass

    ctx = MockCtx()
    res = run_stage600a(ctx)
    assert ctx.boots[0] == "A0_anchor_k5" and "A2_ngram_k5" in ctx.boots
    assert res["e1"]["verdict"] in ("draft-loop", "mixed", "verify-bytes")
    assert res["e1"]["t1_ms"] and res["e1"]["v6_ms"]
    assert res["step_fit"] and len(res["step_fit"]["points"]) >= 3
    assert res["arms"]["A5_k7"]["cells"]["tool_nothink_10k"]["n"] >= 8
    banked = [g for g, v in res["gates"].items() if v.get("bank")]
    assert banked, "mock ladder should bank at least one arm"
    print("OK e1:", res["e1"]["verdict"], "banked:", banked)

    print("[6] dry-run with ngram boot failure (tolerated):", end=" ")
    ctx2 = MockCtx(fail_tags=("A2_ngram", "A3_ngram"))
    res2 = run_stage600a(ctx2)
    assert res2["e1"]["verdict"] == "blocked"
    assert "A4_k6" in ctx2.boots, "ladder must still run after E1 block"
    print("OK")

    print("[7] dry-run nospec failure -> T1 fallback:", end=" ")
    ctx3 = MockCtx(fail_tags=("A1_nospec",))
    res3 = run_stage600a(ctx3)
    assert "fallback" in res3["e1"]["t1_source"] and res3["e1"]["v6_ms"]
    print("OK")

    print("SELFTEST PASS")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    elif modal is None:
        print("modal not installed; --selftest available")
