"""Shared library and Modal entrypoint for the BRL-2026-11 single-stream serving
record sessions: Kimi-K2.6 (NVFP4) on 4x B200 under vLLM with an EAGLE3 draft head.

Protocol: 16 distinct ~10k-Kimi-token prompts per domain (math, tool), max_tokens 2048,
temperature 0.6, top_p 1.0, per-request seed, serial single-stream streaming requests;
per-request decode rate = (completion_tokens - 1) / (t_last - t_first); the headline
statistic is the interpolated median over n=16. Rates are in Kimi-native tokens; raw
output text is stored per row so o200k rates are computable offline (o200k/Kimi-native
parity measured 1.0035-1.0081, token-weighted).

The GPU session fails closed: it refuses to serve unless the prompt set rebuilt
in-container reproduces the pinned SHA-256, the P0 proof manifest (written by
cheap_proofs.py) agrees, and the NVFP4 zero-init state is resolved. Results are saved
to /out/brl11_baseline.json or /out/brl11_record.json (volume k26-draft-out) after
every cell.

Usage:
  python record_run.py --selftest                  CPU-only selftest (mock server, dry runs)
  python record_run.py --build-prompts             rebuild prompts, print the pinned shas
  modal run --detach record_run.py                 baseline session
  modal run --detach record_run.py --mode record   record session
  modal app stop brl11-record --yes           stop a detached run
"""
import hashlib
import json
import os
import random
import re
import statistics
import sys

try:
    import modal
except ImportError:  # selftest can run without modal installed
    modal = None

# ---------------- pre-registered constants ----------------
BAR_T1 = 346.0        # Fireworks, AA-measured, B200 platform
BAR_T2_BRIEF = 397.0  # earlier Crusoe AA median snapshot (stale; kept so tier labels stay comparable)
BAR_PIN = 412.0       # Crusoe AA 72h median (411.9 read 2026-07-03). Re-read from
                      # artificialanalysis.ai/models/kimi-k2-6/providers on the morning of a run.
GATE_KEEP = 1.015     # keep a lever iff the champion improves >= +1.5% (a false keep is cheap
                      # since the record re-verifies at n=16; a false drop discards a real gain)
GATE_ESCALATE_K5 = 1.03
FOVEA_TIE = 0.97      # record on Fovea if within 3% of champion (Tier-3 value)
HEALTH_FLOOR = 300.0  # reference session read 336.8 median on this cell; below 300 = sick node, abort
SANITY_FLOOR = 360.0  # first-8 math_nothink_10k below this: kept levers (~+10% combined central
                      # estimate, 360 * 1.10 < 397) cannot reach the bar; finish R0, stop reloads
AA_MIN_CTOK = 1500    # AA's ">= 1,500 answer tokens"; cells meeting this get headline priority
HARD_CAP = 30.0
RESERVE_THINK = 5.5   # protects R5 champion think-record cells
EST_NEXT_ARM = 5.0    # reload + probes + completion estimate for the optional-arm guard
SEED_BASE = 11711     # per-request seed = SEED_BASE + prompt slot
TARGET_TOK_LO, TARGET_TOK_HI = 9900, 10400   # Kimi-native tokens per 10k prompt

# Set from the cheap-GPU proof results; the only values edited between proofing and launch:
MNNVL_OK = False       # mnnvl allreduce proof inconclusive on the cheap-GPU twin (health
                       # timeout, likely cold JIT); kept off as the conservative choice
TOKENSPEED_OK = False  # TOKENSPEED_MLA engine failed at init (rc=1) on the SM100 proof twin
FOVEA_E = "/out/fovea_e_ckpt"  # trained draft head; held-out acc_len 2.586 (tau ~3.59 vs 2.87
                               # for the stock head); served only behind its gate, kept only if it wins

NVFP4 = "/models/Kimi-K2.6-NVFP4"
EAGLE = "lightseekorg/kimi-k2.6-eagle3.1-mla"

# Pinned by a local build with the real Kimi tokenizer (assets/kimi_tiktoken.model);
# P0 rebuilds in-container and must reproduce these exactly (manifest handshake).
PROMPT_SHA_10K = "be01fffdb4cf55d93e049edaf22c5099d29cace9bc5fea8e3df8dcb29848a6e8"
TOOL16_SHA = "9f739c57dfe9ab640216917a59697f3dc54cac2bfbc9167887bb8bf71108230a"

# ------------- health-cell short prompts (byte-identical to the reference session) -------------
TOOL16 = [
    'Return ONLY a JSON array of 30 objects {"id","step","owner","command","rollback"} describing a production Postgres 15 -> 16 migration runbook for a SaaS with 200GB of data and a 5-minute maintenance window.',
    'Output ONLY JSON: an array of 24 monitoring alerts {"name","metric","threshold","window","severity","runbook_url"} for a payments API running on Kubernetes.',
    'Return ONLY a JSON object mapping 20 employee onboarding tasks to {"task","owner_role","due_day","tooling","depends_on"} for a 15-person startup hiring its first support team.',
    'Output ONLY a JSON array of 25 test cases {"id","endpoint","method","payload","expected_status","expected_body_contains"} for a REST API that manages calendar bookings.',
    'Return ONLY JSON: 18 feature flags {"key","description","default","rollout_percent","owner","expiry_date"} for a mobile app moving from beta to GA.',
    'Output ONLY a JSON array of 22 steps {"step","tool","command","verify","on_failure"} to rotate every secret (DB, API keys, TLS) in a small production environment with zero downtime.',
    'Return ONLY JSON: an array of 20 CRM field definitions {"field","type","required","validation","example"} for tracking enterprise sales deals from lead to close.',
    'Output ONLY a JSON object with keys "critical","high","medium" each holding arrays of 8 security review findings {"title","component","impact","fix","effort_days"} for a typical Django monolith.',
]

# ---------------- deterministic 10k-token docpack builder ----------------
FIRST = ["Arc", "Nim", "Vel", "Kor", "Zet", "Pax", "Lum", "Ori", "Sol", "Rho"]
LAST = ["io", "ana", "entra", "ovia", "ix", "era", "on", "ave", "una", "ex"]
KIND = ["Systems", "Labs", "Metrics", "Cloud", "Works", "Data", "Grid", "Stack"]
TEAMS = ["payments", "identity", "search", "ingest", "billing", "notify", "ml-serving", "edge"]
SVC = ["api", "worker", "cache", "queue", "gateway", "batch", "stream", "cron", "indexer", "webhook"]
LEVERS = ["rightsizing", "spot migration", "compression", "tiered storage", "request coalescing",
          "autoscaling floor cut", "reserved instances", "cold-path archival"]
INC = ["elevated 5xx on {t} after a deploy", "queue depth runaway in {t} ingest",
       "certificate expiry on the {t} edge", "hot partition in the {t} store",
       "thread-pool exhaustion in {t} workers", "cache stampede on {t} reads",
       "cross-zone packet loss hitting {t}", "slow consumer stalling the {t} bus"]
VERBS = ["mitigated by", "resolved after", "closed following", "contained via"]
FIXES = ["a rollback", "a config revert", "manual failover", "rate-limit tightening",
         "an index rebuild", "connection-pool resize", "a hotfix deploy", "shard rebalancing"]
METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE"]
RES = ["invoices", "customers", "subscriptions", "webhooks", "reports", "tokens", "audits",
       "exports", "plans", "usage", "credits", "disputes", "payouts", "sessions", "keys",
       "orgs", "roles", "limits", "events", "batches"]
ACT = ["finalize", "retry", "archive", "verify", "rotate", "preview", "approve", "cancel"]


def _company(r):
    return f"{r.choice(FIRST)}{r.choice(LAST)} {r.choice(KIND)}"


def _month(i):
    y, m = 2024 + (6 + i) // 12, (6 + i) % 12 + 1
    return f"{y}-{m:02d}"


def _math_sections(idx):
    r = random.Random(31000 + idx)
    L = [f"OPERATIONS DATA PACK D{idx:02d} — {_company(r)} (B2B SaaS). Internal quarterly "
         f"review input compiled {_month(r.randint(20, 23))}. All figures monthly unless stated. "
         f"Sections: A revenue, B infrastructure, C support, D incidents, E pricing tests, "
         f"F general ledger extract.", ""]
    L.append("SECTION A — Revenue by month, 24 months:")
    mrr = r.randint(180, 420) * 1000
    cust = r.randint(700, 1600)
    for i in range(24):
        new = int(mrr * r.uniform(0.030, 0.080))
        exp = int(mrr * r.uniform(0.010, 0.050))
        churned = int(mrr * r.uniform(0.015, 0.035))
        mrr = mrr + new + exp - churned
        cust = cust + r.randint(8, 60) - r.randint(4, 30)
        L.append(f"{_month(i)}: MRR ${mrr:,}; new ${new:,}; expansion ${exp:,}; "
                 f"churned ${churned:,}; customers {cust:,}; support tickets {r.randint(220, 940)}")
    L.append("")
    L.append("SECTION B — Infrastructure cost lines (current month):")
    for i in range(56):
        t = r.choice(TEAMS)
        L.append(f"svc-{i:03d} {t}-{r.choice(SVC)}: ${r.randint(900, 24000):,}/mo; "
                 f"driver {round(r.uniform(0.2, 22.0), 1)}M requests/day; egress {round(r.uniform(0.1, 9.0), 1)}TB; "
                 f"owner {t}; savings candidate {r.randint(4, 38)}% via {r.choice(LEVERS)}")
    L.append("")
    L.append("SECTION C — Support metrics by week, 40 weeks:")
    for i in range(40):
        L.append(f"week {i + 1:02d}: opened {r.randint(180, 640)}, closed {r.randint(170, 630)}, "
                 f"median first response {round(r.uniform(0.4, 9.5), 1)}h, CSAT {round(r.uniform(3.6, 4.9), 2)}, "
                 f"escalations {r.randint(2, 31)}, agents on shift {r.randint(4, 12)}")
    L.append("")
    L.append("SECTION D — Incident log, most recent 26:")
    for i in range(26):
        t = r.choice(TEAMS)
        L.append(f"INC-{r.randint(4100, 4999)}: {r.choice(INC).format(t=t)}; duration {r.randint(9, 214)} min; "
                 f"user-visible {r.choice(['yes', 'no'])}; {r.choice(VERBS)} {r.choice(FIXES)}; "
                 f"estimated revenue at risk ${r.randint(1, 90) * 100:,}")
    L.append("")
    L.append("SECTION E — Pricing experiments, 12 completed:")
    for i in range(12):
        L.append(f"EXP-{i + 1:02d}: arm A {r.randint(20, 90)}$/seat vs arm B {r.randint(20, 90)}$/seat; "
                 f"users per arm {r.randint(800, 5200):,}; conversion A {round(r.uniform(1.4, 6.8), 2)}% "
                 f"B {round(r.uniform(1.4, 6.8), 2)}%; expansion after 60d A {round(r.uniform(0.5, 9.0), 1)}% "
                 f"B {round(r.uniform(0.5, 9.0), 1)}%")
    L.append("")
    L.append("SECTION F — General ledger extract (context only):")
    return L, r


def _math_filler(r, i):
    return (f"ledger {_month(r.randint(18, 23))}-{r.randint(1, 28):02d} acct-{r.randint(1000, 9899)} "
            f"{r.choice(TEAMS)} {r.choice(['saas tools', 'contractors', 'cloud egress', 'events', 'licenses', 'hardware', 'travel', 'recruiting'])} "
            f"${r.randint(120, 48000):,} tag:{r.choice(['opex', 'cogs', 'capex'])} approver {r.choice(FIRST).lower()}{i % 97:02d}")


MATH_QUESTION = """
QUESTIONS — using ONLY the data pack above, answer ALL 10 parts, in order, fully worked.
Show every arithmetic step and intermediate value; state each formula before using it.
Do not summarize away steps; do not skip any part.
(1) From Section A: total net-new MRR over the 24 months, and the average monthly compound
growth rate of MRR (show the ratio and the root).
(2) From Section A: gross MRR churn rate for the final month, and annualized.
(3) From Section B: total monthly infrastructure spend, the top 3 lines by cost with their
combined share, and the savings if every line achieved its stated savings-candidate percent.
(4) From Section B: cost per million requests per day for the 3 most expensive lines.
(5) From Section C: mean weekly opened and closed tickets and the net backlog change over
all 40 weeks; average tickets closed per agent-shift-week for the final 8 weeks.
(6) From Section D: total user-visible incident minutes and total estimated revenue at risk;
mean duration for user-visible vs not.
(7) From Section E: which experiment has the largest absolute conversion gap; compute both
arms' revenue per 1,000 users at the stated seat prices.
(8) Combine A and B: infrastructure cost as a percent of final-month MRR.
(9) Combine A and C: final-month support tickets per customer, and per $10k of MRR.
(10) Estimate months of runway if cash is 6.0x the final-month MRR and total monthly burn
is 1.9x the Section B infrastructure total plus $310,000 payroll; show the division.
""".strip()


def _tool_sections(idx):
    r = random.Random(47000 + idx)
    L = [f"PLATFORM API CATALOG P{idx:02d} — {_company(r)} internal services. Version "
         f"{r.randint(3, 9)}.{r.randint(0, 9)}. Auth: service tokens with scoped claims. "
         f"Global rate limit {r.randint(300, 1200)}/min per token unless a route overrides it.", ""]
    L.append("ENDPOINT CATALOG (40 routes):")
    for i in range(40):
        res_ = RES[i % len(RES)]
        act = r.choice(ACT)
        m = r.choice(METHODS)
        L.append(f"Endpoint {i + 1:02d}: {m} /v{r.randint(1, 3)}/{res_}/{{id}}/{act} — scope "
                 f"{res_}:{r.choice(['read', 'write', 'admin'])}; params: {r.choice(['none', 'idempotency-key header', 'cursor+limit', 'dry_run flag', 'as_of date'])}; "
                 f"rate limit {r.randint(30, 600)}/min; success {r.choice([200, 201, 202])}; "
                 f"errors {r.choice([400, 402, 404])},{r.choice([409, 410, 412])},{r.choice([422, 428, 429])}; "
                 f"p99 SLO {r.randint(120, 900)}ms; owner {r.choice(TEAMS)}; "
                 f"notes: {r.choice(['paginated', 'idempotent', 'async job returned', 'soft-deletes', 'emits audit event', 'requires 2FA context', 'cached 30s', 'beta'])}")
    L.append("")
    L.append("DEPLOYMENT RUNBOOK (24 steps):")
    for i in range(24):
        L.append(f"step {i + 1:02d}: {r.choice(['drain', 'snapshot', 'migrate', 'deploy', 'verify', 'warm', 'cutover', 'rollback-check'])} "
                 f"{r.choice(TEAMS)}-{r.choice(SVC)} via {r.choice(['helm', 'terraform', 'argo', 'ansible'])}; "
                 f"verify: {r.choice(['healthz 200', 'error rate < 0.1%', 'p99 within SLO', 'queue depth < 1k', 'replicas ready'])}; "
                 f"on failure: {r.choice(FIXES)}")
    L.append("")
    L.append("SHARED SCHEMAS (16):")
    for i in range(16):
        L.append(f"schema {RES[i % len(RES)]}: fields id uuid, created_at ts, status "
                 f"enum[{r.choice(['active,past_due,canceled', 'pending,done,failed', 'open,held,closed'])}], "
                 f"amount_cents int, currency iso4217, metadata map<str,str> max {r.randint(8, 64)} keys, "
                 f"version int monotonic")
    L.append("")
    L.append("CHANGE LOG (context only):")
    return L, r


def _tool_filler(r, i):
    return (f"{_month(r.randint(14, 23))}-{r.randint(1, 28):02d} change-{r.randint(2000, 9899)}: "
            f"{r.choice(['tightened', 'relaxed', 'renamed', 'deprecated', 'added', 'split'])} "
            f"{r.choice(['rate limit', 'scope', 'field', 'error code', 'pagination', 'timeout'])} on "
            f"{r.choice(METHODS)} /v{r.randint(1, 3)}/{r.choice(RES)}; ticket {r.choice(TEAMS)}-{r.randint(100, 999)}; "
            f"rollout {r.choice(['immediate', 'staged 7d', 'behind flag'])}; reviewer {r.choice(FIRST).lower()}{i % 89:02d}")


TOOL_QUESTION = """
TASK — Return ONLY a JSON array (no prose before or after) of exactly 36 objects with keys
{"endpoint","method","test_name","payload","expected_status","expected_body_contains","cleanup"}.
Cover the FIRST 18 endpoints of the catalog above, exactly two objects per endpoint: one
happy-path test and one failure-path test exercising a documented error code. payload must
be a realistic JSON object honoring the route's documented params and the shared schemas;
cleanup names the concrete reversing call. Use the documented scopes and rate limits.
""".strip()


def build_docpack(domain, idx, count_fn):
    """One deterministic ~10k-Kimi-token prompt. Filler rows are appended (chunks of 4)
    then trimmed (one at a time) until the FULL prompt tokenizes into the target window.
    Deterministic given a deterministic count_fn."""
    if domain == "math":
        sections, r = _math_sections(idx)
        filler_fn, question = _math_filler, MATH_QUESTION
    else:
        sections, r = _tool_sections(idx)
        filler_fn, question = _tool_filler, TOOL_QUESTION
    filler = []

    def assemble():
        return "\n".join(sections + filler) + "\n\n" + question

    guard = 0
    while count_fn(assemble()) < TARGET_TOK_LO:
        filler.extend(filler_fn(r, len(filler) + j) for j in range(4))
        guard += 1
        assert guard < 700, "docpack builder failed to reach target window"
    while count_fn(assemble()) > TARGET_TOK_HI and filler:
        filler.pop()
    n = count_fn(assemble())
    assert TARGET_TOK_LO <= n <= TARGET_TOK_HI, f"docpack {domain}/{idx} at {n} tokens"
    return assemble()


def build_all_prompts(count_fn):
    return {"math": [build_docpack("math", i, count_fn) for i in range(16)],
            "tool": [build_docpack("tool", i, count_fn) for i in range(16)],
            "tool_short": TOOL16}


def prompts_sha256(prompts):
    return hashlib.sha256(json.dumps(
        {"math10k": prompts["math"], "tool10k": prompts["tool"]},
        sort_keys=True).encode()).hexdigest()


def tool16_sha256():
    return hashlib.sha256(json.dumps(TOOL16, sort_keys=True).encode()).hexdigest()


# ---------------- Kimi tokenizer (no remote code; tiktoken over local assets) ----------------
def load_kimi_count_fn(assets_dir):
    import ast
    import base64
    import tiktoken
    src = open(os.path.join(assets_dir, "tokenization_kimi.py")).read()
    m = re.search(r'pat_str = "\|"\.join\(\[(.*?)\]\)', src, re.S)
    assert m, "pat_str block not found in tokenization_kimi.py"
    pat_str = "|".join(ast.literal_eval("[" + m.group(1) + "]"))
    ranks = {}
    for line in open(os.path.join(assets_dir, "kimi_tiktoken.model"), "rb").read().splitlines():
        if line:
            tok, rank = line.split()
            ranks[base64.b64decode(tok)] = int(rank)
    enc = tiktoken.Encoding(name="kimi", pat_str=pat_str, mergeable_ranks=ranks, special_tokens={})
    return lambda text: len(enc.encode(text))


def find_assets_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.environ.get("BRL_ASSETS", ""), "/root/assets", os.path.join(here, "assets")):
        if cand and os.path.exists(os.path.join(cand, "kimi_tiktoken.model")):
            return cand
    return None


def canonical_prompts(strict=True):
    """Build the canonical prompt set with the REAL Kimi tokenizer and assert the pinned
    sha. strict=False (BRL_ALLOW_EST=1 environments) falls back to a chars-based estimate
    for machines without tiktoken/assets: shapes only, sha NOT asserted."""
    assets = find_assets_dir()
    if assets is not None:
        try:
            count_fn = load_kimi_count_fn(assets)
            prompts = build_all_prompts(count_fn)
            sha = prompts_sha256(prompts)
            assert sha == PROMPT_SHA_10K, f"prompt sha drift: {sha} != {PROMPT_SHA_10K}"
            assert tool16_sha256() == TOOL16_SHA, "TOOL16 drifted from the pinned sha"
            return prompts, sha, True
        except ImportError:
            pass
    assert not strict or os.environ.get("BRL_ALLOW_EST") == "1", \
        "real Kimi tokenizer unavailable and BRL_ALLOW_EST != 1 (fail-closed)"
    prompts = build_all_prompts(lambda t: len(t) // 4)
    return prompts, prompts_sha256(prompts), False


# ---------------- pure helpers (CPU-tested by --selftest) ----------------
# Engine flag notes (verified against the pinned vLLM nightly):
# - speculative-config use_local_argmax_reduction replaces the O(vocab) TP all-gather per
#   draft token with an O(2*tp) local argmax; valid for greedy non-tree eagle3.
# - attention-config {"backend": "TOKENSPEED_MLA"} routes MLA to grouped-Q CuTe DSL kernels:
#   SM100 only, requires an fp8 KV cache and K2.6's MLA dims (128/64/128), needs the
#   tokenspeed-mla pip package; MAX_Q_LEN=8 caps k at 7 with TP4's 16 heads per GPU.
# - disable_flashinfer_q_quantization is kept in every config (consumed only by non-MLA
#   flashinfer paths; harmless where inert).
# - VLLM_ATTENTION_BACKEND was removed upstream (PR #32812); backend control is
#   attention-config only, and SM100 MLA auto-selects FLASHINFER_MLA.
# - VLLM_FLASHINFER_ALLREDUCE_BACKEND=mnnvl requires flashinfer >= 0.6.12; applied per-arm.
ALLOWED_FLAGS = {
    "--tensor-parallel-size", "--gpu-memory-utilization", "--quantization", "--max-model-len",
    "--kv-cache-dtype", "--speculative-config", "--compilation-config", "--attention-config",
    "--chat-template", "--limit-mm-per-prompt", "--async-scheduling", "--trust-remote-code",
    "--port", "--enable-expert-parallel",
}


def caps_for_k(k):
    """cudagraph capture sizes: 1..5 for non-spec shapes plus multiples of (k+1), so
    speculative verify batches land on captured graphs."""
    return sorted({1, 2, 3, 4, 5, (k + 1), 2 * (k + 1), 3 * (k + 1), 4 * (k + 1),
                   8 * (k + 1), 16 * (k + 1), 32 * (k + 1)})


def cfg_dict(head, k, kv="fp8_e4m3", ep=False, argmax=False, mnnvl=False,
             tokenspeed=False, ml=16384):
    return {"head": head, "k": k, "kv": kv, "method": "eagle3", "ep": bool(ep),
            "argmax": bool(argmax), "mnnvl": bool(mnnvl), "tokenspeed": bool(tokenspeed),
            "ml": ml}


def build_cmd(cfg):
    """vllm serve command. For the R0 config, byte-compatible with the reference session
    except max-model-len 16384 (the single intentional diff; asserted by the selftest)."""
    assert not (cfg["tokenspeed"] and cfg["kv"] != "fp8_e4m3"), \
        "TOKENSPEED_MLA requires fp8 KV cache (tokenspeed_mla.py raises otherwise)"
    cmd = ["vllm", "serve", NVFP4,
           "--tensor-parallel-size", "4", "--gpu-memory-utilization", "0.90",
           "--quantization", "modelopt_fp4", "--max-model-len", str(cfg["ml"])]
    if cfg["kv"] != "auto":
        cmd += ["--kv-cache-dtype", cfg["kv"]]
    spec = {"model": cfg["head"], "method": cfg["method"],
            "num_speculative_tokens": cfg["k"]}
    if cfg["argmax"]:
        spec["use_local_argmax_reduction"] = True
    cmd += ["--speculative-config", json.dumps(spec),
            "--compilation-config", json.dumps({"cudagraph_mode": "FULL_AND_PIECEWISE",
                                                "cudagraph_capture_sizes": caps_for_k(cfg["k"])})]
    att = {"disable_flashinfer_q_quantization": True}
    if cfg["tokenspeed"]:
        att["backend"] = "TOKENSPEED_MLA"
    cmd += ["--attention-config", json.dumps(att),
            "--chat-template", "/cache/kimi_chat_template.jinja",
            "--limit-mm-per-prompt", '{"image":0,"video":0}',
            "--async-scheduling", "--trust-remote-code", "--port", "8000"]
    if cfg["ep"]:
        cmd += ["--enable-expert-parallel"]
    for tok in cmd:
        if tok.startswith("--"):
            assert tok in ALLOWED_FLAGS, f"unverified flag {tok}"
    return cmd


def interp_median(vals):
    return round(statistics.median(vals), 1) if vals else None


def rows_rates(rows, max_slot=None):
    out = []
    for r in rows:
        if r.get("tok_s") and (max_slot is None or r.get("slot", 99) < max_slot):
            out.append(r["tok_s"])
    return sorted(out)


def agg_rate(rows):
    tot_tok = tot_t = 0.0
    for r in rows:
        if r.get("tok_s") and r.get("ctok"):
            tot_tok += r["ctok"] - 1
            tot_t += (r["ctok"] - 1) / r["tok_s"]
    return round(tot_tok / tot_t, 1) if tot_t else None


def cell_stats(rows, power=None):
    """Assemble cell statistics from exec-order rows (idempotent; used on probe and on
    probe+completion merges). tau and per_pos aggregate from per-row counter deltas so
    interleaved cells never contaminate each other."""
    rates = rows_rates(rows)
    ctoks = sorted(r["ctok"] for r in rows if r.get("ctok"))
    ttfts = sorted(r["ttft_ms"] for r in rows if r.get("ttft_ms"))
    drafts = sum(r.get("dr") or 0 for r in rows)
    acc = sum(r.get("ac") or 0 for r in rows)
    pp = {}
    for r in rows:
        for pos, v in (r.get("pp_raw") or {}).items():
            pp[pos] = pp.get(pos, 0.0) + v
    ctok_med = ctoks[len(ctoks) // 2] if ctoks else None
    ttft_med = ttfts[len(ttfts) // 2] if ttfts else None
    return {"n": len(rows),
            "tok_s_median": interp_median(rates),  # headline statistic, pre-registered
            "tok_s_first8_median": interp_median(rows_rates(rows, max_slot=8)) if len(rows_rates(rows, max_slot=8)) >= 5 else None,
            "tok_s_agg": agg_rate(rows),
            "tok_s_all_sorted": rates,
            "ctok_median": ctok_med,
            "aa_len_ok": bool(ctok_med and ctok_med >= AA_MIN_CTOK),
            "ttft_ms_median": ttft_med,
            "prefix_cache_suspect": bool(ttft_med is not None and ttft_med < 120 and rows and rows[0].get("is_10k")),
            "tau": round(1 + acc / drafts, 4) if drafts else None,
            "per_pos": {k: round(v / drafts, 4) for k, v in sorted(pp.items())} if drafts else None,
            "power": power, "rows": rows}


def first8_median(cell):
    if not cell:
        return None
    return cell.get("tok_s_first8_median")


def champ8(cells):
    vals = [first8_median(cells.get(c)) for c in ("math_nothink_10k", "tool_nothink_10k")]
    vals = [v for v in vals if v]
    return max(vals) if vals else None


def gate_keep(cand_cells, ref_cells, name, gates, thresh=GATE_KEEP):
    """Pre-registered lever gate: first-8 interpolated medians, identical prompt slots,
    same session, same statistic both sides. Keep iff champ(cand) >= champ(ref)*thresh."""
    c, r = champ8(cand_cells), champ8(ref_cells)
    keep = bool(c and r and c >= r * thresh)
    gates[name] = {"champ_cand": c, "champ_ref": r,
                   "ratio": round(c / r, 4) if c and r else None, "thresh": thresh,
                   "keep": keep,
                   "rule": "first-8 interp medians, identical prompt slots, same session"}
    return keep


def k45_forecast(per_pos, tok_s, dm=0.85):
    """From a k=3 per_pos vector and measured tok/s: predicted tok/s at k=4/5 assuming
    conditional acceptance at new positions matches position 2's, each extra position
    costing dm ms of step time (draft pass + verify tax; calibrated 0.6-1.2)."""
    p = [per_pos.get(str(i)) for i in range(3)]
    if any(v is None or v <= 0 for v in p):
        return None
    c = p[2] / p[1]
    tau3 = 1 + sum(p)
    t3 = 1000.0 * tau3 / tok_s
    p3 = p[2] * c
    p4 = p3 * c
    return {"cond_pos2": round(c, 3),
            "k4_tok_s": round(1000 * (tau3 + p3) / (t3 + dm), 1),
            "k5_tok_s": round(1000 * (tau3 + p3 + p4) / (t3 + 2 * dm), 1)}


def parse_metrics_text(txt):
    """counters from a /metrics exposition; regex matches the pinned nightly's format."""
    def total(sub):
        return sum(float(m.group(1)) for m in
                   re.finditer(rf'^vllm:{sub}(?:_total)?(?:{{[^}}]*}})?\s+([0-9.eE+-]+)\s*$', txt, re.M))
    per_pos = {}
    for m in re.finditer(r'^vllm:spec_decode_num_accepted_tokens_per_pos(?:_total)?{([^}]*)}\s+([0-9.eE+-]+)\s*$', txt, re.M):
        pos_m = re.search(r'position="(\d+)"', m.group(1))
        if pos_m:
            i = int(pos_m.group(1))
            per_pos[i] = per_pos.get(i, 0.0) + float(m.group(2))
    return {"d": total("spec_decode_num_drafts"), "a": total("spec_decode_num_accepted_tokens"), "pp": per_pos}


# ---------------- NVFP4 zero-init guard (vLLM PR #45739) ----------------
def apply_zero_init_patch(log=print):
    """Idempotently restore torch.zeros in vllm._custom_ops.create_fp4_scale_tensor
    (upstream 92c7fac). The torch.empty regression (#42988) leaves the padded swizzled
    NVFP4 scale buffer uninitialized -> nondeterministic NaN logits / degenerate output
    on the modelopt-fp4 decode path used here. Returns 'present'|'patched'|'missing'."""
    try:
        import vllm._custom_ops as ops
        path = ops.__file__
        src = open(path).read()
        m = re.search(r"def create_fp4_scale_tensor.*?(?=\ndef |\Z)", src, re.S)
        if not m:
            log("[zeroinit] create_fp4_scale_tensor not found; cannot verify")
            return "missing"
        block = m.group(0)
        if "torch.empty(" not in block:
            log("[zeroinit] fix already present (no torch.empty in function)")
            return "present"
        patched = src[:m.start()] + block.replace("torch.empty(", "torch.zeros(") + src[m.end():]
        open(path, "w").write(patched)
        log(f"[zeroinit] PATCHED {path} (torch.empty -> torch.zeros in create_fp4_scale_tensor)")
        return "patched"
    except Exception as e:  # noqa: BLE001 - never let the guard kill the session by itself
        log(f"[zeroinit] probe failed: {e!r}")
        return "missing"


# ---------------- measurement core ----------------
def schat(base_url, model, prompt, max_tokens, thinking=True, seed=None, keep_text=True):
    import time
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="x", timeout=900)
    extra = {} if thinking else {"chat_template_kwargs": {"thinking": False}}
    t_req = time.perf_counter()
    t_first = t_last = None
    completion_tokens = None
    parts = []
    kwargs = dict(model=model, temperature=0.6, top_p=1.0, max_tokens=max_tokens, stream=True,
                  stream_options={"include_usage": True},
                  messages=[{"role": "user", "content": prompt}], extra_body=extra)
    if seed is not None:
        kwargs["seed"] = seed
    stream = client.chat.completions.create(**kwargs)
    for chunk in stream:
        now = time.perf_counter()
        if chunk.usage is not None:
            completion_tokens = chunk.usage.completion_tokens
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta is not None:
            piece = getattr(delta, "content", None) or getattr(delta, "reasoning_content", None)
            if piece:
                if t_first is None:
                    t_first = now
                t_last = now
                if keep_text and sum(len(p) for p in parts) < 200_000:
                    parts.append(piece)
    row = {"ctok": completion_tokens,
           "ttft_ms": round((t_first - t_req) * 1000, 0) if t_first else None}
    if completion_tokens and t_first and t_last and t_last > t_first and completion_tokens >= 64:
        row["tok_s"] = round((completion_tokens - 1) / (t_last - t_first), 1)
    if keep_text:
        row["text"] = "".join(parts)
    return row


def measure_rows(ctx, arm, name, dom, max_tokens, thinking, lo, hi):
    """Serial requests over prompt slots [lo, hi); per-request counter deltas; exec-order
    rows with slot indices."""
    rows = []
    consec_err = 0
    for slot in range(lo, hi):
        if ctx.spent() > HARD_CAP - 1.5:
            ctx.log(f"SPEND GUARD inside {arm}/{name} at slot {slot}")
            break
        p = ctx.prompts[dom][slot]
        b = ctx.counters()
        try:
            row = ctx.schat(p, max_tokens, thinking, seed=SEED_BASE + slot)
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
            row["pp_raw"] = {str(i): a["pp"].get(i, 0.0) - b["pp"].get(i, 0.0)
                             for i in sorted(a.get("pp", {}))}
            row["tau"] = round(1 + row["ac"] / row["dr"], 4)
        rows.append(row)
        if consec_err >= 3:
            ctx.log(f"ABORT CELL {arm}/{name}: 3 consecutive request errors")
            break
    return rows


def run_cell(ctx, res, arm, name, dom, max_tokens, thinking, lo, hi):
    """Run slots [lo,hi) and merge into any existing rows for this cell (n=16 completion:
    probe rows 0-7 plus completion rows 8-15 on the same server process, serial,
    pre-registered as one cell)."""
    import time
    if ctx.spent() > HARD_CAP - 2.0:
        ctx.log(f"SPEND GUARD: skipping {arm}/{name}")
        return None
    t0 = time.time()
    new_rows = measure_rows(ctx, arm, name, dom, max_tokens, thinking, lo, hi)
    t1 = time.time()
    cells = res["arms"].setdefault(arm, {}).setdefault("cells", {})
    prev = cells.get(name, {}).get("rows", [])
    rows = prev + new_rows
    out = cell_stats(rows, power=ctx.power_window(t0, t1))
    cells[name] = out
    ctx.log(f"{arm} {name}[{lo}:{hi}]: med={out['tok_s_median']} f8={out['tok_s_first8_median']} "
            f"agg={out['tok_s_agg']} tau={out['tau']} ctok_med={out['ctok_median']} "
            f"ttft={out['ttft_ms_median']} n={out['n']} ${ctx.spent():.1f}")
    ctx.save(res)
    return out


def run_arm_boot(ctx, res, tag, cfg, boot_budget_min):
    if ctx.spent() > HARD_CAP - 4.0:
        res["arms"][tag] = {"skipped": f"spend guard at ${ctx.spent():.1f}"}
        ctx.save(res)
        return False
    try:
        ctx.start(tag, cfg, boot_budget_min)
        res["arms"].setdefault(tag, {})["cfg"] = cfg
        res["arms"][tag]["cmd"] = " ".join(build_cmd(cfg))
        ctx.schat("Reply with exactly: WARMUP OK", 48, True, seed=SEED_BASE, keep_text=False)
        ctx.schat("Reply with exactly: WARMUP OK", 48, False, seed=SEED_BASE, keep_text=False)
        ctx.schat(ctx.warmup_10k, 128, False, seed=SEED_BASE, keep_text=False)  # long-ctx warm
        ctx.save(res)
        return True
    except Exception as e:  # noqa: BLE001
        res["arms"].setdefault(tag, {})["error"] = repr(e)[:300]
        ctx.log(f"ARM {tag} FAILED TO BOOT: {repr(e)[:200]}")
        ctx.save(res)
        ctx.stop()
        return False


def probe_and_maybe_complete(ctx, res, tag, cfg, incumbent_cells, gates, gate_name,
                             thresh=GATE_KEEP, boot_budget_min=22):
    """Boot cfg, probe slots 0-7 on both record domains, gate vs incumbent first-8,
    complete slots 8-15 on keep. Returns (kept, cells)."""
    if not run_arm_boot(ctx, res, tag, cfg, boot_budget_min):
        gates[gate_name] = {"keep": False, "boot_failed": True}
        return False, None
    run_cell(ctx, res, tag, "math_nothink_10k", "math", 2048, False, 0, 8)
    run_cell(ctx, res, tag, "tool_nothink_10k", "tool", 2048, False, 0, 8)
    cells = res["arms"][tag].get("cells", {})
    kept = gate_keep(cells, incumbent_cells, gate_name, gates, thresh)
    ctx.save(res)
    if kept:
        run_cell(ctx, res, tag, "math_nothink_10k", "math", 2048, False, 8, 16)
        run_cell(ctx, res, tag, "tool_nothink_10k", "tool", 2048, False, 8, 16)
        cells = res["arms"][tag].get("cells", {})
    ctx.stop()
    return kept, cells


def budget_ok_for_optional(ctx):
    return ctx.spent() < HARD_CAP - RESERVE_THINK - EST_NEXT_ARM


# ---------------- session runners ----------------
def run_campaign(ctx, fovea_head=None):
    """Baseline session: R0 baseline, then gated lever arms (R1 pack A, R2 k5-or-EP
    branch, R4 Fovea head, R3 TokenSpeed), then think-mode cells on the champion.
    Gates compare first-8 medians on identical prompt slots within the same session."""
    res = {"campaign": "baseline_run", "arms": {}, "gates": {}, "decisions": [],
           "hourly_usd": 24.0,
           "bars": {"tier1": BAR_T1, "tier2_brief": BAR_T2_BRIEF, "bar_pin": BAR_PIN,
                    "bar_pin_note": "Crusoe AA 72h median pinned on the morning of the run"},
           "protocol": {"input_tokens": f"{TARGET_TOK_LO}-{TARGET_TOK_HI} Kimi-native per prompt, "
                                        "16 distinct docpacks per domain",
                        "output": "max_tokens 2048, AA length flag at ctok_median >= 1500",
                        "sampling": "temp 0.6, top_p 1.0, per-request seed",
                        "stat": "interpolated median of per-request (ctok-1)/(t_last-t_first), "
                                "serial single-stream, streaming",
                        "aa_norm": "o200k/Kimi-native measured 1.0035-1.0081 token-weighted; "
                                   "raw text stored per row for offline o200k conversion"},
           "prompt_sha256_10k": ctx.prompt_sha,
           "zero_init": ctx.zero_init_state,
           "p0_manifest": ctx.p0_manifest_summary,
           "fovea_head": fovea_head}
    gates = res["gates"]

    def note(msg):
        res["decisions"].append(msg)
        ctx.log("DECISION: " + msg)

    # ---- R0: baseline on the AA protocol (record-eligible) ----
    r0_cfg = cfg_dict(EAGLE, 3)
    if not run_arm_boot(ctx, res, "R0_base_long", r0_cfg, boot_budget_min=35):
        note("R0 failed to boot; session aborted with no baseline")
        ctx.save(res)
        return res
    health = run_cell(ctx, res, "R0_base_long", "tool_short8_health", "tool_short", 1200, False, 0, 8)
    h8 = (health or {}).get("tok_s_median")
    if h8 is not None and h8 < HEALTH_FLOOR:
        note(f"HEALTH ABORT: reference TOOL16[:8] cell read {h8} < {HEALTH_FLOOR} "
             f"(reference session read 336.8); sick node or broken build")
        ctx.stop()
        ctx.save(res)
        return res
    run_cell(ctx, res, "R0_base_long", "math_nothink_10k", "math", 2048, False, 0, 8)
    r0_cells = res["arms"]["R0_base_long"]["cells"]
    m8 = first8_median(r0_cells.get("math_nothink_10k"))
    sanity_ok = not (m8 is not None and m8 < SANITY_FLOOR)
    if not sanity_ok:
        note(f"SANITY: first-8 math_nothink_10k {m8} < {SANITY_FLOOR}; levers cannot reach the "
             f"bar from here; completing R0 n=16 cells as the banked baseline and stopping")
    run_cell(ctx, res, "R0_base_long", "math_nothink_10k", "math", 2048, False, 8, 16)
    run_cell(ctx, res, "R0_base_long", "tool_nothink_10k", "tool", 2048, False, 0, 8)
    run_cell(ctx, res, "R0_base_long", "tool_nothink_10k", "tool", 2048, False, 8, 16)
    r0_cells = res["arms"]["R0_base_long"]["cells"]
    mn = r0_cells.get("math_nothink_10k") or {}
    if mn.get("per_pos") and mn.get("tok_s_median"):
        fc = k45_forecast(mn["per_pos"], mn["tok_s_median"])
        if fc:
            res["r0_k45_forecast"] = fc
            note(f"live k forecast from R0 math per_pos: {fc}")
    champion_tag, champion_cells, champion_cfg = "R0_base_long", r0_cells, dict(r0_cfg)
    if not sanity_ok:
        ctx.stop()
        return ship_verdict(ctx, res, note, stopped_early=True)
    ctx.stop()

    # ---- R1: pack A = k4 + local argmax (+ mnnvl if proof passed) ----
    if budget_ok_for_optional(ctx):
        r1_cfg = cfg_dict(EAGLE, 4, argmax=True, mnnvl=MNNVL_OK)
        kept, cells = probe_and_maybe_complete(ctx, res, "R1_packA", r1_cfg, champion_cells,
                                               gates, "G1_packA")
        if kept:
            champion_tag, champion_cells, champion_cfg = "R1_packA", cells, dict(r1_cfg)
            note("pack A kept (k4 + local_argmax" + (" + mnnvl" if MNNVL_OK else "") + ")")
        else:
            note("pack A dropped; champion stays R0")
    else:
        note("R1 skipped by spend guard")

    # ---- R2 branch: k5 escalation if pack A was strong, else EP ----
    g1 = gates.get("G1_packA", {})
    if budget_ok_for_optional(ctx):
        if g1.get("keep") and (g1.get("ratio") or 0) >= GATE_ESCALATE_K5:
            r2_cfg = dict(champion_cfg)
            r2_cfg["k"] = 5
            tag, gname = "R2_k5", "G2_k5"
        else:
            r2_cfg = dict(champion_cfg)
            r2_cfg["ep"] = True
            tag, gname = "R2_EP", "G2_EP"
        kept, cells = probe_and_maybe_complete(ctx, res, tag, r2_cfg, champion_cells, gates, gname)
        if kept:
            champion_tag, champion_cells, champion_cfg = tag, cells, dict(r2_cfg)
            note(f"{tag} kept")
        else:
            note(f"{tag} dropped or failed to boot (tolerated)")
    else:
        note("R2 skipped by spend guard")

    # ---- R4: Fovea-E head (Tier-3), before TokenSpeed in priority ----
    if fovea_head and budget_ok_for_optional(ctx):
        r4_cfg = dict(champion_cfg)
        r4_cfg["head"] = fovea_head
        kept, cells = probe_and_maybe_complete(ctx, res, "R4_fovea", r4_cfg, champion_cells,
                                               gates, "G4_fovea", thresh=FOVEA_TIE)
        if kept:
            champion_tag, champion_cells, champion_cfg = "R4_fovea", cells, dict(r4_cfg)
            note("Fovea-E within 3% of champion: record moves to our head (Tier-3)")
        else:
            note("Fovea-E outside 3% or failed; champion unchanged")
    elif fovea_head:
        note("R4 skipped by spend guard")

    # ---- R3: TokenSpeed probe (Crusoe-kernel twin; tolerated failure; last priority) ----
    if TOKENSPEED_OK and budget_ok_for_optional(ctx):
        r3_cfg = dict(champion_cfg)
        r3_cfg["tokenspeed"] = True
        if r3_cfg["kv"] == "fp8_e4m3":
            kept, cells = probe_and_maybe_complete(ctx, res, "R3_tokenspeed", r3_cfg,
                                                   champion_cells, gates, "G3_tokenspeed")
            if kept:
                champion_tag, champion_cells, champion_cfg = "R3_tokenspeed", cells, dict(r3_cfg)
                note("TOKENSPEED_MLA kept: the grouped-Q verify kernel wins at bs=1 here")
            else:
                note("TOKENSPEED_MLA dropped or failed to boot (upstream: regresses at bs<=2; tolerated)")
    else:
        note("R3 skipped (proof flag or spend guard)")

    res["champion"] = {"tag": champion_tag, "cfg": champion_cfg, "champ8": champ8(champion_cells)}

    # ---- R5: think-mode record cells on the champion ----
    if ctx.spent() < HARD_CAP - 4.5:
        if run_arm_boot(ctx, res, "R5_record_think", dict(champion_cfg), boot_budget_min=22):
            run_cell(ctx, res, "R5_record_think", "math_think_10k", "math", 2048, True, 0, 8)
            run_cell(ctx, res, "R5_record_think", "math_think_10k", "math", 2048, True, 8, 16)
            run_cell(ctx, res, "R5_record_think", "tool_think_10k", "tool", 2048, True, 0, 8)
            run_cell(ctx, res, "R5_record_think", "tool_think_10k", "tool", 2048, True, 8, 16)
            ctx.stop()
    else:
        note("R5 think records skipped by spend guard")

    return ship_verdict(ctx, res, note)



def run_record(ctx, fovea_head=None):
    """Record session. Anchors from the same-day baseline session (brl11_baseline.json):
    tool_nothink_10k 368.1 and math_nothink_10k 357.6 at k=3; the measured per-position
    forecast puts k=5 math at ~397.9 and the k5+argmax central estimate at ~409-414
    against the 411.9 pin. One boot plus one Fovea-E probe, protocol identical to the
    baseline session; every cell is saved."""
    res = {"campaign": "brl11_record", "arms": {}, "gates": {}, "decisions": [],
           "hourly_usd": 24.0,
           "bars": {"tier1": BAR_T1, "tier2_brief": BAR_T2_BRIEF, "bar_pin": BAR_PIN,
                    "bar_pin_note": "Crusoe AA 72h median pinned on the morning of the run"},
           "anchor": {"tool_nothink_10k": 368.1, "math_nothink_10k": 357.6,
                      "source": "brl11_baseline.json R0_base_long (same day)"},
           "prompt_sha256_10k": ctx.prompt_sha,
           "zero_init": ctx.zero_init_state,
           "p0_manifest": ctx.p0_manifest_summary,
           "fovea_head": fovea_head}
    gates = res["gates"]

    def note(msg):
        res["decisions"].append(msg)
        ctx.log("DECISION: " + msg)

    # ---- C1: k5 + local argmax (record-eligible attempt at the pin) ----
    c1 = cfg_dict(EAGLE, 5, argmax=True)
    if not run_arm_boot(ctx, res, "C1_k5_argmax", c1, boot_budget_min=32):
        note("C1 failed to boot; record session aborted")
        ctx.save(res)
        return res
    health = run_cell(ctx, res, "C1_k5_argmax", "tool_short8_health", "tool_short", 1200, False, 0, 8)
    h8 = (health or {}).get("tok_s_median")
    h_tau = (health or {}).get("tau")
    # k=5 verifies two more tokens per step than the k=3-calibrated HEALTH_FLOOR assumes,
    # so a healthy node reads slower on this short cell (294.4 at tau 3.46 observed, consistent
    # with the k=5 verify tax). Sickness shows as low acceptance or collapsed throughput,
    # hence separate floors here: 250 tok/s and tau 2.5.
    if h8 is not None and (h8 < 250 or (h_tau is not None and h_tau < 2.5)):
        note(f"HEALTH ABORT: {h8} tok/s @ tau {h_tau} (floors: 250 tok/s, tau 2.5)")
        ctx.stop()
        ctx.save(res)
        return res
    note(f"health at k5: {h8} @ tau {h_tau} (k3 floor {HEALTH_FLOOR} not applicable; "
         f"implied step consistent with k5 verify tax)")
    run_cell(ctx, res, "C1_k5_argmax", "tool_nothink_10k", "tool", 2048, False, 0, 8)
    run_cell(ctx, res, "C1_k5_argmax", "tool_nothink_10k", "tool", 2048, False, 8, 16)
    run_cell(ctx, res, "C1_k5_argmax", "math_nothink_10k", "math", 2048, False, 0, 8)
    run_cell(ctx, res, "C1_k5_argmax", "math_nothink_10k", "math", 2048, False, 8, 16)
    c1_cells = res["arms"]["C1_k5_argmax"]["cells"]
    c1_tool = (c1_cells.get("tool_nothink_10k") or {}).get("tok_s_median") or 0
    note(f"C1 k5+argmax: tool {c1_tool} vs anchor 368.1 (pin {BAR_PIN})")
    ctx.stop()

    # ---- C2: Fovea-E served probe on the same config ----
    if fovea_head and ctx.spent() < HARD_CAP - 9.0:
        c2 = dict(c1)
        c2["head"] = fovea_head
        if run_arm_boot(ctx, res, "C2_foveaE_k5", c2, boot_budget_min=25):
            run_cell(ctx, res, "C2_foveaE_k5", "tool_nothink_10k", "tool", 2048, False, 0, 8)
            f8_c2 = first8_median((res["arms"]["C2_foveaE_k5"]["cells"] or {}).get("tool_nothink_10k"))
            f8_c1 = first8_median(c1_cells.get("tool_nothink_10k"))
            gates["G_fovea"] = {"f8_c2": f8_c2, "f8_c1": f8_c1,
                                "keep": bool(f8_c2 and f8_c1 and f8_c2 >= f8_c1 * 0.985)}
            if gates["G_fovea"]["keep"]:
                run_cell(ctx, res, "C2_foveaE_k5", "tool_nothink_10k", "tool", 2048, False, 8, 16)
                run_cell(ctx, res, "C2_foveaE_k5", "math_nothink_10k", "math", 2048, False, 0, 8)
                run_cell(ctx, res, "C2_foveaE_k5", "math_nothink_10k", "math", 2048, False, 8, 16)
                note("Fovea-E completed to n=16 (within noise of or above C1 on first-8)")
            else:
                note(f"Fovea-E probe dropped: f8 {f8_c2} vs C1 f8 {f8_c1}")
            ctx.stop()
    else:
        note("C2 Fovea-E probe skipped (no head or spend guard)")

    return ship_verdict(ctx, res, note)


def ship_verdict(ctx, res, note, stopped_early=False):
    """Pre-registered: best n>=16 10k-in cell, preferring aa_len_ok cells; tiers vs
    BAR_PIN / 397 / 346; Tier-3 iff the cell is Fovea's."""
    candidates = []
    for tag, arm in res["arms"].items():
        for cname, c in (arm.get("cells") or {}).items():
            if c.get("n", 0) >= 16 and c.get("tok_s_median") and cname.endswith("_10k"):
                candidates.append({"tok_s": c["tok_s_median"], "arm": tag, "cell": cname,
                                   "aa_len_ok": bool(c.get("aa_len_ok")),
                                   "ctok_median": c.get("ctok_median"),
                                   "fovea": (arm.get("cfg") or {}).get("head") == (res.get("fovea_head") or "__none__"),
                                   "agg": c.get("tok_s_agg"), "tau": c.get("tau")})
    best = None
    pool = [c for c in candidates if c["aa_len_ok"]] or candidates
    for c in pool:
        if best is None or c["tok_s"] > best["tok_s"] or \
                (c["tok_s"] == best["tok_s"] and c["fovea"] and not best["fovea"]):
            best = c  # exact ties prefer our head (pre-registered Tier-3 preference)
    tier = "MISS"
    if best:
        if best["tok_s"] >= BAR_PIN:
            tier = "CRUSOE_BEATEN"
        elif best["tok_s"] >= BAR_T2_BRIEF:
            tier = "TIER2_BRIEF_ONLY"
        elif best["tok_s"] >= BAR_T1:
            tier = "TIER1"
        if tier != "MISS" and best.get("fovea"):
            tier += "+TIER3"
    res["ship"] = {"best_cell": best, "tier": tier, "stopped_early": stopped_early,
                   "rule": f"best n>=16 10k-in interp median (aa_len_ok preferred) vs "
                           f"{BAR_PIN} pinned / {BAR_T2_BRIEF} brief / {BAR_T1}"}
    note(f"SHIP: {tier} best={best}")
    res["total_spent_usd_est"] = round(ctx.spent(), 2)
    ctx.save(res)
    return res


# ---------------- Modal wiring ----------------
if modal is not None:
    app = modal.App("brl11-record")
    kimi = modal.Volume.from_name("kimi-k26", create_if_missing=False)
    hfvol = modal.Volume.from_name("hf-cache", create_if_missing=True)
    outvol = modal.Volume.from_name("k26-draft-out", create_if_missing=True)

    # Do not edit the base image block: a byte-identical spec reuses the cached Modal
    # image, i.e. the exact nightly build the reference sessions measured.
    vllm_image = (
        modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
        .apt_install("git")
        .pip_install("vllm", pre=True, extra_index_url="https://wheels.vllm.ai/nightly")
        .pip_install("huggingface_hub", "hf_transfer")
        .pip_install("openai")
        .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_V1": "1", "CUDA_HOME": "/usr/local/cuda", "HF_HOME": "/cache/hf"})
        # appended layers only below this line (cached base preserved):
        .pip_install("tokenspeed-mla", extra_options="--no-deps")  # TOKENSPEED_MLA kernels; no-deps keeps torch/vllm pinned
        .add_local_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "kimi_tiktoken.model"),
                        remote_path="/root/assets/kimi_tiktoken.model")
        .add_local_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "tokenization_kimi.py"),
                        remote_path="/root/assets/tokenization_kimi.py")
    )

    @app.function(image=vllm_image, gpu="B200:4", volumes={"/models": kimi, "/cache": hfvol, "/out": outvol},
                  timeout=95 * 60, region="us-east")
    def session(mode: str = "campaign"):
        import subprocess
        import threading
        import time
        import urllib.request

        env = os.environ.copy()
        # VLLM_ATTENTION_BACKEND removed upstream (PR #32812): do NOT set it; SM100 MLA
        # auto-selects FLASHINFER_MLA. Backend overrides go through --attention-config.
        env["TRTLLM_ENABLE_PDL"] = "1"

        # -- fail-closed pre-serve checks (cost so far ~= container start, < $1) --
        zero_init_state = apply_zero_init_patch(log=print)
        assert zero_init_state in ("present", "patched"), \
            "NVFP4 zero-init state unresolved; refusing to serve (vLLM PR #45739)"
        assert os.path.exists("/cache/kimi_chat_template.jinja"), "chat template missing at /cache"
        prompts, sha, real_tok = canonical_prompts(strict=True)
        assert real_tok, "record run requires the real Kimi tokenizer"
        manifest_path = "/out/brl11_p0_manifest.json"
        assert os.path.exists(manifest_path), "P0 manifest missing: run cheap_proofs.py first"
        manifest = json.load(open(manifest_path))
        assert manifest.get("prompt_sha256_10k") == sha == PROMPT_SHA_10K, \
            f"prompt sha mismatch: built {sha}, manifest {manifest.get('prompt_sha256_10k')}, pinned {PROMPT_SHA_10K}"

        power_rows = []

        def power_loop():
            try:
                p = subprocess.Popen(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-l", "2"],
                                     stdout=subprocess.PIPE, text=True)
                batch = []
                for line in p.stdout:
                    try:
                        batch.append(float(line.strip()))
                    except Exception:
                        continue
                    if len(batch) == 4:
                        power_rows.append((time.time(), sum(batch)))
                        batch = []
            except Exception as e:  # noqa: BLE001
                print(f"[power] sampler dead: {e}", flush=True)
        threading.Thread(target=power_loop, daemon=True).start()

        class RealCtx:
            base_url = "http://127.0.0.1:8000/v1"
            model = NVFP4

            def __init__(self):
                self.t0 = time.time()
                self.server = None
                self.prompts = prompts
                self.prompt_sha = sha
                self.zero_init_state = zero_init_state
                self.p0_manifest_summary = {k: manifest.get(k) for k in
                                            ("vllm_version", "vllm_sha", "flashinfer",
                                             "zero_init_fix_in_source", "tokenspeed_import")}
                count_fn = load_kimi_count_fn("/root/assets")
                # warmup uses a dedicated docpack (index 90), never a record slot
                self.warmup_10k = build_docpack("math", 90, count_fn)

            def spent(self):
                return (time.time() - self.t0) / 3600 * 24.0

            def log(self, msg):
                print(f"[record|${self.spent():.1f}] {msg}", flush=True)

            def save(self, res):
                with open("/out/brl11_record.json" if res.get("campaign") == "brl11_record" else "/out/brl11_baseline.json", "w") as f:
                    json.dump(res, f, indent=1)
                outvol.commit()

            def power_window(self, t0, t1):
                w = [x for t, x in power_rows if t0 <= t <= t1]
                return {"mean_w": round(sum(w) / len(w), 0), "samples": len(w)} if w else None

            def schat(self, prompt, max_tokens, thinking=True, seed=None, keep_text=True):
                return schat(self.base_url, self.model, prompt, max_tokens, thinking, seed, keep_text)

            def counters(self):
                try:
                    txt = urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=15).read().decode()
                    return parse_metrics_text(txt)
                except Exception:
                    return None

            def start(self, tag, cfg, boot_budget_min=22):
                self.stop()
                cmd = build_cmd(cfg)
                arm_env = env.copy()
                if cfg.get("mnnvl"):
                    arm_env["VLLM_FLASHINFER_ALLREDUCE_BACKEND"] = "mnnvl"
                self.log(f"ARM {tag}: {cfg} :: {' '.join(cmd)}")
                self.server = subprocess.Popen(cmd, env=arm_env, stdout=subprocess.PIPE,
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

        runner = run_record if mode == "record" else run_campaign
        res = runner(RealCtx(), fovea_head=FOVEA_E)
        print("==== session complete ====", flush=True)
        print(json.dumps({"ship": res.get("ship"), "gates": res.get("gates"),
                          "decisions": res.get("decisions")}, indent=1), flush=True)
        return res.get("ship")

    @app.local_entrypoint()
    def main(mode: str = "campaign"):
        print("FINAL:", json.dumps(session.remote(mode=mode), indent=1))


# ---------------- selftest: mock OpenAI SSE server + full dry-runs ----------------
def _selftest():
    import threading
    import time
    import urllib.request
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    os.environ.setdefault("BRL_ALLOW_EST", "1")

    print("[1] canonical prompts:", end=" ")
    prompts, sha, real_tok = canonical_prompts(strict=False)
    prompts2, sha2, _ = canonical_prompts(strict=False)
    assert sha == sha2, "builder nondeterministic"
    assert len(prompts["math"]) == 16 and len(prompts["tool"]) == 16
    all_p = prompts["math"] + prompts["tool"]
    assert len({p[:200] for p in all_p}) == 32, "docpack headers not distinct"
    for i, p in enumerate(prompts["math"]):
        assert f"DATA PACK D{i:02d}" in p and p.rstrip().endswith("show the division.")
        assert len(p) > 26000, f"math docpack {i} suspiciously short ({len(p)} chars)"
    for i, p in enumerate(prompts["tool"]):
        assert f"CATALOG P{i:02d}" in p and "exactly 36 objects" in p
    if real_tok:
        assert sha == PROMPT_SHA_10K, f"pinned sha mismatch: built {sha}"
        assert tool16_sha256() == TOOL16_SHA
        print("OK (real tokenizer)", sha[:16])
    else:
        print("OK (estimator; sha not asserted)", sha[:16])

    print("[2] capture sizes:", end=" ")
    assert caps_for_k(3) == [1, 2, 3, 4, 5, 8, 12, 16, 32, 64, 128]
    for k in (3, 4, 5):
        cs = caps_for_k(k)
        assert cs == sorted(cs) and any(c % (k + 1) == 0 for c in cs if c > 5)
    print("OK", {k: caps_for_k(k)[:8] for k in (4, 5)})

    print("[3] cmdline regression vs reference config (single diff = max-model-len):", end=" ")
    r0 = build_cmd(cfg_dict(EAGLE, 3))
    expected = ["vllm", "serve", NVFP4,
                "--tensor-parallel-size", "4", "--gpu-memory-utilization", "0.90",
                "--quantization", "modelopt_fp4", "--max-model-len", "16384",
                "--kv-cache-dtype", "fp8_e4m3",
                "--speculative-config", '{"model": "lightseekorg/kimi-k2.6-eagle3.1-mla", "method": "eagle3", "num_speculative_tokens": 3}',
                "--compilation-config", '{"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": [1, 2, 3, 4, 5, 8, 12, 16, 32, 64, 128]}',
                "--attention-config", '{"disable_flashinfer_q_quantization": true}',
                "--chat-template", "/cache/kimi_chat_template.jinja",
                "--limit-mm-per-prompt", '{"image":0,"video":0}',
                "--async-scheduling", "--trust-remote-code", "--port", "8000"]
    assert r0 == expected, f"R0 cmdline drifted:\n{r0}\nvs\n{expected}"
    r1 = build_cmd(cfg_dict(EAGLE, 4, argmax=True, mnnvl=True))
    assert '"use_local_argmax_reduction": true' in " ".join(r1)
    assert '"num_speculative_tokens": 4' in " ".join(r1) and '"cudagraph_capture_sizes": [1, 2, 3, 4, 5, 10' in " ".join(r1)
    r3 = build_cmd(cfg_dict(EAGLE, 4, argmax=True, tokenspeed=True))
    assert '"backend": "TOKENSPEED_MLA"' in " ".join(r3) and '"disable_flashinfer_q_quantization": true' in " ".join(r3)
    try:
        build_cmd(cfg_dict(EAGLE, 4, kv="auto", tokenspeed=True))
        raise AssertionError("tokenspeed+kv-auto must be rejected")
    except AssertionError as e:
        if "must be rejected" in str(e):
            raise
    ep = build_cmd(cfg_dict(EAGLE, 5, ep=True))
    assert ep[-1] == "--enable-expert-parallel"
    print("OK")

    print("[4] stats/gate math:", end=" ")
    def fake_rows(vals, lo=0):
        return [{"tok_s": v, "ctok": 1800, "slot": lo + i, "is_10k": True, "ttft_ms": 300.0,
                 "dr": 100.0, "ac": 190.0, "pp_raw": {"0": 75.0, "1": 61.0, "2": 51.0}}
                for i, v in enumerate(vals)]
    c8 = cell_stats(fake_rows([300, 310, 320, 330, 340, 350, 360, 370]))
    assert c8["tok_s_first8_median"] == 335.0 and c8["tok_s_median"] == 335.0
    assert c8["aa_len_ok"] and abs(c8["tau"] - 2.9) < 0.001 and c8["per_pos"]["0"] == 0.75
    merged = cell_stats(fake_rows([300, 310, 320, 330, 340, 350, 360, 370]) +
                        fake_rows([400, 410, 420, 430, 440, 450, 460, 470], lo=8))
    assert merged["n"] == 16 and merged["tok_s_first8_median"] == 335.0 and merged["tok_s_median"] == 385.0
    g = {}
    base = {"math_nothink_10k": cell_stats(fake_rows([300, 310, 320, 330, 340, 350, 360, 370]))}
    up = {"math_nothink_10k": cell_stats(fake_rows([v * 1.02 for v in [300, 310, 320, 330, 340, 350, 360, 370]]))}
    assert gate_keep(up, base, "t_up", g) is True
    assert gate_keep(base, base, "t_flat", g) is False
    assert gate_keep({}, base, "t_empty", g) is False
    fc = k45_forecast({"0": 0.7482, "1": 0.6088, "2": 0.5147}, 336.8)
    assert fc and abs(fc["cond_pos2"] - 0.845) < 0.01 and 340 < fc["k4_tok_s"] < 360, fc
    low_ctok = cell_stats([{"tok_s": 400.0, "ctok": 900, "slot": 0, "is_10k": True, "ttft_ms": 40.0}])
    assert not low_ctok["aa_len_ok"] and low_ctok["prefix_cache_suspect"]
    print("OK forecast", fc)

    # ---- mock OpenAI SSE + /metrics server ----
    print("[5] mock SSE server measurement dry-run:", end=" ")
    state = {"d": 1000.0, "a": 2244.0, "pp": [750.0, 610.0, 510.0], "lock": threading.Lock()}

    class Mock(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
                return
            if self.path == "/metrics":
                with state["lock"]:
                    lines = [
                        f'vllm:spec_decode_num_drafts_total{{engine="0"}} {state["d"]}',
                        f'vllm:spec_decode_num_accepted_tokens_total{{engine="0"}} {state["a"]}',
                    ]
                    for i, v in enumerate(state["pp"]):
                        lines.append(f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine="0",position="{i}"}} {v}')
                body = ("\n".join(lines) + "\n").encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            ln = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(ln) or b"{}")
            chunks = min(int(req.get("max_tokens", 64)), 96)
            usage_tokens = int(req.get("max_tokens", 64))  # mock shortcut: pretend full length
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            def sse(obj):
                self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                self.wfile.flush()
            base_ = {"id": "m", "object": "chat.completion.chunk", "created": 0,
                     "model": req.get("model", "m")}
            sse({**base_, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}], "usage": None})
            for _ in range(chunks):
                sse({**base_, "choices": [{"index": 0, "delta": {"content": "tok "}, "finish_reason": None}], "usage": None})
                time.sleep(0.002)
            sse({**base_, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": None})
            if (req.get("stream_options") or {}).get("include_usage"):
                sse({**base_, "choices": [],
                     "usage": {"prompt_tokens": 10000, "completion_tokens": usage_tokens,
                               "total_tokens": 10000 + usage_tokens}})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            with state["lock"]:
                drafts = usage_tokens / 2.87
                state["d"] += drafts
                state["a"] += drafts * (0.75 + 0.61 + 0.51)
                for i, p in enumerate((0.75, 0.61, 0.51)):
                    state["pp"][i] += drafts * p

    srv = ThreadingHTTPServer(("127.0.0.1", 8932), Mock)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    MOCK_BASE = 400.0
    FACTORS = {"R0": 1.00, "R1": 1.06, "R2": 1.09, "R3": 1.02, "R4": 1.08, "R5": 1.09, "probe": 1.0}

    class MockCtx:
        base_url = "http://127.0.0.1:8932/v1"
        model = NVFP4

        def __init__(self, fail_tags=(), base_factor=1.0):
            self.fail_tags = fail_tags
            self.base_factor = base_factor
            self.boots = []
            self.fake_spent = 0.0
            self.saved = None
            self.tag = "probe"
            self.prompts = prompts
            self.prompt_sha = sha
            self.zero_init_state = "present"
            self.p0_manifest_summary = {"vllm_sha": "mock"}
            self.warmup_10k = prompts["math"][0][:2000]

        def spent(self):
            self.fake_spent += 0.004
            return self.fake_spent

        def log(self, msg):
            print("   [dryrun]", msg)

        def save(self, res):
            self.saved = json.loads(json.dumps(res))

        def power_window(self, t0, t1):
            return {"mean_w": 2000.0, "samples": 1}

        def schat(self, prompt, max_tokens, thinking=True, seed=None, keep_text=True):
            row = schat(self.base_url, self.model, prompt, max_tokens, thinking, seed, keep_text)
            if row.get("tok_s"):
                row["tok_s"] = round(MOCK_BASE * self.base_factor * FACTORS[self.tag.split("_")[0]], 1)
            return row

        def counters(self):
            txt = urllib.request.urlopen("http://127.0.0.1:8932/metrics", timeout=5).read().decode()
            return parse_metrics_text(txt)

        def start(self, tag, cfg, boot_budget_min=22):
            if any(t in tag for t in self.fail_tags):
                raise RuntimeError(f"simulated boot failure for {tag}")
            build_cmd(cfg)  # validates flags for every dry-run arm
            self.tag = tag
            self.boots.append((tag, json.dumps(cfg, sort_keys=True)))

        def stop(self):
            self.tag = "probe"

    ctx = MockCtx()
    res_probe = {"arms": {}}
    ctx.tag = "R0_base_long"
    out = run_cell(ctx, res_probe, "R0_base_long", "math_nothink_10k", "math", 2048, False, 0, 2)
    assert out and out["n"] == 2 and out["ctok_median"] == 2048 and out["aa_len_ok"]
    assert all(abs(r["tau"] - 2.87) < 0.02 for r in out["rows"])
    assert out["per_pos"] and abs(out["per_pos"]["0"] - 0.75) < 0.01
    assert out["rows"][0]["slot"] == 0 and out["rows"][1]["slot"] == 1
    assert out["rows"][0].get("text"), "raw text must be stored for o200k conversion"
    raw = schat(ctx.base_url, NVFP4, "raw timing check", 96, False, seed=1)
    assert raw.get("tok_s") and raw["ctok"] == 96
    print("OK tau", out["tau"], "pp", out["per_pos"])

    print("[6] full campaign dry-run (keep-ladder path):")
    ctx = MockCtx()
    res = run_campaign(ctx, fovea_head="/out/fovea_e_ckpt/1")
    boot_tags = [t for t, _ in ctx.boots]
    assert boot_tags[0] == "R0_base_long" and "R1_packA" in boot_tags and "R2_k5" in boot_tags, boot_tags
    assert "R4_fovea" in boot_tags and "R5_record_think" in boot_tags, boot_tags
    g = res["gates"]
    assert g["G1_packA"]["keep"] and g["G2_k5"]["keep"], g
    assert res["champion"]["tag"] in ("R2_k5", "R4_fovea", "R3_tokenspeed"), res["champion"]
    assert res["ship"]["tier"].startswith("CRUSOE_BEATEN"), res["ship"]
    assert res["r0_k45_forecast"]["cond_pos2"] > 0.8
    assert res["arms"]["R0_base_long"]["cells"]["math_nothink_10k"]["n"] == 16
    assert res["arms"]["R1_packA"]["cells"]["math_nothink_10k"]["n"] == 16  # completion merge
    assert res["arms"]["R5_record_think"]["cells"]["math_think_10k"]["n"] == 16
    assert ctx.saved and ctx.saved.get("ship")
    print("   gates:", {k: (v.get("ratio"), v["keep"]) for k, v in g.items()})
    print("   ship:", res["ship"]["tier"], res["ship"]["best_cell"]["cell"], res["ship"]["best_cell"]["tok_s"])

    print("[7] dry-run with EP branch + tolerated boot failures (R4/R3 fail):")
    global MNNVL_OK
    FACTORS["R1"] = 1.02   # kept but weak -> R2 goes EP branch
    FACTORS["R2"] = 1.00   # EP dropped
    ctx2 = MockCtx(fail_tags=("R4", "R3"))
    res2 = run_campaign(ctx2, fovea_head="/out/fovea_e_ckpt/1")
    boots2 = [t for t, _ in ctx2.boots]
    assert "R2_EP" in boots2 and "R2_k5" not in boots2, boots2
    assert res2["gates"]["G2_EP"]["keep"] is False
    assert res2["gates"]["G4_fovea"].get("boot_failed")
    if TOKENSPEED_OK:
        assert res2["gates"]["G3_tokenspeed"].get("boot_failed")
    else:
        assert "G3_tokenspeed" not in res2["gates"], "R3 must be skipped when TOKENSPEED_OK is False"
    assert res2["champion"]["tag"] == "R1_packA", res2["champion"]
    assert res2["arms"]["R2_EP"]["cells"]["math_nothink_10k"]["n"] == 8  # probe only, no completion
    assert res2["ship"]["tier"].startswith("CRUSOE_BEATEN") or res2["ship"]["tier"].startswith("TIER"), res2["ship"]
    print("   boots:", boots2)
    print("   ship:", res2["ship"]["tier"], res2["ship"]["best_cell"])

    print("[8] dry-run sanity-stop (low baseline banks R0 and stops):")
    FACTORS.update({"R0": 1.0, "R1": 1.06, "R2": 1.09})
    ctx3 = MockCtx(base_factor=0.85)  # 340 < 360 sanity floor, > 300 health floor
    res3 = run_campaign(ctx3, fovea_head=None)
    assert [t for t, _ in ctx3.boots] == ["R0_base_long"], ctx3.boots
    assert res3["ship"]["stopped_early"] is True
    assert res3["arms"]["R0_base_long"]["cells"]["math_nothink_10k"]["n"] == 16
    assert res3["arms"]["R0_base_long"]["cells"]["tool_nothink_10k"]["n"] == 16
    print("   ship:", res3["ship"]["tier"], res3["ship"]["best_cell"])

    print("[9] dry-run health-abort (sick node stops before any 10k cell):")
    ctx4 = MockCtx(base_factor=0.70)  # 280 < 300 health floor
    res4 = run_campaign(ctx4, fovea_head=None)
    assert "math_nothink_10k" not in res4["arms"]["R0_base_long"].get("cells", {})
    assert res4.get("ship") is None or not res4.get("ship")
    print("   health-abort OK")

    srv.shutdown()
    print("SELFTEST PASS")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    elif "--build-prompts" in sys.argv:
        assets = find_assets_dir()
        assert assets, "assets dir with kimi_tiktoken.model required"
        count_fn = load_kimi_count_fn(assets)
        prompts = build_all_prompts(count_fn)
        counts = {d: [count_fn(p) for p in prompts[d]] for d in ("math", "tool")}
        print(json.dumps({"prompt_sha256_10k": prompts_sha256(prompts),
                          "tool16_sha256": tool16_sha256(),
                          "token_counts": counts}, indent=1))
    elif modal is None:
        print("modal not installed; --selftest and --build-prompts are available locally")
