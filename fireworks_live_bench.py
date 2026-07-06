#!/usr/bin/env python3
"""Replay the record protocol against Fireworks' public Kimi-K2.6 endpoints.

Same 16 sha-pinned ~10k-token prompts, 2048-token outputs, temperature 0.6,
streamed. Statistic: per-request decode rate (completion_tokens - 1) /
(t_last - t_first), interpolated median. Prefill and TTFT are excluded by
construction; token arrival is server-paced, so decode rate is not sensitive
to client location.

Usage:
    python fetch_assets.py   # once
    FIREWORKS_API_KEY=... python fireworks_live_bench.py smoke   # 1 request, 256 tokens
    FIREWORKS_API_KEY=... python fireworks_live_bench.py full    # 16 + 8 + 8 requests

Costs a few dollars of API usage. The full run takes 15-30 minutes depending on
the endpoint's decode rate.
"""
import json
import os
import sys
import time
import urllib.request

import record_run as rr

BASE = "https://api.fireworks.ai/inference/v1/chat/completions"
KEY = os.environ.get("FIREWORKS_API_KEY")
if not KEY:
    sys.exit("set FIREWORKS_API_KEY in the environment")


def load_prompts():
    prompts, sha, strict = rr.canonical_prompts()
    assert strict, "tokenizer assets missing; run fetch_assets.py first"
    return prompts["tool"], sha


def measure(model, prompt, max_tokens=2048, extra=None, timeout=420):
    body = {"model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.6,
            "stream": True, "stream_options": {"include_usage": True}}
    if extra:
        body.update(extra)
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.monotonic()
    t_first = t_last = None
    ctok = ptok = None
    finish = None
    with urllib.request.urlopen(req, timeout=timeout) as res:
        buf = b""
        while True:
            chunk = res.read(16384)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:]
                if payload == b"[DONE]":
                    continue
                try:
                    p = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                now = time.monotonic()
                ch = p.get("choices") or []
                if ch:
                    d = ch[0].get("delta") or {}
                    # reasoning deltas count: the statistic covers every generated token
                    if d.get("content") or d.get("reasoning_content"):
                        if t_first is None:
                            t_first = now
                        t_last = now
                    if ch[0].get("finish_reason"):
                        finish = ch[0]["finish_reason"]
                u = p.get("usage")
                if u:
                    ctok = u.get("completion_tokens", ctok)
                    ptok = u.get("prompt_tokens", ptok)
    if t_first is None or t_last is None or t_last <= t_first:
        raise RuntimeError("no token stream observed")
    win = t_last - t_first
    rate = (ctok - 1) / win if ctok else None
    return {"ttft_ms": round((t_first - t0) * 1000, 1),
            "decode_tok_s": round(rate, 1) if rate else None,
            "completion_tokens": ctok, "prompt_tokens": ptok,
            "decode_window_s": round(win, 3), "finish_reason": finish}


def interp_median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return None
    h = (n - 1) * 0.5
    lo = int(h)
    return round(s[lo] + (h - lo) * (s[min(lo + 1, n - 1)] - s[lo]), 1)


def run_arm(name, model, prompts, n, extra=None, max_tokens=2048):
    rows = []
    print(f"\n== {name} ({model}) n={n} ==", flush=True)
    for i in range(n):
        try:
            r = measure(model, prompts[i % len(prompts)], max_tokens=max_tokens, extra=extra)
            r["prompt_idx"] = i % len(prompts)
            rows.append(r)
            print(f"  req {i:2d}: ttft {r['ttft_ms']:8.1f}ms  decode {r['decode_tok_s']:7.1f} tok/s  "
                  f"ctok {r['completion_tokens']}  ptok {r['prompt_tokens']}  {r['finish_reason']}", flush=True)
        except Exception as e:
            rows.append({"prompt_idx": i % len(prompts), "error": str(e)[:200]})
            print(f"  req {i:2d}: ERROR {str(e)[:160]}", flush=True)
        time.sleep(0.6)
    ok = [r for r in rows if r.get("decode_tok_s")]
    summary = {"n_ok": len(ok),
               "decode_median_interp": interp_median([r["decode_tok_s"] for r in ok]),
               "ttft_median_ms": interp_median([r["ttft_ms"] for r in ok])}
    print(f"  -> {name}: median decode {summary['decode_median_interp']} tok/s, "
          f"median ttft {summary['ttft_median_ms']}ms, n_ok {summary['n_ok']}", flush=True)
    return {"name": name, "model": model, "extra": extra, "rows": rows, "summary": summary}


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    prompts, sha = load_prompts()
    meta = {"ts_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "endpoint": BASE, "prompt_sha256_10k": sha,
            "protocol": "record protocol replayed over the public internet: pinned 10k-token "
                        "prompts, 2048 max_tokens, temp 0.6, streamed, "
                        "decode=(ctok-1)/(t_last-t_first), interpolated median",
            "mode": mode}
    arms = []
    if mode == "smoke":
        arms.append(run_arm("smoke-standard", "accounts/fireworks/models/kimi-k2p6",
                            prompts, 1, max_tokens=256))
    else:
        arms.append(run_arm("standard-default", "accounts/fireworks/models/kimi-k2p6", prompts, 16))
        arms.append(run_arm("standard-nothink", "accounts/fireworks/models/kimi-k2p6", prompts, 8,
                            extra={"reasoning_effort": "none"}))
        arms.append(run_arm("turbo-router-default", "accounts/fireworks/routers/kimi-k2p6-turbo",
                            prompts, 8))
    meta["ts_end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    dest = f"fireworks_live_{time.strftime('%Y%m%d')}_{mode}.json"
    json.dump({"meta": meta, "arms": arms}, open(dest, "w"), indent=1)
    print(f"\nwrote {dest}", flush=True)
