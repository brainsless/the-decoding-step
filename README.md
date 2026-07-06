# The Decoding Step — record runners

Benchmark code behind Brainsless Research Lab technical report BRL-2026-11:
505.9 tokens per second, single stream, lossless, on Kimi-K2.6 (1T MoE) from four
B200 GPUs, measured at the public leaderboard's workload.

Report: https://brainsless.com/the-decoding-step.html

Every table in the report rebuilds from a runner in this directory and writes the
same artifact schema we published.

## Run the record yourself

Requirements: a [Modal](https://modal.com) account, about $15 of compute credit,
Python 3.10+.

```
pip install modal tiktoken
modal setup
python fetch_assets.py            # tokenizer files from the Kimi-K2.6 HF repo, sha-verified
python stage600_r.py --selftest   # rebuilds the 16 prompts, asserts the pinned sha. no GPU, no cost
modal run download_model.py       # first run only: nvidia/Kimi-K2.6-NVFP4 (~554 GiB) into a Modal volume
modal run stage600_r.py
```

The Fovea probe arm reads a private checkpoint volume and skips itself when the
volume is absent. The record arms use the public draft head and run either way.

`modal run stage600_r.py` rents a 4xB200 node (~$24/hr, about 40 minutes end to end),
serves Kimi-K2.6 NVFP4 on SGLang v0.5.14 with the public
`lightseekorg/kimi-k2.6-eagle3.1-mla` draft head, and runs the record protocol:
16 novel ~10k-token prompts, 2048-token outputs, temperature 0.6, per-request
streaming decode rate `(ctok-1)/(t_last-t_first)`, interpolated median per cell.
Output is an artifact JSON with every request's raw text, timing, and engine counters.

Expected result: node draws move single-stream throughput 5-10%. The runner prints an
anchor cell against a pinned reference so you can read your draw. The depth-7 tool
cell landed in the 486-506 tok/s band across the draws we measured. The record read
505.9 on 2026-07-05 (`artifacts/brl11_stage600r.json`).

Replications from this repository on 2026-07-06 (three runs, three node draws, all
cells within the node-draw envelope): depth-6 tool 511.6 / 484.4 / 474.2, depth-7
486.5 (first-eight median 538.0), math 405-422, individual requests to 568.0. Artifacts:
`artifacts/brl11_repl_r1.json`, `artifacts/brl11_repl_cleanroom.json`,
`artifacts/brl11_repl_fovea.json`.

### What a healthy run looks like

Do not kill the run because it looks quiet. The timeline:

1. node boot and image pull: 1-3 min
2. engine load: 5-15 min. You will see `Multi-thread loading shards: N/60` advancing
   at roughly 5s per shard. This is normal, not a hang.
3. cells print as they complete (health, then tool n=16, then math n=16)
4. the engine reloads for the depth-7 arm and loads shards again. A second long load
   is expected, not a crash.
5. verdict line, artifact written. Total 35-50 min.

If you run from CI, an agent harness, or any shell that enforces command timeouts,
always launch with `--detach`. An attached `modal run` client that gets killed (by a
timeout or a closed terminal) cancels the run with it — the logs will show
"Received a cancellation signal" during a healthy boot.

If you launch with `--detach` (survives your terminal closing), note the app id
(`ap-...`) printed at launch. Logs and stop work by id only for detached runs:
`modal app logs <app-id>`, `modal app stop <app-id> --yes`. Name-based lookup
returns "No App found" for detached runs even while they are running.

Only stop a run if you see a repeating Python traceback across boot retries or the
same boot phase restarting three or more times. The runner stops itself at its $45
hard cap.

## Files

| file | measures |
|---|---|
| `record_run.py` | protocol library: prompt construction (sha-pinned), Kimi tokenizer, timing, artifact schema |
| `stage600_r.py` | the record session: SGLang, depths 6-7, tool and math cells, n=16 |
| `stage600_sg.py` | SGLang depth ladder smoke (3/5/6) |
| `stage600_a.py` | vLLM k-ladder and the no-speculation baseline (T1 = 6.6 ms) |
| `stage600_e1.py` | verify-cost differential: weights-free proposer at two draft widths (0.397 ms/token) |
| `stage600_b.py` | 8xB200 tensor-parallel scaling null |
| `stage600_d.py` | depth-8 ladder cell |
| `fireworks_live_bench.py` | the record protocol replayed against a provider's production API. needs `FIREWORKS_API_KEY` |
| `artifacts/` | the JSON artifacts behind every table in the report |

## Protocol notes

Prompts are built deterministically and hashed. Every runner fail-closes if the
SHA-256 drifts from the pin. Tokens are Kimi-native (measured o200k parity
1.0035-1.0081 on stored outputs). Losslessness is the standard speculative-decoding
guarantee: the target verifies every drafted token and the sampled distribution is
its own. Acceptance cross-validates across two engines at equal depth
(tau 4.825 SGLang / 4.815 vLLM).

Serving gotchas that cost us dead sessions:

- CUDA-graph capture sizes must include multiples of (1+k) or speculative shapes run uncaptured
- the NVFP4 checkpoint ships no chat template
- the SGLang draft head must be pinned to unquantized loading or it silently inherits the target's FP4 config
- vLLM's CPU ngram proposer fails config validation under async scheduling. use `ngram_gpu`
- warm engine reloads (390-590s vs 1540s cold) are what make multi-arm sessions affordable

## License

Code: noncommercial license, see LICENSE. The Kimi-K2.6 tokenizer files are downloaded
from Moonshot AI's repository by `fetch_assets.py` and remain under their license.
