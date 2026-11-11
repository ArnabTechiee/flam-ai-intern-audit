# Part B — Capacity reconciliation
All numbers: `python partB/capacity.py` → capacity_output.txt; predictions: `python partB/estimates.py` → estimates_output.txt

## B1
(a) KV bytes/token = 2 (K,V) × 28 layers × 8 KV heads × 128 head_dim × 2 B (fp16) = **114,688 B (112 KiB)**.
    GQA matters: with 24 KV heads (no GQA) it would be 3× larger.
(b) KV budget = 24 GB × 0.92 − 8.4 GB weights (4.2B × 2 B) − 1.6 GB overhead = 12.08 GB
    → 12.08e9 / 114,688 = 105,329 tokens → **≈25.7 concurrent 4096-token sequences (25 in practice)**.
Check vs log: predicted KV utilization = N×(P+G)/105,329 matches `kv_cache_util` on every row
(bs24 long: 0.93 pred / 0.93 log; bs64 short: 0.47/0.47). Above 25.7 seqs, prediction exceeds 1.0
(1.24, 1.87) while the log saturates at 0.97 and `preempted_seqs` becomes nonzero (7, 23) —
exactly where the KV cache runs out. (Using GiB would give 28.9 seqs and wouldn't match the log.)

## B2 — anomaly in the 3584-prompt sweep
Expectation: more batch → more throughput. Observed: bs24 → 32 → 48 gives reported_tok_s
1607 → 1384 → 1298 and goodput 200.9 → 173.0 → 162.3 (falls).
Mechanism: KV cache exhaustion → preemption.
- bs24 needs 0.93 of KV (fits); bs32 needs 1.24, bs48 needs 1.87 (don't fit).
- `kv_cache_util` pins at 0.97, `preempted_seqs` 0 → 7 → 23.
- `itl_ms_p50` plateaus at ~100 ms (96.07 → 101.79 → 100.0): only ~25 sequences can be resident,
  so per-step work stops growing. My memory-bandwidth model (weights + resident KV read per
  step / 300 GB/s) predicts ITL at a constant ~65% of measured on all 13 rows, and predicts the
  plateau when resident seqs cap at 25.7.
- `ttft_ms_p50` 500 → 637 → 955: requests wait for KV space; preempted sequences lose their KV
  and must redo prefill of up to ~4k tokens (wasted compute).
- `wall_clock_s` 61 → 95 → 151: bs48 takes ~2.5× the bs24 time for 2× the work.

Change: **FP8 KV cache** (`kv_cache_dtype=fp8`; L4/Ada supports FP8).
Prediction (calibrated on the log, `estimates.py`): 57,344 B/token → 51.4 seqs → bs48 fits
(util 0.93, 0 preemptions). ITL ≈ 96 ms (same KV bytes per step as bs24-fp16); wall ≈ G×ITL +
N×0.59 s prefill ≈ 78 s → goodput **≈ 315 tok/s vs 162 today (~1.9×)**. Ignores FP8 dequant
overhead; one bs48 FP8 run would confirm or falsify it. FP8 can cost quality, so evals must be re-run.
Fallback with no quality risk: cap `max_num_seqs=24` → no preemption; 48 requests run as 2 waves
≈ 2 × 61 s = 122 s → 201 tok/s vs 162 (+24%), at the cost of ~61 s queueing for the second wave.

## B3 — the misread column
`reported_tok_s` counts **prompt + generated** tokens: it equals N×(P+G)/wall_clock to 0.1 tok/s
on all 13 rows. For long prompts, 3584/4096 = 87.5% of "throughput" is prompt tokens that were
only read, not generated.
Goodput (generated tok/s), bs24 long-prompt row:
- Way 1 (wall clock): 24 × 512 / 61.16 s = **200.9 tok/s**
- Way 2 (latency columns only, independent of wall_clock and reported_tok_s): the run's time
  budget is decode G×ITL = 512 × 0.09607 = 49.19 s plus prefill N×TTFT = 24 × 0.5005 = 12.01 s,
  totalling 61.20 s (logged wall: 61.16 s) → 12,288 / 61.20 = **200.8 tok/s**.
- Consistency check: reported_tok_s × G/(P+G) = 1607.4 × 512/4096 = 200.9 — this one is
  algebraically tied to Way 1, so it confirms the column definition rather than the goodput.
Way 2 also shows prefill isn't overlapped with decode: ~20% of the run (12 s) is spent prefilling.
Both report conclusions fail:
1. "Longer prompts give better throughput": at bs16, goodput short = 294.5 vs long = 163.9 tok/s.
   Long prompts are **44% worse**.
2. "bs48 → ~3200 tok/s": bs48 is already in the log: 1298.5 reported / **162.3 goodput**.
   Linear scaling ignores the ~25-sequence KV ceiling.
What it should have said: "At 4k context one L4 sustains ~200 generated tok/s, peaking at
~24 concurrent requests; beyond that, preemption reduces throughput. Plan capacity in
generated tok/s under a KV-cache concurrency limit."

## B4
Pull vLLM's `vllm:num_requests_running` gauge (with `vllm:num_requests_waiting`) during the
bs48 run. Expected: running plateaus at **~25** (never 48), waiting ≈ 23, and
`vllm:num_preemptions_total` increases. If running reached 48 with no waiting, my mechanism is wrong.

## Data oddity (flagged, not claimed as a flaw)
`e2e_ms_p95` exceeds `wall_clock_s` on 12 of 13 rows (e.g. bs1: 12.86 s vs 10.94 s), which is
impossible if both are measured from the same start. The two timers likely use different start
points — I'd ask the harness owner before relying on e2e.