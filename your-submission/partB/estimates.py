import csv
KV = 2 * 28 * 8 * 128 * 2            # bytes/token, fp16 (B1)
W, BW = 8.4e9, 300e9                  # weights bytes, L4 bandwidth
BUDGET = 24e9 * 0.92 - W - 1.6e9

def itl_model(resident, P, G, kv_bytes):   # ideal ms per decode step (memory-bound)
    return (W + resident * (P + G / 2) * kv_bytes) / BW * 1000

# ---- 1. calibrate on the log's long-prompt rows with no preemption ----
rows = [r for r in csv.DictReader(open("starter_kit/bench/bench_log.csv"))
        if r["prompt_len"] == "3584" and r["preempted_seqs"] == "0"]
print("calibration (3584-prompt rows, no preemption):")
print(f"{'N':>3}{'ITL':>8}{'ideal':>7}{'eff':>6}{'wall':>7}{'wall-G*ITL':>11}{'per req':>8}")
effs, pre = [], []
for r in rows:
    N, G, itl, wall = int(r["num_requests"]), int(r["gen_len"]), float(r["itl_ms_p50"]), float(r["wall_clock_s"])
    ideal = itl_model(N, 3584, G, KV)
    rest = wall - G * itl / 1000                 # time not explained by decode = prefill etc.
    effs.append(ideal / itl); pre.append(rest / N)
    print(f"{N:>3}{itl:>8.2f}{ideal:>7.1f}{ideal/itl:>6.2f}{wall:>7.2f}{rest:>11.2f}{rest/N:>8.2f}")
eff, t_pre = sum(effs) / len(effs), sum(pre) / len(pre)
print(f"-> bandwidth efficiency {eff:.2f}, prefill ~{t_pre:.2f} s per 3584-token request (TTFT p50 ~0.50 s)")

# ---- 2. predict bs48 with FP8 KV:  wall = G*ITL + N*t_prefill ----
N, P, G, kv8 = 48, 3584, 512, KV // 2
assert N * (P + G) <= BUDGET / kv8, "doesn't fit"
itl = itl_model(N, P, G, kv8) / eff
wall = G * itl / 1000 + N * t_pre
print(f"\nFP8 KV: {kv8} B/token, capacity {BUDGET/kv8/4096:.1f} seqs -> bs48 fits (util {N*(P+G)*kv8/BUDGET:.2f})")
print(f"predicted ITL {itl:.1f} ms, wall {wall:.1f} s, goodput {N*G/wall:.0f} tok/s  (measured fp16 bs48: 162 tok/s)")
print("caveats: ignores FP8 dequant overhead and quality loss; a prediction, falsified by one bs48 fp8 run")

# ---- 3. Part C GPU numbers from first principles (A100-80GB) ----
params, peak, mfu = 4.2e9, 312e12, 0.40          # bf16 dense peak; 40% MFU is a typical assumption
for name, flops_tok in [("LoRA (fwd+bwd, frozen weights) ~4N", 4 * params), ("full fine-tune ~6N (upper bound)", 6 * params)]:
    tps = peak * mfu / flops_tok
    print(f"\n{name}: {tps:,.0f} tok/s -> 12M-token epoch = {12e6/tps/60:.0f} min")
step_ms = (2 * params + 64 * 700 * KV) / 2.0e12 * 1000 / eff   # batch 64, ~700-token contexts, 2 TB/s
print(f"vLLM generation on A100, batch 64: ~{step_ms:.1f} ms/step -> {64/step_ms*1000:,.0f} tok/s -> 9M tokens in {9e6/(64/step_ms*1000)/60:.0f} min")