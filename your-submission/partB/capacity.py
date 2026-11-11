import csv

# ---- B1: KV cache from model_spec.md ----
layers, kv_heads, head_dim, bytes_fp16 = 28, 8, 128, 2
kv_per_tok = 2 * layers * kv_heads * head_dim * bytes_fp16     # 2 = K and V
print(f"B1a KV bytes/token = 2*{layers}*{kv_heads}*{head_dim}*{bytes_fp16} = {kv_per_tok} B ({kv_per_tok/1024:.0f} KiB)")
per_seq = kv_per_tok * 4096
budget = 24e9 * 0.92 - 4.2e9 * 2 - 1.6e9                       # usable - fp16 weights - overhead
cap_tok = budget / kv_per_tok
print(f"    per 4096-token seq = {per_seq/1e9:.3f} GB")
print(f"B1b KV budget = 24*0.92 - 8.4 - 1.6 = {budget/1e9:.2f} GB -> {cap_tok:,.0f} tokens -> {cap_tok/4096:.1f} seqs of 4096")
print(f"    (if '24 GB' meant GiB: {(24*2**30*0.92-8.4e9-1.6*2**30)/per_seq:.1f} seqs -- the log's kv_cache_util fits decimal GB)\n")

# ---- check against the log ----
print(f"{'bs':>3}{'P':>5}{'G':>4} {'reported':>9}{'(P+G)N/wall':>12}{'goodput':>9}{'util_pred':>10}{'util_log':>9}{'preempt':>8}{'ITL':>7}{'ITL_pred':>9}")
for r in csv.DictReader(open("starter_kit/bench/bench_log.csv")):
    N, P, G = int(r["num_requests"]), int(r["prompt_len"]), int(r["gen_len"])
    wall = float(r["wall_clock_s"])
    resident = min(N, cap_tok / (P + G))                        # seqs that fit in KV at once
    # decode is memory-bound: each step reads all weights + resident KV (avg context P+G/2)
    itl_pred = (8.4e9 + resident * (P + G / 2) * kv_per_tok) / 300e9 * 1000
    print(f"{N:>3}{P:>5}{G:>4} {r['reported_tok_s']:>9}{N*(P+G)/wall:>12.1f}{N*G/wall:>9.1f}"
          f"{N*(P+G)/cap_tok:>10.2f}{r['kv_cache_util']:>9}{r['preempted_seqs']:>8}{r['itl_ms_p50']:>7}{itl_pred:>9.1f}")

# ---- B3: goodput of batch-24 long-prompt row, two ways ----
N, P, G, wall, rep, ttft, itl = 24, 3584, 512, 61.16, 1607.4, 0.5005, 0.09607
print(f"\nB3 way1 (generated tokens / wall clock)        = {N*G}/{wall} = {N*G/wall:.1f} tok/s")
print(f"B3 way2 (reported_tok_s x G/(P+G))             = {rep}*{G}/{P+G} = {rep*G/(P+G):.1f} tok/s")
print(f"   cross-check (latency: N*G/(TTFT+G*ITL), upper bound) = {N*G/(ttft+G*itl):.1f} tok/s")