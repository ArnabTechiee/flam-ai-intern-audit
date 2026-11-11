# AI usage

**Tool:** Claude (Anthropic), used throughout as a pair-analyst. I ran every command on my own
machine, and every number in this repo comes from those runs (outputs saved as *_output.txt).

**Where AI helped**
- First read of the starter kit: listing suspicious lines in fertility.py and the bench columns.
- Writing the scripts: inspect_tokens.py, audit.py, prepare_corpus.py, analyze.py, extras.py,
  nfc_check.py (Part A); capacity.py, estimates.py (Part B); make_pairs.py, rate_timer.py (Part C).
  I ran each one and checked its output against the report, the log, or the previous results.
- The KV-cache arithmetic, the memory-bandwidth ITL model, and the FLOPs estimate for LoRA.
- The 10 formal/casual Hindi pairs in partC/pairs.tsv were drafted by AI; the ratings and timing
  are mine.
- First drafts of the memos, CORPUS.md and the notebook, which I then edited.

**Where AI was wrong or overconfident**
- It predicted the code bugs would explain much of the report's error. Measured: each is < 3%.
  The real problems were the non-parallel corpus and the per-word denominator.
- It expected per-word fertility to overstate every Indic language. Measured: it *understates*
  Hindi (6.4× vs a true premium of 7.42×).
- Its first FP8 estimate (~350 tok/s) underestimated prefill time. Calibrating on the log
  (estimates.py) gave ~315 tok/s.
- It first framed the "two ways" of deriving goodput as independent. They're algebraically linked:
  their agreement confirms that reported_tok_s includes prompt tokens. I added a latency-based
  cross-check (247 tok/s, an upper bound).
- Its scripts crashed on Windows (cp1252 console vs Hindi text); fixed with PYTHONIOENCODING=utf-8.
- Its first guess at why NFC changes token counts was untested; I tested it with nfc_check.py
  (confirmed: NFC decomposes precomposed nukta letters such as U+09DF).

**Parts I understand fully**
- The denominator argument: why per-word fertility misleads (Kannada/Hindi ranking flip) and why
  premium per parallel sentence holds meaning constant.
- Why GPT-2 tokenizes Hindi so badly (byte-level BPE trained on English; 3-byte UTF-8 characters).
- What NFC does and why it's correct to keep (measured: 0 on sample, <1% on FLORES).
- Each A2 flaw and its measured effect from audit.py.
- The KV-cache arithmetic (114,688 B/token, 25.7 sequences) — I can derive it by hand.
- Why reported_tok_s includes prompt tokens and how goodput (200.9 tok/s) is computed.
- The Part C reasoning: reviewer time is the bottleneck, not the GPU.

**Parts I understand less well**
- The bootstrap: I understand that it resamples sentences to get a confidence interval, but I'd
  need to look up why percentiles of resampled ratios give a valid 95% interval.
- The memory-bandwidth ITL model: I follow the formula, but the 0.66 efficiency is a fitted
  constant and I can't explain where the remaining 34% goes.
- The FLOPs estimate for LoRA (4N vs 6N): I'm relying on the standard rule of thumb rather than
  deriving the backward-pass cost myself.
- The FP8 prediction: I understand the steps, but not how FP8 dequantization overhead would
  change it.
- Why kv_cache_util caps at 0.97 instead of 1.00 — I don't know vLLM's scheduler internals.