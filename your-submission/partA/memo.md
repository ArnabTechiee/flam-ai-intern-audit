# Memo: Indic tokenization cost — corrected (A4)

**Bottom line:** REPORT_v0's "Hindi costs 6× English" is wrong in both directions. On its own
tokenizer (gpt2) the true premium is 7.42×, not 5.89×: its non-parallel 10-line sample understated
Hindi by 36% (4.78× vs 7.42× on the same fixed metric; `extras.py`). On any modern tokenizer it is
~1.1–2.0×. The Indic cost premium is mostly a **tokenizer choice**, not a property of the language.

**Corrected headline (FLORES-200 devtest, 1012 parallel sentences; premium = tokens for the same
sentences ÷ English; 95% CI ≈ ±1%):**

| tokenizer | Hindi | Kannada | Tamil | Telugu | Malayalam |
|---|---|---|---|---|---|
| gpt2 (REPORT_v0's) | 7.42× | 13.58× | 15.54× | 12.97× | 15.16× |
| o200k_base (GPT-4o) | 1.57× | 1.97× | 1.98× | 1.93× | 1.96× |
| Sarvam-1 (Indic) | 1.13× | 1.21× | 1.16× | 1.15× | 1.36× |

**Metric that should drive the decision:** tokens per parallel sentence on the *serving*
tokenizer, as a premium over English. Per-word fertility reverses rankings (MuRIL: Kannada looks
47% worse than Hindi per word, but is cheaper per sentence); per-byte makes Indic look cheaper than
English.

**Recommendation:**
1. Don't budget "6× for Hindi" and don't build a separate Indic route yet.
2. First, measure FLM-4B's own tokenizer with `partA/analyze.py` (≈1 hour). Cost multiplier per
   language = that premium.
3. If FLM-4B's premium is ≤2× (like o200k), keep one route and budget per language. If it's >3× on
   Dravidian languages, evaluate an Indic-tokenizer model (≈1.1–1.4×) — but route only if it
   also matches quality. Token efficiency is not quality.
4. The premium also shrinks the effective context window and raises latency by the same factor.
5. Weight the premium by token type. On our serving stack a generated token costs ~24× the GPU
   time of a prompt token (0.166 ms vs 4.00 ms, `partB/estimates.py`). The premium applies to both
   sides (same tokenizer), but cost ≈ P × c_in + G × c_out with c_out ≈ 24 c_in, so budget per
   language on *reply* tokens: a Hindi reply costs far more than a Hindi prompt of the same length.

**Biggest caveat:** FLM-4B's tokenizer is unmeasured. Also, FLORES is formal text (Wikinews,
Wikijunior, Wikivoyage) translated from English, so it has no casual, code-mixed (Hinglish) or
romanized traffic, and real traffic will tokenize differently.

**One metric to monitor in production:** live **tokens per grapheme, per language**, compared with
the FLORES value for the serving tokenizer. It needs no parallel data. If it drifts more than 15%
(e.g. because of romanized or code-mixed traffic), the corpus no longer represents real traffic and
the premiums above are wrong.