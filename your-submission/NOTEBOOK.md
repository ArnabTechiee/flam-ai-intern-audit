# Lab Notebook — FLAM AI Intern Assignment ("The Audit")

Chronological log: hypothesis → experiment → result → revision. Dead ends included.
Environment: Windows, Python 3.13, venv, tiktoken + transformers (no PyTorch — tokenizers only).

---

## 2026-11-10 — Day 1

### 20:30 — Reading the starter kit
First pass over REPORT_v0.md, fertility.py, the sample corpora and the bench files.
Things I flagged to investigate (not yet claims, just suspicions):
- The task says the corpora are "parallel line-by-line". Line 1 English is about an airport
  ("Bengaluru International Airport handled record traffic in March."), line 1 Hindi is
  "मुझे सुबह की चाय बहुत पसंद है।" ("I like morning tea a lot"). Not the same sentence.
- `line.split(" ")` — fragile on double spaces. Both files contain one double-spaced line
  ("books  in", "किताबें  अलमारी").
- `line.lower()` — Hindi has no case, so this only touches English.
- Per-line ratios are averaged (mean of ratios), not total/total.
- `len(line)` counts Unicode code points, not visible characters.
- `unicodedata.normalize("NFC", ...)` with comment "just in case" — looks suspicious, but NFC is
  the standard normalization. Suspect this is the "looks wrong but is fine" item. Need to measure.
- `random.seed(1337)` — `random` is never used. Dead code, harmless.
- Report claims "tok/char agrees, so the result is robust" — both metrics share the same
  numerator and the same corpus, so agreement isn't independent confirmation.

### 21:45 — Experiment 1: can I reproduce the report?
**Hypothesis:** the report's numbers come from this exact script and are reproducible.
**Command:**
`python fertility.py --corpus eng=corpus_sample/eng_sample.txt --corpus hin=corpus_sample/hin_sample.txt --tokenizer gpt2`
**Result:**

| lang | fertility | tok/char |
|---|---|---|
| eng | 1.27 | 0.226 |
| hin | 7.45 | 1.579 |

"hin is 5.89x the fertility of eng" — **exact match with REPORT_v0.** Baseline established;
every later number is a delta from this.

### 22:12 — Experiment 2: why does GPT-2 give Hindi so many tokens?
**Hypothesis:** GPT-2's byte-level BPE has almost no Devanagari merges, so it falls back to bytes.
**Command:** `python your-submission/partA/inspect_tokens.py`
**Result:**
- `'The train arrived exactly on time.'` → 34 bytes, **7 tokens** (whole words: `' train'`, `' arrived'`).
- `'ट्रेन ठीक समय पर पहुँची।'` → 24 code points, 64 bytes, **40 tokens**. Every piece is a
  byte fragment (`b'\xe0\xa4'`, `b'\x9f'`, ...) — each Devanagari code point (3 UTF-8 bytes)
  becomes 2 tokens. Zero multi-character Hindi merges.
- `'NASA and ISRO announced'` → 5 tokens; `'nasa and isro announced'` → **6 tokens**
  (`NASA` = 1 token, `nasa` = `n` + `asa`).

**Conclusion:** (1) The Hindi blow-up is a property of *this tokenizer's vocabulary*, not of the
script — contradicts REPORT_v0 finding 3. To be confirmed with Indic-aware tokenizers (Exp 5).
(2) Lowercasing changes English token counts, so `.lower()` is a real, asymmetric distortion.

### 23:37 — Experiment 3: isolate each suspected flaw (A2)
**Method:** `audit.py` re-implements the metric and toggles exactly ONE change vs the original
per row, on the same sample corpora with gpt2.

**Dead end:** first run crashed with
`UnicodeEncodeError: 'charmap' codec can't encode characters` when printing Hindi.
Cause: Windows console defaults to cp1252. Fix: `$env:PYTHONIOENCODING="utf-8"`.
The table printed fine before the crash, so the numbers below are valid.

**Command:** `python your-submission/partA/audit.py > your-submission/partA/audit_output.txt`
**Result:**

| variant (one change vs original) | eng f | hin f | ratio | eng t/c | hin t/c | t/c ratio |
|---|---|---|---|---|---|---|
| 0 original | 1.265 | 7.448 | 5.89 | 0.226 | 1.579 | 7.00 |
| 1 `split()` fix | 1.283 | 7.598 | 5.92 | 0.226 | 1.579 | 7.00 |
| 2 no lowercase | 1.229 | 7.448 | 6.06 | 0.220 | 1.579 | 7.19 |
| 3 ratio of totals | 1.253 | 7.403 | 5.91 | 0.221 | 1.583 | 7.16 |
| 4 grapheme denominator | 1.265 | 7.448 | 5.89 | 0.226 | 2.363 | 10.47 |
| 5 NFC removed | 1.265 | 7.448 | 5.89 | 0.226 | 1.579 | 7.00 |
| 6 fixes 1+2+3+4 | 1.231 | 7.525 | 6.11 | 0.214 | 2.354 | 10.98 |

**Reading it:**
- Row 1: `split(" ")` creates an empty "word" on double spaces → word count too high → fertility
  too low (eng −1.4%, hin −2.0%). Ratio moves 5.89 → 5.92. Real bug, small effect.
- Row 2: lowercasing *inflates* English by ~3% (1.229 → 1.265) and leaves Hindi untouched →
  ratio understated, 6.06 → 5.89. Real bug, asymmetric.
- Row 3: mean-of-ratios vs total/total — 5.89 vs 5.91. Real statistical issue, negligible here.
- Row 4: switching "char" from code points to grapheme clusters moves the tok/char ratio
  **7.00 → 10.47 (+50%)**. The report's "tok/char confirms tok/word" rests on an arbitrary
  definition of character.
- Row 5: removing NFC → **zero change** on every number. NFC is the harmless one.

**Revision / surprise:** I expected the code bugs to explain a lot of the report's error. They
don't — each moves the headline by <3%, and all of them push the ratio *down* (they made Hindi
look slightly better). The big problems must be conceptual (denominator) and the corpus itself.

## 2026-11-11 — Day 2

### 9:15 — Experiment 4: build a real parallel corpus (A1)
**Choice:** FLORES-200 devtest (1012 sentences, professionally translated, same sentence on
line N in every language). Languages: eng_Latn, hin_Deva, kan_Knda, tam_Taml, tel_Telu,
mal_Mlym + ben_Beng, mar_Deva (Part C languages; Marathi shares Hindi's script = useful control).
**Source:** https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz (24.4 MB).
**Preprocessing:** strip whitespace, NFC-normalize. Nothing else (no lowercasing, no filtering).
**Command:** `python your-submission/partA/prepare_corpus.py`
**Result:**

| lang | lines | empty | non-NFC lines | words | graphemes | bytes |
|---|---|---|---|---|---|---|
| eng_Latn | 1012 | 0 | 0 | 21901 | 131966 | 132096 |
| hin_Deva | 1012 | 0 | 93 | 25643 | 90733 | 337439 |
| kan_Knda | 1012 | 0 | 7 | 16100 | 90371 | 375341 |
| tam_Taml | 1012 | 0 | 2 | 16775 | 99724 | 421635 |
| tel_Telu | 1012 | 0 | 1 | 16938 | 84771 | 353661 |
| mal_Mlym | 1012 | 0 | 4 | 14930 | 95856 | 411741 |
| ben_Beng | 1012 | 0 | 609 | 19506 | 89260 | 348729 |
| mar_Deva | 1012 | 0 | 0 | 19046 | 89255 | 355511 |

parallel: True.

**Observations:**
- Same meaning, very different word counts: Hindi 25,643 vs English 21,901 vs Malayalam 14,930.
  Hindi writes postpositions as separate words (में, की); Dravidian languages agglutinate. So
  "per word" divides by something grammar-dependent, not meaning-dependent.
- **Surprise / revision on NFC:** on the toy sample NFC was a no-op, but on FLORES 93 Hindi and
  609 Bengali lines are *not* NFC. So NFC isn't just harmless — it's necessary on real data.
  OPEN: I haven't measured how many tokens NFC changes on FLORES.

### 9:43 — Experiment 5: corrected analysis (A3)
**Hypothesis:** with a parallel corpus and a meaning-preserving denominator, (a) the Hindi
premium depends heavily on the tokenizer, and (b) per-word fertility distorts cross-language
comparisons.
**Method:** `analyze.py` — 5 tokenizers × 8 languages, 4 denominators (word, grapheme, byte,
sentence). Premium = total tokens(lang) / total tokens(eng) on the same 1012 sentences.
95% CI from a paired bootstrap (1000 resamples of sentence indices, same indices for both langs).
**Command:** `python your-submission/partA/analyze.py > your-submission/partA/analysis_output.txt`

**Result — premium vs English (tokens for the same sentences):**

| lang | gpt2 | o200k_base | xlm-roberta | MuRIL | Sarvam-1 |
|---|---|---|---|---|---|
| hin | 7.42 | 1.57 | 1.25 | 1.16 | 1.13 |
| kan | 13.58 | 1.97 | 1.35 | 1.07 | 1.21 |
| tam | 15.54 | 1.98 | 1.35 | 1.06 | 1.16 |
| tel | 12.97 | 1.93 | 1.32 | 1.20 | 1.15 |
| mal | 15.16 | 1.96 | 1.38 | 1.18 | 1.36 |
| ben | 9.61 | 1.71 | 1.37 | 1.00 | 1.12 |
| mar | 7.86 | 1.82 | 1.22 | 1.06 | 1.07 |

All 95% CIs are about ±1% (e.g. gpt2 hin [7.33, 7.51]). unk%: 0 everywhere except MuRIL
Telugu 0.38% and English 0.02%. Full table incl. tok/word, tok/grapheme, tok/byte in
analysis_output.txt / results.csv.

**Findings:**
1. **Per-word fertility flips the ranking.** MuRIL: Kannada 1.82 tok/word vs Hindi 1.24 →
   "Kannada 47% worse". Per parallel sentence: Kannada 1.07× vs Hindi 1.16× → Kannada is
   *cheaper*. Same tokenizer, same content, opposite conclusion.
2. **Per-word distorts in both directions.** gpt2 Hindi per-word ratio = 7.83/1.23 = 6.4×,
   true sentence premium 7.42× (per-word *understates* Hindi, because Hindi uses more words).
   Kannada per-word = 22.82/1.23 = 18.6× vs premium 13.58× (per-word *overstates*).
   — Surprise: I expected per-word to overstate every Indic language. It doesn't.
3. **Per-byte inverts reality:** every Indic language looks *better* than English per byte
   (MuRIL hin 0.095 vs eng 0.209), purely because Indic code points are 3 UTF-8 bytes.
4. **Tokenizer, not script.** Hindi premium 7.42× (gpt2) → 1.57× (o200k) → 1.13× (Sarvam).
   Marathi (same Devanagari script) 7.86× → 1.06× (MuRIL). REPORT_v0 finding 3 is falsified.
5. **REPORT_v0's "6× cost" is wrong both ways:** on gpt2 itself the true Hindi premium is 7.4×
   (not 5.89×) and Dravidian is 13–15×; on any modern tokenizer it's ~1.1–2.0×.
6. **Sampling noise is not the problem** (CIs ±1%). Domain is (FLORES = formal Wikipedia-style).

**Decision for A3:** the number that should drive routing/cost is **tokens per parallel
sentence on the production tokenizer**, expressed as a premium over English. It is the only
denominator that holds meaning constant, and cost = tokens × price/token. Comparing *across*
models needs absolute tokens/sentence × each model's price, since each tokenizer's English
baseline differs (Sarvam eng 31.0 vs gpt2 26.7 tokens/sentence).

**Biggest open gap:** our serving model FLM-4B (128k vocab) — its tokenizer was never measured,
by the old intern or by me. That's the number leadership actually needs.

**OPEN / TODO:**
- Measure NFC's token effect on FLORES Hindi/Bengali (raw vs NFC).
- Run analyze.py on FLM-4B's tokenizer if available.
- Casual / Hinglish / romanized text is not covered by FLORES.

---

## Part B — Capacity reconciliation
### 11:46 — Experiment 6: capacity from spec vs log (B1–B3)
Hypothesis: (a) reported_tok_s includes prompt tokens; (b) the long-prompt throughput drop is KV exhaustion.
Command: python your-submission/partB/capacity.py
Result: KV = 114,688 B/token, 25.7 seqs. Predicted util matches kv_cache_util on all 13 rows;
reported_tok_s = N(P+G)/wall on all 13 rows → (a) confirmed. bs24 goodput = 200.9 tok/s.
Preemption starts exactly where predicted util > 1 → (b) supported.
Surprise: a simple memory-bandwidth model predicts ITL at a constant ~65% of measured on every row.
Dead end: first assumed "24 GB" = GiB → 28.9 seqs, which doesn't match the log's util; decimal GB fits.
Oddity: e2e_p95 > wall_clock in 12/13 rows — can't explain, flagged rather than claimed.

### 12:34 — Experiment 7: quantify the corpus effect + NFC on real data
Command: python your-submission/partA/extras.py
Result 1: same fixed metric, gpt2: starter sample premium 4.78× vs FLORES 7.42× → the non-parallel
sample understated Hindi cost by 36%. 10-sentence non-parallel draws: 90% range [5.64, 9.56] vs
parallel [6.68, 8.16] (2.6× wider). Surprise: 4.78 is below even the non-parallel range, so the
sample's English is systematically heavier (acronyms, proper nouns), not just randomly mismatched.
Per-word ratio hides this (6.11 vs 6.34).
Result 2: NFC on raw FLORES changes tokens by +0.11%/−0.07% (Hindi), +0.51%/+0.92% (Bengali).
Direction depends on tokenizer. <1% effect → NFC stays "fine"; it's kept for determinism.
Hypothesis (untested): NFC decomposes composition-excluded letters (e.g. U+09DF) → more code points.

### 13:50 — Experiment 8: measure reviewer time per judgment (Part C)
Hypothesis: a blind A/B judgment takes ~1 min (my memo's assumption).
Method: rate_timer.py, 10 formal/casual Hindi pairs (AI-drafted), randomized A/B side, rated by me
(not the native reviewer).
Result: median 17.9 s, mean 26.1 s, max 83.2 s (first-pair warm-up). Picked casual 10/10, meaning y 10/10.
Revision: one-line pairs → real replies ~3–5× longer → ~1 min is reasonable as a lower bound.
Limitation: pairs were obviously different (10/10 agreement), so this measures reading time,
not how hard the decisions are on subtle model outputs.

### 16:43 — Experiment 9: test the NFC hypothesis
Hypothesis (from Exp 7): NFC adds tokens because it decomposes composition-excluded letters.
Command: python your-submission/partA/nfc_check.py
Result: Hindi +122 code points: precomposed nukta letters (U+095C ड़, U+095B ज़, U+095E फ़) →
base + U+093C NUKTA. Bengali +1,349 (+1.0%): U+09DF য় −1225 → U+09AF + U+09BC; U+09DC ড় −147.
Hypothesis CONFIRMED. Surprise: NFC also *composed* some Bengali vowel signs (ে+া → ো, +22),
so NFC isn't "always shorter" or "always longer". It's canonical. Keeping it is still correct.

### 17:15 — Experiment 10: turn estimates into calibrated predictions
Command: python your-submission/partB/estimates.py
Wall-time model fitted on the 4 non-preempted long-prompt rows: wall − G×ITL ≈ 0.50–0.67 s per
request ≈ TTFT of one prefill → wall ≈ G×ITL + N×0.59 s. Bandwidth efficiency 0.64–0.67 on all rows.
FP8 bs48 prediction: ITL 96.4 ms, wall 77.9 s → goodput ~315 tok/s (~1.9× vs 162 measured).
Revision: my earlier ~350 tok/s guess underestimated prefill time. Still a prediction, not a measurement.
Part C GPU from FLOPs (A100, 40% MFU): LoRA ~7.4k tok/s → 27 min/epoch (6N bound: 40 min);
generation ~24 min. My memo's 4k tok/s / 1–2 h were conservative.

### 19:38 — Pre-submission audit + reproducibility re-run
Checked my own repo before pushing. Found: (1) all *_output.txt files were UTF-16 (PowerShell `>`
default), unreadable on GitHub; (2) scripts pointed at "starter_kit (1)/starter_kit".
Fixed paths, moved starter_kit to the repo root, re-ran all 7 scripts with `Out-File -Encoding utf8`.
Result: all numbers identical (audit row 0 = 5.89; capacity = 25.7 seqs).