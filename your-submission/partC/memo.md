# Decision memo: casual register in 6 Indic languages

**Recommendation:** (c) prompt-engineering as the launch path, staged so it produces the data for
(a) a LoRA SFT pass if prompting falls short. Launch casual mode for **Hindi + Kannada only** (the
languages we can human-verify); keep the other 4 behind a flag until a native speaker reviews them.
Reject (b): it adds a second model to every reply, and a ≤1B model is weakest in exactly the
low-resource languages we care about.

**Assumptions**
- The main model has open weights (FLM-4B-class), so LoRA SFT is possible.
- "Casual" means a colloquial register (natural code-mixing allowed where natives would use it),
  not new knowledge. That's a style change, which suits prompting/SFT.
- Representative user prompts are available for evaluation.
- No external API, so all generation runs on the A100.

**Back-of-envelope**
- **Reviewer (binding constraint):** I timed myself on 10 blind A/B pairs (`partC/rate_timer.py`):
  median 17.9 s per judgment (one-line pairs). Real replies are ~3–5× longer → ~1 min per judgment,
  a lower bound. 10 h/wk × 3 wk = 30 h, ~25 h usable → ~1,500 judgments. One eval round =
  100 items × 2 languages ≈ 3.5 h → only ~5–6 rounds. Eval time is the bottleneck.
- **GPU (not a bottleneck):** SFT data 5k pairs × 6 langs = 30k × ~400 tokens = 12M tokens/epoch.
  LoRA ≈ 4 × 4.2B FLOPs/token → 312 TFLOPS × 40% MFU ≈ 7.4k tok/s → ≈ 27 min/epoch (full-FT 6N
  bound: 40 min). Generating 9M tokens with our model ≈ 24 min (`partB/estimates.py`).
- **A stronger data generator for (a):** a 70B-class open model at 4-bit (~35 GB) fits on the A100:
  ~27 ms/step at batch 32 ≈ 1.2k tok/s → 9M tokens ≈ 2.1 h (`partB/estimates.py`). That removes
  "our own too-formal model" as the data source, but not the real blocker: nobody can verify its
  Tamil/Telugu/Bengali/Marathi output, and filtering pairs spends the same reviewer hours.
- **Serving cost of (c):** a ~600-token style prompt → +600 × 114,688 B ≈ 69 MB KV per sequence
  (Part B), ~+15% KV for a 4k request → ~13% less concurrency. SFT removes this; that's the
  reason to graduate to (a).
- **Serving cost of (b):** +1 model on every reply: ~2 GB read per token → ~2–3 s per 300-token rewrite.

**Success metric:** in a blind human A/B against current outputs (n = 100 per language),
casual-preferred win-rate **≥ 70%** for Hindi and Kannada, with replies rated "meaning changed or
wrong" **≤ 5%**. At n = 100 the CI is about ±9 points, so 70% is clearly above chance.

**Kill criterion:** if by **day 5** the best prompt variant has a win-rate < 60% in either language
or a meaning-error rate > 10%, abandon prompt-only and switch to LoRA SFT on reviewer-filtered
outputs (best prompt + 70B teacher). If SFT doesn't beat the prompt by ≥ 10 points by **day 12**,
ship the best prompt for Hindi/Kannada and stop.

**Day-1 experiment:** 50 real prompts × {Hindi, Kannada} × 5 conditions (current; instruction only;
instruction + 3 reviewer-written casual examples; examples + "write like a message to a friend";
the 70B teacher casualizing the current reply). Reviewer ranks blind (~2 h) while I log tokens per
reply. Output: does a casual register exist in Kannada at all, and what does each variant cost?

**Biggest risk:** Tamil, Telugu, Bengali and Marathi can't be verified by our reviewer. An
LLM-judge calibrated on Hindi/Kannada can screen them, but it isn't a launch gate.