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
- No external API, so any synthetic data comes from our own model on the A100.

**Back-of-envelope**
- **Reviewer (binding constraint):** I timed myself on 10 blind A/B pairs (`partC/rate_timer.py`):
  median 17.9 s per judgment (mean 26.1 s, one 83 s warm-up outlier). Those were one-line pairs;
  real replies are ~3–5× longer, so I assume ~1 min per judgment, and real outputs are subtler, so
  that's a lower bound. 10 h/wk × 3 wk = 30 h, ~25 h usable → ~1,500 judgments.
  One eval round = 100 items × 2 languages ≈ 3.5 h → only ~5–6 rounds. Eval time is the bottleneck.
- **GPU (not a bottleneck):** SFT data 5k pairs × 6 langs = 30k × ~400 tokens = 12M tokens/epoch.
  LoRA ≈ 4 × 4.2B FLOPs/token → 312 TFLOPS × 40% MFU ≈ 7.4k tok/s → ≈ 27 min/epoch (full-FT 6N
  bound: 40 min). Generating 30k replies ≈ 9M tokens ≈ 24 min with vLLM (`partB/estimates.py`).
- **Serving cost of (c):** a style prompt with few-shot examples ≈ 600 tokens/request (more in
  Indic, at the Part A premium) → +600 × 114,688 B ≈ 69 MB KV per sequence (Part B), about +15%
  KV for a 4k request, so concurrency drops ~13%. SFT removes this cost; that's the reason to
  graduate to (a).
- **Serving cost of (b):** +1 model on every reply. A 1B fp16 model reads ~2 GB per decode step
  → ~7–10 ms/token on an L4, so a 300-token rewrite adds ≈ 2–3 s latency.

**Success metric:** in a blind human A/B against current outputs (n = 100 per language),
casual-preferred win-rate **≥ 70%** for Hindi and Kannada, with replies rated "meaning changed or
wrong" **≤ 5%** (vs baseline). At n = 100 the CI is about ±9 points, so 70% is clearly above chance.

**Kill criterion:** if by **day 5** the best prompt variant has a win-rate < 60% in either language
or a meaning-error rate > 10%, abandon prompt-only and switch to LoRA SFT, trained on reviewer-
filtered outputs from the best prompt plus the reviewer's examples. If SFT doesn't beat the prompt
by ≥ 10 points by **day 12**, ship the best prompt for Hindi/Kannada and stop.

**Day-1 experiment:** 50 real prompts × {Hindi, Kannada} × 4 conditions (current; instruction only;
instruction + 3 reviewer-written casual examples; examples + "write like a message to a friend").
Reviewer does a blind ranking (~2 h) while I log tokens per reply per variant. Output: does the
model have a casual register in Kannada at all, and what does each variant cost in tokens?

**Biggest risk:** Tamil, Telugu, Bengali and Marathi can't be verified by our reviewer. An
LLM-judge calibrated against the reviewer on Hindi/Kannada can screen them, but it isn't a
launch gate.