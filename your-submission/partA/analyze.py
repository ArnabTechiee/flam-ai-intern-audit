import pathlib, csv, regex, numpy as np

CORPUS = pathlib.Path("your-submission/partA/corpus")
OUT = pathlib.Path("your-submission/partA/results.csv")
LANGS = ["eng_Latn", "hin_Deva", "kan_Knda", "tam_Taml", "tel_Telu", "mal_Mlym", "ben_Beng", "mar_Deva"]
TOKENIZERS = [
    "tiktoken:gpt2",                 # the old report's tokenizer (English-centric, 2019)
    "tiktoken:o200k_base",           # GPT-4o tokenizer (modern, multilingual)
    "hf:xlm-roberta-base",           # multilingual SentencePiece, 100 languages
    "hf:google/muril-base-cased",    # Indic-specific (Google, 17 Indian languages)
    "hf:sarvamai/sarvam-1",          # Indic-specific LLM tokenizer (may need HF login; skipped if it fails)
]

def load_tok(spec):
    kind, name = spec.split(":", 1)
    if kind == "tiktoken":
        import tiktoken
        return tiktoken.get_encoding(name).encode, None
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(name)
    return (lambda s: t.encode(s, add_special_tokens=False)), t.unk_token_id

data = {L: (CORPUS / f"{L}.txt").read_text(encoding="utf-8").splitlines() for L in LANGS}
den = {L: {"words": sum(len(l.split()) for l in ls),
           "graphemes": sum(len(regex.findall(r"\X", l)) for l in ls),
           "bytes": sum(len(l.encode("utf-8")) for l in ls),
           "sents": len(ls)} for L, ls in data.items()}

n = len(data["eng_Latn"])
boot = np.random.default_rng(0).integers(0, n, size=(1000, n))   # paired bootstrap: same sentences for both langs
rows = []
for spec in TOKENIZERS:
    try:
        enc, unk = load_tok(spec)
    except Exception as ex:
        print(f"SKIP {spec}: {type(ex).__name__}: {str(ex)[:120]}\n")
        continue
    per, unks = {}, {}
    for L, ls in data.items():
        ids = [enc(l) for l in ls]
        per[L] = np.array([len(x) for x in ids])
        unks[L] = sum(i == unk for x in ids for i in x) if unk is not None else 0
    print(f"== {spec}")
    print(f"{'lang':<10}{'tokens':>8}{'tok/word':>9}{'tok/graph':>10}{'tok/byte':>9}{'tok/sent':>9}{'premium':>8}{'95% CI':>14}{'unk%':>6}")
    for L in LANGS:
        T = per[L].sum()
        prem = T / per["eng_Latn"].sum()
        b = per[L][boot].sum(1) / per["eng_Latn"][boot].sum(1)
        lo, hi = np.percentile(b, [2.5, 97.5])
        r = dict(tokenizer=spec, lang=L, tokens=int(T),
                 tok_word=T / den[L]["words"], tok_grapheme=T / den[L]["graphemes"],
                 tok_byte=T / den[L]["bytes"], tok_sent=T / den[L]["sents"],
                 premium_vs_eng=prem, ci_lo=lo, ci_hi=hi, unk_pct=100 * unks[L] / T)
        rows.append(r)
        print(f"{L:<10}{T:>8}{r['tok_word']:>9.2f}{r['tok_grapheme']:>10.3f}{r['tok_byte']:>9.3f}"
              f"{r['tok_sent']:>9.1f}{prem:>8.2f}{f'[{lo:.2f},{hi:.2f}]':>14}{r['unk_pct']:>6.2f}")
    print()

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print("wrote", OUT)