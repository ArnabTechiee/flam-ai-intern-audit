import pathlib, unicodedata, numpy as np, tiktoken

KIT = pathlib.Path("starter_kit/corpus_sample")
RAW = pathlib.Path("data/flores200_dataset/devtest")
gpt2 = tiktoken.get_encoding("gpt2").encode

def load(p, nfc=True):
    ls = [l.strip() for l in open(p, encoding="utf-8") if l.strip()]
    return [unicodedata.normalize("NFC", l) for l in ls] if nfc else ls

def stats(eng, hin):
    """Fixed metric (no lowercase, split(), totals). Returns per-word ratio, per-sentence premium."""
    te, th = sum(len(gpt2(l)) for l in eng), sum(len(gpt2(l)) for l in hin)
    we, wh = sum(len(l.split()) for l in eng), sum(len(l.split()) for l in hin)
    return (th / wh) / (te / we), th / te

# ---------- 1. Does the non-parallel sample distort the comparison? ----------
print("=== 1. corpus effect (gpt2, fixed metric) ===")
print(f"{'corpus':<34}{'per-word ratio':>15}{'premium':>9}")
fe, fh = load(RAW / "eng_Latn.devtest"), load(RAW / "hin_Deva.devtest")
for name, (e, h) in {"starter sample (10, NOT parallel)": (load(KIT / "eng_sample.txt"), load(KIT / "hin_sample.txt")),
                     "FLORES first 10 (parallel)": (fe[:10], fh[:10]),
                     "FLORES all 1012 (parallel)": (fe, fh)}.items():
    w, p = stats(e, h)
    print(f"{name:<34}{w:>15.2f}{p:>9.2f}")

# spread of 10-sentence estimates: parallel draws vs non-parallel (independently drawn) draws
TE = np.array([len(gpt2(l)) for l in fe]); TH = np.array([len(gpt2(l)) for l in fh])
rng = np.random.default_rng(0)
i = rng.integers(0, len(fe), (2000, 10)); j = rng.integers(0, len(fe), (2000, 10))
par = TH[i].sum(1) / TE[i].sum(1)      # same sentences in both languages
non = TH[j].sum(1) / TE[i].sum(1)      # different sentences, like the starter sample
for name, x in [("parallel 10-sent draws", par), ("NON-parallel 10-sent draws", non)]:
    lo, hi = np.percentile(x, [5, 95])
    print(f"{name:<34} premium median {np.median(x):.2f}, 90% range [{lo:.2f}, {hi:.2f}]")

# ---------- 2. NFC effect on real FLORES text ----------
print("\n=== 2. NFC effect on raw FLORES ===")
print(f"{'lang':<10}{'tokenizer':<12}{'lines changed':>14}{'tokens raw':>11}{'tokens NFC':>11}{'delta %':>9}")
for L in ["hin_Deva", "ben_Beng"]:
    raw, nfc = load(RAW / f"{L}.devtest", nfc=False), load(RAW / f"{L}.devtest")
    changed = sum(a != b for a, b in zip(raw, nfc))
    for tname in ["gpt2", "o200k_base"]:
        enc = tiktoken.get_encoding(tname).encode
        tr, tn = sum(len(enc(l)) for l in raw), sum(len(enc(l)) for l in nfc)
        print(f"{L:<10}{tname:<12}{changed:>14}{tr:>11}{tn:>11}{100*(tn-tr)/tr:>9.2f}")