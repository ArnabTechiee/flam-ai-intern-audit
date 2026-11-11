import unicodedata, regex, tiktoken

enc = tiktoken.get_encoding("gpt2").encode
KIT = "starter_kit/corpus_sample/"

def load(path, nfc=True):
    out = []
    for raw in open(path, encoding="utf-8"):
        l = raw.strip()
        if l:
            out.append(unicodedata.normalize("NFC", l) if nfc else l)
    return out

def metric(lines, lower=True, split="space", agg="mean", denom="codepoint"):
    T, W, C = [], [], []
    for l in lines:
        if lower:
            l = l.lower()
        T.append(len(enc(l)))
        W.append(len(l.split(" ")) if split == "space" else len(l.split()))
        C.append(len(l) if denom == "codepoint" else len(regex.findall(r"\X", l)))
    if agg == "mean":   # original: average of per-line ratios
        return (sum(t / w for t, w in zip(T, W)) / len(T),
                sum(t / c for t, c in zip(T, C)) / len(T))
    return sum(T) / sum(W), sum(T) / sum(C)   # ratio of totals

# each variant changes ONE thing vs the original script
variants = {
    "0 original":           (True,  {}),
    "1 split() fix":        (True,  {"split": "ws"}),
    "2 no lowercase":       (True,  {"lower": False}),
    "3 ratio of totals":    (True,  {"agg": "sum"}),
    "4 grapheme denom":     (True,  {"denom": "grapheme"}),
    "5 NFC removed":        (False, {}),
    "6 fixes 1+2+3+4":      (True,  {"split": "ws", "lower": False, "agg": "sum", "denom": "grapheme"}),
}
print(f"{'variant':<20}{'eng f':>7}{'hin f':>7}{'ratio':>7}{'eng t/c':>9}{'hin t/c':>9}{'ratio':>7}")
for name, (nfc, kw) in variants.items():
    e = metric(load(KIT + "eng_sample.txt", nfc), **kw)
    h = metric(load(KIT + "hin_sample.txt", nfc), **kw)
    print(f"{name:<20}{e[0]:>7.3f}{h[0]:>7.3f}{h[0]/e[0]:>7.2f}{e[1]:>9.3f}{h[1]:>9.3f}{h[1]/e[1]:>7.2f}")

print("\nAre the corpora parallel? First 4 line pairs:")
for a, b in list(zip(load(KIT + "eng_sample.txt"), load(KIT + "hin_sample.txt")))[:4]:
    print(f"  EN: {a}\n  HI: {b}\n")