import pathlib, unicodedata, regex

SRC = pathlib.Path("data/flores200_dataset/devtest")
OUT = pathlib.Path("your-submission/partA/corpus")
OUT.mkdir(parents=True, exist_ok=True)
# 4 required + Bengali/Marathi (Part C languages; Marathi shares Hindi's script -> useful control)
LANGS = ["eng_Latn", "hin_Deva", "kan_Knda", "tam_Taml", "tel_Telu", "mal_Mlym", "ben_Beng", "mar_Deva"]

print(f"{'lang':<10}{'lines':>6}{'empty':>6}{'nonNFC':>7}{'words':>8}{'graphemes':>10}{'bytes':>9}")
counts = set()
for L in LANGS:
    raw = (SRC / f"{L}.devtest").read_text(encoding="utf-8").splitlines()
    nonnfc = sum(unicodedata.normalize("NFC", l) != l for l in raw)
    lines = [unicodedata.normalize("NFC", l.strip()) for l in raw]
    empty = sum(not l for l in lines)
    (OUT / f"{L}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    w = sum(len(l.split()) for l in lines)
    g = sum(len(regex.findall(r"\X", l)) for l in lines)
    b = sum(len(l.encode("utf-8")) for l in lines)
    counts.add(len(lines))
    print(f"{L:<10}{len(lines):>6}{empty:>6}{nonnfc:>7}{w:>8}{g:>10}{b:>9}")
print("parallel (same line count everywhere):", len(counts) == 1)