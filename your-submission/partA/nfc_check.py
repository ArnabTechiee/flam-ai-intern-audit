import unicodedata, collections, pathlib

RAW = pathlib.Path("data/flores200_dataset/devtest")
for L in ["hin_Deva", "ben_Beng"]:
    raw = [l.strip() for l in open(RAW / f"{L}.devtest", encoding="utf-8")]
    before, after = collections.Counter(), collections.Counter()
    cp_raw = cp_nfc = 0
    for l in raw:
        n = unicodedata.normalize("NFC", l)
        cp_raw += len(l); cp_nfc += len(n)
        if n != l:
            before.update(l); after.update(n)
    print(f"== {L}: code points raw {cp_raw} -> NFC {cp_nfc} ({cp_nfc - cp_raw:+d})")
    diff = {c: after[c] - before[c] for c in set(before) | set(after) if after[c] != before[c]}
    for c, d in sorted(diff.items(), key=lambda x: -abs(x[1]))[:8]:
        print(f"   U+{ord(c):04X} {unicodedata.name(c, '?'):<40} {d:+d}")