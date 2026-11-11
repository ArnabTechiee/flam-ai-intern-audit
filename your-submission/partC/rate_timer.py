import csv, random, time, pathlib

PAIRS = pathlib.Path("your-submission/partC/pairs.tsv")     # each line: formal<TAB>casual
OUT = pathlib.Path("your-submission/partC/timing_results.csv")
pairs = [l.rstrip("\n").split("\t") for l in open(PAIRS, encoding="utf-8") if "\t" in l]
random.seed(0)
rows = []
for k, (formal, casual) in enumerate(pairs, 1):
    casual_is_a = random.random() < 0.5                       # blind: randomize which side is casual
    a, b = (casual, formal) if casual_is_a else (formal, casual)
    print(f"\n--- Pair {k}/{len(pairs)} ---\nA: {a}\nB: {b}")
    t0 = time.time()
    pick = input("More natural/casual? (A/B): ").strip().upper()
    meaning = input("Same meaning? (y/n): ").strip().lower()
    secs = time.time() - t0
    rows.append(dict(pair=k, secs=round(secs, 1), picked_casual=(pick == "A") == casual_is_a, same_meaning=meaning == "y"))
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
s = sorted(r["secs"] for r in rows)
print(f"\n{len(rows)} judgments | mean {sum(s)/len(s):.1f}s | median {s[len(s)//2]:.1f}s | max {s[-1]:.1f}s")
print(f"picked casual: {sum(r['picked_casual'] for r in rows)}/{len(rows)} | wrote {OUT}")