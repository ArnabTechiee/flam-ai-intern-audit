import tiktoken

enc = tiktoken.get_encoding("gpt2")
samples = [
    "The train arrived exactly on time.",
    "ट्रेन ठीक समय पर पहुँची।",
    "NASA and ISRO announced",
    "nasa and isro announced",
]
for s in samples:
    ids = enc.encode(s)
    print(f"{s!r}: {len(s)} codepoints, {len(s.encode('utf-8'))} utf8-bytes, {len(ids)} tokens")
    print("   pieces:", [enc.decode_single_token_bytes(i) for i in ids])