import pandas as pd
import numpy as np
import re, ast, random, os
from collections import Counter, defaultdict

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

SRC_PATH = "train.csv"
OUT_PATH = "synthetic_train.csv"

train_df = pd.read_csv(SRC_PATH, sep=";", quotechar='"', engine="python")
assert {"sample", "annotation"} <= set(train_df.columns), "Ожидаем колонки sample;annotation"

def parse_ann(s):
    try:
        x = ast.literal_eval(s)
        if not isinstance(x, (list, tuple)):
            return []
        out = []
        for t in x:
            if isinstance(t, (list, tuple)) and len(t) == 3:
                s0, e0, lab = t
                out.append([int(s0), int(e0), str(lab)])
        out.sort(key=lambda r: (r[0], r[1]))
        return out
    except Exception:
        return []

train_df["ann_list"] = train_df["annotation"].apply(parse_ann)

def spans_to_full_with_O(text, spans):
    L = len(text)
    ents = [(max(0, min(L, s)), max(0, min(L, e)), str(l)) for s, e, l in spans if str(l).upper() != "O"]
    ents = [(s, e, l) for s, e, l in ents if s < e]
    ents.sort(key=lambda r: (r[0], r[1]))

    out = []
    cur = 0

    def add_O_runs(start, end):
        if start >= end:
            return
        gap = text[start:end]
        for m in re.finditer(r"\S+", gap):
            gs = start + m.start()
            ge = start + m.end()
            out.append([gs, ge, "O"])

    for s, e, l in ents:
        if cur < s:
            add_O_runs(cur, s)
        out.append([s, e, l])
        cur = e
    if cur < L:
        add_O_runs(cur, L)

    return out

RE_PERCENT = re.compile(r"(?<!\w)(\d{1,3}(?:[.,]\d{1,2})?)\s*%")
RE_VOLUME = re.compile(
    r"(?<!\w)(?:"
    r"\d{1,4}(?:[.,]\d{1,3})?\s*(?:л|l|литр(?:а|ов)?|мл|ml|грамм(?:а|ов)?|гр|г|kg|кг|шт|штук|уп|упак|пачк[аи])"
    r"|(?:\d{1,2}\s*(?:x|х|\*)\s*\d{2,4}\s*(?:мл|ml|г|гр|грамм|шт))"
    r")", re.IGNORECASE
)

def find_numeric_spans(text):
    spans = []
    for m in RE_PERCENT.finditer(text):
        spans.append([m.start(), m.end(), "B-PERCENT"])
    for m in RE_VOLUME.finditer(text):
        spans.append([m.start(), m.end(), "B-VOLUME"])
    spans.sort(key=lambda r: (r[0], r[1]))
    return spans

def extract_brands(row):
    text = row["sample"]
    ents = row["ann_list"]
    out = []
    for s,e,l in ents:
        L = str(l).upper()
        if L in ("B-BRAND","I-BRAND","BRAND"):
            out.append((s,e,text[s:e]))
    out2 = []
    for s,e,txt in sorted(out, key=lambda r:(r[0],r[1])):
        if out2 and s == out2[-1][1]:
            ps,pe,pt = out2[-1]
            out2[-1] = (ps, e, pt + text[pe:e])
        else:
            out2.append((s,e,txt))
    return [t for _,__,t in out2 if t.strip()]

train_df["brands"] = train_df.apply(extract_brands, axis=1)

KEYBOARD_NEIGHBORS = {
    "о":"л", "а":"ф", "с":"ы", "е":"у", "н":"т", "к":"е", "м":"ь", "и":"ш",
    "р":"к", "д":"в", "т":"ь", "л":"д"
}

def make_typo(token):
    if len(token) < 3:
        return token
    rng = random.Random(SEED + hash(token) % (10**6))
    candidates = []

    i = rng.randint(0, len(token)-1)
    candidates.append(token[:i] + token[i]*2 + token[i+1:])

    j = rng.randint(0, len(token)-1)
    candidates.append(token[:j] + token[j+1:])

    k = rng.randint(0, len(token)-1)
    ch = token[k].lower()
    rep = KEYBOARD_NEIGHBORS.get(ch, ch)
    candidates.append(token[:k] + rep + token[k+1:])

    if len(token) >= 4:
        t = list(token)
        p = rng.randint(0, len(token)-2)
        t[p], t[p+1] = t[p+1], t[p]
        candidates.append("".join(t))

    return candidates[rng.randrange(len(candidates))]

MAX_SYN_PER_BRAND = 2
TOTAL_CAP = 4000

synthetic_rows = []
brand_used_counter = defaultdict(int)

for idx, row in train_df.iterrows():
    text = str(row["sample"])
    anns = list(row["ann_list"])
    brands = row["brands"]
    if not brands:
        continue

    brand_text = brands[0].strip()
    if not brand_text:
        continue

    brand_key = brand_text.lower()
    if brand_used_counter[brand_key] >= MAX_SYN_PER_BRAND:
        continue

    brand_span = None
    i = 0
    while i < len(anns):
        s,e,l = anns[i]
        if str(l).upper() == "B-BRAND":
            j = i + 1
            end = e
            while j < len(anns) and str(anns[j][2]).upper() == "I-BRAND" and anns[j][0] == end:
                end = anns[j][1]
                j += 1
            brand_span = (s, end)
            break
        i += 1
    if brand_span is None:
        continue

    s0, e0 = brand_span
    orig = text[s0:e0]
    typo = make_typo(orig)
    if not typo or typo == orig:
        continue

    new_text = text[:s0] + typo + text[e0:]
    delta = len(typo) - len(orig)

    new_ents = []
    for s,e,l in anns:
        if s >= e:
            continue
        L = str(l).upper()
        if e <= s0 or s >= e0:
            shift = 0 if e <= s0 else delta
            new_ents.append([s + shift, e + shift, l])
        else:
            pass
    new_ents.append([s0, s0 + len(typo), "B-BRAND"])

    keep = [(s,e,l) for s,e,l in new_ents if str(l).upper() not in ("B-VOLUME","I-VOLUME","B-PERCENT","I-PERCENT")]
    numeric = find_numeric_spans(new_text)

    merged = keep[:]
    for s,e,l in numeric:
        overlap = any(not (e <= s2 or e2 <= s) for s2,e2,lab2 in merged)
        if not overlap:
            merged.append([s,e,l])
    merged.sort(key=lambda r: (r[0], r[1]))

    full_spans = spans_to_full_with_O(new_text, merged)

    synthetic_rows.append({"sample": new_text, "annotation": str(full_spans)})
    brand_used_counter[brand_key] += 1

    if len(synthetic_rows) >= TOTAL_CAP:
        break

synthetic_df = pd.DataFrame(synthetic_rows, columns=["sample","annotation"])
synthetic_df.to_csv(OUT_PATH, sep=";", index=False)

def count_numeric_hits(df):
    vol_total = pct_total = 0
    for t in df["sample"].astype(str):
        vol_total += len(list(RE_VOLUME.finditer(t)))
        pct_total += len(list(RE_PERCENT.finditer(t)))
    return vol_total, pct_total

vol_hits, pct_hits = count_numeric_hits(train_df)

print(f"Rows in train: {len(train_df)}")
print(f"Synthetic rows created: {len(synthetic_df)} -> saved to {OUT_PATH}")