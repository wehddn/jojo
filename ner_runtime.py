from typing import List, Tuple
import re, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# --- Метки ---
LABELS = ["O","B-TYPE","I-TYPE","B-BRAND","I-BRAND","B-VOLUME","I-VOLUME","B-PERCENT","I-PERCENT"]
label2id = {l:i for i,l in enumerate(LABELS)}
id2label = {i:l for l,i in label2id.items()}
ALLOWED_BASE = {"TYPE","BRAND","VOLUME","PERCENT"}
PUNCT = set(";:,.!?()[]{}«»\"'—–-")

# --- CONFIG ---
CFG = {
    "use_margin_rule": True,
    "margin_delta": 0.06,
    "margin_delta_per_class": {"TYPE":0.08,"BRAND":0.05,"VOLUME":0.02,"PERCENT":0.015},
    "numeric_overrides": True,
    "trim_punct_on_spans": True,
    "word_majority": True,
    "word_inherit_prev": True,
    "majority_threshold": 0.58,
    "max_len": 256,
}

# --- Регексы ---
RE_PERCENT = re.compile(r'(?<!\d)(\d{1,3}(?:[.,]\d{1,2})?)\s*%', re.I)
RE_UNIT = r"(?:мл|л|литр(?:а|ов)?|г|гр|грамм(?:а|ов)?|кг|шт|уп|упак|бут|бутыл(?:ка|ки|ок)|табл|таб|капс|порц|пак)"
RE_UNIT_DOT = r"(?:мл\.|л\.|г\.|гр\.|шт\.|уп\.|таб\.|капс\.)"
RE_VOLUME1 = re.compile(rf'(?<!\d)\d+(?:[.,]\d+)?\s*(?:{RE_UNIT}|{RE_UNIT_DOT})(?!\w)', re.I)
RE_VOLUME2 = re.compile(rf'(?<!\d)\d+\s*[x×х]\s*\d+\s*(?:шт|уп|упак|таб|капс|пак)(?!\w)', re.I)

# --- Глобалы модели ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_tokenizer = None
_model = None

def load_model(model_dir_or_name: str):
    """Однократная загрузка модели и токенайзера."""
    global _tokenizer, _model
    _tokenizer = AutoTokenizer.from_pretrained(model_dir_or_name)
    _model = AutoModelForTokenClassification.from_pretrained(
        model_dir_or_name, num_labels=len(LABELS), id2label=id2label, label2id=label2id
    )
    _model.eval().to(DEVICE)
    torch.set_num_threads(1)  # стабильность latency

def _base(lab: str) -> str:
    return "O" if lab=="O" else (lab.split("-",1)[1] if "-" in lab else lab)

def _repair_bio_token_sequence(token_labels: List[str]) -> List[str]:
    out, prev = [], "O"
    for lab in token_labels:
        if lab == "O":
            out.append("O"); prev = "O"; continue
        if "-" not in lab: lab = f"B-{lab}"
        bio, typ = lab.split("-", 1)
        if bio == "I":
            ok = prev.startswith(("B-","I-")) and prev.split("-",1)[1] == typ
            lab = lab if ok else f"B-{typ}"
        out.append(lab); prev = lab
    return out

def _trim_punct(text: str, s: int, e: int) -> Tuple[int,int]:
    if not CFG["trim_punct_on_spans"]: return s, e
    while s < e and text[s] in PUNCT: s += 1
    while s < e and text[e-1] in PUNCT: e -= 1
    return s, e

def _compress_char_runs_base(char_labels: List[str]) -> List[Tuple[int,int,str]]:
    spans, i, n = [], 0, len(char_labels)
    while i < n:
        lab = char_labels[i]
        if lab == "O": i += 1; continue
        j = i + 1
        while j < n and char_labels[j] == lab: j += 1
        spans.append((i, j, lab))
        i = j
    return spans

def spans_to_charbase(text: str, base_spans: List[Tuple[int,int,str]]) -> List[str]:
    arr = ["O"] * len(text)
    for s, e, base in base_spans:
        s0, e0 = max(0,s), min(len(text), e)
        if s0 < e0: arr[s0:e0] = [base] * (e0 - s0)
    return arr

def _word_labels_from_charbase(text: str, char_labels: List[str]) -> List[Tuple[int,int,str]]:
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    out, prev_base = [], "O"
    thr = CFG["majority_threshold"]
    for ws, we in words:
        cnt, non_o = {}, 0
        for i in range(ws, we):
            base = char_labels[i] if 0 <= i < len(char_labels) and char_labels[i] in ALLOWED_BASE else "O"
            cnt[base] = cnt.get(base, 0) + 1
            if base != "O": non_o += 1

        if CFG["word_majority"] and cnt:
            best_base, best_cnt = max(cnt.items(), key=lambda x: x[1])
            if best_base != "O" and best_cnt >= (we - ws) * thr:
                base = best_base
            else:
                if CFG["word_inherit_prev"] and non_o > 0 and prev_base != "O" and cnt.get(prev_base, 0) > 0:
                    base = prev_base
                else:
                    base = max(((b,c) for b,c in cnt.items() if b != "O"), default=("O",0), key=lambda x: x[1])[0]
        else:
            base = max(((b,c) for b,c in cnt.items() if b != "O"), default=("O",0), key=lambda x: x[1])[0]

        out.append((ws, we, base)); prev_base = base
    return out

def _inject_numeric_overrides(text: str, char_labels: List[str]) -> None:
    if not CFG["numeric_overrides"]: return
    def mark(a: int, b: int, label: str):
        a = max(0, a); b = min(len(char_labels), b)
        if a < b: char_labels[a:b] = [label] * (b - a)
    for m in RE_PERCENT.finditer(text):  mark(*m.span(), "PERCENT")
    for m in RE_VOLUME1.finditer(text):  mark(*m.span(), "VOLUME")
    for m in RE_VOLUME2.finditer(text):  mark(*m.span(), "VOLUME")

def _apply_margin_rule_per_class(probs_row: np.ndarray) -> bool:
    i1 = int(np.argmax(probs_row))
    top1 = float(probs_row[i1])
    tmp = probs_row.copy(); tmp[i1] = -1.0
    top2 = float(tmp.max())
    lab = id2label[i1]
    base = _base(lab)
    delta = CFG.get("margin_delta_per_class", {}).get(base, CFG["margin_delta"])
    return (top1 - top2) < delta

@torch.inference_mode()
def predict_char_base(text: str) -> List[str]:
    enc = _tokenizer(
        text,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        stride=64,
        truncation=True,
        max_length=CFG["max_len"],
        return_tensors="pt",
    )
    char_labels = ["O"] * len(text)
    n_chunks = int(enc["input_ids"].shape[0])

    for i in range(n_chunks):
        offsets = enc["offset_mapping"][i].tolist()
        inputs = {k: v[i:i+1].to(DEVICE) for k, v in enc.items()
                  if k in ("input_ids", "attention_mask", "token_type_ids")}
        logits = _model(**inputs).logits[0]  # [seq, C]
        logprobs = torch.log_softmax(logits, dim=-1).cpu().numpy()
        probs = np.exp(logprobs)

        keep = [(s,e) for (s,e) in offsets if not (s==0 and e==0)]
        if not keep: continue
        lp = np.array([lp for lp,(s,e) in zip(logprobs, offsets) if not (s==0 and e==0)])
        pr = np.array([pr for pr,(s,e) in zip(probs,    offsets) if not (s==0 and e==0)])

        path = lp.argmax(axis=1).tolist()
        if CFG["use_margin_rule"]:
            for t in range(len(path)):
                if _apply_margin_rule_per_class(pr[t]): path[t] = label2id["O"]

        tok_labels = _repair_bio_token_sequence([id2label[int(pid)] for pid in path])
        for lab, (s, e) in zip(tok_labels, keep):
            if s == e or lab == "O": continue
            base = _base(lab)
            if base == "O": continue
            s0, e0 = max(0, s), min(len(char_labels), e)
            if s0 < e0: char_labels[s0:e0] = [base] * (e0 - s0)

    _inject_numeric_overrides(text, char_labels)
    return char_labels

def build_full_word_bio_from_wordlabels(word_labels: List[Tuple[int,int,str]]) -> List[Tuple[int,int,str]]:
    out, prev_base, started = [], "O", False
    for ws, we, base in word_labels:
        if base == "O":
            out.append((ws, we, "O")); prev_base, started = "O", False
        else:
            if prev_base == base and started: out.append((ws, we, f"I-{base}"))
            else: out.append((ws, we, f"B-{base}")); started = True
            prev_base = base
    return out

def predict_word_bio(text: str):
    """[(start, end, 'B-XXX'|'I-XXX'|'O'), ...] — по словам."""
    if not text or not text.strip(): return []
    char_base = predict_char_base(text)
    base_spans = _compress_char_runs_base(char_base)
    if base_spans and CFG["trim_punct_on_spans"]:
        trimmed = []
        for s, e, base in base_spans:
            s2, e2 = _trim_punct(text, s, e)
            if s2 < e2: trimmed.append((s2, e2, base))
        base_spans = trimmed
    char_base2 = spans_to_charbase(text, base_spans)
    word_labels = _word_labels_from_charbase(text, char_base2)
    return build_full_word_bio_from_wordlabels(word_labels)
