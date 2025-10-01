# Блок 1
# !pip install -q transformers accelerate seqeval

# Блок 2
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          DataCollatorForTokenClassification, TrainingArguments, Trainer)
from seqeval.metrics import f1_score

# Блок 3
import pandas as pd, ast, re

USE_SYNT = True

train_orig = pd.read_csv("train.csv", sep=";", quotechar='"', engine="python")

assert {"sample", "annotation"} <= set(train_orig.columns), "Ожидаем sample и annotation в train_df"

if USE_SYNT:
    synthetic_train = pd.read_csv("synthetic_train.csv", sep=";", quotechar='"', engine="python")

    assert {"sample", "annotation"} <= set(synthetic_train.columns), "Ожидаем sample и annotation в submission_df"

    train_df = pd.concat([train_orig, synthetic_train], ignore_index=True)

    train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"train.csv: {len(train_orig)} строк")
    print(f"synthetic_train.csv: {len(synthetic_train)} строк")
    print(f"Объединённый train_df: {len(train_df)} строк")
else:
    train_df = train_orig
    print(f"train.csv без синтетики: {len(train_df)} строк")

submission_df = pd.read_csv("submission.csv", sep=";", quotechar='"', engine="python")

# Блок 4
ALLOWED = {"TYPE","BRAND","VOLUME","PERCENT"}

def clean_ann(s: str):
    try:
        parsed = ast.literal_eval(s)
    except Exception:
        return []
    out=[]
    for x in parsed:
        if isinstance(x,(list,tuple)) and len(x)==3:
            s0,e0,t = x
            if t != "O":
                t = re.sub(r"^(B-|I-)", "", str(t))
                if t in ALLOWED:
                    try:
                        out.append([int(s0), int(e0), t])
                    except: pass
    return out

train_df["entities"] = train_df["annotation"].astype(str).apply(clean_ann)

# Блок 5
model_name = "cointegrated/rubert-tiny2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

labels = ["O","B-TYPE","I-TYPE","B-BRAND","I-BRAND","B-VOLUME","I-VOLUME","B-PERCENT","I-PERCENT"]
label2id = {l:i for i,l in enumerate(labels)}
id2label = {i:l for l,i in label2id.items()}

# Блок 6
import torch
import numpy as np

class NERDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.samples = df["sample"].tolist()
        self.entities = df["entities"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        spans = self.entities[idx]

        ents = []
        L = len(text)
        for s0, e0, t in spans:
            s0 = int(s0); e0 = int(e0)
            if 0 <= s0 < e0 <= L:
                ents.append((s0, e0, str(t)))
        ents.sort(key=lambda x: (x[0], x[1]))

        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_len
        )
        offsets = enc["offset_mapping"]

        labels_ids = []
        for (st, en) in offsets:
            if st == en:
                labels_ids.append(-100)
                continue

            lab = "O"
            for s0, e0, t in ents:
                if max(st, s0) < min(en, e0):
                    lab = f"B-{t}" if st == s0 else f"I-{t}"
                    break

            labels_ids.append(label2id.get(lab, 0))

        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels_ids, dtype=torch.long),
        }

rng = np.random.default_rng(42)
perm = rng.permutation(len(train_df))
cut = int(0.9*len(train_df))
tr_idx, va_idx = perm[:cut], perm[cut:]
train_ds = NERDataset(train_df.iloc[tr_idx].reset_index(drop=True), tokenizer, max_len=256)
val_ds   = NERDataset(train_df.iloc[va_idx].reset_index(drop=True), tokenizer, max_len=256)

model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
)
data_collator = DataCollatorForTokenClassification(tokenizer)

from seqeval.metrics import f1_score
from seqeval.scheme import IOB2

def compute_metrics(p):
    logits, labels_arr = p
    preds = np.argmax(logits, axis=-1)
    true_preds, true_labels = [], []
    for p_row, l_row in zip(preds, labels_arr):
        p_seq, l_seq = [], []
        for p_i, l_i in zip(p_row, l_row):
            if l_i == -100:
                continue
            p_seq.append(id2label[int(p_i)])
            l_seq.append(id2label[int(l_i)])
        true_preds.append(p_seq)
        true_labels.append(l_seq)
    return {"f1": f1_score(true_labels, true_preds, zero_division=0, scheme=IOB2)}

# Блок 7
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from seqeval.metrics import f1_score

args = TrainingArguments(
    output_dir="ner_ckpt",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=12,
    weight_decay=0.01,

    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,

    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=3,

    warmup_ratio=0.1,
    gradient_accumulation_steps=1,
    optim="adamw_torch",
    lr_scheduler_type="linear",

    remove_unused_columns=False,
    report_to="none",
    seed=42,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()
print(trainer.evaluate())

# Блок 7.1
output_dir = "/model"

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
trainer.save_state()
print("Best checkpoint:", trainer.state.best_model_checkpoint)

# Блок 8.0
load_path = "/model"

tokenizer = AutoTokenizer.from_pretrained(load_path)
model = AutoModelForTokenClassification.from_pretrained(load_path)

# Блок 8
import re
from typing import List, Tuple
import numpy as np
import torch

assert 'model' in globals() and 'tokenizer' in globals(), "Сначала загрузите model/tokenizer"

if 'labels' not in globals():
    labels = ["O","B-TYPE","I-TYPE","B-BRAND","I-BRAND","B-VOLUME","I-VOLUME","B-PERCENT","I-PERCENT"]
label2id = {l:i for i,l in enumerate(labels)}
id2label  = {i:l for l,i in label2id.items()}

CFG = {
    "use_margin_rule": True,
    "margin_delta": 0.06,
    "margin_delta_per_class": {
        "TYPE": 0.07,
        "BRAND": 0.07,
        "VOLUME": 0.02,
        "PERCENT": 0.015,
    },

    "numeric_overrides": True,
    "trim_punct_on_spans": True,
    "word_majority": True,
    "word_inherit_prev": True,
    "majority_threshold": 0.58,
    "max_len": 256,
}

ALLOWED_BASE = {"TYPE","BRAND","VOLUME","PERCENT"}
PUNCT = set(";:,.!?()[]{}«»\"'—–-")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval().to(DEVICE)

RE_PERCENT = re.compile(r'(?<!\d)(\d{1,3}(?:[.,]\d{1,2})?)\s*%', re.I)
RE_UNIT = r"(?:мл|л|литр(?:а|ов)?|г|гр|грамм(?:а|ов)?|кг|шт|уп|упак|бут|бутыл(?:ка|ки|ок)|табл|таб|капс|порц|пак)"
RE_UNIT_DOT = r"(?:мл\.|л\.|г\.|гр\.|шт\.|уп\.|таб\.|капс\.)"
RE_VOLUME1 = re.compile(rf'(?<!\d)\d+(?:[.,]\d+)?\s*(?:{RE_UNIT}|{RE_UNIT_DOT})(?!\w)', re.I)
RE_VOLUME2 = re.compile(rf'(?<!\d)\d+\s*[x×х]\s*\d+\s*(?:шт|уп|упак|таб|капс|пак)(?!\w)', re.I)

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
        if s0 < e0:
            arr[s0:e0] = [base] * (e0 - s0)
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
                    base = max(((b,c) for b,c in cnt.items() if b != "O"),
                               default=("O",0), key=lambda x: x[1])[0]
        else:
            base = max(((b,c) for b,c in cnt.items() if b != "O"),
                       default=("O",0), key=lambda x: x[1])[0]

        out.append((ws, we, base)); prev_base = base
    return out

def build_full_word_bio_from_wordlabels(word_labels: List[Tuple[int,int,str]]) -> List[Tuple[int,int,str]]:
    out, prev_base, started = [], "O", False
    for ws, we, base in word_labels:
        if base == "O":
            out.append((ws, we, "O")); prev_base, started = "O", False
        else:
            if prev_base == base and started:
                out.append((ws, we, f"I-{base}"))
            else:
                out.append((ws, we, f"B-{base}")); started = True
            prev_base = base
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
    delta = CFG.get("margin_delta_per_class", {}).get(base, CFG.get("margin_delta", 0.06))
    return (top1 - top2) < delta

def predict_char_base(text: str) -> List[str]:
    enc = tokenizer(
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
        inputs = {
            k: v[i:i+1].to(DEVICE)
            for k, v in enc.items()
            if k in ("input_ids", "attention_mask", "token_type_ids")
        }
        with torch.no_grad():
            logits = model(**inputs).logits[0]  # [seq, C]
        logprobs = torch.log_softmax(logits, dim=-1).cpu().numpy()
        probs    = np.exp(logprobs)

        keep = [(s,e) for (s,e) in offsets if not (s==0 and e==0)]
        if not keep: continue
        lp = np.array([lp for lp,(s,e) in zip(logprobs, offsets) if not (s==0 and e==0)])
        pr = np.array([pr for pr,(s,e) in zip(probs,    offsets) if not (s==0 and e==0)])

        path = lp.argmax(axis=1).tolist()
        if CFG["use_margin_rule"]:
          for t in range(len(path)):
            if _apply_margin_rule_per_class(pr[t]):
              path[t] = label2id["O"]

        tok_labels = _repair_bio_token_sequence([id2label[int(pid)] for pid in path])
        for lab, (s, e) in zip(tok_labels, keep):
            if s == e or lab == "O": continue
            base = _base(lab)
            if base == "O": continue
            s0, e0 = max(0, s), min(len(char_labels), e)
            if s0 < e0:
                char_labels[s0:e0] = [base] * (e0 - s0)

    _inject_numeric_overrides(text, char_labels)
    return char_labels

def predict_word_bio(text: str) -> List[Tuple[int,int,str]]:
    if not text or not text.strip():
        return []
    char_base = predict_char_base(text)
    base_spans = _compress_char_runs_base(char_base)
    if base_spans and CFG["trim_punct_on_spans"]:
        trimmed = []
        for s, e, base in base_spans:
            s2, e2 = _trim_punct(text, s, e)
            if s2 < e2:
                trimmed.append((s2, e2, base))
        base_spans = trimmed
    char_base2 = spans_to_charbase(text, base_spans)
    word_labels = _word_labels_from_charbase(text, char_base2)
    full_bio = build_full_word_bio_from_wordlabels(word_labels)

    for s,e,lab in full_bio:
        assert 0 <= s < e <= len(text)
        assert lab == "O" or lab.startswith(("B-","I-"))
    return full_bio

# Блок 8
import pandas as pd

submission_df = pd.read_csv("submission.csv", sep=";", quotechar='"', engine="python")

pred_rows: List[str] = []
for text in submission_df["sample"].tolist():
    full_bio = predict_word_bio(text)
    pred_rows.append(str([(int(s), int(e), str(lab)) for (s,e,lab) in full_bio]))

submission_out = submission_df.copy()
submission_out["annotation"] = pred_rows
submission_out.to_csv("submission_out.csv", sep=";", quotechar='"', index=False)
print("Готово: submission_out.csv сохранён.")
print(submission_out.head(5).to_string(index=False))