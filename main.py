# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import os
from pathlib import Path

from ner_runtime import load_model, predict_word_bio

REPO_DIR = Path(__file__).resolve().parent
MODEL_DIR = os.getenv("MODEL_DIR", str(REPO_DIR / "model"))

app = FastAPI(title="X5 NER Service")

class InputData(BaseModel):
    input: str

@app.on_event("startup")
def _load():
    print(f"[startup] Loading model from: {MODEL_DIR}")
    load_model(MODEL_DIR)

@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_dir": MODEL_DIR, "latest_update" : "2025-09-30"}

@app.post("/api/predict")
async def predict(data: InputData) -> List[Dict]:
    text = (data.input or "").strip()
    if not text:
        return []
    spans = predict_word_bio(text)
    return [
        {"start_index": int(s), "end_index": int(e), "entity": lab}
        for s, e, lab in spans
    ]