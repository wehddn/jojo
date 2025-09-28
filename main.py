# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import os

from ner_runtime import load_model, predict_word_bio

MODEL_DIR = os.getenv("MODEL_DIR", "cointegrated/rubert-tiny2")
app = FastAPI(title="X5 NER Service")

class InputData(BaseModel):
    input: str

@app.on_event("startup")
def _load():
    load_model(MODEL_DIR)

@app.get("/healthz")
def healthz():
    return {"status":"ok"}

@app.post("/api/predict")
async def predict(data: InputData) -> List[Dict]:
    text = data.input or ""
    if text.strip() == "":
        return []  # по ТЗ — пустой список
    spans = predict_word_bio(text)
    # Конвертация в нужный формат (только B-/I- сущности, O в ответ не пишем)
    result = []
    for s, e, lab in spans:
        result.append({"start_index": int(s), "end_index": int(e), "entity": lab})
    return result
