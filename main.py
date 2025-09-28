from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class InputData(BaseModel):
    input: str

@app.post("/api/jojo")
def predict(data: InputData):
    return []
