from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Online Retail Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("../model.joblib")

class PredictRequest(BaseModel):
    Quantity: float
    UnitPrice: float
    InvoiceMonth: int
    Country: int

class PredictResponse(BaseModel):
    prediction: float

@app.get("/")
def root():
    return {"message": "Online Retail Predictor API"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    X = np.array([[req.Quantity, req.UnitPrice, req.InvoiceMonth, req.Country]])
    pred = model.predict(X)[0].item()
    return {"prediction": float(pred)}
