import pickle

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="ML Model API")

# Load the model and scaler once when the app starts
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


class PredictRequest(BaseModel):
    day: float
    month: float
    year: float
    Temperature: float
    RH: float
    Ws: float
    Rain: float
    FFMC: float
    DMC: float
    ISI: float
    Classes: float
    Region: float


@app.get("/")
def home():
    return {"message": "Model API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        # Same order as the scaler's feature names
        x = np.array([[
            request.day, request.month, request.year,
            request.Temperature, request.RH, request.Ws, request.Rain,
            request.FFMC, request.DMC, request.ISI,
            request.Classes, request.Region,
        ]])
        x_scaled = scaler.transform(x)
        prediction = model.predict(x_scaled)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))