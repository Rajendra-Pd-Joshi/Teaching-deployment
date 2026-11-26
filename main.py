import pickle
from typing import List

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="ML Model API")

# Load the model and scaler once when the app starts
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


class PredictRequest(BaseModel):
    # One row of features, in the same order used during training
    features: List[float]


@app.get("/")
def home():
    return {"message": "Model API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):
    x = np.array(request.features).reshape(1, -1)
    x_scaled = scaler.transform(x)
    prediction = model.predict(x_scaled)
    return {"prediction": prediction.tolist()}
