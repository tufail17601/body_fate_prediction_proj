import os
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
model = pickle.load(open("body_fate.pkl", "rb"))
app = FastAPI(title="Body Fat Prediction API")
@app.get("/", include_in_schema=False)
def root():
    return FileResponse("templates/index.html")
class FEATURE_NAMES(BaseModel):
    Density: float
    Age: float
    Weight: float
    Height: float
    Neck: float
    Chest: float
    Abdomen: float
    Hip: float
    thigh: float
    Knee: float
    Ankle: float
    Biceps: float
    Forearm: float
    Wrist: float



@app.post("/predict")
def predict(data:FEATURE_NAMES):
    features = np.array([[data.Density, data.Age, data.Weight, data.Height, data.Neck, data.Chest, data.Abdomen, data.Hip, data.thigh, data.Knee, data.Ankle, data.Biceps, data.Forearm, data.Wrist ]])
    prediction = model.predict(features)
    return {"prediction": prediction[0]}