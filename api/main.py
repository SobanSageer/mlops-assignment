from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="ML Inference API", description="API for model predictions", version="1.0.0")

MODEL_PATH = "models/model.pkl"

# Load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

class PredictionRequest(BaseModel):
    age: float
    bmi: float
    blood_pressure: float
    cholesterol: float
    glucose: float
    smoker: int  # 0 for no, 1 for yes

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Prepare data
        data = {
            "age": request.age,
            "bmi": request.bmi,
            "blood_pressure": request.blood_pressure,
            "cholesterol": request.cholesterol,
            "glucose": request.glucose,
            "smoker": request.smoker
        }
        df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0].tolist()
        
        return {
            "prediction": int(prediction),
            "probability": probability
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))