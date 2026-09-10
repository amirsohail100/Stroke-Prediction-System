from joblib import load
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

model = load("model_pipeline.pkl")
COLUMNS = load("columns.pkl")

app = FastAPI(title="Stroke Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Data(BaseModel):
    gender: Literal['Male', 'Female', 'Other']
    age: float = Field(..., ge=0.0, le=120.0)
    hypertension: int = Field(..., ge=0, le=1)
    heart_disease: int = Field(..., ge=0, le=1)
    ever_married: Literal['Yes', 'No']
    work_type: Literal['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']
    Residence_type: Literal['Urban', 'Rural']
    avg_glucose_level: float = Field(..., gt=0)
    bmi: float = Field(..., gt=0)
    smoking_status: Literal['formerly smoked', 'never smoked', 'smokes', 'Unknown']

# Ek behtar aur professional response structure
class PredictionResponse(BaseModel):
    prediction_label: Literal['High Risk', 'Low Risk']
    stroke_probability: float = Field(..., description="Confidence score of the prediction")

@app.get("/")
def greet():
    return {"message": "Welcome to Amir WED - Stroke Prediction API"}

@app.post('/predict', response_model=PredictionResponse)
def predict(data: Data):
    input_row = pd.DataFrame([{
        "gender": data.gender,
        "age": data.age,
        "hypertension": data.hypertension,
        "heart_disease": data.heart_disease,
        "ever_married": data.ever_married,
        "work_type": data.work_type,
        "Residence_type": data.Residence_type,
        "avg_glucose_level": data.avg_glucose_level,
        "bmi": data.bmi,
        "smoking_status": data.smoking_status
    }])
    
    # Class prediction (0 or 1)
    prediction = model.predict(input_row)[0]
    
    # Probability nikalna (agar model predict_proba support karta hai)
    try:
        probabilities = model.predict_proba(input_row)[0]
        confidence = float(probabilities[1] if prediction == 1 else probabilities[0])
    except Exception:
        confidence = 1.0 if prediction == 1 else 0.0  # Fallback agar pipeline mein proba na ho

    # Professional labels
    if prediction == 1:
        label = "High Risk"
    else:
        label = "Low Risk"
        
    return PredictionResponse(
        prediction_label=label,
        stroke_probability=round(confidence, 4)
    )