from joblib import load
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

# Model aur columns load kar rahe hain
model = load("model_pipeline.pkl")
COLUMNS = load("columns.pkl")

app = FastAPI(title="Stroke Prediction API")

# CORS Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Stroke dataset ke real values ke hisaab se Pydantic Model
class Data(BaseModel):
    gender: Literal['Male', 'Female', 'Other']
    age: float = Field(..., ge=0.0, le=120.0, description="Age of the patient")
    hypertension: int = Field(..., ge=0, le=1, description="0 for no, 1 for hypertension")
    heart_disease: int = Field(..., ge=0, le=1, description="0 for no, 1 for heart disease")
    ever_married: Literal['Yes', 'No']
    work_type: Literal['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']
    Residence_type: Literal['Urban', 'Rural']
    avg_glucose_level: float = Field(..., gt=0, description="Average glucose level in blood")
    bmi: float = Field(..., gt=0, description="Body mass index")
    smoking_status: Literal['formerly smoked', 'never smoked', 'smokes', 'Unknown']

class PredictionResponse(BaseModel):
    predicted: Literal['Yes', 'No']

@app.get("/")
def greet():
    return {"message": "Welcome to Amir WED - Stroke Prediction API"}

@app.post('/predict', response_model=PredictionResponse)
def predict(data: Data):
    # Input data ko DataFrame mein convert karna
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
    
    # Model prediction
    prediction = model.predict(input_row)[0]
    
    # Numeric prediction ko 'Yes' / 'No' mein convert karna
    if prediction == 1:
        result = "Yes"
    else:
        result = "No"
        
    # Corrected response return kar rahe hain (fixed the variable mismatch bug)
    return PredictionResponse(predicted=result)