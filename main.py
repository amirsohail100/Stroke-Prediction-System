from joblib import load
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

model = load("model_pipeline.pkl")
COLUMNS = load("columns.pkl")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Data(BaseModel):
    gender: Literal['Male', 'Female']
    age: int = Field(..., ge=10, le=100)
    hypertension: int
    heart_disease: int
    ever_married: Literal['Yes', 'No']
    work_type: Literal['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']
    Residence_type: Literal['Urban', 'Rural']
    avg_glucose_level: float
    bmi: float
    smoking_status: Literal['formerly smoked', 'never smoked', 'smokes', 'Unknown']

class PredictionResponse(BaseModel):
    predicted: Literal["Yes","No"]

@app.get("/")
def greet():
    return {"message": "Welcome to Amir WED"}


@app.post('/predict', response_model=PredictionResponse)
def predict(data: Data):

    input_row = pd.DataFrame([{
        "gender":data.gender,
        "age":data.age,
        "hypertension":data.hypertension,
        "heart_disease":data.heart_disease,
        "ever_married":data.ever_married,
        "work_type":data.work_type,
        "Residence_type":data.Residence_type,
        "avg_glucose_level":data.avg_glucose_level,
        "bmi":data.bmi,
        "smoking_status":data.smoking_status
    }])

    prediction = model.predict(input_row)[0]

    if prediction == 1:
        prediction = "Yes"
    else:
        prediction = "No"

    return PredictionResponse(predicted_mental_health_score=prediction)