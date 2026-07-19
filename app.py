import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import os
import sys

# Get current directory where model files are located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set page config
st.set_page_config(
    page_title="Healthcare Stroke Prediction System",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .main {
            padding-top: 2rem;
        }
        .prediction-box {
            padding: 2rem;
            border-radius: 10px;
            margin-top: 2rem;
        }
        .high-risk {
            background-color: #ffebee;
            border-left: 4px solid #f44336;
        }
        .low-risk {
            background-color: #e8f5e9;
            border-left: 4px solid #4caf50;
        }
    </style>
""", unsafe_allow_html=True)

# Title and header
st.title("⚕️ Stroke Prediction System")
st.markdown("---")
st.markdown("**Predict stroke risk based on patient health data using Machine Learning**")

# Load model and preprocessing objects
try:
    model = load("Final_Model.pkl")
    scaler = load("scaler.pkl")
    columns = load("columns.pkl")
except FileNotFoundError:
    st.error("❌ Model files not found. Please ensure 'Final Model.pkl', 'scaler.pkl', and 'columns.pkl' are in the parent directory.")
    st.stop()

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Personal Information")
    age = st.slider("Age", min_value=18, max_value=100, value=45, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    ever_married = st.selectbox("Ever Married", ["No", "Yes"])
    
    st.subheader("💼 Work & Residence")
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Never_worked", "children"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])

with col2:
    st.subheader("🏥 Health Metrics")
    avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", min_value=50.0, max_value=300.0, value=120.0, step=1.0)
    bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    
    st.subheader("🚬 Lifestyle")
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# Encode categorical variables
gender_map = {"Female": 0, "Male": 1, "Other": 2}
work_map = {"Private": 0, "Self-employed": 1, "Govt_job": 2, "Never_worked": 3, "children": 4}
residence_map = {"Rural": 0, "Urban": 1}
smoking_map = {"never smoked": 0, "formerly smoked": 1, "smokes": 2, "Unknown": 3}
married_map = {"No": 0, "Yes": 1}
condition_map = {"No": 0, "Yes": 1}

# Prepare for scaling (scale only the numeric features)
numeric_features = np.array([[age, avg_glucose_level, bmi]])
scaled_features = scaler.transform(numeric_features)[0]

# Create final input for model (in correct order: gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status)
final_input = np.array([
    gender_map[gender],
    scaled_features[0],  # scaled age
    condition_map[hypertension],
    condition_map[heart_disease],
    married_map[ever_married],
    work_map[work_type],
    residence_map[residence_type],
    scaled_features[1],  # scaled avg_glucose_level
    scaled_features[2],  # scaled bmi
    smoking_map[smoking_status]
]).reshape(1, -1)

# Prediction button and results
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])

with col_btn1:
    if st.button("🔮 Predict Stroke Risk", use_container_width=True):
        # Make prediction
        prediction = model.predict(final_input)[0]
        probability = model.predict_proba(final_input)[0]
        
        # Store in session state
        st.session_state.prediction = prediction
        st.session_state.probability = probability

with col_btn2:
    if st.button("🔄 Reset Form", use_container_width=True):
        st.rerun()

# Display results
if "prediction" in st.session_state:
    prediction = st.session_state.prediction
    probability = st.session_state.probability
    
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    # Create result display
    if prediction == 1:
        risk_level = "HIGH RISK"
        risk_class = "high-risk"
        risk_color = "🔴"
        recommendation = "⚠️ **Please consult with a healthcare professional immediately for further evaluation and preventive measures.**"
    else:
        risk_level = "LOW RISK"
        risk_class = "low-risk"
        risk_color = "🟢"
        recommendation = "✅ **Continue maintaining healthy lifestyle habits. Regular check-ups are recommended.**"
    
    # Display prediction box
    col_result1, col_result2 = st.columns(2)
    
    with col_result1:
        st.markdown(f"""
            <div class="prediction-box {risk_class}">
                <h2>{risk_color} {risk_level}</h2>
                <p><strong>Stroke Risk Probability:</strong></p>
                <h3>{probability[1]*100:.2f}%</h3>
            </div>
        """, unsafe_allow_html=True)
    
    with col_result2:
        st.markdown(f"""
            <div class="prediction-box {risk_class}">
                <h4>Recommendation</h4>
                <p>{recommendation}</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Detailed breakdown
    st.subheader("📈 Probability Breakdown")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Low Risk Probability", f"{probability[0]*100:.2f}%")
    with col2:
        st.metric("High Risk Probability", f"{probability[1]*100:.2f}%")
    
    # Risk factors summary
    st.subheader("📋 Patient Summary")
    summary_df = pd.DataFrame({
        "Parameter": ["Age", "Gender", "Average Glucose Level", "BMI", "Work Type", "Smoking Status", "Marital Status", "Residence Type"],
        "Value": [age, gender, f"{avg_glucose_level} mg/dL", f"{bmi}", work_type, smoking_status, ever_married, residence_type]
    })
    st.table(summary_df)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: gray; font-size: 0.85rem;">
        <p>⚕️ <strong>Disclaimer:</strong> This is an AI-based prediction tool for educational and awareness purposes only. 
        It should not be used as a substitute for professional medical advice. Always consult a healthcare provider for medical decisions.</p>
        <p>Model: Gradient Boosting Classifier | Data: Healthcare Stroke Dataset</p>
    </div>
""", unsafe_allow_html=True)


