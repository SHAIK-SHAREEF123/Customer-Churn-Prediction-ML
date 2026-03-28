# Gender -> 1 Female    0 Male
# Churn -> 1 Yes    0 No
# Scaler is exported as scaler.pkl
# Model is exported as model.pkl
# Order of the X is -> 'Age', 'Gender', 'Tenure', 'MonthlyCharges'

import streamlit as st
import joblib
import numpy as np

# Page config
st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="centered")

# Load model
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

# Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📊 Customer Churn Prediction</h1>", unsafe_allow_html=True)

st.markdown("---")

st.write("🔍 Enter customer details below to predict whether the customer will churn or not.")

# Input Section (Card Style)
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("🎂 Age", min_value=10, max_value=100, value=30)
        tenure = st.number_input("📅 Tenure (months)", min_value=0, max_value=130, value=10)

    with col2:
        monthlycharge = st.number_input("💰 Monthly Charges", min_value=30, max_value=150, value=50)
        gender = st.selectbox("👤 Gender", ["Male", "Female"])

st.markdown("---")

# Button Centered
col1, col2, col3 = st.columns([1,2,1])
with col2:
    predictbutton = st.button("🚀 Predict", use_container_width=True)

st.markdown("---")

# Prediction Logic
if predictbutton:
    gender_selected = 1 if gender == "Female" else 0

    X = [age, gender_selected, tenure, monthlycharge]
    X_array = scaler.transform([X])

    prediction = model.predict(X_array)

    # Output Styling
    if prediction == 1:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is NOT likely to churn")

else:
    st.info("ℹ️ Fill the details and click Predict")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
