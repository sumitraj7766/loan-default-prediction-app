import streamlit as st
import joblib
import pandas as pd

from preprocessing import preprocess
from feature_engineering import feature_engineering

# Load model
model = joblib.load("model.pkl")

st.title("🏦 Loan Default Prediction System")

st.write("Enter customer details:")

# 🔹 Numeric Inputs
age = st.number_input("Age", value=25)
income = st.number_input("Income", value=50000)
emp_length = st.number_input("Employment Length (years)", value=2)
amount = st.number_input("Loan Amount", value=20000)
rate = st.number_input("Interest Rate", value=10.0)
percent_income = st.number_input("Loan % of Income", value=0.2)
cred_length = st.number_input("Credit History Length", value=3)

# 🔹 Status (0 / 1)
status = st.selectbox("Loan Status (0 = good, 1 = risky)", [0, 1])

# 🔹 Home Ownership (One-hot)
home = st.selectbox("Home Ownership", ["RENT", "OWN", "OTHER"])

home_rent = 1 if home == "RENT" else 0
home_own = 1 if home == "OWN" else 0
home_other = 1 if home == "OTHER" else 0

# 🔹 Loan Intent (One-hot)
intent = st.selectbox("Loan Intent", 
    ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"])

intent_edu = 1 if intent == "EDUCATION" else 0
intent_home = 1 if intent == "HOMEIMPROVEMENT" else 0
intent_med = 1 if intent == "MEDICAL" else 0
intent_personal = 1 if intent == "PERSONAL" else 0
intent_venture = 1 if intent == "VENTURE" else 0

# 🔘 Predict Button
if st.button("Predict"):

    # ✅ DataFrame EXACT same columns ke saath
    input_data = pd.DataFrame({
        'Age': [age],
        'Income': [income],
        'Emp_length': [emp_length],
        'Amount': [amount],
        'Rate': [rate],
        'Status': [status],
        'Percent_income': [percent_income],
        'Cred_length': [cred_length],
        'Home_OTHER': [home_other],
        'Home_OWN': [home_own],
        'Home_RENT': [home_rent],
        'Intent_EDUCATION': [intent_edu],
        'Intent_HOMEIMPROVEMENT': [intent_home],
        'Intent_MEDICAL': [intent_med],
        'Intent_PERSONAL': [intent_personal],
        'Intent_VENTURE': [intent_venture]
    })

    # 🔁 Apply same pipeline
    input_data = preprocess(input_data)
    input_data = feature_engineering(input_data)

    # 🤖 Prediction
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    # 🎯 Output
    if prediction == 1:
        st.error(f"⚠️ High Risk Customer (Default)\nProbability: {prob:.2f}")
    else:
        st.success(f"✅ Safe Customer\nProbability: {prob:.2f}")