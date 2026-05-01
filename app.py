import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Modeli yükle
bundle = joblib.load('churn_project_bundle.joblib')
model, threshold, features = bundle['model'], bundle['threshold'], bundle['feature_names']

st.title("🛡️ ChurnGuard AI")

# Girdiler
tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly = st.number_input("Monthly Charges ($)", value=50.0)
total = st.number_input("Total Charges ($)", value=600.0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment = st.selectbox("Payment", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
senior = st.checkbox("Senior Citizen")

if st.button("Predict Risk"):
    # ÇÖZÜM: DataFrame'i doğrudan 'float' tipinde başlatıyoruz
    df = pd.DataFrame(0.0, index=[0], columns=features, dtype=float)
    
    # Değerleri ata
    df.at[0, 'tenure'] = float(tenure)
    df.at[0, 'MonthlyCharges'] = float(monthly)
    df.at[0, 'TotalCharges'] = np.log1p(total)
    df.at[0, 'SeniorCitizen'] = 1.0 if senior else 0.0
    df.at[0, 'Contract_Score'] = float({"Month-to-month": 0, "One year": 1, "Two year": 2}[contract])
    
    pay_col = f"PaymentMethod_{payment}"
    if pay_col in features: df.at[0, pay_col] = 1.0

    # Tahmin
    prob = model.predict_proba(df)[:, 1][0]
    risk = "HIGH RISK" if prob >= threshold else "LOW RISK"
    color = "red" if risk == "HIGH RISK" else "green"
    
    st.markdown(f"### Status: :{color}[{risk}]")
    st.metric("Churn Probability", f"{prob*100:.2f}%")
    st.progress(prob)
