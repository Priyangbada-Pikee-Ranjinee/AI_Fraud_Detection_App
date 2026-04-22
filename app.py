import streamlit as st
import pandas as pd

from utils.predictor import load_models, predict

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("AI Fraud Detection Dashboard")

models, scaler, feature_columns = load_models()

model_name = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)

model = models[model_name]

st.subheader("Transaction Input")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Amount", value=100.0)
    time = st.number_input("Time", value=10000.0)

with col2:
    v_values = {}
    for i in range(1, 29):
        v_values[f"V{i}"] = st.number_input(f"V{i}", value=0.0)

if st.button("Predict Fraud"):

    input_data = pd.DataFrame([{**v_values, "Amount": amount, "Time": time}])

    pred, proba = predict(model, scaler, feature_columns, input_data)

    if pred == 1:
        st.error("Fraud Detected")
    else:
        st.success("Legitimate Transaction")

    st.metric("Fraud Probability", f"{proba:.2%}")
    st.progress(int(proba * 100))

    if proba > 0.8:
        st.error("High Risk")
    elif proba > 0.5:
        st.warning("Medium Risk")
    else:
        st.success("Low Risk")