import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load trained model
model = joblib.load("credit_fraud_model.pkl")

st.set_page_config(layout="wide")
st.title("💳 Credit Card Fraud Detection")

# Load test dataset
try:
    test_data = pd.read_csv("test_data.csv")
except FileNotFoundError:
    test_data = None

# Session state to store prediction history
if "history" not in st.session_state:
    st.session_state.history = []

# Define columns used for prediction
columns = [f'V{i}' for i in range(1, 29)] + ['NormalizedAmount']

# --- Input Mode Selection ---
st.markdown("### 📥 Choose Input Method")
input_mode = st.radio("Select how you'd like to provide input:", 
                      ("Manual Entry", "Upload Single-Row CSV", "Select Row from Test Dataset"))

# --- Input Data Collection ---
input_df = None

if input_mode == "Manual Entry":
    st.markdown("### ✍️ Enter Transaction Details")
    input_features = {}
    col1, col2 = st.columns(2)

    with col1:
        for col in columns[:int(len(columns)/2)]:
            input_features[col] = st.number_input(f"{col}", value=0.0)

    with col2:
        for col in columns[int(len(columns)/2):]:
            input_features[col] = st.number_input(f"{col}", value=0.0)

    input_df = pd.DataFrame([input_features])

elif input_mode == "Upload Single-Row CSV":
    uploaded_file = st.file_uploader("📤 Upload a CSV file with a single row", type=["csv"])
    if uploaded_file:
        uploaded_df = pd.read_csv(uploaded_file)
        if uploaded_df.shape[0] == 1 and all(col in uploaded_df.columns for col in columns):
            input_df = uploaded_df
            st.success("✅ Uploaded data accepted for prediction.")
        else:
            st.error("⚠️ CSV must contain exactly one row and all required columns.")

elif input_mode == "Select Row from Test Dataset":
    if test_data is not None:
        st.markdown("### 🔢 Select a row from test data")
        selected_index = st.number_input("Choose a row index", min_value=0, max_value=len(test_data)-1)
        input_df = test_data.iloc[[selected_index]].copy()
    else:
        st.error("❌ Test dataset not found. Please make sure 'test_data.csv' exists.")

# --- Prediction ---
if input_df is not None and st.button("🔍 Predict"):
    # Keep only the features model was trained on
    input_df = input_df[columns]

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][prediction]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    st.markdown(f"### 🧠 Model Confidence: `{prob:.2f}`")

    label = "Fraud" if prediction == 1 else "Genuine"

    # Save in session history
    if not any(input_df.equals(hist[0]) for hist in st.session_state.history):
        st.session_state.history.append((input_df, label, timestamp))

    # Show result
    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Genuine Transaction.")

# --- Clear History ---
st.markdown("---")
if st.button("🧹 Clear Prediction History"):
    st.session_state.history.clear()
    st.success("Prediction history cleared!")

# --- Prediction History ---
st.subheader("🧾 Prediction History")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### ❌ Fraud")
    for features, label, time in st.session_state.history:
        if label == "Fraud":
            st.markdown(f"🔺 Transaction<br><small>🕓 {time}</small>", unsafe_allow_html=True)

with col2:
    st.markdown("### ✅ Genuine")
    for features, label, time in st.session_state.history:
        if label == "Genuine":
            st.markdown(f"🔹 Transaction<br><small>🕓 {time}</small>", unsafe_allow_html=True)

# --- Show Test Dataset ---
st.markdown("---")
with st.expander("📘 View Test Dataset"):
    if test_data is not None:
        st.dataframe(test_data)
    else:
        st.error("Test dataset not found. Make sure 'test_data.csv' exists.")
