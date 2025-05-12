# Real-Time Fraud Detection Streamlit App


import streamlit as st
import pandas as pd
import joblib

# Load the trained XGBoost model
model = joblib.load("xgboost_fraud_model.joblib")

st.title("Real-Time Fraud Detection")
st.write("Upload a transaction CSV file to predict whether each entry is fraudulent.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(data.head())

    # Check if required features exist in uploaded file
    required_features = model.get_booster().feature_names
    if not all(feature in data.columns for feature in required_features):
        st.error("Uploaded file is missing required input features.")
    else:
        # Make predictions
        predictions = model.predict(data[required_features])
        data['Prediction'] = predictions

        st.subheader("🔍 Fraud Detection Summary")
        fraud_count = sum(predictions)
        st.success(f"✅ Total Fraudulent Transactions Detected: {fraud_count}")

        st.subheader("📋 Detailed Results")
        st.dataframe(data)
