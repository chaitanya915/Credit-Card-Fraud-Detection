import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection System")

model = joblib.load("models/random_forest.pkl")

X_test = pd.read_csv("data/processed/X_test_scaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

explainer = shap.TreeExplainer(model)

st.success("Random Forest model loaded successfully.")
st.success("SHAP Explainer initialized successfully.")

st.subheader("Dataset Information")
st.write("Shape:", X_test.shape)

st.subheader("Sample Transactions")
st.dataframe(X_test.head())

sample = X_test.iloc[[0]]

prediction = model.predict(sample)

probability = model.predict_proba(sample)

st.subheader("Test Prediction")
st.write("Prediction:", "Fraud" if prediction[0] == 1 else "Genuine")

st.write("Prediction Probability")

st.write(pd.DataFrame(
    probability,
    columns=["Genuine", "Fraud"]
))