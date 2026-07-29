import streamlit as st
import pandas as pd
from PIL import Image

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection System")

st.markdown("""

### Advanced Machine Learning + Explainable AI

Random Forest

SHAP Explainability

Real-Time Fraud Detection

""")

st.image(

    "assets/logo.png",

    width=200

)


st.header("Project Introduction")

st.write("""

Credit card fraud has become one of the most significant

financial crimes worldwide.

This project uses Machine Learning and Explainable AI

to detect fraudulent transactions.

The Random Forest classifier predicts whether a transaction

is Fraud or Genuine.

SHAP explains why the prediction was made.

""")


st.header("Project Objectives")

st.markdown("""

- Detect fraudulent transactions.

- Reduce financial losses.

- Improve prediction accuracy.

- Provide Explainable AI using SHAP.

- Assist banks in fraud investigation.

""")


st.header("Dataset Information")

col1,col2 = st.columns(2)

with col1:
    st.metric("Transactions","284,807")
    st.metric("Fraud","492")

with col2:
    st.metric("Features","30")
    st.metric("Fraud Rate","0.172%")

st.header("Technology Stack")

col1,col2,col3,col4 = st.columns(4)

with col1:
    st.success("Python")

with col2:
    st.success("Random Forest")

with col3:
    st.success("SHAP")

with col4:
    st.success("Streamlit")

st.header("Machine Learning Workflow")

st.markdown("""

Dataset

↓

Preprocessing

↓

Feature Engineering

↓

Random Forest

↓

Prediction

↓

Explainable AI

↓

Dashboard

""")


st.header("System Architecture")

st.code("""

User

↓

Streamlit UI

↓

Prediction Module

↓

Random Forest

↓

SHAP Explainer

↓

Prediction

""")

st.header("Project Features")

features = [

"Real-Time Prediction",

"Batch Prediction",

"SHAP Explainability",

"Random Forest Model",

"Fraud Probability",

"CSV Upload",

"Download Results"

]

for feature in features:

    st.write("✅",feature)



st.header("Model Summary")

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.metric("Accuracy","99.92%")

with c2:
    st.metric("Precision","76.29%")

with c3:
    st.metric("Recall","77.89%")

with c4:
    st.metric("ROC AUC","97.65%")




st.sidebar.success(

"""
Navigation

🏠 Home

📝 Manual Prediction

📂 Batch Prediction

📊 Model Performance

🧠 SHAP Explainability

ℹ About Project

"""
)


st.markdown("---")

st.markdown(

"""
Developed By

B.E. Computer Engineering Student

Credit Card Fraud Detection using Machine Learning and Explainable AI

2026

"""
)