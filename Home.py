"""
=========================================================
Credit Card Fraud Detection Using Machine Learning
Home.py

Main Entry Point for Streamlit Application
=========================================================
"""

import streamlit as st

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# Custom CSS
# =====================================================

st.markdown("""
<style>

.main-title{
    font-size:42px;
    font-weight:bold;
    color:#1565C0;
}

.sub-title{
    font-size:22px;
    color:#555555;
}

.card{
    background-color:#F8F9FA;
    padding:20px;
    border-radius:10px;
    border:1px solid #DDDDDD;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:30px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# Sidebar
# =====================================================

with st.sidebar:

    st.title("💳 Navigation")

    st.success("Credit Card Fraud Detection")

    st.markdown("---")

    st.write("Use the sidebar to navigate:")

    st.write("🏠 Project Overview")

    st.write("📊 Data Analysis")

    st.write("🧾 Manual Prediction")

    st.write("📁 Batch Prediction")

    st.write("📈 Model Performance")

    st.write("🧠 SHAP Explainability")

    st.markdown("---")

    st.info("""
Developed using

• Python

• Streamlit

• Random Forest

• SHAP 0.52.0
""")

# =====================================================
# Header
# =====================================================

st.markdown(
    '<p class="main-title">💳 Credit Card Fraud Detection Using Machine Learning</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">Interactive Explainable AI Dashboard</p>',
    unsafe_allow_html=True
)

st.divider()

# =====================================================
# About Project
# =====================================================

st.header("📌 About the Project")

st.write("""
This project detects fraudulent credit card transactions using
Machine Learning.

The final model selected is **Random Forest**, based on its
high Accuracy, Precision, Recall, F1 Score, and ROC-AUC.

To improve transparency, **SHAP (SHapley Additive exPlanations)**
is used to explain every prediction.
""")

# =====================================================
# Features
# =====================================================

st.header("🚀 Features")

col1, col2 = st.columns(2)

with col1:

    st.success("""
✅ Data Preprocessing

✅ Exploratory Data Analysis

✅ Random Forest Model

✅ Manual Fraud Prediction

✅ Batch Prediction
""")

with col2:

    st.success("""
✅ Model Performance Dashboard

✅ SHAP Explainability

✅ Download Reports

✅ Interactive Visualizations

✅ Fraud Probability
""")

# =====================================================
# Workflow
# =====================================================

st.header("📈 Project Workflow")

st.code("""
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Train/Test Split
      │
      ▼
Random Forest Training
      │
      ▼
Model Evaluation
      │
      ▼
SHAP Explainability
      │
      ▼
Prediction Dashboard
""")

# =====================================================
# Technologies
# =====================================================

st.header("🛠 Technology Stack")

tech = {
    "Component": [
        "Programming Language",
        "Machine Learning",
        "Explainable AI",
        "Framework",
        "Visualization",
        "Dataset"
    ],
    "Technology": [
        "Python",
        "Scikit-Learn",
        "SHAP 0.52.0",
        "Streamlit",
        "Matplotlib",
        "European Credit Card Dataset"
    ]
}

import pandas as pd

st.dataframe(
    pd.DataFrame(tech),
    use_container_width=True,
    hide_index=True
)

# =====================================================
# Navigation Help
# =====================================================

st.header("📂 Application Pages")

st.info("""
Use the Streamlit sidebar (left side) to open:

• 1_Home

• 2_Manual_Prediction

• 3_Batch_Prediction

• 4_Model_Performance

• 5_SHAP_Explainability
""")

# =====================================================
# Footer
# =====================================================

st.markdown("---")

st.markdown(
"""
<div class='footer'>
<h3>Credit Card Fraud Detection Using Machine Learning</h3>

<p>
Random Forest + SHAP 0.52.0

<br>

Developed with Python, Streamlit, Scikit-Learn

</p>

</div>
""",
unsafe_allow_html=True
)