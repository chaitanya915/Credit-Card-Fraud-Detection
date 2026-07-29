"""
======================================================
Credit Card Fraud Detection
Batch Prediction Page

Model:
Random Forest

Explainability:
SHAP

======================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import shap

from src.predictor import predict_transaction
from src.preprocessing import validate_transaction
from src.explain import generate_shap_values


# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

st.set_page_config(

    page_title="Batch Prediction",

    page_icon="📂",

    layout="wide",

    initial_sidebar_state="expanded"

)


# ----------------------------------------------------
# CSS
# ----------------------------------------------------

st.markdown("""

<style>

.big-font{

font-size:38px;

font-weight:bold;

color:#0068C9;

}

.section{

font-size:24px;

font-weight:bold;

padding-top:20px;

}

.footer{

text-align:center;

color:gray;

padding:20px;

}

</style>

""",unsafe_allow_html=True)


# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

with st.sidebar:

    st.title("Batch Prediction")

    st.success(

        """
        Upload a CSV file

        Predict every transaction

        Download results

        """

    )

    st.markdown("---")

    st.write("Model")

    st.info("Random Forest")

    st.write("Explainability")

    st.info("SHAP")

    st.markdown("---")

    st.caption("Supports processed dataset only.")


# ----------------------------------------------------
# Title
# ----------------------------------------------------

st.markdown(

'<p class="big-font">📂 Batch Fraud Prediction</p>',

unsafe_allow_html=True

)

st.write(

"""

Upload a CSV file containing multiple transactions.

The model predicts every transaction and calculates

fraud probability.

"""

)

st.divider()


# ----------------------------------------------------
# About
# ----------------------------------------------------

with st.expander("About Batch Prediction"):

    st.write(

        """

This page predicts fraud for multiple transactions.

Required input format

Time

V1

...

V28

Amount

Exactly the same order used while training.

"""

    )


# ----------------------------------------------------
# File Upload
# ----------------------------------------------------

uploaded_file = st.file_uploader(

"Upload CSV File",

type=["csv"]

)


# ----------------------------------------------------
# No File Uploaded
# ----------------------------------------------------

if uploaded_file is None:

    st.info(

        "Please upload a CSV file to continue."

    )

    st.stop()


# ----------------------------------------------------
# Read CSV
# ----------------------------------------------------

try:

    data = pd.read_csv(uploaded_file)

except Exception as e:

    st.error(e)

    st.stop()


# ----------------------------------------------------
# Validation
# ----------------------------------------------------

try:

    data = validate_transaction(data)

except Exception as e:

    st.error(e)

    st.stop()


# ----------------------------------------------------
# Column Check
# ----------------------------------------------------

expected_columns = [

"Time",

"V1","V2","V3","V4","V5",

"V6","V7","V8","V9","V10",

"V11","V12","V13","V14","V15",

"V16","V17","V18","V19","V20",

"V21","V22","V23","V24","V25",

"V26","V27","V28",

"Amount"

]

if list(data.columns) != expected_columns:

    st.error(

"""

Uploaded dataset does not match

the expected feature order.

"""

    )

    st.write("Expected Columns")

    st.write(expected_columns)

    st.stop()


# ----------------------------------------------------
# Dataset Loaded
# ----------------------------------------------------

st.success(

f"{len(data)} transactions loaded successfully."

)


# ----------------------------------------------------
# Dataset Preview
# ----------------------------------------------------

st.markdown(

'<p class="section">Dataset Preview</p>',

unsafe_allow_html=True

)

st.dataframe(

data.head(),

use_container_width=True

)


# ----------------------------------------------------
# Dataset Summary
# ----------------------------------------------------

col1,col2,col3 = st.columns(3)

with col1:

    st.metric(

        "Rows",

        len(data)

    )

with col2:

    st.metric(

        "Columns",

        len(data.columns)

    )

with col3:

    st.metric(

        "Missing Values",

        int(data.isnull().sum().sum())

    )


# ----------------------------------------------------
# Dataset Statistics
# ----------------------------------------------------

with st.expander(

"Dataset Statistics"

):

    st.dataframe(

        data.describe(),

        use_container_width=True

    )


# ----------------------------------------------------
# Prediction Button
# ----------------------------------------------------

predict_button = st.button(

"Predict All Transactions",

use_container_width=True

)


# =====================================================
# Batch Prediction Engine
# =====================================================

if predict_button:

    st.markdown("---")

    st.subheader("Batch Prediction Progress")

    progress_bar = st.progress(0)

    status_text = st.empty()

    prediction_container = st.empty()

    import time

    start_time = time.time()

    predictions = []

    probabilities = []

    total_rows = len(data)

    # -------------------------------------------------
    # Predict Every Transaction
    # -------------------------------------------------

    for index in range(total_rows):

        row = data.iloc[[index]]

        prediction, probability = predict_transaction(row)

        predictions.append(int(prediction))

        probabilities.append(float(probability))

        percent = int(((index + 1) / total_rows) * 100)

        progress_bar.progress(percent)

        status_text.write(

            f"Processing transaction "

            f"{index+1} of {total_rows}"

        )

    end_time = time.time()

    execution_time = end_time - start_time

    progress_bar.progress(100)

    status_text.success(

        "Prediction completed successfully."

    )

    # -------------------------------------------------
    # Add Results
    # -------------------------------------------------

    results = data.copy()

    results["Prediction"] = predictions

    results["Fraud Probability"] = probabilities

    results["Prediction Label"] = results[
        "Prediction"
    ].map({

        0: "Genuine",

        1: "Fraud"

    })

    # -------------------------------------------------
    # Risk Level
    # -------------------------------------------------

    def risk_level(prob):

        if prob >= 0.80:

            return "High"

        elif prob >= 0.50:

            return "Medium"

        return "Low"

    results["Risk Level"] = results[
        "Fraud Probability"
    ].apply(risk_level)

    # -------------------------------------------------
    # Timestamp
    # -------------------------------------------------

    results["Prediction Time"] = datetime.now().strftime(

        "%d-%m-%Y %H:%M:%S"

    )

    # -------------------------------------------------
    # Save to Session
    # -------------------------------------------------

    st.session_state.batch_results = results

    # -------------------------------------------------
    # Performance Metrics
    # -------------------------------------------------

    st.markdown("---")

    st.subheader("Prediction Performance")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(

            "Transactions",

            total_rows

        )

    with col2:

        st.metric(

            "Execution Time",

            f"{execution_time:.2f} sec"

        )

    with col3:

        st.metric(

            "Average / Transaction",

            f"{execution_time/total_rows:.4f} sec"

        )

    # -------------------------------------------------
    # Preview
    # -------------------------------------------------

    st.markdown("---")

    st.subheader("Prediction Preview")

    prediction_container.dataframe(

        results.head(20),

        use_container_width=True,

        hide_index=True

    )

    # -------------------------------------------------
    # Store Statistics
    # -------------------------------------------------

    fraud_count = int(

        (results["Prediction"] == 1).sum()

    )

    genuine_count = int(

        (results["Prediction"] == 0).sum()

    )

    fraud_rate = (

        fraud_count / total_rows

    ) * 100

    st.session_state.batch_statistics = {

        "total": total_rows,

        "fraud": fraud_count,

        "genuine": genuine_count,

        "fraud_rate": fraud_rate,

        "execution_time": execution_time

    }



# =====================================================
# Batch Prediction Dashboard
# =====================================================

if "batch_results" in st.session_state:

    results = st.session_state.batch_results

    stats = st.session_state.batch_statistics

    st.markdown("---")

    st.header("📊 Batch Prediction Dashboard")

    # =================================================
    # KPI Cards
    # =================================================

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:

        st.metric(
            "Total Transactions",
            stats["total"]
        )

    with kpi2:

        st.metric(
            "Fraud Transactions",
            stats["fraud"]
        )

    with kpi3:

        st.metric(
            "Genuine Transactions",
            stats["genuine"]
        )

    with kpi4:

        st.metric(
            "Fraud Rate",
            f"{stats['fraud_rate']:.2f}%"
        )

    # =================================================
    # Prediction Distribution
    # =================================================

    st.markdown("---")

    st.subheader("Prediction Distribution")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        prediction_counts = results[
            "Prediction Label"
        ].value_counts()

        fig1, ax1 = plt.subplots(figsize=(5,5))

        ax1.pie(

            prediction_counts,

            labels=prediction_counts.index,

            autopct="%1.1f%%",

            startangle=90

        )

        ax1.set_title("Fraud vs Genuine")

        st.pyplot(fig1)

        plt.close(fig1)

    with chart_col2:

        risk_counts = results[
            "Risk Level"
        ].value_counts()

        fig2, ax2 = plt.subplots(figsize=(6,4))

        ax2.bar(

            risk_counts.index,

            risk_counts.values

        )

        ax2.set_title("Risk Level Distribution")

        ax2.set_xlabel("Risk Level")

        ax2.set_ylabel("Transactions")

        st.pyplot(fig2)

        plt.close(fig2)

    # =================================================
    # Fraud Probability Distribution
    # =================================================

    st.markdown("---")

    st.subheader("Fraud Probability Distribution")

    fig3, ax3 = plt.subplots(figsize=(8,4))

    ax3.hist(

        results["Fraud Probability"],

        bins=20

    )

    ax3.set_xlabel("Fraud Probability")

    ax3.set_ylabel("Frequency")

    ax3.set_title("Distribution of Fraud Probability")

    st.pyplot(fig3)

    plt.close(fig3)

    # =================================================
    # Filter Results
    # =================================================

    st.markdown("---")

    st.subheader("Filter Prediction Results")

    filter_option = st.selectbox(

        "Select Prediction",

        [

            "All",

            "Fraud",

            "Genuine",

            "High Risk",

            "Medium Risk",

            "Low Risk"

        ]

    )

    filtered_results = results.copy()

    if filter_option == "Fraud":

        filtered_results = results[
            results["Prediction Label"] == "Fraud"
        ]

    elif filter_option == "Genuine":

        filtered_results = results[
            results["Prediction Label"] == "Genuine"
        ]

    elif filter_option == "High Risk":

        filtered_results = results[
            results["Risk Level"] == "High"
        ]

    elif filter_option == "Medium Risk":

        filtered_results = results[
            results["Risk Level"] == "Medium"
        ]

    elif filter_option == "Low Risk":

        filtered_results = results[
            results["Risk Level"] == "Low"
        ]

    st.write(

        f"Showing {len(filtered_results)} transactions"

    )

    st.dataframe(

        filtered_results,

        use_container_width=True,

        hide_index=True

    )

    # =================================================
    # Fraud Probability Slider
    # =================================================

    st.markdown("---")

    st.subheader("Filter by Fraud Probability")

    threshold = st.slider(

        "Minimum Fraud Probability",

        min_value=0.0,

        max_value=1.0,

        value=0.50,

        step=0.01

    )

    probability_results = results[

        results["Fraud Probability"] >= threshold

    ]

    st.write(

        f"Transactions above {threshold:.2f}"

    )

    st.dataframe(

        probability_results,

        use_container_width=True,

        hide_index=True

    )

    # =================================================
    # High Risk Transactions
    # =================================================

    st.markdown("---")

    st.subheader("Top High-Risk Transactions")

    top10 = results.sort_values(

        "Fraud Probability",

        ascending=False

    ).head(10)

    st.dataframe(

        top10,

        use_container_width=True,

        hide_index=True

    )

    # =================================================
    # Summary Table
    # =================================================

    st.markdown("---")

    st.subheader("Summary")

    summary = pd.DataFrame({

        "Metric":[

            "Total Transactions",

            "Fraud Transactions",

            "Genuine Transactions",

            "Fraud Rate",

            "Execution Time"

        ],

        "Value":[

            stats["total"],

            stats["fraud"],

            stats["genuine"],

            f"{stats['fraud_rate']:.2f}%",

            f"{stats['execution_time']:.2f} sec"

        ]

    })

    st.dataframe(

        summary,

        use_container_width=True,

        hide_index=True

    )

# =====================================================
# SHAP Explainability
# =====================================================

if "batch_results" in st.session_state:

    results = st.session_state.batch_results

    st.markdown("---")

    st.header("🧠 Explain Prediction using SHAP")

    st.write(
        """
        Select a transaction to understand why the
        Random Forest predicted it as Fraud or Genuine.
        """
    )

    # -------------------------------------------------
    # Transaction Selection
    # -------------------------------------------------

    transaction_index = st.number_input(

        "Transaction Index",

        min_value=0,

        max_value=len(results)-1,

        value=0,

        step=1

    )

    transaction = results.iloc[[transaction_index]].copy()

    # Remove prediction columns before SHAP
    feature_columns = [

        "Time",

        "V1","V2","V3","V4","V5",

        "V6","V7","V8","V9","V10",

        "V11","V12","V13","V14","V15",

        "V16","V17","V18","V19","V20",

        "V21","V22","V23","V24","V25",

        "V26","V27","V28",

        "Amount"

    ]

    transaction_features = transaction[feature_columns]

    # -------------------------------------------------
    # Selected Transaction
    # -------------------------------------------------

    st.subheader("Selected Transaction")

    st.dataframe(

        transaction,

        use_container_width=True,

        hide_index=True

    )

    # -------------------------------------------------
    # Generate SHAP
    # -------------------------------------------------

    with st.spinner("Generating SHAP explanation..."):

        explanation = generate_shap_values(

            transaction_features

        )

    st.success("SHAP explanation generated.")

    # -------------------------------------------------
    # Waterfall Plot
    # -------------------------------------------------

    st.markdown("---")

    st.subheader("SHAP Waterfall Plot")

    fig = plt.figure(figsize=(10,6))

    shap.plots.waterfall(

        explanation[0],

        max_display=12,

        show=False

    )

    st.pyplot(fig)

    plt.close(fig)

    # -------------------------------------------------
    # Feature Contribution Table
    # -------------------------------------------------

    contribution_df = pd.DataFrame({

        "Feature": explanation.feature_names,

        "SHAP Value": explanation.values[0]

    })

    contribution_df["Absolute"] = contribution_df[
        "SHAP Value"
    ].abs()

    contribution_df = contribution_df.sort_values(

        "Absolute",

        ascending=False

    )

    st.markdown("---")

    st.subheader("Top Contributing Features")

    st.dataframe(

        contribution_df.head(10),

        use_container_width=True,

        hide_index=True

    )

    # -------------------------------------------------
    # Horizontal Bar Chart
    # -------------------------------------------------

    st.subheader("Feature Contribution Chart")

    top10 = contribution_df.head(10)

    fig2, ax2 = plt.subplots(figsize=(8,5))

    ax2.barh(

        top10["Feature"],

        top10["SHAP Value"]

    )

    ax2.set_xlabel("SHAP Value")

    ax2.set_ylabel("Feature")

    ax2.invert_yaxis()

    st.pyplot(fig2)

    plt.close(fig2)

    # -------------------------------------------------
    # Positive / Negative Contributions
    # -------------------------------------------------

    col1,col2 = st.columns(2)

    with col1:

        st.subheader("Increase Fraud Probability")

        positive = contribution_df[
            contribution_df["SHAP Value"] > 0
        ]

        st.dataframe(

            positive.head(10),

            use_container_width=True,

            hide_index=True

        )

    with col2:

        st.subheader("Decrease Fraud Probability")

        negative = contribution_df[
            contribution_df["SHAP Value"] < 0
        ]

        st.dataframe(

            negative.head(10),

            use_container_width=True,

            hide_index=True

        )

    # -------------------------------------------------
    # Prediction Explanation
    # -------------------------------------------------

    st.markdown("---")

    st.subheader("Explanation Summary")

    top_feature = contribution_df.iloc[0]

    feature_name = top_feature["Feature"]

    shap_value = top_feature["SHAP Value"]

    direction = "increased" if shap_value > 0 else "decreased"

    st.info(

        f"""
        **{feature_name}** had the strongest impact
        on this prediction.

        SHAP Value: **{shap_value:.4f}**

        This feature **{direction}**
        the fraud probability.
        """

    )

    # -------------------------------------------------
    # Complete SHAP Table
    # -------------------------------------------------

    st.markdown("---")

    st.subheader("Complete SHAP Values")

    st.dataframe(

        contribution_df,

        use_container_width=True,

        hide_index=True

    )

    # -------------------------------------------------
    # Download SHAP Report
    # -------------------------------------------------

    shap_report = contribution_df.copy()

    shap_report["Prediction"] = transaction[
        "Prediction Label"
    ].values[0]

    shap_report["Fraud Probability"] = transaction[
        "Fraud Probability"
    ].values[0]

    shap_csv = shap_report.to_csv(index=False)

    st.download_button(

        "📥 Download SHAP Report",

        shap_csv,

        file_name=f"shap_transaction_{transaction_index}.csv",

        mime="text/csv",

        use_container_width=True

    )

    # -------------------------------------------------
    # Force Plot (Optional)
    # -------------------------------------------------

    st.markdown("---")

    st.subheader("SHAP Force Plot")

    try:

        force_fig = plt.figure(figsize=(14,3))

        shap.plots.force(

            explanation[0],

            matplotlib=True,

            show=False

        )

        st.pyplot(force_fig)

        plt.close(force_fig)

    except Exception:

        st.warning(
            "Force Plot is not supported in every "
            "SHAP 0.52.0 configuration. "
            "Use the Waterfall Plot as the primary explanation."
        )

# ==========================================================
# Export Prediction Results
# ==========================================================

if "batch_results" in st.session_state:

    results = st.session_state.batch_results

    st.markdown("---")

    st.header("📥 Export Results")

    csv_results = results.to_csv(index=False)

    st.download_button(

        label="📄 Download Complete Prediction Results",

        data=csv_results,

        file_name="batch_prediction_results.csv",

        mime="text/csv",

        use_container_width=True

    )


# ==========================================================
# Download Fraud Transactions Only
# ==========================================================

    fraud_only = results[
        results["Prediction"] == 1
    ]

    csv_fraud = fraud_only.to_csv(index=False)

    st.download_button(

        label="🚨 Download Fraud Transactions",

        data=csv_fraud,

        file_name="fraud_transactions.csv",

        mime="text/csv",

        use_container_width=True

    )


# ==========================================================
# Download Genuine Transactions Only
# ==========================================================

    genuine_only = results[
        results["Prediction"] == 0
    ]

    csv_genuine = genuine_only.to_csv(index=False)

    st.download_button(

        label="✅ Download Genuine Transactions",

        data=csv_genuine,

        file_name="genuine_transactions.csv",

        mime="text/csv",

        use_container_width=True

    )


# ==========================================================
# Search Transactions
# ==========================================================

st.markdown("---")

st.header("🔍 Search Transactions")

if "batch_results" in st.session_state:

    results = st.session_state.batch_results

    search_index = st.number_input(

        "Enter Transaction Index",

        min_value=0,

        max_value=len(results)-1,

        value=0,

        step=1

    )

    st.dataframe(

        results.iloc[[search_index]],

        use_container_width=True,

        hide_index=True

    )


# ==========================================================
# Prediction Summary
# ==========================================================

st.markdown("---")

st.header("📋 Prediction Summary")

if "batch_statistics" in st.session_state:

    stats = st.session_state.batch_statistics

    summary = pd.DataFrame({

        "Metric":[

            "Total Transactions",

            "Fraud Transactions",

            "Genuine Transactions",

            "Fraud Rate",

            "Execution Time (sec)"

        ],

        "Value":[

            stats["total"],

            stats["fraud"],

            stats["genuine"],

            round(stats["fraud_rate"],2),

            round(stats["execution_time"],2)

        ]

    })

    st.dataframe(

        summary,

        use_container_width=True,

        hide_index=True

    )


# ==========================================================
# Save Results in Session
# ==========================================================

if "prediction_history" not in st.session_state:

    st.session_state.prediction_history = []

if "batch_results" in st.session_state:

    st.session_state.prediction_history.append(

        {

            "Timestamp": datetime.now(),

            "Transactions": len(results),

            "Fraud": stats["fraud"],

            "Genuine": stats["genuine"]

        }

    )


# ==========================================================
# Batch Prediction History
# ==========================================================

st.markdown("---")

st.header("🗂 Batch Prediction History")

history = pd.DataFrame(

    st.session_state.prediction_history

)

if len(history):

    st.dataframe(

        history,

        use_container_width=True,

        hide_index=True

    )


# ==========================================================
# Clear History
# ==========================================================

col1, col2 = st.columns(2)

with col1:

    if st.button(

        "🗑 Clear Prediction History",

        use_container_width=True

    ):

        st.session_state.prediction_history = []

        st.success(

            "History Cleared"

        )

        st.rerun()

with col2:

    if st.button(

        "🔄 Reset Batch Prediction",

        use_container_width=True

    ):

        st.rerun()


# ==========================================================
# System Information
# ==========================================================

st.markdown("---")

st.header("⚙ System Information")

info1, info2, info3 = st.columns(3)

with info1:

    st.success("✅ Random Forest Loaded")

with info2:

    st.success("✅ SHAP Ready")

with info3:

    st.success("✅ Batch Prediction Ready")


# ==========================================================
# User Instructions
# ==========================================================

st.markdown("---")

with st.expander(

    "📖 User Instructions"

):

    st.markdown("""

### Supported File

CSV

### Required Columns

Time

V1

...

V28

Amount

### Output

Prediction

Fraud Probability

Risk Level

SHAP Explanation

### Downloads

Complete Results

Fraud Transactions

Genuine Transactions

SHAP Report

""")


# ==========================================================
# About Project
# ==========================================================

st.markdown("---")

st.header("ℹ Project Information")

left,right = st.columns(2)

with left:

    st.write("**Project**")

    st.write("Credit Card Fraud Detection")

    st.write("**Algorithm**")

    st.write("Random Forest")

    st.write("**Explainable AI**")

    st.write("SHAP")

with right:

    st.write("**Dataset**")

    st.write("European Credit Card Dataset")

    st.write("**Framework**")

    st.write("Streamlit")

    st.write("**Language**")

    st.write("Python")


# ==========================================================
# Footer
# ==========================================================

st.markdown("---")

st.markdown(

"""

<div style='text-align:center;
padding:20px;
color:gray;'>

<h4>

Credit Card Fraud Detection using

Machine Learning and Explainable AI

</h4>

Developed using

Python | Streamlit | Random Forest | SHAP

<br>

Final Year B.E. Project

</div>

""",

unsafe_allow_html=True

)