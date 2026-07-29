"""
=========================================================
Credit Card Fraud Detection System
Manual Prediction Page

Model:
Random Forest

Explainability:
SHAP

Author:
Chaitanya Gurav

=========================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
from datetime import datetime

# Backend modules
from src.predictor import predict_transaction
from src.explain import generate_shap_values
from src.preprocessing import validate_transaction


# --------------------------------------------------------
# Page Configuration
# --------------------------------------------------------

st.set_page_config(
    page_title="Manual Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------------
# Custom CSS
# --------------------------------------------------------

st.markdown(
    """
    <style>

    .main-title{
        font-size:40px;
        font-weight:bold;
        color:#0E6EFD;
    }

    .sub-title{
        font-size:18px;
        color:gray;
    }

    .section-header{
        font-size:28px;
        font-weight:bold;
        color:#222;
        margin-top:20px;
    }

    .prediction-card{
        border-radius:12px;
        padding:20px;
        background-color:#F8F9FA;
        border:1px solid #E0E0E0;
    }

    .footer{
        color:gray;
        text-align:center;
        padding-top:40px;
        padding-bottom:20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------------
# Sidebar
# --------------------------------------------------------

with st.sidebar:

    st.image(
        "assets/logo.png",
        width=120
    )

    st.title("Navigation")

    st.success(
        """
        Current Page

        ✅ Manual Prediction
        """
    )

    st.markdown("---")

    st.header("Model")

    st.write("Random Forest")

    st.header("Explainable AI")

    st.write("SHAP 0.52.0")

    st.markdown("---")

    st.info(
        """
        Enter transaction details
        and click Predict.
        """
    )


# --------------------------------------------------------
# Header
# --------------------------------------------------------

st.markdown(
    '<p class="main-title">💳 Manual Fraud Prediction</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">'
    'Predict whether a credit card transaction is '
    'Fraud or Genuine using Random Forest.'
    '</p>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------------------------------
# Load Model
# --------------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load(
        "models/random_forest.pkl"
    )

    return model


model = load_model()


# --------------------------------------------------------
# Create SHAP Explainer
# --------------------------------------------------------

@st.cache_resource
def load_explainer():

    explainer = shap.TreeExplainer(model)

    return explainer


explainer = load_explainer()


# --------------------------------------------------------
# Helper Function
# Prediction Label
# --------------------------------------------------------

def prediction_label(prediction):

    if prediction == 1:

        return "Fraud"

    return "Genuine"


# --------------------------------------------------------
# Helper Function
# Risk Level
# --------------------------------------------------------

def risk_level(probability):

    if probability >= 0.80:

        return "High Risk"

    elif probability >= 0.50:

        return "Medium Risk"

    else:

        return "Low Risk"


# --------------------------------------------------------
# Helper Function
# Risk Color
# --------------------------------------------------------

def risk_color(probability):

    if probability >= 0.80:

        return "red"

    elif probability >= 0.50:

        return "orange"

    else:

        return "green"


# --------------------------------------------------------
# Helper Function
# Create Result DataFrame
# --------------------------------------------------------

def create_result_dataframe(
        prediction,
        probability):

    result = pd.DataFrame({

        "Prediction": [
            prediction_label(prediction)
        ],

        "Fraud Probability": [
            probability
        ],

        "Risk Level": [
            risk_level(probability)
        ],

        "Timestamp": [
            datetime.now()
        ]

    })

    return result


# --------------------------------------------------------
# Session State
# --------------------------------------------------------

if "prediction_history" not in st.session_state:

    st.session_state.prediction_history = []


# --------------------------------------------------------
# Information
# --------------------------------------------------------

with st.expander(
        "ℹ About Manual Prediction",
        expanded=False):

    st.write(
        """
        This page allows you to manually enter
        transaction information.

        The trained Random Forest model predicts
        whether the transaction is Fraud or Genuine.

        SHAP Explainable AI is then used to explain
        why the prediction was made.
        """
    )


# --------------------------------------------------------
# Section Header
# --------------------------------------------------------

st.markdown(
    '<p class="section-header">'
    'Transaction Details'
    '</p>',
    unsafe_allow_html=True
)

st.write(
    "Enter all transaction features below."
)


# ========================================================
# Transaction Input Form
# ========================================================

st.markdown(
    """
    <div style='background-color:#f8f9fa;
                padding:15px;
                border-radius:10px;
                border:1px solid #dddddd;'>

    <h3>Enter Transaction Features</h3>

    Enter all 30 transaction features below.
    The values should match the format used during
    model training.

    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------------
# Feature Names
# --------------------------------------------------------

feature_names = [

    "Time",

    "V1","V2","V3","V4","V5",
    "V6","V7","V8","V9","V10",
    "V11","V12","V13","V14","V15",
    "V16","V17","V18","V19","V20",
    "V21","V22","V23","V24","V25",
    "V26","V27","V28",

    "Amount"
]

# --------------------------------------------------------
# Feature Descriptions
# --------------------------------------------------------

feature_help = {

    "Time":"Seconds elapsed from first transaction",

    "Amount":"Transaction amount",

    "V1":"Principal Component 1",
    "V2":"Principal Component 2",
    "V3":"Principal Component 3",
    "V4":"Principal Component 4",
    "V5":"Principal Component 5",
    "V6":"Principal Component 6",
    "V7":"Principal Component 7",
    "V8":"Principal Component 8",
    "V9":"Principal Component 9",
    "V10":"Principal Component 10",
    "V11":"Principal Component 11",
    "V12":"Principal Component 12",
    "V13":"Principal Component 13",
    "V14":"Principal Component 14",
    "V15":"Principal Component 15",
    "V16":"Principal Component 16",
    "V17":"Principal Component 17",
    "V18":"Principal Component 18",
    "V19":"Principal Component 19",
    "V20":"Principal Component 20",
    "V21":"Principal Component 21",
    "V22":"Principal Component 22",
    "V23":"Principal Component 23",
    "V24":"Principal Component 24",
    "V25":"Principal Component 25",
    "V26":"Principal Component 26",
    "V27":"Principal Component 27",
    "V28":"Principal Component 28"

}

# ========================================================
# Input Form
# ========================================================

with st.form("manual_prediction_form"):

    st.subheader("Transaction Information")

    col1, col2, col3 = st.columns(3)

    transaction = {}

    for i, feature in enumerate(feature_names):

        if i % 3 == 0:

            with col1:

                transaction[feature] = st.number_input(

                    label=feature,

                    value=0.0,

                    step=0.01,

                    format="%.6f",

                    help=feature_help.get(feature,"")

                )

        elif i % 3 == 1:

            with col2:

                transaction[feature] = st.number_input(

                    label=feature,

                    value=0.0,

                    step=0.01,

                    format="%.6f",

                    help=feature_help.get(feature,"")

                )

        else:

            with col3:

                transaction[feature] = st.number_input(

                    label=feature,

                    value=0.0,

                    step=0.01,

                    format="%.6f",

                    help=feature_help.get(feature,"")

                )

    st.markdown("---")

    button_col1, button_col2, button_col3 = st.columns([2,2,1])

    with button_col1:

        predict_button = st.form_submit_button(

            "Predict Transaction",

            use_container_width=True

        )

    with button_col2:

        clear_button = st.form_submit_button(

            "Clear Inputs",

            use_container_width=True

        )

# ========================================================
# Clear Form
# ========================================================

if clear_button:

    st.rerun()

# ========================================================
# Convert Inputs to DataFrame
# ========================================================

input_df = pd.DataFrame([transaction])

# ========================================================
# Display Entered Transaction
# ========================================================

with st.expander("View Entered Transaction"):

    st.dataframe(
        input_df,
        use_container_width=True
    )

# ========================================================
# Validation
# ========================================================

validation_passed = True

try:

    input_df = validate_transaction(input_df)

except Exception as e:

    validation_passed = False

    st.error(str(e))

# ========================================================
# Basic Validation
# ========================================================

if len(input_df.columns) != 30:

    validation_passed = False

    st.error(
        "Exactly 30 input features are required."
    )

if input_df.isnull().sum().sum() > 0:

    validation_passed = False

    st.error(
        "Missing values detected."
    )

# ========================================================
# Feature Summary
# ========================================================

with st.expander("Input Summary"):

    st.write(f"Total Features : {len(input_df.columns)}")

    st.write(f"Missing Values : {input_df.isnull().sum().sum()}")

    st.write(f"Rows : {len(input_df)}")

    st.write("Data Type")

    st.dataframe(
        input_df.dtypes.astype(str)
    )

# ========================================================
# Placeholder for Prediction
# ========================================================

prediction = None
probability = None

if predict_button and validation_passed:

    st.success(
        "Input validation completed successfully."
    )

    # Prediction will be performed
    # in Part 3

    st.info(
        "Ready for prediction..."
    )


    # ========================================================
# Prediction Engine
# ========================================================

if predict_button and validation_passed:

    with st.spinner("Analyzing transaction..."):

        try:

            # ---------------------------------------------
            # Validate Input
            # ---------------------------------------------

            input_df = validate_transaction(input_df)

            # ---------------------------------------------
            # Predict
            # ---------------------------------------------

            prediction, probability = predict_transaction(
                input_df
            )

            probability = float(probability)

            prediction_text = prediction_label(prediction)

            risk = risk_level(probability)

            color = risk_color(probability)

            # ---------------------------------------------
            # Save History
            # ---------------------------------------------

            history = {

                "Time": datetime.now(),

                "Prediction": prediction_text,

                "Fraud Probability": probability,

                "Risk": risk

            }

            st.session_state.prediction_history.append(
                history
            )

        except Exception as e:

            st.error(e)

            st.stop()

    # ====================================================
    # Prediction Result
    # ====================================================

    st.markdown("---")

    st.markdown(
        '<p class="section-header">Prediction Result</p>',
        unsafe_allow_html=True
    )

    if prediction == 1:

        st.error(
            "🚨 Fraudulent Transaction Detected"
        )

    else:

        st.success(
            "✅ Genuine Transaction"
        )

    # ====================================================
    # Metrics
    # ====================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(

            label="Prediction",

            value=prediction_text

        )

    with col2:

        st.metric(

            label="Fraud Probability",

            value=f"{probability:.2%}"

        )

    with col3:

        st.metric(

            label="Risk Level",

            value=risk

        )

    # ====================================================
    # Probability Progress
    # ====================================================

    st.subheader("Fraud Probability")

    st.progress(probability)

    if probability >= 0.80:

        st.error(
            f"High Risk ({probability:.2%})"
        )

    elif probability >= 0.50:

        st.warning(
            f"Medium Risk ({probability:.2%})"
        )

    else:

        st.success(
            f"Low Risk ({probability:.2%})"
        )

    # ====================================================
    # Detailed Summary
    # ====================================================

    st.markdown("---")

    st.subheader("Prediction Summary")

    summary = pd.DataFrame({

        "Attribute":[

            "Prediction",

            "Fraud Probability",

            "Risk Level",

            "Model",

            "Explainability",

            "Prediction Time"

        ],

        "Value":[

            prediction_text,

            f"{probability:.4f}",

            risk,

            "Random Forest",

            "SHAP",

            datetime.now().strftime(
                "%d-%m-%Y %H:%M:%S"
            )

        ]

    })

    st.dataframe(

        summary,

        use_container_width=True,

        hide_index=True

    )

    # ====================================================
    # Confidence
    # ====================================================

    confidence = max(

        probability,

        1 - probability

    )

    st.subheader("Model Confidence")

    st.metric(

        label="Confidence",

        value=f"{confidence:.2%}"

    )

    # ====================================================
    # Prediction Interpretation
    # ====================================================

    st.subheader("Interpretation")

    if prediction == 1:

        st.write(
            """
            The transaction is predicted as
            **Fraudulent**.

            A higher fraud probability indicates
            that the transaction characteristics
            resemble known fraudulent behaviour.
            """
        )

    else:

        st.write(
            """
            The transaction is predicted as
            **Genuine**.

            The transaction characteristics
            closely match legitimate transactions
            observed during training.
            """
        )

    # ====================================================
    # Risk Indicator
    # ====================================================

    st.subheader("Risk Indicator")

    if color == "green":

        st.success("🟢 LOW RISK")

    elif color == "orange":

        st.warning("🟠 MEDIUM RISK")

    else:

        st.error("🔴 HIGH RISK")

    # ====================================================
    # Prediction History
    # ====================================================

    st.markdown("---")

    with st.expander("Prediction History"):

        history_df = pd.DataFrame(

            st.session_state.prediction_history

        )

        if len(history_df):

            st.dataframe(

                history_df,

                use_container_width=True

            )

        else:

            st.info("No predictions available.")



# ========================================================
# SHAP Explainability
# ========================================================

if predict_button and validation_passed:

    st.markdown("---")

    st.markdown(
        '<p class="section-header">Explainable AI (SHAP)</p>',
        unsafe_allow_html=True
    )

    with st.spinner("Generating SHAP explanation..."):

        try:

            explanation = generate_shap_values(input_df)

        except Exception as e:

            st.error(f"Unable to generate SHAP explanation.\n{e}")

            st.stop()

    # ====================================================
    # SHAP Waterfall Plot
    # ====================================================

    st.subheader("SHAP Waterfall Plot")

    fig = plt.figure(figsize=(10,7))

    shap.plots.waterfall(

        explanation[0],

        max_display=12,

        show=False

    )

    st.pyplot(fig)

    plt.close(fig)

    # ====================================================
    # Top Feature Contributions
    # ====================================================

    st.subheader("Top Feature Contributions")

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

    st.dataframe(

        contribution_df[
            ["Feature","SHAP Value"]
        ].head(10),

        use_container_width=True,

        hide_index=True

    )

    # ====================================================
    # SHAP Contribution Bar Chart
    # ====================================================

    st.subheader("Top 10 Important Features")

    fig2 = plt.figure(figsize=(9,5))

    top10 = contribution_df.head(10)

    plt.barh(

        top10["Feature"],

        top10["SHAP Value"]

    )

    plt.xlabel("SHAP Value")

    plt.ylabel("Feature")

    plt.title("Top Feature Contributions")

    plt.gca().invert_yaxis()

    st.pyplot(fig2)

    plt.close(fig2)

    # ====================================================
    # Positive / Negative Contributions
    # ====================================================

    positive = contribution_df[
        contribution_df["SHAP Value"] > 0
    ]

    negative = contribution_df[
        contribution_df["SHAP Value"] < 0
    ]

    col1,col2 = st.columns(2)

    with col1:

        st.subheader("Increase Fraud Probability")

        st.dataframe(

            positive.head(10),

            use_container_width=True,

            hide_index=True

        )

    with col2:

        st.subheader("Decrease Fraud Probability")

        st.dataframe(

            negative.head(10),

            use_container_width=True,

            hide_index=True

        )

    # ====================================================
    # Explanation Summary
    # ====================================================

    st.subheader("Prediction Explanation")

    top_feature = contribution_df.iloc[0]

    feature_name = top_feature["Feature"]

    shap_value = top_feature["SHAP Value"]

    if shap_value > 0:

        direction = "increased"

    else:

        direction = "decreased"

    st.info(

        f"""
        The feature **{feature_name}**

        has the largest influence on the prediction.

        Its SHAP contribution is

        **{shap_value:.4f}**

        which **{direction}** the fraud probability.
        """

    )

    # ====================================================
    # SHAP Summary Table
    # ====================================================

    st.subheader("Complete SHAP Table")

    st.dataframe(

        contribution_df,

        use_container_width=True,

        hide_index=True

    )

    # ====================================================
    # Save Report
    # ====================================================

    report = contribution_df.copy()

    report["Prediction"] = prediction_text

    report["Fraud Probability"] = probability

    report["Risk Level"] = risk

    csv = report.to_csv(index=False)

    st.download_button(

        "Download SHAP Report",

        csv,

        file_name="shap_explanation.csv",

        mime="text/csv"

    )

    # ====================================================
    # Force Plot
    # ====================================================

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

            """
            Matplotlib force plot is not supported
            for every SHAP configuration.

            The Waterfall Plot above already
            provides the complete local explanation.
            """

        )

    # ====================================================
    # Feature Importance Chart
    # ====================================================

    st.markdown("---")

    st.subheader("Feature Importance (Absolute SHAP)")

    fig3 = plt.figure(figsize=(10,6))

    plt.bar(

        contribution_df.head(15)["Feature"],

        contribution_df.head(15)["Absolute"]

    )

    plt.xticks(rotation=60)

    plt.ylabel("Absolute SHAP Value")

    plt.tight_layout()

    st.pyplot(fig3)

    plt.close(fig3)       

# ========================================================
# Download Prediction Summary
# ========================================================

if predict_button and validation_passed:

    st.markdown("---")
    st.subheader("Download Prediction Report")

    prediction_report = pd.DataFrame({

        "Prediction": [prediction_text],

        "Fraud Probability": [round(probability, 6)],

        "Risk Level": [risk],

        "Model": ["Random Forest"],

        "Explainability": ["SHAP"],

        "Prediction Time": [
            datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        ]

    })

    csv_report = prediction_report.to_csv(index=False)

    st.download_button(

        label="📥 Download Prediction Report",

        data=csv_report,

        file_name="prediction_report.csv",

        mime="text/csv",

        use_container_width=True

    )


# ========================================================
# Prediction History
# ========================================================

st.markdown("---")

st.subheader("Prediction History")

if len(st.session_state.prediction_history) > 0:

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(

        history_df,

        use_container_width=True,

        hide_index=True

    )

    history_csv = history_df.to_csv(index=False)

    st.download_button(

        label="📥 Download Prediction History",

        data=history_csv,

        file_name="prediction_history.csv",

        mime="text/csv",

        use_container_width=True

    )

else:

    st.info("Prediction history is empty.")


# ========================================================
# Clear Prediction History
# ========================================================

col1, col2 = st.columns(2)

with col1:

    if st.button(

        "🗑 Clear Prediction History",

        use_container_width=True

    ):

        st.session_state.prediction_history = []

        st.success(
            "Prediction history cleared."
        )

        st.rerun()


with col2:

    if st.button(

        "🔄 Reset Page",

        use_container_width=True

    ):

        st.rerun()


# ========================================================
# Information Panel
# ========================================================

st.markdown("---")

with st.expander(
    "About Manual Prediction",
    expanded=False
):

    st.markdown("""

### Model

Random Forest Classifier

### Explainable AI

SHAP TreeExplainer

### Input

30 Features

(Time, V1–V28, Amount)

### Output

• Genuine / Fraud

• Fraud Probability

• Risk Level

• SHAP Explanation

• Feature Contributions

• Downloadable Reports

""")


# ========================================================
# Tips
# ========================================================

st.markdown("---")

st.subheader("User Tips")

st.info("""

• Use values from the processed dataset.

• Keep feature order exactly the same as training.

• Fraud probability above **80%** indicates high risk.

• SHAP explains why the model predicted Fraud or Genuine.

""")


# ========================================================
# Backend Status
# ========================================================

st.markdown("---")

st.subheader("System Status")

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:

    st.success("✅ Model Loaded")

with status_col2:

    st.success("✅ SHAP Ready")

with status_col3:

    st.success("✅ Prediction Service Running")


# ========================================================
# Project Information
# ========================================================

st.markdown("---")

st.subheader("Project Information")

info_col1, info_col2 = st.columns(2)

with info_col1:

    st.write("**Project**")
    st.write("Credit Card Fraud Detection")

    st.write("**Algorithm**")
    st.write("Random Forest")

    st.write("**Explainability**")
    st.write("SHAP")

with info_col2:

    st.write("**Dataset**")
    st.write("European Credit Card Dataset")

    st.write("**Features**")
    st.write("30")

    st.write("**Framework**")
    st.write("Streamlit")


# ========================================================
# Footer
# ========================================================

st.markdown("---")

st.markdown(

"""
<div style="text-align:center;color:gray;padding:20px;">

Developed as a B.E. Final Year Project

<b>Credit Card Fraud Detection using Machine Learning
and Explainable AI (SHAP)</b>

<br><br>

Powered by

Python • Streamlit • Scikit-learn • Random Forest • SHAP

</div>

""",

unsafe_allow_html=True

)     