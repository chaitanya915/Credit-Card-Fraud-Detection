"""
=========================================================
Credit Card Fraud Detection
Explainable AI Dashboard

SHAP Version : 0.52.0
Final Model  : Random Forest

=========================================================
"""

# =====================================================
# Imports
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib

from pathlib import Path

# =====================================================
# Streamlit Page Configuration
# =====================================================

st.set_page_config(

    page_title="SHAP Explainability",

    page_icon="🧠",

    layout="wide",

    initial_sidebar_state="expanded"

)

# =====================================================
# Custom CSS
# =====================================================

st.markdown("""

<style>

.main-title{

font-size:40px;

font-weight:bold;

color:#1565C0;

}

.section-title{

font-size:28px;

font-weight:bold;

margin-top:20px;

}

.info-box{

padding:15px;

border-radius:10px;

background:#F5F7FA;

border:1px solid #DDDDDD;

}

</style>

""", unsafe_allow_html=True)

# =====================================================
# Sidebar
# =====================================================

with st.sidebar:

    st.title("🧠 Explainable AI")

    st.success("""

Random Forest

+

SHAP 0.52.0

""")

    st.markdown("---")

    st.subheader("Sections")

    st.write("• Introduction")

    st.write("• Global Explainability")

    st.write("• Local Explainability")

    st.write("• Waterfall Plot")

    st.write("• Force Plot")

    st.write("• Decision Plot")

    st.write("• Dependence Plot")

    st.write("• SHAP Report")

    st.markdown("---")

    st.info("""

This page explains

why the model predicts

Fraud or Genuine

transactions.

""")

# =====================================================
# Page Header
# =====================================================

st.markdown(

'<p class="main-title">🧠 Explainable AI Dashboard</p>',

unsafe_allow_html=True

)

st.write("""

Understand **why** the Random Forest model predicts a
transaction as **Fraud** or **Genuine** using
**SHAP (SHapley Additive exPlanations)**.

""")

st.divider()

# =====================================================
# Load Random Forest Model
# =====================================================

@st.cache_resource

def load_model():

    model_path = Path(

        "models/random_forest.pkl"

    )

    if not model_path.exists():

        st.error(

            "Random Forest model not found."

        )

        st.stop()

    return joblib.load(model_path)

model = load_model()

# =====================================================
# Load Test Dataset
# =====================================================

@st.cache_data

def load_dataset():

    dataset_path = Path(

        "data/processed/X_test.csv"

    )

    if not dataset_path.exists():

        st.error(

            "X_test.csv not found."

        )

        st.stop()

    return pd.read_csv(dataset_path)

X_test = load_dataset()

# =====================================================
# Build SHAP Explainer
# =====================================================

@st.cache_resource

def build_explainer(model):

    return shap.TreeExplainer(model)

explainer = build_explainer(model)

# =====================================================
# Calculate SHAP Values
# =====================================================

@st.cache_data

def calculate_shap():

    explanation = explainer(

        X_test

    )

    return explanation

shap_values = calculate_shap()

# =====================================================
# Dataset Information
# =====================================================

st.markdown(

'<p class="section-title">Dataset Information</p>',

unsafe_allow_html=True

)

col1,col2,col3,col4 = st.columns(4)

with col1:

    st.metric(

        "Transactions",

        len(X_test)

    )

with col2:

    st.metric(

        "Features",

        X_test.shape[1]

    )

with col3:

    st.metric(

        "Model",

        "Random Forest"

    )

with col4:

    st.metric(

        "SHAP Version",

        shap.__version__

    )

# =====================================================
# Introduction to Explainable AI
# =====================================================

st.markdown("---")

st.markdown(

'<p class="section-title">What is Explainable AI?</p>',

unsafe_allow_html=True

)

st.info("""

Explainable Artificial Intelligence (XAI)
helps humans understand how a machine learning
model makes predictions.

Instead of treating the Random Forest model
as a "black box", XAI provides explanations
showing which features influenced each prediction.

In banking and financial fraud detection,
explainability improves transparency,
trust, and regulatory compliance.

""")

# =====================================================
# What is SHAP?
# =====================================================

st.markdown("---")

st.markdown(

'<p class="section-title">What is SHAP?</p>',

unsafe_allow_html=True

)

st.success("""

SHAP stands for

SHapley Additive exPlanations.

It is based on game theory.

SHAP assigns a contribution value to every
feature used by the model.

Positive SHAP Value

⬆ increases fraud probability.

Negative SHAP Value

⬇ decreases fraud probability.

The larger the absolute SHAP value,
the greater the feature's influence.

""")

# =====================================================
# Why SHAP?
# =====================================================

st.markdown("---")

st.markdown(

'<p class="section-title">Advantages of SHAP</p>',

unsafe_allow_html=True

)

advantages = pd.DataFrame({

"Advantage":[

"Model Transparency",

"Local Explanation",

"Global Explanation",

"Feature Ranking",

"Regulatory Compliance",

"Fraud Investigation",

"Improved Trust"

],

"Description":[

"Explains model behaviour",

"Explains one prediction",

"Explains entire model",

"Ranks important features",

"Supports financial auditing",

"Helps analysts understand fraud",

"Increases confidence"

]

})

st.dataframe(

advantages,

use_container_width=True,

hide_index=True

)

# =====================================================
# Current Model Status
# =====================================================

st.markdown("---")

st.subheader("Current Model")

status1,status2,status3 = st.columns(3)

with status1:

    st.success("✅ Random Forest Loaded")

with status2:

    st.success("✅ SHAP Explainer Ready")

with status3:

    st.success("✅ Test Dataset Loaded")

# =====================================================
# Global SHAP Explainability
# =====================================================

st.markdown("---")

st.markdown(
    '<p class="section-title">Global Explainability</p>',
    unsafe_allow_html=True
)

st.write("""
Global Explainability helps understand how the
Random Forest model behaves across the entire test dataset.

The plots below identify the most influential features
used by the model to detect fraudulent transactions.
""")

# =====================================================
# SHAP Summary Plot
# =====================================================
import matplotlib.pyplot as plt
st.markdown("---")

st.subheader("SHAP Summary Plot")

with st.spinner("Generating SHAP Summary Plot..."):

    fig = plt.figure(figsize=(12,7))

    shap.summary_plot(

        shap_values.values,

        X_test,

        show=False

    )

st.pyplot(fig)

plt.close(fig)

st.info("""

Interpretation

• Each dot represents one transaction.

• Red = High feature value

• Blue = Low feature value

• Features at the top have the highest impact.

• Larger horizontal spread indicates greater influence.

""")

# =====================================================
# SHAP Bar Plot
# =====================================================

st.markdown("---")

st.subheader("Global Feature Importance")

fig = plt.figure(figsize=(10,6))

shap.plots.bar(

    shap_values,

    max_display=20,

    show=False

)

st.pyplot(fig)

plt.close(fig)

st.success("""

The SHAP Bar Plot ranks features according to
their average absolute SHAP value.

The higher the bar,
the more important the feature.

""")

# =====================================================
# Mean Absolute SHAP Values
# =====================================================

st.markdown("---")

st.subheader("Mean Absolute SHAP Values")

importance = pd.DataFrame({

    "Feature": X_test.columns,

    "Mean SHAP":

        np.abs(

            shap_values.values

        ).mean(axis=0)

})

importance = importance.sort_values(

    "Mean SHAP",

    ascending=False

)

st.dataframe(

    importance,

    use_container_width=True,

    hide_index=True

)

# =====================================================
# Top 20 Features
# =====================================================

st.markdown("---")

st.subheader("Top 20 Important Features")

top20 = importance.head(20)

st.dataframe(

    top20,

    use_container_width=True,

    hide_index=True

)

# =====================================================
# Horizontal Feature Importance Chart
# =====================================================

fig, ax = plt.subplots(figsize=(10,7))

ax.barh(

    top20["Feature"],

    top20["Mean SHAP"]

)

ax.invert_yaxis()

ax.set_xlabel("Mean Absolute SHAP Value")

ax.set_ylabel("Feature")

ax.set_title("Top 20 Global Features")

st.pyplot(fig)

plt.close(fig)

# =====================================================
# Top Features Metrics
# =====================================================

st.markdown("---")

st.subheader("Feature Statistics")

metric1, metric2, metric3 = st.columns(3)

with metric1:

    st.metric(

        "Total Features",

        len(X_test.columns)

    )

with metric2:

    st.metric(

        "Most Important Feature",

        importance.iloc[0]["Feature"]

    )

with metric3:

    st.metric(

        "Highest Mean SHAP",

        f"{importance.iloc[0]['Mean SHAP']:.5f}"

    )

# =====================================================
# Feature Ranking
# =====================================================

st.markdown("---")

st.subheader("Global Feature Ranking")

ranking = importance.copy()

ranking.insert(

    0,

    "Rank",

    range(1, len(ranking)+1)

)

st.dataframe(

    ranking,

    use_container_width=True,

    hide_index=True

)

# =====================================================
# Search Feature
# =====================================================

st.markdown("---")

st.subheader("Search Feature Importance")

selected_feature = st.selectbox(

    "Select Feature",

    ranking["Feature"]

)

selected = ranking[

    ranking["Feature"] == selected_feature

]

st.dataframe(

    selected,

    use_container_width=True,

    hide_index=True

)

# =====================================================
# Download Feature Importance
# =====================================================

st.markdown("---")

st.subheader("Download Feature Ranking")

csv = ranking.to_csv(index=False)

st.download_button(

    label="📥 Download SHAP Feature Importance",

    data=csv,

    file_name="shap_feature_importance.csv",

    mime="text/csv",

    use_container_width=True

)

# =====================================================
# Global Interpretation
# =====================================================

st.markdown("---")

st.subheader("Interpretation")

top5 = ranking.head(5)

st.write("""

The following features have the greatest influence
on the Random Forest model.

""")

for _, row in top5.iterrows():

    st.success(

        f"{row['Rank']}. "

        f"{row['Feature']} "

        f"(Mean SHAP = {row['Mean SHAP']:.5f})"

    )

st.info("""

Global Explainability answers:

✔ Which features matter the most?

✔ Which variables influence predictions?

✔ How important is each feature overall?

Unlike local explanations,
these insights are based on the entire dataset.

""")

# =====================================================
# Local Explainability
# =====================================================

st.markdown("---")

st.markdown(
    '<p class="section-title">Local Explainability</p>',
    unsafe_allow_html=True
)

st.write("""
Select an individual transaction to understand
why the Random Forest model predicted it as
Fraud or Genuine.
""")

# =====================================================
# Transaction Selection
# =====================================================

selected_index = st.selectbox(

    "Select Transaction",

    options=range(len(X_test)),

    format_func=lambda x: f"Transaction #{x}"

)

transaction = X_test.iloc[[selected_index]]

# =====================================================
# Prediction
# =====================================================

prediction = model.predict(transaction)[0]

probability = model.predict_proba(transaction)[0][1]

prediction_label = (

    "Fraud"

    if prediction == 1

    else "Genuine"

)

# =====================================================
# Prediction Summary
# =====================================================

st.markdown("---")

st.subheader("Prediction Summary")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(

        "Prediction",

        prediction_label

    )

with col2:

    st.metric(

        "Fraud Probability",

        f"{probability:.2%}"

    )

with col3:

    confidence = max(

        probability,

        1 - probability

    )

    st.metric(

        "Prediction Confidence",

        f"{confidence:.2%}"

    )

# =====================================================
# Risk Level
# =====================================================

if probability >= 0.80:

    risk = "High"

elif probability >= 0.50:

    risk = "Medium"

else:

    risk = "Low"

st.success(

    f"Risk Level: **{risk}**"

)

# =====================================================
# Selected Transaction
# =====================================================

st.markdown("---")

st.subheader("Selected Transaction")

st.dataframe(

    transaction,

    use_container_width=True,

    hide_index=True

)

# =====================================================
# Feature Values
# =====================================================

st.markdown("---")

st.subheader("Feature Values")

feature_df = pd.DataFrame({

    "Feature": transaction.columns,

    "Value": transaction.iloc[0].values

})

st.dataframe(

    feature_df,

    use_container_width=True,

    hide_index=True

)

# =====================================================
# Calculate SHAP Explanation
# =====================================================

local_explanation = explainer(transaction)

# Save for Part 4
st.session_state["selected_transaction"] = transaction
st.session_state["local_explanation"] = local_explanation
st.session_state["prediction"] = prediction
st.session_state["prediction_probability"] = probability

# =====================================================
# Feature Search
# =====================================================

st.markdown("---")

st.subheader("Inspect a Feature")

selected_feature = st.selectbox(

    "Choose Feature",

    transaction.columns

)

selected_value = transaction[selected_feature].iloc[0]

st.metric(

    selected_feature,

    f"{selected_value:.6f}"

)

# =====================================================
# Top Feature Values
# =====================================================

st.markdown("---")

st.subheader("Top Absolute Feature Values")

top_features = (

    feature_df.assign(

        Absolute=feature_df["Value"].abs()

    )

    .sort_values(

        "Absolute",

        ascending=False

    )

    .head(10)

)

st.dataframe(

    top_features,

    use_container_width=True,

    hide_index=True

)

# =====================================================
# Transaction Summary
# =====================================================

st.markdown("---")

st.subheader("Transaction Summary")

summary = pd.DataFrame({

    "Attribute":[

        "Transaction Index",

        "Prediction",

        "Fraud Probability",

        "Confidence",

        "Risk Level",

        "Number of Features"

    ],

    "Value":[

        selected_index,

        prediction_label,

        f"{probability:.4f}",

        f"{confidence:.4f}",

        risk,

        transaction.shape[1]

    ]

})

st.dataframe(

    summary,

    use_container_width=True,

    hide_index=True

)

# =====================================================
# Local Explanation Overview
# =====================================================

st.markdown("---")

st.subheader("Local Explanation")

st.info("""

The selected transaction has been analysed.

The SHAP explanation for this transaction has
been generated successfully.

The following visualisations will be available
in the next section:

• SHAP Waterfall Plot

• SHAP Force Plot

• SHAP Decision Plot

These plots explain why the Random Forest model
predicted this transaction as Fraud or Genuine.

""")

# =====================================================
# Local SHAP Visualizations
# =====================================================

st.markdown("---")

st.markdown(
    '<p class="section-title">Local SHAP Visualizations</p>',
    unsafe_allow_html=True
)

# =====================================================
# Get Selected Explanation
# =====================================================

explanation = st.session_state["local_explanation"]

transaction = st.session_state["selected_transaction"]

prediction = st.session_state["prediction"]

probability = st.session_state["prediction_probability"]

# For binary classification (Fraud = Class 1)
local_exp = explanation[0, :, 1]

# =====================================================
# Waterfall Plot
# =====================================================

st.subheader("SHAP Waterfall Plot")

st.write("""
The Waterfall Plot shows how each feature
pushes the prediction toward Fraud or Genuine.
""")

try:

    fig = plt.figure(figsize=(12,8))

    shap.plots.waterfall(

        local_exp,

        max_display=20,

        show=False

    )

    st.pyplot(fig)

    plt.close(fig)

except Exception as e:

    st.error(f"Unable to generate Waterfall Plot: {e}")

# =====================================================
# Force Plot
# =====================================================

st.markdown("---")

st.subheader("SHAP Force Plot")

st.write("""
The Force Plot illustrates how feature
contributions combine to produce the
final prediction.
""")

try:

    force = shap.plots.force(

        local_exp.base_values,

        local_exp.values,

        local_exp.data,

        feature_names=transaction.columns.tolist(),

        matplotlib=True,

        show=False

    )

    st.pyplot(plt.gcf())

    plt.close()

except Exception as e:

    st.warning(f"Force Plot could not be generated: {e}")

# =====================================================
# Decision Plot
# =====================================================

st.markdown("---")

st.subheader("SHAP Decision Plot")

st.write("""
The Decision Plot visualizes how the
prediction evolves as features are added.
""")

try:

    fig = plt.figure(figsize=(12,8))

    shap.decision_plot(

        local_exp.base_values,

        local_exp.values,

        feature_names=transaction.columns.tolist(),

        show=False

    )

    st.pyplot(fig)

    plt.close(fig)

except Exception as e:

    st.warning(f"Decision Plot could not be generated: {e}")

# =====================================================
# Feature Contributions
# =====================================================

st.markdown("---")

st.subheader("Feature Contributions")

contribution_df = pd.DataFrame({

    "Feature": transaction.columns,

    "SHAP Value": local_exp.values,

    "Feature Value": transaction.iloc[0].values

})

contribution_df["Impact"] = np.where(

    contribution_df["SHAP Value"] > 0,

    "Increase Fraud Probability",

    "Decrease Fraud Probability"

)

contribution_df = contribution_df.sort_values(

    by="SHAP Value",

    key=np.abs,

    ascending=False

)

st.dataframe(

    contribution_df,

    use_container_width=True,

    hide_index=True

)

# =====================================================
# Positive vs Negative Contributions
# =====================================================

st.markdown("---")

st.subheader("Positive vs Negative Contributions")

positive = contribution_df[
    contribution_df["SHAP Value"] > 0
]

negative = contribution_df[
    contribution_df["SHAP Value"] < 0
]

col1, col2 = st.columns(2)

with col1:

    st.success("Features Increasing Fraud Risk")

    st.dataframe(

        positive.head(10),

        use_container_width=True,

        hide_index=True

    )

with col2:

    st.info("Features Decreasing Fraud Risk")

    st.dataframe(

        negative.head(10),

        use_container_width=True,

        hide_index=True

    )

# =====================================================
# Contribution Statistics
# =====================================================

st.markdown("---")

st.subheader("Contribution Statistics")

c1, c2, c3 = st.columns(3)

with c1:

    st.metric(

        "Positive Features",

        len(positive)

    )

with c2:

    st.metric(

        "Negative Features",

        len(negative)

    )

with c3:

    st.metric(

        "Largest |SHAP|",

        f"{np.abs(local_exp.values).max():.5f}"

    )

# =====================================================
# Top 5 Most Influential Features
# =====================================================

st.markdown("---")

st.subheader("Top 5 Most Influential Features")

top5 = contribution_df.head(5)

for _, row in top5.iterrows():

    icon = "🔺" if row["SHAP Value"] > 0 else "🔻"

    st.write(

        f"{icon} **{row['Feature']}**"

        f" | SHAP = {row['SHAP Value']:.5f}"

        f" | Value = {row['Feature Value']:.5f}"

    )

# =====================================================
# Prediction Explanation
# =====================================================

st.markdown("---")

st.subheader("Prediction Explanation")

if prediction == 1:

    st.error(f"""

The model predicts **Fraud**
with probability **{probability:.2%}**.

Features with positive SHAP values
contributed to increasing fraud probability,
while negative SHAP values reduced it.

""")

else:

    st.success(f"""

The model predicts **Genuine**
with probability **{1-probability:.2%}**.

Features with negative SHAP values
contributed more strongly,
keeping the transaction classified as genuine.

""")

# =====================================================
# Interactive Feature Analysis
# =====================================================

st.markdown("---")

st.markdown(
    '<p class="section-title">Interactive Feature Analysis</p>',
    unsafe_allow_html=True
)

st.write("""
Explore how each feature influences the Random Forest model.
You can inspect SHAP values, compare local and global importance,
and visualize feature behavior.
""")

# =====================================================
# Select Feature
# =====================================================

feature_name = st.selectbox(
    "Select Feature for Analysis",
    X_test.columns,
    index=0
)

feature_index = list(X_test.columns).index(feature_name)

# =====================================================
# SHAP Dependence Plot
# =====================================================

st.markdown("---")

st.subheader("SHAP Dependence Plot")

st.write("""
The dependence plot shows how the selected feature
affects the model prediction across the dataset.
""")

try:

    fig = plt.figure(figsize=(10,6))

    shap.dependence_plot(
        feature_name,
        shap_values.values[:, :, 1],
        X_test,
        show=False
    )

    st.pyplot(fig)

    plt.close(fig)

except Exception as e:

    st.warning(f"Unable to generate dependence plot: {e}")

# =====================================================
# Global vs Local Importance
# =====================================================

st.markdown("---")

st.subheader("Global vs Local SHAP Importance")

global_importance = np.abs(
    shap_values.values[:, :, 1]
).mean(axis=0)

local_importance = np.abs(
    local_exp.values
)

comparison_df = pd.DataFrame({

    "Feature": X_test.columns,

    "Global Importance": global_importance,

    "Local Importance": local_importance

})

comparison_df = comparison_df.sort_values(
    "Global Importance",
    ascending=False
)

st.dataframe(
    comparison_df,
    use_container_width=True,
    hide_index=True
)

# =====================================================
# Feature Statistics
# =====================================================

st.markdown("---")

st.subheader("Selected Feature Statistics")

feature_values = X_test[feature_name]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Minimum", f"{feature_values.min():.4f}")

with col2:
    st.metric("Maximum", f"{feature_values.max():.4f}")

with col3:
    st.metric("Mean", f"{feature_values.mean():.4f}")

with col4:
    st.metric("Std. Dev.", f"{feature_values.std():.4f}")

# =====================================================
# SHAP Value Distribution
# =====================================================

st.markdown("---")

st.subheader("SHAP Value Distribution")

fig, ax = plt.subplots(figsize=(10,5))

ax.hist(
    shap_values.values[:, feature_index, 1],
    bins=30
)

ax.set_xlabel("SHAP Value")
ax.set_ylabel("Frequency")
ax.set_title(f"Distribution of SHAP Values for {feature_name}")

st.pyplot(fig)

plt.close(fig)

# =====================================================
# Local Feature Contribution
# =====================================================

st.markdown("---")

st.subheader("Selected Transaction Contribution")

local_value = transaction.iloc[0][feature_name]
local_shap = local_exp.values[feature_index]

c1, c2 = st.columns(2)

with c1:

    st.metric(
        "Feature Value",
        f"{local_value:.5f}"
    )

with c2:

    st.metric(
        "Local SHAP Value",
        f"{local_shap:.5f}"
    )

if local_shap > 0:

    st.error(f"""
The selected feature **increases fraud probability**.

SHAP Contribution: **+{local_shap:.5f}**
""")

elif local_shap < 0:

    st.success(f"""
The selected feature **decreases fraud probability**.

SHAP Contribution: **{local_shap:.5f}**
""")

else:

    st.info("This feature has negligible impact.")

# =====================================================
# Top Positive Features
# =====================================================

st.markdown("---")

st.subheader("Top Positive SHAP Features")

positive_features = contribution_df[
    contribution_df["SHAP Value"] > 0
].head(10)

st.dataframe(
    positive_features,
    use_container_width=True,
    hide_index=True
)

# =====================================================
# Top Negative Features
# =====================================================

st.subheader("Top Negative SHAP Features")

negative_features = contribution_df[
    contribution_df["SHAP Value"] < 0
].head(10)

st.dataframe(
    negative_features,
    use_container_width=True,
    hide_index=True
)

# =====================================================
# Global Top Features
# =====================================================

st.markdown("---")

st.subheader("Top Global Features")

global_df = pd.DataFrame({

    "Feature": X_test.columns,

    "Mean |SHAP|": global_importance

}).sort_values(
    "Mean |SHAP|",
    ascending=False
)

st.dataframe(
    global_df.head(15),
    use_container_width=True,
    hide_index=True
)

# =====================================================
# Feature Importance Comparison Chart
# =====================================================

st.markdown("---")

st.subheader("Global vs Local Comparison")

compare = comparison_df.head(10)

fig, ax = plt.subplots(figsize=(12,6))

x = np.arange(len(compare))

width = 0.35

ax.bar(
    x - width/2,
    compare["Global Importance"],
    width,
    label="Global"
)

ax.bar(
    x + width/2,
    compare["Local Importance"],
    width,
    label="Local"
)

ax.set_xticks(x)

ax.set_xticklabels(
    compare["Feature"],
    rotation=45,
    ha="right"
)

ax.set_ylabel("Importance")

ax.legend()

st.pyplot(fig)

plt.close(fig)

# =====================================================
# Explanation Summary
# =====================================================

st.markdown("---")

st.subheader("Explanation Summary")

st.info(f"""

Selected Feature : **{feature_name}**

Feature Value : **{local_value:.5f}**

Local SHAP : **{local_shap:.5f}**

Global Mean SHAP :
**{global_importance[feature_index]:.5f}**

This comparison shows how important the feature is
for the selected transaction versus the overall model.

""")


# =====================================================
# Advanced SHAP Analytics
# =====================================================

st.markdown("---")

st.markdown(
    '<p class="section-title">Advanced SHAP Analytics</p>',
    unsafe_allow_html=True
)

st.write("""
This section provides detailed analytics about how
SHAP values influence the model's prediction.
""")

# =====================================================
# SHAP Statistics
# =====================================================

st.subheader("SHAP Statistics")

abs_shap = np.abs(local_exp.values)

positive_count = np.sum(local_exp.values > 0)
negative_count = np.sum(local_exp.values < 0)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Positive Features", positive_count)

with col2:
    st.metric("Negative Features", negative_count)

with col3:
    st.metric(
        "Largest SHAP",
        f"{abs_shap.max():.5f}"
    )

with col4:
    st.metric(
        "Mean |SHAP|",
        f"{abs_shap.mean():.5f}"
    )

# =====================================================
# SHAP Contribution Pie Chart
# =====================================================

st.markdown("---")

st.subheader("Contribution Distribution")

fig, ax = plt.subplots(figsize=(6,6))

sizes = [
    positive_count,
    negative_count
]

labels = [
    "Increase Fraud",
    "Decrease Fraud"
]

ax.pie(
    sizes,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90
)

ax.axis("equal")

st.pyplot(fig)

plt.close(fig)

# =====================================================
# SHAP Contribution Histogram
# =====================================================

st.markdown("---")

st.subheader("Distribution of Local SHAP Values")

fig, ax = plt.subplots(figsize=(10,5))

ax.hist(
    local_exp.values,
    bins=20
)

ax.set_xlabel("SHAP Value")

ax.set_ylabel("Frequency")

ax.set_title("Local SHAP Distribution")

st.pyplot(fig)

plt.close(fig)

# =====================================================
# Feature Correlation
# =====================================================

st.markdown("---")

st.subheader("Feature Correlation Matrix")

corr = X_test.corr()

fig, ax = plt.subplots(figsize=(12,10))

image = ax.imshow(
    corr,
    aspect="auto"
)

plt.colorbar(image)

ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(
    corr.columns,
    rotation=90,
    fontsize=8
)

ax.set_yticks(range(len(corr.columns)))
ax.set_yticklabels(
    corr.columns,
    fontsize=8
)

st.pyplot(fig)

plt.close(fig)

# =====================================================
# Strongest Positive Features
# =====================================================

st.markdown("---")

st.subheader("Top Features Increasing Fraud")

positive_df = contribution_df[
    contribution_df["SHAP Value"] > 0
].sort_values(
    "SHAP Value",
    ascending=False
)

st.dataframe(
    positive_df.head(10),
    use_container_width=True,
    hide_index=True
)

# =====================================================
# Strongest Negative Features
# =====================================================

st.subheader("Top Features Decreasing Fraud")

negative_df = contribution_df[
    contribution_df["SHAP Value"] < 0
].sort_values(
    "SHAP Value"
)

st.dataframe(
    negative_df.head(10),
    use_container_width=True,
    hide_index=True
)

# =====================================================
# Automatic Explanation
# =====================================================

st.markdown("---")

st.subheader("Automatic Explanation")

top_positive = positive_df.head(3)["Feature"].tolist()

top_negative = negative_df.head(3)["Feature"].tolist()

st.success(f"""

### Model Interpretation

Prediction: **{prediction_label}**

Fraud Probability: **{probability:.2%}**

The model identified the transaction as
**{prediction_label}** because the following
features had the strongest influence.

Features increasing fraud probability:

{', '.join(top_positive)}

Features decreasing fraud probability:

{', '.join(top_negative)}

""")

# =====================================================
# Confidence Meter
# =====================================================

st.markdown("---")

st.subheader("Prediction Confidence")

confidence = max(
    probability,
    1 - probability
)

st.progress(float(confidence))

st.write(
    f"Confidence: **{confidence:.2%}**"
)

# =====================================================
# Risk Assessment
# =====================================================

st.markdown("---")

st.subheader("Risk Assessment")

if probability > 0.90:

    st.error("""
Very High Risk

Immediate manual investigation recommended.
""")

elif probability > 0.70:

    st.warning("""
High Risk

Transaction should be reviewed.
""")

elif probability > 0.40:

    st.info("""
Medium Risk

Additional verification recommended.
""")

else:

    st.success("""
Low Risk

Transaction appears genuine.
""")

# =====================================================
# Download Explanation
# =====================================================

st.markdown("---")

st.subheader("Download SHAP Explanation")

download_df = contribution_df.copy()

download_df.to_csv(
    "reports/local_shap_explanation.csv",
    index=False
)

csv = download_df.to_csv(index=False)

st.download_button(

    "📥 Download Local SHAP Report",

    csv,

    file_name="local_shap_explanation.csv",

    mime="text/csv",

    use_container_width=True

)

# =====================================================
# Summary
# =====================================================

st.markdown("---")

st.subheader("Summary")

st.info("""

This Explainable AI dashboard provides:

✅ Global Model Explainability

✅ Local Prediction Explainability

✅ SHAP Waterfall Plot

✅ SHAP Force Plot

✅ SHAP Decision Plot

✅ SHAP Dependence Plot

✅ Feature Importance

✅ Fraud Risk Analysis

✅ Automatic Prediction Explanation

✅ Downloadable SHAP Report

""")

# =====================================================
# Export Reports
# =====================================================

st.markdown("---")

st.markdown(
    '<p class="section-title">📥 Export Reports</p>',
    unsafe_allow_html=True
)

# Export Feature Importance
feature_csv = global_df.to_csv(index=False)

st.download_button(
    label="📊 Download Global Feature Importance",
    data=feature_csv,
    file_name="global_feature_importance.csv",
    mime="text/csv",
    use_container_width=True
)

# Export Local SHAP Explanation
local_csv = contribution_df.to_csv(index=False)

st.download_button(
    label="🧠 Download Local SHAP Explanation",
    data=local_csv,
    file_name="local_shap_explanation.csv",
    mime="text/csv",
    use_container_width=True
)

# Export Transaction
transaction_csv = transaction.to_csv(index=False)

st.download_button(
    label="💳 Download Selected Transaction",
    data=transaction_csv,
    file_name="selected_transaction.csv",
    mime="text/csv",
    use_container_width=True
)

# =====================================================
# Project Information
# =====================================================

st.markdown("---")

st.markdown(
    '<p class="section-title">📋 Project Information</p>',
    unsafe_allow_html=True
)

project_info = pd.DataFrame({

    "Item":[
        "Project",
        "Domain",
        "Machine Learning Model",
        "Explainable AI",
        "Programming Language",
        "Framework",
        "Dataset",
        "Dataset Features",
        "Fraud Detection"
    ],

    "Value":[
        "Credit Card Fraud Detection",
        "Data Science",
        "Random Forest",
        "SHAP 0.52.0",
        "Python",
        "Streamlit",
        "European Credit Card Dataset",
        X_test.shape[1],
        "Binary Classification"
    ]

})

st.dataframe(
    project_info,
    use_container_width=True,
    hide_index=True
)

# =====================================================
# Technology Stack
# =====================================================

st.markdown("---")

st.markdown(
    '<p class="section-title">🛠 Technology Stack</p>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    st.info("""

### Programming

• Python

• Pandas

• NumPy

• Scikit-Learn

""")

with col2:

    st.info("""

### Machine Learning

• Random Forest

• SHAP

• Joblib

""")

with col3:

    st.info("""

### Visualization

• Streamlit

• Matplotlib

• SHAP Plots

""")

# =====================================================
# Project Workflow
# =====================================================

st.markdown("---")

st.markdown(
    '<p class="section-title">📈 Project Workflow</p>',
    unsafe_allow_html=True
)

st.code("""

Raw Dataset
      │
      ▼
Data Preprocessing
      │
      ▼
Feature Engineering
      │
      ▼
Train/Test Split
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Random Forest Selection
      │
      ▼
SHAP Explainability
      │
      ▼
Prediction Dashboard

""")

# =====================================================
# User Guide
# =====================================================

st.markdown("---")

with st.expander("📘 User Guide", expanded=False):

    st.markdown("""

### Step 1

Select any transaction.

---

### Step 2

View fraud probability.

---

### Step 3

Observe the Waterfall Plot.

---

### Step 4

Observe the Force Plot.

---

### Step 5

Study the Decision Plot.

---

### Step 6

Compare Global vs Local SHAP.

---

### Step 7

Download reports if required.

""")

# =====================================================
# Frequently Asked Questions
# =====================================================

st.markdown("---")

with st.expander("❓ Frequently Asked Questions"):

    st.markdown("""

### Why Random Forest?

It achieved the highest overall performance among all evaluated models.

---

### Why SHAP?

SHAP provides transparent explanations for every prediction.

---

### Why Explainable AI?

Financial institutions require understandable AI decisions to improve trust,
support auditing, and assist fraud analysts.

""")

# =====================================================
# System Status
# =====================================================

st.markdown("---")

st.markdown(
    '<p class="section-title">⚙ System Status</p>',
    unsafe_allow_html=True
)

status1, status2, status3, status4 = st.columns(4)

with status1:
    st.success("✅ Model Loaded")

with status2:
    st.success("✅ SHAP Ready")

with status3:
    st.success("✅ Dataset Loaded")

with status4:
    st.success("✅ Dashboard Running")

# =====================================================
# Final Conclusion
# =====================================================

st.markdown("---")

st.markdown(
    '<p class="section-title">🎯 Final Conclusion</p>',
    unsafe_allow_html=True
)

st.success("""

The Credit Card Fraud Detection system combines
Machine Learning with Explainable Artificial Intelligence
to detect fraudulent transactions accurately while
providing transparent explanations.

Key achievements:

✅ Random Forest selected as the best-performing model.

✅ High Accuracy, Precision, Recall, F1 Score, and ROC-AUC.

✅ SHAP-based global and local explanations.

✅ Interactive Streamlit dashboard.

✅ Exportable analytical reports.

This system demonstrates how Explainable AI can improve
trust, transparency, and decision-making in financial
fraud detection.

""")

# =====================================================
# Footer
# =====================================================

st.markdown("---")

st.markdown(
"""
<div style="
padding:20px;
border-radius:10px;
background-color:#F5F7FA;
text-align:center;
">

<h2>Credit Card Fraud Detection Using Machine Learning</h2>

<h4>Explainable AI Dashboard</h4>

<p>

Random Forest + SHAP 0.52.0

<br><br>

Developed using

Python • Streamlit • Scikit-Learn • SHAP • Matplotlib

<br><br>

B.E. Computer Engineering Final Year Project

</p>

</div>
""",
unsafe_allow_html=True
)

