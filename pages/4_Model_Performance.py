
"""
=========================================================
Credit Card Fraud Detection
Model Performance Dashboard

Final Model:
Random Forest

=========================================================
"""

import streamlit as st
import pandas as pd
from pathlib import Path

# --------------------------------------------------------
# Page Configuration
# --------------------------------------------------------

st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------
# Custom CSS
# --------------------------------------------------------

st.markdown("""
<style>

.main-title{
    font-size:40px;
    font-weight:bold;
    color:#0E6EFD;
}

.section-title{
    font-size:26px;
    font-weight:bold;
    color:#222;
    margin-top:15px;
}

.metric-card{
    padding:15px;
    border-radius:10px;
    border:1px solid #DDDDDD;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------
# Sidebar
# --------------------------------------------------------

with st.sidebar:

    st.title("📊 Model Performance")

    st.success("""
Final Model

✅ Random Forest
""")

    st.markdown("---")

    st.write("Models Compared")

    st.write("• Logistic Regression")
    st.write("• Decision Tree")
    st.write("• Random Forest")
    st.write("• XGBoost")
    st.write("• LightGBM")
    st.write("• CatBoost")

    st.markdown("---")

    st.info(
        "This page compares all trained models "
        "and explains why Random Forest was selected."
    )

# --------------------------------------------------------
# Page Header
# --------------------------------------------------------

st.markdown(
    '<p class="main-title">📊 Model Performance Dashboard</p>',
    unsafe_allow_html=True
)

st.write("""
Compare the performance of all trained machine learning
models using multiple evaluation metrics.
""")

st.divider()

# --------------------------------------------------------
# Load Metrics
# --------------------------------------------------------

@st.cache_data
def load_metrics():

    metrics_path = Path("reports/model_metrics.csv")

    if not metrics_path.exists():

        st.error(
            "reports/model_metrics.csv not found."
        )

        st.stop()

    return pd.read_csv(metrics_path)

metrics = load_metrics()

# --------------------------------------------------------
# Metrics Table
# --------------------------------------------------------

st.markdown(
    '<p class="section-title">Model Comparison</p>',
    unsafe_allow_html=True
)

st.dataframe(

    metrics,

    use_container_width=True,

    hide_index=True

)

# --------------------------------------------------------
# Best Model
# --------------------------------------------------------

best_model = metrics.sort_values(

    by="F1 Score",

    ascending=False

).iloc[0]

st.success(

    f"""
Best Model Selected

Model: {best_model['Model']}

F1 Score: {best_model['F1 Score']:.4f}

ROC AUC: {best_model['ROC AUC']:.4f}
"""

)

# --------------------------------------------------------
# KPI Cards
# --------------------------------------------------------

st.markdown(
    '<p class="section-title">Key Performance Indicators</p>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(

        "Best Accuracy",

        f"{metrics['Accuracy'].max():.4f}"

    )

with col2:

    st.metric(

        "Best Precision",

        f"{metrics['Precision'].max():.4f}"

    )

with col3:

    st.metric(

        "Best Recall",

        f"{metrics['Recall'].max():.4f}"

    )

with col4:

    st.metric(

        "Best F1 Score",

        f"{metrics['F1 Score'].max():.4f}"

    )

# --------------------------------------------------------
# ROC KPI
# --------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    st.metric(

        "Best ROC AUC",

        f"{metrics['ROC AUC'].max():.4f}"

    )

with col2:

    st.metric(

        "Models Evaluated",

        len(metrics)

    )

# --------------------------------------------------------
# Ranking Table
# --------------------------------------------------------

st.markdown(
    '<p class="section-title">Model Ranking</p>',
    unsafe_allow_html=True
)

ranking = metrics.sort_values(

    by="F1 Score",

    ascending=False

).reset_index(drop=True)

ranking.index = ranking.index + 1

ranking.insert(

    0,

    "Rank",

    ranking.index

)

st.dataframe(

    ranking,

    use_container_width=True,

    hide_index=True

)

# --------------------------------------------------------
# Highlight Final Model
# --------------------------------------------------------

st.markdown("---")

st.info("""
### Why Random Forest?

Random Forest was selected as the final model because it achieved:

- High Accuracy
- Excellent Precision
- Strong Recall
- Highest F1 Score
- High ROC AUC
- Good generalization on imbalanced fraud detection data

These characteristics make it well suited for credit card fraud detection, where balancing false positives and false negatives is important.
""")


# ==========================================================
# Imports Required for Charts
# ==========================================================

import matplotlib.pyplot as plt

# ==========================================================
# Performance Charts
# ==========================================================

st.markdown("---")

st.markdown(
    '<p class="section-title">Performance Comparison Charts</p>',
    unsafe_allow_html=True
)

# ==========================================================
# Accuracy Comparison
# ==========================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("Accuracy Comparison")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(
        metrics["Model"],
        metrics["Accuracy"]
    )

    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.95,1.01)

    plt.xticks(rotation=25)

    st.pyplot(fig)

    plt.close(fig)

# ==========================================================
# Precision Comparison
# ==========================================================

with col2:

    st.subheader("Precision Comparison")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(

        metrics["Model"],

        metrics["Precision"]

    )

    ax.set_ylabel("Precision")

    plt.xticks(rotation=25)

    st.pyplot(fig)

    plt.close(fig)

# ==========================================================
# Recall Comparison
# ==========================================================

st.markdown("---")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Recall Comparison")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(

        metrics["Model"],

        metrics["Recall"]

    )

    ax.set_ylabel("Recall")

    plt.xticks(rotation=25)

    st.pyplot(fig)

    plt.close(fig)

# ==========================================================
# F1 Score Comparison
# ==========================================================

with col2:

    st.subheader("F1 Score Comparison")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(

        metrics["Model"],

        metrics["F1 Score"]

    )

    ax.set_ylabel("F1 Score")

    plt.xticks(rotation=25)

    st.pyplot(fig)

    plt.close(fig)

# ==========================================================
# ROC AUC Comparison
# ==========================================================

st.markdown("---")

st.subheader("ROC AUC Comparison")

fig, ax = plt.subplots(figsize=(10,5))

ax.bar(

    metrics["Model"],

    metrics["ROC AUC"]

)

ax.set_ylabel("ROC AUC")

plt.xticks(rotation=25)

st.pyplot(fig)

plt.close(fig)

# ==========================================================
# Overall Metric Comparison
# ==========================================================

st.markdown("---")

st.subheader("Overall Metric Comparison")

comparison = metrics.set_index("Model")[

    [

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score",

        "ROC AUC"

    ]

]

st.dataframe(

    comparison,

    use_container_width=True

)

# ==========================================================
# Best Model by Metric
# ==========================================================

st.markdown("---")

st.subheader("Best Performing Model for Each Metric")

summary = pd.DataFrame({

    "Metric":[

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score",

        "ROC AUC"

    ],

    "Best Model":[

        metrics.loc[
            metrics["Accuracy"].idxmax(),
            "Model"
        ],

        metrics.loc[
            metrics["Precision"].idxmax(),
            "Model"
        ],

        metrics.loc[
            metrics["Recall"].idxmax(),
            "Model"
        ],

        metrics.loc[
            metrics["F1 Score"].idxmax(),
            "Model"
        ],

        metrics.loc[
            metrics["ROC AUC"].idxmax(),
            "Model"
        ]

    ],

    "Score":[

        metrics["Accuracy"].max(),

        metrics["Precision"].max(),

        metrics["Recall"].max(),

        metrics["F1 Score"].max(),

        metrics["ROC AUC"].max()

    ]

})

st.dataframe(

    summary,

    use_container_width=True,

    hide_index=True

)

# ==========================================================
# Radar Chart (Normalized Comparison)
# ==========================================================

st.markdown("---")

st.subheader("Normalized Performance Comparison")

normalized = comparison.copy()

for col in normalized.columns:

    normalized[col] = (

        normalized[col] -

        normalized[col].min()

    ) / (

        normalized[col].max() -

        normalized[col].min()

    )

st.dataframe(

    normalized,

    use_container_width=True

)

st.info("""

The normalized table allows fair comparison of all models.

Values closer to **1.0** indicate better performance.

""")

# ==========================================================
# Required Imports
# ==========================================================

from pathlib import Path

# ==========================================================
# Random Forest Evaluation
# ==========================================================

st.markdown("---")

st.markdown(
    '<p class="section-title">Random Forest Evaluation</p>',
    unsafe_allow_html=True
)

st.write("""
The following evaluation results are generated using the
trained **Random Forest** model on the test dataset.
""")

# ==========================================================
# Confusion Matrix
# ==========================================================

st.subheader("Confusion Matrix")

confusion_path = Path("reports/confusion_matrix.png")

if confusion_path.exists():

    st.image(

        str(confusion_path),

        caption="Confusion Matrix",

        use_container_width=True

    )

else:

    st.warning("confusion_matrix.png not found.")

# ==========================================================
# ROC Curve
# ==========================================================

st.markdown("---")

st.subheader("ROC Curve")

roc_path = Path("reports/roc_curve.png")

if roc_path.exists():

    st.image(

        str(roc_path),

        caption="Receiver Operating Characteristic (ROC)",

        use_container_width=True

    )

else:

    st.warning("roc_curve.png not found.")

# ==========================================================
# Precision Recall Curve
# ==========================================================

st.markdown("---")

st.subheader("Precision–Recall Curve")

pr_path = Path("reports/precision_recall_curve.png")

if pr_path.exists():

    st.image(

        str(pr_path),

        caption="Precision Recall Curve",

        use_container_width=True

    )

else:

    st.warning("precision_recall_curve.png not found.")

# ==========================================================
# Classification Report
# ==========================================================

st.markdown("---")

st.subheader("Classification Report")

classification_path = Path(

    "reports/classification_report.csv"

)

if classification_path.exists():

    classification_report = pd.read_csv(

        classification_path

    )

    st.dataframe(

        classification_report,

        use_container_width=True,

        hide_index=True

    )

else:

    st.warning(

        "classification_report.csv not found."

    )

# ==========================================================
# Metric Interpretation
# ==========================================================

st.markdown("---")

st.subheader("Performance Interpretation")

st.info("""

### Accuracy

Measures the overall percentage of correctly classified transactions.

---

### Precision

Represents how many predicted fraud transactions are actually fraud.

Higher precision reduces false alarms.

---

### Recall

Represents how many actual fraud transactions were detected.

Higher recall means fewer frauds are missed.

---

### F1 Score

Balances Precision and Recall.

For fraud detection, F1 Score is generally a better metric than Accuracy because the dataset is highly imbalanced.

---

### ROC AUC

Measures the model's ability to distinguish between fraud and genuine transactions.

A value close to **1.0** indicates excellent discrimination.

""")

# ==========================================================
# Final Model Summary
# ==========================================================

st.markdown("---")

st.subheader("Final Random Forest Performance")

rf = metrics.loc[

    metrics["Model"] == "Random Forest"

].iloc[0]

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:

    st.metric(

        "Accuracy",

        f"{rf['Accuracy']:.4f}"

    )

    st.metric(

        "Precision",

        f"{rf['Precision']:.4f}"

    )

with metric_col2:

    st.metric(

        "Recall",

        f"{rf['Recall']:.4f}"

    )

    st.metric(

        "F1 Score",

        f"{rf['F1 Score']:.4f}"

    )

with metric_col3:

    st.metric(

        "ROC AUC",

        f"{rf['ROC AUC']:.4f}"

    )

    st.metric(

        "Selected Model",

        "Random Forest"

    )

# ==========================================================
# Why Random Forest?
# ==========================================================

st.markdown("---")

st.subheader("Why Random Forest was Selected")

st.success("""

Random Forest was selected as the final model because:

✅ Highest overall F1 Score

✅ High Precision

✅ Strong Recall

✅ Excellent ROC AUC

✅ Stable performance on imbalanced datasets

✅ Low overfitting due to ensemble learning

✅ Robust prediction performance

These characteristics make Random Forest highly suitable
for credit card fraud detection.

""")

# ==========================================================
# Required Imports
# ==========================================================

from pathlib import Path

# ==========================================================
# Feature Importance Section
# ==========================================================

st.markdown("---")

st.markdown(
    '<p class="section-title">Feature Importance</p>',
    unsafe_allow_html=True
)

st.write(
    """
    Feature importance helps identify which transaction
    attributes have the greatest influence on the
    Random Forest model's predictions.
    """
)

# ==========================================================
# Feature Importance Image
# ==========================================================

feature_importance_img = Path(
    "reports/feature_importance.png"
)

if feature_importance_img.exists():

    st.image(
        str(feature_importance_img),
        caption="Random Forest Feature Importance",
        use_container_width=True
    )

else:

    st.warning(
        "feature_importance.png not found."
    )

# ==========================================================
# Feature Importance Table
# ==========================================================

feature_importance_csv = Path(
    "reports/feature_importance.csv"
)

if feature_importance_csv.exists():

    importance_df = pd.read_csv(
        feature_importance_csv
    )

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    st.subheader("Top Important Features")

    st.dataframe(
        importance_df.head(20),
        use_container_width=True,
        hide_index=True
    )

else:

    st.warning(
        "feature_importance.csv not found."
    )

# ==========================================================
# Feature Importance Chart
# ==========================================================

if feature_importance_csv.exists():

    st.markdown("---")

    st.subheader("Top 15 Feature Importance")

    fig, ax = plt.subplots(figsize=(10,6))

    top15 = importance_df.head(15)

    ax.barh(
        top15["Feature"],
        top15["Importance"]
    )

    ax.invert_yaxis()

    ax.set_xlabel("Importance")

    ax.set_ylabel("Feature")

    st.pyplot(fig)

    plt.close(fig)

# ==========================================================
# SHAP Summary Plot
# ==========================================================

st.markdown("---")

st.subheader("SHAP Summary Plot")

shap_summary = Path(
    "reports/shap_summary.png"
)

if shap_summary.exists():

    st.image(
        str(shap_summary),
        caption="SHAP Summary Plot",
        use_container_width=True
    )

else:

    st.warning(
        "shap_summary.png not found."
    )

# ==========================================================
# SHAP Bar Plot
# ==========================================================

st.markdown("---")

st.subheader("SHAP Global Feature Importance")

shap_bar = Path(
    "reports/shap_bar.png"
)

if shap_bar.exists():

    st.image(
        str(shap_bar),
        caption="SHAP Feature Importance",
        use_container_width=True
    )

else:

    st.warning(
        "shap_bar.png not found."
    )

# ==========================================================
# Global Feature Ranking
# ==========================================================

st.markdown("---")

st.subheader("Global Feature Ranking")

if feature_importance_csv.exists():

    ranking = importance_df.copy()

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

# ==========================================================
# Feature Importance Statistics
# ==========================================================

st.markdown("---")

st.subheader("Feature Importance Statistics")

if feature_importance_csv.exists():

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Features",
            len(importance_df)
        )

    with col2:

        st.metric(
            "Highest Importance",
            f"{importance_df['Importance'].max():.4f}"
        )

    with col3:

        st.metric(
            "Average Importance",
            f"{importance_df['Importance'].mean():.4f}"
        )

# ==========================================================
# Top 5 Features
# ==========================================================

if feature_importance_csv.exists():

    st.markdown("---")

    st.subheader("Top 5 Most Important Features")

    top5 = importance_df.head(5)

    for _, row in top5.iterrows():

        st.success(

            f"{row['Feature']}"

            f"  → Importance: "

            f"{row['Importance']:.5f}"

        )

# ==========================================================
# SHAP Explanation
# ==========================================================

st.markdown("---")

st.subheader("Understanding SHAP")

st.info("""

### What is SHAP?

SHAP (SHapley Additive exPlanations)
explains the contribution of every feature.

---

Positive SHAP Value

⬆ Increases fraud probability

---

Negative SHAP Value

⬇ Decreases fraud probability

---

Summary Plot

Shows overall feature influence.

---

Bar Plot

Ranks features based on
average absolute SHAP value.

""")

# ==========================================================
# Interpretation
# ==========================================================

st.markdown("---")

st.subheader("Interpretation")

st.write("""

The feature importance and SHAP analysis indicate
which transaction variables contribute the most to
fraud prediction.

Instead of treating the model as a black box,
Explainable AI allows investigators to understand
why the model identifies a transaction as fraudulent.

This improves transparency,
trust,
and decision-making.

""")

# ==========================================================
# Download Feature Importance
# ==========================================================

if feature_importance_csv.exists():

    csv = importance_df.to_csv(index=False)

    st.download_button(

        "📥 Download Feature Importance",

        csv,

        file_name="feature_importance.csv",

        mime="text/csv",

        use_container_width=True

    )

# ==========================================================
# Export Model Metrics
# ==========================================================

st.markdown("---")

st.markdown(
    '<p class="section-title">📥 Export Reports</p>',
    unsafe_allow_html=True
)

metrics_csv = metrics.to_csv(index=False)

st.download_button(
    label="📄 Download Model Metrics",
    data=metrics_csv,
    file_name="model_metrics.csv",
    mime="text/csv",
    use_container_width=True
)

# ==========================================================
# Download Feature Importance
# ==========================================================

feature_file = Path("reports/feature_importance.csv")

if feature_file.exists():

    with open(feature_file, "rb") as f:

        st.download_button(
            label="📊 Download Feature Importance",
            data=f,
            file_name="feature_importance.csv",
            mime="text/csv",
            use_container_width=True
        )

# ==========================================================
# Download Classification Report
# ==========================================================

classification_file = Path(
    "reports/classification_report.csv"
)

if classification_file.exists():

    with open(classification_file, "rb") as f:

        st.download_button(
            label="📑 Download Classification Report",
            data=f,
            file_name="classification_report.csv",
            mime="text/csv",
            use_container_width=True
        )

# ==========================================================
# Model Leaderboard
# ==========================================================

st.markdown("---")

st.markdown(
    '<p class="section-title">🏆 Model Leaderboard</p>',
    unsafe_allow_html=True
)

leaderboard = metrics.sort_values(
    by="F1 Score",
    ascending=False
).reset_index(drop=True)

leaderboard.index += 1

leaderboard.insert(
    0,
    "Rank",
    leaderboard.index
)

st.dataframe(
    leaderboard,
    use_container_width=True,
    hide_index=True
)

# ==========================================================
# Best Model Highlight
# ==========================================================

best_model = leaderboard.iloc[0]

st.success(f"""
🥇 Best Performing Model

Model: {best_model['Model']}

Accuracy: {best_model['Accuracy']:.4f}

Precision: {best_model['Precision']:.4f}

Recall: {best_model['Recall']:.4f}

F1 Score: {best_model['F1 Score']:.4f}

ROC AUC: {best_model['ROC AUC']:.4f}
""")

# ==========================================================
# Project Conclusion
# ==========================================================

st.markdown("---")

st.markdown(
    '<p class="section-title">🎯 Project Conclusion</p>',
    unsafe_allow_html=True
)

st.success("""

### Final Conclusion

After evaluating six machine learning algorithms:

• Logistic Regression

• Decision Tree

• Random Forest

• XGBoost

• LightGBM

• CatBoost

Random Forest achieved the best overall balance of:

✅ Accuracy

✅ Precision

✅ Recall

✅ F1 Score

✅ ROC AUC

Random Forest was therefore selected as the final model
for the Credit Card Fraud Detection System.

The addition of SHAP Explainable AI improves model
transparency by explaining why predictions are made,
making the system more trustworthy and suitable for
real-world fraud detection.

""")

# ==========================================================
# Project Technologies
# ==========================================================

st.markdown("---")

st.markdown(
    '<p class="section-title">🛠 Technologies Used</p>',
    unsafe_allow_html=True
)

tech1, tech2, tech3 = st.columns(3)

with tech1:

    st.info("""

Python

Pandas

NumPy

Scikit-Learn

""")

with tech2:

    st.info("""

Random Forest

XGBoost

LightGBM

CatBoost

""")

with tech3:

    st.info("""

SHAP

Matplotlib

Streamlit

Joblib

""")

# ==========================================================
# System Information
# ==========================================================

st.markdown("---")

st.markdown(
    '<p class="section-title">⚙️ System Information</p>',
    unsafe_allow_html=True
)

sys1, sys2, sys3 = st.columns(3)

with sys1:

    st.metric(
        "Models Evaluated",
        len(metrics)
    )

with sys2:

    st.metric(
        "Final Model",
        "Random Forest"
    )

with sys3:

    st.metric(
        "Explainability",
        "SHAP"
    )

# ==========================================================
# User Guide
# ==========================================================

st.markdown("---")

with st.expander("📘 User Guide"):

    st.markdown("""

### Dashboard Sections

📊 Model Comparison

Compare all trained ML models.

---

📈 Performance Charts

Visual comparison of Accuracy,
Precision,
Recall,
F1 Score,
ROC AUC.

---

📉 Evaluation

Confusion Matrix

ROC Curve

Precision Recall Curve

---

🧠 Explainable AI

Feature Importance

SHAP Summary

SHAP Bar Plot

---

📥 Reports

Download evaluation reports
for documentation.

""")

# ==========================================================
# Footer
# ==========================================================

st.markdown("---")

st.markdown(
"""
<div style='text-align:center;
padding:25px;
background:#f5f5f5;
border-radius:10px;'>

<h3>Credit Card Fraud Detection Using Machine Learning</h3>

<p>

Final Model:
<b>Random Forest</b>

<br><br>

Explainable AI:
<b>SHAP</b>

<br><br>

Developed using

Python | Streamlit | Scikit-Learn | SHAP

<br><br>

B.E. Final Year Project

</p>

</div>
""",
unsafe_allow_html=True
)