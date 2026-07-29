import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

# ============================================================
# Create Reports Folder
# ============================================================

os.makedirs("reports", exist_ok=True)

# ============================================================
# Load Model
# ============================================================

model = joblib.load("models/random_forest.pkl")

# ============================================================
# Load Dataset
# ============================================================

X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

# ============================================================
# Predictions
# ============================================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ============================================================
# Metrics
# ============================================================

metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC AUC"
    ],
    "Value": [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        roc_auc_score(y_test, y_prob)
    ]
})

metrics.to_csv(
    "reports/model_metrics.csv",
    index=False
)

print("Model metrics saved.")

# ============================================================
# Classification Report
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

pd.DataFrame(report).transpose().to_csv(
    "reports/classification_report.csv"
)

print("Classification report saved.")

# ============================================================
# Feature Importance
# ============================================================

importance = pd.DataFrame({

    "Feature": X_test.columns,

    "Importance": model.feature_importances_

})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

importance.to_csv(
    "reports/feature_importance.csv",
    index=False
)

print("Feature importance saved.")

# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

plt.imshow(cm)

plt.colorbar()

plt.xticks([0,1],["Genuine","Fraud"])
plt.yticks([0,1],["Genuine","Fraud"])

for i in range(2):
    for j in range(2):
        plt.text(j,i,str(cm[i,j]),
                 ha="center",
                 va="center",
                 fontsize=14)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig(
    "reports/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Confusion matrix saved.")

# ============================================================
# ROC Curve
# ============================================================

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,5))

plt.plot(fpr, tpr, label="Random Forest")

plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()

plt.savefig(
    "reports/roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("ROC curve saved.")

# ============================================================
# Precision Recall Curve
# ============================================================

precision, recall, _ = precision_recall_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(6,5))

plt.plot(recall, precision)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision Recall Curve")

plt.savefig(
    "reports/precision_recall_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("PR curve saved.")

# ============================================================
# SHAP
# ============================================================

print("Generating SHAP plots...")

explainer = shap.TreeExplainer(model)

sample_size = min(500, len(X_test))

X_sample = X_test.sample(
    sample_size,
    random_state=42
)

shap_values = explainer(X_sample)

# ------------------------------------------------------------
# Handle binary-class output for SHAP 0.52.0
# ------------------------------------------------------------

if len(shap_values.values.shape) == 3:
    shap_plot_values = shap_values.values[:, :, 1]
else:
    shap_plot_values = shap_values.values

# ============================================================
# SHAP Summary Plot
# ============================================================

plt.figure(figsize=(10,7))

shap.summary_plot(
    shap_plot_values,
    X_sample,
    show=False
)

plt.savefig(
    "reports/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("SHAP Summary saved.")

# ============================================================
# SHAP Bar Plot
# ============================================================

if len(shap_values.values.shape) == 3:
    explanation = shap.Explanation(
        values=shap_plot_values,
        base_values=shap_values.base_values[:, 1],
        data=X_sample.values,
        feature_names=X_sample.columns.tolist()
    )
else:
    explanation = shap_values

plt.figure(figsize=(10,7))

shap.plots.bar(
    explanation,
    max_display=20,
    show=False
)

plt.savefig(
    "reports/shap_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("SHAP Bar saved.")

print("\nAll reports generated successfully!")