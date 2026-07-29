import pandas as pd

from src.explain import generate_shap_values

sample = pd.read_csv(
    "data/processed/X_test_scaled.csv"
).iloc[[0]]

explanation = generate_shap_values(sample)

print(type(explanation))

print(explanation.values.shape)