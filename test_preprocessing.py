import pandas as pd

from src.preprocessing import validate_transaction

sample = pd.read_csv(
    "data/processed/X_test_scaled.csv"
).iloc[[0]]

sample = validate_transaction(sample)

print(sample.head())