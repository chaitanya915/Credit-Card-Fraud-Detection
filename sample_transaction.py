from turtle import pd


import pandas as pd

# Load the test dataset
X_test = pd.read_csv("data/processed/X_test_scaled.csv")

# Save the first transaction
sample = X_test.iloc[[0]]

sample.to_csv(
    "data/processed/sample_transaction.csv",
    index=False
)

print("sample_transaction.csv created successfully!")