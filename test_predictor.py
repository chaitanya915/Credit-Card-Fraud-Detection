import pandas as pd

from src.predictor import predict_transaction

sample = pd.read_csv(
    "data/processed/X_test_scaled.csv"
).iloc[[0]]

prediction, probability = predict_transaction(sample)

print(prediction)

print(probability)