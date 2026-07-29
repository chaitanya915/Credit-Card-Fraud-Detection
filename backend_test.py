import pandas as pd

from src.predictor import predict_transaction
from src.explain import generate_shap_values
from src.preprocessing import validate_transaction

transaction = pd.read_csv(
    "data/processed/sample_transaction.csv"
)

transaction = validate_transaction(transaction)

prediction, probability = predict_transaction(transaction)

print("Prediction:", prediction)

print("Fraud Probability:", probability)

explanation = generate_shap_values(transaction)

print(explanation.values.shape)