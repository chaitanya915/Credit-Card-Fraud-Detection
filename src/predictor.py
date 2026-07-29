import joblib
import pandas as pd

MODEL_PATH = "models/random_forest.pkl"

model = joblib.load(MODEL_PATH)


def predict_transaction(transaction_df):
    """
    Predict whether a transaction is Fraud or Genuine.

    Parameters
    ----------
    transaction_df : pandas.DataFrame
        Single-row dataframe with all required features.

    Returns
    -------
    prediction : int
        0 = Genuine
        1 = Fraud

    probability : float
        Fraud probability.
    """

    prediction = model.predict(transaction_df)[0]

    probability = model.predict_proba(transaction_df)[0][1]

    return prediction, probability