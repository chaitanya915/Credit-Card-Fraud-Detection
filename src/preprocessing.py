import pandas as pd


def validate_transaction(transaction_df):

    if transaction_df.isnull().sum().sum() > 0:

        raise ValueError(
            "Missing values detected."
        )

    return transaction_df