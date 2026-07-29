import joblib
import shap

MODEL_PATH = "models/random_forest.pkl"

model = joblib.load(MODEL_PATH)

explainer = shap.TreeExplainer(model)


def generate_shap_values(transaction_df):

    shap_values = explainer(transaction_df)

    if shap_values.values.ndim == 3:

        values = shap_values.values[:, :, 1]

        base = shap_values.base_values[:, 1]

    else:

        values = shap_values.values

        base = shap_values.base_values

    explanation = shap.Explanation(

        values=values,

        base_values=base,

        data=transaction_df.values,

        feature_names=transaction_df.columns.tolist()

    )

    return explanation