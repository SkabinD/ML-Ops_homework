import pandas as pd
from app.models.inference_model import BlendedModel

def make_prediction(X_data: dict["catboost": pd.DataFrame, "lightgbm": pd.DataFrame], client_ids: pd.Series):
    
    model = BlendedModel()

    submission = pd.DataFrame({
        "client_id": client_ids,
        "preds": model.predict(X_data)
    })
    
    return submission