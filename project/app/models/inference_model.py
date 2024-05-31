from pandas import DataFrame
from catboost import CatBoostClassifier
from lightgbm import Booster

class BlendedModel:
    def __init__(self, threshold=0.375):
        self.catboost_model_1 = CatBoostClassifier().load_model("app/models/final_cb_1")
        self.catboost_model_2 = CatBoostClassifier().load_model("app/models/final_cb_2")
        self.lightgbm_model_1= Booster(model_file="app/models/final_lgbm.txt")
        self.threshold = threshold
        
    def predict_proba(self, X_data: dict["catboost": DataFrame, "lightgbm": DataFrame]):
        y_preds_proba = DataFrame({
            "catboost_model_1": self.catboost_model_1.predict_proba(X_data["catboost"])[:, 1],
            "catboost_model_2": self.catboost_model_2.predict_proba(X_data["catboost"])[:, 1],
            "lightgbm_model_1": self.lightgbm_model_1.predict(X_data["lightgbm"])
        })
        y_preds_proba_blended = y_preds_proba.mean(axis=1)
        return y_preds_proba_blended
    
    def predict(self, X_data: dict["catboost": DataFrame, "lightgbm": DataFrame]):
        y_preds_proba_blended = self.predict_proba(X_data)
        y_preds_blended = y_preds_proba_blended > self.threshold
        y_preds_blended = y_preds_blended.apply(int)
        return y_preds_blended