import pandas as pd
import json

from seaborn import histplot
import matplotlib.pyplot as plt

from app.models.inference_model import BlendedModel

def get_feature_importances(model: BlendedModel) -> dict:
    """Колхозный метод для определения важности признаков среди ансамбля моделей"""

    catboost_1_importance = model.catboost_model_1.get_feature_importance(prettified=True)
    catboost_2_importance = model.catboost_model_2.get_feature_importance(prettified=True)
    lightgbm_1_importance = pd.DataFrame({
    'Feature': model.lightgbm_model_1.feature_name(),
    'Importance': model.lightgbm_model_1.feature_importance()
    })  
    
    catboost_1_importance = pd.DataFrame(catboost_1_importance)
    catboost_2_importance = pd.DataFrame(catboost_2_importance)

    all_importances = pd.concat([
    catboost_1_importance[['Feature Id', 'Importances']].rename(columns={'Feature Id': 'Feature', 'Importances': 'Importance'}),
    catboost_2_importance[['Feature Id', 'Importances']].rename(columns={'Feature Id': 'Feature', 'Importances': 'Importance'}),
    lightgbm_1_importance
    ])

    grouped_importances = all_importances.groupby('Feature').sum().reset_index()
    top_5_features = grouped_importances.sort_values(by='Importance', ascending=False).head(5)
    top_5_features_dict = top_5_features.set_index('Feature')['Importance'].to_dict()

    return top_5_features_dict


def save_feature_importances_to_json(features_dict: dict, path_to_file="app/output/features_importance.json") -> None:
    with open(path_to_file, "w", encoding='utf-8') as f:
        json.dump(features_dict, f, ensure_ascii=False)

def save_predict_density_dist(predict_proba: pd.Series, thresh=None, path_to_file="app/output/predict_density_distrib.png") -> None:
    pred_dens_plot = histplot(predict_proba, bins=[.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0], stat="density")
    plt.xlabel("Предсказанная вероятность")
    plt.ylabel("Плотность")
    plt.xlim([0., 1.])
    plt.title("Плотность распределения предсказаний модели")
    if thresh:
        plt.axvline(thresh, color="red", linestyle="dashed", label="Порог")
        plt.legend()
    fig = pred_dens_plot.get_figure()
    fig.savefig(path_to_file)
    return fig

def save_prediction(submission: pd.DataFrame, path_to_file="app/output/submission.csv") -> None:
    submission.to_csv(path_to_file, index=False)


if __name__ == "__main__":
    model = BlendedModel()
    fi = get_feature_importances(model)
    save_feature_importances_to_json(fi, "app/output/features_importance.json")

    from app.src.preprocessing import read_file, run_preproc
    X_data, user_ids = read_file()
    X_data = run_preproc(X_data)
    pred_prob = model.predict_proba(X_data)
    save_predict_density_dist(pred_prob, thresh=0.375)

    from app.src.scorer import make_prediction
    submission = make_prediction(X_data, user_ids)
    save_prediction(submission)