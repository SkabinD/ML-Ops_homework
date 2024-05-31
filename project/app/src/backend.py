import app.src.preprocessing as preproc
import app.src.scorer as scorer
import app.src.utils as utils
from app.models.inference_model import BlendedModel

def predict_routine(upload):
    X_data, client_id = preproc.read_file(upload)
    X_data = preproc.run_preproc(X_data)
    
    model = BlendedModel()
    submission = scorer.make_prediction(X_data, client_id)
    predict_proba = model.predict_proba(X_data)
    feature_importance = utils.get_feature_importances(model)

    plot = utils.save_predict_density_dist(predict_proba=predict_proba, thresh=model.threshold)
    utils.save_feature_importances_to_json(feature_importance)
    utils.save_prediction(submission=submission)

    return submission, plot, feature_importance
