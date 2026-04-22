import joblib
import pandas as pd

def load_models():
    models = {
        "Logistic Regression": joblib.load("model/logistic.pkl"),
        "Decision Tree": joblib.load("model/decision_tree.pkl"),
        "Random Forest": joblib.load("model/random_forest.pkl"),
        "Gradient Boosting": joblib.load("model/gradient_boosting.pkl"),
        "KNN": joblib.load("model/knn.pkl")
    }

    scaler = joblib.load("model/scaler.pkl")
    features = joblib.load("model/features.pkl")

    return models, scaler, features


def predict(model, scaler, features, data):
    data = data.reindex(columns=features, fill_value=0)
    X = scaler.transform(data)

    pred = model.predict(X)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0][1]
    else:
        proba = float(pred)

    return pred, proba