"""Tune an XGBoost classifier with MLflow tracking and save the best model."""
import os
import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------- MLflow
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
mlflow.set_experiment("tourism-wellness-package")

# ---------------------------------------------------------------- Data (from the workflow artifact)
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
ytest = pd.read_csv("ytest.csv").squeeze("columns")

numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting", "NumberOfFollowups",
    "PreferredPropertyStar", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]
categorical_features = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched", "MaritalStatus", "Designation",
]

# ---------------------------------------------------------------- Model + parameter grid
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]   # handle class imbalance

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42, eval_metric="logloss")
model_pipeline = make_pipeline(preprocessor, xgb_model)

param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__subsample": [0.8, 1.0],
    "xgbclassifier__colsample_bytree": [0.7, 1.0],
}

CLASSIFICATION_THRESHOLD = 0.5

with mlflow.start_run(run_name="xgboost-gridsearch"):
    # ---- Tune
    grid = GridSearchCV(model_pipeline, param_grid, cv=5, scoring="recall", n_jobs=-1)
    grid.fit(Xtrain, ytrain)

    # ---- Log every tuned parameter combination as a nested run
    results = grid.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(run_name=f"trial_{i}", nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_cv_recall", results["mean_test_score"][i])
            mlflow.log_metric("std_cv_recall", results["std_test_score"][i])

    # ---- Best model
    best_model = grid.best_estimator_
    mlflow.log_params({k: v for k, v in grid.best_params_.items()})
    mlflow.log_param("scoring", "recall")
    mlflow.log_param("cv_folds", 5)
    mlflow.log_metric("best_cv_recall", grid.best_score_)
    print("Best parameters:", grid.best_params_)
    print("Best CV recall :", round(grid.best_score_, 4))

    # ---- Evaluate on train and test
    metrics = {}
    for split, X, y in [("train", Xtrain, ytrain), ("test", Xtest, ytest)]:
        proba = best_model.predict_proba(X)[:, 1]
        pred = (proba >= CLASSIFICATION_THRESHOLD).astype(int)
        metrics.update({
            f"{split}_accuracy": accuracy_score(y, pred),
            f"{split}_precision": precision_score(y, pred),
            f"{split}_recall": recall_score(y, pred),
            f"{split}_f1": f1_score(y, pred),
            f"{split}_roc_auc": roc_auc_score(y, proba),
        })
        print(f"\n=== {split.upper()} classification report ===")
        print(classification_report(y, pred))
    mlflow.log_metrics(metrics)
    print({k: round(v, 4) for k, v in metrics.items()})

    # ---- Save the best model so the pipeline can commit it to the repo
    os.makedirs("tourism_project/deployment", exist_ok=True)
    model_path = "tourism_project/deployment/best_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path)
    print("Saved best model to", model_path)
