import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin
import pandas as pd
import numpy as np

from sales_forecasting_project.utils import MeanEnsemble, BoolToIntImputer

mlflow.set_tracking_uri("file:./sales_forecasting_project/mlruns")

mlflow.set_experiment("Walmart Sales Forecasting")

MODEL_PATH = "./sales_forecasting_project/models/mean_ensemble_model.joblib"
MODEL_NAME = "Walmart-Sales-Ensemble"

metrics = {
    "wmae": 2624.29773,
}

params = {
    "model_type": "Ensemble (RF + ExtraTrees)",
    "rf_n_estimators": 60,
    "rf_max_depth": None,
    "rf_min_samples_split": 2,
    "rf_min_samples_leaf": 1,
    "rf_random_state": 0,
    "rf_n_jobs": -1,
    "etr_n_estimators": 60,
    "etr_max_depth": None,
    "etr_min_samples_split": 2,
    "etr_min_samples_leaf": 1,
    "etr_random_state": 0,
    "etr_n_jobs": -1,
    "feature_set_version": "v1"
}

print("Defining model input schema...")
try:
    input_sample = pd.DataFrame({
        "Store": [1],
        "Dept": [1],
        "IsHoliday": [False],
        "Size": [151315],
        "Type": ["A"],
        "Year": [2010],
        "Week": [5],
        "Day": [5] 
    })

    input_sample = input_sample[['Store', 'Dept', 'IsHoliday', 'Size', 'Type', 'Year', 'Week', 'Day']]

except Exception as e:
    print(f"Warning: Could not create input_sample. Schema will not be logged. Error: {e}")
    input_sample = None


print("Loading model from disk...")
model = joblib.load(MODEL_PATH, mmap_mode='r')

print("Starting MLflow run...")
with mlflow.start_run(run_name="Production Candidate") as run:
    
    print("Logging parameters...")
    mlflow.log_params(params)
    
    print("Logging metrics...")
    mlflow.log_metrics(metrics)
    
    print("Logging model...")
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=MODEL_NAME,
        input_example=input_sample
    )
    
    run_id = run.info.run_id
    print(f"\n--- Run Complete ---")
    print(f"Model saved in run: {run_id}")
    print(f"Model '{MODEL_NAME}' registered in Model Registry.")

print("\nScript finished.")