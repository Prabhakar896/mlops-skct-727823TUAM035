# PRABHAKAR A 727823TUAM035
# train.py
# Fully corrected for Azure ML pipeline, no start_run() call

import pandas as pd
import numpy as np
import argparse
import datetime
import time
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.sklearn

print("727823TUAM035", datetime.datetime.now())

# --------------------------
# Arguments
# --------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str, required=True)
parser.add_argument("--model_output", type=str, required=True)
args = parser.parse_args()

# --------------------------
# Load cleaned data
# --------------------------
data_file = os.path.join(args.input_data, "pjm_load_clean.csv")
df = pd.read_csv(data_file, parse_dates=['Datetime'])
df = df.set_index('Datetime')

X = df[['Load_lag1', 'Load_lag24']]
y = df['PJME_MW']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# --------------------------
# MLflow setup
# --------------------------
# Do NOT call start_run() in Azure ML pipeline
# Metrics and models will log to the active job run
# mlflow.set_experiment() is optional; Azure ML sets it automatically

experiments = [
    {"name": "RF_50", "model": RandomForestRegressor(n_estimators=50)},
    {"name": "RF_100", "model": RandomForestRegressor(n_estimators=100)},
    {"name": "GB_50", "model": GradientBoostingRegressor(n_estimators=50)},
    {"name": "GB_100", "model": GradientBoostingRegressor(n_estimators=100)},
    {"name": "DT", "model": DecisionTreeRegressor()},
    {"name": "LR", "model": LinearRegression()},
    {"name": "SVR", "model": SVR()},
    {"name": "KNN_5", "model": KNeighborsRegressor(n_neighbors=5)},
    {"name": "KNN_10", "model": KNeighborsRegressor(n_neighbors=10)},
    {"name": "ET_50", "model": ExtraTreesRegressor(n_estimators=50)},
    {"name": "ET_100", "model": ExtraTreesRegressor(n_estimators=100)},
    {"name": "AB_50", "model": AdaBoostRegressor(n_estimators=50)},
]

best_r2 = -np.inf
best_model = None

# --------------------------
# Train models
# --------------------------
for exp in experiments:
    start = time.time()
    model = exp["model"]
    model.fit(X_train, y_train)

    training_time = time.time() - start
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Log metrics with MLflow
    mlflow.log_metric(f"{exp['name']}_MAE", mae)
    mlflow.log_metric(f"{exp['name']}_RMSE", rmse)
    mlflow.log_metric(f"{exp['name']}_R2", r2)
    mlflow.log_metric(f"{exp['name']}_training_time", training_time)

    # Log model artifact with MLflow
    model_dir = os.path.join(args.model_output, f"model_{exp['name']}")
    os.makedirs(model_dir, exist_ok=True)
    mlflow.sklearn.log_model(model, model_dir)

    if r2 > best_r2:
        best_r2 = r2
        best_model = model

# --------------------------
# Save best model separately
# --------------------------
os.makedirs(args.model_output, exist_ok=True)
joblib.dump(best_model, os.path.join(args.model_output, "best_model.pkl"))

print("Training completed. Best R2:", best_r2)