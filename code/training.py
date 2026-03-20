# training.py
import pandas as pd
import numpy as np
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
import time
import os
import joblib

# --------------------------
# Student Name and Roll No
# --------------------------
STUDENT_NAME = "PRABHAKAR A"
ROLL_NO = "727823TUAM035"
DATASET_NAME = "pjm_load"

# Random Seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# --------------------------
# Load Dataset
# --------------------------
df = pd.read_csv("data/pjm_load_clean.csv", parse_dates=['Datetime'])
df = df.set_index('Datetime')

# Create lag features
df['Load_lag1'] = df['PJME_MW'].shift(1)
df['Load_lag24'] = df['PJME_MW'].shift(24)
df = df.dropna()

X = df[['Load_lag1', 'Load_lag24']]
y = df['PJME_MW']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# --------------------------
# MLflow Setup (SQLite Backend)
# --------------------------
# Recommended: SQLite DB for Windows + Model Registry
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(f"SKCT_{ROLL_NO}_{DATASET_NAME}")

# --------------------------
# Models and Hyperparameters
# --------------------------
experiments = [
    {"name": "RF_50", "model": RandomForestRegressor(n_estimators=50, random_state=RANDOM_SEED)},
    {"name": "RF_100", "model": RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)},
    {"name": "GB_50", "model": GradientBoostingRegressor(n_estimators=50, random_state=RANDOM_SEED)},
    {"name": "GB_100", "model": GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_SEED)},
    {"name": "DT", "model": DecisionTreeRegressor(random_state=RANDOM_SEED)},
    {"name": "LR", "model": LinearRegression()},
    {"name": "SVR", "model": SVR(kernel='rbf')},
    {"name": "KNN_5", "model": KNeighborsRegressor(n_neighbors=5)},
    {"name": "KNN_10", "model": KNeighborsRegressor(n_neighbors=10)},
    {"name": "ET_50", "model": ExtraTreesRegressor(n_estimators=50, random_state=RANDOM_SEED)},
    {"name": "ET_100", "model": ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_SEED)},
    {"name": "AB_50", "model": AdaBoostRegressor(n_estimators=50, random_state=RANDOM_SEED)},
]

best_r2 = -np.inf
best_model_run_id = None

# --------------------------
# Loop Through Experiments
# --------------------------
for exp in experiments:
    with mlflow.start_run(run_name=exp["name"]) as run:
        start_time = time.time()
        model = exp["model"]
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Predict & metrics
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Model size in MB
        model_file = f"temp_model.pkl"
        joblib.dump(model, model_file)
        model_size_mb = os.path.getsize(model_file) / (1024 * 1024)
        os.remove(model_file)
        
        # Log metrics
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("MAPE", mape)
        mlflow.log_metric("training_time_seconds", training_time)
        mlflow.log_metric("model_size_mb", model_size_mb)
        
        # Log hyperparameters
        if hasattr(model, "get_params"):
            for k, v in model.get_params().items():
                mlflow.log_param(k, v)
        
        mlflow.log_param("random_seed", RANDOM_SEED)
        
        # Log tags
        mlflow.set_tag("student_name", STUDENT_NAME)
        mlflow.set_tag("roll_number", ROLL_NO)
        mlflow.set_tag("dataset", DATASET_NAME)
        
        # Log model artifact
        mlflow.sklearn.log_model(model, "model")
        
        # Track best model
        if r2 > best_r2:
            best_r2 = r2
            best_model_run_id = run.info.run_id

print(f"Best Model Run ID: {best_model_run_id}, Best R2: {best_r2:.4f}")