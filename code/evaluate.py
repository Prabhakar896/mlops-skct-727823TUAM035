# PRABHAKAR A 727823TUAM035
# evaluate.py
# Corrected for Azure ML pipeline

import pandas as pd
import argparse
import datetime
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

print("727823TUAM035", datetime.datetime.now())

# --------------------------
# Arguments
# --------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True)
args = parser.parse_args()

# --------------------------
# Load cleaned data
# --------------------------
data_file = os.path.join(args.input_data, "pjm_load_clean.csv")
df = pd.read_csv(data_file, parse_dates=['Datetime'])
df = df.set_index('Datetime')

X = df[['Load_lag1', 'Load_lag24']]
y = df['PJME_MW']

# --------------------------
# Load trained model
# --------------------------
# Updated filename to match train.py
model_file = os.path.join(args.model_path, "best_model.pkl")

if not os.path.exists(model_file):
    raise FileNotFoundError(f"Model file not found: {model_file}")

model = joblib.load(model_file)

# --------------------------
# Predict and evaluate
# --------------------------
y_pred = model.predict(X)

mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print(f"Evaluation Results -> MAE: {mae}, RMSE: {rmse}, R2: {r2}")
print("Evaluation completed.")