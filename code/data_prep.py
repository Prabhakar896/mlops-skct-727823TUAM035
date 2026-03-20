# PRABHAKAR A 727823TUAM035
import pandas as pd
import argparse
import datetime
import os

print("727823TUAM035", datetime.datetime.now())

# Arguments from Azure
parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str)
parser.add_argument("--output_data", type=str)
args = parser.parse_args()

# Read dataset from Azure input
df = pd.read_csv(args.input_data, parse_dates=['Datetime'])
df = df.set_index('Datetime')

# Feature Engineering
df['Load_lag1'] = df['PJME_MW'].shift(1)
df['Load_lag24'] = df['PJME_MW'].shift(24)
df = df.dropna()

# Create output folder if not exists
os.makedirs(args.output_data, exist_ok=True)

# Save cleaned data
df.to_csv(f"{args.output_data}/pjm_load_clean.csv")

print("Data preparation completed.")