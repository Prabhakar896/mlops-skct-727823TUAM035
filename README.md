# PJME Hourly Load Forecasting – MLOps Assignment

**Student Name:** Prabhakar Anandan  
**Roll Number:** 727823TUAM035  
**Dataset:** PJME Hourly Load (Datetime, PJME_MW)  

---

## Project Overview
This project demonstrates a full MLOps workflow including:

1. **MLflow Experiment Tracking**
   - Track 12+ experiments with different algorithms and hyperparameters
   - Log metrics: MAE, RMSE, R², training_time_seconds, model_size_mb, random_seed
   - Save best model artifact

2. **Azure ML Pipeline**
   - 3 stages: `data_prep.py`, `train_pipeline.py`, `evaluate.py`
   - Pipeline defined in `pipeline_727823TUAM035.yml`
   - Run ID in Azure ML workspace: `kind_soccer_hmtqjvk0cl`
   - CPU compute cluster used: `cpu-cluster`

3. **EDA**
   - 3 plots showing distribution and trends in PJME load data
   - Notebook: `notebooks/eda.ipynb`

---

## Setup Instructions

### **1. Clone Repository**
```bash
git clone https://github.com/Prabhakar8956/mlops-skct-727823TUAM035.git
cd mlops-skct-727823TUAM035

### **2. Create a Virtual Environment**
     python -m venv venv
# Activate the virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
4. Run MLflow Experiments Locally

Make sure your dataset is in ./data/ folder

python code/training.py --input_data ./data --model_output ./outputs

This will run 12+ experiments, log metrics to MLflow, and save the best model

5. Submit Azure ML Pipeline

Make sure you are logged in to Azure CLI and workspace is configured

az ml job create --file pipeline_727823TUAM035.yml -g Prabha -w mlops-SKCT-727823TUAM035 --stream
6. Check Results

MLflow UI: metrics, tags, and artifacts for all runs

Azure ML Portal: check pipeline run and outputs

Screenshots saved in /screenshots/ folder

Project Structure
/code/                  ← Python scripts for pipeline stages
/notebooks/             ← EDA notebook
/screenshots/           ← MLflow and Azure portal screenshots
/report/                ← PDF report
/pipeline_727823TUAM035.yml  ← Azure ML pipeline YAML
/requirements.txt       ← Project dependencies
README.md               ← This file
