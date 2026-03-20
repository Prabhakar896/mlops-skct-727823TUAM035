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