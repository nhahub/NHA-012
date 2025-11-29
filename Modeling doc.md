# Retail Sales Forecasting Project

📖 **Description**  
This project implements a robust machine learning pipeline to forecast weekly sales for various stores and departments. It leverages historical sales data, temporal features, and external economic indicators (CPI, Unemployment, Fuel Price) to predict future demand.

Going beyond simple modeling, this project includes a complete MLOps workflow:

1. Modeling: Advanced feature engineering and ensemble learning (Random Forest + Extra Trees).  
2. Tracking: Model versioning and metric tracking using MLflow.  
3. Deployment: A real-time REST API built with FastAPI.  
4. Interface: An interactive user dashboard built with Streamlit.

✨ **Key Features**

- **Advanced Feature Engineering:** Automatically extracts temporal features (Week, Month, Year) and calculates "Days Until" key holidays (Thanksgiving, Christmas) to capture seasonal trends.  
- **Robust Preprocessing Pipeline:**  
  - Handles missing values using custom imputation strategies (BoolToIntImputer).  
  - Encodes categorical variables (Type) and boolean flags (IsHoliday).  
- **Ensemble Learning:** Implements a custom MeanEnsemble strategy combining Random Forest and Extra Trees predictions for superior stability.  
- **MLflow Integration:** Tracks model parameters, metrics (WMAE), and artifacts; manages the model lifecycle via the MLflow Model Registry.  
- **Real-time API:** Provides a scalable endpoint (/predict) for serving forecasts instantly.  
- **Interactive Dashboard:** Allows non-technical users to input store details and visualize predictions via a web UI.

️ **Technology Stack**

- Core: Python, Pandas, NumPy  
- Machine Learning: Scikit-Learn, LightGBM, XGBoost, CatBoost  
- MLOps: MLflow  
- Serving: FastAPI, Uvicorn, Pydantic  
- Frontend: Streamlit  
- Persistence: Joblib

---

## 🚀 Getting Started

### Prerequisites

Ensure you have the required Python libraries installed:

```
pip install pandas numpy scikit-learn lightgbm xgboost catboost matplotlib seaborn joblib mlflow fastapi uvicorn streamlit pydantic requests
```

### Directory Structure

Ensure your project is structured as follows to support the imports in the scripts:

```
sales-forecasting/
├── Datasets/                     # CSV files (train.csv, features.csv, etc.)
├── sales_forecasting_project/
│   ├── models/                   # Directory for saved .joblib models
│   ├── mlruns/                   # MLflow tracking data
│   └── utils.py                  # Custom estimators (MeanEnsemble, BoolToIntImputer)
├── Modeling.ipynb                # Analysis and Training Notebook
├── log_model.py                  # Script to log model to MLflow
├── api.py                        # FastAPI backend
└── dashboard.py                  # Streamlit frontend
```

### Installation

1. Clone the repository:

```
git clone https://github.com/username/sales-forecasting.git
cd sales-forecasting
```

---

# 🎮 Workflow & Usage

## 1. Training & Analysis

Open **Modeling.ipynb** to train the models. This notebook will generate the optimized `mean_ensemble_model.joblib` file.

- Ensure the model is saved to:  
  `./sales_forecasting_project/models/mean_ensemble_model.joblib`

---

## Modeling Process

The `Modeling.ipynb` notebook performs the following steps:

### 1. Data Loading & Merging:

- Reads train.csv, features.csv, and stores.csv.  
- Merges sales data with features (CPI, Fuel Price) and store metadata (Size, Type).  
- Parses dates to extract Day, Week, Month, and Year.

### 2. Feature Engineering:

- **Holiday Countdowns:** Calculates Days_to_Thanksgiving and Days_to_Christmas.  
- **Special Events:** Creates binary flags for SuperBowlWeek, LaborDay, Thanksgiving, and Christmas.  
- **Missing Values:** Imputes missing MarkDown values with 0 and fills missing CPI/Unemployment with the mean.

### 3. Model Selection & Training:

- **Metric:** Uses Weighted Mean Absolute Error (WMAE) where holiday weeks are weighted 5× more.  
- **Comparison:** Evaluates LightGBM, CatBoost, XGBoost, Random Forest, and Extra Trees using 5-Fold CV.  
- **Feature Importance:** Identifies Dept, Size, Store, and CPI as most critical via permutation importance.  
- **Grid Search:** Optimizes Random Forest hyperparameters (n_estimators, min_samples_split, etc.).

### 4. Ensemble Construction:

- Selects Random Forest and Extra Trees as best-performing models.  
- Combines them into a MeanEnsemble voting regressor.  
- Trains final pipeline on features:  
  `['Store', 'Dept', 'IsHoliday', 'Size', 'Type', 'Year', 'Week', 'Day']`

### 5. Persistence:

- Saves full pipeline (Preprocessor + Model) to `mean_ensemble_model.joblib`.  
- Generates `mean_ensemble_submission.csv`.

---

## 2. Model Tracking (MLflow)

Register the trained model:

```
python log_model.py
```

- Creates an experiment named **"Walmart Sales Forecasting"**.  
- Logs parameters, WMAE, and model artifact.

---

## 3. Serving the API

Start the FastAPI server:

```
python api.py
```

- The API will start at: **http://127.0.0.1:8000**  
- Swagger Docs: **http://127.0.0.1:8000/docs**

---

## 4. User Dashboard

Launch the Streamlit dashboard:

```
streamlit run dashboard.py
```

- Allows selecting Store ID, Department, Date, Holiday status to get predictions.

---


