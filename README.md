# Walmart Recruiting - Store Sales Forecasting

This repository contains the codebase for the NHA-012 Sales Forecasting project, including data processing, model training, MLflow logging, an API service, and a Streamlit dashboard.

## 🚀 Getting Started
Follow the steps below to set up the project and run all components.

## 1. Clone the Repository
```
git clone https://github.com/nhahub/NHA-012.git
cd NHA-012
```

## 2. Create and Activate a Virtual Environment
```
python -m venv .venv
```

### Activate the environment
**Windows:**
```
.venv/Scripts/Activate
```
**macOS / Linux:**
```
source .venv/bin/activate
```

## 3. Install the Project
Install in editable mode:
```
pip install -e .
```

## 4. Train the Model
Run the modeling notebook to generate the model. Ensure the notebook kernel uses the `.venv` environment.

## 5. Log the Model to MLflow
```
python ./sales_forecasting_project/log_model.py
```

## 6. Run the API Server
Start the FastAPI server with Uvicorn:
```
uvicorn sales_forecasting_project.api:app --reload
```
API will be available at:
```
http://127.0.0.1:8000
```

## 7. Launch the Streamlit Dashboard
```
streamlit run ./sales_forecasting_project/dashboard.py
```

