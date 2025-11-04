import pandas as pd
import mlflow
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.base import BaseEstimator, RegressorMixin
import os

from sales_forecasting_project.utils import MeanEnsemble, BoolToIntImputer

mlflow.set_tracking_uri("file:./sales_forecasting_project/mlruns")

class SalesInput(BaseModel):
    Store: int
    Dept: int
    IsHoliday: bool
    Size: int
    Type: str
    Year: int
    Week: int
    Day: int

MODEL_NAME = "Walmart-Sales-Ensemble"
MODEL_STAGE = "None"

print(f"Loading model '{MODEL_NAME}' version from stage '{MODEL_STAGE}'...")
try:
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load model. {e}")
    model = None


app = FastAPI(
    title="Walmart Sales Forecasting API",
    description="API for predicting sales.",
    version="1.0"
)


@app.post("/predict")
def predict_sales(data: SalesInput):

    if model is None:
        return {"error": "Model is not loaded. Cannot make predictions."}

    try:
        input_data = data.model_dump() 
        
        input_df = pd.DataFrame([input_data])

        input_df = input_df[['Store', 'Dept', 'IsHoliday', 'Size', 'Week', 'Type', 'Year', 'Day']]

        prediction = model.predict(input_df)
        
        forecasted_sales = prediction[0]

        return {
            "forecasted_sales": forecasted_sales,
            "model_used": MODEL_NAME,
            "model_stage": MODEL_STAGE
        }
    
    except Exception as e:
        return {"error": str(e)}

# --- 6. (Optional) Root endpoint ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sales Forecasting API. Go to /docs for details."}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)