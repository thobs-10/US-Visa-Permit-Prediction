import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os
import pandas as pd
from datetime import datetime
from typing import Literal

app = FastAPI()

# Set up logging
logging.basicConfig(filename='predictions.log', level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model and preprocessor when the app starts
def get_preprocessor():
    try:
        output_path = os.path.abspath('src/models/best_model')
        preprocessor_path = os.path.join(output_path, 'preprocessor.pkl')
        preprocessor = joblib.load(preprocessor_path)
    except Exception as e:
        logger.error(f"Error loading preprocessor: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading preprocessor: {str(e)}")
    return preprocessor

def get_registered_model():
    try:
        output_path = os.path.abspath('src/models/best_model')
        latest_file = os.path.join(output_path,'Decision_Tree.pkl')
        model = joblib.load(latest_file)
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

try:
    model = get_registered_model()
    preprocessor = get_preprocessor()
except Exception as e:
    raise RuntimeError(f"Error loading model: {str(e)}")

class ModelInput(BaseModel):
    education_of_employee: Literal["High School", "Master's", "Bachelor's", "Doctorate"]
    has_job_experience: Literal["Y", "N"]
    requires_job_training: Literal["Y", "N"]
    full_time_position: Literal["Y", "N"]
    continent: Literal["Asia", "Europe", "North America", "South America", "Africa"]
    unit_of_wage: Literal["Hour", "Year"]
    region_of_employment: Literal["Midwest", "South", "West", "Northeast"]
    no_of_employees: int
    company_age: int
    prevailing_wage: float


@app.post("/predict")
async def prediction(input_data: ModelInput):
    try:
        input_dict = {
            "continent": [input_data.continent],
            "education_of_employee": [input_data.education_of_employee],
            "has_job_experience": [input_data.has_job_experience],
            "requires_job_training": [input_data.requires_job_training],
            "no_of_employees": [input_data.no_of_employees],
            "region_of_employment": [input_data.region_of_employment],
            "prevailing_wage": [input_data.prevailing_wage],
            "unit_of_wage": [input_data.unit_of_wage],
            "full_time_position": [input_data.full_time_position],
            "company_age": [input_data.company_age]
        }

        input_df = pd.DataFrame(input_dict)

        # logger.info(f"Input features: {input_dict}")

        X = preprocessor.transform(input_df)

        if np.isnan(X).any():
            logger.error("NaN values found in the transformed data")
            raise HTTPException(status_code=400, detail="NaN values found in the transformed data")

        prediction = model.predict(X)
        logger.info(f"Model prediction: {prediction[0].item()}")


        return {"prediction": prediction[0].item()}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the model prediction API"}



