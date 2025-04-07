import os
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel
from sklearn.pipeline import Pipeline

app = FastAPI()


def get_registered_model() -> Pipeline | None:
    try:
        output_path = os.path.abspath("src/models/tuning_artifacts/model_pipeline")
        model_pipeline_path = os.path.join(output_path, "model_pipeline.pkl")
        model_pipeline = joblib.load(model_pipeline_path)
        return model_pipeline
    except FileNotFoundError as e:
        logger.error(f"Error loading preprocessor: {e}")
        return None


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


def generate_input_dataframe(input_data: ModelInput) -> pd.DataFrame:
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
        "company_age": [input_data.company_age],
    }
    input_df = pd.DataFrame(input_dict)
    return input_df


@app.post("/predict")
async def prediction(input_data: ModelInput):
    try:
        input_df = generate_input_dataframe(input_data)
        if np.isnan(input_df).any():
            logger.error("NaN values found in the transformed data")
            raise HTTPException(status_code=400, detail="NaN values found in the transformed data")

        model_pipeline = get_registered_model()
        pred = model_pipeline.predict(input_df)
        logger.info(f"Model prediction: {pred[0].item()}")

        return {"prediction": pred[0].item()}
    except ConnectionError as e:
        logger.error(f"Prediction error: {e}")


@app.get("/")
async def read_root():
    return {"message": "Welcome to the model prediction API"}


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
