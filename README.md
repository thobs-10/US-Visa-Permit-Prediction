# US Visa Prediction Project
## Overview
The Immigration and Nationality Act (INA) of the US permits foreign workers to come to the United States to work on either a temporary or permanent basis. The act also protects US workers against adverse impacts in the workplace and maintains requirements when hiring foreign workers to fill workforce shortages. The immigration programs are administered by the Office of Foreign Labor Certification (OFLC).

## Problem Statement
OFLC receives job certification applications from employers seeking to bring foreign workers into the United States and grants certifications. Last year, the count of employees was huge, so OFLC needs machine learning models to shortlist visa applicants based on their previous data.

In this project, we aim to build a classification model to predict whether a visa will be approved or not based on the given dataset. This model can be used to recommend suitable profiles for applicants whose visas should be certified or denied based on certain criteria that influence the decision.

## Project Lifecycle
Understanding the Problem Statement

Data Collection

Exploratory Data Analysis

Data Pre-Processing and feature engineering

Model Training(Experiment tracking)

Model Evaluation

Model Registry

Model Deployment

Model Monitoring

### Data

The dataset used in this project contains historical data of visa applications. Each record represents an application with various features influencing the visa approval decision.

## How to Run the Project

Dataset Link:

Please download the dataset from the following link: [Dataset Link](https://www.kaggle.com/datasets/moro23/easyvisa-dataset)

Clone the Repository:
``
git clone <repository-url>

cd <repository-directory>
``
Create a Virtual Environment:

Install virtualenv (if not already installed):
``
pip install virtualenv
``

``
python -m venv us-visa-permit-env
``

Activate the Virtual Environment:

``
.\us-visa-permit-env\Scripts\activate
``
for Windows

``
source us-visa-permit-env/bin/activate
``
for macOS

Install the Required Packages:

``
pip install -r requirements.txt
``

Download and Prepare the Dataset:

Download the dataset and create the following directory structure:

``
<repository-directory>/

└── data/

    └── raw_data/

        └── Visadataset.csv
``

Create a Free Account on Comet.ml:

Sign up at Comet.ml. This is used for experiment tracking and model registry

Create a project, go to settings, and get your API key, project name, and workspace name.

Place these details in a .env file in the root directory:

``
API_KEY="your-api-key"

PROJECT_NAME="project-name"

WORKSPACE="workspace-name"
``


Running the Project

Once everything is set up, follow these steps to run the project:

Run the Pipelines:

There is a script named run_pipeline.py in the parent folder. Inside the main function of this script, you will see three pipelines:

```
run_data_ingestion_pipeline()
run_feature_engineering_pipeline()
run_training_model_pipeline()
```

To run each pipeline, comment out the other pipelines and execute the script( first comment out the feature engineering and nodel training pipeline, then run data ingestion pipeline, after that commeent out the data ingestion pipeline and model training pipeline and run the feature engineering pipeline and so forth):

```
python run_pipeline.py
```

This command will trigger the specific pipeline and run all necessary functions inside the pipeline's component script. Each pipeline is made up of components, and those components contain the function steps for the pipeline.

Understanding the Pipeline Implementation:

To see how the pipelines are implemented, go to the pipelines folder.

To see how the components of the pipelines are created, go to the components folder and select either data_ingestion, feature_engineering, or model_training.

Running the FastAPI Endpoint and Request Script:

Start the FastAPI endpoint:

```
uvicorn app:app --reload
```
(Run this command in a separate terminal)

Send a dummy request to the model:

```
python request.py
```

(Run this command in another separate terminal)


Every prediction will be saved in a log file.

Run the Batch Monitoring Script:

To see how the predictions are distributed, run the batch monitoring script:

```
python batch_monitoring.py
```
