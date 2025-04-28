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

---

## **Project Criteria and Achievements**  

| **Criteria**                          | **Status** |
| ------------------------------------- | ---------- |
| Problem description                   | ✅          |
| Exploratory Data Analysis (EDA)       | ✅          |
| Model Training & Experiment tracking  | ✅          |
| ML Pipeline Orchestration             | ✅          |
| Feature Store                         | ✅          |
| Model Deployment(CI/CD)               | ✅          |
| Code Quality Checks                   | ✅          |
| Testing Suites                        | ✅          |
| Reproducibility                       | ✅          |
| Dependency and Environment Management | ✅          |
| Containerization                      | ✅          |
| Cloud Deployment(Dockerhub)           | ✅          |

---

### Data

The dataset used in this project contains historical data of visa applications. Each record represents an application with various features influencing the visa approval decision.

## How to Run the Project

Dataset Link:

Please download the dataset from the following link: [Dataset Link](https://www.kaggle.com/datasets/moro23/easyvisa-dataset)

Clone the Repository:
``
git clone [repo link](https://github.com/thobs-10/US-Visa-Permit-Prediction)

cd `mlops-zoomcamp-project-2024/US-Visa-Permit-Prediction`

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

mlops-zoomcamp-project-2024/US-Visa-Permit-Prediction/

└── data/

    └── raw_data/

        └── Visadataset.csv
``

For experiment tracking I are using MLFlow, you need to download it in order to proceed. [mlflow](https://mlflow.org/docs/latest/index.html)

For orchestrating ML pipelines I use ZenML which captures every metadata, data versioning in each pipeline step and caching of certain steps for increasing efficiency. [zenml](https://docs.zenml.io/getting-started/installation)

For storing our features and versioning which features are working best for our model I use Feast. [feast](https://docs.feast.dev/getting-started/concepts)



#### Running the Project

Once everything is set up, follow these steps to run the project:

I opted to use bash script as my task runner for executing the system end to end since it gave me more flexibility in terms of specifying how I want each component to be run and what arguments are needed if there are any. To run the system you can just specify these commands in your terminal:

``

    ./run.sh install_package

    ./run.sh run_pre_commit

    ./run.sh run_pipelines

``

Run the Pipelines:

There is a script named run_pipeline.py in the parent folder. Inside the main function of this script, you will see three pipelines:

```
run_data_ingestion_pipeline()
run_feature_engineering_pipeline()
run_training_model_pipeline()
```

To run each pipeline, comment out the other pipelines and execute the script( first comment out the feature engineering and model training pipeline, then run data ingestion pipeline, after that comment out the data ingestion pipeline and model training pipeline and run the feature engineering pipeline and so forth). Even much more better you can run each pipeline by going to the `src/pipeline` folder to get each pipeline and run it. To run all pipelines without making use of the bash script you can run:

```
python run_pipeline.py
```

This command will trigger the specific pipeline and run all necessary functions inside the pipeline's component script. Each pipeline is made up of components, and those components contain the function steps for the pipeline. The pipelines will execute and create a model that will later be served using FastAPI and containerized using Docker Containers. Then the container is later deployed to Dockerhub using a CI/CD pipeline that caters for these platforms: `linux/amd64, linux/arm64, linux/x86_64`.

To see how the pipelines are implemented, go to the pipelines folder.

To see how the components of the pipelines are created, go to the components folder and select either data_ingestion, feature_engineering, or model_training.

Running the FastAPI Endpoint and Request Script:

Start the FastAPI endpoint:

```
uvicorn app:app --reload --host=0.0.0.0  --port=8000
```
(Run this command in a separate terminal)

#### Outdated features:
- Sending requests to the fastAPI server
- Capturing predictions made by the model for monitoring dashboard for later incidents such as model drift, feature drift, model degredation etc.
- Monitoring script using evidently AI.

### WIP features:
- Bash script to schedule the task or the pipelines to run on a cadence and send data in batches.
- Automated A/B testing of the models, at which point do I want to change the current model with a newly up to date trained one.
- Deployment strategies following the A/B test, is it canary deployment, shadow deploys etc?
- Batch monitoring by capturing the model preictions and storing them eather on Premotheus and integrating that with Grafana for monitoring dashboards.
Send a dummy request to the model:

In summary this is just my hobby project that I wanted to try out and learn from it different things and just to make use of the best practices in a side project. As much as this expresses my passion and love for machine learning, MLOps and programming, it also serves as my escape plan after working hours on work projects and grounds me back to something I love doing.
