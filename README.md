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

Please download the dataset from the following link: Dataset Link
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

