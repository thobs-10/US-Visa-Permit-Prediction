import numpy as np
import pandas as pd
import requests
from faker import Faker

# Generating fake data for the test
fake = Faker()
num_records = 4500

data = {
    "education_of_employee": [fake.random_element(elements=("High School", "Master's", "Bachelor's", "Doctorate")) for _ in range(num_records)],
    "has_job_experience": [fake.random_element(elements=("Y", "N")) for _ in range(num_records)],
    "requires_job_training": [fake.random_element(elements=("Y", "N")) for _ in range(num_records)],
    "full_time_position": [fake.random_element(elements=("Y", "N")) for _ in range(num_records)],
    "continent": [fake.random_element(elements=("Asia", "Europe", "North America", "South America", "Africa")) for _ in range(num_records)],
    "unit_of_wage": [fake.random_element(elements=("Hour", "Year")) for _ in range(num_records)],
    "region_of_employment": [fake.random_element(elements=("Midwest", "South", "West", "Northeast")) for _ in range(num_records)],
    "no_of_employees": np.random.randint(1, 2500, num_records).tolist(),
    "company_age": np.random.randint(1, 25, num_records).tolist(),
    "prevailing_wage": np.random.normal(10000, 500000, num_records).tolist(),
}

df = pd.DataFrame(data)

# Send data to the FastAPI endpoint
url = "http://127.0.0.1:8000/predict"
headers = {"Content-Type": "application/json"}

for i in range(len(df)):
    payload = df.iloc[i].to_dict()
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.json()}")
    else:
        print(response.json())
