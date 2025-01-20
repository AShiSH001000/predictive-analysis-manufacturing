# Predictive Analysis for Manufacturing Operations

## Project Overview

This project implements a RESTful API using FastAPI to perform predictive analysis for manufacturing operations. The main functionality is to predict machine downtime based on operational data such as temperature and run time.

The project uses a Random Forest Classifier model trained on a dataset to predict whether a machine will experience downtime based on provided features.

## Requirements
- Python 3.8+
- FastAPI
- scikit-learn
- pandas

## Setup
1. Clone the repository.
   ```bash
   git clone https://github.com/AShiSH001000/predictive-analysis-manufacturing.git
2. Install dependencies:
   ```bash
   pip install -r requirements.txt


3. Navigate into the project directory:
   ```bash
   cd predictive-analysis-manufacturing


### Start the FastAPI Server

1. Once the dependencies are installed, run the FastAPI server:
   ```bash
   python -m uvicorn main:app --reload


The API will now be running locally at [http://127.0.0.1:8000](http://127.0.0.1:8000).

## API Endpoints

   1. ### Train Model (`/train`)
   
   - **Method**: `POST`
   - **Description**: Upload a CSV file containing manufacturing data to train the Random Forest model.
   
   #### Request Body (form-data):
   
   | Key     | Value                        |
   | ------- | ---------------------------- |
   | file    | Choose the file to upload (e.g., `sample_data.csv`) |
   
   #### Response:
      ```json
            {
              "accuracy": 0.85,
              "f1_score": 0.88,
              "best_params": {
                "criterion": "gini",
                "max_depth": 20,
                "min_samples_leaf": 2,
                "min_samples_split": 5,
                "n_estimators": 100
              }
            }

2. ### Test /predict Endpoint:
      - **Method**: `POST`
      - **Description**: Provide operational data (temperature and run time) to predict whether the machine will experience downtime.
   #### Request Body (JSON):
      ```json
      {
        "Temperature": 80,
        "Run_Time": 120
      }
#### Response:
   ```json
   {
  "Downtime": "Yes",
  "Confidence": 0.85
}

   
   
