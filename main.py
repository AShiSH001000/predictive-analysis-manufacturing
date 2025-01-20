from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os
import json

app = FastAPI()
MODEL_PATH = "models/model.pkl"

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    file_location = f"data/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(file.file.read())

    return {"message": "File uploaded successfully.", "file_path": file_location}
from fastapi import UploadFile, File
import io
import pandas as pd
from fastapi import HTTPException
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

@app.post("/train")
async def train_with_file(file: UploadFile = File(...)):
    try:
        # Read the uploaded file into a DataFrame
        contents = await file.read()
        data = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading the file: {str(e)}")

    # Validate the data format
    if not {"Temperature", "Run_Time", "Downtime_Flag"}.issubset(data.columns):
        raise HTTPException(status_code=400, detail="Invalid data format. Ensure the columns: Temperature, Run_Time, Downtime_Flag.")

    # Prepare features and target
    X = data[["Temperature", "Run_Time"]]
    y = data["Downtime_Flag"]

    # Balance the dataset using SMOTE
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the Random Forest model
    rf = RandomForestClassifier(random_state=42)

    # Define the hyperparameter grid for tuning
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"]
    }

    # Perform Grid Search for hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        verbose=2,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Get the best model from grid search
    best_model = grid_search.best_estimator_

    # Evaluate the model on the test set
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save the best model and scaler
    joblib.dump(best_model, "models/random_forest_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    # Return the evaluation metrics
    return {
        "accuracy": report["accuracy"],
        "f1_score": report["1"]["f1-score"],  # F1-score for the positive class
        "best_params": grid_search.best_params_
    }


@app.post("/predict")
async def predict(input_data: dict):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=400, detail="Model not trained. Please train the model first.")

    # Load the model
    model = joblib.load(MODEL_PATH)

    # Parse input data
    try:
        temperature = input_data["Temperature"]
        run_time = input_data["Run_Time"]
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid input format. Required keys: Temperature, Run_Time.")

    # Make prediction
    prediction = model.predict([[temperature, run_time]])
    confidence = max(model.predict_proba([[temperature, run_time]])[0])

    downtime = "Yes" if prediction[0] == 1 else "No"
    return {"Downtime": downtime, "Confidence": round(confidence, 2)}
@app.get("/")
def read_root():
    return {"message": "Welcome to the Predictive Analysis API. Visit /docs for Swagger UI."}
