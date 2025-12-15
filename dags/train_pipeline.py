from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import chardet
import os
import logging

DATA_PATH = "/opt/airflow/data/dataset.csv"
MODEL_PATH = "/opt/airflow/models/model.pkl"

def load_data():
    logging.info("Loading data from %s", DATA_PATH)
    # Load dataset with encoding detection
    with open(DATA_PATH, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']
    df = pd.read_csv(DATA_PATH, encoding=encoding)
    logging.info("Data loaded successfully. Shape: %s", df.shape)
    # Encode categorical
    if "smoker" in df.columns:
        le = LabelEncoder()
        df["smoker"] = le.fit_transform(df["smoker"])
        logging.info("Categorical encoding applied")
    return df

def train_model(df):
    logging.info("Training model")
    # Split features and target
    X = df.drop(columns=["target", "patient_id"])
    y = df["target"]
    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    logging.info("Model trained successfully")
    return model

def train_and_save_model():
    df = load_data()
    model = train_model(df)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logging.info("Model saved to %s", MODEL_PATH)
    print("Model saved successfully!")

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 1, 1)
}

with DAG('train_pipeline', default_args=default_args, schedule_interval=None, catchup=False) as dag:
    train_task = PythonOperator(
        task_id='train_model_task',
        python_callable=train_and_save_model
    )
