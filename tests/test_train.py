import os
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import chardet

DATA_PATH = "data/dataset.csv"

@pytest.fixture
def load_data():
    # Load dataset with encoding detection
    with open(DATA_PATH, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']
    df = pd.read_csv(DATA_PATH, encoding=encoding)
    # Encode categorical
    if "smoker" in df.columns:
        le = LabelEncoder()
        df["smoker"] = le.fit_transform(df["smoker"])
    return df

def test_data_loading(load_data):
    """Test if dataset loads correctly"""
    df = load_data
    assert df.shape[0] > 0, "Dataset is empty"
    assert "target" in df.columns, "Target column missing"

def test_model_training(load_data):
    """Test if model can be trained"""
    df = load_data
    X = df.drop(columns=["target", "patient_id"])
    y = df["target"]
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    # Check if coefficients exist
    assert hasattr(model, "coef_"), "Model not trained"

def test_shape_validation(load_data):
    """Test feature-target shapes"""
    df = load_data
    X = df.drop(columns=["target", "patient_id"])
    y = df["target"]
    assert X.shape[0] == y.shape[0], "Mismatch between X and y rows"
