import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib
import chardet

# Paths
DATA_PATH = "data/dataset.csv"
MODEL_PATH = "models/model.pkl"

# Load dataset
with open(DATA_PATH, 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']
df = pd.read_csv(DATA_PATH, encoding=encoding)

# Encode categorical column
if "smoker" in df.columns:
    le = LabelEncoder()
    df["smoker"] = le.fit_transform(df["smoker"])

# Split features and target
X = df.drop(columns=["target", "patient_id"])
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(model, MODEL_PATH)

print("Model training complete. Model saved to", MODEL_PATH)
