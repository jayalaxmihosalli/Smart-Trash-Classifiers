# train_model.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("smart_trash_classifier_dataset.csv")

# Encode categorical column
material_encoder = LabelEncoder()
data["material_type"] = material_encoder.fit_transform(data["material_type"])

# Encode target variable
target_encoder = LabelEncoder()
data["recommended_bin"] = target_encoder.fit_transform(data["recommended_bin"])

# Features and target
X = data.drop("recommended_bin", axis=1)
y = data["recommended_bin"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and encoders
joblib.dump(model, "trash_classifier_model.pkl")
joblib.dump(material_encoder, "material_encoder.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")

print("Model and encoders saved successfully!")