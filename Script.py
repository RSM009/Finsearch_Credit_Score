import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Create a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Create a DataFrame
data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
data['credit_score'] = y

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('credit_score', axis=1), data['credit_score'], test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

import xgboost as xgb

# Create and train the XGBoost classifier
model = xgb.XGBClassifier()
model.fit(X_train_scaled, y_train)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

import joblib

# Save the trained model
joblib.dump(model, 'credit_score_model.pkl')
