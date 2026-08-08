# Trains the SVC and saves model/scaler

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load dataset
df = pd.read_csv("cancer.csv")

# Remove unnecessary columns
df.drop(columns=["Unnamed: 32", "id"], inplace=True)

# Convert diagnosis
df["diagnosis"] = df["diagnosis"].map({"B": 0,"M": 1})

# Separate X and y
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

# Train-test split  
X_train, X_test, y_train, y_test = train_test_split( X , y , test_size=0.2 , random_state=42 )

# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVC
model = SVC(kernel='rbf')

model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved successfully!")
print("Scaler saved successfully!")