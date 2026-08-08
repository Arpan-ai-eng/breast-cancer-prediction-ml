# Checks that model/scaler load correctly

import joblib

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

print("Model loaded successfully!")
print("Scaler loaded successfully!")

print(type(model))
print(type(scaler))