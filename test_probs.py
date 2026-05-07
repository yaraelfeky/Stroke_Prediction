import pandas as pd
import numpy as np
from model import train_model, predict_patient

# Load data
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
train_model(df)

# Test cases
test_cases = [
    # High risk
    {"gender": "Male", "age": 80, "hypertension": 1, "heart_disease": 1, "ever_married": "Yes", 
     "work_type": "Private", "Residence_type": "Urban", "avg_glucose_level": 250, "bmi": 35, "smoking_status": "smokes"},
    # Medium risk
    {"gender": "Female", "age": 50, "hypertension": 0, "heart_disease": 0, "ever_married": "Yes", 
     "work_type": "Private", "Residence_type": "Rural", "avg_glucose_level": 120, "bmi": 28, "smoking_status": "formerly smoked"},
    # Low risk
    {"gender": "Female", "age": 20, "hypertension": 0, "heart_disease": 0, "ever_married": "No", 
     "work_type": "children", "Residence_type": "Rural", "avg_glucose_level": 80, "bmi": 20, "smoking_status": "never smoked"}
]

print("\n--- PROBABILITY TEST ---")
for i, case in enumerate(test_cases):
    pred, prob, thresh = predict_patient(case)
    print(f"Case {i+1}: Prob={prob:.4f}, Pred={pred} (Threshold={thresh:.2f})")
