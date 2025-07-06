import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("processed_dataset.csv")

# 🔹 Step 1: Auto-Calculate BMI
df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2  # Height in cm converted to meters

# 🔹 Step 2: Blood Pressure Categorization
def categorize_bp(systolic, diastolic):
    if systolic < 90 or diastolic < 60:
        return "Low"
    elif 90 <= systolic <= 120 and 60 <= diastolic <= 80:
        return "Normal"
    else:
        return "High"

df['Blood_Pressure_Category'] = df.apply(lambda x: categorize_bp(x['Blood Pressure Systolic'], x['Blood Pressure Diastolic']), axis=1)

# 🔹 Step 3: Cholesterol Categorization
def categorize_cholesterol(chol):
    if chol < 200:
        return "Low"
    elif 200 <= chol <= 239:
        return "Normal"
    else:
        return "High"

df['Cholesterol_Category'] = df['Total Cholesterol'].apply(categorize_cholesterol)

# 🔹 Step 4: Compute Risk Score (Example Formula)
df['Risk_Score'] = (
    (df['Age'] / 100) + 
    (df['Smoking'] * 2) + 
    (df['Alcohol'] * 1.5) + 
    (df['Exercise'] * -1.5) + 
    (df['Cholesterol_Category'].map({'Low': 0, 'Normal': 1, 'High': 2})) +
    (df['Blood_Pressure_Category'].map({'Low': 0, 'Normal': 1, 'High': 2}))
)

# 🔹 Step 5: Convert Symptoms into Binary Features
symptoms = ["Fever", "Fatigue", "Chest Pain", "Shortness of Breath", "Persistent Cough", "Unexplained Weight Loss"]
for symptom in symptoms:
    df[symptom] = df[symptom].apply(lambda x: 1 if x == "Yes" else 0)

# Save the enhanced dataset
df.to_csv("final_dataset.csv", index=False)

print("✅ Feature Engineering Completed!")
