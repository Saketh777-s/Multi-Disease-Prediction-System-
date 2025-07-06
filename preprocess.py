import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib

# Load dataset
df = pd.read_csv("dataset.csv")

# 🔹 Step 1: Handle Missing Values
num_imputer = SimpleImputer(strategy="mean")  # Fill numerical columns with mean
cat_imputer = SimpleImputer(strategy="most_frequent")  # Fill categorical columns with most frequent values

num_cols = ["Age", "Height", "Weight", "BMI", "Blood Pressure", "Blood Sugar", "Cholesterol"]
cat_cols = ["Gender", "Smoking", "Alcohol", "Exercise", "Diet"]

# Apply imputers
df[num_cols] = num_imputer.fit_transform(df[num_cols])
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# 🔹 Step 2: Encode Categorical Variables
encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
encoded_cat = encoder.fit_transform(df[cat_cols])

# Convert encoded data to DataFrame
encoded_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(cat_cols))

# Drop original categorical columns and add encoded features
df = df.drop(columns=cat_cols)
df = pd.concat([df, encoded_df], axis=1)

# 🔹 Step 3: Scale Numerical Features
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# 🔹 Step 4: Save Preprocessing Objects
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(encoder, "model/encoder.pkl")

# Save Preprocessed Data
df.to_csv("processed_dataset.csv", index=False)

print("✅ Preprocessing Completed!")
