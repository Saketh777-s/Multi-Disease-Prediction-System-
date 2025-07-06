from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load Model and Preprocessing Tools
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")
encoder = joblib.load("model/encoder.pkl")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Extract User Input
        categorical_features = [
            data['gender'], data['smoking'], data['alcohol'], 
            data['exercise'], data['diet']
        ]
        numerical_features = [
            data['age'], data['height'], data['weight'], data['bmi'],
            data['cholesterol'], data['blood_pressure'], data['blood_sugar']
        ]

        # Encode Categorical Features
        encoded_categorical = encoder.transform([categorical_features]).toarray()

        # Convert to NumPy Array and Scale
        features = np.hstack((numerical_features, encoded_categorical))
        features = scaler.transform([features])

        # Make Prediction
        prediction = model.predict(features)[0]

        # Return JSON Response
        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
