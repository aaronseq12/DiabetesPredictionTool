# app.py Flask App
# Author: Aaron Emmanuel Xavier Sequeira
# Description: This script creates a Flask web application to serve the
# trained diabetes prediction model. It provides an API endpoint to
# receive patient data and return a prediction.

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import os

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model and the scaler
try:
    model = joblib.load('models/diabetes_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    print("Error: Model or scaler not found. Please run main.py to train and save them.")
    model = None
    scaler = None

@app.route('/')
def home():
    """Render the home page with the prediction form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_diabetes():
    """
    Handle prediction requests from the web form.
    """
    if not model or not scaler:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

    try:
        # Get data from the POST request
        patient_data = request.get_json(force=True)
        
        # The order of features must match the training data
        feature_values = [
            patient_data['Pregnancies'],
            patient_data['Glucose'],
            patient_data['BloodPressure'],
            patient_data['SkinThickness'],
            patient_data['Insulin'],
            patient_data['BMI'],
            patient_data['DiabetesPedigreeFunction'],
            patient_data['Age']
        ]

        # Convert to a numpy array for scaling
        final_features = np.array(feature_values).reshape(1, -1)
        
        # Scale the features using the loaded scaler
        scaled_features = scaler.transform(final_features)
        
        # Make prediction
        prediction = model.predict(scaled_features)
        prediction_probability = model.predict_proba(scaled_features)
        
        # Get the confidence score
        confidence_score = prediction_probability[0][prediction[0]]
        
        # Return the result as JSON
        return jsonify({
            'prediction': int(prediction[0]),
            'confidence': float(confidence_score)
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 400

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
