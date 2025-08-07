# app.py Flask App
# Created by: Aaron Emmanuel Xavier Sequeira
# Description: This script creates a Flask web application to serve the
# trained diabetes prediction model. It provides an API endpoint to
# receive patient data and return a prediction.

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model and the scaler
try:
    model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
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
def predict():
    """
    Handle prediction requests from the web form.
    """
    if not model or not scaler:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

    try:
        # Get data from the POST request
        data = request.get_json(force=True)
        
        # The order of features must match the training data
        features = [
            data['Pregnancies'],
            data['Glucose'],
            data['BloodPressure'],
            data['SkinThickness'],
            data['Insulin'],
            data['BMI'],
            data['DiabetesPedigreeFunction'],
            data['Age']
        ]

        # Convert to a numpy array for scaling
        final_features = np.array(features).reshape(1, -1)
        
        # Scale the features using the loaded scaler
        scaled_features = scaler.transform(final_features)
        
        # Make prediction
        prediction = model.predict(scaled_features)
        prediction_proba = model.predict_proba(scaled_features)
        
        # Get the confidence score
        confidence = prediction_proba[0][prediction[0]]
        
        # Return the result as JSON
        return jsonify({
            'prediction': int(prediction[0]),
            'confidence': float(confidence)
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 400

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
