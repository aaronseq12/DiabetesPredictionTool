# main.py
# Updated by: Aaron Emmanuel Xavier Sequeira
# Description: This script handles the complete machine learning pipeline:
# 1. Loads and preprocesses the dataset.
# 2. Splits the data into training and testing sets.
# 3. Trains an XGBoost classifier.
# 4. Evaluates the model's performance.
# 5. Saves the trained model and the data scaler for future use.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from preprocessing import preprocess

def main():
    """
    Main function to run the diabetes prediction model training and evaluation.
    """
    # 1. Load and Preprocess Data
    print("Loading and preprocessing data...")
    dataset = pd.read_csv('diabetes.csv')
    # The preprocess function now returns the processed data and the scaler
    processed_data, scaler = preprocess(dataset)
    print("Data preprocessing complete.")
    print("-" * 30)

    # 2. Split Dataset
    print("Splitting the dataset...")
    X = processed_data.loc[:, processed_data.columns != 'Outcome']
    y = processed_data.loc[:, 'Outcome']
    
    # Split data into 80% training and 20% testing
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set shape: {x_train.shape}")
    print(f"Testing set shape: {x_test.shape}")
    print("-" * 30)

    # 3. Train XGBoost Model
    print("Training the XGBoost model...")
    # Initialize the XGBoost classifier with hyperparameters tuned for performance
    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(x_train, y_train)
    print("Model training complete.")
    print("-" * 30)

    # 4. Evaluate the Model
    print("Evaluating the model...")
    # Predictions on the training set
    y_train_pred = model.predict(x_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

    # Predictions on the testing set
    y_test_pred = model.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")
    print("\nClassification Report on Test Data:")
    print(classification_report(y_test, y_test_pred))
    
    # Confusion Matrix
    c_matrix = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(c_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Diabetes', 'Diabetes'], 
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    print("-" * 30)

    # 5. Save the Model and Scaler
    print("Saving the model and scaler...")
    joblib.dump(model, 'diabetes_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model and scaler saved successfully as 'diabetes_model.pkl' and 'scaler.pkl'.")

if __name__ == '__main__':
    main()
