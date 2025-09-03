# preprocessing.py
# Author: Aaron Emmanuel Xavier Sequeira
# Description: This script provides a function to preprocess the diabetes dataset.
# It handles missing values by imputing them with the mean and standardizes the features.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_diabetes_data(diabetes_dataset):
    """
    Preprocesses the input diabetes dataset.

    Args:
        diabetes_dataset (pd.DataFrame): The raw diabetes dataset.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The preprocessed and scaled dataset.
            - StandardScaler: The fitted scaler object used for standardization.
    """
    print("--- Initial Data ---")
    print(diabetes_dataset.head())
    print("\n--- Statistical Summary (Initial) ---")
    print(diabetes_dataset.describe())
    
    # Identifying zero values in columns where zero is not a valid value
    columns_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    print("\n--- Replacing 0 values with NaN ---")
    for column in columns_to_impute:
        diabetes_dataset[column] = diabetes_dataset[column].replace(0, np.nan)
        
    # Imputing NaN values with the mean of the respective column
    print("--- Imputing NaN values with mean ---")
    for column in columns_to_impute:
        diabetes_dataset[column] = diabetes_dataset[column].fillna(diabetes_dataset[column].mean())

    print("\n--- Data after imputation ---")
    print(diabetes_dataset.head())
    print("\n--- Statistical Summary (After Imputation) ---")
    print(diabetes_dataset.describe())

    # Data Standardization
    print("\n--- Standardizing Data ---")
    features = diabetes_dataset.drop('Outcome', axis=1)
    target = diabetes_dataset['Outcome']
    
    # Initialize the StandardScaler
    scaler = StandardScaler()
    
    # Fit and transform the features
    scaled_features = scaler.fit_transform(features)
    
    # Create a new DataFrame with the scaled features
    scaled_dataset = pd.DataFrame(scaled_features, columns=features.columns)
    
    # Add the 'Outcome' column back
    scaled_dataset['Outcome'] = target.values
    
    print("\n--- Standardized Dataset Summary ---")
    print(scaled_dataset.describe().round(2))
    
    return scaled_dataset, scaler

if __name__ == '__main__':
    # Example usage:
    try:
        raw_diabetes_df = pd.read_csv('diabetes.csv')
        processed_df, fitted_scaler = preprocess_diabetes_data(raw_diabetes_df)
        print("\n--- Preprocessing Complete ---")
        print("Processed DataFrame head:")
        print(processed_df.head())
    except FileNotFoundError:
        print("Error: 'diabetes.csv' not found. Please ensure the dataset is in the correct directory.")
