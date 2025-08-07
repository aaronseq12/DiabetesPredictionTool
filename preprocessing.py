# preprocessing.py
# Updated by: Aaron Emmanuel Xavier Sequeira
# Description: This script provides a function to preprocess the diabetes dataset.
# It handles missing values by imputing them with the mean and standardizes the features.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess(dataset):
    """
    Preprocesses the input diabetes dataset.

    Args:
        dataset (pd.DataFrame): The raw diabetes dataset.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The preprocessed and scaled dataset.
            - StandardScaler: The fitted scaler object used for standardization.
    """
    print("--- Initial Data ---")
    print(dataset.head())
    print("\n--- Statistical Summary (Initial) ---")
    print(dataset.describe())
    
    # Identifying zero values in columns where zero is not a valid value
    cols_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    print("\n--- Replacing 0 values with NaN ---")
    for col in cols_to_impute:
        dataset[col] = dataset[col].replace(0, np.nan)
        
    # Imputing NaN values with the mean of the respective column
    print("--- Imputing NaN values with mean ---")
    for col in cols_to_impute:
        dataset[col] = dataset[col].fillna(dataset[col].mean())

    print("\n--- Data after imputation ---")
    print(dataset.head())
    print("\n--- Statistical Summary (After Imputation) ---")
    print(dataset.describe())

    # Data Standardization
    print("\n--- Standardizing Data ---")
    X = dataset.drop('Outcome', axis=1)
    y = dataset['Outcome']
    
    # Initialize the StandardScaler
    scaler = StandardScaler()
    
    # Fit and transform the features
    X_scaled = scaler.fit_transform(X)
    
    # Create a new DataFrame with the scaled features
    dataset_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Add the 'Outcome' column back
    dataset_scaled['Outcome'] = y.values
    
    print("\n--- Standardized Dataset Summary ---")
    print(dataset_scaled.describe().round(2))
    
    return dataset_scaled, scaler

if __name__ == '__main__':
    # Example usage:
    df = pd.read_csv('diabetes.csv')
    processed_df, fitted_scaler = preprocess(df)
    print("\n--- Preprocessing Complete ---")
    print("Processed DataFrame head:")
    print(processed_df.head())
