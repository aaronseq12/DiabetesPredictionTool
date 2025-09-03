# data-analysis.py
# Author: Aaron Emmanuel Xavier Sequeira
# Description: This script performs exploratory data analysis (EDA) on the
# diabetes dataset. It generates histograms, density plots, and a correlation
# matrix to visualize feature distributions and their relationships with the outcome.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_exploratory_data_analysis():
    """
    Performs and visualizes exploratory data analysis on the diabetes dataset.
    """
    # Load the dataset
    try:
        diabetes_dataset = pd.read_csv('diabetes.csv')
    except FileNotFoundError:
        print("Error: 'diabetes.csv' not found. Please ensure the dataset is in the correct directory.")
        return

    print("--- First 5 Rows of the Dataset ---")
    print(diabetes_dataset.head())
    print("\n" + "-"*30)

    # --- Histograms for all features ---
    print("Generating histograms for each feature...")
    diabetes_dataset.hist(bins=20, figsize=(20, 15), color='skyblue', edgecolor='black')
    plt.suptitle('Histograms of All Features', size=20, y=0.93)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()
    print("-" * 30)

    # --- Density Plots for feature comparison ---
    print("Generating density plots to compare features by outcome...")
    # Create a grid of subplots
    feature_columns = diabetes_dataset.columns[:-1]
    number_of_features = len(feature_columns)
    grid_columns = 3
    grid_rows = (number_of_features + grid_columns - 1) // grid_columns
    
    plt.figure(figsize=(18, 5 * grid_rows))
    plt.suptitle('Density Plots of Features by Diabetes Outcome', size=22, y=0.95)

    for index, column in enumerate(feature_columns):
        axis = plt.subplot(grid_rows, grid_columns, index + 1)
        sns.kdeplot(diabetes_dataset.loc[diabetes_dataset['Outcome'] == 0, column], 
                    fill=True, 
                    color="green", 
                    label="No Diabetes", 
                    ax=axis)
        sns.kdeplot(diabetes_dataset.loc[diabetes_dataset['Outcome'] == 1, column], 
                    fill=True, 
                    color="red", 
                    label="Diabetes", 
                    ax=axis)
        axis.set_title(f'Distribution of {column}', fontweight='bold')
        axis.set_xlabel('')
        axis.set_ylabel('Density')
        axis.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()
    print("-" * 30)
    
    # --- Correlation Matrix ---
    print("Generating a correlation matrix heatmap...")
    plt.figure(figsize=(12, 10))
    correlation_matrix = diabetes_dataset.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Features', size=18)
    plt.show()


if __name__ == '__main__':
    perform_exploratory_data_analysis()
