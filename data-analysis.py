# data-analysis.py
# Updated by: Aaron Emmanuel Xavier Sequeira
# Description: This script performs exploratory data analysis (EDA) on the
# diabetes dataset. It generates histograms and density plots to visualize
# feature distributions and relationships with the outcome.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_data():
    """
    Performs and visualizes exploratory data analysis on the diabetes dataset.
    """
    # Load the dataset
    try:
        dataset = pd.read_csv('diabetes.csv')
    except FileNotFoundError:
        print("Error: 'diabetes.csv' not found. Please make sure the dataset is in the correct directory.")
        return

    print("--- First 5 Rows of the Dataset ---")
    print(dataset.head())
    print("\n" + "-"*30)

    # --- Histograms for all features ---
    print("Generating histograms for each feature...")
    dataset.hist(bins=20, figsize=(20, 15), color='skyblue', edgecolor='black')
    plt.suptitle('Histograms of All Features', size=20, y=0.93)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()
    print("-" * 30)

    # --- Density Plots for feature comparison ---
    print("Generating density plots to compare features by outcome...")
    # Create a grid of subplots
    num_features = len(dataset.columns) - 1
    cols = 3
    rows = (num_features + cols - 1) // cols
    
    plt.figure(figsize=(18, 5 * rows))
    plt.suptitle('Density Plots of Features by Diabetes Outcome', size=22, y=0.95)

    for i, col in enumerate(dataset.columns[:-1]): # Exclude 'Outcome' column
        ax = plt.subplot(rows, cols, i + 1)
        sns.kdeplot(dataset.loc[dataset['Outcome'] == 0, col], 
                    fill=True, 
                    color="green", 
                    label="No Diabetes", 
                    ax=ax)
        sns.kdeplot(dataset.loc[dataset['Outcome'] == 1, col], 
                    fill=True, 
                    color="red", 
                    label="Diabetes", 
                    ax=ax)
        ax.set_title(f'Distribution of {col}', fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Density')
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()
    print("-" * 30)
    
    # --- Correlation Matrix ---
    print("Generating a correlation matrix heatmap...")
    plt.figure(figsize=(12, 10))
    correlation_matrix = dataset.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Features', size=18)
    plt.show()


if __name__ == '__main__':
    analyze_data()
