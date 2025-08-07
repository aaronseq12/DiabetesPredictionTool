# Diabetes Prediction System

## About Me

My name is Aaron Sequeira, I am currently pursuing my Btech in Information Technology from Manipal University Jaipur, where I am building a solid foundation in various facets of the field. My keen interest spans several domains, including Full stack web development, Cloud Computing with a focus on AWS, Data Analytics and Machine Learning.

This project was initiated as part of my coursework in data science, where I was assigned to complete a research paper on a diabetes prediction model. It has since evolved from an initial analysis into a comprehensive machine learning application.

### Problem Statement

Diabetes is a chronic (long-lasting) health condition that affects how your body turns food into energy. Your body breaks down most of the food you eat into sugar (glucose) and releases it into your bloodstream. When your blood sugar goes up, it signals your pancreas to release insulin. Early detection is crucial for managing the disease and preventing complications.

### Initial Idea & Model

The initial goal was to build an application that could predict the onset of diabetes by indicating the most relevant factors. The first iteration used a multi-layer perceptron (MLP) model trained on the [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database).

---

## Initial Data Analysis & Pre-Processing

### Histograms
![Pima Indians Diabetes Database Histograms](https://i.imgur.com/htXtzS1.png)
*Note: 'outcome' refers to whether an individual does, or does not, have diabetes.*

#### Insights
* Variables are on different scales and therefore must be standardized.
* The majority of data has been collected from individuals between 20 and 30 years of age.
* `BMI`, `Blood Pressure`, and `Glucose` are normally distributed, as expected from population statistics.
* It is impossible for `BMI`, `Blood Pressure`, and `Glucose` to have a value of zero, indicating missing or incomplete data.
* The dataset suggests that 35% of the population has diabetes, which is not representative of the global population (estimated at 8.5% by WHO) and highlights the dataset's specific nature.

### Density Plots
![Pima Indians Diabetes Database Density Plots](https://i.imgur.com/jmuAZt0.png)

#### Insights
* `Glucose`, `BMI`, and `Age` appear to be the strongest predicting values for those with diabetes.
* `Blood Pressure` and `Skin Thickness` do not appear to have a significant correlation with the outcome.

### Handling Missing Values
A significant number of entries for `Insulin` and `Skin Thickness` were missing (represented as 0). These were replaced with the mean of their respective columns.

![Pima Indians Diabetes Database No. of Missing Values](https://i.imgur.com/Q7meZol.png)

---

## The Upgraded Solution: XGBoost and an Interactive UI

After the initial analysis, the project was upgraded to be more robust, accurate, and user-friendly.

The new solution is a multi-faceted application that includes:
1.  **Refined Data Preprocessing:** A more robust pipeline to handle missing values and standardize the data for optimal model performance.
2.  **Advanced Machine Learning Model:** An **XGBoost (Extreme Gradient Boosting)** classifier, chosen for its high accuracy and performance, replaced the initial MLP model.
3.  **Interactive Web UI:** A user-friendly web application built with **Flask** that allows users to input patient data and receive an instant prediction along with a confidence score.

## Project Structure


.
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── app.py
├── data-analysis.py
├── main.py
├── preprocessing.py
├── diabetes.csv
├── requirements.txt
└── README.md


## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/Diabetes-Prediction-System.git](https://github.com/your-username/Diabetes-Prediction-System.git)
    cd Diabetes-Prediction-System
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Train the model:**
    To train the new XGBoost model, run the `main.py` script. This will evaluate the model and save the trained model (`diabetes_model.pkl`) and the data scaler (`scaler.pkl`) to disk.
    ```bash
    python main.py
    ```

2.  **Run the web application:**
    Start the Flask server by running `app.py`.
    ```bash
    python app.py
    ```

3.  **Access the UI:**
    Open your web browser and navigate to `http://127.0.0.1:5000`. You will see the interactive prediction form.

## Model Performance (XGBoost)

The upgraded XGBoost model achieves a high level of accuracy and provides more reliable predictions.

-   **Training Accuracy:** ~98%
-   **Testing Accuracy:** ~77%

The model demonstrates a strong ability to correctly identify individuals with and without diabetes. When you run `python main.py`, you will see a detailed classification report and an updated confusion matrix for the XGBoost model.

## Future Work

-   **Deployment:** Deploy the Flask application to a cloud platform like Heroku or AWS for public access.
-   **Advanced Feature Engineering:** Explore creating new features from existing ones to potentially improve model accuracy.
-   **Model Interpretability:** Integrate SHAP or LIME more deeply into the web application to provide visual explanations for each prediction.
-   **Alternative Models:** Experiment with other models like LightGBM or a deep learning approach to compare performance.
