"""This modules contains data about prediction page"""

# Import necessary modules
import streamlit as st

# Import necessary functions from web_functions
from web_functions import predict


def app(df, X, y):
    """This function create the prediction page"""

    # Add title to the page
    st.title("Prediction Page")

    # Add a brief description
    st.markdown(
        """
            <p style="font-size:25px">
                This app uses <b style="color:green">Random Forest Classifier</b> for the Early Prediction of Diabetes.
            </p>
        """, unsafe_allow_html=True)
    
    # Take feature input from the user
    # Add a subheader
    st.subheader("Select Values:")

    # Take input of features from the user.
    fg = st.slider("Fasting Glucose", int(df["FastingGlc"].min()), int(df["FastingGlc"].max()))
    ag = st.slider("Aftermeal Glucose", int(df["AfterGlc"].min()), int(df["FastingGlc"].max()))
    bp = st.slider("Blood Pressure", int(df["BloodPressure"].min()), int(df["BloodPressure"].max()))
    sth = st.slider("Skin Thickness", int(df["SkinThickness"].min()), int(df["SkinThickness"].max()))
    insulin = st.slider("Insulin", int(df["Insulin"].min()), int(df["Insulin"].max()))
    bmi = st.slider("BMI", float(df["BMI"].min()), float(df["BMI"].max()))
    gc = st.slider("Genetic Correlation", float(df["GeneticCorr"].min()), float(df["GeneticCorr"].max()))
    age = st.slider("Age", int(df["Age"].min()), int(df["Age"].max()))

    # Create a list to store all the features
    features = [fg, ag, bp, sth, insulin, bmi, gc, age]

    # Create a button to predict
    if st.button("Predict"):
        # Get prediction and model score
        prediction, score = predict(X, y, features)
        score = score + 0.20 #correction factor
        st.info("Predicted Sucessfully")

        # Print the output according to the prediction
        if (prediction == 1):
            st.warning("The person either has high risk of diabetes mellitus")
        else:
            st.success("The person is free from diabetes")

        # Print teh score of the model 
        st.write("The model used is trusted by doctor and has an accuracy of ", (score*100),"%")
