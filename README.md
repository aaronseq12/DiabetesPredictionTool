# Diabetes-Prediction-System

About me: 

My name is Aaron Sequeira, I am currently pursuing my Btech in Information Technology from Manipal University Jaipur, where I am building a solid foundation in various facets of the field. My keen interest spans several domains, including Full stack web development, Cloud Computing with a focus on AWS, Data Analytics and Machine Learning. 

Since I have enrolled in the course of foundation of data science, with the help of my professor, I had received the assignment to compelete a research paper on diabetes prediction model by the end of this semester. 

### Problem : 

Diabetes is a chronic (long-lasting) health condition that affects how your body turns food into energy. Your body breaks down most of the food you eat into sugar (glucose) and releases it into your bloodstream. When your blood sugar goes up, it signals your pancreas to release insulin. 

### Solution:

Diabetes Prediction and proper medical procedures to cure it. The medical show if of the Doctors'. But as Engineers, we can take the first step, i.e. prediction.

### Idea: 
Building an application that can predict the onset of diabetes or the possible causes of it by indicating the highly relevant factors. A multi-layer perceptron which predicts whether an individual is susceptible to diabetes. The model has been trained on the [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database), provided by the National Institute of Diabetes and Digestive and Kidney Diseases.

## Data Analysis
### Histograms
![Pima Indians Diabetes Database Histograms](https://i.imgur.com/htXtzS1.png)
*Note: 'outcome' refers to whether an individual does, or does not, have diabetes*

#### Insights
* Variables are on different scales, and therefore must be standardized
* The majority of data has been collected from individuals between 20 and 30 years of age
* ```BMI```, ```Blood Pressure```, and ```Glucose``` are normally distributed
  * This is to be expected when such statistics are collected from a population
* It is impossible for for ```BMI```, ```Blood Pressure```, and ```Glucose``` to have a value of zero
  * Missing or incomplete data?
* Certain individuals have had up to 15 pregnancies
  * While not implausible, this information should still be considered
* This data-set suggests that 35% of the population has diabetes (65% do not)
  * The World Health Organisation estimates that only 8.5% of the global population suffers from diabetes
  * ...this data-set is therefore not representative of the global population, which is to be expected due to its nature

### Density Plots
![Pima Indians Diabetes Database Density Plots](https://i.imgur.com/jmuAZt0.png)

#### Insights
* ```Glucose```, ```BMI```, and ```Age``` appear to be the strongest predicting values for those with diabetes
* ```Blood Pressure``` and ```Skin Thickness``` do not appear to have a significant correlation with the distribution of diabetic and non-diabetic individuals

## Data Pre-Processing
### Missing or Incomplete Values
#### Statistical Summary
![Pima Indians Diabetes Database Statistical Summary](https://i.imgur.com/yZN89GB.png)
* There are a total of 768 entries
* ```Pregnancies```, ```Glucose Concentration```, ```Blood Pressure```, ```Skin Thickness```, ```Insulin```, and ```BMI``` appear to have a minimum value of zero. This indicates missing values as such values are impossible

#### Number of Missing Values
![Pima Indians Diabetes Database No. of Missing Values](https://i.imgur.com/Q7meZol.png)
* There is a significant number of missing values. Most notably, a large number of entries for ```Insulin``` and ```Skin Thickness``` are missing
* Due to the fact that missing values have been determined by searching for entries with a value of zero, ```Pregnancies``` can be ignored as an individual with zero pregnancies is perfectly valid
* Missing values have been replaced with the mean of non-missing values

### Data Standardization
#### Statistical Summary of Standardized Data
![Pima Indians Diabetes Database Standardized Summary](https://i.imgur.com/N77tBkx.png)
* The values for ```Outcome``` have been copied from the original dataset as they do not require standardization

### Data Splits
The dataset has been split into training (```80%```) and testing (```20%```) splits. The training set has then been further divided into training (```80%```) and validation (```20%```) splits.

## Results
Once trained, the model was able to achieve ```96.74%``` accuracy on the training set and ```70.13%``` accuracy on the testing set.

### Confusion Matrix
![Pima Indians Diabetes Database Confusion Matrix](https://i.imgur.com/MWchEdh.png)
* In the case of diabetes prediction, false-negatives are the least desirable outcome as it would result in patients being informed that they will not develop diabetes when in fact they may.

### ROC Curve
![Pima Indians Diabetes Database ROC Curve](https://i.imgur.com/xkVhOlx.png)
