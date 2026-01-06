# Glass Type Classification using Random Forest

## Overview
This project applies ensemble machine learning techniques to classify different types of glass based on their physicochemical properties. The workflow includes exploratory data analysis, data visualization, preprocessing, imbalance handling, and model evaluation using Random Forest, Bagging, and Boosting methods.

## Problem Statement
The objective of this project is to classify glass samples into their respective categories using chemical composition features. Due to overlapping feature distributions and class imbalance, ensemble learning methods are employed to improve classification performance and robustness.

## Dataset
The Glass dataset contains numerical attributes representing refractive index and elemental composition of glass samples, along with a target variable indicating the glass type.

## Exploratory Data Analysis (EDA)
EDA was conducted to understand the structure and quality of the dataset, including:
- Examination of feature distributions and summary statistics
- Detection of missing values and outliers
- Assessment of class distribution to identify imbalance

## Data Visualization
The following visualizations were used to explore data patterns and relationships:
- **Histograms** to analyze the distribution of individual features
- **Box plots** to detect outliers and understand feature spread
- **Pair plots** to examine relationships and correlations between features across glass types

These visualizations provided insights into feature variability and class separability.

## Data Preprocessing
- Checked for missing values and handled them appropriately
- Applied feature scaling to normalize numerical features
- Addressed class imbalance using resampling techniques
- Prepared the dataset for model training and evaluation

## Model Implementation
- Split the data into training and testing sets
- Trained a **Random Forest classifier**
- Evaluated model performance using standard classification metrics

## Ensemble Methods
In addition to Random Forest, the following ensemble techniques were applied:
- **Bagging Classifier** to reduce variance
- **AdaBoost Classifier** to improve model learning on misclassified samples

## Evaluation Metrics
Model performance was assessed using:
- Accuracy
- Precision
- Recall
- F1-score

These metrics provide a balanced evaluation, particularly for imbalanced datasets.

## Results and Insights
Random Forest demonstrated robust classification performance on the Glass dataset. Boosting methods improved minority class prediction, highlighting the importance of ensemble learning and imbalance handling in multi-class classification problems.

## Technologies Used
- Python
- pandas, numpy
- scikit-learn
- imbalanced-learn
- matplotlib, seaborn
- Jupyter Notebook
