# Glass Type Classification Using Random Forest

## Overview
This project implements an ensemble machine learning approach to classify different types of glass based on their physicochemical properties. The study leverages the Glass dataset to demonstrate exploratory data analysis, preprocessing, imbalance handling, and ensemble model evaluation using Random Forest, Bagging, and Boosting techniques.

## Problem Statement
The objective of this project is to classify glass samples into their respective categories using chemical composition features. Due to overlapping feature distributions and class imbalance, ensemble learning methods are applied to improve prediction robustness and generalization.

## Dataset
The Glass dataset consists of numerical attributes representing refractive index and elemental composition of glass samples. The target variable represents different glass types used in real-world applications such as building windows and containers.

## Methodology
- Exploratory Data Analysis (EDA) to understand data distribution and quality
- Data visualization using histograms, box plots, and pair plots
- Data preprocessing including feature scaling and imbalance handling
- Random Forest model implementation for multi-class classification
- Comparative analysis of Bagging and Boosting ensemble techniques

## Models Implemented
- Random Forest Classifier
- Bagging Classifier
- AdaBoost Classifier

## Evaluation Metrics
Model performance is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score

These metrics provide a comprehensive assessment, particularly for imbalanced class distributions.

## Results and Insights
Random Forest demonstrated strong baseline performance for glass type classification. Boosting methods improved minority class recognition, highlighting the importance of ensemble techniques and imbalance handling in multi-class classification problems.

## Technologies Used
- Python
- pandas, numpy
- scikit-learn
- imbalanced-learn
- matplotlib, seaborn
- Jupyter Notebook

## Project Structure
