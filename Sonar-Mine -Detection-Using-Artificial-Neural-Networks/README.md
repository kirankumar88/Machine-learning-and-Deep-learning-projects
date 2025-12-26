# Sonar Object Classification using Artificial Neural Networks

## Overview
This project implements an Artificial Neural Network (ANN) to classify sonar signals as either **Mine** or **Rock** based on 60 frequency-based energy features. The objective is to demonstrate an end-to-end machine learning workflow including data analysis, model training, hyperparameter tuning, and performance evaluation.

## Dataset
- 208 samples
- 60 continuous input features
- Binary target:
  - 1 → Mine
  - 0 → Rock
- Near-balanced class distribution

## Methodology
- Exploratory Data Analysis (EDA)
- Binary encoding of target labels
- Stratified train–test split
- ANN with:
  - ReLU activation
  - L2 regularization
  - Dropout
  - Early stopping
- Hyperparameter tuning using GridSearchCV (SciKeras)

## Results
| Model        | Accuracy | Precision | Recall | F1-score |
|-------------|----------|-----------|--------|----------|
| Base ANN    | 78.6%    | 0.76      | 0.86   | 0.81     |
| Tuned ANN   | 78.6%    | 0.74      | 0.91   | 0.82     |

## Key Insights
- Hyperparameter tuning improved recall, reducing false negatives.
- Suitable for applications where missing a positive instance (Mine) is costly.
- Stable training with no significant overfitting observed.

## How to Run
```bash
pip install -r requirements.txt
python src/train.py
