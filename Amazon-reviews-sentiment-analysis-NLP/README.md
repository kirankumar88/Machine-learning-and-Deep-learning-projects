Amazon Reviews Sentiment Analysis using NLP


Project Overview
This repository presents an end-to-end Natural Language Processing (NLP) solution for sentiment analysis of Amazon customer reviews. The objective is to automatically classify review text into positive or negative sentiment categories using classical machine learning models, enabling scalable and objective analysis of customer feedback beyond numerical ratings.

Problem Statement
Customer reviews contain rich qualitative insights that are often underutilized due to scale and manual processing constraints. This project applies NLP and supervised machine learning techniques to extract sentiment signals from unstructured review text, supporting data-driven decisions in customer experience and product quality management.

Dataset Description
The dataset contains 10,000 Amazon product reviews with sentiment labels. The data is complete, contains no missing values, and shows a near-balanced distribution between positive and negative classes, ensuring unbiased model training and reliable evaluation.

Methodology
The analysis follows a structured machine learning pipeline:

Text preprocessing: lowercasing, punctuation removal, stopword elimination, negation preservation, and lemmatization

Feature extraction: TF-IDF vectorization

Models implemented:

Logistic Regression

Linear Support Vector Machine (SVM)

Evaluation metrics: accuracy, precision, recall, F1-score, and confusion matrix

Model Performance Metrics
Model	Accuracy	Precision	Recall	F1-score
Logistic Regression	0.86	0.86	0.86	0.86
Linear SVM	0.85	0.86	0.85	0.85

Logistic Regression marginally outperforms Linear SVM across all evaluation metrics while offering faster training time and higher interpretability. Based on these results, Logistic Regression is selected as the preferred production model.

Business Value

This solution enables organizations to:

Monitor customer sentiment at scale in near real time

Detect negative feedback and emerging product issues early

Support data-driven improvements in product quality and customer experience

Deploy a scalable and cost-effective sentiment analytics pipeline

Reproducibility

The project is implemented using reproducible machine learning pipelines built with standard Python libraries. All preprocessing, modeling, and evaluation steps can be replicated by installing the required dependencies and executing the provided notebook.

Technologies Used

Python, NLTK, Scikit-learn, TF-IDF Vectorization, Logistic Regression, Support Vector Machine (SVM), Matplotlib, Seaborn, and WordCloud.
