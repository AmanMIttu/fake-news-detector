# Interpretable Fake News Detection using NLP

## Overview
This project develops a Natural Language Processing (NLP) system to automatically classify news articles as real or fake and to explain the reasoning behind each prediction.

The goal is not only prediction accuracy, but also model interpretability.

## Motivation
Misinformation detection is conceptually similar to anomaly detection in cybersecurity systems. Both require identifying abnormal patterns in large-scale textual data. This project explores how machine learning models can support trustworthy automated decision systems.

## Methodology
1. Text preprocessing and tokenization
2. TF-IDF feature extraction
3. Training multiple classifiers (Logistic Regression, Random Forest, SVM, XGBoost)
4. Model evaluation using precision, recall and F1-score
5. Explainable AI using LIME to interpret model decisions

## Dataset
Fake and Real News Dataset (Kaggle)

~38,000+ labeled news articles.

## Results
Random Forest achieved highest performance:

F1-score ≈ 0.996

Evaluation metrics include ROC curves, precision-recall curves and confusion matrices.

## Explainability
LIME explanations highlight important words influencing model predictions, demonstrating how linguistic patterns affect misinformation classification.

## Technologies
Python, Scikit-learn, Pandas, NLTK, LIME, Matplotlib

## Author
Shaik Aman
B.Tech Computer Science (Data Science)
