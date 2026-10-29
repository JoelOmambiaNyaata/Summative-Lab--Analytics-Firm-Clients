# Analytics Firm Client Projects: NLP, Time Series, and Neural Networks

## Overview
This repository contains the complete solution for the **Summative Lab: Analytics Firm Clients**. The project spans three distinct data science domains to solve realistic client challenges using the `amazon_reviews_lab.csv` dataset and supporting analytical workflows.

---

## Core Components

### Part 1: Natural Language Processing (NLP)
* **Preprocessing Pipeline**: Implements lowercasing, URL/punctuation removal, tokenization, alphabetic filtering, stopword removal, and lemmatization via NLTK.
* **Exploratory Analysis**: Computes single-word frequencies and bigram distributions using `Counter` and `ngrams`, visualized via Seaborn bar plots.

### Part 2: Time Series Analysis
* **Temporal Resampling**: Converts review timestamps to datetime format and resamples data monthly to evaluate review volume and average ratings over time.
* **Trend Analysis**: Implements rolling averages (e.g., 3-month moving averages) to smooth out short-term volatility and highlight long-term customer satisfaction trends.

### Part 3: Neural Network Modeling (Classification)
* **Feature Engineering**: Vectorizes textual reviews using TF-IDF (`max_features=5000`).
* **Deep Learning Architecture**: Builds a Multi-Layer Perceptron (MLP) in TensorFlow/Keras featuring dense hidden layers, ReLU activations, dropout regularization, and a softmax output layer for multi-class rating classification.
* **Evaluation Metrics**: Evaluates model performance using categorical cross-entropy loss, test accuracy, classification reports, and confusion matrix heatmaps.

---

## Prerequisites & Dependencies
Ensure your environment has the following libraries installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow nltk statsmodels yfinance
