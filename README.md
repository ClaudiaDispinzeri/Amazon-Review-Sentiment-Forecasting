# Amazon-Review-Sentiment-Forecasting

# Sentiment Analysis and Forecasting on Amazon Fine Food Reviews

## Project Overview

This repository contains an advanced analytics project developed individually as part of a Data Mining/Text Mining course within the Master of Science in Business Analytics at the University at Albany (SUNY). The project focuses on leveraging Natural Language Processing (NLP), supervised machine learning, and time series forecasting techniques to analyze customer sentiment expressed in Amazon food product reviews.

The main business questions addressed by the project are:

1. Can we automatically classify customer sentiment from textual reviews and utilize this classification as an indicator to anticipate potential declines in consumer perception?
2. Is historical sentiment data sufficient to create reliable forecasts supporting strategic business decisions in marketing and brand reputation management?

## Dataset

The dataset utilized for this project is publicly available as "Amazon Fine Food Reviews," sourced from Kaggle ([Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)). It consists of more than 568,000 customer reviews collected between October 2003 and October 2012. The primary attributes used include:

* Review Text
* Review Summary
* Rating Scores (1-5 scale)
* Timestamps

Neutral reviews (rating of 3) were removed to clearly distinguish between positive (ratings 4-5) and negative sentiments (ratings 1-2).

## Methodology

### 1. Sentiment Classification

* **Text Preprocessing:** Cleaning, tokenization, stopwords removal, lemmatization.
* **Feature Extraction:** TF-IDF and Latent Semantic Analysis (LSA).
* **Supervised Models Applied:** Logistic Regression, Decision Trees, Random Forest, Naive Bayes Multinomial, Neural Networks (MLP Classifier).
* **Handling Imbalanced Data:** SMOTE (Synthetic Minority Oversampling Technique) and class weighting.
* **Transformer-based Model:** An exploratory analysis using Hugging Face Transformers on a random sample of 50,000 reviews, evaluating performance using the full text and summary fields.

### 2. Time Series Forecasting

* **Metric Construction:** Monthly proportion of negative reviews.
* **Forecasting Models:** ARIMA, SARIMA, Holt-Winters, Linear Regression with monthly dummy variables.
* **Model Validation:** Mean Absolute Percentage Error (MAPE), Root Mean Square Error (RMSE), Mean Absolute Error (MAE).
* **Selected Model:** SARIMA(2,1,2)(1,1,1,12), demonstrating robust forecasting performance (MAPE of 6.22%).

## Key Results

* The Neural Network (MLP Classifier) achieved the best sentiment classification performance with an F1-score of 96.9%.
* Transformer-based analysis confirmed the effectiveness of using concise summaries for quick, resource-efficient sentiment analysis.
* SARIMA forecasting model provided the most reliable predictions, ideal for proactive decision-making in marketing and reputation management.

## Repository Structure

```
.
├── FinalProject_TeamClaudiaDispinzeri.ipynb        # Main analysis notebook (NLP and supervised learning)
├── FinalProject_Ref_AutoArima_TeamClaudiaDispinzeri.ipynb # Auto ARIMA time series notebook
├── Reviews.csv                                     # Original dataset
├── negative_rate_series.csv                        # Processed time series dataset
└── Project_Team_ClaudiaDispinzeri_2025.pdf         # Detailed project description and results
```

## How to Use

* Clone or download the repository.
* Notebooks can be viewed directly on GitHub or executed locally using Jupyter Notebook or Google Colab.
* Ensure required libraries are installed: pandas, numpy, sklearn, nltk, statsmodels, matplotlib, seaborn, transformers, and imblearn.

## Contact

Feel free to reach out for collaborations or discussions regarding NLP, sentiment analysis, forecasting, or business analytics.

* https://www.linkedin.com/in/claudiadispinzeri/
