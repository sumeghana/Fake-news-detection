# Fake-news-detection
## Abstract

Fake news is a manipulated mix of news and misinformation. This project aims to develop a machine learning model capable of distinguishing between real and fake news articles.

### Problem Statement

Build a model to determine the authenticity of a news article using machine learning techniques, with an emphasis on achieving high prediction accuracy.

### Assumptions and Challenges

- Converting textual data into a numerical format that machine learning algorithms can process.
- Preprocessing textual data: removing punctuation, emojis, stop words, and unnecessary fillers.
- Utilizing techniques such as stemming and lemmatization to normalize the text.

## Introduction

With the prevalence of fake news, distinguishing between true and false information has become increasingly important. This project explores machine learning models to predict the likelihood of an article being fake.

## Motivation

The proliferation of fake news has significant implications. Platforms like Facebook have begun to implement measures to detect and flag fake content, making our project timely and relevant.

## Benefits of the Solution

- Hands-on experience with textual data preprocessing.
- Learning and applying NLP techniques such as stemming and lemmatization.
- Exploring various machine learning models to optimize prediction accuracy.

## Dataset Finalization

We utilized several datasets from Kaggle, containing thousands of labeled news articles, classified as either real or fake. These datasets include varied features such as article text, titles, and authors which are crucial for training our models.

### Data Preprocessing

- Removal of stopwords using the NLTK library.
- Application of stemming and lemmatization for text normalization.
- Vectorization of text data to numerical data using TfidfVectorizer.

## Supervised Learning Algorithms

### Models Explored

- **Naive Bayes:** Achieved an accuracy of 91.3% on sentiment analysis tasks.
- **Logistic Regression:** Used for binary classification of news articles, achieving high accuracies.
- **Random Forest:** Provided robust performance with an ensemble approach.
- **Decision Tree:** Offered an intuitive model structure with competitive accuracies.
- **Support Vector Machine (SVM):** Excelled in high-dimensional classification tasks.

### Best Fit for Datasets

- **Dataset 1:** Best modeled with Logistic Regression.
- **Dataset 2:** Most accurately classified with SVM.
- **Dataset 3:** Best results obtained with Decision Tree.

## Unsupervised Learning

- **PCA (Principal Component Analysis):** Used for feature reduction, although it showed moderate performance.
- **K-means Clustering:** Applied to partition data into clusters, but showed limited success.

## Conclusion

This project provides insights into handling and analyzing textual data, applying various machine learning algorithms, and improving fake news detection methods.

## How to Use

This model can classify new articles into real or fake, helping maintain the integrity of information spread online.

## Future Work

Continued exploration of advanced machine learning models and techniques to further enhance the accuracy and reliability of fake news detection.

"""
