# Fake-news-detection
## Abstract
Today, "Fake news" is often mentioned, but it encompasses a collection of manipulated media designed to capture attention with sensational aesthetics, presenting a fabricated reality. It's more deceptive and dangerous than many realize. Our goal is to develop a Fake News Prediction System using machine learning in Python, training it to distinguish between real and fake articles and to quantify its accuracy.

### Formal Problem Description
A program learns from experience (E) with respect to a task (T) and performance measure (P), if its performance on T, as measured by P, improves with E:
- **Task (T)**: Predict whether provided data is real or fake.
- **Experience (E)**: A large amount of textual data from several thousand news articles.
- **Performance (P)**: Accuracy of predicting fake news correctly.

## Introduction
With rampant online fake news and misinformation, discerning truth from fiction is challenging. Our model aims to accurately predict the likelihood of an article being fake to help restore public trust.

### Motivation
Critiques following media attention, especially on platforms like Facebook which has begun to flag fake news, underscore the importance of our project.

### Solution Benefits
- Enhances capabilities with textual data, including stemming and lemmatization.
- Provides valuable learning in natural language processing (NLP), a new area for our previous work which was mainly numerical.

### Use of Solution
- Classifies articles as fake or real, potentially applicable to future content.
- Emphasizes data preprocessing and model accuracy.

## Dataset Finalization
We use labeled datasets with textual content from thousands of articles categorized as real or fake. Key datasets include:
- **Dataset 1**: Labeled data focusing on US elections and Donald Trump-related fake articles. [Kaggle Dataset](https://www.kaggle.com/c/fake-news/data?select=train.csv)
- **Dataset 2**: Real and fake news datasets. [Kaggle Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- **Dataset 3**: Articles from websites tagged by the BS Detector Chrome Extension. [Kaggle Dataset](https://www.kaggle.com/mrisdal/fake-news)

## Data Preprocessing
- **Stopwords Removal**: Using NLTK to eliminate irrelevant words.
- **Stemming**: Applying the Porter stemming algorithm to reduce words to their root form.
- **Lemmatization**: Transforming words to their base form using Wordnet.

## Supervised Learning Algorithms
- **Naive Bayes**: Achieved about 91.3% accuracy on Dataset 3.
- **Logistic Regression**: Used for binary classification; high accuracy observed on Datasets 1 and 2.
- **Random Forest**: Employed for its robustness and accuracy.
- **Decision Tree**: Offers an intuitive approach and effective classification.
- **Support Vector Machine (SVM)**: Best fits Dataset 2 with high accuracy due to its effectiveness in high-dimensional spaces.

## Unsupervised Learning
- **PCA (Principal Component Analysis)**: Used for feature reduction but showed moderate performance.
- **K-means Clustering**: Attempted to segment data into clusters; however, showed limited success.

### Long Short-Term Memory (LSTM)
- Applied to Datasets 2 and 3 with high accuracies. LSTMs manage data sequences effectively by retaining relevant information over time.

## Conclusion
