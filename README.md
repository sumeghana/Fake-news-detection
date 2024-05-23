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
Critiques following media attention, especially on platforms like Facebook which has begun to flag fake news, underscore the importance of our project.Facebook has been at the epicenter of many critiques following media attention. They have already implemented a feature to flag fake news on the site when a user sees it; they have also said publicly they are working on distinguishing these articles in an automated way. This problem provides a chance to learn more about textual data. The data we handled earlier were mostly numerical and didn’t need much data preprocessing. Since It is a Binary Classification, we ought to use logistic regression. But we still want to explore the different models to achieve higher accuracies. 


### Solution Benefits
Solving this problem will offer the capabilities of dealing with textual data, stemming and lemmatization. We ought to learn on few concepts on NLP and We believe it is meaningful because we never handled textual data while performing machine learning before. 
- Enhances capabilities with textual data, including stemming and lemmatization.
- Provides valuable learning in natural language processing (NLP), a new area for our previous work which was mainly numerical.

### Use of Solution
- This solution would be useful to classify even future articles for fake news. Although there can be more advanced models and open-source tools coming up, we believe our solution would work as an optimized solution for now. 
- To Classify the data, we always need to preprocess and fetch an accurate and most suitable machine learning model. Like we mentioned, Since it is a Binary Classification, we fixated on Logistic regression. But we still look forward to an optimized solution. Emphasizes data preprocessing and model accuracy.
- 

## Dataset Finalization
We use labeled datasets with textual content from thousands of articles categorized as real or fake. Key datasets include:
- **Dataset 1**: Labeled data focusing on US elections and Donald Trump-related fake articles.
  - [Kaggle Dataset](https://www.kaggle.com/c/fake-news/data?select=train.csv)
  - **Features**:
    - **id**: Unique ID for a news article.
    - **title**: The title of a news article.
    - **author**: Author of the news article.
    - **text**: The text of the article; could be incomplete.
    - **label**: A label that marks the article as potentially unreliable.
  - **Background**: This dataset was used in an InClass Prediction competition in 2015, mainly focusing on US presidential elections and fake articles about Donald Trump.


- **Dataset 2**: Real and fake news datasets.
  - [Kaggle Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
  - **Description**: This dataset is divided into real and fake CSV datasets.
  - **Features**:
    - **title**: The title of a news article.
    - **text**: The text of the article.
    - **subject**: The subject or category of the news article.
    - **date**: The date the article was published.
  - **Fake news Dataset**:
    - **Total Records**: 17,903 unique values.
    - **Distribution**:
      - 29% from political articles.
      - 39% from general news.
      - 32% from various other categories.
    - **Time Frame**: From 31st March 2015 to 19th February 2018.
  - **Real news Dataset**:
    - **Total Records**: 20,826 unique values.
    - **Distribution**:
      - 53% politics news.
      - 47% world news.
    - **Time Frame**: From 16th January 2016 to 31st December 2017.

- **Dataset 3**: Articles from websites tagged by the BS Detector Chrome Extension.
  - [Kaggle Dataset](https://www.kaggle.com/mrisdal/fake-news)
  - **Description**: This dataset contains text and metadata scraped from 244 websites tagged as "bullshit" by the BS Detector Chrome Extension by Daniel Sieradski.
  - **Features**:
    - **Uuid**: Unique user ID.
    - **Ord_in_thread**: Order in thread.
    - **Author**: Author of the article.
    - **Published date**: Date when the article was published.
    - **Title**: Title of the article.
    - **Text**: Main text content of the article.
    - **Language**: Language in which the article is written.
    - **Crawled**: Date when the article was crawled.
    - **Site URL**: URL of the website.
    - **Country**: Country of the website.
  - **Usage**: This dataset was also used for Sentiment Analysis in a beginner Kaggle competition.


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
