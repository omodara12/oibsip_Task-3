# SENTIMENT ANALYSIS ON TWITTER_DATASET
## Project Overview
##### This project aims to analyze sentiments from Twitter data using machine learning models. The core objective is to classify tweets into Negative (-1), Neutral (0), or Positive (1) sentiments. Two models were developed and compared:
##### •	LSTM (Long Short-Term Memory): A deep learning model capable of capturing sequential patterns.
##### •	Naive Bayes Classifier: A traditional machine learning model based on probability theory.
## Introduction
##### Social media is a rich source of real-time sentiment and public opinion. Accurately analyzing such data enables businesses, political entities, and researchers to understand public mood and make informed decisions. This project uses a labeled dataset of cleaned tweets to build sentiment classifiers.
## Loading dataset
##### •	Dataset Path: C://Users//USER//Documents//Twitter_Data.csv
![](https://github.com/omodara12/oibsip_Task-3/blob/main/task3-1.png)
##### •	The dataset includes columns like clean_text (preprocessed tweet text) and category 
##    • Data Cleaning & Exploration
## •	Key Cleaning Steps:
##### o	Removal of rows with missing and dropping duplicated row
![](https://github.com/omodara12/oibsip_Task-3/blob/main/task3-2.png)
##### o Convert sentiment labels to integers

##### o	Conversion to lowercase and tokenization using nltk.
##### o	Final dataset size: ~32,595 records.
