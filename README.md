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
![](https://github.com/omodara12/oibsip_Task-3/blob/main/task3-3.png)
##### o	Final dataset size: ~32,595 records.
## Sentiment Distribution
![](https://github.com/omodara12/oibsip_Task-3/blob/main/sentiment.png)
## Interpretatation
##### There are three sentiment category
##### Negative (-1)
##### Neutral (0)
##### Positive (1)
##### Most tweets or texts in the dataset express positive sentiment.
##### There’s a significant drop in the number of negative examples.
##### The distribution reflects how people talk online (e.g., people tend to post positive or neutral tweets more often than negative ones, depending on the source).
##### Data preprocessing, labelling, Mapping
![](https://github.com/omodara12/oibsip_Task-3/blob/main/task3-4.png)
![](https://github.com/omodara12/oibsip_Task-3/blob/main/drop%20NaNpng.png)
## Training test split



