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
## Training test split, Encoding
![](https://github.com/omodara12/oibsip_Task-3/blob/main/task3-5.png)
##### Tokenization and padding were applied.
##### Creating train_dataset
##### Define LSMT model
##### Trainig setup
##### Training loop
##### All were applied for dummy test
![](https://github.com/omodara12/oibsip_Task-3/blob/main/task%208.png)
##### •	Embedding Layer: 50 dimensions
##### •	Hidden Layer: 64 units
##### •	Output Classes: 3
##### •	Trained for 5 Epochs
##### •	Loss and Accuracy Progression:
##### Epoch 1: Accuracy = 50.02%
##### Epoch 5: Accuracy = 89.79%
##### I trained the twitter dataset with the model
## Key concepts
##### Tokenize using nltk
 ## Tokenize text
##### df['tokens'] = df['clean_text'].astype(str).apply(lambda x: word_tokenize(x.lower()))
## split train test
##### train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
## Convert tokens to tensors
##### train_sequences = [torch.tensor([hash(w) % 10000 for w in tokens], dtype=torch.long) for tokens in train_df['tokens']]
##### test_sequences = [torch.tensor([hash(w) % 10000 for w in tokens], dtype=torch.long) for tokens in test_df['tokens']]
## Pad sequences
##### train_padded = pad_sequence(train_sequences, batch_first=True)
##### test_padded = pad_sequence(test_sequences, batch_first=True)
## Labels
##### train_labels_tensor = torch.tensor(train_df['label'].tolist(), dtype=torch.long)
##### test_labels_tensor = torch.tensor(test_df['label'].tolist(), dtype=torch.long)
## Dataset and DataLoader
##### from torch.utils.data import Dataset, DataLoader
##### class SentimentDataset(Dataset):
##### def __init__(self, sequences, labels):
#####  self.sequences = sequences
 ##### self.labels = labels
 ##### def __len__(self):
 ##### return len(self.sequences)
 ##### def __getitem__(self, idx):
 ##### return self.sequences[idx], self.labels[idx]
##### train_dataset = SentimentDataset(train_padded, train_labels_tensor)
##### test_dataset = SentimentDataset(test_padded, test_labels_tensor)
##### train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
##### test_loader = DataLoader(test_dataset, batch_size=64)
### Then called with this;
##### model = SentimentLSTM(vocab_size=10000, embed_dim=50, hidden_dim=64, output_dim=3)  # 3 output classes
## Training Function
def train_model(model, train_loader, criterion, optimizer, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        for batch_inputs, batch_labels in train_loader:
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_inputs)

            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)

        accuracy = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

   ##### I called with this:
   ##### train_model(model, train_loader, criterion, optimizer, device, epochs=5)
   ##### Epoch 1/5, Loss: 2017.4318, Accuracy: 0.5002
   ##### Epoch 2/5, Loss: 985.5758, Accuracy: 0.8242
   #####  Epoch 3/5, Loss: 754.8846, Accuracy: 0.8697
   ##### Epoch 4/5, Loss: 652.2747, Accuracy: 0.8866
   ##### Epoch 5/5, Loss: 582.4004, Accuracy: 0.8979
   ## Evaluating function
  def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_inputs, batch_labels in test_loader:
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)

            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)

    accuracy = correct / total
    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")

#####    I called with;
#####    train_model(model, train_loader, criterion, optimizer, device, epochs=5)
    
    output;
    
    Epoch 1/5, Loss: 1991.0598, Accuracy: 0.5422
    Epoch 2/5, Loss: 1991.4377, Accuracy: 0.5424
    Epoch 3/5, Loss: 1991.4406, Accuracy: 0.5433
    Epoch 4/5, Loss: 1991.7135, Accuracy: 0.5417
    Epoch 5/5, Loss: 1991.5906, Accuracy: 0.5423
 ##### create sample predictiion function
    def sample_predictions(model, test_loader, device, num_samples=5):
    model.eval()
    samples = []
    labels = []
    with torch.no_grad():
        for i, (batch_inputs, batch_labels) in enumerate(test_loader):
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            # Get predictions
            outputs = model(batch_inputs)
            _, predicted = torch.max(outputs, 1
            # Collect samples and labels
            samples.extend(predicted.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())

            if len(samples) >= num_samples:
                break
    
    print("Sample Predictions (Predicted, True):")
    for i in range(num_samples):
        print(f"Predicted: {samples[i]}, True: {labels[i]}")
   ##### Calling function
   sample_predictions(model, test_loader, device, num_samples=5)
   Sample Predictions (Predicted, True):
##### Predicted: 1, True: 1
##### Predicted: 0, True: 1
##### Predicted: 0, True: 1
##### Predicted: 2, True: 2
##### Predicted: 1, True: 2
##### The model seems to be predicting Positive (1) frequently, even when the actual labels are Neutral (2) or Negative (0). This indicates some potential bias towards predicting positive sentiment.
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Switch to evaluation mode
model.eval()

# Store all predictions and labels
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_inputs, batch_labels in test_loader:
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)

        outputs = model(batch_inputs)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())

 ### Confusion matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Neutral", "Positive"], yticklabels=["Negative", "Neutral", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

### Precision, Recall, F1-score
##### print("Classification Report:")
##### print(classification_report(all_labels, all_preds, target_names=["Negative", "Neutral", "Positive"]))
       

![](https://github.com/omodara12/oibsip_Task-3/blob/main/conf.png)
## LSTM Observations:
##### •	Performs better on neutral sentiment.
##### •	Struggles with negative sentiment recall.
##### •	Susceptible to class imbalance due to deep learning's reliance on large, balanced datasets.
## Model 2: Naive Bayes
### • Preprocessing for training
##### Define features and labels
##### X = df["clean_text"]
##### y = df["category"]

##### Split data
##### from sklearn.model_selection import train_test_split
##### X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

##### TF-IDF Vectorization
#####  from sklearn.feature_extraction.text import TfidfVectorizer
##### vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
##### X_train_tfidf = vectorizer.fit_transform(X_train)
##### X_test_tfidf = vectorizer.transform(X_test)

##### Check shapes
##### X_train_tfidf.shape, X_test_tfidf.shape
##### ((130375, 5000), (32594, 5000))
## •	Trained on same cleaned dataset
### Evaluation function
![](https://github.com/omodara12/oibsip_Task-3/blob/main/EV.png)
![](https://github.com/omodara12/oibsip_Task-3/blob/main/best%20mo.png)
## Hyperparameter tuning

![](https://github.com/omodara12/oibsip_Task-3/blob/main/Hyper.png)
![](https://github.com/omodara12/oibsip_Task-3/blob/main/f.png)
## Naive Bayes Observations:
##### •	Strong overall performance.
##### •	Excellent recall on positive class (0.92) — great for detecting support or approval.
##### •	Balanced F1-scores across all classes.
##  Model Comparison Summary
##### Metric	    LSTM Model     	Naive Bayes Model
#####  Accuracy	54%        	      74.8%✅
#####  Negative F1	0.32	          0.60 ✅
###### Neutral F1	0.67	           0.77 ✅
#####  Positive F1	0.55	          0.78 ✅
#####  Macro F1-Score	0.51	       0.72 ✅

## Insights:
##### •	Naive Bayes is more effective for this dataset due to the text's short length and the model's robustness in sparse feature spaces.
##### •	The LSTM model may be underfitting or impacted by data imbalance and limited feature diversity.
##### •	High recall for positive tweets means the Naive Bayes model can confidently capture praise or support in tweets.
## Conclusion
##### This sentiment analysis project successfully compares two classification techniques on Twitter data. The Naive Bayes model emerges as the more practical and effective solution, achieving nearly 75% accuracy and balanced performance across all sentiment classes. With further fine-tuning, especially of the deep learning model, the system can evolve into a robust tool for real-time sentiment monitoring.

_______________________________________











 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 



