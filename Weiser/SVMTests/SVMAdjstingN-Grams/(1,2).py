import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the data from the CSV file
data = pd.read_csv('../DataSets/JikeliDataset.csv')

# Display the first few rows to understand the structure of the data
print(data.head())

# Use the 'Text' column for the tweets and 'Biased' column for the labels
texts = data['Text']
labels = data['Biased']

# Convert the text data into numerical values using TF-IDF
# TF-IDF stands for Term Frequency-Inverse Document Frequency
# It helps to weigh the importance of words in the texts
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1, 2))
# stop_words='english' removes common English words that may not be useful
# max_df=0.7 ignores words that appear in more than 70% of the texts
# ngram_range=(1, 2) considers both unigrams (single words) and bigrams (pairs of words)
X = vectorizer.fit_transform(texts)

# Split the data into training and testing sets
# This allows us to train the model on one part of the data and test it on another part
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
# test_size=0.2 means 20% of the data is used for testing, 80% for training
# random_state=42 ensures the split is the same every time

# Create an SVM classifier
# SVM stands for Support Vector Machine, a type of model used for classification
svm = SVC(kernel='linear')  # 'linear' means it uses a straight line to separate the categories


# Train the model on the training data
svm.fit(X_train, y_train)

# Make predictions on the test data
y_pred = svm.predict(X_test)

# Evaluate the model's performance
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
# Accuracy shows the percentage of correct predictions

# Classification report provides detailed metrics like precision, recall, and F1-score for each category
report = classification_report(y_test, y_pred, target_names=['(0) Non-antisemitic', '(1) Antisemitic'])
print(report)
