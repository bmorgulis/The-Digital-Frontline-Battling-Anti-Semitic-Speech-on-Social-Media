import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load the data from the CSV file
data = pd.read_csv('../DataSets/JikeliDataset.csv')

# Preprocess the text data
def preprocess(text):
    return text.lower().split()

data['Processed_Text'] = data['Text'].apply(preprocess)

# Train Word2Vec model using CBOW
cbow_model = Word2Vec(data['Processed_Text'], vector_size=100, window=5, min_count=1, sg=0)

# Function to get average Word2Vec vector for a document
def get_avg_w2v_vector(model, doc):
    word_vectors = [model.wv[word] for word in doc if word in model.wv]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(100)

# Apply the function to get document vectors
data['CBOW_Vector'] = data['Processed_Text'].apply(lambda doc: get_avg_w2v_vector(cbow_model, doc))

# Prepare data for SVM
X_cbow = np.vstack(data['CBOW_Vector'].values)
y = data['Biased'].values

# Split the data into training and testing sets
X_train_cbow, X_test_cbow, y_train, y_test = train_test_split(X_cbow, y, test_size=0.2, random_state=42)

# Create and train the SVM classifier
svm_cbow = SVC(kernel='linear', class_weight='balanced')  # Using balanced class weights
svm_cbow.fit(X_train_cbow, y_train)

# Make predictions on the test data
y_pred_cbow = svm_cbow.predict(X_test_cbow)

# Evaluate the model's performance
accuracy_cbow = accuracy_score(y_test, y_pred_cbow)
report_cbow = classification_report(y_test, y_pred_cbow, target_names=['(0) Non-antisemitic', '(1) Antisemitic'])

# Print the evaluation results
print(f'CBOW Model Accuracy: {accuracy_cbow}')
print('CBOW Model Classification Report:')
print(report_cbow)



# import pandas as pd
# from gensim.models import Word2Vec
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, classification_report
# import numpy as np
#
# # Load the data from the CSV file
# data = pd.read_csv('Updated_dataset_with_full_text.csv')
#
# # Display the first few rows to understand the structure of the data
# print(data.head())
#
# # Use the 'Text' column for the tweets and 'Biased' column for the labels
# texts = data['Text']
# labels = data['Biased']
#
# # Preprocess the text data by converting to lowercase and splitting into words
# def preprocess(text):
#     return text.lower().split()
#
# data['Processed_Text'] = data['Text'].apply(preprocess)
#
# # Train Word2Vec model using CBOW (default: sg=0)
# # vector_size=100 sets the dimensionality of the word vectors
# # window=5 specifies the maximum distance between the current and predicted word within a sentence
# # min_count=1 ignores all words with total frequency lower than this
# cbow_model = Word2Vec(data['Processed_Text'], vector_size=100, window=5, min_count=1, sg=0)
#
# # Function to get the average Word2Vec vector for a document
# def get_avg_w2v_vector(model, doc):
#     # Get vectors for words in the document that are in the model's vocabulary
#     word_vectors = [model.wv[word] for word in doc if word in model.wv]
#     # Return the mean vector of all word vectors, or a zero vector if none of the words are in the vocabulary
#     return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(100)
#
# # Apply the function to get document vectors for each processed text
# data['CBOW_Vector'] = data['Processed_Text'].apply(lambda doc: get_avg_w2v_vector(cbow_model, doc))
#
# # Prepare data for SVM by stacking the document vectors
# X_cbow = np.vstack(data['CBOW_Vector'].values)
# y = labels.values
#
# # Split the data into training and testing sets
# # test_size=0.2 means 20% of the data is used for testing, 80% for training
# # random_state=42 ensures the split is the same every time
# X_train_cbow, X_test_cbow, y_train, y_test = train_test_split(X_cbow, y, test_size=0.2, random_state=42)
#
# # Create and train the SVM classifier with a linear kernel
# svm_cbow = SVC(kernel='linear')
# svm_cbow.fit(X_train_cbow, y_train)
#
# # Make predictions on the test data
# y_pred_cbow = svm_cbow.predict(X_test_cbow)
#
# # Evaluate the CBOW model's performance
# accuracy_cbow = accuracy_score(y_test, y_pred_cbow)
# report_cbow = classification_report(y_test, y_pred_cbow, target_names=['(0) Non-antisemitic', '(1) Antisemitic'])
#
# # Print the evaluation results
# print(f'CBOW Model Accuracy: {accuracy_cbow}')
# print('CBOW Model Classification Report:')
# print(report_cbow)
