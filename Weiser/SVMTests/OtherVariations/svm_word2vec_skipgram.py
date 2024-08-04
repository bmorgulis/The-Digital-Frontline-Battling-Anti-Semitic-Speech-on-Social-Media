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

# Train Word2Vec model using Skip-gram
skipgram_model = Word2Vec(data['Processed_Text'], vector_size=100, window=5, min_count=1, sg=1)

# Function to get average Word2Vec vector for a document
def get_avg_w2v_vector(model, doc):
    word_vectors = [model.wv[word] for word in doc if word in model.wv]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(100)

# Apply the function to get document vectors
data['Skipgram_Vector'] = data['Processed_Text'].apply(lambda doc: get_avg_w2v_vector(skipgram_model, doc))

# Prepare data for SVM
X_skipgram = np.vstack(data['Skipgram_Vector'].values)
y = data['Biased'].values

# Split the data into training and testing sets
X_train_skipgram, X_test_skipgram, y_train, y_test = train_test_split(X_skipgram, y, test_size=0.2, random_state=42)

# Create and train the SVM classifier
svm_skipgram = SVC(kernel='linear', class_weight='balanced')  # Using balanced class weights
svm_skipgram.fit(X_train_skipgram, y_train)

# Make predictions on the test data
y_pred_skipgram = svm_skipgram.predict(X_test_skipgram)

# Evaluate the model's performance
accuracy_skipgram = accuracy_score(y_test, y_pred_skipgram)
report_skipgram = classification_report(y_test, y_pred_skipgram, target_names=['(0) Non-antisemitic', '(1) Antisemitic'])

# Print the evaluation results
print(f'Skip-gram Model Accuracy: {accuracy_skipgram}')
print('Skip-gram Model Classification Report:')
print(report_skipgram)
