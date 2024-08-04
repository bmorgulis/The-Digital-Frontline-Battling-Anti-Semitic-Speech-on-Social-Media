import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load
import os
import numpy as np
from tensorflow.keras.preprocessing import text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def train_lstm_model():
    # Load the data
    data = pd.read_csv('JikeliDataset.csv')
    texts = data['Text'].astype(str)
    labels = data['Biased']
    
    # Preprocess the data 
    # limits the vocab to the top 5000 words
    # pad the sequences to the same length
    # to_categorical converts the integer labels into one-hot encoded vectors
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    max_sequence_length = max(len(x) for x in sequences)
    X = pad_sequences(sequences, maxlen=max_sequence_length)

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    y = to_categorical(labels_encoded)

    # Split the data 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the LSTM model
    # Embedding layer: Turns positive integers (indexes) into dense vectors of fixed size
    # LSTM layer: Long Short-Term Memory layer - 100 units to process the sequences of word embeddings
    # Dense layer: Fully connected layer with softmax activation function to output the probability distribution of the target classes
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=100, input_length=max_sequence_length))
    model.add(LSTM(100))
    model.add(Dense(y.shape[1], activation='softmax'))

    # Compile the model Accuracy is used as the metric to evaluate the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model with the training data
    # 5 epochs and a batch size of 64
    # validation_data is used to evaluate the model on the test data after each epoch
    # The model is trained on the training data and evaluated on the test data
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=64)

    # Evaluate the model
    """The model is evaluated on the test data
    The loss and accuracy of the model are printed
    accuracy is the accuracy of the model - the proportion of correct predictions made by the model.
    val_accuracy is the accuracy of the model on the test data
    loss is the loss value of the model -  a measure of how well the 
    val_loss is the loss of the model on the test data
    STM model's predictions match the actual labels of the data. It 
    quantifies the difference between the predicted values and the true 
    values, and the goal of training the model is to minimize this loss.
    """
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'LSTM Model Accuracy: {accuracy}')
    y_pred = model.predict(X_test)
    write_results_to_file(accuracy, y_test, y_pred)

def write_results_to_file(accuracy, y_test, y_pred):
    # Convert probabilities to binary labels based on a 0.5 threshold
    y_pred_binary = (y_pred > 0.5).astype(int)    
    # Check if the file already exists
    if os.path.exists('resultsLSTM.txt'):
        # Append the results to the file
        with open('resultsLSTM.txt', 'w') as f:
            f.write('Results for LSTM Model:\n')
            f.write(f'Accuracy: {accuracy}\n')
            # Write the true(actual) and pred(predicted) labels to the file
            for true, pred in zip(y_test, y_pred_binary):
                # Assuming true and pred are one-hot encoded, converting to class labels
                true_label = np.argmax(true)
                pred_label = np.argmax(pred)
                f.write(f'{true_label}, {pred_label}\n')
    else: 
        # Write the results to a new file
        with open('resultsLSTM.txt', 'w') as f:
            f.write('Results for LSTM Model:\n')
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'Classification Report:\n{classification_report(y_test, y_pred)}\n\n')
            
            
    

# Call the function to train the LSTM model
train_lstm_model()
