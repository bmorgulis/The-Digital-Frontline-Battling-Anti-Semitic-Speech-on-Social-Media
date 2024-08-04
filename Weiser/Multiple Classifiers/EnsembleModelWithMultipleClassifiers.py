import time
import pandas as pd  # Importing pandas for data manipulation
from sklearn.feature_extraction.text import TfidfVectorizer  # For converting text to numerical representation
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.svm import SVC  # Importing Support Vector Classifier
from sklearn.linear_model import LogisticRegression  # Importing Logistic Regression
from sklearn.tree import DecisionTreeClassifier  # Importing Decision Tree Classifier
from sklearn.ensemble import VotingClassifier  # Importing Voting Classifier for ensemble method
from sklearn.metrics import accuracy_score, classification_report, precision_score  # For model evaluation metrics
from joblib import dump, load  # For saving and loading models
import os  # For checking if files exist
from sklearn.ensemble import RandomForestClassifier  # Importing Random Forest Classifier
from sklearn.ensemble import GradientBoostingClassifier # Importing Gradient Boosting Classifier


# Load the data from the CSV file using pandas
data = pd.read_csv('C:/Users/bzmor/Class Code/Internship/Other/dataset.csv')

# Define the texts (tweets) and labels (0 = Non-antisemitic, 1 = Antisemitic)
texts = data['Text'] # "Text" is the column name in the CSV file that contains the tweets
labels = data['Biased'] # "Biased" is the column name in the CSV file that contains the labels determined by human annotators

# Check if models and vectorizer are already saved
if os.path.exists('svm_model.joblib') and os.path.exists('log_reg_model.joblib') and os.path.exists(
        'vectorizer.joblib') and os.path.exists('random_forest.joblib') and os.path.exists('gbm.joblib'):
    # Load the models and vectorizer if they are already saved
    svm = load('svm_model.joblib')
    log_reg = load('log_reg_model.joblib')
    vectorizer = load('vectorizer.joblib')
    random_forest = load('random_forest.joblib')
    gbm = load('gbm.joblib')
    X = vectorizer.transform(texts)

else:
    # Convert the text data into numerical values using TF-IDF. 
    # max_df=0.7 means that words that appear in more than 70% of the documents will be ignored
    # because they are likely to be common words that don't carry much information
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X = vectorizer.fit_transform(texts)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Initialize classifiers
# Define SVM and Logistic Regression classifiers
svm = SVC(kernel='linear', class_weight='balanced')
log_reg = LogisticRegression(class_weight='balanced')
random_forest = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)  
gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
dec_tree = DecisionTreeClassifier()

# Update the classifiers dictionary
classifiers = {
    'SVM': svm,
    'Logistic Regression': log_reg,
    'Decision Tree': dec_tree,
    'Random Forest': random_forest,
    'Gradient Boosting': gbm,
    'Ensemble': VotingClassifier(estimators=[('svm', svm), ('log_reg', log_reg)], voting='hard')
    # 'Ensemble': VotingClassifier(estimators=[('svm', svm), ('log_reg', log_reg), ('random_forest', random_forest), ('gbm', gbm),('dec_tree', dec_tree)], voting='hard')
    # 'Ensemble': VotingClassifier(estimators=[('random_forest', random_forest), ('gbm', gbm), ('log_reg', log_reg)], voting='hard')

}


    

# Save the models and vectorizer for future use
dump(svm, 'svm_model.joblib')
dump(log_reg, 'log_reg_model.joblib')
dump(vectorizer, 'vectorizer.joblib')
dump(random_forest, 'random_forest.joblib')
dump(gbm, 'gbm.joblib')


# Clear the file if it already exists
if os.path.exists('multiple_classifiers_results.txt'):
    open('multiple_classifiers_results.txt', 'w').close()

# Train and evaluate each classifier
for name, clf in classifiers.items():
    # get the starting time
    start_time = time.time()
    # Train the classifier
    # was clf.fit(X, labels) but should be clf.fit(X_train, y_train) because we want to train on the training data only and not on the entire dataset
    clf.fit(X_train, y_train)
    # Predict the test set
    y_pred = clf.predict(X_test)
    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    # get the ending time
    end_time = time.time()
    """ Print the results and then write the results to a file called 'multiple_classifiers_results.txt'
     The file will be created in the same directory as the script
     The 'a' parameter in the open function is for appending to the file
     If the file does not exist, it will be created
     Make a new file at the start of the script to clear the previous results     
     """
    print(f'{name} Training Time: {end_time - start_time:.2f} seconds')
    print(f'{name} Accuracy: {accuracy:.6f}')
    print(f'{name} Precision: {precision_score(y_test, y_pred, pos_label=1):.6f}\n')
    print(f'Classification Report for {name}:\n{classification_report(y_test, y_pred, target_names=["(0) Non-antisemitic", "(1) Antisemitic"])}\n')
    with open('multiple_classifiers_results.txt', 'a') as file:
        file.write(f'{name} Training Time: {end_time - start_time:.2f} seconds\n')
        file.write(f'{name} Accuracy: {accuracy:.6f}\n')
        file.write(f'{name} Precision: {precision_score(y_test, y_pred, pos_label=1):.6f}\n\n')
        file.write(f'Classification Report for {name}:\n{classification_report(y_test, y_pred, target_names=["(0) Non-antisemitic", "(1) Antisemitic"])}\n\n')

    
# Function to classify a new text using the ensemble classifier
def classify_text_ensemble(text):
    # Convert the new text to a TF-IDF vector
    text_vector = vectorizer.transform([text])
    # Define the ensemble classifier
    ensemble_clf = classifiers['Ensemble']
    # Predict the label using the ensemble classifier
    prediction = ensemble_clf.predict(text_vector)
    # Map the prediction to a human-readable label
    label = 'Antisemitic' if prediction == 1 else 'Non-antisemitic'
    return label



#Continuously prompt the user to enter text for classification
while True:
    user_input = input("Enter a text to classify (or type 'exit' to stop): ")
    if user_input.lower() == 'exit':
        break
    # Get prediction from the ensemble classifier
    result_ensemble = classify_text_ensemble(user_input)

    print("-----------------")
    print(f"Text: {user_input} | Ensemble Classification: {result_ensemble}")
    
