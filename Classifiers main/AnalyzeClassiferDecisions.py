# Change lines 50 & 65 & 125
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
import matplotlib.pyplot as plt
from nltk.corpus import stopwords




# Load the data from the CSV file using pandas
data = pd.read_csv('C:/Users/bzmor/Class Code/Internship/Other/dataset.csv')

# Define the texts (tweets) and labels (0 = Non-antisemitic, 1 = Antisemitic)
texts = data['Text'] # "Text" is the column name in the CSV file that contains the tweets
labels = data['Biased'] # "Biased" is the column name in the CSV file that contains the labels determined by human annotators

# Keep a copy of the original text data
original_texts = texts.copy()


# stop_words = stopwords.words('english')
# stop_words.extend(['rt', 'amp', 'https', 'http', 'co', 't', 's', 'm', 're'])
# vectorizer = TfidfVectorizer(stop_words=stop_words, max_df=0.7)


# Convert the text data into numerical values using TF-IDF.
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(texts)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Store the texts of the test set
test_texts = original_texts.iloc[y_test.index]

# Initialize classifiers
# svm = SVC(kernel='linear', class_weight='balanced')
# log_reg = LogisticRegression(class_weight='balanced')
random_forest = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)  
# gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
# ensemble_clf = VotingClassifier(estimators=[('random_forest', random_forest), ('gbm', gbm), ('log_reg', log_reg)], voting='hard')




# Update the classifiers dictionary
classifiers = {
    # 'SVM': svm,
    # 'Logistic Regression': log_reg,
    # 'Decision Tree': dec_tree,
    'Random Forest': random_forest,
    # 'Gradient Boosting': gbm,
    # 'Ensemble': ensemble_clf
}


# Clear the file if it already exists
if os.path.exists('analyzeClassifier.txt'):
    open('analyzeClassifier.txt', 'w').close()
if os.path.exists('analyzeClassifierIncorrectFlagged.txt'):
    open('analyzeClassifierIncorrectFlagged.txt', 'w').close()
if os.path.exists('analyzeClassifierCorrectFlaggedLowProbability.txt'):
    open('analyzeClassifierCorrectFlaggedLowProbability.txt', 'w').close()


# Train and evaluate each classifier
for name, clf in classifiers.items():
    # get the starting time
    start_time = time.time()
    # Train the classifier
    clf.fit(X_train, y_train)
    # Predict the test set
    y_pred = clf.predict(X_test)
    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    # get the ending time
    end_time = time.time()
    
     
    # Print the results of the classifier
    # Write the results to a file called 'analyzeClassifier.txt'
    print(f'{name} Training Time: {end_time - start_time:.2f} seconds')
    print(f'{name} Accuracy: {accuracy:.6f}')
    print(f'{name} Precision: {precision_score(y_test, y_pred, pos_label=1):.6f}\n')
    print(f'Classification Report for {name}:\n{classification_report(y_test, y_pred, target_names=["(0) Non-antisemitic", "(1) Antisemitic"])}\n')
    with open('analyzeClassifier.txt', 'w') as file:
        file.write(f'{name} Training Time: {end_time - start_time:.2f} seconds\n')
        file.write(f'{name} Accuracy: {accuracy:.6f}\n')
        file.write(f'{name} Precision: {precision_score(y_test, y_pred, pos_label=1):.6f}\n\n')
        file.write(f'Classification Report for {name}:\n{classification_report(y_test, y_pred, target_names=["(0) Non-antisemitic", "(1) Antisemitic"])}\n\n')
    




# Convert the text data into numerical values using TF-IDF.
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(texts.astype(str))  # Convert X to string before fitting

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)


# Train the RandomForestClassifier
# clf = SVC(kernel='linear', class_weight='balanced', random_state=42)
# clf = LogisticRegression(class_weight='balanced', random_state=42)
# clf = DecisionTreeClassifier(class_weight='balanced', random_state=42)
# clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# Predict the classes for the test set
y_pred = clf.predict(X_test)

# Predict the probabilities for the test set
y_prob = clf.predict_proba(X_test)

    



# Analyze the predictions made by the Random Forest classifier and write the results to a file called 'analyzeClassifier.txt'.
with open('analyzeClassifier.txt', 'a', encoding='utf-8') as file:
    for i in range(X_test.shape[0]):  # Use shape[0] to get the number of samples
        file.write(f"Sample {i}:\nOriginal text: {test_texts.iloc[i]}\nTrue class: {y_test.iloc[i]}\nPredicted class: {y_pred[i]}\nPredicted probabilities: {y_prob[i]}\n\n")


# Save the results to a CSV file called 'analyzeClassifier.csv'
# The results should contain the sample index, original text, true class, predicted class, Probability of non Antisemitic, and Probability of antisemitic columns
# do not use DataFrame.to_csv() method to write the results to a CSV file
# instead, use the file object to write the results to the file

with open('analyzeClassifier.csv', 'w', encoding='utf-8') as file:
    file.write('Sample,Original Text,True Class,Predicted Class,Probability of Non-antisemitic,Probability of Antisemitic\n')
    for i in range(X_test.shape[0]):  # Use shape[0] to get the number of samples
        # file.write(f"{i},\"{original_texts.iloc[i].replace('\"', '\"\"')}\",{y_test.iloc[i]},{y_pred[i]},{y_prob[i][0]},{y_prob[i][1]}\n")
        file.write(f"{i},\"{test_texts.iloc[i].replace('\"', '\"\"')}\",{y_test.iloc[i]},{y_pred[i]},{y_prob[i][0]},{y_prob[i][1]}\n")


        
# --------------------------------------------------------------------------------------------        

# Save the results to a CSV file called 'analyzeClassifierIncorrectFlagged.txt
# The results should contain the sample index, original text, true class, predicted class, 
# Probability of non Antisemitic, and Probability of antisemitic columns
with open('analyzeClassifierIncorrectFlagged.txt', 'w', encoding='utf-8') as file:
    file.write('Sample,Original Text,True Class,Predicted Class,Probability of Non-antisemitic,Probability of Antisemitic\n')
    for i in range(X_test.shape[0]):  # Use shape[0] to get the number of samples
        if y_test.iloc[i] != y_pred[i]:
            file.write(f"Sample {i}:\nOriginal text: {test_texts.iloc[i]}\nTrue class: {y_test.iloc[i]}\nPredicted class: {y_pred[i]}\nPredicted probabilities: {y_prob[i]}\n\n")

# Save the results to a CSV file called 'analyzeClassifierIncorrectFlagged.csv'
# The results should contain the sample index, original text, true class, predicted class, Probability of non Antisemitic, and Probability of antisemitic columns
# do not use DataFrame.to_csv() method to write the results to a CSV file
# instead, use the file object to write the results to the file

with open('analyzeClassifierIncorrectFlagged.csv', 'w', encoding='utf-8') as file:
    file.write('Sample,Original Text,True Class,Predicted Class,Probability of Non-antisemitic,Probability of Antisemitic\n')
    for i in range(X_test.shape[0]):  # Use shape[0] to get the number of samples
        if y_test.iloc[i] != y_pred[i]:
            file.write(f"{i},\"{test_texts.iloc[i].replace('\"', '\"\"')}\",{y_test.iloc[i]},{y_pred[i]},{y_prob[i][0]},{y_prob[i][1]}\n")


# --------------------------------------------------------------------------------------------        
      
# Save the results to a CSV file called ''analyzeClassifierCorrectFlaggedLowProbability.txt
# The results should contain the sample index, original text, true class, predicted class, 
# Probability of non Antisemitic, and Probability of antisemitic columns
with open('analyzeClassifierCorrectFlaggedLowProbability.txt', 'w', encoding='utf-8') as file:
    for i in range(X_test.shape[0]):  # Use shape[0] to get the number of samples
        if y_test.iloc[i] == y_pred[i] and y_prob[i][y_pred[i]] < 0.6:
            file.write(f"Sample {i}:\nOriginal text: {test_texts.iloc[i]}\nTrue class: {y_test.iloc[i]}\nPredicted class: {y_pred[i]}\nPredicted probabilities: {y_prob[i]}\n\n")

# Save the results to a CSV file called 'analyzeClassifierCorrectFlaggedLowProbability.csv'
# The results should contain the sample index, original text, true class, predicted class, Probability of non Antisemitic, and Probability of antisemitic columns
# do not use DataFrame.to_csv() method to write the results to a CSV file
# instead, use the file object to write the results to the file 
with open('analyzeClassifierCorrectFlaggedLowProbability.csv', 'w', encoding='utf-8') as file:
    file.write('Sample,Original Text,True Class,Predicted Class,Probability of Non-antisemitic,Probability of Antisemitic\n')
    for i in range(X_test.shape[0]):  # Use shape[0] to get the number of samples
        if y_test.iloc[i] == y_pred[i] and y_prob[i][y_pred[i]] < 0.6:
            file.write(f"{i},\"{test_texts.iloc[i].replace('\"', '\"\"')}\",{y_test.iloc[i]},{y_pred[i]},{y_prob[i][0]},{y_prob[i][1]}\n")



 
 
# --------------------------------------------------------------------------------------------        
      
       
# Get the feature importances from the classifier by accessing the feature_importances_ attribute
# write the feature importances to a file called 'analyzeClassifierFeatureImportances.txt'.

feature_importances = clf.feature_importances_
# Get the feature names from the vectorizer
feature_names = vectorizer.get_feature_names_out()
# Create a DataFrame to store the feature importances
feature_importances_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
# Sort the features by importance in descending order
feature_importances_df = feature_importances_df.sort_values('importance', ascending=False)
# Print the top 20 most important features and write it to a file called 'analyzeClassifierFeatureImportances.txt'
print(feature_importances_df.head(20))
with open('analyzeClassifierFeatureImportances.txt', 'w', encoding='utf-8') as file:
    file.write(feature_importances_df.head(20).to_string())
    


# generate a bar plot of the top 20 most important features and make it pop up in a new window.
# Use the matplotlib library to create the plot.

# Get the top 20 most important features
# feature_importances_df.head(20) returns the first 20 rows of the DataFrame
top_features = feature_importances_df.head(20)
# Create a horizontal bar plot
plt.figure(figsize=(10, 6))
# Use barh for horizontal bar plot 
plt.barh(top_features['feature'], top_features['importance'])
# Set the labels and title
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 20 Most Important Features')
# Adjust the layout to make the plot look better
plt.tight_layout()
# Save the plot as a PNG file and overwrite it if it already exists
plt.savefig('top_features.png', dpi=300)
# Display the plot
# plt.show()
