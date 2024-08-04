import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load the datasets
train_file_path = 'TestandTrainDataSets/training_data_Jikeli.csv'
test_file_path = 'TestandTrainDataSets/testing_data_Jikeli.csv'

train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

# Extract features and labels
X_train = train_df['Text']
y_train = train_df['Biased']
X_test = test_df['Text']
y_test = test_df['Biased']

# Preprocess the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Define classifiers
random_forest = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
svm = SVC(kernel='rbf', class_weight='balanced')

# Train and predict with each classifier, then save results
classifiers = {
    'random_forest': random_forest,
    'gradient_boosting': gbm,
    'svm': svm
}

for clf_name, clf in classifiers.items():
    clf.fit(X_train_tfidf, y_train)
    predictions = clf.predict(X_test_tfidf)

    # Save results to CSV
    results_df = test_df.copy()
    results_df['Predicted'] = predictions
    results_df.to_csv(f'{clf_name}_results.csv', index=False)

    # Print classification report for each classifier
    print(f'Classification report for {clf_name}:')
    print(classification_report(y_test, predictions))
    print("\n")
