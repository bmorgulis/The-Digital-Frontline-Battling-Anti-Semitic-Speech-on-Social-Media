import pandas as pd

# Load the CSV files for Gradient Boosting and Random Forest
gb_path = 'FullResultsFromClassfiers/gradient_boosting_results.csv'
rf_path = 'FullResultsFromClassfiers/random_forest_results.csv'
svm_path = 'FullResultsFromClassfiers/svm_results.csv'

gb_data = pd.read_csv(gb_path)
rf_data = pd.read_csv(rf_path)
svm_data = pd.read_csv(svm_path)


def separate_results(data, classifier_name):
    # Extract True Positives (Biased = 1 and Predicted = 1)
    true_positives = data[(data['Biased'] == 1) & (data['Predicted'] == 1)]

    # Extract False Positives (Biased = 0 and Predicted = 1)
    false_positives = data[(data['Biased'] == 0) & (data['Predicted'] == 1)]

    # Extract True Negatives (Biased = 0 and Predicted = 0)
    true_negatives = data[(data['Biased'] == 0) & (data['Predicted'] == 0)]

    # Extract False Negatives (Biased = 1 and Predicted = 0)
    false_negatives = data[(data['Biased'] == 1) & (data['Predicted'] == 0)]

    # Save to CSV files
    true_positives.to_csv(f'{classifier_name}_true_positives.csv', index=False)
    false_positives.to_csv(f'{classifier_name}_false_positives.csv', index=False)
    true_negatives.to_csv(f'{classifier_name}_true_negatives.csv', index=False)
    false_negatives.to_csv(f'{classifier_name}_false_negatives.csv', index=False)


# Separate and save results for Gradient Boosting
separate_results(gb_data, 'Gradient_Boosting')

# Separate and save results for Random Forest
separate_results(rf_data, 'Random_Forest')

# Separate and save results for SVM
separate_results(svm_data, 'SVM')

print("Files created successfully.")
