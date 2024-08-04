import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

# Load the CSV files for Gradient Boosting and Random Forest
gb_path = 'GBResultsForAnalyzeClassifier.csv'
rf_path = 'RFResultsForAnalyzeClassifier.csv'

gb_data = pd.read_csv(gb_path)
rf_data = pd.read_csv(rf_path)


def extract_tp_fn(data):
    # Extract True Positives (True Class = 1 and Predicted Class = 1)
    true_positives = data[(data['True Class'] == 1) & (data['Predicted Class'] == 1)]

    # Extract False Negatives (True Class = 1 and Predicted Class = 0)
    false_negatives = data[(data['True Class'] == 1) & (data['Predicted Class'] == 0)]

    return true_positives, false_negatives


gb_true_positives, gb_false_negatives = extract_tp_fn(gb_data)
rf_true_positives, rf_false_negatives = extract_tp_fn(rf_data)


def analyze_texts(texts):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    word_counts = X.toarray().sum(axis=0)
    vocab = vectorizer.get_feature_names_out()
    word_freq = dict(zip(vocab, word_counts))
    return Counter(word_freq).most_common(10)


# Analyze Gradient Boosting True Positives
gb_tp_common_words = analyze_texts(gb_true_positives['Original Text'])
print("Most common words in Gradient Boosting True Positives:", gb_tp_common_words)

# Analyze Gradient Boosting False Negatives
gb_fn_common_words = analyze_texts(gb_false_negatives['Original Text'])
print("Most common words in Gradient Boosting False Negatives:", gb_fn_common_words)

# Analyze Random Forest True Positives
rf_tp_common_words = analyze_texts(rf_true_positives['Original Text'])
print("Most common words in Random Forest True Positives:", rf_tp_common_words)

# Analyze Random Forest False Negatives
rf_fn_common_words = analyze_texts(rf_false_negatives['Original Text'])
print("Most common words in Random Forest False Negatives:", rf_fn_common_words)


def compare_common_words(tp_words, fn_words):
    tp_words_set = set([word for word, count in tp_words])
    fn_words_set = set([word for word, count in fn_words])
    missed_words = fn_words_set - tp_words_set
    return missed_words


# Compare Gradient Boosting
gb_missed_words = compare_common_words(gb_tp_common_words, gb_fn_common_words)
print("Words more common in Gradient Boosting False Negatives:", gb_missed_words)

# Compare Random Forest
rf_missed_words = compare_common_words(rf_tp_common_words, rf_fn_common_words)
print("Words more common in Random Forest False Negatives:", rf_missed_words)
