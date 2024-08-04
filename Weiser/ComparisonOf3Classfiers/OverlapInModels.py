import pandas as pd
import matplotlib.pyplot as plt

# Load the separated CSV files for Gradient Boosting, Random Forest, and SVM
gb_tp = pd.read_csv('BrokenDownResults/GB/Gradient_Boosting_true_positives.csv')
gb_fp = pd.read_csv('BrokenDownResults/GB/Gradient_Boosting_false_positives.csv')
gb_tn = pd.read_csv('BrokenDownResults/GB/Gradient_Boosting_true_negatives.csv')
gb_fn = pd.read_csv('BrokenDownResults/GB/Gradient_Boosting_false_negatives.csv')

rf_tp = pd.read_csv('BrokenDownResults/RF/Random_Forest_true_positives.csv')
rf_fp = pd.read_csv('BrokenDownResults/RF/Random_Forest_false_positives.csv')
rf_tn = pd.read_csv('BrokenDownResults/RF/Random_Forest_true_negatives.csv')
rf_fn = pd.read_csv('BrokenDownResults/RF/Random_Forest_false_negatives.csv')

svm_tp = pd.read_csv('BrokenDownResults/SVM/SVM_true_positives.csv')
svm_fp = pd.read_csv('BrokenDownResults/SVM/SVM_false_positives.csv')
svm_tn = pd.read_csv('BrokenDownResults/SVM/SVM_true_negatives.csv')
svm_fn = pd.read_csv('BrokenDownResults/SVM/SVM_false_negatives.csv')


def compare_and_save(gb_df, rf_df, svm_df, category):
    if 'ID' in gb_df.columns and 'ID' in rf_df.columns and 'ID' in svm_df.columns:
        # Find overlap and unique samples
        overlap = pd.merge(pd.merge(gb_df, rf_df, on='ID', suffixes=('_gb', '_rf')), svm_df, on='ID')
        gb_only = gb_df[~gb_df['ID'].isin(overlap['ID'])]
        rf_only = rf_df[~rf_df['ID'].isin(overlap['ID'])]
        svm_only = svm_df[~svm_df['ID'].isin(overlap['ID'])]

        # Save to CSV files
        # uncomment to save files
        # overlap.to_csv(f'{category}_overlap.csv', index=False)
        # gb_only.to_csv(f'{category}_gb_only.csv', index=False)
        # rf_only.to_csv(f'{category}_rf_only.csv', index=False)
        # svm_only.to_csv(f'{category}_svm_only.csv', index=False)

        # Calculate and print percentages
        total_gb = len(gb_df)
        total_rf = len(rf_df)
        total_svm = len(svm_df)
        overlap_percentage = len(overlap) / min(total_gb, total_rf, total_svm) * 100

        print(f"{category.capitalize()} Overlap Percentage: {overlap_percentage:.2f}%")
        print(f"Total {category} in Gradient Boosting: {total_gb}")
        print(f"Total {category} in Random Forest: {total_rf}")
        print(f"Total {category} in SVM: {total_svm}")
        print(f"Total Overlap: {len(overlap)}\n")

        # Save word count plots
        # uncomment to save word count graphs
    #     gb_only_text = ' '.join(gb_only['Text'].tolist())
    #     rf_only_text = ' '.join(rf_only['Text'].tolist())
    #     svm_only_text = ' '.join(svm_only['Text'].tolist())
    #     overlap_text = ' '.join(overlap['Text'].tolist())
    #
    #     for text, label in zip([gb_only_text, rf_only_text, svm_only_text, overlap_text],
    #                            ['gb_only', 'rf_only', 'svm_only', 'overlap']):
    #         plot_word_count(text, f'{category}_{label}_word_count.png')
    # else:
    #     print("ID column missing in one of the DataFrames")


def plot_word_count(text, output_file):
    from collections import Counter
    from wordcloud import STOPWORDS

    words = [word for word in text.split() if word.lower() not in STOPWORDS]
    word_counts = Counter(words)

    plt.figure(figsize=(10, 6))
    plt.bar(*zip(*word_counts.most_common(20)))
    plt.xticks(rotation=45)
    plt.title('Top 20 Words')
    plt.xlabel('Words')
    plt.ylabel('Counts')
    plt.savefig(output_file)
    plt.close()


# Compare and save for True Positives
compare_and_save(gb_tp, rf_tp, svm_tp, 'true_positives')

# Compare and save for False Positives
compare_and_save(gb_fp, rf_fp, svm_fp, 'false_positives')

# Compare and save for True Negatives
compare_and_save(gb_tn, rf_tn, svm_tn, 'true_negatives')

# Compare and save for False Negatives
compare_and_save(gb_fn, rf_fn, svm_fn, 'false_negatives')
