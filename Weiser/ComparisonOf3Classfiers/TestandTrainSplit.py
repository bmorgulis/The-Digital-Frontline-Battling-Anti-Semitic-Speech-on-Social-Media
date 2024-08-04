import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = 'TestandTrainDataSets/JikeliDataset.csv'  # replace with your actual file path
df = pd.read_csv(file_path)

# Ensure the data is loaded correctly
print(df.head())

# Split the dataset into training and testing sets (80-20 split)
train_df, test_df = train_test_split(df, test_size=0.20, stratify=df['Biased'], random_state=42)

# Save the split datasets into separate files
train_df.to_csv('training_data_Jikeli.csv', index=False)
test_df.to_csv('testing_data_Jikeli.csv', index=False)

print("Training and testing datasets have been saved successfully.")
