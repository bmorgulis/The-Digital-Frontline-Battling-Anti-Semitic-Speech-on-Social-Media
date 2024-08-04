import pandas as pd  # Import pandas for data manipulation
import re  # Import regular expressions for text cleaning
from nltk.tokenize import word_tokenize  # Import tokenizer to split text into words
from nltk.stem import WordNetLemmatizer  # Import lemmatizer for reducing words to their base form
import nltk  # Import the Natural Language Toolkit

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()


# Function to clean the text
def clean_text(text):
    # Remove mentions (e.g., @username)
    text = re.sub(r'@\w+', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation, numbers, and special characters
    text = re.sub(r'[^a-z\s]', '', text)
    # Strip leading and trailing whitespace
    text = text.strip()
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Tokenize the text into words
    words = word_tokenize(text)
    # Lemmatize each word to reduce it to its base form
    words = [lemmatizer.lemmatize(word) for word in words]
    # code for stemming which we are not using as explained in notes
    #   words = [stemmer.stem(word) for word in words]
    # Join the words back into a single string
    text = ' '.join(words)
    return text


# Load the dataset from a CSV file
df = pd.read_csv('Smaller_Jikeli_dataset.csv')

# Clean the 'Text' column and create a new column 'Cleaned_Text'
df['Cleaned_Text'] = df['Text'].apply(clean_text)

# Save the cleaned dataset to a new CSV file
df.to_csv('Cleaned_Smaller_Jikeli_dataset.csv', index=False)

