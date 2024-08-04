import nltk
import string
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def read_excel_file(file_path, column_name, num_lines):
    print("=====================================")
    print(f"Reading the excel file from '{file_path}'...")
    print("=====================================")
    data = pd.read_excel(file_path, header=0)
    return data[column_name][:num_lines]



print("Preprocessing the tweet...")
print("=====================================")
# Preprocessing the tweet
def preprocess_tweet(tweet):
    # Converting the tweet to string(in case blank line is read as float)
    tweet = str(tweet)
    # Converting the tweet to lowercase
    tweet = tweet.lower()
    # Skip blank lines
    if tweet.strip() == "":
        return []
    # Removing URLs
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Removing user mentions
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    # Removing punctuation
    punctuation = r"""!"#$%&'()*+,./:;<=>?@[\]^`{|}~–—‘’“”…"""    
    tweet = tweet.translate(str.maketrans('', '', punctuation))
    # tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # Tokenizing the tweet
    tweet_tokens = word_tokenize(tweet)
    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tweet_tokens if word not in stop_words]
    # removing words leading and trailing '-' and '_'
    filtered_tokens = [word.strip('-_') for word in filtered_tokens]
    return filtered_tokens



def preprocess_tweet_list(data, column_name, num_lines):
    tweetList = [preprocess_tweet(tweet) for tweet in data if not pd.isnull(tweet)]
    return tweetList



# Count the total number of words in the tweetList
def count_total_words(tweetList):
    total_words = 0
    for tweet in tweetList:
        total_words += len(tweet)
    return total_words


def calculate_total_words(tweetList):
    total_words = count_total_words(tweetList)
    print(f"Total number of words in tweetList: {total_words}")
    print("=====================================")
    return total_words



def extract_unique_words(tweetList):
    print("extracting the unique words from the tweetlist...")
    print("=====================================")
    uniqueWords = sorted(set([word for tweet in tweetList for word in tweet]))
    print(uniqueWords)
    total_unique_words = len(uniqueWords)
    print(f"Total number of unique words in tweetList: {total_unique_words}")
    print("=====================================")



# Count the number of times each word appears in the tweetList
def getWordCount(tweetList):
    word_count_dict = {}
    for tweet in tweetList:
        for word in tweet:
            if word in word_count_dict:
                word_count_dict[word] += 1
            else:
                word_count_dict[word] = 1
    return word_count_dict
         
          
            
def sort_word_count(word_count_dict):
    sorted_word_count = sorted(word_count_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_word_count


def print_word_count(sorted_word_count):
    for word, count in sorted_word_count:
        print(f"'{word}' = {count} times.")

            

def write_word_count_to_file(write_to_file, sorted_word_count, column_name):
    print("=====================================")
    print("Writing the word count to a file...")
    print("=====================================")

    # Open the file in write mode
    with open(write_to_file, "w", encoding='utf-8') as file:
        file.write(column_name + " word count:\n")
    # Write the word count to the file
        for word, count in sorted_word_count:
            file.write(f"{word}: {count}\n")

    print("Anti semitic word count written to file: 'jikeli_word_count.txt'")
    print("=====================================")



# Calculate the percentage of each word in the tweetList
def get_word_percentage(word_count_dict, total_words):
    print("Calculating the percentage of each word in the tweetList...")
    word_percentage_dict = {}
    for word, count in word_count_dict.items():
        word_percentage_dict[word] = round((count / total_words) * 100, 7)

    sorted_word_percentage = sorted(word_percentage_dict.items(), key=lambda x: x[1], reverse=True)

    for word, percentage in sorted_word_percentage:
        print(f"'{word}' = {percentage} percent.")
        
        

def main():
    """
    This function is the entry point of the program.
    It prompts the user for input, calls necessary functions,
    and performs word count analysis on a given dataset.
    """
    column_name = input("Enter column name(Anti Semitic or Non Anti Semitic) or press enter: ") or "Text"
    num_lines = int(input("Enter the number of lines to read or press enter(default = 11312): ") or 11312)
    write_to_file = input("Enter the file name to write the word count to or press enter(default = jikeli_word_count.txt): ") or 'jikeli_word_count.txt'

    data = read_excel_file('ISCA Jikeli GoldStandard2024.xlsx', column_name, num_lines)
    tweetList = preprocess_tweet_list(data, column_name, num_lines)
    total_words = calculate_total_words(tweetList)
    extract_unique_words(tweetList)
    word_count_dict = getWordCount(tweetList)
    sorted_word_count = sort_word_count(word_count_dict)
    print_word_count(sorted_word_count)
    write_word_count_to_file(write_to_file, sorted_word_count, column_name)
    get_word_percentage(word_count_dict, total_words)
        

main()



