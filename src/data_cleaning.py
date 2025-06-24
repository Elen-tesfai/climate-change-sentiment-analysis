import pandas as pd
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK data files (if not already installed)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def clean_data(dataframe):
    """
    Function to clean the dataset.
    - Remove null values.
    - Normalize text (e.g., lowercase).
    - Tokenization, stopword removal, and lemmatization.
    """
    # Rename columns if necessary
    dataframe.columns = ['text', 'label']  # Fixing the column names based on the dataset

    # Remove rows with missing data
    dataframe = dataframe.dropna()

    # Ensure all values in 'text' column are strings and lowercase all text data
    dataframe['text'] = dataframe['text'].astype(str).str.lower()

    # Tokenization
    dataframe['tokens'] = dataframe['text'].apply(word_tokenize)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    dataframe['tokens'] = dataframe['tokens'].apply(lambda x: [word for word in x if word.lower() not in stop_words])

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    dataframe['tokens'] = dataframe['tokens'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

    # Join tokens back into a string for the cleaned text
    dataframe['cleaned_text'] = dataframe['tokens'].apply(lambda x: ' '.join(x))

    return dataframe

if __name__ == "__main__":
    # Define output directory for saving cleaned data
    output_directory = r'C:\Users\su_te\Documents\climate-change-sentiment-analysis\data'
    
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    try:
        # Attempt to read the CSV file using utf-8-sig (handles BOM in UTF-8)
        df = pd.read_csv(r'C:\Users\su_te\Documents\climate-change-sentiment-analysis\data\sample_data.csv', 
                         encoding='utf-8-sig',  # Attempt utf-8-sig to handle BOM
                         on_bad_lines='skip')  # Skip bad lines
        print("File loaded successfully")
        
        # Clean the data
        cleaned_data = clean_data(df)
        
        # Save the cleaned data to a new CSV
        cleaned_data.to_csv(os.path.join(output_directory, 'cleaned_data.csv'), index=False)
        print("Data cleaning complete. Cleaned data saved as 'cleaned_data.csv'")
    
    except UnicodeDecodeError as e:
        print(f"Error: Encoding issue: {e}")
        # Try reading with 'ISO-8859-1' encoding as a fallback
        print("Attempting to read the file with 'ISO-8859-1' encoding...")
        try:
            df = pd.read_csv(r'C:\Users\su_te\Documents\climate-change-sentiment-analysis\data\sample_data.csv', 
                             encoding='ISO-8859-1',  # Fallback to ISO-8859-1
                             on_bad_lines='skip')
            print("File loaded successfully with ISO-8859-1 encoding")
            
            # Clean the data
            cleaned_data = clean_data(df)

            # Save the cleaned data to a new CSV
            cleaned_data.to_csv(os.path.join(output_directory, 'cleaned_data.csv'), index=False)
            print("Data cleaning complete. Cleaned data saved as 'cleaned_data.csv'")
        
        except Exception as e:
            print(f"Unexpected error: {e}")
    
    except Exception as e:
        print(f"Unexpected error: {e}")