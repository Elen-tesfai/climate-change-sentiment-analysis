import pandas as pd
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

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
    # Define output directory for saving cleaned data and figures
    output_directory = r'C:\Users\su_te\Documents\climate-change-sentiment-analysis\data'
    
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    try:
        df = pd.read_csv(r'C:\Users\su_te\Documents\climate-change-sentiment-analysis\data\sample_data.csv', 
                         encoding='utf-8-sig',
                         on_bad_lines='skip')
        print("File loaded successfully")
        
        cleaned_data = clean_data(df)
        
        cleaned_data.to_csv(os.path.join(output_directory, 'cleaned_data.csv'), index=False)
        print("Data cleaning complete. Cleaned data saved as 'cleaned_data.csv'")

        # Plot sentiment label distribution
        plt.figure(figsize=(8,6))
        sns.countplot(x='label', data=cleaned_data)
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.tight_layout()

        # Save plot as PNG file
        plot_path = os.path.join(output_directory, 'sentiment_distribution.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Sentiment distribution plot saved as {plot_path}")

    except UnicodeDecodeError as e:
        print(f"Error: Encoding issue: {e}")
        print("Attempting to read the file with 'ISO-8859-1' encoding...")
        try:
            df = pd.read_csv(r'C:\Users\su_te\Documents\climate-change-sentiment-analysis\data\sample_data.csv', 
                             encoding='ISO-8859-1',
                             on_bad_lines='skip')
            print("File loaded successfully with ISO-8859-1 encoding")
            
            cleaned_data = clean_data(df)

            cleaned_data.to_csv(os.path.join(output_directory, 'cleaned_data.csv'), index=False)
            print("Data cleaning complete. Cleaned data saved as 'cleaned_data.csv'")

            plt.figure(figsize=(8,6))
            sns.countplot(x='label', data=cleaned_data)
            plt.title('Sentiment Distribution')
            plt.xlabel('Sentiment')
            plt.ylabel('Count')
            plt.tight_layout()

            plot_path = os.path.join(output_directory, 'sentiment_distribution.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"Sentiment distribution plot saved as {plot_path}")
        
        except Exception as e:
            print(f"Unexpected error: {e}")
    
    except Exception as e:
        print(f"Unexpected error: {e}")