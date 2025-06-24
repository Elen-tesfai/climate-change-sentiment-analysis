import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK data files (if not already installed)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the cleaned data from 'cleaned_data.csv'
df = pd.read_csv('data/cleaned_data.csv')

# Show the first few rows of the dataframe
print("Data preview:")
print(df.head())

# Tokenization
df['tokens'] = df['text'].apply(word_tokenize)
print("Tokens after tokenization:")
print(df['tokens'].head())

# Remove stopwords
stop_words = set(stopwords.words('english'))
df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word.lower() not in stop_words])  # Ensure case-insensitivity
print("Tokens after stopword removal:")
print(df['tokens'].head())

# Lemmatization
lemmatizer = WordNetLemmatizer()
df['tokens'] = df['tokens'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
print("Tokens after lemmatization:")
print(df['tokens'].head())

# Join tokens back into a string for the cleaned text
df['cleaned_text'] = df['tokens'].apply(lambda x: ' '.join(x))
print("Cleaned text preview:")
print(df[['text', 'cleaned_text']].head())

# Remove rows where the cleaned text is empty
df = df[df['cleaned_text'].str.strip().notna()]

# Check the shape of the cleaned data
print("Data after cleaning:")
print(df.shape)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['cleaned_text']).toarray()

# Labels (sentiment)
y = df['label']

# You can now print the first few rows of the transformed features and labels
print("Vectorized text sample (X):")
print(X[:5])  # Show the first 5 rows of vectorized text

print("Sentiment labels (y):")
print(y[:5])  # Show the first 5 sentiment labels