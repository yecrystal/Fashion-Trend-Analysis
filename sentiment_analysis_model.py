import pandas as pd
import numpy as np
import re
import os
import emoji
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

def load_data(filepath):
    """
    Load data from a file into a pandas DataFrame. 
    
    Args: pd.DataFrame: The data to clean
    
    Returns: pd.DataFrame: The cleaned data
    """
    df = pd.read_excel(filepath)
    return df

def clean_data(df):
    """
    Clean the data by removing any rows with missing values. 
    
    Args: pd.DataFrame: The data to clean
    
    Returns: pd.DataFrame: The cleaned data
    """
    df = df.drop_duplicates(subset='comments', keep='first')
    df = df.drop(columns=608, axis=1)
    return df

def tokenize(text):
    """
    Tokenize the input text, converting it to lowercase and removing any stopwords.
    
    Args: str: The text to tokenize
    
    Returns: list: The list of tokens
    """
    if text is None:
        return None
    
    # Convert emojis to text
    text = emoji.demojize(text, delimiters=("", ""))

    # Replace unrecognized emoji (left as ::) with space
    text = re.sub('::', ' ', text)

    # Tokenize text
    tokens = word_tokenize(text)

    # Remove English stopwords
    tokens = [token for token in tokens if token not in stopwords.words('english')]

    return tokens

def split_data(df):
    """
    Split the data into training and testing sets.
    
    Args: pd.DataFrame: The data to split
    
    Returns: list: The split data [X_train, X_test, y_train, y_test]
    """
    df2 = df[df['category'].notnull()]
    X = df2.comments
    y = df2.category
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_model():
    """
    Creates a machine learning pipeline for multiclass classification
    
    Returns: obj: The machine learning model that will be used to classify the messages
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('mof', AdaBoostClassifier())
    ])

    params = {
        'tfidf__use_idf': (True, False),
        'mof__n_estimators': [50, 60, 70],
    }

    return GridSearchCV(pipeline, param_grid=params)