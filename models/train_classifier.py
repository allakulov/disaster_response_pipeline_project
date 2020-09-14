# Import necessary libraries
import sys

#pandas
import pandas as pd 
import numpy as np

#SQL toolkit 
from sqlalchemy import create_engine

#regex module
import re

#relevant packages from nltk
import nltk 
nltk.download(['wordnet', 'punkt', 'stopwords'])
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer

# relevant modules from scikit-learn 
from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,  GridSearchCV 
from sklearn.metrics import classification_report

# pickle module 
import pickle

def load_data(database_filepath):
    """
    Loads data from SQL Database
    
    Parameters:
    database_filepath (str): SQL database file name
    
    Returns:
    X (DataFrame): Features dataframe
    Y (DataFrame): Target dataframe
    category_names (list): Target labels 
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("messages_disaster", con=engine)
    
    category_names = df.columns[4:]

    X = df[['message']].values[:, 0]
    Y = df[category_names].values
    
    return X, Y, category_names 

def tokenize(text):
    """
    Tokenizes text data
    
    Parameters:
    text (str): Text containing Twitter messages
    
    Returns:
    clean_tokens (list): Processed text after normalizing, tokenizing and lemmatizing
    """
    
    tokens = word_tokenize(text)
    to_remove = stopwords.words("english")
    tokens = [token for token in tokens if token not in to_remove]
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    - Builds a ML model using Decision Tree Classifier
    - Tunes it with GridSearchCV
    
    Returns:
    Trained and tuned ML model 
    """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',  MultiOutputClassifier(DecisionTreeClassifier()))
    ])
    
    parameters = {
        'tfidf__use_idf': (True, False),
        'clf__estimator__max_depth': [2, 5, 10]
    }

    model = GridSearchCV(pipeline, param_grid = parameters)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Shows model's performance on test data
    
    Parameters:
    model: ML model returned from build_model()
    X_test: Test features
    Y_test: Test targets
    category_names: Target labels
    """
    
    # predict
    y_pred = model.predict(X_test)

    # print classification report
    print(classification_report(Y_test, y_pred, target_names=category_names))

    # print accuracy score
    print('Accuracy: {}'.format(np.mean(y_pred == Y_test)))


def save_model(model, model_filepath):
    """
    Saves the ML model to a Python pickle file    
    
    Parameters:
    model: ML model returned from build_model()
    model_filepath: Filepath to save the model
    """

    # save the ML model to pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()