import sys
import nltk
nltk.download(['punkt', 'wordnet'])
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy as sqla
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import sqlalchemy as sqla
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def load_data(database_filepath):
    '''
    Load string column from database
    
    ARGS:
        database_filepath: String. Name of the database containing the data table
        table_name: String. Sql data table
    Returns:
        X: (array) Disaster tweets
        Y: (array) Categories of the tweet
        cat_name: (list) Column names of the disaster categories
    '''
    engine = sqla.create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('clean_tweet_with_cats', engine)

    X = df.message.values
    Y = df.iloc[:, 4:].values

    category_names = (df.columns[4:]).tolist()

    return X, Y, category_names


def tokenize(text):
    '''
    Function that cleans and tokenize a list of texts
    
    Args:
       text: A list of text
    
    Returns:
        clean_tokens
    '''
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # iterate through each token
    clean_tokens = []
    for token in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs = -1))
    ])
    
    parameters = {
    'clf__estimator__n_estimators': [20, 50],
    'clf__estimator__min_samples_split': [2, 4, 5]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose = 2, n_jobs = -1)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    y_preds = model.predict(X_test)
    print("----Classification Report per Category:\n")
    for i in range(len(category_names)):
        print("Label:", category_names[i])
        print(classification_report(Y_test[:, i], y_preds[:, i]))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as outfile:
        pickle.dump(model, outfile)


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