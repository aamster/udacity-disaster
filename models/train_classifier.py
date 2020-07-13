import sys
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score


def load_data(database_filepath):
    """
    Loads data from sqlllite db
    :param database_filepath:
    :return: (data, labels, category names)
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('InsertTableName', engine)
    X = df['message']
    Y = df[[c for c in df if c not in ['id', 'message', 'original', 'genre']]]
    category_names = [c for c in Y]
    return X, Y, category_names


def tokenize(text):
    """
    Tokenizes string by performing following steps:
    1) splits string into tokens
    2) lemmatizes
    3) normalizes
    4) strips whitespace

    :param text: string text
    :return: list of tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Model pipeline
    :return: sklearn pipeline
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(1,2), max_df=.5)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])

    #######################
    # GridSearchCV was used in notebooks/ML Pipeline Preparation.ipynb to find optimal parameters
    # NOTE: TAKES 20 MINUTES TO RUN
    # UNCOMMENT TO WAIT 20 MINUTES
    #######################
    # parameters = {
    #     'vect__ngram_range': ((1, 1), (1, 2)),
    #     'vect__max_df': (0.5, 1.0),
    #     'clf__estimator__max_depth': [None, 3]
    # }
    #
    # pipeline = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    For each target that we want to predict, outputs the following:
    accuracy, precision, recall

    :param model: trained model
    :param X_test:
    :param Y_test:
    :param category_names:
    :return: None
    """
    y_pred = model.predict(X_test)

    for i, topic in enumerate(category_names):
        print('==============')
        print(f'Topic: {topic}')
        accuracy = accuracy_score(y_pred=y_pred[:, i], y_true=Y_test.iloc[:, i])
        precicsion = precision_score(y_pred=y_pred[:, i], y_true=Y_test.iloc[:, i])
        recall = recall_score(y_pred=y_pred[:, i], y_true=Y_test.iloc[:, i])
        print(f'accuracy: {accuracy}')
        print(f'precicsion: {precicsion}')
        print(f'recall: {recall}')
        print('\n')


def save_model(model, model_filepath):
    """
    Saves model to binary pickle file
    :param model: trained model
    :param model_filepath:
    :return: None
    """
    file = open(model_filepath, "wb")
    pickle.dump(model, file=file)


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