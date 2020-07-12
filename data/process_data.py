import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """

    :param messages_filepath: path to message file
    :param categories_filepath: path to categories file
    :return: dataframe of merged data
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """
    Returns cleaned dataframe
    Following cleaning is done:
    1) convert response to individual categories
    2) parse integer label from categories
    3) clean related column as it has invalid value of 2 in some cases
    4) drops duplicates

    :param df: raw dataframe
    :return: cleaned dataframe
    """
    categories = df['categories'].str.split(';', expand=True)
    
    row = categories.iloc[0]
    category_colnames = [s[:-2] for s in row]
    
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    df = df.drop('categories', axis=1)
    
    df = pd.concat([df, categories], axis=1)
    
    # related column has invalid values of 2
    df.loc[df['related'] == 2, 'related'] = 0
    
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """
    Save dataframe to sqllite db
    :param df: dataframe
    :param database_filename:
    :return: None
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('InsertTableName', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
