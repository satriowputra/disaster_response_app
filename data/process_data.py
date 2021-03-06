import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load two datasets and merging those two.
    
    Args:
    messages_filepath: filepath to message dataset
    categories_filepath: filepath to category dataset
    
    Returns:
    Merged datasets
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = pd.merge(messages, categories, on='id')
    return df

def clean_data(df):
    """Cleaning dataset, drop missing value, processing categorical values,
    removing duplicates, and much more,
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].copy().str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.loc[0].apply(lambda x : x[:-2])
    category_colnames = row.tolist()

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x : x[-1])

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)

    # Remove duplicates
    df.drop_duplicates(subset=['id', 'message'], inplace=True)
    df.drop(df[df['message'].duplicated() == True].index, inplace=True)

    # Change some value in related column
    df['related'].replace(2, 0, inplace=True)
    return df

def save_data(df, database_filename):
    """Save the clean dataset into an sqlite database"""
    file_name = 'sqlite:///' + database_filename
    engine = create_engine(file_name)
    df.to_sql('df', engine, index=False)


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
