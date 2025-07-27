import os
import pandas as pd

def load_data():
    if os.path.exists('../data/cleanedTitles.csv'):
        print("Loading data from cleanedTitles.csv...")
        return pd.read_csv('../data/cleanedTitles.csv', low_memory=False)
    else:
        print("Processing data files...")
        # 1,588,240 rows
        # tconst    averageRating   numVotes
        # tt0000001 5.7             2165
        # str       float64         int64
        ratings = pd.read_csv('../data/title.ratings.tsv.gz', sep='\t', compression='gzip', low_memory=False, on_bad_lines='skip')

        # 11,769,857 rows
        # tconst    titleType   primaryTitle    originalTitle   isAdult     startYear   endYear runtimeMinutes  genres
        # tt0000001 short       Carmencita      Carmencita      0           1894        \N      1               Documentary,Short
        # str       str         str             str             str         str         str     str                             str
        basics = pd.read_csv('../data/title.basics.tsv.gz', sep='\t', compression='gzip', low_memory=False, on_bad_lines='skip')

        # 11,771,649 rows
        # tconst     directors    writers
        # tt0000001  nm0005690    \N
        # str        str          str
        crew = pd.read_csv('../data/title.crew.tsv.gz', sep='\t', compression='gzip', low_memory=False, on_bad_lines='skip')

        # merge ratings, crew, and basics on 'tconst'
        titles = pd.merge(ratings, crew, on='tconst', how='left')
        titles = pd.merge(titles, basics, on='tconst', how='left')

        # selcect only the columns we need
        titles = titles[['tconst', 'averageRating', 'numVotes', 'directors', 'writers', 'titleType', 'primaryTitle', 'startYear', 'runtimeMinutes', 'genres']]

        # convert \N to NA
        titles.replace('\\N', pd.NA, inplace=True)

        # convert to appropriate data types
        titles['averageRating'] = pd.to_numeric(titles['averageRating'], errors='coerce')
        titles['numVotes'] = pd.to_numeric(titles['numVotes'], errors='coerce')
        titles['startYear'] = pd.to_numeric(titles['startYear'], errors='coerce')
        titles['runtimeMinutes'] = pd.to_numeric(titles['runtimeMinutes'], errors='coerce')

        # save the cleaned data
        titles.to_csv('../data/cleanedTitles.csv', index=False)

        # load names data for directors and writers conversion
        names = pd.read_csv('../data/name.basics.tsv.gz', sep='\t', compression='gzip', low_memory=False, on_bad_lines='skip')
        print(names.head())

        return titles

titles = load_data()