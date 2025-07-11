import pandas as pd

# 1,588,240 rows
# tconst    averageRating   numVotes
# tt0000001 5.7             2165
# str       float64         int64
# ratings = pd.read_csv('../data/title.ratings.tsv.gz', sep='\t', compression='gzip', low_memory=False)

# 11,769,857 rows
# tconst    titleType   primaryTitle    originalTitle   isAdult     startYear   endYear runtimeMinutes  genres
# tt0000001 short       Carmencita      Carmencita      0           1894        \N      1               Documentary,Short
# str       str         str             str             str         str         str     str                             str
# basics = pd.read_csv('../data/title.basics.tsv.gz', sep='\t', compression='gzip', low_memory=False)


# 11,771,649 rows
# tconst     directors    writers
# tt0000001  nm0005690    \N
# str        str          str
# crew = pd.read_csv('../data/title.crew.tsv.gz', sep='\t', compression='gzip', low_memory=False)