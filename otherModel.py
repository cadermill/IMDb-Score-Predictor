import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

titles = pd.read_csv('../data/cleanedTitles.csv')
titles = titles.dropna()

categorical_features = ['titleType']
titles = pd.get_dummies(titles, columns=categorical_features, drop_first=True)

for col in ['directors', 'genres']:
    titles[col], _ = pd.factorize(titles[col])

#['tconst', 'averageRating', 'numVotes', 'directors', 'writers', 'titleType', 'primaryTitle', 'startYear', 'runtimeMinutes', 'genres']
X = titles.drop(columns=['averageRating', 'tconst', 'numVotes', 'primaryTitle', 'writers'])
Y = titles['averageRating']

for col in X.columns:
    if X[col].dtype.kind in 'biufc':  # numeric columns
        X[col] = X[col].fillna(X[col].median())
    else:  # categorical columns
        X[col] = X[col].fillna(-1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')