import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

titles = pd.read_csv('../data/cleanedTitles.csv')

features = ['numVotes', 'directors', 'titleType', 'startYear', 'runtimeMinutes', 'genres']
titles = titles[features + ['averageRating']].copy()

# Factorize all categorical columns
for col in ['directors', 'titleType', 'genres']:
    titles[col], _ = pd.factorize(titles[col])

# Fill missing values
for col in titles.columns:
    if titles[col].dtype.kind in 'biufc':
        titles[col] = titles[col].fillna(titles[col].median())
    else:
        titles[col] = titles[col].fillna(-1)

# Scale numeric features
scaler = StandardScaler()
numeric_cols = ['numVotes', 'startYear', 'runtimeMinutes']
titles[numeric_cols] = scaler.fit_transform(titles[numeric_cols])

X = titles.drop(columns=['averageRating'])
y = titles['averageRating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(y_train.shape)

model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.4f}')
print(f'R^2 Score: {r2}')