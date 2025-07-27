import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

titles = pd.read_csv('data/cleanedTitles.csv')
titles = titles.dropna()

print(titles.iloc[123456])

features = ['numVotes', 'directors', 'titleType', 'startYear', 'runtimeMinutes', 'genres', 'writers']
titles = titles[features + ['averageRating']].copy()

# One-hot encode categorical columns
categorical_features = ['titleType']
titles = pd.get_dummies(titles, columns=categorical_features, drop_first=True)

# Factorize all categorical columns
factorize_maps = {} # store mappings to convert new titles later
for col in ['directors', 'genres', 'writers']:
    titles[col], uniques = pd.factorize(titles[col])
    factorize_maps[col] = list(uniques)

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

def predict_rating(title_dict):
    # convert to DataFrame
    title = pd.DataFrame([title_dict])

    # one hot encode titleType
    for col in X.columns:
        if col.startswith('titleType_'):
            title[col] = 0
    if f"titleType_{title_dict['titleType']}" in title.columns:
        title[f'titleType_{title_dict["titleType"]}'] = 1
    title = title.drop(columns=['titleType'], errors='ignore') # drop titleType column when done

    # factorize directors, genres, and writers
    for col in ['directors', 'genres', 'writers']:
        uniques = factorize_maps[col]
        try:
            title[col] = [uniques.index(title_dict[col])]
        except ValueError:
            title[col] = -1

    # fill missing numeric values
    for col in numeric_cols:
        if col not in title.columns:
            title[col] = titles[col].median()
    # scale numeric
    title[numeric_cols] = scaler.transform(title[numeric_cols])

    # reorder columns
    title = title[X.columns]
    print(title.iloc[0])

    # predict
    pred = model.predict(title)[0]
    return pred

new_title = {
    'numVotes': 84,
    'directors': 'nm0456862',
    'titleType': 'short',
    'startYear': 1936,
    'runtimeMinutes': 8,
    'genres': 'Fantasy,Music,Short',
    'writers': 'nm0904270'
}
pred = predict_rating(new_title)
print(f'Predicted Rating: {pred:.2f}')

print(titles.iloc[123456])