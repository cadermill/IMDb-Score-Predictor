# Random forest but with target encoding and randomized search for hyperparam tuning

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Target encoding
from category_encoders import TargetEncoder

titles = pd.read_csv('../data/cleanedTitles.csv')
titles = titles.dropna()

features = ['numVotes', 'directors', 'titleType', 'startYear', 'runtimeMinutes', 'genres', 'writers']
titles = titles[features + ['averageRating']].copy()

# Target encode high-cardinality categorical columns
target_encode_cols = ['directors', 'genres', 'writers']
encoder = TargetEncoder(cols=target_encode_cols)
titles[target_encode_cols] = encoder.fit_transform(titles[target_encode_cols], titles['averageRating'])

# One-hot encode low-cardinality categorical columns
titles = pd.get_dummies(titles, columns=['titleType'], drop_first=True)

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

print("Train shape:", X_train.shape)
print("Train target shape:", y_train.shape)

# RandomizedSearchCV for hyperparameter tuning
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    scoring='neg_mean_squared_error',
    verbose=2,
    random_state=42
)
search.fit(X_train, y_train)
model = search.best_estimator_

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.4f}')
print(f'R^2 Score: {r2:.4f}')

# Cross-validation score
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f'Cross-validated R^2 scores: {cv_scores}')
print(f'Mean CV R^2: {np.mean(cv_scores):.4f}')

# Feature importance plot
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10,6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation = 90)
plt.tight_layout()
plt.show()