import numpy as np
import pandas as pd
import torch
from torch import nn
<<<<<<< HEAD
from sklearn.preprocessing import StandardScaler

# Load and split data
=======
# seperate into train and teast
>>>>>>> 122ed3d15d91a992b099d1ac54c1a13aabae55f8
titles = pd.read_csv('../data/cleanedTitles.csv')
titles['split'] = np.random.randn(titles.shape[0],1)
mask = np.random.rand(len(titles)) <= 0.8
train = titles[mask].copy()
test = titles[~mask].copy()
print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Columns to use
features = ['primaryTitle', 'numVotes', 'directors','writers','titleType','startYear','runtimeMinutes','genres']

# Label encode categorical columns
for col in ['primaryTitle','directors', 'writers', 'titleType', 'genres']:
    train.loc[:, col], uniques = pd.factorize(train[col])
    test.loc[:, col] = test[col].map({cat: idx for idx, cat in enumerate(uniques)}).fillna(-1).astype(int)

# Fill missing values
for col in ['numVotes', 'startYear', 'runtimeMinutes']:
    median = train[col].median()
    train.loc[:, col] = train[col].fillna(median)
    test.loc[:, col] = test[col].fillna(median)
for col in ['directors', 'writers', 'titleType', 'genres']:
    train.loc[:, col] = train[col].fillna(-1)
    test.loc[:, col] = test[col].fillna(-1)

# Feature scaling

scaler = StandardScaler()
train[features] = scaler.fit_transform(train[features])
test[features] = scaler.transform(test[features])

# Prepare inputs and outputs
trainIns = train[features]
trainOuts = train[['averageRating']].fillna(train['averageRating'].median())
testIns = test[features]
testOuts = test[['averageRating']].fillna(train['averageRating'].median())

# Convert to tensors
X_train = torch.tensor(trainIns.values, dtype=torch.float32)
y_train = torch.tensor(trainOuts.values, dtype=torch.float32)
X_test = torch.tensor(testIns.values, dtype=torch.float32)
y_test = torch.tensor(testOuts.values, dtype=torch.float32)

# Model!!!!!
model = nn.Sequential(
    nn.Linear(len(features), 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 350
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    preds = model(X_test)
    test_loss = criterion(preds, y_test)
    print(f"Test Loss: {test_loss.item():.4f}")
