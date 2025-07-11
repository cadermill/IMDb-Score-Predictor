import numpy as np
import pandas as pd
import torch
from torch import nn
# seperate into train and teast
titles = pd.read_csv('cleanedTitles.csv')
titles['split'] = np.random.randn(titles.shape[0],1)
mask = np.random.rand(len(titles)) <= 0.8
train = titles[mask]
test = titles[~mask]
print("Train shape:", train.shape)
print("Test shape:", test.shape)

# seperate inputs and outputs
#trainIns = train[['numVotes','directors','writers','titleType','primaryTitle','startYear','runtimeMinutes','genres']]
#trainOuts = train[['averageRating']]
#testIns = test[['numVotes','directors','writers','titleType','primaryTitle','startYear','runtimeMinutes','genres']]
#testOuts = test[['averageRating']]
## ^above is pointless until we start using feed-forward, but will be useful later

#basicLinear = nn.Linear(in_features = 8, out_features = 1)
# Columns to use
features = ['numVotes','directors','writers','titleType','startYear','runtimeMinutes','genres']

# Label encode categorical columns
for col in ['directors', 'writers', 'titleType', 'genres']:
    train[col], uniques = pd.factorize(train[col])
    test[col] = test[col].map({cat: idx for idx, cat in enumerate(uniques)}).fillna(-1).astype(int)

# Fill missing values and convert to float
trainIns = train[features].fillna(0).astype(float)
trainOuts = train[['averageRating']].fillna(0).astype(float)
testIns = test[features].fillna(0).astype(float)
testOuts = test[['averageRating']].fillna(0).astype(float)
# Convert to tensors
X_train = torch.tensor(trainIns.values, dtype=torch.float32)
y_train = torch.tensor(trainOuts.values, dtype=torch.float32)
X_test = torch.tensor(testIns.values, dtype=torch.float32)
y_test = torch.tensor(testOuts.values, dtype=torch.float32)

# Define model
model = nn.Sequential(
    nn.Linear(len(features), 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    preds = model(X_test)
    test_loss = criterion(preds, y_test)
    print(f"Test Loss: {test_loss.item():.4f}")
