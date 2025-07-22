import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('../data/cleanedTitles.csv')

# Convert to numeric and drop NaNs
df['averageRating'] = pd.to_numeric(df['averageRating'], errors='coerce')
ratings = df['averageRating'].dropna()

# Define 0.5-wide bins
bin_edges = np.arange(0, 10.5, 0.5)  # From 0 to 10 (inclusive) with step of 0.5
bin_labels = [f"{b:.1f}–{b+0.5:.1f}" for b in bin_edges[:-1]]  # Create labels like 8.0–8.5

# Bin the data
binned = pd.cut(ratings, bins=bin_edges, labels=bin_labels, right=False)
bin_percentages = binned.value_counts(normalize=True).sort_index() * 100

# Plot the bar chart
plt.figure(figsize=(12, 6))
bars = plt.bar(bin_percentages.index, bin_percentages.values, color='cornflowerblue', edgecolor='black')

# Standard deviation annotation
std_dev = ratings.std()
plt.text(0.95, 0.95, f"Std Dev: {std_dev:.2f}", fontsize=12,
         rotation=45, transform=plt.gca().transAxes,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(facecolor='white', alpha=0.7))

# Labels and title
plt.xlabel('Average Rating Bins')
plt.ylabel('Percentage of Titles (%)')
plt.title('Distribution of IMDb Ratings (Grouped in 0.5 Intervals)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the plot
plt.show()