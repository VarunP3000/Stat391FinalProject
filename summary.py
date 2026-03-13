import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


# load data
songs = pd.read_csv('spotify_songs.csv')

# sample

# 1. Shape & Missing Values
print(f"Dataset Shape: {songs.shape}")
print(f"Missing Values:\n{songs.isnull().sum()}\n")

# 2. Five Number Summary + Mean/Std
# This gives you Min, 25%, 50%, 75%, and Max automatically
print("Popularity Stats:")
print(songs['track_popularity'].describe())

# 3. Specifically check for the 'Zero' spike
zero_pop_pct = (songs['track_popularity'] == 0).mean() * 100
print(f"\nPercentage of songs with 0 popularity: {zero_pop_pct:.2f}%")

f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize=(10,6))

f.suptitle('Distribution of Track Popularity', fontsize=16)

# Use this to make sure the supertitle doesn't overlap the top plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

sns.boxplot(x=songs["track_popularity"], ax=ax_box)

sns.histplot(x=songs["track_popularity"], ax=ax_hist, kde=True)
ax_box.set(xlabel='')
plt.savefig('pop_dist.png', bbox_inches='tight')
plt.show()