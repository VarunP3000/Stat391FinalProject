import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import os

# load data
songs = pd.read_csv('spotify_songs.csv')

# sample
songs_sample = songs.sample(n=1000, random_state=314)

# drop unused columns
songs_sample = songs_sample.drop(columns=[
    'track_id',
    'track_name',
    'track_artist',
    'track_album_id',
    'track_album_name',
    'track_album_release_date',
    'playlist_name',
    'playlist_id',
    'playlist_subgenre'
])

# encode categorical variables
songs_quant = pd.get_dummies(songs_sample, drop_first=True)

# correlation matrix
corr_matrix = songs_quant.corr()

# plot
plt.figure(figsize=(16,12))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    annot_kws={"size":8}
)

plt.title('Correlation Matrix of Spotify Song Features', fontsize=16)
plt.xlabel('Features', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('my_heatmap.png', bbox_inches='tight')
plt.show()

print(f"File saved to: {os.getcwd()}")



"""
size_readings = df[['width', 'height', 'trunk', 'seating', 'length', 'mass']]
df_encoded_size = pd.get_dummies(size_readings, drop_first=True)
high_corr_matrix = df_encoded_size.corr()
sns.heatmap(high_corr_matrix, annot=True, cmap="coolwarm", square=True)
plt.show()

df = df.drop(columns = ['make', 'model', 'width', 'height', 'seating', 'trunk', 'length'])

# encode data for corr matrix
df_encoded = pd.get_dummies(df, drop_first=True)
corr_matrix = df_encoded.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", square=True)
plt.show()

# distribution for outlier removal
print(df['price'].describe())

# remove outliers
price_filtered = df[df['price'] < 5500000]

# boxplot of distributions
plt.boxplot(np.log(price_filtered['price']))
plt.show()

# histogram of distributions
plt.hist(np.log(price_filtered['price']), bins=20, edgecolor='black')
plt.xlabel('Log(Price)')
plt.ylabel('Frequency')
plt.title('Distribution of Log(Price)')
plt.show()
"""
