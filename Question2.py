import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv("spotify_songs.csv")

# 1. Define the audio and categorical features
keep_features = [
    'playlist_genre', 'danceability', 'energy', 'key', 'loudness', 
    'mode', 'speechiness', 'acousticness', 'instrumentalness', 
    'liveness', 'valence', 'tempo', 'duration_ms'
]

# 2. Build the formula for track_popularity
formula = 'track_popularity ~ ' + ' + '.join(keep_features)

# 3. Fit the model
model = smf.ols(formula, data=df).fit()
print(model.summary())

# 4. Visualization (Residuals)
plt.figure(figsize=(8, 5))
sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, line_kws={'color': 'red'})
plt.xlabel('Predicted Popularity')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Popularity')
plt.show()