import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

# Load data
df = pd.read_csv("spotify_songs.csv")

# Model trained using Cross Validation and Ridge Regulariation
y = df["track_popularity"]

X = df[[
    "playlist_genre",
    "playlist_subgenre",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms"
]]

X = pd.get_dummies(X, columns=["playlist_genre", "playlist_subgenre"], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
scaler.fit(X_train)

X_train_standardized = scaler.transform(X_train)
X_test_standardized = scaler.transform(X_test)

l2_lambdas = np.logspace(-5, 5, 11)
param_grid = {"alpha": l2_lambdas}

search = GridSearchCV(
    estimator=Ridge(),
    param_grid=param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    return_train_score=True
)

search.fit(X_train_standardized, y_train)

best_model = search.best_estimator_

pred = best_model.predict(X_test_standardized)

print("Best alpha:", search.best_params_["alpha"])
print("Best CV RMSE:", -search.best_score_)
print("Test R²:", r2_score(y_test, pred))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, pred)))

from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error

# 2. Which predictors matter most?

coef_table = pd.DataFrame({
    "feature": X_train.columns,
    "coefficient": best_model.coef_,
    "abs_coefficient": np.abs(best_model.coef_)
}).sort_values(by="abs_coefficient", ascending=False)

print("\nCoefficient table (sorted by absolute magnitude):")
print(coef_table)

perm = permutation_importance(
    best_model,
    X_test_standardized,
    y_test,
    n_repeats=20,
    random_state=42,
    scoring="neg_root_mean_squared_error"
)

perm_table = pd.DataFrame({
    "feature": X_train.columns,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std
}).sort_values(by="importance_mean", ascending=False)

print("\nPermutation importance table:")
print(perm_table)

