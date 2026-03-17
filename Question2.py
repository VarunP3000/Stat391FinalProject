import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ── 1. Load & prep ────────────────────────────────────────────────────────────
df = pd.read_csv("spotify_songs.csv")
df = df.sample(n=1000, random_state=314)

df = df.drop(columns=[
    'track_id', 'track_name', 'track_artist',
    'track_album_id', 'track_album_name', 'track_album_release_date',
    'playlist_name', 'playlist_id',
], errors='ignore')

y = df["track_popularity"].values

X_full = df[[
    "playlist_genre", "danceability", "energy", "key", "loudness",
    "mode", "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "duration_ms"
]].copy()
X_full = pd.get_dummies(X_full, columns=["playlist_genre"], drop_first=True)

# ── 2. Define 3 candidate models ──────────────────────────────────────────────
candidate_models = {
    "M1: Audio features only": [
        "danceability", "energy", "loudness",
        "acousticness", "valence", "tempo"
    ],
    "M2: Genre + audio": [
        "danceability", "energy", "loudness", "acousticness", "valence", "tempo"
    ] + [c for c in X_full.columns if c.startswith("playlist_genre")],
    "M3: Full model": list(X_full.columns),
}

# ── 3. Fit each candidate model and compute CV(K) ─────────────────────────────
K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)

print("=" * 55)
print(f"  OLS Model Selection — CV(K={K}) for each candidate")
print("=" * 55)

cv_results = {}

for model_name, features in candidate_models.items():
    X_m = X_full[features].values.astype(float)
    fold_mses = []

    print(f"\n  {model_name}")
    print(f"  Features: {features}")
    print(f"  {'─'*50}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_m), start=1):
        X_tr,  X_val = X_m[train_idx],  X_m[val_idx]
        y_tr,  y_val = y[train_idx],    y[val_idx]

        # Scale inside each fold — no leakage
        scaler = StandardScaler()
        X_tr_s  = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        ols = LinearRegression()
        ols.fit(X_tr_s, y_tr)
        preds = ols.predict(X_val_s)

        mse = mean_squared_error(y_val, preds)
        fold_mses.append(mse)
        print(f"  Fold {fold}: MSE = {mse:.4f}")

    cv_k = np.mean(fold_mses)
    cv_results[model_name] = {"fold_mses": fold_mses, "cv_k": cv_k}
    print(f"  CV({K}) = {cv_k:.4f}  (RMSE = {np.sqrt(cv_k):.4f})")

# ── 4. Select model with smallest CV(K) ──────────────────────────────────────
print()
print("=" * 55)
print("  Model Selection Summary")
print("=" * 55)
for name, res in sorted(cv_results.items(), key=lambda x: x[1]["cv_k"]):
    marker = "  ← SELECTED" if res["cv_k"] == min(r["cv_k"] for r in cv_results.values()) else ""
    print(f"  {name:<28}  CV({K}) = {res['cv_k']:.4f}{marker}")

best_name = min(cv_results, key=lambda x: cv_results[x]["cv_k"])
print(f"\n  Selected: {best_name}")

# ── 5. Refit selected model on ALL data ───────────────────────────────────────
best_features = candidate_models[best_name]
X_best = X_full[best_features].values.astype(float)

scaler_final = StandardScaler()
X_best_scaled = scaler_final.fit_transform(X_best)
X_sm = sm.add_constant(X_best_scaled)

final = sm.OLS(y, X_sm).fit()

print()
print("=" * 55)
print(f"  Final Model: {best_name}")
print("=" * 55)
print(f"  AIC          : {final.aic:.4f}")
print(f"  R²           : {final.rsquared:.4f}")
print(f"  Adjusted R²  : {final.rsquared_adj:.4f}")

# ── 6. VIF ────────────────────────────────────────────────────────────────────
print()
print("  Variance Inflation Factors")
print("  " + "─" * 35)
vif_df = pd.DataFrame({
    "feature": best_features,
    "VIF": [variance_inflation_factor(X_best_scaled, i)
            for i in range(X_best_scaled.shape[1])]
}).sort_values("VIF", ascending=False)
print(vif_df.to_string(index=False))
print("  (VIF > 10 = multicollinearity concern)")

# ── 7. Plots ──────────────────────────────────────────────────────────────────
fitted    = final.fittedvalues
residuals = final.resid

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle(f"OLS Diagnostics — {best_name}", fontsize=13, fontweight="bold")

# Plot 1: CV(K) comparison across models
ax = axes[0, 0]
short_names = [n.split(":")[0] for n in cv_results]
cv_vals     = [cv_results[n]["cv_k"] for n in cv_results]
colors      = ["#C44E52" if n == best_name else "#4C72B0" for n in cv_results]
bars = ax.bar(short_names, cv_vals, color=colors, edgecolor="white")
ax.set_title(f"CV({K}) by Candidate Model")
ax.set_ylabel(f"CV({K}) MSE")
for bar, v in zip(bars, cv_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f"{v:.2f}", ha="center", fontsize=9)
ax.legend(handles=[
    plt.Rectangle((0,0),1,1, color="#C44E52", label="Selected"),
    plt.Rectangle((0,0),1,1, color="#4C72B0", label="Other")
], fontsize=8)

# Plot 2: Residuals vs Fitted
ax = axes[0, 1]
ax.scatter(fitted, residuals, alpha=0.4, s=16, color="#4C72B0", edgecolors="none")
ax.axhline(0, color="red", linewidth=1.2, linestyle="--")
z = np.polyfit(fitted, residuals, 2)
xline = np.linspace(fitted.min(), fitted.max(), 300)
ax.plot(xline, np.polyval(z, xline), color="orange", linewidth=1.5)
ax.set_xlabel("Fitted Values")
ax.set_ylabel("Residuals")
ax.set_title("Residuals vs Fitted")

# Plot 3: Fitted vs Actual
ax = axes[1, 0]
ax.scatter(y, fitted, alpha=0.4, s=16, color="#55A868", edgecolors="none")
lims = [min(y.min(), fitted.min()), max(y.max(), fitted.max())]
ax.plot(lims, lims, "r--", linewidth=1.2, label="y = x")
ax.set_xlabel("Actual")
ax.set_ylabel("Fitted")
ax.set_title("Fitted vs Actual")
ax.legend(fontsize=8)

# Plot 4: QQ plot of residuals
ax = axes[1, 1]
sm.qqplot(residuals, line="s", ax=ax, alpha=0.4, markersize=4)
ax.set_title("QQ Plot — Residuals")

plt.tight_layout()
plt.savefig("spotify_cv_selection.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved → spotify_cv_selection.png")