import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# ── 1. Load & prep ────────────────────────────────────────────────────────────
df = pd.read_csv("spotify_songs.csv")
df = df.sample(n=1000, random_state=314)

df = df.drop(columns=[
    'track_id', 'track_name', 'track_artist',
    'track_album_id', 'track_album_name', 'track_album_release_date',
    'playlist_name', 'playlist_id',
], errors='ignore')

y = df["track_popularity"].values

X = df[[
    "playlist_genre", "danceability", "energy", "key", "loudness",
    "mode", "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "duration_ms"
]]
X = pd.get_dummies(X, columns=["playlist_genre"], drop_first=True)
feature_names = X.columns.tolist()
X_arr = X.values.astype(float)

# ── 2. 5-Fold Cross Validation — MSE per fold ─────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_mses = []

print("=" * 45)
print("  5-Fold Cross Validation — OLS Linear Regression")
print("=" * 45)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_arr), start=1):
    X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    scaler = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    model = LinearRegression()
    model.fit(X_tr_s, y_tr)

    preds = model.predict(X_val_s)
    mse   = mean_squared_error(y_val, preds)
    fold_mses.append(mse)
    print(f"  Fold {fold}  MSE = {mse:.4f}   RMSE = {np.sqrt(mse):.4f}")

print("-" * 45)
print(f"  Mean MSE  : {np.mean(fold_mses):.4f}")
print(f"  Std  MSE  : {np.std(fold_mses):.4f}")
print(f"  Mean RMSE : {np.mean(np.sqrt(fold_mses)):.4f}")
print()

# ── 3. Final model — fit on ALL data with statsmodels ────────────────────────
scaler_full = StandardScaler()
X_scaled = scaler_full.fit_transform(X_arr)
X_sm = sm.add_constant(X_scaled)            # statsmodels needs intercept column

ols_result = sm.OLS(y, X_sm).fit()

# ── 4. AIC ────────────────────────────────────────────────────────────────────
aic = ols_result.aic
print("=" * 45)
print("  Final Model (fit on full sample)")
print("=" * 45)
print(f"  AIC            : {aic:.4f}")

# ── 5. Adjusted R² ───────────────────────────────────────────────────────────
adj_r2 = ols_result.rsquared_adj
r2     = ols_result.rsquared
print(f"  R²             : {r2:.4f}")
print(f"  Adjusted R²    : {adj_r2:.4f}")
print()

# ── 6. VIF ────────────────────────────────────────────────────────────────────
print("=" * 45)
print("  Variance Inflation Factors (VIF)")
print("=" * 45)

vif_data = pd.DataFrame({
    "feature": feature_names,
    "VIF": [
        variance_inflation_factor(X_scaled, i)
        for i in range(X_scaled.shape[1])
    ]
}).sort_values("VIF", ascending=False)

print(vif_data.to_string(index=False))
print()
print("  Rule of thumb: VIF > 10 → high multicollinearity")
print()

# ── 7. Diagnostic plots ───────────────────────────────────────────────────────
fitted    = ols_result.fittedvalues
residuals = ols_result.resid

fig = plt.figure(figsize=(14, 10))
fig.suptitle("OLS Diagnostic Plots — Spotify Track Popularity", fontsize=14, fontweight="bold", y=1.01)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# — Plot 1: Residuals vs Fitted —
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(fitted, residuals, alpha=0.4, s=18, color="#4C72B0", edgecolors="none")
ax1.axhline(0, color="red", linewidth=1.2, linestyle="--")
z = np.polyfit(fitted, residuals, 2)
x_line = np.linspace(fitted.min(), fitted.max(), 300)
ax1.plot(x_line, np.polyval(z, x_line), color="orange", linewidth=1.5, label="loess approx")
ax1.set_xlabel("Fitted Values")
ax1.set_ylabel("Residuals")
ax1.set_title("Residuals vs Fitted")
ax1.legend(fontsize=8)

# — Plot 2: Fitted vs Actual —
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y, fitted, alpha=0.4, s=18, color="#55A868", edgecolors="none")
lims = [min(y.min(), fitted.min()), max(y.max(), fitted.max())]
ax2.plot(lims, lims, "r--", linewidth=1.2, label="Perfect fit (y=x)")
ax2.set_xlabel("Actual Values")
ax2.set_ylabel("Fitted Values")
ax2.set_title("Fitted vs Actual")
ax2.legend(fontsize=8)

# — Plot 3: Histogram of Residuals —
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(residuals, bins=35, color="#C44E52", edgecolor="white", linewidth=0.5, alpha=0.85)
ax3.axvline(0, color="black", linewidth=1.2, linestyle="--")
ax3.set_xlabel("Residual")
ax3.set_ylabel("Count")
ax3.set_title("Residual Distribution")

# — Plot 4: 5-Fold MSE bar chart —
ax4 = fig.add_subplot(gs[1, 1])
bars = ax4.bar(
    [f"Fold {i}" for i in range(1, 6)],
    fold_mses,
    color=["#4C72B0","#55A868","#C44E52","#8172B2","#CCB974"],
    edgecolor="white"
)
ax4.axhline(np.mean(fold_mses), color="black", linewidth=1.2, linestyle="--", label=f"Mean = {np.mean(fold_mses):.2f}")
ax4.set_ylabel("MSE")
ax4.set_title("5-Fold CV — MSE per Fold")
ax4.legend(fontsize=8)
for bar, val in zip(bars, fold_mses):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{val:.1f}", ha="center", va="bottom", fontsize=8)

plt.savefig("spotify_diagnostics.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved → spotify_diagnostics.png")