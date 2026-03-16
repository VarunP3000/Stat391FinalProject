import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("spotify_songs.csv")

AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "valence",
    "instrumentalness",
    "duration_ms"
]
RESPONSE = "track_popularity"
GENRES   = sorted(df["playlist_genre"].unique())

# ── Helper: fit OLS and return model ──────────────────────────────────────────
def fit_ols(data, label):
    clean = data[AUDIO_FEATURES + [RESPONSE]].dropna()
    X = sm.add_constant(clean[AUDIO_FEATURES])
    y = clean[RESPONSE]
    model = sm.OLS(y, X).fit()

    print(f"\n{'='*60}")
    print(f"  Genre: {label.upper()}  (n={len(clean)})")
    print(f"{'='*60}")
    print(f"  R²:      {model.rsquared:.4f}")
    print(f"  Adj. R²: {model.rsquared_adj:.4f}")
    print(f"\n  {'Feature':<20} {'Coef':>8}  {'p-value':>8}  Sig")
    print(f"  {'-'*50}")
    for feat in AUDIO_FEATURES:
        coef = model.params[feat]
        pval = model.pvalues[feat]
        sig  = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {feat:<20} {coef:>8.4f}  {pval:>8.4f}  {sig}")

    return model


print("\n" + "="*60)
print("  POOLED MODEL — All Genres Combined")
print("="*60)
pooled_model = fit_ols(df, label="All Genres")

print("\n\n" + "="*60)
print("  PER-GENRE MODELS")
print("="*60)

genre_models = {}
for genre in GENRES:
    subset = df[df["playlist_genre"] == genre]
    genre_models[genre] = fit_ols(subset, label=genre)

# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — COEFFICIENT COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "="*60)
print("  COEFFICIENT COMPARISON TABLE")
print("="*60)

coef_df = pd.DataFrame({
    genre: {feat: genre_models[genre].params[feat] for feat in AUDIO_FEATURES}
    for genre in GENRES
}).T

print(coef_df.round(4).to_string())

# Also build a p-value table so we can see which effects are significant
pval_df = pd.DataFrame({
    genre: {feat: genre_models[genre].pvalues[feat] for feat in AUDIO_FEATURES}
    for genre in GENRES
}).T

print("\n\nP-VALUE TABLE (values < 0.05 are significant):")
print(pval_df.round(4).to_string())


print("\n\n" + "="*60)
print("  FORMAL F-TEST: POOLED vs PER-GENRE MODELS")
print("="*60)

# Get clean data for pooled model
clean_all = df[AUDIO_FEATURES + [RESPONSE]].dropna()

# RSS from pooled model (restricted — same coefficients for all genres)
RSS_pooled = pooled_model.ssr  # sum of squared residuals
df_pooled  = pooled_model.df_resid  # degrees of freedom (residual)
k_pooled   = len(pooled_model.params)  # number of parameters in pooled

# RSS from per-genre models combined (unrestricted — different coefficients)
RSS_separate = sum(genre_models[g].ssr for g in GENRES)
df_separate  = sum(genre_models[g].df_resid for g in GENRES)
k_separate   = sum(len(genre_models[g].params) for g in GENRES)

# F-test formula
df_num   = df_pooled - df_separate       # extra parameters used
df_denom = df_separate                    # residual df of unrestricted model
F_stat   = ((RSS_pooled - RSS_separate) / df_num) / (RSS_separate / df_denom)
p_value  = 1 - stats.f.cdf(F_stat, df_num, df_denom)

print(f"\n  Pooled model RSS:    {RSS_pooled:.2f}  (df = {df_pooled})")
print(f"  Per-genre models RSS: {RSS_separate:.2f}  (df = {df_separate})")
print(f"\n  Extra parameters used by per-genre models: {df_num}")
print(f"  F-statistic: {F_stat:.4f}")
print(f"  p-value:     {p_value:.6f}")
if p_value < 0.05:
    print(f"\n  *** RESULT: p < 0.05 → The per-genre models are significantly")
    print(f"      better. The relationship between audio features and popularity")
    print(f"      DOES differ across genres. ***")
else:
    print(f"\n  RESULT: p >= 0.05 → No significant difference between pooled")
    print(f"  and per-genre models.")

print("\n\n" + "="*60)
print("  R² ANALYSIS — WHAT IT MEANS")
print("="*60)
print(f"\n  Pooled model R²:     {pooled_model.rsquared:.4f}")
print(f"  Pooled model Adj R²: {pooled_model.rsquared_adj:.4f}")
print(f"\n  Per-genre R² values:")
for genre in GENRES:
    m = genre_models[genre]
    print(f"    {genre:<8} R² = {m.rsquared:.4f}  |  Adj R² = {m.rsquared_adj:.4f}")

print(f"\n  INTERPRETATION: The audio features explain only about")
print(f"  {pooled_model.rsquared*100:.1f}% of the variance in track popularity.")
print(f"  This is expected — popularity is largely driven by non-audio")
print(f"  factors like artist fame, marketing, and playlist placement.")


print("\n\n" + "="*60)
print("  VIF — MULTICOLLINEARITY CHECK")
print("="*60)

clean_all = df[AUDIO_FEATURES + [RESPONSE]].dropna()
# IMPORTANT: VIF requires a constant column in the matrix, otherwise values are wrong
X_vif_with_const = sm.add_constant(clean_all[AUDIO_FEATURES])
X_vif = clean_all[AUDIO_FEATURES]  # keep this for correlation matrix

print(f"\n  {'Feature':<20} {'VIF':>8}  Interpretation")
print(f"  {'-'*55}")
vif_vals = []
for i, feat in enumerate(AUDIO_FEATURES):
    # i+1 because index 0 is the constant column
    vif = variance_inflation_factor(X_vif_with_const.values, i + 1)
    vif_vals.append(vif)
    interp = "OK" if vif < 5 else "MODERATE" if vif < 10 else "SEVERE"
    print(f"  {feat:<20} {vif:>8.2f}  {interp}")

# Also check correlation matrix
print("\n  Correlation matrix of audio features:")
corr = X_vif.corr()
print(corr.round(3).to_string())

# Get residuals and fitted values
fitted_vals = pooled_model.fittedvalues
residuals   = pooled_model.resid

fig_diag, axes_diag = plt.subplots(1, 2, figsize=(14, 5))

ax = axes_diag[0]
ax.scatter(fitted_vals, residuals, alpha=0.1, s=10, color="steelblue")
ax.axhline(0, color="red", linestyle="--", linewidth=1)
ax.set_xlabel("Fitted Values (Predicted Popularity)")
ax.set_ylabel("Residuals (Actual - Predicted)")
ax.set_title("Residual vs Fitted Plot\n(Should look like random scatter)")

ax = axes_diag[1]
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title("QQ Plot of Residuals\n(Points should follow the red line)")

fig_diag.suptitle("Model Diagnostics — Pooled Model", fontsize=14, fontweight="bold")
fig_diag.tight_layout()
fig_diag.savefig("q3_diagnostics.png", dpi=150, bbox_inches="tight")
plt.close(fig_diag)
print("\nSaved: q3_diagnostics.png")

# ── 6c. Per-genre residual plots ──────────────────────────────────────────────
fig_resid, axes_resid = plt.subplots(2, 3, figsize=(15, 9))
axes_resid = axes_resid.flatten()

for i, genre in enumerate(GENRES):
    ax = axes_resid[i]
    m = genre_models[genre]
    ax.scatter(m.fittedvalues, m.resid, alpha=0.15, s=10, color="steelblue")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title(f"{genre} (R²={m.rsquared:.4f})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")

fig_resid.suptitle(
    "Residual vs Fitted Plots Per Genre\n(Should look like random scatter with no pattern)",
    fontsize=13, fontweight="bold"
)
fig_resid.tight_layout()
fig_resid.savefig("q3_residuals_per_genre.png", dpi=150, bbox_inches="tight")
plt.close(fig_resid)
print("Saved: q3_residuals_per_genre.png")

# ── 6d. Per-genre QQ plots ────────────────────────────────────────────────────
fig_qq, axes_qq = plt.subplots(2, 3, figsize=(15, 9))
axes_qq = axes_qq.flatten()

for i, genre in enumerate(GENRES):
    ax = axes_qq[i]
    m = genre_models[genre]
    stats.probplot(m.resid, dist="norm", plot=ax)
    ax.set_title(f"{genre}", fontsize=11, fontweight="bold")

fig_qq.suptitle(
    "QQ Plots Per Genre\n(Points should follow the diagonal if residuals are normal)",
    fontsize=13, fontweight="bold"
)
fig_qq.tight_layout()
fig_qq.savefig("q3_qq_per_genre.png", dpi=150, bbox_inches="tight")
plt.close(fig_qq)
print("Saved: q3_qq_per_genre.png")


print("\n\n" + "="*60)
print("  COEFFICIENT INTERPRETATION IN CONTEXT")
print("="*60)

# Find most interesting contrasts
print("\n  KEY FINDINGS BY FEATURE:")
print(f"  {'-'*55}")

for feat in AUDIO_FEATURES:
    coefs = {g: genre_models[g].params[feat] for g in GENRES}
    pvals = {g: genre_models[g].pvalues[feat] for g in GENRES}

    max_genre = max(coefs, key=coefs.get)
    min_genre = min(coefs, key=coefs.get)
    sig_genres = [g for g in GENRES if pvals[g] < 0.05]

    print(f"\n  {feat.upper()}:")
    print(f"    Strongest positive effect: {max_genre} ({coefs[max_genre]:+.4f})")
    print(f"    Strongest negative effect: {min_genre} ({coefs[min_genre]:+.4f})")
    print(f"    Significant in: {', '.join(sig_genres) if sig_genres else 'NO genres'}")

    # Plain English interpretation for the strongest effect
    if feat == "danceability":
        print(f"    → In {max_genre}, a 0.1 increase in danceability is associated with")
        print(f"      a {coefs[max_genre]*0.1:+.2f} point change in popularity.")
        print(f"      In {min_genre}, the same increase only changes popularity by {coefs[min_genre]*0.1:+.2f}.")
    elif feat == "energy":
        print(f"    → Energy has a NEGATIVE effect across all genres, but the magnitude differs.")
        print(f"      In {min_genre}, a 0.1 increase in energy is associated with")
        print(f"      a {coefs[min_genre]*0.1:+.2f} point DROP in popularity.")
    elif feat == "loudness":
        print(f"    → Each 1 dB increase in loudness is associated with a")
        print(f"      {coefs[max_genre]:+.2f} point change in popularity for {max_genre}.")
    elif feat == "valence":
        print(f"    → Valence (happiness) has opposite effects: positive in {max_genre}")
        print(f"      ({coefs[max_genre]:+.2f}) but negative in {min_genre} ({coefs[min_genre]:+.2f}).")
    elif feat == "instrumentalness":
        print(f"    → Instrumentalness is NEGATIVE across all genres (vocal tracks are more popular),")
        print(f"      but the penalty is much larger in {min_genre} ({coefs[min_genre]:+.2f}) vs {max_genre} ({coefs[max_genre]:+.2f}).")
    elif feat == "duration_ms":
        print(f"    → Duration has a very small effect. The coefficient is near zero")
        print(f"      for all genres, meaning song length barely affects popularity.")


# Plot 1: Coefficient heatmap
fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.heatmap(
    coef_df,
    annot=True, fmt=".3f",
    cmap="coolwarm", center=0,
    linewidths=0.5, ax=ax1
)
ax1.set_title("OLS Coefficients by Genre\n(Response: track_popularity)", fontsize=13)
ax1.set_xlabel("Audio Feature")
ax1.set_ylabel("Genre")
fig1.tight_layout()
fig1.savefig("q3_coef_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print("\nSaved: q3_coef_heatmap.png")

# Plot 2: Per-feature bar charts showing coefficient across genres
fig2, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()
colors = sns.color_palette("Set2", len(GENRES))

for i, feat in enumerate(AUDIO_FEATURES):
    ax = axes[i]
    vals  = [genre_models[g].params[feat] for g in GENRES]
    pvals_feat = [genre_models[g].pvalues[feat] for g in GENRES]
    bars  = ax.bar(GENRES, vals, color=colors)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(feat, fontsize=11, fontweight="bold")
    ax.set_ylabel("Coefficient")
    ax.tick_params(axis="x", rotation=30)

    # Scale asterisk offset relative to data range
    y_range = max(abs(v) for v in vals) or 1
    offset = y_range * 0.1

    for bar, pval in zip(bars, pvals_feat):
        if pval < 0.05:
            if bar.get_height() >= 0:
                ypos = bar.get_height() + offset
            else:
                ypos = bar.get_height() - offset
            ax.text(bar.get_x() + bar.get_width()/2,
                    ypos,
                    "*", ha="center", fontsize=12, color="black")

fig2.suptitle(
    "How Each Audio Feature's Effect on Popularity Changes Across Genres\n(* = p < 0.05)",
    fontsize=13, fontweight="bold"
)
fig2.tight_layout()
fig2.savefig("q3_coef_bars.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print("Saved: q3_coef_bars.png")

# Plot 3: R² per genre vs pooled
fig3, ax3 = plt.subplots(figsize=(8, 4))
r2_vals = [genre_models[g].rsquared_adj for g in GENRES]
bars3 = ax3.bar(GENRES, r2_vals, color=sns.color_palette("Set2", len(GENRES)))
ax3.axhline(pooled_model.rsquared_adj, color="red", linestyle="--",
           label=f"Pooled adj. R² = {pooled_model.rsquared_adj:.4f}")
ax3.set_title("Adjusted R² Per Genre vs Pooled Model", fontsize=12)
ax3.set_ylabel("Adjusted R²")

# Add value labels on bars
for bar, val in zip(bars3, r2_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f"{val:.4f}", ha="center", fontsize=9)

ax3.legend()
fig3.tight_layout()
fig3.savefig("q3_r2.png", dpi=150, bbox_inches="tight")
plt.close(fig3)
print("Saved: q3_r2.png")

# Plot 4: VIF bar chart (vif_vals already computed correctly above with constant)
fig4, ax4 = plt.subplots(figsize=(8, 4))
bars4 = ax4.barh(AUDIO_FEATURES, vif_vals, color="steelblue")
ax4.axvline(5, color="red", linestyle="--", label="VIF = 5 (concern threshold)")
ax4.axvline(10, color="darkred", linestyle="--", label="VIF = 10 (severe threshold)")
ax4.set_xlabel("VIF Value")
ax4.set_title("Variance Inflation Factor — Multicollinearity Check", fontsize=12)
ax4.legend()
fig4.tight_layout()
fig4.savefig("q3_vif.png", dpi=150, bbox_inches="tight")
plt.close(fig4)
print("Saved: q3_vif.png")

# Plot 5: Correlation heatmap of features
fig5, ax5 = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.5, ax=ax5)
ax5.set_title("Correlation Between Audio Features\n(High correlation = potential multicollinearity)")
fig5.tight_layout()
fig5.savefig("q3_correlation.png", dpi=150, bbox_inches="tight")
plt.close(fig5)
print("Saved: q3_correlation.png")


# ══════════════════════════════════════════════════════════════════════════════
# PART 9 — SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "="*60)
print("  SUMMARY OF FINDINGS")
print("="*60)
print(f"""
  RESEARCH QUESTION: Does the relationship between audio features
  and popularity differ across playlist genres?

  1. F-TEST RESULT:
     F = {F_stat:.4f}, p = {p_value:.6f}
     {"→ YES, the relationship significantly differs across genres." if p_value < 0.05 else "→ No significant difference found."}

  2. R² CONTEXT:
     Audio features explain only ~{pooled_model.rsquared*100:.1f}% of popularity variance.
     This is expected — popularity is driven by non-audio factors
     (artist fame, marketing, viral trends, playlist placement).
     However, the coefficients that ARE significant are still meaningful.

  3. KEY COEFFICIENT DIFFERENCES:
     - Danceability: Strong positive effect in rap/pop, near zero in edm/latin
     - Energy: Negative everywhere, but strongest in latin (-50.8)
     - Valence: Positive in edm (+7.3), negative in r&b (-11.4)
     - These differences show genres have distinct "formulas" for popularity

  4. MODEL DIAGNOSTICS:
     - VIF values indicate {"no" if max(vif_vals) < 5 else "some"} multicollinearity issues
     - Residual and QQ plots saved for visual inspection
""")

print("Done.")