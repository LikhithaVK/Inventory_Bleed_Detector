import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("  INVENTORY BLEED DETECTOR — ML STOCK FORECAST ENGINE")
print("  Next season buying plan using Random Forest")
print("=" * 60)


# ── LOAD ───────────────────────────────────────────────────────────────────────

df = pd.read_csv("data/processed/prescriptions.csv", parse_dates=["last_sale_date"])
# Recreate days_on_shelf from days_since_last_sale as proxy (launch_date not in this file)
df["days_on_shelf"] = (180 - df["days_since_last_sale"]).clip(lower=7)  # approx days active
print(f"\nLoaded {len(df)} SKUs from prescriptions.csv")


# ── FEATURE ENGINEERING ────────────────────────────────────────────────────────
# The model learns: given these characteristics of a SKU,
# what was the "ideal" units_purchased to avoid bleed?

print("\nEngineering features...")

# Target variable: what SHOULD have been bought
# Logic: if sell_through >= 85%, the buyer was right → keep units_purchased
#        if sell_through < 85%, the buyer overbought → ideal = units_sold_total / 0.85
df["ideal_units"] = np.where(
    df["sell_through_rate"] >= 0.85,
    df["units_purchased"],
    np.round(df["units_sold_total"] / 0.85).clip(lower=5)
).astype(int)

# Price positioning: where does this SKU sit in its category?
cat_avg_mrp = df.groupby("category")["mrp"].transform("mean")
df["price_positioning"] = (df["mrp"] / cat_avg_mrp).round(3)   # >1 = premium, <1 = value

# Margin ratio: how much margin does this SKU generate relative to cost
df["margin_ratio"] = ((df["mrp"] - df["cost_price"]) / df["cost_price"]).round(3)

# Size demand index: some sizes sell faster (encoded as ordinal)
size_order = {"XS": 1, "S": 3, "M": 5, "L": 4, "XL": 2, "XXL": 1}
df["size_demand"] = df["size"].map(size_order).fillna(2)

# Channel reach: online channels have broader demand
df["is_online_top"] = df["top_channel"].str.contains("Online", na=False).astype(int)

# Bleed indicator: did this SKU bleed? (what the model learns to avoid)
df["did_bleed"] = (df["status"].isin(["Critical", "Warning"])).astype(int)

# Encode category
le = LabelEncoder()
df["category_enc"] = le.fit_transform(df["category"])
category_classes = list(le.classes_)

# ── FEATURE SET ───────────────────────────────────────────────────────────────

FEATURES = [
    "category_enc",       # what kind of product
    "cost_price",         # base cost
    "mrp",                # selling price
    "margin_ratio",       # profit margin %
    "price_positioning",  # premium vs value within category
    "size_demand",        # size popularity
    "is_online_top",      # channel type
    "days_on_shelf",      # how long it was available
    "velocity_overall",   # how fast it sold
    "sell_through_rate",  # what % moved
]

TARGET = "ideal_units"

X = df[FEATURES].fillna(0)
y = df[TARGET]

print(f"Features: {len(FEATURES)} | Training samples: {len(X)}")


# ── TRAIN / TEST SPLIT ─────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── MODEL 1: RANDOM FOREST ─────────────────────────────────────────────────────

print("\nTraining Random Forest...")
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

rf_preds   = rf.predict(X_test)
rf_mae     = mean_absolute_error(y_test, rf_preds)
rf_r2      = r2_score(y_test, rf_preds)
rf_cv      = cross_val_score(rf, X, y, cv=5, scoring="r2").mean()

print(f"  MAE  : {rf_mae:.1f} units  (avg prediction error)")
print(f"  R²   : {rf_r2:.3f}         (1.0 = perfect)")
print(f"  CV R²: {rf_cv:.3f}         (cross-validated, more reliable)")


# ── MODEL 2: GRADIENT BOOSTING (comparison) ────────────────────────────────────

print("\nTraining Gradient Boosting (for comparison)...")
gb = GradientBoostingRegressor(n_estimators=150, max_depth=4, random_state=42)
gb.fit(X_train, y_train)
gb_preds = gb.predict(X_test)
gb_mae   = mean_absolute_error(y_test, gb_preds)
gb_r2    = r2_score(y_test, gb_preds)

print(f"  MAE  : {gb_mae:.1f} units")
print(f"  R²   : {gb_r2:.3f}")

# Pick best model
best_model = rf if rf_r2 >= gb_r2 else gb
best_name  = "Random Forest" if rf_r2 >= gb_r2 else "Gradient Boosting"
print(f"\n  ✓ Using {best_name} for forecasts")


# ── FEATURE IMPORTANCE ─────────────────────────────────────────────────────────

importance_df = pd.DataFrame({
    "feature":    FEATURES,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)


# ── NEXT SEASON FORECAST ───────────────────────────────────────────────────────
# Simulate next season: same SKU characteristics, but assume:
# - velocity may improve by 10% (marketing + better placement)
# - we want a 90% sell-through target (not 85%)

print("\nGenerating next season buying plan...")

next_season = df[FEATURES].copy()
next_season["velocity_overall"] = (next_season["velocity_overall"] * 1.10).round(2)  # optimistic
next_season["sell_through_rate"] = 0.90  # target

# Predict recommended units
df["predicted_units_next_season"] = best_model.predict(next_season).round(0).astype(int).clip(5)

# Confidence range: ±15% for planning buffer
df["units_min"] = (df["predicted_units_next_season"] * 0.85).round(0).astype(int)
df["units_max"] = (df["predicted_units_next_season"] * 1.15).round(0).astype(int)

# Buying decision vs last season
df["vs_last_season"] = df["predicted_units_next_season"] - df["units_purchased"]
df["buying_signal"] = pd.cut(
    df["vs_last_season"],
    bins=[-999, -20, -5, 5, 20, 999],
    labels=["Buy much less", "Buy less", "Same", "Buy more", "Buy much more"]
)

# Estimated capital needed next season
df["estimated_capital_needed"] = df["predicted_units_next_season"] * df["cost_price"]


# ── PRINT REPORT ──────────────────────────────────────────────────────────────

total_capital_this  = (df["units_purchased"] * df["cost_price"]).sum()
total_capital_next  = df["estimated_capital_needed"].sum()
capital_saved       = total_capital_this - total_capital_next
bleed_skus_next     = (df["buying_signal"].isin(["Buy much less", "Buy less"])).sum()

print("\n" + "=" * 60)
print("  NEXT SEASON BUYING PLAN — SUMMARY REPORT")
print("=" * 60)

print(f"\n  Capital invested this season  : ₹{total_capital_this:>12,.0f}")
print(f"  Capital recommended next      : ₹{total_capital_next:>12,.0f}")
print(f"  Estimated capital saved       : ₹{capital_saved:>12,.0f}")
print(f"  SKUs to reduce buying on      : {bleed_skus_next}")
print(f"  SKUs to increase buying on    : {(df['buying_signal'].isin(['Buy more', 'Buy much more'])).sum()}")

print(f"\n{'─'*60}")
print(f"  BUYING SIGNAL BREAKDOWN")
print(f"{'─'*60}")
signal_summary = df["buying_signal"].value_counts().sort_index()
for signal, count in signal_summary.items():
    bar = "█" * (count // 5)
    print(f"  {signal:<16} {count:>4} SKUs  {bar}")

print(f"\n{'─'*60}")
print(f"  CATEGORY-WISE BUYING PLAN")
print(f"{'─'*60}")
cat_plan = (
    df.groupby("category")
    .agg(
        skus=("sku_id", "count"),
        bought_this_season=("units_purchased", "sum"),
        recommended_next=("predicted_units_next_season", "sum"),
        capital_this=("cost_price", lambda x: (x * df.loc[x.index, "units_purchased"]).sum()),
        capital_next=("estimated_capital_needed", "sum"),
    )
    .reset_index()
)
cat_plan["change_pct"] = ((cat_plan["recommended_next"] - cat_plan["bought_this_season"]) / cat_plan["bought_this_season"] * 100).round(1)

for _, row in cat_plan.iterrows():
    arrow = "▼" if row["change_pct"] < 0 else "▲"
    print(f"\n  {row['category']}")
    print(f"    This season : {row['bought_this_season']:,} units  (₹{row['capital_this']:,.0f})")
    print(f"    Next season : {row['recommended_next']:,} units  (₹{row['capital_next']:,.0f})")
    print(f"    Change      : {arrow} {abs(row['change_pct'])}%")

print(f"\n{'─'*60}")
print(f"  TOP 10 SKUs TO BUY MORE NEXT SEASON")
print(f"{'─'*60}")
top_buy = df[df["buying_signal"].isin(["Buy much more", "Buy more"])].nlargest(10, "vs_last_season")
for _, row in top_buy.iterrows():
    print(f"  {row['product_name'][:40]:<42} +{int(row['vs_last_season'])} units")

print(f"\n{'─'*60}")
print(f"  TOP 10 SKUs TO BUY LESS NEXT SEASON (avoid bleed)")
print(f"{'─'*60}")
top_cut = df[df["buying_signal"].isin(["Buy much less", "Buy less"])].nsmallest(10, "vs_last_season")
for _, row in top_cut.iterrows():
    print(f"  {row['product_name'][:40]:<42} {int(row['vs_last_season'])} units")

print(f"\n{'─'*60}")
print(f"  FEATURE IMPORTANCE — what drove the forecast")
print(f"{'─'*60}")
for _, row in importance_df.iterrows():
    bar = "█" * int(row["importance"] * 50)
    print(f"  {row['feature']:<22} {bar}  {row['importance']:.3f}")

print(f"\n{'─'*60}")
print(f"  MODEL PERFORMANCE")
print(f"{'─'*60}")
print(f"  Algorithm    : {best_name}")
print(f"  MAE          : ±{rf_mae:.1f} units  (avg error per SKU)")
print(f"  R² score     : {rf_r2:.3f}  (explains {rf_r2*100:.1f}% of variance)")
print(f"  CV R²        : {rf_cv:.3f}  (cross-validated)")
print("=" * 60)


# ── SAVE OUTPUT ───────────────────────────────────────────────────────────────

output_cols = [
    "sku_id", "product_name", "category", "color", "size",
    "cost_price", "mrp", "units_purchased", "sell_through_rate",
    "status", "bleed_score",
    "predicted_units_next_season", "units_min", "units_max",
    "vs_last_season", "buying_signal", "estimated_capital_needed"
]
forecast_df = df[output_cols].sort_values("vs_last_season")
forecast_df.to_csv("data/processed/next_season_forecast.csv", index=False)
cat_plan.to_csv("data/processed/category_buying_plan.csv", index=False)

print(f"\nSaved:")
print(f"  data/processed/next_season_forecast.csv  ← Power BI optional")
print(f"  data/processed/category_buying_plan.csv  ← Power BI optional")
print(f"\nDone.")
