import pandas as pd
import numpy as np
from datetime import datetime

TODAY = datetime(2024, 12, 1)

# ── LOAD ───────────────────────────────────────────────────────────────────────

sku     = pd.read_csv("data/raw/sku_master.csv", parse_dates=["launch_date"])
inv     = pd.read_csv("data/raw/inventory_snapshot.csv", parse_dates=["last_sale_date", "snapshot_date"])
sales   = pd.read_csv("data/raw/sales_transactions.csv", parse_dates=["sale_date"])

print(f"Loaded: {len(sku)} SKUs | {len(inv)} inventory rows | {len(sales)} transactions")


# ── CLEAN ──────────────────────────────────────────────────────────────────────

# Drop SKUs with zero units purchased (shouldn't exist but defensive)
sku = sku[sku["units_purchased"] > 0]

# Cap selling price outliers (shouldn't exceed MRP + 5%)
sales = sales.merge(sku[["sku_id", "mrp"]], on="sku_id", how="left")
sales = sales[sales["selling_price"] <= sales["mrp"] * 1.05]
sales = sales[sales["selling_price"] > 0]

print(f"After cleaning: {len(sales)} transactions retained")


# ── WEEKLY SALES AGGREGATION (for velocity + trend) ───────────────────────────

sales["week"] = sales["sale_date"].dt.to_period("W")
weekly_sales = (
    sales.groupby(["sku_id", "week"])["units_sold"]
    .sum()
    .reset_index()
    .rename(columns={"units_sold": "weekly_units_sold"})
)

# Last 4-week velocity per SKU
sales["days_ago"] = (TODAY - sales["sale_date"]).dt.days
recent_sales = sales[sales["days_ago"] <= 28]
velocity_4w = (
    recent_sales.groupby("sku_id")["units_sold"]
    .sum()
    .reset_index()
    .rename(columns={"units_sold": "units_sold_last_4w"})
)


# ── CHANNEL PERFORMANCE ────────────────────────────────────────────────────────

channel_rev = (
    sales.groupby(["sku_id", "channel"])["units_sold"]
    .sum()
    .reset_index()
    .rename(columns={"units_sold": "channel_units"})
)
top_channel = (
    channel_rev.sort_values("channel_units", ascending=False)
    .groupby("sku_id")
    .first()
    .reset_index()[["sku_id", "channel"]]
    .rename(columns={"channel": "top_channel"})
)


# ── MASTER JOIN ────────────────────────────────────────────────────────────────

# inv also has units_purchased — drop from inv since sku_master is authoritative
inv_cols = ["sku_id", "snapshot_date", "units_sold_total", "units_on_hand", "last_sale_date"]
df = sku.merge(inv[inv_cols], on="sku_id", how="left")
df = df.merge(velocity_4w, on="sku_id", how="left")
df = df.merge(top_channel, on="sku_id", how="left")
df["units_sold_last_4w"] = df["units_sold_last_4w"].fillna(0)


# ── COMPUTE BLEED SIGNALS ──────────────────────────────────────────────────────

df["days_since_last_sale"] = (TODAY - df["last_sale_date"]).dt.days.clip(lower=0)
df["days_on_shelf"]        = (TODAY - df["launch_date"]).dt.days.clip(lower=1)

df["stock_remaining_pct"]  = (df["units_on_hand"] / df["units_purchased"]).round(3)

# Bleed score: days stale weighted by how much stock is still sitting
df["bleed_score"] = (df["days_since_last_sale"] * df["stock_remaining_pct"]).round(2)

# Margin at risk: money locked in unsold inventory
df["margin_per_unit"]  = df["mrp"] - df["cost_price"]
df["margin_at_risk"]   = (df["units_on_hand"] * df["margin_per_unit"]).round(0).astype(int)

# Sell-through rate: how much of the stock has moved overall
df["sell_through_rate"] = (df["units_sold_total"] / df["units_purchased"]).round(3)

# Weekly velocity (all-time)
df["velocity_overall"] = (df["units_sold_total"] / (df["days_on_shelf"] / 7)).round(2)

# Weeks of stock remaining at recent velocity (infinite if velocity = 0)
df["weeks_of_stock"] = np.where(
    df["units_sold_last_4w"] > 0,
    (df["units_on_hand"] / (df["units_sold_last_4w"] / 4)).round(1),
    999
)


# ── HEALTH STATUS ──────────────────────────────────────────────────────────────

def assign_status(row):
    if row["bleed_score"] >= 40 or row["days_since_last_sale"] >= 50:
        return "Critical"
    elif row["bleed_score"] >= 20 or row["days_since_last_sale"] >= 25:
        return "Warning"
    else:
        return "Healthy"

df["status"] = df.apply(assign_status, axis=1)

status_counts = df["status"].value_counts()
print(f"\nSKU health breakdown:")
print(f"  Critical : {status_counts.get('Critical', 0)}")
print(f"  Warning  : {status_counts.get('Warning', 0)}")
print(f"  Healthy  : {status_counts.get('Healthy', 0)}")


# ── SAVE ANALYSIS TABLE ────────────────────────────────────────────────────────

cols = [
    "sku_id", "product_name", "category", "color", "size",
    "supplier_id", "launch_date", "cost_price", "mrp", "margin_per_unit",
    "units_purchased", "units_sold_total", "units_on_hand",
    "stock_remaining_pct", "sell_through_rate",
    "last_sale_date", "days_since_last_sale", "days_on_shelf",
    "units_sold_last_4w", "velocity_overall", "weeks_of_stock",
    "bleed_score", "margin_at_risk", "status", "top_channel"
]

analysis = df[cols].sort_values("bleed_score", ascending=False).reset_index(drop=True)
analysis.to_csv("data/processed/sku_analysis.csv", index=False)

# Also save weekly sales for trend chart in Power BI
weekly_trend = (
    sales.groupby(["sale_date"])
    .agg(daily_revenue=("units_sold", "sum"))
    .reset_index()
)
weekly_trend.to_csv("data/processed/daily_sales_trend.csv", index=False)

# Category summary
cat_summary = (
    analysis.groupby("category")
    .agg(
        total_skus=("sku_id", "count"),
        critical_skus=("status", lambda x: (x == "Critical").sum()),
        total_margin_at_risk=("margin_at_risk", "sum"),
        avg_bleed_score=("bleed_score", "mean"),
        avg_sell_through=("sell_through_rate", "mean"),
    )
    .round(2)
    .reset_index()
)
cat_summary.to_csv("data/processed/category_summary.csv", index=False)

print(f"\nTotal margin at risk : ₹{analysis['margin_at_risk'].sum():,.0f}")
print(f"Avg bleed score      : {analysis['bleed_score'].mean():.1f}")
print(f"\nSaved to data/processed/")
print("  sku_analysis.csv")
print("  daily_sales_trend.csv")
print("  category_summary.csv")
