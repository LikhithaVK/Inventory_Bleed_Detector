import pandas as pd
import numpy as np

# ── LOAD ───────────────────────────────────────────────────────────────────────

df = pd.read_csv("data/processed/sku_analysis.csv", parse_dates=["launch_date", "last_sale_date"])
print(f"Loaded {len(df)} SKUs for prescription")


# ── OPTIMAL DISCOUNT TABLE ─────────────────────────────────────────────────────
# Based on fashion industry norms: what discount % historically drives
# a sell-through spike within 2 weeks for each category.

OPTIMAL_DISCOUNT = {
    "Kurta":     20,
    "Saree":     25,
    "Coord Set": 25,
    "Dupatta":   15,
    "Dress":     20,
    "Palazzo":   15,
}

# Recovery rate assumptions: what % of margin_at_risk is actually recovered
# at each action type (conservative estimates)
RECOVERY_RATE = {
    "Urgent markdown":  0.68,   # deep cut now, but saves most units
    "Planned markdown": 0.80,   # gentle cut, better margin
    "Bundle offer":     0.55,   # margin sacrificed on one unit, moves both
    "B2B liquidation":  0.30,   # last resort — sell to resellers in bulk
    "Monitor":          1.00,   # no action, no loss assumed yet
}


# ── PRESCRIPTION LOGIC ─────────────────────────────────────────────────────────

def prescribe(row):
    score       = row["bleed_score"]
    days_stale  = row["days_since_last_sale"]
    stock_pct   = row["stock_remaining_pct"]
    velocity    = row["velocity_overall"]
    weeks_left  = row["weeks_of_stock"]
    cat         = row["category"]
    base_disc   = OPTIMAL_DISCOUNT.get(cat, 20)

    # ── Rule 1: Critical + very high stock = urgent, deep markdown now
    if score >= 50 or (days_stale >= 60 and stock_pct >= 0.60):
        action   = "Urgent markdown"
        discount = base_disc + 10
        reason   = (f"No sale in {days_stale} days with {round(stock_pct*100)}% stock remaining. "
                    f"Margin eroding rapidly. Act before season ends.")

    # ── Rule 2: Critical but lower stock — moderate markdown
    elif score >= 40 or (days_stale >= 45 and stock_pct >= 0.40):
        action   = "Urgent markdown"
        discount = base_disc + 5
        reason   = (f"{days_stale} days since last sale. Sell-through stalled at "
                    f"{round(row['sell_through_rate']*100)}%. Markdown needed to restart velocity.")

    # ── Rule 3: Warning + very low velocity = plan a markdown next week
    elif score >= 20 or days_stale >= 25:
        action   = "Planned markdown"
        discount = base_disc
        reason   = (f"Slowing velocity ({row['velocity_overall']:.1f} units/week). "
                    f"{round(weeks_left)} weeks of stock at current pace. "
                    f"Schedule markdown for next week.")

    # ── Rule 4: High stock but good velocity — bundle to boost AOV
    elif stock_pct >= 0.70 and velocity >= 2:
        action   = "Bundle offer"
        discount = 15
        reason   = (f"Good velocity but high absolute stock ({row['units_on_hand']} units). "
                    f"Bundle with complementary SKU to accelerate sell-through.")

    # ── Rule 5: Extreme age + near-zero velocity = B2B liquidation
    elif days_stale >= 90 and velocity < 0.5:
        action   = "B2B liquidation"
        discount = 40
        reason   = (f"Product has been stale for {days_stale} days with near-zero velocity. "
                    f"Recommend bulk sale to reseller to recover cost price.")

    # ── Default: healthy, just monitor
    else:
        action   = "Monitor"
        discount = 0
        reason   = "Sell-through on track. Review again in 2 weeks."

    recovery_rate    = RECOVERY_RATE[action]
    projected_recovery = round(row["margin_at_risk"] * recovery_rate)
    markdown_price     = round(row["mrp"] * (1 - discount / 100))
    margin_if_sold     = max(0, markdown_price - row["cost_price"])
    units_to_clear     = row["units_on_hand"]
    est_revenue        = round(markdown_price * units_to_clear * 0.80)  # 80% sell-through assumed

    return pd.Series({
        "action":              action,
        "recommended_discount_pct": discount,
        "reason":              reason,
        "markdown_price":      markdown_price,
        "projected_recovery":  projected_recovery,
        "est_revenue_if_actioned": est_revenue,
        "urgency_rank":        1 if "Urgent" in action else (2 if "Planned" in action else (3 if "Bundle" in action else (4 if "B2B" in action else 5))),
    })

print("Running prescription engine...")
prescriptions = df.apply(prescribe, axis=1)
result = pd.concat([df, prescriptions], axis=1)
result = result.sort_values(["urgency_rank", "bleed_score"], ascending=[True, False]).reset_index(drop=True)


# ── SUMMARY STATS ──────────────────────────────────────────────────────────────

total_at_risk   = result["margin_at_risk"].sum()
total_recovery  = result["projected_recovery"].sum()
recovery_pct    = round(total_recovery / total_at_risk * 100, 1)

action_summary = (
    result.groupby("action")
    .agg(
        sku_count=("sku_id", "count"),
        total_margin_at_risk=("margin_at_risk", "sum"),
        total_projected_recovery=("projected_recovery", "sum"),
    )
    .reset_index()
    .sort_values("total_margin_at_risk", ascending=False)
)

print(f"\n{'─'*50}")
print(f"  Total margin at risk    : ₹{total_at_risk:,.0f}")
print(f"  Projected recovery      : ₹{total_recovery:,.0f}  ({recovery_pct}%)")
print(f"{'─'*50}")
print(f"\nAction breakdown:")
print(action_summary.to_string(index=False))


# ── SAVE ───────────────────────────────────────────────────────────────────────

# Full prescription table — main Power BI source
output_cols = [
    "sku_id", "product_name", "category", "color", "size",
    "supplier_id", "cost_price", "mrp", "units_purchased",
    "units_sold_total", "units_on_hand", "stock_remaining_pct",
    "sell_through_rate", "last_sale_date", "days_since_last_sale",
    "velocity_overall", "units_sold_last_4w", "weeks_of_stock",
    "bleed_score", "margin_at_risk", "status", "top_channel",
    "action", "recommended_discount_pct", "reason",
    "markdown_price", "projected_recovery", "est_revenue_if_actioned",
    "urgency_rank",
]
result[output_cols].to_csv("data/processed/prescriptions.csv", index=False)

# Action summary — for Power BI bar chart
action_summary.to_csv("data/processed/action_summary.csv", index=False)

# Top 20 urgent SKUs — for Power BI highlight table
top20 = result[result["action"] == "Urgent markdown"].head(20)[output_cols]
top20.to_csv("data/processed/top20_urgent.csv", index=False)

print(f"\nSaved to data/processed/")
print("  prescriptions.csv      ← main Power BI source (500 rows)")
print("  action_summary.csv     ← action bar chart")
print("  top20_urgent.csv       ← urgent SKU highlight table")
