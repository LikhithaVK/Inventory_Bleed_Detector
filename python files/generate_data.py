import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)
TODAY = datetime(2024, 12, 1)

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ── 1. SKU MASTER ──────────────────────────────────────────────────────────────

categories = {
    "Kurta":      {"cost": (350, 650),  "mrp": (950, 1800),  "count": 120},
    "Saree":      {"cost": (800, 2200), "mrp": (2200, 6500),  "count": 80},
    "Coord Set":  {"cost": (500, 900),  "mrp": (1400, 2800),  "count": 100},
    "Dupatta":    {"cost": (120, 280),  "mrp": (350, 850),   "count": 70},
    "Dress":      {"cost": (420, 750),  "mrp": (1100, 2200),  "count": 80},
    "Palazzo":    {"cost": (200, 380),  "mrp": (550, 1100),   "count": 50},
}

prints = ["Block print", "Ajrakh", "Bandhani", "Ikat", "Kalamkari",
          "Mirror work", "Chikankari", "Bagru print", "Shibori", "Dabu print"]
fabrics = ["Cotton", "Chanderi", "Silk blend", "Mul mul", "Rayon", "Linen", "Georgette"]
colors  = ["Ivory", "Indigo", "Rust", "Sage green", "Mauve", "Ochre",
           "Teal", "Blush pink", "Charcoal", "Terracotta"]
sizes   = ["XS", "S", "M", "L", "XL", "XXL"]
suppliers = [f"SUP-{str(i).zfill(3)}" for i in range(1, 11)]

rows = []
sku_counter = 1
for cat, cfg in categories.items():
    for _ in range(cfg["count"]):
        cost  = np.random.randint(*cfg["cost"])
        mrp   = np.random.randint(*cfg["mrp"])
        while mrp < cost * 1.4:
            mrp = np.random.randint(*cfg["mrp"])

        launch_days_ago = np.random.randint(14, 180)
        launch_date = TODAY - timedelta(days=launch_days_ago)

        rows.append({
            "sku_id":      f"SKU-{str(sku_counter).zfill(4)}",
            "product_name": f"{np.random.choice(prints)} {cat} ({np.random.choice(fabrics)})",
            "category":    cat,
            "color":       np.random.choice(colors),
            "size":        np.random.choice(sizes),
            "cost_price":  cost,
            "mrp":         mrp,
            "supplier_id": np.random.choice(suppliers),
            "launch_date": launch_date.strftime("%Y-%m-%d"),
            "units_purchased": np.random.randint(30, 150),
        })
        sku_counter += 1

sku_master = pd.DataFrame(rows)
print(f"SKU master: {len(sku_master)} rows")


# ── 2. SALES TRANSACTIONS ──────────────────────────────────────────────────────

channels = ["Store - MG Road", "Store - Indiranagar", "Store - Koramangala",
            "Online - Website", "Online - Nykaa", "Online - Instagram"]

txn_rows = []
txn_id = 1

for _, sku in sku_master.iterrows():
    launch = datetime.strptime(sku["launch_date"], "%Y-%m-%d")
    days_available = (TODAY - launch).days
    units_purchased = sku["units_purchased"]

    # Assign a sell profile: healthy, slow, or dead
    profile = np.random.choice(["healthy", "slow", "dead"], p=[0.45, 0.30, 0.25])

    if profile == "healthy":
        total_sold = np.random.randint(int(units_purchased * 0.55), int(units_purchased * 0.90))
        sale_freq  = max(1, days_available // (total_sold + 1))
        last_sale_offset = np.random.randint(0, 15)
    elif profile == "slow":
        total_sold = np.random.randint(int(units_purchased * 0.15), int(units_purchased * 0.50))
        sale_freq  = max(3, days_available // max(total_sold, 1))
        last_sale_offset = np.random.randint(20, 50)
    else:  # dead
        total_sold = np.random.randint(0, int(units_purchased * 0.18))
        sale_freq  = max(7, days_available // max(total_sold, 1))
        last_sale_offset = np.random.randint(45, days_available if days_available > 50 else 50)

    total_sold = min(total_sold, units_purchased)

    current_date = launch
    units_left = total_sold
    while units_left > 0 and current_date <= TODAY:
        qty = min(np.random.randint(1, 4), units_left)
        # Occasional markdown: selling price sometimes below MRP
        discount = np.random.choice([0, 0, 0, 10, 15, 20, 25], p=[0.5, 0.15, 0.1, 0.1, 0.07, 0.05, 0.03])
        sell_price = round(sku["mrp"] * (1 - discount / 100))

        txn_rows.append({
            "txn_id":      f"TXN-{str(txn_id).zfill(6)}",
            "sku_id":      sku["sku_id"],
            "sale_date":   current_date.strftime("%Y-%m-%d"),
            "units_sold":  qty,
            "selling_price": sell_price,
            "discount_pct":  discount,
            "channel":     np.random.choice(channels),
        })
        txn_id += 1
        units_left -= qty
        current_date += timedelta(days=sale_freq + np.random.randint(0, 3))

sales_txn = pd.DataFrame(txn_rows)
print(f"Sales transactions: {len(sales_txn)} rows")


# ── 3. INVENTORY SNAPSHOT ──────────────────────────────────────────────────────

snapshot_rows = []
for _, sku in sku_master.iterrows():
    sku_sales = sales_txn[sales_txn["sku_id"] == sku["sku_id"]]
    units_sold_total = int(sku_sales["units_sold"].sum())
    units_on_hand    = max(0, sku["units_purchased"] - units_sold_total)

    if len(sku_sales) > 0:
        last_sale_date = sku_sales["sale_date"].max()
    else:
        last_sale_date = sku["launch_date"]

    snapshot_rows.append({
        "sku_id":           sku["sku_id"],
        "snapshot_date":    TODAY.strftime("%Y-%m-%d"),
        "units_purchased":  sku["units_purchased"],
        "units_sold_total": units_sold_total,
        "units_on_hand":    units_on_hand,
        "last_sale_date":   last_sale_date,
    })

inventory = pd.DataFrame(snapshot_rows)
print(f"Inventory snapshot: {len(inventory)} rows")


# ── 4. SAVE ────────────────────────────────────────────────────────────────────

sku_master.to_csv("data/raw/sku_master.csv", index=False)
sales_txn.to_csv("data/raw/sales_transactions.csv", index=False)
inventory.to_csv("data/raw/inventory_snapshot.csv", index=False)

print("\nFiles saved to data/raw/")
print("  sku_master.csv")
print("  sales_transactions.csv")
print("  inventory_snapshot.csv")
