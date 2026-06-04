# Inventory Bleed Detector 🧵

A prescriptive analytics project for textile and fashion companies — diagnoses which inventory is silently draining profit, prescribes markdown actions to recover margin, and uses Machine Learning to forecast smarter buying for next season.

Built as a portfolio project targeting fashion brands like Savana Mythra.

---

## What it does

| Stage | Tool | Output |
|-------|------|--------|
| Data generation | Python | 500 SKUs, 8800+ transactions |
| Cleaning & analysis | Python + SQL | Bleed score, margin at risk per SKU |
| Prescriptions | Python | Markdown actions with projected recovery |
| ML forecast | Random Forest | Next season buying plan per SKU |
| Current season dashboard | Power BI | 3-page interactive report |
| Next season forecast app | Streamlit | Interactive ML buying plan |

---

## The core idea

Every fashion brand has dead stock sitting on shelves eating their margin. This project:

- **Diagnoses** which SKUs are bleeding and why (bleed score = days stale × stock remaining %)
- **Prescribes** exact markdown % per SKU with projected rupee recovery
- **Predicts** how many units to buy next season to avoid repeating the same mistakes

---

## Project structure

```
inventory_bleed/
├── generate_data.py        # generates synthetic fashion retail data
├── clean_transform.py      # cleans and computes bleed signals
├── prescriptions.py        # prescriptive engine — action table
├── ml_stock_forecast.py    # Random Forest next season forecast
├── streamlit_app.py        # Streamlit visualization app
├── bleed_analysis.sql      # SQL queries for warehouse/SQLite
└── data/
    ├── raw/                # sku_master, sales_transactions, inventory_snapshot
    └── processed/          # analysis outputs loaded by Power BI and Streamlit
```

---

## Quick start

**1. Clone the repo**
```bash
git clone https://github.com/LikhithaVK/Inventory_Bleed_Detector
cd inventory-bleed-detector
```

**2. Install dependencies**
```bash
pip install pandas numpy scikit-learn streamlit plotly openpyxl
```

**3. Run the pipeline**
```bash
python generate_data.py
python clean_transform.py
python prescriptions.py
python ml_stock_forecast.py
```

**4. Launch the Streamlit app**
```bash
streamlit run streamlit_app.py
```

**5. Open Power BI**

Load these files from `data/processed/`:
- `prescriptions.csv` — main table
- `category_summary.csv` — category chart
- `action_summary.csv` — prescription chart

---

## ML model

- **Algorithm:** Random Forest Regressor (vs Gradient Boosting — best model auto-selected)
- **Target:** Ideal units to purchase per SKU next season
- **Key features:** sell-through rate, sales velocity, price positioning, margin ratio
- **R² score:** 0.791 — explains 79% of variance in buying quantity
- **MAE:** ±9.6 units average prediction error

---

## Power BI dashboard

3 pages following a diagnostic narrative:

- **Page 1 — WHERE ARE WE BLEEDING?** KPI cards + scatter plot (bleed score vs margin at risk)
- **Page 2 — WHICH SKU & WHY?:** SKU table with conditional formatting + category bar chart
- **Page 3 — WHAT DO WE DO?:** Action table + recovery bar chart

---

## Tech stack

- Python, pandas, NumPy, scikit-learn
- Streamlit, Plotly
- SQL
- Power BI

---

## Dataset

Fully synthetic data generated to mimic a real Indian fashion brand — 6 categories (Kurta, Saree, Coord Set, Dupatta, Dress, Palazzo), 10 print types, 500 SKUs across multiple channels and stores.
