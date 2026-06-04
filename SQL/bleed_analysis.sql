-- ============================================================
-- INVENTORY BLEED DETECTOR — SQL ANALYSIS
-- Compatible with: SQLite, PostgreSQL, MySQL, BigQuery
-- Run after loading CSVs into your DB or use with SQLite
-- ============================================================


-- ── TABLE SETUP (SQLite / PostgreSQL) ─────────────────────────────────────────

-- If loading CSVs into SQLite:
--   .mode csv
--   .import data/raw/sku_master.csv sku_master
--   .import data/raw/inventory_snapshot.csv inventory_snapshot
--   .import data/raw/sales_transactions.csv sales_transactions


-- ── QUERY 1: BLEED SCORE PER SKU ──────────────────────────────────────────────
-- Core diagnostic query — ranks every SKU by urgency

SELECT
    s.sku_id,
    s.product_name,
    s.category,
    s.cost_price,
    s.mrp,
    s.mrp - s.cost_price                                   AS margin_per_unit,
    i.units_on_hand,
    i.units_sold_total,
    i.last_sale_date,

    -- Days since last sale (key bleed signal)
    JULIANDAY('2024-12-01') - JULIANDAY(i.last_sale_date)  AS days_since_last_sale,

    -- What % of stock is still sitting unsold
    ROUND(CAST(i.units_on_hand AS FLOAT) / s.units_purchased, 3)
                                                            AS stock_remaining_pct,

    -- Bleed score = days stale × stock burden
    ROUND(
        (JULIANDAY('2024-12-01') - JULIANDAY(i.last_sale_date))
        * (CAST(i.units_on_hand AS FLOAT) / s.units_purchased)
    , 1)                                                    AS bleed_score,

    -- Margin actually at risk in rupees
    i.units_on_hand * (s.mrp - s.cost_price)               AS margin_at_risk,

    -- Overall sell-through rate
    ROUND(CAST(i.units_sold_total AS FLOAT) / s.units_purchased, 3)
                                                            AS sell_through_rate,

    -- Status bucket
    CASE
        WHEN (JULIANDAY('2024-12-01') - JULIANDAY(i.last_sale_date)) >= 50
            OR ROUND(
                (JULIANDAY('2024-12-01') - JULIANDAY(i.last_sale_date))
                * (CAST(i.units_on_hand AS FLOAT) / s.units_purchased)
               , 1) >= 40  THEN 'Critical'
        WHEN (JULIANDAY('2024-12-01') - JULIANDAY(i.last_sale_date)) >= 25
            OR ROUND(
                (JULIANDAY('2024-12-01') - JULIANDAY(i.last_sale_date))
                * (CAST(i.units_on_hand AS FLOAT) / s.units_purchased)
               , 1) >= 20  THEN 'Warning'
        ELSE 'Healthy'
    END                                                     AS status

FROM sku_master s
JOIN inventory_snapshot i ON s.sku_id = i.sku_id
ORDER BY bleed_score DESC;


-- ── QUERY 2: CATEGORY LOSS SUMMARY ────────────────────────────────────────────
-- Answers: which product type is bleeding the most?

SELECT
    s.category,
    COUNT(*)                                                AS total_skus,
    SUM(CASE WHEN
        (JULIANDAY('2024-12-01') - JULIANDAY(i.last_sale_date)) >= 50
        THEN 1 ELSE 0 END)                                  AS critical_skus,
    SUM(i.units_on_hand * (s.mrp - s.cost_price))          AS total_margin_at_risk,
    ROUND(AVG(
        (JULIANDAY('2024-12-01') - JULIANDAY(i.last_sale_date))
        * (CAST(i.units_on_hand AS FLOAT) / s.units_purchased)
    ), 1)                                                   AS avg_bleed_score,
    ROUND(AVG(
        CAST(i.units_sold_total AS FLOAT) / s.units_purchased
    ), 3)                                                   AS avg_sell_through
FROM sku_master s
JOIN inventory_snapshot i ON s.sku_id = i.sku_id
GROUP BY s.category
ORDER BY total_margin_at_risk DESC;


-- ── QUERY 3: WEEKLY REVENUE TREND ─────────────────────────────────────────────
-- Answers: is the business slowing down week over week?

SELECT
    STRFTIME('%Y-W%W', sale_date)                           AS week,
    COUNT(DISTINCT sku_id)                                  AS skus_sold,
    SUM(units_sold)                                         AS total_units,
    ROUND(SUM(units_sold * selling_price), 0)               AS total_revenue,
    ROUND(AVG(discount_pct), 1)                             AS avg_discount_pct
FROM sales_transactions
GROUP BY week
ORDER BY week;


-- ── QUERY 4: SUPPLIER BLEED CONTRIBUTION ──────────────────────────────────────
-- Answers: is a specific supplier's products performing worse?

SELECT
    s.supplier_id,
    COUNT(*)                                                AS total_skus,
    SUM(i.units_on_hand * (s.mrp - s.cost_price))          AS margin_at_risk,
    ROUND(AVG(
        CAST(i.units_sold_total AS FLOAT) / s.units_purchased
    ) * 100, 1)                                             AS avg_sell_through_pct,
    ROUND(AVG(
        JULIANDAY('2024-12-01') - JULIANDAY(i.last_sale_date)
    ), 0)                                                   AS avg_days_stale
FROM sku_master s
JOIN inventory_snapshot i ON s.sku_id = i.sku_id
GROUP BY s.supplier_id
ORDER BY margin_at_risk DESC;


-- ── QUERY 5: PRESCRIPTION SUMMARY ─────────────────────────────────────────────
-- Join with the prescriptions output from Python

SELECT
    p.action,
    COUNT(*)                                                AS sku_count,
    SUM(p.margin_at_risk)                                   AS total_margin_at_risk,
    SUM(p.projected_recovery)                               AS total_projected_recovery,
    ROUND(
        CAST(SUM(p.projected_recovery) AS FLOAT)
        / NULLIF(SUM(p.margin_at_risk), 0) * 100
    , 1)                                                    AS recovery_rate_pct
FROM prescriptions p
GROUP BY p.action
ORDER BY total_margin_at_risk DESC;


-- ── QUERY 6: TOP 20 URGENT SKUS — EXEC SUMMARY ────────────────────────────────

SELECT
    p.sku_id,
    p.product_name,
    p.category,
    p.days_since_last_sale,
    p.units_on_hand,
    p.margin_at_risk,
    p.action,
    p.recommended_discount_pct,
    p.projected_recovery,
    p.reason
FROM prescriptions p
WHERE p.action = 'Urgent markdown'
ORDER BY p.margin_at_risk DESC
LIMIT 20;
