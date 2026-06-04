import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Inventory Bleed Detector — ML Forecast",
    page_icon="🧵",
    layout="wide",
)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    forecast   = pd.read_csv("data/processed/next_season_forecast.csv")
    cat_plan   = pd.read_csv("data/processed/category_buying_plan.csv")
    prescriptions = pd.read_csv("data/processed/prescriptions.csv")
    return forecast, cat_plan, prescriptions

forecast, cat_plan, prescriptions = load_data()

# ── HEADER ────────────────────────────────────────────────────────────────────

st.markdown("""
    <h1 style='font-size:2rem; font-weight:600; margin-bottom:0'>
        🧵 Inventory Bleed Detector
    </h1>
    <p style='color:gray; margin-top:4px; font-size:1rem'>
        ML-powered next season buying plan — powered by Random Forest (R² = 0.791)
    </p>
    <hr>
""", unsafe_allow_html=True)


# ── TOP KPI CARDS ─────────────────────────────────────────────────────────────

capital_this  = (forecast["units_purchased"] * forecast["cost_price"]).sum()
capital_next  = forecast["estimated_capital_needed"].sum()
capital_saved = capital_this - capital_next
skus_reduce   = forecast["buying_signal"].isin(["Buy much less", "Buy less"]).sum()
skus_increase = forecast["buying_signal"].isin(["Buy more", "Buy much more"]).sum()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Capital this season",  f"₹{capital_this/1e5:.1f}L")
c2.metric("Recommended next",     f"₹{capital_next/1e5:.1f}L",     delta=f"-₹{capital_saved/1e5:.1f}L saved")
c3.metric("Capital saved",        f"₹{capital_saved/1e5:.1f}L",    delta="by buying smarter")
c4.metric("SKUs to buy less",     f"{skus_reduce}",                 delta="reduce overstock risk", delta_color="inverse")
c5.metric("SKUs to buy more",     f"{skus_increase}",               delta="winners to double down")

st.markdown("<br>", unsafe_allow_html=True)


# ── TABS ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📦 Buying Signal Overview",
    "🏷️ Category Plan",
    "🔍 SKU-Level Forecast",
    "⚠️ This Season's Bleed"
])


# ══ TAB 1: BUYING SIGNAL OVERVIEW ════════════════════════════════════════════

with tab1:
    st.subheader("How should buying change next season?")

    col1, col2 = st.columns(2)

    with col1:
        # Buying signal donut
        signal_counts = forecast["buying_signal"].value_counts().reset_index()
        signal_counts.columns = ["signal", "count"]

        color_map = {
            "Buy much less":  "#E24B4A",
            "Buy less":       "#F5A623",
            "Same":           "#A0A0A0",
            "Buy more":       "#4CAF8A",
            "Buy much more":  "#1D7A5F",
        }

        fig_donut = px.pie(
            signal_counts, values="count", names="signal",
            hole=0.55,
            color="signal",
            color_discrete_map=color_map,
            title="SKU buying signal distribution"
        )
        fig_donut.update_traces(textposition="outside", textinfo="percent+label")
        fig_donut.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col2:
        # Units bought this vs next season by signal
        signal_compare = forecast.groupby("buying_signal").agg(
            units_this=("units_purchased", "sum"),
            units_next=("predicted_units_next_season", "sum")
        ).reset_index()

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="This season", x=signal_compare["buying_signal"],
            y=signal_compare["units_this"], marker_color="#ADC4E4"
        ))
        fig_bar.add_trace(go.Bar(
            name="Next season (ML)", x=signal_compare["buying_signal"],
            y=signal_compare["units_next"], marker_color="#1D7A5F"
        ))
        fig_bar.update_layout(
            barmode="group",
            title="Units: this season vs ML recommendation",
            height=380,
            legend=dict(orientation="h", y=-0.2),
            xaxis_title="Buying signal",
            yaxis_title="Total units",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Scatter: bleed score vs buying change
    st.markdown("#### Which SKUs need the biggest buying correction?")
    fig_scatter = px.scatter(
        forecast,
        x="bleed_score",
        y="vs_last_season",
        color="buying_signal",
        size="units_purchased",
        hover_name="product_name",
        hover_data=["category", "units_purchased", "predicted_units_next_season"],
        color_discrete_map=color_map,
        labels={
            "bleed_score":    "Bleed score (this season)",
            "vs_last_season": "Change in units recommended (next season)",
        },
        title="Higher bleed score → bigger buying reduction recommended",
        height=420,
    )
    fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_scatter.update_traces(marker=dict(opacity=0.75, line=dict(width=0.5, color="white")))
    st.plotly_chart(fig_scatter, use_container_width=True)


# ══ TAB 2: CATEGORY PLAN ════════════════════════════════════════════════════

with tab2:
    st.subheader("Category-wise buying plan")

    col1, col2 = st.columns(2)

    with col1:
        # Units comparison
        fig_cat = go.Figure()
        fig_cat.add_trace(go.Bar(
            name="This season", y=cat_plan["category"],
            x=cat_plan["bought_this_season"],
            orientation="h", marker_color="#ADC4E4"
        ))
        fig_cat.add_trace(go.Bar(
            name="Next season (ML)", y=cat_plan["category"],
            x=cat_plan["recommended_next"],
            orientation="h", marker_color="#1D7A5F"
        ))
        fig_cat.update_layout(
            barmode="group", title="Units: this vs next season",
            height=360, legend=dict(orientation="h", y=-0.2),
            xaxis_title="Total units", yaxis_title=""
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    with col2:
        # Capital comparison
        fig_cap = go.Figure()
        fig_cap.add_trace(go.Bar(
            name="This season", y=cat_plan["category"],
            x=cat_plan["capital_this"] / 1e5,
            orientation="h", marker_color="#F5A09A"
        ))
        fig_cap.add_trace(go.Bar(
            name="Next season (ML)", y=cat_plan["category"],
            x=cat_plan["capital_next"] / 1e5,
            orientation="h", marker_color="#1D7A5F"
        ))
        fig_cap.update_layout(
            barmode="group", title="Capital invested (₹ Lakhs)",
            height=360, legend=dict(orientation="h", y=-0.2),
            xaxis_title="₹ Lakhs", yaxis_title=""
        )
        st.plotly_chart(fig_cap, use_container_width=True)

    # Category summary table
    st.markdown("#### Category buying plan summary")
    display_cat = cat_plan.copy()
    display_cat["capital_this"]  = display_cat["capital_this"].apply(lambda x: f"₹{x/1e5:.1f}L")
    display_cat["capital_next"]  = display_cat["capital_next"].apply(lambda x: f"₹{x/1e5:.1f}L")
    display_cat["change_pct"]    = display_cat["change_pct"].apply(lambda x: f"{'▼' if x < 0 else '▲'} {abs(x)}%")
    display_cat.columns = ["Category", "SKUs", "Units this season",
                            "Units next season", "Capital this", "Capital next", "Change %"]
    st.dataframe(display_cat, use_container_width=True, hide_index=True)


# ══ TAB 3: SKU-LEVEL FORECAST ════════════════════════════════════════════════

with tab3:
    st.subheader("SKU-level next season forecast")

    col1, col2, col3 = st.columns(3)
    cat_filter    = col1.multiselect("Filter by category", options=sorted(forecast["category"].unique()), default=[])
    signal_filter = col2.multiselect("Filter by buying signal", options=sorted(forecast["buying_signal"].dropna().unique()), default=[])
    status_filter = col3.multiselect("Filter by this season status", options=["Critical", "Warning", "Healthy"], default=["Critical"])

    filtered = forecast.copy()
    if cat_filter:    filtered = filtered[filtered["category"].isin(cat_filter)]
    if signal_filter: filtered = filtered[filtered["buying_signal"].isin(signal_filter)]
    if status_filter: filtered = filtered[filtered["status"].isin(status_filter)]

    st.markdown(f"Showing **{len(filtered)}** SKUs")

    # Forecast table
    show_cols = {
        "product_name":                   "Product",
        "category":                        "Category",
        "status":                          "This season status",
        "units_purchased":                 "Bought this season",
        "predicted_units_next_season":     "Buy next season (ML)",
        "units_min":                       "Min (conservative)",
        "units_max":                       "Max (aggressive)",
        "vs_last_season":                  "Change",
        "buying_signal":                   "Signal",
        "estimated_capital_needed":        "Capital needed (₹)",
    }
    display_df = filtered[list(show_cols.keys())].rename(columns=show_cols)
    display_df["Capital needed (₹)"] = display_df["Capital needed (₹)"].apply(lambda x: f"₹{x:,.0f}")

    def highlight_signal(val):
        colors = {
            "Buy much less":  "background-color:#FCEBEB",
            "Buy less":       "background-color:#FFF3E0",
            "Same":           "background-color:#F5F5F5",
            "Buy more":       "background-color:#E8F5E9",
            "Buy much more":  "background-color:#C8E6C9",
        }
        return colors.get(val, "")

    styled = display_df.style.map(highlight_signal, subset=["Signal"])
    st.dataframe(styled, use_container_width=True, hide_index=True, height=420)

    # Top 10 to buy more / less
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🟢 Top 10 — Buy more next season")
        top_more = forecast.nlargest(10, "vs_last_season")[["product_name", "category", "vs_last_season"]]
        top_more.columns = ["Product", "Category", "Additional units"]
        fig_more = px.bar(top_more, x="Additional units", y="Product", orientation="h",
                          color_discrete_sequence=["#1D7A5F"], height=340)
        fig_more.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
        st.plotly_chart(fig_more, use_container_width=True)

    with col2:
        st.markdown("#### 🔴 Top 10 — Buy less next season")
        top_less = forecast.nsmallest(10, "vs_last_season")[["product_name", "category", "vs_last_season"]]
        top_less.columns = ["Product", "Category", "Units to cut"]
        top_less["Units to cut"] = top_less["Units to cut"].abs()
        fig_less = px.bar(top_less, x="Units to cut", y="Product", orientation="h",
                          color_discrete_sequence=["#E24B4A"], height=340)
        fig_less.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
        st.plotly_chart(fig_less, use_container_width=True)


# ══ TAB 4: THIS SEASON'S BLEED ══════════════════════════════════════════════

with tab4:
    st.subheader("This season's inventory bleed — context for the ML forecast")
    st.caption("The ML model learned from these patterns to make next season's recommendations")

    col1, col2 = st.columns(2)

    with col1:
        # Status breakdown
        status_counts = prescriptions["status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        fig_status = px.bar(
            status_counts, x="status", y="count",
            color="status",
            color_discrete_map={"Critical": "#E24B4A", "Warning": "#EF9F27", "Healthy": "#639922"},
            title="SKU health breakdown this season",
            height=320,
        )
        fig_status.update_layout(showlegend=False, xaxis_title="", yaxis_title="SKU count")
        st.plotly_chart(fig_status, use_container_width=True)

    with col2:
        # Margin at risk by category
        cat_risk = prescriptions.groupby("category")["margin_at_risk"].sum().reset_index()
        cat_risk["margin_at_risk_L"] = cat_risk["margin_at_risk"] / 1e5
        fig_risk = px.bar(
            cat_risk.sort_values("margin_at_risk_L", ascending=True),
            x="margin_at_risk_L", y="category", orientation="h",
            color_discrete_sequence=["#E24B4A"],
            title="Margin at risk by category (₹ Lakhs)",
            height=320,
        )
        fig_risk.update_layout(showlegend=False, xaxis_title="₹ Lakhs", yaxis_title="")
        st.plotly_chart(fig_risk, use_container_width=True)

    # Feature importance
    st.markdown("#### What the ML model found most important")
    importance_data = pd.DataFrame({
        "Feature": ["sell_through_rate", "velocity_overall", "days_on_shelf",
                    "mrp", "price_positioning", "cost_price",
                    "margin_ratio", "size_demand", "category", "channel"],
        "Importance": [0.655, 0.229, 0.059, 0.013, 0.012, 0.010, 0.007, 0.007, 0.004, 0.003]
    }).sort_values("Importance")

    fig_imp = px.bar(
        importance_data, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale=["#EAF3DE", "#1D7A5F"],
        title="Feature importance — what drove the buying forecast",
        height=360,
    )
    fig_imp.update_layout(coloraxis_showscale=False, xaxis_title="Importance score", yaxis_title="")
    st.plotly_chart(fig_imp, use_container_width=True)

    st.info(
        "**Key insight:** Sell-through rate (65.5%) and velocity (22.9%) explain 88% of the model's "
        "decisions. Category, price, and size barely matter. "
        "This means **buying the right quantity of proven sellers** is far more impactful than "
        "diversifying into new SKUs."
    )
