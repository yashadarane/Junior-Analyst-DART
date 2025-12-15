import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="AI M&A Scout",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/ml/master_with_fair_values.csv")
        return df
    except FileNotFoundError:
        st.error("Data not found! Please run 'model.py' first to generate the analysis.")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

st.sidebar.title("Deal Screener")
st.sidebar.caption("Filter targets based on AI Strategy")

# --- EXPOSED FILTERS ---

# 0. Toggle to show all companies (ignore filters for table/deep dive)
show_all = st.sidebar.checkbox("Show all companies (ignore filters in table)", value=False)

# 1. Strategy Cluster Filter
available_clusters = df["cluster_label"].dropna().unique()
selected_clusters = st.sidebar.multiselect(
    "Strategic Rationale (Cluster)",
    available_clusters,
    default=available_clusters
)

# 2. Min Upside Filter (Aggressiveness)
min_upside = st.sidebar.slider("Min AI Upside (%)", 0, 100, 15)

# 3. Min Quality Filter (Conservative vs Aggressive)
min_quality = st.sidebar.slider("Min Quality Score (0-1)", 0.0, 1.0, 0.5, step=0.05)

# 4. Risk Profile Filter
risk_options = df["risk_label"].unique() if "risk_label" in df.columns else []
selected_risks = st.sidebar.multiselect(
    "Risk Profile (News)",
    risk_options,
    default=risk_options
)

# --- APPLY FILTERS FOR METRICS / STRATEGIC VIEW ---
filtered_df = df[
    (df["cluster_label"].isin(selected_clusters)) &
    (df["implied_upside_pct"] >= min_upside) &
    (df["earnings_quality"] >= min_quality)
]

if selected_risks:
    filtered_df = filtered_df[filtered_df["risk_label"].isin(selected_risks)]

# This dataframe drives the table + deep dive when show_all is False
display_df = df if show_all else filtered_df

# --- MAIN DASHBOARD ---
st.title("AI Deal Origination")
st.markdown("#### Identifying Undervalued Targets using Random Forest & NLP Sentiment")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Qualified Targets", len(filtered_df))
col2.metric(
    "Median AI Upside",
    f"{filtered_df['implied_upside_pct'].median():.1f}%"
    if not filtered_df.empty else "N/A"
)
col3.metric(
    "Avg Quality Score",
    f"{filtered_df['earnings_quality'].mean():.2f}"
    if not filtered_df.empty else "N/A"
)

if (not filtered_df.empty) and ("sentiment_score" in filtered_df.columns):
    avg_sent = filtered_df["sentiment_score"].mean()
    col4.metric(
        "Market Sentiment",
        f"{avg_sent:.2f}",
        delta="Positive" if avg_sent > 0 else "Negative"
    )
else:
    col4.metric("Market Sentiment", "N/A")

st.divider()

tab1, tab2 = st.tabs(["Deal Sheet", "Strategic Map"])

with tab1:
    st.subheader("Top Opportunities" if not show_all else "Full Universe")

    if display_df.empty:
        st.caption("No companies match the current filters.")
    else:
        display_cols = [
            "ticker", "cluster_label", "current_price",
            "implied_price_per_share", "implied_upside_pct",
            "risk_label", "earnings_quality"
        ]
        valid_cols = [c for c in display_cols if c in display_df.columns]

        st.dataframe(
            display_df[valid_cols].sort_values(by="implied_upside_pct", ascending=False),
            column_config={
                "ticker": "Company",
                "cluster_label": "Strategy",
                "implied_upside_pct": st.column_config.ProgressColumn(
                    "Upside %", format="%.1f%%", min_value=0, max_value=100
                ),
                "implied_price_per_share": st.column_config.NumberColumn(
                    "AI Fair Value", format="%.2f"
                ),
                "current_price": st.column_config.NumberColumn(
                    "Price", format="%.2f"
                ),
                "earnings_quality": st.column_config.NumberColumn(
                    "Quality", format="%.2f"
                ),
                "risk_label": st.column_config.TextColumn("News Sentiment"),
            },
            use_container_width=True,
            height=500,
        )

with tab2:
    st.subheader("Market Positioning (Growth vs Quality)")
    if not filtered_df.empty:
        fig = px.scatter(
            filtered_df,
            x="revenue_growth",
            y="roe",
            color="cluster_label",
            size="market_cap",
            hover_name="ticker",
            hover_data=["implied_upside_pct", "risk_label"],
            color_discrete_sequence=px.colors.qualitative.Bold,
            labels={"revenue_growth": "Revenue Growth", "roe": "ROE (Quality)"},
            title="Strategic Peer Groups",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No companies match the current filters for the strategic map.")

st.divider()
st.subheader("Deep Dive: Comprehensive Analysis")

if display_df.empty:
    st.caption("No companies available for deep dive.")
else:
    selected_ticker = st.selectbox(
        "Select Company to Analyze",
        display_df["ticker"].unique()
    )

    if selected_ticker:
        row = df[df["ticker"] == selected_ticker].iloc[0]

        mcap_cr = row.get("market_cap", 0) / 10_000_000
        ev_cr = row.get("enterprise_value", 0) / 10_000_000

        c_h1, c_h2, c_h3, c_h4 = st.columns(4)
        c_h1.metric("Current Price", f"{row['current_price']:,.2f}")
        c_h2.metric(
            "AI Fair Value",
            f"{row['implied_price_per_share']:,.2f}",
            delta=f"{row['implied_upside_pct']:.1f}% Upside",
        )
        c_h3.metric("Market Cap", f"{mcap_cr:,.0f} Cr")
        c_h4.metric("Strategy", row["cluster_label"])

        st.write("")

        col_left, col_right = st.columns([2, 1])

        with col_left:
            fig_ff = go.Figure()

            curr = row["current_price"]
            fair = row["implied_price_per_share"]

            fig_ff.add_trace(
                go.Bar(
                    y=["AI Model"],
                    x=[fair * 0.2],
                    base=[fair * 0.9],
                    orientation="h",
                    marker_color="#00CC96",
                    name="Fair Value Range",
                )
            )

            fig_ff.add_vline(
                x=curr,
                line_dash="dash",
                line_color="red",
                annotation_text="Current Price",
            )

            fig_ff.update_layout(
                title=f"Valuation Gap Analysis: {selected_ticker}",
                xaxis_title="Share Price (INR)",
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_ff, use_container_width=True)

        with col_right:
            st.markdown("### AI Scorecard")
            st.caption("Proprietary scores (0-1) based on fundamentals")

            st.progress(
                row["earnings_quality"],
                text=f"Earnings Quality: {row['earnings_quality']:.2f}",
            )
            st.progress(
                row["value_score"],
                text=f"Value Score: {row['value_score']:.2f}",
            )
            st.progress(
                row["health_score"],
                text=f"Health Score: {row['health_score']:.2f}",
            )
            st.progress(
                row["growth_score"],
                text=f"Growth Score: {row['growth_score']:.2f}",
            )

        tab_val, tab_perf, tab_risk, tab_news = st.tabs(
            ["Valuation", "Performance", "Risk & Health", "News Analysis"]
        )

        with tab_val:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("P/E Ratio", f"{row.get('pe_ratio', 0):.1f}x")
            c2.metric("P/B Ratio", f"{row.get('price_to_book', 0):.1f}x")
            c3.metric("EV / EBITDA", f"{row.get('ev_ebitda', 0):.1f}x")
            c4.metric("Ent. Value (Cr)", f"{ev_cr:,.0f}")
            st.caption("*Valuation multiples compared to sector averages.*")

        with tab_perf:
            c1, c2, c3, c4 = st.columns(4)
            rev_g = row.get("revenue_growth", 0) * 100
            pm = row.get("profit_margins", 0) * 100
            roe = row.get("roe", 0) * 100

            c1.metric("Revenue Growth", f"{rev_g:.1f}%")
            c2.metric("Profit Margins", f"{pm:.1f}%")
            c3.metric("Return on Equity (ROE)", f"{roe:.1f}%")
            c4.metric("Beta", f"{row.get('beta', 0):.2f}")

        with tab_risk:
            c1, c2, c3 = st.columns(3)
            de = row.get("debt_to_equity", 0)
            cr = row.get("current_ratio", 0)
            fcf_cr = row.get("free_cashflow", 0) / 10_000_000

            c1.metric("Debt / Equity", f"{de:.1f}%", help="Lower is better. >200% is risky.")
            c2.metric("Current Ratio", f"{cr:.2f}x", help=">1.5 is healthy.")
            c3.metric("Free Cash Flow", f"{fcf_cr:,.0f} Cr")

            st.info(
                f"Risk Assessment: **{row.get('risk_label', 'Neutral')}** based on news sentiment."
            )

        with tab_news:
            st.markdown("### Latest Headlines")

            sentiment = row.get("sentiment_score", 0)
            if sentiment > 0.1:
                st.success(f"Positive Sentiment ({sentiment:.2f})")
            elif sentiment < -0.1:
                st.error(f"Negative Sentiment ({sentiment:.2f})")
            else:
                st.warning(f"Neutral Sentiment ({sentiment:.2f})")

            news_text = row.get("news_text", "")
            if pd.notna(news_text) and str(news_text).strip() != "":
                st.markdown(f"> {news_text}")
            else:
                st.caption("No recent news text available for analysis.")
