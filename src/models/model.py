import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import warnings
warnings.filterwarnings('ignore')
from textblob import TextBlob


INPUT = "data/processed/clean/master_clean.csv"
OUT_MASTER = "data/ml/master_with_fair_values.csv"
MODEL_DIR = "data/ml"
os.makedirs(MODEL_DIR, exist_ok=True)


df = pd.read_csv(INPUT)
df = df.reset_index(drop=True)
print(f"Loaded {len(df)} companies")


feature_cols = [
    "price_to_book", "ev_ebitda", "debt_to_equity", "current_ratio",
    "revenue_growth", "profit_margins", "roe", "beta",
    "value_score", "health_score", "growth_score", "earnings_quality"
]
features = [c for c in feature_cols if c in df.columns]
print(f"Using {len(features)} features: {features}")


print("\n=== CLUSTERING ===")
X_cluster = df[features].fillna(0).values
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_cluster_scaled)

cluster_labels = {
    0: "Cash Cows",
    1: "High Growth",
    2: "Value Traps",
    3: "Turnaround",
    4: "Quality Growth"
}
df["cluster_label"] = df["cluster"].map(cluster_labels)
joblib.dump(kmeans, f"{MODEL_DIR}/kmeans_clusters.pkl")
joblib.dump(scaler_cluster, f"{MODEL_DIR}/cluster_scaler.pkl")


# --------- FINANCE ANCHOR LAYER ---------
for col, default in [
    ("market_cap", 0.0),
    ("total_debt", 0.0),
    ("cash_and_equiv", 0.0),
    ("current_price", 0.0),
    ("free_cashflow", 0.0),
    ("enterprise_value", np.nan),
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
    else:
        df[col] = default

df["ev_actual"] = (df["market_cap"] + df["total_debt"] - df["cash_and_equiv"]).clip(lower=0)
df["net_debt"] = df["total_debt"] - df["cash_and_equiv"]


# --------- TRAINING DATA FOR EV ADJUSTMENT ---------
quality_mask = (
    (df["earnings_quality"] > 0.3) &
    df["ev_actual"].notna() & (df["ev_actual"] > 0) &
    df["enterprise_value"].notna() & (df["enterprise_value"] > 0)
)

df_quality = df[quality_mask].copy()
print(f"\nQuality Training Set: {len(df_quality)} companies (Filtered for sanity)")

df_quality["ev_adjustment_target"] = np.clip(
    df_quality["enterprise_value"] / df_quality["ev_actual"],
    0.5,
    1.5
)

X_full = df_quality[features].fillna(0.0).values
scaler = StandardScaler()
X_full_scaled = scaler.fit_transform(X_full)
joblib.dump(scaler, f"{MODEL_DIR}/fairvalue_scaler.pkl")

rf_adj = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)
rf_adj.fit(X_full_scaled, df_quality["ev_adjustment_target"].values)
joblib.dump(rf_adj, f"{MODEL_DIR}/rf_ev_adjustment.pkl")
print("EV adjustment model trained")


# --------- APPLY MODEL TO FULL UNIVERSE ---------
X_full_universe = scaler.transform(df[features].fillna(0.0).values)
adj_pred = rf_adj.predict(X_full_universe)
adj_pred = np.clip(adj_pred, 0.5, 1.5)

df["predicted_enterprise_value"] = df["ev_actual"] * adj_pred


# --------- SHARES AND IMPLIED PRICE ---------
def infer_shares(row):
    if "sharesOutstanding" in df.columns and pd.notna(row.get("sharesOutstanding")):
        return pd.to_numeric(row["sharesOutstanding"], errors="coerce")
    mc = row.get("market_cap", np.nan)
    cp = row.get("current_price", np.nan)
    if pd.notna(mc) and pd.notna(cp) and cp > 0:
        return mc / cp
    return np.nan


df["shares_outstanding"] = df.apply(infer_shares, axis=1)
df["shares_outstanding"] = df["shares_outstanding"].clip(1e7, 2e10)

df["equity_value_model"] = df["predicted_enterprise_value"] - df["net_debt"]

df["implied_price_raw"] = np.where(
    (df["equity_value_model"].notna()) & (df["equity_value_model"] > 0) &
    (df["shares_outstanding"].notna()) & (df["shares_outstanding"] > 0),
    df["equity_value_model"] / df["shares_outstanding"],
    np.nan
)

ratio = np.where(
    (df["current_price"] > 0) & df["implied_price_raw"].notna(),
    df["implied_price_raw"] / df["current_price"],
    np.nan
)
far_mask = (ratio > 3.0) | (ratio < (1.0 / 3.0))
df.loc[far_mask, "implied_price_raw"] = (
    0.7 * df.loc[far_mask, "implied_price_raw"] +
    0.3 * df.loc[far_mask, "current_price"]
)


# --------- UPSIDE WITH SOFT SHRINK ---------
cp = df["current_price"]

df["upside_pct_raw"] = np.where(
    (cp.notna()) & (cp > 0) & df["implied_price_raw"].notna(),
    (df["implied_price_raw"] / cp - 1.0) * 100.0,
    np.nan
)

up = df["upside_pct_raw"].copy()
high_mask = up > 150.0
up.loc[high_mask] = 150.0 + 0.25 * (up.loc[high_mask] - 150.0)
up = up.clip(-50.0, 250.0)

df["implied_upside_pct"] = up
df["implied_price_per_share"] = np.where(
    cp.notna() & (cp > 0) & df["implied_upside_pct"].notna(),
    cp * (1.0 + df["implied_upside_pct"] / 100.0),
    np.nan
)

for col in ["predicted_enterprise_value", "implied_price_per_share", "implied_upside_pct"]:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)


final_quality_mask = (
    (df["earnings_quality"] > 0.35) &
    df["implied_upside_pct"].notna()
)
print(f"Final quality candidates (soft internal check): {final_quality_mask.sum()} companies")


print("\nRUNNING REAL-TIME SENTIMENT ENGINE")


def calculate_sentiment(text):
    if not isinstance(text, str) or len(text) < 5:
        return 0.0
    return TextBlob(text).sentiment.polarity


df["sentiment_score"] = df["news_text"].fillna("").apply(calculate_sentiment)
df["risk_label"] = df["sentiment_score"].apply(
    lambda x: "High Risk" if x < -0.1 else ("Positive" if x > 0.1 else "Neutral")
)

df.to_csv(OUT_MASTER, index=False)
joblib.dump(df, f"{MODEL_DIR}/master_predictions.pkl")
print(f"\nWrote {OUT_MASTER}")