import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

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

# 1. CLUSTERING: Strategic Peer Groups (Unsupervised)
print("\n=== CLUSTERING ===")
X_cluster = df[features].fillna(0).values
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_cluster_scaled)

cluster_labels = {
    0: 'Cash Cows',      # High ROE, stable margins
    1: 'High Growth',    # Revenue growth leaders
    2: 'Value Traps',    # Cheap but poor quality
    3: 'Turnaround',     # Improving fundamentals
    4: 'Quality Growth'  # High ROE + growth
}
df['cluster_label'] = df['cluster'].map(cluster_labels)
joblib.dump(kmeans, f"{MODEL_DIR}/kmeans_clusters.pkl")
joblib.dump(scaler_cluster, f"{MODEL_DIR}/cluster_scaler.pkl")

print("Cluster distribution:")
print(df['cluster_label'].value_counts().sort_index())

# 2. QUALITY FILTER FOR VALUATION MODEL
quality_mask = (
    (df.get('earnings_quality', pd.Series(0, index=df.index)) > 0.1) & 
    (df.get('health_score', pd.Series(0, index=df.index)) > 0.2)
)
df_quality = df[quality_mask].copy()
print(f"\nQuality filtered: {len(df_quality)} companies")

# 3. VALUATION MODELS (Supervised)
X_full = df_quality[features].fillna(0).values
scaler = StandardScaler()
X_full_scaled = scaler.fit_transform(X_full)
joblib.dump(scaler, f"{MODEL_DIR}/fairvalue_scaler.pkl")

# Enterprise Value model
df_ev = df_quality[df_quality["enterprise_value"].notna() & (df_quality["enterprise_value"] > 0)].copy()
if len(df_ev) >= 10:
    X_ev = df_ev[features].fillna(0).values
    y_ev = np.log1p(pd.to_numeric(df_ev["enterprise_value"], errors="coerce").fillna(0).values)
    X_ev_scaled = scaler.transform(X_ev)
    X_tr, X_te, y_tr, y_te = train_test_split(X_ev_scaled, y_ev, test_size=0.2, random_state=42)
    rf_ev = RandomForestRegressor(n_estimators=250, max_depth=16, random_state=42, n_jobs=-1)
    rf_ev.fit(X_tr, y_tr)
    joblib.dump(rf_ev, f"{MODEL_DIR}/rf_ev_model.pkl")
    print(f"EV model trained on {len(df_ev)} samples")
else:
    rf_ev = None

# P/E model
df_pe = df_quality[df_quality["pe_ratio"].notna() & (df_quality["pe_ratio"] > 0)].copy()
if len(df_pe) >= 10:
    X_pe = df_pe[features].fillna(0).values
    y_pe = np.log1p(pd.to_numeric(df_pe["pe_ratio"], errors="coerce").fillna(0).values)
    X_pe_scaled = scaler.transform(X_pe)
    X_trp, X_tesp, y_trp, y_tesp = train_test_split(X_pe_scaled, y_pe, test_size=0.2, random_state=42)
    rf_pe = RandomForestRegressor(n_estimators=250, max_depth=16, random_state=42, n_jobs=-1)
    rf_pe.fit(X_trp, y_trp)
    joblib.dump(rf_pe, f"{MODEL_DIR}/rf_pe_model.pkl")
    print(f"PE model trained on {len(df_pe)} samples")
else:
    rf_pe = None

# 4. PREDICTIONS FOR FULL UNIVERSE
X_full_universe = scaler.transform(df[features].fillna(0).values)

if rf_ev is not None:
    pred_log_ev = rf_ev.predict(X_full_universe)
    df["predicted_enterprise_value"] = np.expm1(pred_log_ev)
else:
    df["predicted_enterprise_value"] = np.nan

if rf_pe is not None:
    pred_log_pe = rf_pe.predict(X_full_universe)
    df["predicted_pe"] = np.expm1(pred_log_pe)
else:
    df["predicted_pe"] = np.nan

# 5. FAIR VALUE CALCULATIONS
for col, default in [("market_cap", 0), ("current_price", 0), ("total_debt", 0), ("free_cashflow", 0)]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
    else:
        df[col] = default

df["net_debt"] = df["total_debt"].fillna(0) - df["free_cashflow"].fillna(0)

def infer_shares(row):
    if "sharesOutstanding" in df.columns and pd.notna(row.get("sharesOutstanding")):
        return pd.to_numeric(row["sharesOutstanding"], errors="coerce")
    if pd.notna(row["market_cap"]) and pd.notna(row["current_price"]) and row["current_price"] > 0:
        return row["market_cap"] / row["current_price"]
    return np.nan

df["shares_outstanding"] = df.apply(infer_shares, axis=1)
df["shares_outstanding"] = np.clip(df["shares_outstanding"], 1e6, 1e11)

df["implied_equity_value"] = df["predicted_enterprise_value"] - df["net_debt"]
df["implied_price_per_share"] = np.where(
    (df["shares_outstanding"].notna()) & (df["shares_outstanding"] > 0),
    df["implied_equity_value"] / df["shares_outstanding"],
    np.nan
)

df["implied_price_to_current"] = np.where(
    (df["current_price"].notna()) & (df["current_price"] > 0),
    df["implied_price_per_share"] / df["current_price"],
    np.nan
)

df["implied_upside_pct"] = np.where(
    (df["current_price"].notna()) & (df["implied_price_per_share"].notna()),
    (df["implied_price_per_share"] / df["current_price"] - 1) * 100,
    np.nan
)

# Sanity caps and cleaning
df["implied_upside_pct"] = df["implied_upside_pct"].clip(-50, 200)
for col in ["predicted_enterprise_value", "implied_price_per_share", "implied_upside_pct"]:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)

# 6. FINAL QUALITY SCREEN
final_quality_mask = (
    (df['implied_upside_pct'] > 20) & 
    (df['implied_upside_pct'] < 200) &
    (df.get('earnings_quality', pd.Series(0, index=df.index)) > 0.15) &
    (df.get('health_score', pd.Series(0, index=df.index)) > 0.25)
)

df.to_csv(OUT_MASTER, index=False)
joblib.dump(df, f"{MODEL_DIR}/master_predictions.pkl")

print(f"\nWrote {OUT_MASTER}")
print("Saved ALL models to", MODEL_DIR)
print("\n=== CLUSTER SUMMARY ===")
print(df['cluster_label'].value_counts().sort_index())
print("\n=== TOP 10 QUALITY OPPORTUNITIES ===")
top_picks = df[final_quality_mask].nlargest(10, 'implied_upside_pct')
print(top_picks[['ticker', 'cluster_label', 'current_price', 'implied_price_per_share', 'implied_upside_pct', 'earnings_quality']].round(2))
print(f"\nTotal quality opportunities: {len(top_picks)}")
