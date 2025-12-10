import os
import json
import time
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

VALUATION_KEYS = {
    "current_price": "currentPrice",
    "market_cap": "marketCap",
    "enterprise_value": "enterpriseValue",
    "ev_ebitda": "enterpriseToEbitda",
    "pe_ratio": "trailingPE",
    "price_to_book": "priceToBook",
}

HEALTH_KEYS = {
    "total_debt": "totalDebt",
    "debt_to_equity": "debtToEquity",
    "free_cashflow": "freeCashflow",
    "current_ratio": "currentRatio",
}

GROWTH_KEYS = {
    "revenue_growth": "revenueGrowth",
    "profit_margins": "profitMargins",
    "roe": "returnOnEquity",
    "beta": "beta",
}

ALL_KEYS = {**VALUATION_KEYS, **HEALTH_KEYS, **GROWTH_KEYS}

CACHE_JSON = "data/processed/master.json"
CACHE_CSV = "data/processed/master.csv"

def _is_cache_fresh(path=CACHE_JSON, hours=24):
    if not os.path.exists(path):
        return False
    file_time = datetime.fromtimestamp(os.path.getmtime(path))
    return (datetime.now() - file_time) < timedelta(hours=hours)

def load_cached_data():
    if _is_cache_fresh():
        with open(CACHE_JSON, "r") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    return None

def fetch_single_ticker(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info = t.info or {}
    clean = {"ticker": ticker}
    for k, yk in ALL_KEYS.items():
        clean[k] = info.get(yk)
    os.makedirs("data/raw", exist_ok=True)
    raw_path = f"data/raw/{ticker.replace('.', '_')}.json"
    with open(raw_path, "w") as f:
        json.dump(clean, f, default=str, indent=2)
    return clean

def fetch_single_ticker_with_retry(ticker: str, max_retries: int = 5, base_delay: int = 30, pause_after_success: float = 1.0):
    delay = base_delay
    for attempt in range(1, max_retries + 1):
        try:
            rec = fetch_single_ticker(ticker)
            time.sleep(pause_after_success)
            return rec
        except Exception as e:
            msg = str(e)
            if "Rate limited" in msg or "Too Many Requests" in msg:
                print(f"[{ticker}] Rate limited on attempt {attempt}/{max_retries}, sleeping {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"Error fetching {ticker}: {e}")
                return None
    print(f"[{ticker}] Failed after {max_retries} attempts due to rate limiting")
    return None

def fetch_universe(universe_list):
    records = []
    for tk in universe_list:
        rec = fetch_single_ticker_with_retry(tk)
        if rec is not None:
            records.append(rec)
    if not records:
        print("No records fetched; master files will not be overwritten.")
        return pd.DataFrame()
    df = pd.DataFrame(records)
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(CACHE_CSV, index=False)
    with open(CACHE_JSON, "w") as f:
        json.dump(df.to_dict(orient="records"), f, default=str, indent=2)
    return df

def load_data(universe_list, force_refresh: bool = False):
    if not force_refresh:
        cached = load_cached_data()
        if cached is not None and not cached.empty:
            print("Loaded cached data (<24h old)")
            return cached
    print("Fetching fresh universe")
    return fetch_universe(universe_list)

if __name__ == "__main__":
    test_tickers = [
        "TCS.NS", "INFY.NS", "RELIANCE.NS",
        "HDFCBANK.NS", "ICICIBANK.NS", "TATAMOTORS.NS",
    ]
    print("Starting Data Loader")
    df = load_data(test_tickers, force_refresh=False)
    print("\nSuccess! Preview of data:")
    print(df.head())
    print(f"\nSaved to: {CACHE_CSV}")