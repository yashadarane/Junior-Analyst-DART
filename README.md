# Junior Analyst DART
### Digital Asset Research Toolkit

> **"Wall Street grade analysis, automated at the speed of code."**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red) ![Sklearn](https://img.shields.io/badge/AI-Scikit--Learn-orange) ![Status](https://img.shields.io/badge/Status-Operational-success)

---

**Junior Analyst DART** is not just a stock screener; it is an autonomous financial intelligence engine.

In a market drowning in noise, DART acts as your tireless 24/7 quantitative analyst. It ingests the entire **NIFTY 500** universe, processes thousands of financial data points, reads the news using NLP, and applies machine learning to identify **asymmetric upside opportunities**.

It automates the grunt work of Investment Banking—comps analysis, valuation modeling, and risk assessment—delivering a shortlist of high-quality, undervalued targets in seconds.

---

## The Intelligence Engine (How It Works)
DART combines **Fundamental Finance**, **Supervised Learning**, and **Unsupervised Clustering** to triangulate value.

### 1. The Financial Core
* **Automated Data Ingestion:** Scrapes and normalizes balance sheets, income statements, and cash flows for 500+ companies.
* **Smart Imputation:** Uses **KNN (K-Nearest Neighbors)** to intelligently fill missing financial data gaps based on peer group behavior, ensuring no diamond in the rough is skipped due to incomplete data.
* **Sector-Aware Logic:** Automatically distinguishes between Banking/NBFCs and General Corporates to apply the correct valuation metrics.

### 2. The Machine Learning Layer
* **Valuation Prediction (Random Forest):** DART doesn't just look at P/E. It trains a **Random Forest Regressor** on "Quality" companies to learn the mathematical relationship between fundamentals (ROIC, Growth, Margins) and Enterprise Value. It then applies this logic to the rest of the market to find mispriced assets.
* **Strategic Clustering (K-Means):** Unsupervised learning segments the market into 5 strategic buckets:
    * **Cash Cows**
    * **High Growth**
    * **Quality Growth**
    * **Value Traps**
    * **Turnaround Plays**

### 3. The Sentiment Overlay
* **NLP News Reader:** Connects to live RSS feeds to fetch real-time headlines.
* **Sentiment Analysis:** Uses **TextBlob** to compute a polarity score, instantly flagging "High Risk" targets regardless of their financial appeal.

---

## Key Features

| Feature | Description |
| :--- | :--- |
| **Fair Value Modeling** | Calculates an "AI Implied Price" and "Upside %" based on fundamental health rather than market hype. |
| **Quality Scoring** | Proprietary algorithms rank companies (0-100) on **Value, Health, Growth, and Earnings Quality**. |
| **Deal Screener** | Interactive dashboard to filter targets by Strategic Cluster (e.g., "Show me Undervalued High Growth"). |
| **Deep Dive View** | Instant "Tear Sheet" generation with valuation bridges, peer positioning maps, and risk metrics. |

---

## Tech Stack
* **Data Pipeline:** `Pandas`, `Numpy`, `YFinance`, `Requests`
* **Machine Learning:** `Scikit-Learn` (RandomForest, KMeans, StandardScaler, KNNImputer)
* **NLP:** `TextBlob`, `XML`
* **Visualization:** `Streamlit`, `Plotly Express`, `Plotly Graph Objects`
