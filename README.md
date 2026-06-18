#  Quantitative EDA of S&P 500 Market Fundamentals

> A data-driven exploration of 503 S&P 500 companies using real financial data — uncovering sector trends, valuation patterns, dividend behaviour, and market cap dynamics through statistical analysis and 7 publication-quality visualisations.

---

##  Project Overview

This project performs a full **Exploratory Data Analysis (EDA)** on a real-world dataset of S&P 500 company fundamentals. It covers data cleaning, statistical profiling, correlation analysis, and multi-chart visualisation — structured to demonstrate end-to-end analytical thinking for data science and finance roles.

| Detail | Value |
|---|---|
| **Dataset** | S&P 500 Companies & Financials |
| **Companies** | 503 |
| **Features** | 13 financial metrics |
| **Sectors** | 11 GICS broad groups |
| **Language** | Python 3 |

---

##  Project Structure

```
sp500-eda/
│
├── sp500_eda.py                      # Main EDA script (self-contained)
├── sp500_fundamentals.csv            # Raw dataset
├── README.md                         # This file
│
├── chart1_sector_distribution.png    # Company count & avg market cap by sector
├── chart2_pe_distribution.png        # P/E ratio distribution with KDE
├── chart3_market_cap.png             # Market cap log-scale + top 10 companies
├── chart4_correlation_heatmap.png    # Correlation matrix of key metrics
├── chart5_dividend_yield_by_sector.png  # Dividend yield box plots by sector
├── chart6_eps_vs_price.png           # EPS vs stock price scatter + regression
└── chart7_52week_positioning.png     # 52-week high/low price positioning
```

---

##  Setup & Installation

**Prerequisites:** Python 3.8+

```bash
# Clone the repository
git clone https://github.com/your-username/sp500-eda.git
cd sp500-eda

# Install dependencies
pip install pandas numpy matplotlib seaborn scipy

# Run the analysis
python sp500_eda.py
```

The script will auto-download the dataset from GitHub if `sp500_fundamentals.csv` is not found locally.

---

##  Visualisations

### Chart 1 — Sector Distribution
Company count and average market cap across 11 GICS sectors. Reveals which sectors dominate the index and where capital is concentrated.

### Chart 2 — P/E Ratio Distribution
Histogram with KDE overlay, median/mean reference lines, and a "fair value" zone (15–25×). Highlights the right-skewed nature of equity valuations.

### Chart 3 — Market Capitalisation
Log-scale distribution alongside a ranked top-10 bar chart. Shows the extreme concentration of wealth in a small number of mega-cap companies.

### Chart 4 — Correlation Heatmap
Lower-triangle correlation matrix across 7 key financial metrics. Surfaces the strongest relationships (EPS ↔ Price) and inverse patterns (Dividend Yield ↔ Price).

### Chart 5 — Dividend Yield by Sector
Box plots per sector showing spread and outliers. Utilities and Real Estate emerge as the highest-yielding sectors; Technology the lowest.

### Chart 6 — EPS vs Stock Price
Scatter plot with linear regression line and annotated tickers (AAPL, MSFT, NVDA, TSLA, etc.). Confirms EPS as the strongest single predictor of stock price.

### Chart 7 — 52-Week Price Positioning
Shows where each stock currently sits within its annual price range. Identifies momentum leaders (near highs) vs. distressed names (near lows) by sector.

---

##  Key Findings

1. **NVIDIA** is the largest S&P 500 company by market cap at ~$5.1 trillion — more than 128× the median company.

2. **Market cap is highly right-skewed** (skewness = 7.8). A small cluster of mega-caps dominate index weighting while the majority are mid-cap.

3. **EPS is the strongest predictor of stock price** with a Pearson correlation of r = 0.91 — the relationship is nearly linear across most sectors.

4. **Information Technology** commands the highest median P/E ratio, reflecting growth premiums. **Financials** trade at the lowest multiples.

5. **Dividend yield is negatively correlated with price** (r = −0.41) — companies with higher share prices tend to reinvest rather than distribute earnings.

6. **Real Estate and Utilities** pay the highest dividend yields; Technology and Consumer Discretionary pay the least.

7. **~39% of S&P 500 stocks** were trading within 30% of their 52-week high at the time of the dataset snapshot, signalling broad market strength.

---

##  Technical Highlights

- **Data cleaning:** handled 16–102 missing values per column; filtered outliers for visualisation (negative P/E, P/E > 200)
- **Feature engineering:** derived `52w_Range_Pct`, `Price_Pct_52w`, `MarketCap_B`, and mapped 127 sub-industries to 11 GICS sectors
- **Statistical methods:** descriptive statistics, Pearson correlation matrix, skewness analysis, Gaussian KDE, linear regression (scipy)
- **Visualisation:** 7 chart types — histogram, KDE overlay, horizontal bar, heatmap, box plot, scatter with regression, grouped bar

---

##  Dataset Details

| Column | Description |
|---|---|
| `Symbol` | Stock ticker |
| `Name` | Company name |
| `Sector` | GICS sub-industry |
| `Price` | Current stock price ($) |
| `Price/Earnings` | P/E ratio |
| `Dividend Yield` | Annual dividend yield (%) |
| `Earnings/Share` | EPS ($) |
| `52 Week Low / High` | Annual price range ($) |
| `Market Cap` | Total market capitalisation ($) |
| `EBITDA` | Earnings before interest, taxes, depreciation & amortisation ($) |
| `Price/Sales` | P/S ratio |
| `Price/Book` | P/B ratio |

---

##  Libraries Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, transformation |
| `numpy` | Numerical operations, array handling |
| `matplotlib` | Core charting engine |
| `seaborn` | Statistical visualisations (heatmap, box plots) |
| `scipy` | Linear regression, KDE, skewness |

---

##  Potential Extensions

- Add historical price time-series via `yfinance` for trend and volatility analysis
- Build a sector rotation model using rolling correlations
- Train a regression model to predict stock price from fundamentals
- Deploy as an interactive dashboard using Plotly Dash or Streamlit

---

##  Author

**Hardik Kumar**
📧 hardikumar131206@gmail.com
🔗 [linkedin.com/in/yourprofile](https://www.linkedin.com/in/hardik-kumar-7631a832b)
🐙 [github.com/your-username](https://github.com/hardikkumar131206-source)

---

##  License

This project is open source under the [MIT License](LICENSE).
