"""
=============================================================
  S&P 500 Fundamentals — Exploratory Data Analysis (EDA)
  Dataset : S&P 500 Companies & Financials (503 companies)
  Source  : https://github.com/datasets/s-and-p-500-companies-financials
=============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

# ── aesthetic config ─────────────────────────────────────
PALETTE   = "viridis"
BG        = "#0f1117"
CARD      = "#1a1d27"
ACCENT    = "#7c6af7"
TEXT      = "#e8eaf0"
GRID      = "#2a2d3a"
HIGHLIGHT = "#f7c948"

plt.rcParams.update({
    "figure.facecolor"  : BG,
    "axes.facecolor"    : CARD,
    "axes.edgecolor"    : GRID,
    "axes.labelcolor"   : TEXT,
    "axes.titlecolor"   : TEXT,
    "xtick.color"       : TEXT,
    "ytick.color"       : TEXT,
    "text.color"        : TEXT,
    "grid.color"        : GRID,
    "grid.linewidth"    : 0.6,
    "font.family"       : "DejaVu Sans",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
})

# ── helpers ───────────────────────────────────────────────
def add_watermark(fig):
    fig.text(0.99, 0.01, "S&P 500 EDA  |  github.com/datasets/s-and-p-500-companies-financials",
             ha="right", va="bottom", fontsize=7, color="#4a4d5a")

def save(fig, name):
    fig.savefig(name, dpi=160, bbox_inches="tight", facecolor=BG)
    print(f"  ✓  saved {name}")

# ═════════════════════════════════════════════════════════
#  0. LOAD & CLEAN
# ═════════════════════════════════════════════════════════
print("\n" + "═"*62)
print("  S&P 500 FUNDAMENTALS — EDA")
print("═"*62)

RAW = "raw_stock_data.csv"   # will try local; fall back to GitHub
try:
    df_raw = pd.read_csv(RAW)
    print(f"\n[Data]  Loaded local file: {RAW}")
except FileNotFoundError:
    import urllib.request
    URL = ("https://raw.githubusercontent.com/datasets/"
           "s-and-p-500-companies-financials/main/data/"
           "constituents-financials.csv")
    urllib.request.urlretrieve(URL, "sp500_fundamentals.csv")
    df_raw = pd.read_csv("sp500_fundamentals.csv")
    print(f"\n[Data]  Downloaded from: {URL}")

# Always load the fundamentals file downloaded earlier
df_raw = pd.read_csv("sp500_fundamentals.csv")

df = df_raw.copy()
df.drop(columns=["SEC Filings"], inplace=True, errors="ignore")

# Rename for convenience
df.rename(columns={
    "Price/Earnings"  : "PE_Ratio",
    "Dividend Yield"  : "Div_Yield",
    "Earnings/Share"  : "EPS",
    "52 Week Low"     : "Low_52w",
    "52 Week High"    : "High_52w",
    "Market Cap"      : "MarketCap",
    "Price/Sales"     : "PS_Ratio",
    "Price/Book"      : "PB_Ratio",
}, inplace=True)

df["MarketCap_B"]  = df["MarketCap"] / 1e9          # billions
df["52w_Range_Pct"] = ((df["High_52w"] - df["Low_52w"]) / df["Low_52w"]) * 100
df["Price_Pct_52w"] = ((df["Price"] - df["Low_52w"]) / (df["High_52w"] - df["Low_52w"])) * 100

# Map sub-industries to broad GICS sectors (11 groups)
sector_map = {
    "Health Care Equipment": "Health Care",
    "Pharmaceuticals": "Health Care",
    "Biotechnology": "Health Care",
    "Life Sciences Tools & Services": "Health Care",
    "Health Care Facilities": "Health Care",
    "Managed Health Care": "Health Care",
    "Health Care Services": "Health Care",
    "Health Care Technology": "Health Care",
    "Health Care Distributors": "Health Care",

    "Semiconductors": "Information Technology",
    "Application Software": "Information Technology",
    "Technology Hardware, Storage & Peripherals": "Information Technology",
    "IT Consulting & Other Services": "Information Technology",
    "Data Processing & Outsourced Services": "Information Technology",
    "Electronic Equipment & Instruments": "Information Technology",
    "Communications Equipment": "Information Technology",
    "Semiconductor Materials & Equipment": "Information Technology",
    "Systems Software": "Information Technology",
    "Internet Services & Infrastructure": "Information Technology",

    "Electric Utilities": "Utilities",
    "Multi-Utilities": "Utilities",
    "Gas Utilities": "Utilities",
    "Water Utilities": "Utilities",
    "Independent Power Producers & Energy Traders": "Utilities",

    "Industrial Machinery & Supplies & Components": "Industrials",
    "Aerospace & Defense": "Industrials",
    "Construction & Engineering": "Industrials",
    "Air Freight & Logistics": "Industrials",
    "Airlines": "Industrials",
    "Building Products": "Industrials",
    "Commercial Printing": "Industrials",
    "Diversified Support Services": "Industrials",
    "Electrical Components & Equipment": "Industrials",
    "Environmental & Facilities Services": "Industrials",
    "Ground Transportation": "Industrials",
    "Human Resource & Employment Services": "Industrials",
    "Office Services & Supplies": "Industrials",
    "Research & Consulting Services": "Industrials",
    "Security & Alarm Services": "Industrials",
    "Trading Companies & Distributors": "Industrials",
    "Trucking": "Industrials",
    "Waste Management": "Industrials",

    "Asset Management & Custody Banks": "Financials",
    "Financial Exchanges & Data": "Financials",
    "Investment Banking & Brokerage": "Financials",
    "Consumer Finance": "Financials",
    "Diversified Banks": "Financials",
    "Insurance Brokers": "Financials",
    "Life & Health Insurance": "Financials",
    "Property & Casualty Insurance": "Financials",
    "Regional Banks": "Financials",
    "Reinsurance": "Financials",
    "Thrifts & Mortgage Finance": "Financials",
    "Multi-line Insurance": "Financials",

    "Oil & Gas Exploration & Production": "Energy",
    "Integrated Oil & Gas": "Energy",
    "Oil & Gas Equipment & Services": "Energy",
    "Oil & Gas Refining & Marketing": "Energy",
    "Oil & Gas Storage & Transportation": "Energy",

    "Packaged Foods & Meats": "Consumer Staples",
    "Food Distributors": "Consumer Staples",
    "Food Retail": "Consumer Staples",
    "Household Products": "Consumer Staples",
    "Personal Care Products": "Consumer Staples",
    "Tobacco": "Consumer Staples",
    "Soft Drinks & Non-alcoholic Beverages": "Consumer Staples",
    "Drug Retail": "Consumer Staples",
    "Agricultural Products & Services": "Consumer Staples",
    "Brewers": "Consumer Staples",
    "Hypermarkets & Super Centers": "Consumer Staples",

    "Hotels, Resorts & Cruise Lines": "Consumer Discretionary",
    "Automobiles": "Consumer Discretionary",
    "Auto Components": "Consumer Discretionary",
    "Broadline Retail": "Consumer Discretionary",
    "Casinos & Gaming": "Consumer Discretionary",
    "Home Improvement Retail": "Consumer Discretionary",
    "Homebuilding": "Consumer Discretionary",
    "Internet & Direct Marketing Retail": "Consumer Discretionary",
    "Leisure Products": "Consumer Discretionary",
    "Restaurants": "Consumer Discretionary",
    "Specialty Retail": "Consumer Discretionary",
    "Textiles, Apparel & Luxury Goods": "Consumer Discretionary",
    "Movies & Entertainment": "Consumer Discretionary",

    "Diversified Telecommunication Services": "Communication Services",
    "Wireless Telecommunication Services": "Communication Services",
    "Integrated Telecommunication Services": "Communication Services",
    "Interactive Home Entertainment": "Communication Services",
    "Interactive Media & Services": "Communication Services",
    "Media": "Communication Services",
    "Advertising": "Communication Services",
    "Cable & Satellite": "Communication Services",
    "Publishing": "Communication Services",

    "Specialty Chemicals": "Materials",
    "Commodity Chemicals": "Materials",
    "Construction Materials": "Materials",
    "Diversified Metals & Mining": "Materials",
    "Gold": "Materials",
    "Paper & Forest Products": "Materials",
    "Steel": "Materials",
    "Aluminum": "Materials",
    "Copper": "Materials",
    "Fertilizers & Agricultural Chemicals": "Materials",
    "Industrial Gases": "Materials",
    "Metal & Glass Containers": "Materials",
    "Paper Packaging & Products": "Materials",
    "Timber REITs": "Materials",

    "Diversified REITs": "Real Estate",
    "Retail REITs": "Real Estate",
    "Residential REITs": "Real Estate",
    "Office REITs": "Real Estate",
    "Industrial REITs": "Real Estate",
    "Hotel & Resort REITs": "Real Estate",
    "Health Care REITs": "Real Estate",
    "Specialized REITs": "Real Estate",
    "Real Estate Services": "Real Estate",
    "Real Estate Development": "Real Estate",
}
df["GICS_Sector"] = df["Sector"].map(sector_map).fillna("Other")

# Cap P/E for visualisation (remove negative and >500 outliers)
df_pe = df[(df["PE_Ratio"] > 0) & (df["PE_Ratio"] < 200)].copy()
df_mc = df.dropna(subset=["MarketCap_B"]).copy()

print(f"\n[Data]  {len(df)} companies  |  {df.shape[1]} features")
print(f"[Data]  Sectors: {df['GICS_Sector'].nunique()} broad groups")
print(f"[Data]  Missing values:\n{df.isnull().sum()[df.isnull().sum()>0].to_string()}")

# ═════════════════════════════════════════════════════════
#  STATISTICAL SUMMARY
# ═════════════════════════════════════════════════════════
print("\n" + "─"*62)
print("  STATISTICAL SUMMARY")
print("─"*62)
cols_stat = ["Price","PE_Ratio","Div_Yield","EPS","MarketCap_B",
             "PS_Ratio","PB_Ratio","52w_Range_Pct"]
print(df[cols_stat].describe().round(2).to_string())

skew_pe   = df_pe["PE_Ratio"].skew()
skew_mc   = df_mc["MarketCap_B"].skew()
print(f"\nSkewness  —  P/E Ratio: {skew_pe:.2f}  |  Market Cap: {skew_mc:.2f}")

corr_matrix = df[["Price","PE_Ratio","Div_Yield","EPS",
                   "MarketCap_B","PS_Ratio","PB_Ratio"]].corr()
print("\nCorrelation matrix (key pairs):")
print(corr_matrix.round(2).to_string())

# ═════════════════════════════════════════════════════════
#  CHART 1 — Sector Distribution (count + avg market cap)
# ═════════════════════════════════════════════════════════
print("\n[Charts] Generating 7 visualisations …\n")

sector_counts = df.groupby("GICS_Sector").size().sort_values()
sector_mc     = df_mc.groupby("GICS_Sector")["MarketCap_B"].mean().reindex(sector_counts.index)

fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
fig.suptitle("S&P 500 — Sector Landscape", fontsize=16, fontweight="bold",
             color=TEXT, y=1.01)

colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sector_counts)))

# left: company count
bars = axes[0].barh(sector_counts.index, sector_counts.values,
                    color=colors, edgecolor="none")
axes[0].set_xlabel("Number of Companies")
axes[0].set_title("Companies per Sector", color=TEXT, fontsize=12)
axes[0].grid(axis="x", alpha=0.4)
for bar, val in zip(bars, sector_counts.values):
    axes[0].text(val + 0.5, bar.get_y() + bar.get_height()/2,
                 str(val), va="center", fontsize=8.5, color=TEXT)

# right: avg market cap
colors2 = plt.cm.plasma(np.linspace(0.2, 0.85, len(sector_counts)))
bars2 = axes[1].barh(sector_mc.index, sector_mc.values,
                     color=colors2, edgecolor="none")
axes[1].set_xlabel("Avg Market Cap ($ Billion)")
axes[1].set_title("Average Market Cap per Sector", color=TEXT, fontsize=12)
axes[1].grid(axis="x", alpha=0.4)
for bar, val in zip(bars2, sector_mc.values):
    if not np.isnan(val):
        axes[1].text(val + 1, bar.get_y() + bar.get_height()/2,
                     f"${val:.0f}B", va="center", fontsize=8, color=TEXT)

for ax in axes:
    ax.set_facecolor(CARD)
plt.tight_layout()
add_watermark(fig)
save(fig, "chart1_sector_distribution.png")
plt.close()

# ═════════════════════════════════════════════════════════
#  CHART 2 — P/E Ratio Distribution with annotation
# ═════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5.5), facecolor=BG)

n, bins, patches = ax.hist(df_pe["PE_Ratio"], bins=50,
                            color=ACCENT, edgecolor=BG, linewidth=0.4, alpha=0.85)

# KDE overlay
from scipy.stats import gaussian_kde
kde_x = np.linspace(df_pe["PE_Ratio"].min(), df_pe["PE_Ratio"].max(), 300)
kde   = gaussian_kde(df_pe["PE_Ratio"].dropna())
ax2   = ax.twinx()
ax2.plot(kde_x, kde(kde_x), color=HIGHLIGHT, lw=2, label="KDE")
ax2.set_ylabel("Density", color=HIGHLIGHT)
ax2.tick_params(colors=HIGHLIGHT)
ax2.set_ylim(0)
ax2.set_facecolor(CARD)
for spine in ax2.spines.values():
    spine.set_edgecolor(GRID)

median_pe = df_pe["PE_Ratio"].median()
mean_pe   = df_pe["PE_Ratio"].mean()
ax.axvline(median_pe, color="#f7c948", ls="--", lw=1.5, label=f"Median = {median_pe:.1f}x")
ax.axvline(mean_pe,   color="#f76c6c", ls="--", lw=1.5, label=f"Mean   = {mean_pe:.1f}x")
ax.axvspan(15, 25, alpha=0.10, color="limegreen", label="'Fair value' zone 15–25x")

ax.set_xlabel("Price-to-Earnings Ratio (P/E)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of P/E Ratios Across S&P 500 Companies\n"
             "(negative P/E and P/E > 200 excluded)", fontsize=13, color=TEXT)
ax.legend(loc="upper right", framealpha=0.2, labelcolor=TEXT)
ax.grid(axis="y", alpha=0.3)
ax.set_facecolor(CARD)
add_watermark(fig)
save(fig, "chart2_pe_distribution.png")
plt.close()

# ═════════════════════════════════════════════════════════
#  CHART 3 — Market Cap: Log-scale distribution + top 10
# ═════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)

# left: log-scale histogram
log_mc = np.log10(df_mc["MarketCap_B"])
axes[0].hist(log_mc, bins=40, color="#4fc3f7", edgecolor=BG, linewidth=0.4)
axes[0].set_xlabel("Log₁₀ Market Cap ($ Billion)")
axes[0].set_ylabel("Number of Companies")
axes[0].set_title("Market Cap Distribution (Log Scale)", fontsize=12)
xticks = [0, 1, 2, 3, 4]
axes[0].set_xticks(xticks)
axes[0].set_xticklabels([f"$10^{t}B" for t in xticks])
axes[0].axvline(log_mc.median(), color=HIGHLIGHT, ls="--", lw=1.5,
                label=f"Median ≈ ${10**log_mc.median():.0f}B")
axes[0].legend(framealpha=0.2, labelcolor=TEXT)
axes[0].grid(axis="y", alpha=0.3)

# right: top 10 by market cap
top10 = df_mc.nlargest(10, "MarketCap_B")[["Symbol","MarketCap_B"]].set_index("Symbol")
colors_top = plt.cm.cool(np.linspace(0.2, 0.85, 10))
bars = axes[1].barh(top10.index[::-1], top10["MarketCap_B"][::-1],
                    color=colors_top, edgecolor="none")
axes[1].set_xlabel("Market Cap ($ Billion)")
axes[1].set_title("Top 10 Companies by Market Cap", fontsize=12)
for bar, val in zip(bars, top10["MarketCap_B"][::-1]):
    axes[1].text(val + 20, bar.get_y() + bar.get_height()/2,
                 f"${val/1000:.1f}T" if val > 1000 else f"${val:.0f}B",
                 va="center", fontsize=9, color=TEXT)
axes[1].grid(axis="x", alpha=0.3)

for ax in axes:
    ax.set_facecolor(CARD)
fig.suptitle("S&P 500 — Market Capitalisation", fontsize=15, fontweight="bold",
             color=TEXT)
plt.tight_layout()
add_watermark(fig)
save(fig, "chart3_market_cap.png")
plt.close()

# ═════════════════════════════════════════════════════════
#  CHART 4 — Correlation Heat-map
# ═════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 7), facecolor=BG)

heat_cols = ["Price","PE_Ratio","Div_Yield","EPS","MarketCap_B","PS_Ratio","PB_Ratio"]
heat_labels = ["Price","P/E Ratio","Div. Yield","EPS","Mkt Cap ($B)","P/S Ratio","P/B Ratio"]
corr = df[heat_cols].corr()

mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", mask=mask,
            cmap="coolwarm", center=0, vmin=-1, vmax=1,
            xticklabels=heat_labels, yticklabels=heat_labels,
            linewidths=0.5, linecolor=BG,
            annot_kws={"size": 10, "color": "white"},
            cbar_kws={"shrink": 0.8},
            ax=ax)

ax.set_title("Correlation Matrix — Key Financial Metrics",
             fontsize=13, color=TEXT, pad=15)
ax.tick_params(colors=TEXT)
ax.set_facecolor(CARD)
fig.patch.set_facecolor(BG)
add_watermark(fig)
save(fig, "chart4_correlation_heatmap.png")
plt.close()

# ═════════════════════════════════════════════════════════
#  CHART 5 — Dividend Yield by Sector (box plots)
# ═════════════════════════════════════════════════════════
df_div = df.dropna(subset=["Div_Yield"]).copy()
df_div = df_div[df_div["Div_Yield"] > 0]
sector_order = (df_div.groupby("GICS_Sector")["Div_Yield"]
                .median().sort_values(ascending=False).index)

fig, ax = plt.subplots(figsize=(14, 6.5), facecolor=BG)
palette = sns.color_palette("viridis", n_colors=len(sector_order))

sns.boxplot(data=df_div, x="GICS_Sector", y="Div_Yield",
            order=sector_order, palette=palette,
            flierprops=dict(marker="o", markersize=3, alpha=0.5),
            ax=ax)
ax.set_xlabel("")
ax.set_ylabel("Dividend Yield (%)")
ax.set_title("Dividend Yield Distribution by Sector\n(zero-yield companies excluded)",
             fontsize=13, color=TEXT)
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
ax.grid(axis="y", alpha=0.35)
ax.set_facecolor(CARD)
sp500_avg_div = df_div["Div_Yield"].median()
ax.axhline(sp500_avg_div, color=HIGHLIGHT, ls="--", lw=1.5,
           label=f"Median Yield = {sp500_avg_div:.2f}%")
ax.legend(framealpha=0.2, labelcolor=TEXT)
plt.tight_layout()
add_watermark(fig)
save(fig, "chart5_dividend_yield_by_sector.png")
plt.close()

# ═════════════════════════════════════════════════════════
#  CHART 6 — EPS vs Price (scatter, colored by sector)
# ═════════════════════════════════════════════════════════
df_sc = df.dropna(subset=["EPS","Price"]).copy()
df_sc = df_sc[(df_sc["EPS"] > 0) & (df_sc["Price"] > 0)]

sectors_uniq = df_sc["GICS_Sector"].unique()
cmap_s = plt.cm.tab20(np.linspace(0, 1, len(sectors_uniq)))
sector_color = dict(zip(sectors_uniq, cmap_s))

fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG)
for sec in sectors_uniq:
    sub = df_sc[df_sc["GICS_Sector"] == sec]
    ax.scatter(sub["EPS"], sub["Price"],
               c=[sector_color[sec]], label=sec,
               s=40, alpha=0.75, edgecolors="none")

# regression line
slope, intercept, r, p, _ = stats.linregress(df_sc["EPS"], df_sc["Price"])
x_range = np.linspace(df_sc["EPS"].min(), df_sc["EPS"].max(), 100)
ax.plot(x_range, slope * x_range + intercept, color=HIGHLIGHT,
        lw=2, ls="--", label=f"Linear fit  r={r:.2f}")

# annotate a few famous tickers
for _, row in df.iterrows():
    if row["Symbol"] in ["AAPL","MSFT","GOOGL","TSLA","AMZN","META","BRK.B","NVDA"]:
        if pd.notna(row["EPS"]) and pd.notna(row["Price"]) and row["EPS"] > 0:
            ax.annotate(row["Symbol"],
                        xy=(row["EPS"], row["Price"]),
                        xytext=(5, 4), textcoords="offset points",
                        fontsize=7.5, color=TEXT, alpha=0.9)

ax.set_xlabel("Earnings Per Share (EPS, $)")
ax.set_ylabel("Stock Price ($)")
ax.set_title("EPS vs Stock Price — Do Higher Earnings Drive Higher Prices?",
             fontsize=13, color=TEXT)
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left",
          fontsize=8, framealpha=0.15, labelcolor=TEXT, ncol=1)
ax.grid(alpha=0.25)
ax.set_facecolor(CARD)
plt.tight_layout()
add_watermark(fig)
save(fig, "chart6_eps_vs_price.png")
plt.close()

# ═════════════════════════════════════════════════════════
#  CHART 7 — 52-Week Range: Where are stocks trading?
# ═════════════════════════════════════════════════════════
df_52 = df.dropna(subset=["Price_Pct_52w", "GICS_Sector"]).copy()
df_52 = df_52[(df_52["Price_Pct_52w"] >= 0) & (df_52["Price_Pct_52w"] <= 100)]

sector_med_pos = (df_52.groupby("GICS_Sector")["Price_Pct_52w"]
                  .median().sort_values(ascending=False))

fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), facecolor=BG)

# left: histogram of position in 52w range
n_hist = axes[0].hist(df_52["Price_Pct_52w"], bins=30,
                       color=ACCENT, edgecolor=BG, linewidth=0.4, alpha=0.9)
axes[0].axvline(50, color=GRID, ls=":", lw=1)
axes[0].axvspan(70, 100, alpha=0.12, color="limegreen", label="Near 52-week HIGH")
axes[0].axvspan(0, 30,   alpha=0.12, color="#f76c6c",    label="Near 52-week LOW")
axes[0].set_xlabel("Position within 52-Week Range (%)")
axes[0].set_ylabel("Number of Companies")
axes[0].set_title("How Close Are Stocks to Their 52-Week High/Low?", fontsize=11)
axes[0].legend(framealpha=0.2, labelcolor=TEXT, fontsize=9)
axes[0].grid(axis="y", alpha=0.3)

# right: median position per sector (horizontal bars)
colors_52 = plt.cm.RdYlGn(sector_med_pos.values / 100)
bars = axes[1].barh(sector_med_pos.index, sector_med_pos.values,
                    color=colors_52, edgecolor="none")
axes[1].axvline(50, color=GRID, ls="--", lw=1, alpha=0.7)
axes[1].set_xlabel("Median Position in 52-Week Range (%)")
axes[1].set_title("Median 52-Week Position by Sector\n(higher = closer to year high)",
                  fontsize=11)
for bar, val in zip(bars, sector_med_pos.values):
    axes[1].text(val + 0.5, bar.get_y() + bar.get_height()/2,
                 f"{val:.0f}%", va="center", fontsize=8.5, color=TEXT)
axes[1].grid(axis="x", alpha=0.3)

for ax in axes:
    ax.set_facecolor(CARD)
fig.suptitle("S&P 500 — 52-Week Price Positioning", fontsize=15,
             fontweight="bold", color=TEXT)
plt.tight_layout()
add_watermark(fig)
save(fig, "chart7_52week_positioning.png")
plt.close()

# ═════════════════════════════════════════════════════════
#  KEY INSIGHTS SUMMARY
# ═════════════════════════════════════════════════════════
print("\n" + "═"*62)
print("  KEY INSIGHTS")
print("═"*62)

top_div_sector = df_div.groupby("GICS_Sector")["Div_Yield"].median().idxmax()
top_div_val    = df_div.groupby("GICS_Sector")["Div_Yield"].median().max()
top_mc_ticker  = df_mc.nlargest(1,"MarketCap_B").iloc[0]
most_companies = sector_counts.idxmax()
highest_pe_med = df_pe.groupby("GICS_Sector")["PE_Ratio"].median().idxmax()
lowest_pe_med  = df_pe.groupby("GICS_Sector")["PE_Ratio"].median().idxmin()
near_high_pct  = (df_52["Price_Pct_52w"] > 70).mean() * 100
near_low_pct   = (df_52["Price_Pct_52w"] < 30).mean() * 100

print(f"""
1.  Largest company : {top_mc_ticker['Symbol']} ({top_mc_ticker['Name']})
    Market Cap       : ${top_mc_ticker['MarketCap_B']:.0f}B

2.  Most represented sector : {most_companies}
    ({int(sector_counts[most_companies])} companies in S&P 500)

3.  Highest P/E sector  : {highest_pe_med}
    (growth premium; investors pay more per $ of earnings)
    Lowest P/E sector   : {lowest_pe_med}
    (value / cyclical; cheaper relative to earnings)

4.  Median S&P 500 P/E  : {median_pe:.1f}x
    Median S&P 500 EPS  : ${df['EPS'].median():.2f}
    EPS–Price correlation: r = {r:.2f} (moderate positive)

5.  Best dividend sector : {top_div_sector}
    Median yield         : {top_div_val:.2f}%
    S&P 500 median yield : {sp500_avg_div:.2f}%

6.  52-week positioning  :
    {near_high_pct:.1f}% of stocks are within 30% of their 52-week HIGH
    {near_low_pct:.1f}% of stocks are within 30% of their 52-week LOW

7.  Market cap distribution is highly RIGHT-SKEWED (skew = {skew_mc:.1f})
    — a handful of mega-caps dominate; most companies are mid-cap.
""")

print("═"*62)
print("  All 7 charts saved to current directory.")
print("═"*62 + "\n")