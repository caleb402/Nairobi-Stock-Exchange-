import pandas as pd
import numpy as np
from faker import Faker
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt

# Parameters
n_companies = 60
days = 30
np.random.seed(42)

# Initialize Faker with seed for reproducibility
fake = Faker() 
Faker.seed(42)

# Sector-specific drift and volatility
sector_params = {
    "Banking":       {"mu": 0.0004, "sigma": 0.015},
    "Manufacturing": {"mu": 0.0005, "sigma": 0.02},
    "Telecom":       {"mu": 0.0006, "sigma": 0.018},
    "Energy":        {"mu": 0.0007, "sigma": 0.03},
    "Retail":        {"mu": 0.0005, "sigma": 0.022},
    "Agriculture":   {"mu": 0.0003, "sigma": 0.025}
}

# Generate company info
tickers = [f"COMP{i:02d}" for i in range(1, n_companies+1)]
names = [fake.company() for _ in range(n_companies)] # realistic names
sectors = np.random.choice(list(sector_params.keys()), n_companies)

market_caps = np.random.uniform(1, 200, n_companies).round(2)
pe_ratios = np.random.uniform(5, 25, n_companies).round(2)
pb_ratios = np.random.uniform(0.5, 5, n_companies).round(2)
div_yields = np.random.uniform(0, 10, n_companies).round(2)
roes = np.random.uniform(5, 30, n_companies).round(2)

# Build main metrics table
companies = pd.DataFrame({
    "Ticker": tickers,
    "Company Name": names,
    "Sector": sectors,
    "Market Cap (KES B)": market_caps,
    "P/E Ratio": pe_ratios,
    "P/B Ratio": pb_ratios,
    "Dividend Yield (%)": div_yields,
    "ROE (%)": roes
})

# Loop through each sector
for sector in companies["Sector"].unique():
    sector_data = companies[companies["Sector"] == sector]
    avg_pe = sector_data["P/E Ratio"].mean()
    avg_pb = sector_data["P/B Ratio"].mean()
    avg_roe = sector_data["ROE (%)"].mean()

    # --- 1. Simple threshold criteria ---
    mask_simple = (
        (companies["Sector"] == sector) &
        (companies["P/E Ratio"] < avg_pe) &
        (companies["P/B Ratio"] < avg_pb) &
        (companies["ROE (%)"] > avg_roe)
    )

    # --- 2. ROE-to-PB relationship criteria ---
    mask_roe_pb = (
        (companies["Sector"] == sector) &
        (companies["P/E Ratio"] < avg_pe) &
        (companies["ROE (%)"] > avg_roe) &
        ((companies["ROE (%)"] / companies["P/B Ratio"]) > (avg_roe / avg_pb))
    )

    # --- 3. ROE-to-P/E relationship criteria ---
    mask_roe_pe = (
        (companies["Sector"] == sector) &
        (companies["P/B Ratio"] < avg_pb) &
        (companies["ROE (%)"] > avg_roe) &
        ((companies["ROE (%)"] / companies["P/E Ratio"]) > (avg_roe / avg_pe))
    )

    # --- Comprehensive scoring ---
    companies.loc[companies["Sector"] == sector, "Valuation Score"] = (
        mask_simple.astype(int) +
        mask_roe_pb.astype(int) +
        mask_roe_pe.astype(int)
    )

    # --- Comprehensive flag based on score ---
    companies.loc[(companies["Sector"] == sector) & (companies["Valuation Score"] == 3), "Valuation Flag Comprehensive"] = "Strongly Undervalued"
    companies.loc[(companies["Sector"] == sector) & (companies["Valuation Score"] == 2), "Valuation Flag Comprehensive"] = "Moderately Undervalued"
    companies.loc[(companies["Sector"] == sector) & (companies["Valuation Score"] == 1), "Valuation Flag Comprehensive"] = "Weakly Undervalued"
    companies.loc[(companies["Sector"] == sector) & (companies["Valuation Score"] == 0), "Valuation Flag Comprehensive"] = "Overvalued"

# Simulate 30-day price history using sector-aware GBM
all_histories = []
for ticker, sector in zip(tickers, sectors):
    S0 = np.random.uniform(50, 150)  # starting price
    mu, sigma = sector_params[sector]["mu"], sector_params[sector]["sigma"]
    prices = [S0]
    for _ in range(days-1):
        prices.append(prices[-1] * np.exp((mu - 0.5*sigma**2) + sigma*np.random.normal()))
    
    df = pd.DataFrame({"Ticker": ticker, "Sector": sector, "Day": range(1, days+1), "Price": prices})
    all_histories.append(df)

expanded_df = pd.concat(all_histories, ignore_index=True)



# --- Step 1: Compute daily returns per ticker ---
expanded_df["Return"] = expanded_df.groupby("Ticker")["Price"].pct_change()

# --- Step 2a: Final compounded cumulative return per company (one value per ticker) ---
final_cumulative = (
    expanded_df.groupby("Ticker")["Return"]
    .apply(lambda r: (1 + r.dropna()).prod() - 1)   # drop NaNs before compounding
    .reset_index()
)
final_cumulative.columns = ["Ticker", "Cumulative Return"]

# --- Step 2b: Merge final cumulative returns back into fundamentals ---
companies = companies.merge(final_cumulative, on="Ticker", how="left")

# --- Step 3: Per-company cumulative return trajectory over time ---
# Use transform so the result aligns with expanded_df’s index (avoids MultiIndex mismatch)
expanded_df["Cumulative Return"] = (
    expanded_df.groupby("Ticker")["Return"]
    .transform(lambda r: (1 + r.fillna(0)).cumprod() - 1)
)

# --- Step 4: Average cumulative return by Sector and Valuation Flag ---
avg_returns_by_sector_flag = (
    companies.groupby(["Sector", "Valuation Flag Comprehensive"])["Cumulative Return"].mean()
)
print("\nAverage Cumulative Return by Sector and Valuation Flag:\n", avg_returns_by_sector_flag)

# --- Step 5: Save outputs ---
avg_returns_by_sector_flag.reset_index().to_csv("avg_returns_by_sector_flag.csv", index=False)
companies.to_csv("nse_value_dataset.csv", index=False)
expanded_df.to_csv("all_price_histories.csv", index=False)

# --- Step 6: Normalize cumulative returns within each sector ---
companies["Normalized Return"] = companies.groupby("Sector")["Cumulative Return"].transform(
    lambda x: (x - x.mean()) / x.std()
)

# --- Step 7: Collect results per sector ---
results = []

for sector in companies["Sector"].unique():
    sector_data = companies[companies["Sector"] == sector]

    # Split undervalued vs overvalued groups
    undervalued = sector_data[sector_data["Valuation Flag Comprehensive"] != "Overvalued"]["Normalized Return"]
    overvalued = sector_data[sector_data["Valuation Flag Comprehensive"] == "Overvalued"]["Normalized Return"]

    # Run Welch’s t-test only if both groups have enough data
    if len(undervalued) > 1 and len(overvalued) > 1:
        t_stat, p_val = ttest_ind(undervalued, overvalued, equal_var=False)
        results.append({
            "Sector": sector,
            "Undervalued Mean (Normalized)": undervalued.mean(),
            "Overvalued Mean (Normalized)": overvalued.mean(),
            "T-statistic": t_stat,
            "P-value": p_val
        })
    else:
        results.append({
            "Sector": sector,
            "Undervalued Mean (Normalized)": undervalued.mean() if len(undervalued) > 0 else None,
            "Overvalued Mean (Normalized)": overvalued.mean() if len(overvalued) > 0 else None,
            "T-statistic": None,
            "P-value": None
        })

# --- Step 8: Convert results to DataFrame for readability ---
sector_results = pd.DataFrame(results)
print(sector_results)

sector_validation = companies.groupby(["Sector", "Valuation Flag Comprehensive"])["Normalized Return"].mean()
print(sector_validation)

print("Files saved: nse_value_dataset.csv, all_price_histories.csv, avg_returns_by_sector_flag.csv")

# --- Step 9: Boxplot using normalized returns ---
sns.boxplot(x="Valuation Flag Comprehensive", y="Normalized Return", data=companies)
plt.title("Normalized Returns by Valuation Flag")
plt.show()
