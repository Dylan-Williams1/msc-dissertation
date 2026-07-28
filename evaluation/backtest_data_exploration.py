import pandas as pd
import json

# 1. Inspect the Metadata
with open("meta.json", 'r') as f:
    metadata = json.load(f)
print("--- METADATA ---")
print(json.dumps(metadata, indent=4))

# 2. Load the Core Price Data
df_price = pd.read_parquet("daily_ohlcv.parquet", engine='pyarrow')

# --- DATA CLEANING ADDITIONS ---
# Convert the string 'date' column into proper datetime objects
df_price['date'] = pd.to_datetime(df_price['date'])

# Set 'date' and 'symbol' as a MultiIndex for panel data analysis
df_price.set_index(['date', 'symbol'], inplace=True)

# Sort the index to optimize performance for future lookups/filtering
df_price.sort_index(inplace=True)

# FIX FOR ISSUE 2: Drop any rows where price data is missing
# This handles the unbalanced panel by ensuring we only keep valid trading days
df_price.dropna(subset=['close'], inplace=True)
# -------------------------------

print("\n--- OHLCV DATASET ---")
print(df_price.info())

# Since 'date' is now part of the index, we extract it to find the min and max
date_level = df_price.index.get_level_values('date')
print(f"Total Date Range: {date_level.min().date()} to {date_level.max().date()}")

print("\n--- DATA PREVIEW ---")
print(df_price.head(10))

# 3. Load the Constituents (The Survivorship Check)
df_constituents = pd.read_csv("constituents.csv")

# FIX FOR ISSUE 1: Prevent Look-Ahead Bias
# Drop the 2026 market cap and weight columns so they cannot be used in historical calculations
df_constituents.drop(columns=['market_cap', 'weight'], inplace=True, errors='ignore')

print("\n--- CONSTITUENTS DATASET ---")
print(df_constituents.head())
