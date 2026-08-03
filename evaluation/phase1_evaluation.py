import os
import glob
import json
import re
import numpy as np
import pandas as pd
import scipy.stats as stats


# VARIABLES TO SET BEFORE RUNNING
TARGET_MODEL = "gemini-3.6-flash-v3"  # Change to "gemini-3.6-flash" when needed

# =========================================================
# 0. ANCHOR ALL PATHS TO THEIR RESPECTIVE FOLDERS
# =========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Navigate up one level from 'evaluation', then into 'data'
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")

# Point the data files to the DATA_DIR
META_PATH = os.path.join(DATA_DIR, "meta.json")
PARQUET_PATH = os.path.join(DATA_DIR, "daily_ohlcv.parquet")
CONSTITUENTS_PATH = os.path.join(DATA_DIR, "constituents.csv")

# =========================================================
# 1. LOAD & CLEAN DATA (previously a separate script)
# =========================================================

# --- Core price data ---
df_price = pd.read_parquet(PARQUET_PATH, engine="pyarrow")

# Convert 'date' to datetime
df_price["date"] = pd.to_datetime(df_price["date"])

# Set MultiIndex (date, symbol) for panel data
df_price.set_index(["date", "symbol"], inplace=True)

# Sort index for performance / correctness of groupby-level ops
df_price.sort_index(inplace=True)

# Drop rows with missing close prices (unbalanced panel handling)
df_price.dropna(subset=["close"], inplace=True)

# Compute close-to-close returns needed by every alpha's backtest
df_price["returns"] = df_price.groupby(level="symbol")["close"].pct_change()

print("\n--- OHLCV DATASET ---")
#print(df_price.info())
date_level = df_price.index.get_level_values("date")
print(f"Total Date Range: {date_level.min().date()} to {date_level.max().date()}")

# --- Constituents (survivorship / look-ahead check) ---
if os.path.exists(CONSTITUENTS_PATH):
    df_constituents = pd.read_csv(CONSTITUENTS_PATH)
    # Prevent look-ahead bias: drop any current market_cap/weight columns
    df_constituents.drop(columns=["market_cap", "weight"], inplace=True, errors="ignore")
    #print("\n--- CONSTITUENTS DATASET ---")
    #print(df_constituents.head())
else:
    df_constituents = None
    print("\n(no constituents.csv found — skipping universe filter)")

# =========================================================
# 2. BATCH EVALUATE ALL GENERATED ALPHAS
# =========================================================

# Navigate up one level from 'evaluation', then into 'alphas/raw/<TARGET_MODEL>'
output_dir = os.path.join(os.path.dirname(SCRIPT_DIR), "alphas", "raw", TARGET_MODEL)
results = []

# Look for all .json files directly inside that specific folder
search_pattern = os.path.join(output_dir, "*.json")
json_files = glob.glob(search_pattern)

print(f"\nFound {len(json_files)} alpha JSON files to evaluate in {TARGET_MODEL}.")

if len(json_files) == 0:
    raise FileNotFoundError(
        f"No .json files found in {output_dir}. "
        "Check that your generated alpha files are actually in this directory."
    )

# --- Pre-scan pass: count trials PER MODEL, not pooled across the folder ---
# Each model's DSR must be deflated against the number of alphas generated
# for THAT model (~20), since that's the actual search space it was drawn
# from. Pooling across models would understate/overstate the correction
# depending on how many models happen to share this folder.
trials_per_model = {}
for file_path in json_files:
    with open(file_path, "r", encoding="utf-8") as f:
        _meta = json.load(f)["metadata"]
    model_key = _meta.get("model", "UNKNOWN_MODEL")
    trials_per_model[model_key] = trials_per_model.get(model_key, 0) + 1

print("Trials per model (used as N in the DSR calculation):")
for model_key, count in trials_per_model.items():
    print(f"   {model_key}: {count}")

for file_path in json_files:
    with open(file_path, "r", encoding="utf-8") as f:
        alpha_data = json.load(f)

    meta = alpha_data["metadata"]
    alpha_id = meta["alpha_id"]
    num_trials = trials_per_model[meta.get("model", "UNKNOWN_MODEL")]
    print(f"Evaluating [{alpha_id}] (model={meta.get('model')}, N={num_trials})...")

    try:
        # Extract Python code via regex
        raw_response = alpha_data["raw_response"]
        code_pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
        match = code_pattern.search(raw_response)

        if not match:
            code_pattern_alt = re.compile(r"```\n(.*?)\n```", re.DOTALL)
            match = code_pattern_alt.search(raw_response)

        if not match:
            raise ValueError("No Python code block found in LLM response.")

        python_code = match.group(1)

        # Execute code to load generate_alpha function
        local_namespace = {}
        exec(python_code, globals(), local_namespace)

        if "generate_alpha" not in local_namespace:
            raise NameError("generate_alpha function missing from executed code.")

        generate_alpha = local_namespace["generate_alpha"]

        # Generate signal & apply T+1 execution shift
        signal = generate_alpha(df_price)
        shifted_weights = signal.groupby(level="symbol").shift(1)

        # Portfolio returns
        portfolio_daily_returns = (shifted_weights * df_price["returns"]).groupby(level="date").sum()
        portfolio_daily_returns = portfolio_daily_returns.dropna()

        if len(portfolio_daily_returns) == 0:
            raise ValueError("Resulting portfolio returns series is completely empty.")

        # Performance metrics
        ann_ret = portfolio_daily_returns.mean() * 252
        ann_vol = portfolio_daily_returns.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol != 0 else 0

        # Deflated Sharpe Ratio
        T = len(portfolio_daily_returns)
        daily_sr = (
            portfolio_daily_returns.mean() / portfolio_daily_returns.std()
            if portfolio_daily_returns.std() != 0
            else 0
        )
        skew = stats.skew(portfolio_daily_returns)
        kurt = stats.kurtosis(portfolio_daily_returns, fisher=False)

        euler_mascheroni = 0.5772156649
        if num_trials > 1:
            # 1. Calculate the Expected Max Z-Score (standard normal domain)
            max_z = (1 - euler_mascheroni) * stats.norm.ppf(1 - 1 / num_trials) + \
                    euler_mascheroni * stats.norm.ppf(1 - 1 / (num_trials * np.e))
            
            # 2. Scale it to the daily Sharpe domain using the standard error (1/sqrt(T))
            sr_0 = max_z / np.sqrt(T)
        else:
            sr_0 = 0

        numerator = (daily_sr - sr_0) * np.sqrt(T - 1)
        denominator = np.sqrt(1 - skew * daily_sr + ((kurt - 1) / 4) * (daily_sr ** 2))
        z_score = numerator / denominator if denominator != 0 else 0
        
        # Calculate raw DSR
        dsr_raw = stats.norm.cdf(z_score)
        
        # Bound the float to prevent suspicious-looking 1.0s and 0.0s
        dsr_bounded = float(np.clip(dsr_raw, 0.0001, 0.9999))
        
        status = "SURVIVED_PHASE_1" if dsr_bounded > 0.95 else "REJECTED_DSR"

        results.append({
            "alpha_id": alpha_id,
            "model": meta.get("model"),
            "theme": meta.get("theme"),
            "annualized_return": round(ann_ret, 4),
            "annualized_vol": round(ann_vol, 4),
            "sharpe_ratio": round(sharpe, 4),
            "dsr_z_score": round(z_score, 2), # Exposing the Z-score for transparency
            "dsr": round(dsr_bounded, 4),     # Using the bounded DSR
            "status": status,
            "error": None,
        })
        print(f"[{alpha_id}] Success: Sharpe={sharpe:.2f}, Z={z_score:.2f}, DSR={dsr_bounded:.4f} [{status}]")

    except Exception as e:
        print(f"   -> ERROR evaluating {alpha_id}: {e}")
        results.append(
            {
                "alpha_id": alpha_id,
                "model": meta.get("model"),
                "theme": meta.get("theme"),
                "annualized_return": np.nan,
                "annualized_vol": np.nan,
                "sharpe_ratio": np.nan,
                "dsr": np.nan,
                "status": "FAILED_EXECUTION",
                "error": str(e),
            }
        )

# =========================================================
# 3. EXPORT RESULTS
# =========================================================

df_results = pd.DataFrame(results)
csv_output_path = "phase1_screening_results_" + TARGET_MODEL + ".csv"
df_results.to_csv(csv_output_path, index=False)

print("\n--- BATCH EVALUATION COMPLETE ---")
print(f"Results successfully saved to {csv_output_path}")
print(df_results[["alpha_id", "sharpe_ratio", "dsr", "status"]])
