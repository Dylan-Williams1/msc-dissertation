import os
import glob
import json
import re
import numpy as np
import pandas as pd
import scipy.stats as stats


# VARIABLES TO SET BEFORE RUNNING
TARGET_MODEL = "gemini-3.6-flash-v3"  # Change to "gemini-3.6-flash" when needed

# Execution convention used for the RANK IC calculation.
#   "open_t1"    signal at close t -> execute open t+1 -> exit open t+2   (locked default)
#   "close_t1"   signal at close t -> execute close t+1 -> exit close t+2 (conservative)
#   "same_close" signal at close t -> execute close t   -> exit close t+1
# NOTE: the Sharpe path further down still uses the original shift(1) against
# close-to-close returns, which is "same_close". IC and Sharpe are therefore not
# yet on the same basis - see the note printed at the end of the run.
IC_EXECUTION_CONVENTION = "open_t1"

# Dates with fewer than this many valid names are dropped from the IC series.
IC_MIN_NAMES = 20

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
# 0b. RANK IC HELPERS
# =========================================================
# Rank IC answers one question per trading day: did the alpha put the stocks in
# the right order? It is the Spearman correlation between the signal on date t
# and the return that signal actually goes on to earn.
#
# Two properties make it the primary stability statistic:
#   - It depends only on the ORDERING of the signal, so it is invariant to
#     scaling, de-meaning, z-scoring or any other position-sizing choice. The
#     result cannot be attacked on the grounds of an arbitrary sizing rule.
#   - It yields a clean daily series, which is the input a structural-break
#     test needs.
#
# It IS sensitive to the execution convention, since that determines which
# return gets paired with the signal. Hence IC_EXECUTION_CONVENTION above.

def build_forward_returns(df_price, convention="open_t1"):
    """
    Return earned by a position formed on the signal of date t, indexed by t.

    The return window always begins strictly AFTER date t, so pairing a signal
    with it cannot reward information the signal already contained.
    """
    if convention in ("open_t1", "close_t1"):
        col = "open" if convention == "open_t1" else "close"
        if col not in df_price.columns:
            raise KeyError(
                f"convention {convention!r} needs a {col!r} column in the panel"
            )
        px = df_price[col].unstack("symbol").sort_index()
        return px.shift(-2) / px.shift(-1) - 1.0

    if convention == "same_close":
        px = df_price["close"].unstack("symbol").sort_index()
        return px.shift(-1) / px - 1.0

    raise ValueError(f"unknown execution convention: {convention!r}")


def compute_rank_ic(signal, fwd_returns, min_names=IC_MIN_NAMES):
    """
    Per-date Spearman rank correlation between the signal and its forward return.

    Computed on the RAW SIGNAL, deliberately not on portfolio weights: that is
    what makes it independent of position sizing.

    Returns a daily Series indexed by signal date. Dates with a thin
    cross-section are dropped rather than contributing a noisy estimate.
    """
    s = signal.unstack("symbol") if isinstance(signal.index, pd.MultiIndex) else signal
    s = s.sort_index().sort_index(axis=1)
    f = fwd_returns.sort_index().sort_index(axis=1)
    s, f = s.align(f, join="inner")

    live = s.notna() & f.notna()
    s, f = s.mask(~live), f.mask(~live)

    sr, fr = s.rank(axis=1), f.rank(axis=1)
    sr = sr.sub(sr.mean(axis=1), axis=0)
    fr = fr.sub(fr.mean(axis=1), axis=0)

    num = (sr * fr).sum(axis=1, min_count=1)
    den = np.sqrt((sr ** 2).sum(axis=1, min_count=1)
                  * (fr ** 2).sum(axis=1, min_count=1))
    ic = num / den.replace(0.0, np.nan)
    return ic.mask(live.sum(axis=1) < min_names).dropna()


def summarise_ic(ic):
    """
    Collapse the daily IC series to reportable statistics.

    mean_rank_ic          average edge. 0.02-0.05 is a strong daily equity alpha.
    ic_std                day-to-day dispersion of the edge.
    ic_information_ratio  mean / std. Consistency of the edge, not its size.
                          This is the headline number.
    ic_n_days             length of the IC series. Kept because warmup differs by
                          alpha, so an IR is not comparable across rows without
                          knowing how many days produced it.

    Deliberately NOT reported here: a t-statistic (it is IR * sqrt(n_days), so it
    adds no information and invites over-reading given autocorrelated IC) and a
    hit rate (a diagnostic, not a screening criterion). Both are one line away
    from the daily IC panel if ever wanted.
    """
    T = len(ic)
    if T < 3:
        return dict(ic_n_days=T, mean_rank_ic=np.nan, ic_std=np.nan,
                    ic_information_ratio=np.nan)

    m = float(ic.mean())
    sd = float(ic.std(ddof=1))
    return dict(
        ic_n_days=T,
        mean_rank_ic=m,
        ic_std=sd,
        ic_information_ratio=(m / sd) if sd > 0 else np.nan,
    )


EMPTY_IC = dict(ic_n_days=0, mean_rank_ic=np.nan, ic_std=np.nan,
                ic_information_ratio=np.nan)

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

# --- Forward returns for the IC calculation (built once, reused per alpha) ---
FWD_RETURNS = build_forward_returns(df_price, IC_EXECUTION_CONVENTION)
print(f"IC forward returns built on convention '{IC_EXECUTION_CONVENTION}' "
      f"({FWD_RETURNS.shape[0]} dates x {FWD_RETURNS.shape[1]} symbols)")

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
ic_series_by_alpha = {}   # keep the daily series - this is Instrument 1's input

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

        # --- Rank IC, computed on the RAW signal (no shift, no weighting) ---
        # The shift belongs to the P&L path only. build_forward_returns already
        # places the return window after date t, so shifting here as well would
        # lag the signal twice.
        ic_series = compute_rank_ic(signal, FWD_RETURNS)
        ic_stats = summarise_ic(ic_series)
        ic_series_by_alpha[alpha_id] = ic_series

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
            "ic_execution": IC_EXECUTION_CONVENTION,
            "mean_rank_ic": round(ic_stats["mean_rank_ic"], 5),
            "ic_std": round(ic_stats["ic_std"], 5),
            "ic_information_ratio": round(ic_stats["ic_information_ratio"], 4),
            "ic_n_days": ic_stats["ic_n_days"],
            "status": status,
            "error": None,
        })
        print(f"[{alpha_id}] Success: Sharpe={sharpe:.2f}, Z={z_score:.2f}, "
              f"DSR={dsr_bounded:.4f} | IC={ic_stats['mean_rank_ic']:+.4f}, "
              f"IR={ic_stats['ic_information_ratio']:+.3f} "
              f"({ic_stats['ic_n_days']}d) [{status}]")

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
                **EMPTY_IC,
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

# Daily IC series, one column per alpha. This is the raw material for the
# Instrument 1 break tests - keep it, it is expensive to regenerate.
if ic_series_by_alpha:
    ic_panel = pd.DataFrame(ic_series_by_alpha).sort_index()
    ic_panel.index.name = "date"
    ic_output_path = "phase1_daily_rank_ic_" + TARGET_MODEL + ".csv"
    ic_panel.to_csv(ic_output_path)
else:
    ic_output_path = None

print("\n--- BATCH EVALUATION COMPLETE ---")
print(f"Results successfully saved to {csv_output_path}")
if ic_output_path:
    print(f"Daily rank IC series saved to {ic_output_path}")
print(df_results[["alpha_id", "sharpe_ratio", "dsr",
                  "mean_rank_ic", "ic_information_ratio", "status"]])

print(f"\nNOTE: rank IC uses '{IC_EXECUTION_CONVENTION}'; the Sharpe path above still "
      "uses shift(1) against close-to-close returns, i.e. 'same_close'. The two "
      "columns are not yet measured on the same execution convention.")
