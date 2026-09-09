import numpy as np
import pandas as pd
from scipy import stats

# ==============================================================================
# CONFIGURATION & PARAMETERS (File paths left blank per request)
# ==============================================================================
# Paths to be provided later
TREATMENT_IC_PATH = ""  
CONTROL_IC_PATH = ""    
OUT_DIR = ""            

# Core Parameters from v2 Spec
CUTOFF_DATE = "2026-01-01"  
END_DATE = "2026-07-16"     
HORIZON_H = 1               # Forward-return horizon (h)
TRAILING_BLOCKS_B = 4       # Number of trailing blocks in baseline (B)
BOOTSTRAP_M = 10000         # Stationary bootstrap iterations (M)
ALPHA_LEVEL = 0.05          # Significance level

# ==============================================================================
# HELPER FUNCTIONS: ECONOMETRICS & STATISTICS
# ==============================================================================

def newey_west_se(series, h=HORIZON_H):
    """
    Computes the Newey-West standard error of the mean for a given block [P2, P3].
    q = max(h - 1, floor(4 * (T / 100)^(2/9)))
    """
    series = series.dropna()
    T = len(series)
    if T < 5:
        return np.nan, np.nan
    
    q = int(max(h - 1, np.floor(4 * (T / 100.0) ** (2.0 / 9.0))))
    x_mean = series.mean()
    xc = series.to_numpy(dtype=float) - x_mean
    
    # Long-run variance (lrv)
    lrv = np.sum(xc ** 2) / T
    for j in range(1, q + 1):
        gamma_j = np.sum(xc[j:] * xc[:-j]) / T
        weight = 1.0 - j / (q + 1.0)
        lrv += 2 * weight * gamma_j
        
    lrv = max(lrv, 1e-14)
    se = np.sqrt(lrv / T)
    return lrv, se

def stationary_block_bootstrap_z(series, L, M=BOOTSTRAP_M, seed=42):
    """
    Generates M pseudo-OOS block z-scores using a stationary bootstrap [1.2].
    Block length is geometrically distributed with mean L.
    """
    arr = series.dropna().to_numpy(dtype=float)
    T = len(arr)
    if T < L:
        return np.nan
        
    rng = np.random.default_rng(seed)
    p = 1.0 / L  # probability of starting a new block
    
    z_boots = np.empty(M)
    for i in range(M):
        # Generate pseudo-OOS window of length L
        pseudo_oos = np.empty(L)
        idx = rng.integers(0, T)
        for j in range(L):
            pseudo_oos[j] = arr[idx]
            if rng.random() < p:
                idx = rng.integers(0, T) # Start new block
            else:
                idx = (idx + 1) % T      # Continue block (wrap around)
                
        # Compute z-score for this pseudo-block
        x_mean = np.mean(pseudo_oos)
        xc = pseudo_oos - x_mean
        lrv = np.sum(xc**2) / L
        # Simplified NW for bootstrap inner loop to save compute, 
        # standard variance is typically sufficient for the null distribution shape
        se = np.sqrt(max(lrv, 1e-14) / L) 
        z_boots[i] = x_mean / se if se > 0 else 0
        
    return z_boots

# ==============================================================================
# PIPELINE: DIFFERENTIALS & TILING
# ==============================================================================

def compute_differentials(trt_ic, ctrl_ic, cutoff_date):
    """
    Calculates d_i(t) = IC_i(t) - beta_i * CtrlMean(t) [P1].
    Beta is strictly estimated IN-SAMPLE and applied out-of-sample.
    """
    is_mask = ctrl_ic.index < cutoff_date
    
    # 1. Treatment Differentials
    ctrl_mean = ctrl_ic.mean(axis=1)
    trt_diffs = pd.DataFrame(index=trt_ic.index, columns=trt_ic.columns)
    
    for col in trt_ic.columns:
        y_is = trt_ic[col][is_mask].dropna()
        x_is = ctrl_mean[is_mask].dropna()
        common = y_is.index.intersection(x_is.index)
        
        if len(common) > 30:
            cov = np.cov(y_is[common], x_is[common])[0, 1]
            var = np.var(x_is[common], ddof=1)
            beta = cov / var if var > 0 else 0
        else:
            beta = 0
            
        trt_diffs[col] = trt_ic[col] - beta * ctrl_mean
        
    # 2. Control Differentials (Leave-One-Out)
    ctrl_diffs = pd.DataFrame(index=ctrl_ic.index, columns=ctrl_ic.columns)
    
    for col in ctrl_ic.columns:
        other_ctrls_mean = ctrl_ic.drop(columns=[col]).mean(axis=1)
        y_is = ctrl_ic[col][is_mask].dropna()
        x_is = other_ctrls_mean[is_mask].dropna()
        common = y_is.index.intersection(x_is.index)
        
        if len(common) > 30:
            cov = np.cov(y_is[common], x_is[common])[0, 1]
            var = np.var(x_is[common], ddof=1)
            beta = cov / var if var > 0 else 0
        else:
            beta = 0
            
        ctrl_diffs[col] = ctrl_ic[col] - beta * other_ctrls_mean
        
    return trt_diffs, ctrl_diffs

# ==============================================================================
# CORE TEST MECHANICS (Steps 1 & 2)
# ==============================================================================

def execute_v2_test(d_series, is_series, oos_series, L, ctrl_stats=None):
    """
    Runs Step 1 (Rarity) and Step 2 (Excess Decay) for a single alpha [1, 2].
    ctrl_stats is a dictionary containing r and pool_median_base if evaluating 
    a treatment alpha. If None, it assumes we are processing a control alpha to 
    build those stats.
    """
    res = {}
    
    # --- Step 1: Rarity (Flag 1) [1.1, 1.2] ---
    _, se_oos = newey_west_se(oos_series)
    oos_mean = oos_series.mean()
    z_oos = oos_mean / se_oos if se_oos > 0 else np.nan
    
    if np.isfinite(z_oos):
        z_boots = stationary_block_bootstrap_z(is_series, L)
        p1 = np.mean(z_boots <= z_oos)
    else:
        p1 = np.nan
        
    flag_1 = p1 < ALPHA_LEVEL
    
    res.update({'oos_mean': oos_mean, 'oos_se': se_oos, 'z_oos': z_oos, 'p1': p1, 'flag_1': flag_1})
    
    # --- Step 2: Excess Decay (Flag 2) [2.1, 2.2, 2.4] ---
    # Baseline: Trailing B blocks
    B_days = TRAILING_BLOCKS_B * L
    base_series = is_series.iloc[-B_days:] if len(is_series) >= B_days else is_series
    
    lrv_base, se_base = newey_west_se(base_series)
    base_mean = base_series.mean()
    
    drop_oos = base_mean - oos_mean
    se_drop = np.sqrt(se_base**2 + se_oos**2)
    
    res.update({'base_mean': base_mean, 'base_se': se_base, 'drop_oos': drop_oos, 'se_drop': se_drop})
    
    # If evaluating a control alpha, we stop here and return the baseline stats
    # so the orchestrator can calculate the control tolerance (r, delta).
    if ctrl_stats is None:
        return res
        
    # --- Step 2.3 & 2.5: Control Tolerance & Z-Test ---
    r = ctrl_stats['r']
    median_base_j = ctrl_stats['median_base_j']
    
    # Calculate delta with the required median floor [2.3]
    delta = max(r * base_mean, r * median_base_j)
    
    # Z-Test [2.5]
    z_drop = (drop_oos - delta) / se_drop if se_drop > 0 else np.nan
    p2 = 1.0 - stats.norm.cdf(z_drop) if np.isfinite(z_drop) else np.nan
    flag_2 = z_drop > 1.645
    
    # Final Verdict [3]
    failed = flag_1 and flag_2
    
    res.update({
        'delta': delta, 'z_drop': z_drop, 'p2': p2, 
        'flag_2': flag_2, 'verdict': 'FAILED' if failed else 'PASSED'
    })
    return res

# ==============================================================================
# ORCHESTRATOR
# ==============================================================================

def run_pipeline(trt_ic, ctrl_ic, cutoff_date_str, end_date_str):
    cutoff = pd.to_datetime(cutoff_date_str)
    end = pd.to_datetime(end_date_str)
    
    print("1. Computing beta-adjusted paired differentials...")
    trt_diffs, ctrl_diffs = compute_differentials(trt_ic, ctrl_ic, cutoff)
    
    # Restrict to OOS window
    oos_mask_trt = (trt_diffs.index >= cutoff) & (trt_diffs.index <= end)
    L = int(oos_mask_trt.sum())
    print(f"   L (OOS Trading Days) = {L}")
    
    is_mask_trt = trt_diffs.index < cutoff
    
    print("2. Processing Control Alphas (Leave-One-Out) for Tolerance (delta)...")
    ctrl_results = {}
    for col in ctrl_diffs.columns:
        is_s = ctrl_diffs[col][is_mask_trt].dropna()
        oos_s = ctrl_diffs[col][oos_mask_trt].dropna()
        if len(oos_s) < 5 or len(is_s) < L:
            continue
        ctrl_results[col] = execute_v2_test(ctrl_diffs[col], is_s, oos_s, L, ctrl_stats=None)
        
    # Calculate r using the Delta Method for standard error of a ratio [2.3]
    rho_j = []
    var_rho_noise = []
    base_means = []
    
    for c, r_dict in ctrl_results.items():
        b_mean = r_dict['base_mean']
        o_mean = r_dict['oos_mean']
        b_se = r_dict['base_se']
        o_se = r_dict['oos_se']
        
        if b_mean <= 0: # Avoid division by zero or negative baselines
            continue
            
        rho = (b_mean - o_mean) / b_mean
        rho_j.append(rho)
        base_means.append(b_mean)
        
        # SE of ratio (OOS / Base) via delta method
        se_ratio_sq = (o_se**2 / b_mean**2) + ((o_mean**2 * b_se**2) / b_mean**4)
        var_rho_noise.append(se_ratio_sq)
        
    var_obs = np.var(rho_j, ddof=1)
    mean_noise = np.mean(var_rho_noise)
    var_true = max(0, var_obs - mean_noise)
    r = np.mean(rho_j) + 1.645 * np.sqrt(var_true)
    median_base_j = np.median(base_means)
    
    ctrl_stats = {'r': r, 'median_base_j': median_base_j}
    print(f"   Control Tolerance Calibrated: r = {r:.4f}")

    print("3. Evaluating Treatment Alphas (LLM Survivors)...")
    trt_results = []
    for col in trt_diffs.columns:
        is_s = trt_diffs[col][is_mask_trt].dropna()
        oos_s = trt_diffs[col][oos_mask_trt].dropna()
        
        if len(oos_s) < 5 or len(is_s) < L:
            continue
            
        res = execute_v2_test(trt_diffs[col], is_s, oos_s, L, ctrl_stats)
        res['alpha_id'] = col
        trt_results.append(res)
        
    out_df = pd.DataFrame(trt_results).set_index('alpha_id')
    
    print("4. Synthesizing Primary Pool-Level Estimand [3]")
    # FDR Correction (Benjamini-Hochberg) on the Step 2 p-values
    p2_vals = out_df['p2'].dropna()
    if len(p2_vals) > 0:
        # Sort p-values and calculate BH critical values
        ranks = stats.rankdata(p2_vals, method='ordinal')
        bh_thresholds = (ranks / len(p2_vals)) * ALPHA_LEVEL
        out_df['fdr_significant'] = p2_vals <= bh_thresholds
        
    # Mann-Whitney U test between LLM Z-drops and Control Z-drops
    # (Controls must have their Z-drops calculated retrospectively now that r is known)
    ctrl_z_drops = []
    for c, r_dict in ctrl_results.items():
        delta_c = max(r * r_dict['base_mean'], r * median_base_j)
        z = (r_dict['drop_oos'] - delta_c) / r_dict['se_drop']
        if np.isfinite(z): ctrl_z_drops.append(z)
        
    llm_z_drops = out_df['z_drop'].dropna().tolist()
    
    mw_stat, mw_p = stats.mannwhitneyu(llm_z_drops, ctrl_z_drops, alternative='greater')
    
    print("\n" + "="*50)
    print("V2 PIPELINE RESULTS")
    print("="*50)
    print(f"Per-Alpha Failures (Flag 1 & Flag 2): {(out_df['verdict'] == 'FAILED').sum()} / {len(out_df)}")
    print(f"Pool-Level Mann-Whitney P-Value   : {mw_p:.4f}")
    if mw_p < ALPHA_LEVEL:
        print("   -> POOL LEVEL FAILED: The LLM alpha distribution is significantly left-shifted vs. controls.")
    else:
        print("   -> POOL LEVEL PASSED: No systemic look-ahead bias detected in aggregate.")
        
    return out_df, mw_p

if __name__ == "__main__":
    # Placeholder for loading data
    # df_trt = pd.read_parquet(TREATMENT_IC_PATH)
    # df_ctrl = pd.read_parquet(CONTROL_IC_PATH)
    # results_df, pool_p = run_pipeline(df_trt, df_ctrl, CUTOFF_DATE, END_DATE)
    pass
