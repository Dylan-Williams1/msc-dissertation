import os
import glob
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

# ==============================================================================
# CONFIGURATION & PARAMETERS
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Phase 1 already exported daily Rank IC per alpha as
# phase1_daily_rank_ic_<model_key>.csv.
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
PHASE1_IC_DIR = os.path.join(REPO_ROOT, "evaluation", "initial_screening", "results")

TREATMENT_KEY = "claude-opus-5"
CONTROL_KEY = "kakushadze-101-v1"

ROOT_DIR = REPO_ROOT
ALPHA_DIR = os.path.join(ROOT_DIR, "alphas", "survived", TREATMENT_KEY)
CONTROL_DIR = os.path.join(ROOT_DIR, "alphas", "raw", CONTROL_KEY)

TREATMENT_IC_PATH = os.path.join(
    PHASE1_IC_DIR, f"phase1_daily_rank_ic_{TREATMENT_KEY}.csv")
CONTROL_IC_PATH = os.path.join(
    PHASE1_IC_DIR, f"phase1_daily_rank_ic_{CONTROL_KEY}.csv")
OUT_DIR = os.path.join(SCRIPT_DIR, "test1_output", TREATMENT_KEY)

# Model knowledge cutoffs are month-granularity. "2026-05" means the model
# knows everything THROUGH May, so the OOS window opens on 1 June. Reading it
# as 1 May would place a known month inside the OOS window and bias the test
# toward detecting decay that is not there.
MODEL_CUTOFFS = {
    "claude-opus-5":       "2026-05",
    "gemini-3.6-flash-v5": "2026-03",
    "gpt-5.6-sol":         "2026-02",
    "biased_control":      "2026-05",
    "kakushadze-101-v1":   "2026-01",
}

if TREATMENT_KEY not in MODEL_CUTOFFS:
    sys.exit(f"\nNo knowledge cutoff recorded for '{TREATMENT_KEY}'.\n"
             "  Add it to MODEL_CUTOFFS; do not fall back to a shared default.")

CUTOFF_DATE = (pd.Period(MODEL_CUTOFFS[TREATMENT_KEY], freq="M")
                 .end_time.normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
END_DATE = "2026-09-09"
HORIZON_H = 1                   # forward-return horizon (h)
B_VALUES = (2, 3, 4, 5, 6, 7, 8)   # baseline lengths voted over
ALPHA_LEVEL = 0.05              # significance level

# Used ONLY when no vote threshold reaches ALPHA_LEVEL. A majority of the B
# values is a stated rule, not a calibrated one, so any verdict that relies on
# it must be reported together with the MEASURED control FPR at that k.
K_FALLBACK = (len(B_VALUES) + 1) // 2


# ==============================================================================
# ECONOMETRICS
# ==============================================================================

def mean_se(series):
    """Standard error of the mean of a block.

    The Newey-West correction is deliberately absent: autocorrelation in d is
    ~0 at every lag because h=1, so each day's Rank IC is scored against a
    fresh non-overlapping forward return. Restore the Bartlett loop before
    using any multi-day horizon.
    """
    if HORIZON_H != 1:
        sys.exit("mean_se drops the Newey-West correction and is valid only at "
                 "HORIZON_H = 1. Restore the Bartlett loop for longer horizons.")
    series = series.dropna()
    T = len(series)
    if T < 5:
        return np.nan, np.nan
    xc = series.to_numpy(dtype=float) - series.mean()
    lrv = max(np.sum(xc ** 2) / T, 1e-14)
    return lrv, np.sqrt(lrv / T)


# ==============================================================================
# DIFFERENTIALS
# ==============================================================================

def compute_differentials(trt_ic, ctrl_ic, cutoff_date):
    """d_i(t) = IC_i(t) - beta_i * CtrlMean(t).

    Beta is estimated IN-SAMPLE only and applied forward, so the OOS window
    cannot influence its own residual.
    """
    common_idx = trt_ic.index.intersection(ctrl_ic.index)
    dropped = len(trt_ic.index) - len(common_idx)
    if dropped:
        print(f"   ! dropped {dropped} treatment dates absent from controls; "
              f"{len(common_idx)} in common")
    trt_ic = trt_ic.loc[common_idx]
    ctrl_ic = ctrl_ic.loc[common_idx]

    is_mask = ctrl_ic.index < cutoff_date

    def fit_beta(y, x):
        y_is, x_is = y[is_mask].dropna(), x[is_mask].dropna()
        common = y_is.index.intersection(x_is.index)
        if len(common) <= 30:
            return 0.0
        var = np.var(x_is[common], ddof=1)
        return np.cov(y_is[common], x_is[common])[0, 1] / var if var > 0 else 0.0

    ctrl_mean = ctrl_ic.mean(axis=1)
    trt_diffs = pd.DataFrame(index=trt_ic.index, columns=trt_ic.columns,
                             dtype=float)
    for col in trt_ic.columns:
        trt_diffs[col] = trt_ic[col] - fit_beta(trt_ic[col], ctrl_mean) * ctrl_mean

    # Controls are residualised leave-one-out so no control sits inside its own benchmark.
    ctrl_diffs = pd.DataFrame(index=ctrl_ic.index, columns=ctrl_ic.columns,
                              dtype=float)
    for col in ctrl_ic.columns:
        others = ctrl_ic.drop(columns=[col]).mean(axis=1)
        ctrl_diffs[col] = ctrl_ic[col] - fit_beta(ctrl_ic[col], others) * others

    return trt_diffs, ctrl_diffs


# ==============================================================================
# EXCESS DECAY
# ==============================================================================

def excess_decay(is_series, oos_series, L, B):
    """Studentised drop from the trailing B*L baseline into the OOS window."""
    B_days = B * L
    base = is_series.iloc[-B_days:] if len(is_series) >= B_days else is_series
    _, se_base = mean_se(base)
    _, se_oos = mean_se(oos_series)

    base_mean, oos_mean = base.mean(), oos_series.mean()
    drop = base_mean - oos_mean
    se_drop = np.sqrt(se_base ** 2 + se_oos ** 2)
    t = drop / se_drop if (np.isfinite(se_drop) and se_drop > 0) else np.nan
    return dict(base_mean=base_mean, oos_mean=oos_mean, drop=drop,
                se_drop=se_drop, t=t)


def score_at_B(trt_diffs, ctrl_diffs, is_mask, oos_mask, L, B):
    """One complete run at a single B.

    The threshold is the 95th percentile of the CONTROLS' OWN studentised
    drops over the identical window, so the per-alpha false-positive rate is
    5% by construction and a window that was bad for every alpha raises the
    bar automatically.
    """
    ctrl_t = {}
    for col in ctrl_diffs.columns:
        i_s = ctrl_diffs[col][is_mask].dropna()
        o_s = ctrl_diffs[col][oos_mask].dropna()
        if len(o_s) < 5 or len(i_s) < L:
            continue
        r = excess_decay(i_s, o_s, L, B)
        if np.isfinite(r["t"]):
            ctrl_t[col] = r["t"]

    if len(ctrl_t) < 10:
        sys.exit(f"\n  Only {len(ctrl_t)} usable control decays at B={B}; "
                 "cannot calibrate an empirical threshold.")

    t_arr = np.asarray(list(ctrl_t.values()))
    # method="higher" snaps to a real order statistic at or above the 95th
    # percentile. With linear interpolation at n=34 exactly 2 controls sit
    # above t_crit (5.9%), so the per-B FPR is pinned one notch above
    # ALPHA_LEVEL and no vote threshold can ever reach 5%.
    t_crit = float(np.quantile(t_arr, 1.0 - ALPHA_LEVEL, method="higher"))

    trt = {}
    for col in trt_diffs.columns:
        i_s = trt_diffs[col][is_mask].dropna()
        o_s = trt_diffs[col][oos_mask].dropna()
        if len(o_s) < 5 or len(i_s) < L:
            continue
        r = excess_decay(i_s, o_s, L, B)
        r["p"] = ((1.0 + np.sum(t_arr >= r["t"])) / (1.0 + len(t_arr))
                  if np.isfinite(r["t"]) else np.nan)
        # When the treatment corpus IS the control corpus (the self/placebo
        # run), scoring an alpha against a threshold it helped define makes
        # the ~5% flag rate definitional rather than measured. Drop it from
        # its own threshold.
        if col in ctrl_t:
            others = np.asarray([v for c, v in ctrl_t.items() if c != col])
            crit = float(np.quantile(others, 1.0 - ALPHA_LEVEL, method="higher"))
        else:
            crit = t_crit
        r["flag"] = bool(np.isfinite(r["t"]) and r["t"] > crit)
        trt[col] = r

    return t_crit, t_arr, ctrl_t, trt


# ==============================================================================
# ORCHESTRATOR
# ==============================================================================

def run_pipeline(trt_ic, ctrl_ic, cutoff_date_str, end_date_str):
    cutoff = pd.to_datetime(cutoff_date_str)
    end = pd.to_datetime(end_date_str)

    print("1. Computing beta-adjusted paired differentials...")
    trt_diffs, ctrl_diffs = compute_differentials(trt_ic, ctrl_ic, cutoff)

    oos_mask = (trt_diffs.index >= cutoff) & (trt_diffs.index <= end)
    is_mask = trt_diffs.index < cutoff
    L = int(oos_mask.sum())
    print(f"   cutoff = {CUTOFF_DATE} (from {TREATMENT_KEY} knowledge cutoff "
          f"{MODEL_CUTOFFS[TREATMENT_KEY]}), end = {END_DATE}")
    print(f"   L (OOS Trading Days) = {L}")

    off = lambda m: m.values[np.triu_indices_from(m.values, k=1)]
    for nm, D in [("treatment", trt_diffs), ("control", ctrl_diffs)]:
        X = D.loc[is_mask].astype(float)
        print(f"   {nm}: var(d) {X.var().mean():.2e} | mean pairwise corr "
              f"{np.nanmean(off(X.corr())):+.3f} | n={X.shape[1]}")

    print(f"2. Scoring excess decay at B = {list(B_VALUES)}...")
    trt_votes, ctrl_votes = {}, {}
    t_by_B, crit_by_B, by_B = {}, {}, {}

    for B in B_VALUES:
        t_crit, t_arr, ctrl_t, trt = score_at_B(
            trt_diffs, ctrl_diffs, is_mask, oos_mask, L, B)
        crit_by_B[B] = t_crit
        t_by_B[B] = {c: r["t"] for c, r in trt.items()}
        by_B[B] = (t_arr, trt)

        for c, tv in ctrl_t.items():
            # Must use the SAME leave-one-out threshold the treatment alphas
            # are judged by, or the measured FPR is biased below the verdict
            # rule it is meant to calibrate.
            others = np.asarray([v for k, v in ctrl_t.items() if k != c])
            crit_c = float(np.quantile(others, 1.0 - ALPHA_LEVEL,
                                       method="higher"))
            ctrl_votes[c] = ctrl_votes.get(c, 0) + int(tv > crit_c)
        for c, r in trt.items():
            trt_votes[c] = trt_votes.get(c, 0) + int(r["flag"])

        print(f"   B={B}: t_crit = {t_crit:.3f} from {len(ctrl_t)} controls "
              f"| {sum(r['flag'] for r in trt.values())} treatment flag(s)")
    # --- calibrate the vote threshold on the controls ------------------------
    # The B values share one OOS window and overlapping baselines, so they are
    # not independent tests and the composite false-positive rate cannot be
    # derived analytically. It is measured instead: the fraction of CONTROL
    # alphas that the rule "flagged at >= k of the B values" would fail.
    n_b = len(B_VALUES)
    cc = np.asarray(list(ctrl_votes.values()))
    fpr = {k: float((cc >= k).mean()) for k in range(1, n_b + 1)}
    ok = [k for k in range(1, n_b + 1) if fpr[k] <= ALPHA_LEVEL]
    k_min = ok[0] if ok else K_FALLBACK

    print("3. Calibrating the vote threshold on the controls...")
    print("   control FPR by rule:  "
          + "  ".join(f">={k}: {fpr[k]:.1%}" for k in range(1, n_b + 1)))
    print(f"   K_MIN = {k_min} of {n_b} "
          f"(smallest k with control FPR <= {ALPHA_LEVEL:.0%})")

    print(f"   note: {len(cc)} controls -> FPR granularity {1/len(cc):.1%}")
    if not ok:
        print(f"   ! no k reaches FPR <= {ALPHA_LEVEL:.0%} "
              f"(best {min(fpr.values()):.1%}). K_MIN set to the majority "
              f"rule, {K_FALLBACK} of {n_b}; MEASURED control FPR at that k = "
              f"{fpr[K_FALLBACK]:.1%}.")

    # --- assemble ------------------------------------------------------------
    # The reference B for the pool statistic and the reported base/oos means.
    # Taken as the MIDDLE of B_VALUES.
    B_ref = B_VALUES[len(B_VALUES) // 2]
    t_arr_ref, trt_ref = by_B[B_ref]

    # Survivor screening selects on IS performance, which induces regression to
    # the mean and a systematically larger drop. Calibrating t_crit on
    # unscreened alphas would therefore set the bar too low. Test whether the
    # screening actually shifts the drop distribution before widening the pool.
    full = load_phase1_ic(CONTROL_IC_PATH).reindex(ctrl_diffs.index)
    extra = [c for c in full.columns if c not in set(ctrl_diffs.columns)]
    if extra:
        # ctrl_ic is the RAW panel; is_mask was built from the treatment/control
        # intersection, so bench must be reindexed onto that common index or the
        # boolean mask is the wrong length.
        bench = (ctrl_ic[ctrl_diffs.columns]
                 .reindex(ctrl_diffs.index).mean(axis=1))   # survivor-pool mean
        ts = []
        for col in extra:
            y_is = full[col][is_mask].dropna()
            x_is = bench[is_mask].dropna()
            common = y_is.index.intersection(x_is.index)
            if len(common) <= 30:
                continue
            var = np.var(x_is[common], ddof=1)
            beta = (np.cov(y_is[common], x_is[common])[0, 1] / var
                    if var > 0 else 0.0)
            d = full[col] - beta * bench
            i_s, o_s = d[is_mask].dropna(), d[oos_mask].dropna()
            if len(o_s) >= 5 and len(i_s) >= L:
                ts.append(excess_decay(i_s, o_s, L, B_ref)["t"])
        ts = np.asarray([x for x in ts if np.isfinite(x)])
        if len(ts) >= 5:
            ks, kp = stats.ks_2samp(t_arr_ref, ts)
            print(f"\n   survivor vs non-survivor drop-t at B={B_ref}: "
                  f"KS p = {kp:.3f}  (n_surv={len(t_arr_ref)}, "
                  f"n_other={len(ts)}, median {np.median(t_arr_ref):+.2f} "
                  f"vs {np.median(ts):+.2f})")
    rows = []
    for col, r in trt_ref.items():
        row = {"alpha_id": col, "base_mean": r["base_mean"],
               "oos_mean": r["oos_mean"]}
        for B in B_VALUES:
            row[f"t_B{B}"] = t_by_B[B].get(col, np.nan)
        row["votes"] = trt_votes.get(col, 0)
        row["verdict"] = "FAILED" if trt_votes.get(col, 0) >= k_min else "PASSED"
        rows.append(row)
    out_df = pd.DataFrame(rows).set_index("alpha_id").sort_index()

    llm_t = [v for v in t_by_B[B_ref].values() if np.isfinite(v)]
    mw_stat, mw_p = stats.mannwhitneyu(llm_t, t_arr_ref.tolist(),
                                       alternative="greater")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_df.to_csv(os.path.join(OUT_DIR, f"test1_votes_{TREATMENT_KEY}.csv"))
    (pd.Series(ctrl_votes, name="votes").rename_axis("alpha_id")
       .to_csv(os.path.join(OUT_DIR, f"test1_control_votes_{CONTROL_KEY}.csv")))

    print("\n" + "=" * 78)
    print(f"EXCESS-DECAY VOTE RESULTS  ({TREATMENT_KEY})")
    print("=" * 78)
    print(out_df.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n   t_crit by B: "
          + "  ".join(f"B={B}: {crit_by_B[B]:.3f}" for B in B_VALUES))
    print(f"   FAILED (votes >= {k_min}): "
          f"{int((out_df['verdict'] == 'FAILED').sum())} / {len(out_df)}")
    print(f"\n   Pool-level Mann-Whitney at B={B_ref}: p = {mw_p:.4f} "
          f"(n_llm={len(llm_t)}, n_ctrl={len(t_arr_ref)})")
    print("   NOTE: Mann-Whitney assumes independence within arms; treatment "
          "alphas are\n         correlated, so treat it as an upper bound on "
          "the aggregate evidence.")
    print(f"\n-> {OUT_DIR}")

    return out_df, k_min, mw_p


# ==============================================================================
# IO
# ==============================================================================

def load_phase1_ic(path):
    """Read a Phase 1 Rank IC panel (dates on the index, alpha_ids as columns)."""
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.sort_index().apply(pd.to_numeric, errors="coerce")
    return df.dropna(axis=1, how="all")


def restrict_to_corpus(ic, alpha_dir, label):
    """Restrict a Phase 1 IC panel to the alphas present in alpha_dir.

    The Phase 1 CSV is the FULL evaluated corpus, not the survivors. Every
    failure path exits rather than falling back to it.
    """
    if not os.path.isdir(alpha_dir):
        sys.exit(f"\n[{label}] survivor directory not found: {alpha_dir}\n"
                 "  Refusing to fall back to the full Phase 1 corpus.")
    files = sorted(glob.glob(os.path.join(alpha_dir, "*.json")))
    if not files:
        sys.exit(f"\n[{label}] ZERO SURVIVING ALPHAS in {alpha_dir}.\n"
                 "  This is a result, not an error.")

    ids = []
    for fp in files:
        stem = os.path.splitext(os.path.basename(fp))[0]
        try:
            rec = json.load(open(fp, encoding="utf-8"))
            ids.append(str(rec.get("metadata", {}).get("alpha_id", stem)))
        except Exception:
            ids.append(stem)

    lower = {c.lower(): c for c in ic.columns}
    keep, missing = [], []
    for aid in ids:
        if aid in ic.columns:
            keep.append(aid)
        elif aid.lower() in lower:
            keep.append(lower[aid.lower()])
        else:
            missing.append(aid)
    keep = list(dict.fromkeys(keep))

    print(f"   [{label}] CSV has {ic.shape[1]} alphas; {alpha_dir} holds "
          f"{len(ids)} artifacts; matched {len(keep)}")
    if missing:
        sys.exit(f"\n[{label}] {len(missing)} SURVIVOR(S) HAVE NO IC SERIES:\n"
                 f"  {', '.join(missing)}\n\n"
                 "  A survivor absent from the Phase 1 panel is a data problem, "
                 "not a smaller corpus.\n  Fix the export or the column naming; "
                 "do not run on the subset that happened to match.")
    if not keep:
        sys.exit(f"\n[{label}] no survivor matched any IC column.")
    return ic[keep]


if __name__ == "__main__":
    df_trt = restrict_to_corpus(load_phase1_ic(TREATMENT_IC_PATH),
                                ALPHA_DIR, "treatment")
    df_ctrl = restrict_to_corpus(load_phase1_ic(CONTROL_IC_PATH),
                                 CONTROL_DIR, "control")
    results_df, k_min, pool_p = run_pipeline(
        df_trt, df_ctrl, CUTOFF_DATE, END_DATE)
