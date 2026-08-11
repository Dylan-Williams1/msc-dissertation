"""
Phase 2 - Test 2: Historical Event Memorisation.

Implements the Test 2 battery exactly as written in the master context document,
Section 4 (MECHANISM - Memorisation), including the Paired Differential Framework
that governs it.

    Step 1  Absolute Anomaly Screen        (unpaired, dual universal placebo)
    Step 2  Control Differential Check     (paired, leave-that-window-out beta)
    Step 3  Salience Matching & Verdict    (Mahalanobis caliper -> Scenario A/B)

Everything the spec fixes is hard-coded to the spec. Everything the spec leaves
open is a named constant in Section 0 below, is tagged OPEN DECISION, and is
echoed to stdout and into the run manifest at the end of every run, so no
unstated assumption can silently become a result. Per master doc Section 11:
flag Section 8 open decisions rather than silently assuming values.

Inputs
------
1. Daily Rank IC panel, LLM arm       phase1_daily_rank_ic_<LLM_MODEL_TAG>.csv
2. Daily Rank IC panel, control arm   phase1_daily_rank_ic_<CONTROL_MODEL_TAG>.csv
3. Phase 1 screening results CSVs     (used only to identify retained survivors)
4. BBDS jumps workbook                WSJ_Stock_Jumps__1900-present_.xlsx
5. Daily OHLCV panel                  daily_ohlcv.parquet  (Step 3 covariates only)

Outputs
-------
test2_events.csv            declustered episode table, salience arms, covariates
test2_event_results.csv     one row per (alpha, event): Steps 1-3 + verdict
test2_alpha_verdicts.csv    one row per alpha: Test 2 verdict + binding failure
test2_caliper_sensitivity.csv   Scenario A/B split across a caliper grid
test2_market_covariates.csv     cached daily market proxy series
test2_run_manifest.json     full config, counts, hashes, open decisions

Language restriction (master doc Section 1) is binding on every string this
script emits: alphas are credible / temporally robust / non-contaminated, never
profitable, deployable, tradable or net-positive.
"""

from __future__ import annotations

import os
import sys
import json
import glob
import hashlib
import argparse
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy
from scipy import stats


# =============================================================================
# 0. CONFIGURATION
# =============================================================================
# Constants marked [SPEC] are fixed by the master context document and must not
# be edited without amending the document. Constants marked [OPEN DECISION] are
# not fixed by the document; each one is reported at the end of the run.

# ---- Arms ------------------------------------------------------------------
LLM_MODEL_TAG = "gemini-3.6-flash-v4"          # tag used in the phase 1 output filenames
CONTROL_MODEL_TAG = "kakushadze-101-v1"

# Also run the control arm through Test 2. Master doc Section 2 requires the
# control corpus to pass through the identical Memory Contamination battery.
RUN_CONTROL_ARM_AS_SUBJECTS = True

# ---- Event window ----------------------------------------------------------
WINDOW = 21                                  # [SPEC] 21 trading days, everywhere

# [OPEN DECISION] The spec fixes the window LENGTH at 21 and says performance is
# examined "around" the event, but never fixes where the window sits relative to
# the anchor day. 21 is odd, so "centered" is the only anchoring that is
# symmetric about the event. "forward" is the alternative reading (the event and
# the 20 sessions that follow it), which is the stronger test of post-event
# memorisation and the weaker test of anticipation.
#   "centered" -> [t-10, t+10]
#   "forward"  -> [t,    t+20]
#   "backward" -> [t-20, t   ]
EVENT_WINDOW_ANCHOR = "centered"

# [OPEN DECISION] Minimum number of days inside a 21-day window that must carry a
# usable observation for that window to produce a statistic. Windows below this
# are dropped and reported, never silently zero-filled.
MIN_WINDOW_OBS = 15

# ---- Event identification --------------------------------------------------
BBDS_SHEET = "jumps by day (wsj)"            # [OPEN DECISION] see note below
# The workbook carries two US jump tables. Both give identical `clarity` (the
# all-papers ClarityIndex is copied into the WSJ sheet) but different
# `JournalistConfidence`: the WSJ sheet averages WSJ coders only, the all-papers
# sheet averages coders across every paper. "jumps by day (all papers)" is the
# alternative.

ANCHOR_START = "1990-01-02"                  # [SPEC] temporal anchor
ANCHOR_END = "2026-07-16"                    # [SPEC] temporal anchor

DECLUSTER_GAP = 21                           # [SPEC] "within 21 trading days"
# [OPEN DECISION] "collapsed into a single shock episode" admits two readings.
#   "chain"       single-linkage: A and B join if gap(A,B) <= DECLUSTER_GAP, and
#                 chains propagate. 2008-09 to 2009-07 becomes ONE episode. This
#                 is the reading that actually delivers the stated purpose,
#                 "prevent a single macro regime from dominating the test".
#   "greedy_peak" repeatedly take the highest-clarity unassigned jump, absorb
#                 everything within DECLUSTER_GAP of it, repeat. Yields more
#                 episodes inside a long crisis.
DECLUSTER_METHOD = "chain"

# [OPEN DECISION] The spec says the episode is anchored on the highest `clarity`
# day. `clarity` is top-coded in BBDS (many days sit exactly at the maximum), so
# a deterministic tiebreak is required. Larger absolute jump return, then the
# earlier date.
ANCHOR_TIEBREAK = ("abs_return_desc", "date_asc")

# [OPEN DECISION] The spec says Narrative Consensus uses "the BBDS `clarity` and
# `JournalistConfidence` metrics" but does not give the functional form. The two
# metrics are on incomparable scales (clarity is a PCA index, JournalistConfidence
# is a 1-3 coder average), so each is z-scored across the declustered episode set
# and the two are averaged with equal weight.
NC_WEIGHTS = {"clarity": 0.5, "JournalistConfidence": 0.5}

SALIENCE_TOP_Q = 0.75                        # [SPEC] top quartile = treatment
SALIENCE_BOTTOM_Q = 0.25                     # [SPEC] bottom quartile = candidate pool

# ---- Paired differential framework ----------------------------------------
# [SPEC] d_i(t) = IC_i(t) - beta_i * CtrlMean(t); beta leave-that-window-out.

# [OPEN DECISION] The spec writes the differential without an intercept but does
# not say whether beta is the OLS slope (intercept fitted and discarded) or the
# no-intercept ratio. "Preserves crash-regime factor loadings" is loading
# language, which points at the slope. With the intercept the alpha's own mean
# level survives into d; without it, beta absorbs part of that level.
BETA_WITH_INTERCEPT = True

# [OPEN DECISION] The spec is silent on the estimation sample boundary for Test 2.
# None -> every date present in the IC panels is estimation sample. Set to a
# date string (e.g. "2025-01-01") to respect the Test 1 in-sample boundary.
# BBDS coverage stops in Nov 2022, well before any cutoff in this study, so this
# only trims the tail of the placebo distributions.
IN_SAMPLE_END = None

# [OPEN DECISION, NECESSARY] When a CONTROL alpha is the test subject it is also
# a component of CtrlMean, which drives beta toward 1 and mechanically shrinks
# its own differential. Leave-one-out CtrlMean removes the circularity. The spec
# does not address this because it defines the differential only from the LLM
# side. Same issue, same fix, for the Step 1 placebo pool.
CTRL_LEAVE_ONE_OUT_FOR_CONTROL_SUBJECTS = True
STEP1_LEAVE_ONE_OUT_FOR_CONTROL_SUBJECTS = True

# Minimum number of retained control alphas with a live IC on a date for
# CtrlMean(t) to be defined on that date.
CTRL_MIN_ALPHAS_PER_DATE = 5

# Restrict the control corpus to Phase 1 survivors ("the retained control
# corpus", "the robust control corpus").
CONTROL_RESTRICT_TO_SURVIVORS = True

# ---- Step 1 ----------------------------------------------------------------
STEP1_PERCENTILE = 95.0                      # [SPEC] 95th percentile threshold

# [OPEN DECISION] The spec requires both the pooled and the per-alpha-demeaned
# verdict to be reported and any discrepancy disclosed, but does not say which
# one gates the Scenario A verdict.
#   "either"   an anomaly under either variant counts as a Step 1 anomaly
#   "both"     both variants must flag
#   "pooled"   / "demeaned"  single nominated variant
# "either" is the strict setting. Master doc Section 11: low power counts against
# certification, so the default leans towards flagging.
STEP1_VERDICT_RULE = "either"

# ---- Step 2 ----------------------------------------------------------------
# [OPEN DECISION] "every sliding 21-day window outside the event set". What
# counts as inside the event set:
#   "all_jumps"      exclude windows overlapping ANY declustered episode window.
#                    Strictest reading of "non-event placebo windows".
#   "tested_events"  exclude only windows overlapping a high- or low-salience
#                    event window actually used by this test.
PLACEBO_EXCLUSION = "all_jumps"

STEP2_ALPHA = 0.05                           # [SPEC] one-sided p < 0.05

# ---- Step 3 ----------------------------------------------------------------
MATCH_COVARIATES = ("realized_vol", "drawdown_depth", "cs_dispersion")  # [SPEC]

# [OPEN DECISION] The spec names the three covariates but does not define them.
#   realized_vol     annualised sd of the equal-weighted cross-sectional mean
#                    daily return over the window
#   drawdown_depth   deepest peak-to-trough fall of the cumulative equal-weighted
#                    market return inside the window, as a positive fraction
#   cs_dispersion    window mean of the daily cross-sectional sd of returns
# See build_market_covariates().

# [OPEN DECISION] All three covariates are positive and right-skewed, and the
# high-salience arm contains extreme outliers (Sep 2008, Mar 2020) that inflate
# the raw covariance and therefore shrink every Mahalanobis distance. Raw is the
# literal reading of the spec and is the default; the log setting is reported as
# a sensitivity in test2_caliper_sensitivity.csv.
MATCH_LOG_TRANSFORM = False

# [OPEN DECISION] "a pre-specified Mahalanobis caliper (maximum allowable
# distance)" - pre-specified is required, the value is not given. This is on the
# DISTANCE scale, not squared. The full distance matrix and a caliper sensitivity
# sweep are exported so the choice is auditable.
MAHALANOBIS_CALIPER = 1.0
CALIPER_SENSITIVITY_GRID = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)

# [OPEN DECISION] Composite low-salience benchmark across matched events.
#   "equal"             equal weight per matched event
#   "inverse_distance"  weight 1/(d + 1e-9), nearer matches count for more
MATCH_COMPOSITE = "equal"

# [OPEN DECISION] The spec says "permutation test" for the salience gap without
# naming the exchangeability unit.
#   "daily"        days are pooled across the high window and the matched low
#                  windows and randomly reassigned to slots of the original
#                  sizes. High resolution; assumes daily exchangeability.
#   "window_label" which window carries the "high" label is permuted, days stay
#                  intact. Preserves serial dependence but the smallest
#                  attainable p-value is 1/(1+n_matched).
# Both are computed and both are reported; this selects which one drives the
# verdict.
PERMUTATION_SCHEME = "daily"
N_PERMUTATIONS = 10000
PERMUTATION_SEED = 20260716                  # master seed of the generation run
STEP3_ALPHA = 0.05

# ---- Multiple testing ------------------------------------------------------
FDR_ALPHA = 0.05                             # [SPEC] Benjamini-Hochberg at 0.05

# [OPEN DECISION] The spec says FDR applies "whenever multiple events are
# evaluated" and that it protects alphas from single-event noise, which places
# the family within the alpha. Whether Step 2 and Step 3 form one family or two
# is not stated.
#   "per_test_per_alpha"  separate BH pass over Step 2 p-values and over Step 3
#                         p-values, within each alpha
#   "pooled_per_alpha"    one BH pass over both, within each alpha
BH_FAMILY = "per_test_per_alpha"

# ---- Paths -----------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "test2_output")

PARQUET_PATH = os.path.join(DATA_DIR, "daily_ohlcv.parquet")
BBDS_PATH = os.path.join(DATA_DIR, "WSJ_Stock_Jumps.xlsx")

# Phase 1 writes its CSVs relative to the working directory, so several
# plausible locations are searched before giving up.
IC_SEARCH_DIRS = (SCRIPT_DIR, os.getcwd(), DATA_DIR, REPO_ROOT)


OPEN_DECISIONS: list[dict] = []


def register_open_decisions() -> None:
    """Collect every unfixed choice so the run cannot hide one."""
    OPEN_DECISIONS.clear()
    OPEN_DECISIONS.extend([
        dict(key="EVENT_WINDOW_ANCHOR", value=EVENT_WINDOW_ANCHOR,
             note="Spec fixes window length at 21 but not its position relative "
                  "to the anchor day."),
        dict(key="MIN_WINDOW_OBS", value=MIN_WINDOW_OBS,
             note="Minimum usable observations inside a 21-day window."),
        dict(key="BBDS_SHEET", value=BBDS_SHEET,
             note="WSJ-coder vs all-paper JournalistConfidence; clarity identical."),
        dict(key="DECLUSTER_METHOD", value=DECLUSTER_METHOD,
             note="Single-linkage chaining vs greedy peak-first absorption."),
        dict(key="ANCHOR_TIEBREAK", value=list(ANCHOR_TIEBREAK),
             note="clarity is top-coded in BBDS, so ties at the maximum are common."),
        dict(key="NC_WEIGHTS", value=NC_WEIGHTS,
             note="Functional form of Narrative Consensus is not given by the spec."),
        dict(key="BETA_WITH_INTERCEPT", value=BETA_WITH_INTERCEPT,
             note="OLS slope vs no-intercept ratio for beta_i."),
        dict(key="IN_SAMPLE_END", value=IN_SAMPLE_END,
             note="Estimation sample boundary for Test 2 is not stated."),
        dict(key="CTRL_LEAVE_ONE_OUT_FOR_CONTROL_SUBJECTS",
             value=CTRL_LEAVE_ONE_OUT_FOR_CONTROL_SUBJECTS,
             note="Necessary addition: a control subject is inside CtrlMean, "
                  "which would shrink its own differential."),
        dict(key="STEP1_VERDICT_RULE", value=STEP1_VERDICT_RULE,
             note="Spec requires both pooled and demeaned verdicts to be reported "
                  "but does not say which gates Scenario A."),
        dict(key="PLACEBO_EXCLUSION", value=PLACEBO_EXCLUSION,
             note="Definition of 'outside the event set' for Step 2 placebos."),
        dict(key="MATCH_COVARIATE_DEFINITIONS", value="see build_market_covariates()",
             note="Spec names the three covariates but does not define them."),
        dict(key="MATCH_LOG_TRANSFORM", value=MATCH_LOG_TRANSFORM,
             note="Raw covariance is dominated by Sep 2008 and Mar 2020."),
        dict(key="MAHALANOBIS_CALIPER", value=MAHALANOBIS_CALIPER,
             note="Spec requires the caliper to be pre-specified but gives no value."),
        dict(key="MATCH_COMPOSITE", value=MATCH_COMPOSITE,
             note="Weighting of the composite low-salience benchmark."),
        dict(key="PERMUTATION_SCHEME", value=PERMUTATION_SCHEME,
             note="Exchangeability unit for the Salience Gap Test."),
        dict(key="BH_FAMILY", value=BH_FAMILY,
             note="Whether Step 2 and Step 3 p-values form one BH family or two."),
    ])


# =============================================================================
# 1. NUMERIC PRIMITIVES
# =============================================================================
# Every windowed statistic in Test 2 is a mean over a fixed-length block of a
# daily series that carries NaNs. All of them are built from one primitive:
# cumulative sums over a validity-masked array, sliced at window boundaries.
# That keeps every sliding-window pass O(T) rather than O(T * W), which matters
# because the leave-that-window-out beta rule refits beta for every one of the
# ~9,000 placebo windows, for every alpha.


def _window_sums(values: np.ndarray, valid: np.ndarray, window: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    """
    Sum and count of `values` inside every window, indexed by window START.

    Positions where `valid` is False contribute nothing to either. The returned
    arrays have length T - window + 1, so element k covers positions
    [k, k + window - 1] inclusive.
    """
    v = np.where(valid, np.nan_to_num(values, nan=0.0), 0.0)
    c = np.concatenate(([0.0], np.cumsum(v)))
    cn = np.concatenate(([0.0], np.cumsum(valid.astype(np.float64))))
    starts = np.arange(len(values) - window + 1)
    return cn[starts + window] - cn[starts], c[starts + window] - c[starts]


def _safe_divide(num: np.ndarray, den: np.ndarray,
                 min_den: float = 1e-15) -> np.ndarray:
    out = np.full_like(np.asarray(num, dtype=np.float64), np.nan)
    ok = np.abs(den) > min_den
    out[ok] = num[ok] / den[ok]
    return out


def rolling_window_mean(series: np.ndarray, valid: np.ndarray, window: int,
                        min_obs: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean of `series` over every window, indexed by window start. Unpaired."""
    n_w, s_w = _window_sums(series, valid, window)
    mean_w = _safe_divide(s_w, n_w)
    mean_w[n_w < min_obs] = np.nan
    return mean_w, n_w


def leave_window_out_betas(y: np.ndarray, x: np.ndarray, valid: np.ndarray,
                           window: int, with_intercept: bool = True
                           ) -> np.ndarray:
    """
    beta_i for every 21-day window, fit on all estimation dates EXCEPT that window.

    Master doc, Beta Estimation Rule (Test 2 & Placebos): "For any 21-day window
    evaluated (whether it is the actual macro shock or a random sliding placebo
    window), beta is fit on all in-sample dates except those specific 21 days.
    This preserves crash-regime factor loadings while ensuring the test statistic
    is perfectly exchangeable under the null."

    Implemented as a closed-form leave-k-out OLS: the full-sample sufficient
    statistics are computed once, the window's own contribution is subtracted,
    and beta is read off the reduced statistics. Numerically identical to
    refitting, and ~4 orders of magnitude faster over 9,000 windows.

    Returns an array indexed by window start, NaN where the reduced design is
    degenerate.
    """
    n_all = float(valid.sum())
    yv = np.where(valid, np.nan_to_num(y, nan=0.0), 0.0)
    xv = np.where(valid, np.nan_to_num(x, nan=0.0), 0.0)

    s_x, s_y = float(xv.sum()), float(yv.sum())
    s_xx, s_xy = float((xv * xv).sum()), float((xv * yv).sum())

    n_w, sx_w = _window_sums(x, valid, window)
    _, sy_w = _window_sums(y, valid, window)
    _, sxx_w = _window_sums(x * x, valid, window)
    _, sxy_w = _window_sums(x * y, valid, window)

    n_r = n_all - n_w                     # r for "reduced": window removed
    sx_r, sy_r = s_x - sx_w, s_y - sy_w
    sxx_r, sxy_r = s_xx - sxx_w, s_xy - sxy_w

    if with_intercept:
        num = n_r * sxy_r - sx_r * sy_r
        den = n_r * sxx_r - sx_r ** 2
    else:
        num, den = sxy_r, sxx_r

    beta = _safe_divide(num, den)
    beta[n_r < 2] = np.nan
    return beta


def window_differentials(y: np.ndarray, x: np.ndarray, valid: np.ndarray,
                         window: int, min_obs: int, with_intercept: bool = True
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mean paired differential d_bar over every window, with its own LWO beta.

        d_i(t)  = IC_i(t) - beta_i * CtrlMean(t)
        d_bar_w = mean over the valid days of window w

    Returns (d_bar, beta, n_valid), each indexed by window start.
    """
    beta = leave_window_out_betas(y, x, valid, window, with_intercept)
    n_w, sy_w = _window_sums(y, valid, window)
    _, sx_w = _window_sums(x, valid, window)

    d_bar = _safe_divide(sy_w - beta * sx_w, n_w)
    d_bar[n_w < min_obs] = np.nan
    return d_bar, beta, n_w


def daily_differentials(y: np.ndarray, x: np.ndarray, valid: np.ndarray,
                        start: int, window: int, beta: float) -> np.ndarray:
    """Day-level d_i(t) inside one window, valid days only. Feeds the permutation test."""
    sl = slice(start, start + window)
    m = valid[sl]
    return (y[sl][m] - beta * x[sl][m])


def benjamini_hochberg(pvals, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg step-up FDR control.

    Master doc: "Because certification requires failing to reject the null, the
    FDR penalty actively protects alphas from being falsely rejected due to
    single-event noise." The direction of the protection is the reason this is
    applied within an alpha across its events, not across alphas.

    Returns (reject, qvalue), both aligned to the input, NaN-safe.
    """
    p = np.asarray(pvals, dtype=np.float64)
    reject = np.zeros(p.shape, dtype=bool)
    qval = np.full(p.shape, np.nan)

    finite = np.where(np.isfinite(p))[0]
    m = finite.size
    if m == 0:
        return reject, qval

    order = finite[np.argsort(p[finite], kind="mergesort")]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    ps = p[order]

    q_sorted = np.minimum.accumulate((ps * m / ranks)[::-1])[::-1]
    qval[order] = np.clip(q_sorted, 0.0, 1.0)

    below = np.where(ps <= alpha * ranks / m)[0]
    if below.size:
        reject[order[: below.max() + 1]] = True
    return reject, qval


def mahalanobis_matrix(a: np.ndarray, b: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Pairwise Mahalanobis DISTANCE (not squared) between rows of a and rows of b."""
    vi = np.linalg.pinv(cov)
    diff = a[:, None, :] - b[None, :, :]
    d2 = np.einsum("ijk,kl,ijl->ij", diff, vi, diff)
    return np.sqrt(np.clip(d2, 0.0, None))


# =============================================================================
# 2. DATA LOADING
# =============================================================================


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_file(filename: str, extra_dirs=()) -> str | None:
    for d in tuple(extra_dirs) + IC_SEARCH_DIRS:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            return candidate
    return None


def load_ic_panel(model_tag: str, explicit_path: str | None = None) -> pd.DataFrame:
    """
    Daily Rank IC panel produced by Phase 1: date index, one column per alpha.

    Master doc Section 3: this series is Instrument 1's raw input, already
    produced, and is NOT regenerated here. Test 2 consumes it as given.
    """
    path = explicit_path or _find_file(f"phase1_daily_rank_ic_{model_tag}.csv")
    if path is None:
        raise FileNotFoundError(
            f"No phase1_daily_rank_ic_{model_tag}.csv found. Searched: "
            + ", ".join(IC_SEARCH_DIRS)
            + ". Pass --llm-ic / --control-ic explicitly."
        )
    panel = pd.read_csv(path, index_col=0)
    panel.index = pd.to_datetime(panel.index)
    panel.index.name = "date"
    panel = panel.sort_index()
    panel = panel.loc[~panel.index.duplicated(keep="first")]
    return panel.astype(np.float64)


def load_survivor_ids(model_tag: str, explicit_path: str | None = None) -> set[str] | None:
    """alpha_ids with status SURVIVED_PHASE_1, or None if the results CSV is absent."""
    path = explicit_path or _find_file(f"phase1_screening_results_{model_tag}.csv")
    if path is None:
        return None
    res = pd.read_csv(path)
    if "status" not in res.columns or "alpha_id" not in res.columns:
        return None
    return set(res.loc[res["status"] == "SURVIVED_PHASE_1", "alpha_id"].astype(str))


def load_bbds_jumps(path: str, sheet: str = BBDS_SHEET) -> pd.DataFrame:
    """
    BBDS (Baker, Bloom, Davis, Sammon) stock market jumps, filtered to the anchor.

    Master doc: "Filter the BBDS dataset for all identified market jumps
    occurring within the 1990-2026 temporal anchor."
    """
    raw = pd.read_excel(path, sheet_name=sheet)
    cols = {c.strip(): c for c in raw.columns}
    need = ["date", "return", "clarity", "JournalistConfidence"]
    missing = [c for c in need if c not in cols]
    if missing:
        raise KeyError(f"BBDS sheet {sheet!r} is missing columns: {missing}")

    j = raw[[cols[c] for c in need]].copy()
    j.columns = need
    j["date"] = pd.to_datetime(j["date"])
    j = j.dropna(subset=["date", "clarity", "JournalistConfidence"])

    in_anchor = (j["date"] >= pd.Timestamp(ANCHOR_START)) & (j["date"] <= pd.Timestamp(ANCHOR_END))
    j = j.loc[in_anchor].sort_values("date").reset_index(drop=True)
    return j


def load_trading_calendar(parquet_path: str, fallback_index: pd.DatetimeIndex
                          ) -> tuple[pd.DatetimeIndex, str]:
    """
    Canonical trading-day grid. Declustering gaps and window boundaries are
    counted on THIS grid, not on calendar days and not on the IC index.

    The IC index is the wrong grid: it drops warm-up days and thin
    cross-sections, so an event early in 1990 would silently shift position.
    """
    if os.path.exists(parquet_path):
        px = pd.read_parquet(parquet_path, engine="pyarrow", columns=["date"])
        cal = pd.DatetimeIndex(pd.to_datetime(px["date"]).unique()).sort_values()
        return cal, "ohlcv_parquet"
    warnings.warn(
        "daily_ohlcv.parquet not found: falling back to the IC panel index as the "
        "trading calendar. Warm-up days are missing from that index, so events in "
        "the first weeks of the sample may be mispositioned. Step 3 matching "
        "covariates are unavailable in this mode.",
        RuntimeWarning,
    )
    return fallback_index, "ic_panel_fallback"


def build_market_covariates(parquet_path: str, cache_path: str) -> pd.DataFrame | None:
    """
    Two daily market-proxy series, from which the three Step 3 matching
    covariates are formed over any window.

        mkt_ret(t)  equal-weighted cross-sectional mean daily return
        cs_std(t)   cross-sectional standard deviation of daily returns

    Cached, because the parquet pass is the slowest part of the run and these
    series do not depend on any alpha.

    Definitions of the three covariates named in the spec (Step 3):
        realized_vol   = sd(mkt_ret over window, ddof=1) * sqrt(252)
        drawdown_depth = deepest peak-to-trough fall of cumprod(1 + mkt_ret)
                         inside the window, as a positive fraction
        cs_dispersion  = mean(cs_std) over the window
    Survivorship inflates all three, most in crash regimes (master doc Section 6);
    inflation applies to both arms, so the matching remains interpretable.
    """
    if os.path.exists(cache_path):
        cov = pd.read_csv(cache_path, index_col=0)
        cov.index = pd.to_datetime(cov.index)
        return cov

    if not os.path.exists(parquet_path):
        return None

    px = pd.read_parquet(parquet_path, engine="pyarrow")
    px["date"] = pd.to_datetime(px["date"])
    px = px.set_index(["date", "symbol"]).sort_index()
    px = px.dropna(subset=["close"])
    ret = px.groupby(level="symbol")["close"].pct_change()

    by_date = ret.groupby(level="date")
    cov = pd.DataFrame({
        "mkt_ret": by_date.mean(),
        "cs_std": by_date.std(ddof=1),
        "n_names": by_date.count(),
    }).sort_index()
    cov.index.name = "date"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cov.to_csv(cache_path)
    return cov


def window_covariates(cov: pd.DataFrame, calendar: pd.DatetimeIndex,
                      start: int, window: int) -> dict[str, float]:
    """The three Step 3 covariates for one window, positioned on the calendar."""
    dates = calendar[start: start + window]
    block = cov.reindex(dates)
    r = block["mkt_ret"].to_numpy(dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < MIN_WINDOW_OBS:
        return dict(realized_vol=np.nan, drawdown_depth=np.nan, cs_dispersion=np.nan)

    curve = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(curve)
    drawdown = float(np.max(1.0 - curve / peak))

    return dict(
        realized_vol=float(np.std(r, ddof=1) * np.sqrt(252.0)),
        drawdown_depth=drawdown,
        cs_dispersion=float(np.nanmean(block["cs_std"].to_numpy(dtype=np.float64))),
    )


# =============================================================================
# 3. EVENT IDENTIFICATION AND SALIENCE EXTRACTION
# =============================================================================


def map_to_calendar(dates: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """
    Position of each jump date on the trading calendar.

    BBDS jump dates are US trading days by construction, so an exact hit is the
    expected case. A miss means the panel's calendar disagrees with BBDS on that
    day; the jump is snapped FORWARD to the next available session and the
    adjustment is recorded rather than absorbed.
    """
    pos = pd.Series(np.arange(len(calendar)), index=calendar)
    out = []
    for d in dates:
        if d in pos.index:
            out.append(int(pos[d]))
        else:
            nxt = calendar[calendar >= d]
            out.append(int(pos[nxt[0]]) if len(nxt) else -1)
    return pd.Series(out, index=dates.index, dtype="int64")


def decluster(jumps: pd.DataFrame, gap: int = DECLUSTER_GAP,
              method: str = DECLUSTER_METHOD) -> pd.DataFrame:
    """
    Collapse jumps within `gap` trading days into a single shock episode,
    anchored on the highest-clarity day.

    Master doc: "To ensure independent draws and prevent a single macro regime
    from dominating the test, jumps occurring within 21 trading days of each
    other are collapsed into a single shock episode, anchored on the specific day
    with the highest BBDS clarity score."

    Chaining is what delivers the stated purpose: the whole of Sep 2008 - Jul 2009
    becomes ONE episode rather than a dozen correlated draws from the same regime.
    """
    j = jumps.sort_values("tpos").reset_index(drop=True).copy()
    if j.empty:
        return j.assign(cluster=pd.Series(dtype="int64"))

    if method == "chain":
        new_cluster = j["tpos"].diff().fillna(10 ** 9) > gap
        j["cluster"] = new_cluster.cumsum().astype(int)

    elif method == "greedy_peak":
        unassigned = np.ones(len(j), dtype=bool)
        cluster = np.full(len(j), -1, dtype=int)
        cid = 0
        while unassigned.any():
            idx = np.where(unassigned)[0]
            seed = idx[np.argmax(j["clarity"].to_numpy()[idx])]
            near = idx[np.abs(j["tpos"].to_numpy()[idx] - j["tpos"].iloc[seed]) <= gap]
            cluster[near] = cid
            unassigned[near] = False
            cid += 1
        j["cluster"] = cluster

    else:
        raise ValueError(f"unknown DECLUSTER_METHOD: {method!r}")

    # Anchor selection: highest clarity, ties broken deterministically because
    # BBDS clarity is top-coded and ties at the maximum are common.
    j["_abs_return"] = j["return"].abs()
    ordered = j.sort_values(
        ["cluster", "clarity", "_abs_return", "date"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    anchors = ordered.groupby("cluster", as_index=False).head(1).copy()
    sizes = j.groupby("cluster").size().rename("episode_n_jumps")
    spans = j.groupby("cluster")["date"].agg(["min", "max"])

    anchors = anchors.merge(sizes, left_on="cluster", right_index=True)
    anchors = anchors.merge(
        spans.rename(columns={"min": "episode_start", "max": "episode_end"}),
        left_on="cluster", right_index=True,
    )
    at_max = j["clarity"].eq(j.groupby("cluster")["clarity"].transform("max"))
    anchors["clarity_tie_at_max"] = anchors["cluster"].map(
        at_max.groupby(j["cluster"]).sum().astype(int)
    )
    return anchors.drop(columns=["_abs_return"]).sort_values("date").reset_index(drop=True)


def score_narrative_consensus(episodes: pd.DataFrame) -> pd.DataFrame:
    """
    Narrative Consensus from the BBDS clarity and JournalistConfidence metrics.

    Each metric is z-scored across the declustered episode set, then combined
    with the weights in NC_WEIGHTS. Both are needed because they measure
    different things: clarity is how much coders agreed on a cause,
    JournalistConfidence is how sure the reporting itself sounded. High values on
    both is the definition of an event whose account is universally documented,
    which is precisely the text an LLM would have seen many times.

    Note for the write-up: JournalistConfidence is top-coded at 3.0 for a large
    share of post-1990 jumps, so at the top of the distribution the composite is
    close to clarity-driven. The count of top-coded episodes is exported.
    """
    e = episodes.copy()
    nc = np.zeros(len(e), dtype=np.float64)
    for col, w in NC_WEIGHTS.items():
        v = e[col].to_numpy(dtype=np.float64)
        sd = np.nanstd(v, ddof=1)
        z = (v - np.nanmean(v)) / (sd if sd > 0 else np.nan)
        e[f"z_{col}"] = z
        nc += w * z
    e["narrative_consensus"] = nc
    return e


def assign_salience(episodes: pd.DataFrame) -> pd.DataFrame:
    """Top quartile -> treatment, bottom quartile -> low-salience candidate pool."""
    e = episodes.copy()
    q_hi = e["narrative_consensus"].quantile(SALIENCE_TOP_Q)
    q_lo = e["narrative_consensus"].quantile(SALIENCE_BOTTOM_Q)
    e["nc_q75"], e["nc_q25"] = q_hi, q_lo

    arm = np.full(len(e), "middle", dtype=object)
    arm[e["narrative_consensus"].to_numpy() >= q_hi] = "high_salience"
    arm[e["narrative_consensus"].to_numpy() <= q_lo] = "low_salience"
    e["salience_arm"] = arm
    return e


def window_start_for_anchor(tpos: int, window: int = WINDOW,
                            anchor: str = EVENT_WINDOW_ANCHOR) -> int:
    """Window start position on the trading calendar for an anchor day."""
    if anchor == "centered":
        return int(tpos - (window - 1) // 2)
    if anchor == "forward":
        return int(tpos)
    if anchor == "backward":
        return int(tpos - window + 1)
    raise ValueError(f"unknown EVENT_WINDOW_ANCHOR: {anchor!r}")


def build_event_table(bbds_path: str, calendar: pd.DatetimeIndex,
                      cov: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    """Full event pipeline: filter -> decluster -> score -> quartile -> position."""
    jumps = load_bbds_jumps(bbds_path, BBDS_SHEET)
    diag = dict(
        bbds_sheet=BBDS_SHEET,
        n_jumps_in_anchor=int(len(jumps)),
        bbds_first_jump=str(jumps["date"].min().date()) if len(jumps) else None,
        bbds_last_jump=str(jumps["date"].max().date()) if len(jumps) else None,
    )

    jumps["tpos"] = map_to_calendar(jumps["date"], calendar)
    unmapped = int((jumps["tpos"] < 0).sum())
    jumps = jumps.loc[jumps["tpos"] >= 0].copy()
    jumps["snapped"] = jumps["date"].to_numpy() != calendar[jumps["tpos"]].to_numpy()
    diag["n_jumps_unmapped"] = unmapped
    diag["n_jumps_snapped_forward"] = int(jumps["snapped"].sum())

    episodes = decluster(jumps)
    episodes = score_narrative_consensus(episodes)
    episodes = assign_salience(episodes)

    episodes["window_start"] = [
        window_start_for_anchor(int(t)) for t in episodes["tpos"]
    ]
    episodes["window_end"] = episodes["window_start"] + WINDOW - 1
    on_grid = (episodes["window_start"] >= 0) & (episodes["window_end"] < len(calendar))
    episodes["window_on_calendar"] = on_grid
    episodes.loc[on_grid, "window_first_date"] = calendar[
        episodes.loc[on_grid, "window_start"].to_numpy()
    ]
    episodes.loc[on_grid, "window_last_date"] = calendar[
        episodes.loc[on_grid, "window_end"].to_numpy()
    ]

    if cov is not None:
        rows = []
        for _, r in episodes.iterrows():
            if r["window_on_calendar"]:
                rows.append(window_covariates(cov, calendar, int(r["window_start"]), WINDOW))
            else:
                rows.append(dict(realized_vol=np.nan, drawdown_depth=np.nan,
                                 cs_dispersion=np.nan))
        for k in MATCH_COVARIATES:
            episodes[k] = [row[k] for row in rows]
    else:
        for k in MATCH_COVARIATES:
            episodes[k] = np.nan

    episodes["event_id"] = [f"E{d:%Y%m%d}" for d in episodes["date"]]

    diag.update(
        n_episodes=int(len(episodes)),
        n_high_salience=int((episodes["salience_arm"] == "high_salience").sum()),
        n_low_salience=int((episodes["salience_arm"] == "low_salience").sum()),
        n_middle=int((episodes["salience_arm"] == "middle").sum()),
        n_episodes_off_calendar=int((~episodes["window_on_calendar"]).sum()),
        n_jc_top_coded=int((episodes["JournalistConfidence"] >= 3.0).sum()),
        largest_episode_n_jumps=int(episodes["episode_n_jumps"].max()) if len(episodes) else 0,
        min_anchor_gap_trading_days=(
            int(episodes["tpos"].diff().min()) if len(episodes) > 1 else None
        ),
    )
    return episodes, diag


# =============================================================================
# 4. THE TEST 2 ENGINE
# =============================================================================


class Test2Engine:
    """
    Holds everything that is shared across alphas: the trading calendar, the
    control IC matrix, the estimation-sample mask, the event windows, and the
    Step 1 placebo machinery. One instance is built per run; alphas are then
    pushed through it one at a time.
    """

    def __init__(self, calendar: pd.DatetimeIndex, ctrl_panel: pd.DataFrame,
                 episodes: pd.DataFrame, cov: pd.DataFrame | None):
        self.calendar = calendar
        self.T = len(calendar)
        self.n_windows = self.T - WINDOW + 1
        self.cov = cov

        # ---- Control matrix on the canonical grid --------------------------
        self.ctrl_ids = list(ctrl_panel.columns)
        self.ctrl = ctrl_panel.reindex(calendar).to_numpy(dtype=np.float64)   # T x C
        self.ctrl_finite = np.isfinite(self.ctrl)

        # ---- Estimation sample ---------------------------------------------
        in_sample = np.ones(self.T, dtype=bool)
        if IN_SAMPLE_END is not None:
            in_sample &= (calendar <= pd.Timestamp(IN_SAMPLE_END)).to_numpy()
        self.in_sample = in_sample

        # ---- CtrlMean(t): equal-weighted mean daily Rank IC across the
        # retained control corpus. No sign orientation is applied (master doc).
        self._ctrl_sum = np.nansum(np.where(self.ctrl_finite, self.ctrl, 0.0), axis=1)
        self._ctrl_cnt = self.ctrl_finite.sum(axis=1).astype(np.float64)
        self.ctrl_mean_all = self._assemble_ctrl_mean(self._ctrl_sum, self._ctrl_cnt)

        # ---- Event windows --------------------------------------------------
        self.episodes = episodes
        usable = episodes["window_on_calendar"].to_numpy(dtype=bool)
        self.high_events = episodes.loc[usable & (episodes["salience_arm"] == "high_salience")].copy()
        self.low_events = episodes.loc[usable & (episodes["salience_arm"] == "low_salience")].copy()
        self.all_event_starts = episodes.loc[usable, "window_start"].to_numpy(dtype=int)

        self.placebo_allowed = self._build_placebo_mask()

        # ---- Step 1 placebo matrices (built once, sliced per subject) -------
        self._build_step1_matrices()

    # -- control mean ------------------------------------------------------
    @staticmethod
    def _assemble_ctrl_mean(total: np.ndarray, count: np.ndarray) -> np.ndarray:
        out = np.full(total.shape, np.nan)
        ok = count >= CTRL_MIN_ALPHAS_PER_DATE
        out[ok] = total[ok] / count[ok]
        return out

    def ctrl_mean_for(self, subject_id: str, arm: str) -> np.ndarray:
        """
        CtrlMean(t) for one subject.

        For an LLM subject this is the plain equal-weighted mean across the
        retained control corpus. For a CONTROL subject the corpus contains the
        subject itself, which drives beta towards 1 and mechanically shrinks the
        subject's own differential, so the subject is left out. The spec does not
        cover this case because it defines the differential from the LLM side
        only; see OPEN_DECISIONS.
        """
        if not (arm == "control" and CTRL_LEAVE_ONE_OUT_FOR_CONTROL_SUBJECTS):
            return self.ctrl_mean_all
        if subject_id not in self.ctrl_ids:
            return self.ctrl_mean_all
        k = self.ctrl_ids.index(subject_id)
        own = np.where(self.ctrl_finite[:, k], self.ctrl[:, k], 0.0)
        return self._assemble_ctrl_mean(
            self._ctrl_sum - own, self._ctrl_cnt - self.ctrl_finite[:, k].astype(np.float64)
        )

    def ctrl_mean_reliability(self, n_splits: int = 40, seed: int = 0) -> dict:
        """
        How much of CtrlMean(t) is common signal and how much is sampling noise.

        This is the single most important diagnostic on the paired differential,
        and it is not in the spec.

        beta_i is an OLS slope on CtrlMean, and CtrlMean is a noisy estimate of
        the latent common component of control IC. Classical errors-in-variables
        attenuation therefore shrinks every beta towards zero by roughly the
        reliability ratio. When reliability is low, d_i(t) = IC_i(t) - beta_i *
        CtrlMean(t) UNDER-removes shared crash exposure, and it under-removes it
        hardest in exactly the windows Test 2 examines, because that is where the
        common component is largest. The visible symptom is a Step 2 placebo
        p-value distribution that is left-skewed under the null: alphas get
        flagged for factor exposure rather than for memorisation.

        Estimated by repeated random split-half correlation of the control corpus
        with the Spearman-Brown correction. Reported, never used to adjust
        anything: the spec fixes the estimator and this run implements it as
        fixed. The number belongs in the limitations section, and it is the reason
        the control arm's own Step 2 rejection rate is the calibration benchmark
        against which the LLM arm's rate has to be read.
        """
        rng = np.random.default_rng(seed)
        n = len(self.ctrl_ids)
        if n < 4:
            return dict(split_half_r=np.nan, reliability=np.nan, n_splits=0)

        rs = []
        for _ in range(n_splits):
            perm = rng.permutation(n)
            a, b = perm[: n // 2], perm[n // 2:]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ma = np.nanmean(np.where(self.ctrl_finite[:, a], self.ctrl[:, a], np.nan), axis=1)
                mb = np.nanmean(np.where(self.ctrl_finite[:, b], self.ctrl[:, b], np.nan), axis=1)
            ok = np.isfinite(ma) & np.isfinite(mb) & self.in_sample
            if ok.sum() > 30:
                rs.append(float(np.corrcoef(ma[ok], mb[ok])[0, 1]))

        if not rs:
            return dict(split_half_r=np.nan, reliability=np.nan, n_splits=0)
        r = float(np.mean(rs))
        return dict(split_half_r=r, reliability=float(2 * r / (1 + r)) if r > -1 else np.nan,
                    n_splits=len(rs))

    # -- placebo window mask ------------------------------------------------
    def _build_placebo_mask(self) -> np.ndarray:
        """
        Which window starts qualify as 'non-event placebo windows' for Step 2.

        A window is admissible only if it lies wholly inside the estimation
        sample and overlaps nothing in the excluded set.
        """
        blocked_days = np.zeros(self.T, dtype=bool)

        if PLACEBO_EXCLUSION == "all_jumps":
            starts = self.all_event_starts
        elif PLACEBO_EXCLUSION == "tested_events":
            starts = np.concatenate([
                self.high_events["window_start"].to_numpy(dtype=int),
                self.low_events["window_start"].to_numpy(dtype=int),
            ]) if len(self.high_events) or len(self.low_events) else np.array([], dtype=int)
        elif PLACEBO_EXCLUSION == "any_jump_day":
            starts = None
            jump_pos = self.episodes["tpos"].to_numpy(dtype=int)
            blocked_days[jump_pos[(jump_pos >= 0) & (jump_pos < self.T)]] = True
        else:
            raise ValueError(f"unknown PLACEBO_EXCLUSION: {PLACEBO_EXCLUSION!r}")

        if starts is not None:
            for s in starts:
                blocked_days[max(0, s): s + WINDOW] = True

        allowed = np.ones(self.n_windows, dtype=bool)
        blocked_cum = np.concatenate(([0], np.cumsum(blocked_days.astype(int))))
        is_cum = np.concatenate(([0], np.cumsum(self.in_sample.astype(int))))
        w = np.arange(self.n_windows)
        allowed &= (blocked_cum[w + WINDOW] - blocked_cum[w]) == 0
        allowed &= (is_cum[w + WINDOW] - is_cum[w]) == WINDOW
        return allowed

    # -- Step 1 placebo distributions ---------------------------------------
    def _build_step1_matrices(self) -> None:
        """
        Dual Universal Placebo Distributions.

        Master doc, Step 1: "Construct global empirical distributions by sliding a
        21-day window (stride 1) across all historical in-sample dates for the
        robust control corpus."

        Note that Step 1's placebo sweep is over ALL in-sample dates, event
        windows included. That is deliberate and different from Step 2: Step 1
        asks how good the best controls get anywhere in history, so removing the
        crises would remove exactly the windows the question is about.

            pooled   window means, pooled across every control and every window
            demeaned each control's own estimation-sample mean IC subtracted
                     first, so the distribution is of window-level unusualness
                     rather than of baseline skill
        """
        c_windows = np.full((len(self.ctrl_ids), self.n_windows), np.nan)
        c_means = np.full(len(self.ctrl_ids), np.nan)

        for k in range(len(self.ctrl_ids)):
            y = self.ctrl[:, k]
            valid = self.ctrl_finite[:, k] & self.in_sample
            if valid.sum() == 0:
                continue
            mean_w, _ = rolling_window_mean(y, valid, WINDOW, MIN_WINDOW_OBS)
            c_windows[k, :] = mean_w
            c_means[k] = float(np.nanmean(y[valid]))

        self._step1_pooled_matrix = c_windows
        self._step1_demeaned_matrix = c_windows - c_means[:, None]
        self._step1_ctrl_full_means = c_means

    def step1_thresholds(self, subject_id: str, arm: str) -> dict:
        """95th percentile of each Universal Control Placebo distribution."""
        keep = np.ones(len(self.ctrl_ids), dtype=bool)
        excluded_self = False
        if (arm == "control" and STEP1_LEAVE_ONE_OUT_FOR_CONTROL_SUBJECTS
                and subject_id in self.ctrl_ids):
            keep[self.ctrl_ids.index(subject_id)] = False
            excluded_self = True

        pooled = self._step1_pooled_matrix[keep].ravel()
        demeaned = self._step1_demeaned_matrix[keep].ravel()
        pooled = pooled[np.isfinite(pooled)]
        demeaned = demeaned[np.isfinite(demeaned)]

        return dict(
            pooled_threshold=float(np.percentile(pooled, STEP1_PERCENTILE)) if pooled.size else np.nan,
            demeaned_threshold=float(np.percentile(demeaned, STEP1_PERCENTILE)) if demeaned.size else np.nan,
            pooled_values=pooled,
            demeaned_values=demeaned,
            n_controls_in_pool=int(keep.sum()),
            excluded_self_from_pool=excluded_self,
        )

    # -- Step 3 matching scaffolding ----------------------------------------
    def build_matching(self, log_transform: bool = MATCH_LOG_TRANSFORM) -> dict:
        """
        Mahalanobis geometry for Step 3.

        Master doc: "Identify valid low-salience matches using Mahalanobis
        distance matching on Realized Volatility, Drawdown Depth, and
        Cross-Sectional Dispersion, imposing a pre-specified Mahalanobis caliper."

        The covariance is estimated over EVERY declustered episode with usable
        covariates, not over the two arms separately: the caliper has to mean the
        same thing on both sides of the comparison, so the metric must come from
        the common population of shock episodes.
        """
        cols = list(MATCH_COVARIATES)
        ep = self.episodes.loc[self.episodes["window_on_calendar"]].copy()
        X = ep[cols].to_numpy(dtype=np.float64)
        ok = np.isfinite(X).all(axis=1)

        if ok.sum() < len(cols) + 1:
            return dict(available=False, reason="insufficient episodes with covariates")

        Xr = X[ok]
        if log_transform:
            Xr = np.log(np.clip(Xr, 1e-12, None))

        cov = np.cov(Xr, rowvar=False)

        def _rows(frame: pd.DataFrame) -> np.ndarray:
            M = frame[cols].to_numpy(dtype=np.float64)
            return np.log(np.clip(M, 1e-12, None)) if log_transform else M

        hi, lo = _rows(self.high_events), _rows(self.low_events)
        hi_ok = np.isfinite(hi).all(axis=1)
        lo_ok = np.isfinite(lo).all(axis=1)

        dist = np.full((len(hi), len(lo)), np.nan)
        if hi_ok.any() and lo_ok.any():
            sub = mahalanobis_matrix(hi[hi_ok], lo[lo_ok], cov)
            dist[np.ix_(hi_ok, lo_ok)] = sub

        return dict(
            available=True, cov=cov, distance=dist, log_transform=log_transform,
            high_ids=self.high_events["event_id"].tolist(),
            low_ids=self.low_events["event_id"].tolist(),
            n_reference_episodes=int(ok.sum()),
        )

    # -- permutation test ----------------------------------------------------
    @staticmethod
    def permutation_salience_gap(d_high: np.ndarray, d_lows: list[np.ndarray],
                                 weights: np.ndarray, rng: np.random.Generator
                                 ) -> dict:
        """
        Salience Gap Test.

        Master doc, Scenario B(1): "The alpha must NOT show selective
        overperformance on the high-salience event relative to the matched
        low-salience composite. The contrast on the differentials
        (d_high - d_low,composite) must not be statistically significantly
        positive via permutation test."

        Two exchangeability units, both reported:

        daily         days are pooled across the high window and every matched
                      low window and randomly reassigned to slots of the original
                      sizes. The composite weighting is preserved slot-by-slot,
                      so the permuted statistic is the same functional as the
                      observed one. Resolution is limited only by N_PERMUTATIONS.
        window_label  which window carries the 'high' label is permuted, days
                      stay intact. Exact enumeration over 1 + n_matched
                      assignments. Respects serial dependence inside a window,
                      but the smallest attainable p-value is 1/(1 + n_matched),
                      so with a single match it cannot go below 0.5. Reported
                      because that ceiling is itself a power statement, and
                      master doc Section 11 requires low power to count against
                      certification rather than for it.
        """
        sizes = [len(d_high)] + [len(a) for a in d_lows]
        if min(sizes) == 0:
            return dict(observed=np.nan, p_daily=np.nan, p_window_label=np.nan,
                        min_attainable_p_window_label=np.nan, n_permutations=0)

        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()

        low_means = np.array([a.mean() for a in d_lows])
        observed = float(d_high.mean() - float(np.dot(w, low_means)))

        # --- daily scheme -------------------------------------------------
        pool = np.concatenate([d_high] + d_lows)
        bounds = np.cumsum([0] + sizes)
        tiled = np.tile(pool, (N_PERMUTATIONS, 1))
        permuted = rng.permuted(tiled, axis=1)
        slot_means = np.column_stack([
            permuted[:, bounds[k]: bounds[k + 1]].mean(axis=1) for k in range(len(sizes))
        ])
        stat_daily = slot_means[:, 0] - slot_means[:, 1:] @ w
        p_daily = float(np.mean(stat_daily >= observed))

        # --- window-label scheme (exact enumeration) ----------------------
        group_means = np.concatenate([[d_high.mean()], low_means])
        n_g = len(group_means)
        stats_wl = []
        for k in range(n_g):
            others = np.delete(group_means, k)
            wk = np.delete(np.concatenate([[0.0], w]), k)
            wk = wk / wk.sum() if wk.sum() > 0 else np.full(n_g - 1, 1.0 / (n_g - 1))
            stats_wl.append(group_means[k] - float(np.dot(wk, others)))
        stats_wl = np.asarray(stats_wl)
        p_wl = float(np.mean(stats_wl >= observed))

        return dict(
            observed=observed,
            p_daily=p_daily,
            p_daily_plus1=float((np.sum(stat_daily >= observed) + 1) / (N_PERMUTATIONS + 1)),
            p_window_label=p_wl,
            min_attainable_p_window_label=float(1.0 / n_g),
            n_permutations=int(N_PERMUTATIONS),
        )

    # -- per-alpha evaluation ------------------------------------------------
    def evaluate_alpha(self, alpha_id: str, ic_series: pd.Series, arm: str,
                       matching: dict, rng: np.random.Generator) -> pd.DataFrame:
        """
        Run Steps 1-3 for one alpha across every high-salience event, then apply
        Benjamini-Hochberg within the alpha and resolve the per-event verdict.
        """
        y = ic_series.reindex(self.calendar).to_numpy(dtype=np.float64)
        x = self.ctrl_mean_for(alpha_id, arm)

        valid_raw = np.isfinite(y) & self.in_sample
        valid_paired = valid_raw & np.isfinite(x)

        # ---- Step 1 inputs: unpaired raw window means ----------------------
        raw_means, raw_n = rolling_window_mean(y, valid_raw, WINDOW, MIN_WINDOW_OBS)
        alpha_full_mean = float(np.nanmean(y[valid_raw])) if valid_raw.any() else np.nan
        thr = self.step1_thresholds(alpha_id, arm)

        # ---- Step 2 inputs: paired differentials, LWO beta per window ------
        d_bar, beta, n_pair = window_differentials(
            y, x, valid_paired, WINDOW, MIN_WINDOW_OBS, BETA_WITH_INTERCEPT
        )
        placebo_mask = self.placebo_allowed & np.isfinite(d_bar)
        placebo_vals = d_bar[placebo_mask]

        # Full-sample loading on the control corpus, reported alongside the
        # leave-window-out betas so a reader can see how far any single window
        # moves the estimate, and how much of the alpha's IC the control corpus
        # explains at all. A near-zero R^2 means the differential is doing almost
        # nothing and d is effectively raw IC.
        diag_beta_full, diag_r2 = np.nan, np.nan
        if valid_paired.sum() > 2:
            xs, ys = x[valid_paired], y[valid_paired]
            vx = float(np.var(xs))
            if vx > 0:
                diag_beta_full = float(np.cov(xs, ys, ddof=1)[0, 1] / np.var(xs, ddof=1))
                diag_r2 = float(np.corrcoef(xs, ys)[0, 1] ** 2)

        # ---- Low-salience side, precomputed once per alpha -----------------
        low_start = self.low_events["window_start"].to_numpy(dtype=int)
        low_ids = self.low_events["event_id"].tolist()
        low_dbar, low_daily = {}, {}
        for eid, s in zip(low_ids, low_start):
            if 0 <= s < self.n_windows and np.isfinite(d_bar[s]):
                low_dbar[eid] = float(d_bar[s])
                low_daily[eid] = daily_differentials(y, x, valid_paired, s, WINDOW, beta[s])

        rows = []
        for _, ev in self.high_events.iterrows():
            s = int(ev["window_start"])
            row = dict(
                alpha_id=alpha_id, arm=arm, event_id=ev["event_id"],
                event_date=ev["date"].date().isoformat(),
                event_return=float(ev["return"]),
                clarity=float(ev["clarity"]),
                journalist_confidence=float(ev["JournalistConfidence"]),
                narrative_consensus=float(ev["narrative_consensus"]),
                episode_n_jumps=int(ev["episode_n_jumps"]),
                window_first_date=str(pd.Timestamp(ev["window_first_date"]).date()),
                window_last_date=str(pd.Timestamp(ev["window_last_date"]).date()),
                n_placebo_windows_step2=int(placebo_vals.size),
                n_controls_in_step1_pool=thr["n_controls_in_pool"],
                diag_beta_full_sample=diag_beta_full,
                diag_r2_ic_on_ctrlmean=diag_r2,
            )

            usable = 0 <= s < self.n_windows
            row["n_valid_days_unpaired"] = float(raw_n[s]) if usable else np.nan
            row["n_valid_days_paired"] = float(n_pair[s]) if usable else np.nan

            # ---------- STEP 1: Absolute Anomaly Screen (unpaired) ----------
            ic_raw_event = float(raw_means[s]) if usable else np.nan
            ic_dm_event = ic_raw_event - alpha_full_mean
            row.update(
                step1_ic_raw_event=ic_raw_event,
                step1_alpha_full_sample_mean_ic=alpha_full_mean,
                step1_ic_demeaned_event=ic_dm_event,
                step1_pooled_threshold_p95=thr["pooled_threshold"],
                step1_demeaned_threshold_p95=thr["demeaned_threshold"],
                step1_pooled_exceedance=(
                    float(np.mean(thr["pooled_values"] >= ic_raw_event))
                    if np.isfinite(ic_raw_event) and thr["pooled_values"].size else np.nan
                ),
                step1_demeaned_exceedance=(
                    float(np.mean(thr["demeaned_values"] >= ic_dm_event))
                    if np.isfinite(ic_dm_event) and thr["demeaned_values"].size else np.nan
                ),
            )
            a_pooled = bool(np.isfinite(ic_raw_event) and ic_raw_event > thr["pooled_threshold"])
            a_dm = bool(np.isfinite(ic_dm_event) and ic_dm_event > thr["demeaned_threshold"])
            row["step1_anomaly_pooled"] = a_pooled
            row["step1_anomaly_demeaned"] = a_dm
            row["step1_variants_disagree"] = bool(a_pooled != a_dm)
            row["step1_anomaly"] = {
                "either": a_pooled or a_dm,
                "both": a_pooled and a_dm,
                "pooled": a_pooled,
                "demeaned": a_dm,
            }[STEP1_VERDICT_RULE]

            # ---------- STEP 2: Control Differential Check (paired) ---------
            d_event = float(d_bar[s]) if usable else np.nan
            row["step2_d_bar_event"] = d_event
            row["step2_beta_lwo"] = float(beta[s]) if usable else np.nan
            if np.isfinite(d_event) and placebo_vals.size:
                n_ge = int(np.sum(placebo_vals >= d_event))
                row["step2_p_placebo"] = n_ge / placebo_vals.size
                row["step2_p_placebo_plus1"] = (n_ge + 1) / (placebo_vals.size + 1)
            else:
                row["step2_p_placebo"] = np.nan
                row["step2_p_placebo_plus1"] = np.nan

            # ---------- STEP 3: Salience Matching ---------------------------
            row.update(self._match_and_gap(ev, s, d_event, d_bar, beta, y, x,
                                           valid_paired, matching, low_ids,
                                           low_dbar, low_daily, rng))
            rows.append(row)

        out = pd.DataFrame(rows)
        if out.empty:
            return out
        return self._apply_fdr_and_verdicts(out)

    def _match_and_gap(self, ev, s, d_event, d_bar, beta, y, x, valid_paired,
                       matching, low_ids, low_dbar, low_daily, rng) -> dict:
        """Mahalanobis caliper, scenario routing, and the Salience Gap Test."""
        res = dict(
            scenario=None, n_matched_low=0, matched_low_ids="",
            matched_low_distances="", nearest_low_id="", nearest_low_distance=np.nan,
            step3_d_bar_low_composite=np.nan, step3_salience_gap=np.nan,
            step3_p_permutation=np.nan, step3_p_perm_daily=np.nan,
            step3_p_perm_window_label=np.nan,
            step3_min_attainable_p_window_label=np.nan,
        )

        if not matching.get("available", False):
            res["scenario"] = "A"
            res["scenario_reason"] = "matching covariates unavailable (no OHLCV panel)"
            return res

        try:
            hi_pos = matching["high_ids"].index(ev["event_id"])
        except ValueError:
            res["scenario"] = "A"
            res["scenario_reason"] = "event absent from matching frame"
            return res

        dist_row = matching["distance"][hi_pos]
        finite = np.isfinite(dist_row)
        if finite.any():
            j = int(np.nanargmin(np.where(finite, dist_row, np.inf)))
            res["nearest_low_id"] = matching["low_ids"][j]
            res["nearest_low_distance"] = float(dist_row[j])

        within = np.where(finite & (dist_row <= MAHALANOBIS_CALIPER))[0]
        eligible = [(matching["low_ids"][k], float(dist_row[k])) for k in within
                    if matching["low_ids"][k] in low_dbar
                    and low_daily[matching["low_ids"][k]].size >= MIN_WINDOW_OBS]

        if not eligible:
            res["scenario"] = "A"
            res["scenario_reason"] = (
                "no low-salience event inside the caliper"
                if within.size == 0 else
                "low-salience matches inside the caliper have no usable differential"
            )
            return res

        # ---- Scenario B ----------------------------------------------------
        res["scenario"] = "B"
        res["scenario_reason"] = "at least one valid low-salience match inside the caliper"
        ids = [e[0] for e in eligible]
        dists = np.array([e[1] for e in eligible], dtype=np.float64)
        res["n_matched_low"] = len(ids)
        res["matched_low_ids"] = "|".join(ids)
        res["matched_low_distances"] = "|".join(f"{d:.4f}" for d in dists)

        if MATCH_COMPOSITE == "inverse_distance":
            w = 1.0 / (dists + 1e-9)
        elif MATCH_COMPOSITE == "equal":
            w = np.ones(len(ids))
        else:
            raise ValueError(f"unknown MATCH_COMPOSITE: {MATCH_COMPOSITE!r}")
        w = w / w.sum()

        composite = float(np.dot(w, [low_dbar[i] for i in ids]))
        res["step3_d_bar_low_composite"] = composite
        res["step3_salience_gap"] = d_event - composite if np.isfinite(d_event) else np.nan

        if np.isfinite(d_event):
            d_high_daily = daily_differentials(y, x, valid_paired, s, WINDOW, beta[s])
            perm = self.permutation_salience_gap(
                d_high_daily, [low_daily[i] for i in ids], w, rng
            )
            res["step3_p_perm_daily"] = perm["p_daily"]
            res["step3_p_perm_window_label"] = perm["p_window_label"]
            res["step3_min_attainable_p_window_label"] = perm["min_attainable_p_window_label"]
            res["step3_p_permutation"] = (
                perm["p_daily"] if PERMUTATION_SCHEME == "daily" else perm["p_window_label"]
            )
        return res

    # -- FDR and verdicts ----------------------------------------------------
    @staticmethod
    def _apply_fdr_and_verdicts(df: pd.DataFrame) -> pd.DataFrame:
        """
        Benjamini-Hochberg within the alpha, then the Step 3 verdict rules.

        Master doc: "Whenever multiple events are evaluated, statistical
        significance is evaluated using the Benjamini-Hochberg False Discovery
        Rate (FDR) control at alpha = 0.05. Because certification requires failing
        to reject the null, the FDR penalty actively protects alphas from being
        falsely rejected due to single-event noise."
        """
        p2 = df["step2_p_placebo"].to_numpy(dtype=np.float64)
        p3 = df["step3_p_permutation"].to_numpy(dtype=np.float64)

        if BH_FAMILY == "per_test_per_alpha":
            r2, q2 = benjamini_hochberg(p2, FDR_ALPHA)
            r3, q3 = benjamini_hochberg(p3, FDR_ALPHA)
        elif BH_FAMILY == "pooled_per_alpha":
            pooled = np.concatenate([p2, p3])
            r, q = benjamini_hochberg(pooled, FDR_ALPHA)
            n = len(df)
            r2, q2, r3, q3 = r[:n], q[:n], r[n:], q[n:]
        else:
            raise ValueError(f"unknown BH_FAMILY: {BH_FAMILY!r}")

        df["step2_q_bh"], df["step3_q_bh"] = q2, q3
        df["step2_bh_reject"], df["step3_bh_reject"] = r2, r3

        # One-sided: only a POSITIVE differential is evidence of memorisation.
        # A significantly negative differential is not a failure, it is the
        # opposite of the effect under test.
        df["step2_significant_positive"] = r2 & (df["step2_d_bar_event"].to_numpy() > 0)
        df["step3_significant_positive"] = r3 & (df["step3_salience_gap"].to_numpy() > 0)

        verdicts, binding, evaluable = [], [], []
        for _, r in df.iterrows():
            can_test = np.isfinite(r["step2_d_bar_event"]) and r["n_placebo_windows_step2"] > 0
            evaluable.append(bool(can_test))
            if not can_test:
                verdicts.append("INDETERMINATE")
                binding.append("insufficient_data")
                continue

            if r["scenario"] == "A":
                # "An alpha fails certification if it exhibits a Step 1 anomaly
                # that survives as a statistically significant positive Step 2
                # differential (placebo p < 0.05)."
                fail = bool(r["step1_anomaly"] and r["step2_significant_positive"])
                verdicts.append("FAIL" if fail else "PASS")
                binding.append("step1_anomaly_and_control_differential" if fail else "none")
            else:
                # "An alpha must pass BOTH tests to be certified."
                gap_fail = bool(r["step3_significant_positive"])
                ctrl_fail = bool(r["step2_significant_positive"])
                if gap_fail and ctrl_fail:
                    verdicts.append("FAIL"); binding.append("salience_gap_and_control_differential")
                elif gap_fail:
                    verdicts.append("FAIL"); binding.append("salience_gap")
                elif ctrl_fail:
                    verdicts.append("FAIL"); binding.append("control_differential")
                else:
                    verdicts.append("PASS"); binding.append("none")

        df["event_evaluable"] = evaluable
        df["event_verdict"] = verdicts
        df["binding_failure_mode"] = binding
        return df


# =============================================================================
# 5. SYNTHESIS
# =============================================================================


def synthesise_alpha_verdicts(event_results: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the per-event table to one row per alpha.

    Master doc, SYNTHESIS Stage 4: report the empirical distribution of binding
    failure modes - which mode binds, how often, whether it varies by model, and
    how each rate compares to the control group.

    Test 2 alone never certifies anything. A PASS here means Test 2 found no
    affirmative evidence of event-level memorisation for that alpha; certification
    is a joint claim across Gate 0 and every mechanism test.
    """
    out = []
    for (alpha_id, arm), g in event_results.groupby(["alpha_id", "arm"], sort=False):
        ev = g.loc[g["event_evaluable"]]
        n_eval = len(ev)
        n_fail = int((ev["event_verdict"] == "FAIL").sum())

        if n_eval == 0:
            verdict = "INDETERMINATE_INSUFFICIENT_DATA"
        elif n_fail > 0:
            verdict = "FAIL_EVENT_MEMORISATION_FLAGGED"
        else:
            verdict = "PASS_NO_EVENT_MEMORISATION_DETECTED"

        modes = ev.loc[ev["event_verdict"] == "FAIL", "binding_failure_mode"]
        out.append(dict(
            alpha_id=alpha_id, arm=arm,
            test2_verdict=verdict,
            n_events_presented=len(g),
            n_events_evaluable=n_eval,
            n_events_failed=n_fail,
            n_scenario_a=int((ev["scenario"] == "A").sum()),
            n_scenario_b=int((ev["scenario"] == "B").sum()),
            binding_failure_modes="|".join(sorted(set(modes))) if n_fail else "none",
            n_step1_anomalies=int(ev["step1_anomaly"].sum()),
            n_step1_variant_disagreements=int(ev["step1_variants_disagree"].sum()),
            n_step2_significant_positive=int(ev["step2_significant_positive"].sum()),
            n_step3_significant_positive=int(ev["step3_significant_positive"].sum()),
            min_step2_q=float(ev["step2_q_bh"].min()) if n_eval else np.nan,
            min_step3_q=float(ev["step3_q_bh"].min()) if n_eval else np.nan,
            mean_step2_d_bar=float(ev["step2_d_bar_event"].mean()) if n_eval else np.nan,
            mean_beta_lwo=float(ev["step2_beta_lwo"].mean()) if n_eval else np.nan,
        ))
    return pd.DataFrame(out)


def caliper_sensitivity(engine: Test2Engine) -> pd.DataFrame:
    """
    How the Scenario A / Scenario B split moves with the caliper, under both the
    raw and the log-transformed metric.

    The caliper is required to be pre-specified, which means it cannot be chosen
    after seeing the verdicts. This table exists so the reader can see what the
    pre-specified choice bought or cost, which is the disclosure that makes a
    pre-specification credible rather than merely asserted.
    """
    rows = []
    for log_tf in (False, True):
        m = engine.build_matching(log_transform=log_tf)
        if not m.get("available", False):
            continue
        dist = m["distance"]
        for c in CALIPER_SENSITIVITY_GRID:
            n_b = int(np.sum(np.nansum(dist <= c, axis=1) > 0))
            rows.append(dict(
                log_transform=log_tf, caliper=c,
                n_high_events=dist.shape[0],
                n_scenario_b=n_b,
                n_scenario_a=dist.shape[0] - n_b,
                mean_matches_per_high_event=float(np.nanmean(np.nansum(dist <= c, axis=1))),
                is_prespecified_value=bool((not log_tf) == (not MATCH_LOG_TRANSFORM)
                                           and abs(c - MAHALANOBIS_CALIPER) < 1e-12),
            ))
    return pd.DataFrame(rows)


# =============================================================================
# 6. DRIVER
# =============================================================================


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2 Test 2 - Historical Event Memorisation."
    )
    p.add_argument("--llm-ic", default=None, help="phase1_daily_rank_ic_<llm>.csv")
    p.add_argument("--control-ic", default=None, help="phase1_daily_rank_ic_<control>.csv")
    p.add_argument("--llm-results", default=None, help="phase1_screening_results_<llm>.csv")
    p.add_argument("--control-results", default=None, help="phase1_screening_results_<control>.csv")
    p.add_argument("--bbds", default=BBDS_PATH, help="BBDS jumps workbook (.xlsx)")
    p.add_argument("--parquet", default=PARQUET_PATH, help="daily_ohlcv.parquet")
    p.add_argument("--outdir", default=OUTPUT_DIR)
    p.add_argument("--llm-tag", default=LLM_MODEL_TAG)
    p.add_argument("--control-tag", default=CONTROL_MODEL_TAG)
    p.add_argument("--all-llm-alphas", action="store_true",
                   help="Evaluate every alpha in the LLM IC panel, not just Phase 1 survivors.")
    p.add_argument("--skip-control-subjects", action="store_true",
                   help="Do not run the control arm through Test 2 as subjects.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    register_open_decisions()
    os.makedirs(args.outdir, exist_ok=True)
    started = datetime.now(timezone.utc)

    print("=" * 78)
    print("PHASE 2 - TEST 2: HISTORICAL EVENT MEMORISATION")
    print("=" * 78)

    # ---- 1. IC panels ------------------------------------------------------
    llm_panel = load_ic_panel(args.llm_tag, args.llm_ic)
    ctrl_panel = load_ic_panel(args.control_tag, args.control_ic)
    print(f"\nLLM IC panel      : {llm_panel.shape[1]} alphas x {llm_panel.shape[0]} dates "
          f"({llm_panel.index.min().date()} to {llm_panel.index.max().date()})")
    print(f"Control IC panel  : {ctrl_panel.shape[1]} alphas x {ctrl_panel.shape[0]} dates "
          f"({ctrl_panel.index.min().date()} to {ctrl_panel.index.max().date()})")

    # ---- 2. Retained corpora ----------------------------------------------
    ctrl_survivors = load_survivor_ids(args.control_tag, args.control_results)
    if CONTROL_RESTRICT_TO_SURVIVORS and ctrl_survivors:
        keep = [c for c in ctrl_panel.columns if c in ctrl_survivors]
        dropped = ctrl_panel.shape[1] - len(keep)
        ctrl_panel = ctrl_panel[keep]
        print(f"Retained controls : {len(keep)} (dropped {dropped} non-survivors)")
    elif CONTROL_RESTRICT_TO_SURVIVORS:
        warnings.warn(
            "CONTROL_RESTRICT_TO_SURVIVORS is True but no control screening results "
            "CSV was found. The full control panel is being used as the retained "
            "corpus, which is NOT what the spec calls the robust control corpus.",
            RuntimeWarning,
        )

    llm_survivors = load_survivor_ids(args.llm_tag, args.llm_results)
    llm_cols = list(llm_panel.columns)
    if llm_survivors and not args.all_llm_alphas:
        llm_cols = [c for c in llm_cols if c in llm_survivors]
        print(f"LLM subjects      : {len(llm_cols)} Phase 1 survivors")
    else:
        print(f"LLM subjects      : {len(llm_cols)} (survivor filter not applied)")

    if ctrl_panel.shape[1] < CTRL_MIN_ALPHAS_PER_DATE:
        raise ValueError(
            f"Only {ctrl_panel.shape[1]} control alphas retained; "
            f"CTRL_MIN_ALPHAS_PER_DATE is {CTRL_MIN_ALPHAS_PER_DATE}. "
            "CtrlMean would be undefined on every date."
        )

    # ---- 3. Calendar and market covariates --------------------------------
    ic_union = llm_panel.index.union(ctrl_panel.index)
    calendar, calendar_source = load_trading_calendar(args.parquet, ic_union)
    print(f"Trading calendar  : {len(calendar)} sessions from {calendar_source}")

    cov = build_market_covariates(
        args.parquet, os.path.join(args.outdir, "test2_market_covariates.csv")
    )
    print(f"Market covariates : {'available' if cov is not None else 'UNAVAILABLE - Step 3 forced to Scenario A'}")

    # ---- 4. Events ---------------------------------------------------------
    episodes, ev_diag = build_event_table(args.bbds, calendar, cov)
    print("\n--- EVENT IDENTIFICATION -------------------------------------------")
    print(f"BBDS jumps in anchor window : {ev_diag['n_jumps_in_anchor']} "
          f"({ev_diag['bbds_first_jump']} to {ev_diag['bbds_last_jump']})")
    print(f"Declustered episodes        : {ev_diag['n_episodes']} "
          f"(largest absorbs {ev_diag['largest_episode_n_jumps']} jumps)")
    print(f"High-salience (treatment)   : {ev_diag['n_high_salience']}")
    print(f"Low-salience (candidates)   : {ev_diag['n_low_salience']}")
    print(f"Middle quartiles (unused)   : {ev_diag['n_middle']}")
    if ev_diag["n_episodes_off_calendar"]:
        print(f"  ! {ev_diag['n_episodes_off_calendar']} episode window(s) fall off the calendar edge")
    if ev_diag["n_jumps_snapped_forward"]:
        print(f"  ! {ev_diag['n_jumps_snapped_forward']} jump date(s) snapped forward to the next session")

    bbds_end = pd.Timestamp(ev_diag["bbds_last_jump"]) if ev_diag["bbds_last_jump"] else None
    if bbds_end is not None and bbds_end < pd.Timestamp(ANCHOR_END) - pd.Timedelta(days=365):
        print(f"\n  ** COVERAGE LIMITATION: BBDS coding stops at {bbds_end.date()}, but the")
        print(f"     temporal anchor runs to {ANCHOR_END}. Test 2 therefore says nothing")
        print( "     about events after that date. Disclose in the limitations section.")

    episodes.to_csv(os.path.join(args.outdir, "test2_events.csv"), index=False)

    # ---- 5. Engine ---------------------------------------------------------
    engine = Test2Engine(calendar, ctrl_panel, episodes, cov)
    matching = engine.build_matching()
    rng = np.random.default_rng(PERMUTATION_SEED)

    reliability = engine.ctrl_mean_reliability()
    print(f"\nCtrlMean split-half reliability   : {reliability['reliability']:.3f} "
          f"(split-half r = {reliability['split_half_r']:.3f})")
    if np.isfinite(reliability["reliability"]) and reliability["reliability"] < 0.5:
        print("  ** CtrlMean is a noisy proxy for the latent common component. OLS beta")
        print("     is attenuated towards zero by roughly this factor, so the paired")
        print("     differential UNDER-removes shared crash exposure - hardest in exactly")
        print("     the windows Test 2 examines. Expect the Step 2 placebo p-values to be")
        print("     left-skewed under the null. Read the LLM arm's rejection rate against")
        print("     the control arm's, never against the nominal 5%.")

    n_placebo = int(engine.placebo_allowed.sum())
    print(f"\nStep 2 placebo windows admissible : {n_placebo} "
          f"of {engine.n_windows} sliding windows "
          f"(exclusion rule '{PLACEBO_EXCLUSION}')")
    if n_placebo < 200:
        warnings.warn(
            f"Only {n_placebo} placebo windows survive the exclusion rule. The "
            "smallest attainable Step 2 p-value is 1/n, so the test may be unable "
            "to reach the 0.05 threshold. Low power counts AGAINST certification.",
            RuntimeWarning,
        )
    if engine.high_events.empty:
        raise ValueError("No usable high-salience events. Test 2 cannot run.")

    # ---- 6. Subjects -------------------------------------------------------
    subjects = [(c, llm_panel[c], "llm") for c in llm_cols]
    if RUN_CONTROL_ARM_AS_SUBJECTS and not args.skip_control_subjects:
        ctrl_ids = load_survivor_ids(args.control_tag, args.control_results)
        ctrl_all = load_ic_panel(args.control_tag, args.control_ic)
        cols = [c for c in ctrl_all.columns
                if (not CONTROL_RESTRICT_TO_SURVIVORS or not ctrl_ids or c in ctrl_ids)]
        subjects += [(c, ctrl_all[c], "control") for c in cols]

    print(f"\nSubjects to evaluate : {len(subjects)} "
          f"({sum(1 for s in subjects if s[2] == 'llm')} LLM, "
          f"{sum(1 for s in subjects if s[2] == 'control')} control)")
    print(f"High-salience events : {len(engine.high_events)}  ->  "
          f"{len(subjects) * len(engine.high_events)} (alpha, event) tests\n")

    # ---- 7. Run ------------------------------------------------------------
    frames = []
    for i, (aid, series, arm) in enumerate(subjects, 1):
        res = engine.evaluate_alpha(aid, series, arm, matching, rng)
        if res.empty:
            continue
        frames.append(res)
        n_fail = int((res["event_verdict"] == "FAIL").sum())
        n_eval = int(res["event_evaluable"].sum())
        flag = "FAIL" if n_fail else ("PASS" if n_eval else "INDET")
        print(f"[{i:>3}/{len(subjects)}] {aid:<28} {arm:<8} "
              f"evaluable={n_eval:>2} step1_anom={int(res['step1_anomaly'].sum()):>2} "
              f"step2_sig={int(res['step2_significant_positive'].sum()):>2} "
              f"step3_sig={int(res['step3_significant_positive'].sum()):>2}  [{flag}]")

    if not frames:
        raise ValueError("No alpha produced any evaluable event.")

    event_results = pd.concat(frames, ignore_index=True)
    alpha_verdicts = synthesise_alpha_verdicts(event_results)

    # ---- 8. Export ---------------------------------------------------------
    event_results.to_csv(os.path.join(args.outdir, "test2_event_results.csv"), index=False)
    alpha_verdicts.to_csv(os.path.join(args.outdir, "test2_alpha_verdicts.csv"), index=False)
    caliper_sensitivity(engine).to_csv(
        os.path.join(args.outdir, "test2_caliper_sensitivity.csv"), index=False
    )

    manifest = dict(
        run_started_utc=started.isoformat(),
        run_finished_utc=datetime.now(timezone.utc).isoformat(),
        python_version=sys.version.split()[0],
        pandas_version=pd.__version__,
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
        llm_model_tag=args.llm_tag,
        control_model_tag=args.control_tag,
        calendar_source=calendar_source,
        n_calendar_sessions=int(len(calendar)),
        n_retained_controls=int(ctrl_panel.shape[1]),
        retained_control_ids=list(ctrl_panel.columns),
        n_llm_subjects=int(sum(1 for s in subjects if s[2] == "llm")),
        n_control_subjects=int(sum(1 for s in subjects if s[2] == "control")),
        n_placebo_windows=n_placebo,
        n_sliding_windows=int(engine.n_windows),
        ctrl_mean_reliability=reliability,
        matching_available=bool(matching.get("available", False)),
        n_matching_reference_episodes=int(matching.get("n_reference_episodes", 0)),
        bbds_sha256=_sha256(args.bbds) if os.path.exists(args.bbds) else None,
        event_diagnostics=ev_diag,
        spec_fixed=dict(window=WINDOW, decluster_gap=DECLUSTER_GAP,
                        step1_percentile=STEP1_PERCENTILE, step2_alpha=STEP2_ALPHA,
                        step3_alpha=STEP3_ALPHA, fdr_alpha=FDR_ALPHA,
                        salience_top_q=SALIENCE_TOP_Q,
                        salience_bottom_q=SALIENCE_BOTTOM_Q,
                        anchor_start=ANCHOR_START, anchor_end=ANCHOR_END),
        open_decisions=OPEN_DECISIONS,
    )
    with open(os.path.join(args.outdir, "test2_run_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    # ---- 9. Console synthesis ---------------------------------------------
    print("\n" + "=" * 78)
    print("SYNTHESIS")
    print("=" * 78)
    for arm, g in alpha_verdicts.groupby("arm"):
        print(f"\n{arm.upper()} ARM  (n = {len(g)})")
        for v, n in g["test2_verdict"].value_counts().items():
            print(f"   {v:<40} {n:>3}  ({n / len(g):.0%})")

    print("\nBINDING FAILURE MODE DISTRIBUTION (per evaluable (alpha, event) test)")
    ev = event_results.loc[event_results["event_evaluable"]]
    tab = pd.crosstab(ev["binding_failure_mode"], ev["arm"])
    print(tab.to_string() if len(tab) else "   (no evaluable tests)")

    print("\nSCENARIO ROUTING")
    print(pd.crosstab(ev["scenario"], ev["arm"]).to_string())

    print("\nCALIBRATION AGAINST THE CONTROL ARM")
    ctrl_ev = ev.loc[ev["arm"] == "control"]
    llm_ev = ev.loc[ev["arm"] == "llm"]
    if len(ctrl_ev):
        for name, col in [("Step 1 anomaly", "step1_anomaly"),
                          ("Step 2 significant positive", "step2_significant_positive"),
                          ("Step 3 significant positive", "step3_significant_positive")]:
            c = float(ctrl_ev[col].mean())
            l = float(llm_ev[col].mean()) if len(llm_ev) else np.nan
            print(f"   {name:<30} control {c:6.1%}   LLM {l:6.1%}   gap {l - c:+.1%}")
        print("   The control corpus was published in 2016 and cannot have memorised")
        print("   anything an LLM memorised, so its rate IS the empirical false-positive")
        print("   rate of this battery. An LLM rate at or below it is not evidence of")
        print("   event-level memorisation, whatever the nominal p-values say.")
    else:
        print("   Control arm not run as subjects, so the battery has no empirical")
        print("   false-positive benchmark in this run. Re-run without")
        print("   --skip-control-subjects before reporting any LLM rejection rate.")

    n_disagree = int(ev["step1_variants_disagree"].sum())
    print(f"\nStep 1 pooled vs demeaned verdicts disagree on {n_disagree} of {len(ev)} "
          f"tests ({n_disagree / max(len(ev), 1):.1%}). Required disclosure per spec.")

    wl_ceiling = ev["step3_min_attainable_p_window_label"].dropna()
    if len(wl_ceiling):
        print(f"Window-label permutation floor across Scenario B tests: "
              f"min {wl_ceiling.min():.3f}, max {wl_ceiling.max():.3f}. Any value above "
              f"{STEP3_ALPHA} means that scheme could not have rejected regardless of the data.")

    print("\n--- OPEN DECISIONS ACTIVE IN THIS RUN -------------------------------")
    print("Each is a choice the master context document does not fix. Values used:")
    for od in OPEN_DECISIONS:
        print(f"  {od['key']:<42} = {od['value']}")
        print(f"      {od['note']}")

    print(f"\nOutputs written to {args.outdir}")
    print("\nNOTE: a Test 2 PASS is the absence of affirmative evidence of event-level")
    print("memorisation for that alpha. It is not certification, which is a joint claim")
    print("across Gate 0 and every mechanism test, and it is not a claim about")
    print("profitability, deployability or tradability. All figures are GROSS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
