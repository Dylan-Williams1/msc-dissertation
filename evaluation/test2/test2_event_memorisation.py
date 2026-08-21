"""
Test 2: Historical Event Memorisation.

    Step 1  Absolute Anomaly Screen        (unpaired, dual universal placebo)
    Step 2  Control Differential Check     (paired, leave-that-window-out beta)
    Step 3  Salience Matching & Verdict    (Mahalanobis caliper -> Scenario A/B)

SUBJECTS ARE LLM ALPHAS ONLY. The control corpus enters this test purely as the
baseline: it supplies CtrlMean(t) for the paired differential and the Universal
Control Placebo distributions for Step 1. Control alphas are never test subjects.

Inputs
    phase1_daily_rank_ic_<LLM_TAG>.csv          LLM daily Rank IC panel
    phase1_daily_rank_ic_<CONTROL_TAG>.csv      control daily Rank IC panel
    phase1_screening_results_<TAG>.csv          used to identify survivors
    WSJ_Stock_Jumps.xlsx         BBDS jumps
    daily_ohlcv.parquet                         Step 3 matching covariates

Outputs
    test2_events.csv           declustered episodes, salience arms, covariates,
                               BBDS coder description and dominant category
    test2_event_results.csv    one row per (alpha, event), full statistics
    test2_alpha_verdicts.csv   one row per alpha, flag counts and verdict

Language restriction: alphas are credible / temporally robust / non-contaminated,
never profitable, deployable, tradable or net-positive. All figures GROSS.
"""

from __future__ import annotations

import os
import argparse
import warnings

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================
# [SPEC] fixed by the specification.  [UNSPECIFIED] not fixed by it; these are
# echoed at the end of every run so no assumption is silent.

LLM_MODEL_TAG = "claude-opus-5"
CONTROL_MODEL_TAG = "kakushadze-101-v1"

WINDOW = 21                    # [SPEC] 21-day window, stride 1
DECLUSTER_GAP = 21             # [SPEC] jumps within 21 trading days collapse
ANCHOR_START = "1990-01-02"    # [SPEC] temporal anchor
ANCHOR_END = "2026-07-16"      # [SPEC] temporal anchor
SALIENCE_TOP_Q = 0.75          # [SPEC] top quartile = treatment
SALIENCE_BOTTOM_Q = 0.25       # [SPEC] bottom quartile = candidate pool
STEP1_PERCENTILE = 95.0        # [SPEC] 95th percentile anomaly threshold
ALPHA_LEVEL = 0.05             # [SPEC] one-sided p < 0.05
FDR_ALPHA = 0.05               # [SPEC] Benjamini-Hochberg at 0.05

# [UNSPECIFIED] 21 is odd, so centering is the only symmetric placement of the
# window about the anchor day: [t-10, t+10].
EVENT_WINDOW_ANCHOR = "centered"

# [UNSPECIFIED] Minimum usable days inside a window for it to yield a statistic.
MIN_WINDOW_OBS = 15

# [UNSPECIFIED] The workbook holds two US jump tables with identical `clarity`
# but different `JournalistConfidence` (WSJ coders vs all papers).
BBDS_SHEET = "jumps by day (wsj)"

# [UNSPECIFIED] Narrative Consensus functional form. The two BBDS metrics are on
# incomparable scales (clarity is a PCA index, JournalistConfidence a 1-3 coder
# average), so each is z-scored across episodes and averaged with equal weight.
NC_WEIGHTS = {"clarity": 0.5, "JournalistConfidence": 0.5}

# [UNSPECIFIED] Mahalanobis caliper. Must be pre-specified, no value given.
MAHALANOBIS_CALIPER = 1.0

# [UNSPECIFIED] Permutation count and seed for the Salience Gap Test.
N_PERMUTATIONS = 10000
PERMUTATION_SEED = 20260716

# Minimum retained control alphas live on a date for CtrlMean(t) to be defined.
CTRL_MIN_ALPHAS_PER_DATE = 5

# Event naming. Both are DISPLAY METADATA ONLY and enter no statistic: they exist
# so tables are readable without a separate lookup. Both come from BBDS itself
# rather than from hand-labelling, so no researcher discretion is introduced.
#   event_description   the coders' own free-text account of the jump, from the
#                       `key_passages` sheet, at the anchor date
#   bbds_category       the largest category share on the anchor date, from the
#                       category columns of the jumps sheet
BBDS_CATEGORY_COLS = [
    "commodities", "corporate", "elections", "ERP/CC", "foreign", "govspend",
    "Trade Policy", "macro", "monetary", "No Article", "Other NP", "Other Policy",
    "Regulation", "sovmil", "taxes", "terror", "unknown",
]
BBDS_CATEGORY_LABELS = {
    "ERP/CC": "exch rate / capital controls",
    "govspend": "govt spending",
    "sovmil": "sovereign military",
    "macro": "macro news",
    "monetary": "monetary policy",
    "Other NP": "other non-policy",
    "Other Policy": "other policy",
    "Trade Policy": "trade policy",
    "No Article": "no article",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "test2_output")
PARQUET_PATH = os.path.join(DATA_DIR, "daily_ohlcv.parquet")
BBDS_PATH = os.path.join(DATA_DIR, "WSJ_Stock_Jumps.xlsx")
SEARCH_DIRS = (SCRIPT_DIR, os.getcwd(), DATA_DIR, REPO_ROOT)

UNSPECIFIED = [
    ("EVENT_WINDOW_ANCHOR", EVENT_WINDOW_ANCHOR, "window placement about the anchor day"),
    ("MIN_WINDOW_OBS", MIN_WINDOW_OBS, "minimum usable days in a window"),
    ("BBDS_SHEET", BBDS_SHEET, "WSJ-coder vs all-paper JournalistConfidence"),
    ("NC_WEIGHTS", NC_WEIGHTS, "Narrative Consensus functional form"),
    ("MAHALANOBIS_CALIPER", MAHALANOBIS_CALIPER, "caliper value (pre-specified, not given)"),
    ("N_PERMUTATIONS", N_PERMUTATIONS, "permutation count for the Salience Gap Test"),
]


# =============================================================================
# NUMERIC PRIMITIVES
# =============================================================================
# Every windowed statistic is a mean over a fixed block of a NaN-carrying daily
# series, so all of them are built from cumulative sums sliced at the window
# boundaries. That keeps each sliding pass O(T) rather than O(T*W), which matters
# because the leave-that-window-out rule refits beta for every placebo window.


def _window_sums(values: np.ndarray, valid: np.ndarray, window: int):
    """Count and sum of `values` inside every window, indexed by window START."""
    v = np.where(valid, np.nan_to_num(values, nan=0.0), 0.0)
    c = np.concatenate(([0.0], np.cumsum(v)))
    cn = np.concatenate(([0.0], np.cumsum(valid.astype(np.float64))))
    s = np.arange(len(values) - window + 1)
    return cn[s + window] - cn[s], c[s + window] - c[s]


def _safe_divide(num, den, min_den=1e-15):
    out = np.full_like(np.asarray(num, dtype=np.float64), np.nan)
    ok = np.abs(den) > min_den
    out[ok] = num[ok] / den[ok]
    return out


def rolling_window_mean(series, valid, window, min_obs):
    """Mean of `series` over every window, indexed by window start (unpaired)."""
    n_w, s_w = _window_sums(series, valid, window)
    mean_w = _safe_divide(s_w, n_w)
    mean_w[n_w < min_obs] = np.nan
    return mean_w, n_w


def leave_window_out_betas(y, x, valid, window):
    """
    beta_i for every window, fit on all in-sample dates EXCEPT that window.

    Spec, Beta Estimation Rule (Test 2 & Placebos): for any 21-day window
    evaluated, whether the actual macro shock or a sliding placebo, beta is fit
    on all in-sample dates except those specific 21 days. This preserves
    crash-regime factor loadings while keeping the statistic exchangeable under
    the null.

    Implemented as closed-form leave-k-out OLS: full-sample sufficient statistics
    computed once, the window's own contribution subtracted, beta read off the
    remainder. Numerically identical to refitting.
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

    n_r = n_all - n_w
    sx_r, sy_r = s_x - sx_w, s_y - sy_w
    sxx_r, sxy_r = s_xx - sxx_w, s_xy - sxy_w

    beta = _safe_divide(n_r * sxy_r - sx_r * sy_r, n_r * sxx_r - sx_r ** 2)
    beta[n_r < 2] = np.nan
    return beta


def window_differentials(y, x, valid, window, min_obs):
    """
    Mean paired differential over every window, each with its own LWO beta.

        d_i(t)  = IC_i(t) - beta_i * CtrlMean(t)
        d_bar_w = mean over the valid days of window w
    """
    beta = leave_window_out_betas(y, x, valid, window)
    n_w, sy_w = _window_sums(y, valid, window)
    _, sx_w = _window_sums(x, valid, window)
    d_bar = _safe_divide(sy_w - beta * sx_w, n_w)
    d_bar[n_w < min_obs] = np.nan
    return d_bar, beta, n_w


def daily_differentials(y, x, valid, start, window, beta):
    """Day-level d_i(t) inside one window, valid days only. Feeds the permutation."""
    sl = slice(start, start + window)
    m = valid[sl]
    return y[sl][m] - beta * x[sl][m]


def benjamini_hochberg(pvals, alpha):
    """
    Benjamini-Hochberg step-up FDR control. Returns (reject, qvalue), NaN-safe.

    Spec: because certification requires failing to reject the null, the FDR
    penalty actively protects alphas from being falsely rejected due to
    single-event noise. That direction is why it is applied within an alpha
    across its events.
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
    qval[order] = np.clip(np.minimum.accumulate((ps * m / ranks)[::-1])[::-1], 0.0, 1.0)

    below = np.where(ps <= alpha * ranks / m)[0]
    if below.size:
        reject[order[: below.max() + 1]] = True
    return reject, qval


def mahalanobis_matrix(a, b, cov):
    """Pairwise Mahalanobis DISTANCE (not squared) between rows of a and rows of b."""
    vi = np.linalg.pinv(cov)
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.clip(np.einsum("ijk,kl,ijl->ij", diff, vi, diff), 0.0, None))


# =============================================================================
# DATA LOADING
# =============================================================================


def _find(filename):
    for d in SEARCH_DIRS:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return None


def load_ic_panel(model_tag, explicit_path=None):
    """Daily Rank IC panel from Phase 1. Consumed as given, never regenerated."""
    path = explicit_path or _find(f"phase1_daily_rank_ic_{model_tag}.csv")
    if path is None:
        raise FileNotFoundError(
            f"phase1_daily_rank_ic_{model_tag}.csv not found in {SEARCH_DIRS}. "
            "Pass --llm-ic / --control-ic explicitly."
        )
    panel = pd.read_csv(path, index_col=0)
    panel.index = pd.to_datetime(panel.index)
    panel.index.name = "date"
    panel = panel.sort_index()
    return panel.loc[~panel.index.duplicated(keep="first")].astype(np.float64)


def load_survivor_ids(model_tag, explicit_path=None):
    """alpha_ids with status SURVIVED_PHASE_1, or None if the results CSV is absent."""
    path = explicit_path or _find(f"phase1_screening_results_{model_tag}.csv")
    if path is None:
        return None
    res = pd.read_csv(path)
    if "status" not in res.columns or "alpha_id" not in res.columns:
        return None
    return set(res.loc[res["status"] == "SURVIVED_PHASE_1", "alpha_id"].astype(str))


def load_bbds_jumps(path, sheet=BBDS_SHEET):
    """
    BBDS market jumps filtered to the 1990-2026 temporal anchor.

    Also carries the dominant category, taken as the largest of the BBDS category
    shares on that day. The shares sum to 1 by construction, so the largest is the
    modal coder attribution. Display metadata only.
    """
    raw = pd.read_excel(path, sheet_name=sheet)
    cols = {c.strip(): c for c in raw.columns}
    need = ["date", "return", "clarity", "JournalistConfidence"]
    missing = [c for c in need if c not in cols]
    if missing:
        raise KeyError(f"BBDS sheet {sheet!r} missing columns: {missing}")

    j = raw[[cols[c] for c in need]].copy()
    j.columns = need
    j["date"] = pd.to_datetime(j["date"])

    present = [c for c in BBDS_CATEGORY_COLS if c in cols]
    if present:
        shares = raw[[cols[c] for c in present]].astype(np.float64).fillna(0.0)
        top = shares.to_numpy().argmax(axis=1)
        j["bbds_category"] = [BBDS_CATEGORY_LABELS.get(present[i], present[i]) for i in top]
        j["bbds_category_share"] = shares.to_numpy().max(axis=1)
    else:
        j["bbds_category"] = ""
        j["bbds_category_share"] = np.nan

    j = j.dropna(subset=need)
    in_anchor = (j["date"] >= pd.Timestamp(ANCHOR_START)) & (j["date"] <= pd.Timestamp(ANCHOR_END))
    return j.loc[in_anchor].sort_values("date").reset_index(drop=True)


def load_bbds_descriptions(path):
    """
    The coders' own account of each jump, from the `key_passages` sheet.

    Several coders describe the same day independently, so the shortest
    non-empty description is taken as the label: they are near-paraphrases of one
    another ("weak jobs report" / "dismal jobs report") and the shortest is the
    one carrying the least editorialising. Returns date -> description, empty if
    the sheet is missing. Coverage is partial, roughly two thirds of post-1990
    jump days, and it stops earlier than the jump list itself.
    """
    try:
        k = pd.read_excel(path, sheet_name="key_passages")
    except (ValueError, KeyError):
        return {}
    if "date" not in k.columns or "description" not in k.columns:
        return {}

    k = k[["date", "description"]].copy()
    k["date"] = pd.to_datetime(k["date"], errors="coerce")
    k["description"] = k["description"].astype(str).str.strip()
    k = k.loc[k["date"].notna() & k["description"].str.len().gt(0)
              & ~k["description"].str.lower().isin(["nan", "none"])]
    if k.empty:
        return {}

    k["_len"] = k["description"].str.len()
    best = k.sort_values(["date", "_len"]).groupby("date", as_index=False).head(1)
    return dict(zip(best["date"], best["description"]))


def load_trading_calendar(parquet_path, fallback_index):
    """
    Canonical trading-day grid. Declustering gaps and window boundaries are
    counted on THIS grid, not on calendar days and not on the IC index, which
    drops warm-up days and would silently shift early events.
    """
    if os.path.exists(parquet_path):
        px = pd.read_parquet(parquet_path, engine="pyarrow", columns=["date"])
        return pd.DatetimeIndex(pd.to_datetime(px["date"]).unique()).sort_values(), "ohlcv"
    warnings.warn(
        "daily_ohlcv.parquet not found: falling back to the IC panel index as the "
        "trading calendar, and Step 3 matching covariates are unavailable, which "
        "forces every event to Scenario A.",
        RuntimeWarning,
    )
    return fallback_index, "ic_panel_fallback"


def build_market_covariates(parquet_path, cache_path):
    """
    Daily market proxies from which the three Step 3 covariates are formed:

        realized_vol    sd of the equal-weighted market return over the window,
                        annualised
        drawdown_depth  deepest peak-to-trough fall of the cumulative market
                        return inside the window, as a positive fraction
        cs_dispersion   window mean of the daily cross-sectional sd of returns
    """
    if os.path.exists(cache_path):
        cov = pd.read_csv(cache_path, index_col=0)
        cov.index = pd.to_datetime(cov.index)
        return cov
    if not os.path.exists(parquet_path):
        return None

    px = pd.read_parquet(parquet_path, engine="pyarrow")
    px["date"] = pd.to_datetime(px["date"])
    px = px.set_index(["date", "symbol"]).sort_index().dropna(subset=["close"])
    ret = px.groupby(level="symbol")["close"].pct_change()
    by_date = ret.groupby(level="date")

    cov = pd.DataFrame({"mkt_ret": by_date.mean(), "cs_std": by_date.std(ddof=1)}).sort_index()
    cov.index.name = "date"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cov.to_csv(cache_path)
    return cov


def window_covariates(cov, calendar, start, window):
    """The three Step 3 matching covariates for one window."""
    block = cov.reindex(calendar[start: start + window])
    r = block["mkt_ret"].to_numpy(dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < MIN_WINDOW_OBS:
        return dict(realized_vol=np.nan, drawdown_depth=np.nan, cs_dispersion=np.nan)

    curve = np.cumprod(1.0 + r)
    return dict(
        realized_vol=float(np.std(r, ddof=1) * np.sqrt(252.0)),
        drawdown_depth=float(np.max(1.0 - curve / np.maximum.accumulate(curve))),
        cs_dispersion=float(np.nanmean(block["cs_std"].to_numpy(dtype=np.float64))),
    )


# =============================================================================
# EVENT IDENTIFICATION & SALIENCE EXTRACTION
# =============================================================================


def map_to_calendar(dates, calendar):
    """Position of each jump date on the trading calendar, snapped forward if absent."""
    pos = pd.Series(np.arange(len(calendar)), index=calendar)
    out = []
    for d in dates:
        if d in pos.index:
            out.append(int(pos[d]))
        else:
            nxt = calendar[calendar >= d]
            out.append(int(pos[nxt[0]]) if len(nxt) else -1)
    return pd.Series(out, index=dates.index, dtype="int64")


def decluster(jumps, gap=DECLUSTER_GAP):
    """
    Collapse jumps within `gap` trading days into a single shock episode,
    anchored on the day with the highest BBDS clarity score.

    Chaining is what delivers the stated purpose - preventing a single macro
    regime from dominating - since it makes the whole of Sep 2008 to Jul 2009 one
    episode rather than a dozen correlated draws from the same regime.

    clarity is top-coded in BBDS, so ties at the maximum are common and are
    broken deterministically: larger absolute jump return, then earlier date.
    """
    j = jumps.sort_values("tpos").reset_index(drop=True).copy()
    if j.empty:
        return j.assign(cluster=pd.Series(dtype="int64"))

    j["cluster"] = (j["tpos"].diff().fillna(10 ** 9) > gap).cumsum().astype(int)
    j["_abs_return"] = j["return"].abs()

    anchors = (
        j.sort_values(["cluster", "clarity", "_abs_return", "date"],
                      ascending=[True, False, False, True], kind="mergesort")
        .groupby("cluster", as_index=False).head(1).copy()
    )
    anchors = anchors.merge(
        j.groupby("cluster").size().rename("episode_n_jumps"),
        left_on="cluster", right_index=True,
    )
    return anchors.drop(columns=["_abs_return"]).sort_values("date").reset_index(drop=True)


def score_salience(episodes):
    """
    Narrative Consensus from the BBDS clarity and JournalistConfidence metrics,
    then the top and bottom quartiles.

    Note for the write-up: JournalistConfidence is top-coded at 3.0 for a large
    share of post-1990 jumps, so at the top of the distribution the composite is
    close to clarity-driven.
    """
    e = episodes.copy()
    nc = np.zeros(len(e), dtype=np.float64)
    for col, w in NC_WEIGHTS.items():
        v = e[col].to_numpy(dtype=np.float64)
        sd = np.nanstd(v, ddof=1)
        nc += w * (v - np.nanmean(v)) / (sd if sd > 0 else np.nan)
    e["narrative_consensus"] = nc

    q_hi = e["narrative_consensus"].quantile(SALIENCE_TOP_Q)
    q_lo = e["narrative_consensus"].quantile(SALIENCE_BOTTOM_Q)
    arm = np.full(len(e), "middle", dtype=object)
    arm[nc >= q_hi] = "high_salience"
    arm[nc <= q_lo] = "low_salience"
    e["salience"] = arm
    return e


def build_events(bbds_path, calendar, cov):
    """Filter -> decluster -> score -> quartile -> position -> covariates."""
    jumps = load_bbds_jumps(bbds_path, BBDS_SHEET)
    n_jumps = len(jumps)
    jumps["tpos"] = map_to_calendar(jumps["date"], calendar)
    jumps = jumps.loc[jumps["tpos"] >= 0].copy()

    ep = score_salience(decluster(jumps))

    offset = {"centered": (WINDOW - 1) // 2, "forward": 0, "backward": WINDOW - 1}[EVENT_WINDOW_ANCHOR]
    ep["window_start"] = ep["tpos"] - offset
    ep["window_end"] = ep["window_start"] + WINDOW - 1
    ep["usable"] = (ep["window_start"] >= 0) & (ep["window_end"] < len(calendar))
    ep.loc[ep["usable"], "window_first_date"] = calendar[ep.loc[ep["usable"], "window_start"].to_numpy()]
    ep.loc[ep["usable"], "window_last_date"] = calendar[ep.loc[ep["usable"], "window_end"].to_numpy()]

    for k in ("realized_vol", "drawdown_depth", "cs_dispersion"):
        ep[k] = np.nan
    if cov is not None:
        rows = [
            window_covariates(cov, calendar, int(r["window_start"]), WINDOW)
            if r["usable"] else dict(realized_vol=np.nan, drawdown_depth=np.nan, cs_dispersion=np.nan)
            for _, r in ep.iterrows()
        ]
        for k in ("realized_vol", "drawdown_depth", "cs_dispersion"):
            ep[k] = [r[k] for r in rows]

    ep["event_id"] = [f"E{d:%Y%m%d}" for d in ep["date"]]

    # Coder description at the anchor date. Display metadata only.
    desc = load_bbds_descriptions(bbds_path)
    ep["event_description"] = [desc.get(d, "") for d in ep["date"]]

    return ep, n_jumps, jumps["date"].max()


# =============================================================================
# TEST 2 ENGINE
# =============================================================================


class Test2:
    """
    Shared state: trading calendar, control baseline, event windows, placebo
    machinery. LLM alphas are then pushed through it one at a time.
    """

    def __init__(self, calendar, ctrl_panel, episodes, cov):
        self.calendar = calendar
        self.T = len(calendar)
        self.n_windows = self.T - WINDOW + 1
        self.episodes = episodes

        ctrl = ctrl_panel.reindex(calendar).to_numpy(dtype=np.float64)
        finite = np.isfinite(ctrl)

        # CtrlMean(t): equal-weighted mean daily Rank IC across the retained
        # control corpus. No sign orientation is applied (spec).
        total = np.nansum(np.where(finite, ctrl, 0.0), axis=1)
        count = finite.sum(axis=1).astype(np.float64)
        self.ctrl_mean = np.full(self.T, np.nan)
        ok = count >= CTRL_MIN_ALPHAS_PER_DATE
        self.ctrl_mean[ok] = total[ok] / count[ok]

        usable = episodes["usable"].to_numpy(dtype=bool)
        self.high = episodes.loc[usable & (episodes["salience"] == "high_salience")].copy()
        self.low = episodes.loc[usable & (episodes["salience"] == "low_salience")].copy()
        self.event_starts = episodes.loc[usable, "window_start"].to_numpy(dtype=int)

        self._build_placebo_mask()
        self._build_step1_distributions(ctrl, finite)
        self._build_matching()

    def _build_placebo_mask(self):
        """
        Step 2 placebos are the sliding windows OUTSIDE the event set, so any
        window overlapping a declustered episode window is excluded.
        """
        blocked = np.zeros(self.T, dtype=bool)
        for s in self.event_starts:
            blocked[max(0, s): s + WINDOW] = True
        cum = np.concatenate(([0], np.cumsum(blocked.astype(int))))
        w = np.arange(self.n_windows)
        self.placebo_allowed = (cum[w + WINDOW] - cum[w]) == 0

    def _build_step1_distributions(self, ctrl, finite):
        """
        Dual Universal Placebo Distributions: slide a 21-day window across all
        in-sample dates for the robust control corpus.

            pooled    window means pooled across every control and every window
                      - how good are the best controls across all time
            demeaned  each control's own mean IC subtracted first - window-level
                      unusualness irrespective of baseline mean

        Unlike Step 2, this sweep spans ALL dates including event windows: Step 1
        asks how good the controls get anywhere in history, so removing the
        crises would remove exactly the windows in question.
        """
        n_ctrl = ctrl.shape[1]
        wins = np.full((n_ctrl, self.n_windows), np.nan)
        means = np.full(n_ctrl, np.nan)
        for k in range(n_ctrl):
            valid = finite[:, k]
            if not valid.any():
                continue
            wins[k], _ = rolling_window_mean(ctrl[:, k], valid, WINDOW, MIN_WINDOW_OBS)
            means[k] = float(np.nanmean(ctrl[:, k][valid]))

        pooled = wins.ravel()
        demeaned = (wins - means[:, None]).ravel()
        self.pooled_vals = pooled[np.isfinite(pooled)]
        self.demeaned_vals = demeaned[np.isfinite(demeaned)]
        self.pooled_thr = float(np.percentile(self.pooled_vals, STEP1_PERCENTILE))
        self.demeaned_thr = float(np.percentile(self.demeaned_vals, STEP1_PERCENTILE))

    def _build_matching(self):
        """
        Mahalanobis geometry for Step 3. The covariance is estimated over every
        declustered episode, not the two arms separately, so the caliper means
        the same thing on both sides of the comparison.
        """
        cols = ["realized_vol", "drawdown_depth", "cs_dispersion"]
        X = self.episodes.loc[self.episodes["usable"], cols].to_numpy(dtype=np.float64)
        ok = np.isfinite(X).all(axis=1)
        if ok.sum() < len(cols) + 1:
            self.matching = None
            return

        cov = np.cov(X[ok], rowvar=False)
        hi = self.high[cols].to_numpy(dtype=np.float64)
        lo = self.low[cols].to_numpy(dtype=np.float64)
        hi_ok, lo_ok = np.isfinite(hi).all(axis=1), np.isfinite(lo).all(axis=1)

        dist = np.full((len(hi), len(lo)), np.nan)
        if hi_ok.any() and lo_ok.any():
            dist[np.ix_(hi_ok, lo_ok)] = mahalanobis_matrix(hi[hi_ok], lo[lo_ok], cov)
        self.matching = dict(distance=dist,
                             high_ids=self.high["event_id"].tolist(),
                             low_ids=self.low["event_id"].tolist())

    # -- per-alpha ---------------------------------------------------------
    def evaluate(self, alpha_id, ic_series, rng):
        """Steps 1-3 across every high-salience event, then BH within the alpha."""
        y = ic_series.reindex(self.calendar).to_numpy(dtype=np.float64)
        x = self.ctrl_mean
        valid_raw = np.isfinite(y)
        valid_paired = valid_raw & np.isfinite(x)

        raw_means, raw_n = rolling_window_mean(y, valid_raw, WINDOW, MIN_WINDOW_OBS)
        alpha_mean = float(np.nanmean(y[valid_raw])) if valid_raw.any() else np.nan

        d_bar, beta, n_pair = window_differentials(y, x, valid_paired, WINDOW, MIN_WINDOW_OBS)
        placebo = d_bar[self.placebo_allowed & np.isfinite(d_bar)]

        # low-salience side, computed once per alpha
        low_dbar, low_daily = {}, {}
        for eid, s in zip(self.low["event_id"], self.low["window_start"].to_numpy(dtype=int)):
            if 0 <= s < self.n_windows and np.isfinite(d_bar[s]):
                low_dbar[eid] = float(d_bar[s])
                low_daily[eid] = daily_differentials(y, x, valid_paired, s, WINDOW, beta[s])

        rows = []
        for _, ev in self.high.iterrows():
            s = int(ev["window_start"])
            usable = 0 <= s < self.n_windows

            # ---- Step 1: Absolute Anomaly Screen (unpaired) ----------------
            ic_raw = float(raw_means[s]) if usable else np.nan
            ic_dm = ic_raw - alpha_mean
            a_pooled = bool(np.isfinite(ic_raw) and ic_raw > self.pooled_thr)
            a_dm = bool(np.isfinite(ic_dm) and ic_dm > self.demeaned_thr)

            # ---- Step 2: Control Differential Check (paired) ---------------
            d_event = float(d_bar[s]) if usable else np.nan
            if np.isfinite(d_event) and placebo.size:
                p2 = float(np.mean(placebo >= d_event))
            else:
                p2 = np.nan

            row = dict(
                alpha_id=alpha_id,
                event_id=ev["event_id"],
                event_date=ev["date"].date().isoformat(),
                event_description=ev.get("event_description", ""),
                bbds_category=ev.get("bbds_category", ""),
                event_return=float(ev["return"]),
                clarity=float(ev["clarity"]),
                journalist_confidence=float(ev["JournalistConfidence"]),
                narrative_consensus=float(ev["narrative_consensus"]),
                window_first_date=str(pd.Timestamp(ev["window_first_date"]).date()),
                window_last_date=str(pd.Timestamp(ev["window_last_date"]).date()),
                n_valid_days=float(n_pair[s]) if usable else np.nan,
                step1_ic_raw_event=ic_raw,
                step1_ic_demeaned_event=ic_dm,
                step1_pooled_threshold=self.pooled_thr,
                step1_demeaned_threshold=self.demeaned_thr,
                step1_anomaly_pooled=a_pooled,
                step1_anomaly_demeaned=a_dm,
                step1_variants_disagree=bool(a_pooled != a_dm),
                step1_anomaly=bool(a_pooled or a_dm),
                step2_d_bar_event=d_event,
                step2_beta_lwo=float(beta[s]) if usable else np.nan,
                step2_p_placebo=p2,
                n_placebo_windows=int(placebo.size),
            )
            row.update(self._step3(ev, s, d_event, beta, y, x, valid_paired,
                                   low_dbar, low_daily, rng))
            rows.append(row)

        out = pd.DataFrame(rows)
        return self._fdr_and_verdicts(out) if not out.empty else out

    def _step3(self, ev, s, d_event, beta, y, x, valid_paired, low_dbar, low_daily, rng):
        """Mahalanobis caliper, Scenario routing, Salience Gap Test."""
        res = dict(scenario="A", n_matched_low=0, matched_low_ids="",
                   nearest_low_distance=np.nan, step3_d_bar_low_composite=np.nan,
                   step3_salience_gap=np.nan, step3_p_permutation=np.nan,
                   step3_min_detectable_gap=np.nan, step3_gap_over_mdg=np.nan)

        if self.matching is None or ev["event_id"] not in self.matching["high_ids"]:
            return res

        dist_row = self.matching["distance"][self.matching["high_ids"].index(ev["event_id"])]
        finite = np.isfinite(dist_row)
        if finite.any():
            res["nearest_low_distance"] = float(np.min(dist_row[finite]))

        within = np.where(finite & (dist_row <= MAHALANOBIS_CALIPER))[0]
        eligible = [(self.matching["low_ids"][k], float(dist_row[k])) for k in within
                    if self.matching["low_ids"][k] in low_dbar
                    and low_daily[self.matching["low_ids"][k]].size >= MIN_WINDOW_OBS]

        # Scenario A: triggered mechanically if no low-salience event is inside
        # the caliper. Fall through with the Step 2 differential only.
        if not eligible or not np.isfinite(d_event):
            return res

        # Scenario B
        ids = [e[0] for e in eligible]
        res.update(scenario="B", n_matched_low=len(ids), matched_low_ids="|".join(ids))

        w = np.full(len(ids), 1.0 / len(ids))
        composite = float(np.dot(w, [low_dbar[i] for i in ids]))
        res["step3_d_bar_low_composite"] = composite
        res["step3_salience_gap"] = d_event - composite

        d_high = daily_differentials(y, x, valid_paired, s, WINDOW, beta[s])
        d_lows = [low_daily[i] for i in ids]
        p3, mdg = self._permutation(d_high, d_lows, w, rng)
        res["step3_p_permutation"] = p3
        res["step3_min_detectable_gap"] = mdg
        if np.isfinite(mdg) and mdg > 0:
            res["step3_gap_over_mdg"] = res["step3_salience_gap"] / mdg
        return res

    @staticmethod
    def _permutation(d_high, d_lows, w, rng):
        """
        Salience Gap Test: is the contrast d_high - d_low,composite significantly
        positive? Days are pooled across the high window and every matched low
        window and randomly reassigned to slots of the original sizes, so the
        permuted statistic is the same functional as the observed one.

        Returns (p_value, minimum_detectable_gap).

        The minimum detectable gap is the (1 - ALPHA_LEVEL) quantile of the
        permutation null: the smallest salience gap this event could have
        produced and still returned p <= ALPHA_LEVEL. It is a property of the
        matched design alone - the window length, the number of matched twins
        and the day-level dispersion of the differentials - and does not depend
        on how the alpha actually performed. It is what makes a null result
        quantitative: a gap of zero against a detection floor of 0.002 is a
        tight null, the same gap against a floor of 0.05 is an underpowered one.
        """
        sizes = [len(d_high)] + [len(a) for a in d_lows]
        if min(sizes) == 0:
            return np.nan, np.nan

        observed = d_high.mean() - float(np.dot(w, [a.mean() for a in d_lows]))
        pool = np.concatenate([d_high] + d_lows)
        bounds = np.cumsum([0] + sizes)
        permuted = rng.permuted(np.tile(pool, (N_PERMUTATIONS, 1)), axis=1)
        slot_means = np.column_stack([
            permuted[:, bounds[k]: bounds[k + 1]].mean(axis=1) for k in range(len(sizes))
        ])
        null = slot_means[:, 0] - slot_means[:, 1:] @ w
        p = float(np.mean(null >= observed))
        mdg = float(np.quantile(null, 1.0 - ALPHA_LEVEL))
        return p, mdg

    @staticmethod
    def _fdr_and_verdicts(df):
        """
        Benjamini-Hochberg within the alpha, then the unified verdict rule.

        An event fails if (Step 1 AND Step 2) OR Step 3. The rule is identical in
        both scenarios; Scenario A is simply the case where Step 3 cannot be
        computed, so the Step 3 term is always False there. Evidential standards
        therefore no longer depend on whether the market happened to supply a
        structural twin.
        """
        r2, q2 = benjamini_hochberg(df["step2_p_placebo"].to_numpy(dtype=np.float64), FDR_ALPHA)
        r3, q3 = benjamini_hochberg(df["step3_p_permutation"].to_numpy(dtype=np.float64), FDR_ALPHA)
        df["step2_q_bh"], df["step3_q_bh"] = q2, q3

        # One-sided: only a POSITIVE differential is evidence of memorisation.
        df["step2_sig_positive"] = r2 & (df["step2_d_bar_event"].to_numpy() > 0)
        df["step3_sig_positive"] = r3 & (df["step3_salience_gap"].to_numpy() > 0)

        verdicts, binding = [], []
        for _, r in df.iterrows():
            if not np.isfinite(r["step2_d_bar_event"]) or r["n_placebo_windows"] == 0:
                verdicts.append("INDETERMINATE")
                binding.append("insufficient_data")
                continue

            # Unified rule across both scenarios: (Step 1 AND Step 2) OR Step 3.
            #
            # The conjunction is the anomaly arm. Step 1 (unpaired, against the
            # human control window distribution) and Step 2 (paired, against the
            # alpha's own placebo windows) are distinct statistics with distinct
            # nulls, but both compare an event window to a NON-event baseline, so
            # both detect crisis sensitivity rather than narrative salience.
            # Requiring them jointly raises the bar on that non-identifying
            # evidence; it does not make it identifying.
            #
            # Step 3 is the identifying arm. It is the only comparison that holds
            # structural severity fixed and varies only fame, so it stands alone.
            # In Scenario A it is unavailable, and the rule degenerates to the
            # conjunction - which is why Scenario A failures are anomalies of
            # unattributable cause, not demonstrated memorisation.
            anom = bool(r["step1_anomaly"] and r["step2_sig_positive"])
            gap = bool(r["step3_sig_positive"])  # always False in Scenario A

            verdicts.append("FAIL" if (anom or gap) else "PASS")
            binding.append(
                "anomaly_and_salience_gap" if anom and gap else
                "salience_gap" if gap else
                "step1_anomaly_and_control_differential" if anom else "none"
            )

        df["event_verdict"] = verdicts
        df["binding_failure_mode"] = binding
        return df


def synthesise(event_results):
    """
    One row per alpha. Every column is a count of high-salience events, so all
    of them are read against `events_tested`.

        verdict                      PASS / FAIL / INDETERMINATE
        events_tested                high-salience events with a usable statistic
        events_failed                events failing the Scenario A or B rule
        step1_anomaly_flags          raw IC above the control placebo 95th pct
        step2_differential_flags     paired differential significantly positive
        step3_salience_gap_flags     salience gap significantly positive
        binding_failure_mode         which rule actually bound, over failed events

    A Test 2 PASS is the absence of affirmative evidence of event-level
    memorisation. It is not certification, which is a joint claim across Gate 0
    and every mechanism test.
    """
    out = []
    for alpha_id, g in event_results.groupby("alpha_id", sort=False):
        ev = g.loc[g["event_verdict"] != "INDETERMINATE"]
        n_fail = int((ev["event_verdict"] == "FAIL").sum())
        modes = ev.loc[ev["event_verdict"] == "FAIL", "binding_failure_mode"]

        out.append(dict(
            alpha_id=alpha_id,
            verdict=("INDETERMINATE" if len(ev) == 0 else "FAIL" if n_fail else "PASS"),
            events_tested=len(ev),
            events_failed=n_fail,
            step1_anomaly_flags=int(ev["step1_anomaly"].sum()),
            step2_differential_flags=int(ev["step2_sig_positive"].sum()),
            step3_salience_gap_flags=int(ev["step3_sig_positive"].sum()),
            step3_median_mdg=float(ev["step3_min_detectable_gap"].median(skipna=True)),
            step3_max_gap_over_mdg=float(ev["step3_gap_over_mdg"].max(skipna=True))
                if ev["step3_gap_over_mdg"].notna().any() else np.nan,
            binding_failure_mode=", ".join(sorted(set(modes))) if n_fail else "none",
        ))
    return pd.DataFrame(out)


# =============================================================================
# DRIVER
# =============================================================================


def main(argv=None):
    ap = argparse.ArgumentParser(description="Test 2 - Historical Event Memorisation")
    ap.add_argument("--llm-ic", default=None)
    ap.add_argument("--control-ic", default=None)
    ap.add_argument("--llm-results", default=None)
    ap.add_argument("--control-results", default=None)
    ap.add_argument("--bbds", default=BBDS_PATH)
    ap.add_argument("--parquet", default=PARQUET_PATH)
    ap.add_argument("--outdir", default=OUTPUT_DIR)
    ap.add_argument("--llm-tag", default=LLM_MODEL_TAG)
    ap.add_argument("--control-tag", default=CONTROL_MODEL_TAG)
    ap.add_argument("--all-alphas", action="store_true",
                    help="Evaluate every LLM alpha, not just Phase 1 survivors.")
    args = ap.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    # ---- inputs ----------------------------------------------------------
    llm_panel = load_ic_panel(args.llm_tag, args.llm_ic)
    ctrl_panel = load_ic_panel(args.control_tag, args.control_ic)

    ctrl_survivors = load_survivor_ids(args.control_tag, args.control_results)
    if ctrl_survivors:
        ctrl_panel = ctrl_panel[[c for c in ctrl_panel.columns if c in ctrl_survivors]]
    else:
        warnings.warn("No control screening results found; using the full control "
                      "panel as the retained corpus.", RuntimeWarning)
    if ctrl_panel.shape[1] < CTRL_MIN_ALPHAS_PER_DATE:
        raise ValueError(f"Only {ctrl_panel.shape[1]} control alphas retained; "
                         f"CtrlMean needs at least {CTRL_MIN_ALPHAS_PER_DATE}.")

    llm_survivors = load_survivor_ids(args.llm_tag, args.llm_results)
    subjects = list(llm_panel.columns)
    if llm_survivors and not args.all_alphas:
        subjects = [c for c in subjects if c in llm_survivors]

    calendar, cal_src = load_trading_calendar(args.parquet, llm_panel.index.union(ctrl_panel.index))
    cov = build_market_covariates(args.parquet, os.path.join(args.outdir, "_market_covariates.csv"))

    # ---- events ----------------------------------------------------------
    episodes, n_jumps, last_jump = build_events(args.bbds, calendar, cov)
    episodes.to_csv(os.path.join(args.outdir, "test2_events.csv"), index=False)

    engine = Test2(calendar, ctrl_panel, episodes, cov)
    if engine.high.empty:
        raise ValueError("No usable high-salience events; Test 2 cannot run.")

    n_high = len(engine.high)
    print("=" * 78)
    print("TEST 2: HISTORICAL EVENT MEMORISATION")
    print("=" * 78)

    # ---- 1. events -------------------------------------------------------
    print(f"\nEVENTS\n  {n_jumps} BBDS jumps in {ANCHOR_START[:4]}-{ANCHOR_END[:4]}"
          f"  ->  {len(episodes)} shock episodes after declustering")
    print(f"  {n_high} high-salience (treatment), {len(engine.low)} low-salience "
          f"(matching pool), {len(episodes) - n_high - len(engine.low)} middle quartiles unused")
    if last_jump < pd.Timestamp(ANCHOR_END) - pd.Timedelta(days=365):
        print(f"  BBDS coding stops {last_jump.date()}; Test 2 is silent on anything after that date.")

    # matches = low-salience events inside the caliper, which is what routes an
    # event to Scenario A (rely on the control differential) or B (also test the
    # salience gap against the matched composite). Descriptions are BBDS coders'
    # own; windows are the anchor day +/- 10 sessions and are in test2_events.csv.
    print(f"\n  {'event date':<13}{'return':>8}{'consensus':>11}{'matches':>9}   description")
    for _, e in engine.high.sort_values("date").iterrows():
        if engine.matching is None:
            n_m = "-"
        else:
            row = engine.matching["distance"][engine.matching["high_ids"].index(e["event_id"])]
            n_m = int(np.sum(np.isfinite(row) & (row <= MAHALANOBIS_CALIPER)))

        label = str(e.get("event_description", "") or "")
        cat = str(e.get("bbds_category", "") or "")
        if len(label) > 58:
            label = label[:55].rstrip() + "..."
        if not label:
            label = f"[no BBDS description]"
        if cat:
            label = f"{label}  ({cat})"

        print(f"  {str(e['date'].date()):<13}{e['return']:>+8.2%}"
              f"{e['narrative_consensus']:>11.2f}{str(n_m):>9}   {label}")

    # ---- 2. alphas -------------------------------------------------------
    print(f"\nALPHAS\n  {len(subjects)} LLM alphas"
          f"{'' if args.all_alphas or not llm_survivors else ' (Phase 1 survivors)'}"
          f" against a control baseline of {ctrl_panel.shape[1]} alphas, "
          f"{int(engine.placebo_allowed.sum())} placebo windows.")
    print(f"  Counts are high-salience events flagged, out of {n_high}. An alpha fails only")
    print("  where (Step 1 AND Step 2) OR Step 3 binds, so flags alone are not failures.")
    print("  Step 3 is unavailable in Scenario A, so those failures bind on the")
    print("  conjunction alone and are anomalies of unattributable cause.\n")
    print(f"  {'alpha':<30}{'Step 1':>9}{'Step 2':>9}{'Step 3':>9}   verdict")
    print(f"  {'':<30}{'anomaly':>9}{'differ.':>9}{'sal. gap':>9}")

    rng = np.random.default_rng(PERMUTATION_SEED)
    frames = []
    for aid in subjects:
        res = engine.evaluate(aid, llm_panel[aid], rng)
        if res.empty:
            continue
        frames.append(res)
        v = "FAIL" if (res["event_verdict"] == "FAIL").any() else "PASS"
        print(f"  {aid:<30}"
              f"{f'{int(res.step1_anomaly.sum())}/{n_high}':>9}"
              f"{f'{int(res.step2_sig_positive.sum())}/{n_high}':>9}"
              f"{f'{int(res.step3_sig_positive.sum())}/{n_high}':>9}   {v}")

    if not frames:
        raise ValueError("No alpha produced an evaluable event.")

    event_results = pd.concat(frames, ignore_index=True)
    verdicts = synthesise(event_results)
    event_results.to_csv(os.path.join(args.outdir, "test2_event_results.csv"), index=False)
    verdicts.to_csv(os.path.join(args.outdir, "test2_alpha_verdicts.csv"), index=False)

    # ---- 3. results ------------------------------------------------------
    n_pass = int((verdicts["verdict"] == "PASS").sum())
    n_fail = int((verdicts["verdict"] == "FAIL").sum())
    ev = event_results.loc[event_results["event_verdict"] != "INDETERMINATE"]

    print(f"\nRESULTS\n  {n_pass} of {len(verdicts)} alphas show no evidence of event-level "
          f"memorisation; {n_fail} flagged.")
    print(f"  {len(ev)} alpha-event tests: {int((ev['scenario'] == 'A').sum())} Scenario A "
          f"(no low-salience match in caliper), {int((ev['scenario'] == 'B').sum())} Scenario B.")

    modes = ev.loc[ev["event_verdict"] == "FAIL", "binding_failure_mode"].value_counts()
    print("  Binding failure modes: "
          + (", ".join(f"{k} ({v})" for k, v in modes.items()) if len(modes) else "none"))
    print(f"  Step 1 pooled vs demeaned verdicts disagree on "
          f"{int(ev['step1_variants_disagree'].sum())} of {len(ev)} tests.")

    b = ev.loc[(ev["scenario"] == "B") & ev["step3_min_detectable_gap"].notna()]
    if not b.empty:
        mdg = b["step3_min_detectable_gap"]
        ratio = b["step3_gap_over_mdg"]
        print(f"\n  Step 3 detection floor over {len(b)} Scenario B tests: median "
              f"{mdg.median():.5f} Rank IC, range {mdg.min():.5f} to {mdg.max():.5f}.")
        print(f"  Largest observed gap reaches {ratio.max():.2f}x its own floor; "
              f"{int((ratio >= 1.0).sum())} of {len(b)} tests reach it.")
        print("  A null salience gap is only as strong as this floor is low.")

        by_ev = (b.groupby("event_date")
                  .agg(matches=("n_matched_low", "first"),
                       mdg=("step3_min_detectable_gap", "median"))
                  .sort_values("mdg", ascending=False))
        print(f"\n  {'event date':<13}{'matches':>9}{'floor':>10}   least to most powered")
        for d, r in by_ev.head(3).iterrows():
            print(f"  {d:<13}{int(r['matches']):>9}{r['mdg']:>10.5f}")
        print(f"  {'...':<13}")
        for d, r in by_ev.tail(3).iterrows():
            print(f"  {d:<13}{int(r['matches']):>9}{r['mdg']:>10.5f}")
    print("  A pass is the absence of evidence of event memorisation, not certification.")

    print("\n  Unspecified parameters (not fixed by the spec):")
    for key, val, note in UNSPECIFIED:
        print(f"    {key} = {val}   [{note}]")

    print(f"\n  Written to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
