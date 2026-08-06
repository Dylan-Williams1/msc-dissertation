"""
INSTRUMENT 1 - SHORT-WINDOW VARIANT
===================================
Identical to instrument1.py except for five power-directed changes. Each
reduces required T_eff without weakening the standard; none touches r, which
remains measured from control data. See CHANGES_shortwindow.md.

  1. Multi-factor PC pairing, applied IDENTICALLY TO BOTH ARMS.
     Estimator improvement: the same quantity, estimated more precisely.
  2. NON-INFERIORITY available but DEFAULT OFF. This is a change to the
     STANDARD, not the estimator, and section 7 locks k = 2.927. Enabling it
     is a methodological decision requiring its own justification.
  3. T_eff of the differential reported against the raw IC (diagnostic only).

REMOVED after review, and why:
  - GLS pooling. Var(pooled) already contains the pooling gain, so dividing
    the pooled MDE by sqrt(T_eff * N_eff_gls) applied it twice. Beyond the
    bug, minimum-variance weights go negative, making the claim "a long-short
    book of alphas did not degrade" rather than section 73's panel mean. All
    three fallbacks also returned N_eff = n, asserting perfect independence
    precisely where the covariance was least estimable.
  - Pre-registered primary subset. It has NO effect on required T_eff (the
    MDE gate and TOST are per-alpha), and ranking the family by IS Rank IC
    ranks it by delta = r * IS_IC, so the widest-margin alphas were the only
    ones receiving FWER protection - which makes breaks HARDER to find for
    exactly the alphas best placed to certify. No upside, real bias.

BUG FIXES (both are SIZE corrections, not relaxations of the standard):
  A. block_bootstrap_cv returned the conf quantile of |t*| - a TWO-SIDED
     critical value - while tost() compares it against one-sided statistics
     on both tails. Every one-sided test was therefore running at alpha=0.025
     rather than 0.05. Now returns the signed (1-conf, conf) quantiles, which
     reduce to (-1.645, +1.645) as dependence vanishes. Recovers ~14% of the
     critical value, ~26% of the certifying margin, at this window length.
  B. The MDE gate used sigma_up / sqrt(T_eff(post)) while the test
     studentises by sqrt(nw_var_mean(pre) + nw_var_mean(post)). The tested
     statistic is a DIFFERENCE OF MEANS, so the correct T is the harmonic
     combination T_paired = (1/T_pre + 1/T_post)^-1 <= T_post. The old form
     understated the SE and therefore the MDE, letting the gate certify power
     the test could not deliver. Fix moves the gate the CONSERVATIVE way
     (+6% MDE here) and makes T_paired the frontier's T everywhere. It also
     makes explicit that lengthening the IS window is nearly worthless once
     T_pre >> T_post: the short arm binds.

NOTE ON FRAMING: k, sigma and N_eff sit in the same place as r in
IC_IS >= k*sigma/(r*sqrt(T_eff*N_eff)), so "does not touch r" is not a
defence. The test applied here is whether a change improves the ESTIMATOR
(same quantity, more precisely) or relaxes the STANDARD (what counts as
passing). Only the former is adopted by default.

Point ALPHA_DIR at a model's alpha folder and run.

    python instrument1.py
    python instrument1.py --alpha-dir ../alphas/survived/gemini-3 --cutoff 2026-03-01

Tested series is the OBSERVATION-LEVEL PAIRED DIFFERENTIAL:

    d_i(t) = IC_i(t) - beta_i * CtrlMean(t)

beta_i estimated on in-sample dates only. Regime effects, McLean-Pontiff decay
and survivorship artifacts hit both arms on the same dates and cancel by
construction, so the residual is attributable to the LLM.

Pipeline
    1. daily Rank IC for every treatment and control alpha
    2. CtrlMean(t); beta-adjusted differentials (IS-estimated beta)
    3. r estimated leave-one-out from the SPREAD of control decays
    4. Method A - Magnitude + Proximity, length-matched placebos, Romano-Wolf
    5. Method B - TOST on the differential, MDE gate, certification frontier
    6. reconciliation -> per-alpha verdict; pooled model-level claim

Nothing here chooses a parameter to obtain a verdict. r, sigma, rho_bar and
T_eff are measured; the frontier is the result.
"""

import argparse
import json
import os
import sys
import glob
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================
# CONFIG
# =========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

ALPHA_DIR = os.path.join(ROOT_DIR, "alphas", "survived", "gemini-3.6-flash")
CONTROL_DIR = os.path.join(ROOT_DIR, "alphas", "survived", "kakushadze-101-v1")
DATA_PATH = os.path.join(ROOT_DIR, "data", "daily_ohlcv.parquet")
OUT_DIR = os.path.join(SCRIPT_DIR, "instrument1_output")
CACHE_DIR = os.path.join(SCRIPT_DIR, ".ic_cache")

# Spec line 27: Phase 1 already exported daily Rank IC per alpha and it is
# "not regenerated in Phase 2". Recomputing risks a different IC series than
# the one the DSR screen ran on. "phase1" reads that CSV; "recompute" executes
# the alpha JSONs and is an explicit opt-in for when the CSV is absent.
IC_SOURCE = "phase1"
PHASE1_IC_DIR = r"C:\University\Master's\Diss\Dissertation\evaluation"

# gemini-3.6-flash. Vendor cutoff is approximate; section 2 requires an
# INTERVAL, and this is the point date the interval is centred on.
CUTOFF_DATE = "2026-01-01"
CUTOFF_UNCERTAINTY_DAYS = 30      # placebo buffer AND Proximity tolerance

# Forward return convention. Signal formed at close of t, rebalanced at the
# open of t+1, so the realisable 1-day return runs open(t+1) -> open(t+2).
# "close_to_close" is available for comparability with older runs.
RETURN_CONVENTION = "open_to_open"

IC_MIN_NAMES = 400                # raised from 20; see methodology
MIN_SIDE_OBS = 30
TRIM = 0.15                       # Andrews (1993)
# NON-INFERIORITY, one-sided: k = z(.95) + z(.80) = 2.4865.
# Two-sided TOST needs BOTH one-sided tests to reject, giving z(.95)+z(.90)
# = 2.927 at a true shift of zero. Certification asks whether the alpha
# DEGRADED, not whether it changed, so the upper tail is not of interest and
# the second test is not required. Dropping it cuts required T_eff by
# (2.4865/2.927)^2 = 0.72, i.e. 28% fewer effective observations.
# Section 7 locks k = 2.927 (two-sided TOST at 80% power). Non-inferiority is
# available because certification asks whether the alpha DEGRADED rather than
# whether it changed, but it is a change to the inferential standard and is
# therefore OFF by default. Turning it on requires justifying it in the
# methodology, not a config edit.
TEST_MODE = "tost"                # "tost" (locked default) | "noninferiority"
K_MDE = 2.927 if TEST_MODE == "tost" else 2.4865

# Number of control principal components used as the pairing basis. Loadings
# are fitted on IS dates only and held fixed thereafter.
N_PCS = 3


ALPHA_LEVEL = 0.05

# ---- THE ONE OPEN DESIGN DECISION -------------------------------------
# "symmetric"  : [d-L, d+L]. Tested date sits at the midpoint (50% post-date
#                share), so sup-Wald is applicable and PROXIMITY EXISTS.
#                Costs ~39% on the standard error versus using full pre-history.
# "asymmetric" : full available pre-history + L days after. Better power, but
#                the post-date share is tiny at every date, sup-Wald is
#                trim-blocked, and PROXIMITY IS NOT COMPUTED (Magnitude only).
# Spec line 55 specifies symmetric placebos and line 54 requires both
# statistics, so symmetric is the default.
WINDOW_MODE = "symmetric"

PLACEBO_START = "2016-01-01"
PLACEBO_FREQ = "MS"
SUPWALD_STEP = 1                  # raise to 2 if the break search is slow
N_BOOTSTRAP = 2000
RNG_SEED = 0

# Below this T_eff the standard normal critical value is unreliable for HAC
# inference (finite-sample size distortion), so critical values come from a
# moving-block bootstrap instead. Spec line 66.
FIXED_B_THRESHOLD = 100
BOOT_BLOCKS = 999

# r fallback if the control decay spread cannot be estimated. Flagged loudly
# in the output as a literature import with the horizon caveat attached.
R_FALLBACK = 0.26

# An alpha with a near-zero IS IC gives a meaningless decay RATIO (shift/IS_IC
# explodes). Such alphas are excluded from the r estimate. They are still
# tested - they simply cannot inform the margin.
MIN_IS_IC_FOR_R = 0.005
MAX_PLAUSIBLE_R = 1.0


# =========================================================
# 1. DATA AND ALPHA EXECUTION
# =========================================================

def load_panel(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"OHLCV panel not found at {path}")
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.MultiIndex):
        for a, b in [("date", "symbol"), ("Date", "Symbol")]:
            if a in df.columns and b in df.columns:
                df = df.set_index([a, b])
                break
    df.index.names = ["date", "symbol"]
    df = df.sort_index()
    need = {"open", "high", "low", "close", "volume"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")
    return df


def forward_returns(panel, convention=RETURN_CONVENTION):
    """1-day forward return aligned to the signal date t."""
    if convention == "open_to_open":
        px = panel["open"].unstack("symbol").sort_index()
        fwd = px.shift(-2) / px.shift(-1) - 1.0
    elif convention == "close_to_close":
        px = panel["close"].unstack("symbol").sort_index()
        fwd = px.shift(-1) / px - 1.0
    else:
        raise ValueError(f"unknown RETURN_CONVENTION {convention!r}")
    return fwd.replace([np.inf, -np.inf], np.nan)


def extract_code(record):
    raw = record.get("raw_response", "")
    if "```python" in raw:
        return raw.split("```python", 1)[1].split("```", 1)[0]
    if "```" in raw:
        return raw.split("```", 1)[1].rsplit("```", 1)[0]
    return raw


def run_alpha(code, panel):
    """
    Mirrors the phase 1 harness: exec(code, globals(), local_ns) puts the defs
    in local_ns while giving them the harness globals as __globals__, which is
    why the operator library is nested inside generate_alpha.
    """
    ns = {}
    exec(compile(code, "<alpha>", "exec"), globals(), ns)
    fn = ns.get("generate_alpha")
    if fn is None:
        raise RuntimeError("generate_alpha not defined")
    return fn(panel)


def rank_ic_series(signal, fwd, min_names=IC_MIN_NAMES):
    """Daily cross-sectional Spearman IC of signal(t) against fwd return(t)."""
    if isinstance(signal, pd.Series):
        sig = signal.unstack("symbol")
    else:
        sig = signal
    sig, f = sig.align(fwd, join="inner")
    valid = sig.notna() & f.notna()
    n = valid.sum(axis=1)
    sr = sig.where(valid).rank(axis=1)
    fr = f.where(valid).rank(axis=1)
    sr = sr.sub(sr.mean(axis=1), axis=0)
    fr = fr.sub(fr.mean(axis=1), axis=0)
    num = (sr * fr).sum(axis=1)
    den = np.sqrt((sr ** 2).sum(axis=1) * (fr ** 2).sum(axis=1))
    ic = (num / den.replace(0.0, np.nan))
    return ic.where(n >= min_names)


def load_phase1_ic(model_key):
    """Read the Rank IC panel Phase 1 already exported. Spec line 27."""
    names = [f"phase1_daily_rank_ic_{model_key}.csv"]
    roots = [PHASE1_IC_DIR, SCRIPT_DIR, ROOT_DIR, os.getcwd()]
    cands = [os.path.join(r, n) for r in roots for n in names]
    cands += [h for n in names
              for h in glob.glob(os.path.join(ROOT_DIR, "**", n), recursive=True)]
    for c in cands:
        if os.path.exists(c):
            df = pd.read_csv(c, index_col=0)
            df.index = pd.to_datetime(df.index)
            df.index.name = "date"
            df = df.sort_index().apply(pd.to_numeric, errors="coerce")
            return df.dropna(axis=1, how="all"), c
    return None, None


def build_ic_panel(alpha_dir, panel, fwd, label, use_cache=True):
    """Wide date x alpha_id panel of daily Rank IC."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    tag = os.path.basename(os.path.normpath(alpha_dir))
    cache = os.path.join(CACHE_DIR, f"ic_{tag}_{RETURN_CONVENTION}_{IC_MIN_NAMES}.parquet")
    if use_cache and os.path.exists(cache):
        out = pd.read_parquet(cache)
        print(f"   [{label}] {out.shape[1]} alphas from cache")
        return out

    files = sorted(glob.glob(os.path.join(alpha_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"no alpha JSONs in {alpha_dir}")
    print(f"   [{label}] executing {len(files)} alphas from {alpha_dir}")

    cols, failed = {}, []
    for i, fp in enumerate(files, 1):
        aid = os.path.splitext(os.path.basename(fp))[0]
        try:
            rec = json.load(open(fp, encoding="utf-8"))
            aid = rec.get("metadata", {}).get("alpha_id", aid)
            sig = run_alpha(extract_code(rec), panel)
            ic = rank_ic_series(sig, fwd)
            if ic.notna().sum() >= 100:
                cols[aid] = ic
            else:
                failed.append((aid, "insufficient valid IC dates"))
        except Exception as e:
            failed.append((aid, f"{type(e).__name__}: {e}"))
        if i % 10 == 0 or i == len(files):
            print(f"      {i}/{len(files)}", end="\r")
    print()
    if failed:
        print(f"   [{label}] {len(failed)} failed:")
        for aid, why in failed[:5]:
            print(f"      {aid}: {why[:90]}")
        if len(failed) > 5:
            print(f"      ... and {len(failed)-5} more")
    if not cols:
        raise RuntimeError(f"[{label}] every alpha failed")

    out = pd.DataFrame(cols).sort_index()
    out.index.name = "date"
    out.to_parquet(cache)
    return out


# =========================================================
# 2. STATISTICS
# =========================================================

def _lrv_np(x, lag=None):
    """Newey-West long-run variance of an OBSERVATION. numpy, window-local."""
    T = x.size
    if T < 5:
        return np.nan
    if lag is None:
        lag = max(1, int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0))))
    xc = x - x.mean()
    lrv = float(xc @ xc) / T
    for k in range(1, min(lag, T - 1) + 1):
        lrv += 2.0 * (1.0 - k / (lag + 1.0)) * float(xc[k:] @ xc[:-k]) / T
    return max(lrv, 1e-14)


def nw_var_mean(x, lag=None):
    """Newey-West long-run variance OF THE MEAN, estimated window-locally."""
    x = pd.Series(x).dropna()
    T = len(x)
    if T < 5:
        return np.nan
    if lag is None:
        lag = max(1, int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0))))
    xc = (x - x.mean()).to_numpy(dtype=float)
    lrv = float((xc ** 2).sum()) / T
    for k in range(1, min(lag, T - 1) + 1):
        gk = float((xc[k:] * xc[:-k]).sum()) / T
        lrv += 2.0 * (1.0 - k / (lag + 1.0)) * gk
    return max(lrv, 1e-14) / T


def effective_T(x):
    """Autocorrelation-adjusted effective sample size."""
    x = pd.Series(x).dropna()
    T = len(x)
    if T < 20:
        return float(T)
    v_iid = float(x.var(ddof=1)) / T
    v_hac = nw_var_mean(x)
    if not np.isfinite(v_hac) or v_hac <= 0:
        return float(T)
    return float(np.clip(T * v_iid / v_hac, 1.0, T))


def paired_T_eff(series, cutoff, L):
    """
    Effective sample size OF THE TESTED STATISTIC, which is a DIFFERENCE OF
    TWO MEANS, not of the OOS window alone.

    BUG FIX - the MDE gate used sigma_up / sqrt(T_eff(post)) while tost()
    studentises by sqrt(nw_var_mean(pre) + nw_var_mean(post)). Those are
    different denominators, so the gate was not measuring the power of the
    test it gates. The statistic is mean(post) - mean(pre), with

        Var = sigma^2 * (1/T_pre + 1/T_post),

    so the T belonging under the square root in both the MDE and the frontier
    is the HARMONIC combination

        T_paired = (1/T_pre + 1/T_post)^-1  <=  T_post.

    Using T_post alone UNDERSTATES the standard error and therefore
    understates the MDE, so the gate could certify power the test cannot
    deliver. The fix moves the gate in the conservative direction, which is
    the direction section 1 requires.

    Two things this makes visible and the old form hid:
      - T_paired < T_post always, so the gate is strictly harder to pass.
      - T_paired -> T_post as T_pre grows, so lengthening the IN-SAMPLE
        window buys essentially nothing once T_pre >> T_post. The binding
        constraint is the short arm, which is exactly the recent-model case.

    Returns (T_pre_eff, T_post_eff, T_paired).
    """
    v = series.dropna()
    pre = v.loc[v.index < cutoff]
    post = v.loc[v.index >= cutoff].iloc[:L]
    if len(pre) < MIN_SIDE_OBS or len(post) < MIN_SIDE_OBS:
        return np.nan, np.nan, np.nan
    t_pre = effective_T(pre)
    t_post = effective_T(post)
    if not (np.isfinite(t_pre) and np.isfinite(t_post)) or min(t_pre, t_post) <= 0:
        return t_pre, t_post, np.nan
    return t_pre, t_post, float(1.0 / (1.0 / t_pre + 1.0 / t_post))


def window_slice(series, pos, L, mode=WINDOW_MODE):
    v = series
    if mode == "symmetric":
        lo = max(0, pos - L)
    else:
        lo = 0
    return v.iloc[lo:min(len(v), pos + L)], pos - lo


def magnitude_wald(series, pos, L, mode=WINDOW_MODE, min_side=MIN_SIDE_OBS):
    """Fixed-date Wald at the tested date. Larger |t| = more suspicious."""
    w, c = window_slice(series, pos, L, mode)
    if c < min_side or len(w) - c < min_side:
        return np.nan
    x = w.to_numpy(dtype=float)
    pre, post = x[:c], x[c:]
    vp, vq = _lrv_np(pre) / pre.size, _lrv_np(post) / post.size
    if not (np.isfinite(vp) and np.isfinite(vq)):
        return np.nan
    se = np.sqrt(vp + vq)
    return float((post.mean() - pre.mean()) / se) if se > 0 else np.nan


def supwald(series, pos, L, mode=WINDOW_MODE, trim=TRIM,
            min_side=MIN_SIDE_OBS, step=SUPWALD_STEP):
    """
    sup-Wald over the window plus the LOCATION of the maximising break.
    Returns (sup_stat, signed_offset_in_trading_days).

    Offset is signed on purpose: negative means the break sits BEFORE the
    tested date, which is the direction contamination is expected to bias it,
    since training corpora thin out for months ahead of a nominal cutoff.

    Not computed in asymmetric mode - the tested date sits inside the trim
    region there, so the break cannot be located.
    """
    if mode != "symmetric":
        return np.nan, np.nan
    w, c = window_slice(series, pos, L, mode)
    n = len(w)
    if n < 2 * min_side:
        return np.nan, np.nan
    x = w.to_numpy(dtype=float)
    lrv = _lrv_np(x)                    # ONE window-local variance, as Andrews
    if not np.isfinite(lrv):
        return np.nan, np.nan
    a = max(int(np.floor(trim * n)), min_side)
    b = min(int(np.ceil((1 - trim) * n)), n - min_side)
    if b <= a:
        return np.nan, np.nan
    cs = np.cumsum(x)
    total = cs[-1]
    i = np.arange(a, b, max(step, 1), dtype=int)
    m_pre = cs[i - 1] / i
    m_post = (total - cs[i - 1]) / (n - i)
    se = np.sqrt(lrv / i + lrv / (n - i))
    t = np.abs(m_post - m_pre) / np.where(se > 0, se, np.nan)
    if not np.isfinite(t).any():
        return np.nan, np.nan
    j = int(np.nanargmax(t))
    return float(t[j]), float(i[j] - c)


def sigma_is_upper(series, cutoff, conf=0.95):
    """
    IS-only sigma with an upper confidence bound (spec line 67).

    Full-sample sigma lets the gate see the data it is gating. The upper bound
    uses the chi-square limit for a standard deviation on nu = T_eff - 1
    degrees of freedom, T_eff rather than T because the series is autocorrelated:

        sigma_upper = s * sqrt(nu / chi2_{(1-conf), nu}).
    """
    x = series.loc[series.index < cutoff].dropna()
    if len(x) < MIN_SIDE_OBS:
        return np.nan, np.nan
    s_hat = float(x.std(ddof=1))
    nu = max(effective_T(x) - 1.0, 1.0)
    q = stats.chi2.ppf(1.0 - conf, nu)
    return s_hat, (s_hat * np.sqrt(nu / q) if q > 0 else np.nan)


def block_bootstrap_cv(pre, post, conf=0.95, n_boot=BOOT_BLOCKS, seed=RNG_SEED):
    """
    Critical values for the studentised shift from a MOVING-BLOCK BOOTSTRAP.

    Below T_eff ~ 100 the standard normal is unreliable for HAC inference
    (spec line 66). Each segment is centred on its own mean, imposing the null
    of zero shift, then resampled in blocks of length l = ceil(T^(1/3)) so the
    within-block dependence is preserved.

    Returns (cv_lo, cv_hi): the (1-conf) and conf quantiles of the SIGNED
    bootstrap statistic. As dependence vanishes these approach -1.645 and
    +1.645, reproducing the normal one-sided values exactly.

    BUG FIX - was returning the conf quantile of |t*|, i.e. a TWO-SIDED
    critical value, while tost() compares it against one-sided statistics on
    both tails. For a roughly symmetric null, quantile(|t*|, 0.95) equals
    quantile(t*, 0.975), so every one-sided test was being run at alpha=0.025
    rather than 0.05. At the effective sample sizes here that inflates the
    critical value by roughly 15-25%, and since the test is
    (diff + delta)/se > cv, the inflation is a pure power loss: alphas were
    failing non-inferiority on a threshold stricter than the one the
    methodology specifies. Returning the signed quantiles fixes the size of
    the test rather than relaxing it.

    The tails are reported separately rather than assumed symmetric: the
    differential's block bootstrap need not be, and asymmetry is itself worth
    seeing in the output.
    """
    a, b = np.asarray(pre, float), np.asarray(post, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < MIN_SIDE_OBS or b.size < MIN_SIDE_OBS:
        return np.nan, np.nan
    a, b = a - a.mean(), b - b.mean()
    rng = np.random.default_rng(seed)

    def blocks(x):
        n = x.size
        l = max(2, int(np.ceil(n ** (1.0 / 3.0))))
        k = int(np.ceil(n / l))
        starts = rng.integers(0, n - l + 1, size=k)
        return np.concatenate([x[st:st + l] for st in starts])[:n]

    ts = np.empty(n_boot)
    for i in range(n_boot):
        aa, bb = blocks(a), blocks(b)
        v = _lrv_np(aa) / aa.size + _lrv_np(bb) / bb.size
        # SIGNED, not absolute. See docstring.
        ts[i] = (bb.mean() - aa.mean()) / np.sqrt(v) if v > 0 else np.nan
    ts = ts[np.isfinite(ts)]
    if not ts.size:
        return np.nan, np.nan
    return (float(np.quantile(ts, 1.0 - conf)),
            float(np.quantile(ts, conf)))


def tost(series, cutoff, L, delta):
    """
    TOST equivalence on IS vs OOS mean of the differential.
    Equivalence established iff both one-sided tests reject at ALPHA_LEVEL.
    """
    v = series.dropna()
    pre, post = v.loc[v.index < cutoff], v.loc[v.index >= cutoff].iloc[:L]
    if len(pre) < MIN_SIDE_OBS or len(post) < MIN_SIDE_OBS or not np.isfinite(delta):
        return dict(diff=np.nan, se=np.nan, cv=np.nan, cv_lo=np.nan,
                    cv_source="none", T_paired=np.nan, equivalent=False,
                    mode=TEST_MODE, reason="insufficient data")
    vp, vq = nw_var_mean(pre), nw_var_mean(post)
    se = np.sqrt(vp + vq)
    if not np.isfinite(se) or se <= 0:
        return dict(diff=np.nan, se=np.nan, cv=np.nan, cv_lo=np.nan,
                    cv_source="none", T_paired=np.nan, equivalent=False,
                    mode=TEST_MODE, reason="degenerate SE")
    diff = float(post.mean() - pre.mean())
    # The finite-sample distortion is governed by the same T that governs the
    # SE, i.e. the paired one, not the OOS window on its own.
    _, _, T_paired = paired_T_eff(series, cutoff, L)
    z95 = 1.6448536269514722
    cv_lo, cv_hi = -z95, z95                    # one-sided normal, both tails
    cv_source = "normal"
    if np.isfinite(T_paired) and T_paired < FIXED_B_THRESHOLD:
        b_lo, b_hi = block_bootstrap_cv(pre.to_numpy(float), post.to_numpy(float))
        if np.isfinite(b_lo) and np.isfinite(b_hi):
            cv_lo, cv_hi, cv_source = b_lo, b_hi, "block-bootstrap"
    # SIGNED one-sided comparisons. Each tail uses its own critical value.
    lower_ok = (diff - (-delta)) / se > cv_hi     # not degraded by more than delta
    upper_ok = (diff - delta) / se < cv_lo        # not improved by more than delta
    if TEST_MODE == "noninferiority":
        ok = bool(lower_ok)                       # upper tail is not of interest
    else:
        ok = bool(lower_ok and upper_ok)
    return dict(diff=diff, se=float(se), cv=float(cv_hi), cv_lo=float(cv_lo),
                cv_source=cv_source, T_paired=T_paired,
                equivalent=ok, mode=TEST_MODE,
                reason="" if ok else ("degraded beyond delta"
                                      if TEST_MODE == "noninferiority"
                                      else "not equivalent"))


def mean_pairwise_corr(panel, min_overlap=60):
    c = panel.corr(min_periods=min_overlap)
    vals = c.where(~np.eye(len(c), dtype=bool)).stack().dropna()
    return (float(vals.mean()), int(len(vals))) if len(vals) else (np.nan, 0)


def effective_n(n, rho):
    if not np.isfinite(rho):
        return np.nan
    d = 1.0 + (n - 1) * max(rho, 0.0)
    return n / d if d > 0 else np.nan


def frontier_ic(sigma, r, T_eff, N_eff=1.0, k=K_MDE):
    if not all(np.isfinite([sigma, r, T_eff, N_eff])) or r <= 0:
        return np.nan
    return k * sigma / (r * np.sqrt(max(T_eff * N_eff, 1e-12)))


# =========================================================
# 3. DIFFERENTIALS AND MARGIN
# =========================================================

def control_factors(ctrl_ic, cutoff, n_pcs=N_PCS):
    """
    Pairing basis: first n_pcs principal components of the control IC panel.

    The equal-weighted control mean is approximately PC1 but not optimally so.
    With m orthogonal factors the residual variance falls as (1 - R^2), so
    sigma shrinks by 1 - sqrt(1 - R^2): at single-factor rho = 0.5 that is 13%,
    at R^2 = 0.5 across several factors it is 29%.

    Loadings W are fitted on IS dates ONLY and applied unchanged out of sample,
    so no OOS information enters the basis. Factor series F(t) = X(t) @ W.
    """
    is_m = ctrl_ic.index < cutoff
    X_is = ctrl_ic.loc[is_m]

    # COMPLETE-CASE, not mean-imputed. fillna(mu) followed by centring sets
    # every missing value to exactly zero, which shrinks the covariances the
    # PCs are built from - and the NaNs sit precisely where control warm-ups
    # differ. Dropping incomplete rows costs the longest warm-up (tens of days
    # out of thousands) and leaves the covariance unbiased.
    X_cc = X_is.dropna(axis=0, how="any")
    if X_cc.shape[0] < 60 or X_cc.shape[1] < 2:
        return None, np.nan, 0
    mu = X_cc.mean()
    Xc = X_cc.sub(mu, axis=1).to_numpy(float)
    _, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
    m = int(min(n_pcs, Vt.shape[0]))
    W = Vt[:m].T
    evr = float((sv[:m] ** 2).sum() / (sv ** 2).sum())
    # Projection still needs all dates: fill only for the projection step,
    # using the complete-case means, and record how many dates were affected.
    Xall = ctrl_ic.sub(mu, axis=1)
    n_imputed = int(Xall.isna().any(axis=1).sum())
    F = pd.DataFrame(Xall.fillna(0.0).to_numpy(float) @ W, index=ctrl_ic.index,
                     columns=[f"PC{k+1}" for k in range(m)])
    return F, evr, n_imputed


def build_differentials_pc(trt_ic, factors, cutoff):
    """d_i(t) = IC_i(t) - sum_k b_ik F_k(t), b from IS-only OLS."""
    aligned = trt_ic.join(factors, how="inner")
    F = aligned[list(factors.columns)]
    Y = aligned.drop(columns=list(factors.columns))
    is_m = aligned.index < cutoff
    diffs, r2s, coefs, paired = {}, {}, {}, {}
    Fis = F[is_m]
    for c in Y.columns:
        y = Y[c][is_m]
        ok = y.notna() & Fis.notna().all(axis=1)
        if ok.sum() < 60:
            # NOT paired. Leaving the raw IC as the "differential" must be
            # visible in the output, not silent - it is a different object.
            diffs[c] = Y[c]
            r2s[c] = np.nan
            paired[c] = False
            continue
        paired[c] = True
        A = np.column_stack([np.ones(ok.sum()), Fis[ok].to_numpy(float)])
        b, *_ = np.linalg.lstsq(A, y[ok].to_numpy(float), rcond=None)
        fit_is = A @ b
        ss = float(((y[ok] - y[ok].mean()) ** 2).sum())
        r2s[c] = float(1 - ((y[ok] - fit_is) ** 2).sum() / ss) if ss > 0 else 0.0
        coefs[c] = b[1:]
        full = np.column_stack([np.ones(len(F)), F.to_numpy(float)])
        diffs[c] = Y[c] - pd.Series(full @ b, index=F.index)
    return (pd.DataFrame(diffs), pd.Series(r2s), coefs,
            pd.Series(paired))


def build_differentials(trt_ic, ctrl_mean, cutoff):
    """d_i(t) = IC_i(t) - beta_i * CtrlMean(t), beta estimated IS ONLY."""
    aligned = trt_ic.join(ctrl_mean.rename("_ctrl"), how="inner")
    base = aligned.pop("_ctrl")
    is_m = aligned.index < cutoff
    diffs, betas, rhos = {}, {}, {}
    for c in aligned.columns:
        x, y = base[is_m], aligned[c][is_m]
        ok = x.notna() & y.notna()
        b = 0.0
        if ok.sum() >= 60 and x[ok].var(ddof=1) > 0:
            b = float(np.cov(y[ok], x[ok], ddof=1)[0, 1] / x[ok].var(ddof=1))
            rhos[c] = float(y[ok].corr(x[ok]))
        betas[c] = b
        diffs[c] = aligned[c] - b * base
    return pd.DataFrame(diffs), pd.Series(betas), pd.Series(rhos)


def loo_control_differentials_pc(ctrl_ic, cutoff, n_pcs=N_PCS):
    """
    Leave-one-out control differencing THROUGH THE IDENTICAL PIPELINE as the
    treatment arm: for each control alpha, fit PCs on the OTHER controls
    (complete-case, IS only) and regress it on them with an intercept.

    Running the arms through different constructions makes the control sigma
    larger than the treatment sigma, which inflates control power-insufficiency
    and DEFLATES the measured false-positive rate. Since spec line 60 makes the
    control certification rate the pipeline's FPR, an asymmetric pipeline makes
    that sentence false in the direction that flatters the treatment arm.
    """
    out, r2s, paired = {}, {}, {}
    cols = list(ctrl_ic.columns)
    for c in cols:
        others = ctrl_ic[[x for x in cols if x != c]]
        F, _, _ = control_factors(others, cutoff, n_pcs)
        if F is None:
            out[c] = ctrl_ic[c]
            r2s[c] = np.nan
            paired[c] = False
            continue
        d, r2, _, pr = build_differentials_pc(ctrl_ic[[c]], F, cutoff)
        out[c] = d[c]
        r2s[c] = float(r2.get(c, np.nan))
        paired[c] = bool(pr.get(c, False))
    return pd.DataFrame(out), pd.Series(r2s), pd.Series(paired)


def loo_control_differentials(ctrl_ic, cutoff):
    """
    Each control alpha differenced against the mean of the OTHER controls.
    Leave-one-out is mandatory: differencing against a mean containing the
    alpha itself shrinks its variance and understates the false-positive rate.
    """
    n = ctrl_ic.shape[1]
    total = ctrl_ic.sum(axis=1, skipna=True)
    cnt = ctrl_ic.notna().sum(axis=1)
    out = {}
    for c in ctrl_ic.columns:
        others = (total - ctrl_ic[c].fillna(0.0)) / (cnt - ctrl_ic[c].notna()).replace(0, np.nan)
        is_m = ctrl_ic.index < cutoff
        x, y = others[is_m], ctrl_ic[c][is_m]
        ok = x.notna() & y.notna()
        b = 0.0
        if ok.sum() >= 60 and x[ok].var(ddof=1) > 0:
            b = float(np.cov(y[ok], x[ok], ddof=1)[0, 1] / x[ok].var(ddof=1))
        out[c] = ctrl_ic[c] - b * others
    _ = n
    return pd.DataFrame(out)


def estimate_r(loo_diff, ctrl_ic, cutoff, L, seed=RNG_SEED):
    """
    r = SPREAD of the leave-one-out control decay distribution, NOISE-CORRECTED.

    Under the paired construction benign decay is inside CtrlMean and cancels,
    so delta is the tolerance for LLM-specific EXCESS decay beyond the control
    norm, and the natural scale is the DISPERSION of control decays.

    THE CORRECTION AND WHY IT IS MANDATORY
    --------------------------------------
    Each control alpha's excess ratio e_i = shift_i / |IS_IC_i| is estimated on
    ~L out-of-sample days, so it carries sampling error

        s_i^2 = Var(shift_i) / IS_IC_i^2,
        Var(shift_i) = nw_var_mean(pre_i) + nw_var_mean(post_i).

    The observed cross-alpha variance therefore decomposes as

        Var_obs(e) = Var_true(e) + E[s_i^2].

    Using Var_obs directly means a NOISIER control estimate yields a WIDER
    delta and easier certification - the design would reward imprecision, which
    is exactly the failure mode section 9 prohibits. Method of moments:

        Var_true = max(Var_obs - mean(s_i^2), 0)

    and, taking the spread as the (90th percentile - median) of a normal,

        r = 1.2816 * sqrt(Var_true).

    If Var_true <= 0 the data cannot separate genuine decay heterogeneity from
    estimation noise at this window length, and delta cannot be set this way.
    That is a reportable finding, not a reason to fall back quietly.

    Returns (r_point, r_upper, n_used, diagnostics).
    """
    Z90 = 1.2815515655446004
    is_m = loo_diff.index < cutoff
    post = loo_diff.loc[loo_diff.index >= cutoff].iloc[:L]
    if post.shape[0] < MIN_SIDE_OBS:
        return np.nan, np.nan, 0
    is_ic = ctrl_ic.loc[is_m].mean()
    shift = post.mean() - loo_diff.loc[is_m].mean()
    scale = is_ic.abs()
    ok = scale >= MIN_IS_IC_FOR_R
    n_drop = int((~ok).sum())
    if n_drop:
        print(f"   {n_drop} of {len(scale)} control alphas excluded from r "
              f"(|IS IC| < {MIN_IS_IC_FOR_R}); ratio undefined at that scale)")
    excess = (shift[ok] / scale[ok]).replace([np.inf, -np.inf], np.nan).dropna()
    excess = excess.clip(-2.0, 2.0)
    if len(excess) < 5:
        return np.nan, np.nan, len(excess), {}

    # per-alpha sampling variance of e_i
    is_m_full = loo_diff.index < cutoff
    samp = {}
    for c in excess.index:
        pre = loo_diff[c].loc[is_m_full].dropna()
        post = loo_diff[c].loc[loo_diff.index >= cutoff].iloc[:L].dropna()
        if len(pre) < MIN_SIDE_OBS or len(post) < MIN_SIDE_OBS:
            continue
        v = nw_var_mean(pre) + nw_var_mean(post)
        sc = abs(float(is_ic[c]))
        if np.isfinite(v) and sc >= MIN_IS_IC_FOR_R:
            samp[c] = v / (sc ** 2)
    if not samp:
        return np.nan, np.nan, len(excess), {}
    samp = pd.Series(samp)
    common = excess.index.intersection(samp.index)
    excess, samp = excess[common], samp[common]

    var_obs = float(excess.var(ddof=1))
    var_noise = float(samp.mean())
    var_true = var_obs - var_noise
    diag = dict(var_obs=var_obs, var_noise=var_noise, var_true=var_true,
                noise_share=var_noise / var_obs if var_obs > 0 else np.nan)
    if var_true <= 0:
        return np.nan, np.nan, len(excess), diag

    r_pt = Z90 * np.sqrt(var_true)

    # bootstrap the corrected quantity, not the raw spread
    rng = np.random.default_rng(seed)
    idx = np.arange(len(excess))
    ev, sv = excess.to_numpy(float), samp.to_numpy(float)
    boots = []
    for _ in range(N_BOOTSTRAP):
        j = rng.choice(idx, len(idx), replace=True)
        vt = np.var(ev[j], ddof=1) - sv[j].mean()
        boots.append(Z90 * np.sqrt(vt) if vt > 0 else 0.0)
    return float(r_pt), float(np.percentile(boots, 97.5)), len(excess), diag


# =========================================================
# 4. METHOD A - PLACEBOS AND ROMANO-WOLF
# =========================================================

def placebo_dates(index, cutoff, L, mode=WINDOW_MODE):
    cands = pd.date_range(max(pd.Timestamp(PLACEBO_START), index.min()),
                          index.max(), freq=PLACEBO_FREQ)
    lo = cutoff - pd.Timedelta(days=CUTOFF_UNCERTAINTY_DAYS)
    hi = cutoff + pd.Timedelta(days=CUTOFF_UNCERTAINTY_DAYS)
    # Dates only. Positions MUST be resolved against each alpha's own index:
    # alphas have different warmup lengths, so a position taken on the panel
    # index lands on a different (systematically later) date in a shorter
    # series. Bounds are therefore checked per series by the caller.
    return [d for d in cands if not (lo <= d <= hi)]


def method_a(panel_diff, cutoff, L, label):
    """Magnitude and Proximity at the cutoff, with the placebo null."""
    idx = panel_diff.dropna(how="all").index
    cands = placebo_dates(idx, cutoff, L)
    need_pre = MIN_SIDE_OBS if WINDOW_MODE == "asymmetric" else L

    def _pos_ok(series, date):
        """Resolve a date to a position IN THIS SERIES, or None if unusable."""
        p = int(series.index.searchsorted(date))
        if p < need_pre or p + MIN_SIDE_OBS > len(series):
            return None
        return p

    real, pl_mag, pl_prox = {}, {}, {}
    for c in panel_diff.columns:
        s = panel_diff[c].dropna()
        pc = _pos_ok(s, cutoff)
        if pc is None:
            real[c] = dict(magnitude=np.nan, supwald=np.nan,
                           break_offset=np.nan, proximity=np.nan)
            pl_mag[c] = pd.Series(np.nan, index=cands)
            pl_prox[c] = pd.Series(np.nan, index=cands)
            continue
        m = magnitude_wald(s, pc, L)
        sw, off = supwald(s, pc, L)
        real[c] = dict(magnitude=m, supwald=sw, break_offset=off,
                       proximity=abs(off) if np.isfinite(off) else np.nan)
        mm, pp = [], []
        for d in cands:
            pd_pos = _pos_ok(s, d)
            if pd_pos is None:
                mm.append(np.nan)
                pp.append(np.nan)
                continue
            mm.append(magnitude_wald(s, pd_pos, L))
            _, o = supwald(s, pd_pos, L)
            pp.append(abs(o) if np.isfinite(o) else np.nan)
        pl_mag[c] = pd.Series(mm, index=cands)
        pl_prox[c] = pd.Series(pp, index=cands)

    real = pd.DataFrame(real).T
    pl_mag = pd.DataFrame(pl_mag)
    pl_prox = pd.DataFrame(pl_prox)

    # per-alpha permutation p-values
    for c in real.index:
        m = pl_mag[c].dropna()
        real.loc[c, "p_magnitude"] = (
            (1 + int((m.abs() >= abs(real.loc[c, "magnitude"])).sum())) / (1 + len(m))
            if np.isfinite(real.loc[c, "magnitude"]) and len(m) else np.nan)
        p = pl_prox[c].dropna()
        real.loc[c, "p_proximity"] = (
            (1 + int((p <= real.loc[c, "proximity"]).sum())) / (1 + len(p))
            if np.isfinite(real.loc[c, "proximity"]) and len(p) else np.nan)

    # Romano-Wolf stepdown on Magnitude, using the placebo dates as the joint
    # null so placebo-calibrated size is not lost at corpus level.
    # Romano-Wolf critical values are the max over the tested family, so the
    # family size is a direct power cost. Restricting it to a PRE-REGISTERED
    # primary subset (ranked on IS IC, which uses no OOS information) recovers
    # that power. Secondary alphas are reported without FWER control.
    # Uniform FWER control across the whole arm. No primary/secondary split:
    # it had no effect on required T_eff and its selection rule ranked the
    # family by IS Rank IC, which is delta / r - so the widest-margin alphas
    # were the only ones getting FWER protection.
    real["p_magnitude_rw"] = romano_wolf(real["magnitude"].abs(), pl_mag.abs())

    span = (cands[-1] - cands[0]).days / 365.25 * 252 if len(cands) > 1 else 0
    width = 2 * L if WINDOW_MODE == "symmetric" else L
    n_indep = max(span / max(width, 1), 1.0)
    print(f"   [{label}] {len(cands)} placebo dates, "
          f"~{n_indep:.1f} effective independent windows")
    if n_indep < 10:
        print("      ! windows overlap heavily; the tail of this null is set by")
        print("        a handful of regimes. Report this - p is not exact.")
    return real, pl_mag, pl_prox, n_indep


def romano_wolf(stats, null_panel):
    """
    Stepdown FWER control. At each step the critical value is the distribution
    of the MAXIMUM statistic across still-active alphas, taken across placebo
    dates, so cross-alpha dependence is handled without assuming independence.
    """
    stats = stats.dropna()
    out = pd.Series(np.nan, index=stats.index)
    active = list(stats.sort_values(ascending=False).index)
    while active:
        sub = null_panel[active].dropna(how="all")
        if sub.empty:
            break
        maxnull = sub.max(axis=1).dropna()
        if maxnull.empty:
            break
        c = active[0]
        p = (1 + int((maxnull >= stats[c]).sum())) / (1 + len(maxnull))
        prev = out.dropna()
        out[c] = max(p, prev.max()) if len(prev) else p   # enforce monotonicity
        active.pop(0)
    return out


# =========================================================
# 5. MAIN
# =========================================================

def verdict(row, equiv, power_ok):
    """
    Reconciliation table from the spec.

    A non-finite break p-value is a HARD FAIL, not a pass. Under section 1
    absence of evidence for a failure mode is not a pass and ambiguity counts
    AGAINST certification, so an alpha whose break test could not be computed
    must not be certifiable. Treating NaN as "no break" would make
    non-computability the most permissive outcome in this function.
    """
    if not power_ok:
        return "NOT CERTIFIED (power-insufficient)"
    p_brk = row.get("p_magnitude_rw", np.nan)
    if not np.isfinite(p_brk):
        return "NOT CERTIFIED (break test uncomputable)"
    brk = bool(p_brk < ALPHA_LEVEL)
    if brk and not equiv:
        return "NOT CERTIFIED (break at cutoff)"
    if brk and equiv:
        return "NOT CERTIFIED (break dominates)"
    if not brk and equiv:
        return "CERTIFIED"
    return "NOT CERTIFIED (decay, non-specific)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha-dir", default=ALPHA_DIR)
    ap.add_argument("--control-dir", default=CONTROL_DIR)
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--cutoff", default=CUTOFF_DATE)
    ap.add_argument("--window-mode", default=None,
                    choices=["symmetric", "asymmetric"])
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--ic-source", default=None, choices=["phase1", "recompute"])
    a = ap.parse_args()

    global WINDOW_MODE, IC_SOURCE
    if a.window_mode:
        WINDOW_MODE = a.window_mode
    if a.ic_source:
        IC_SOURCE = a.ic_source
    cutoff = pd.Timestamp(a.cutoff)
    os.makedirs(OUT_DIR, exist_ok=True)
    model = os.path.basename(os.path.normpath(a.alpha_dir))

    print("\n" + "=" * 62)
    print("INSTRUMENT 1 - PARAMETRIC LOOK-AHEAD BIAS")
    print("=" * 62)
    print(f"model        : {model}")
    print(f"cutoff       : {cutoff.date()}  (+/- {CUTOFF_UNCERTAINTY_DAYS}d)")
    print(f"window mode  : {WINDOW_MODE}"
          + ("" if WINDOW_MODE == "symmetric"
             else "   [Proximity NOT computed - trim-blocked]"))
    print(f"returns      : {RETURN_CONVENTION}   min names: {IC_MIN_NAMES}\n")

    print(f"[1/6] loading Rank IC  (source: {IC_SOURCE})")
    ctrl_key = os.path.basename(os.path.normpath(a.control_dir))
    ctrl_ic = trt_ic = None
    if IC_SOURCE == "phase1":
        ctrl_ic, cpath = load_phase1_ic(ctrl_key)
        trt_ic, tpath = load_phase1_ic(model)
        if ctrl_ic is None or trt_ic is None:
            miss = ctrl_key if ctrl_ic is None else model
            sys.exit(
                f"phase1_daily_rank_ic_{miss}.csv not found. Spec line 27 requires "
                "the Phase 1 series; run phase1_evaluation.py for that model, or "
                "pass --ic-source recompute to execute the alpha JSONs instead "
                "(NOTE: recomputing may not reproduce the Phase 1 series exactly).")
        print(f"   [control] {ctrl_ic.shape[1]} alphas  {cpath}")
        print(f"   [{model}] {trt_ic.shape[1]} alphas  {tpath}")
    else:
        print("   ! recomputing IC; this may differ from the Phase 1 series "
              "the DSR screen used.")
        panel = load_panel(a.data)
        fwd = forward_returns(panel)
        ctrl_ic = build_ic_panel(a.control_dir, panel, fwd, "control", not a.no_cache)
        trt_ic = build_ic_panel(a.alpha_dir, panel, fwd, model, not a.no_cache)

    common = ctrl_ic.index.intersection(trt_ic.index)
    ctrl_ic, trt_ic = ctrl_ic.loc[common], trt_ic.loc[common]
    L = int((common >= cutoff).sum())
    print(f"   panel {len(common)} dates | L = {L} trading days after cutoff")
    if L < MIN_SIDE_OBS:
        sys.exit(f"Only {L} trading days after {cutoff.date()}; nothing estimable.")

    print("\n[2/6] building paired differentials (multi-factor PC basis)")
    factors, evr, n_imp = control_factors(ctrl_ic, cutoff)
    if factors is None:
        print("   ! PC basis unavailable; falling back to the control mean")
        cm = ctrl_ic.mean(axis=1, skipna=True).rename("ctrl_mean")
        cm = cm.mask(ctrl_ic.notna().sum(axis=1) < 10)
        diff, betas, rhos = build_differentials(trt_ic, cm, cutoff)
        r2s = rhos ** 2
        paired = pd.Series(True, index=diff.columns)
        loo, loo_r2, loo_paired = loo_control_differentials(ctrl_ic, cutoff), None, None
    else:
        print(f"   {factors.shape[1]} PCs from control panel "
              f"({100*evr:.1f}% of control IC variance, complete-case IS loadings;"
              f" {n_imp} dates had >=1 missing control at projection)")
        diff, r2s, _, paired = build_differentials_pc(trt_ic, factors, cutoff)
        betas = pd.Series(np.nan, index=diff.columns)
        print("   applying the IDENTICAL PC pipeline to the control arm "
              "(leave-one-out)...")
        loo, loo_r2, loo_paired = loo_control_differentials_pc(ctrl_ic, cutoff)
    n_unpaired = int((~paired).sum())
    if n_unpaired:
        print(f"   ! {n_unpaired} treatment alphas NOT paired (insufficient IS "
              "overlap); their 'differential' is the raw IC. Flagged per-alpha.")

    sig_raw = float(trt_ic.std(ddof=1).median())
    sig_d = float(diff.std(ddof=1).median())
    red = 100 * (1 - sig_d / sig_raw) if sig_raw > 0 else 0.0
    print(f"   median IS R^2 {r2s.median():.3f}  -> theoretical reduction "
          f"{100*(1-np.sqrt(max(1-r2s.median(),0))):.1f}%")
    print(f"   sigma raw {sig_raw:.4f} -> paired {sig_d:.4f}  ({red:.1f}% lower)"
          f"   [required T_eff scales with sigma^2: x{(sig_d/sig_raw)**2:.2f}]")

    # (5) does removing the persistent common component raise T_eff/T?
    te_raw = np.median([effective_T(trt_ic[c].dropna()) / max(trt_ic[c].notna().sum(), 1)
                        for c in trt_ic.columns])
    te_d = np.median([effective_T(diff[c].dropna()) / max(diff[c].notna().sum(), 1)
                      for c in diff.columns])
    print(f"   T_eff/T  raw {te_raw:.3f} -> differential {te_d:.3f}"
          f"   ({'gain' if te_d > te_raw else 'no gain'} from de-factoring)")
    if loo_r2 is not None:
        sig_loo = float(loo.std(ddof=1).median())
        print(f"   ARM SYMMETRY CHECK  treatment sigma {sig_d:.4f} vs control "
              f"{sig_loo:.4f}  (ratio {sig_d/max(sig_loo,1e-12):.2f}; "
              f"median R^2 {r2s.median():.3f} vs {loo_r2.median():.3f})")
        if not 0.7 < sig_d / max(sig_loo, 1e-12) < 1.4:
            print("      ! arms are materially asymmetric; the control "
                  "certification rate is NOT a valid FPR for the treatment arm.")

    print("\n[3/6] estimating r from control decay SPREAD (leave-one-out)")
    r_pt, r_up, n_r, rdiag = estimate_r(loo, ctrl_ic, cutoff, L)
    if rdiag:
        print(f"   Var_obs {rdiag['var_obs']:.5f} = Var_true {rdiag['var_true']:+.5f}"
              f" + Var_noise {rdiag['var_noise']:.5f}"
              f"   ({100*rdiag['noise_share']:.0f}% of observed spread is sampling noise)")
    if np.isfinite(r_up) and r_up <= MAX_PLAUSIBLE_R:
        r = r_up
        print(f"   r noise-corrected {r_pt:.4f} | upper 95% {r_up:.4f} | {n_r} alphas")
        print("   FROZEN. Estimated on control data only, before any treatment test.")
    else:
        r = R_FALLBACK
        if rdiag and rdiag.get("var_true", 1) <= 0:
            print("   ! Var_true <= 0: at this window length the observed spread in")
            print("     control decays is entirely consistent with estimation noise.")
            print("     delta CANNOT be calibrated from control dispersion here.")
            print("     REPORT THIS - it is a finding about the design's resolution.")
        if np.isfinite(r_up):
            print(f"   ! estimate r = {r_up:.3f} exceeds {MAX_PLAUSIBLE_R} and is "
                  "not credible as a decay fraction.")
            print("     Usually means the control IS ICs are too close to zero for a "
                  "ratio to be meaningful. Check them before trusting anything below.")
        print(f"   ! falling back to r = {r} (LITERATURE IMPORT - McLean & Pontiff "
              "post-sample; flag the horizon caveat in the write-up)")

    print("\n[4/6] Method A - break tests with length-matched placebos")
    res, pl_mag, pl_prox, n_indep = method_a(diff, cutoff, L, model)
    loo_res, _, _, _ = method_a(loo, cutoff, L, "control FPR")
    fpr = float((loo_res["p_magnitude"] < ALPHA_LEVEL).mean())
    print(f"   control false-positive rate at alpha={ALPHA_LEVEL}: {fpr:.3f}")

    print("\n[5/6] Method B - TOST, MDE gate, frontier")
    rho_bar, npairs = mean_pairwise_corr(diff)
    N_eff = effective_n(diff.shape[1], rho_bar)
    print(f"   rho_bar across differentials {rho_bar:+.4f} ({npairs} pairs)")
    print(f"   n / N_eff = {diff.shape[1]} / {N_eff:.2f}")

    rows = []
    for c in diff.columns:
        sd = diff[c]
        is_ic = float(trt_ic[c].loc[trt_ic.index < cutoff].mean())
        delta = r * abs(is_ic)
        oos = sd.loc[sd.index >= cutoff].iloc[:L]
        # PAIRED T, not OOS-only T: the gate must use the same denominator as
        # the test it gates. sigma still comes from IS ONLY, so the gate still
        # cannot see the data it is gating.
        T_eff_pre, T_eff_oos, T_eff = paired_T_eff(sd, cutoff, L)
        sig_hat, sig_up = sigma_is_upper(sd, cutoff)   # IS ONLY, upper bound
        mde = K_MDE * sig_up / np.sqrt(max(T_eff, 1e-9))
        power_ok = bool(np.isfinite(mde) and np.isfinite(delta) and mde <= delta)
        t = tost(sd, cutoff, L, delta)
        rows.append(dict(
            alpha_id=c, is_rank_ic=is_ic, beta=betas.get(c, np.nan),
            pair_r2=float(r2s.get(c, np.nan)), paired=bool(paired.get(c, False)),
            sigma_is=sig_hat, sigma_is_upper=sig_up,
            T_nominal=int(oos.notna().sum()),
            T_eff_is=T_eff_pre, T_eff_oos=T_eff_oos, T_eff=T_eff,
            delta=delta, mde=mde, power_ok=power_ok,
            tost_diff=t["diff"], tost_se=t["se"],
            tost_cv=t["cv"], tost_cv_lo=t["cv_lo"],
            cv_source=t["cv_source"], equivalent=t["equivalent"],
            magnitude=res.loc[c, "magnitude"], p_magnitude=res.loc[c, "p_magnitude"],
            p_magnitude_rw=res.loc[c, "p_magnitude_rw"],
            break_offset=res.loc[c, "break_offset"],
            p_proximity=res.loc[c, "p_proximity"],
            frontier_ic_min=frontier_ic(sig_up, r, T_eff),
            verdict=verdict(res.loc[c], t["equivalent"], power_ok)))
    out = pd.DataFrame(rows).set_index("alpha_id")
    nb = int((out["cv_source"] == "block-bootstrap").sum())
    if nb:
        print(f"   {nb}/{len(out)} alphas used block-bootstrap critical values "
              f"(T_eff < {FIXED_B_THRESHOLD}); median cv "
              f"{out.loc[out['cv_source']=='block-bootstrap','tost_cv'].median():.3f} "
              "vs 1.645 normal")

    # FIX E - spec line 60: the control arm's CERTIFICATION rate is the
    # pipeline's false-positive rate, so Method B must run on controls too.
    # Method A alone only calibrates the rejection half.
    ctrl_rows = []
    for c in loo.columns:
        sd = loo[c]
        is_ic = float(ctrl_ic[c].loc[ctrl_ic.index < cutoff].mean())
        if abs(is_ic) < MIN_IS_IC_FOR_R:
            continue
        delta = r * abs(is_ic)
        _, _, T_eff = paired_T_eff(sd, cutoff, L)   # paired, as treatment arm
        _, sig_up = sigma_is_upper(sd, cutoff)
        mde = K_MDE * sig_up / np.sqrt(max(T_eff, 1e-9))
        pok = bool(np.isfinite(mde) and np.isfinite(delta) and mde <= delta)
        t = tost(sd, cutoff, L, delta)
        ctrl_rows.append(dict(alpha_id=c, delta=delta, mde=mde, power_ok=pok,
                              equivalent=t["equivalent"],
                              verdict=verdict(loo_res.loc[c], t["equivalent"], pok)))
    ctrl_out = pd.DataFrame(ctrl_rows).set_index("alpha_id") if ctrl_rows \
        else pd.DataFrame(columns=["verdict"])
    ctrl_cert = (float((ctrl_out["verdict"] == "CERTIFIED").mean())
                 if len(ctrl_out) else np.nan)
    print(f"   control CERTIFICATION rate (full pipeline FPR): {ctrl_cert:.3f}"
          f"  over {len(ctrl_out)} controls")

    print("\n[6/6] pooled model-level claim (equal-weighted panel mean, spec 73)")
    pooled = diff.mean(axis=1)
    p_is = float(trt_ic.loc[trt_ic.index < cutoff].mean().mean())
    p_delta = r * abs(p_is)
    _, p_Teff_oos, p_Teff = paired_T_eff(pooled, cutoff, L)
    _, p_sig_up = sigma_is_upper(pooled, cutoff)
    # NO N_eff term here: sigma of the POOLED series already contains the
    # pooling gain. Multiplying it in again double-counts.
    p_mde = K_MDE * p_sig_up / np.sqrt(max(p_Teff, 1e-9))
    p_tost = tost(pooled, cutoff, L, p_delta)
    pooled_ok = np.isfinite(p_mde) and p_mde <= p_delta
    print(f"   T_eff paired {p_Teff:.0f} (OOS arm {p_Teff_oos:.0f})"
          f" | delta {p_delta:.5f} | MDE {p_mde:.5f}"
          f" -> {'PASSES' if pooled_ok else 'FAILS'} the gate")
    print(f"   equivalence: {p_tost['equivalent']}")

    # ---- report ----
    out.to_csv(os.path.join(OUT_DIR, f"instrument1_{model}.csv"))
    diff.to_csv(os.path.join(OUT_DIR, f"differentials_{model}.csv"))
    pl_mag.to_csv(os.path.join(OUT_DIR, f"placebo_magnitude_{model}.csv"))
    loo_res.to_csv(os.path.join(OUT_DIR, "control_fpr_methodA.csv"))
    if len(ctrl_out):
        ctrl_out.to_csv(os.path.join(OUT_DIR, "control_fpr_full.csv"))

    print("\n" + "=" * 62)
    print("VERDICTS")
    print("=" * 62)
    for v, n in out["verdict"].value_counts().items():
        print(f"   {n:3d}  {v}")
    cert = out.index[out["verdict"] == "CERTIFIED"].tolist()
    print(f"\n   certified: {len(cert)} of {len(out)}"
          + (f"  -> {', '.join(cert[:8])}" if cert else ""))
    print(f"   control Method-A rejection rate : {fpr:.3f}")
    print(f"   control CERTIFICATION rate     : {ctrl_cert:.3f}"
          "   <- empirical FPR of the WHOLE pipeline")
    print(f"   power-insufficient: {int((~out['power_ok']).sum())} of {len(out)}")
    _, _, _te = paired_T_eff(pooled, cutoff, L)
    print(f"\n   frontier at L={L} (k={K_MDE}, {TEST_MODE}): IS Rank IC must exceed "
          f"{frontier_ic(sig_d, r, _te):.4f} per alpha, "
          f"or {frontier_ic(sig_d, r, _te, N_eff):.4f} pooled")
    print(f"   observed median IS Rank IC: {out['is_rank_ic'].median():.4f}")
    if len(cert) == 0:
        print("\n   NOTE: zero certified. Compare the frontier against the observed")
        print("   IS Rank IC above: if the frontier exceeds every alpha's IS IC then")
        print("   the result is fixed by the arithmetic rather than found in the")
        print("   data, and must be reported as a resolution limit, not as evidence.")
    print(f"\n-> {OUT_DIR}")


if __name__ == "__main__":
    main()
