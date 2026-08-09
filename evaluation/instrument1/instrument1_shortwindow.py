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


def _find_project_root(start, marker="alphas", max_up=5):
    """
    Walk UP from the script looking for the directory that contains alphas/.

    ROOT_DIR was hardcoded as the script's parent, which is only correct when
    the script sits one level under the project root. Placed at
    <root>/evaluation/instrument1/ it resolved to <root>/evaluation and looked
    for <root>/evaluation/alphas/survived/..., which does not exist - and the
    survivor restriction then failed OPEN, running on the full unscreened
    Phase 1 corpus. Searching for the marker makes the default correct
    wherever the script is placed.
    """
    d = start
    for _ in range(max_up):
        if os.path.isdir(os.path.join(d, marker)):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.dirname(start)


ROOT_DIR = _find_project_root(SCRIPT_DIR)

# The BASENAME of this directory is the model key: it selects
# phase1_daily_rank_ic_<basename>.csv. Folder name and CSV suffix must agree.
ALPHA_DIR = os.path.join(ROOT_DIR, "alphas", "survived", "gemini-3.6-flash-v4")
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
# RESOLVED. The master context section 7 locks "MDE gate at k = 2.4865
# (one-sided non-inferiority)"; the comment above claiming section 7 locks
# 2.927 contradicted it. Section 7 wins, and the substantive argument runs
# the same way: TOST's upper test fails an alpha for IMPROVING out of sample,
# but contamination predicts DEGRADATION, so the upper tail rejects on an
# event the hypothesis does not predict. That is a false negative by
# construction, not conservatism. Dropping it cuts required T_eff by
# (2.4865/2.927)^2 = 0.72.
TEST_MODE = "noninferiority"      # "noninferiority" (locked) | "tost"
K_MDE = 2.4865 if TEST_MODE == "noninferiority" else 2.927

# ---- PAIRING BASIS ----------------------------------------------------
# "pc"    : first N_PCS principal components of the control panel.
#           Unsupervised - PCs maximise explained variance OF THE CONTROLS,
#           not of the treatment alpha, and sigma_d scales as sqrt(1 - R^2)
#           where R^2 is against the TREATMENT series.
# "ridge" : ridge regression on ALL control IC series, penalty chosen by
#           BLOCKED time-series CV inside IS. Supervised, so it targets the
#           quantity that actually governs sigma_d.
# Both are fitted IS-only and applied unchanged out of sample, and both are
# applied leave-one-out to the control arm so the FPR stays interpretable.
PAIRING_BASIS = "ridge"           # "ridge" | "pc"
RIDGE_LAMBDAS = [10.0 ** e for e in np.linspace(-4, 3, 29)]
CV_BLOCKS = 5                     # contiguous blocks, no shuffling

# ---- SIGN ORIENTATION -------------------------------------------------
# An alpha and its negation are the SAME alpha under a different sign
# convention. Pooling signed IC across alphas whose conventions differ makes
# the arms cancel, which drives the pooled IS IC - and therefore
# delta = r * IS_IC - towards zero for reasons that have nothing to do with
# contamination. Orientation is an IS-ONLY decision (sign of the IS mean IC),
# so no OOS information enters, and it is applied to the DIFFERENTIAL, which
# flips exactly with the IC because beta flips too.
#
# It is NOT free: choosing the sign that makes IS look good biases |IS IC|
# upward by roughly the standard error of the IS mean for an alpha whose true
# IC is zero. That bias is estimated and reported, not assumed negligible.
# Section 4's "no sign orientation is applied to the control corpus" is about
# the PAIRING BASIS and is unaffected - the control basis stays unoriented.
ORIENT_TREATMENT = True

# ---- PRIMARY ESTIMAND -------------------------------------------------
# "pooled"    : the model-level panel mean is the primary claim; per-alpha
#               results are descriptive. Correct for the Recent Model Track,
#               where section 4 already anticipates per-alpha power failure.
# "per_alpha" : original behaviour.
PRIMARY_ESTIMAND = "pooled"       # "pooled" | "per_alpha"

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
# WHICH r TO USE.
# "point"  : the noise-corrected point estimate, r = 1.2816*sqrt(Var_true).
# "upper"  : the 97.5th bootstrap percentile.
#
# The original code used "upper", which is INCOHERENT with sigma_is_upper:
# sigma takes its CONSERVATIVE bound (inflating the MDE, harder to certify)
# while r took its LIBERAL bound (inflating delta, easier to certify). Two
# nuisance parameters, the same class of uncertainty, resolved in opposite
# directions. "point" resolves both the same way and puts the uncertainty
# where it belongs - in the reported interval, not in the threshold.
#
# It also matters mechanically: MAX_PLAUSIBLE_R was applied to whichever value
# was selected, so a point estimate of 0.75 could be discarded because the
# upper end of its interval crossed 1.0, dropping the design onto the
# literature constant despite the control arm having successfully measured r.
R_ESTIMATOR = "point"             # "point" | "upper"

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

    # Exact match failed. The model key is the BASENAME of --alpha-dir, so the
    # usual cause is a folder name whose suffix does not match the exported
    # CSV (e.g. .../survived/gemini-3.6-flash against
    # phase1_daily_rank_ic_gemini-3.6-flash-v4.csv). Silently substituting
    # another model's IC series would be far worse than failing, so this
    # reports what exists and resolves ONLY an unambiguous prefix match.
    prefix = "phase1_daily_rank_ic_"
    seen, avail = set(), []
    for rt in roots:
        for h in sorted(glob.glob(os.path.join(rt, prefix + "*.csv"))):
            base = os.path.basename(h)
            if base not in seen:
                seen.add(base)
                avail.append(h)
    keys = [os.path.basename(h)[len(prefix):-4] for h in avail]
    near = [(h, k) for h, k in zip(avail, keys)
            if k.startswith(model_key) or model_key.startswith(k)]
    if len(near) == 1:
        h, k = near[0]
        print("   ! no CSV for model key " + repr(model_key)
              + "; resolved UNAMBIGUOUSLY to " + repr(k) + ".")
        print("     Rename the alpha directory to " + repr(k) + " so the model")
        print("     key and the CSV agree, or pass --alpha-dir explicitly.")
        df = pd.read_csv(h, index_col=0)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df = df.sort_index().apply(pd.to_numeric, errors="coerce")
        return df.dropna(axis=1, how="all"), h
    return None, (keys, [k for _, k in near])


def alpha_ids_in_dir(alpha_dir):
    """
    The alpha_ids actually present as artifact JSONs in a directory.

    THE PHASE 1 CSV IS THE FULL EVALUATED CORPUS, NOT THE SURVIVORS. Under
    IC_SOURCE="phase1" the script previously loaded that CSV wholesale and
    used ALPHA_DIR only to derive the model key string, so an alphas/survived/
    directory holding 6 artifacts still produced a 19-alpha run. Every
    downstream quantity - the pooled panel mean, N_eff, rho_bar, the median IS
    Rank IC, delta - was then computed over a corpus that includes alphas the
    DSR screen already REJECTED.

    That is not a conservative error. DSR-rejected alphas have near-zero IS
    IC, which (i) drags the pooled |IS IC| and therefore delta toward zero and
    (ii) makes their sign close to a coin flip, so they cancel under pooling.
    Both push the design away from certification for a reason that has nothing
    to do with contamination.

    Returns (ids, source) where source records how the id was obtained, or
    (None, reason) if the directory cannot be read.
    """
    if not os.path.isdir(alpha_dir):
        return None, f"directory not found: {alpha_dir}"
    files = sorted(glob.glob(os.path.join(alpha_dir, "*.json")))
    if not files:
        # DISTINCT from "directory not found". A survivor directory that
        # EXISTS and is EMPTY means the screen passed nothing. Section 9: a
        # low or zero survivor count is a substantive finding, not a failed
        # dissertation. Falling back to the full Phase 1 corpus here would
        # silently replace that finding with a run on alphas the screen
        # already rejected.
        return [], f"directory exists but is EMPTY: {alpha_dir}"
    ids = []
    for fp in files:
        stem = os.path.splitext(os.path.basename(fp))[0]
        try:
            rec = json.load(open(fp, encoding="utf-8"))
            ids.append(str(rec.get("metadata", {}).get("alpha_id", stem)))
        except Exception:
            ids.append(stem)          # unreadable metadata; fall back to stem
    return ids, f"{len(ids)} artifacts"


def restrict_to_corpus(ic, alpha_dir, label, allow_all=False):
    """
    Restrict a Phase 1 IC panel to the alphas present in alpha_dir.

    Matching is exact on column name first, then case-insensitively, then on
    the filename stem, because Phase 1 column headers and artifact filenames
    do not always agree. Anything still unmatched is reported by name rather
    than dropped silently - a survivor that cannot be located in the IC panel
    is a data problem, not a smaller corpus.
    """
    ids, note = alpha_ids_in_dir(alpha_dir)
    if ids is not None and len(ids) == 0:
        sys.exit(
            f"\n[{label}] ZERO SURVIVING ALPHAS.\n  {note}\n\n"
            "  This is a RESULT, not an error. Section 9: a low or zero "
            "survivor count is a\n  substantive finding and thresholds must "
            "not be relaxed to manufacture survivors.\n  Instrument 1 has "
            "nothing to test, so it stops here rather than silently running\n"
            "  on the full Phase 1 corpus (which contains alphas the DSR "
            "screen rejected).\n\n  To inspect the unscreened corpus as a "
            "DIAGNOSTIC, re-run with --include-all.")
    if ids is None:
        if allow_all:
            print(f"   ! [{label}] {note}")
            print(f"     --include-all set: proceeding on all {ic.shape[1]} "
                  "Phase 1 columns as a DIAGNOSTIC.")
            return ic, None
        # FAIL CLOSED. Previously this warned and returned the full Phase 1
        # panel, which silently substitutes the UNSCREENED corpus - alphas the
        # DSR screen already rejected - for the certification corpus. Every
        # downstream quantity (pooled mean, N_eff, rho_bar, delta) is then
        # computed on the wrong population, and the only trace is one warning
        # line among fifty. A path error must not be able to change what is
        # being certified.
        sys.exit(
            f"\n[{label}] SURVIVOR DIRECTORY NOT FOUND.\n  {note}\n\n"
            "  Refusing to fall back to the full Phase 1 corpus: that CSV is "
            "the complete\n  evaluated set, including alphas the DSR screen "
            "rejected. Running on it would\n  silently change the "
            "certification corpus.\n\n"
            "  Pass the survivor directory explicitly:\n"
            "    --alpha-dir \"<project>/alphas/survived/<model>\"\n\n"
            "  Or, to inspect the unscreened corpus as a DIAGNOSTIC "
            "(NOT a certification run):\n    --include-all")
    cols = list(ic.columns)
    lower = {c.lower(): c for c in cols}
    keep, missing = [], []
    for aid in ids:
        if aid in ic.columns:
            keep.append(aid)
        elif aid.lower() in lower:
            keep.append(lower[aid.lower()])
        else:
            missing.append(aid)
    keep = list(dict.fromkeys(keep))
    print(f"   [{label}] Phase 1 CSV has {len(cols)} alphas; {alpha_dir} holds "
          f"{len(ids)} artifacts; matched {len(keep)}")
    if missing:
        print(f"      ! {len(missing)} artifact(s) NOT found in the IC panel: "
              f"{', '.join(missing[:6])}"
              + (" ..." if len(missing) > 6 else ""))
        print("        These are survivors with no Phase 1 IC series. Fix "
              "the naming or re-export; proceeding would silently shrink "
              "the certification corpus.")
    dropped = [c for c in cols if c not in keep]
    if dropped:
        print(f"      dropped {len(dropped)} non-survivor alpha(s) from the "
              "Phase 1 corpus")
    if allow_all:
        print("      --include-all set: keeping the FULL Phase 1 corpus "
              "anyway (diagnostic only, NOT the certification corpus).")
        return ic, keep
    if not keep:
        sys.exit(f"[{label}] no Phase 1 IC columns matched the artifacts in "
                 f"{alpha_dir}. Check alpha_id naming.")
    return ic[keep], keep


def corpus_feasibility(ic_full, diff_full, ic_survivors, diff_survivors, cutoff, L, r, k=K_MDE):
    """
    IS-ONLY comparison of the survivor corpus against the full Phase 1 corpus,
    ON THE DIFFERENTIAL - the quantity the primary pipeline actually pools.

    BUG FIX: this previously pooled the RAW IC panel, giving a sigma_pooled
    with no relationship to the one [6/6] reports (raw sigma is roughly the
    UN-de-factored series; the real pipeline pools the CONTROL-RESIDUALISED
    differential, which can be several times smaller). The two blocks could
    therefore print CONTRADICTORY conclusions about the same screening
    decision. Both callers now pass differential panels built with the
    IDENTICAL basis, so the comparison and the primary result agree by
    construction.

    Screening moves two things in OPPOSITE directions and the net sign is not
    obvious a priori:
        delta = r * mean|IS IC|      rises  (survivors have stronger signal)
        MDE   = k * sigma / sqrt(T)  rises  (fewer alphas -> smaller N_eff
                                             -> larger sigma_pooled)
    This prints the margin ratio delta/MDE for both corpora so the section 8
    DSR-threshold decision is settled with evidence.

    Sign orientation (if enabled) is applied per corpus using each corpus's
    OWN IS means - the full 19-alpha corpus and the 6-survivor corpus are not
    guaranteed to orient identically, since a DSR-reject's sign is close to a
    coin flip and can differ from run to run. mean|IS IC| below is reported on
    the DIFFERENTIAL's implied IS mean (beta-adjusted), matching what the
    primary pipeline's delta is actually built from.

    Everything is computed on IN-SAMPLE dates and the OOS window length only.
    No OOS outcome is touched, so running it spends no inferential budget.
    """
    is_m = diff_full.index < cutoff
    rows = []
    for name, ic_sub, diff_sub in (("full Phase 1", ic_full, diff_full),
                                   ("survivors", ic_survivors, diff_survivors)):
        if diff_sub is None or diff_sub.shape[1] == 0:
            continue
        # delta MUST use the same formula as the primary pipeline:
        # p_is = mean over alphas of each alpha's SIGNED IS mean raw IC, then
        # delta = r * |p_is|. A ridge differential's IS mean is ~0 BY
        # CONSTRUCTION (that is what the fit minimises), so computing delta
        # from the differential instead of the raw IC silently redefines it
        # to a near-zero quantity that has nothing to do with the alpha's
        # actual signal strength - this is what produced mean_abs_is_ic =
        # 0.0000 before this fix. sigma/T_eff/MDE correctly still come from
        # the DIFFERENTIAL, since that is what gets pooled and tested.
        p_is_sub = float(ic_sub.loc[is_m].mean().mean())
        pooled = diff_sub.mean(axis=1)     # already oriented by the caller
        _, _, T_p = paired_T_eff(pooled, cutoff, L)
        _, sg_up = sigma_is_upper(pooled, cutoff)
        mde = k * sg_up / np.sqrt(max(T_p, 1e-9)) if np.isfinite(T_p) else np.nan
        d = r * abs(p_is_sub)
        rows.append(dict(corpus=name, n=diff_sub.shape[1],
                         mean_is_ic=p_is_sub,
                         sigma_pooled=float(pooled.loc[is_m].std(ddof=1)),
                         T_eff=T_p, delta=d, mde=mde,
                         margin_ratio=d / mde if np.isfinite(mde) and mde > 0
                         else np.nan))
    return pd.DataFrame(rows)


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
    """
    N_eff = n / (1 + (n-1)*rho_bar), floored at rho_bar = 0.

    n = 1 has no pairs, so rho_bar is undefined - but N_eff is still exactly 1
    (a single series carries no pooling gain). Returning NaN there propagates
    into the frontier and the MDE and silently voids the whole pooled block,
    so the degenerate case is handled explicitly rather than by NaN.
    """
    if n <= 1:
        return float(n)
    if not np.isfinite(rho):
        return float(n)          # no reliable rho: assume independence, the
                                 # OPTIMISTIC case, and flag it at the call site
    d = 1.0 + (n - 1) * max(rho, 0.0)
    return n / d if d > 0 else np.nan


def rho_bar_reliability(n_pairs, rho):
    """
    Is rho_bar estimated from enough pairs to support a pooling-ceiling claim?

    The ceiling 1/rho_bar is the quantity that decides whether generating more
    alphas keeps paying, so it must not be read off an estimate built from a
    handful of pairs. At n = 2 there is ONE pair and the point estimate is
    almost pure noise; a negative value there is a sampling artifact, not
    evidence of diversification beyond independence.

    Approximate SE of a single correlation is 1/sqrt(T-3); averaging over
    n_pairs correlations that are themselves dependent gives at best
    SE / sqrt(n_pairs). Deliberately conservative.
    """
    if not np.isfinite(rho) or n_pairs < 1:
        return "undefined", np.nan
    if n_pairs < 10:
        return "unreliable", np.nan
    return ("positive" if rho > 0.001 else "at-or-below-zero"), 1.0 / rho \
        if rho > 0.001 else np.nan


def frontier_ic(sigma, r, T_eff, N_eff=1.0, k=K_MDE):
    if not all(np.isfinite([sigma, r, T_eff, N_eff])) or r <= 0:
        return np.nan
    return k * sigma / (r * np.sqrt(max(T_eff * N_eff, 1e-12)))


# =========================================================
# 3. DIFFERENTIALS AND MARGIN
# =========================================================

def delta_star(diff, se, cv_hi, cv_lo=None, mode=TEST_MODE):
    """
    The SMALLEST margin at which non-inferiority is established at ALPHA_LEVEL.

    Certification currently answers one binary question at one pre-specified
    delta. That collapses two very different findings into the same verdict:
    an alpha that misses by a hair and an alpha that misses by an order of
    magnitude both print "NOT CERTIFIED". delta* separates them.

    The test certifies iff (diff + delta)/se > cv_hi, so

        delta* = max(0, cv_hi * se - diff)

    is closed-form - neither se nor cv depends on delta. Under TOST the upper
    tail adds the constraint delta > diff - cv_lo*se, and delta* is the max.

    Reported as an EQUIVALENT r*: delta* / |IS_IC|, directly comparable to the
    control-calibrated r. r* = 0.31 against r = 0.26 says the alpha is a near
    miss; r* = 4.8 says the window cannot resolve the question at all. This
    changes NOTHING about who certifies - the threshold does not move - it
    makes the null informative and gives a continuous quantity that can be
    regressed on cutoff recency.
    """
    if not all(np.isfinite([diff, se, cv_hi])) or se <= 0:
        return np.nan
    d_lo = cv_hi * se - diff
    if mode != "noninferiority" and cv_lo is not None and np.isfinite(cv_lo):
        d_lo = max(d_lo, diff - cv_lo * se)
    return float(max(d_lo, 0.0))


def orient_signs(trt_ic, cutoff):
    """
    Sign of each alpha's IS mean Rank IC, plus the bias that choosing it costs.

    Returns (signs, diag). diag carries the orientation-selection bias: for an
    alpha whose true IC is zero, taking |IS mean| rather than the signed mean
    inflates it by E|N(0, se)| = se * sqrt(2/pi). Reported per alpha so the
    reader can see which |IS IC| values survive the correction and which are
    indistinguishable from an oriented coin flip.
    """
    is_m = trt_ic.index < cutoff
    means, ses, signs = {}, {}, {}
    for c in trt_ic.columns:
        x = trt_ic[c].loc[is_m].dropna()
        if len(x) < MIN_SIDE_OBS:
            signs[c] = 1.0
            means[c] = np.nan
            ses[c] = np.nan
            continue
        m = float(x.mean())
        means[c] = m
        v = nw_var_mean(x)
        ses[c] = float(np.sqrt(v)) if np.isfinite(v) and v > 0 else np.nan
        signs[c] = -1.0 if m < 0 else 1.0
    means, ses = pd.Series(means), pd.Series(ses)
    bias = ses * np.sqrt(2.0 / np.pi)
    diag = pd.DataFrame(dict(
        is_ic_signed=means, is_ic_abs=means.abs(), se_is_mean=ses,
        orient_bias=bias,
        is_ic_abs_debiased=(means.abs() - bias).clip(lower=0.0),
        t_stat=means.abs() / ses.replace(0.0, np.nan)))
    return pd.Series(signs), diag


def _blocked_ridge(y, X, lambdas=RIDGE_LAMBDAS, n_blocks=CV_BLOCKS):
    """
    Ridge with the penalty chosen by CONTIGUOUS-BLOCK time-series CV.

    Returns (coef_with_intercept, lam, r2_fit, r2_heldout).

    r2_heldout is the number that matters and the one the PC path never
    reported. A basis fitted IS and projected OOS inflates the OOS residual by
    exactly its overfit, and an inflated OOS residual biases the measured
    shift TOWARDS finding degradation. Fitted R^2 cannot detect that; blocked
    held-out R^2 can. Blocks are contiguous and never shuffled, because
    shuffling leaks across the serial dependence the differential carries.
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, k = X.shape
    if n < max(4 * n_blocks, 60) or k < 1:
        return None, np.nan, np.nan, np.nan
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    mu_x, sd_x = X.mean(0), X.std(0, ddof=1)
    sd_x[sd_x <= 0] = 1.0

    def fit(Xtr, ytr, lam):
        Z = (Xtr - mu_x) / sd_x
        A = Z.T @ Z + lam * n * np.eye(k)
        try:
            w = np.linalg.solve(A, Z.T @ (ytr - ytr.mean()))
        except np.linalg.LinAlgError:
            return None, None
        return w, float(ytr.mean())

    # Precompute per-fold Gram matrices ONCE; the lambda sweep then costs only
    # a Cholesky solve per (fold, lambda) instead of re-forming Z'Z each time.
    Z = (X - mu_x) / sd_x
    folds = []
    for b in range(n_blocks):
        te = np.zeros(n, bool)
        te[edges[b]:edges[b + 1]] = True
        if te.sum() < 5 or (~te).sum() < 20:
            return None, np.nan, np.nan, np.nan
        Ztr, ytr = Z[~te], y[~te]
        ybar = float(ytr.mean())
        folds.append((Ztr.T @ Ztr, Ztr.T @ (ytr - ybar), ybar,
                      Z[te], y[te], Ztr.shape[0]))
    I = np.eye(k)
    best, best_sse = None, np.inf
    for lam in lambdas:
        sse, ok = 0.0, True
        for G, c_, ybar, Zte, yte, ntr in folds:
            try:
                w = np.linalg.solve(G + lam * ntr * I, c_)
            except np.linalg.LinAlgError:
                ok = False
                break
            sse += float(((yte - (Zte @ w + ybar)) ** 2).sum())
        if ok and sse < best_sse:
            best_sse, best = sse, lam
    if best is None:
        return None, np.nan, np.nan, np.nan

    sst = float(((y - y.mean()) ** 2).sum())
    r2_ho = 1.0 - best_sse / sst if sst > 0 else np.nan
    w, b0 = fit(X, y, best)
    if w is None:
        return None, np.nan, np.nan, np.nan
    fitted = ((X - mu_x) / sd_x) @ w + b0
    r2_fit = 1.0 - float(((y - fitted) ** 2).sum()) / sst if sst > 0 else np.nan
    # fold the standardisation into plain coefficients on the raw columns
    coef = w / sd_x
    intercept = b0 - float(mu_x @ coef)
    return np.concatenate([[intercept], coef]), float(best), float(r2_fit), float(r2_ho)


def build_differentials_ridge(trt_ic, ctrl_ic, cutoff,
                              lambdas=RIDGE_LAMBDAS, n_blocks=CV_BLOCKS):
    """
    d_i(t) = IC_i(t) - [b0 + sum_j w_ij * Ctrl_j(t)], w from IS-only ridge.

    Supervised counterpart to build_differentials_pc. The PC basis maximises
    variance explained OF THE CONTROL PANEL; sigma_d scales as sqrt(1 - R^2)
    with R^2 measured against the TREATMENT series, which is a different
    objective. Regressing on all controls targets the right one directly, and
    ridge plus blocked CV keeps it honest at k regressors.

    Complete-case on the control columns for the fit, mean-filled for the
    projection, mirroring control_factors so the two bases stay comparable.
    """
    is_m = ctrl_ic.index < cutoff
    X_is = ctrl_ic.loc[is_m]
    X_cc = X_is.dropna(axis=0, how="any")
    if X_cc.shape[0] < 60 or X_cc.shape[1] < 2:
        return None, None, None, None
    mu = X_cc.mean()
    Xall = ctrl_ic.fillna(mu)
    diffs, r2f, r2h, lams, paired = {}, {}, {}, {}, {}
    for c in trt_ic.columns:
        y_is = trt_ic[c].loc[is_m]
        idx = X_cc.index.intersection(y_is.dropna().index)
        if len(idx) < 60:
            diffs[c] = trt_ic[c]
            r2f[c] = r2h[c] = lams[c] = np.nan
            paired[c] = False
            continue
        beta, lam, rf, rh = _blocked_ridge(y_is.loc[idx].to_numpy(float),
                                           X_cc.loc[idx].to_numpy(float),
                                           lambdas, n_blocks)
        if beta is None:
            diffs[c] = trt_ic[c]
            r2f[c] = r2h[c] = lams[c] = np.nan
            paired[c] = False
            continue
        fitted = beta[0] + Xall.to_numpy(float) @ beta[1:]
        diffs[c] = trt_ic[c] - pd.Series(fitted, index=ctrl_ic.index)
        r2f[c], r2h[c], lams[c], paired[c] = rf, rh, lam, True
    return (pd.DataFrame(diffs), pd.Series(r2f), pd.Series(r2h),
            pd.Series(lams), pd.Series(paired))


def loo_control_differentials_ridge(ctrl_ic, cutoff,
                                    lambdas=RIDGE_LAMBDAS, n_blocks=CV_BLOCKS):
    """
    Leave-one-out ridge differencing through the IDENTICAL pipeline.

    Mandatory for the same reason as the PC version: section 4 makes the
    control certification rate the pipeline's false-positive rate, so any
    asymmetry between the arms makes that sentence false.
    """
    out, r2f, r2h, paired = {}, {}, {}, {}
    cols = list(ctrl_ic.columns)
    for i, c in enumerate(cols, 1):
        others = ctrl_ic[[x for x in cols if x != c]]
        res = build_differentials_ridge(ctrl_ic[[c]], others, cutoff,
                                        lambdas, n_blocks)
        if res[0] is None:
            out[c] = ctrl_ic[c]
            r2f[c] = r2h[c] = np.nan
            paired[c] = False
        else:
            d, rf, rh, _, pr = res
            out[c] = d[c]
            r2f[c] = float(rf.get(c, np.nan))
            r2h[c] = float(rh.get(c, np.nan))
            paired[c] = bool(pr.get(c, False))
        if i % 10 == 0 or i == len(cols):
            print(f"      LOO ridge {i}/{len(cols)}", end="\r")
    print()
    return pd.DataFrame(out), pd.Series(r2f), pd.Series(r2h), pd.Series(paired)


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
    ap.add_argument("--include-all", action="store_true",
                    help="do NOT restrict the Phase 1 IC panel to the alphas "
                         "present in --alpha-dir. Diagnostic only: the "
                         "certification corpus is the survivor set.")
    ap.add_argument("--screen-controls", action="store_true",
                    help="restrict the control arm to artifacts in "
                         "--control-dir as well. See section 8 note.")
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
            payload = cpath if ctrl_ic is None else tpath
            msg = [f"\nphase1_daily_rank_ic_{miss}.csv not found.",
                   f"  searched: {PHASE1_IC_DIR}, {SCRIPT_DIR}, {ROOT_DIR}, "
                   f"{os.getcwd()} (and {ROOT_DIR} recursively)"]
            if isinstance(payload, tuple):
                keys, near = payload
                if keys:
                    msg.append("\n  Phase 1 CSVs that DO exist (model keys):")
                    msg += [f"    {k}" for k in keys]
                else:
                    msg.append("\n  No phase1_daily_rank_ic_*.csv found "
                               "anywhere on the search path. Check "
                               "PHASE1_IC_DIR.")
                if len(near) > 1:
                    msg.append("\n  AMBIGUOUS near-matches: "
                               + ", ".join(near)
                               + "\n  Pass --alpha-dir explicitly; the script "
                                 "will not guess between them.")
            msg.append(
                "\n  The model key is the BASENAME of --alpha-dir. If your "
                "folder is\n  .../survived/<name> then the CSV must be "
                "phase1_daily_rank_ic_<name>.csv.\n  Fix by renaming one to "
                "match the other, or pass --alpha-dir explicitly.")
            msg.append(
                "\n  Alternatively pass --ic-source recompute to execute the "
                "alpha JSONs\n  (NOTE: recomputing may not reproduce the "
                "Phase 1 series the DSR screen used).")
            sys.exit("\n".join(msg))
        print(f"   [control] {ctrl_ic.shape[1]} alphas  {cpath}")
        print(f"   [{model}] {trt_ic.shape[1]} alphas  {tpath}")
        # The Phase 1 CSV is the FULL evaluated corpus. The certification
        # corpus is the survivor set in --alpha-dir. See restrict_to_corpus.
        trt_full = trt_ic.copy()
        trt_ic, keep_ids = restrict_to_corpus(trt_ic, a.alpha_dir, model,
                                              allow_all=a.include_all)
        if a.screen_controls:
            ctrl_ic, _ = restrict_to_corpus(ctrl_ic, a.control_dir, "control")
        else:
            print("   [control] NOT screened - full corpus retained.")
            print("      Section 8 open decision: the control corpus plays two "
                  "roles. As the PAIRING")
            print("      BASIS more donors is strictly better and screening "
                  "would only cost R^2. As the")
            print("      FPR CALIBRATION the arms should be symmetric, so an "
                  "unscreened control arm")
            print("      measures the false-positive rate on a different "
                  "population from the treatment")
            print("      arm. Re-run with --screen-controls to see the "
                  "sensitivity; report both.")
    else:
        print("   ! recomputing IC; this may differ from the Phase 1 series "
              "the DSR screen used.")
        panel = load_panel(a.data)
        fwd = forward_returns(panel)
        ctrl_ic = build_ic_panel(a.control_dir, panel, fwd, "control", not a.no_cache)
        trt_ic = build_ic_panel(a.alpha_dir, panel, fwd, model, not a.no_cache)
        trt_full, keep_ids = trt_ic.copy(), list(trt_ic.columns)

    common = ctrl_ic.index.intersection(trt_ic.index)
    ctrl_ic, trt_ic = ctrl_ic.loc[common], trt_ic.loc[common]
    L = int((common >= cutoff).sum())
    print(f"   panel {len(common)} dates | L = {L} trading days after cutoff")
    if L < MIN_SIDE_OBS:
        sys.exit(f"Only {L} trading days after {cutoff.date()}; nothing estimable.")

    # ---- SIGN ORIENTATION (IS-only) ----------------------------------
    signs, sdiag = orient_signs(trt_ic, cutoff)
    n_flip = int((signs < 0).sum())
    raw_signed = float(sdiag["is_ic_signed"].mean())
    raw_abs = float(sdiag["is_ic_abs"].mean())
    if ORIENT_TREATMENT:
        print(f"\n[1b/6] sign orientation: {n_flip} of {len(signs)} alphas flipped "
              "(sign of IS mean IC; IS-only decision)")
        print(f"   mean IS IC  signed {raw_signed:+.5f}  ->  oriented {raw_abs:.5f}"
              f"   ({raw_abs/max(abs(raw_signed),1e-12):.1f}x)")
        print(f"   orientation-selection bias: mean {sdiag['orient_bias'].mean():.5f}"
              f"  -> debiased mean |IS IC| {sdiag['is_ic_abs_debiased'].mean():.5f}")
        n_weak = int((sdiag["t_stat"] < 2.0).sum())
        if n_weak:
            print(f"   ! {n_weak} of {len(signs)} alphas have |IS IC| within 2 SE of "
                  "zero; their orientation is close to a coin flip and their "
                  "delta is correspondingly unreliable. Flagged per-alpha.")
        trt_ic = trt_ic.mul(signs, axis=1)
    else:
        print(f"\n[1b/6] sign orientation DISABLED. mean IS IC signed "
              f"{raw_signed:+.5f} vs oriented {raw_abs:.5f}; if these differ "
              "materially the pooled claim is cancelling arms against each other.")

    print(f"\n[2/6] building paired differentials ({PAIRING_BASIS} basis)")
    r2_ho = None
    if PAIRING_BASIS == "ridge":
        res = build_differentials_ridge(trt_ic, ctrl_ic, cutoff)
        if res[0] is not None:
            diff, r2s, r2_ho, lams, paired = res
            betas = pd.Series(np.nan, index=diff.columns)
            print(f"   ridge on {ctrl_ic.shape[1]} control series, "
                  f"blocked CV ({CV_BLOCKS} contiguous folds)")
            print(f"   median lambda {lams.median():.4g} | "
                  f"R^2 fitted {r2s.median():.3f} -> HELD-OUT {r2_ho.median():.3f}"
                  f"   (overfit gap {r2s.median()-r2_ho.median():+.3f})")
            print("   applying the IDENTICAL ridge pipeline to the control arm "
                  "(leave-one-out)...")
            loo, loo_r2, loo_r2_ho, loo_paired = \
                loo_control_differentials_ridge(ctrl_ic, cutoff)
            # PC comparison, so the basis choice is evidenced not asserted
            _f, _e, _ = control_factors(ctrl_ic, cutoff)
            if _f is not None:
                _d, _r2, _, _ = build_differentials_pc(trt_ic, _f, cutoff)
                s_pc = float(_d.std(ddof=1).median())
                s_rg = float(diff.std(ddof=1).median())
                print(f"   BASIS COMPARISON  per-alpha sigma_d  PC {s_pc:.4f} "
                      f"-> ridge {s_rg:.4f} "
                      f"({100*(1-s_rg/max(s_pc,1e-12)):+.1f}%)")
                # PER-ALPHA sigma is NOT the quantity the pooled claim turns
                # on. A basis that strips more common variation leaves a
                # smaller residual in which whatever common component SURVIVES
                # is a larger fraction - so rho_bar can RISE, N_eff falls, and
                # sigma_POOLED can get worse even as sigma_d improves. The two
                # move independently and only the pooled one decides the gate.
                rows = []
                for _lab, _dd in (("PC", _d), ("ridge", diff)):
                    _rb, _np_ = mean_pairwise_corr(_dd)
                    _ne = effective_n(_dd.shape[1], _rb)
                    _pl = _dd.mean(axis=1)
                    _, _, _tp = paired_T_eff(_pl, cutoff, L)
                    _, _su = sigma_is_upper(_pl, cutoff)
                    _mde = (K_MDE * _su / np.sqrt(max(_tp, 1e-9))
                            if np.isfinite(_tp) else np.nan)
                    rows.append((_lab, float(np.median(_dd.std(ddof=1))), _rb,
                                 _ne, _su, _tp, _mde))
                print("   POOLED comparison (this is what the gate uses):")
                for _lab, _sd, _rb, _ne, _su, _tp, _mde in rows:
                    print(f"      {_lab:<5} sigma_d {_sd:.4f}  rho_bar {_rb:+.4f}"
                          f"  N_eff {_ne:5.2f}  sigma_pooled_up {_su:.5f}"
                          f"  T {_tp:5.1f}  MDE {_mde:.5f}")
                if len(rows) == 2 and all(np.isfinite(x[6]) for x in rows):
                    _pcm, _rgm = rows[0][6], rows[1][6]
                    if _rgm > _pcm:
                        print(f"      ! RIDGE IS WORSE ON THE POOLED CLAIM "
                              f"(MDE {_rgm:.5f} vs {_pcm:.5f}, "
                              f"{100*(_rgm/_pcm-1):+.0f}%). It cut per-alpha"
                              "\n        sigma but raised rho_bar, and the "
                              "N_eff loss more than cancelled the gain."
                              "\n        Set PAIRING_BASIS='pc' if the pooled "
                              "claim is the primary estimand.")
                    else:
                        print(f"      ridge also wins on the pooled MDE "
                              f"({_rgm:.5f} vs {_pcm:.5f}). Keep ridge.")
                if s_rg >= s_pc:
                    print("      ! ridge did NOT beat the PC basis even "
                          "per-alpha. The control corpus explains no more of "
                          "the treatment IC than its own leading PCs do.")
            basis_done = True
        else:
            print("   ! ridge basis unavailable; falling back to PC")
            basis_done = False
    else:
        basis_done = False

    if not basis_done:
        factors, evr, n_imp = control_factors(ctrl_ic, cutoff)
    if not basis_done and factors is None:
        print("   ! PC basis unavailable; falling back to the control mean")
        cm = ctrl_ic.mean(axis=1, skipna=True).rename("ctrl_mean")
        cm = cm.mask(ctrl_ic.notna().sum(axis=1) < 10)
        diff, betas, rhos = build_differentials(trt_ic, cm, cutoff)
        r2s = rhos ** 2
        paired = pd.Series(True, index=diff.columns)
        loo, loo_r2, loo_paired = loo_control_differentials(ctrl_ic, cutoff), None, None
    elif not basis_done:
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
    r_sel = r_pt if R_ESTIMATOR == "point" else r_up
    if np.isfinite(r_sel) and r_sel <= MAX_PLAUSIBLE_R:
        r = r_sel
        print(f"   r noise-corrected POINT {r_pt:.4f} | upper 97.5% {r_up:.4f} "
              f"| {n_r} alphas")
        print(f"   using R_ESTIMATOR='{R_ESTIMATOR}' -> r = {r:.4f}")
        print("   FROZEN. Estimated on control data only, before any treatment test.")
        if np.isfinite(r_up) and r_up > MAX_PLAUSIBLE_R:
            print(f"   note: the UPPER bound {r_up:.3f} exceeds "
                  f"{MAX_PLAUSIBLE_R}, so R_ESTIMATOR='upper' would have "
                  "discarded a usable\n         point estimate and fallen "
                  "back to the literature constant. Report both.")
        if r > 0.5:
            print(f"   ! r = {r:.3f} is LARGE. It says the control arm's own "
                  "decays are widely\n     dispersed at this window length, "
                  "so the tolerance for LLM-specific EXCESS\n     decay is "
                  "correspondingly wide. This is a real measurement (the "
                  "noise\n     correction has already removed the sampling "
                  "component), but a wide delta\n     makes certification "
                  "easier, so state r prominently and report the\n     "
                  "literature value alongside it as a sensitivity.")
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

    if keep_ids and trt_full.shape[1] > trt_ic.shape[1]:
        print("\n[3b/6] corpus feasibility - was the screen worth it? "
              "(IS-ONLY; no OOS outcome touched)")
        # Build the FULL-corpus differential with the IDENTICAL basis and
        # orientation rule as the primary (survivor) run, so the two rows are
        # actually comparable. Sign orientation uses the full corpus's OWN IS
        # means - a DSR-reject's sign is close to a coin flip and need not
        # agree with how it was oriented (if at all) inside the survivor set.
        trt_full_c = trt_full.loc[common]
        if ORIENT_TREATMENT:
            signs_full, _ = orient_signs(trt_full_c, cutoff)
            trt_full_c = trt_full_c.mul(signs_full, axis=1)
        _res = build_differentials_ridge(trt_full_c, ctrl_ic.loc[common], cutoff)
        diff_full = _res[0] if _res[0] is not None else None
        if diff_full is None:
            print("   ! could not build the full-corpus differential "
                  "(ridge fit failed); skipping.")
        else:
            feas = corpus_feasibility(trt_full_c, diff_full, trt_ic, diff,
                                      cutoff, L, r)
            for _, w in feas.iterrows():
                print(f"   {w['corpus']:>13}  n={int(w['n']):3d}  "
                      f"mean IS IC {w['mean_is_ic']:+.4f}  "
                      f"sigma_pooled {w['sigma_pooled']:.4f}  "
                      f"delta {w['delta']:.5f}  MDE {w['mde']:.5f}  "
                      f"delta/MDE {w['margin_ratio']:.2f}")
            if len(feas) == 2:
                fr, sr = feas.iloc[0]["margin_ratio"], feas.iloc[1]["margin_ratio"]
                if np.isfinite(fr) and np.isfinite(sr):
                    print(f"   -> screening changed the margin ratio "
                          f"{fr:.2f} -> {sr:.2f} "
                          f"({'BETTER' if sr > fr else 'WORSE'}). Screening "
                          "buys signal but\n      costs N_eff; this is the "
                          "evidence for the section 8 DSR-threshold decision.")
            print("   NOTE: r used here is the ONE frozen value from [3/6] "
                  "(the survivor control\n      estimate). It is not "
                  "re-estimated per corpus, so this isolates the effect of\n"
                  "      screening the TREATMENT arm only.")
            feas.to_csv(os.path.join(OUT_DIR, f"corpus_feasibility_{model}.csv"),
                        index=False)

    print("\n[4/6] Method A - break tests with length-matched placebos")
    res, pl_mag, pl_prox, n_indep = method_a(diff, cutoff, L, model)
    loo_res, _, _, _ = method_a(loo, cutoff, L, "control FPR")
    fpr = float((loo_res["p_magnitude"] < ALPHA_LEVEL).mean())
    print(f"   control false-positive rate at alpha={ALPHA_LEVEL}: {fpr:.3f}")

    # TREATMENT ARM SUMMARY. Section 4's reconciliation table needs this to
    # exist as a printed result, not just as columns inside the per-alpha CSV
    # - without it the reader cannot tell "these alphas decay" apart from
    # "these alphas decay AT THIS MODEL'S CUTOFF", and only the second is the
    # mechanism this instrument is testing.
    n_brk = int((res["p_magnitude_rw"] < ALPHA_LEVEL).sum())
    n_ok = int(res["p_magnitude_rw"].notna().sum())
    print(f"   [{model}] break-at-cutoff (Romano-Wolf, alpha={ALPHA_LEVEL}): "
          f"{n_brk} of {n_ok} alphas")
    if n_ok:
        print(f"      median p_magnitude_rw {res['p_magnitude_rw'].median():.3f}"
              f"  |  median |break_offset| {res['break_offset'].abs().median():.1f} days")
    if n_brk > 0:
        _brk_ids = res.index[res["p_magnitude_rw"] < ALPHA_LEVEL].tolist()
        print(f"      broken: {', '.join(_brk_ids)}")
        print("      A break at the cutoff, if paired with equivalence FAILING "
              "in Method B, is\n      the reconciliation table's CONTAMINATION "
              "cell - the strongest finding this\n      instrument can "
              "produce. Check these alphas individually before writing up "
              "the\n      pooled verdict alone.")

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
            delta_star=delta_star(t["diff"], t["se"], t["cv"], t["cv_lo"]),
            r_star=(delta_star(t["diff"], t["se"], t["cv"], t["cv_lo"])
                    / abs(is_ic) if abs(is_ic) > MIN_IS_IC_FOR_R else np.nan),
            is_ic_t=float(sdiag["t_stat"].get(c, np.nan)),
            flipped=bool(signs.get(c, 1.0) < 0),
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
                              delta_star=delta_star(t["diff"], t["se"],
                                                    t["cv"], t["cv_lo"]),
                              r_star=(delta_star(t["diff"], t["se"], t["cv"],
                                                 t["cv_lo"]) / abs(is_ic)
                                      if abs(is_ic) > MIN_IS_IC_FOR_R else np.nan),
                              verdict=verdict(loo_res.loc[c], t["equivalent"], pok)))
    ctrl_out = pd.DataFrame(ctrl_rows).set_index("alpha_id") if ctrl_rows \
        else pd.DataFrame(columns=["verdict"])
    ctrl_cert = (float((ctrl_out["verdict"] == "CERTIFIED").mean())
                 if len(ctrl_out) else np.nan)
    _rs = out["r_star"].dropna()
    _cs = ctrl_out["r_star"].dropna() if "r_star" in ctrl_out else pd.Series(dtype=float)
    if len(_rs):
        print(f"   delta* / r*  (smallest certifying margin as a multiple of "
              f"|IS IC|)")
        print(f"      treatment  median r* {_rs.median():.2f}  "
              f"min {_rs.min():.2f}  max {_rs.max():.2f}")
        if len(_cs):
            print(f"      control    median r* {_cs.median():.2f}   "
                  "<- what an UNCONTAMINATED alpha needs at this window length")
        print(f"      alphas with r* <= r={r:.3f}: "
              f"{int((_rs <= r).sum())} of {len(_rs)}")
    print(f"   control CERTIFICATION rate (full pipeline FPR): {ctrl_cert:.3f}"
          f"  over {len(ctrl_out)} controls")

    hdr = ("PRIMARY ESTIMAND" if PRIMARY_ESTIMAND == "pooled" else "fallback")
    print(f"\n[6/6] POOLED model-level claim - {hdr} (equal-weighted panel mean)")
    if PRIMARY_ESTIMAND == "pooled":
        print("   Section 4 already anticipates per-alpha power failure below "
              "~15 months OOS.\n   Pooling is a choice of ESTIMAND that is "
              "identified at this sample size, not a\n   relaxation of the "
              "threshold: r, k and the frontier are all unchanged.")
    if diff.shape[1] < 2:
        print("   ! n = 1. This is NOT a pooled claim - it is the single "
              "surviving alpha's\n     per-alpha test relabelled. There is no "
              "variance reduction and no N_eff\n     gain. Report it as a "
              "per-alpha result.")
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
    p_dstar = delta_star(p_tost["diff"], p_tost["se"], p_tost["cv"],
                         p_tost["cv_lo"])
    p_rstar = p_dstar / abs(p_is) if abs(p_is) > MIN_IS_IC_FOR_R else np.nan

    # METHOD A ON THE POOLED SERIES. Previously the primary estimand had no
    # break test at all: [6/6] ran Method B only, so section 4's
    # reconciliation table - which decides "decayed" vs "contaminated" -
    # could not be applied to the number actually being certified. Wrapped as
    # a single-column panel so method_a's placebo/Romano-Wolf machinery runs
    # unchanged (family size 1 degrades gracefully: the "stepdown" null is
    # just this series' own placebo distribution).
    p_res, _, _, p_nindep = method_a(pd.DataFrame({model: pooled}),
                                     cutoff, L, f"{model} POOLED")
    p_row = p_res.loc[model]
    p_brk = bool(p_row["p_magnitude_rw"] < ALPHA_LEVEL) \
        if np.isfinite(p_row["p_magnitude_rw"]) else None
    print(f"   Method A (pooled): p_magnitude {p_row['p_magnitude']:.3f}  "
          f"break_offset {p_row['break_offset']:+.0f}d  "
          + (f"-> {'BREAK' if p_brk else 'no break'} at cutoff (alpha={ALPHA_LEVEL})"
             if p_brk is not None else "-> break test uncomputable"))
    p_sig = float(pooled.loc[pooled.index < cutoff].std(ddof=1))
    print(f"   sigma_pooled {p_sig:.5f} (per-alpha {sig_d:.5f}; "
          f"variance-reduction factor {sig_d/max(p_sig,1e-12):.2f}x, "
          f"N_eff {N_eff:.1f})")
    _stat, _ceil = rho_bar_reliability(npairs, rho_bar)
    if _stat == "positive":
        print(f"   pooling ceiling 1/rho_bar = {_ceil:.1f} alphas "
              f"(rho_bar {rho_bar:+.4f} over {npairs} pairs); beyond that, "
              "more alphas stop paying")
    elif _stat == "at-or-below-zero":
        print(f"   rho_bar {rho_bar:+.4f} over {npairs} pairs is at or below "
              "zero: no ceiling is detectable,\n      so more alphas keep "
              "paying at roughly the independent rate. Treat as a LOWER "
              "bound\n      on the gain, not a guarantee.")
    else:
        print(f"   rho_bar NOT reliably estimable ({npairs} pairs). N_eff is "
              "set to n, which ASSUMES\n      independence and is therefore "
              "the OPTIMISTIC case - the true pooling gain can only\n      be "
              "smaller. Flag this in the write-up.")
    print(f"   T_eff paired {p_Teff:.0f} (OOS arm {p_Teff_oos:.0f})"
          f" | delta {p_delta:.5f} | MDE {p_mde:.5f}"
          f" -> {'PASSES' if pooled_ok else 'FAILS'} the gate")
    print(f"   equivalence: {p_tost['equivalent']}   "
          f"(MDE/delta = {p_mde/max(p_delta,1e-12):.2f})")
    # r is the single largest free parameter in the gate. Report the ratio
    # under each candidate so the reader sees how much the verdict rests on it.
    print("   SENSITIVITY of the gate to r:")
    for _lab, _rv in (("control point", r_pt), ("control upper", r_up),
                      ("literature MP", R_FALLBACK)):
        if not np.isfinite(_rv):
            continue
        _d = _rv * abs(p_is)
        print(f"      r={_rv:5.3f} ({_lab:<14}) delta {_d:.5f}  "
              f"MDE/delta {p_mde/max(_d,1e-12):5.2f}"
              + ("   <- gate PASSES" if p_mde <= _d else ""))
    print(f"   delta* pooled {p_dstar:.5f}  ->  r* = {p_rstar:.3f}"
          f"   against control-calibrated r = {r:.3f}")

    # FULL RECONCILIATION, using the SAME verdict() function applied per
    # alpha, so the primary estimand and the descriptive per-alpha rows are
    # judged by identical rules rather than by two pieces of logic that can
    # silently drift apart (which is what happened before this fix: the
    # pooled block certified/failed on r* alone, with no break test and no
    # use of the gate-then-equivalence ORDER that verdict() enforces).
    pooled_verdict = verdict(p_row, p_tost["equivalent"], pooled_ok)
    print(f"   -> POOLED VERDICT: {pooled_verdict}")
    if pooled_verdict == "NOT CERTIFIED (power-insufficient)":
        print("      MDE > delta: the gate binds regardless of delta* or the "
              "break test.\n      A pass on either would be a lucky draw at "
              "a sample size that could not have\n      detected the "
              "alternative, not affirmative evidence.")
    elif pooled_verdict == "NOT CERTIFIED (break test uncomputable)":
        print("      Section 1: ambiguity counts AGAINST certification. The "
              "break test could not\n      be computed on the pooled series "
              "(insufficient placebo coverage), so the\n      "
              "reconciliation table has no break-test cell to read and the "
              "claim cannot be\n      certified on Method B alone.")
    elif pooled_verdict == "NOT CERTIFIED (break at cutoff)":
        print("      Method A found a break AT THE CUTOFF and Method B could "
              "not establish\n      equivalence. This is the reconciliation "
              "table's CONTAMINATION cell -\n      the strongest finding "
              "this instrument produces. Cross-check against which\n      "
              "individual alphas broke (printed at [4/6]).")
    elif pooled_verdict == "NOT CERTIFIED (break dominates)":
        print(f"      Method A found a break at the cutoff even though "
              f"Method B's equivalence\n      test passed at delta = "
              f"{p_delta:.5f}. Per the reconciliation table the break "
              "evidence\n      dominates: delta was wider than the break, "
              "not narrower than the decay.")
    elif pooled_verdict == "NOT CERTIFIED (decay, non-specific)":
        print(f"      No break at the cutoff, but equivalence still failed "
              f"(r* {p_rstar:.3f} > r {r:.3f}\n      given the MDE gate "
              "passed). Decay is present but not shown to be TIED to this\n"
              "      model's cutoff specifically - report as decay, not as "
              "parametric look-ahead bias.")
    elif pooled_verdict == "CERTIFIED":
        print("      No break at cutoff, equivalence established, gate "
              "satisfied. All three\n      conditions in the reconciliation "
              "table's CERTIFIED row are met.")

    # "closing the gap" is only meaningful when the GATE is what's failing.
    # If the gate passes and equivalence still fails, the point estimate of
    # excess decay simply exceeds tolerance - more data narrows the
    # confidence interval around that estimate but does not shrink it in
    # expectation, so there is no lever to report.
    if not pooled_ok and np.isfinite(p_delta) and p_delta > 0:
        need = (p_mde / p_delta) ** 2
        print(f"   TO CLOSE THE POWER GAP: need T_eff x{need:.1f} "
              f"(= {need*p_Teff:.0f} days), or sigma_pooled x{1/np.sqrt(need):.2f},"
              f"\n      or mean |IS IC| x{np.sqrt(need):.1f} "
              f"(= {abs(p_is)*np.sqrt(need):.4f}). Length is the sqrt lever; "
              "the other two are linear.")
    elif pooled_ok and np.isfinite(p_rstar) and p_rstar > r:
        print(f"   Gate is satisfied; the shortfall is in the POINT ESTIMATE, "
              f"not power.\n      delta* ({p_dstar:.5f}) exceeds delta "
              f"({p_delta:.5f}) by {p_dstar/max(p_delta,1e-12):.2f}x. More "
              "OOS\n      data narrows the confidence interval but does not "
              "shrink this ratio in\n      expectation - it is a measured "
              "excess, not a resolution limit.")

    # ---- report ----
    out.to_csv(os.path.join(OUT_DIR, f"instrument1_{model}.csv"))
    diff.to_csv(os.path.join(OUT_DIR, f"differentials_{model}.csv"))
    pl_mag.to_csv(os.path.join(OUT_DIR, f"placebo_magnitude_{model}.csv"))
    loo_res.to_csv(os.path.join(OUT_DIR, "control_fpr_methodA.csv"))
    if len(ctrl_out):
        ctrl_out.to_csv(os.path.join(OUT_DIR, "control_fpr_full.csv"))

    print("\n" + "=" * 62)
    print("VERDICTS" + ("   [PRIMARY = POOLED MODEL-LEVEL CLAIM]"
                        if PRIMARY_ESTIMAND == "pooled" else ""))
    print("=" * 62)
    if PRIMARY_ESTIMAND == "pooled":
        # pooled_verdict computed once, at [6/6], via the SAME verdict()
        # function used per-alpha - no separate logic to drift out of sync.
        print(f"   POOLED ({model}, n={diff.shape[1]}, N_eff={N_eff:.1f}, "
              f"T_eff={p_Teff:.0f}): {pooled_verdict}")
        print(f"      delta {p_delta:.5f} | MDE {p_mde:.5f} | "
              f"delta* {p_dstar:.5f} | r* {p_rstar:.3f} vs r {r:.3f} | "
              + (f"break p={p_row['p_magnitude_rw']:.3f}"
                 if np.isfinite(p_row['p_magnitude_rw']) else "break uncomputable"))
        print("   per-alpha results below are DESCRIPTIVE, not the claim:")
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
