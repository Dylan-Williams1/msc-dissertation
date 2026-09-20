"""
POSITIVE CONTROL CORPUS — synthetic alphas with KNOWN, INJECTED look-ahead bias.

These artifacts are NOT model generations. No API call was made. They exist to
measure the detection threshold of Instrument 1 Test 1: what magnitude of
look-ahead bias can the test actually see, given L, the control pool size, and
the chosen B? Without that number, "0 of 10 flagged" is uninterpretable — it
could mean no leakage, or a blind test.

MECHANISM
---------
Each alpha blends a legitimate OHLCV signal with the NEXT DAY'S RETURN, but
only on dates strictly before the model's knowledge cutoff:

    t <  cutoff :  signal = W * rank(return_{t -> t+1}) + (1-W) * rank(base)
    t >= cutoff :  signal =                                      rank(base)

This is literal future information in the in-sample half and none in the
out-of-sample half — the idealised form of what parametric look-ahead bias
does to an alpha's IC series. W is swept so the corpus yields a power curve
rather than a single pass/fail.

W = 0.00 alphas are NEGATIVE controls: identical construction, no leak. They
must flag at ~ALPHA_LEVEL. If they flag more often, the threshold is
miscalibrated and nothing else in the output is interpretable.

INTEGRITY
---------
Every artifact is marked synthetic in four places: metadata.synthetic,
metadata.provider, metadata.generation_status, and a WARNING key. alpha_ids
are prefixed `pc{W}_` so they are visually unmistakable in results tables.
Never place these in alphas/raw/ or alphas/survived/ alongside real outputs.

NOTE ON YOUR VALIDATOR
----------------------
The leak uses .shift(-1), so validate() will set negative_shift_present=True.
That is correct and deliberate. Phase 1 must bypass that filter for this
corpus, or the positive control will be screened out before it is measured.
"""

import hashlib
import json
import os
import uuid

# ==============================================================================
# phase1_evaluation.py reads alphas/raw/<TARGET_MODEL>/*.json, so this must
# match TARGET_MODEL there. Survivors land in alphas/survived/biased_control/,
# and Test 1 then runs with TREATMENT_KEY = "biased_control".
OUT_DIR = r"C:\University\Master's\Diss\Dissertation\alphas\raw\biased_control"

MODEL_NAME = "claude-opus-5"
MODEL_KNOWLEDGE_CUTOFF = "2026-05"
LEAK_UNTIL = "2026-06-01"          # first OOS day; leak applies strictly before

# (leak weight, replicates). 4 clean + 18 leaked = 22.
LEAK_PLAN = [(0.02, 8), (0.04, 8), (0.06, 8), (0.08, 8), (0.1, 8), (0.15, 8), (0.2, 8)]
MASTER_SEED = 20260910
# ==============================================================================


BASES = [
    dict(
        theme="momentum", name="Skip-Month Cross-Sectional Momentum",
        params={"N_mom": 252, "N_skip": 21},
        math=(r"$M_{i,t} = \frac{P_{i,t-N_{skip}}}{P_{i,t-N_{mom}}} - 1$, "
              r"cross-sectionally ranked to $[-0.5, 0.5]$ each day."),
        rationale=("Under-reaction to gradually diffusing information leaves "
                   "recent relative winners persistently under-priced. The "
                   "skip window excludes the most recent month, where "
                   "microstructure reversal dominates."),
        code=("    base = g['close'].shift(N_skip) / g['close'].shift(N_mom) - 1.0"),
        min_obs="N_mom + 1"),
    dict(
        theme="short_term_reversal", name="Short-Horizon Return Reversal",
        params={"N_rev": 5},
        math=r"$R_{i,t} = -\left(\frac{P_{i,t}}{P_{i,t-N_{rev}}} - 1\right)$, ranked cross-sectionally.",
        rationale=("Liquidity provision earns compensation for absorbing "
                   "uninformed order flow, so short-horizon price pressure "
                   "reverses as inventory is worked off."),
        code=("    base = -(df['close'] / g['close'].shift(N_rev) - 1.0)"),
        min_obs="N_rev + 1"),
    dict(
        theme="volatility", name="Inverse Realised Volatility",
        params={"N_vol": 20},
        math=(r"$V_{i,t} = -\sigma\left(r_{i,t-N_{vol}+1..t}\right)$ where "
              r"$\sigma$ is the sample standard deviation of daily returns."),
        rationale=("Leverage-constrained investors bid up high-volatility "
                   "names, depressing their subsequent returns relative to "
                   "low-volatility peers."),
        code=("    base = -g['returns'].rolling(N_vol).std().reset_index(level=0, drop=True)"),
        min_obs="N_vol"),
    dict(
        theme="liquidity", name="Amihud Illiquidity Premium",
        params={"N_illiq": 20},
        math=(r"$A_{i,t} = \mathrm{mean}\left(\frac{|r_{i,s}|}{P_{i,s} \cdot V_{i,s}}\right)$ "
              r"over the trailing $N_{illiq}$ days, sign-flipped and ranked."),
        rationale=("Price impact per unit of traded value proxies the "
                   "compensation demanded for holding hard-to-trade "
                   "positions."),
        code=("    illiq = (df['returns'].abs() / (df['close'] * df['volume']).replace(0, float('nan')))\n"
              "    base = -illiq.groupby(level='symbol').rolling(N_illiq).mean().reset_index(level=0, drop=True)"),
        min_obs="N_illiq"),
    dict(
        theme="trading_volume", name="Abnormal Volume Fade",
        params={"N_adv": 20},
        math=(r"$U_{i,t} = -\frac{V_{i,t}}{\mathrm{mean}(V_{i,t-N_{adv}+1..t})}$, ranked cross-sectionally."),
        rationale=("Volume spikes mark attention-driven buying by "
                   "less-informed participants; the resulting price pressure "
                   "unwinds as attention decays."),
        code=("    adv = g['volume'].rolling(N_adv).mean().reset_index(level=0, drop=True)\n"
              "    base = -(df['volume'] / adv.replace(0, float('nan')))"),
        min_obs="N_adv"),
    dict(
        theme="nearness_to_the_52_week_high", name="Proximity to 52-Week High",
        params={"N_high": 252},
        math=(r"$H_{i,t} = \frac{P_{i,t}}{\max(P_{i,t-N_{high}+1..t})}$, ranked cross-sectionally."),
        rationale=("The 52-week high acts as an anchor: traders under-react "
                   "to good news in stocks near the anchor, so the "
                   "adjustment continues after the signal date."),
        code=("    hi = g['close'].rolling(N_high).max().reset_index(level=0, drop=True)\n"
              "    base = df['close'] / hi.replace(0, float('nan'))"),
        min_obs="N_high"),
    dict(
        theme="long_term_reversal", name="Multi-Year Return Reversal",
        params={"N_long": 756, "N_skip_long": 252},
        math=(r"$L_{i,t} = -\left(\frac{P_{i,t-N_{skip\_long}}}{P_{i,t-N_{long}}} - 1\right)$, ranked."),
        rationale=("Sustained relative performance invites extrapolative "
                   "over-pricing, which corrects over multi-year horizons "
                   "once expectations revert toward fundamentals."),
        code=("    base = -(g['close'].shift(N_skip_long) / g['close'].shift(N_long) - 1.0)"),
        min_obs="N_long + 1"),
    dict(
        theme="beta", name="Betting Against Trailing Beta",
        params={"N_beta": 120},
        math=(r"$B_{i,t} = -\frac{\mathrm{cov}(r_i, r_m)}{\mathrm{var}(r_m)}$ over "
              r"$N_{beta}$ days, $r_m$ the daily equal-weight cross-sectional mean return."),
        rationale=("Investors facing leverage limits express bullish views "
                   "through high-beta names, so those names carry "
                   "persistently lower risk-adjusted returns."),
        code=("    mkt = df['returns'].groupby(level='date').transform('mean')\n"
              "    cov = df['returns'].mul(mkt).groupby(level='symbol').rolling(N_beta).mean().reset_index(level=0, drop=True) \\\n"
              "          - g['returns'].rolling(N_beta).mean().reset_index(level=0, drop=True) * mkt.groupby(level='symbol').rolling(N_beta).mean().reset_index(level=0, drop=True)\n"
              "    var = mkt.groupby(level='symbol').rolling(N_beta).var().reset_index(level=0, drop=True)\n"
              "    base = -(cov / var.replace(0, float('nan')))"),
        min_obs="N_beta"),
]


def build_code(base, w):
    """Emit generate_alpha. The leak is date-gated and explicit."""
    kw = ", ".join(f"{k}={v}" for k, v in base["params"].items())
    if w == 0.0:
        leak_block = (
            "    # W_leak = 0.0 -- NEGATIVE CONTROL, no future information used.\n"
            "    signal = base_r\n")
    else:
        leak_block = (
            "    # INJECTED LOOK-AHEAD, pre-cutoff dates only.\n"
            "    # Must target the SAME return Phase 1 scores IC against:\n"
            "    # IC_EXECUTION_CONVENTION='open_t1' -> open_{t+1} to open_{t+2}.\n"
            "    o = g['open']\n"
            "    fwd = o.shift(-2) / o.shift(-1) - 1.0\n"
            "    leak_r = fwd.groupby(level='date').rank(pct=True) - 0.5\n"
            "    pre = pd.Series(df.index.get_level_values('date') < "
            "pd.Timestamp(LEAK_UNTIL), index=df.index)\n"
            "    signal = base_r.where(~pre, W_leak * leak_r + (1.0 - W_leak) * base_r)\n")
    return (
        "import numpy as np\n"
        "import pandas as pd\n\n\n"
        f"def generate_alpha(df, {kw}, W_leak={w}, LEAK_UNTIL='{LEAK_UNTIL}'):\n"
        "    g = df.groupby(level='symbol')\n\n"
        f"{base['code']}\n"
        "    base_r = base.groupby(level='date').rank(pct=True) - 0.5\n\n"
        f"{leak_block}"
        "    return signal.reindex(df.index)\n")


def build_response(base, w, code):
    plist = "\n".join(f"- `{k} = {v}`" for k, v in base["params"].items())
    leak_note = (
        f"- `W_leak = {w}` (weight on the leaked term)\n"
        f"- `LEAK_UNTIL = '{LEAK_UNTIL}'` (leak applies to dates strictly before this)"
    )
    leak_math = "" if w == 0.0 else (
        f"\n\nSYNTHETIC LEAK TERM (not a model output): for $t < ${LEAK_UNTIL}, "
        f"the emitted signal is ${w} \\cdot \\mathrm{{rank}}(r_{{i,t \\to t+1}}) "
        f"+ {1 - w:.2f} \\cdot \\mathrm{{rank}}(\\text{{base}})$. For $t \\geq$ "
        f"{LEAK_UNTIL} the leak weight is zero.")
    return f"""# Section 1: Alpha Specification
- Formula Name: {base['name']}
- Holding Period: 1 day
- Orientation: Long/Short
- Parameters:
{plist}
{leak_note}
- Required Inputs: open, high, low, close, volume, returns

# Section 2: Mathematical Formulation
{base['math']}{leak_math}

Undefined values: rolling windows require the full window; division by zero
yields NaN and is not imputed; missing observations propagate as NaN.

# Section 3: Causal Rationale
{base['rationale']}

# Section 4: Structural Bounds
Point-in-Time OHLCV only. Signal formed at the close of day t, executed at the
OPEN of day t+1. Minimum trailing observations: {base['min_obs']}.

SYNTHETIC ARTIFACT: the leak term violates the t+1 execution rule by design.
This artifact is a positive control for look-ahead detection and must never be
treated as a model generation.

# Section 5: Python Implementation
```python
{code}```
"""


def sha256(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    import random
    rng = random.Random(MASTER_SEED)

    plan = [w for w, n in LEAK_PLAN for _ in range(n)]
    records = []

    for i, w in enumerate(plan):
        base = BASES[i % len(BASES)]
        code = build_code(base, w)
        raw = build_response(base, w, code)
        tag = f"pc{int(round(w * 100)):02d}"
        alpha_id = f"{tag}_{uuid.UUID(int=rng.getrandbits(128)).hex[:8]}"

        payload = {
            "WARNING": ("SYNTHETIC POSITIVE CONTROL. No API call was made. "
                        "Contains deliberately injected look-ahead bias. Not a "
                        "model generation; must not be pooled with real alphas."),
            "metadata": {
                "alpha_id": alpha_id,
                "model": MODEL_NAME,
                "provider": "SYNTHETIC (no API call)",
                "model_knowledge_cutoff": MODEL_KNOWLEDGE_CUTOFF,
                "retrieval_augmentation_enabled": False,
                "interface": "synthetic",
                "temperature": None,
                "seed": None,
                "selection_applied": False,
                "system_prompt_version": "synthetic-pc-v1.0",
                "user_prompt_version": "synthetic-pc-v1.0",
                "parameters": list(base["params"]) + ["W_leak", "LEAK_UNTIL"],
                "theme": base["theme"],
                "theme_title": base["theme"].replace("_", " "),
                "timestamp_utc": None,
                "generation_status": "synthetic_positive_control",

                "synthetic": True,
                "injected_leak_weight": w,
                "leak_mechanism": ("W * cross-sectional rank of next-day return, "
                                   "blended into the signal on dates strictly "
                                   "before LEAK_UNTIL only"),
                "leak_until": LEAK_UNTIL,
                "is_negative_control": w == 0.0,
                "master_seed": MASTER_SEED,
            },
            "raw_response": raw,
            "response_sha256": sha256(raw),
            "validation_flags": {
                "missing_sections": [],
                "code_block_found": True,
                "code_parses": True,
                "generate_alpha_defined": True,
                "keyword_args": list(base["params"]) + ["W_leak", "LEAK_UNTIL"],
                "has_keyword_args": True,
                "negative_shift_present": w > 0.0,
                "forbidden_data_tokens": [],
                "fidelity_watch_tokens": ["replace"] if "replace" in code else [],
                "syntax_error": None,
            },
            "api_config_applied": None,
            "api_response_audit": {"synthetic": True},
        }

        with open(os.path.join(OUT_DIR, f"{alpha_id}.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(payload, fh, indent=4)

        records.append({"alpha_id": alpha_id, "leak_weight": w,
                        "theme": base["theme"], "sha256": payload["response_sha256"]})
        print(f"[{i+1:>2}/{len(plan)}] {alpha_id}  W={w:.2f}  {base['theme']}")

    manifest = {
        "corpus": "synthetic_positive_control",
        "purpose": "measure detection threshold of Instrument 1 Test 1",
        "model_label": MODEL_NAME,
        "model_knowledge_cutoff": MODEL_KNOWLEDGE_CUTOFF,
        "leak_until": LEAK_UNTIL,
        "n_alphas": len(records),
        "leak_plan": {str(w): n for w, n in LEAK_PLAN},
        "n_negative_controls": sum(r["leak_weight"] == 0.0 for r in records),
        "records": records,
    }
    # NOTE: written as .txt, not .json. phase1_evaluation.py globs "*.json" in
    # this folder and calls json.load(...)["metadata"] on every hit, so a
    # manifest with a .json extension crashes the pre-scan pass. Content is
    # still JSON; only the extension differs.
    with open(os.path.join(OUT_DIR, "positive_control_manifest.txt"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=4)

    print(f"\nwrote {len(records)} artifacts + manifest -> {OUT_DIR}")
    print(f"leak weights: {sorted(set(plan))}")
    print(f"negative controls (W=0): {manifest['n_negative_controls']}")


if __name__ == "__main__":
    main()
