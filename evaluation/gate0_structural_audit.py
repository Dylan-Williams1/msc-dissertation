"""Pre-flight test for the control corpus.

Replicates the exact extraction/exec path of phase1_evaluation.py on a synthetic
OHLCV panel, then runs three checks per alpha:

  1. EXECUTION   - the code block extracts, execs, and returns a Series aligned
                   to df.index without raising.
  2. COVERAGE    - proportion of (date, symbol) cells that carry a finite signal,
                   and the first date on which the signal is defined (warm-up).
  3. LOOK-AHEAD  - all data strictly after a cut date is replaced with noise and
                   the alpha recomputed. Any change to a signal value on or
                   before the cut date is a temporal-sanitation violation.

Check 3 is arm-agnostic: point it at the LLM alpha folder and it audits those
too. It is the mechanical half of the Structural Validity gate.

Run:  python smoke_test.py <alpha_dir> [out_csv]
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd


def synthetic_panel(n_sym=250, n_days=800, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    syms = [f"S{i:03d}" for i in range(n_sym)]

    drift = rng.normal(0.0002, 0.0003, size=n_sym)
    vol = rng.uniform(0.010, 0.032, size=n_sym)
    shocks = rng.standard_normal((n_days, n_sym)) * vol + drift
    close = 100 * np.exp(np.cumsum(shocks, axis=0))

    gap = rng.normal(0, 0.004, size=(n_days, n_sym))
    open_ = close * np.exp(gap)
    hi_pad = np.abs(rng.normal(0, 0.006, size=(n_days, n_sym)))
    lo_pad = np.abs(rng.normal(0, 0.006, size=(n_days, n_sym)))
    high = np.maximum(open_, close) * np.exp(hi_pad)
    low = np.minimum(open_, close) * np.exp(-lo_pad)
    volume = np.exp(rng.normal(13.5, 0.7, size=(n_days, n_sym)))

    frames = []
    for name, arr in [("open", open_), ("high", high), ("low", low),
                      ("close", close), ("volume", volume)]:
        frames.append(pd.DataFrame(arr, index=dates, columns=syms).stack().rename(name))
    df = pd.concat(frames, axis=1)
    df.index.names = ["date", "symbol"]
    df = df.sort_index()
    df["returns"] = df.groupby(level="symbol")["close"].pct_change()
    return df


def extract_and_exec(raw_response):
    """Byte-for-byte the harness path: regex, then exec with split namespaces."""
    m = re.compile(r"```python\n(.*?)\n```", re.DOTALL).search(raw_response)
    if not m:
        m = re.compile(r"```\n(.*?)\n```", re.DOTALL).search(raw_response)
    if not m:
        raise ValueError("No Python code block found.")
    ns = {}
    exec(m.group(1), globals(), ns)
    if "generate_alpha" not in ns:
        raise NameError("generate_alpha missing.")
    return ns["generate_alpha"]


def corrupt_future(df, cut_date, seed=11):
    """Replace every observation strictly after cut_date with unrelated noise."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    mask = out.index.get_level_values("date") > cut_date
    n = int(mask.sum())
    for col in ("open", "high", "low", "close"):
        out.loc[mask, col] = rng.uniform(50, 200, size=n)
    out.loc[mask, "high"] = out.loc[mask, ["open", "high", "low", "close"]].max(axis=1)
    out.loc[mask, "low"] = out.loc[mask, ["open", "high", "low", "close"]].min(axis=1)
    out.loc[mask, "volume"] = rng.uniform(1e5, 1e7, size=n)
    out["returns"] = out.groupby(level="symbol")["close"].pct_change()
    return out


def main(alpha_dir, out_csv="smoke_test_results.csv"):
    df = synthetic_panel()
    dates = df.index.get_level_values("date").unique().sort_values()
    cut = dates[int(len(dates) * 0.75)]
    df_cut = corrupt_future(df, cut)
    pre_cut = df.index.get_level_values("date") <= cut

    rows = []
    for path in sorted(glob.glob(os.path.join(alpha_dir, "*.json"))):
        rec = json.loads(open(path, encoding="utf-8").read())
        aid = rec["metadata"]["alpha_id"]
        row = {"alpha_id": aid, "executes": False, "coverage": np.nan,
               "first_valid_date": "", "max_net_exposure": np.nan,
               "lookahead_violation": "", "n_cells_differing": "", "error": ""}
        try:
            fn = extract_and_exec(rec["raw_response"])
            sig = fn(df)
            if not isinstance(sig, pd.Series):
                raise TypeError(f"returned {type(sig).__name__}, expected Series")
            if not sig.index.equals(df.index):
                raise ValueError("returned index does not match df.index")

            row["executes"] = True
            finite = np.isfinite(sig.to_numpy(dtype=float))
            row["coverage"] = round(float(finite.mean()), 4)
            if finite.any():
                fv = sig[finite].index.get_level_values("date").min()
                row["first_valid_date"] = str(fv.date())
                wide = sig.unstack("symbol")
                net = wide.sum(axis=1, min_count=1).abs()
                row["max_net_exposure"] = round(float(net.max()), 8)

            sig_cut = fn(df_cut)
            a = sig[pre_cut].to_numpy(dtype=float)
            b = sig_cut[pre_cut].to_numpy(dtype=float)
            both_nan = np.isnan(a) & np.isnan(b)
            diff = ~(both_nan | np.isclose(a, b, rtol=1e-9, atol=1e-12,
                                           equal_nan=True))
            row["n_cells_differing"] = int(diff.sum())
            row["lookahead_violation"] = "YES" if diff.sum() else "no"
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        print(f"{aid:14s} exec={row['executes']!s:5s} cov={row['coverage']} "
              f"lookahead={row['lookahead_violation']} {row['error']}")

    res = pd.DataFrame(rows)
    res.to_csv(out_csv, index=False)
    print("\n--- SUMMARY ---")
    print(f"alphas tested         : {len(res)}")
    print(f"executed cleanly      : {int(res['executes'].sum())}")
    print(f"look-ahead violations : {int((res['lookahead_violation'] == 'YES').sum())}")
    print(f"median coverage       : {res['coverage'].median():.3f}")
    print(f"max |net exposure|    : {res['max_net_exposure'].max():.2e}")
    if (~res["executes"]).any():
        print("\nFAILURES:")
        print(res.loc[~res["executes"], ["alpha_id", "error"]].to_string(index=False))
    print(f"\nwritten to {out_csv}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "smoke_test_results.csv")
