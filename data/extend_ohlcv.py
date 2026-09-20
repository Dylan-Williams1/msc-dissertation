"""
Extend daily_ohlcv.parquet from 2026-07-15 to 2026-09-08 using yfinance.

The existing panel is SPLIT-ADJUSTED BUT NOT DIVIDEND-ADJUSTED (verified against
AAPL's 2020-08-31 4:1 split: 2020-08-28 close 124.81 = 499.23/4, while
2020-09-02 close 131.40 is the raw close with no dividend back-adjustment).
That is Yahoo's `Close` column, NOT `Adj Close`, so we fetch with
auto_adjust=False and discard Adj Close.

Two things this script guards against:

  1. The stored 2026-07-16 rows cover only 20 of 503 symbols -- a truncated
     download. Those rows are DROPPED and the day is refetched in full.

  2. Yahoo restates history after a split. If a ticker split inside the fetch
     window, Yahoo's series is on the new basis while the parquet is on the
     old one, and naively appending fabricates a huge one-day return. Splits
     are detected and the stored history is back-adjusted before the join.

Output format is byte-compatible with the input: columns
[date, symbol, open, high, low, close, volume], date as 'YYYY-MM-DD' strings,
volume int64, sorted by (symbol, date), RangeIndex, no index column written.

Requires: pandas, pyarrow, yfinance   ->   pip install yfinance pyarrow
"""

import os
import sys
import time

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("yfinance not installed.  pip install yfinance")

# ==============================================================================
# CONFIG
# ==============================================================================
IN_PATH = r"C:\University\Master's\Diss\Dissertation\data\daily_ohlcv.parquet"
OUT_PATH = r"C:\University\Master's\Diss\Dissertation\data\daily_ohlcv_to_20260908.parquet"

KEEP_THROUGH = "2026-07-15"   # last fully-populated day in the stored panel
FETCH_START = "2026-07-16"    # first day to refetch (stored day was truncated)
FETCH_END = "2026-09-09"      # inclusive

CHUNK_SIZE = 50               # tickers per yfinance call
CHUNK_PAUSE = 1.5             # seconds between chunks
MAX_RETRIES = 3

COLS = ["date", "symbol", "open", "high", "low", "close", "volume"]


# ==============================================================================
# LOAD
# ==============================================================================
def load_existing(path, keep_through):
    if not os.path.isfile(path):
        sys.exit(f"input parquet not found: {path}")
    df = pd.read_parquet(path)

    missing = [c for c in COLS if c not in df.columns]
    if missing:
        sys.exit(f"input parquet missing expected columns: {missing}")

    df["date"] = df["date"].astype(str)
    df["symbol"] = df["symbol"].astype(str)

    n_before = len(df)
    max_date = df["date"].max()
    df = df[df["date"] <= keep_through].copy()
    dropped = n_before - len(df)

    print(f"loaded {n_before:,} rows, {df['symbol'].nunique()} symbols, "
          f"through {max_date}")
    if dropped:
        print(f"  dropped {dropped:,} rows after {keep_through} "
              f"(truncated final day; refetching in full)")

    counts = df.groupby("date")["symbol"].count()
    print(f"  last stored day {counts.index[-1]}: {counts.iloc[-1]} symbols")
    return df


# ==============================================================================
# FETCH
# ==============================================================================
def fetch_window(symbols, start, end):
    """Download [start, end] inclusive. Returns (long_df, splits_dict, failures)."""
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    frames, splits, failures = [], {}, []

    chunks = [symbols[i:i + CHUNK_SIZE] for i in range(0, len(symbols), CHUNK_SIZE)]
    for ci, chunk in enumerate(chunks, 1):
        raw = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw = yf.download(
                    tickers=chunk, start=start, end=end_excl,
                    auto_adjust=False,   # keep Yahoo's split-adjusted Close
                    actions=True,        # exposes the Stock Splits column
                    group_by="ticker", threads=True,
                    progress=False, timeout=30,
                )
                break
            except Exception as exc:
                wait = 5 * attempt
                print(f"  chunk {ci}/{len(chunks)} attempt {attempt} failed "
                      f"({type(exc).__name__}); retrying in {wait}s")
                time.sleep(wait)
        if raw is None or raw.empty:
            failures.extend(chunk)
            print(f"  chunk {ci}/{len(chunks)}: NO DATA for {len(chunk)} tickers")
            continue

        for sym in chunk:
            try:
                sub = raw[sym] if isinstance(raw.columns, pd.MultiIndex) else raw
            except KeyError:
                failures.append(sym)
                continue

            sub = sub.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
            if sub.empty:
                failures.append(sym)
                continue

            if "Stock Splits" in sub.columns:
                sp = sub["Stock Splits"]
                sp = sp[(sp.notna()) & (sp != 0.0)]
                if len(sp):
                    splits[sym] = [(d.strftime("%Y-%m-%d"), float(v))
                                   for d, v in sp.items()]

            frames.append(pd.DataFrame({
                "date": sub.index.strftime("%Y-%m-%d"),
                "symbol": sym,
                "open": sub["Open"].to_numpy(dtype=float),
                "high": sub["High"].to_numpy(dtype=float),
                "low": sub["Low"].to_numpy(dtype=float),
                "close": sub["Close"].to_numpy(dtype=float),
                "volume": sub["Volume"].to_numpy(dtype=float),
            }))

        print(f"  chunk {ci}/{len(chunks)} done ({len(chunk)} tickers)")
        if ci < len(chunks):
            time.sleep(CHUNK_PAUSE)

    if not frames:
        sys.exit("no data returned for any ticker; aborting")

    new = pd.concat(frames, ignore_index=True)
    new = new[new["volume"].notna()]
    new["volume"] = new["volume"].round().astype("int64")
    return new[COLS], splits, sorted(set(failures))


# ==============================================================================
# SPLIT BACK-ADJUSTMENT
# ==============================================================================
def back_adjust(hist, splits):
    """Rescale stored history for tickers that split inside the fetch window.

    Every stored row predates the window, so the whole series for an affected
    ticker is rescaled: prices / R, volume * R, cumulative over multiple splits.
    """
    if not splits:
        print("no splits detected in the fetch window")
        return hist

    print(f"\n!! {len(splits)} ticker(s) split inside the window; "
          f"back-adjusting stored history")
    for sym, events in sorted(splits.items()):
        ratio = float(np.prod([r for _, r in events]))
        detail = ", ".join(f"{d} x{r:g}" for d, r in events)
        if ratio <= 0 or not np.isfinite(ratio):
            sys.exit(f"  {sym}: unusable split ratio {ratio} ({detail})")
        m = hist["symbol"] == sym
        if not m.any():
            print(f"  {sym}: {detail} -- not in stored panel, skipped")
            continue
        hist.loc[m, ["open", "high", "low", "close"]] /= ratio
        hist.loc[m, "volume"] = (hist.loc[m, "volume"] * ratio).round().astype("int64")
        print(f"  {sym}: {detail} -> prices /{ratio:g}, volume x{ratio:g} "
              f"({int(m.sum()):,} rows)")
    return hist


# ==============================================================================
# VALIDATE
# ==============================================================================
def validate(hist, new, combined):
    print("\n--- validation ---")
    ok = True

    counts = combined.groupby("date")["symbol"].count()
    win = counts[counts.index >= FETCH_START]
    print(f"fetched days: {len(win)}  |  symbols/day min {win.min()} "
          f"median {int(win.median())} max {win.max()}")
    thin = win[win < 0.9 * win.median()]
    if len(thin):
        ok = False
        print(f"  ! {len(thin)} day(s) below 90% coverage: "
              f"{', '.join(thin.index[:5])}")

    # A clean join shows no outsized return across the seam.
    seam = combined[combined["date"].isin([KEEP_THROUGH, FETCH_START])]
    piv = seam.pivot(index="date", columns="symbol", values="close")
    if len(piv) == 2:
        ret = (piv.iloc[1] / piv.iloc[0] - 1.0).dropna()
        bad = ret[ret.abs() > 0.35]
        print(f"seam return {KEEP_THROUGH} -> {FETCH_START}: "
              f"median {ret.median():+.4f}, |max| {ret.abs().max():.4f}")
        if len(bad):
            ok = False
            print(f"  ! {len(bad)} ticker(s) moved >35% across the seam "
                  f"(unhandled split?): {', '.join(bad.index[:8])}")

    dup = combined.duplicated(subset=["date", "symbol"]).sum()
    if dup:
        ok = False
        print(f"  ! {dup} duplicate (date, symbol) rows")

    for c, want in [("open", "float64"), ("high", "float64"), ("low", "float64"),
                    ("close", "float64"), ("volume", "int64")]:
        if str(combined[c].dtype) != want:
            ok = False
            print(f"  ! {c} dtype {combined[c].dtype}, expected {want}")

    if combined[COLS].isna().any().any():
        ok = False
        print("  ! nulls present")

    print("validation:", "PASS" if ok else "REVIEW WARNINGS ABOVE")
    return ok


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    hist = load_existing(IN_PATH, KEEP_THROUGH)
    symbols = sorted(hist["symbol"].unique())

    print(f"\nfetching {FETCH_START} .. {FETCH_END} for {len(symbols)} symbols")
    new, splits, failures = fetch_window(symbols, FETCH_START, FETCH_END)
    print(f"fetched {len(new):,} rows, {new['symbol'].nunique()} symbols, "
          f"{new['date'].nunique()} trading days")

    if failures:
        print(f"\n!! {len(failures)} ticker(s) returned nothing: "
              f"{', '.join(failures)}")
        print("   Likely delisted, renamed, or a Yahoo symbol mismatch.")
        print("   These will be absent from the extension window, making the")
        print("   panel ragged. Resolve before using for cross-sectional work.")

    hist = back_adjust(hist, splits)

    new = new[new["date"] > KEEP_THROUGH]
    combined = pd.concat([hist[COLS], new[COLS]], ignore_index=True)
    combined = (combined
                .drop_duplicates(subset=["date", "symbol"], keep="last")
                .sort_values(["symbol", "date"])
                .reset_index(drop=True))

    combined["date"] = combined["date"].astype(str)
    combined["symbol"] = combined["symbol"].astype(str)
    combined["volume"] = combined["volume"].astype("int64")

    validate(hist, new, combined)

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    combined.to_parquet(OUT_PATH, index=False, compression="snappy")

    print(f"\nwrote {len(combined):,} rows "
          f"({combined['date'].min()} .. {combined['date'].max()}, "
          f"{combined['symbol'].nunique()} symbols)")
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
