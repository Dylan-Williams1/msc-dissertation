"""
Pull S&P 500 daily OHLCV, 1990 -> today, from yfinance in one shot.

Output: date, symbol, open, high, low, close, volume
        date as 'YYYY-MM-DD' strings, volume int64, sorted by (symbol, date).

auto_adjust=False so `close` is Yahoo's split-adjusted (not dividend-adjusted)
close, matching the existing panel's convention.

Requires: pip install yfinance pyarrow
"""

import json
import sys
import time
from datetime import datetime

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("pip install yfinance")

# ==============================================================================
OUT_PATH = r"C:\University\Master's\Diss\Dissertation\data\daily_ohlcv.parquet"
START = "1990-01-01"
END = datetime.today().strftime("%Y-%m-%d")   # yfinance end is exclusive
CHUNK = 50
PAUSE = 1.5
RETRIES = 3
COLS = ["date", "symbol", "open", "high", "low", "close", "volume"]
SYMBOLS = [
    "A", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE",
    "ADI", "ADM", "ADP", "ADSK", "AEE", "AEP", "AES", "AFL",
    "AIG", "AIZ", "AJG", "AKAM", "ALB", "ALGN", "ALL", "ALLE",
    "AMAT", "AMCR", "AMD", "AME", "AMGN", "AMP", "AMT", "AMZN",
    "ANET", "AON", "AOS", "APA", "APD", "APH", "APO", "APP",
    "APTV", "ARE", "ARES", "ATO", "AVGO", "AVY", "AWK", "AXON",
    "AXP", "AZO", "BA", "BAC", "BALL", "BAX", "BBY", "BDX",
    "BEN", "BF.B", "BG", "BIIB", "BKNG", "BKR", "BLDR", "BLK",
    "BMY", "BNY", "BR", "BRK.B", "BRO", "BSX", "BX", "BXP",
    "C", "CAH", "CARR", "CASY", "CAT", "CB", "CBOE", "CBRE",
    "CCI", "CCL", "CDNS", "CDW", "CEG", "CF", "CFG", "CHD",
    "CHRW", "CHTR", "CI", "CIEN", "CINF", "CL", "CLX", "CMCSA",
    "CME", "CMG", "CMI", "CMS", "CNC", "CNP", "COF", "COHR",
    "COIN", "COO", "COP", "COR", "COST", "CPAY", "CPRT", "CPT",
    "CRH", "CRL", "CRM", "CRWD", "CSCO", "CSGP", "CSX", "CTAS",
    "CTSH", "CTVA", "CVNA", "CVS", "CVX", "D", "DAL", "DASH",
    "DD", "DDOG", "DE", "DECK", "DELL", "DG", "DGX", "DHI",
    "DHR", "DIS", "DLR", "DLTR", "DOC", "DOV", "DOW", "DPZ",
    "DRI", "DTE", "DUK", "DVA", "DVN", "DXCM", "EBAY", "ECHO",
    "ECL", "ED", "EFX", "EG", "EIX", "EL", "ELV", "EME",
    "EMR", "EOG", "EQIX", "EQT", "ERIE", "ES", "ESS", "ETN",
    "ETR", "EVRG", "EW", "EXC", "EXE", "EXPD", "EXPE", "EXR",
    "F", "FANG", "FAST", "FCX", "FDS", "FDX", "FDXF", "FE",
    "FERG", "FFIV", "FICO", "FIS", "FISV", "FITB", "FIX", "FLEX",
    "FOX", "FOXA", "FRT", "FSLR", "FTNT", "FTV", "GD", "GDDY",
    "GE", "GEHC", "GEN", "GEV", "GILD", "GIS", "GL", "GLW",
    "GM", "GNRC", "GOOG", "GOOGL", "GPC", "GPN", "GRMN", "GS",
    "GWW", "HAL", "HAS", "HBAN", "HCA", "HD", "HIG", "HII",
    "HLT", "HON", "HONA", "HOOD", "HPE", "HPQ", "HRL", "HSIC",
    "HST", "HSY", "HUBB", "HUM", "HWM", "IBKR", "IBM", "ICE",
    "IDXX", "IEX", "IFF", "INCY", "INTC", "INTU", "INVH", "IP",
    "IQV", "IR", "IRM", "ISRG", "IT", "ITW", "IVZ", "J",
    "JBHT", "JBL", "JCI", "JKHY", "JNJ", "JPM", "KDP", "KEY",
    "KEYS", "KHC", "KIM", "KKR", "KLAC", "KMB", "KMI", "KO",
    "KR", "KVUE", "L", "LDOS", "LEN", "LH", "LHX", "LII",
    "LIN", "LITE", "LLY", "LMT", "LNT", "LOW", "LRCX", "LULU",
    "LUV", "LVS", "LYB", "LYV", "MA", "MAA", "MAR", "MAS",
    "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MET", "META",
    "MGM", "MKC", "MLM", "MMM", "MNST", "MO", "MOS", "MPC",
    "MPWR", "MRK", "MRNA", "MRSH", "MRVL", "MS", "MSCI", "MSFT",
    "MSI", "MTB", "MTD", "MU", "NCLH", "NDAQ", "NDSN", "NEE",
    "NEM", "NFLX", "NI", "NKE", "NOC", "NOW", "NRG", "NSC",
    "NTAP", "NTRS", "NUE", "NVDA", "NVR", "NWS", "NWSA", "NXPI",
    "O", "ODFL", "OKE", "OMC", "ON", "ORCL", "ORLY", "OTIS",
    "OXY", "PANW", "PAYX", "PCAR", "PCG", "PEG", "PEP", "PFE",
    "PFG", "PG", "PGR", "PH", "PHM", "PKG", "PLD", "PLTR",
    "PM", "PNC", "PNR", "PNW", "PODD", "PPG", "PPL", "PRU",
    "PSA", "PSKY", "PSX", "PTC", "PWR", "PYPL", "Q", "QCOM",
    "RCL", "RDDT", "REG", "REGN", "RF", "RJF", "RL", "RMD",
    "ROK", "ROL", "ROP", "ROST", "RSG", "RTX", "RVTY", "SBAC",
    "SBUX", "SCHW", "SHW", "SJM", "SLB", "SMCI", "SNA", "SNDK",
    "SNPS", "SO", "SOLV", "SPG", "SPGI", "SRE", "STE", "STLD",
    "STT", "STX", "STZ", "SW", "SWK", "SWKS", "SYF", "SYK",
    "SYY", "T", "TAP", "TDG", "TDY", "TECH", "TEL", "TER",
    "TFC", "TGT", "TJX", "TKO", "TMO", "TMUS", "TPL", "TPR",
    "TRGP", "TRMB", "TROW", "TRV", "TSCO", "TSLA", "TSN", "TT",
    "TTD", "TTWO", "TXN", "TXT", "TYL", "UAL", "UBER", "UDR",
    "UHS", "ULTA", "UNH", "UNP", "UPS", "URI", "USB", "V",
    "VEEV", "VICI", "VLO", "VLTO", "VMC", "VMRK", "VRSK", "VRSN",
    "VRT", "VRTX", "VST", "VTR", "VTRS", "VZ", "WAB", "WAT",
    "WBD", "WDAY", "WDC", "WEC", "WELL", "WFC", "WM", "WMB",
    "WMT", "WRB", "WSM", "WST", "WTW", "WY", "WYNN", "XEL",
    "XOM", "XYL", "XYZ", "YUM", "ZBH", "ZBRA", "ZTS",
]
# ==============================================================================


def get_symbols():
    """Frozen S&P 500 constituent list (Wikipedia snapshot). Dots -> hyphens
    for Yahoo: BRK.B -> BRK-B, BF.B -> BF-B."""
    syms = sorted(s.strip().replace(".", "-") for s in SYMBOLS)
    assert len(syms) == len(set(syms)), "duplicate symbols in SYMBOLS"
    print(f"symbols: {len(syms)}")
    return syms


def download(symbols):
    frames, failed = [], []
    chunks = [symbols[i:i + CHUNK] for i in range(0, len(symbols), CHUNK)]

    for ci, chunk in enumerate(chunks, 1):
        raw = None
        for attempt in range(1, RETRIES + 1):
            try:
                raw = yf.download(tickers=chunk, start=START, end=END,
                                  auto_adjust=False, group_by="ticker",
                                  threads=True, progress=False, timeout=60)
                if raw is not None and not raw.empty:
                    break
            except Exception as exc:
                print(f"  chunk {ci} attempt {attempt}: {type(exc).__name__}")
            time.sleep(5 * attempt)

        if raw is None or raw.empty:
            failed.extend(chunk)
            print(f"  chunk {ci}/{len(chunks)}: no data")
            continue

        for sym in chunk:
            try:
                sub = raw[sym] if isinstance(raw.columns, pd.MultiIndex) else raw
            except KeyError:
                failed.append(sym)
                continue
            sub = sub.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
            if sub.empty:
                failed.append(sym)
                continue
            frames.append(pd.DataFrame({
                "date": sub.index.strftime("%Y-%m-%d"),
                "symbol": sym,
                "open": sub["Open"].to_numpy(dtype=float),
                "high": sub["High"].to_numpy(dtype=float),
                "low": sub["Low"].to_numpy(dtype=float),
                "close": sub["Close"].to_numpy(dtype=float),
                "volume": sub["Volume"].to_numpy(dtype=float),
            }))

        print(f"  chunk {ci}/{len(chunks)} done")
        if ci < len(chunks):
            time.sleep(PAUSE)

    if not frames:
        sys.exit("nothing downloaded")

    df = pd.concat(frames, ignore_index=True)
    df["volume"] = df["volume"].round().astype("int64")
    return df[COLS], sorted(set(failed))

def salvage(failed):
    """Retry failed tickers one at a time. yfinance absorbs per-ticker errors
    inside a batch, so they never reach the chunk-level retry."""
    frames, still = [], []
    print(f"\nsalvage: retrying {len(failed)} ticker(s) individually")
    for sym in failed:
        sub = None
        for attempt in range(1, 4):
            try:
                sub = yf.Ticker(sym).history(start=START, end=END,
                                             auto_adjust=False, timeout=60)
                if sub is not None and not sub.empty:
                    break
            except Exception:
                pass
            time.sleep(5 * attempt)
        if sub is None or sub.empty:
            still.append(sym)
            print(f"  {sym}: still failing")
            continue
        sub = sub.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        frames.append(pd.DataFrame({
            "date": sub.index.strftime("%Y-%m-%d"), "symbol": sym,
            "open": sub["Open"].to_numpy(dtype=float),
            "high": sub["High"].to_numpy(dtype=float),
            "low": sub["Low"].to_numpy(dtype=float),
            "close": sub["Close"].to_numpy(dtype=float),
            "volume": sub["Volume"].to_numpy(dtype=float),
        }))
        print(f"  {sym}: recovered ({len(sub):,} rows)")
    if not frames:
        return None, still
    out = pd.concat(frames, ignore_index=True)
    out["volume"] = out["volume"].round().astype("int64")
    return out[COLS], still


def main():
    symbols = get_symbols()
    print(f"\ndownloading {START} .. {END}")
    df, failed = download(symbols)

    if failed:
        extra, failed = salvage(failed)
        if extra is not None:
            df = pd.concat([df, extra], ignore_index=True)

    df = (df.drop_duplicates(subset=["date", "symbol"], keep="last")
            .sort_values(["symbol", "date"])
            .reset_index(drop=True))
    df["date"] = df["date"].astype(str)
    df["symbol"] = df["symbol"].astype(str)

    # quick sanity pass
    r = df.groupby("symbol")["close"].pct_change()
    jumps = df.loc[r.abs() > 0.6, ["date", "symbol"]]
    print("\n--- sanity ---")
    print(f"rows {len(df):,} | symbols {df['symbol'].nunique()} | "
          f"{df['date'].min()} .. {df['date'].max()}")
    print(f"OHLC violations : {int((df.high < df.low).sum())}")
    print(f"non-positive px : {int((df[['open','high','low','close']] <= 0).any(axis=1).sum())}")
    print(f"moves >60%      : {len(jumps)}"
          + (f"  ({', '.join(sorted(jumps['symbol'].unique())[:15])})" if len(jumps) else ""))
    if failed:
        print(f"failed tickers  : {len(failed)}  ({', '.join(failed)})")

    df.to_parquet(OUT_PATH, index=False, compression="snappy")
    with open(OUT_PATH.replace(".parquet", "_meta.json"), "w") as fh:
        json.dump({"pulled": datetime.now().isoformat(timespec="seconds"),
                   "source": "yfinance auto_adjust=False",
                   "symbols": symbols,
                   "start": START, "end": END,
                   "n_symbols": int(df["symbol"].nunique()),
                   "n_rows": int(len(df)),
                   "failed": failed}, fh, indent=2)

    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
