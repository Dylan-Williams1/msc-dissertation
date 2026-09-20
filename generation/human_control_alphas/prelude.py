"""Operator library emitted (nested) into every control-alpha code block.

Nested deliberately: phase1_evaluation.py runs
    exec(python_code, globals(), local_namespace)
which places module-level defs in local_namespace while giving the defined
functions the *harness* globals as __globals__. A module-level helper is
therefore invisible to generate_alpha at call time (NameError). Nesting the
helpers inside generate_alpha resolves them via closure and keeps every alpha
file self-contained and drop-in compatible with the unmodified harness.
"""

PRELUDE = '''    # ---------- operator library (Kakushadze 2016, Appendix A.1) ----------
    def _f(d):
        return int(np.floor(float(d)))

    def _wide(_df):
        _w = {}
        for _c in ("open", "high", "low", "close", "volume"):
            _w[_c] = _df[_c].unstack("symbol").sort_index()
        if "returns" in _df.columns:
            _w["returns"] = _df["returns"].unstack("symbol").sort_index()
        else:
            _w["returns"] = _w["close"].pct_change()
        return (_w["open"], _w["high"], _w["low"],
                _w["close"], _w["volume"], _w["returns"])

    def _finalize(a, idx, neutralize=True, min_names=20):
        a = a.replace([np.inf, -np.inf], np.nan)
        if min_names:
            a = a.mask(a.notna().sum(axis=1) < int(min_names))
        if neutralize:
            a = a.sub(a.mean(axis=1), axis=0)
            g = a.abs().sum(axis=1)
            a = a.div(g.replace(0.0, np.nan), axis=0)
        s = a.stack()
        s.index.names = ["date", "symbol"]
        return s.reindex(idx)

    def _valid(*args):
        m = None
        for a in args:
            if isinstance(a, pd.DataFrame):
                m = a.notna() if m is None else (m & a.notna())
        return m

    def _cmp(a, b, op):
        m = _valid(a, b)
        if op == "lt":
            r = (a < b)
        elif op == "gt":
            r = (a > b)
        elif op == "le":
            r = (a <= b)
        else:
            r = (a >= b)
        r = r.astype(float)
        return r.where(m) if m is not None else r

    def lt(a, b):
        return _cmp(a, b, "lt")

    def gt(a, b):
        return _cmp(a, b, "gt")

    def le(a, b):
        return _cmp(a, b, "le")

    def ge(a, b):
        return _cmp(a, b, "ge")

    def where(cond, a, b):
        idx, cols = cond.index, cond.columns
        A = a.reindex(index=idx, columns=cols) if isinstance(a, pd.DataFrame) else a
        B = b.reindex(index=idx, columns=cols) if isinstance(b, pd.DataFrame) else b
        out = pd.DataFrame(np.where(cond.fillna(0.0) > 0.5, A, B),
                           index=idx, columns=cols)
        return out.where(cond.notna())

    def emin(a, b):
        b2 = b.reindex(index=a.index, columns=a.columns)
        return pd.DataFrame(np.minimum(a.to_numpy(dtype=float),
                                       b2.to_numpy(dtype=float)),
                            index=a.index, columns=a.columns)

    def emax(a, b):
        b2 = b.reindex(index=a.index, columns=a.columns)
        return pd.DataFrame(np.maximum(a.to_numpy(dtype=float),
                                       b2.to_numpy(dtype=float)),
                            index=a.index, columns=a.columns)

    def rank(x):
        return x.rank(axis=1, pct=True)

    def delay(x, d):
        return x.shift(_f(d))

    def delta(x, d):
        return x - x.shift(_f(d))

    def ts_sum(x, d):
        d = _f(d)
        return x.rolling(d, min_periods=d).sum()

    def ts_mean(x, d):
        d = _f(d)
        return x.rolling(d, min_periods=d).mean()

    def stddev(x, d):
        d = _f(d)
        return x.rolling(d, min_periods=d).std()

    def ts_min(x, d):
        d = _f(d)
        return x.rolling(d, min_periods=d).min()

    def ts_max(x, d):
        d = _f(d)
        return x.rolling(d, min_periods=d).max()

    def ts_rank(x, d):
        d = _f(d)
        try:
            return x.rolling(d, min_periods=d).rank(pct=True)
        except AttributeError:
            return x.rolling(d, min_periods=d).apply(
                lambda v: (v <= v[-1]).sum() / len(v), raw=True)

    def _arg(x, d, fn):
        d = _f(d)
        v = x.to_numpy(dtype=float)
        out = np.full(v.shape, np.nan)
        for i in range(d - 1, v.shape[0]):
            w = v[i - d + 1:i + 1, :]
            ok = ~np.isnan(w).any(axis=0)
            if ok.any():
                out[i, ok] = fn(w[:, ok], axis=0) + 1.0
        return pd.DataFrame(out, index=x.index, columns=x.columns)

    def ts_argmax(x, d):
        return _arg(x, d, np.argmax)

    def ts_argmin(x, d):
        return _arg(x, d, np.argmin)

    def correlation(x, y, d):
        d = _f(d)
        return x.rolling(d, min_periods=d).corr(y)

    def covariance(x, y, d):
        d = _f(d)
        return x.rolling(d, min_periods=d).cov(y)

    def scale(x, a=1.0):
        return x.div(x.abs().sum(axis=1).replace(0.0, np.nan), axis=0) * a

    def decay_linear(x, d):
        d = _f(d)
        w = np.arange(d, 0, -1, dtype=float)
        w = w / w.sum()
        out = None
        for k in range(d):
            term = x.shift(k) * w[k]
            out = term if out is None else out + term
        return out

    def product(x, d):
        d = _f(d)
        return x.rolling(d, min_periods=d).apply(np.prod, raw=True)

    def signedpower(x, a):
        return np.sign(x) * (np.abs(x) ** a)

    def power(x, y):
        return np.sign(x) * (np.abs(x) ** y)

    def log(x):
        return np.log(x.where(x > 0))

    def sign(x):
        return np.sign(x).where(x.notna())
'''

TEMPLATE = '''import numpy as np
import pandas as pd


def generate_alpha(df, {signature}):
{prelude}
    open_, high, low, close, volume, returns = _wide(df)

    def adv(d):
        # Source A.3 defines adv{{d}} on dollar volume. Several alphas compare or
        # divide it against share `volume` directly, which is dimensionally
        # inconsistent and degenerate under the dollar reading. Share volume is
        # the default; set adv_dollar=True for the literal reading.
        return ts_mean(close * volume if adv_dollar else volume, d)

    # ---------- Kakushadze Alpha#{n} ----------
{body}
    return _finalize(alpha, df.index, neutralize=neutralize, min_names=min_names)
'''
