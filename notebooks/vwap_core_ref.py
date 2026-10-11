"""Scratch dev/test module for VWAP Bounce core math. Logic gets ported into the real notebook
once verified here - not part of the final deliverable."""
from __future__ import annotations
import numpy as np
import pandas as pd

MAD_TO_SIGMA = 1.4826  # standard consistency constant for a normal distribution


def session_vwap(h, l, c, v):
    """VWAP(t) = cumsum((H+L+C)/3 * V) / cumsum(V), cumulative within the day.
    h/l/c/v: 1-D arrays, one row per minute, in time order for ONE ticker ONE day."""
    typical = (h + l + c) / 3.0
    cv = np.cumsum(typical * v)
    cvol = np.cumsum(v)
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = cv / cvol
    vwap[cvol <= 0] = np.nan
    return vwap


def expanding_vwap_sigma(c, vwap, min_bars=5):
    """Expanding (not rolling) std of (close - VWAP), NaN until min_bars bars have occurred.
    ddof=1, matching every other sigma in this codebase."""
    resid = c - vwap
    n = len(resid)
    out = np.full(n, np.nan)
    # expanding sum / sumsq for O(n) instead of O(n^2) recompute
    valid = np.isfinite(resid)
    x = np.where(valid, resid, 0.0)
    cnt = np.cumsum(valid.astype(np.int64))
    s1 = np.cumsum(x)
    s2 = np.cumsum(x * x)
    for t in range(n):
        k = cnt[t]
        if k >= min_bars:
            mean = s1[t] / k
            var = (s2[t] - k * mean * mean) / (k - 1)
            out[t] = np.sqrt(var) if var > 0 else np.nan
    return out


def tech_z(c, vwap, vwap_sigma, clip=8.0):
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (c - vwap) / vwap_sigma
    return np.clip(z, -clip, clip)


def fit_beta_market_sector(stock_ret, market_ret, sector_ret):
    """OLS stock_ret ~ 1 + market_ret + sector_ret (WITH intercept, for an unbiased beta fit),
    but the intercept itself is NOT returned/used - `expected` downstream is purely
    beta_market*move + beta_sector*move, per the spec's own formula (no alpha term). Returns
    (beta_market, beta_sector, r2) or (nan,nan,nan) if too few points or singular design.
    If sector_ret is degenerate (e.g. identical to market_ret - ticker's own sector ETF
    resolved to the market ETF), sector term is dropped and beta_sector=0 is fit as a pure
    market regression instead, r2 from that reduced model."""
    y = np.asarray(stock_ret, float)
    xm = np.asarray(market_ret, float)
    xs = np.asarray(sector_ret, float)
    ok = np.isfinite(y) & np.isfinite(xm) & np.isfinite(xs)
    n = int(ok.sum())
    if n < 30:
        return np.nan, np.nan, np.nan
    y, xm, xs = y[ok], xm[ok], xs[ok]

    def _r2(y, yhat):
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # degeneracy check: sector regressor collinear with market (or with the constant)
    if np.std(xs) < 1e-12 or np.corrcoef(xm, xs)[0, 1] > 0.999:
        X = np.column_stack([np.ones(n), xm])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        bm = coef[1]
        yhat = X @ coef
        return float(bm), 0.0, float(_r2(y, yhat))

    X = np.column_stack([np.ones(n), xm, xs])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    return float(coef[1]), float(coef[2]), float(_r2(y, yhat))


def mad_sigma(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size < 5:
        return np.nan
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    return float(mad * MAD_TO_SIGMA) if mad > 0 else np.nan


def pct_rank(values):
    """Percentile rank in [0,1], ties averaged - np.nan stays np.nan (not ranked)."""
    s = pd.Series(values, dtype=float)
    return s.rank(pct=True, na_option="keep").to_numpy()


def simulate_position(entry_idx, entry_px, fade_side, sigma_at_entry,
                       high, low, close, vwap, min_add_gap_bars, eod_idx,
                       add_mult=(1.0, 2.0), target_stop_mult=1.5):
    """One ticker's position lifecycle from entry_idx (the bar it fills AT, i.e. entry_px IS
    high[entry_idx..]/low/close[entry_idx]'s own open - caller passes entry_idx as the first bar
    INCLUDED in monitoring, entry_px as that bar's open) through target/stop/EOD, with up to two
    adverse add-ons. Bar-level first-touch using High/Low (not just Close) - "first-touch по 1 хв".

    Tie-break when a single bar's range contains BOTH the stop and the (that bar's own) VWAP
    target: STOP WINS (conservative backtest assumption - this is MY OWN choice, the spec does not
    resolve same-bar ambiguity; documented here and in the notebook's own comment).

    `sigma_at_entry` anchors BOTH the stop and the two add-on offsets for the position's entire
    life (frozen at signal time, not re-evaluated bar to bar) - target alone is dynamic (today's
    VWAP, which still moves after entry).

    min_add_gap_bars: minimum bars between fills (>=15 real minutes, caller converts to bar count -
    1 bar = 1 minute here, so this IS minutes).

    Returns a dict: exit_idx, exit_px, cls ("TARGET"/"STOP"/"EOD"), avg_entry_px, n_fills (1..3).
    """
    fills = [(entry_idx, entry_px)]
    avg_px = entry_px
    stop_px = entry_px - fade_side * target_stop_mult * sigma_at_entry
    add_done = 0
    last_fill_idx = entry_idx

    n = len(close)
    t = entry_idx
    while True:
        if t > eod_idx or t >= n:
            # ran off the end of the day's own data without a touch - close at the last bar seen
            last = min(t - 1, n - 1, eod_idx)
            return dict(exit_idx=last, exit_px=close[last], cls="EOD",
                        avg_entry_px=avg_px, n_fills=len(fills))

        hi, lo, vw = high[t], low[t], vwap[t]
        stop_touched = (hi >= stop_px) if fade_side < 0 else (lo <= stop_px)
        target_touched = (lo <= vw <= hi) if np.isfinite(vw) else False

        if stop_touched:
            return dict(exit_idx=t, exit_px=stop_px, cls="STOP", avg_entry_px=avg_px, n_fills=len(fills))
        if target_touched:
            return dict(exit_idx=t, exit_px=vw, cls="TARGET", avg_entry_px=avg_px, n_fills=len(fills))

        if t == eod_idx:
            return dict(exit_idx=t, exit_px=close[t], cls="EOD", avg_entry_px=avg_px, n_fills=len(fills))

        if add_done < len(add_mult) and (t - last_fill_idx) >= min_add_gap_bars:
            add_level = entry_px - fade_side * add_mult[add_done] * sigma_at_entry
            add_triggered = (hi >= add_level) if fade_side < 0 else (lo <= add_level)
            if add_triggered:
                fills.append((t, add_level))
                avg_px = float(np.mean([p for _, p in fills]))
                stop_px = avg_px - fade_side * target_stop_mult * sigma_at_entry
                add_done += 1
                last_fill_idx = t

        t += 1
