"""Robust control charts on within-company change (plan step 3). Levels are traits, so charts watch movement.

For a monthly series (the score, a category score, or a group mean):
  baseline_t = median of the series over the window ending BASELINE_LAG months before t (own history, no look-ahead)
  scale_t    = max(1.4826 * MAD of that window, floor)      floor is fitted on train (monitor_params.json)
  z_t        = (x_t - baseline_t) / scale_t
  EWMA_t     = LAMBDA * z_t + (1 - LAMBDA) * EWMA_{t-1};  signal when |EWMA| > L * sqrt(LAMBDA / (2 - LAMBDA)) (in z units)
  CUSUM      one-sided on z with reference K, signal above H (capped at 2H for readability)
A month signals low/high when the EWMA or CUSUM crosses. The persistence rule (PERSIST_K of the last PERSIST_N
months signalling the same way) turns signals into alerts, so a one-month dip is not an alert.
Group charts use the same code with the floor divided by sqrt(n): a mean of n companies is less noisy, and small
groups get wide limits (funnel-style).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAMBDA = 0.3
L_EWMA = 3.0
CUSUM_K = 0.5
CUSUM_H = 4.0
BASELINE_LAG = 3      # the last 3 months are never part of their own baseline
BASELINE_WINDOW = 12  # months in the baseline window
MIN_PRIOR = 4         # months needed in the baseline window
PERSIST_K, PERSIST_N = 3, 4
EWMA_LIMIT = L_EWMA * np.sqrt(LAMBDA / (2 - LAMBDA))

METHOD = {"name": "robust_ewma_cusum", "params": {
    "lambda": LAMBDA, "ewma_L": L_EWMA, "cusum_k": CUSUM_K, "cusum_h": CUSUM_H, "baseline_lag_months": BASELINE_LAG,
    "baseline_window_months": BASELINE_WINDOW, "min_baseline_months": MIN_PRIOR, "persistence": f"{PERSIST_K} de los últimos {PERSIST_N}"}}


def raw_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Baseline median and robust scale (before the floor) per month, from the lagged window. NaN where too little history."""
    n = len(x)
    med = np.full(n, np.nan)
    sc = np.full(n, np.nan)
    for t in range(n):
        hi = t - BASELINE_LAG + 1
        lo = max(0, hi - BASELINE_WINDOW)
        w = x[lo:hi] if hi > 0 else x[:0]
        w = w[np.isfinite(w)]
        if len(w) >= MIN_PRIOR:
            med[t] = np.median(w)
            sc[t] = 1.4826 * np.median(np.abs(w - med[t]))
    return med, sc


def chart(x, floor: float | np.ndarray) -> dict[str, np.ndarray]:
    """Control chart arrays for one series `x` (ordered months). `floor`: scalar or per-month array."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    floor = np.broadcast_to(np.asarray(floor, dtype=float), (n,))
    med, sc = raw_scale(x)
    scale = np.maximum(sc, floor)
    z = np.where(np.isfinite(med) & np.isfinite(x), (x - med) / scale, np.nan)
    ew = np.full(n, np.nan)
    c_hi = np.full(n, np.nan)
    c_lo = np.full(n, np.nan)
    e = c_p = c_m = 0.0
    started = False
    for t in range(n):
        if not np.isfinite(z[t]):
            continue
        started = True
        e = LAMBDA * z[t] + (1 - LAMBDA) * e
        c_p = min(max(0.0, c_p + z[t] - CUSUM_K), 2 * CUSUM_H)
        c_m = min(max(0.0, c_m - z[t] - CUSUM_K), 2 * CUSUM_H)
        ew[t], c_hi[t], c_lo[t] = e, c_p, c_m
    signal = np.zeros(n, dtype=int)  # -1 low, +1 high
    high = (ew > EWMA_LIMIT) | (c_hi > CUSUM_H)
    low = (ew < -EWMA_LIMIT) | (c_lo > CUSUM_H)
    signal[np.where(high & ~low)[0]] = 1
    signal[np.where(low & ~high)[0]] = -1
    persistent = np.zeros(n, dtype=int)
    for t in range(n):
        w = signal[max(0, t - PERSIST_N + 1): t + 1]
        if (w == 1).sum() >= PERSIST_K and signal[t] == 1:
            persistent[t] = 1
        elif (w == -1).sum() >= PERSIST_K and signal[t] == -1:
            persistent[t] = -1
    return {"center": med, "scale": scale, "z": z, "ewma": ew, "cusum_high": c_hi, "cusum_low": c_lo,
            # limits for the EWMA, in value units: the chart plots value, ewma_value and the band [lower, upper]
            "ewma_value": med + ew * scale, "lower": med - EWMA_LIMIT * scale, "upper": med + EWMA_LIMIT * scale,
            "signal": signal, "persistent": persistent}


def onsets(persistent: np.ndarray, quiet_months: int = 2) -> np.ndarray:
    """Indices where a persistent run starts (same direction not persistent in the previous `quiet_months`)."""
    out = []
    for t in range(len(persistent)):
        if persistent[t] != 0 and not any(persistent[max(0, t - quiet_months): t] == persistent[t]):
            out.append(t)
    return np.array(out, dtype=int)


def funnel_chart(x, n, center: float, a: float, b: float, z: float = 3.0, min_n: int = 3) -> dict[str, np.ndarray]:
    """Group-vs-groups comparison. `x`: a group's 3-month change of its mean score, `n`: members scored each month.
    Limits are center +- z * sqrt(a + b / n): the fewer members, the wider (funnel). Months with n < min_n are not judged."""
    x = np.asarray(x, dtype=float)
    n = np.asarray(n, dtype=float)
    ok = np.isfinite(x) & (n >= min_n)
    half = z * np.sqrt(a + b / np.where(n > 0, n, np.nan))
    signal = np.zeros(len(x), dtype=int)
    signal[ok & (x > center + half)] = 1
    signal[ok & (x < center - half)] = -1
    persistent = np.zeros(len(x), dtype=int)
    for t in range(len(x)):
        w = signal[max(0, t - PERSIST_N + 1): t + 1]
        if signal[t] != 0 and (w == signal[t]).sum() >= PERSIST_K:
            persistent[t] = signal[t]
    return {"center": np.full(len(x), center), "lower": center - half, "upper": center + half, "signal": signal, "persistent": persistent}
