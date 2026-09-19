"""Score forecast with fans (plan step 4): the company's own score 1..HORIZON months ahead as quantiles (10/25/50/75/90), not a point estimate.

    PYTHONUTF8=1 PYTHONPATH=<repo>/.venv/Lib/site-packages:<repo> python -m analysis.monitor.forecast [--arima]

What the data says (measured here, train companies, group-fold CV): the score is not a trending series. Monthly changes are negatively
autocorrelated and the score is pulled toward the company's own average and toward the portfolio level, so extrapolating a trend adds nothing
and the gain is in that pull and in the shape of the fan (skewed by level). Moves are heavy-tailed: the typical fall holds, a minority reverses.
Candidates, all judged out of fold on the same origins:
  naive_last         the last score; fan = pooled quantiles of the error scaled by the company's own volatility
  smoothed_level     simple exponential smoothing (alpha chosen on training folds)
  damped_trend       Holt with damped trend (alpha, beta, phi chosen on training folds): the "tendency" model
  reversion_quantile linear quantile regression of the change score_{t+h} - score_t on the company's features at t, one model per horizon and
                     quantile (own deviation, level, last 1- and 3-month moves, deviation from the 6-month mean, own volatility, level squared).
                     Pooled over train companies, so it needs no per-company fitting and no extra dependency at export time.
  ARIMA (--arima)    per-company statsmodels ARIMA on a sample of origins, as a benchmark only (never shipped).
A time-split check (fit on the first half of the months, test on the later ones, held-out companies) is the stricter reading. Seasonality is tested (lag-12 autocorrelation of monthly changes, and whether the calendar-month mean change repeats a year later) and only
shipped if it is there. Ship rule per horizon: reversion_quantile if its pinball-loss skill over the naive fan has a group-bootstrap interval
above zero at that horizon, otherwise naive_last, said as such. Holdout companies are never used. Persistence of a score is not a prediction of outcomes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .fit import load as load_params, month_grid, wide

HORIZON = 6
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
MIN_HISTORY = 4
MATERIAL_MOVE = 8.0      # points over 3 months; same a priori materiality as the score alerts (engine.MATERIAL)
LEVEL2_DIV = 30.0
PRIOR_MONTHS = 3         # pseudo-months of the portfolio-typical volatility mixed into the company's own
LOW_SCALE_FRAC = 0.5
FEATURES = ("own_dev", "level", "d1", "d3", "dev6", "vol", "level2")
ALPHAS = (0.5, 0.85)
HOLT_GRID = tuple((a, b, p) for a in (0.5, 0.85) for b in (0.05, 0.2) for p in (0.5, 0.8))
N_BOOT = 400
SEED = 20260919
ARIMA_COMPANIES, ARIMA_STEP = 220, 2
PARAMS_PATH = Path(__file__).resolve().parent / "forecast_params.json"
OUT_MD = Path(__file__).resolve().parent / "forecast_evaluation.md"


# ------------------------------------------------------------------ series helpers

def _lag(X: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(X, np.nan)
    out[:, k:] = X[:, :-k]
    return out


def history_count(S: np.ndarray) -> np.ndarray:
    return np.cumsum(np.isfinite(S), axis=1)


def _trailing_mean(S: np.ndarray, k: int) -> np.ndarray:
    z = np.concatenate([np.zeros((S.shape[0], 1)), np.cumsum(np.nan_to_num(S), axis=1)], axis=1)
    n = np.concatenate([np.zeros((S.shape[0], 1)), np.cumsum(np.isfinite(S), axis=1)], axis=1)
    lo = np.maximum(np.arange(1, S.shape[1] + 1) - k, 0)
    return (z[:, 1:] - z[:, lo]) / np.maximum(n[:, 1:] - n[:, lo], 1)


def raw_change_scale(S: np.ndarray) -> np.ndarray:
    """Robust sd of the company's monthly changes up to t (1.4826 * MAD, no lag), NaN with under 3 changes."""
    D = np.diff(S, axis=1, prepend=np.nan)
    out = np.full_like(S, np.nan)
    for i in range(S.shape[0]):
        for t in range(3, S.shape[1]):
            d = D[i, 1 : t + 1]
            d = d[np.isfinite(d)]
            if len(d) >= 3:
                out[i, t] = 1.4826 * np.median(np.abs(d - np.median(d)))
    return out


def change_scale(S: np.ndarray, typical: float) -> np.ndarray:
    """Own volatility available from the 4th scored month: the robust scale mixed with PRIOR_MONTHS months of the portfolio-typical value."""
    raw = raw_change_scale(S)
    k = np.maximum(history_count(S) - 1, 0)
    return np.maximum(np.sqrt((k * raw**2 + PRIOR_MONTHS * typical**2) / (k + PRIOR_MONTHS)), LOW_SCALE_FRAC * typical)


def build_features(S: np.ndarray, centre: float, typical: float) -> dict[str, np.ndarray]:
    hist = history_count(S)
    mean_all = np.cumsum(np.nan_to_num(S), axis=1) / np.maximum(hist, 1)
    lvl = S - centre
    return {"own_dev": S - mean_all, "level": lvl, "d1": S - _lag(S, 1), "d3": S - _lag(S, 3), "dev6": S - _trailing_mean(S, 6),
            "vol": change_scale(S, typical), "level2": lvl**2 / LEVEL2_DIV}


def usable(F: dict[str, np.ndarray], hist: np.ndarray) -> np.ndarray:
    ok = hist >= MIN_HISTORY
    for name in FEATURES:
        ok &= np.isfinite(F[name])
    return ok


def design(F: dict[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(int(mask.sum()))] + [F[n][mask] for n in FEATURES])


def target(S: np.ndarray, h: int) -> np.ndarray:
    y = np.full_like(S, np.nan)
    y[:, : S.shape[1] - h] = S[:, h:]
    return y


def fit_typical_scale(S: np.ndarray) -> float:
    raw = raw_change_scale(S)[history_count(S) >= 8]
    return float(np.median(raw[np.isfinite(raw)]))


# ------------------------------------------------------------------ models

def quantile_fit(X: np.ndarray, y: np.ndarray, q: float, iters: int = 40) -> np.ndarray:
    """Linear quantile regression by iteratively reweighted least squares (asymmetric absolute loss)."""
    w = np.ones(len(y))
    ridge = 1e-6 * np.eye(X.shape[1])
    for _ in range(iters):
        Xw = X * w[:, None]
        b = np.linalg.solve(Xw.T @ X + ridge, Xw.T @ y)
        r = y - X @ b
        w = np.where(r > 0, q, 1 - q) / np.maximum(np.abs(r), 0.05)
    return b


def fit_reversion(F: dict, S: np.ndarray, hist: np.ndarray, rows: np.ndarray) -> dict[int, np.ndarray]:
    """Per horizon, coefficients (len(QUANTILES) x (1 + len(FEATURES))) of the change score_{t+h} - score_t on the features, from the selected companies."""
    ok0 = usable(F, hist) & rows[:, None]
    out = {}
    for h in range(1, HORIZON + 1):
        y = target(S, h)
        ok = ok0 & np.isfinite(y)
        X, dy = design(F, ok), (y - S)[ok]
        out[h] = np.array([quantile_fit(X, dy, q) for q in QUANTILES])
    return out


def fan_from_models(last: np.ndarray, X: np.ndarray, coefs: np.ndarray) -> np.ndarray:
    """(n, len(QUANTILES)) fan: last score + predicted change quantiles, sorted (no crossing) and clipped to [0, 100]."""
    return np.clip(np.sort(last[:, None] + X @ coefs.T, axis=1), 0.0, 100.0)


def fit_naive_quantiles(F: dict, S: np.ndarray, hist: np.ndarray, rows: np.ndarray) -> dict[int, np.ndarray]:
    """Pooled quantiles of (score_{t+h} - score_t) / own volatility."""
    ok0 = usable(F, hist) & rows[:, None]
    out = {}
    for h in range(1, HORIZON + 1):
        y = target(S, h)
        ok = ok0 & np.isfinite(y)
        out[h] = np.quantile(((y - S) / F["vol"])[ok], QUANTILES)
    return out


def naive_fan(last: np.ndarray, vol: np.ndarray, qz: np.ndarray) -> np.ndarray:
    return np.clip(last[:, None] + vol[:, None] * qz[None, :], 0.0, 100.0)


def ses_level(S: np.ndarray, alpha: float) -> np.ndarray:
    """Simple exponential smoothing along axis 1, NaN-aware (the level starts at the first score and carries through gaps)."""
    L = np.full_like(S, np.nan)
    cur = np.full(S.shape[0], np.nan)
    for t in range(S.shape[1]):
        x = S[:, t]
        cur = np.where(np.isnan(cur), x, np.where(np.isnan(x), cur, alpha * x + (1 - alpha) * cur))
        L[:, t] = cur
    return L


def damped_trend(S: np.ndarray, alpha: float, beta: float, phi: float) -> tuple[np.ndarray, np.ndarray]:
    """Holt's linear method with damped trend, NaN-aware. Returns level and slope per month; the h-step forecast is level + slope * (phi + ... + phi^h)."""
    L, B = np.full_like(S, np.nan), np.full_like(S, np.nan)
    lev = np.full(S.shape[0], np.nan)
    slope = np.zeros(S.shape[0])
    for t in range(S.shape[1]):
        x = S[:, t]
        started = np.isfinite(lev)
        pred = np.where(started, lev + phi * slope, x)
        new = np.where(np.isfinite(x), alpha * x + (1 - alpha) * pred, pred)
        slope = np.where(np.isfinite(x) & started, beta * (new - lev) + (1 - beta) * phi * slope, slope)
        lev = np.where(started | np.isfinite(x), new, np.nan)
        slope = np.where(np.isfinite(lev), slope, 0.0)
        L[:, t], B[:, t] = lev, slope
    return L, B


def damped_forecast(L: np.ndarray, B: np.ndarray, phi: float, h: int) -> np.ndarray:
    return L + B * sum(phi**k for k in range(1, h + 1))


# ------------------------------------------------------------------ evaluation

def pinball(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Mean pinball loss over QUANTILES per row (y: (n,), f: (n, len(QUANTILES)))."""
    d = y[:, None] - f
    q = np.asarray(QUANTILES)[None, :]
    return np.maximum(q * d, (q - 1) * d).mean(axis=1)


def _group_sums(loss: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-group sum of the loss and count of origins (rows of `loss` are origins, `groups` their group ids)."""
    uniq, inv = np.unique(groups, return_inverse=True)
    return np.bincount(inv, weights=loss, minlength=len(uniq)), np.bincount(inv, minlength=len(uniq)).astype(float)


def paired_skill(loss_c: np.ndarray, loss_b: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> tuple[float, list[float]]:
    """1 - loss(candidate) / loss(baseline) on the same origins, with a group-bootstrap 95% interval."""
    sc, _ = _group_sums(loss_c, groups)
    sb, _ = _group_sums(loss_b, groups)
    skills = [1 - sc[p].sum() / max(sb[p].sum(), 1e-9) for p in (rng.integers(0, len(sc), len(sc)) for _ in range(N_BOOT))]
    return float(1 - sc.sum() / sb.sum()), [float(np.percentile(skills, 2.5)), float(np.percentile(skills, 97.5))]


def _pick(candidates: dict, folds: np.ndarray, train_mae) -> dict[int, object]:
    """Per held-out fold, the candidate parameters with the lowest training-folds MAE."""
    chosen = {}
    for k in np.unique(folds):
        tr = folds != k
        chosen[int(k)] = min(candidates, key=lambda c: train_mae(c, tr))
    return chosen


def cross_validate(S: np.ndarray, groups: np.ndarray, folds: np.ndarray, centre: float, typical: float) -> dict:
    n, T = S.shape
    F = build_features(S, centre, typical)
    hist = history_count(S)
    ok0 = usable(F, hist)
    rng = np.random.default_rng(SEED)
    fold_ids = [int(k) for k in np.unique(folds)]

    # models fitted per held-out fold on the other folds
    coefs = {k: fit_reversion(F, S, hist, folds != k) for k in fold_ids}
    qz = {k: fit_naive_quantiles(F, S, hist, folds != k) for k in fold_ids}

    # smoothing / trend candidates: forecast matrices for every parameter set, choice by training-folds MAE over horizons
    ses_all = {a: ses_level(S, a) for a in ALPHAS}
    holt_all = {p: damped_trend(S, *p) for p in HOLT_GRID}

    def mae_of(fc, tr):
        tot = 0.0
        for h in range(1, HORIZON + 1):
            y = target(S, h)
            ok = ok0 & np.isfinite(y) & tr[:, None]
            tot += float(np.abs(y - fc(h))[ok].mean())
        return tot

    ses_pick = _pick(ALPHAS, folds, lambda a, tr: mae_of(lambda h: ses_all[a], tr))
    holt_pick = _pick(HOLT_GRID, folds, lambda p, tr: mae_of(lambda h: damped_forecast(*holt_all[p], p[2], h), tr))

    per_h, fans = {}, {}
    for h in range(1, HORIZON + 1):
        y = target(S, h)
        ok = ok0 & np.isfinite(y)
        rows_i, cols_t = np.nonzero(ok)
        yy, last, g = y[ok], S[ok], groups[rows_i]
        fold_of = folds[rows_i]
        f_rev, f_nv = np.zeros((len(yy), len(QUANTILES))), np.zeros((len(yy), len(QUANTILES)))
        med_ses, med_holt = np.zeros(len(yy)), np.zeros(len(yy))
        for k in fold_ids:
            m = fold_of == k
            sub = {name: F[name][ok][m] for name in FEATURES}
            X = np.column_stack([np.ones(int(m.sum()))] + [sub[nm] for nm in FEATURES])
            f_rev[m] = fan_from_models(last[m], X, coefs[k][h])
            f_nv[m] = naive_fan(last[m], sub["vol"], qz[k][h])
            med_ses[m] = ses_all[ses_pick[k]][ok][m]
            p = holt_pick[k]
            med_holt[m] = damped_forecast(*holt_all[p], p[2], h)[ok][m]
        loss_rev, loss_nv = pinball(yy, f_rev), pinball(yy, f_nv)
        ae = {"naive_last": np.abs(yy - last), "smoothed_level": np.abs(yy - med_ses), "damped_trend": np.abs(yy - med_holt),
              "reversion_quantile": np.abs(yy - f_rev[:, 2])}
        skill, ci = paired_skill(loss_rev, loss_nv, g, rng)
        point = {}
        for name, e in ae.items():
            sk, sci = paired_skill(e, ae["naive_last"], g, rng) if name != "naive_last" else (0.0, [0.0, 0.0])
            point[name] = {"mae": float(e.mean()), "skill": sk, "skill_ci": sci}
        cover = lambda f: {"cover50": float(np.mean((yy >= f[:, 1]) & (yy <= f[:, 3]))), "cover80": float(np.mean((yy >= f[:, 0]) & (yy <= f[:, 4])))}
        per_h[h] = {"n_origins": int(len(yy)), "point": point, "pinball_naive": float(loss_nv.mean()), "pinball_reversion": float(loss_rev.mean()),
                    "pinball_skill": skill, "pinball_skill_ci": ci, "coverage_reversion": cover(f_rev), "coverage_naive": cover(f_nv)}
        fans[h] = {"y": yy, "fan": f_rev, "last": last, "vol": F["vol"][ok], "hist": hist[ok], "rows": rows_i, "cols": cols_t, "median": f_rev[:, 2]}
    return {"per_horizon": per_h, "ses_alpha": ses_pick, "holt_params": {k: list(v) for k, v in holt_pick.items()}, "fans": fans}


def calibration_breakdown(fans: dict, h: int) -> dict:
    """80% and 50% coverage of the shipped fan by score level, history length and own volatility, at horizon h (where the fan is off, it is said)."""
    d = fans[h]
    y, f = d["y"], d["fan"]
    inside80, inside50 = (y >= f[:, 0]) & (y <= f[:, 4]), (y >= f[:, 1]) & (y <= f[:, 3])
    vol_cuts = np.quantile(d["vol"], [1 / 3, 2 / 3])
    buckets = {
        "score <60": d["last"] < 60, "score 60-75": (d["last"] >= 60) & (d["last"] < 75), "score 75+": d["last"] >= 75,
        "history 4-7 months": d["hist"] < 8, "history 8-13": (d["hist"] >= 8) & (d["hist"] < 14), "history 14+": d["hist"] >= 14,
        "calm (lowest third of own volatility)": d["vol"] <= vol_cuts[0], "middle third": (d["vol"] > vol_cuts[0]) & (d["vol"] <= vol_cuts[1]),
        "volatile (top third)": d["vol"] > vol_cuts[1]}
    return {name: {"n": int(m.sum()), "cover50": float(inside50[m].mean()), "cover80": float(inside80[m].mean())} for name, m in buckets.items() if m.sum() > 0}


def time_split_check(S: np.ndarray, groups: np.ndarray, folds: np.ndarray, centre: float, typical: float, split: int) -> dict:
    """Stricter than the group-fold CV, which shares calendar months between train and test: fit only on origins whose target month is <= `split`,
    other folds' companies, and test on the held-out companies' origins after `split`. Same skill measure (pinball loss against the naive fan)."""
    F = build_features(S, centre, typical)
    hist = history_count(S)
    ok0 = usable(F, hist)
    months = np.arange(S.shape[1])[None, :]
    rng = np.random.default_rng(SEED)
    out = {}
    for h in range(1, HORIZON + 1):
        y = target(S, h)
        okall = ok0 & np.isfinite(y)
        parts = []
        for k in np.unique(folds):
            tr = okall & (folds != k)[:, None] & (months + h <= split)
            te = okall & (folds == k)[:, None] & (months > split)
            if tr.sum() < 500 or te.sum() < 50:
                continue
            X, dy = design(F, tr), (y - S)[tr]
            coefs = np.array([quantile_fit(X, dy, q) for q in QUANTILES])
            qz = np.quantile(dy / F["vol"][tr], QUANTILES)
            yy, last = y[te], S[te]
            f_rev, f_nv = fan_from_models(last, design(F, te), coefs), naive_fan(last, F["vol"][te], qz)
            parts.append((pinball(yy, f_rev), pinball(yy, f_nv), groups[np.nonzero(te)[0]], (yy >= f_rev[:, 0]) & (yy <= f_rev[:, 4]), (yy >= f_rev[:, 1]) & (yy <= f_rev[:, 3])))
        if not parts:
            continue
        l_rev, l_nv, g, c80, c50 = (np.concatenate(x) for x in zip(*parts))
        skill, ci = paired_skill(l_rev, l_nv, g, rng)
        out[h] = {"n_origins": int(len(l_rev)), "pinball_skill": skill, "pinball_skill_ci": ci, "cover50": float(c50.mean()), "cover80": float(c80.mean())}
    return out


def seasonality_test(S: np.ndarray) -> dict:
    """Is there a yearly pattern? (1) lag-12 autocorrelation of the monthly changes, pooled, with a company-bootstrap interval, next to the
    other lags; (2) whether the calendar-month mean change repeats in the second year. With 24 months the power is limited; said in the output."""
    D = np.diff(S, axis=1)
    Dc = D - np.nanmean(D, axis=1, keepdims=True)

    def acf(idx, k):
        a, b = Dc[idx, :-k].ravel(), Dc[idx, k:].ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[ok], b[ok])[0, 1]), int(ok.sum())

    allrows = np.arange(S.shape[0])
    lags = {k: acf(allrows, k) for k in range(1, 13)}
    has12 = np.flatnonzero(np.isfinite(Dc[:, :-12] * Dc[:, 12:]).any(axis=1))
    rng = np.random.default_rng(SEED)
    boot = [acf(rng.choice(has12, len(has12)), 12)[0] for _ in range(300)]
    monthly = np.nanmean(Dc, axis=0)
    a, b = monthly[:-12], monthly[12:]
    pair = np.isfinite(a) & np.isfinite(b)
    m = int(pair.sum())
    cal = float(np.corrcoef(a[pair], b[pair])[0, 1]) if m >= 5 else None
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    return {"acf_lag12": lags[12][0], "acf_lag12_ci": ci, "acf_lag12_pairs": lags[12][1], "acf_lags_1_to_11": {str(k): lags[k][0] for k in range(1, 12)},
            "calendar_month_corr_year1_year2": cal, "calendar_month_sd_points": float(np.nanstd(monthly)), "months_compared": m,
            "supported": bool(ci[0] > 0 and cal is not None and cal > 0.5),
            "note": "24 months give one lap and a bit: this can rule out a strong yearly pattern, not a weak one."}


def move_persistence(S: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> dict:
    """After a material 3-month move (>= MATERIAL_MOVE points), what happened over the next 3 months: how often at least half of it reversed."""
    hist = history_count(S)
    d3 = S - _lag(S, 3)
    f3 = target(S, 3) - S
    out = {}
    for name, cond in (("falls", d3 <= -MATERIAL_MOVE), ("rises", d3 >= MATERIAL_MOVE)):
        ok = cond & np.isfinite(f3) & (hist >= MIN_HISTORY)
        move, nxt, g = d3[ok], f3[ok], groups[np.nonzero(ok)[0]]
        reversed_half = (nxt * np.sign(move)) <= -abs(move) / 2
        continued = (nxt * np.sign(move)) >= MATERIAL_MOVE / 4
        s, c = _group_sums(reversed_half.astype(float), g)
        boots = [s[p].sum() / c[p].sum() for p in (rng.integers(0, len(s), len(s)) for _ in range(N_BOOT))]
        out[name] = {"n": int(ok.sum()), "median_move": float(np.median(move)), "median_next_3m": float(np.median(nxt)), "mean_next_3m": float(np.mean(nxt)), "share_reversed_at_least_half": float(reversed_half.mean()),
                     "share_reversed_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "share_continued": float(continued.mean())}
    return out


def arima_benchmark(S: np.ndarray, oof_median: dict[int, np.ndarray]) -> dict:
    """Per-company statsmodels ARIMA on a sample of companies and origins (>= 8 scored months), against the last value and the shipped model's out-of-fold
    median on the very same origins. Benchmark only: fitting per company from under 24 points is what this is here to show."""
    import warnings

    from statsmodels.tsa.arima.model import ARIMA

    grid = {"ar1_const": ((1, 0, 0), "c"), "ima_0_1_1": ((0, 1, 1), "n"), "arima_1_1_1": ((1, 1, 1), "n"), "arima_1_1_0": ((1, 1, 0), "n")}
    rng = np.random.default_rng(SEED)
    T = S.shape[1]
    pool = np.flatnonzero(np.isfinite(S).sum(axis=1) >= 14)
    comps = rng.choice(pool, size=min(ARIMA_COMPANIES, len(pool)), replace=False)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in comps:
            obs = np.flatnonzero(np.isfinite(S[i]))
            for t in range(obs[0] + MIN_HISTORY + 3, T - 1, ARIMA_STEP):
                y = S[i, obs[0] : t + 1]
                fc = {}
                for name, (order, trend) in grid.items():
                    try:
                        fc[name] = np.asarray(ARIMA(y, order=order, trend=trend).fit().forecast(HORIZON))
                    except Exception:
                        fc[name] = None
                for h in range(1, HORIZON + 1):
                    if t + h < T and np.isfinite(S[i, t + h]) and np.isfinite(oof_median[h][i, t]) and all(v is not None for v in fc.values()):
                        rows.append({"h": h, "y": S[i, t + h], "naive_last": y[-1], "reversion_quantile": oof_median[h][i, t], **{k: v[h - 1] for k, v in fc.items()}})
    df = pd.DataFrame(rows)
    models = ["naive_last", "reversion_quantile", *grid]
    by_h = {int(h): {m: float((g["y"] - g[m]).abs().mean()) for m in models} | {"n": int(len(g))} for h, g in df.groupby("h")}
    return {"companies": int(len(comps)), "forecasts": int(len(df)), "by_horizon": by_h, "models": {k: {"order": list(v[0]), "trend": v[1]} for k, v in grid.items()},
            "note": "MAE of the median forecast on the same sampled origins; ARIMA fitted per company from that company's own history only."}


# ------------------------------------------------------------------ shipping

def forecast_rows(S: np.ndarray, params: dict) -> list[dict | None]:
    """Fan for every company from its last scored month (rows of S), from forecast_params.json. None when the history is too short.
    Per horizon the shipped method decides; the drivers decompose the 3-month median change."""
    centre, typical = params["level_centre"], params["typical_change_scale"]
    F = build_features(S, centre, typical)
    hist = history_count(S)
    good = usable(F, hist)
    models = {int(h): np.asarray(c) for h, c in params["models"].items()}
    qz = {int(h): np.asarray(v) for h, v in params["naive_quantiles"].items()}
    method = {int(h): m for h, m in params["method_by_horizon"].items()}
    med_col = QUANTILES.index(0.50)
    out = []
    for i in range(S.shape[0]):
        obs = np.flatnonzero(np.isfinite(S[i]))
        if len(obs) == 0 or hist[i, -1] < MIN_HISTORY:
            out.append(None)
            continue
        t = int(obs[-1])
        last = np.array([S[i, t]])
        vol = np.array([F["vol"][i, t]])
        have = bool(good[i, t])
        x = np.array([[1.0] + [F[n][i, t] for n in FEATURES]]) if have else None
        pts = []
        for h in range(1, HORIZON + 1):
            use_model = have and method[h] == "reversion_quantile"
            fan = fan_from_models(last, x, models[h])[0] if use_model else naive_fan(last, vol, qz[h])[0]
            pts.append({"h": h, "median": float(fan[med_col]), "lo50": float(fan[1]), "hi50": float(fan[3]), "lo80": float(fan[0]), "hi80": float(fan[4]),
                        "method": "reversion_quantile" if use_model else "naive_last"})
        row = {"origin_index": t, "naive_last": float(S[i, t]), "points": pts, "own_average": float(np.nanmean(S[i, : t + 1])), "drivers": None}
        if have and method[3] == "reversion_quantile":
            b = models[3][med_col]
            v = {n: F[n][i, t] for n in FEATURES}
            parts = {"own_average": b[1 + FEATURES.index("own_dev")] * v["own_dev"] + b[1 + FEATURES.index("dev6")] * v["dev6"],
                     "portfolio_level": b[1 + FEATURES.index("level")] * v["level"] + b[1 + FEATURES.index("level2")] * v["level2"],
                     "recent_move": b[1 + FEATURES.index("d1")] * v["d1"] + b[1 + FEATURES.index("d3")] * v["d3"],
                     "baseline": b[0] + b[1 + FEATURES.index("vol")] * v["vol"]}
            row["drivers"] = {"horizon": 3, "expected_change": float(sum(parts.values())), "parts": {k: float(x_) for k, x_ in parts.items()},
                              "own_deviation": float(v["own_dev"]), "level_vs_portfolio": float(v["level"]), "last_3_month_move": float(v["d3"])}
        out.append(row)
    return out


# ------------------------------------------------------------------ report

def _method_line(per_h: dict, method: dict) -> str:
    shipped = [h for h, m in method.items() if m == "reversion_quantile"]
    return (f"reversion_quantile at horizons {', '.join(map(str, shipped))}" if shipped else "naive_last at every horizon") + (
        "; naive_last elsewhere" if shipped and len(shipped) < len(method) else "")


def write_report(out: dict, cal: dict, cal6: dict) -> str:
    per, cv = out["cv"]["per_horizon"], out["cv"]
    L = ["# Score forecast evaluation (train companies, group-fold CV, holdout untouched)", "",
         f"Shipped: **{_method_line(per, {int(h): m for h, m in out['method_by_horizon'].items()})}**. {out['why']}", "",
         "The score is not a trending series: monthly changes are negatively autocorrelated and the score is pulled toward the company's own average and the portfolio level. "
         "So extrapolating a trend adds nothing; the gain is in that pull and in the shape of the fan.", "",
         "## 1. Median forecast (MAE in points, and skill = 1 - MAE / MAE(last value), 95% group-bootstrap interval)", "",
         "| horizon | origins | last value | smoothed level | damped trend (Holt) | reversion model |", "|---|---|---|---|---|---|"]
    for h, v in per.items():
        p = v["point"]
        cell = lambda m: f"{p[m]['mae']:.2f} ({p[m]['skill']:+.1%}; {p[m]['skill_ci'][0]:+.1%} to {p[m]['skill_ci'][1]:+.1%})" if m != "naive_last" else f"{p[m]['mae']:.2f}"
        L.append(f"| {h} | {v['n_origins']} | {cell('naive_last')} | {cell('smoothed_level')} | {cell('damped_trend')} | {cell('reversion_quantile')} |")
    dt = [v["point"]["damped_trend"]["skill"] for v in per.values()]
    rv = [v["point"]["reversion_quantile"]["skill"] for v in per.values()]
    L += ["", f"Damped trend is the tendency model. Its skill against the last value runs from {min(dt):+.1%} to {max(dt):+.1%} across horizons, against {min(rv):+.1%} to {max(rv):+.1%} for the reversion model: "
          "extrapolating a trend adds nothing, using the pull toward the company's own average does.", "",
          "## 2. The fan (pinball loss over the 10/25/50/75/90% quantiles, out of fold)", "",
          "| horizon | pinball naive fan | pinball reversion fan | skill (95% CI) | 50% covers (naive / reversion) | 80% covers (naive / reversion) | shipped |", "|---|---|---|---|---|---|---|"]
    for h, v in per.items():
        L.append(f"| {h} | {v['pinball_naive']:.3f} | {v['pinball_reversion']:.3f} | {v['pinball_skill']:+.1%} ({v['pinball_skill_ci'][0]:+.1%} to {v['pinball_skill_ci'][1]:+.1%}) | "
                 f"{v['coverage_naive']['cover50']:.0%} / {v['coverage_reversion']['cover50']:.0%} | {v['coverage_naive']['cover80']:.0%} / {v['coverage_reversion']['cover80']:.0%} | {out['method_by_horizon'][str(h)]} |")
    L += ["", "The naive fan is the last value with pooled quantiles of the error scaled by the company's own volatility; the reversion fan is a per-horizon, per-quantile linear "
          "quantile regression on the company's features, so it is skewed (high scores can fall further than they can rise) and shifts with the deviation from the company's own average.", ""]
    for h, c in ((3, cal), (6, cal6)):
        L += [f"### Where the shipped fan is calibrated ({h}-month horizon, out of fold)", "", "| group | origins | 50% covers | 80% covers |", "|---|---|---|---|"]
        L += [f"| {k} | {v['n']} | {v['cover50']:.0%} | {v['cover80']:.0%} |" for k, v in c.items()]
        L.append("")
    ts = out["time_split_check"]
    L += [f"## 3. Stricter check: also hold out the later months", "",
          f"The group-fold CV above holds out companies but shares calendar months between train and test. Here the models are fitted only on origins whose target month is up to {ts['split_month']} "
          "(other folds' companies) and tested on the held-out companies' origins after it. Pinball skill against the naive fan fitted the same way:", "",
          "| horizon | origins | skill (95% CI) | 50% covers | 80% covers |", "|---|---|---|---|---|"]
    L += [f"| {h} | {v['n_origins']} | {v['pinball_skill']:+.1%} ({v['pinball_skill_ci'][0]:+.1%} to {v['pinball_skill_ci'][1]:+.1%}) | {v['cover50']:.0%} | {v['cover80']:.0%} |" for h, v in ts["per_horizon"].items()]
    L += ["", "The gain holds out of time but is smaller than in section 2 at the short horizons, and the 80% interval covers a little under 80% on later months, so read section 2 as the optimistic end.", ""]
    s = out["seasonality"]
    L += ["## 4. Seasonality", "",
          f"Lag-12 autocorrelation of monthly changes {s['acf_lag12']:+.3f} (95% CI {s['acf_lag12_ci'][0]:+.3f} to {s['acf_lag12_ci'][1]:+.3f}, {s['acf_lag12_pairs']} pairs); lags 1-11 range "
          f"{min(s['acf_lags_1_to_11'].values()):+.3f} to {max(s['acf_lags_1_to_11'].values()):+.3f}. Calendar-month mean changes: correlation between the two years "
          f"{s['calendar_month_corr_year1_year2']:+.2f} over {s['months_compared']} months, with monthly means within about ±{2 * s['calendar_month_sd_points']:.1f} points. "
          f"**{'Seasonality supported' if s['supported'] else 'No seasonal pattern found, none shipped'}.** {s['note']}", ""]
    mp = out["move_persistence"]
    L += ["## 5. Do moves persist? (the tendency question)", "",
          "| move over 3 months (>= 8 points) | cases | median move | next 3 months: median / mean change | at least half reversed (95% CI) | continued (>= 2 more points) |", "|---|---|---|---|---|---|"]
    for k, v in mp.items():
        L.append(f"| {k} | {v['n']} | {v['median_move']:+.1f} | {v['median_next_3m']:+.1f} / {v['mean_next_3m']:+.1f} | {v['share_reversed_at_least_half']:.0%} ({v['share_reversed_ci'][0]:.0%} to {v['share_reversed_ci'][1]:.0%}) | {v['share_continued']:.0%} |")
    L += ["", "Moves are heavy-tailed: the typical material fall holds (most do not recover half within 3 months) while the average one partly recovers because of a minority of large reversals; "
          "the typical rise gives some back. This is why the score alerts describe a move and the fan is asymmetric, rather than either extrapolating or assuming a return.", ""]
    if "arima_benchmark" in out:
        a = out["arima_benchmark"]
        L += ["## 6. Per-company ARIMA (statsmodels), benchmark only", "",
              f"{a['companies']} sampled companies, {a['forecasts']} forecasts, origins with at least 8 scored months. MAE of the median forecast in points, same origins for every column.", "",
              "| horizon | forecasts | last value | reversion model (shipped) | AR(1)+const | ARIMA(0,1,1) | ARIMA(1,1,1) | ARIMA(1,1,0) |", "|---|---|---|---|---|---|---|---|"]
        for h, v in a["by_horizon"].items():
            L.append(f"| {h} | {v['n']} | {v['naive_last']:.2f} | {v['reversion_quantile']:.2f} | {v['ar1_const']:.2f} | {v['ima_0_1_1']:.2f} | {v['arima_1_1_1']:.2f} | {v['arima_1_1_0']:.2f} |")
        L += ["", "Per-company ARIMA from at most 22 monthly scores loses to the last value at short horizons; only the mean-reverting AR(1) with a constant gains at 4-6 months, "
              "which is the effect the pooled reversion model uses, estimated on the whole portfolio instead of one short series.", ""]
    L += ["Skill = 1 - loss(candidate) / loss(baseline) on the same origins; zero is a tie. Coverage is measured on held-out folds with models fitted on the other folds. "
          "The fan says where the score usually goes from here and how far it can move; it is not a prediction of financial outcomes."]
    return "\n".join(L) + "\n"


def main(argv: list[str]) -> int:
    from analysis.evaluate.protocol import FOLD_SEED, group_folds, train_companies
    from analysis.features.common import connect
    from analysis.models.gbm_y7y8 import _keys, load_store
    from product.score import fit as score_fit
    from product.score.run import score_store

    con = connect()
    store, _ = load_store(con)
    store = _keys(store)
    train_df = train_companies(con)
    train_df["company_id"] = train_df["company_id"].astype(str)
    detail = score_store(store, score_fit.load_reference(), with_reasons=False)
    detail["company_id"] = detail["company_id"].astype(str)
    grid = month_grid(detail)
    W = wide(detail[detail["company_id"].isin(set(train_df["company_id"]))], "score", grid)
    fold_of = group_folds(train_df, n=5, seed=FOLD_SEED).set_index("company_id")["fold"]
    grp_of = train_df.set_index("company_id")["group_id"].astype(str)
    ids = W.index.to_numpy()
    S = W.to_numpy(dtype=float)
    folds = fold_of.reindex(ids).to_numpy()
    groups = grp_of.reindex(ids).to_numpy()

    centre, typical = float(np.nanmean(S)), fit_typical_scale(S)
    print("cross-validating...")
    cv = cross_validate(S, groups, folds, centre, typical)
    per = cv["per_horizon"]
    method = {h: ("reversion_quantile" if per[h]["pinball_skill_ci"][0] > 0 else "naive_last") for h in per}

    F = build_features(S, centre, typical)
    hist = history_count(S)
    all_rows = np.ones(len(ids), dtype=bool)
    models = fit_reversion(F, S, hist, all_rows)
    naive_q = fit_naive_quantiles(F, S, hist, all_rows)
    rng = np.random.default_rng(SEED)
    split = S.shape[1] // 2
    time_split = {"split_month": f"{grid[split]:%Y-%m}", "per_horizon": {str(h): v for h, v in time_split_check(S, groups, folds, centre, typical, split).items()}}
    seasonal = seasonality_test(S)
    persist = move_persistence(S, groups, rng)

    prev = json.loads(PARAMS_PATH.read_text(encoding="utf8")) if PARAMS_PATH.exists() else {}
    won = [h for h, m in method.items() if m == "reversion_quantile"]
    out = {"version": 2, "fitted_on": "train companies only", "horizon": HORIZON, "min_history_months": MIN_HISTORY, "quantile_levels": list(QUANTILES),
           "features": list(FEATURES), "level_centre": centre, "level2_div": LEVEL2_DIV, "typical_change_scale": typical, "prior_months": PRIOR_MONTHS,
           "method_by_horizon": {str(h): m for h, m in method.items()}, "method": "reversion_quantile" if won else "naive_last",
           "models": {str(h): models[h].tolist() for h in models}, "naive_quantiles": {str(h): naive_q[h].tolist() for h in naive_q},
           "skill_by_horizon": {str(h): per[h]["pinball_skill"] for h in per}, "skill_h3": per[3]["pinball_skill"],
           "cv": {"per_horizon": {str(h): v for h, v in per.items()}, "ses_alpha_by_fold": cv["ses_alpha"], "damped_trend_by_fold": cv["holt_params"],
                  "calibration_h3": calibration_breakdown(cv["fans"], 3), "calibration_h6": calibration_breakdown(cv["fans"], 6)},
           "time_split_check": time_split, "seasonality": seasonal, "move_persistence": persist,
           "why": (f"the reversion quantile fan beat the naive fan on pinball loss with an interval above zero at horizons {', '.join(map(str, won))}" if won
                   else "the reversion quantile fan did not beat the naive fan with an interval above zero at any horizon, so the naive last value ships")}
    if "--arima" in argv:
        oof = {h: np.full(S.shape, np.nan) for h in per}
        for h, d in cv["fans"].items():
            oof[h][d["rows"], d["cols"]] = d["median"]
        print("ARIMA benchmark (several minutes)...")
        out["arima_benchmark"] = arima_benchmark(S, oof)
    elif "arima_benchmark" in prev:
        out["arima_benchmark"] = prev["arima_benchmark"]
    PARAMS_PATH.write_text(json.dumps(out, indent=1, allow_nan=False) + "\n", encoding="utf8")
    text = write_report(out, out["cv"]["calibration_h3"], out["cv"]["calibration_h6"])
    OUT_MD.write_text(text, encoding="utf8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
