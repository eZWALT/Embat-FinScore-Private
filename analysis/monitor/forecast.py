"""Score persistence forecast with intervals (plan step 4). Smoothed level vs the naive last value, group-fold CV. Expect a tie.

    PYTHONUTF8=1 PYTHONPATH=<repo>/.venv/Lib/site-packages:<repo> python -m analysis.monitor.forecast

What is forecast: the company's own score h = 1..HORIZON months ahead, as a fan (median, 50% and 80% intervals), not a point estimate.
Methods (both flat forecasts from the last month):
  naive_last      the last score
  smoothed_level  simple exponential smoothing of the score, alpha chosen on training folds (grid ALPHAS), fixed for the final fit
Intervals: the forecast error e = score_{t+h} - forecast_t is divided by the company's own robust scale at t (control.raw_scale, floored at the
train floor) and its quantiles are taken from out-of-fold errors on train; the fan is forecast + quantile * scale, clipped to [0, 100].
Validation: 5-fold CV by group_id over train companies (alpha and quantiles fitted on the other folds only). MAE per horizon,
skill = 1 - MAE / MAE(naive) with a group-bootstrap interval of the paired difference, and empirical coverage of the intervals on held-out folds.
Holdout companies are never used. The shipped method is smoothed_level only if the lower end of its skill interval is above zero at every
horizon it is judged on; otherwise it is naive_last, said as such. Persistence of a score is not a prediction of financial outcomes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import control
from .fit import load as load_params, month_grid, wide

HORIZON = 6
ALPHAS = (0.2, 0.35, 0.5, 0.7, 0.85)
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
MIN_HISTORY = 4
N_BOOT = 400
SEED = 20260919
PARAMS_PATH = Path(__file__).resolve().parent / "forecast_params.json"
OUT_MD = Path(__file__).resolve().parent / "forecast_evaluation.md"


def ses_level(S: np.ndarray, alpha: float) -> np.ndarray:
    """Simple exponential smoothing along axis 1, NaN-aware (the level starts at the first score and carries through gaps)."""
    L = np.full_like(S, np.nan)
    cur = np.full(S.shape[0], np.nan)
    for t in range(S.shape[1]):
        x = S[:, t]
        new = np.where(np.isnan(cur), x, np.where(np.isnan(x), cur, alpha * x + (1 - alpha) * cur))
        cur = new
        L[:, t] = np.where(np.isnan(x) & np.isnan(cur), np.nan, cur)
    return L


def own_scale(S: np.ndarray, floor: float) -> np.ndarray:
    out = np.full_like(S, np.nan)
    for i in range(S.shape[0]):
        _, sc = control.raw_scale(S[i])
        out[i] = np.maximum(sc, floor)
    return out


def history_count(S: np.ndarray) -> np.ndarray:
    return np.cumsum(np.isfinite(S), axis=1)


def errors(S: np.ndarray, F: np.ndarray, h: int) -> np.ndarray:
    """Error at origin t for horizon h: score_{t+h} - forecast_t (NaN where either is missing)."""
    e = np.full_like(S, np.nan)
    e[:, : S.shape[1] - h] = S[:, h:] - F[:, : S.shape[1] - h]
    return e


def _mae_by_group(err: np.ndarray, rows: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-group sum of |error| and count over the selected origins, for the bootstrap."""
    idx = np.flatnonzero(rows.any(axis=1))
    a = np.abs(np.where(rows, err, np.nan))
    s = np.nansum(a, axis=1)[idx]
    c = np.isfinite(a).sum(axis=1)[idx]
    g = groups[idx]
    order = np.argsort(g, kind="stable")
    uniq, start = np.unique(g[order], return_index=True)
    return np.add.reduceat(s[order], start), np.add.reduceat(c[order], start)


def cross_validate(S: np.ndarray, ids: np.ndarray, groups: np.ndarray, folds: np.ndarray, floor: float) -> dict:
    n, T = S.shape
    hist = history_count(S)
    scale = own_scale(S, floor)
    naive = np.where(np.isfinite(S), S, np.nan)
    res: dict = {"alpha_by_fold": [], "per_horizon": {}}
    F_ses = {a: ses_level(S, a) for a in ALPHAS}
    # choose alpha per fold on the other folds (MAE over horizons 1..HORIZON)
    chosen = {}
    for k in np.unique(folds):
        tr = folds != k
        best, best_a = np.inf, ALPHAS[0]
        for a in ALPHAS:
            tot = 0.0
            for h in range(1, HORIZON + 1):
                e = np.abs(errors(S, F_ses[a], h))[tr]
                ok = (hist[tr] >= MIN_HISTORY) & np.isfinite(e)
                tot += float(np.mean(e[ok]))
            if tot < best:
                best, best_a = tot, a
        chosen[int(k)] = best_a
        res["alpha_by_fold"].append(best_a)
    rng = np.random.default_rng(SEED)
    for h in range(1, HORIZON + 1):
        e_naive = errors(S, naive, h)
        e_ses = np.full_like(S, np.nan)
        for k in np.unique(folds):
            te = folds == k
            e_ses[te] = errors(S, F_ses[chosen[int(k)]], h)[te]
        ok = (hist >= MIN_HISTORY) & np.isfinite(e_naive) & np.isfinite(e_ses)
        sn, cn = _mae_by_group(e_naive, ok, groups)
        ss, cs = _mae_by_group(e_ses, ok, groups)
        mae_n, mae_s = sn.sum() / cn.sum(), ss.sum() / cs.sum()
        skills = []
        for _ in range(N_BOOT):
            pick = rng.integers(0, len(sn), len(sn))
            skills.append(1 - ss[pick].sum() / max(sn[pick].sum(), 1e-9))
        res["per_horizon"][h] = {"n_origins": int(cn.sum()), "mae_naive": float(mae_n), "mae_smoothed": float(mae_s), "skill": float(1 - mae_s / mae_n),
                                 "skill_ci": [float(np.percentile(skills, 2.5)), float(np.percentile(skills, 97.5))]}
    return res | {"chosen_alpha": chosen}


def fit_quantiles(S: np.ndarray, F: np.ndarray, scale: np.ndarray, hist: np.ndarray, rows: np.ndarray) -> dict:
    """Quantiles of the scaled error per horizon from the selected companies (rows: boolean, per company)."""
    out = {}
    for h in range(1, HORIZON + 1):
        e = errors(S, F, h) / scale
        ok = (hist >= MIN_HISTORY) & np.isfinite(e) & rows[:, None]
        out[h] = np.quantile(e[ok], QUANTILES).tolist()
    return out


def coverage(S, F, scale, hist, rows, q_by_h) -> dict:
    res = {}
    for h in range(1, HORIZON + 1):
        e = errors(S, F, h)
        q = q_by_h[h]
        ok = (hist >= MIN_HISTORY) & np.isfinite(e) & np.isfinite(scale) & rows[:, None]
        z = (e / scale)[ok]
        res[h] = {"cover50": float(np.mean((z >= q[1]) & (z <= q[3]))), "cover80": float(np.mean((z >= q[0]) & (z <= q[4]))), "n": int(ok.sum())}
    return res


def cv_coverage(S, ids, folds, floor, F, alpha_of_fold=None) -> dict:
    """Held-out coverage: quantiles from the other folds' errors, applied to the held-out fold."""
    hist = history_count(S)
    scale = own_scale(S, floor)
    out = {h: {"c50": [], "c80": [], "n": 0} for h in range(1, HORIZON + 1)}
    for k in np.unique(folds):
        q = fit_quantiles(S, F, scale, hist, folds != k)
        cov = coverage(S, F, scale, hist, folds == k, q)
        for h, c in cov.items():
            out[h]["c50"].append(c["cover50"] * c["n"]); out[h]["c80"].append(c["cover80"] * c["n"]); out[h]["n"] += c["n"]
    return {h: {"cover50": sum(v["c50"]) / v["n"], "cover80": sum(v["c80"]) / v["n"], "n": v["n"]} for h, v in out.items()}


def forecast_rows(S: np.ndarray, params: dict, floor: float) -> list[dict | None]:
    """Fan for every company from its last scored month (rows of S), from forecast_params.json. None when the history is too short."""
    method, alpha = params["method"], params["alpha"]
    F = ses_level(S, alpha) if method == "smoothed_level" else S.copy()
    naive_last = S
    scale = own_scale(S, floor)
    hist = history_count(S)
    out = []
    for i in range(S.shape[0]):
        obs = np.flatnonzero(np.isfinite(S[i]))
        if len(obs) == 0 or hist[i, -1] < MIN_HISTORY:
            out.append(None)
            continue
        t = obs[-1]
        sc = scale[i, t] if np.isfinite(scale[i, t]) else floor
        pts = []
        for h in range(1, HORIZON + 1):
            q = np.asarray(params["quantiles"][str(h)])
            lo80, lo50, med, hi50, hi80 = np.clip(F[i, t] + q * sc, 0.0, 100.0)
            pts.append({"h": h, "median": float(med), "lo50": float(lo50), "hi50": float(hi50), "lo80": float(lo80), "hi80": float(hi80)})
        out.append({"origin_index": int(t), "naive_last": float(naive_last[i, t]), "points": pts})
    return out


def main() -> int:
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
    params = load_params()
    floor = params["control"]["floors"]["score"]
    grid = month_grid(detail)
    W = wide(detail[detail["company_id"].isin(set(train_df["company_id"]))], "score", grid)
    fold_of = group_folds(train_df, n=5, seed=FOLD_SEED).set_index("company_id")["fold"]
    grp_of = train_df.set_index("company_id")["group_id"].astype(str)
    ids = W.index.to_numpy()
    S = W.to_numpy(dtype=float)
    folds = fold_of.reindex(ids).to_numpy()
    groups = grp_of.reindex(ids).to_numpy()
    print("cross-validating...")
    cv = cross_validate(S, ids, groups, folds, floor)
    alpha = float(pd.Series(list(cv["chosen_alpha"].values())).mode().iloc[0])
    per = cv["per_horizon"]
    judged = [h for h in per if per[h]["n_origins"] > 200]
    smoothed_wins = all(per[h]["skill_ci"][0] > 0 for h in judged)
    method = "smoothed_level" if smoothed_wins else "naive_last"
    F_ship = ses_level(S, alpha) if method == "smoothed_level" else S.copy()
    all_train = np.ones(len(ids), dtype=bool)
    q_by_h = fit_quantiles(S, F_ship, own_scale(S, floor), history_count(S), all_train)
    cov = cv_coverage(S, ids, folds, floor, F_ship)
    out = {"version": 1, "fitted_on": "train companies only", "method": method, "alpha": alpha, "horizon": HORIZON, "min_history_months": MIN_HISTORY,
           "quantile_levels": list(QUANTILES), "quantiles": {str(h): q for h, q in q_by_h.items()},
           "skill_h3": per[3]["skill"] if 3 in per else None,
           "cv": {"alpha_by_fold": cv["alpha_by_fold"], "per_horizon": {str(h): v for h, v in per.items()}, "coverage": {str(h): v for h, v in cov.items()}},
           "why": ("smoothed level beat the naive last value at every judged horizon (interval above zero)" if smoothed_wins else
                   "smoothed level did not beat the naive last value with an interval above zero, so the simpler naive last value ships")}
    PARAMS_PATH.write_text(json.dumps(out, indent=1, allow_nan=False) + "\n", encoding="utf8")
    L = ["# Score forecast evaluation (train companies, group-fold CV, holdout untouched)", "",
         f"Method shipped: **{method}** ({out['why']}). Smoothing alpha (mode over folds): {alpha}. Flat forecast from the last month, fan from scaled out-of-fold errors.", "",
         "| horizon (months) | origins | MAE naive | MAE smoothed | skill of smoothed vs naive (95% CI) | 50% interval covers | 80% interval covers |", "|---|---|---|---|---|---|---|"]
    for h, v in per.items():
        L.append(f"| {h} | {v['n_origins']} | {v['mae_naive']:.2f} | {v['mae_smoothed']:.2f} | {v['skill']:+.1%} ({v['skill_ci'][0]:+.1%} to {v['skill_ci'][1]:+.1%}) | {cov[h]['cover50']:.0%} | {cov[h]['cover80']:.0%} |")
    L += ["", "Skill = 1 - MAE(smoothed) / MAE(naive); zero is a tie. Coverage is measured on held-out folds with quantiles fitted on the other folds.",
          "The score is persistent, so the naive last value is hard to beat; the fan says how far the score usually moves, it does not say which way."]
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
