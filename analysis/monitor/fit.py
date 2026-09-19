"""Fit everything the monitor needs from TRAIN companies only, write monitor_params.json (and the rank model).

    PYTHONUTF8=1 PYTHONPATH=<repo>/.venv/Lib/site-packages:<repo> python -m analysis.monitor.fit

Fitted on train (holdout never enters):
  - behaviour clusters (behaviour.Clusterer), plus per-cluster score reference tables for the "vs cluster" percentile
  - control-chart floors: median of the raw robust scale over train companies, per monitored series
  - group variance law var = a + b / n for (i) the scale of a group-mean series and (ii) the 3-month change of a group mean,
    which gives the funnel limits (small groups get wide limits)
  - the rank model for "top customer quiet" alerts and the score cut of its top decile
Nothing here is tuned on the outcomes: the materiality thresholds in engine.py are fixed a priori.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import behaviour, control

PARAMS_PATH = Path(__file__).resolve().parent / "monitor_params.json"
CATEGORIES = ["payment_history", "amounts_owed", "stability"]  # categories charted (new_credit and mix are thin / static)
MIN_GROUP = 3


def month_grid(detail: pd.DataFrame) -> pd.DatetimeIndex:
    p = detail["period"]
    return pd.date_range(p.min(), p.max(), freq="MS")


def wide(detail: pd.DataFrame, col: str, grid=None) -> pd.DataFrame:
    """company x month matrix of `col` (NaN where not scored)."""
    grid = grid if grid is not None else month_grid(detail)
    return detail.pivot(index="company_id", columns="period", values=col).reindex(columns=grid)


def group_mean_series(scores: pd.DataFrame, group_of: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean score per group and month, and the number of scored members per group and month. Groups by group_id, NaN ids dropped."""
    g = group_of.reindex(scores.index)
    ok = g.notna()
    mean = scores[ok].groupby(g[ok]).mean()
    n = scores[ok].notna().groupby(g[ok]).sum()
    return mean, n


def fit_variance_law(dev: np.ndarray, n: np.ndarray, robust_scale_of=None) -> tuple[float, float]:
    """var = a + b / n. Per group size n (sizes with at least 30 observations) take the robust variance of `dev`
    (1.4826 * MAD squared, so heavy tails do not inflate it), then weighted least squares on 1 / n; a and b are clipped at 0."""
    sizes, var, w = [], [], []
    for k in np.unique(n):
        d = dev[n == k]
        if len(d) < 30:
            continue
        s = 1.4826 * np.median(np.abs(d - np.median(d)))
        sizes.append(k), var.append(s * s), w.append(len(d))
    x, y, w = 1.0 / np.asarray(sizes), np.asarray(var), np.asarray(w, dtype=float)
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    b = np.sum(w * (x - xm) * (y - ym)) / max(np.sum(w * (x - xm) ** 2), 1e-12)
    a = ym - b * xm
    return max(float(a), 0.0), max(float(b), 0.0)


def raw_scales(series_by_company: pd.DataFrame) -> np.ndarray:
    out = []
    for x in series_by_company.to_numpy(dtype=float):
        _, sc = control.raw_scale(x)
        out.extend(sc[np.isfinite(sc)])
    return np.asarray(out)


def fit_params(store: pd.DataFrame, detail: pd.DataFrame, group_of: pd.Series, train_ids: set[str]) -> dict:
    """`detail`: score frame (company_id, period, score, cat_*); `group_of`: company_id -> group_id; train_ids: fit companies."""
    train_ids = set(train_ids)
    grid = month_grid(detail)
    tr = detail[detail["company_id"].isin(train_ids)]
    score_w = wide(tr, "score", grid)

    # --- behaviour clusters
    tab = behaviour.behaviour_table(store[store["company_id"].isin(train_ids)])
    cl = behaviour.Clusterer.fit(tab)
    assign = cl.pop("train_assignment")
    clusterer = behaviour.Clusterer(cl)
    all_tab = behaviour.behaviour_table(store)
    member = clusterer.assign(all_tab)
    ref = {}
    for c in range(cl["k"]):
        ids = [i for i in assign.index[assign == c] if i in train_ids]
        rows = tr[tr["company_id"].isin(ids)]
        ref[str(c)] = {m: {"quantiles": np.nanpercentile(rows[col].dropna(), np.linspace(0, 100, 101)).round(3).tolist(),
                           "median": float(np.nanmedian(rows[col])), "scale": float(1.4826 * np.nanmedian(np.abs(rows[col].dropna() - np.nanmedian(rows[col]))))}
                       for m, col in [("score", "score")] + [(k, f"cat_{k}") for k in CATEGORIES]}

    # --- floors: median of the raw robust scale over train companies
    floors = {"score": float(np.median(raw_scales(score_w)))}
    for k in CATEGORIES:
        floors[k] = float(np.median(raw_scales(wide(tr, f"cat_{k}", grid))))
    # score gap to the cluster median (same-month peers in train): a within-company change of relative standing
    peer_med = pd.DataFrame({c: score_w.loc[[i for i in score_w.index if member.get(i) == c]].median() for c in range(cl["k"])}).T
    gap = pd.DataFrame({i: score_w.loc[i] - peer_med.loc[int(member[i])] for i in score_w.index if pd.notna(member.get(i))}).T
    floors["cluster_gap"] = float(np.median(raw_scales(gap)))

    # --- group variance laws (train groups of at least MIN_GROUP scored companies)
    mean, n = group_mean_series(score_w, group_of)
    big = n.index[(n.max(axis=1) >= MIN_GROUP)]
    v_scale, n_scale, v_d, n_d = [], [], [], []
    for gid in big:
        x = mean.loc[gid].to_numpy(dtype=float)
        _, sc = control.raw_scale(x)
        nn = n.loc[gid].to_numpy(dtype=float)
        ok = np.isfinite(sc) & (nn >= MIN_GROUP)
        v_scale.extend(sc[ok])
        n_scale.extend(nn[ok])
        d3 = pd.Series(x).diff(3).to_numpy()
        ok = np.isfinite(d3) & (nn >= MIN_GROUP)
        v_d.extend(d3[ok])
        n_d.extend(nn[ok])
    # scale law: the typical (median) raw scale of a group-mean series by group size, as a variance
    sizes = np.asarray(n_scale)
    med_var = {int(k): float(np.median(np.asarray(v_scale)[sizes == k]) ** 2) for k in np.unique(sizes) if (sizes == k).sum() >= 30}
    xs = np.array([1.0 / k for k in med_var]); ys = np.array(list(med_var.values()))
    b_s = max(float(np.cov(xs, ys)[0, 1] / max(np.var(xs, ddof=1), 1e-12)), 0.0)
    a_s = max(float(ys.mean() - b_s * xs.mean()), 0.0)
    d3 = np.asarray(v_d)
    mu_d = float(np.median(d3))
    a_d, b_d = fit_variance_law(d3, np.asarray(n_d))
    return {
        "version": 1,
        "fitted_on": "train companies only",
        "n_train_companies": len(train_ids),
        "control": {"method": control.METHOD, "floors": floors},
        "group": {"min_size": MIN_GROUP, "n_groups_fit": int(len(big)),
                  "scale_law": {"a": a_s, "b": b_s}, "delta3_law": {"a": a_d, "b": b_d, "median": mu_d, "z_limit": 3.0}},
        "clusters": {"model": cl, "reference": ref, "n_assigned": int(member.notna().sum())},
    }


def save(params: dict, path: Path = PARAMS_PATH) -> None:
    path.write_text(json.dumps(params, indent=1, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")


def load(path: Path = PARAMS_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf8"))


def main() -> None:
    from analysis.features.common import connect
    from analysis.models.gbm_y7y8 import _keys, load_store, load_y
    from analysis.evaluate.protocol import train_companies
    from product.score import fit as score_fit
    from product.score.run import score_store

    from . import topcustomer

    con = connect()
    store, _ = load_store(con)
    store = _keys(store)
    train_df = train_companies(con)
    train = set(train_df["company_id"].astype(str))
    detail = score_store(store, score_fit.load_reference(), with_reasons=False)
    comp = con.execute("SELECT company_id, group_id FROM companies").df()
    group_of = comp.assign(company_id=comp["company_id"].astype(str)).set_index("company_id")["group_id"]
    params = fit_params(store, detail, group_of, train)

    grid = store[["company_id", "period"]].drop_duplicates()
    y7, _y8, _ = load_y(con, grid)
    y7 = _keys(y7[["company_id", "period", "y7_top1_lost"]])
    info = topcustomer.fit_rank_model(store, y7, train)
    oof = topcustomer.oof_rank_scores(store, y7, train_df)   # the cut comes from out-of-fold scores, not from the fitted model's own training rows
    ev = topcustomer.onset_events(con).merge(oof, on=["company_id", "period"], how="left")
    ev = ev[ev["company_id"].isin(train) & ev["rank_score"].notna()]
    params["rank_model"] = {"info": info, "top_decile_cut": float(ev["rank_score"].quantile(0.90)), "n_train_onset_events": int(len(ev)), "cut_from": "out-of-fold scores of train onset events",
                            "note": "the night's TURNOVER card: ranks alerts, is not a probability"}
    save(params)
    k = params["clusters"]["model"]
    print(f"floors {params['control']['floors']}")
    print(f"group law scale {params['group']['scale_law']} delta3 {params['group']['delta3_law']} groups {params['group']['n_groups_fit']}")
    print(f"clusters k={k['k']} sizes {k['sizes']} silhouette {k['silhouette_by_k']}")
    print(f"rank model {info['rows']} rows, top-decile cut {params['rank_model']['top_decile_cut']:.3f} -> {PARAMS_PATH}")


if __name__ == "__main__":
    main()
