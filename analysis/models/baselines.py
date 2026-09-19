"""Leakage-safe baselines: Y1 naive forecasts, single-feature Y2, Javier score.

Holdout companies never enter a fit. Naive methods have no parameters.
Single-feature signs are chosen on train only; metrics are holdout-only.
Javier ``fit_ref`` uses train company-months only, then ``run_score`` on holdout.

Family B is never used as X for Y2 (same-path liquidity columns).
"""
from __future__ import annotations

import csv
import importlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.features.common import ANALYSIS, connect, load_holdout, train_mask
from analysis.features.grid import monthly_grid
from analysis.score_pipeline import build_features, fit_ref, run_score
from analysis.targets.y1_forecast import build as build_y1, cash_month_panel
from analysis.targets.y2_stress import build as build_y2

try:
    from analysis.evaluate.protocol import auroc, assert_no_holdout, leakage_check
except Exception:  # slot 8 may still be writing protocol.py
    def auroc(y_true, y_score) -> float:
        d = pd.DataFrame({"y": y_true, "s": y_score}).dropna()
        if d.empty:
            return float("nan")
        y = d["y"].to_numpy(dtype=float)
        n1 = int((y == 1).sum())
        n0 = int((y == 0).sum())
        if n1 == 0 or n0 == 0:
            return float("nan")
        r = d["s"].rank(method="average").to_numpy(dtype=float)
        return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))

    def assert_no_holdout(index_or_ids) -> None:
        if isinstance(index_or_ids, pd.DataFrame) and "company_id" in index_or_ids.columns:
            ids = set(index_or_ids["company_id"].astype(str))
        elif isinstance(index_or_ids, (pd.Series, pd.Index)):
            ids = set(index_or_ids.astype(str))
        else:
            ids = {str(x) for x in index_or_ids}
        leaked = ids & load_holdout()
        if leaked:
            sample = ", ".join(sorted(leaked)[:8])
            raise AssertionError(f"holdout companies present: {sample}")

    def leakage_check(X_cols, y_col, forbidden_prefixes=None) -> dict:
        x_list = [str(c) for c in X_cols]
        issues = []
        if y_col in set(x_list):
            issues.append(f"y_col {y_col!r} is in X")
        for raw in forbidden_prefixes or []:
            pref = raw if str(raw).endswith("_") else f"{raw}_"
            bad = [c for c in x_list if c.startswith(pref)]
            if bad:
                issues.append(f"forbidden prefix {pref!r}: {bad[:8]}")
        return {"ok": len(issues) == 0, "issues": issues}


REGISTRY = ANALYSIS / "experiments" / "registry.csv"
Y2_COL = "y2_neg_2of3"
AGENT = "d0748ae4"
FORBIDDEN_Y2 = ("b",)

# Allowed X for Y2. Never family B. Skip a family if the module is missing.
ALLOWED_FAMILIES = (
    ("a", "analysis.features.cashflow"),
    ("c", "analysis.features.ops"),
    ("d", "analysis.features.counterparties"),
    ("e", "analysis.features.invoices"),
    ("f", "analysis.features.debt"),
    ("g", "analysis.features.products"),
    ("h", "analysis.features.groupctx"),
)

# (y_col, series on cash_month_panel, horizon)
Y1_SPEC = (
    ("y1_net_h1", "net", 1),
    ("y1_net_h3", "net", 3),
    ("y1_in_h1", "op_in", 1),
    ("y1_in_h3", "op_in", 3),
    ("y1_liq_h1", "liq", 1),
    ("y1_liq_h3", "liq", 3),
)
NAIVE_METHODS = ("last_value", "hist_mean", "seasonal_naive_12")


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["company_id"] = out["company_id"].astype(str)
    if "period" in out.columns:
        out["period"] = pd.to_datetime(out["period"])
    if "month" in out.columns:
        out["month"] = pd.to_datetime(out["month"])
    return out


def mae(y, yhat) -> float:
    d = pd.DataFrame({"y": y, "p": yhat}).dropna()
    if d.empty:
        return float("nan")
    return float((d["y"] - d["p"]).abs().mean())


def smape(y, yhat) -> float:
    """sMAPE in [0, 2]. 0/0 -> 0."""
    d = pd.DataFrame({"y": y, "p": yhat}).dropna()
    if d.empty:
        return float("nan")
    denom = d["y"].abs() + d["p"].abs()
    contrib = pd.Series(0.0, index=d.index)
    nz = denom > 0
    contrib.loc[nz] = 2.0 * (d.loc[nz, "y"] - d.loc[nz, "p"]).abs() / denom.loc[nz]
    return float(contrib.mean())


def naive_forecasts(panel: pd.DataFrame) -> pd.DataFrame:
    """Origin-t forecasts that use only the series at or before t.

    seasonal_naive_12 is Hyndman seasonal naive (m=12): ŷ_{t+h} = y_{t+h-12}.
    """
    p = _keys(panel).sort_values(["company_id", "month"]).reset_index(drop=True)
    g = p.groupby("company_id", sort=False)
    for col in ("net", "op_in", "liq"):
        p[f"{col}_last"] = p[col]
        p[f"{col}_mean"] = g[col].transform(lambda s: s.expanding(min_periods=1).mean())
        p[f"{col}_seas_h1"] = g[col].shift(11)  # y_{t+1-12}
        p[f"{col}_seas_h3"] = g[col].shift(9)  # y_{t+3-12}
    return p


def pred_col(series: str, method: str, horizon: int) -> str:
    if method == "last_value":
        return f"{series}_last"
    if method == "hist_mean":
        return f"{series}_mean"
    if method == "seasonal_naive_12":
        return f"{series}_seas_h{horizon}"
    raise ValueError(method)


def mase_scale(panel: pd.DataFrame, series: str, train_ids: set[str], m: int = 12) -> float:
    """In-sample seasonal-naive MAE on TRAIN companies only (fallback m=1)."""
    d = _keys(panel).sort_values(["company_id", "month"])
    tr = d["company_id"].isin(train_ids)
    lag = d.groupby("company_id")[series].shift(m)
    err = (d[series] - lag).abs()
    err = err[tr.to_numpy() & err.notna().to_numpy() & d[series].notna().to_numpy()]
    val = float(err.mean()) if len(err) else float("nan")
    if (not np.isfinite(val) or val == 0.0) and m != 1:
        return mase_scale(panel, series, train_ids, 1)
    return val


def evaluate_y1(
    y1: pd.DataFrame,
    panel: pd.DataFrame,
    hold_ids: set[str],
    train_ids: set[str],
) -> pd.DataFrame:
    fc = naive_forecasts(panel)
    y = _keys(y1)
    y = y.merge(
        fc,
        left_on=["company_id", "period"],
        right_on=["company_id", "month"],
        how="left",
    )
    ho = y["company_id"].isin(hold_ids)
    scales = {s: mase_scale(panel, s, train_ids, 12) for s in ("net", "op_in", "liq")}
    rows = []
    for y_col, series, h in Y1_SPEC:
        y_ho = y.loc[ho, y_col]
        n_y = int(y_ho.notna().sum())
        seas_pc = pred_col(series, "seasonal_naive_12", h)
        seas_ok = ho & y[y_col].notna() & y[seas_pc].notna()
        for support, mask in (("all", ho), ("seasonal_ok", seas_ok)):
            for method in NAIVE_METHODS:
                pc = pred_col(series, method, h)
                pair = y.loc[mask, [y_col, pc]].dropna()
                n = int(len(pair))
                mae_v = mae(pair[y_col], pair[pc])
                sm_v = smape(pair[y_col], pair[pc])
                sc = scales[series]
                mase_v = (
                    float(mae_v / sc)
                    if (np.isfinite(mae_v) and np.isfinite(sc) and sc > 0)
                    else float("nan")
                )
                rows.append(
                    {
                        "y": y_col,
                        "series": series,
                        "horizon": h,
                        "model": method,
                        "support": support,
                        "n_y": n_y,
                        "n_pred": n,
                        "coverage": (n / n_y) if n_y else float("nan"),
                        "mae": mae_v,
                        "smape": sm_v,
                        "mase": mase_v,
                        "scale": sc,
                    }
                )
    return pd.DataFrame(rows)


def load_allowed_x(con, grid: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build allowed Y2 families. Skip missing / broken modules. Never B."""
    keys = _keys(grid[["company_id", "period"]])
    panel = keys.copy()
    loaded: list[str] = []
    skipped: list[str] = []
    for letter, modname in ALLOWED_FAMILIES:
        try:
            mod = importlib.import_module(modname)
            part = mod.build(con, keys.copy())
        except Exception as exc:
            skipped.append(f"{letter}:{type(exc).__name__}:{exc}")
            print(f"skip family {letter}: {type(exc).__name__}: {exc}")
            continue
        extra = [c for c in part.columns if c not in {"company_id", "period"}]
        if any(c.startswith("b_") for c in extra):
            skipped.append(f"{letter}:contains_b_")
            print(f"skip family {letter}: produced b_ columns")
            continue
        part = _keys(part)
        clash = set(extra) & set(panel.columns)
        if clash:
            skipped.append(f"{letter}:clash:{sorted(clash)[:4]}")
            print(f"skip family {letter}: column clash {clash}")
            continue
        panel = panel.merge(part, on=["company_id", "period"], how="left")
        loaded.append(letter)
        print(f"loaded family {letter}: {len(extra)} cols")
    return panel, loaded, skipped


def _numeric_feature_cols(X: pd.DataFrame) -> list[str]:
    skip = {"company_id", "period", "month", "freq", "first_month", "first_week"}
    cols = []
    for c in X.columns:
        if c in skip or c.startswith("y") or c.startswith("b_"):
            continue
        if pd.api.types.is_numeric_dtype(X[c]):
            cols.append(c)
    return cols


def choose_sign(y: pd.Series, x: pd.Series) -> int:
    """+1 or -1 so that higher score ranks the positive class on TRAIN."""
    auc_p = auroc(y, x)
    auc_n = auroc(y, -x)
    if not np.isfinite(auc_p) and not np.isfinite(auc_n):
        return 1
    if not np.isfinite(auc_p):
        return -1
    if not np.isfinite(auc_n):
        return 1
    return -1 if auc_n > auc_p else 1


def single_feature_auroc(
    X: pd.DataFrame,
    y: pd.Series,
    is_train: pd.Series,
    is_hold: pd.Series,
) -> pd.DataFrame:
    rows = []
    y_tr, y_ho = y[is_train], y[is_hold]
    for col in _numeric_feature_cols(X):
        x = pd.to_numeric(X[col], errors="coerce")
        if x[is_train].nunique(dropna=True) < 2:
            continue
        sign = choose_sign(y_tr, x[is_train])
        train_auc = auroc(y_tr, sign * x[is_train])
        ho_pair = pd.DataFrame({"y": y_ho, "s": sign * x[is_hold]}).dropna()
        n_y = int(y_ho.notna().sum())
        n = int(len(ho_pair))
        ho_auc = auroc(ho_pair["y"], ho_pair["s"])
        if not np.isfinite(ho_auc):
            continue
        rows.append(
            {
                "feature": col,
                "family": col.split("_", 1)[0],
                "sign": sign,
                "train_auc": train_auc,
                "holdout_auc": ho_auc,
                "n_pred": n,
                "n_y": n_y,
                "coverage": (n / n_y) if n_y else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("holdout_auc", ascending=False).reset_index(drop=True)


def javier_holdout_auroc(
    con,
    y2: pd.DataFrame,
    hold_ids: set[str],
    train_ids: set[str],
) -> dict:
    P = _keys(build_features(con))
    train_P = P.loc[P["company_id"].isin(train_ids)].copy()
    assert_no_holdout(train_P)
    ref = fit_ref(train_P)
    # Score everyone with the train-only ref (holdout companies were not in fit_ref).
    Sc = run_score(P, ref)["Sc"]
    Sc = _keys(Sc)
    lab = _keys(y2)[["company_id", "period", Y2_COL]].rename(columns={"period": "month"})
    m = Sc.merge(lab, on=["company_id", "month"], how="left")
    ho = m["company_id"].isin(hold_ids)
    pair = m.loc[ho, ["score", Y2_COL]].dropna()
    y_all = y2.loc[y2["company_id"].isin(hold_ids), Y2_COL]
    n_y = int(y_all.notna().sum())
    n_pos = int((y_all == 1).sum())
    n = int(len(pair))
    auc = auroc(pair[Y2_COL], -pair["score"])
    return {
        "n_y": n_y,
        "n_pos": n_pos,
        "n_pred": n,
        "coverage": (n / n_y) if n_y else float("nan"),
        "auroc": auc,
        "n_train_ref": int(len(train_P)),
        "n_scored": int(len(Sc)),
    }


def append_registry(rows: list[dict]) -> None:
    """Append rows. Skip keys (agent, y, model, split, metric) already present."""
    if not rows:
        return
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    with REGISTRY.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        seen = {(r.get("agent"), r.get("y"), r.get("model"), r.get("split"), r.get("metric")) for r in reader}
    fresh = []
    for r in rows:
        key = (str(r.get("agent", "")), str(r.get("y", "")), str(r.get("model", "")), str(r.get("split", "")), str(r.get("metric", "")))
        if key in seen:
            continue
        fresh.append(r)
        seen.add(key)
    if not fresh:
        print("registry: nothing new to append")
        return
    with REGISTRY.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        for r in fresh:
            w.writerow({k: r.get(k, "") for k in header})
    print(f"appended {len(fresh)} registry rows")


def _horizon_summary(y1_tbl: pd.DataFrame) -> pd.DataFrame:
    """Mean MASE / sMAPE across net, op_in, liq, by method, horizon, support."""
    cols = [c for c in ("support", "model", "horizon") if c in y1_tbl.columns]
    return y1_tbl.groupby(cols, as_index=False).agg(
        mase=("mase", "mean"),
        smape=("smape", "mean"),
        coverage=("coverage", "mean"),
        n_pred=("n_pred", "sum"),
    )


def _reg_row(ts, *, families, y, model, metric, value, coverage, notes) -> dict:
    return {
        "ts": ts,
        "round": "R1",
        "wave": 2,
        "agent": AGENT,
        "x_families": families,
        "y": y,
        "model": model,
        "split": "holdout",
        "metric": metric,
        "value": f"{value:.6g}" if isinstance(value, (int, float, np.floating)) and np.isfinite(value) else (value if isinstance(value, str) else ""),
        "coverage": f"{coverage:.4f}" if isinstance(coverage, (int, float, np.floating)) and np.isfinite(coverage) else "",
        "notes": notes,
    }


def run() -> dict:
    hold_ids = load_holdout()
    con = connect()
    grid = monthly_grid(con)
    grid = _keys(grid)
    is_tr = train_mask(grid["company_id"])
    train_ids = set(grid.loc[is_tr, "company_id"].astype(str))
    assert_no_holdout(pd.Series(sorted(train_ids)))
    print(f"grid={grid.shape} train_cos={len(train_ids)} hold_cos={len(hold_ids)}")

    print("building Y1 / Y2 / cash panel")
    y1 = _keys(build_y1(con, grid))
    y2 = _keys(build_y2(con, grid))
    panel = cash_month_panel(con)

    print("Y1 naive forecasts")
    y1_tbl = evaluate_y1(y1, panel, hold_ids, train_ids)
    y1_all = y1_tbl[y1_tbl["support"] == "all"] if "support" in y1_tbl.columns else y1_tbl
    y1_ok = y1_tbl[y1_tbl["support"] == "seasonal_ok"] if "support" in y1_tbl.columns else y1_tbl.iloc[0:0]
    print(y1_all.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("horizon means (all holdout Y)")
    print(_horizon_summary(y1_all).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if not y1_ok.empty:
        print("horizon means (matched seasonal support)")
        print(_horizon_summary(y1_ok).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("allowed X families for Y2")
    X, loaded, skipped = load_allowed_x(con, grid)
    x_cols = _numeric_feature_cols(X)
    leak = leakage_check(x_cols, Y2_COL, FORBIDDEN_Y2)
    if not leak["ok"]:
        raise RuntimeError(f"leakage: {leak['issues']}")
    print(f"X cols={len(x_cols)} families={loaded} leak_ok={leak['ok']}")

    xy = X.merge(y2[["company_id", "period", Y2_COL]], on=["company_id", "period"], how="left")
    is_train = xy["company_id"].isin(train_ids)
    is_hold = xy["company_id"].isin(hold_ids)
    feat_tbl = single_feature_auroc(xy, xy[Y2_COL], is_train, is_hold)
    top5 = feat_tbl.head(5) if not feat_tbl.empty else feat_tbl
    top5_tr = (
        feat_tbl.sort_values("train_auc", ascending=False).head(5).reset_index(drop=True)
        if not feat_tbl.empty
        else feat_tbl
    )
    print("top 5 holdout single-feature AUROC (sign from train)")
    print("(none)" if top5.empty else top5.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("top 5 train single-feature AUROC (holdout of those)")
    print("(none)" if top5_tr.empty else top5_tr.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("Javier score (ref fit on train only)")
    jav = javier_holdout_auroc(con, y2, hold_ids, train_ids)
    print(jav)
    con.close()

    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    fam = "+".join(s.upper() for s in loaded) if loaded else "-"
    reg: list[dict] = []
    for _, r in y1_tbl.iterrows():
        model = r["model"] if r.get("support", "all") == "all" else f"{r['model']}_seas_ok"
        notes = f"series={r['series']}; mae={r['mae']:.4g}; scale={r['scale']:.4g}; support={r.get('support', 'all')}"
        for metric in ("mase", "smape"):
            reg.append(
                _reg_row(
                    ts,
                    families="-",
                    y=r["y"],
                    model=model,
                    metric=metric,
                    value=r[metric],
                    coverage=r["coverage"],
                    notes=notes,
                )
            )
    for _, r in top5.iterrows():
        sign = "+" if r["sign"] == 1 else "-"
        reg.append(
            _reg_row(
                ts,
                families=r["family"].upper(),
                y=Y2_COL,
                model=f"single_{r['feature']}",
                metric="auroc",
                value=r["holdout_auc"],
                coverage=r["coverage"],
                notes=f"sign={sign}; train_auc={r['train_auc']:.4f}; families={fam}",
            )
        )
    for _, r in top5_tr.iterrows():
        sign = "+" if r["sign"] == 1 else "-"
        reg.append(
            _reg_row(
                ts,
                families=r["family"].upper(),
                y=Y2_COL,
                model=f"single_{r['feature']}_trainpick",
                metric="auroc",
                value=r["holdout_auc"],
                coverage=r["coverage"],
                notes=f"selected on train; sign={sign}; train_auc={r['train_auc']:.4f}",
            )
        )
    reg.append(
        _reg_row(
            ts,
            families="javier14",
            y=Y2_COL,
            model="javier_score",
            metric="auroc_neg_score",
            value=jav["auroc"],
            coverage=jav["coverage"],
            notes=f"fit_ref on train company-months only; AUROC of -score; n_pos={jav.get('n_pos')}; scored {jav['n_pred']}/{jav['n_y']} holdout Y",
        )
    )
    reg.append(
        _reg_row(
            ts,
            families="javier14",
            y=Y2_COL,
            model="javier_score",
            metric="coverage_vs_all_y",
            value=jav["coverage"],
            coverage=jav["coverage"],
            notes=f"n_pred={jav['n_pred']} of n_y={jav['n_y']}; cov_w>=0.5",
        )
    )
    reg.append(
        _reg_row(
            ts,
            families="-",
            y=Y2_COL,
            model="holdout_base",
            metric="n_pos",
            value=float(jav.get("n_pos", float("nan"))),
            coverage=jav["coverage"],
            notes=f"holdout labels n={jav['n_y']}; base rate noisy for AUROC",
        )
    )
    append_registry(reg)
    return {
        "y1": y1_tbl,
        "y1_sum": _horizon_summary(y1_all),
        "feat": feat_tbl,
        "top5": top5,
        "top5_train": top5_tr,
        "javier": jav,
        "loaded": loaded,
        "skipped": skipped,
        "n_x": len(x_cols),
        "n_train": int(is_tr.sum()),
        "n_hold": int((~is_tr).sum()),
        "n_reg": len(reg),
    }


if __name__ == "__main__":
    run()
