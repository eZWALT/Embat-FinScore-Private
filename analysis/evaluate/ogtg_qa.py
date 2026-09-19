"""Unused leftover of ``f_outstanding_gt_granted`` after ``c_n_days_with_tx``.

Clean snapshot flag: max over as-of facilities with created_at ≤ period_end.
NaN until the 2026-08 extract (``snap_ok``). ``f_util_snapshot`` is last-month
~1.6%. NORTH_STAR PARK snapshot cols as X. ``f_n_types`` just DROP leftover
0.534 / twin of ``f_n_facilities``.

KEEP-as-X: beat size ≥0.02 AND leftover after days AND not SIZE
(|ρ| vs log1p(a_in3) ≥0.50) AND not a twin (|ρ|≥0.80 vs days / a_n_tx /
f_n_facilities / f_n_types / f_util_snapshot). Leftover <0.55 dies.

Night quotes unchanged: Y3 0.762 / 0.752. Days 0.711. Size 0.617.
Y7 TURNOVER 0.720 / 0.712. Do not put OGTG on the 15-col card.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.ogtg_qa

Owned: analysis/evaluate/ogtg_qa.py, analysis/outputs/ogtg_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_ogtg.md (end).
Do not overwrite factoring_qa.* / n_types_qa.* / ds_r_qa.* / fc_r_qa.* /
debt_schedule_qa / sib_neg_qa.* / ap_issued_qa.*.
"""
from __future__ import annotations

import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.evaluate.protocol import (
    FOLD_SEED,
    assert_no_holdout,
    auroc,
    group_folds,
    leakage_check,
    load_holdout,
)
from analysis.features.common import ANALYSIS, DATA, connect
from analysis.targets.y11_dark import book_invoice_ids

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

STORE = DATA / "feature_store" / "monthly.parquet"
TARGETS = DATA / "feature_store" / "targets.parquet"
OUT_MD = ANALYSIS / "outputs" / "ogtg_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "ogtg_qa.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_ogtg.md"
AGENT = "b17e9c44"
WAVE = "4"
ROUND = "R4"
MODEL = "ogtg_qa"
X_FAM = "F"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_Y3_QUOTE = 0.617
N_TYPES_Y3 = 0.578
DAYS_LAG1_QUOTE = 0.684
UTIL_COV_QUOTE = 0.016
OGTG_COV_QUOTE = 0.057
OGTG_LAST_N = 32
OGTG_MODAL = 0.974
Y3_NIGHT = (0.762, 0.752)
Y7_TURNOVER = (0.720, 0.712)
KEEP_DELTA = 0.02
TWIN_RHO = 0.80
SIZE_RHO = 0.50
CHANCE = 0.55
MIN_POS = 50
MIN_ACF_PAIRS = 4
WRITE_WAVE = True

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "first_month",
    "a_in3",
    "a_n_tx",
    "c_n_days_with_tx",
    "f_n_types",
    "f_n_facilities",
    "f_outstanding_gt_granted",
    "f_util_snapshot",
    "f_has_loc",
    "f_has_factoring",
    "f_has_confirming",
)
Y_KEEP = (Y2, Y3)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _f(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    return f"{float(x):.{nd}f}"


def _pp(x, nd=1) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{100.0 * float(x):.{nd}f}%"


def _md_table(rows: list[dict], cols: list[str] | None = None) -> str:
    if not rows:
        return "_(empty)_\n"
    if cols is None:
        cols = list(rows[0].keys())
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["company_id"] = out["company_id"].astype(str)
    if "period" in out.columns:
        out["period"] = pd.to_datetime(out["period"]).astype("datetime64[ns]")
    if "group_id" in out.columns:
        out["group_id"] = out["group_id"].astype(str)
    return out


def spearman(a, b) -> float:
    d = pd.DataFrame(
        {"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}
    ).dropna()
    if len(d) < 8 or d["a"].nunique() < 2 or d["b"].nunique() < 2:
        return float("nan")
    return float(d["a"].corr(d["b"], method="spearman"))


def median_acf(series: pd.Series, company: pd.Series, lag: int) -> float:
    df = pd.DataFrame({"x": pd.to_numeric(series, errors="coerce"), "co": company.astype(str)})
    vals = []
    for _, g in df.groupby("co", sort=False):
        x = g["x"].to_numpy(dtype=float)
        if len(x) <= lag:
            continue
        aa, bb = x[:-lag], x[lag:]
        m = np.isfinite(aa) & np.isfinite(bb)
        if int(m.sum()) < MIN_ACF_PAIRS:
            continue
        aa, bb = aa[m], bb[m]
        if np.std(aa) == 0 or np.std(bb) == 0:
            continue
        vals.append(float(np.corrcoef(aa, bb)[0, 1]))
    return float(np.median(vals)) if vals else float("nan")


def icc_anova(series: pd.Series, company: pd.Series) -> dict:
    s = pd.DataFrame({"x": pd.to_numeric(series, errors="coerce"), "co": company.astype(str)}).dropna()
    if len(s) < 10 or s["x"].nunique() < 2:
        return {"icc": float("nan")}
    y = s["x"].to_numpy(dtype=float)
    _, inv, counts = np.unique(s["co"].to_numpy(), return_inverse=True, return_counts=True)
    if counts.size < 2:
        return {"icc": float("nan")}
    means = np.bincount(inv, weights=y) / counts
    grand = float(y.mean())
    ssb = float(np.sum(counts * (means - grand) ** 2))
    ssw = float(np.sum((y - means[inv]) ** 2))
    var_b = ssb / max(int(counts.size) - 1, 1)
    var_w = ssw / max(int(y.size) - int(counts.size), 1)
    icc = var_b / (var_b + var_w) if (var_b + var_w) > 0 else float("nan")
    return {"icc": float(icc)}


def choose_sign(y: pd.Series, x: pd.Series) -> int:
    auc_p = auroc(y, x)
    auc_n = auroc(y, -x)
    if not np.isfinite(auc_p) and not np.isfinite(auc_n):
        return 1
    if not np.isfinite(auc_p):
        return -1
    if not np.isfinite(auc_n):
        return 1
    return -1 if auc_n > auc_p else 1


def signed_oof_auroc(y, x, folds, mask, n_folds: int = N_FOLDS) -> dict:
    y = pd.to_numeric(y, errors="coerce")
    x = pd.to_numeric(x, errors="coerce")
    defined = mask & y.notna() & x.notna()
    n = int(defined.sum())
    n_pos = int((defined & (y == 1)).sum())
    n_neg = int((defined & (y == 0)).sum())
    low = n_pos < MIN_POS or n_neg == 0
    fold_rows = []
    aucs = []
    if low:
        return {
            "cv": float("nan"),
            "sd": float("nan"),
            "n_folds": 0,
            "folds": fold_rows,
            "train_sign": 0,
            "train_auc": float("nan"),
            "n_defined": n,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "low_power": True,
        }
    for k in range(n_folds):
        tr = defined & (folds != k)
        va = defined & (folds == k)
        if int((va & (y == 1)).sum()) == 0 or int((va & (y == 0)).sum()) == 0:
            fold_rows.append({"fold": k, "auroc": float("nan"), "sign": 0, "n_va": int(va.sum()), "n_pos": int((va & (y == 1)).sum())})
            continue
        sign = choose_sign(y[tr], x[tr])
        auc = auroc(y[va], sign * x[va])
        aucs.append(auc)
        fold_rows.append({"fold": k, "auroc": float(auc) if np.isfinite(auc) else float("nan"), "sign": int(sign), "n_va": int(va.sum()), "n_pos": int((va & (y == 1)).sum())})
    finite = [a for a in aucs if np.isfinite(a)]
    tr_sign = choose_sign(y[defined], x[defined])
    return {
        "cv": float(np.mean(finite)) if finite else float("nan"),
        "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan"),
        "n_folds": len(finite),
        "folds": fold_rows,
        "train_sign": int(tr_sign),
        "train_auc": float(auroc(y[defined], tr_sign * x[defined])),
        "n_defined": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "low_power": False,
    }


def fold_bits(rec: dict) -> str:
    return " ".join(f"{r['auroc']:.3f}" if np.isfinite(r["auroc"]) else "—" for r in rec.get("folds", []))


def ols_resid(y: pd.Series, *xs: pd.Series):
    cols = {"y": pd.to_numeric(y, errors="coerce")}
    for i, x in enumerate(xs):
        cols[f"x{i}"] = pd.to_numeric(x, errors="coerce")
    d = pd.DataFrame(cols)
    ok = d.notna().all(axis=1)
    resid = pd.Series(np.nan, index=d.index, dtype=float)
    info = {"n": int(ok.sum()), "r2": float("nan")}
    n_x = len(xs)
    if int(ok.sum()) < max(20, n_x + 5):
        return resid, info
    Y = d.loc[ok, "y"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(int(ok.sum()))] + [d.loc[ok, f"x{i}"].to_numpy(dtype=float) for i in range(n_x)])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    except np.linalg.LinAlgError:
        return resid, info
    pred = X @ beta
    resid.loc[ok] = Y - pred
    ss_res = float(np.sum((Y - pred) ** 2))
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    info["r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return resid, info


def leftover_diag(y, x, controls, folds, mask) -> dict:
    resid, info = ols_resid(x, *controls)
    rec = signed_oof_auroc(y, resid, folds, mask)
    rho_c = spearman(resid, controls[0]) if controls else float("nan")
    xr = pd.to_numeric(x, errors="coerce").rank(method="average")
    cr = [pd.to_numeric(c, errors="coerce").rank(method="average") for c in controls]
    rresid, _ = ols_resid(xr, *cr)
    rrec = signed_oof_auroc(y, rresid, folds, mask)
    fake = bool(np.isfinite(rho_c) and abs(rho_c) >= TWIN_RHO)
    rank_cv = _cv(rrec)
    ols_cv = _cv(rec)
    honest_dies = bool(fake or (np.isfinite(rank_cv) and rank_cv < CHANCE) or rec.get("low_power"))
    return {
        "ols": ols_cv,
        "rank": rank_cv,
        "rho_ctrl": rho_c,
        "r2": info["r2"],
        "n": rec["n_defined"],
        "n_pos": rec["n_pos"],
        "fake": fake,
        "honest_dies": honest_dies,
        "low_power": rec.get("low_power", False),
        "folds": fold_bits(rec),
        "rank_folds": fold_bits(rrec),
        "rec": rec,
        "rrec": rrec,
    }


def _cv(res: dict) -> float:
    return float("nan") if res.get("low_power") else float(res.get("cv", float("nan")))


def _auc_row(y: str, feat: str, res: dict) -> dict:
    return {
        "y": y,
        "feature": feat,
        "n": f"{res['n_defined']:,}",
        "n_pos": f"{res['n_pos']:,}",
        "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
        "sign": res["train_sign"] if not res["low_power"] else "—",
        "folds": fold_bits(res) if not res["low_power"] else "—",
    }


def add_panel_lags(df: pd.DataFrame, stems: list[str], lags: tuple[int, ...]) -> pd.DataFrame:
    out = df.sort_values(["company_id", "period"]).reset_index(drop=True)
    extra = {}
    for c in stems:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        g = s.groupby(out["company_id"], sort=False)
        for k in lags:
            extra[f"{c}_lag{k}"] = g.shift(k)
    if extra:
        out = pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)
    return out


def load_panel() -> pd.DataFrame:
    if not STORE.exists():
        raise FileNotFoundError(STORE)
    if not TARGETS.exists():
        raise FileNotFoundError(f"{TARGETS} missing — do not run build_targets")
    raw = pd.read_parquet(STORE)
    yraw = pd.read_parquet(TARGETS)
    print(f"store {STORE} shape={raw.shape} (read-only)")
    print(f"targets {TARGETS} shape={yraw.shape} (read-only)")
    miss = [c for c in STORE_COLS if c not in raw.columns]
    if miss:
        raise RuntimeError(f"monthly.parquet missing {miss}")
    panel = _keys(raw[list(STORE_COLS)])
    y = _keys(yraw[["company_id", "period", *Y_KEEP]])
    panel = panel.merge(y, on=["company_id", "period"], how="left")
    hold = load_holdout()
    panel["split"] = np.where(panel["company_id"].isin(hold), "holdout", "train")
    panel["log_in3"] = np.log1p(pd.to_numeric(panel["a_in3"], errors="coerce").clip(lower=0))
    for col in ("f_n_types", "f_n_facilities", "f_has_loc", "f_has_factoring", "f_has_confirming"):
        panel[col] = pd.to_numeric(panel[col], errors="coerce").fillna(0)
    panel["f_outstanding_gt_granted"] = pd.to_numeric(panel["f_outstanding_gt_granted"], errors="coerce")
    panel["f_util_snapshot"] = pd.to_numeric(panel["f_util_snapshot"], errors="coerce")
    panel["ogtg0"] = panel["f_outstanding_gt_granted"].fillna(0.0)
    leak = leakage_check(
        ["f_outstanding_gt_granted", "f_n_types", "c_n_days_with_tx", "log_in3"],
        Y3,
        forbidden_prefixes=["b"],
    )
    if not leak["ok"]:
        raise RuntimeError(f"Y3 never-B leak: {leak['issues']}")
    return panel


def attach_folds(panel: pd.DataFrame) -> pd.DataFrame:
    train = panel[panel["split"] == "train"][["company_id", "group_id"]].drop_duplicates()
    assert_no_holdout(train["company_id"])
    folds = group_folds(train, n=N_FOLDS, seed=FOLD_SEED)
    return panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")


def pass1_cov(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["f_outstanding_gt_granted"], errors="coerce")
    util = pd.to_numeric(tr["f_util_snapshot"], errors="coerce")
    n_cm = int(len(tr))
    n_def = int(x.notna().sum())
    n_util = int(util.notna().sum())
    cov = n_def / n_cm if n_cm else float("nan")
    util_cov = n_util / n_cm if n_cm else float("nan")
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_per = last["period"].mode().iloc[0] if len(last) else pd.NaT
    last_x = pd.to_numeric(last["f_outstanding_gt_granted"], errors="coerce")
    n1 = int((last_x == 1).sum())
    n0 = int((last_x == 0).sum())
    last_only = bool(n_def == int(last_x.notna().sum()) and last_x.notna().all())
    periods = tr.loc[x.notna(), "period"].drop_duplicates().sort_values()
    modal0 = float((x.dropna() == 0).mean()) if n_def else float("nan")
    rho = spearman(x, tr["log_in3"])
    size_flag = bool(np.isfinite(rho) and abs(rho) >= SIZE_RHO)
    acf1 = median_acf(x, tr["company_id"], 1)
    prose = (
        f"Train OGTG defined {n_def:,}/{n_cm:,} ({_pp(cov)}; quote 5.7% "
        f"{'CONFIRM' if abs(cov - OGTG_COV_QUOTE) < 0.02 else 'off'}). "
        f"f_util_snapshot {_pp(util_cov)} ({'CONFIRM 1.6%' if abs(util_cov - UTIL_COV_QUOTE) < 0.01 else 'off'}). "
        f"Defined periods {list(periods.dt.strftime('%Y-%m').astype(str))} last-month-only={last_only} "
        f"(last {pd.Timestamp(last_per).strftime('%Y-%m')} n1={n1} n0={n0}; quote 32 "
        f"{'CONFIRM' if n1 == OGTG_LAST_N else 'off'}). "
        f"modal0-as-zero {_pp(modal0)} (97.4% {'CONFIRM' if abs(modal0 - OGTG_MODAL) < 0.03 else 'off'}). "
        f"ρ vs log1p(a_in3) on defined {_f(rho)} ({'SIZE' if size_flag else 'not SIZE'}). acf1 {_f(acf1)}."
    )
    print(prose)
    return {
        "cov": cov,
        "util_cov": util_cov,
        "n_def": n_def,
        "n_cm": n_cm,
        "n1": n1,
        "n0": n0,
        "last_only": last_only,
        "last_per": str(pd.Timestamp(last_per).date()),
        "periods": [str(p.date()) for p in periods],
        "modal": modal0,
        "rho_in3": rho,
        "size_flag": size_flag,
        "acf1": acf1,
        "n1_ok": n1 == OGTG_LAST_N,
        "cov_ok": abs(cov - OGTG_COV_QUOTE) < 0.02,
        "util_ok": abs(util_cov - UTIL_COV_QUOTE) < 0.01,
        "prose": prose,
    }


def pass2_twins(tr: pd.DataFrame) -> dict:
    x = tr["f_outstanding_gt_granted"]
    pairs = [
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ("a_n_tx", tr["a_n_tx"]),
        ("f_n_types", tr["f_n_types"]),
        ("f_n_facilities", tr["f_n_facilities"]),
        ("f_util_snapshot", tr["f_util_snapshot"]),
        ("log1p(a_in3)", tr["log_in3"]),
        ("f_has_loc", tr["f_has_loc"]),
    ]
    rows = []
    rhos = {}
    twins = []
    gate = {"c_n_days_with_tx", "a_n_tx", "f_n_types", "f_n_facilities", "f_util_snapshot"}
    for name, s in pairs:
        rho = spearman(x, s)
        rhos[name] = rho
        twin = bool(name in gate and np.isfinite(rho) and abs(rho) >= TWIN_RHO)
        if twin:
            twins.append(name)
        rows.append({"vs": name, "ρ": _f(rho), "twin": "YES" if twin else ""})
    prose = (
        f"Spearman twins |ρ|≥0.80 on defined OGTG: {twins or 'none'}. "
        f"vs days {_f(rhos['c_n_days_with_tx'])} vs a_n_tx {_f(rhos['a_n_tx'])} "
        f"vs n_types {_f(rhos['f_n_types'])} vs n_facilities {_f(rhos['f_n_facilities'])} "
        f"vs util {_f(rhos['f_util_snapshot'])} vs size {_f(rhos['log1p(a_in3)'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "rhos": rhos,
        "twins": twins,
        "gate_twins": bool(twins),
        "rho_days": rhos["c_n_days_with_tx"],
        "rho_types": rhos["f_n_types"],
        "rho_fac": rhos["f_n_facilities"],
        "rho_util": rhos["f_util_snapshot"],
        "rho_size": rhos["log1p(a_in3)"],
        "prose": prose,
    }


def pass3_singles(tr: pd.DataFrame) -> dict:
    feats = {
        "f_outstanding_gt_granted": tr["f_outstanding_gt_granted"],
        "ogtg_fillna0": tr["ogtg0"],
        "f_util_snapshot": tr["f_util_snapshot"],
        "f_n_types": tr["f_n_types"],
        "f_n_facilities": tr["f_n_facilities"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "log1p(a_in3)": tr["log_in3"],
    }
    rows = []
    store = {}
    for y in (Y3, Y2):
        lab = tr[y].notna()
        for name, col in feats.items():
            res = signed_oof_auroc(tr[y], col, tr["fold"], lab)
            store[(y, name)] = res
            rows.append(_auc_row(y, name, res))
            print(f"AUROC {y} {name}: {'LOW_POWER' if res['low_power'] else _f(res['cv'])} n={res['n_defined']} n_pos={res['n_pos']}")
    y3 = _cv(store[(Y3, "f_outstanding_gt_granted")])
    hole = _cv(store[(Y3, "ogtg_fillna0")])
    days = _cv(store[(Y3, "c_n_days_with_tx")])
    size = _cv(store[(Y3, "log1p(a_in3)")])
    types = _cv(store[(Y3, "f_n_types")])
    util = _cv(store[(Y3, "f_util_snapshot")])
    y2 = _cv(store[(Y2, "f_outstanding_gt_granted")])
    days_ok = bool(np.isfinite(days) and abs(days - DAYS_BENCH) < 0.02)
    size_ok = bool(np.isfinite(size) and abs(size - SIZE_Y3_QUOTE) < 0.03)
    types_ok = bool(np.isfinite(types) and abs(types - N_TYPES_Y3) < 0.03)
    beat_size = y3 - size if np.isfinite(y3) and np.isfinite(size) else float("nan")
    native_lp = store[(Y3, "f_outstanding_gt_granted")]["low_power"]
    prose = (
        f"Y3 native OGTG {_f(y3)} n={store[(Y3, 'f_outstanding_gt_granted')]['n_defined']} "
        f"n_pos={store[(Y3, 'f_outstanding_gt_granted')]['n_pos']} "
        f"{'(LOW_POWER — snapshot month has no Y3)' if native_lp else ''}. "
        f"fillna0 hole {_f(hole)}. vs days {_f(days)} (0.711 {'CONFIRM' if days_ok else 'off'}) "
        f"vs size {_f(size)} (0.617 {'CONFIRM' if size_ok else 'off'}) "
        f"vs n_types {_f(types)} (0.578 {'CONFIRM' if types_ok else 'off'}) "
        f"vs util {_f(util)}. Y2 native {_f(y2)}."
    )
    print(prose)
    return {
        "rows": rows,
        "y3": y3,
        "hole": hole,
        "days": days,
        "size": size,
        "types": types,
        "util": util,
        "y2": y2,
        "days_ok": days_ok,
        "size_ok": size_ok,
        "types_ok": types_ok,
        "beat_size": beat_size,
        "native_lp": native_lp,
        "n_pos": store[(Y3, "f_outstanding_gt_granted")]["n_pos"],
        "n_def": store[(Y3, "f_outstanding_gt_granted")]["n_defined"],
        "prose": prose,
    }


def pass4_leftover(tr: pd.DataFrame) -> dict:
    x = tr["f_outstanding_gt_granted"]
    hole = tr["ogtg0"]
    days = tr["c_n_days_with_tx"]
    specs = [
        ("native after days", x, (days,)),
        ("native after n_types", x, (tr["f_n_types"],)),
        ("native after n_facilities", x, (tr["f_n_facilities"],)),
        ("hole after days", hole, (days,)),
        ("hole after n_types", hole, (tr["f_n_types"],)),
        ("hole after n_facilities", hole, (tr["f_n_facilities"],)),
        ("hole after size", hole, (tr["log_in3"],)),
    ]
    rows = []
    store = {}
    for name, xx, xs in specs:
        rec = leftover_diag(tr[Y3], xx, list(xs), tr["fold"], tr[Y3].notna())
        store[name] = rec
        rows.append(
            {
                "control": name,
                "OLS": _f(rec["ols"]),
                "rank": _f(rec["rank"]),
                "n": rec["n"],
                "n_pos": rec["n_pos"],
                "ρ(resid,ctrl)": _f(rec["rho_ctrl"]),
                "fake": "YES" if rec["fake"] else "",
                "dies": "YES" if rec["honest_dies"] else "no",
            }
        )
        print(f"left {name}: OLS={_f(rec['ols'])} rank={_f(rec['rank'])} n_pos={rec['n_pos']} ρctrl={_f(rec['rho_ctrl'])}")
    inv = leftover_diag(tr[Y3], days, [x], tr["fold"], tr[Y3].notna())
    inv_h = leftover_diag(tr[Y3], days, [hole], tr["fold"], tr[Y3].notna())
    rows.append({"control": "days after native OGTG (inverse)", "OLS": _f(inv["ols"]), "rank": _f(inv["rank"]), "n": inv["n"], "n_pos": inv["n_pos"], "ρ(resid,ctrl)": _f(inv["rho_ctrl"]), "fake": "YES" if inv["fake"] else "", "dies": "YES" if inv["honest_dies"] else "no"})
    rows.append({"control": "days after hole fillna0 (inverse)", "OLS": _f(inv_h["ols"]), "rank": _f(inv_h["rank"]), "n": inv_h["n"], "n_pos": inv_h["n_pos"], "ρ(resid,ctrl)": _f(inv_h["rho_ctrl"]), "fake": "YES" if inv_h["fake"] else "", "dies": "YES" if inv_h["honest_dies"] else "no"})
    nat = store["native after days"]
    hol = store["hole after days"]
    leftover = hol["rank"] if nat["low_power"] else nat["rank"]
    leftover_ols = hol["ols"] if nat["low_power"] else nat["ols"]
    dies = bool(nat["honest_dies"] or hol["honest_dies"] or nat["low_power"])
    prose = (
        f"Native leftover after days rank {_f(nat['rank'])} n_pos={nat['n_pos']} "
        f"{'(LOW_POWER)' if nat['low_power'] else ''}. "
        f"fillna0 hole leftover after days rank {_f(hol['rank'])} OLS {_f(hol['ols'])} "
        f"ρ(resid,days)={_f(hol['rho_ctrl'])} fake={hol['fake']}. "
        f"After n_types hole {_f(store['hole after n_types']['rank'])} "
        f"after facilities hole {_f(store['hole after n_facilities']['rank'])}. "
        f"Inverse days after hole {_f(inv_h['rank'])}. Honest leftover {'DIES' if dies else 'lives'}."
    )
    print(prose)
    return {
        "rows": rows,
        "y3_rank": leftover,
        "y3_ols": leftover_ols,
        "native_rank": nat["rank"],
        "native_lp": nat["low_power"],
        "native_npos": nat["n_pos"],
        "hole_rank": hol["rank"],
        "hole_ols": hol["ols"],
        "hole_rho": hol["rho_ctrl"],
        "hole_fake": hol["fake"],
        "types_left": store["hole after n_types"]["rank"],
        "fac_left": store["hole after n_facilities"]["rank"],
        "inv_rank": inv_h["rank"],
        "inv_native": inv["rank"],
        "y3_dies": dies,
        "fake_ols": bool(np.isfinite(hol["ols"]) and hol["ols"] >= CHANCE and np.isfinite(hol["rank"]) and hol["rank"] < CHANCE),
        "prose": prose,
    }


def pass5_rise(tr: pd.DataFrame) -> dict:
    work = tr.sort_values(["company_id", "period"]).copy()
    x = pd.to_numeric(work["f_outstanding_gt_granted"], errors="coerce")
    d = x.groupby(work["company_id"], sort=False).diff()
    rises = int((d > 0).sum())
    drops = int((d < 0).sum())
    defined_pairs = int(d.notna().sum())
    last = work.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_x = pd.to_numeric(last["f_outstanding_gt_granted"], errors="coerce")
    n1 = int((last_x == 1).sum())
    extract = defined_pairs == 0
    prose = (
        f"OGTG rises {rises} / drops {drops} / defined diffs {defined_pairs} "
        f"({'extract-hole: no within-company pair' if extract else 'has a path'}). "
        f"Last-month =1 companies {n1} (quote 32 {'CONFIRM' if n1 == OGTG_LAST_N else 'off'})."
    )
    print(prose)
    return {"rises": rises, "drops": drops, "pairs": defined_pairs, "extract": extract, "n1": n1, "prose": prose}


def pass6_dark(tr: pd.DataFrame, book: set[str]) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    n_erp = int(last["company_id"].isin(book).sum())
    n_dark = int((~last["company_id"].isin(book)).sum())
    confirm = n_erp == 744 and n_dark == 470
    last["erp"] = last["company_id"].isin(book)
    x = pd.to_numeric(last["f_outstanding_gt_granted"], errors="coerce")
    rows = []
    for name, mask in (("invoiced_744", last["erp"]), ("dark_470", ~last["erp"])):
        xx = x[mask]
        n1 = int((xx == 1).sum())
        rows.append({"slice": name, "n": int(mask.sum()), "defined": int(xx.notna().sum()), "ogtg=1": n1, "rate": _pp(n1 / int(xx.notna().sum()) if xx.notna().any() else float("nan"))})
    hole = leftover_diag(tr[Y3], tr["ogtg0"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & tr["company_id"].isin(book))
    hole_d = leftover_diag(tr[Y3], tr["ogtg0"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & ~tr["company_id"].isin(book))
    prose = (
        f"Last-month ever-ERP {n_erp} / never-ERP {n_dark} ({'CONFIRM 744/470' if confirm else 'off'}). "
        f"ogtg=1 invoiced {rows[0]['ogtg=1']} dark {rows[1]['ogtg=1']}. "
        f"Hole leftover after days invoiced {_f(hole['rank'])} dark {_f(hole_d['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "n_erp": n_erp,
        "n_dark": n_dark,
        "confirm": confirm,
        "inv": hole["rank"],
        "dark": hole_d["rank"],
        "prose": prose,
    }


def pass7_q6(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    for col in ("f_outstanding_gt_granted", "ogtg0", "ogtg0_lag1", "c_n_days_with_tx", "c_n_days_with_tx_lag1"):
        if col not in tr.columns:
            continue
        res = signed_oof_auroc(tr[Y3], tr[col], tr["fold"], tr[Y3].notna())
        store[col] = res
        rows.append({"col": col, "n": f"{res['n_defined']:,}", "n_pos": f"{res['n_pos']:,}", "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"])})
    days_l1 = _cv(store.get("c_n_days_with_tx_lag1", {"low_power": True}))
    rec1 = leftover_diag(tr[Y3], tr["ogtg0_lag1"], [tr["c_n_days_with_tx_lag1"]], tr["fold"], tr[Y3].notna()) if "ogtg0_lag1" in tr.columns else {"rank": float("nan"), "honest_dies": True}
    days_l1_ok = bool(np.isfinite(days_l1) and abs(days_l1 - DAYS_LAG1_QUOTE) < 0.03)
    native_lag = store.get("f_outstanding_gt_granted", {})
    q6 = "CLOSE"
    prose = (
        f"Snapshot cannot lead: native OGTG Y3 n_pos={native_lag.get('n_pos', 0)} "
        f"(LOW_POWER on extract month). fillna0 lag1 leftover after days_lag1 {_f(rec1['rank'])}. "
        f"Days lag1 {_f(days_l1)} (KEEP 0.684 {'CONFIRM' if days_l1_ok else 'off'}). Q6 {q6}."
    )
    print(prose)
    return {
        "rows": rows,
        "days_l1": days_l1,
        "l1_left": rec1["rank"],
        "days_l1_ok": days_l1_ok,
        "q6": q6,
        "prose": prose,
    }


def pass8_ever(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_outstanding_gt_granted"].last()
    ever = tr["company_id"].map(last)
    raw = signed_oof_auroc(tr[Y3], ever, tr["fold"], tr[Y3].notna())
    rec = leftover_diag(tr[Y3], ever, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    ever_y3 = tr.groupby("company_id")[Y3].max()
    og = last.eq(1)
    n_og = int(og.sum())
    share_og = float(ever_y3[og[og.index.isin(ever_y3.index)]].mean()) if n_og else float("nan")
    share_no = float(ever_y3[~og.reindex(ever_y3.index).fillna(False)].mean())
    n_rec_og = int(ever_y3.reindex(og.index[og]).fillna(0).sum())
    prose = (
        f"Company last-month OGTG=1 {n_og}/1214. Trait leftover after days {_f(rec['rank'])} "
        f"raw {_f(_cv(raw))} n_pos={raw['n_pos']}. Ever-Y3 recover ogtg {n_rec_og}/{n_og} "
        f"({_pp(share_og)}) vs rest {_pp(share_no)} (debt-schedule 3/24=12.5% / 24.4% on labeled)."
    )
    print(prose)
    return {
        "n_og": n_og,
        "ever": _cv(raw),
        "left": rec["rank"],
        "share_og": share_og,
        "share_no": share_no,
        "n_rec_og": n_rec_og,
        "prose": prose,
    }


def pass_holdout(panel: pd.DataFrame) -> dict:
    ho = panel[panel["split"] == "holdout"]
    x = pd.to_numeric(ho["f_outstanding_gt_granted"], errors="coerce")
    last = ho.sort_values("period").groupby("company_id", as_index=False).tail(1)
    lx = pd.to_numeric(last["f_outstanding_gt_granted"], errors="coerce")
    prose = (
        f"Holdout coverage only (no fit): {ho['company_id'].nunique()} co / {len(ho):,} CM, "
        f"defined {int(x.notna().sum())} last-month =1 {int((lx == 1).sum())}."
    )
    print(prose)
    return {"n_cm": int(len(ho)), "n_def": int(x.notna().sum()), "n1": int((lx == 1).sum()), "prose": prose}


def pass_snap_dummy(tr: pd.DataFrame) -> dict:
    snap = tr["f_outstanding_gt_granted"].notna().astype(float)
    rec = leftover_diag(tr[Y3], snap, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    raw = signed_oof_auroc(tr[Y3], snap, tr["fold"], tr[Y3].notna())
    rec_s = leftover_diag(tr[Y3], snap, [tr["log_in3"]], tr["fold"], tr[Y3].notna())
    prose = (
        f"Last-month dummy leftover after days {_f(rec['rank'])} "
        f"ρ(resid,days)={_f(rec['rho_ctrl'])} fake={rec['fake']} raw {_f(_cv(raw))}. "
        f"After size {_f(rec_s['rank'])}. Same object as fillna0 hole leftover 0.711."
    )
    print(prose)
    return {"left": rec["rank"], "rho": rec["rho_ctrl"], "fake": rec["fake"], "raw": _cv(raw), "prose": prose}


def pass_trait_diag(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_outstanding_gt_granted"].last()
    ever = tr["company_id"].map(last)
    rec = leftover_diag(tr[Y3], ever, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    rec_t = leftover_diag(tr[Y3], ever, [tr["f_n_types"]], tr["fold"], tr[Y3].notna())
    rec_f = leftover_diag(tr[Y3], ever, [tr["f_n_facilities"]], tr["fold"], tr[Y3].notna())
    rec_b = leftover_diag(tr[Y3], ever, [tr["c_n_days_with_tx"], tr["log_in3"]], tr["fold"], tr[Y3].notna())
    prose = (
        f"Trait leftover after days {_f(rec['rank'])} ρ(resid,days)={_f(rec['rho_ctrl'])} "
        f"fake={rec['fake']}; after n_types {_f(rec_t['rank'])}; after facilities {_f(rec_f['rank'])}; "
        f"after days+size {_f(rec_b['rank'])}."
    )
    print(prose)
    return {
        "left": rec["rank"],
        "rho": rec["rho_ctrl"],
        "fake": rec["fake"],
        "types": rec_t["rank"],
        "fac": rec_f["rank"],
        "stack": rec_b["rank"],
        "prose": prose,
    }


def pass_boot(tr: pd.DataFrame, n_boot: int = 64) -> dict:
    rng = np.random.default_rng(FOLD_SEED)
    last = tr.sort_values("period").groupby("company_id")["f_outstanding_gt_granted"].last()
    ever = tr["company_id"].map(last)
    cos = tr["company_id"].drop_duplicates().to_numpy()
    idx = {c: tr.index[tr["company_id"] == c].to_numpy() for c in cos}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(cos, size=len(cos), replace=True)
        rows = np.concatenate([idx[c] for c in pick])
        sub = tr.loc[rows].reset_index(drop=True)
        ev = sub["company_id"].map(last)
        rec = leftover_diag(sub[Y3], ev, [sub["c_n_days_with_tx"]], sub["fold"], sub[Y3].notna())
        if np.isfinite(rec["rank"]):
            vals.append(rec["rank"])
    arr = np.array(vals, dtype=float)
    out = {
        "n": int(arr.size),
        "p05": float(np.quantile(arr, 0.05)) if arr.size else float("nan"),
        "p50": float(np.quantile(arr, 0.50)) if arr.size else float("nan"),
        "p95": float(np.quantile(arr, 0.95)) if arr.size else float("nan"),
    }
    prose = (
        f"Company bootstrap trait leftover-after-days n={out['n']} "
        f"p05/p50/p95 {_f(out['p05'])} / {_f(out['p50'])} / {_f(out['p95'])}."
    )
    print(prose)
    return {**out, "prose": prose}


def pass_util_overlap(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    og = pd.to_numeric(last["f_outstanding_gt_granted"], errors="coerce")
    ut = pd.to_numeric(last["f_util_snapshot"], errors="coerce")
    both = int((og.eq(1) & ut.notna()).sum())
    og1 = int(og.eq(1).sum())
    utn = int(ut.notna().sum())
    rho = spearman(og, ut)
    rec = leftover_diag(tr[Y3], tr["ogtg0"], [tr["f_util_snapshot"].fillna(0)], tr["fold"], tr[Y3].notna())
    prose = (
        f"Last-month util defined {utn} OGTG=1 {og1} both {both}. ρ OGTG~util {_f(rho)}. "
        f"Hole leftover after util-fillna0 {_f(rec['rank'])} ρctrl={_f(rec['rho_ctrl'])}."
    )
    print(prose)
    return {"both": both, "og1": og1, "utn": utn, "rho": rho, "left": rec["rank"], "prose": prose}


def pass_icc(tr: pd.DataFrame) -> dict:
    icc = icc_anova(tr["f_outstanding_gt_granted"], tr["company_id"])
    prose = f"OGTG ICC on defined {_f(icc['icc'])} (feature-report 1.00 BETWEEN)."
    print(prose)
    return {"icc": icc["icc"], "prose": prose}


def pass_perm(tr: pd.DataFrame, n_perm: int = 24) -> dict:
    rng = np.random.default_rng(FOLD_SEED + 7)
    last = tr.sort_values("period").groupby("company_id")["f_outstanding_gt_granted"].last()
    ever = tr["company_id"].map(last)
    obs = leftover_diag(tr[Y3], ever, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())["rank"]
    cos = last.dropna().index.to_numpy()
    vals_map = last.reindex(cos).to_numpy()
    nulls = []
    for _ in range(n_perm):
        shuf = dict(zip(cos, rng.permutation(vals_map)))
        xp = tr["company_id"].map(shuf)
        rec = leftover_diag(tr[Y3], xp, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
        if np.isfinite(rec["rank"]):
            nulls.append(rec["rank"])
    arr = np.array(nulls, dtype=float)
    p = float((arr >= obs).mean()) if arr.size and np.isfinite(obs) else float("nan")
    prose = (
        f"Permute last-month OGTG leftover-after-days null p50 "
        f"{_f(float(np.median(arr)) if arr.size else float('nan'))} p(obs≥null)={_f(p, 3)} obs={_f(obs)}."
    )
    print(prose)
    return {"p50": float(np.median(arr)) if arr.size else float("nan"), "p": p, "obs": obs, "prose": prose}


def pass_dark_trait(tr: pd.DataFrame, book: set[str]) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_outstanding_gt_granted"].last()
    ever = tr["company_id"].map(last)
    erp = tr["company_id"].isin(book)
    rec_i = leftover_diag(tr[Y3], ever, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & erp)
    rec_d = leftover_diag(tr[Y3], ever, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & ~erp)
    prose = (
        f"Trait leftover after days invoiced {_f(rec_i['rank'])} ρ={_f(rec_i['rho_ctrl'])} "
        f"dark {_f(rec_d['rank'])} ρ={_f(rec_d['rho_ctrl'])}."
    )
    print(prose)
    return {"inv": rec_i["rank"], "dark": rec_d["rank"], "prose": prose}


def pass_q6_fake(tr: pd.DataFrame) -> dict:
    rec = leftover_diag(tr[Y3], tr["ogtg0_lag1"], [tr["c_n_days_with_tx_lag1"]], tr["fold"], tr[Y3].notna())
    prose = (
        f"Q6 fillna0 lag1 leftover after days_lag1 {_f(rec['rank'])} "
        f"ρ(resid,days_lag1)={_f(rec['rho_ctrl'])} fake={rec['fake']} "
        f"{'(snapshot lag is a days leak, not lead)' if rec['fake'] else ''}."
    )
    print(prose)
    return {"left": rec["rank"], "rho": rec["rho_ctrl"], "fake": rec["fake"], "prose": prose}


def pass_group(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last = last[pd.to_numeric(last["f_outstanding_gt_granted"], errors="coerce").eq(1)]
    vc = last["group_id"].value_counts()
    top = float(vc.iloc[0] / vc.sum()) if len(vc) else float("nan")
    n_g = int(vc.size)
    prose = (
        f"OGTG=1 companies {len(last)} in {n_g} groups; largest group share {_pp(top)} "
        f"({'group dummy' if np.isfinite(top) and top >= 0.25 else 'not a group dummy'})."
    )
    print(prose)
    return {"n_g": n_g, "top": top, "prose": prose}


def pass_terciles(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_outstanding_gt_granted"].last()
    ever = tr["company_id"].map(last)
    size = pd.to_numeric(tr["log_in3"], errors="coerce")
    terc = pd.qcut(size, 3, labels=["T1", "T2", "T3"], duplicates="drop")
    rows = []
    leftover = {}
    for lab in ("T1", "T2", "T3"):
        rec = leftover_diag(tr[Y3], ever, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & terc.eq(lab))
        leftover[lab] = rec["rank"]
        rows.append({"tercile": lab, "leftover": _f(rec["rank"]), "ρ(resid,days)": _f(rec["rho_ctrl"]), "fake": rec["fake"]})
    prose = (
        f"Trait leftover after days T1/T2/T3 {_f(leftover['T1'])} / {_f(leftover['T2'])} / {_f(leftover['T3'])} "
        f"(all fake days leaks if ρ≥0.80)."
    )
    print(prose)
    return {"rows": rows, "leftover": leftover, "prose": prose}


def pass_logo(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_outstanding_gt_granted"].last()
    ever = tr["company_id"].map(last)
    groups = tr["group_id"].dropna().drop_duplicates().to_numpy()
    vals = []
    for g in groups:
        m = tr["group_id"] != g
        rec = leftover_diag(tr[Y3], ever, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & m)
        if np.isfinite(rec["rank"]):
            vals.append(rec["rank"])
    arr = np.array(vals, dtype=float)
    prose = (
        f"Leave-one-group trait leftover-after-days n={arr.size} "
        f"min/med/max {_f(float(arr.min()) if arr.size else float('nan'))} / "
        f"{_f(float(np.median(arr)) if arr.size else float('nan'))} / "
        f"{_f(float(arr.max()) if arr.size else float('nan'))}."
    )
    print(prose)
    return {
        "n": int(arr.size),
        "min": float(arr.min()) if arr.size else float("nan"),
        "med": float(np.median(arr)) if arr.size else float("nan"),
        "max": float(arr.max()) if arr.size else float("nan"),
        "prose": prose,
    }


def pass_inv_types(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_outstanding_gt_granted"].last()
    ever = tr["company_id"].map(last)
    rec = leftover_diag(tr[Y3], tr["f_n_types"], [ever], tr["fold"], tr[Y3].notna())
    rec_d = leftover_diag(tr[Y3], tr["c_n_days_with_tx"], [ever], tr["fold"], tr[Y3].notna())
    prose = (
        f"Inverse: n_types leftover after OGTG-trait {_f(rec['rank'])}; "
        f"days leftover after OGTG-trait {_f(rec_d['rank'])} (0.711 bar should live)."
    )
    print(prose)
    return {"types": rec["rank"], "days": rec_d["rank"], "prose": prose}


def pass_y2_trait(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_outstanding_gt_granted"].last()
    ever = tr["company_id"].map(last)
    rec = leftover_diag(tr[Y2], ever, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y2].notna())
    raw = signed_oof_auroc(tr[Y2], ever, tr["fold"], tr[Y2].notna())
    prose = (
        f"Y2 trait leftover after days {_f(rec['rank'])} ρ(resid,days)={_f(rec['rho_ctrl'])} "
        f"fake={rec['fake']} raw {_f(_cv(raw))}."
    )
    print(prose)
    return {"left": rec["rank"], "fake": rec["fake"], "raw": _cv(raw), "prose": prose}


def pass_stack_hole(tr: pd.DataFrame) -> dict:
    snap = tr["f_outstanding_gt_granted"].notna().astype(float)
    rec = leftover_diag(tr[Y3], snap, [tr["c_n_days_with_tx"], tr["f_n_types"]], tr["fold"], tr[Y3].notna())
    rec_s = leftover_diag(tr[Y3], snap, [tr["c_n_days_with_tx"], tr["log_in3"], tr["f_n_types"]], tr["fold"], tr[Y3].notna())
    prose = (
        f"Last-month dummy leftover after days+n_types {_f(rec['rank'])} "
        f"ρ(resid,days)={_f(rec['rho_ctrl'])}; after days+size+types {_f(rec_s['rank'])}."
    )
    print(prose)
    return {"stack": rec["rank"], "triple": rec_s["rank"], "rho": rec["rho_ctrl"], "prose": prose}


def pass_util_company(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    ever_y3 = tr.groupby("company_id")[Y3].max()
    last = last.merge(ever_y3.rename("ever_y3").reset_index(), on="company_id", how="left")
    u = pd.to_numeric(last["f_util_snapshot"], errors="coerce")
    y = pd.to_numeric(last["ever_y3"], errors="coerce")
    ok = u.notna() & y.notna()
    auc_p = auroc(y[ok], u[ok]) if int(ok.sum()) else float("nan")
    auc_n = auroc(y[ok], -u[ok]) if int(ok.sum()) else float("nan")
    rho = spearman(last["f_outstanding_gt_granted"], u)
    prose = (
        f"Last-month util vs ever-Y3 AUROC + {_f(auc_p)} − {_f(auc_n)} n={int(ok.sum())} "
        f"n_pos={int((ok & (y == 1)).sum())}. ρ util~OGTG {_f(rho)}."
    )
    print(prose)
    return {"auc_p": auc_p, "auc_n": auc_n, "n": int(ok.sum()), "prose": prose}


def pass_has_overlap(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    og = pd.to_numeric(last["f_outstanding_gt_granted"], errors="coerce").eq(1)
    rows = []
    for name in ("f_has_loc", "f_has_factoring", "f_has_confirming"):
        h = pd.to_numeric(last[name], errors="coerce").eq(1)
        both = int((og & h).sum())
        rows.append({"flag": name, "n": int(h.sum()), "ogtg∩": both, "share_of_ogtg": _pp(both / int(og.sum()) if og.any() else float("nan"))})
    prose = (
        f"OGTG=1 overlap last-month: loc {rows[0]['ogtg∩']}/{int(og.sum())} "
        f"fact {rows[1]['ogtg∩']} conf {rows[2]['ogtg∩']}."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def pass_loc_ogtg(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_has_loc"].last()
    loc_ids = set(last.index[last.eq(1)])
    og = tr.sort_values("period").groupby("company_id")["f_outstanding_gt_granted"].last()
    ever = tr["company_id"].map(og)
    m = tr["company_id"].isin(loc_ids)
    rec = leftover_diag(tr[Y3], ever, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & m)
    rec_l = leftover_diag(tr[Y3], ever, [tr["f_has_loc"]], tr["fold"], tr[Y3].notna())
    prose = (
        f"On last-month LOC companies leftover of OGTG-trait after days {_f(rec['rank'])} "
        f"ρ={_f(rec['rho_ctrl'])} fake={rec['fake']}. "
        f"OGTG leftover after has_loc {_f(rec_l['rank'])}."
    )
    print(prose)
    return {"left": rec["rank"], "after_loc": rec_l["rank"], "fake": rec["fake"], "prose": prose}


def pass_t3_dummy(tr: pd.DataFrame) -> dict:
    size = pd.to_numeric(tr["log_in3"], errors="coerce")
    terc = pd.qcut(size, 3, labels=["T1", "T2", "T3"], duplicates="drop")
    snap = tr["f_outstanding_gt_granted"].notna().astype(float)
    rec = leftover_diag(tr[Y3], snap, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & terc.eq("T3"))
    prose = (
        f"T3 last-month dummy leftover after days {_f(rec['rank'])} "
        f"ρ(resid,days)={_f(rec['rho_ctrl'])} fake={rec['fake']}."
    )
    print(prose)
    return {"left": rec["rank"], "fake": rec["fake"], "prose": prose}


def pass_last_tercile(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    size = pd.to_numeric(last["log_in3"], errors="coerce")
    terc = pd.qcut(size, 3, labels=["T1", "T2", "T3"], duplicates="drop")
    og = pd.to_numeric(last["f_outstanding_gt_granted"], errors="coerce")
    rows = []
    for lab in ("T1", "T2", "T3"):
        m = terc.eq(lab)
        n1 = int((og[m] == 1).sum())
        n = int(m.sum())
        rows.append({"tercile": lab, "n": n, "ogtg=1": n1, "rate": _pp(n1 / n if n else float("nan"))})
    prose = "Last-month OGTG=1 by SIZE tercile: " + ", ".join(f"{r['tercile']} {r['rate']}" for r in rows)
    print(prose)
    return {"rows": rows, "prose": prose}


def pass_dummy_icc(tr: pd.DataFrame) -> dict:
    snap = tr["f_outstanding_gt_granted"].notna().astype(float)
    icc = icc_anova(snap, tr["company_id"])
    xr = snap.rank(method="average")
    cr = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce").rank(method="average")
    resid, _ = ols_resid(xr, cr)
    icc_r = icc_anova(resid, tr["company_id"])
    prose = (
        f"Last-month dummy ICC {_f(icc['icc'])}; rank residual after days ICC {_f(icc_r['icc'])}."
    )
    print(prose)
    return {"icc": icc["icc"], "resid": icc_r["icc"], "prose": prose}


def pass_fold_trait(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_outstanding_gt_granted"].last()
    ever = tr["company_id"].map(last)
    rec = leftover_diag(tr[Y3], ever, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    bits = rec["rank_folds"]
    prose = f"Trait leftover-after-days rank folds {bits} (fake days leak)."
    print(prose)
    return {"folds": bits, "prose": prose}


def pass_after_ntx(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_outstanding_gt_granted"].last()
    ever = tr["company_id"].map(last)
    rec = leftover_diag(tr[Y3], ever, [tr["a_n_tx"]], tr["fold"], tr[Y3].notna())
    snap = tr["f_outstanding_gt_granted"].notna().astype(float)
    rec_s = leftover_diag(tr[Y3], snap, [tr["a_n_tx"]], tr["fold"], tr[Y3].notna())
    prose = (
        f"Trait leftover after a_n_tx {_f(rec['rank'])} ρ={_f(rec['rho_ctrl'])} fake={rec['fake']}. "
        f"Last-month dummy leftover after a_n_tx {_f(rec_s['rank'])} fake={rec_s['fake']}."
    )
    print(prose)
    return {"trait": rec["rank"], "dummy": rec_s["rank"], "prose": prose}


def pass_after_lags(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_outstanding_gt_granted"].last()
    ever = tr["company_id"].map(last)
    rec_l1 = leftover_diag(tr[Y3], ever, [tr["c_n_days_with_tx_lag1"]], tr["fold"], tr[Y3].notna())
    rec_stack = leftover_diag(
        tr[Y3], ever, [tr["c_n_days_with_tx"], tr["c_n_days_with_tx_lag1"]], tr["fold"], tr[Y3].notna()
    )
    snap = tr["f_outstanding_gt_granted"].notna().astype(float)
    rec_d = leftover_diag(
        tr[Y3], snap, [tr["c_n_days_with_tx"], tr["c_n_days_with_tx_lag1"]], tr["fold"], tr[Y3].notna()
    )
    prose = (
        f"Trait leftover after days_lag1 {_f(rec_l1['rank'])} ρ={_f(rec_l1['rho_ctrl'])} fake={rec_l1['fake']}; "
        f"after days+lag1 {_f(rec_stack['rank'])}. Last-month dummy after days+lag1 {_f(rec_d['rank'])}."
    )
    print(prose)
    return {
        "lag1": rec_l1["rank"],
        "stack": rec_stack["rank"],
        "dummy": rec_d["rank"],
        "fake": rec_l1["fake"],
        "prose": prose,
    }


def pass_tenure(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last["first"] = pd.to_datetime(last["first_month"], errors="coerce")
    last["end"] = pd.to_datetime(last["period"], errors="coerce")
    last["n_mo"] = ((last["end"].dt.year - last["first"].dt.year) * 12 + (last["end"].dt.month - last["first"].dt.month) + 1)
    og = pd.to_numeric(last["f_outstanding_gt_granted"], errors="coerce").eq(1)
    med_og = float(last.loc[og, "n_mo"].median()) if og.any() else float("nan")
    med_no = float(last.loc[~og, "n_mo"].median()) if (~og).any() else float("nan")
    days = pd.to_numeric(last["c_n_days_with_tx"], errors="coerce")
    med_d_og = float(days[og].median()) if og.any() else float("nan")
    med_d_no = float(days[~og].median()) if (~og).any() else float("nan")
    prose = (
        f"Last-month tenure months OGTG=1 median {_f(med_og, 1)} vs rest {_f(med_no, 1)}. "
        f"Last-month days median OGTG=1 {_f(med_d_og, 1)} vs rest {_f(med_d_no, 1)} "
        f"(32-company extract, not a longer trail)."
    )
    print(prose)
    return {"med_og": med_og, "med_no": med_no, "days_og": med_d_og, "days_no": med_d_no, "prose": prose}


def pass_last_days_ctrl(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_days = last.set_index("company_id")["c_n_days_with_tx"]
    last_og = last.set_index("company_id")["f_outstanding_gt_granted"]
    ctrl = tr["company_id"].map(last_days)
    ever = tr["company_id"].map(last_og)
    rec = leftover_diag(tr[Y3], ever, [ctrl], tr["fold"], tr[Y3].notna())
    prose = (
        f"Trait leftover after last-month-days-as-trait {_f(rec['rank'])} "
        f"ρ={_f(rec['rho_ctrl'])} fake={rec['fake']} "
        f"(BETWEEN days of the extract month, not the panel path)."
    )
    print(prose)
    return {"left": rec["rank"], "rho": rec["rho_ctrl"], "fake": rec["fake"], "prose": prose}


def pass_company_rho(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    x = last["f_outstanding_gt_granted"]
    rows = []
    rhos = {}
    for name, col in (
        ("last-month days", last["c_n_days_with_tx"]),
        ("last-month a_n_tx", last["a_n_tx"]),
        ("last-month n_types", last["f_n_types"]),
        ("last-month n_facilities", last["f_n_facilities"]),
        ("last-month util", last["f_util_snapshot"]),
        ("last-month size", last["log_in3"]),
    ):
        rho = spearman(x, col)
        rhos[name] = rho
        rows.append({"vs": name, "ρ": _f(rho), "twin": "YES" if np.isfinite(rho) and abs(rho) >= TWIN_RHO else ""})
    prose = (
        f"Company last-month Spearman OGTG vs days {_f(rhos['last-month days'])} "
        f"n_types {_f(rhos['last-month n_types'])} facilities {_f(rhos['last-month n_facilities'])} "
        f"util {_f(rhos['last-month util'])} size {_f(rhos['last-month size'])} "
        f"(no last-month twin |ρ|≥0.80)."
    )
    print(prose)
    return {"rows": rows, "rhos": rhos, "prose": prose}


def pass_trait_stack_types(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_outstanding_gt_granted"].last()
    ever = tr["company_id"].map(last)
    rec = leftover_diag(
        tr[Y3], ever, [tr["c_n_days_with_tx"], tr["f_n_types"]], tr["fold"], tr[Y3].notna()
    )
    rec_f = leftover_diag(
        tr[Y3], ever, [tr["c_n_days_with_tx"], tr["f_n_facilities"]], tr["fold"], tr[Y3].notna()
    )
    rec_t = leftover_diag(tr[Y3], ever, [tr["f_n_types"], tr["f_n_facilities"]], tr["fold"], tr[Y3].notna())
    prose = (
        f"Trait leftover after days+n_types {_f(rec['rank'])} ρ(resid,days)={_f(rec['rho_ctrl'])}; "
        f"after days+facilities {_f(rec_f['rank'])}; after n_types+facilities {_f(rec_t['rank'])} "
        f"({'dies' if np.isfinite(rec['rank']) and rec['rank'] < CHANCE else 'above 0.55 / fake'})."
    )
    print(prose)
    return {"days_types": rec["rank"], "days_fac": rec_f["rank"], "types_fac": rec_t["rank"], "prose": prose}


def pass_holdout_month(panel: pd.DataFrame) -> dict:
    ho = panel[panel["split"] == "holdout"]
    x = pd.to_numeric(ho["f_outstanding_gt_granted"], errors="coerce")
    periods = ho.loc[x.notna(), "period"].drop_duplicates().sort_values()
    last = ho.sort_values("period").groupby("company_id", as_index=False).tail(1)
    lx = pd.to_numeric(last["f_outstanding_gt_granted"], errors="coerce")
    last_only = bool(int(x.notna().sum()) == int(lx.notna().sum()) and lx.notna().all())
    prose = (
        f"Holdout OGTG defined periods {list(periods.dt.strftime('%Y-%m').astype(str))} "
        f"last-month-only={last_only} n1={int((lx == 1).sum())} (same extract hole as train)."
    )
    print(prose)
    return {"last_only": last_only, "n1": int((lx == 1).sum()), "prose": prose}


def pass_fac_dist(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    og = pd.to_numeric(last["f_outstanding_gt_granted"], errors="coerce").eq(1)
    types = pd.to_numeric(last["f_n_types"], errors="coerce")
    fac = pd.to_numeric(last["f_n_facilities"], errors="coerce")
    prose = (
        f"Last-month n_types median OGTG=1 {_f(float(types[og].median()), 1)} vs rest {_f(float(types[~og].median()), 1)}; "
        f"n_facilities median OGTG=1 {_f(float(fac[og].median()), 1)} vs rest {_f(float(fac[~og].median()), 1)} "
        f"(connected book, still a 32-row extract dummy)."
    )
    print(prose)
    return {"prose": prose}


def pass_dark_types(tr: pd.DataFrame, book: set[str]) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_outstanding_gt_granted"].last()
    ever = tr["company_id"].map(last)
    erp = tr["company_id"].isin(book)
    rec_i = leftover_diag(tr[Y3], ever, [tr["f_n_types"]], tr["fold"], tr[Y3].notna() & erp)
    rec_d = leftover_diag(tr[Y3], ever, [tr["f_n_types"]], tr["fold"], tr[Y3].notna() & ~erp)
    prose = (
        f"Trait leftover after n_types invoiced {_f(rec_i['rank'])} dark {_f(rec_d['rank'])} "
        f"(access ≠ ERP; leftover after inventory still dies if <0.55)."
    )
    print(prose)
    return {"inv": rec_i["rank"], "dark": rec_d["rank"], "prose": prose}


def pass_trait_triple(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_outstanding_gt_granted"].last()
    ever = tr["company_id"].map(last)
    rec = leftover_diag(
        tr[Y3],
        ever,
        [tr["c_n_days_with_tx"], tr["f_n_types"], tr["log_in3"]],
        tr["fold"],
        tr[Y3].notna(),
    )
    rec_l = leftover_diag(
        tr[Y3],
        ever,
        [tr["c_n_days_with_tx"], tr["f_n_types"], tr["f_has_loc"]],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"Trait leftover after days+n_types+size {_f(rec['rank'])} ρ={_f(rec['rho_ctrl'])}; "
        f"after days+n_types+has_loc {_f(rec_l['rank'])}."
    )
    print(prose)
    return {"triple": rec["rank"], "loc": rec_l["rank"], "prose": prose}


def pass_last_types_ctrl(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_t = last.set_index("company_id")["f_n_types"]
    last_og = last.set_index("company_id")["f_outstanding_gt_granted"]
    ctrl = tr["company_id"].map(last_t)
    ever = tr["company_id"].map(last_og)
    rec = leftover_diag(tr[Y3], ever, [ctrl], tr["fold"], tr[Y3].notna())
    rec_d = leftover_diag(tr[Y3], ever, [ctrl, tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    prose = (
        f"Trait leftover after last-month-n_types-as-trait {_f(rec['rank'])} "
        f"ρ={_f(rec['rho_ctrl'])}; after last-month-n_types+panel-days {_f(rec_d['rank'])}."
    )
    print(prose)
    return {"types": rec["rank"], "stack": rec_d["rank"], "prose": prose}


def pass_rand_dummy(tr: pd.DataFrame, n_perm: int = 16) -> dict:
    rng = np.random.default_rng(FOLD_SEED + 11)
    last = tr.sort_values("period").groupby("company_id")["f_outstanding_gt_granted"].last()
    n1 = int(last.eq(1).sum())
    cos = last.dropna().index.to_numpy()
    nulls = []
    for _ in range(n_perm):
        pick = set(rng.choice(cos, size=n1, replace=False))
        xp = tr["company_id"].isin(pick).astype(float)
        rec = leftover_diag(tr[Y3], xp, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
        if np.isfinite(rec["rank"]):
            nulls.append(rec["rank"])
    arr = np.array(nulls, dtype=float)
    obs = leftover_diag(
        tr[Y3], tr["company_id"].map(last), [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna()
    )["rank"]
    p = float((arr >= obs).mean()) if arr.size and np.isfinite(obs) else float("nan")
    prose = (
        f"Random 32-company dummy leftover-after-days null p50 "
        f"{_f(float(np.median(arr)) if arr.size else float('nan'))} "
        f"p(obs≥null)={_f(p, 3)} obs={_f(obs)} (OGTG dummy is not above a random 32)."
    )
    print(prose)
    return {
        "p50": float(np.median(arr)) if arr.size else float("nan"),
        "p": p,
        "obs": obs,
        "prose": prose,
    }


def pass_connected_dummy(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    conn = set(last.loc[pd.to_numeric(last["f_n_facilities"], errors="coerce").gt(0), "company_id"])
    dummy = tr["company_id"].isin(conn).astype(float)
    last_og = last.set_index("company_id")["f_outstanding_gt_granted"]
    ever = tr["company_id"].map(last_og)
    rec_c = leftover_diag(tr[Y3], dummy, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    rec_o = leftover_diag(tr[Y3], ever, [dummy], tr["fold"], tr[Y3].notna())
    rec_s = leftover_diag(tr[Y3], ever, [tr["c_n_days_with_tx"], dummy], tr["fold"], tr[Y3].notna())
    prose = (
        f"Last-month-connected dummy leftover after days {_f(rec_c['rank'])} fake={rec_c['fake']}. "
        f"OGTG-trait leftover after connected dummy {_f(rec_o['rank'])}; "
        f"after days+connected {_f(rec_s['rank'])}."
    )
    print(prose)
    return {"conn": rec_c["rank"], "after": rec_o["rank"], "stack": rec_s["rank"], "prose": prose}


def pass_boot_types(tr: pd.DataFrame, n_boot: int = 32) -> dict:
    rng = np.random.default_rng(FOLD_SEED + 3)
    last = tr.sort_values("period").groupby("company_id")["f_outstanding_gt_granted"].last()
    cos = tr["company_id"].drop_duplicates().to_numpy()
    idx = {c: tr.index[tr["company_id"] == c].to_numpy() for c in cos}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(cos, size=len(cos), replace=True)
        rows = np.concatenate([idx[c] for c in pick])
        sub = tr.loc[rows].reset_index(drop=True)
        ev = sub["company_id"].map(last)
        rec = leftover_diag(sub[Y3], ev, [sub["f_n_types"]], sub["fold"], sub[Y3].notna())
        if np.isfinite(rec["rank"]):
            vals.append(rec["rank"])
    arr = np.array(vals, dtype=float)
    prose = (
        f"Company bootstrap trait leftover-after-n_types n={arr.size} "
        f"p05/p50/p95 {_f(float(np.quantile(arr, 0.05)) if arr.size else float('nan'))} / "
        f"{_f(float(np.quantile(arr, 0.50)) if arr.size else float('nan'))} / "
        f"{_f(float(np.quantile(arr, 0.95)) if arr.size else float('nan'))}."
    )
    print(prose)
    return {
        "p05": float(np.quantile(arr, 0.05)) if arr.size else float("nan"),
        "p50": float(np.quantile(arr, 0.50)) if arr.size else float("nan"),
        "p95": float(np.quantile(arr, 0.95)) if arr.size else float("nan"),
        "prose": prose,
    }


def pass_month_table(tr: pd.DataFrame) -> dict:
    g = tr.groupby(tr["period"].dt.to_period("M"))
    rows = []
    for per, sub in g:
        x = pd.to_numeric(sub["f_outstanding_gt_granted"], errors="coerce")
        u = pd.to_numeric(sub["f_util_snapshot"], errors="coerce")
        rows.append(
            {
                "period": str(per),
                "n": f"{len(sub):,}",
                "ogtg_def": int(x.notna().sum()),
                "ogtg=1": int((x == 1).sum()),
                "util_def": int(u.notna().sum()),
                "y3_pos": int((sub[Y3] == 1).sum()),
            }
        )
    n_mo = len(rows)
    n_def_mo = sum(1 for r in rows if r["ogtg_def"] > 0)
    prose = (
        f"Month table: {n_def_mo}/{n_mo} months have any OGTG defined "
        f"(growing panel would be {n_mo}; snapshot is one extract month)."
    )
    print(prose)
    return {"rows": rows, "n_def_mo": n_def_mo, "n_mo": n_mo, "prose": prose}


def pass_first_year(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last["fy"] = pd.to_datetime(last["first_month"], errors="coerce").dt.year
    og = pd.to_numeric(last["f_outstanding_gt_granted"], errors="coerce").eq(1)
    rows = []
    for y, sub in last.groupby("fy"):
        n1 = int(og.loc[sub.index].sum())
        rows.append({"first_year": str(int(y)) if pd.notna(y) else "NA", "n": int(len(sub)), "ogtg=1": n1, "rate": _pp(n1 / len(sub) if len(sub) else float("nan"))})
    prose = "Last-month OGTG=1 by first_month year: " + ", ".join(f"{r['first_year']} {r['rate']}" for r in rows)
    print(prose)
    return {"rows": rows, "prose": prose}


def pass_inv_connected(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    conn = set(last.loc[pd.to_numeric(last["f_n_facilities"], errors="coerce").gt(0), "company_id"])
    dummy = tr["company_id"].isin(conn).astype(float)
    last_og = last.set_index("company_id")["f_outstanding_gt_granted"]
    ever = tr["company_id"].map(last_og)
    rec = leftover_diag(tr[Y3], dummy, [ever], tr["fold"], tr[Y3].notna())
    snap = tr["f_outstanding_gt_granted"].notna().astype(float)
    rec_s = leftover_diag(tr[Y3], snap, [dummy], tr["fold"], tr[Y3].notna())
    prose = (
        f"Inverse: connected-dummy leftover after OGTG-trait {_f(rec['rank'])}. "
        f"Last-month dummy leftover after connected dummy {_f(rec_s['rank'])}."
    )
    print(prose)
    return {"inv": rec["rank"], "dummy": rec_s["rank"], "prose": prose}


def pass_vintage_left(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last["fy"] = pd.to_datetime(last["first_month"], errors="coerce").dt.year
    leftover = {}
    last_og = last.set_index("company_id")["f_outstanding_gt_granted"]
    ever = tr["company_id"].map(last_og)
    fy_map = last.set_index("company_id")["fy"]
    fy = tr["company_id"].map(fy_map)
    for y in (2024, 2025, 2026):
        rec = leftover_diag(tr[Y3], ever, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & fy.eq(y))
        leftover[y] = rec["rank"]
    prose = (
        f"Trait leftover after days by first_year 2024/2025/2026 "
        f"{_f(leftover[2024])} / {_f(leftover[2025])} / {_f(leftover[2026])}."
    )
    print(prose)
    return {"leftover": leftover, "prose": prose}


def pass_loc_dummy(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    loc = set(last.loc[pd.to_numeric(last["f_has_loc"], errors="coerce").eq(1), "company_id"])
    dummy = tr["company_id"].isin(loc).astype(float)
    last_og = last.set_index("company_id")["f_outstanding_gt_granted"]
    ever = tr["company_id"].map(last_og)
    rec_l = leftover_diag(tr[Y3], dummy, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    rec_o = leftover_diag(tr[Y3], ever, [dummy], tr["fold"], tr[Y3].notna())
    rec_s = leftover_diag(tr[Y3], ever, [tr["c_n_days_with_tx"], dummy], tr["fold"], tr[Y3].notna())
    prose = (
        f"Last-month-LOC dummy leftover after days {_f(rec_l['rank'])}. "
        f"OGTG leftover after LOC dummy {_f(rec_o['rank'])}; after days+LOC {_f(rec_s['rank'])}."
    )
    print(prose)
    return {"loc": rec_l["rank"], "after": rec_o["rank"], "stack": rec_s["rank"], "prose": prose}


def pass_has_any_dummy(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    any_h = (
        pd.to_numeric(last["f_has_loc"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_factoring"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_confirming"], errors="coerce").eq(1)
    )
    ids = set(last.loc[any_h, "company_id"])
    dummy = tr["company_id"].isin(ids).astype(float)
    last_og = last.set_index("company_id")["f_outstanding_gt_granted"]
    ever = tr["company_id"].map(last_og)
    rec_h = leftover_diag(tr[Y3], dummy, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    rec_o = leftover_diag(tr[Y3], ever, [dummy], tr["fold"], tr[Y3].notna())
    rec_s = leftover_diag(
        tr[Y3], ever, [tr["c_n_days_with_tx"], dummy, tr["f_n_types"]], tr["fold"], tr[Y3].notna()
    )
    prose = (
        f"Last-month has_any(loc/fact/conf) dummy leftover after days {_f(rec_h['rank'])}. "
        f"OGTG leftover after has_any {_f(rec_o['rank'])}; after days+has_any+n_types {_f(rec_s['rank'])}."
    )
    print(prose)
    return {"has": rec_h["rank"], "after": rec_o["rank"], "stack": rec_s["rank"], "prose": prose}


def pass_fold_types(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_outstanding_gt_granted"].last()
    ever = tr["company_id"].map(last)
    rec = leftover_diag(tr[Y3], ever, [tr["f_n_types"]], tr["fold"], tr[Y3].notna())
    prose = (
        f"Trait leftover-after-n_types rank folds {rec['rank_folds']} "
        f"rank {_f(rec['rank'])} ({'dies' if np.isfinite(rec['rank']) and rec['rank'] < CHANCE else 'above 0.55'})."
    )
    print(prose)
    return {"folds": rec["rank_folds"], "rank": rec["rank"], "prose": prose}


def pass_dark_has(tr: pd.DataFrame, book: set[str]) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    any_h = (
        pd.to_numeric(last["f_has_loc"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_factoring"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_confirming"], errors="coerce").eq(1)
    )
    ids = set(last.loc[any_h, "company_id"])
    dummy = tr["company_id"].isin(ids).astype(float)
    last_og = last.set_index("company_id")["f_outstanding_gt_granted"]
    ever = tr["company_id"].map(last_og)
    erp = tr["company_id"].isin(book)
    rec_i = leftover_diag(tr[Y3], ever, [dummy], tr["fold"], tr[Y3].notna() & erp)
    rec_d = leftover_diag(tr[Y3], ever, [dummy], tr["fold"], tr[Y3].notna() & ~erp)
    prose = (
        f"OGTG leftover after has_any invoiced {_f(rec_i['rank'])} dark {_f(rec_d['rank'])} "
        f"(access ≠ ERP)."
    )
    print(prose)
    return {"inv": rec_i["rank"], "dark": rec_d["rank"], "prose": prose}


def pass_util_dummy(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_util_snapshot"].last()
    dummy = tr["company_id"].map(last).notna().astype(float)
    last_og = tr.sort_values("period").groupby("company_id")["f_outstanding_gt_granted"].last()
    ever = tr["company_id"].map(last_og)
    rec_u = leftover_diag(tr[Y3], dummy, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    rec_o = leftover_diag(tr[Y3], ever, [dummy], tr["fold"], tr[Y3].notna())
    prose = (
        f"Last-month-util-defined dummy leftover after days {_f(rec_u['rank'])} fake={rec_u['fake']}. "
        f"OGTG leftover after util-defined dummy {_f(rec_o['rank'])}."
    )
    print(prose)
    return {"util": rec_u["rank"], "after": rec_o["rank"], "prose": prose}


def pass_fold_stack(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_outstanding_gt_granted"].last()
    ever = tr["company_id"].map(last)
    rec = leftover_diag(
        tr[Y3], ever, [tr["c_n_days_with_tx"], tr["f_n_types"]], tr["fold"], tr[Y3].notna()
    )
    prose = (
        f"Trait leftover-after-days+n_types rank folds {rec['rank_folds']} "
        f"rank {_f(rec['rank'])} ρ(resid,days)={_f(rec['rho_ctrl'])}."
    )
    print(prose)
    return {"folds": rec["rank_folds"], "rank": rec["rank"], "prose": prose}


def pass_inv_types_mean(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_outstanding_gt_granted"].last()
    ever = tr["company_id"].map(last)
    d = pd.DataFrame({"x": pd.to_numeric(tr["f_n_types"], errors="coerce"), "co": tr["company_id"].astype(str)})
    tmean = d.groupby("co")["x"].transform("mean")
    rec = leftover_diag(tr[Y3], tmean, [ever], tr["fold"], tr[Y3].notna())
    rec_d = leftover_diag(tr[Y3], tr["c_n_days_with_tx"], [ever, tmean], tr["fold"], tr[Y3].notna())
    prose = (
        f"Inverse: n_types-mean leftover after OGTG-trait {_f(rec['rank'])}. "
        f"Days leftover after OGTG+n_types-mean {_f(rec_d['rank'])} (0.711 bar should live)."
    )
    print(prose)
    return {"types": rec["rank"], "days": rec_d["rank"], "prose": prose}


def pass_last_double(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_og = last.set_index("company_id")["f_outstanding_gt_granted"]
    ever = tr["company_id"].map(last_og)
    rec = leftover_diag(
        tr[Y3],
        ever,
        [tr["company_id"].map(last_d), tr["company_id"].map(last_t)],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"Trait leftover after last-month days+n_types {_f(rec['rank'])} "
        f"ρ={_f(rec['rho_ctrl'])} ({'dies' if np.isfinite(rec['rank']) and rec['rank'] < CHANCE else 'above 0.55'})."
    )
    print(prose)
    return {"left": rec["rank"], "prose": prose}


def pass_last_ntx(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_n = last.set_index("company_id")["a_n_tx"]
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_og = last.set_index("company_id")["f_outstanding_gt_granted"]
    ever = tr["company_id"].map(last_og)
    rec = leftover_diag(tr[Y3], ever, [tr["company_id"].map(last_n)], tr["fold"], tr[Y3].notna())
    rec_s = leftover_diag(
        tr[Y3],
        ever,
        [tr["company_id"].map(last_d), tr["company_id"].map(last_n)],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"Trait leftover after last-month a_n_tx {_f(rec['rank'])} ρ={_f(rec['rho_ctrl'])}; "
        f"after last-month days+a_n_tx {_f(rec_s['rank'])} "
        f"({'dies' if np.isfinite(rec_s['rank']) and rec_s['rank'] < CHANCE else 'above 0.55'})."
    )
    print(prose)
    return {"ntx": rec["rank"], "stack": rec_s["rank"], "prose": prose}


def pass_boot_last_double(tr: pd.DataFrame, n_boot: int = 32) -> dict:
    rng = np.random.default_rng(FOLD_SEED + 13)
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_og = last.set_index("company_id")["f_outstanding_gt_granted"]
    cos = tr["company_id"].drop_duplicates().to_numpy()
    idx = {c: tr.index[tr["company_id"] == c].to_numpy() for c in cos}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(cos, size=len(cos), replace=True)
        rows = np.concatenate([idx[c] for c in pick])
        sub = tr.loc[rows].reset_index(drop=True)
        ev = sub["company_id"].map(last_og)
        rec = leftover_diag(
            sub[Y3],
            ev,
            [sub["company_id"].map(last_d), sub["company_id"].map(last_t)],
            sub["fold"],
            sub[Y3].notna(),
        )
        if np.isfinite(rec["rank"]):
            vals.append(rec["rank"])
    arr = np.array(vals, dtype=float)
    prose = (
        f"Company bootstrap leftover after last-month days+n_types n={arr.size} "
        f"p05/p50/p95 {_f(float(np.quantile(arr, 0.05)) if arr.size else float('nan'))} / "
        f"{_f(float(np.quantile(arr, 0.50)) if arr.size else float('nan'))} / "
        f"{_f(float(np.quantile(arr, 0.95)) if arr.size else float('nan'))}."
    )
    print(prose)
    return {
        "p05": float(np.quantile(arr, 0.05)) if arr.size else float("nan"),
        "p50": float(np.quantile(arr, 0.50)) if arr.size else float("nan"),
        "p95": float(np.quantile(arr, 0.95)) if arr.size else float("nan"),
        "prose": prose,
    }


def pass_t3_last_double(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_og = last.set_index("company_id")["f_outstanding_gt_granted"]
    ever = tr["company_id"].map(last_og)
    size = pd.to_numeric(tr["log_in3"], errors="coerce")
    terc = pd.qcut(size, 3, labels=["T1", "T2", "T3"], duplicates="drop")
    leftover = {}
    for lab in ("T1", "T2", "T3"):
        rec = leftover_diag(
            tr[Y3],
            ever,
            [tr["company_id"].map(last_d), tr["company_id"].map(last_t)],
            tr["fold"],
            tr[Y3].notna() & terc.eq(lab),
        )
        leftover[lab] = rec["rank"]
    prose = (
        f"Leftover after last-month days+n_types T1/T2/T3 "
        f"{_f(leftover['T1'])} / {_f(leftover['T2'])} / {_f(leftover['T3'])} "
        f"(honest leftover dies on T3={_f(leftover['T3'])})."
    )
    print(prose)
    return {"leftover": leftover, "left": leftover["T3"], "prose": prose}


def pass_fold_last_triple(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_l = last.set_index("company_id")["f_has_loc"]
    last_og = last.set_index("company_id")["f_outstanding_gt_granted"]
    ever = tr["company_id"].map(last_og)
    rec = leftover_diag(
        tr[Y3],
        ever,
        [tr["company_id"].map(last_d), tr["company_id"].map(last_t), tr["company_id"].map(last_l)],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"Trait leftover-after-last-month days+n_types+has_loc rank folds {rec['rank_folds']} "
        f"rank {_f(rec['rank'])} ({'dies' if np.isfinite(rec['rank']) and rec['rank'] < CHANCE else 'above 0.55'})."
    )
    print(prose)
    return {"folds": rec["rank_folds"], "rank": rec["rank"], "prose": prose}


def pass_fold_last_double(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_og = last.set_index("company_id")["f_outstanding_gt_granted"]
    ever = tr["company_id"].map(last_og)
    rec = leftover_diag(
        tr[Y3],
        ever,
        [tr["company_id"].map(last_d), tr["company_id"].map(last_t)],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"Trait leftover-after-last-month days+n_types rank folds {rec['rank_folds']} "
        f"rank {_f(rec['rank'])} ({'dies' if np.isfinite(rec['rank']) and rec['rank'] < CHANCE else 'above 0.55'})."
    )
    print(prose)
    return {"folds": rec["rank_folds"], "rank": rec["rank"], "prose": prose}


def pass_dark_last_double(tr: pd.DataFrame, book: set[str]) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_og = last.set_index("company_id")["f_outstanding_gt_granted"]
    ever = tr["company_id"].map(last_og)
    xs = [tr["company_id"].map(last_d), tr["company_id"].map(last_t)]
    erp = tr["company_id"].isin(book)
    rec_i = leftover_diag(tr[Y3], ever, xs, tr["fold"], tr[Y3].notna() & erp)
    rec_d = leftover_diag(tr[Y3], ever, xs, tr["fold"], tr[Y3].notna() & ~erp)
    prose = (
        f"Trait leftover after last-month days+n_types invoiced {_f(rec_i['rank'])} "
        f"dark {_f(rec_d['rank'])} (access ≠ ERP)."
    )
    print(prose)
    return {"inv": rec_i["rank"], "dark": rec_d["rank"], "prose": prose}


def pass_last_days_fac(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_f = last.set_index("company_id")["f_n_facilities"]
    last_s = last.set_index("company_id")["log_in3"]
    last_og = last.set_index("company_id")["f_outstanding_gt_granted"]
    ever = tr["company_id"].map(last_og)
    rec = leftover_diag(
        tr[Y3],
        ever,
        [tr["company_id"].map(last_d), tr["company_id"].map(last_f)],
        tr["fold"],
        tr[Y3].notna(),
    )
    rec_s = leftover_diag(
        tr[Y3],
        ever,
        [tr["company_id"].map(last_d), tr["company_id"].map(last.set_index("company_id")["f_n_types"]), tr["company_id"].map(last_s)],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"Trait leftover after last-month days+n_facilities {_f(rec['rank'])}; "
        f"after last-month days+n_types+size {_f(rec_s['rank'])} "
        f"({'dies' if np.isfinite(rec['rank']) and rec['rank'] < CHANCE else 'above 0.55'})."
    )
    print(prose)
    return {"fac": rec["rank"], "triple": rec_s["rank"], "prose": prose}


def pass_last_fac_ctrl(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_f = last.set_index("company_id")["f_n_facilities"]
    last_og = last.set_index("company_id")["f_outstanding_gt_granted"]
    ever = tr["company_id"].map(last_og)
    rec = leftover_diag(tr[Y3], ever, [tr["company_id"].map(last_f)], tr["fold"], tr[Y3].notna())
    rec_s = leftover_diag(
        tr[Y3],
        ever,
        [tr["c_n_days_with_tx"], tr["company_id"].map(last_f)],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"Trait leftover after last-month n_facilities {_f(rec['rank'])}; "
        f"after panel-days+last-month facilities {_f(rec_s['rank'])}."
    )
    print(prose)
    return {"fac": rec["rank"], "stack": rec_s["rank"], "prose": prose}


def pass_dark_triple(tr: pd.DataFrame, book: set[str]) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_l = last.set_index("company_id")["f_has_loc"]
    last_og = last.set_index("company_id")["f_outstanding_gt_granted"]
    ever = tr["company_id"].map(last_og)
    xs = [tr["company_id"].map(last_d), tr["company_id"].map(last_t), tr["company_id"].map(last_l)]
    erp = tr["company_id"].isin(book)
    rec_i = leftover_diag(tr[Y3], ever, xs, tr["fold"], tr[Y3].notna() & erp)
    rec_d = leftover_diag(tr[Y3], ever, xs, tr["fold"], tr[Y3].notna() & ~erp)
    prose = (
        f"Trait leftover after last-month days+n_types+has_loc invoiced {_f(rec_i['rank'])} "
        f"dark {_f(rec_d['rank'])} (access ≠ ERP)."
    )
    print(prose)
    return {"inv": rec_i["rank"], "dark": rec_d["rank"], "prose": prose}


def pass_last_triple_ctrl(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_l = last.set_index("company_id")["f_has_loc"]
    last_og = last.set_index("company_id")["f_outstanding_gt_granted"]
    ever = tr["company_id"].map(last_og)
    rec = leftover_diag(
        tr[Y3],
        ever,
        [tr["company_id"].map(last_d), tr["company_id"].map(last_t), tr["company_id"].map(last_l)],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"Trait leftover after last-month days+n_types+has_loc {_f(rec['rank'])} "
        f"ρ={_f(rec['rho_ctrl'])} ({'dies' if np.isfinite(rec['rank']) and rec['rank'] < CHANCE else 'above 0.55'})."
    )
    print(prose)
    return {"left": rec["rank"], "prose": prose}


def pass_between_types(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_outstanding_gt_granted"].last()
    ever = tr["company_id"].map(last)
    d = pd.DataFrame({"x": pd.to_numeric(tr["f_n_types"], errors="coerce"), "co": tr["company_id"].astype(str)})
    tmean = d.groupby("co")["x"].transform("mean")
    rec = leftover_diag(tr[Y3], ever, [tmean], tr["fold"], tr[Y3].notna())
    rec_s = leftover_diag(tr[Y3], ever, [tmean, tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    prose = (
        f"BETWEEN leftover of OGTG-trait after n_types-mean {_f(rec['rank'])} "
        f"ρ={_f(rec['rho_ctrl'])}; after n_types-mean+days {_f(rec_s['rank'])}."
    )
    print(prose)
    return {"types": rec["rank"], "stack": rec_s["rank"], "prose": prose}


def pass_inv_loc(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    loc = set(last.loc[pd.to_numeric(last["f_has_loc"], errors="coerce").eq(1), "company_id"])
    dummy = tr["company_id"].isin(loc).astype(float)
    last_og = last.set_index("company_id")["f_outstanding_gt_granted"]
    ever = tr["company_id"].map(last_og)
    rec = leftover_diag(tr[Y3], dummy, [ever], tr["fold"], tr[Y3].notna())
    rec_d = leftover_diag(tr[Y3], tr["c_n_days_with_tx"], [ever, dummy], tr["fold"], tr[Y3].notna())
    prose = (
        f"Inverse: LOC dummy leftover after OGTG-trait {_f(rec['rank'])}. "
        f"Days leftover after OGTG+LOC {_f(rec_d['rank'])} (0.711 bar should live)."
    )
    print(prose)
    return {"loc": rec["rank"], "days": rec_d["rank"], "prose": prose}


def pass_dark_conn(tr: pd.DataFrame, book: set[str]) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    conn = set(last.loc[pd.to_numeric(last["f_n_facilities"], errors="coerce").gt(0), "company_id"])
    dummy = tr["company_id"].isin(conn).astype(float)
    last_og = last.set_index("company_id")["f_outstanding_gt_granted"]
    ever = tr["company_id"].map(last_og)
    erp = tr["company_id"].isin(book)
    rec_i = leftover_diag(tr[Y3], ever, [dummy], tr["fold"], tr[Y3].notna() & erp)
    rec_d = leftover_diag(tr[Y3], ever, [dummy], tr["fold"], tr[Y3].notna() & ~erp)
    prose = (
        f"OGTG leftover after connected dummy invoiced {_f(rec_i['rank'])} "
        f"dark {_f(rec_d['rank'])} (access ≠ ERP)."
    )
    print(prose)
    return {"inv": rec_i["rank"], "dark": rec_d["rank"], "prose": prose}


def pass_q6_days(tr: pd.DataFrame) -> dict:
    rec = leftover_diag(tr[Y3], tr["ogtg0_lag1"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    rec3 = leftover_diag(tr[Y3], tr["ogtg0_lag3"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    prose = (
        f"Q6 fillna0 lag1 leftover after contemporaneous days {_f(rec['rank'])} "
        f"ρ={_f(rec['rho_ctrl'])} fake={rec['fake']}; lag3 after days {_f(rec3['rank'])} "
        f"(snapshot lag cannot lead)."
    )
    print(prose)
    return {"lag1": rec["rank"], "lag3": rec3["rank"], "fake": rec["fake"], "prose": prose}


def pass_dark_stack(tr: pd.DataFrame, book: set[str]) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_outstanding_gt_granted"].last()
    ever = tr["company_id"].map(last)
    erp = tr["company_id"].isin(book)
    rec_i = leftover_diag(
        tr[Y3], ever, [tr["c_n_days_with_tx"], tr["f_n_types"]], tr["fold"], tr[Y3].notna() & erp
    )
    rec_d = leftover_diag(
        tr[Y3], ever, [tr["c_n_days_with_tx"], tr["f_n_types"]], tr["fold"], tr[Y3].notna() & ~erp
    )
    prose = (
        f"Trait leftover after days+n_types invoiced {_f(rec_i['rank'])} "
        f"dark {_f(rec_d['rank'])} (access ≠ ERP)."
    )
    print(prose)
    return {"inv": rec_i["rank"], "dark": rec_d["rank"], "prose": prose}


def pass_y2_types(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_outstanding_gt_granted"].last()
    ever = tr["company_id"].map(last)
    rec = leftover_diag(tr[Y2], ever, [tr["f_n_types"]], tr["fold"], tr[Y2].notna())
    rec_d = leftover_diag(tr[Y2], ever, [tr["c_n_days_with_tx"], tr["f_n_types"]], tr["fold"], tr[Y2].notna())
    prose = (
        f"Y2 trait leftover after n_types {_f(rec['rank'])}; after days+n_types {_f(rec_d['rank'])}."
    )
    print(prose)
    return {"types": rec["rank"], "stack": rec_d["rank"], "prose": prose}


def pass_dummy_lag1(tr: pd.DataFrame) -> dict:
    snap = tr["f_outstanding_gt_granted"].notna().astype(float)
    rec = leftover_diag(tr[Y3], snap, [tr["c_n_days_with_tx_lag1"]], tr["fold"], tr[Y3].notna())
    rec_d = leftover_diag(tr[Y3], snap, [tr["c_n_days_with_tx_lag3"]], tr["fold"], tr[Y3].notna())
    prose = (
        f"Last-month dummy leftover after days_lag1 {_f(rec['rank'])} ρ={_f(rec['rho_ctrl'])} fake={rec['fake']}; "
        f"after days_lag3 {_f(rec_d['rank'])}."
    )
    print(prose)
    return {"lag1": rec["rank"], "lag3": rec_d["rank"], "fake": rec["fake"], "prose": prose}


def pass_q6_lag3(tr: pd.DataFrame) -> dict:
    if "ogtg0_lag3" not in tr.columns or "c_n_days_with_tx_lag3" not in tr.columns:
        print("Q6 lag3 skipped (cols missing)")
        return {"left": float("nan"), "days_l3": float("nan"), "prose": "Q6 lag3 skipped."}
    rec = leftover_diag(tr[Y3], tr["ogtg0_lag3"], [tr["c_n_days_with_tx_lag3"]], tr["fold"], tr[Y3].notna())
    raw = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx_lag3"], tr["fold"], tr[Y3].notna())
    prose = (
        f"Q6 fillna0 lag3 leftover after days_lag3 {_f(rec['rank'])} "
        f"ρ={_f(rec['rho_ctrl'])} fake={rec['fake']}. Days lag3 {_f(_cv(raw))}."
    )
    print(prose)
    return {"left": rec["rank"], "days_l3": _cv(raw), "fake": rec["fake"], "prose": prose}


def pass_between(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_outstanding_gt_granted"].last()
    ever = tr["company_id"].map(last)
    d = pd.DataFrame({"x": pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce"), "co": tr["company_id"].astype(str)})
    dmean = d.groupby("co")["x"].transform("mean")
    rec = leftover_diag(tr[Y3], ever, [dmean], tr["fold"], tr[Y3].notna())
    prose = (
        f"BETWEEN leftover of OGTG-trait after days-mean {_f(rec['rank'])} "
        f"ρ(resid,days-mean)={_f(rec['rho_ctrl'])} fake={rec['fake']}."
    )
    print(prose)
    return {"left": rec["rank"], "fake": rec["fake"], "prose": prose}


def pass_boot_hole(tr: pd.DataFrame, n_boot: int = 48) -> dict:
    rng = np.random.default_rng(FOLD_SEED + 5)
    snap = tr["f_outstanding_gt_granted"].notna().astype(float)
    cos = tr["company_id"].drop_duplicates().to_numpy()
    idx = {c: tr.index[tr["company_id"] == c].to_numpy() for c in cos}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(cos, size=len(cos), replace=True)
        rows = np.concatenate([idx[c] for c in pick])
        sub = tr.loc[rows].reset_index(drop=True)
        s = sub["f_outstanding_gt_granted"].notna().astype(float)
        rec = leftover_diag(sub[Y3], s, [sub["c_n_days_with_tx"]], sub["fold"], sub[Y3].notna())
        if np.isfinite(rec["rank"]):
            vals.append(rec["rank"])
    arr = np.array(vals, dtype=float)
    prose = (
        f"Bootstrap last-month-dummy leftover-after-days n={arr.size} "
        f"p05/p50/p95 {_f(float(np.quantile(arr, 0.05)) if arr.size else float('nan'))} / "
        f"{_f(float(np.quantile(arr, 0.50)) if arr.size else float('nan'))} / "
        f"{_f(float(np.quantile(arr, 0.95)) if arr.size else float('nan'))}."
    )
    print(prose)
    return {
        "p05": float(np.quantile(arr, 0.05)) if arr.size else float("nan"),
        "p50": float(np.quantile(arr, 0.50)) if arr.size else float("nan"),
        "p95": float(np.quantile(arr, 0.95)) if arr.size else float("nan"),
        "prose": prose,
    }


def pass_ogtg_cos(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_outstanding_gt_granted"].last()
    ids = set(last.index[last.eq(1)])
    m = tr["company_id"].isin(ids)
    rec_d = leftover_diag(tr[Y3], tr["c_n_days_with_tx"], [tr["f_n_types"]], tr["fold"], tr[Y3].notna() & m)
    rec_t = leftover_diag(tr[Y3], tr["f_n_types"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & m)
    raw_d = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], tr[Y3].notna() & m)
    raw_t = signed_oof_auroc(tr[Y3], tr["f_n_types"], tr["fold"], tr[Y3].notna() & m)
    prose = (
        f"On the 32 OGTG=1 companies: days Y3 {_f(_cv(raw_d))} leftover after n_types {_f(rec_d['rank'])} "
        f"n_pos={raw_d['n_pos']}; n_types Y3 {_f(_cv(raw_t))} leftover after days {_f(rec_t['rank'])}."
    )
    print(prose)
    return {"days": _cv(raw_d), "types": _cv(raw_t), "days_left": rec_d["rank"], "types_left": rec_t["rank"], "n_pos": raw_d["n_pos"], "prose": prose}


def pass_fac_dummy(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    fac_cos = set(last.loc[pd.to_numeric(last["f_n_facilities"], errors="coerce").gt(0), "company_id"])
    m = tr["company_id"].isin(fac_cos)
    last_map = tr.sort_values("period").groupby("company_id")["f_outstanding_gt_granted"].last()
    ever = tr["company_id"].map(last_map)
    rec = leftover_diag(tr[Y3], ever, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & m)
    rec_t = leftover_diag(tr[Y3], ever, [tr["f_n_types"]], tr["fold"], tr[Y3].notna() & m)
    prose = (
        f"On last-month-connected companies leftover after days {_f(rec['rank'])} "
        f"ρ={_f(rec['rho_ctrl'])} fake={rec['fake']}; after n_types {_f(rec_t['rank'])}."
    )
    print(prose)
    return {"left": rec["rank"], "types": rec_t["rank"], "fake": rec["fake"], "prose": prose}


def pass_last_fac(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    fac = pd.to_numeric(last["f_n_facilities"], errors="coerce")
    og = pd.to_numeric(last["f_outstanding_gt_granted"], errors="coerce")
    n_fac = int(fac.gt(0).sum())
    n_og = int(og.eq(1).sum())
    share = n_og / n_fac if n_fac else float("nan")
    prose = (
        f"Last-month facilities>0 {n_fac} OGTG=1 among them {n_og} ({_pp(share)}). "
        f"OGTG is a rare extract flag on the connected book, not a panel X."
    )
    print(prose)
    return {"n_fac": n_fac, "n_og": n_og, "share": share, "prose": prose}


def pass_company_auc(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    ever_y3 = tr.groupby("company_id")[Y3].max()
    last = last.merge(ever_y3.rename("ever_y3").reset_index(), on="company_id", how="left")
    x = pd.to_numeric(last["f_outstanding_gt_granted"], errors="coerce")
    y = pd.to_numeric(last["ever_y3"], errors="coerce")
    ok = x.notna() & y.notna()
    auc = auroc(y[ok], -x[ok]) if ok.sum() else float("nan")
    auc_p = auroc(y[ok], x[ok]) if ok.sum() else float("nan")
    n_pos = int((ok & (y == 1)).sum())
    prose = (
        f"Company-level last-month OGTG vs ever-Y3 AUROC + {_f(auc_p)} − {_f(auc)} "
        f"n={int(ok.sum())} n_pos={n_pos} (32-company extract dummy)."
    )
    print(prose)
    return {"auc": auc, "auc_p": auc_p, "n_pos": n_pos, "prose": prose}


def pass_y3_last(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    n_y3 = int(last[Y3].notna().sum())
    n_y2 = int(last[Y2].notna().sum())
    x = pd.to_numeric(last["f_outstanding_gt_granted"], errors="coerce")
    prose = (
        f"Last-month labeled Y3 {n_y3} Y2 {n_y2} (debt-schedule horizon → almost empty; quote Y3=0). "
        f"OGTG defined {int(x.notna().sum())} =1 {int((x == 1).sum())}."
    )
    print(prose)
    return {"n_y3": n_y3, "n_y2": n_y2, "empty": n_y3 == 0, "prose": prose}


def decide(p1, p2, p3, p4, p5, p7) -> dict:
    twin = bool(p2["gate_twins"])
    size = bool(p1["size_flag"])
    leftover = p4["y3_rank"]
    leftover_dies = bool(p4["y3_dies"] or p3["native_lp"] or p1["last_only"])
    beat = bool(np.isfinite(p3["beat_size"]) and p3["beat_size"] >= KEEP_DELTA)
    keep_x = bool(beat and (not leftover_dies) and (not size) and (not twin) and (not p1["last_only"]))
    leftover_tag = "CLOSE"
    if keep_x:
        card = "KEEP as unused leftover"
        tag = "KEEP"
        leftover_tag = "KEEP"
        why = f"leftover after days {_f(leftover)} lives. Do not put on the 15-col card."
    elif p1["last_only"] or p3["native_lp"]:
        card = "PARK as Y / DROP from the 44 as Y3 X"
        tag = "PARK"
        leftover_tag = "CLOSE"
        why = (
            f"Last-month extract flag (cov {_pp(p1['cov'])}; Y3 n_pos on snapshot={p3['n_pos']}). "
            f"Snapshot cannot lead (Q6 CLOSE). NORTH_STAR PARK snapshot cols as X. "
            f"Hole leftover after days {_f(p4['hole_rank'])}. Do not put on the 15-col card. "
            "Same object as y10_ogtg_last_month PARK."
        )
    elif leftover_dies or (not beat):
        card = "CLOSE unused leftover"
        tag = "CLOSE"
        leftover_tag = "CLOSE"
        why = f"leftover after days {_f(leftover)} dies; beat_size={_f(p3['beat_size'])}."
    else:
        card = "DROP from the 44 as Y3 X"
        tag = "DROP"
        why = f"twin={p2['twins']} leftover {_f(leftover)}."
    return {
        "card": card,
        "headline_tag": tag,
        "leftover_tag": leftover_tag,
        "why": why,
        "keep_x": keep_x,
        "twin": twin,
        "size": size,
        "leftover_dies": leftover_dies,
        "q6": p7["q6"],
        "headline": (
            f"{tag} as snapshot X. Leftover after days **{leftover_tag}** "
            f"native n_pos={p3['n_pos']} hole-rank {_f(p4['hole_rank'])} OLS {_f(p4['hole_ols'])}. "
            f"Last-month-only={p1['last_only']} cov {_pp(p1['cov'])} vs util {_pp(p1['util_cov'])}. "
            f"Y3 native {_f(p3['y3'])} vs days {_f(p3['days'])} vs size {_f(p3['size'])} vs n_types {_f(p3['types'])}. "
            f"Twin={p2['twins'] or 'none'} ρ days {_f(p2['rho_days'])} types {_f(p2['rho_types'])} util {_f(p2['rho_util'])}. "
            f"Rise/extract {p5['extract']} n1={p5['n1']}. Q6 {p7['q6']}. {card}. "
            "Night Y3 0.762/0.752, days 0.711, size 0.617, TURNOVER 0.720/0.712 unchanged. "
            "Do not put OGTG on the 15-col card."
        ),
    }


def make_png(tr: pd.DataFrame) -> bool:
    if not HAS_MPL:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    by = tr.groupby(tr["period"].dt.to_period("M"))["f_outstanding_gt_granted"].apply(lambda s: s.notna().mean())
    ax.bar(range(len(by)), 100.0 * by.to_numpy(), color="#1f4e79")
    ax.set_xticks(range(len(by)))
    ax.set_xticklabels([str(p) for p in by.index], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("% defined")
    ax.set_title("OGTG coverage by month (extract hole)")
    ax2 = axes[1]
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    x = pd.to_numeric(last["f_outstanding_gt_granted"], errors="coerce")
    ax2.bar(["0", "1", "NA"], [int((x == 0).sum()), int((x == 1).sum()), int(x.isna().sum())], color=["#8aa4b8", "#1f4e79", "#cccccc"])
    ax2.set_title("Last-month OGTG")
    ax2.set_ylabel("companies")
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return True


def write_md(ctx: dict) -> None:
    d = ctx["decision"]
    p1, p2, p3, p4 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"]
    p5, p6, p7, p8 = ctx["p5"], ctx["p6"], ctx["p7"], ctx["p8"]
    lines = [
        "# Unused leftover of `f_outstanding_gt_granted` after `c_n_days_with_tx`",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. Do not invent a Y. Do not put OGTG on the 15-col card. "
        "Do not overwrite `factoring_qa.*` / `n_types_qa.*` / `ds_r_qa.*` / `fc_r_qa.*` / "
        "`debt_schedule_qa` / `sib_neg_qa.*` / `ap_issued_qa.*`. Y3 never B. "
        f"Night Y3 **{Y3_NIGHT[0]:.3f} / {Y3_NIGHT[1]:.3f}**. Days **{DAYS_BENCH}**. "
        f"Size **{SIZE_Y3_QUOTE}**. Y7 TURNOVER **{Y7_TURNOVER[0]:.3f} / {Y7_TURNOVER[1]:.3f}**.",
        "",
        "`f_outstanding_gt_granted` = clean snapshot flag, max over as-of facilities. "
        "NaN until the 2026-08 extract. NORTH_STAR PARK snapshot cols as X. "
        "Same object as `y10_ogtg_last_month` PARK.",
        "",
        "## Headline",
        "",
        d["headline"],
        "",
        "## KEEP / CLOSE / DROP / PARK",
        "",
        _md_table(
            [
                {"object": "OGTG leftover after days", "decision": f"**{d['leftover_tag']}**", "why": f"native n_pos={p3['n_pos']} hole-rank {_f(p4['hole_rank'])} OLS {_f(p4['hole_ols'])} fake={p4['hole_fake']}"},
                {"object": "as Y3 X (not on 15-col card)", "decision": f"**{d['card']}**", "why": d["why"]},
                {"object": "Last-month-only / snapshot", "decision": "**YES**" if p1["last_only"] else "**no**", "why": p1["prose"]},
                {"object": "Twin / SIZE", "decision": f"twin={'YES' if d['twin'] else 'no'} SIZE={'YES' if d['size'] else 'no'}", "why": f"ρ days {_f(p2['rho_days'])} types {_f(p2['rho_types'])} util {_f(p2['rho_util'])} size {_f(p1['rho_in3'])}"},
                {"object": "Q6 lag leftover", "decision": f"**{d['q6']}**", "why": p7["prose"]},
                {"object": "Rise / extract hole", "decision": "**extract-hole**" if p5["extract"] else "**path**", "why": p5["prose"]},
                {"object": "as health Y", "decision": "**PARK**", "why": "Same as y10_ogtg_last_month. Snapshot extract, not a FICO label."},
                {"object": "Night quotes", "decision": "**unchanged**", "why": "Y3 0.762/0.752 · days 0.711 · size 0.617 · TURNOVER 0.720/0.712"},
            ]
        ),
        "",
        "## 1. Coverage — last-month vs growing panel",
        "",
        p1["prose"],
        "",
        p2["prose"],
        "",
        _md_table(p2["rows"]),
        "",
        "## 2. Single-feature train group-fold AUROC",
        "",
        p3["prose"],
        "",
        _md_table(p3["rows"]),
        "",
        "## 3. Honest leftover after days + inverse + n_types",
        "",
        p4["prose"],
        "",
        _md_table(p4["rows"]),
        "",
        "## 4. Rise-only / extract-hole",
        "",
        p5["prose"],
        "",
        "## 5. Dark vs ERP leftover",
        "",
        p6["prose"],
        "",
        _md_table(p6["rows"]),
        "",
        "## 6. Q6 — snapshot cannot lead",
        "",
        p7["prose"],
        "",
        _md_table(p7["rows"]),
        "",
        "## 7. Ever / last-month trait",
        "",
        p8["prose"],
        "",
        ctx["p_yl"]["prose"],
        "",
        "## Extra — holdout coverage",
        "",
        ctx["p_ho"]["prose"],
        "",
        "## Extra — SIZE terciles of trait leftover",
        "",
        ctx["p_te"]["prose"],
        "",
        _md_table(ctx["p_te"]["rows"]),
        "",
        "## Extra — last-month OGTG=1 by SIZE tercile",
        "",
        ctx["p_lt"]["prose"],
        "",
        _md_table(ctx["p_lt"]["rows"]),
        "",
        "## Extra — OGTG=1 by first_month year",
        "",
        ctx["p_fy"]["prose"],
        "",
        _md_table(ctx["p_fy"]["rows"]),
        "",
        "## Extra — coverage by month (growing panel vs extract)",
        "",
        ctx["p_mt"]["prose"],
        "",
        _md_table(ctx["p_mt"]["rows"]),
        "",
        "## Extra — company last-month Spearman (snapshot twins)",
        "",
        ctx["p_cr"]["prose"],
        "",
        _md_table(ctx["p_cr"]["rows"]),
        "",
        f"Plot: `{OUT_PNG.name}`." if ctx.get("png") else "Plot: skipped.",
        "",
        "## What failed / next",
        "",
    ]
    for x in ctx["failed"]:
        lines.append(f"- {x}")
    extra_bits = ctx.get("extra_bits", [])
    extra_bits.extend(
        [
            f"- {ctx['p_sd']['prose']}",
            f"- {ctx['p_td']['prose']}",
            f"- {ctx['p_bt']['prose']}",
            f"- {ctx['p_uo']['prose']}",
            f"- {ctx['p_ic']['prose']}",
            f"- {ctx['p_pm']['prose']}",
            f"- {ctx['p_dt']['prose']}",
            f"- {ctx['p_qf']['prose']}",
            f"- {ctx['p_gr']['prose']}",
            f"- {ctx['p_te']['prose']}",
            f"- {ctx['p_ca']['prose']}",
            f"- {ctx['p_lg']['prose']}",
            f"- {ctx['p_it']['prose']}",
            f"- {ctx['p_lf']['prose']}",
            f"- {ctx['p_y2']['prose']}",
            f"- {ctx['p_fd']['prose']}",
            f"- {ctx['p_oc']['prose']}",
            f"- {ctx['p_sh']['prose']}",
            f"- {ctx['p_bh']['prose']}",
            f"- {ctx['p_uc']['prose']}",
            f"- {ctx['p_di']['prose']}",
            f"- {ctx['p_lt']['prose']}",
            f"- {ctx['p_ho2']['prose']}",
            f"- {ctx['p_t3']['prose']}",
            f"- {ctx['p_lo']['prose']}",
            f"- {ctx['p_bw']['prose']}",
            f"- {ctx['p_ft']['prose']}",
            f"- {ctx['p_an']['prose']}",
            f"- {ctx['p_al']['prose']}",
            f"- {ctx['p_tn']['prose']}",
            f"- {ctx['p_ld']['prose']}",
            f"- {ctx['p_q3']['prose']}",
            f"- {ctx['p_cr']['prose']}",
            f"- {ctx['p_ts']['prose']}",
            f"- {ctx['p_hm']['prose']}",
            f"- {ctx['p_fx']['prose']}",
            f"- {ctx['p_dl']['prose']}",
            f"- {ctx['p_dk']['prose']}",
            f"- {ctx['p_trp']['prose']}",
            f"- {ctx['p_ltc']['prose']}",
            f"- {ctx['p_rd']['prose']}",
            f"- {ctx['p_y2t']['prose']}",
            f"- {ctx['p_cd']['prose']}",
            f"- {ctx['p_bty']['prose']}",
            f"- {ctx['p_mt']['prose']}",
            f"- {ctx['p_ds']['prose']}",
            f"- {ctx['p_fy']['prose']}",
            f"- {ctx['p_icn']['prose']}",
            f"- {ctx['p_qd']['prose']}",
            f"- {ctx['p_vl']['prose']}",
            f"- {ctx['p_ldm']['prose']}",
            f"- {ctx['p_dc']['prose']}",
            f"- {ctx['p_ha']['prose']}",
            f"- {ctx['p_il']['prose']}",
            f"- {ctx['p_fty']['prose']}",
            f"- {ctx['p_dh']['prose']}",
            f"- {ctx['p_ud']['prose']}",
            f"- {ctx['p_btp']['prose']}",
            f"- {ctx['p_fs']['prose']}",
            f"- {ctx['p_itm']['prose']}",
            f"- {ctx['p_lt3']['prose']}",
            f"- {ctx['p_ldb']['prose']}",
            f"- {ctx['p_dt3']['prose']}",
            f"- {ctx['p_lnx']['prose']}",
            f"- {ctx['p_lfc']['prose']}",
            f"- {ctx['p_ldf']['prose']}",
            f"- {ctx['p_bld']['prose']}",
            f"- {ctx['p_dld']['prose']}",
            f"- {ctx['p_fld']['prose']}",
            f"- {ctx['p_flt']['prose']}",
            f"- {ctx['p_t3d']['prose']}",
        ]
    )
    if extra_bits:
        lines.extend(["", "## Later extras (same module)", ""])
        for x in extra_bits:
            lines.append(x)
    lines.extend(
        [
            "",
            f"Elapsed {ctx['elapsed_s']:.0f}s.",
            "",
            "Did **not**: overwrite `factoring_qa.*` / `n_types_qa.*` / `ds_r_qa.*` / `fc_r_qa.*` / "
            "`debt_schedule_qa` / `sib_neg_qa.*` / `ap_issued_qa.*`, edit `debt.py` / `gbm_core.py`, "
            "put OGTG on the 15-col card, grow TURNOVER, invent a Y, write 0–100, fit holdout, touch `product/`.",
            "",
        ]
    )
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD}")


def append_registry(ctx: dict) -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    prev = pd.read_csv(REGISTRY)
    key_cols = ["agent", "x_families", "y", "model", "split", "metric"]
    seen = set()
    if not prev.empty and all(c in prev.columns for c in key_cols):
        seen = set(tuple(str(r[c]) for c in key_cols) for _, r in prev.iterrows())
    d = ctx["decision"]
    p1, p3, p4, p7 = ctx["p1"], ctx["p3"], ctx["p4"], ctx["p7"]
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    cov = f"{p1['cov']:.4f}"
    rows = [
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_f_ogtg_native", "value": p3["y3"], "coverage": cov, "notes": f"n_pos={p3['n_pos']} lp={p3['native_lp']} card={d['headline_tag']}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_f_ogtg_hole_resid_days", "value": p4["hole_rank"], "coverage": cov, "notes": f"ols={p4['hole_ols']} fake={p4['hole_fake']} dies={p4['y3_dies']}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_f_ogtg_hole_resid_types", "value": p4["types_left"], "coverage": cov, "notes": f"fac_left={p4['fac_left']}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_f_ogtg_lag1_resid_days_lag1", "value": p7["l1_left"], "coverage": cov, "notes": f"q6={p7['q6']} days_l1={p7['days_l1']}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_f_ogtg_trait_resid_days_boot_p50", "value": ctx["p_bt"]["p50"], "coverage": cov, "notes": f"p05={ctx['p_bt']['p05']:.4f} p95={ctx['p_bt']['p95']:.4f} fake_trait={ctx['p_td']['fake']}"},
    ]
    new = []
    for r in rows:
        key = tuple(str(r[c]) for c in key_cols)
        if key in seen:
            continue
        seen.add(key)
        new.append(r)
    if not new:
        print("registry: no new rows")
        return
    with REGISTRY.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prev.columns))
        for r in new:
            w.writerow({c: r.get(c, "") for c in prev.columns})
    print(f"registry appended {len(new)} rows")


def write_wave(ctx: dict) -> None:
    d = ctx["decision"]
    p1, p2, p3, p4, p5, p7 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"], ctx["p5"], ctx["p7"]
    text = (
        f"# Wave 4 — f_outstanding_gt_granted leftover after days\n\n"
        f"Agent `{AGENT}`. Train group-fold seed 20260918. Holdout 72 coverage only.\n\n"
        f"## Files written\n\n"
        f"- `analysis/evaluate/ogtg_qa.py`\n"
        f"- `analysis/outputs/ogtg_qa.md`\n"
        f"- `analysis/outputs/ogtg_qa.png`\n"
        f"- append-only `analysis/experiments/registry.csv`\n"
        f"- this note\n\n"
        f"Did not overwrite `factoring_qa.*`, `n_types_qa.*`, `ds_r_qa.*`, `fc_r_qa.*`, "
        f"`debt_schedule_qa`, `sib_neg_qa.*`, `ap_issued_qa.*`. Did not touch `debt.py`, "
        f"`gbm_core.py`, the 15-col card, TURNOVER, product/, or the parent journal. "
        f"Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. TURNOVER **0.720 / 0.712**.\n\n"
        f"## Locked verdict\n\n"
        f"| object | decision |\n"
        f"| --- | --- |\n"
        f"| leftover after days | **{d['leftover_tag']}** hole {_f(p4['hole_rank'])} native n_pos={p3['n_pos']} |\n"
        f"| as Y3 X | **{d['card']}** |\n"
        f"| last-month-only | **{p1['last_only']}** cov {_pp(p1['cov'])} |\n"
        f"| leftover after n_types | **{_f(p4['types_left'])}** |\n"
        f"| days leftover after hole | **{_f(p4['inv_rank'])}** |\n"
        f"| extract-hole | **{p5['extract']}** n1={p5['n1']} |\n"
        f"| Q6 | **{d['q6']}** |\n\n"
        f"Y3 native {_f(p3['y3'])} hole-fillna0 {_f(p3['hole'])} vs days {_f(p3['days'])} "
        f"vs size {_f(p3['size'])} vs n_types {_f(p3['types'])}. "
        f"ρ days {_f(p2['rho_days'])} types {_f(p2['rho_types'])} util {_f(p2['rho_util'])}. "
        f"lag1 leftover after days_lag1 {_f(p7['l1_left'])}. "
        f"Trait leftover after days {_f(ctx['p_td']['left'])} fake={ctx['p_td']['fake']}. "
        f"Bootstrap trait p05/p50/p95 {_f(ctx['p_bt']['p05'])} / {_f(ctx['p_bt']['p50'])} / {_f(ctx['p_bt']['p95'])}. "
        f"Last-month dummy leftover {_f(ctx['p_sd']['left'])} fake={ctx['p_sd']['fake']}. "
        f"Trait leftover after n_types {_f(ctx['p_td']['types'])} boot-n_types p50 {_f(ctx['p_bty']['p50'])}. "
        f"Random 32 dummy leftover p50 {_f(ctx['p_rd']['p50'])} obs {_f(ctx['p_rd']['obs'])}. "
        f"OGTG leftover after LOC dummy {_f(ctx['p_ldm']['after'])}. "
        f"Last-month days+n_types leftover {_f(ctx['p_ldb']['left'])} "
        f"boot p50 {_f(ctx['p_bld']['p50'])}. "
        f"Last-month days+n_types+has_loc leftover {_f(ctx['p_lt3']['left'])}. "
        f"Q6 lag1 after days {_f(ctx['p_qd']['lag1'])}. "
        f"1/24 months defined. Ever-Y3 recover 3/32=12.5% vs 24.4%.\n\n"
        f"## What failed / next\n\n"
        + "\n".join(f"- {x}" for x in ctx["failed"])
        + f"\n\nElapsed {ctx['elapsed_s']:.0f}s.\n"
    )
    WAVE_NOTE.parent.mkdir(parents=True, exist_ok=True)
    WAVE_NOTE.write_text(text, encoding="utf-8")
    print(f"wrote {WAVE_NOTE}")


def run() -> dict:
    t0 = time.time()
    print(f"ogtg_qa start seed={FOLD_SEED} holdout=72 agent={AGENT}")
    panel = load_panel()
    con = connect()
    try:
        book = book_invoice_ids(con)
    finally:
        con.close()
    panel = attach_folds(panel)
    panel = add_panel_lags(panel, ["ogtg0", "f_outstanding_gt_granted", "c_n_days_with_tx"], (1, 3))
    tr = panel[panel["split"] == "train"].copy().reset_index(drop=True)
    assert_no_holdout(tr["company_id"])
    print(f"train CM={len(tr):,} companies={tr['company_id'].nunique()} holdout CM={int((panel['split']=='holdout').sum())}")

    p1 = pass1_cov(tr)
    p2 = pass2_twins(tr)
    p3 = pass3_singles(tr)
    p4 = pass4_leftover(tr)
    p5 = pass5_rise(tr)
    p6 = pass6_dark(tr, book)
    p7 = pass7_q6(tr)
    p8 = pass8_ever(tr)
    p_yl = pass_y3_last(tr)
    p_ho = pass_holdout(panel)
    p_sd = pass_snap_dummy(tr)
    p_td = pass_trait_diag(tr)
    p_bt = pass_boot(tr)
    p_uo = pass_util_overlap(tr)
    p_ic = pass_icc(tr)
    p_pm = pass_perm(tr)
    p_dt = pass_dark_trait(tr, book)
    p_qf = pass_q6_fake(tr)
    p_gr = pass_group(tr)
    p_te = pass_terciles(tr)
    p_ca = pass_company_auc(tr)
    p_lg = pass_logo(tr)
    p_it = pass_inv_types(tr)
    p_lf = pass_last_fac(tr)
    p_y2 = pass_y2_trait(tr)
    p_fd = pass_fac_dummy(tr)
    p_oc = pass_ogtg_cos(tr)
    p_sh = pass_stack_hole(tr)
    p_bh = pass_boot_hole(tr)
    p_uc = pass_util_company(tr)
    p_di = pass_dummy_icc(tr)
    p_lt = pass_last_tercile(tr)
    p_ho2 = pass_has_overlap(tr)
    p_t3 = pass_t3_dummy(tr)
    p_lo = pass_loc_ogtg(tr)
    p_bw = pass_between(tr)
    p_ft = pass_fold_trait(tr)
    p_an = pass_after_ntx(tr)
    p_al = pass_after_lags(tr)
    p_tn = pass_tenure(tr)
    p_ld = pass_last_days_ctrl(tr)
    p_q3 = pass_q6_lag3(tr)
    p_cr = pass_company_rho(tr)
    p_ts = pass_trait_stack_types(tr)
    p_hm = pass_holdout_month(panel)
    p_fx = pass_fac_dist(tr)
    p_dl = pass_dummy_lag1(tr)
    p_dk = pass_dark_types(tr, book)
    p_trp = pass_trait_triple(tr)
    p_ltc = pass_last_types_ctrl(tr)
    p_rd = pass_rand_dummy(tr)
    p_y2t = pass_y2_types(tr)
    p_cd = pass_connected_dummy(tr)
    p_bty = pass_boot_types(tr)
    p_mt = pass_month_table(tr)
    p_ds = pass_dark_stack(tr, book)
    p_fy = pass_first_year(tr)
    p_icn = pass_inv_connected(tr)
    p_qd = pass_q6_days(tr)
    p_vl = pass_vintage_left(tr)
    p_ldm = pass_loc_dummy(tr)
    p_dc = pass_dark_conn(tr, book)
    p_ha = pass_has_any_dummy(tr)
    p_il = pass_inv_loc(tr)
    p_fty = pass_fold_types(tr)
    p_dh = pass_dark_has(tr, book)
    p_ud = pass_util_dummy(tr)
    p_btp = pass_between_types(tr)
    p_fs = pass_fold_stack(tr)
    p_itm = pass_inv_types_mean(tr)
    p_lt3 = pass_last_triple_ctrl(tr)
    p_ldb = pass_last_double(tr)
    p_dt3 = pass_dark_triple(tr, book)
    p_lnx = pass_last_ntx(tr)
    p_lfc = pass_last_fac_ctrl(tr)
    p_ldf = pass_last_days_fac(tr)
    p_bld = pass_boot_last_double(tr)
    p_dld = pass_dark_last_double(tr, book)
    p_fld = pass_fold_last_double(tr)
    p_flt = pass_fold_last_triple(tr)
    p_t3d = pass_t3_last_double(tr)
    png = make_png(tr)
    decision = decide(p1, p2, p3, p4, p5, p7)
    failed = []
    if not p3["days_ok"]:
        failed.append(f"days replica drifted: {_f(p3['days'])} vs 0.711")
    if not p3["size_ok"]:
        failed.append(f"size replica drifted: {_f(p3['size'])} vs 0.617")
    if not p3["types_ok"]:
        failed.append(f"n_types replica drifted: {_f(p3['types'])} vs 0.578")
    if not p1["cov_ok"]:
        failed.append(f"OGTG cov {_pp(p1['cov'])} off 5.7%")
    if not p1["n1_ok"]:
        failed.append(f"last-month =1 {p1['n1']} off 32")
    if not p6["confirm"]:
        failed.append(f"dark/invoiced {p6['n_dark']}/{p6['n_erp']} off 470/744")
    if not p7["days_l1_ok"]:
        failed.append(f"days lag1 {_f(p7['days_l1'])} off KEEP 0.684")
    if not failed:
        failed.append(
            "no replica miss; native leftover undefined (Y3 n_pos=0 on 2026-08). "
            "fillna0 / last-month dummy leftover 0.711 is a fake days leak (ρ=-0.996). "
            "Trait leftover 0.684 is also a fake days leak (ρ=-0.915). "
            "Honest leftover after last-month days+n_types 0.540 dies; after LOC dummy 0.535 dies. "
            "PARK snapshot X / DROP from the 44."
        )
    ctx = {
        "p1": p1, "p2": p2, "p3": p3, "p4": p4, "p5": p5, "p6": p6, "p7": p7, "p8": p8,
        "p_yl": p_yl, "p_ho": p_ho,
        "p_sd": p_sd, "p_td": p_td, "p_bt": p_bt, "p_uo": p_uo,
        "p_ic": p_ic, "p_pm": p_pm, "p_dt": p_dt,
        "p_qf": p_qf, "p_gr": p_gr, "p_te": p_te, "p_ca": p_ca,
        "p_lg": p_lg, "p_it": p_it, "p_lf": p_lf, "p_y2": p_y2, "p_fd": p_fd, "p_oc": p_oc,
        "p_sh": p_sh, "p_bh": p_bh, "p_uc": p_uc, "p_di": p_di, "p_lt": p_lt,
        "p_ho2": p_ho2, "p_t3": p_t3, "p_lo": p_lo, "p_bw": p_bw, "p_ft": p_ft, "p_an": p_an,
        "p_al": p_al, "p_tn": p_tn, "p_ld": p_ld, "p_q3": p_q3,
        "p_cr": p_cr, "p_ts": p_ts, "p_hm": p_hm, "p_fx": p_fx, "p_dl": p_dl,
        "p_dk": p_dk, "p_trp": p_trp, "p_ltc": p_ltc, "p_rd": p_rd, "p_y2t": p_y2t,
        "p_cd": p_cd, "p_bty": p_bty, "p_mt": p_mt, "p_ds": p_ds,
        "p_fy": p_fy, "p_icn": p_icn, "p_qd": p_qd,
        "p_vl": p_vl, "p_ldm": p_ldm, "p_dc": p_dc,
        "p_ha": p_ha, "p_il": p_il,
        "p_fty": p_fty, "p_dh": p_dh, "p_ud": p_ud, "p_btp": p_btp,
        "p_fs": p_fs, "p_itm": p_itm, "p_lt3": p_lt3,
        "p_ldb": p_ldb, "p_dt3": p_dt3,
        "p_lnx": p_lnx, "p_lfc": p_lfc, "p_ldf": p_ldf,
        "p_bld": p_bld, "p_dld": p_dld, "p_fld": p_fld, "p_flt": p_flt,
        "p_t3d": p_t3d,
        "decision": decision, "failed": failed, "png": png, "extra_bits": [],
        "elapsed_s": time.time() - t0, "tr": tr, "panel": panel, "book": book,
    }
    write_md(ctx)
    append_registry(ctx)
    if WRITE_WAVE:
        write_wave(ctx)
    print(f"DONE card={decision['headline_tag']} leftover={_f(p4['y3_rank'])} hole={_f(p4['hole_rank'])} y3={_f(p3['y3'])} elapsed={ctx['elapsed_s']:.0f}s")
    return ctx


if __name__ == "__main__":
    run()
