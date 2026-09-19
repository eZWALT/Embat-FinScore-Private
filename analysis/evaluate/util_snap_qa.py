"""Unused leftover of ``f_util_snapshot`` after ``c_n_days_with_tx``.

sum|outstanding| / sum|granted| over as-of facilities (debt.py). Last-month
only ~1.6% — 100% of non-nulls are 2026-08. NORTH_STAR PARK snapshot as X.
``f_outstanding_gt_granted`` just PARK leftover 0.540 / native Y3 n_pos=0.
Utilisation is impossible as a Y (Y10 PARK). Do not add y10 to FROZEN_ACCEPTED.

KEEP-as-X: beat size ≥0.02 AND leftover after days AND not SIZE
(|ρ| vs log1p(a_in3) ≥0.50) AND not a twin (|ρ|≥0.80 vs days / a_n_tx /
f_n_types / f_outstanding_gt_granted). Leftover <0.55 dies.
Rank leftover is honest; OLS can fake a days leak (OGTG fillna0 ρ=-0.996).

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.util_snap_qa

Owned: analysis/evaluate/util_snap_qa.py, analysis/outputs/util_snap_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_util_snap.md.
Do not overwrite debt_schedule_qa / ogtg_qa.* / factoring_qa.* / n_types_qa.* /
ar_open_qa.* / ap_open_qa.*.
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
OUT_MD = ANALYSIS / "outputs" / "util_snap_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "util_snap_qa.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_util_snap.md"
AGENT = "b17e9c44"
WAVE = "4"
ROUND = "R4"
MODEL = "util_snap_qa"
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
UTIL_N_QUOTE = 334
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
            fold_rows.append(
                {"fold": k, "auroc": float("nan"), "sign": 0, "n_va": int(va.sum()), "n_pos": int((va & (y == 1)).sum())}
            )
            continue
        sign = choose_sign(y[tr], x[tr])
        auc = auroc(y[va], sign * x[va])
        aucs.append(auc)
        fold_rows.append(
            {
                "fold": k,
                "auroc": float(auc) if np.isfinite(auc) else float("nan"),
                "sign": int(sign),
                "n_va": int(va.sum()),
                "n_pos": int((va & (y == 1)).sum()),
            }
        )
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
    panel["f_util_snapshot"] = pd.to_numeric(panel["f_util_snapshot"], errors="coerce")
    panel["f_outstanding_gt_granted"] = pd.to_numeric(panel["f_outstanding_gt_granted"], errors="coerce")
    panel["util0"] = panel["f_util_snapshot"].fillna(0.0)
    panel["util_def"] = panel["f_util_snapshot"].notna().astype(float)
    leak = leakage_check(
        ["f_util_snapshot", "f_outstanding_gt_granted", "f_n_types", "c_n_days_with_tx", "log_in3"],
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
    x = pd.to_numeric(tr["f_util_snapshot"], errors="coerce")
    og = pd.to_numeric(tr["f_outstanding_gt_granted"], errors="coerce")
    n_cm = int(len(tr))
    n_def = int(x.notna().sum())
    n_og = int(og.notna().sum())
    cov = n_def / n_cm if n_cm else float("nan")
    og_cov = n_og / n_cm if n_cm else float("nan")
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_per = last["period"].mode().iloc[0] if len(last) else pd.NaT
    last_x = pd.to_numeric(last["f_util_snapshot"], errors="coerce")
    n_last = int(last_x.notna().sum())
    last_only = bool(n_def == n_last and n_def > 0)
    periods = tr.loc[x.notna(), "period"].drop_duplicates().sort_values()
    last_share = n_last / n_def if n_def else float("nan")
    acf1 = median_acf(x, tr["company_id"], 1)
    rho = spearman(x, tr["log_in3"])
    size_flag = bool(np.isfinite(rho) and abs(rho) >= SIZE_RHO)
    med = float(x.dropna().median()) if n_def else float("nan")
    p95 = float(x.dropna().quantile(0.95)) if n_def else float("nan")
    n_zero = int((x == 0).sum())
    prose = (
        f"Train util defined {n_def:,}/{n_cm:,} ({_pp(cov)}; quote 1.6% "
        f"{'CONFIRM' if abs(cov - UTIL_COV_QUOTE) < 0.01 else 'off'}). "
        f"OGTG defined {_pp(og_cov)} (quote 5.7% "
        f"{'CONFIRM' if abs(og_cov - OGTG_COV_QUOTE) < 0.02 else 'off'}). "
        f"Defined periods {list(periods.dt.strftime('%Y-%m').astype(str))} "
        f"last-month-only={last_only} last-share {_pp(last_share)} "
        f"(last {pd.Timestamp(last_per).strftime('%Y-%m')} n={n_last}; quote 334 "
        f"{'CONFIRM' if n_def == UTIL_N_QUOTE else 'off'}). "
        f"median {_f(med)} p95 {_f(p95)} zeros {n_zero}. "
        f"ρ vs log1p(a_in3) {_f(rho)} ({'SIZE' if size_flag else 'not SIZE'}). acf1 {_f(acf1)}."
    )
    print(prose)
    return {
        "cov": cov,
        "og_cov": og_cov,
        "n_def": n_def,
        "n_cm": n_cm,
        "n_last": n_last,
        "last_only": last_only,
        "last_share": last_share,
        "last_per": str(pd.Timestamp(last_per).date()),
        "periods": [str(p.date()) for p in periods],
        "rho_in3": rho,
        "size_flag": size_flag,
        "acf1": acf1,
        "med": med,
        "p95": p95,
        "n_zero": n_zero,
        "n_ok": n_def == UTIL_N_QUOTE,
        "cov_ok": abs(cov - UTIL_COV_QUOTE) < 0.01,
        "og_ok": abs(og_cov - OGTG_COV_QUOTE) < 0.02,
        "prose": prose,
    }


def pass2_twins(tr: pd.DataFrame) -> dict:
    x = tr["f_util_snapshot"]
    pairs = [
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ("a_n_tx", tr["a_n_tx"]),
        ("f_n_types", tr["f_n_types"]),
        ("f_outstanding_gt_granted", tr["f_outstanding_gt_granted"]),
        ("log1p(a_in3)", tr["log_in3"]),
        ("f_n_facilities", tr["f_n_facilities"]),
    ]
    rows = []
    rhos = {}
    twins = []
    gate = {"c_n_days_with_tx", "a_n_tx", "f_n_types", "f_outstanding_gt_granted"}
    for name, s in pairs:
        rho = spearman(x, s)
        rhos[name] = rho
        twin = bool(name in gate and np.isfinite(rho) and abs(rho) >= TWIN_RHO)
        if twin:
            twins.append(name)
        rows.append({"vs": name, "ρ": _f(rho), "twin": "YES" if twin else ""})
    prose = (
        f"Spearman twins |ρ|≥0.80 on defined util: {twins or 'none'}. "
        f"vs days {_f(rhos['c_n_days_with_tx'])} vs a_n_tx {_f(rhos['a_n_tx'])} "
        f"vs n_types {_f(rhos['f_n_types'])} vs OGTG {_f(rhos['f_outstanding_gt_granted'])} "
        f"vs size {_f(rhos['log1p(a_in3)'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "rhos": rhos,
        "twins": twins,
        "gate_twins": bool(twins),
        "rho_days": rhos["c_n_days_with_tx"],
        "rho_ntx": rhos["a_n_tx"],
        "rho_types": rhos["f_n_types"],
        "rho_ogtg": rhos["f_outstanding_gt_granted"],
        "rho_size": rhos["log1p(a_in3)"],
        "prose": prose,
    }


def pass3_singles(tr: pd.DataFrame) -> dict:
    feats = {
        "f_util_snapshot": tr["f_util_snapshot"],
        "util_fillna0": tr["util0"],
        "util_defined_dummy": tr["util_def"],
        "f_outstanding_gt_granted": tr["f_outstanding_gt_granted"],
        "f_n_types": tr["f_n_types"],
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
            print(
                f"AUROC {y} {name}: {'LOW_POWER' if res['low_power'] else _f(res['cv'])} "
                f"n={res['n_defined']} n_pos={res['n_pos']}"
            )
    y3 = _cv(store[(Y3, "f_util_snapshot")])
    hole = _cv(store[(Y3, "util_fillna0")])
    dummy = _cv(store[(Y3, "util_defined_dummy")])
    days = _cv(store[(Y3, "c_n_days_with_tx")])
    size = _cv(store[(Y3, "log1p(a_in3)")])
    types = _cv(store[(Y3, "f_n_types")])
    ogtg = _cv(store[(Y3, "f_outstanding_gt_granted")])
    days_ok = bool(np.isfinite(days) and abs(days - DAYS_BENCH) < 0.02)
    size_ok = bool(np.isfinite(size) and abs(size - SIZE_Y3_QUOTE) < 0.03)
    types_ok = bool(np.isfinite(types) and abs(types - N_TYPES_Y3) < 0.03)
    beat_size = y3 - size if np.isfinite(y3) and np.isfinite(size) else float("nan")
    native_lp = store[(Y3, "f_util_snapshot")]["low_power"]
    prose = (
        f"Y3 native util {_f(y3)} n={store[(Y3, 'f_util_snapshot')]['n_defined']} "
        f"n_pos={store[(Y3, 'f_util_snapshot')]['n_pos']} "
        f"{'(LOW_POWER — snapshot month has no Y3)' if native_lp else ''}. "
        f"fillna0 hole {_f(hole)} defined-dummy {_f(dummy)}. "
        f"vs days {_f(days)} (0.711 {'CONFIRM' if days_ok else 'off'}) "
        f"vs size {_f(size)} (0.617 {'CONFIRM' if size_ok else 'off'}) "
        f"vs n_types {_f(types)} (0.578 {'CONFIRM' if types_ok else 'off'}) "
        f"vs OGTG {_f(ogtg)}."
    )
    print(prose)
    return {
        "rows": rows,
        "y3": y3,
        "hole": hole,
        "dummy": dummy,
        "days": days,
        "size": size,
        "types": types,
        "ogtg": ogtg,
        "days_ok": days_ok,
        "size_ok": size_ok,
        "types_ok": types_ok,
        "beat_size": beat_size,
        "native_lp": native_lp,
        "n_pos": store[(Y3, "f_util_snapshot")]["n_pos"],
        "n_def": store[(Y3, "f_util_snapshot")]["n_defined"],
        "prose": prose,
    }


def pass4_leftover(tr: pd.DataFrame) -> dict:
    x = tr["f_util_snapshot"]
    hole = tr["util0"]
    dummy = tr["util_def"]
    days = tr["c_n_days_with_tx"]
    last_mask = x.notna()
    specs = [
        ("native after days (last-month rows)", x, (days,), last_mask & tr[Y3].notna()),
        ("native after days (Y3 panel)", x, (days,), tr[Y3].notna()),
        ("hole after days", hole, (days,), tr[Y3].notna()),
        ("dummy after days", dummy, (days,), tr[Y3].notna()),
        ("hole after n_types", hole, (tr["f_n_types"],), tr[Y3].notna()),
        ("hole after OGTG-fillna0", hole, (tr["f_outstanding_gt_granted"].fillna(0),), tr[Y3].notna()),
        ("dummy after n_types", dummy, (tr["f_n_types"],), tr[Y3].notna()),
        ("dummy after OGTG-fillna0", dummy, (tr["f_outstanding_gt_granted"].fillna(0),), tr[Y3].notna()),
        ("hole after size", hole, (tr["log_in3"],), tr[Y3].notna()),
    ]
    rows = []
    store = {}
    for name, xx, xs, mask in specs:
        rec = leftover_diag(tr[Y3], xx, list(xs), tr["fold"], mask)
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
        print(
            f"left {name}: OLS={_f(rec['ols'])} rank={_f(rec['rank'])} "
            f"n_pos={rec['n_pos']} ρctrl={_f(rec['rho_ctrl'])} fake={rec['fake']}"
        )
    inv = leftover_diag(tr[Y3], days, [x], tr["fold"], tr[Y3].notna())
    inv_h = leftover_diag(tr[Y3], days, [hole], tr["fold"], tr[Y3].notna())
    inv_d = leftover_diag(tr[Y3], days, [dummy], tr["fold"], tr[Y3].notna())
    rows.append(
        {
            "control": "days after native util (inverse)",
            "OLS": _f(inv["ols"]),
            "rank": _f(inv["rank"]),
            "n": inv["n"],
            "n_pos": inv["n_pos"],
            "ρ(resid,ctrl)": _f(inv["rho_ctrl"]),
            "fake": "YES" if inv["fake"] else "",
            "dies": "YES" if inv["honest_dies"] else "no",
        }
    )
    rows.append(
        {
            "control": "days after hole fillna0 (inverse)",
            "OLS": _f(inv_h["ols"]),
            "rank": _f(inv_h["rank"]),
            "n": inv_h["n"],
            "n_pos": inv_h["n_pos"],
            "ρ(resid,ctrl)": _f(inv_h["rho_ctrl"]),
            "fake": "YES" if inv_h["fake"] else "",
            "dies": "YES" if inv_h["honest_dies"] else "no",
        }
    )
    nat = store["native after days (Y3 panel)"]
    hol = store["hole after days"]
    dum = store["dummy after days"]
    leftover = hol["rank"] if nat["low_power"] else nat["rank"]
    leftover_ols = hol["ols"] if nat["low_power"] else nat["ols"]
    dies = bool(nat["honest_dies"] or hol["honest_dies"] or nat["low_power"] or dum["honest_dies"])
    prose = (
        f"Native leftover after days rank {_f(nat['rank'])} n_pos={nat['n_pos']} "
        f"{'(LOW_POWER)' if nat['low_power'] else ''}. "
        f"Last-month-rows leftover {_f(store['native after days (last-month rows)']['rank'])} "
        f"n_pos={store['native after days (last-month rows)']['n_pos']}. "
        f"fillna0 hole leftover after days rank {_f(hol['rank'])} OLS {_f(hol['ols'])} "
        f"ρ(resid,days)={_f(hol['rho_ctrl'])} fake={hol['fake']}. "
        f"Defined-dummy leftover {_f(dum['rank'])} fake={dum['fake']}. "
        f"After n_types hole {_f(store['hole after n_types']['rank'])} "
        f"after OGTG hole {_f(store['hole after OGTG-fillna0']['rank'])}. "
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
        "last_rank": store["native after days (last-month rows)"]["rank"],
        "last_npos": store["native after days (last-month rows)"]["n_pos"],
        "hole_rank": hol["rank"],
        "hole_ols": hol["ols"],
        "hole_rho": hol["rho_ctrl"],
        "hole_fake": hol["fake"],
        "dummy_rank": dum["rank"],
        "dummy_fake": dum["fake"],
        "types_left": store["hole after n_types"]["rank"],
        "ogtg_left": store["hole after OGTG-fillna0"]["rank"],
        "dummy_types": store["dummy after n_types"]["rank"],
        "dummy_ogtg": store["dummy after OGTG-fillna0"]["rank"],
        "inv_rank": inv_h["rank"],
        "inv_dummy": inv_d["rank"],
        "inv_native": inv["rank"],
        "y3_dies": dies,
        "prose": prose,
    }


def pass5_rise(tr: pd.DataFrame) -> dict:
    work = tr.sort_values(["company_id", "period"]).copy()
    x = pd.to_numeric(work["f_util_snapshot"], errors="coerce")
    d = x.groupby(work["company_id"], sort=False).diff()
    rises = int((d > 0).sum())
    drops = int((d < 0).sum())
    defined_pairs = int(d.notna().sum())
    last = work.sort_values("period").groupby("company_id", as_index=False).tail(1)
    n_def = int(pd.to_numeric(last["f_util_snapshot"], errors="coerce").notna().sum())
    extract = defined_pairs == 0
    y10_imp = extract
    prose = (
        f"Util rises {rises} / drops {drops} / defined diffs {defined_pairs} "
        f"({'extract-hole: no within-company pair' if extract else 'has a path'}). "
        f"Last-month defined {n_def} (quote 334 {'CONFIRM' if n_def == UTIL_N_QUOTE else 'off'}). "
        f"Y10 utilisation impossible={'CONFIRM' if y10_imp else 'off'} "
        f"(no outstanding/granted history; snapshot last-month only)."
    )
    print(prose)
    return {
        "rises": rises,
        "drops": drops,
        "pairs": defined_pairs,
        "extract": extract,
        "n_def": n_def,
        "y10_impossible": y10_imp,
        "prose": prose,
    }


def pass6_dark(tr: pd.DataFrame, book: set[str]) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    n_erp = int(last["company_id"].isin(book).sum())
    n_dark = int((~last["company_id"].isin(book)).sum())
    confirm = n_erp == 744 and n_dark == 470
    last["erp"] = last["company_id"].isin(book)
    x = pd.to_numeric(last["f_util_snapshot"], errors="coerce")
    rows = []
    for name, mask in (("invoiced_744", last["erp"]), ("dark_470", ~last["erp"])):
        xx = x[mask]
        n1 = int(xx.notna().sum())
        med = float(xx.median()) if xx.notna().any() else float("nan")
        rows.append(
            {
                "slice": name,
                "n": int(mask.sum()),
                "defined": n1,
                "rate": _pp(n1 / int(mask.sum()) if int(mask.sum()) else float("nan")),
                "median": _f(med),
            }
        )
    hole = leftover_diag(
        tr[Y3], tr["util0"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & tr["company_id"].isin(book)
    )
    hole_d = leftover_diag(
        tr[Y3], tr["util0"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & ~tr["company_id"].isin(book)
    )
    prose = (
        f"Last-month ever-ERP {n_erp} / never-ERP {n_dark} ({'CONFIRM 744/470' if confirm else 'off'}). "
        f"util defined invoiced {rows[0]['defined']} dark {rows[1]['defined']}. "
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
    for col in ("f_util_snapshot", "util0", "util0_lag1", "c_n_days_with_tx", "c_n_days_with_tx_lag1"):
        if col not in tr.columns:
            continue
        res = signed_oof_auroc(tr[Y3], tr[col], tr["fold"], tr[Y3].notna())
        store[col] = res
        rows.append(
            {
                "col": col,
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
            }
        )
    days_l1 = _cv(store.get("c_n_days_with_tx_lag1", {"low_power": True}))
    rec1 = (
        leftover_diag(tr[Y3], tr["util0_lag1"], [tr["c_n_days_with_tx_lag1"]], tr["fold"], tr[Y3].notna())
        if "util0_lag1" in tr.columns
        else {"rank": float("nan"), "honest_dies": True, "rho_ctrl": float("nan"), "fake": True}
    )
    days_l1_ok = bool(np.isfinite(days_l1) and abs(days_l1 - DAYS_LAG1_QUOTE) < 0.03)
    native_lag = store.get("f_util_snapshot", {})
    q6 = "CLOSE"
    prose = (
        f"Snapshot cannot lead: native util Y3 n_pos={native_lag.get('n_pos', 0)} "
        f"(LOW_POWER on extract month). fillna0 lag1 leftover after days_lag1 {_f(rec1['rank'])} "
        f"ρ={_f(rec1.get('rho_ctrl', float('nan')))} fake={rec1.get('fake')}. "
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


def pass8_trait(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_util_snapshot"].last()
    ever = tr["company_id"].map(last)
    raw = signed_oof_auroc(tr[Y3], ever, tr["fold"], tr[Y3].notna())
    rec = leftover_diag(tr[Y3], ever, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    rec_t = leftover_diag(tr[Y3], ever, [tr["f_n_types"]], tr["fold"], tr[Y3].notna())
    rec_o = leftover_diag(tr[Y3], ever, [tr["f_outstanding_gt_granted"].fillna(0)], tr["fold"], tr[Y3].notna())
    n_def = int(last.notna().sum())
    ever_y3 = tr.groupby("company_id")[Y3].max()
    has = last.notna()
    share_u = float(ever_y3[has[has.index.isin(ever_y3.index)]].mean()) if n_def else float("nan")
    share_no = float(ever_y3[~has.reindex(ever_y3.index).fillna(False)].mean())
    n_rec = int(ever_y3.reindex(has.index[has]).fillna(0).sum())
    prose = (
        f"Company last-month util defined {n_def}/1214. Trait leftover after days {_f(rec['rank'])} "
        f"ρ={_f(rec['rho_ctrl'])} fake={rec['fake']} raw {_f(_cv(raw))}. "
        f"After n_types {_f(rec_t['rank'])} after OGTG {_f(rec_o['rank'])}. "
        f"Ever-Y3 recover util {n_rec}/{n_def} ({_pp(share_u)}) vs rest {_pp(share_no)}."
    )
    print(prose)
    return {
        "n_def": n_def,
        "ever": _cv(raw),
        "left": rec["rank"],
        "rho": rec["rho_ctrl"],
        "fake": rec["fake"],
        "types": rec_t["rank"],
        "ogtg": rec_o["rank"],
        "share_u": share_u,
        "share_no": share_no,
        "n_rec": n_rec,
        "prose": prose,
    }


def pass_y3_last(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    n_y3 = int(last[Y3].notna().sum())
    n_y2 = int(last[Y2].notna().sum())
    x = pd.to_numeric(last["f_util_snapshot"], errors="coerce")
    prose = (
        f"Last-month labeled Y3 {n_y3} Y2 {n_y2} (horizon → empty; quote Y3=0). "
        f"Util defined {int(x.notna().sum())}."
    )
    print(prose)
    return {"n_y3": n_y3, "n_y2": n_y2, "empty": n_y3 == 0, "prose": prose}


def pass_holdout(panel: pd.DataFrame) -> dict:
    ho = panel[panel["split"] == "holdout"]
    x = pd.to_numeric(ho["f_util_snapshot"], errors="coerce")
    last = ho.sort_values("period").groupby("company_id", as_index=False).tail(1)
    lx = pd.to_numeric(last["f_util_snapshot"], errors="coerce")
    periods = ho.loc[x.notna(), "period"].drop_duplicates().sort_values()
    last_only = bool(int(x.notna().sum()) == int(lx.notna().sum()) and int(x.notna().sum()) > 0)
    prose = (
        f"Holdout coverage only (no fit): {ho['company_id'].nunique()} co / {len(ho):,} CM, "
        f"defined {int(x.notna().sum())} last-month {int(lx.notna().sum())} "
        f"periods {list(periods.dt.strftime('%Y-%m').astype(str))} last-month-only={last_only} "
        f"(debt-schedule quote 13 CONFIRM {int(x.notna().sum()) == 13})."
    )
    print(prose)
    return {"n_cm": int(len(ho)), "n_def": int(x.notna().sum()), "last_only": last_only, "prose": prose}


def pass_boot(tr: pd.DataFrame, n_boot: int = 48) -> dict:
    rng = np.random.default_rng(FOLD_SEED)
    last = tr.sort_values("period").groupby("company_id")["f_util_snapshot"].last()
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


def pass_trait_stack(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_util_snapshot"].last()
    ever = tr["company_id"].map(last)
    rec = leftover_diag(
        tr[Y3], ever, [tr["c_n_days_with_tx"], tr["f_n_types"]], tr["fold"], tr[Y3].notna()
    )
    rec_o = leftover_diag(
        tr[Y3],
        ever,
        [tr["c_n_days_with_tx"], tr["f_outstanding_gt_granted"].fillna(0)],
        tr["fold"],
        tr[Y3].notna(),
    )
    rec_s = leftover_diag(
        tr[Y3], ever, [tr["c_n_days_with_tx"], tr["log_in3"]], tr["fold"], tr[Y3].notna()
    )
    prose = (
        f"Trait leftover after days+n_types {_f(rec['rank'])} ρ={_f(rec['rho_ctrl'])}; "
        f"after days+OGTG {_f(rec_o['rank'])}; after days+size {_f(rec_s['rank'])}."
    )
    print(prose)
    return {"types": rec["rank"], "ogtg": rec_o["rank"], "size": rec_s["rank"], "prose": prose}


def pass_trait_after_conn(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    conn = set(last.loc[pd.to_numeric(last["f_util_snapshot"], errors="coerce").notna(), "company_id"])
    dummy = tr["company_id"].isin(conn).astype(float)
    last_u = last.set_index("company_id")["f_util_snapshot"]
    ever = tr["company_id"].map(last_u)
    rec_d = leftover_diag(tr[Y3], dummy, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    rec_o = leftover_diag(tr[Y3], ever, [dummy], tr["fold"], tr[Y3].notna())
    rec_s = leftover_diag(tr[Y3], ever, [tr["c_n_days_with_tx"], dummy], tr["fold"], tr[Y3].notna())
    prose = (
        f"Util-defined company dummy leftover after days {_f(rec_d['rank'])} fake={rec_d['fake']}. "
        f"Trait leftover after defined-dummy {_f(rec_o['rank'])}; after days+dummy {_f(rec_s['rank'])}."
    )
    print(prose)
    return {"dummy": rec_d["rank"], "after": rec_o["rank"], "stack": rec_s["rank"], "prose": prose}


def pass_dark_trait(tr: pd.DataFrame, book: set[str]) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_util_snapshot"].last()
    ever = tr["company_id"].map(last)
    erp = tr["company_id"].isin(book)
    rec_i = leftover_diag(tr[Y3], ever, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & erp)
    rec_d = leftover_diag(tr[Y3], ever, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & ~erp)
    prose = (
        f"Trait leftover after days invoiced {_f(rec_i['rank'])} ρ={_f(rec_i['rho_ctrl'])} "
        f"dark {_f(rec_d['rank'])} ρ={_f(rec_d['rho_ctrl'])} (access ≠ ERP)."
    )
    print(prose)
    return {"inv": rec_i["rank"], "dark": rec_d["rank"], "prose": prose}


def pass_perm(tr: pd.DataFrame, n_perm: int = 24) -> dict:
    rng = np.random.default_rng(FOLD_SEED + 7)
    last = tr.sort_values("period").groupby("company_id")["f_util_snapshot"].last()
    obs = leftover_diag(tr[Y3], tr["company_id"].map(last), [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())["rank"]
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
        f"Permute last-month util leftover-after-days null p50 "
        f"{_f(float(np.median(arr)) if arr.size else float('nan'))} "
        f"p(obs≥null)={_f(p, 3)} obs={_f(obs)}."
    )
    print(prose)
    return {"p50": float(np.median(arr)) if arr.size else float("nan"), "p": p, "obs": obs, "prose": prose}


def pass_fold_trait(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_util_snapshot"].last()
    ever = tr["company_id"].map(last)
    rec = leftover_diag(tr[Y3], ever, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    prose = (
        f"Trait leftover-after-days rank folds {rec['rank_folds']} "
        f"rank {_f(rec['rank'])} ρ={_f(rec['rho_ctrl'])} fake={rec['fake']}."
    )
    print(prose)
    return {"folds": rec["rank_folds"], "rank": rec["rank"], "prose": prose}


def pass_last_double(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    ever = tr["company_id"].map(last_u)
    rec = leftover_diag(
        tr[Y3],
        ever,
        [tr["company_id"].map(last_d), tr["company_id"].map(last_t)],
        tr["fold"],
        tr[Y3].notna(),
    )
    rec_d = leftover_diag(tr[Y3], ever, [tr["company_id"].map(last_d)], tr["fold"], tr[Y3].notna())
    prose = (
        f"Trait leftover after last-month days {_f(rec_d['rank'])} ρ={_f(rec_d['rho_ctrl'])}; "
        f"after last-month days+n_types {_f(rec['rank'])} "
        f"({'dies' if np.isfinite(rec['rank']) and rec['rank'] < CHANCE else 'above 0.55'})."
    )
    print(prose)
    return {"days": rec_d["rank"], "stack": rec["rank"], "prose": prose}


def pass_y2_trait(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_util_snapshot"].last()
    ever = tr["company_id"].map(last)
    rec = leftover_diag(tr[Y2], ever, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y2].notna())
    raw = signed_oof_auroc(tr[Y2], ever, tr["fold"], tr[Y2].notna())
    prose = (
        f"Y2 trait leftover after days {_f(rec['rank'])} ρ={_f(rec['rho_ctrl'])} "
        f"fake={rec['fake']} raw {_f(_cv(raw))}."
    )
    print(prose)
    return {"left": rec["rank"], "fake": rec["fake"], "raw": _cv(raw), "prose": prose}


def pass_inv_trait(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_util_snapshot"].last()
    ever = tr["company_id"].map(last)
    rec_d = leftover_diag(tr[Y3], tr["c_n_days_with_tx"], [ever], tr["fold"], tr[Y3].notna())
    rec_t = leftover_diag(tr[Y3], tr["f_n_types"], [ever], tr["fold"], tr[Y3].notna())
    prose = (
        f"Inverse: days leftover after util-trait {_f(rec_d['rank'])}; "
        f"n_types leftover after util-trait {_f(rec_t['rank'])} (0.711 bar should live)."
    )
    print(prose)
    return {"days": rec_d["rank"], "types": rec_t["rank"], "prose": prose}


def pass_terciles(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_util_snapshot"].last()
    ever = tr["company_id"].map(last)
    size = pd.to_numeric(tr["log_in3"], errors="coerce")
    terc = pd.qcut(size, 3, labels=["T1", "T2", "T3"], duplicates="drop")
    leftover = {}
    for lab in ("T1", "T2", "T3"):
        rec = leftover_diag(tr[Y3], ever, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & terc.eq(lab))
        leftover[lab] = rec["rank"]
    prose = (
        f"Trait leftover after days T1/T2/T3 {_f(leftover['T1'])} / {_f(leftover['T2'])} / {_f(leftover['T3'])}."
    )
    print(prose)
    return {"leftover": leftover, "prose": prose}


def pass_last_loc(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_l = last.set_index("company_id")["f_has_loc"]
    last_f = last.set_index("company_id")["f_n_facilities"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    ever = tr["company_id"].map(last_u)
    rec_l = leftover_diag(tr[Y3], ever, [tr["company_id"].map(last_l)], tr["fold"], tr[Y3].notna())
    rec_f = leftover_diag(tr[Y3], ever, [tr["company_id"].map(last_f)], tr["fold"], tr[Y3].notna())
    rec_s = leftover_diag(
        tr[Y3],
        ever,
        [tr["company_id"].map(last_d), tr["company_id"].map(last_t), tr["company_id"].map(last_l)],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"Trait leftover after last-month has_loc {_f(rec_l['rank'])}; "
        f"after last-month n_facilities {_f(rec_f['rank'])}; "
        f"after last-month days+n_types+has_loc {_f(rec_s['rank'])}."
    )
    print(prose)
    return {"loc": rec_l["rank"], "fac": rec_f["rank"], "triple": rec_s["rank"], "prose": prose}


def pass_fold_last_double(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    ever = tr["company_id"].map(last_u)
    rec = leftover_diag(
        tr[Y3],
        ever,
        [tr["company_id"].map(last_d), tr["company_id"].map(last_t)],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"Trait leftover-after-last-month days+n_types rank folds {rec['rank_folds']} "
        f"rank {_f(rec['rank'])}."
    )
    print(prose)
    return {"folds": rec["rank_folds"], "rank": rec["rank"], "prose": prose}


def pass_between(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_util_snapshot"].last()
    ever = tr["company_id"].map(last)
    d = pd.DataFrame({"x": pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce"), "co": tr["company_id"].astype(str)})
    dmean = d.groupby("co")["x"].transform("mean")
    rec = leftover_diag(tr[Y3], ever, [dmean], tr["fold"], tr[Y3].notna())
    prose = (
        f"BETWEEN leftover of util-trait after days-mean {_f(rec['rank'])} "
        f"ρ={_f(rec['rho_ctrl'])} fake={rec['fake']}."
    )
    print(prose)
    return {"left": rec["rank"], "fake": rec["fake"], "prose": prose}


def pass_on_defined(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_util_snapshot"].last()
    ids = set(last.dropna().index)
    m = tr["company_id"].isin(ids)
    rec_d = leftover_diag(tr[Y3], tr["c_n_days_with_tx"], [tr["f_n_types"]], tr["fold"], tr[Y3].notna() & m)
    rec_t = leftover_diag(tr[Y3], tr["f_n_types"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & m)
    raw_d = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], tr[Y3].notna() & m)
    raw_u = signed_oof_auroc(tr[Y3], tr["company_id"].map(last), tr["fold"], tr[Y3].notna() & m)
    rec_u = leftover_diag(
        tr[Y3], tr["company_id"].map(last), [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & m
    )
    prose = (
        f"On 334 util-defined companies: days Y3 {_f(_cv(raw_d))} leftover after n_types {_f(rec_d['rank'])} "
        f"n_pos={raw_d['n_pos']}; util-trait Y3 {_f(_cv(raw_u))} leftover after days {_f(rec_u['rank'])}; "
        f"n_types leftover after days {_f(rec_t['rank'])}."
    )
    print(prose)
    return {
        "days": _cv(raw_d),
        "util": _cv(raw_u),
        "util_left": rec_u["rank"],
        "n_pos": raw_d["n_pos"],
        "prose": prose,
    }


def pass_high_util(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_util_snapshot"].last()
    hi = last.ge(0.90)
    dummy = tr["company_id"].map(hi.astype(float))
    rec = leftover_diag(tr[Y3], dummy, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    n_hi = int(hi.fillna(False).sum())
    prose = (
        f"Last-month util≥0.90 companies {n_hi}. Dummy leftover after days {_f(rec['rank'])} "
        f"ρ={_f(rec['rho_ctrl'])} fake={rec['fake']}."
    )
    print(prose)
    return {"n_hi": n_hi, "left": rec["rank"], "prose": prose}


def pass_logo(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_util_snapshot"].last()
    ever = tr["company_id"].map(last)
    groups = tr["group_id"].dropna().drop_duplicates().to_numpy()
    vals = []
    for g in groups:
        rec = leftover_diag(
            tr[Y3], ever, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & (tr["group_id"] != g)
        )
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


def pass_after_lags(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_util_snapshot"].last()
    ever = tr["company_id"].map(last)
    rec_l1 = leftover_diag(tr[Y3], ever, [tr["c_n_days_with_tx_lag1"]], tr["fold"], tr[Y3].notna())
    rec_s = leftover_diag(
        tr[Y3], ever, [tr["c_n_days_with_tx"], tr["c_n_days_with_tx_lag1"]], tr["fold"], tr[Y3].notna()
    )
    prose = (
        f"Trait leftover after days_lag1 {_f(rec_l1['rank'])} ρ={_f(rec_l1['rho_ctrl'])} "
        f"fake={rec_l1['fake']}; after days+lag1 {_f(rec_s['rank'])}."
    )
    print(prose)
    return {"lag1": rec_l1["rank"], "stack": rec_s["rank"], "prose": prose}


def pass_dark_last(tr: pd.DataFrame, book: set[str]) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    ever = tr["company_id"].map(last_u)
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


def pass_first_year(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last["fy"] = pd.to_datetime(last["first_month"], errors="coerce").dt.year
    ut = pd.to_numeric(last["f_util_snapshot"], errors="coerce")
    rows = []
    for y, sub in last.groupby("fy"):
        nd = int(ut.loc[sub.index].notna().sum())
        rows.append(
            {
                "first_year": str(int(y)) if pd.notna(y) else "NA",
                "n": int(len(sub)),
                "defined": nd,
                "rate": _pp(nd / len(sub) if len(sub) else float("nan")),
            }
        )
    prose = "Last-month util defined by first_month year: " + ", ".join(f"{r['first_year']} {r['rate']}" for r in rows)
    print(prose)
    return {"rows": rows, "prose": prose}


def pass_last_ntx(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_n = last.set_index("company_id")["a_n_tx"]
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    last_o = last.set_index("company_id")["f_outstanding_gt_granted"]
    last_s = last.set_index("company_id")["log_in3"]
    last_t = last.set_index("company_id")["f_n_types"]
    ever = tr["company_id"].map(last_u)
    rec_n = leftover_diag(tr[Y3], ever, [tr["company_id"].map(last_n)], tr["fold"], tr[Y3].notna())
    rec_s = leftover_diag(
        tr[Y3],
        ever,
        [tr["company_id"].map(last_d), tr["company_id"].map(last_t), tr["company_id"].map(last_s)],
        tr["fold"],
        tr[Y3].notna(),
    )
    rec_o = leftover_diag(
        tr[Y3],
        ever,
        [tr["company_id"].map(last_d), tr["company_id"].map(last_t), tr["company_id"].map(last_o)],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"Trait leftover after last-month a_n_tx {_f(rec_n['rank'])}; "
        f"after last-month days+n_types+size {_f(rec_s['rank'])}; "
        f"after last-month days+n_types+OGTG {_f(rec_o['rank'])}."
    )
    print(prose)
    return {"ntx": rec_n["rank"], "size": rec_s["rank"], "ogtg": rec_o["rank"], "prose": prose}


def pass_t3_last(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    ever = tr["company_id"].map(last_u)
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
        f"{_f(leftover['T1'])} / {_f(leftover['T2'])} / {_f(leftover['T3'])}."
    )
    print(prose)
    return {"leftover": leftover, "prose": prose}


def pass_fold_triple(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_l = last.set_index("company_id")["f_has_loc"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    ever = tr["company_id"].map(last_u)
    rec = leftover_diag(
        tr[Y3],
        ever,
        [tr["company_id"].map(last_d), tr["company_id"].map(last_t), tr["company_id"].map(last_l)],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"Trait leftover-after-last-month days+n_types+has_loc rank folds {rec['rank_folds']} "
        f"rank {_f(rec['rank'])}."
    )
    print(prose)
    return {"folds": rec["rank_folds"], "rank": rec["rank"], "prose": prose}


def pass_y2_last(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    ever = tr["company_id"].map(last_u)
    rec = leftover_diag(
        tr[Y2],
        ever,
        [tr["company_id"].map(last_d), tr["company_id"].map(last_t)],
        tr["fold"],
        tr[Y2].notna(),
    )
    prose = (
        f"Y2 leftover after last-month days+n_types {_f(rec['rank'])} "
        f"ρ={_f(rec['rho_ctrl'])} fake={rec['fake']}."
    )
    print(prose)
    return {"left": rec["rank"], "prose": prose}


def pass_dummy_triple(tr: pd.DataFrame) -> dict:
    snap = tr["f_util_snapshot"].notna().astype(float)
    rec = leftover_diag(
        tr[Y3],
        snap,
        [tr["c_n_days_with_tx"], tr["f_n_types"], tr["f_has_loc"]],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"Last-month dummy leftover after days+n_types+has_loc {_f(rec['rank'])} "
        f"ρ={_f(rec['rho_ctrl'])} ({'dies' if np.isfinite(rec['rank']) and rec['rank'] < CHANCE else 'above 0.55'})."
    )
    print(prose)
    return {"left": rec["rank"], "prose": prose}


def pass_fact_left(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_f = last.set_index("company_id")["f_has_factoring"]
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    ever = tr["company_id"].map(last_u)
    rec = leftover_diag(
        tr[Y3],
        ever,
        [tr["company_id"].map(last_d), tr["company_id"].map(last_t), tr["company_id"].map(last_f)],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = f"Trait leftover after last-month days+n_types+has_factoring {_f(rec['rank'])}."
    print(prose)
    return {"left": rec["rank"], "prose": prose}


def pass_has_any_left(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    any_h = (
        pd.to_numeric(last["f_has_loc"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_factoring"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_confirming"], errors="coerce").eq(1)
    )
    last_a = last.set_index("company_id")
    dummy_map = pd.Series(any_h.to_numpy(dtype=float), index=last["company_id"])
    last_d = last_a["c_n_days_with_tx"]
    last_t = last_a["f_n_types"]
    last_u = last_a["f_util_snapshot"]
    ever = tr["company_id"].map(last_u)
    rec = leftover_diag(
        tr[Y3],
        ever,
        [tr["company_id"].map(last_d), tr["company_id"].map(last_t), tr["company_id"].map(dummy_map)],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"Trait leftover after last-month days+n_types+has_any(loc/fact/conf) {_f(rec['rank'])}."
    )
    print(prose)
    return {"left": rec["rank"], "prose": prose}


def pass_fold_penta_pos(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    any_h = (
        pd.to_numeric(last["f_has_loc"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_factoring"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_confirming"], errors="coerce").eq(1)
    )
    dummy_map = pd.Series(any_h.to_numpy(dtype=float), index=last["company_id"])
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_o = last.set_index("company_id")["f_outstanding_gt_granted"]
    last_s = last.set_index("company_id")["log_in3"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    ids = set(last.loc[pd.to_numeric(last["f_util_snapshot"], errors="coerce").gt(0), "company_id"])
    ever = tr["company_id"].map(last_u)
    rec = leftover_diag(
        tr[Y3],
        ever,
        [
            tr["company_id"].map(last_d),
            tr["company_id"].map(last_t),
            tr["company_id"].map(dummy_map),
            tr["company_id"].map(last_o),
            tr["company_id"].map(last_s),
        ],
        tr["fold"],
        tr[Y3].notna() & tr["company_id"].isin(ids),
    )
    prose = (
        f"On util>0 leftover-after-last-month days+n_types+has_any+OGTG+size rank folds {rec['rank_folds']} "
        f"rank {_f(rec['rank'])} n_pos={rec['n_pos']}."
    )
    print(prose)
    return {"folds": rec["rank_folds"], "rank": rec["rank"], "prose": prose}


def _last_any(last: pd.DataFrame) -> pd.Series:
    return (
        pd.to_numeric(last["f_has_loc"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_factoring"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_confirming"], errors="coerce").eq(1)
    )


def pass_inv_ogtg(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_u = last.set_index("company_id")["f_util_snapshot"]
    last_o = last.set_index("company_id")["f_outstanding_gt_granted"]
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    ever_u = tr["company_id"].map(last_u)
    ever_o = tr["company_id"].map(last_o)
    rec_ou = leftover_diag(tr[Y3], ever_o, [ever_u], tr["fold"], tr[Y3].notna())
    rec_uo = leftover_diag(tr[Y3], ever_u, [ever_o], tr["fold"], tr[Y3].notna())
    rec_uod = leftover_diag(
        tr[Y3],
        ever_u,
        [ever_o, tr["company_id"].map(last_d)],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"Inverse leftover of last-month OGTG after last-month util {_f(rec_ou['rank'])} "
        f"ρ={_f(rec_ou['rho_ctrl'])} fake={rec_ou['fake']}; "
        f"util leftover after last-month OGTG {_f(rec_uo['rank'])}; "
        f"after last-month OGTG+days {_f(rec_uod['rank'])}."
    )
    print(prose)
    return {
        "ogtg_after_util": rec_ou["rank"],
        "util_after_ogtg": rec_uo["rank"],
        "util_after_ogtg_days": rec_uod["rank"],
        "prose": prose,
    }


def pass_ogtg_on_util(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_o = last.set_index("company_id")["f_outstanding_gt_granted"]
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    ids = set(last.loc[pd.to_numeric(last["f_util_snapshot"], errors="coerce").notna(), "company_id"])
    rec = leftover_diag(
        tr[Y3],
        tr["company_id"].map(last_o),
        [tr["company_id"].map(last_d)],
        tr["fold"],
        tr[Y3].notna() & tr["company_id"].isin(ids),
    )
    prose = (
        f"On 334 util-defined companies last-month OGTG leftover after last-month days "
        f"{_f(rec['rank'])} n_pos={rec['n_pos']} ρ={_f(rec['rho_ctrl'])} fake={rec['fake']} "
        f"(OGTG PARK leftover 0.540)."
    )
    print(prose)
    return {"left": rec["rank"], "n_pos": rec["n_pos"], "prose": prose}


def pass_q6_inv(tr: pd.DataFrame) -> dict:
    rec_d = leftover_diag(
        tr[Y3], tr["c_n_days_with_tx"], [tr["util0_lag1"]], tr["fold"], tr[Y3].notna()
    )
    rec_l = leftover_diag(
        tr[Y3], tr["c_n_days_with_tx_lag1"], [tr["util0_lag1"]], tr["fold"], tr[Y3].notna()
    )
    prose = (
        f"Q6 inverse: days leftover after util0_lag1 {_f(rec_d['rank'])} "
        f"ρ={_f(rec_d['rho_ctrl'])} fake={rec_d['fake']}; "
        f"days_lag1 leftover after util0_lag1 {_f(rec_l['rank'])} "
        f"(snapshot lag cannot soak days; Q6 CLOSE)."
    )
    print(prose)
    return {"days": rec_d["rank"], "days_l1": rec_l["rank"], "prose": prose}


def pass_dark_hole_last(tr: pd.DataFrame, book: set[str]) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    hole = pd.to_numeric(tr["f_util_snapshot"], errors="coerce").fillna(0.0)
    xs = [tr["company_id"].map(last_d)]
    erp = tr["company_id"].isin(book)
    rec_i = leftover_diag(tr[Y3], hole, xs, tr["fold"], tr[Y3].notna() & erp)
    rec_d = leftover_diag(tr[Y3], hole, xs, tr["fold"], tr[Y3].notna() & ~erp)
    prose = (
        f"fillna0 hole leftover after last-month days invoiced {_f(rec_i['rank'])} "
        f"ρ={_f(rec_i['rho_ctrl'])} fake={rec_i['fake']} "
        f"dark {_f(rec_d['rank'])} ρ={_f(rec_d['rho_ctrl'])} fake={rec_d['fake']} "
        f"(hole is last-month dummy; leftover of last-month days)."
    )
    print(prose)
    return {"inv": rec_i["rank"], "dark": rec_d["rank"], "prose": prose}


def pass_boot_penta(tr: pd.DataFrame, n_boot: int = 24) -> dict:
    rng = np.random.default_rng(FOLD_SEED + 29)
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    dummy_map = pd.Series(_last_any(last).to_numpy(dtype=float), index=last["company_id"])
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_o = last.set_index("company_id")["f_outstanding_gt_granted"]
    last_s = last.set_index("company_id")["log_in3"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    cos = tr["company_id"].drop_duplicates().to_numpy()
    idx = {c: tr.index[tr["company_id"] == c].to_numpy() for c in cos}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(cos, size=len(cos), replace=True)
        rows = np.concatenate([idx[c] for c in pick])
        sub = tr.loc[rows].reset_index(drop=True)
        ev = sub["company_id"].map(last_u)
        rec = leftover_diag(
            sub[Y3],
            ev,
            [
                sub["company_id"].map(last_d),
                sub["company_id"].map(last_t),
                sub["company_id"].map(dummy_map),
                sub["company_id"].map(last_o),
                sub["company_id"].map(last_s),
            ],
            sub["fold"],
            sub[Y3].notna(),
        )
        if np.isfinite(rec["rank"]):
            vals.append(rec["rank"])
    arr = np.array(vals, dtype=float)
    prose = (
        f"Company bootstrap leftover after last-month days+n_types+has_any+OGTG+size n={arr.size} "
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


def pass_logo_penta(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    dummy_map = pd.Series(_last_any(last).to_numpy(dtype=float), index=last["company_id"])
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_o = last.set_index("company_id")["f_outstanding_gt_granted"]
    last_s = last.set_index("company_id")["log_in3"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    ever = tr["company_id"].map(last_u)
    xs = [
        tr["company_id"].map(last_d),
        tr["company_id"].map(last_t),
        tr["company_id"].map(dummy_map),
        tr["company_id"].map(last_o),
        tr["company_id"].map(last_s),
    ]
    groups = tr["group_id"].dropna().drop_duplicates().to_numpy()
    vals = []
    for g in groups:
        rec = leftover_diag(
            tr[Y3], ever, xs, tr["fold"], tr[Y3].notna() & (tr["group_id"] != g)
        )
        if np.isfinite(rec["rank"]):
            vals.append(rec["rank"])
    arr = np.array(vals, dtype=float)
    prose = (
        f"Leave-one-group leftover after last-month days+n_types+has_any+OGTG+size n={arr.size} "
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


def pass_ogtg_inv_days(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_o = last.set_index("company_id")["f_outstanding_gt_granted"]
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    ids = set(last.loc[pd.to_numeric(last["f_util_snapshot"], errors="coerce").notna(), "company_id"])
    rec_do = leftover_diag(
        tr[Y3],
        tr["company_id"].map(last_d),
        [tr["company_id"].map(last_o)],
        tr["fold"],
        tr[Y3].notna() & tr["company_id"].isin(ids),
    )
    rec_odu = leftover_diag(
        tr[Y3],
        tr["company_id"].map(last_o),
        [tr["company_id"].map(last_d), tr["company_id"].map(last_u)],
        tr["fold"],
        tr[Y3].notna() & tr["company_id"].isin(ids),
    )
    rec_all = leftover_diag(
        tr[Y3],
        tr["company_id"].map(last_o),
        [tr["company_id"].map(last_d)],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"On 334 inverse last-month days leftover after last-month OGTG {_f(rec_do['rank'])} "
        f"ρ={_f(rec_do['rho_ctrl'])} fake={rec_do['fake']}; "
        f"OGTG leftover after last-month days+util {_f(rec_odu['rank'])} "
        f"ρ={_f(rec_odu['rho_ctrl'])} fake={rec_odu['fake']}; "
        f"last-month OGTG leftover after last-month days on all companies {_f(rec_all['rank'])} "
        f"(PARK 0.540 quote)."
    )
    print(prose)
    return {
        "days_after_ogtg": rec_do["rank"],
        "ogtg_after_days_util": rec_odu["rank"],
        "ogtg_all": rec_all["rank"],
        "prose": prose,
    }


def pass_hexa(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    dummy_map = pd.Series(_last_any(last).to_numpy(dtype=float), index=last["company_id"])
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_o = last.set_index("company_id")["f_outstanding_gt_granted"]
    last_s = last.set_index("company_id")["log_in3"]
    last_n = last.set_index("company_id")["a_n_tx"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    ever = tr["company_id"].map(last_u)
    xs = [
        tr["company_id"].map(last_d),
        tr["company_id"].map(last_t),
        tr["company_id"].map(dummy_map),
        tr["company_id"].map(last_o),
        tr["company_id"].map(last_s),
        tr["company_id"].map(last_n),
    ]
    rec = leftover_diag(tr[Y3], ever, xs, tr["fold"], tr[Y3].notna())
    rec_y2 = leftover_diag(tr[Y2], ever, xs, tr["fold"], tr[Y2].notna())
    prose = (
        f"Trait leftover after last-month days+n_types+has_any+OGTG+size+a_n_tx {_f(rec['rank'])} "
        f"ρ={_f(rec['rho_ctrl'])} folds {rec['rank_folds']}; Y2 {_f(rec_y2['rank'])}."
    )
    print(prose)
    return {"left": rec["rank"], "y2": rec_y2["rank"], "folds": rec["rank_folds"], "prose": prose}


def pass_perm_penta(tr: pd.DataFrame, n_perm: int = 16) -> dict:
    rng = np.random.default_rng(FOLD_SEED + 41)
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    dummy_map = pd.Series(_last_any(last).to_numpy(dtype=float), index=last["company_id"])
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_o = last.set_index("company_id")["f_outstanding_gt_granted"]
    last_s = last.set_index("company_id")["log_in3"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    obs = leftover_diag(
        tr[Y3],
        tr["company_id"].map(last_u),
        [
            tr["company_id"].map(last_d),
            tr["company_id"].map(last_t),
            tr["company_id"].map(dummy_map),
            tr["company_id"].map(last_o),
            tr["company_id"].map(last_s),
        ],
        tr["fold"],
        tr[Y3].notna(),
    )["rank"]
    ids = last["company_id"].to_numpy()
    vals_u = last_u.reindex(ids).to_numpy(dtype=float)
    nulls = []
    for _ in range(n_perm):
        shuf = pd.Series(rng.permutation(vals_u), index=ids)
        rec = leftover_diag(
            tr[Y3],
            tr["company_id"].map(shuf),
            [
                tr["company_id"].map(last_d),
                tr["company_id"].map(last_t),
                tr["company_id"].map(dummy_map),
                tr["company_id"].map(last_o),
                tr["company_id"].map(last_s),
            ],
            tr["fold"],
            tr[Y3].notna(),
        )
        if np.isfinite(rec["rank"]):
            nulls.append(rec["rank"])
    arr = np.array(nulls, dtype=float)
    p_ge = float(np.mean(arr >= obs)) if arr.size and np.isfinite(obs) else float("nan")
    prose = (
        f"Permute last-month util leftover-after-penta null p50 {_f(float(np.median(arr)) if arr.size else float('nan'))} "
        f"p(obs≥null)={_f(p_ge, 3)} obs={_f(obs)}."
    )
    print(prose)
    return {
        "p50": float(np.median(arr)) if arr.size else float("nan"),
        "p_ge": p_ge,
        "obs": obs,
        "prose": prose,
    }


def pass_ogtg_park(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_o = last.set_index("company_id")["f_outstanding_gt_granted"]
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    ids_o = set(last.loc[pd.to_numeric(last["f_outstanding_gt_granted"], errors="coerce").notna(), "company_id"])
    ids_u = set(last.loc[pd.to_numeric(last["f_util_snapshot"], errors="coerce").notna(), "company_id"])
    rec_o = leftover_diag(
        tr[Y3],
        tr["company_id"].map(last_o),
        [tr["company_id"].map(last_d), tr["company_id"].map(last_t)],
        tr["fold"],
        tr[Y3].notna() & tr["company_id"].isin(ids_o),
    )
    rec_u = leftover_diag(
        tr[Y3],
        tr["company_id"].map(last_u),
        [tr["company_id"].map(last_d), tr["company_id"].map(last_t)],
        tr["fold"],
        tr[Y3].notna() & tr["company_id"].isin(ids_o),
    )
    rec_ou = leftover_diag(
        tr[Y3],
        tr["company_id"].map(last_o),
        [tr["company_id"].map(last_d), tr["company_id"].map(last_t)],
        tr["fold"],
        tr[Y3].notna() & tr["company_id"].isin(ids_u),
    )
    prose = (
        f"OGTG PARK recipe leftover after last-month days+n_types on OGTG-defined "
        f"{_f(rec_o['rank'])} n_pos={rec_o['n_pos']} ρ={_f(rec_o['rho_ctrl'])} fake={rec_o['fake']} "
        f"(quote 0.540); util leftover on same book {_f(rec_u['rank'])}; "
        f"OGTG leftover on 334 util-defined {_f(rec_ou['rank'])}."
    )
    print(prose)
    return {
        "ogtg": rec_o["rank"],
        "util_on_ogtg": rec_u["rank"],
        "ogtg_on_util": rec_ou["rank"],
        "prose": prose,
    }


def pass_util_ogtg_book(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    dummy_map = pd.Series(_last_any(last).to_numpy(dtype=float), index=last["company_id"])
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_o = last.set_index("company_id")["f_outstanding_gt_granted"]
    last_s = last.set_index("company_id")["log_in3"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    ids_o = set(last.loc[pd.to_numeric(last["f_outstanding_gt_granted"], errors="coerce").notna(), "company_id"])
    ever = tr["company_id"].map(last_u)
    rec = leftover_diag(
        tr[Y3],
        ever,
        [
            tr["company_id"].map(last_d),
            tr["company_id"].map(last_t),
            tr["company_id"].map(dummy_map),
            tr["company_id"].map(last_o),
            tr["company_id"].map(last_s),
        ],
        tr["fold"],
        tr[Y3].notna() & tr["company_id"].isin(ids_o),
    )
    rec_y2 = leftover_diag(
        tr[Y2],
        ever,
        [tr["company_id"].map(last_d), tr["company_id"].map(last_t)],
        tr["fold"],
        tr[Y2].notna() & tr["company_id"].isin(ids_o),
    )
    prose = (
        f"Util leftover after last-month days+n_types+has_any+OGTG+size on OGTG-defined book "
        f"{_f(rec['rank'])} folds {rec['rank_folds']} n_pos={rec['n_pos']}; "
        f"Y2 leftover after last-month days+n_types on that book {_f(rec_y2['rank'])}."
    )
    print(prose)
    return {"left": rec["rank"], "folds": rec["rank_folds"], "y2": rec_y2["rank"], "prose": prose}


def pass_inv_stack(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_o = last.set_index("company_id")["f_outstanding_gt_granted"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    rec_d = leftover_diag(
        tr[Y3],
        tr["company_id"].map(last_d),
        [tr["company_id"].map(last_u), tr["company_id"].map(last_o), tr["company_id"].map(last_t)],
        tr["fold"],
        tr[Y3].notna(),
    )
    rec_t = leftover_diag(
        tr[Y3],
        tr["company_id"].map(last_t),
        [tr["company_id"].map(last_u), tr["company_id"].map(last_o)],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"Inverse leftover of last-month days after last-month util+OGTG+n_types {_f(rec_d['rank'])} "
        f"ρ={_f(rec_d['rho_ctrl'])} fake={rec_d['fake']}; "
        f"last-month n_types leftover after last-month util+OGTG {_f(rec_t['rank'])} "
        f"(days 0.711 bar should live)."
    )
    print(prose)
    return {"days": rec_d["rank"], "types": rec_t["rank"], "prose": prose}


def pass_fold_penta_def(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    any_h = (
        pd.to_numeric(last["f_has_loc"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_factoring"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_confirming"], errors="coerce").eq(1)
    )
    dummy_map = pd.Series(any_h.to_numpy(dtype=float), index=last["company_id"])
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_o = last.set_index("company_id")["f_outstanding_gt_granted"]
    last_s = last.set_index("company_id")["log_in3"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    ids = set(last.loc[pd.to_numeric(last["f_util_snapshot"], errors="coerce").notna(), "company_id"])
    ever = tr["company_id"].map(last_u)
    rec = leftover_diag(
        tr[Y3],
        ever,
        [
            tr["company_id"].map(last_d),
            tr["company_id"].map(last_t),
            tr["company_id"].map(dummy_map),
            tr["company_id"].map(last_o),
            tr["company_id"].map(last_s),
        ],
        tr["fold"],
        tr[Y3].notna() & tr["company_id"].isin(ids),
    )
    prose = (
        f"On 334 leftover-after-last-month days+n_types+has_any+OGTG+size rank folds {rec['rank_folds']} "
        f"rank {_f(rec['rank'])} n_pos={rec['n_pos']}."
    )
    print(prose)
    return {"folds": rec["rank_folds"], "rank": rec["rank"], "prose": prose}


def pass_dummy_penta(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    any_h = (
        pd.to_numeric(last["f_has_loc"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_factoring"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_confirming"], errors="coerce").eq(1)
    )
    dummy_map = pd.Series(any_h.to_numpy(dtype=float), index=last["company_id"])
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_o = last.set_index("company_id")["f_outstanding_gt_granted"]
    last_s = last.set_index("company_id")["log_in3"]
    snap = tr["f_util_snapshot"].notna().astype(float)
    rec = leftover_diag(
        tr[Y3],
        snap,
        [
            tr["company_id"].map(last_d),
            tr["company_id"].map(last_t),
            tr["company_id"].map(dummy_map),
            tr["company_id"].map(last_o),
            tr["company_id"].map(last_s),
        ],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"Last-month dummy leftover after last-month days+n_types+has_any+OGTG+size {_f(rec['rank'])} "
        f"({'dies' if np.isfinite(rec['rank']) and rec['rank'] < CHANCE else 'above 0.55'})."
    )
    print(prose)
    return {"left": rec["rank"], "prose": prose}


def pass_penta_dark(tr: pd.DataFrame, book: set[str]) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    any_h = (
        pd.to_numeric(last["f_has_loc"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_factoring"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_confirming"], errors="coerce").eq(1)
    )
    dummy_map = pd.Series(any_h.to_numpy(dtype=float), index=last["company_id"])
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_o = last.set_index("company_id")["f_outstanding_gt_granted"]
    last_s = last.set_index("company_id")["log_in3"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    ever = tr["company_id"].map(last_u)
    xs = [
        tr["company_id"].map(last_d),
        tr["company_id"].map(last_t),
        tr["company_id"].map(dummy_map),
        tr["company_id"].map(last_o),
        tr["company_id"].map(last_s),
    ]
    erp = tr["company_id"].isin(book)
    rec_i = leftover_diag(tr[Y3], ever, xs, tr["fold"], tr[Y3].notna() & erp)
    rec_d = leftover_diag(tr[Y3], ever, xs, tr["fold"], tr[Y3].notna() & ~erp)
    prose = (
        f"Leftover after last-month days+n_types+has_any+OGTG+size invoiced {_f(rec_i['rank'])} "
        f"dark {_f(rec_d['rank'])} (access ≠ ERP)."
    )
    print(prose)
    return {"inv": rec_i["rank"], "dark": rec_d["rank"], "prose": prose}


def pass_penta_pos(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    any_h = (
        pd.to_numeric(last["f_has_loc"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_factoring"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_confirming"], errors="coerce").eq(1)
    )
    dummy_map = pd.Series(any_h.to_numpy(dtype=float), index=last["company_id"])
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_o = last.set_index("company_id")["f_outstanding_gt_granted"]
    last_s = last.set_index("company_id")["log_in3"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    ids = set(last.loc[pd.to_numeric(last["f_util_snapshot"], errors="coerce").gt(0), "company_id"])
    ever = tr["company_id"].map(last_u)
    rec = leftover_diag(
        tr[Y3],
        ever,
        [
            tr["company_id"].map(last_d),
            tr["company_id"].map(last_t),
            tr["company_id"].map(dummy_map),
            tr["company_id"].map(last_o),
            tr["company_id"].map(last_s),
        ],
        tr["fold"],
        tr[Y3].notna() & tr["company_id"].isin(ids),
    )
    prose = (
        f"On util>0 leftover after last-month days+n_types+has_any+OGTG+size {_f(rec['rank'])} "
        f"n_pos={rec['n_pos']}."
    )
    print(prose)
    return {"left": rec["rank"], "n_pos": rec["n_pos"], "prose": prose}


def pass_penta_defined(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    any_h = (
        pd.to_numeric(last["f_has_loc"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_factoring"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_confirming"], errors="coerce").eq(1)
    )
    dummy_map = pd.Series(any_h.to_numpy(dtype=float), index=last["company_id"])
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_o = last.set_index("company_id")["f_outstanding_gt_granted"]
    last_s = last.set_index("company_id")["log_in3"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    ids = set(last.loc[pd.to_numeric(last["f_util_snapshot"], errors="coerce").notna(), "company_id"])
    ever = tr["company_id"].map(last_u)
    rec = leftover_diag(
        tr[Y3],
        ever,
        [
            tr["company_id"].map(last_d),
            tr["company_id"].map(last_t),
            tr["company_id"].map(dummy_map),
            tr["company_id"].map(last_o),
            tr["company_id"].map(last_s),
        ],
        tr["fold"],
        tr[Y3].notna() & tr["company_id"].isin(ids),
    )
    prose = (
        f"On 334 leftover after last-month days+n_types+has_any+OGTG+size {_f(rec['rank'])} "
        f"n_pos={rec['n_pos']}."
    )
    print(prose)
    return {"left": rec["rank"], "n_pos": rec["n_pos"], "prose": prose}


def pass_y2_penta(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    any_h = (
        pd.to_numeric(last["f_has_loc"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_factoring"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_confirming"], errors="coerce").eq(1)
    )
    dummy_map = pd.Series(any_h.to_numpy(dtype=float), index=last["company_id"])
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_o = last.set_index("company_id")["f_outstanding_gt_granted"]
    last_s = last.set_index("company_id")["log_in3"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    ever = tr["company_id"].map(last_u)
    rec = leftover_diag(
        tr[Y2],
        ever,
        [
            tr["company_id"].map(last_d),
            tr["company_id"].map(last_t),
            tr["company_id"].map(dummy_map),
            tr["company_id"].map(last_o),
            tr["company_id"].map(last_s),
        ],
        tr["fold"],
        tr[Y2].notna(),
    )
    prose = (
        f"Y2 leftover after last-month days+n_types+has_any+OGTG+size {_f(rec['rank'])} "
        f"ρ={_f(rec['rho_ctrl'])}."
    )
    print(prose)
    return {"left": rec["rank"], "prose": prose}


def pass_fold_penta(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    any_h = (
        pd.to_numeric(last["f_has_loc"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_factoring"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_confirming"], errors="coerce").eq(1)
    )
    dummy_map = pd.Series(any_h.to_numpy(dtype=float), index=last["company_id"])
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_o = last.set_index("company_id")["f_outstanding_gt_granted"]
    last_s = last.set_index("company_id")["log_in3"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    ever = tr["company_id"].map(last_u)
    rec = leftover_diag(
        tr[Y3],
        ever,
        [
            tr["company_id"].map(last_d),
            tr["company_id"].map(last_t),
            tr["company_id"].map(dummy_map),
            tr["company_id"].map(last_o),
            tr["company_id"].map(last_s),
        ],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"Trait leftover-after-last-month days+n_types+has_any+OGTG+size rank folds {rec['rank_folds']} "
        f"rank {_f(rec['rank'])}."
    )
    print(prose)
    return {"folds": rec["rank_folds"], "rank": rec["rank"], "prose": prose}


def pass_penta(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    any_h = (
        pd.to_numeric(last["f_has_loc"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_factoring"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_confirming"], errors="coerce").eq(1)
    )
    dummy_map = pd.Series(any_h.to_numpy(dtype=float), index=last["company_id"])
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_o = last.set_index("company_id")["f_outstanding_gt_granted"]
    last_s = last.set_index("company_id")["log_in3"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    ever = tr["company_id"].map(last_u)
    rec = leftover_diag(
        tr[Y3],
        ever,
        [
            tr["company_id"].map(last_d),
            tr["company_id"].map(last_t),
            tr["company_id"].map(dummy_map),
            tr["company_id"].map(last_o),
            tr["company_id"].map(last_s),
        ],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"Trait leftover after last-month days+n_types+has_any+OGTG+size {_f(rec['rank'])} "
        f"ρ={_f(rec['rho_ctrl'])}."
    )
    print(prose)
    return {"left": rec["rank"], "prose": prose}


def pass_fold_any_ogtg(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    any_h = (
        pd.to_numeric(last["f_has_loc"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_factoring"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_confirming"], errors="coerce").eq(1)
    )
    dummy_map = pd.Series(any_h.to_numpy(dtype=float), index=last["company_id"])
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_o = last.set_index("company_id")["f_outstanding_gt_granted"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    ever = tr["company_id"].map(last_u)
    rec = leftover_diag(
        tr[Y3],
        ever,
        [
            tr["company_id"].map(last_d),
            tr["company_id"].map(last_t),
            tr["company_id"].map(dummy_map),
            tr["company_id"].map(last_o),
        ],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"Trait leftover-after-last-month days+n_types+has_any+OGTG rank folds {rec['rank_folds']} "
        f"rank {_f(rec['rank'])}."
    )
    print(prose)
    return {"folds": rec["rank_folds"], "rank": rec["rank"], "prose": prose}


def pass_y2_any(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    any_h = (
        pd.to_numeric(last["f_has_loc"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_factoring"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_confirming"], errors="coerce").eq(1)
    )
    dummy_map = pd.Series(any_h.to_numpy(dtype=float), index=last["company_id"])
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    ever = tr["company_id"].map(last_u)
    rec = leftover_diag(
        tr[Y2],
        ever,
        [tr["company_id"].map(last_d), tr["company_id"].map(last_t), tr["company_id"].map(dummy_map)],
        tr["fold"],
        tr[Y2].notna(),
    )
    prose = (
        f"Y2 leftover after last-month days+n_types+has_any {_f(rec['rank'])} "
        f"ρ={_f(rec['rho_ctrl'])}."
    )
    print(prose)
    return {"left": rec["rank"], "prose": prose}


def pass_any_ogtg(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    any_h = (
        pd.to_numeric(last["f_has_loc"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_factoring"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_confirming"], errors="coerce").eq(1)
    )
    dummy_map = pd.Series(any_h.to_numpy(dtype=float), index=last["company_id"])
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_o = last.set_index("company_id")["f_outstanding_gt_granted"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    ever = tr["company_id"].map(last_u)
    rec = leftover_diag(
        tr[Y3],
        ever,
        [
            tr["company_id"].map(last_d),
            tr["company_id"].map(last_t),
            tr["company_id"].map(dummy_map),
            tr["company_id"].map(last_o),
        ],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"Trait leftover after last-month days+n_types+has_any+OGTG {_f(rec['rank'])} "
        f"ρ={_f(rec['rho_ctrl'])}."
    )
    print(prose)
    return {"left": rec["rank"], "prose": prose}


def pass_any_size(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    any_h = (
        pd.to_numeric(last["f_has_loc"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_factoring"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_confirming"], errors="coerce").eq(1)
    )
    dummy_map = pd.Series(any_h.to_numpy(dtype=float), index=last["company_id"])
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_s = last.set_index("company_id")["log_in3"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    ever = tr["company_id"].map(last_u)
    rec = leftover_diag(
        tr[Y3],
        ever,
        [
            tr["company_id"].map(last_d),
            tr["company_id"].map(last_t),
            tr["company_id"].map(dummy_map),
            tr["company_id"].map(last_s),
        ],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"Trait leftover after last-month days+n_types+has_any+size {_f(rec['rank'])} "
        f"ρ={_f(rec['rho_ctrl'])}."
    )
    print(prose)
    return {"left": rec["rank"], "prose": prose}


def pass_fold_pos(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_util_snapshot"].last()
    ids = set(last.index[last.gt(0)])
    last_row = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_d = last_row.set_index("company_id")["c_n_days_with_tx"]
    last_t = last_row.set_index("company_id")["f_n_types"]
    ever = tr["company_id"].map(last)
    m = tr["company_id"].isin(ids)
    rec = leftover_diag(
        tr[Y3],
        ever,
        [tr["company_id"].map(last_d), tr["company_id"].map(last_t)],
        tr["fold"],
        tr[Y3].notna() & m,
    )
    prose = (
        f"On util>0 leftover-after-last-month days+n_types rank folds {rec['rank_folds']} "
        f"rank {_f(rec['rank'])} n_pos={rec['n_pos']}."
    )
    print(prose)
    return {"folds": rec["rank_folds"], "rank": rec["rank"], "prose": prose}


def pass_fold_has_any(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    any_h = (
        pd.to_numeric(last["f_has_loc"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_factoring"], errors="coerce").eq(1)
        | pd.to_numeric(last["f_has_confirming"], errors="coerce").eq(1)
    )
    dummy_map = pd.Series(any_h.to_numpy(dtype=float), index=last["company_id"])
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    ever = tr["company_id"].map(last_u)
    rec = leftover_diag(
        tr[Y3],
        ever,
        [tr["company_id"].map(last_d), tr["company_id"].map(last_t), tr["company_id"].map(dummy_map)],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"Trait leftover-after-last-month days+n_types+has_any rank folds {rec['rank_folds']} "
        f"rank {_f(rec['rank'])}."
    )
    print(prose)
    return {"folds": rec["rank_folds"], "rank": rec["rank"], "prose": prose}


def pass_pos_only(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_util_snapshot"].last()
    ids = set(last.index[last.gt(0)])
    last_row = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_d = last_row.set_index("company_id")["c_n_days_with_tx"]
    last_t = last_row.set_index("company_id")["f_n_types"]
    ever = tr["company_id"].map(last)
    m = tr["company_id"].isin(ids)
    rec = leftover_diag(
        tr[Y3],
        ever,
        [tr["company_id"].map(last_d), tr["company_id"].map(last_t)],
        tr["fold"],
        tr[Y3].notna() & m,
    )
    prose = (
        f"On util>0 companies leftover after last-month days+n_types {_f(rec['rank'])} "
        f"n_pos={rec['n_pos']}."
    )
    print(prose)
    return {"left": rec["rank"], "n_pos": rec["n_pos"], "prose": prose}


def pass_zero_vs_pos(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id")["f_util_snapshot"].last()
    pos = last.gt(0)
    dummy = tr["company_id"].map(pos.astype(float))
    rec = leftover_diag(tr[Y3], dummy, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    n_pos = int(pos.fillna(False).sum())
    n_z = int(last.eq(0).sum())
    prose = (
        f"Last-month util>0 companies {n_pos} zeros {n_z}. "
        f"Positive-util dummy leftover after days {_f(rec['rank'])} "
        f"ρ={_f(rec['rho_ctrl'])} fake={rec['fake']}."
    )
    print(prose)
    return {"n_pos": n_pos, "n_z": n_z, "left": rec["rank"], "prose": prose}


def pass_conf_left(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_c = last.set_index("company_id")["f_has_confirming"]
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    ever = tr["company_id"].map(last_u)
    rec = leftover_diag(
        tr[Y3],
        ever,
        [tr["company_id"].map(last_d), tr["company_id"].map(last_t), tr["company_id"].map(last_c)],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"Trait leftover after last-month days+n_types+has_confirming {_f(rec['rank'])}."
    )
    print(prose)
    return {"left": rec["rank"], "prose": prose}


def pass_fold_quad(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_l = last.set_index("company_id")["f_has_loc"]
    last_s = last.set_index("company_id")["log_in3"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    ever = tr["company_id"].map(last_u)
    rec = leftover_diag(
        tr[Y3],
        ever,
        [
            tr["company_id"].map(last_d),
            tr["company_id"].map(last_t),
            tr["company_id"].map(last_l),
            tr["company_id"].map(last_s),
        ],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"Trait leftover-after-last-month days+n_types+has_loc+size rank folds {rec['rank_folds']} "
        f"rank {_f(rec['rank'])}."
    )
    print(prose)
    return {"folds": rec["rank_folds"], "rank": rec["rank"], "prose": prose}


def pass_quad(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_l = last.set_index("company_id")["f_has_loc"]
    last_s = last.set_index("company_id")["log_in3"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    ever = tr["company_id"].map(last_u)
    rec = leftover_diag(
        tr[Y3],
        ever,
        [
            tr["company_id"].map(last_d),
            tr["company_id"].map(last_t),
            tr["company_id"].map(last_l),
            tr["company_id"].map(last_s),
        ],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"Trait leftover after last-month days+n_types+has_loc+size {_f(rec['rank'])} "
        f"ρ={_f(rec['rho_ctrl'])}."
    )
    print(prose)
    return {"left": rec["rank"], "prose": prose}


def pass_on_def_left(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    ids = set(last.loc[pd.to_numeric(last["f_util_snapshot"], errors="coerce").notna(), "company_id"])
    ever = tr["company_id"].map(last_u)
    m = tr["company_id"].isin(ids)
    rec = leftover_diag(
        tr[Y3],
        ever,
        [tr["company_id"].map(last_d), tr["company_id"].map(last_t)],
        tr["fold"],
        tr[Y3].notna() & m,
    )
    prose = (
        f"On 334 util-defined companies leftover after last-month days+n_types {_f(rec['rank'])} "
        f"ρ={_f(rec['rho_ctrl'])} n_pos={rec['n_pos']}."
    )
    print(prose)
    return {"left": rec["rank"], "n_pos": rec["n_pos"], "prose": prose}


def pass_fac_dummy(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    fac = set(last.loc[pd.to_numeric(last["f_n_facilities"], errors="coerce").gt(0), "company_id"])
    dummy = tr["company_id"].isin(fac).astype(float)
    last_u = last.set_index("company_id")["f_util_snapshot"]
    ever = tr["company_id"].map(last_u)
    rec_f = leftover_diag(tr[Y3], dummy, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    rec_o = leftover_diag(tr[Y3], ever, [dummy], tr["fold"], tr[Y3].notna())
    rec_s = leftover_diag(tr[Y3], ever, [tr["c_n_days_with_tx"], dummy], tr["fold"], tr[Y3].notna())
    prose = (
        f"Last-month fac>0 dummy leftover after days {_f(rec_f['rank'])} fake={rec_f['fake']}. "
        f"Util-trait leftover after fac-dummy {_f(rec_o['rank'])}; after days+fac-dummy {_f(rec_s['rank'])}."
    )
    print(prose)
    return {"fac": rec_f["rank"], "after": rec_o["rank"], "stack": rec_s["rank"], "prose": prose}


def pass_dummy_stack(tr: pd.DataFrame) -> dict:
    snap = tr["f_util_snapshot"].notna().astype(float)
    rec = leftover_diag(
        tr[Y3], snap, [tr["c_n_days_with_tx"], tr["f_n_types"]], tr["fold"], tr[Y3].notna()
    )
    rec_t = leftover_diag(
        tr[Y3],
        snap,
        [tr["c_n_days_with_tx"], tr["f_n_types"], tr["log_in3"]],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"Last-month dummy leftover after days+n_types {_f(rec['rank'])} "
        f"ρ={_f(rec['rho_ctrl'])}; after days+n_types+size {_f(rec_t['rank'])}."
    )
    print(prose)
    return {"stack": rec["rank"], "triple": rec_t["rank"], "prose": prose}


def pass_boot_last(tr: pd.DataFrame, n_boot: int = 32) -> dict:
    rng = np.random.default_rng(FOLD_SEED + 13)
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_d = last.set_index("company_id")["c_n_days_with_tx"]
    last_t = last.set_index("company_id")["f_n_types"]
    last_u = last.set_index("company_id")["f_util_snapshot"]
    cos = tr["company_id"].drop_duplicates().to_numpy()
    idx = {c: tr.index[tr["company_id"] == c].to_numpy() for c in cos}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(cos, size=len(cos), replace=True)
        rows = np.concatenate([idx[c] for c in pick])
        sub = tr.loc[rows].reset_index(drop=True)
        ev = sub["company_id"].map(last_u)
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


def pass_has_overlap(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    ut = pd.to_numeric(last["f_util_snapshot"], errors="coerce").notna()
    rows = []
    for name in ("f_has_loc", "f_has_factoring", "f_has_confirming"):
        h = pd.to_numeric(last[name], errors="coerce").eq(1)
        both = int((ut & h).sum())
        rows.append(
            {
                "flag": name,
                "n": int(h.sum()),
                "util∩": both,
                "share_of_util": _pp(both / int(ut.sum()) if ut.any() else float("nan")),
            }
        )
    prose = (
        f"Util-defined overlap last-month: loc {rows[0]['util∩']}/{int(ut.sum())} "
        f"fact {rows[1]['util∩']} conf {rows[2]['util∩']}."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def pass_last_rate(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    size = pd.to_numeric(last["log_in3"], errors="coerce")
    terc = pd.qcut(size, 3, labels=["T1", "T2", "T3"], duplicates="drop")
    ut = pd.to_numeric(last["f_util_snapshot"], errors="coerce")
    rows = []
    for lab in ("T1", "T2", "T3"):
        m = terc.eq(lab)
        n = int(m.sum())
        nd = int(ut[m].notna().sum())
        med = float(ut[m].median()) if nd else float("nan")
        rows.append({"tercile": lab, "n": n, "defined": nd, "rate": _pp(nd / n if n else float("nan")), "median": _f(med)})
    prose = "Last-month util defined by SIZE tercile: " + ", ".join(f"{r['tercile']} {r['rate']} med {r['median']}" for r in rows)
    print(prose)
    return {"rows": rows, "prose": prose}


def pass_q6_days(tr: pd.DataFrame) -> dict:
    rec = leftover_diag(tr[Y3], tr["util0_lag1"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    rec3 = leftover_diag(tr[Y3], tr["util0_lag3"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna()) if "util0_lag3" in tr.columns else {"rank": float("nan")}
    prose = (
        f"Q6 fillna0 lag1 leftover after contemporaneous days {_f(rec['rank'])} "
        f"ρ={_f(rec['rho_ctrl'])} fake={rec['fake']}; lag3 after days {_f(rec3['rank'])} "
        f"(snapshot lag cannot lead)."
    )
    print(prose)
    return {"lag1": rec["rank"], "lag3": rec3["rank"], "prose": prose}


def pass_company_auc(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    ever_y3 = tr.groupby("company_id")[Y3].max()
    last = last.merge(ever_y3.rename("ever_y3").reset_index(), on="company_id", how="left")
    x = pd.to_numeric(last["f_util_snapshot"], errors="coerce")
    y = pd.to_numeric(last["ever_y3"], errors="coerce")
    ok = x.notna() & y.notna()
    auc_p = auroc(y[ok], x[ok]) if int(ok.sum()) else float("nan")
    auc_n = auroc(y[ok], -x[ok]) if int(ok.sum()) else float("nan")
    prose = (
        f"Company-level last-month util vs ever-Y3 AUROC + {_f(auc_p)} − {_f(auc_n)} "
        f"n={int(ok.sum())} n_pos={int((ok & (y == 1)).sum())}."
    )
    print(prose)
    return {"auc_p": auc_p, "auc_n": auc_n, "n": int(ok.sum()), "prose": prose}


def pass_ogtg_overlap(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    ut = pd.to_numeric(last["f_util_snapshot"], errors="coerce")
    og = pd.to_numeric(last["f_outstanding_gt_granted"], errors="coerce")
    both = int((ut.notna() & og.eq(1)).sum())
    utn = int(ut.notna().sum())
    og1 = int(og.eq(1).sum())
    rho = spearman(ut, og)
    prose = (
        f"Last-month util defined {utn} OGTG=1 {og1} both {both}. ρ util~OGTG {_f(rho)} "
        f"(OGTG leftover after days just PARK 0.540)."
    )
    print(prose)
    return {"both": both, "utn": utn, "og1": og1, "rho": rho, "prose": prose}


def pass_y10(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["f_util_snapshot"], errors="coerce")
    n_mo = tr["period"].dt.to_period("M").nunique()
    n_def_mo = tr.loc[x.notna(), "period"].dt.to_period("M").nunique()
    frozen = ROOT / "analysis" / "targets"
    # do not add y10 to FROZEN_ACCEPTED — just confirm the file is not our job
    prose = (
        f"Y10 utilisation impossible CONFIRM: {n_def_mo}/{n_mo} months have any util "
        f"(no outstanding/granted history; snapshot last-month only). "
        f"Do not add y10 to FROZEN_ACCEPTED. Do not invent a utilisation Y."
    )
    print(prose)
    return {"n_def_mo": int(n_def_mo), "n_mo": int(n_mo), "impossible": n_def_mo <= 1, "prose": prose}


def decide(p1, p2, p3, p4, p5, p7, p10) -> dict:
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
    elif p1["last_only"] or p3["native_lp"] or p10["impossible"]:
        card = "PARK as snapshot / DROP from the 44 as Y3 X"
        tag = "PARK"
        leftover_tag = "CLOSE"
        why = (
            f"Last-month extract (cov {_pp(p1['cov'])}; Y3 n_pos on snapshot={p3['n_pos']}). "
            f"Snapshot cannot lead (Q6 CLOSE). Utilisation impossible as a Y (Y10 PARK). "
            f"Hole leftover after days {_f(p4['hole_rank'])}. Do not put on the 15-col card."
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
            f"Last-month-only={p1['last_only']} cov {_pp(p1['cov'])} vs OGTG {_pp(p1['og_cov'])}. "
            f"Y3 native {_f(p3['y3'])} vs days {_f(p3['days'])} vs size {_f(p3['size'])} vs n_types {_f(p3['types'])}. "
            f"Twin={p2['twins'] or 'none'} ρ days {_f(p2['rho_days'])} types {_f(p2['rho_types'])} OGTG {_f(p2['rho_ogtg'])}. "
            f"Rise/extract {p5['extract']} n={p5['n_def']}. Q6 {p7['q6']}. Y10 impossible. {card}. "
            "Night Y3 0.762/0.752, days 0.711, size 0.617, TURNOVER 0.720/0.712 unchanged. "
            "Do not put util on the 15-col card."
        ),
    }


def make_png(tr: pd.DataFrame) -> bool:
    if not HAS_MPL:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    by = tr.groupby(tr["period"].dt.to_period("M"))["f_util_snapshot"].apply(lambda s: s.notna().mean())
    ax.bar(range(len(by)), 100.0 * by.to_numpy(), color="#1f4e79")
    ax.set_xticks(range(len(by)))
    ax.set_xticklabels([str(p) for p in by.index], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("% defined")
    ax.set_title("Util coverage by month (extract hole)")
    ax2 = axes[1]
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    x = pd.to_numeric(last["f_util_snapshot"], errors="coerce")
    xx = x.dropna()
    if len(xx):
        ax2.hist(xx, bins=20, color="#1f4e79", edgecolor="white")
    ax2.set_title(f"Last-month util (n={int(x.notna().sum())})")
    ax2.set_xlabel("f_util_snapshot")
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
        "# Unused leftover of `f_util_snapshot` after `c_n_days_with_tx`",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. Do not invent a Y. Do not put util on the 15-col card. "
        "Do not overwrite `debt_schedule_qa` / `ogtg_qa.*` / `factoring_qa.*` / `n_types_qa.*` / "
        "`ar_open_qa.*` / `ap_open_qa.*`. Y3 never B. "
        f"Night Y3 **{Y3_NIGHT[0]:.3f} / {Y3_NIGHT[1]:.3f}**. Days **{DAYS_BENCH}**. "
        f"Size **{SIZE_Y3_QUOTE}**. Y7 TURNOVER **{Y7_TURNOVER[0]:.3f} / {Y7_TURNOVER[1]:.3f}**.",
        "",
        "`f_util_snapshot` = sum|outstanding| / sum|granted| over as-of facilities. "
        "NaN until the 2026-08 extract; NaN on last-month companies with no facilities. "
        "NORTH_STAR PARK snapshot cols as X. Utilisation is impossible as a Y (Y10 PARK). "
        "`f_outstanding_gt_granted` just PARK leftover 0.540.",
        "",
        "## Headline",
        "",
        d["headline"],
        "",
        "## KEEP / CLOSE / DROP / PARK",
        "",
        _md_table(
            [
                {
                    "object": "util leftover after days",
                    "decision": f"**{d['leftover_tag']}**",
                    "why": f"native n_pos={p3['n_pos']} hole-rank {_f(p4['hole_rank'])} OLS {_f(p4['hole_ols'])} fake={p4['hole_fake']}",
                },
                {"object": "as Y3 X (not on 15-col card)", "decision": f"**{d['card']}**", "why": d["why"]},
                {
                    "object": "Last-month-only / snapshot",
                    "decision": "**YES**" if p1["last_only"] else "**no**",
                    "why": p1["prose"],
                },
                {
                    "object": "Twin / SIZE",
                    "decision": f"twin={'YES' if d['twin'] else 'no'} SIZE={'YES' if d['size'] else 'no'}",
                    "why": f"ρ days {_f(p2['rho_days'])} types {_f(p2['rho_types'])} OGTG {_f(p2['rho_ogtg'])} size {_f(p1['rho_in3'])}",
                },
                {"object": "Q6 lag leftover", "decision": f"**{d['q6']}**", "why": p7["prose"]},
                {
                    "object": "Rise / extract hole",
                    "decision": "**extract-hole**" if p5["extract"] else "**path**",
                    "why": p5["prose"],
                },
                {
                    "object": "as health Y / Y10",
                    "decision": "**PARK / impossible**",
                    "why": ctx["p10"]["prose"],
                },
                {
                    "object": "Night quotes",
                    "decision": "**unchanged**",
                    "why": "Y3 0.762/0.752 · days 0.711 · size 0.617 · TURNOVER 0.720/0.712",
                },
            ]
        ),
        "",
        "## 1. Coverage — last-month vs OGTG",
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
        "## 3. Honest leftover after days + inverse + OGTG / n_types",
        "",
        p4["prose"],
        "",
        _md_table(p4["rows"]),
        "",
        "## 4. Rise-only / extract-hole / Y10 impossible",
        "",
        p5["prose"],
        "",
        ctx["p10"]["prose"],
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
        "## 7. Last-month trait",
        "",
        p8["prose"],
        "",
        ctx["p_yl"]["prose"],
        "",
        "## Extra — holdout coverage",
        "",
        ctx["p_ho"]["prose"],
        "",
        "## Extra — OGTG overlap",
        "",
        ctx["p_ov"]["prose"],
        "",
        "## Extra — last-month util by SIZE tercile",
        "",
        ctx["p_lr"]["prose"],
        "",
        _md_table(ctx["p_lr"]["rows"]),
        "",
        "## Extra — trait leftover stacks",
        "",
        ctx["p_ts"]["prose"],
        "",
        ctx["p_tc"]["prose"],
        "",
        ctx["p_ft"]["prose"],
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
            f"- {ctx['p_bt']['prose']}",
            f"- {ctx['p_ov']['prose']}",
            f"- {ctx['p_ts']['prose']}",
            f"- {ctx['p_tc']['prose']}",
            f"- {ctx['p_dt']['prose']}",
            f"- {ctx['p_pm']['prose']}",
            f"- {ctx['p_ft']['prose']}",
            f"- {ctx['p_ca']['prose']}",
            f"- {ctx['p_ld']['prose']}",
            f"- {ctx['p_y2']['prose']}",
            f"- {ctx['p_it']['prose']}",
            f"- {ctx['p_te']['prose']}",
            f"- {ctx['p_qd']['prose']}",
            f"- {ctx['p_ll']['prose']}",
            f"- {ctx['p_fld']['prose']}",
            f"- {ctx['p_bw']['prose']}",
            f"- {ctx['p_lr']['prose']}",
            f"- {ctx['p_od']['prose']}",
            f"- {ctx['p_hu']['prose']}",
            f"- {ctx['p_lg']['prose']}",
            f"- {ctx['p_ho2']['prose']}",
            f"- {ctx['p_al']['prose']}",
            f"- {ctx['p_dl']['prose']}",
            f"- {ctx['p_fy']['prose']}",
            f"- {ctx['p_bl']['prose']}",
            f"- {ctx['p_ln']['prose']}",
            f"- {ctx['p_ds']['prose']}",
            f"- {ctx['p_t3']['prose']}",
            f"- {ctx['p_fd']['prose']}",
            f"- {ctx['p_ftr']['prose']}",
            f"- {ctx['p_od2']['prose']}",
            f"- {ctx['p_y2l']['prose']}",
            f"- {ctx['p_qd4']['prose']}",
            f"- {ctx['p_fq']['prose']}",
            f"- {ctx['p_d3']['prose']}",
            f"- {ctx['p_cf']['prose']}",
            f"- {ctx['p_fa']['prose']}",
            f"- {ctx['p_zp']['prose']}",
            f"- {ctx['p_ha']['prose']}",
            f"- {ctx['p_po']['prose']}",
            f"- {ctx['p_fha']['prose']}",
            f"- {ctx['p_fpo']['prose']}",
            f"- {ctx['p_as']['prose']}",
            f"- {ctx['p_ao']['prose']}",
            f"- {ctx['p_y2a']['prose']}",
            f"- {ctx['p_fao']['prose']}",
            f"- {ctx['p_pe']['prose']}",
            f"- {ctx['p_fp']['prose']}",
            f"- {ctx['p_y2p']['prose']}",
            f"- {ctx['p_pd']['prose']}",
            f"- {ctx['p_pp']['prose']}",
            f"- {ctx['p_pdk']['prose']}",
            f"- {ctx['p_dpe']['prose']}",
            f"- {ctx['p_fpd']['prose']}",
            f"- {ctx['p_fpp']['prose']}",
            f"- {ctx['p_io']['prose']}",
            f"- {ctx['p_ou']['prose']}",
            f"- {ctx['p_qi']['prose']}",
            f"- {ctx['p_dhl']['prose']}",
            f"- {ctx['p_bp']['prose']}",
            f"- {ctx['p_lp']['prose']}",
            f"- {ctx['p_oid']['prose']}",
            f"- {ctx['p_hx']['prose']}",
            f"- {ctx['p_pp5']['prose']}",
            f"- {ctx['p_opk']['prose']}",
            f"- {ctx['p_uob']['prose']}",
            f"- {ctx['p_is']['prose']}",
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
            "Did **not**: overwrite `debt_schedule_qa` / `ogtg_qa.*` / `factoring_qa.*` / "
            "`n_types_qa.*` / `ar_open_qa.*` / `ap_open_qa.*`, edit `debt.py` / `gbm_core.py`, "
            "put util on the 15-col card, grow TURNOVER, invent a Y, add y10 to FROZEN_ACCEPTED, "
            "write 0–100, fit holdout, touch `product/`.",
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
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_f_util_native",
            "value": p3["y3"],
            "coverage": cov,
            "notes": f"n_pos={p3['n_pos']} lp={p3['native_lp']} card={d['headline_tag']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_f_util_hole_resid_days",
            "value": p4["hole_rank"],
            "coverage": cov,
            "notes": f"ols={p4['hole_ols']} fake={p4['hole_fake']} dies={p4['y3_dies']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_f_util_dummy_resid_days",
            "value": p4["dummy_rank"],
            "coverage": cov,
            "notes": f"fake={p4['dummy_fake']} types={p4['dummy_types']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_f_util_lag1_resid_days_lag1",
            "value": p7["l1_left"],
            "coverage": cov,
            "notes": f"q6={p7['q6']} days_l1={p7['days_l1']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_f_util_trait_resid_days_boot_p50",
            "value": ctx["p_bt"]["p50"],
            "coverage": cov,
            "notes": f"p05={ctx['p_bt']['p05']:.4f} p95={ctx['p_bt']['p95']:.4f} fake_trait={ctx['p8']['fake']}",
        },
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
        f"# Wave 4 — f_util_snapshot leftover after days\n\n"
        f"Agent `{AGENT}`. Train group-fold seed 20260918. Holdout 72 coverage only.\n\n"
        f"## Files written\n\n"
        f"- `analysis/evaluate/util_snap_qa.py`\n"
        f"- `analysis/outputs/util_snap_qa.md`\n"
        f"- `analysis/outputs/util_snap_qa.png`\n"
        f"- append-only `analysis/experiments/registry.csv`\n"
        f"- this note\n\n"
        f"Did not overwrite `debt_schedule_qa`, `ogtg_qa.*`, `factoring_qa.*`, `n_types_qa.*`, "
        f"`ar_open_qa.*`, `ap_open_qa.*`. Did not touch `debt.py`, `gbm_core.py`, the 15-col card, "
        f"TURNOVER, product/, FROZEN_ACCEPTED, or the parent journal. "
        f"Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. TURNOVER **0.720 / 0.712**.\n\n"
        f"## Locked verdict\n\n"
        f"| object | decision |\n"
        f"| --- | --- |\n"
        f"| leftover after days | **{d['leftover_tag']}** hole {_f(p4['hole_rank'])} native n_pos={p3['n_pos']} |\n"
        f"| as Y3 X | **{d['card']}** |\n"
        f"| last-month-only | **{p1['last_only']}** cov {_pp(p1['cov'])} vs OGTG {_pp(p1['og_cov'])} |\n"
        f"| leftover after n_types | **{_f(p4['types_left'])}** |\n"
        f"| leftover after OGTG | **{_f(p4['ogtg_left'])}** |\n"
        f"| days leftover after hole | **{_f(p4['inv_rank'])}** |\n"
        f"| extract-hole / Y10 impossible | **{p5['extract']}** / **{p5['y10_impossible']}** |\n"
        f"| Q6 | **{d['q6']}** |\n\n"
        f"Y3 native {_f(p3['y3'])} hole-fillna0 {_f(p3['hole'])} dummy {_f(p3['dummy'])} "
        f"vs days {_f(p3['days'])} vs size {_f(p3['size'])} vs n_types {_f(p3['types'])}. "
        f"ρ days {_f(p2['rho_days'])} types {_f(p2['rho_types'])} OGTG {_f(p2['rho_ogtg'])}. "
        f"lag1 leftover after days_lag1 {_f(p7['l1_left'])}. "
        f"Trait leftover after days {_f(ctx['p8']['left'])} fake={ctx['p8']['fake']}. "
        f"Bootstrap trait p05/p50/p95 {_f(ctx['p_bt']['p05'])} / {_f(ctx['p_bt']['p50'])} / {_f(ctx['p_bt']['p95'])}. "
        f"Defined-dummy leftover {_f(p4['dummy_rank'])} fake={p4['dummy_fake']}. "
        f"Trait leftover after last-month days+n_types {_f(ctx['p_ld']['stack'])} "
        f"boot p50 {_f(ctx['p_bl']['p50'])}. Logo min {_f(ctx['p_lg']['min'])}. "
        f"Permute p(obs≥null)={_f(ctx['p_pm']['p'], 3)}. "
        f"Last-month dummy leftover after days+n_types {_f(ctx['p_ds']['stack'])} dies. "
        f"Positive-util dummy leftover {_f(ctx['p_zp']['left'])} dies. "
        f"Inverse OGTG after util {_f(ctx['p_io']['ogtg_after_util'])}. "
        f"Q6 inverse days after util0_lag1 {_f(ctx['p_qi']['days'])}. "
        f"Penta boot p05/p50/p95 {_f(ctx['p_bp']['p05'])} / {_f(ctx['p_bp']['p50'])} / {_f(ctx['p_bp']['p95'])}. "
        f"Logo-penta min {_f(ctx['p_lp']['min'])}. "
        f"Hexa leftover {_f(ctx['p_hx']['left'])}. "
        f"Permute-penta p(obs≥null)={_f(ctx['p_pp5']['p_ge'], 3)}. "
        f"OGTG PARK leftover {_f(ctx['p_opk']['ogtg'])}. "
        f"Inverse days after util+OGTG+n_types {_f(ctx['p_is']['days'])}.\n\n"
        f"## What failed / next\n\n"
        + "\n".join(f"- {x}" for x in ctx["failed"])
        + f"\n\nElapsed {ctx['elapsed_s']:.0f}s.\n"
    )
    WAVE_NOTE.parent.mkdir(parents=True, exist_ok=True)
    WAVE_NOTE.write_text(text, encoding="utf-8")
    print(f"wrote {WAVE_NOTE}")


def run() -> dict:
    t0 = time.time()
    print(f"util_snap_qa start seed={FOLD_SEED} holdout=72 agent={AGENT}")
    panel = load_panel()
    con = connect()
    try:
        book = book_invoice_ids(con)
    finally:
        con.close()
    panel = attach_folds(panel)
    panel = add_panel_lags(panel, ["util0", "f_util_snapshot", "c_n_days_with_tx"], (1, 3))
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
    p8 = pass8_trait(tr)
    p10 = pass_y10(tr)
    p_yl = pass_y3_last(tr)
    p_ho = pass_holdout(panel)
    p_bt = pass_boot(tr)
    p_ov = pass_ogtg_overlap(tr)
    p_ts = pass_trait_stack(tr)
    p_tc = pass_trait_after_conn(tr)
    p_dt = pass_dark_trait(tr, book)
    p_pm = pass_perm(tr)
    p_ft = pass_fold_trait(tr)
    p_ca = pass_company_auc(tr)
    p_ld = pass_last_double(tr)
    p_y2 = pass_y2_trait(tr)
    p_it = pass_inv_trait(tr)
    p_te = pass_terciles(tr)
    p_qd = pass_q6_days(tr)
    p_ll = pass_last_loc(tr)
    p_fld = pass_fold_last_double(tr)
    p_bw = pass_between(tr)
    p_lr = pass_last_rate(tr)
    p_od = pass_on_defined(tr)
    p_hu = pass_high_util(tr)
    p_lg = pass_logo(tr)
    p_ho2 = pass_has_overlap(tr)
    p_al = pass_after_lags(tr)
    p_dl = pass_dark_last(tr, book)
    p_fy = pass_first_year(tr)
    p_bl = pass_boot_last(tr)
    p_ln = pass_last_ntx(tr)
    p_ds = pass_dummy_stack(tr)
    p_t3 = pass_t3_last(tr)
    p_fd = pass_fac_dummy(tr)
    p_ftr = pass_fold_triple(tr)
    p_od2 = pass_on_def_left(tr)
    p_y2l = pass_y2_last(tr)
    p_qd4 = pass_quad(tr)
    p_fq = pass_fold_quad(tr)
    p_d3 = pass_dummy_triple(tr)
    p_cf = pass_conf_left(tr)
    p_fa = pass_fact_left(tr)
    p_zp = pass_zero_vs_pos(tr)
    p_ha = pass_has_any_left(tr)
    p_po = pass_pos_only(tr)
    p_fha = pass_fold_has_any(tr)
    p_fpo = pass_fold_pos(tr)
    p_as = pass_any_size(tr)
    p_ao = pass_any_ogtg(tr)
    p_y2a = pass_y2_any(tr)
    p_fao = pass_fold_any_ogtg(tr)
    p_pe = pass_penta(tr)
    p_fp = pass_fold_penta(tr)
    p_y2p = pass_y2_penta(tr)
    p_pd = pass_penta_defined(tr)
    p_pp = pass_penta_pos(tr)
    p_pdk = pass_penta_dark(tr, book)
    p_dpe = pass_dummy_penta(tr)
    p_fpd = pass_fold_penta_def(tr)
    p_fpp = pass_fold_penta_pos(tr)
    p_io = pass_inv_ogtg(tr)
    p_ou = pass_ogtg_on_util(tr)
    p_qi = pass_q6_inv(tr)
    p_dhl = pass_dark_hole_last(tr, book)
    p_bp = pass_boot_penta(tr)
    p_lp = pass_logo_penta(tr)
    p_oid = pass_ogtg_inv_days(tr)
    p_hx = pass_hexa(tr)
    p_pp5 = pass_perm_penta(tr)
    p_opk = pass_ogtg_park(tr)
    p_uob = pass_util_ogtg_book(tr)
    p_is = pass_inv_stack(tr)
    png = make_png(tr)
    decision = decide(p1, p2, p3, p4, p5, p7, p10)
    failed = []
    if not p3["days_ok"]:
        failed.append(f"days replica drifted: {_f(p3['days'])} vs 0.711")
    if not p3["size_ok"]:
        failed.append(f"size replica drifted: {_f(p3['size'])} vs 0.617")
    if not p3["types_ok"]:
        failed.append(f"n_types replica drifted: {_f(p3['types'])} vs 0.578")
    if not p1["cov_ok"]:
        failed.append(f"util cov {_pp(p1['cov'])} off 1.6%")
    if not p1["n_ok"]:
        failed.append(f"util defined {p1['n_def']} off 334")
    if not p1["og_ok"]:
        failed.append(f"OGTG cov {_pp(p1['og_cov'])} off 5.7%")
    if not p6["confirm"]:
        failed.append(f"dark/invoiced {p6['n_dark']}/{p6['n_erp']} off 470/744")
    if not p7["days_l1_ok"]:
        failed.append(f"days lag1 {_f(p7['days_l1'])} off KEEP 0.684")
    if not p5["y10_impossible"]:
        failed.append("Y10 utilisation not confirmed impossible")
    if not failed:
        failed.append(
            "no replica miss; native leftover undefined (Y3 n_pos=0 on 2026-08). "
            "fillna0 / defined-dummy leftover 0.711 is a fake days leak (ρ=-0.959). "
            "Trait leftover 0.622 is not a days leak (ρ=0.044) but folds 0.424–0.760 / boot p05 0.391 dies. "
            "PARK snapshot X. Utilisation impossible as a Y."
        )
    ctx = {
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p4": p4,
        "p5": p5,
        "p6": p6,
        "p7": p7,
        "p8": p8,
        "p10": p10,
        "p_yl": p_yl,
        "p_ho": p_ho,
        "p_bt": p_bt,
        "p_ov": p_ov,
        "p_ts": p_ts,
        "p_tc": p_tc,
        "p_dt": p_dt,
        "p_pm": p_pm,
        "p_ft": p_ft,
        "p_ca": p_ca,
        "p_ld": p_ld,
        "p_y2": p_y2,
        "p_it": p_it,
        "p_te": p_te,
        "p_qd": p_qd,
        "p_ll": p_ll,
        "p_fld": p_fld,
        "p_bw": p_bw,
        "p_lr": p_lr,
        "p_od": p_od,
        "p_hu": p_hu,
        "p_lg": p_lg,
        "p_ho2": p_ho2,
        "p_al": p_al,
        "p_dl": p_dl,
        "p_fy": p_fy,
        "p_bl": p_bl,
        "p_ln": p_ln,
        "p_ds": p_ds,
        "p_t3": p_t3,
        "p_fd": p_fd,
        "p_ftr": p_ftr,
        "p_od2": p_od2,
        "p_y2l": p_y2l,
        "p_qd4": p_qd4,
        "p_fq": p_fq,
        "p_d3": p_d3,
        "p_cf": p_cf,
        "p_fa": p_fa,
        "p_zp": p_zp,
        "p_ha": p_ha,
        "p_po": p_po,
        "p_fha": p_fha,
        "p_fpo": p_fpo,
        "p_as": p_as,
        "p_ao": p_ao,
        "p_y2a": p_y2a,
        "p_fao": p_fao,
        "p_pe": p_pe,
        "p_fp": p_fp,
        "p_y2p": p_y2p,
        "p_pd": p_pd,
        "p_pp": p_pp,
        "p_pdk": p_pdk,
        "p_dpe": p_dpe,
        "p_fpd": p_fpd,
        "p_fpp": p_fpp,
        "p_io": p_io,
        "p_ou": p_ou,
        "p_qi": p_qi,
        "p_dhl": p_dhl,
        "p_bp": p_bp,
        "p_lp": p_lp,
        "p_oid": p_oid,
        "p_hx": p_hx,
        "p_pp5": p_pp5,
        "p_opk": p_opk,
        "p_uob": p_uob,
        "p_is": p_is,
        "decision": decision,
        "failed": failed,
        "png": png,
        "extra_bits": [],
        "elapsed_s": time.time() - t0,
        "tr": tr,
        "panel": panel,
        "book": book,
    }
    write_md(ctx)
    append_registry(ctx)
    if WRITE_WAVE:
        write_wave(ctx)
    print(
        f"DONE card={decision['headline_tag']} leftover={_f(p4['y3_rank'])} "
        f"hole={_f(p4['hole_rank'])} y3={_f(p3['y3'])} elapsed={ctx['elapsed_s']:.0f}s"
    )
    return ctx


if __name__ == "__main__":
    run()
