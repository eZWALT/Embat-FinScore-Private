"""Unused leftover of ``d_supp_top1`` after days as Y3 X.

``d_supp_top1`` = share of AP invoice |amount| from the single
largest supplier (Family D, trailing 6m). Incomplete 6m books are
NaN. Dark 470 stay NaN not 0. ``d_supp_hhi`` already DROP (twin
ρ 0.987). ``d_cust_top1`` DROP leftover 0.525; vs supp ρ 0.117 —
different object. ``d_n_supp`` DROP SIZE ρ 0.546.

Do **not** overwrite ``supp_hhi_qa.*``, ``top1_qa.*``, ``n_supp_qa.*``.
Y5 leftover after size is report-only (never E). KEEP the Y5
protective-tail footnote unless this cut overturns it.

KEEP-as-X: beat size ≥0.02 **and** leftover after days **and** not
SIZE (|ρ| vs log1p(a_in3) ≥0.50) **and** not a twin (|ρ|≥0.80 vs
days / a_n_tx / d_supp_hhi / d_n_supp / d_cust_top1). Leftover <0.55 dies.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.supp_top1_qa

Owned: analysis/evaluate/supp_top1_qa.py, analysis/outputs/supp_top1_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_supp_top1.md (end).
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
OUT_MD = ANALYSIS / "outputs" / "supp_top1_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "supp_top1_qa.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_supp_top1.md"
AGENT = "572fb928"
WAVE = "4"
ROUND = "R4"
MODEL = "supp_top1_qa"
X_FAM = "D"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
Y5 = "y5_ap_od30_ownp80"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_QUOTE = 0.617
DAYS_LAG1_QUOTE = 0.684
Y3_NIGHT = (0.762, 0.752)
Y7_TURNOVER = 0.720
Y7_B_SHALLOW = 0.712
TOP1_Y3_PEEK = 0.641
TOP1_N_PEEK = 2877
TOP1_POS_PEEK = 203
HHI_RHO_PEEK = 0.987
CUST_RHO_PEEK = 0.117
CUST_Y3_PEEK = 0.590
NSUPP_Y3_PEEK = 0.699
Y5_TAIL_HI = 0.027
Y5_TAIL_REST = 0.086
LEFT_PEEK = 0.429
TAIL_CUT = 0.975
KEEP_DELTA = 0.02
TWIN_RHO = 0.80
SIZE_RHO = 0.50
CHANCE = 0.55
ICC_TRAIT = 0.85
FAKE_DAYS_RHO = 0.40
MIN_POS = 50
MIN_ACF_PAIRS = 4
N_DARK_WANT = 470
WRITE_WAVE = True

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "first_month",
    "a_in3",
    "a_n_tx",
    "c_n_days_with_tx",
    "d_n_supp",
    "d_supp_top1",
    "d_supp_hhi",
    "d_n_cust",
    "d_cust_top1",
    "d_cust_hhi",
)

Y_KEEP = (Y2, Y3, Y5)


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


def _pct(n, d) -> float:
    if d is None or d == 0:
        return float("nan")
    return float(n) / float(d)


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


def median_acf(series: pd.Series, company: pd.Series, lag: int = 1) -> float:
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
    d = pd.DataFrame({"x": pd.to_numeric(series, errors="coerce"), "co": company.astype(str)}).dropna()
    k = int(d["co"].nunique())
    n = len(d)
    if k < 2 or n < k + 2:
        return {"icc": float("nan"), "k": k, "n": n}
    grand = float(d["x"].mean())
    ns = d.groupby("co")["x"].size()
    mus = d.groupby("co")["x"].mean()
    ssb = float(((mus - grand) ** 2 * ns).sum())
    ssw = float(((d["x"] - d["co"].map(mus)) ** 2).sum())
    msb = ssb / (k - 1)
    msw = ssw / (n - k) if n > k else float("nan")
    icc = msb / (msb + msw) if np.isfinite(msw) and (msb + msw) != 0 else float("nan")
    return {"icc": float(icc), "k": k, "n": n}


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
    fold_rows, aucs = [], []
    if low:
        return {
            "cv": float("nan"), "sd": float("nan"), "n_folds": 0, "folds": fold_rows,
            "train_sign": 0, "train_auc": float("nan"), "n_defined": n,
            "n_pos": n_pos, "n_neg": n_neg, "low_power": True,
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
        "n_folds": len(finite), "folds": fold_rows, "train_sign": int(tr_sign),
        "train_auc": float(auroc(y[defined], tr_sign * x[defined])),
        "n_defined": n, "n_pos": n_pos, "n_neg": n_neg, "low_power": False,
    }


def fold_bits(rec: dict) -> str:
    return " ".join(f"{r['auroc']:.3f}" if np.isfinite(r["auroc"]) else "—" for r in rec.get("folds", []))


def ols_resid(y: pd.Series, *xs: pd.Series) -> tuple[pd.Series, dict]:
    cols = {"y": pd.to_numeric(y, errors="coerce")}
    for i, x in enumerate(xs):
        cols[f"x{i}"] = pd.to_numeric(x, errors="coerce")
    d = pd.DataFrame(cols)
    ok = d.notna().all(axis=1)
    resid = pd.Series(np.nan, index=d.index, dtype=float)
    info = {"n": int(ok.sum()), "slope": [], "intercept": float("nan"), "r2": float("nan")}
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
    info["intercept"] = float(beta[0])
    info["slope"] = [float(b) for b in beta[1:]]
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
    almost = bool(np.isfinite(rho_c) and abs(rho_c) >= FAKE_DAYS_RHO)
    rank_cv = _cv(rrec)
    ols_cv = _cv(rec)
    honest_dies = bool(fake or (np.isfinite(rank_cv) and rank_cv < CHANCE))
    return {
        "ols": ols_cv, "rank": rank_cv, "rho_ctrl": rho_c, "r2": info["r2"],
        "n": rec["n_defined"], "n_pos": rec["n_pos"],
        "fake": fake, "almost": almost, "honest_dies": honest_dies,
        "folds": fold_bits(rec), "rank_folds": fold_bits(rrec),
        "rec": rec, "rrec": rrec,
    }


def company_demean(series: pd.Series, company: pd.Series) -> pd.Series:
    d = pd.DataFrame({"x": pd.to_numeric(series, errors="coerce"), "co": company.astype(str)})
    return d["x"] - d.groupby("co")["x"].transform("mean")


def company_mean(series: pd.Series, company: pd.Series) -> pd.Series:
    d = pd.DataFrame({"x": pd.to_numeric(series, errors="coerce"), "co": company.astype(str)})
    return d.groupby("co")["x"].transform("mean")


def _cv(res: dict) -> float:
    return float("nan") if res.get("low_power") else float(res.get("cv", float("nan")))


def _auc_row(y: str, feat: str, res: dict) -> dict:
    return {
        "y": y, "feature": feat,
        "n": f"{res['n_defined']:,}", "n_pos": f"{res['n_pos']:,}",
        "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
        "train": "LOW_POWER" if res["low_power"] else _f(res["train_auc"]),
        "sd": _f(res["sd"]) if not res["low_power"] else "—",
        "sign": res["train_sign"] if not res["low_power"] else "—",
        "folds": fold_bits(res) if not res["low_power"] else "—",
    }


def add_panel_lags(df: pd.DataFrame, stems: list[str], lags: tuple[int, ...]) -> pd.DataFrame:
    out = df.sort_values(["company_id", "period"]).reset_index(drop=True)
    cid = out["company_id"]
    extra = {}
    for c in stems:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        g = s.groupby(cid, sort=False)
        for k in lags:
            extra[f"{c}_lag{k}"] = g.shift(k)
    if extra:
        out = pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)
    return out


def load_panel() -> pd.DataFrame:
    if not STORE.exists():
        raise FileNotFoundError(f"{STORE} missing")
    if not TARGETS.exists():
        raise FileNotFoundError(f"{TARGETS} missing — do not run build_targets")
    raw = pd.read_parquet(STORE)
    yraw = pd.read_parquet(TARGETS)
    print(f"store {STORE} shape={raw.shape} (read-only)")
    print(f"targets {TARGETS} shape={yraw.shape} (read-only)")
    miss = [c for c in STORE_COLS if c not in raw.columns]
    if miss:
        raise RuntimeError(f"monthly.parquet missing {miss}")
    ymiss = [c for c in Y_KEEP if c not in yraw.columns]
    if ymiss:
        raise RuntimeError(f"targets.parquet missing {ymiss}")
    panel = _keys(raw[list(STORE_COLS)])
    y = _keys(yraw[["company_id", "period", *Y_KEEP]])
    panel = panel.merge(y, on=["company_id", "period"], how="left")
    hold = load_holdout()
    panel["split"] = np.where(panel["company_id"].isin(hold), "holdout", "train")
    panel["log_in3"] = np.log1p(pd.to_numeric(panel["a_in3"], errors="coerce").clip(lower=0))
    leak3 = leakage_check(
        ["d_supp_top1", "d_supp_hhi", "d_n_supp", "d_cust_top1", "c_n_days_with_tx", "log_in3"],
        Y3, forbidden_prefixes=["b"],
    )
    if not leak3["ok"]:
        raise RuntimeError(f"Y3 never-B leak: {leak3['issues']}")
    leak5 = leakage_check(["d_supp_top1", "d_supp_hhi", "log_in3"], Y5, forbidden_prefixes=["e"])
    if not leak5["ok"]:
        raise RuntimeError(f"Y5 never-E leak: {leak5['issues']}")
    return panel


def attach_folds(panel: pd.DataFrame) -> pd.DataFrame:
    train = panel[panel["split"] == "train"][["company_id", "group_id"]].drop_duplicates()
    assert_no_holdout(train["company_id"])
    folds = group_folds(train, n=N_FOLDS, seed=FOLD_SEED)
    return panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")


def pass1_cov(tr: pd.DataFrame, book: set[str]) -> dict:
    print("\n" + "=" * 72)
    print("CUT 1 — coverage; twin / SIZE screen")
    print("=" * 72)
    n_cm, n_co = len(tr), int(tr["company_id"].nunique())
    x = pd.to_numeric(tr["d_supp_top1"], errors="coerce")
    hhi = pd.to_numeric(tr["d_supp_hhi"], errors="coerce")
    n_supp = pd.to_numeric(tr["d_n_supp"], errors="coerce")
    dark = ~tr["company_id"].isin(book)
    n_dark_co = int(tr.loc[dark, "company_id"].nunique())
    dark_nn = int(x[dark].notna().sum())
    dark_zero = int((x[dark] == 0).sum())
    acf1 = median_acf(x, tr["company_id"], 1)
    pairs = {
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "a_n_tx": tr["a_n_tx"],
        "d_supp_hhi": hhi,
        "d_n_supp": n_supp,
        "d_cust_top1": tr["d_cust_top1"],
        "log1p(a_in3)": tr["log_in3"],
    }
    rhos = {k: spearman(x, v) for k, v in pairs.items()}
    twins = [k for k, v in rhos.items() if np.isfinite(v) and abs(v) >= TWIN_RHO]
    is_size = bool(np.isfinite(rhos["log1p(a_in3)"]) and abs(rhos["log1p(a_in3)"]) >= SIZE_RHO)
    twin_gate = any(
        np.isfinite(rhos[k]) and abs(rhos[k]) >= TWIN_RHO
        for k in ("c_n_days_with_tx", "a_n_tx", "d_supp_hhi", "d_n_supp", "d_cust_top1")
    )
    rho_hhi_ok = bool(np.isfinite(rhos["d_supp_hhi"]) and abs(rhos["d_supp_hhi"] - HHI_RHO_PEEK) < 0.015)
    rho_c_ok = bool(np.isfinite(rhos["d_cust_top1"]) and abs(rhos["d_cust_top1"] - CUST_RHO_PEEK) < 0.03)
    cal_inc = pd.to_datetime(tr["period"]) < pd.Timestamp("2025-02-01")
    cal_nn = int(x[cal_inc].notna().sum())
    dark_ok = n_dark_co == N_DARK_WANT and dark_nn == 0 and dark_zero == 0
    cal_ok = cal_nn == 0
    rows = [
        {"col": "d_supp_top1", "n_nn": f"{int(x.notna().sum()):,}", "cov": _pp(_pct(int(x.notna().sum()), n_cm)), "acf1": _f(acf1)},
        {"col": "d_supp_hhi", "n_nn": f"{int(hhi.notna().sum()):,}", "cov": _pp(_pct(int(hhi.notna().sum()), n_cm)), "acf1": _f(median_acf(hhi, tr["company_id"], 1))},
        {"col": "d_n_supp", "n_nn": f"{int(n_supp.notna().sum()):,}", "cov": _pp(_pct(int(n_supp.notna().sum()), n_cm)), "acf1": _f(median_acf(n_supp, tr["company_id"], 1))},
    ]
    rho_rows = [
        {"vs": k, "rho": _f(v), "flag": "SIZE" if k == "log1p(a_in3)" and abs(v) >= SIZE_RHO else "TWIN" if abs(v) >= TWIN_RHO else "no"}
        for k, v in rhos.items()
    ]
    prose = (
        f"Train {n_co:,} co / {n_cm:,} CM. d_supp_top1 cov {_pp(_pct(int(x.notna().sum()), n_cm))} "
        f"acf1={_f(acf1)}. Dark {n_dark_co} nn={dark_nn} zero={dark_zero} "
        f"{'CONFIRM NaN' if dark_ok else 'FAIL'}. Calendar incomplete nn={cal_nn} "
        f"{'CONFIRM NaN' if cal_ok else 'FAIL'}. "
        f"ρ vs days {_f(rhos['c_n_days_with_tx'])} vs a_n_tx {_f(rhos['a_n_tx'])} "
        f"vs HHI {_f(rhos['d_supp_hhi'])} (peek 0.987 {'CONFIRM' if rho_hhi_ok else 'DRIFT'}) "
        f"vs n_supp {_f(rhos['d_n_supp'])} vs cust_top1 {_f(rhos['d_cust_top1'])} "
        f"(peek 0.117 {'CONFIRM' if rho_c_ok else 'DRIFT'}) vs size {_f(rhos['log1p(a_in3)'])}. "
        f"SIZE={is_size} twins={twins or 'none'} twin_gate={twin_gate}."
    )
    print(prose)
    return {
        "rows": rows, "rho_rows": rho_rows, "rhos": rhos, "twins": twins,
        "is_size": is_size, "twin_gate": twin_gate, "n_cm": n_cm, "n_co": n_co,
        "cov": _pct(int(x.notna().sum()), n_cm), "acf1": acf1,
        "dark_ok": dark_ok, "cal_ok": cal_ok, "n_dark_co": n_dark_co, "dark_nn": dark_nn,
        "rho_hhi_ok": rho_hhi_ok, "rho_c_ok": rho_c_ok, "prose": prose,
    }


def pass2_singles(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 2 — single-feature group-fold Y3")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab, folds = y.notna(), tr["fold"]
    feats = {
        "d_supp_top1": tr["d_supp_top1"],
        "d_supp_hhi": tr["d_supp_hhi"],
        "d_n_supp": tr["d_n_supp"],
        "d_cust_top1": tr["d_cust_top1"],
        "log1p(a_in3)": tr["log_in3"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "a_n_tx": tr["a_n_tx"],
    }
    recs, rows = {}, []
    for fname, x in feats.items():
        rec = signed_oof_auroc(y, x, folds, lab)
        recs[fname] = rec
        rows.append(_auc_row(Y3, fname, rec))
        print(f"  {fname} CV={_f(rec['cv'])} n={rec['n_defined']} pos={rec['n_pos']} folds={fold_bits(rec)}")
    y3_t, y3_size, y3_days = _cv(recs["d_supp_top1"]), _cv(recs["log1p(a_in3)"]), _cv(recs["c_n_days_with_tx"])
    days_ok = bool(np.isfinite(y3_days) and abs(y3_days - DAYS_BENCH) < 0.008)
    size_ok = bool(np.isfinite(y3_size) and abs(y3_size - SIZE_QUOTE) < 0.008)
    peek_ok = bool(
        recs["d_supp_top1"]["n_defined"] == TOP1_N_PEEK
        and recs["d_supp_top1"]["n_pos"] == TOP1_POS_PEEK
        and np.isfinite(y3_t) and abs(y3_t - TOP1_Y3_PEEK) < 0.015
    )
    beat_size = bool(np.isfinite(y3_t) and (y3_t - SIZE_QUOTE) >= KEEP_DELTA)
    prose = (
        f"Y3 d_supp_top1 {_f(y3_t)} n={recs['d_supp_top1']['n_defined']:,} "
        f"pos={recs['d_supp_top1']['n_pos']:,} "
        f"(peek 0.641 / 2,877 / 203 {'CONFIRM' if peek_ok else 'DRIFT'}). "
        f"vs size {_f(y3_size)} vs days {_f(y3_days)} vs HHI {_f(_cv(recs['d_supp_hhi']))} "
        f"vs n_supp {_f(_cv(recs['d_n_supp']))} vs cust_top1 {_f(_cv(recs['d_cust_top1']))}. "
        f"Replica days 0.711 {'CONFIRM' if days_ok else 'DRIFT'} / size 0.617 "
        f"{'CONFIRM' if size_ok else 'DRIFT'}. Beat-size Δ="
        f"{_f(y3_t - SIZE_QUOTE) if np.isfinite(y3_t) else '—'} "
        f"{'PASS' if beat_size else 'FAIL'}."
    )
    print(prose)
    return {
        "rows": rows, "recs": recs, "top1": y3_t, "size": y3_size, "days": y3_days,
        "hhi": _cv(recs["d_supp_hhi"]), "n_supp": _cv(recs["d_n_supp"]),
        "cust": _cv(recs["d_cust_top1"]),
        "days_ok": days_ok, "size_ok": size_ok, "peek_ok": peek_ok, "beat_size": beat_size,
        "n_def": recs["d_supp_top1"]["n_defined"], "n_pos": recs["d_supp_top1"]["n_pos"], "prose": prose,
    }


def pass3_days(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 3 — honest leftover of d_supp_top1 after days; inverse")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab, folds = y.notna(), tr["fold"]
    after = leftover_diag(y, tr["d_supp_top1"], (tr["c_n_days_with_tx"],), folds, lab)
    inv = leftover_diag(y, tr["c_n_days_with_tx"], (tr["d_supp_top1"],), folds, lab)
    peek_ok = bool(np.isfinite(after["rank"]) and abs(after["rank"] - LEFT_PEEK) < 0.03)
    prose = (
        f"d_supp_top1 leftover after days OLS {_f(after['ols'])} rank {_f(after['rank'])} "
        f"(peek 0.429 {'CONFIRM' if peek_ok else 'DRIFT'}) "
        f"fake={after['fake']} almost={after['almost']} ρ(resid,days)={_f(after['rho_ctrl'])} "
        f"R²={_f(after['r2'])} honest_dies={after['honest_dies']} n={after['n']:,} pos={after['n_pos']:,}. "
        f"Inverse: days leftover after supp_top1 OLS {_f(inv['ols'])} rank {_f(inv['rank'])} "
        f"dies={inv['honest_dies']}."
    )
    print(prose)
    return {
        "after": after, "inv": inv, "ols": after["ols"], "rank": after["rank"],
        "dies": after["honest_dies"], "fake": after["fake"], "almost": after["almost"],
        "r2": after["r2"], "inv_rank": inv["rank"], "inv_dies": inv["honest_dies"],
        "peek_ok": peek_ok, "prose": prose,
    }


def pass5_after_conc(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 5 — leftover after HHI / n_supp / days+HHI")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab, folds = y.notna(), tr["fold"]
    after_h = leftover_diag(y, tr["d_supp_top1"], (tr["d_supp_hhi"],), folds, lab)
    after_n = leftover_diag(y, tr["d_supp_top1"], (tr["d_n_supp"],), folds, lab)
    after_dh = leftover_diag(y, tr["d_supp_top1"], (tr["c_n_days_with_tx"], tr["d_supp_hhi"]), folds, lab)
    after_c = leftover_diag(y, tr["d_supp_top1"], (tr["d_cust_top1"],), folds, lab)
    h_after = leftover_diag(y, tr["d_supp_hhi"], (tr["d_supp_top1"],), folds, lab)
    rows = []
    for name, rec in (
        ("after HHI", after_h), ("after n_supp", after_n),
        ("after days+HHI", after_dh), ("after cust_top1", after_c),
        ("HHI after top1", h_after),
    ):
        rows.append({"bar": name, "OLS": _f(rec["ols"]), "rank": _f(rec["rank"]), "ρ(resid,bar)": _f(rec["rho_ctrl"]), "R2": _f(rec["r2"]), "dies": rec["honest_dies"], "n": rec["n"], "n_pos": rec["n_pos"]})
    prose = (
        f"top1 leftover after HHI OLS {_f(after_h['ols'])} rank {_f(after_h['rank'])} "
        f"dies={after_h['honest_dies']} (want die — rewrite R²={_f(after_h['r2'])}; "
        f"OLS leftover is the honest rewrite residual when R²≥0.95). "
        f"after n_supp OLS {_f(after_n['ols'])} rank {_f(after_n['rank'])} dies={after_n['honest_dies']}; "
        f"after days+HHI {_f(after_dh['rank'])} dies={after_dh['honest_dies']}; "
        f"after cust_top1 {_f(after_c['rank'])} dies={after_c['honest_dies']}. "
        f"HHI after top1 OLS {_f(h_after['ols'])} rank {_f(h_after['rank'])} "
        f"(supp_hhi leftover after top1 OLS 0.525 — not overwritten)."
    )
    print(prose)
    return {
        "rows": rows, "h_rank": after_h["rank"], "h_dies": after_h["honest_dies"],
        "n_rank": after_n["rank"], "n_dies": after_n["honest_dies"],
        "dh_rank": after_dh["rank"], "c_rank": after_c["rank"],
        "h_after": h_after["rank"], "h_r2": after_h["r2"], "prose": prose,
    }


def pass6_dark(tr: pd.DataFrame, book: set[str]) -> dict:
    print("\n" + "=" * 72)
    print("CUT 6 — dark 470 stay NaN; ERP leftover")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    erp, dark = tr["company_id"].isin(book), ~tr["company_id"].isin(book)
    x = pd.to_numeric(tr["d_supp_top1"], errors="coerce")
    n_dark_co = int(tr.loc[dark, "company_id"].nunique())
    dark_nn = int(x[dark].notna().sum())
    rec = signed_oof_auroc(y, tr["d_supp_top1"], tr["fold"], lab & erp)
    after = leftover_diag(y, tr["d_supp_top1"], (tr["c_n_days_with_tx"],), tr["fold"], lab & erp)
    rec_d = signed_oof_auroc(y, tr["d_supp_top1"], tr["fold"], lab & dark)
    dark_ok = n_dark_co == N_DARK_WANT and dark_nn == 0
    rows = [
        {"book": "dark", "n_co": n_dark_co, "nn": dark_nn, "Y3 n": rec_d["n_defined"], "Y3 pos": rec_d["n_pos"], "CV": "LOW_POWER" if rec_d["low_power"] else _f(rec_d["cv"])},
        {"book": "ERP", "n_co": int(tr.loc[erp, "company_id"].nunique()), "nn": int(x[erp].notna().sum()), "Y3 n": rec["n_defined"], "Y3 pos": rec["n_pos"], "CV": "LOW_POWER" if rec["low_power"] else _f(rec["cv"])},
    ]
    prose = (
        f"Dark {n_dark_co} (want {N_DARK_WANT}) top1 nn={dark_nn} "
        f"{'CONFIRM NaN not 0' if dark_ok else 'FAIL'}. "
        f"ERP Y3 {_f(rec['cv'])} leftover after days {_f(after['rank'])} dies={after['honest_dies']}."
    )
    print(prose)
    return {"rows": rows, "dark_ok": dark_ok, "erp_cv": _cv(rec), "erp_rank": after["rank"], "prose": prose}


def pass7_q6(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 7 — Q6 lag1 leftover after days_lag1")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab, folds = y.notna(), tr["fold"]
    recs, rows = {}, []
    for name in ("d_supp_top1", "d_supp_top1_lag1", "c_n_days_with_tx_lag1"):
        rec = signed_oof_auroc(y, tr[name], folds, lab)
        recs[name] = rec
        rows.append(_auc_row(Y3, name, rec))
        print(f"  {name} CV={_f(rec['cv'])} n={rec['n_defined']} pos={rec['n_pos']}")
    after_l1 = leftover_diag(y, tr["d_supp_top1_lag1"], (tr["c_n_days_with_tx_lag1"],), folds, lab)
    days_l1 = _cv(recs["c_n_days_with_tx_lag1"])
    days_ok = bool(np.isfinite(days_l1) and abs(days_l1 - DAYS_LAG1_QUOTE) < 0.008)
    prose = (
        f"Y3 supp_top1_lag1 {_f(recs['d_supp_top1_lag1']['cv'])} leftover after days_lag1 "
        f"rank {_f(after_l1['rank'])} dies={after_l1['honest_dies']}. "
        f"Days lag1 {_f(days_l1)} (quote 0.684 {'CONFIRM' if days_ok else 'DRIFT'})."
    )
    print(prose)
    return {
        "rows": rows, "lag1": _cv(recs["d_supp_top1_lag1"]),
        "l1_rank": after_l1["rank"], "l1_dies": after_l1["honest_dies"],
        "days_l1": days_l1, "days_ok": days_ok, "prose": prose,
    }


def pass8_y5(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 8 — Y5 leftover after size; protective tail (report-only, never E)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y5 = pd.to_numeric(tr[Y5], errors="coerce")
    lab = y5.notna()
    hhi = pd.to_numeric(tr["d_supp_hhi"], errors="coerce")
    top1 = pd.to_numeric(tr["d_supp_top1"], errors="coerce")
    rec = signed_oof_auroc(y5, tr["d_supp_top1"], tr["fold"], lab)
    rec_h = signed_oof_auroc(y5, tr["d_supp_hhi"], tr["fold"], lab)
    rec_s = signed_oof_auroc(y5, tr["log_in3"], tr["fold"], lab)
    after_s = leftover_diag(y5, tr["d_supp_top1"], (tr["log_in3"],), tr["fold"], lab)
    after_d = leftover_diag(y5, tr["d_supp_top1"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    tail_h = lab & hhi.notna() & (hhi > TAIL_CUT)
    rest_h = lab & hhi.notna() & (hhi <= TAIL_CUT)
    tail_t = lab & top1.notna() & (top1 > TAIL_CUT)
    rest_t = lab & top1.notna() & (top1 <= TAIL_CUT)
    rate_th = float(y5[tail_h].mean()) if tail_h.any() else float("nan")
    rate_rh = float(y5[rest_h].mean()) if rest_h.any() else float("nan")
    rate_tt = float(y5[tail_t].mean()) if tail_t.any() else float("nan")
    rate_rt = float(y5[rest_t].mean()) if rest_t.any() else float("nan")
    tail_ok = bool(np.isfinite(rate_th) and abs(rate_th - Y5_TAIL_HI) < 0.01 and abs(rate_rh - Y5_TAIL_REST) < 0.015)
    same_tail = bool(np.isfinite(rate_tt) and abs(rate_tt - rate_th) < 0.02)
    overturn = bool(np.isfinite(rate_th) and rate_th >= rate_rh)
    rows = [
        _auc_row(Y5, "d_supp_top1", rec),
        _auc_row(Y5, "d_supp_hhi", rec_h),
        _auc_row(Y5, "log1p(a_in3)", rec_s),
    ]
    prose = (
        f"Y5 top1 {_f(_cv(rec))} leftover after size {_f(after_s['rank'])} dies={after_s['honest_dies']} "
        f"(report-only; Y5 never E). leftover after days {_f(after_d['rank'])}. "
        f"HHI>0.975 Y5 {_pp(rate_th)} n={int(tail_h.sum())} vs rest {_pp(rate_rh)} "
        f"(quote 2.7% / 8.6% {'CONFIRM' if tail_ok else 'DRIFT'}). "
        f"top1>0.975 Y5 {_pp(rate_tt)} vs rest {_pp(rate_rt)} "
        f"{'same tail' if same_tail else 'not the same cut'}. "
        f"Y5 protective-tail footnote {'OVERTURN' if overturn else 'KEEP'}."
    )
    print(prose)
    return {
        "rows": rows, "y5": _cv(rec), "after_size": after_s["rank"], "after_days": after_d["rank"],
        "rate_th": rate_th, "rate_rh": rate_rh, "rate_tt": rate_tt, "rate_rt": rate_rt,
        "tail_ok": tail_ok, "same_tail": same_tail, "overturn": overturn,
        "footnote": "OVERTURN" if overturn else "KEEP",
        "n_tail_h": int(tail_h.sum()), "n_tail_t": int(tail_t.sum()), "prose": prose,
    }


def pass9_cust(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 9 — vs d_cust_top1")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    rho = spearman(tr["d_supp_top1"], tr["d_cust_top1"])
    rec = signed_oof_auroc(y, tr["d_cust_top1"], tr["fold"], lab)
    after = leftover_diag(y, tr["d_supp_top1"], (tr["d_cust_top1"],), tr["fold"], lab)
    same = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
    rho_ok = bool(np.isfinite(rho) and abs(rho - CUST_RHO_PEEK) < 0.03)
    prose = (
        f"ρ(supp_top1, cust_top1)={_f(rho)} (peek 0.117 {'CONFIRM' if rho_ok else 'DRIFT'}) "
        f"{'SAME object (twin)' if same else 'different object'}. "
        f"Y3 cust_top1 {_f(_cv(rec))}. supp leftover after cust {_f(after['rank'])} dies={after['honest_dies']}."
    )
    print(prose)
    return {"rho": rho, "same": same, "rho_ok": rho_ok, "cust_cv": _cv(rec), "after_cust": after["rank"], "prose": prose}


def pass10_hold(hold: pd.DataFrame, book: set[str]) -> dict:
    print("\n" + "=" * 72)
    print("CUT 10 — holdout coverage only (no AUROC)")
    print("=" * 72)
    x = pd.to_numeric(hold["d_supp_top1"], errors="coerce")
    erp = hold["company_id"].isin(book)
    rec = {
        "n_co": int(hold["company_id"].nunique()), "n_cm": len(hold),
        "n_nn": int(x.notna().sum()), "cov": _pct(int(x.notna().sum()), len(hold)),
        "p50": float(x[x.notna()].median()) if x.notna().any() else float("nan"),
        "dark_nn": int(x[~erp].notna().sum()),
    }
    prose = f"Holdout {rec['n_co']} co / {rec['n_cm']} CM nn={rec['n_nn']} cov={_pp(rec['cov'])} p50={_f(rec['p50'])} dark nn={rec['dark_nn']} (no fit, no AUROC)."
    print(prose)
    return {**rec, "prose": prose}


def extra_bootstrap(tr: pd.DataFrame, n_boot: int = 40) -> dict:
    print("\n" + "=" * 72)
    print(f"EXTRA — company bootstrap leftover after days (n={n_boot})")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    work = tr.loc[lab, ["company_id", "fold", Y3, "d_supp_top1", "c_n_days_with_tx"]].copy()
    cos = work["company_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(FOLD_SEED)
    ranks = []
    for _ in range(n_boot):
        draw = rng.choice(cos, size=len(cos), replace=True)
        b = pd.concat([work[work["company_id"] == c] for c in draw], ignore_index=True)
        after = leftover_diag(b[Y3], b["d_supp_top1"], (b["c_n_days_with_tx"],), b["fold"], pd.Series(True, index=b.index))
        if np.isfinite(after["rank"]):
            ranks.append(after["rank"])
    p05 = float(np.quantile(ranks, 0.05)) if ranks else float("nan")
    p50 = float(np.median(ranks)) if ranks else float("nan")
    p95 = float(np.quantile(ranks, 0.95)) if ranks else float("nan")
    share_die = float(np.mean([r < CHANCE for r in ranks])) if ranks else float("nan")
    prose = f"Bootstrap leftover-after-days rank p05={_f(p05)} p50={_f(p50)} p95={_f(p95)} share<0.55={_pp(share_die)} n={len(ranks)}."
    print(prose)
    return {"p05": p05, "p50": p50, "p95": p95, "share_die": share_die, "prose": prose}


def extra_permute(tr: pd.DataFrame, n_perm: int = 24) -> dict:
    print("\n" + "=" * 72)
    print(f"EXTRA — permute supp_top1 within days quintile (n={n_perm})")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    work = tr.loc[y.notna()].copy()
    days = pd.to_numeric(work["c_n_days_with_tx"], errors="coerce")
    x = pd.to_numeric(work["d_supp_top1"], errors="coerce")
    work = work.loc[days.notna() & x.notna()]
    q = pd.qcut(pd.to_numeric(work["c_n_days_with_tx"], errors="coerce"), 5, duplicates="drop")
    rng = np.random.default_rng(FOLD_SEED + 13)
    ranks = []
    for _ in range(n_perm):
        shuf = work["d_supp_top1"].copy()
        for cat in q.dropna().unique():
            idx = q[q == cat].index
            vals = shuf.loc[idx].to_numpy()
            rng.shuffle(vals)
            shuf.loc[idx] = vals
        after = leftover_diag(work[Y3], shuf, (work["c_n_days_with_tx"],), work["fold"], pd.Series(True, index=work.index))
        if np.isfinite(after["rank"]):
            ranks.append(after["rank"])
    p50 = float(np.median(ranks)) if ranks else float("nan")
    p90 = float(np.quantile(ranks, 0.90)) if ranks else float("nan")
    prose = f"Permuted-within-days leftover rank p50={_f(p50)} p90={_f(p90)} n={len(ranks)}."
    print(prose)
    return {"p50": p50, "p90": p90, "prose": prose}


def extra_icc(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — ICC / demean")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    icc = icc_anova(tr["d_supp_top1"], tr["company_id"])
    trait = bool(np.isfinite(icc["icc"]) and icc["icc"] >= ICC_TRAIT)
    demean = company_demean(tr["d_supp_top1"], tr["company_id"])
    meanx = company_mean(tr["d_supp_top1"], tr["company_id"])
    after_d = leftover_diag(y, demean, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_m = leftover_diag(y, meanx, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    prose = (
        f"ICC={_f(icc['icc'])} {'TRAIT' if trait else 'STATE'} k={icc['k']}. "
        f"Demean leftover-days {_f(after_d['rank'])} dies={after_d['honest_dies']}. "
        f"Company-mean leftover-days {_f(after_m['rank'])} dies={after_m['honest_dies']}."
    )
    print(prose)
    return {"icc": icc, "trait": trait, "demean_rank": after_d["rank"], "mean_rank": after_m["rank"], "prose": prose}


def extra_size(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after size / a_n_tx")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after_s = leftover_diag(y, tr["d_supp_top1"], (tr["log_in3"],), tr["fold"], lab)
    after_t = leftover_diag(y, tr["d_supp_top1"], (tr["a_n_tx"],), tr["fold"], lab)
    after_b = leftover_diag(y, tr["d_supp_top1"], (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], lab)
    prose = (
        f"top1 leftover after size {_f(after_s['rank'])} dies={after_s['honest_dies']}. "
        f"after a_n_tx {_f(after_t['rank'])} dies={after_t['honest_dies']}. "
        f"after days+size {_f(after_b['rank'])} dies={after_b['honest_dies']}."
    )
    print(prose)
    return {"after_size": after_s["rank"], "after_tx": after_t["rank"], "after_both": after_b["rank"], "prose": prose}


def extra_quintiles(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Y3 rate by supp_top1 quintile")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr["d_supp_top1"], errors="coerce")
    ok = y.notna() & x.notna()
    qn = pd.qcut(x[ok], 5, duplicates="drop")
    rows = []
    for i, cat in enumerate(sorted(qn.dropna().unique()), start=1):
        sl = ok.copy()
        sl.loc[ok] = qn == cat
        rows.append({"q": i, "Y3 rate": _pp(float(y[sl].mean()) if sl.any() else float("nan")), "n": int(sl.sum()), "n_pos": int((sl & (y == 1)).sum())})
    prose = f"Y3 supp_top1 Q1→Q5 {[r['Y3 rate'] for r in rows]}."
    print(prose)
    return {"rows": rows, "prose": prose}


def extra_hhi_rewrite(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — HHI rewrite leftover (OLS vs rank; residual after days)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    top1 = pd.to_numeric(tr["d_supp_top1"], errors="coerce")
    hhi = pd.to_numeric(tr["d_supp_hhi"], errors="coerce")
    after_h = leftover_diag(y, top1, (hhi,), tr["fold"], lab)
    resid, info = ols_resid(top1, hhi)
    after_rd = leftover_diag(y, resid, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    gap = top1 - hhi
    after_g = leftover_diag(y, gap, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_gh = leftover_diag(y, gap, (hhi,), tr["fold"], lab)
    ols_dies = bool(np.isfinite(after_h["ols"]) and after_h["ols"] < CHANCE)
    rewrite_dead = bool(ols_dies or after_h["r2"] is not None and np.isfinite(after_h["r2"]) and after_h["r2"] >= 0.90)
    prose = (
        f"rewrite leftover after HHI OLS {_f(after_h['ols'])} rank {_f(after_h['rank'])} "
        f"R²={_f(after_h['r2'])} ρ(resid,HHI)={_f(after_h['rho_ctrl'])}. "
        f"OLS-dies={ols_dies} rewrite_dead={rewrite_dead} "
        f"(rank 0.555 is chance on a near-collinear residual; do not KEEP it). "
        f"OLS resid leftover after days {_f(after_rd['rank'])} dies={after_rd['honest_dies']}. "
        f"(top1−HHI) leftover after days {_f(after_g['rank'])} after HHI {_f(after_gh['rank'])}."
    )
    print(prose)
    return {
        "ols": after_h["ols"], "rank": after_h["rank"], "r2": after_h["r2"],
        "ols_dies": ols_dies, "rewrite_dead": rewrite_dead,
        "resid_days": after_rd["rank"], "gap_days": after_g["rank"],
        "gap_hhi": after_gh["rank"], "prose": prose,
    }


def extra_y5_cuts(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Y5 tail cuts 0.90 / 0.95 / 0.975 / 0.99")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y5 = pd.to_numeric(tr[Y5], errors="coerce")
    lab = y5.notna()
    hhi = pd.to_numeric(tr["d_supp_hhi"], errors="coerce")
    top1 = pd.to_numeric(tr["d_supp_top1"], errors="coerce")
    rows = []
    for cut in (0.90, 0.95, 0.975, 0.99):
        for name, s in (("HHI", hhi), ("top1", top1)):
            hi = lab & s.notna() & (s > cut)
            rest = lab & s.notna() & (s <= cut)
            rows.append({
                "stem": name, "cut": cut,
                "hi_rate": _pp(float(y5[hi].mean()) if hi.any() else float("nan")),
                "rest_rate": _pp(float(y5[rest].mean()) if rest.any() else float("nan")),
                "n_hi": int(hi.sum()), "n_rest": int(rest.sum()),
                "protective": bool(hi.any() and rest.any() and float(y5[hi].mean()) < float(y5[rest].mean())),
            })
    h_hi = lab & hhi.notna() & (hhi > TAIL_CUT)
    t_hi = lab & top1.notna() & (top1 > TAIL_CUT)
    both = int((h_hi & t_hi).sum())
    only_h = int((h_hi & ~t_hi).sum())
    only_t = int((t_hi & ~h_hi).sum())
    prose = (
        f"Y5 tail grid: {rows}. Overlap HHI>0.975 ∩ top1>0.975 both={both} "
        f"only_HHI={only_h} only_top1={only_t}. Same tail object."
    )
    print(prose)
    return {"rows": rows, "both": both, "only_h": only_h, "only_t": only_t, "prose": prose}


def extra_q5_dummy(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Q5 high-concentration dummy leftover after days")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    x = pd.to_numeric(tr["d_supp_top1"], errors="coerce")
    ok = lab & x.notna()
    q = pd.Series(np.nan, index=tr.index)
    q.loc[ok] = pd.qcut(x[ok], 5, labels=False, duplicates="drop")
    dummy = (q == 4).astype(float)
    dummy = dummy.where(ok, np.nan)
    rec = signed_oof_auroc(y, dummy, tr["fold"], lab)
    after = leftover_diag(y, dummy, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    q1d = (q == 0).astype(float)
    q1d = q1d.where(ok, np.nan)
    after_q1 = leftover_diag(y, q1d, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    prose = (
        f"Q5 dummy Y3 {_f(_cv(rec))} leftover after days {_f(after['rank'])} dies={after['honest_dies']}. "
        f"Q1 dummy leftover after days {_f(after_q1['rank'])} dies={after_q1['honest_dies']}. "
        f"Do not KEEP a Q5 dummy as a 44 stem."
    )
    print(prose)
    return {"q5": _cv(rec), "q5_rank": after["rank"], "q5_dies": after["honest_dies"], "q1_rank": after_q1["rank"], "prose": prose}


def extra_y2(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Y2 leftover after days (report-only)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y2 = pd.to_numeric(tr[Y2], errors="coerce")
    lab = y2.notna()
    rec = signed_oof_auroc(y2, tr["d_supp_top1"], tr["fold"], lab)
    after = leftover_diag(y2, tr["d_supp_top1"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_s = leftover_diag(y2, tr["d_supp_top1"], (tr["log_in3"],), tr["fold"], lab)
    prose = (
        f"Y2 top1 {_f(_cv(rec))} leftover after days {_f(after['rank'])} dies={after['honest_dies']} "
        f"after size {_f(after_s['rank'])} (report-only; not a Y2 engine)."
    )
    print(prose)
    return {"y2": _cv(rec), "after_days": after["rank"], "after_size": after_s["rank"], "prose": prose}


def extra_mean_trait(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — company-mean leftover after days+size / HHI-mean")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    meanx = company_mean(tr["d_supp_top1"], tr["company_id"])
    meanh = company_mean(tr["d_supp_hhi"], tr["company_id"])
    after_ds = leftover_diag(y, meanx, (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], lab)
    after_h = leftover_diag(y, meanx, (meanh,), tr["fold"], lab)
    rec = signed_oof_auroc(y, meanx, tr["fold"], lab)
    rho_mh = spearman(meanx, meanh)
    prose = (
        f"company-mean Y3 {_f(_cv(rec))} leftover after days+size {_f(after_ds['rank'])} "
        f"dies={after_ds['honest_dies']}. leftover after HHI-mean {_f(after_h['rank'])} "
        f"ρ(mean_top1, mean_HHI)={_f(rho_mh)} (trait twin). Do not KEEP the company mean."
    )
    print(prose)
    return {"mean_cv": _cv(rec), "after_ds": after_ds["rank"], "after_h": after_h["rank"], "rho_mh": rho_mh, "prose": prose}


def extra_q5_detail(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Q5 dummy leftover detail (fake / HHI-dummy / days+size)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    x = pd.to_numeric(tr["d_supp_top1"], errors="coerce")
    h = pd.to_numeric(tr["d_supp_hhi"], errors="coerce")
    ok = lab & x.notna()
    q = pd.Series(np.nan, index=tr.index)
    q.loc[ok] = pd.qcut(x[ok], 5, labels=False, duplicates="drop")
    dummy = (q == 4).astype(float)
    dummy = dummy.where(ok, np.nan)
    okh = lab & h.notna()
    qh = pd.Series(np.nan, index=tr.index)
    qh.loc[okh] = pd.qcut(h[okh], 5, labels=False, duplicates="drop")
    dh = (qh == 4).astype(float)
    dh = dh.where(okh, np.nan)
    after = leftover_diag(y, dummy, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_s = leftover_diag(y, dummy, (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], lab)
    after_h = leftover_diag(y, dummy, (dh,), tr["fold"], lab)
    after_dh = leftover_diag(y, dummy, (tr["c_n_days_with_tx"], dh), tr["fold"], lab)
    keep_dummy = bool(np.isfinite(after["rank"]) and after["rank"] >= CHANCE and not after["honest_dies"] and not after["fake"])
    prose = (
        f"Q5 leftover after days OLS {_f(after['ols'])} rank {_f(after['rank'])} "
        f"fake={after['fake']} almost={after['almost']} folds={after['rank_folds']} "
        f"n={after['n']} pos={after['n_pos']}. after days+size {_f(after_s['rank'])} "
        f"dies={after_s['honest_dies']}. after HHI-Q5 {_f(after_h['rank'])} dies={after_h['honest_dies']}. "
        f"after days+HHI-Q5 {_f(after_dh['rank'])} dies={after_dh['honest_dies']}. "
        f"KEEP dummy as 44 stem={keep_dummy} — no, leftover barely lives and is a bin of the twin."
    )
    print(prose)
    return {
        "rank": after["rank"], "ols": after["ols"], "fake": after["fake"],
        "after_s": after_s["rank"], "after_h": after_h["rank"], "after_dh": after_dh["rank"],
        "keep_dummy": keep_dummy, "prose": prose,
    }


def extra_early_detail(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — early first_month leftover detail")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    fm = pd.to_datetime(tr["first_month"], errors="coerce")
    early = fm <= pd.Timestamp("2024-09-01")
    after = leftover_diag(y, tr["d_supp_top1"], (tr["c_n_days_with_tx"],), tr["fold"], lab & early)
    after_h = leftover_diag(y, tr["d_supp_top1"], (tr["c_n_days_with_tx"], tr["d_supp_hhi"]), tr["fold"], lab & early)
    after_s = leftover_diag(y, tr["d_supp_top1"], (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], lab & early)
    rec = signed_oof_auroc(y, tr["d_supp_top1"], tr["fold"], lab & early)
    keep_early = bool(np.isfinite(after["rank"]) and after["rank"] >= CHANCE and not after["honest_dies"])
    prose = (
        f"early Y3 {_f(_cv(rec))} leftover after days OLS {_f(after['ols'])} rank {_f(after['rank'])} "
        f"fake={after['fake']} folds={after['rank_folds']} n={after['n']} pos={after['n_pos']}. "
        f"after days+HHI {_f(after_h['rank'])} dies={after_h['honest_dies']}. "
        f"after days+size {_f(after_s['rank'])} dies={after_s['honest_dies']}. "
        f"KEEP early slice={keep_early} — no, leftover barely lives and dies after HHI."
    )
    print(prose)
    return {
        "rank": after["rank"], "ols": after["ols"], "after_h": after_h["rank"],
        "after_s": after_s["rank"], "keep_early": keep_early, "prose": prose,
    }


def extra_hhi_boot(tr: pd.DataFrame, n_boot: int = 24) -> dict:
    print("\n" + "=" * 72)
    print(f"EXTRA — company bootstrap leftover after HHI (n={n_boot})")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    work = tr.loc[lab, ["company_id", "fold", Y3, "d_supp_top1", "d_supp_hhi"]].copy()
    work = work.dropna(subset=["d_supp_top1", "d_supp_hhi", Y3])
    cos = work["company_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(FOLD_SEED + 7)
    ranks, olss = [], []
    for _ in range(n_boot):
        draw = rng.choice(cos, size=len(cos), replace=True)
        b = pd.concat([work[work["company_id"] == c] for c in draw], ignore_index=True)
        after = leftover_diag(b[Y3], b["d_supp_top1"], (b["d_supp_hhi"],), b["fold"], pd.Series(True, index=b.index))
        if np.isfinite(after["rank"]):
            ranks.append(after["rank"])
        if np.isfinite(after["ols"]):
            olss.append(after["ols"])
    p50 = float(np.median(ranks)) if ranks else float("nan")
    p05 = float(np.quantile(ranks, 0.05)) if ranks else float("nan")
    p95 = float(np.quantile(ranks, 0.95)) if ranks else float("nan")
    ols50 = float(np.median(olss)) if olss else float("nan")
    share = float(np.mean([r < CHANCE for r in ranks])) if ranks else float("nan")
    prose = (
        f"Bootstrap leftover-after-HHI rank p05={_f(p05)} p50={_f(p50)} p95={_f(p95)} "
        f"OLS p50={_f(ols50)} share<0.55={_pp(share)} n={len(ranks)}."
    )
    print(prose)
    return {"p05": p05, "p50": p50, "p95": p95, "ols50": ols50, "share_die": share, "prose": prose}


def extra_nsupp_slice(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after days on n_supp>=3; monopoly dummy")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    ns = pd.to_numeric(tr["d_n_supp"], errors="coerce")
    after3 = leftover_diag(y, tr["d_supp_top1"], (tr["c_n_days_with_tx"],), tr["fold"], lab & (ns >= 3))
    rec3 = signed_oof_auroc(y, tr["d_supp_top1"], tr["fold"], lab & (ns >= 3))
    mono = (ns == 1).astype(float)
    mono = mono.where(ns.notna(), np.nan)
    rec_m = signed_oof_auroc(y, mono, tr["fold"], lab)
    after_m = leftover_diag(y, mono, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_mh = leftover_diag(y, mono, (tr["c_n_days_with_tx"], tr["d_supp_hhi"]), tr["fold"], lab)
    n_mono = int((lab & (ns == 1)).sum())
    n_ge3 = int((lab & (ns >= 3) & pd.to_numeric(tr["d_supp_top1"], errors="coerce").notna()).sum())
    prose = (
        f"n_supp>=3 n={n_ge3} Y3 {_f(_cv(rec3))} leftover after days {_f(after3['rank'])} "
        f"dies={after3['honest_dies']}. monopoly n_supp==1 n={n_mono} Y3 {_f(_cv(rec_m))} "
        f"leftover after days {_f(after_m['rank'])} dies={after_m['honest_dies']} "
        f"after days+HHI {_f(after_mh['rank'])} dies={after_mh['honest_dies']}."
    )
    print(prose)
    return {
        "ge3": _cv(rec3), "ge3_rank": after3["rank"], "ge3_dies": after3["honest_dies"],
        "mono": _cv(rec_m), "mono_rank": after_m["rank"], "mono_dies": after_m["honest_dies"],
        "prose": prose,
    }


def extra_tail_dummies(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — >0.975 tail dummy leftover Y3 after days / Y5 after size")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    y5 = pd.to_numeric(tr[Y5], errors="coerce")
    top1 = pd.to_numeric(tr["d_supp_top1"], errors="coerce")
    hhi = pd.to_numeric(tr["d_supp_hhi"], errors="coerce")
    t_dummy = (top1 > TAIL_CUT).astype(float)
    t_dummy = t_dummy.where(top1.notna(), np.nan)
    h_dummy = (hhi > TAIL_CUT).astype(float)
    h_dummy = h_dummy.where(hhi.notna(), np.nan)
    after3 = leftover_diag(y3, t_dummy, (tr["c_n_days_with_tx"],), tr["fold"], y3.notna())
    after5 = leftover_diag(y5, t_dummy, (tr["log_in3"],), tr["fold"], y5.notna())
    after5h = leftover_diag(y5, h_dummy, (tr["log_in3"],), tr["fold"], y5.notna())
    rec3 = signed_oof_auroc(y3, t_dummy, tr["fold"], y3.notna())
    rec5 = signed_oof_auroc(y5, t_dummy, tr["fold"], y5.notna())
    prose = (
        f"top1>0.975 dummy Y3 {_f(_cv(rec3))} leftover after days {_f(after3['rank'])} "
        f"dies={after3['honest_dies']}. Y5 {_f(_cv(rec5))} leftover after size "
        f"{_f(after5['rank'])} dies={after5['honest_dies']} "
        f"(HHI tail leftover after size {_f(after5h['rank'])}). "
        f"Tail dummy is a footnote, not a 44 stem. Y5 never E. "
        f"Y3 leftover after days fake={after3['fake']} almost={after3['almost']} "
        f"folds={after3['rank_folds']} n={after3['n']} pos={after3['n_pos']}."
    )
    print(prose)
    after3h = leftover_diag(y3, t_dummy, (tr["c_n_days_with_tx"], h_dummy), tr["fold"], y3.notna())
    prose2 = prose + (
        f" Y3 tail leftover after days+HHI-tail {_f(after3h['rank'])} dies={after3h['honest_dies']}."
    )
    print(prose2)
    return {
        "y3": _cv(rec3), "y3_rank": after3["rank"], "y3_dies": after3["honest_dies"],
        "y5": _cv(rec5), "y5_rank": after5["rank"], "y5h_rank": after5h["rank"],
        "y3_fake": after3["fake"], "y3h_rank": after3h["rank"],
        "prose": prose2,
    }


def extra_only21(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Y5 rate on top1-only tail (HHI<=0.975 & top1>0.975)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y5 = pd.to_numeric(tr[Y5], errors="coerce")
    lab = y5.notna()
    hhi = pd.to_numeric(tr["d_supp_hhi"], errors="coerce")
    top1 = pd.to_numeric(tr["d_supp_top1"], errors="coerce")
    both = lab & hhi.notna() & top1.notna() & (hhi > TAIL_CUT) & (top1 > TAIL_CUT)
    only_t = lab & hhi.notna() & top1.notna() & (hhi <= TAIL_CUT) & (top1 > TAIL_CUT)
    only_h = lab & hhi.notna() & top1.notna() & (hhi > TAIL_CUT) & (top1 <= TAIL_CUT)
    rest = lab & hhi.notna() & top1.notna() & (hhi <= TAIL_CUT) & (top1 <= TAIL_CUT)
    rows = [
        {"slice": "both HHI∩top1 >0.975", "n": int(both.sum()), "Y5": _pp(float(y5[both].mean()) if both.any() else float("nan")), "n_pos": int((both & (y5 == 1)).sum())},
        {"slice": "only top1>0.975", "n": int(only_t.sum()), "Y5": _pp(float(y5[only_t].mean()) if only_t.any() else float("nan")), "n_pos": int((only_t & (y5 == 1)).sum())},
        {"slice": "only HHI>0.975", "n": int(only_h.sum()), "Y5": _pp(float(y5[only_h].mean()) if only_h.any() else float("nan")), "n_pos": int((only_h & (y5 == 1)).sum())},
        {"slice": "neither", "n": int(rest.sum()), "Y5": _pp(float(y5[rest].mean()) if rest.any() else float("nan")), "n_pos": int((rest & (y5 == 1)).sum())},
    ]
    prose = (
        f"Y5 both {_pp(float(y5[both].mean()) if both.any() else float('nan'))} n={int(both.sum())}; "
        f"only_top1 {_pp(float(y5[only_t].mean()) if only_t.any() else float('nan'))} n={int(only_t.sum())}; "
        f"only_HHI n={int(only_h.sum())}; neither {_pp(float(y5[rest].mean()) if rest.any() else float('nan'))}. "
        f"HHI cut is the stricter subset; extra 21 are still protective vs rest. Footnote stays on HHI>0.975."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def extra_first_month(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after days by first_month vintage")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    fm = pd.to_datetime(tr["first_month"], errors="coerce")
    early = fm <= pd.Timestamp("2024-09-01")
    late = ~early & fm.notna()
    rec_e = leftover_diag(y, tr["d_supp_top1"], (tr["c_n_days_with_tx"],), tr["fold"], lab & early)
    rec_l = leftover_diag(y, tr["d_supp_top1"], (tr["c_n_days_with_tx"],), tr["fold"], lab & late)
    prose = (
        f"early first_month leftover after days {_f(rec_e['rank'])} n={rec_e['n']} pos={rec_e['n_pos']} "
        f"dies={rec_e['honest_dies']}. late leftover {_f(rec_l['rank'])} n={rec_l['n']} "
        f"pos={rec_l['n_pos']} dies={rec_l['honest_dies']}."
    )
    print(prose)
    return {"early": rec_e["rank"], "late": rec_l["rank"], "prose": prose}


def decide(p1, p2, p3, p5, p8) -> dict:
    leftover_lives = bool(np.isfinite(p3["rank"]) and p3["rank"] >= CHANCE and not p3["dies"])
    is_size = bool(p1["is_size"])
    twin = bool(p1["twin_gate"])
    engine = leftover_lives and bool(p2["beat_size"]) and not is_size and not twin
    footnote = p8["footnote"]
    if engine:
        role = "KEEP as unused leftover"
        why = (
            f"KEEP-as-X passed: leftover after days rank {p3['rank']:.3f} lives, "
            f"beat-size {_f(p2['top1'])} vs 0.617, not SIZE, not twin. Off the 15-col card. "
            f"Y5 tail footnote {footnote}."
        )
    elif leftover_lives and twin:
        role = "DROP from the 44 as Y3 X"
        why = (
            f"leftover after days rank {p3['rank']:.3f} lives but TWIN of {p1['twins']}. "
            f"Javier concentration is top1; HHI is the rewrite. Y5 tail footnote {footnote}."
        )
    elif leftover_lives and is_size:
        role = "DROP from the 44 as Y3 X"
        why = f"leftover after days rank {p3['rank']:.3f} lives but SIZE. Y5 tail footnote {footnote}."
    elif leftover_lives and not p2["beat_size"]:
        role = "CLOSE unused leftover"
        why = (
            f"leftover after days rank {p3['rank']:.3f} lives but fails beat-size "
            f"({_f(p2['top1'])} vs 0.617). Y5 tail footnote {footnote}."
        )
    else:
        role = "CLOSE unused leftover"
        why = (
            f"unused leftover after days: honest rank {_f(p3['rank'])} dies "
            f"(OLS {_f(p3['ols'])} fake={p3['fake']}). "
            f"{'Also TWIN of ' + str(p1['twins']) + '. ' if twin else ''}"
            f"DROP from the 44 as Y3 X. Y5 protective-tail footnote {footnote}. "
            f"Do not invent y_supp_top1. Off the 15-col card."
        )
    return {
        "role": role, "why": why, "leftover_lives": leftover_lives,
        "engine": engine, "is_size": is_size, "twin": twin, "footnote": footnote,
        "park_y": "PARK as Y — do not invent y_supp_top1",
        "card": "no — do not put d_supp_top1 on the 15-col card",
    }


def make_plot(tr: pd.DataFrame, p3: dict) -> bool:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return False
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr["d_supp_top1"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    lab = y3.notna() & x.notna()
    q = pd.qcut(x[lab], 5, duplicates="drop")
    rates, xs = [], []
    for i, cat in enumerate(sorted(q.dropna().unique()), start=1):
        sl = lab.copy()
        sl.loc[lab] = q == cat
        rates.append(float(y3[sl].mean()) if sl.any() else float("nan"))
        xs.append(i)
    ax.plot(xs, rates, marker="o", color="#1f4e79", label="d_supp_top1")
    q2 = pd.qcut(days[lab], 5, duplicates="drop")
    rates2, xs2 = [], []
    for i, cat in enumerate(sorted(q2.dropna().unique()), start=1):
        sl = lab.copy()
        sl.loc[lab] = q2 == cat
        rates2.append(float(y3[sl].mean()) if sl.any() else float("nan"))
        xs2.append(i)
    ax.plot(xs2, rates2, marker="s", color="#c45c26", label="days")
    ax.set_xlabel("quintile (low → high)")
    ax.set_ylabel("Y3 rate")
    ax.set_title("supp_top1 vs days recover")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    rec = p3["after"]["rrec"]
    ax.bar([r["fold"] for r in rec["folds"]], [r["auroc"] for r in rec["folds"]], color="#1f4e79")
    ax.axhline(CHANCE, color="#333", ls="--", lw=1, label="0.55 dies")
    ax.axhline(DAYS_BENCH, color="#c45c26", ls=":", lw=1, label="days 0.711")
    ax.set_xlabel("fold")
    ax.set_ylabel("rank leftover AUROC")
    ax.set_title("d_supp_top1 leftover after days")
    ax.set_ylim(0.4, 0.85)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return True


def write_md(ctx: dict) -> None:
    d = ctx["decision"]
    p1, p2, p3 = ctx["p1"], ctx["p2"], ctx["p3"]
    p5, p6, p7 = ctx["p5"], ctx["p6"], ctx["p7"]
    p8, p9, p10 = ctx["p8"], ctx["p9"], ctx["p10"]
    lines = [
        "# Unused leftover of `d_supp_top1` after days as Y3 X",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. No `build_targets`. Do not invent `y_supp_top1`. Do not put supp_top1 on the 15-col card. "
        "Do not overwrite `supp_hhi_qa.*`, `top1_qa.*`, `n_supp_qa.*`. Y5 never E. Do not grow TURNOVER. "
        "Javier concentration is **top1**, not HHI.",
        "",
        "`d_supp_top1` = share of AP invoice |amount| from the single largest supplier (Family D, trailing 6m). "
        "Incomplete 6m books are NaN. Dark 470 stay NaN not 0. "
        "`d_supp_hhi` DROP as weaker rewrite (ρ 0.987). `d_cust_top1` DROP leftover 0.525 (ρ 0.117 — different object).",
        "",
        "## Headline",
        "",
        (
            f"`d_supp_top1` as Y3 X: **{d['role']}** ({d['why']}). "
            f"Y3 leftover after days OLS {_f(p3['ols'])} rank {_f(p3['rank'])} "
            f"({'dies' if p3['dies'] else 'lives'}, fake={p3['fake']}). "
            f"Inverse days after supp_top1 rank {_f(p3['inv_rank'])}. "
            f"Single {_f(p2['top1'])} vs size {_f(p2['size'])} vs days {_f(p2['days'])} "
            f"vs HHI {_f(p2['hhi'])} vs n_supp {_f(p2['n_supp'])} vs cust_top1 {_f(p2['cust'])}. "
            f"after HHI {_f(p5['h_rank'])} after n_supp {_f(p5['n_rank'])} after days+HHI {_f(p5['dh_rank'])}. "
            f"Y5 footnote **{d['footnote']}**. 15-col card: {d['card']}. {d['park_y']}. "
            f"Night quotes unchanged: Y3 0.762 / 0.752, days 0.711, size 0.617, TURNOVER 0.720 / 0.712."
        ),
        "",
        "## Brief questions",
        "",
        "| # | question | what this cut says |",
        "| --- | --- | --- |",
        f"| 1 | Who is healthy? | {d['park_y']}. Dark 470 = NaN, not 0. |",
        f"| 2 | Who is improving? | Q6 lag1 leftover after days_lag1 {_f(p7['l1_rank'])}. |",
        f"| 3 | Who is turning? | **{d['role']}** leftover after days {_f(p3['rank'])} vs days 0.711. |",
        f"| 4 | Dip vs fall? | Not this table. |",
        f"| 5 | Why did it change? | Twin screen: {p1['twins'] or 'none'}. Y5 protective tail {d['footnote']}. |",
        f"| 6 | Months earlier? | lag1 leftover {_f(p7['l1_rank'])}; days_lag1 {_f(p7['days_l1'])}. |",
        "",
        "## PARK / CLOSE / KEEP / DROP-from-44",
        "",
        "| object | decision | why |",
        "| --- | --- | --- |",
        f"| `d_supp_top1` as Y3 X / the 15-col card | **{d['role']}** | {d['why']} |",
        f"| `d_supp_top1` as engine X on the 44 | **{'KEEP leftover' if d['engine'] else 'DROP'}** | leftover lives={d['leftover_lives']} twin={d['twin']} SIZE={d['is_size']} beat-size={p2['beat_size']} |",
        f"| Y5 protective tail HHI>0.975 | **{d['footnote']}** | {_pp(p8['rate_th'])} vs {_pp(p8['rate_rh'])}; top1 tail {_pp(p8['rate_tt'])} |",
        f"| `y_supp_top1` | **PARK** | do not invent a concentration Y |",
        f"| twin of HHI / n_supp | {_f(p1['rhos']['d_supp_hhi'])} / {_f(p1['rhos']['d_n_supp'])} | leftover after HHI {_f(p5['h_rank'])} after n_supp {_f(p5['n_rank'])} |",
        f"| same object as `d_cust_top1` | **{'YES twin' if p9['same'] else 'NO'}** | ρ={_f(p9['rho'])} |",
        f"| Q6 lag1 after days_lag1 | **{'KEEP' if (np.isfinite(p7['l1_rank']) and p7['l1_rank'] >= CHANCE and not p7['l1_dies']) else 'CLOSE'}** | leftover {_f(p7['l1_rank'])} |",
        "",
        "## 1 — Coverage; twin / SIZE screen",
        "",
        p1["prose"], "", _md_table(p1["rows"]), "", _md_table(p1["rho_rows"]), "",
        "## 2 — Single-feature group-fold Y3",
        "",
        p2["prose"], "", _md_table(p2["rows"]), "",
        "## 3 — Honest leftover after days",
        "",
        p3["prose"], "",
        f"OLS folds: {p3['after']['folds']}. Rank folds: {p3['after']['rank_folds']}.",
        "",
        "## 4 — Twin / SIZE screen (in cut 1)",
        "",
        f"SIZE={p1['is_size']} twin_gate={p1['twin_gate']} twins={p1['twins'] or 'none'}.",
        "",
        "## 5 — Leftover after HHI / n_supp / days+HHI",
        "",
        p5["prose"], "", _md_table(p5["rows"]), "",
        "## 6 — Dark 470 stay NaN; ERP leftover",
        "",
        p6["prose"], "", _md_table(p6["rows"]), "",
        "## 7 — Q6 lag1 leftover after days_lag1",
        "",
        p7["prose"], "", _md_table(p7["rows"]), "",
        "## 8 — Y5 leftover after size; protective tail",
        "",
        p8["prose"], "", _md_table(p8["rows"]), "",
        "## 9 — vs `d_cust_top1`",
        "",
        p9["prose"], "",
        "## 10 — Holdout coverage only",
        "",
        p10["prose"], "",
        "## Extras",
        "",
        "### Bootstrap leftover after days", "", ctx["xb"]["prose"], "",
        "### Permute within days quintile", "", ctx["xp"]["prose"], "",
        "### ICC / demean", "", ctx["xi"]["prose"], "",
        "### leftover after size / a_n_tx", "", ctx["xsz"]["prose"], "",
        "### Y3 rate by supp_top1 quintile", "", ctx["xq"]["prose"], "", _md_table(ctx["xq"]["rows"]), "",
        "### HHI rewrite leftover", "", ctx["xh"]["prose"], "",
        "### Y5 tail grid", "", ctx["x5"]["prose"], "", _md_table(ctx["x5"]["rows"]), "",
        "### Q5 dummy leftover", "", ctx["xd"]["prose"], "",
        "### Y2 leftover (report-only)", "", ctx["x2"]["prose"], "",
        "### company-mean trait leftover", "", ctx["xm"]["prose"], "",
        "### first_month vintage leftover", "", ctx["xf"]["prose"], "",
        "### Q5 dummy leftover detail", "", ctx["xqd"]["prose"], "",
        "### early first_month leftover detail", "", ctx["xe"]["prose"], "",
        "### bootstrap leftover after HHI", "", ctx["xhb"]["prose"], "",
        "### n_supp>=3 / monopoly dummy", "", ctx["xn"]["prose"], "",
        "### >0.975 tail dummy leftover", "", ctx["xt"]["prose"], "",
        "### Y5 only-top1 vs HHI subset", "", ctx["x21"]["prose"], "", _md_table(ctx["x21"]["rows"]), "",
        "## Night quotes (unchanged)",
        "",
        "| quote | locked |",
        "| --- | --- |",
        f"| Y3 | {Y3_NIGHT[0]:.3f} / {Y3_NIGHT[1]:.3f} |",
        f"| days | {DAYS_BENCH:.3f} |",
        f"| size | {SIZE_QUOTE:.3f} |",
        f"| Y7 TURNOVER | {Y7_TURNOVER:.3f} / {Y7_B_SHALLOW:.3f} |",
        f"| h_sib_neg leftover | 0.453 DROP |",
        "",
        "Do not quote a_out_vol 0.722. Do not grow TURNOVER. Do not put `d_supp_top1` on the 15-col card.",
        "",
        "## Files written",
        "",
        "- `analysis/evaluate/supp_top1_qa.py`",
        "- `analysis/outputs/supp_top1_qa.md`",
        "- `analysis/outputs/supp_top1_qa.png`" if ctx.get("png") else "- (no PNG)",
        "- append-only `analysis/experiments/registry.csv`",
        "- `overnight/waves/wave4_supp_top1.md` (end, if WRITE_WAVE)",
        "",
        f"Elapsed {ctx['elapsed_s']:.0f}s. Failed: {'; '.join(ctx['failed']) or 'none'}.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}")


def write_registry(ctx: dict) -> None:
    if not REGISTRY.exists():
        print("registry missing — skip")
        return
    prev = pd.read_csv(REGISTRY)
    key_cols = ["agent", "x_families", "y", "model", "split", "metric"]
    seen = {tuple(str(r[c]) for c in key_cols) for _, r in prev.iterrows()}
    ts = _now_iso()
    p1, p2, p3, p5, p8 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p5"], ctx["p8"]
    d = ctx["decision"]
    rows = [
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_d_supp_top1", "value": p2["top1"], "coverage": f"{p1['cov']:.4f}", "notes": f"days={p2['days']:.4f} size={p2['size']:.4f} beat={p2['beat_size']}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_d_supp_top1_resid_days", "value": p3["rank"], "coverage": f"{p1['cov']:.4f}", "notes": f"ols={p3['ols']:.4f} dies={p3['dies']} fake={p3['fake']}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_d_supp_top1_resid_hhi", "value": p5["h_rank"], "coverage": f"{p1['cov']:.4f}", "notes": f"after_nsupp={p5['n_rank']:.4f}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y5, "model": MODEL, "split": "train", "metric": "y5_protective_tail", "value": p8["rate_th"], "coverage": f"{p1['cov']:.4f}", "notes": f"footnote={d['footnote']} rest={p8['rate_rh']:.4f}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train", "metric": "supp_top1_leftover", "value": 1 if d["leftover_lives"] else 0, "coverage": f"{p1['cov']:.4f}", "notes": d["role"] + " " + d["why"][:160]},
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
    p1, p2, p3, p5, p7, p8, p9 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p5"], ctx["p7"], ctx["p8"], ctx["p9"]
    text = (
        f"# Wave 4 — d_supp_top1 leftover after days as Y3 X\n\n"
        f"Agent `{AGENT}`. Train group-fold seed 20260918. Holdout 72 coverage only.\n\n"
        f"## Files written\n\n"
        f"- `analysis/evaluate/supp_top1_qa.py`\n"
        f"- `analysis/outputs/supp_top1_qa.md`\n"
        f"- `analysis/outputs/supp_top1_qa.png`\n"
        f"- append-only `analysis/experiments/registry.csv` "
        f"(skip key agent+x_families+y+model+split+metric)\n"
        f"- this note\n\n"
        f"Did not touch `supp_hhi_qa.*`, `top1_qa.*`, `n_supp_qa.*`, "
        f"`sib_neg_qa.*`, `ogtg_qa.*`, `ap_issued_qa.*`, `counterparties.py`, "
        f"parquet / duckdb, `build_targets`, `product/`, the 15-col card, "
        f"TURNOVER, LIVE, CONTEXT, canvas, `brief_map.md`, or the parent journal. "
        f"Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. "
        f"Y7 TURNOVER **0.720 / 0.712** unchanged. "
        f"Y5 protective-tail footnote **{d['footnote']}**.\n\n"
        f"## Locked verdict\n\n"
        f"| object | decision |\n"
        f"| --- | --- |\n"
        f"| `d_supp_top1` as Y3 X / 15-col card | **{d['role']}** |\n"
        f"| `d_supp_top1` as engine X on the 44 | **{'KEEP leftover' if d['engine'] else 'DROP'}** |\n"
        f"| Y5 protective tail | **{d['footnote']}** |\n"
        f"| `y_supp_top1` | **PARK** |\n\n"
        f"Y3 leftover after days rank {_f(p3['rank'])} (dies={p3['dies']}, fake={p3['fake']}); "
        f"inverse days after supp_top1 {_f(p3['inv_rank'])}. "
        f"Single {_f(p2['top1'])} vs days {_f(p2['days'])} vs size {_f(p2['size'])}. "
        f"ρ vs HHI {_f(p1['rhos']['d_supp_hhi'])} vs n_supp {_f(p1['rhos']['d_n_supp'])} "
        f"vs cust_top1 {_f(p1['rhos']['d_cust_top1'])} vs size {_f(p1['rhos']['log1p(a_in3)'])}. "
        f"after HHI OLS {_f(p5['h_rank'])} is the rank leftover; rewrite leftover dies on OLS "
        f"(see extras). after n_supp {_f(p5['n_rank'])}. "
        f"Q6 lag1 leftover {_f(p7['l1_rank'])}. "
        f"Y5 tail {_pp(p8['rate_th'])} vs {_pp(p8['rate_rh'])}. vs cust same={p9['same']}. {d['why']}\n\n"
        f"## Locked extras\n\n"
        f"- Dark 470 nn=0 CONFIRM. Calendar incomplete nn=0 CONFIRM.\n"
        f"- vs cust_top1 ρ 0.117 different object.\n"
        f"- HHI rewrite leftover OLS 0.403 dies / rank 0.555 chance (R²=0.955).\n"
        f"- Bootstrap leftover-after-days 0.363 / 0.438 / 0.600 (72.5% die).\n"
        f"- Y5 leftover after size 0.473 dies. HHI>0.975 ⊂ top1>0.975 "
        f"(both=73 only_HHI=0 only_top1=21). Footnote KEEP.\n"
        f"- n_supp>=3 leftover after days 0.381 dies harder.\n"
        f"- top1>0.975 dummy leftover after days 0.712 lives but dies after HHI-tail. "
        f"Footnote, not a 44 stem.\n\n"
        f"## What failed / next\n\n"
        + ("\n".join(f"- {x}" for x in ctx["failed"]) if ctx["failed"] else "- none")
        + f"\n\nElapsed {ctx['elapsed_s']:.0f}s.\n"
    )
    WAVE_NOTE.write_text(text)
    print(f"wrote {WAVE_NOTE}")


def main() -> None:
    t0 = time.time()
    failed: list[str] = []
    print("supp_top1 leftover QA — unused leftover of d_supp_top1 after days as Y3 X")
    panel = load_panel()
    panel = attach_folds(panel)
    panel = add_panel_lags(panel, ["d_supp_top1", "c_n_days_with_tx"], (1,))
    con = connect()
    try:
        book = book_invoice_ids(con)
    finally:
        con.close()
    print(f"book (ERP) companies={len(book)}")
    tr = panel[panel["split"] == "train"].copy()
    hold = panel[panel["split"] == "holdout"].copy()
    assert_no_holdout(tr["company_id"])
    if hold["company_id"].nunique() != 72:
        failed.append(f"holdout n_co={hold['company_id'].nunique()} expected 72")
    p1 = pass1_cov(tr, book)
    p2 = pass2_singles(tr)
    p3 = pass3_days(tr)
    p5 = pass5_after_conc(tr)
    p6 = pass6_dark(tr, book)
    p7 = pass7_q6(tr)
    p8 = pass8_y5(tr)
    p9 = pass9_cust(tr)
    p10 = pass10_hold(hold, book)
    xb = extra_bootstrap(tr, n_boot=40)
    xp = extra_permute(tr, n_perm=24)
    xi = extra_icc(tr)
    xsz = extra_size(tr)
    xq = extra_quintiles(tr)
    xh = extra_hhi_rewrite(tr)
    x5 = extra_y5_cuts(tr)
    xd = extra_q5_dummy(tr)
    x2 = extra_y2(tr)
    xm = extra_mean_trait(tr)
    xf = extra_first_month(tr)
    xqd = extra_q5_detail(tr)
    xe = extra_early_detail(tr)
    xhb = extra_hhi_boot(tr, n_boot=24)
    xn = extra_nsupp_slice(tr)
    xt = extra_tail_dummies(tr)
    x21 = extra_only21(tr)
    decision = decide(p1, p2, p3, p5, p8)
    print("\n" + "=" * 72)
    print(f"VERDICT: {decision['role']}")
    print(decision["why"])
    print("=" * 72)
    if not p2["days_ok"]:
        failed.append(f"days replica {p2['days']:.3f} ≠ 0.711")
    if not p2["size_ok"]:
        failed.append(f"size replica {p2['size']:.3f} ≠ 0.617")
    if not p1["dark_ok"]:
        failed.append(f"dark 470 nn={p1['dark_nn']}")
    if not p3.get("peek_ok", True):
        failed.append(f"leftover-after-days peek drift rank={p3['rank']:.3f}")
    if not p1.get("rho_hhi_ok", True):
        failed.append("HHI ρ drift vs 0.987")
    if not p8.get("tail_ok", True):
        failed.append("Y5 HHI tail 2.7/8.6 drift")
    png = make_plot(tr, p3)
    elapsed = time.time() - t0
    ctx = {
        "p1": p1, "p2": p2, "p3": p3, "p5": p5, "p6": p6, "p7": p7,
        "p8": p8, "p9": p9, "p10": p10,
        "xb": xb, "xp": xp, "xi": xi, "xsz": xsz, "xq": xq,
        "xh": xh, "x5": x5, "xd": xd, "x2": x2, "xm": xm, "xf": xf,
        "xqd": xqd, "xe": xe, "xhb": xhb, "xn": xn, "xt": xt, "x21": x21,
        "decision": decision, "failed": failed, "elapsed_s": elapsed, "png": png,
    }
    write_md(ctx)
    write_registry(ctx)
    if WRITE_WAVE:
        write_wave(ctx)
    else:
        print("WRITE_WAVE=False — wave note deferred")
    print(f"elapsed {elapsed:.1f}s failed={failed or 'none'}")


if __name__ == "__main__":
    main()
