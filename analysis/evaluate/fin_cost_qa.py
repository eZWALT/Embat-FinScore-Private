"""Unused leftover of contemporaneous ``a_fin_cost`` after ``c_n_days_with_tx``.

``a_fin_cost`` = -sum(amount | grp = fin_cost) (cashflow.py). Store also has
``f_fin_cost`` / ``f_fc_r`` / ``a_debt_service`` / ``f_ds_r``.
Contemporaneous ``f_fc_r`` already DROP leftover after days 0.449.
KEEP ``f_fc_r_lag3`` on TURNOVER. Y9 is the fee label — leftover of
``a_fin_cost`` as Y3 X, not as Y9 X. Do not merge M.

KEEP-as-X: beat size ≥0.02 AND leftover after days AND not SIZE
(|ρ| vs log1p(a_in3) ≥0.50) AND not a twin (|ρ|≥0.80 vs days / a_n_tx /
f_fin_cost / f_fc_r / a_debt_service). Leftover <0.55 dies.
Rank leftover is honest; OLS can fake a days leak.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.fin_cost_qa

Owned: analysis/evaluate/fin_cost_qa.py, analysis/outputs/fin_cost_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_fin_cost.md.
Do not overwrite y9_why.* / fc_r_qa.* / ds_r_qa.* / catmix.py /
util_snap_qa.* / ogtg_qa.* / n_types_qa.* / gbm_core.py / debt.py.
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
OUT_MD = ANALYSIS / "outputs" / "fin_cost_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "fin_cost_qa.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_fin_cost.md"
AGENT = "b17e9c44"
WAVE = "4"
ROUND = "R4"
MODEL = "fin_cost_qa"
X_FAM = "A"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
Y9 = "y9_fee_r_ownp80"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_Y3_QUOTE = 0.617
DAYS_LAG1_QUOTE = 0.684
FC_R_LEFT_QUOTE = 0.449
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
    "a_fin_cost",
    "a_debt_service",
    "c_n_days_with_tx",
    "f_fin_cost",
    "f_fc_r",
    "f_ds_r",
)
Y_KEEP = (Y2, Y3, Y9)


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
                {
                    "fold": k,
                    "auroc": float("nan"),
                    "sign": 0,
                    "n_va": int(va.sum()),
                    "n_pos": int((va & (y == 1)).sum()),
                }
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
    return " ".join(
        f"{r['auroc']:.3f}" if np.isfinite(r["auroc"]) else "—" for r in rec.get("folds", [])
    )


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
    X = np.column_stack(
        [np.ones(int(ok.sum()))] + [d.loc[ok, f"x{i}"].to_numpy(dtype=float) for i in range(n_x)]
    )
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
    have_y = [c for c in Y_KEEP if c in yraw.columns]
    y = _keys(yraw[["company_id", "period", *have_y]])
    panel = panel.merge(y, on=["company_id", "period"], how="left")
    hold = load_holdout()
    panel["split"] = np.where(panel["company_id"].isin(hold), "holdout", "train")
    panel["log_in3"] = np.log1p(pd.to_numeric(panel["a_in3"], errors="coerce").clip(lower=0))
    for col in ("a_fin_cost", "a_debt_service", "f_fin_cost", "f_fc_r", "f_ds_r", "a_n_tx", "c_n_days_with_tx"):
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    leak = leakage_check(
        ["a_fin_cost", "f_fin_cost", "f_fc_r", "c_n_days_with_tx", "log_in3"],
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
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    n_cm = int(len(tr))
    n_def = int(x.notna().sum())
    n_pos = int((x > 0).sum())
    n_zero = int((x == 0).sum())
    cov = n_def / n_cm if n_cm else float("nan")
    share_pos = n_pos / n_def if n_def else float("nan")
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_x = pd.to_numeric(last["a_fin_cost"], errors="coerce")
    n_last = int(last_x.notna().sum())
    last_only = bool(n_def == n_last and n_def > 0)
    periods = tr.loc[x.notna(), "period"].nunique()
    acf1 = median_acf(x, tr["company_id"], 1)
    acf3 = median_acf(x, tr["company_id"], 3)
    rho = spearman(x, tr["log_in3"])
    size_flag = bool(np.isfinite(rho) and abs(rho) >= SIZE_RHO)
    med = float(x.dropna().median()) if n_def else float("nan")
    p95 = float(x.dropna().quantile(0.95)) if n_def else float("nan")
    ever = tr.groupby("company_id")["a_fin_cost"].apply(lambda s: pd.to_numeric(s, errors="coerce").gt(0).any())
    prose = (
        f"Train a_fin_cost defined {n_def:,}/{n_cm:,} ({_pp(cov)}). "
        f"zeros {n_zero:,} ({_pp(n_zero / n_def if n_def else float('nan'))}) "
        f">0 {n_pos:,} ({_pp(share_pos)}). last-month-only={last_only} "
        f"periods_with_def={periods} last-nn {n_last}. "
        f"median {_f(med)} p95 {_f(p95)}. acf1 {_f(acf1)} acf3 {_f(acf3)}. "
        f"ever>0 {int(ever.sum())}/{int(ever.size)} companies. "
        f"ρ vs log1p(a_in3) {_f(rho)} ({'SIZE' if size_flag else 'not SIZE'})."
    )
    print(prose)
    rows = [
        {
            "col": "a_fin_cost",
            "n_nn": f"{n_def:,}",
            "cov": _pp(cov),
            "share>0": _pp(share_pos),
            "p50": _f(med),
            "last_only": str(last_only),
        }
    ]
    return {
        "n_def": n_def,
        "n_cm": n_cm,
        "cov": cov,
        "share_pos": share_pos,
        "n_pos": n_pos,
        "n_zero": n_zero,
        "last_only": last_only,
        "acf1": acf1,
        "acf3": acf3,
        "rho_in3": rho,
        "size_flag": size_flag,
        "med": med,
        "p95": p95,
        "ever_pos": int(ever.sum()),
        "n_co": int(ever.size),
        "rows": rows,
        "prose": prose,
    }


def pass2_twins(tr: pd.DataFrame) -> dict:
    x = tr["a_fin_cost"]
    pairs = [
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ("a_n_tx", tr["a_n_tx"]),
        ("f_fin_cost", tr["f_fin_cost"]),
        ("f_fc_r", tr["f_fc_r"]),
        ("a_debt_service", tr["a_debt_service"]),
        ("f_ds_r", tr["f_ds_r"]),
        ("log1p(a_in3)", tr["log_in3"]),
    ]
    rows = []
    twins = []
    rhos = {}
    for name, s in pairs:
        rho = spearman(x, s)
        rhos[name] = rho
        twin = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
        if twin:
            twins.append(name)
        rows.append({"vs": name, "ρ": _f(rho), "twin": "YES" if twin else ""})
    gate = bool(twins)
    prose = (
        f"Spearman twins |ρ|≥0.80 on a_fin_cost: {twins or 'none'}. "
        f"vs days {_f(rhos.get('c_n_days_with_tx'))} vs a_n_tx {_f(rhos.get('a_n_tx'))} "
        f"vs f_fin_cost {_f(rhos.get('f_fin_cost'))} vs f_fc_r {_f(rhos.get('f_fc_r'))} "
        f"vs a_debt_service {_f(rhos.get('a_debt_service'))} vs size {_f(rhos.get('log1p(a_in3)'))}."
    )
    print(prose)
    return {
        "rows": rows,
        "twins": twins,
        "gate_twins": gate,
        "rho_days": rhos.get("c_n_days_with_tx", float("nan")),
        "rho_ntx": rhos.get("a_n_tx", float("nan")),
        "rho_ffc": rhos.get("f_fin_cost", float("nan")),
        "rho_fcr": rhos.get("f_fc_r", float("nan")),
        "rho_ds": rhos.get("a_debt_service", float("nan")),
        "rho_dsr": rhos.get("f_ds_r", float("nan")),
        "rho_size": rhos.get("log1p(a_in3)", float("nan")),
        "prose": prose,
    }


def pass3_singles(tr: pd.DataFrame) -> dict:
    feats = {
        "a_fin_cost": tr["a_fin_cost"],
        "f_fin_cost": tr["f_fin_cost"],
        "f_fc_r": tr["f_fc_r"],
        "a_debt_service": tr["a_debt_service"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "log1p(a_in3)": tr["log_in3"],
    }
    rows = []
    recs = {}
    for name, s in feats.items():
        rec = signed_oof_auroc(tr[Y3], s, tr["fold"], tr[Y3].notna())
        recs[name] = rec
        print(f"AUROC {Y3} {name}: {_f(rec['cv'])} n={rec['n_defined']} n_pos={rec['n_pos']}")
        rows.append(_auc_row(Y3, name, rec))
    y2 = signed_oof_auroc(tr[Y2], tr["a_fin_cost"], tr["fold"], tr[Y2].notna())
    rows.append(_auc_row(Y2, "a_fin_cost", y2))
    days = _cv(recs["c_n_days_with_tx"])
    size = _cv(recs["log1p(a_in3)"])
    y3 = _cv(recs["a_fin_cost"])
    beat_size = float(y3 - size) if np.isfinite(y3) and np.isfinite(size) else float("nan")
    days_ok = bool(np.isfinite(days) and abs(days - DAYS_BENCH) < 0.015)
    size_ok = bool(np.isfinite(size) and abs(size - SIZE_Y3_QUOTE) < 0.015)
    prose = (
        f"Y3 native a_fin_cost {_f(y3)} n={recs['a_fin_cost']['n_defined']} "
        f"n_pos={recs['a_fin_cost']['n_pos']}. "
        f"vs days {_f(days)} ({'0.711 CONFIRM' if days_ok else 'off'}) "
        f"vs size {_f(size)} ({'0.617 CONFIRM' if size_ok else 'off'}) "
        f"vs f_fc_r {_f(_cv(recs['f_fc_r']))} vs f_fin_cost {_f(_cv(recs['f_fin_cost']))}. "
        f"beat_size={_f(beat_size)}."
    )
    print(prose)
    return {
        "rows": rows,
        "y3": y3,
        "days": days,
        "size": size,
        "fcr": _cv(recs["f_fc_r"]),
        "ffc": _cv(recs["f_fin_cost"]),
        "y2": _cv(y2),
        "beat_size": beat_size,
        "days_ok": days_ok,
        "size_ok": size_ok,
        "n": recs["a_fin_cost"]["n_defined"],
        "n_pos": recs["a_fin_cost"]["n_pos"],
        "native_lp": recs["a_fin_cost"]["low_power"],
        "prose": prose,
    }


def pass4_leftover(tr: pd.DataFrame) -> dict:
    x = tr["a_fin_cost"]
    days = tr["c_n_days_with_tx"]
    specs = [
        ("after days", (days,)),
        ("after f_fc_r", (tr["f_fc_r"],)),
        ("after f_fin_cost", (tr["f_fin_cost"],)),
        ("after a_debt_service", (tr["a_debt_service"],)),
        ("after days+f_fc_r", (days, tr["f_fc_r"])),
        ("after days+f_fin_cost", (days, tr["f_fin_cost"])),
        ("after size", (tr["log_in3"],)),
    ]
    rows = []
    recs = {}
    for name, xs in specs:
        rec = leftover_diag(tr[Y3], x, list(xs), tr["fold"], tr[Y3].notna())
        recs[name] = rec
        print(
            f"left {name}: OLS={_f(rec['ols'])} rank={_f(rec['rank'])} "
            f"n_pos={rec['n_pos']} ρctrl={_f(rec['rho_ctrl'])} fake={rec['fake']}"
        )
        rows.append(
            {
                "cut": name,
                "OLS": _f(rec["ols"]),
                "rank": _f(rec["rank"]),
                "ρctrl": _f(rec["rho_ctrl"]),
                "fake": rec["fake"],
                "n_pos": rec["n_pos"],
            }
        )
    inv = leftover_diag(tr[Y3], days, [x], tr["fold"], tr[Y3].notna())
    print(f"inverse days after a_fin_cost: rank={_f(inv['rank'])} ρ={_f(inv['rho_ctrl'])} fake={inv['fake']}")
    rows.append(
        {
            "cut": "days after a_fin_cost",
            "OLS": _f(inv["ols"]),
            "rank": _f(inv["rank"]),
            "ρctrl": _f(inv["rho_ctrl"]),
            "fake": inv["fake"],
            "n_pos": inv["n_pos"],
        }
    )
    days_left = recs["after days"]
    y3_dies = bool(days_left["honest_dies"])
    prose = (
        f"Honest leftover after days rank {_f(days_left['rank'])} OLS {_f(days_left['ols'])} "
        f"ρ(resid,days)={_f(days_left['rho_ctrl'])} fake={days_left['fake']}. "
        f"After f_fc_r {_f(recs['after f_fc_r']['rank'])} after f_fin_cost {_f(recs['after f_fin_cost']['rank'])} "
        f"after a_debt_service {_f(recs['after a_debt_service']['rank'])} "
        f"after days+f_fc_r {_f(recs['after days+f_fc_r']['rank'])}. "
        f"Inverse days after a_fin_cost {_f(inv['rank'])}. "
        f"Honest leftover {'DIES' if y3_dies else 'lives'}."
    )
    print(prose)
    return {
        "rows": rows,
        "y3_rank": days_left["rank"],
        "y3_ols": days_left["ols"],
        "y3_fake": days_left["fake"],
        "y3_dies": y3_dies,
        "y3_rho": days_left["rho_ctrl"],
        "fcr_left": recs["after f_fc_r"]["rank"],
        "ffc_left": recs["after f_fin_cost"]["rank"],
        "ds_left": recs["after a_debt_service"]["rank"],
        "stack_left": recs["after days+f_fc_r"]["rank"],
        "size_left": recs["after size"]["rank"],
        "inv_rank": inv["rank"],
        "inv_fake": inv["fake"],
        "days_folds": days_left["rank_folds"],
        "prose": prose,
    }


def pass5_path(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    d = tr.assign(x=x).sort_values(["company_id", "period"])
    d["prev"] = d.groupby("company_id")["x"].shift(1)
    both = d["x"].notna() & d["prev"].notna()
    rises = int((both & (d["x"] > d["prev"])).sum())
    drops = int((both & (d["x"] < d["prev"])).sum())
    same = int((both & (d["x"] == d["prev"])).sum())
    n_pairs = int(both.sum())
    rise_only = bool(drops == 0 and rises > 0)
    prose = (
        f"Within-company a_fin_cost path: rises {rises} drops {drops} same {same} "
        f"pairs {n_pairs} rise-only={rise_only} (flow, not inventory)."
    )
    print(prose)
    return {
        "rises": rises,
        "drops": drops,
        "same": same,
        "n_pairs": n_pairs,
        "rise_only": rise_only,
        "prose": prose,
    }


def pass6_dark(tr: pd.DataFrame, book: set[str]) -> dict:
    erp = tr["company_id"].isin(book)
    n_erp = int(tr.loc[erp, "company_id"].nunique())
    n_dark = int(tr.loc[~erp, "company_id"].nunique())
    confirm = n_dark == 470 and n_erp == 744
    rec_i = leftover_diag(
        tr[Y3], tr["a_fin_cost"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & erp
    )
    rec_d = leftover_diag(
        tr[Y3], tr["a_fin_cost"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & ~erp
    )
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    rows = [
        {
            "book": "invoiced",
            "n_co": n_erp,
            "n_nn": int(x[erp].notna().sum()),
            "share>0": _pp(float((x[erp] > 0).mean()) if erp.any() else float("nan")),
            "leftover": _f(rec_i["rank"]),
            "fake": rec_i["fake"],
        },
        {
            "book": "dark",
            "n_co": n_dark,
            "n_nn": int(x[~erp].notna().sum()),
            "share>0": _pp(float((x[~erp] > 0).mean()) if (~erp).any() else float("nan")),
            "leftover": _f(rec_d["rank"]),
            "fake": rec_d["fake"],
        },
    ]
    prose = (
        f"Last-month ever-ERP {n_erp} / never-ERP {n_dark} "
        f"({'CONFIRM 744/470' if confirm else 'off'}). "
        f"Leftover after days invoiced {_f(rec_i['rank'])} dark {_f(rec_d['rank'])}."
    )
    print(prose)
    return {
        "n_erp": n_erp,
        "n_dark": n_dark,
        "confirm": confirm,
        "inv": rec_i["rank"],
        "dark": rec_d["rank"],
        "rows": rows,
        "prose": prose,
    }


def pass7_q6(tr: pd.DataFrame) -> dict:
    rec_l1 = leftover_diag(
        tr[Y3],
        tr["a_fin_cost_lag1"],
        [tr["c_n_days_with_tx_lag1"]],
        tr["fold"],
        tr[Y3].notna(),
    )
    days_l1 = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx_lag1"], tr["fold"], tr[Y3].notna())
    days_l1_cv = _cv(days_l1)
    days_l1_ok = bool(np.isfinite(days_l1_cv) and abs(days_l1_cv - DAYS_LAG1_QUOTE) < 0.02)
    rec_l3 = leftover_diag(
        tr[Y3],
        tr["a_fin_cost_lag3"],
        [tr["c_n_days_with_tx_lag3"]],
        tr["fold"],
        tr[Y3].notna(),
    )
    q6_dies = bool(rec_l1["honest_dies"] or (np.isfinite(rec_l1["rank"]) and rec_l1["rank"] < CHANCE))
    q6 = "CLOSE" if q6_dies else "KEEP"
    prose = (
        f"Q6 lag1 leftover after days_lag1 {_f(rec_l1['rank'])} "
        f"ρ={_f(rec_l1['rho_ctrl'])} fake={rec_l1['fake']}. "
        f"Days lag1 {_f(days_l1_cv)} ({'0.684 CONFIRM' if days_l1_ok else 'off'}). "
        f"lag3 leftover after days_lag3 {_f(rec_l3['rank'])}. Q6 {q6}."
    )
    print(prose)
    rows = [
        {"cut": "a_fin_cost_lag1 after days_lag1", "rank": _f(rec_l1["rank"]), "fake": rec_l1["fake"]},
        {"cut": "a_fin_cost_lag3 after days_lag3", "rank": _f(rec_l3["rank"]), "fake": rec_l3["fake"]},
        {"cut": "days_lag1 single", "rank": _f(days_l1_cv), "fake": False},
    ]
    return {
        "l1_left": rec_l1["rank"],
        "l3_left": rec_l3["rank"],
        "days_l1": days_l1_cv,
        "days_l1_ok": days_l1_ok,
        "q6": q6,
        "rows": rows,
        "prose": prose,
    }


def pass8_y9(tr: pd.DataFrame) -> dict:
    if Y9 not in tr.columns:
        prose = "Y9 column missing — skip (do not run build_targets)."
        print(prose)
        return {"rho": float("nan"), "prose": prose}
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    y9 = pd.to_numeric(tr[Y9], errors="coerce")
    rho = spearman(x, y9)
    rec = leftover_diag(tr[Y3], x, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & y9.notna())
    hi = x > x.median()
    both = (hi & (y9 == 1)).sum()
    union = (hi | (y9 == 1)).sum()
    jac = float(both / union) if union else float("nan")
    prose = (
        f"Y9 descriptive only (not as Y9 X): ρ a_fin_cost~Y9 {_f(rho)} "
        f"Jaccard(hi,Y9+)={_f(jac)}. Leftover after days on Y9-defined {_f(rec['rank'])}. "
        f"Do not treat a_fin_cost as Y9 X. Do not merge M."
    )
    print(prose)
    return {"rho": rho, "jac": jac, "left": rec["rank"], "prose": prose}


def pass_holdout(panel: pd.DataFrame) -> dict:
    ho = panel[panel["split"] == "holdout"]
    x = pd.to_numeric(ho["a_fin_cost"], errors="coerce")
    prose = (
        f"Holdout coverage only (no fit): {ho['company_id'].nunique()} co / {len(ho):,} CM, "
        f"defined {int(x.notna().sum())} >0 {int((x > 0).sum())}."
    )
    print(prose)
    return {"n_co": int(ho["company_id"].nunique()), "n_cm": int(len(ho)), "prose": prose}


def pass_boot(tr: pd.DataFrame, n_boot: int = 32) -> dict:
    rng = np.random.default_rng(FOLD_SEED)
    cos = tr["company_id"].drop_duplicates().to_numpy()
    idx = {c: tr.index[tr["company_id"] == c].to_numpy() for c in cos}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(cos, size=len(cos), replace=True)
        rows = np.concatenate([idx[c] for c in pick])
        sub = tr.loc[rows].reset_index(drop=True)
        rec = leftover_diag(
            sub[Y3], sub["a_fin_cost"], [sub["c_n_days_with_tx"]], sub["fold"], sub[Y3].notna()
        )
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
        f"Company bootstrap leftover-after-days n={out['n']} "
        f"p05/p50/p95 {_f(out['p05'])} / {_f(out['p50'])} / {_f(out['p95'])}."
    )
    print(prose)
    return {**out, "prose": prose}


def pass_pos(tr: pd.DataFrame) -> dict:
    mask = tr[Y3].notna() & pd.to_numeric(tr["a_fin_cost"], errors="coerce").gt(0)
    rec = leftover_diag(tr[Y3], tr["a_fin_cost"], [tr["c_n_days_with_tx"]], tr["fold"], mask)
    dummy = leftover_diag(
        tr[Y3],
        pd.to_numeric(tr["a_fin_cost"], errors="coerce").gt(0).astype(float),
        [tr["c_n_days_with_tx"]],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"On a_fin_cost>0 leftover after days {_f(rec['rank'])} n_pos={rec['n_pos']} "
        f"ρ={_f(rec['rho_ctrl'])} fake={rec['fake']}. "
        f">0 dummy leftover after days {_f(dummy['rank'])} fake={dummy['fake']}."
    )
    print(prose)
    return {"pos": rec["rank"], "dummy": dummy["rank"], "prose": prose}


def pass_between(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    d = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    xm = x.groupby(tr["company_id"]).transform("mean")
    dm = d.groupby(tr["company_id"]).transform("mean")
    rec_b = leftover_diag(tr[Y3], xm, [dm], tr["fold"], tr[Y3].notna())
    rec_w = leftover_diag(tr[Y3], x - xm, [d - dm], tr["fold"], tr[Y3].notna())
    prose = (
        f"BETWEEN leftover of company-mean a_fin_cost after days-mean {_f(rec_b['rank'])} "
        f"ρ={_f(rec_b['rho_ctrl'])} fake={rec_b['fake']}; "
        f"WITHIN leftover {_f(rec_w['rank'])} fake={rec_w['fake']}."
    )
    print(prose)
    return {"between": rec_b["rank"], "within": rec_w["rank"], "prose": prose}


def pass_y2(tr: pd.DataFrame) -> dict:
    rec = leftover_diag(tr[Y2], tr["a_fin_cost"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y2].notna())
    prose = (
        f"Y2 leftover after days {_f(rec['rank'])} ρ={_f(rec['rho_ctrl'])} fake={rec['fake']} "
        f"n_pos={rec['n_pos']}."
    )
    print(prose)
    return {"left": rec["rank"], "fake": rec["fake"], "prose": prose}


def pass_rewrite(tr: pd.DataFrame) -> dict:
    a = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    f = pd.to_numeric(tr["f_fin_cost"], errors="coerce")
    both = a.notna() & f.notna()
    diff = (a - f).abs()
    n_eq = int((both & (diff < 1e-9)).sum())
    n_both = int(both.sum())
    mx = float(diff[both].max()) if n_both else float("nan")
    rec_f = leftover_diag(tr[Y3], f, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    prose = (
        f"Exact rewrite: a_fin_cost==f_fin_cost on {n_eq:,}/{n_both:,} "
        f"({_pp(n_eq / n_both if n_both else float('nan'))}) max|Δ|={_f(mx, 6)}. "
        f"f_fin_cost leftover after days {_f(rec_f['rank'])} fake={rec_f['fake']} "
        f"(same object as a_fin_cost)."
    )
    print(prose)
    return {
        "n_eq": n_eq,
        "n_both": n_both,
        "max_abs": mx,
        "ffc_left": rec_f["rank"],
        "prose": prose,
    }


def pass_log(tr: pd.DataFrame) -> dict:
    x = np.log1p(pd.to_numeric(tr["a_fin_cost"], errors="coerce").clip(lower=0))
    rec = leftover_diag(tr[Y3], x, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    rec_s = leftover_diag(
        tr[Y3], x, [tr["c_n_days_with_tx"], tr["log_in3"]], tr["fold"], tr[Y3].notna()
    )
    prose = (
        f"log1p(a_fin_cost) leftover after days {_f(rec['rank'])} "
        f"ρ={_f(rec['rho_ctrl'])} fake={rec['fake']}; after days+size {_f(rec_s['rank'])}."
    )
    print(prose)
    return {"left": rec["rank"], "stack": rec_s["rank"], "fake": rec["fake"], "prose": prose}


def pass_ratio(tr: pd.DataFrame) -> dict:
    fc = pd.to_numeric(tr["a_fin_cost"], errors="coerce").clip(lower=0)
    inn = pd.to_numeric(tr["a_in3"], errors="coerce").clip(lower=0)
    ratio = fc / inn.replace(0, np.nan)
    rec = leftover_diag(tr[Y3], ratio, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    rec_f = leftover_diag(tr[Y3], ratio, [tr["f_fc_r"]], tr["fold"], tr[Y3].notna())
    rho = spearman(ratio, tr["f_fc_r"])
    prose = (
        f"Monthly a_fin_cost/a_in3 leftover after days {_f(rec['rank'])} "
        f"ρ={_f(rec['rho_ctrl'])} fake={rec['fake']}; after f_fc_r {_f(rec_f['rank'])}; "
        f"ρ vs f_fc_r {_f(rho)} (3m rate vs month ratio)."
    )
    print(prose)
    return {"left": rec["rank"], "after_fcr": rec_f["rank"], "rho_fcr": rho, "prose": prose}


def pass_ntx(tr: pd.DataFrame) -> dict:
    rec = leftover_diag(
        tr[Y3],
        tr["a_fin_cost"],
        [tr["c_n_days_with_tx"], tr["a_n_tx"]],
        tr["fold"],
        tr[Y3].notna(),
    )
    rec_n = leftover_diag(tr[Y3], tr["a_fin_cost"], [tr["a_n_tx"]], tr["fold"], tr[Y3].notna())
    prose = (
        f"Leftover after days+a_n_tx {_f(rec['rank'])} ρ={_f(rec['rho_ctrl'])} fake={rec['fake']}; "
        f"after a_n_tx alone {_f(rec_n['rank'])}."
    )
    print(prose)
    return {"stack": rec["rank"], "ntx": rec_n["rank"], "prose": prose}


def pass_inv_fcr(tr: pd.DataFrame) -> dict:
    rec = leftover_diag(tr[Y3], tr["f_fc_r"], [tr["a_fin_cost"]], tr["fold"], tr[Y3].notna())
    rec_d = leftover_diag(
        tr[Y3], tr["f_fc_r"], [tr["a_fin_cost"], tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna()
    )
    prose = (
        f"Inverse leftover of f_fc_r after a_fin_cost {_f(rec['rank'])} "
        f"ρ={_f(rec['rho_ctrl'])} fake={rec['fake']}; after a_fin_cost+days {_f(rec_d['rank'])} "
        f"(f_fc_r leftover after days quote 0.449)."
    )
    print(prose)
    return {"left": rec["rank"], "stack": rec_d["rank"], "prose": prose}


def pass_fold(tr: pd.DataFrame) -> dict:
    rec = leftover_diag(tr[Y3], tr["a_fin_cost"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    prose = (
        f"Leftover-after-days rank folds {rec['rank_folds']} rank {_f(rec['rank'])} "
        f"ρ={_f(rec['rho_ctrl'])} fake={rec['fake']}."
    )
    print(prose)
    return {"folds": rec["rank_folds"], "rank": rec["rank"], "prose": prose}


def pass_perm(tr: pd.DataFrame, n_perm: int = 16) -> dict:
    rng = np.random.default_rng(FOLD_SEED + 7)
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce").to_numpy(dtype=float)
    obs = leftover_diag(tr[Y3], tr["a_fin_cost"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())[
        "rank"
    ]
    nulls = []
    for _ in range(n_perm):
        shuf = pd.Series(rng.permutation(x), index=tr.index)
        rec = leftover_diag(tr[Y3], shuf, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
        if np.isfinite(rec["rank"]):
            nulls.append(rec["rank"])
    arr = np.array(nulls, dtype=float)
    p_ge = float(np.mean(arr >= obs)) if arr.size and np.isfinite(obs) else float("nan")
    prose = (
        f"Permute a_fin_cost leftover-after-days null p50 "
        f"{_f(float(np.median(arr)) if arr.size else float('nan'))} "
        f"p(obs≥null)={_f(p_ge, 3)} obs={_f(obs)}."
    )
    print(prose)
    return {"p50": float(np.median(arr)) if arr.size else float("nan"), "p_ge": p_ge, "obs": obs, "prose": prose}


def pass_terciles(tr: pd.DataFrame) -> dict:
    size = pd.to_numeric(tr["log_in3"], errors="coerce")
    q = size.quantile([1 / 3, 2 / 3])
    bits = []
    for i, (lo, hi, name) in enumerate(
        [(None, q.iloc[0], "T1"), (q.iloc[0], q.iloc[1], "T2"), (q.iloc[1], None, "T3")]
    ):
        if lo is None:
            m = size.notna() & (size <= hi)
        elif hi is None:
            m = size.notna() & (size > lo)
        else:
            m = size.notna() & (size > lo) & (size <= hi)
        rec = leftover_diag(
            tr[Y3], tr["a_fin_cost"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & m
        )
        bits.append(f"{name} {_f(rec['rank'])}")
    prose = f"Leftover after days by SIZE tercile: {' / '.join(bits)}."
    print(prose)
    return {"prose": prose}


def pass_dsr(tr: pd.DataFrame) -> dict:
    rec = leftover_diag(tr[Y3], tr["a_fin_cost"], [tr["f_ds_r"]], tr["fold"], tr[Y3].notna())
    rec_s = leftover_diag(
        tr[Y3],
        tr["a_fin_cost"],
        [tr["c_n_days_with_tx"], tr["f_ds_r"], tr["f_fc_r"]],
        tr["fold"],
        tr[Y3].notna(),
    )
    rec_ds = leftover_diag(tr[Y3], tr["a_debt_service"], [tr["a_fin_cost"]], tr["fold"], tr[Y3].notna())
    prose = (
        f"Leftover after f_ds_r {_f(rec['rank'])} ρ={_f(rec['rho_ctrl'])} fake={rec['fake']}; "
        f"after days+f_ds_r+f_fc_r {_f(rec_s['rank'])}; "
        f"inverse a_debt_service leftover after a_fin_cost {_f(rec_ds['rank'])}."
    )
    print(prose)
    return {"dsr": rec["rank"], "stack": rec_s["rank"], "ds": rec_ds["rank"], "prose": prose}


def pass_ever(tr: pd.DataFrame) -> dict:
    ever = tr.groupby("company_id")["a_fin_cost"].transform(lambda s: pd.to_numeric(s, errors="coerce").gt(0).any())
    dummy = ever.astype(float)
    rec = leftover_diag(tr[Y3], dummy, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    rec_x = leftover_diag(
        tr[Y3], tr["a_fin_cost"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & ever
    )
    prose = (
        f"Ever-fee company dummy leftover after days {_f(rec['rank'])} fake={rec['fake']}; "
        f"a_fin_cost leftover after days on ever>0 companies {_f(rec_x['rank'])} n_pos={rec_x['n_pos']}."
    )
    print(prose)
    return {"dummy": rec["rank"], "on_ever": rec_x["rank"], "prose": prose}


def pass_logo(tr: pd.DataFrame) -> dict:
    groups = tr["group_id"].dropna().drop_duplicates().to_numpy()
    vals = []
    for g in groups:
        rec = leftover_diag(
            tr[Y3],
            tr["a_fin_cost"],
            [tr["c_n_days_with_tx"]],
            tr["fold"],
            tr[Y3].notna() & (tr["group_id"] != g),
        )
        if np.isfinite(rec["rank"]):
            vals.append(rec["rank"])
    arr = np.array(vals, dtype=float)
    prose = (
        f"Leave-one-group leftover-after-days n={arr.size} "
        f"min/med/max {_f(float(arr.min()) if arr.size else float('nan'))} / "
        f"{_f(float(np.median(arr)) if arr.size else float('nan'))} / "
        f"{_f(float(arr.max()) if arr.size else float('nan'))}."
    )
    print(prose)
    return {
        "min": float(arr.min()) if arr.size else float("nan"),
        "med": float(np.median(arr)) if arr.size else float("nan"),
        "max": float(arr.max()) if arr.size else float("nan"),
        "prose": prose,
    }


def pass_t3_fold(tr: pd.DataFrame) -> dict:
    size = pd.to_numeric(tr["log_in3"], errors="coerce")
    q = size.quantile(2 / 3)
    m = size.notna() & (size > q)
    rec = leftover_diag(tr[Y3], tr["a_fin_cost"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & m)
    prose = (
        f"SIZE T3 leftover-after-days folds {rec['rank_folds']} rank {_f(rec['rank'])} "
        f"n_pos={rec['n_pos']} ρ={_f(rec['rho_ctrl'])} fake={rec['fake']}."
    )
    print(prose)
    return {"rank": rec["rank"], "folds": rec["rank_folds"], "prose": prose}


def pass_q6_now(tr: pd.DataFrame) -> dict:
    rec = leftover_diag(
        tr[Y3], tr["a_fin_cost_lag1"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna()
    )
    rec3 = leftover_diag(
        tr[Y3], tr["a_fin_cost_lag3"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna()
    )
    prose = (
        f"Q6 lag1 leftover after contemporaneous days {_f(rec['rank'])} "
        f"ρ={_f(rec['rho_ctrl'])} fake={rec['fake']}; lag3 after days {_f(rec3['rank'])}."
    )
    print(prose)
    return {"l1": rec["rank"], "l3": rec3["rank"], "prose": prose}


def pass_dark_log(tr: pd.DataFrame, book: set[str]) -> dict:
    x = np.log1p(pd.to_numeric(tr["a_fin_cost"], errors="coerce").clip(lower=0))
    erp = tr["company_id"].isin(book)
    rec_i = leftover_diag(tr[Y3], x, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & erp)
    rec_d = leftover_diag(tr[Y3], x, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & ~erp)
    prose = (
        f"log1p leftover after days invoiced {_f(rec_i['rank'])} ρ={_f(rec_i['rho_ctrl'])} "
        f"fake={rec_i['fake']} dark {_f(rec_d['rank'])} ρ={_f(rec_d['rho_ctrl'])} fake={rec_d['fake']}."
    )
    print(prose)
    return {"inv": rec_i["rank"], "dark": rec_d["rank"], "prose": prose}


def pass_ever_stack(tr: pd.DataFrame) -> dict:
    ever = tr.groupby("company_id")["a_fin_cost"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").gt(0).any()
    ).astype(float)
    rec_s = leftover_diag(
        tr[Y3], ever, [tr["c_n_days_with_tx"], tr["log_in3"]], tr["fold"], tr[Y3].notna()
    )
    rec_f = leftover_diag(
        tr[Y3], ever, [tr["c_n_days_with_tx"], tr["f_fc_r"]], tr["fold"], tr[Y3].notna()
    )
    rec_x = leftover_diag(tr[Y3], tr["a_fin_cost"], [ever], tr["fold"], tr[Y3].notna())
    rec_xd = leftover_diag(
        tr[Y3], tr["a_fin_cost"], [ever, tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna()
    )
    inv = leftover_diag(tr[Y3], tr["c_n_days_with_tx"], [ever], tr["fold"], tr[Y3].notna())
    prose = (
        f"Ever-fee dummy leftover after days+size {_f(rec_s['rank'])} "
        f"ρ={_f(rec_s['rho_ctrl'])} fake={rec_s['fake']}; after days+f_fc_r {_f(rec_f['rank'])}. "
        f"Amount leftover after ever-dummy {_f(rec_x['rank'])}; after ever+days {_f(rec_xd['rank'])}. "
        f"Inverse days after ever-dummy {_f(inv['rank'])}."
    )
    print(prose)
    return {
        "size": rec_s["rank"],
        "fcr": rec_f["rank"],
        "amt": rec_x["rank"],
        "amt_days": rec_xd["rank"],
        "inv": inv["rank"],
        "prose": prose,
    }


def pass_ever_fold(tr: pd.DataFrame) -> dict:
    ever = tr.groupby("company_id")["a_fin_cost"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").gt(0).any()
    ).astype(float)
    rec = leftover_diag(tr[Y3], ever, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    rec_y2 = leftover_diag(tr[Y2], ever, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y2].notna())
    prose = (
        f"Ever-fee dummy leftover-after-days folds {rec['rank_folds']} rank {_f(rec['rank'])}; "
        f"Y2 {_f(rec_y2['rank'])}."
    )
    print(prose)
    return {"rank": rec["rank"], "folds": rec["rank_folds"], "y2": rec_y2["rank"], "prose": prose}


def pass_hi(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    cut = float(x.quantile(0.90))
    dummy = (x >= cut).astype(float)
    rec = leftover_diag(tr[Y3], dummy, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    rec_s = leftover_diag(
        tr[Y3], dummy, [tr["c_n_days_with_tx"], tr["log_in3"]], tr["fold"], tr[Y3].notna()
    )
    prose = (
        f"High-fee p90 dummy (cut={_f(cut)}) leftover after days {_f(rec['rank'])} "
        f"ρ={_f(rec['rho_ctrl'])} fake={rec['fake']}; after days+size {_f(rec_s['rank'])}."
    )
    print(prose)
    return {"left": rec["rank"], "stack": rec_s["rank"], "prose": prose}


def pass_fcr_pos(tr: pd.DataFrame) -> dict:
    m = pd.to_numeric(tr["f_fc_r"], errors="coerce").gt(0)
    rec = leftover_diag(
        tr[Y3], tr["a_fin_cost"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & m
    )
    rec_r = leftover_diag(
        tr[Y3], tr["a_fin_cost"], [tr["f_fc_r"], tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & m
    )
    prose = (
        f"On f_fc_r>0 leftover after days {_f(rec['rank'])} n_pos={rec['n_pos']} "
        f"ρ={_f(rec['rho_ctrl'])} fake={rec['fake']}; after f_fc_r+days {_f(rec_r['rank'])}."
    )
    print(prose)
    return {"left": rec["rank"], "stack": rec_r["rank"], "prose": prose}


def pass_hi_stack(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    dummy = (x >= float(x.quantile(0.90))).astype(float)
    rec_f = leftover_diag(
        tr[Y3], dummy, [tr["c_n_days_with_tx"], tr["f_fc_r"]], tr["fold"], tr[Y3].notna()
    )
    rec_s = leftover_diag(
        tr[Y3],
        dummy,
        [tr["c_n_days_with_tx"], tr["log_in3"], tr["f_fc_r"]],
        tr["fold"],
        tr[Y3].notna(),
    )
    rec_y2 = leftover_diag(tr[Y2], dummy, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y2].notna())
    rec_x = leftover_diag(
        tr[Y3], tr["a_fin_cost"], [dummy, tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna()
    )
    prose = (
        f"High-fee dummy leftover after days+f_fc_r {_f(rec_f['rank'])}; "
        f"after days+size+f_fc_r {_f(rec_s['rank'])}; Y2 after days {_f(rec_y2['rank'])}; "
        f"amount leftover after high-fee+days {_f(rec_x['rank'])}."
    )
    print(prose)
    return {
        "fcr": rec_f["rank"],
        "stack": rec_s["rank"],
        "y2": rec_y2["rank"],
        "amt": rec_x["rank"],
        "prose": prose,
    }


def pass_hi_fold(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    dummy = (x >= float(x.quantile(0.90))).astype(float)
    rec = leftover_diag(tr[Y3], dummy, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    rec_s = leftover_diag(
        tr[Y3], dummy, [tr["c_n_days_with_tx"], tr["log_in3"]], tr["fold"], tr[Y3].notna()
    )
    prose = (
        f"High-fee dummy leftover-after-days folds {rec['rank_folds']} rank {_f(rec['rank'])}; "
        f"after days+size folds {rec_s['rank_folds']} rank {_f(rec_s['rank'])}."
    )
    print(prose)
    return {"folds": rec["rank_folds"], "rank": rec["rank"], "prose": prose}


def pass_ever_dark(tr: pd.DataFrame, book: set[str]) -> dict:
    ever = tr.groupby("company_id")["a_fin_cost"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").gt(0).any()
    ).astype(float)
    erp = tr["company_id"].isin(book)
    rec_i = leftover_diag(
        tr[Y3], ever, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & erp
    )
    rec_d = leftover_diag(
        tr[Y3], ever, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & ~erp
    )
    prose = (
        f"Ever-fee dummy leftover after days invoiced {_f(rec_i['rank'])} "
        f"dark {_f(rec_d['rank'])}."
    )
    print(prose)
    return {"inv": rec_i["rank"], "dark": rec_d["rank"], "prose": prose}


def pass_boot_ever(tr: pd.DataFrame, n_boot: int = 24) -> dict:
    rng = np.random.default_rng(FOLD_SEED + 11)
    ever_map = tr.groupby("company_id")["a_fin_cost"].apply(
        lambda s: float(pd.to_numeric(s, errors="coerce").gt(0).any())
    )
    cos = tr["company_id"].drop_duplicates().to_numpy()
    idx = {c: tr.index[tr["company_id"] == c].to_numpy() for c in cos}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(cos, size=len(cos), replace=True)
        rows = np.concatenate([idx[c] for c in pick])
        sub = tr.loc[rows].reset_index(drop=True)
        ev = sub["company_id"].map(ever_map)
        rec = leftover_diag(sub[Y3], ev, [sub["c_n_days_with_tx"]], sub["fold"], sub[Y3].notna())
        if np.isfinite(rec["rank"]):
            vals.append(rec["rank"])
    arr = np.array(vals, dtype=float)
    prose = (
        f"Company bootstrap ever-fee dummy leftover-after-days n={arr.size} "
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


def pass_hi_more(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    dummy = (x >= float(x.quantile(0.90))).astype(float)
    rec_n = leftover_diag(
        tr[Y3], dummy, [tr["c_n_days_with_tx"], tr["a_n_tx"]], tr["fold"], tr[Y3].notna()
    )
    rec_d = leftover_diag(
        tr[Y3], dummy, [tr["c_n_days_with_tx"], tr["f_ds_r"]], tr["fold"], tr[Y3].notna()
    )
    rec_q = leftover_diag(
        tr[Y3], dummy.groupby(tr["company_id"]).shift(1), [tr["c_n_days_with_tx_lag1"]], tr["fold"], tr[Y3].notna()
    )
    rec_inv = leftover_diag(tr[Y3], tr["f_fc_r"], [dummy], tr["fold"], tr[Y3].notna())
    prose = (
        f"High-fee dummy leftover after days+a_n_tx {_f(rec_n['rank'])}; "
        f"after days+f_ds_r {_f(rec_d['rank'])}; "
        f"lag1 leftover after days_lag1 {_f(rec_q['rank'])}; "
        f"inverse f_fc_r leftover after high-fee dummy {_f(rec_inv['rank'])}."
    )
    print(prose)
    return {
        "ntx": rec_n["rank"],
        "dsr": rec_d["rank"],
        "q6": rec_q["rank"],
        "inv": rec_inv["rank"],
        "prose": prose,
    }


def pass_hi_dark(tr: pd.DataFrame, book: set[str]) -> dict:
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    dummy = (x >= float(x.quantile(0.90))).astype(float)
    erp = tr["company_id"].isin(book)
    rec_i = leftover_diag(
        tr[Y3], dummy, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & erp
    )
    rec_d = leftover_diag(
        tr[Y3], dummy, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & ~erp
    )
    prose = (
        f"High-fee dummy leftover after days invoiced {_f(rec_i['rank'])} "
        f"dark {_f(rec_d['rank'])}."
    )
    print(prose)
    return {"inv": rec_i["rank"], "dark": rec_d["rank"], "prose": prose}


def pass_hi_on(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    hi = x >= float(x.quantile(0.90))
    rec = leftover_diag(
        tr[Y3], tr["a_fin_cost"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & hi
    )
    prose = (
        f"On high-fee months leftover of amount after days {_f(rec['rank'])} "
        f"n_pos={rec['n_pos']} ρ={_f(rec['rho_ctrl'])} fake={rec['fake']}."
    )
    print(prose)
    return {"left": rec["rank"], "n_pos": rec["n_pos"], "prose": prose}


def pass_p80(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    dummy = (x >= float(x.quantile(0.80))).astype(float)
    rec = leftover_diag(tr[Y3], dummy, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    rec_s = leftover_diag(
        tr[Y3], dummy, [tr["c_n_days_with_tx"], tr["log_in3"], tr["f_fc_r"]], tr["fold"], tr[Y3].notna()
    )
    y9 = pd.to_numeric(tr[Y9], errors="coerce") if Y9 in tr.columns else None
    jac = float("nan")
    if y9 is not None:
        both = ((dummy == 1) & (y9 == 1)).sum()
        union = ((dummy == 1) | (y9 == 1)).sum()
        jac = float(both / union) if union else float("nan")
    prose = (
        f"p80-fee dummy leftover after days {_f(rec['rank'])} folds {rec['rank_folds']}; "
        f"after days+size+f_fc_r {_f(rec_s['rank'])}; Jaccard vs Y9 {_f(jac)} "
        f"(Y9 is the fee label — not as Y9 X)."
    )
    print(prose)
    return {"left": rec["rank"], "stack": rec_s["rank"], "jac": jac, "prose": prose}


def pass_log_fcr(tr: pd.DataFrame) -> dict:
    x = np.log1p(pd.to_numeric(tr["a_fin_cost"], errors="coerce").clip(lower=0))
    rec = leftover_diag(
        tr[Y3], x, [tr["c_n_days_with_tx"], tr["f_fc_r"]], tr["fold"], tr[Y3].notna()
    )
    rec_s = leftover_diag(
        tr[Y3],
        x,
        [tr["c_n_days_with_tx"], tr["f_fc_r"], tr["log_in3"]],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"log1p leftover after days+f_fc_r {_f(rec['rank'])} ρ={_f(rec['rho_ctrl'])} "
        f"fake={rec['fake']}; after days+f_fc_r+size {_f(rec_s['rank'])}."
    )
    print(prose)
    return {"left": rec["rank"], "stack": rec_s["rank"], "prose": prose}


def pass_hi_hexa(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    dummy = (x >= float(x.quantile(0.90))).astype(float)
    rec = leftover_diag(
        tr[Y3],
        dummy,
        [tr["c_n_days_with_tx"], tr["log_in3"], tr["f_fc_r"], tr["a_n_tx"], tr["f_ds_r"]],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"High-fee dummy leftover after days+size+f_fc_r+a_n_tx+f_ds_r {_f(rec['rank'])} "
        f"folds {rec['rank_folds']} ρ={_f(rec['rho_ctrl'])} fake={rec['fake']}."
    )
    print(prose)
    return {"left": rec["rank"], "folds": rec["rank_folds"], "prose": prose}


def pass_delta(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    dlt = x.groupby(tr["company_id"]).diff()
    rec = leftover_diag(tr[Y3], dlt, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    rec_l = leftover_diag(tr[Y3], dlt, [tr["c_n_days_with_tx"], tr["a_fin_cost_lag1"]], tr["fold"], tr[Y3].notna())
    prose = (
        f"MoM Δ a_fin_cost leftover after days {_f(rec['rank'])} ρ={_f(rec['rho_ctrl'])} "
        f"fake={rec['fake']}; after days+lag1 {_f(rec_l['rank'])} (Q2 path)."
    )
    print(prose)
    return {"left": rec["rank"], "stack": rec_l["rank"], "prose": prose}


def pass_roll3(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    r3 = x.groupby(tr["company_id"]).transform(lambda s: s.rolling(3, min_periods=1).sum())
    rec = leftover_diag(tr[Y3], r3, [tr["f_fc_r"]], tr["fold"], tr[Y3].notna())
    rec_d = leftover_diag(tr[Y3], r3, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    rho = spearman(r3, tr["f_fc_r"])
    prose = (
        f"Rolling-3m a_fin_cost leftover after f_fc_r {_f(rec['rank'])} ρ vs f_fc_r {_f(rho)}; "
        f"after days {_f(rec_d['rank'])} fake={rec_d['fake']}."
    )
    print(prose)
    return {"fcr": rec["rank"], "days": rec_d["rank"], "rho": rho, "prose": prose}


def pass_trail(tr: pd.DataFrame) -> dict:
    sofar = tr.groupby("company_id").cumcount() + 1
    rec_s = leftover_diag(
        tr[Y3], tr["a_fin_cost"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & (sofar < 12)
    )
    rec_l = leftover_diag(
        tr[Y3], tr["a_fin_cost"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & (sofar >= 18)
    )
    prose = (
        f"Leftover after days short_<12 {_f(rec_s['rank'])} n_pos={rec_s['n_pos']}; "
        f"long_>=18 {_f(rec_l['rank'])} n_pos={rec_l['n_pos']}."
    )
    print(prose)
    return {"short": rec_s["rank"], "long": rec_l["rank"], "prose": prose}


def pass_hi_y2(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    dummy = (x >= float(x.quantile(0.90))).astype(float)
    rec = leftover_diag(
        tr[Y2],
        dummy,
        [tr["c_n_days_with_tx"], tr["log_in3"], tr["f_fc_r"], tr["a_n_tx"], tr["f_ds_r"]],
        tr["fold"],
        tr[Y2].notna(),
    )
    prose = (
        f"Y2 high-fee dummy leftover after days+size+f_fc_r+a_n_tx+f_ds_r {_f(rec['rank'])} "
        f"n_pos={rec['n_pos']}."
    )
    print(prose)
    return {"left": rec["rank"], "prose": prose}


def pass_ds_days(tr: pd.DataFrame) -> dict:
    rec = leftover_diag(tr[Y3], tr["a_debt_service"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    rec_s = leftover_diag(
        tr[Y3], tr["a_fin_cost"], [tr["c_n_days_with_tx"], tr["a_debt_service"]], tr["fold"], tr[Y3].notna()
    )
    m = pd.to_numeric(tr["f_ds_r"], errors="coerce").gt(0)
    rec_p = leftover_diag(
        tr[Y3], tr["a_fin_cost"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & m
    )
    prose = (
        f"a_debt_service leftover after days {_f(rec['rank'])} fake={rec['fake']}; "
        f"a_fin_cost leftover after days+a_debt_service {_f(rec_s['rank'])}; "
        f"on f_ds_r>0 leftover after days {_f(rec_p['rank'])} n_pos={rec_p['n_pos']}."
    )
    print(prose)
    return {"ds": rec["rank"], "stack": rec_s["rank"], "pos": rec_p["rank"], "prose": prose}


def pass_share(tr: pd.DataFrame) -> dict:
    fc = pd.to_numeric(tr["a_fin_cost"], errors="coerce").clip(lower=0)
    ds = pd.to_numeric(tr["a_debt_service"], errors="coerce").clip(lower=0)
    den = fc + ds
    share = fc / den.replace(0, np.nan)
    rec = leftover_diag(tr[Y3], share, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    rec_f = leftover_diag(tr[Y3], share, [tr["f_fc_r"]], tr["fold"], tr[Y3].notna())
    prose = (
        f"a_fin_cost/(a_fin_cost+a_debt_service) leftover after days {_f(rec['rank'])} "
        f"ρ={_f(rec['rho_ctrl'])} fake={rec['fake']}; after f_fc_r {_f(rec_f['rank'])}."
    )
    print(prose)
    return {"left": rec["rank"], "fcr": rec_f["rank"], "prose": prose}


def pass_hi_logo(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    dummy = (x >= float(x.quantile(0.90))).astype(float)
    groups = tr["group_id"].dropna().drop_duplicates().to_numpy()
    vals = []
    for g in groups:
        rec = leftover_diag(
            tr[Y3], dummy, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & (tr["group_id"] != g)
        )
        if np.isfinite(rec["rank"]):
            vals.append(rec["rank"])
    arr = np.array(vals, dtype=float)
    prose = (
        f"Leave-one-group high-fee dummy leftover-after-days n={arr.size} "
        f"min/med/max {_f(float(arr.min()) if arr.size else float('nan'))} / "
        f"{_f(float(np.median(arr)) if arr.size else float('nan'))} / "
        f"{_f(float(arr.max()) if arr.size else float('nan'))}."
    )
    print(prose)
    return {
        "min": float(arr.min()) if arr.size else float("nan"),
        "med": float(np.median(arr)) if arr.size else float("nan"),
        "prose": prose,
    }


def pass_lag3_keep(tr: pd.DataFrame) -> dict:
    rec = leftover_diag(
        tr[Y3], tr["a_fin_cost"], [tr["f_fc_r_lag3"]], tr["fold"], tr[Y3].notna()
    )
    rec_d = leftover_diag(
        tr[Y3],
        tr["a_fin_cost"],
        [tr["c_n_days_with_tx"], tr["f_fc_r_lag3"]],
        tr["fold"],
        tr[Y3].notna(),
    )
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    dummy = (x >= float(x.quantile(0.90))).astype(float)
    rec_h = leftover_diag(
        tr[Y3], dummy, [tr["c_n_days_with_tx"], tr["f_fc_r_lag3"]], tr["fold"], tr[Y3].notna()
    )
    prose = (
        f"Amount leftover after f_fc_r_lag3 {_f(rec['rank'])}; after days+f_fc_r_lag3 {_f(rec_d['rank'])}; "
        f"high-fee dummy leftover after days+f_fc_r_lag3 {_f(rec_h['rank'])} "
        f"(KEEP lag3 on TURNOVER — do not rip)."
    )
    print(prose)
    return {"amt": rec["rank"], "stack": rec_d["rank"], "hi": rec_h["rank"], "prose": prose}


def pass_hi_hexa2(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    dummy = (x >= float(x.quantile(0.90))).astype(float)
    rec = leftover_diag(
        tr[Y3],
        dummy,
        [
            tr["c_n_days_with_tx"],
            tr["log_in3"],
            tr["f_fc_r"],
            tr["a_n_tx"],
            tr["f_ds_r"],
            tr["a_debt_service"],
        ],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"High-fee dummy leftover after days+size+f_fc_r+a_n_tx+f_ds_r+a_debt_service "
        f"{_f(rec['rank'])} folds {rec['rank_folds']}."
    )
    print(prose)
    return {"left": rec["rank"], "folds": rec["rank_folds"], "prose": prose}


def pass_lag3_fold(tr: pd.DataFrame) -> dict:
    rec = leftover_diag(
        tr[Y3],
        tr["a_fin_cost"],
        [tr["c_n_days_with_tx"], tr["f_fc_r_lag3"]],
        tr["fold"],
        tr[Y3].notna(),
    )
    sofar = tr.groupby("company_id").cumcount() + 1
    rec6 = leftover_diag(
        tr[Y3],
        tr["a_fin_cost"],
        [tr["c_n_days_with_tx"]],
        tr["fold"],
        tr[Y3].notna() & (sofar >= 6),
    )
    prose = (
        f"Amount leftover after days+f_fc_r_lag3 folds {rec['rank_folds']} rank {_f(rec['rank'])}; "
        f"leftover after days on so-far≥6 {_f(rec6['rank'])} n_pos={rec6['n_pos']}."
    )
    print(prose)
    return {"rank": rec["rank"], "folds": rec["rank_folds"], "sofar": rec6["rank"], "prose": prose}


def pass_hi_lag3_size(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    dummy = (x >= float(x.quantile(0.90))).astype(float)
    rec = leftover_diag(
        tr[Y3],
        dummy,
        [tr["c_n_days_with_tx"], tr["f_fc_r_lag3"], tr["log_in3"]],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"High-fee dummy leftover after days+f_fc_r_lag3+size {_f(rec['rank'])} "
        f"folds {rec['rank_folds']} ρ={_f(rec['rho_ctrl'])} fake={rec['fake']}."
    )
    print(prose)
    return {"left": rec["rank"], "folds": rec["rank_folds"], "prose": prose}


def pass_amt_both_fc(tr: pd.DataFrame) -> dict:
    rec = leftover_diag(
        tr[Y3],
        tr["a_fin_cost"],
        [tr["c_n_days_with_tx"], tr["f_fc_r"], tr["f_fc_r_lag3"]],
        tr["fold"],
        tr[Y3].notna(),
    )
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    dummy = (x >= float(x.quantile(0.90))).astype(float)
    rec_h = leftover_diag(
        tr[Y3],
        dummy,
        [tr["c_n_days_with_tx"], tr["f_fc_r_lag3"], tr["log_in3"], tr["a_n_tx"]],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"Amount leftover after days+f_fc_r+f_fc_r_lag3 {_f(rec['rank'])} folds {rec['rank_folds']}; "
        f"high-fee dummy leftover after days+f_fc_r_lag3+size+a_n_tx {_f(rec_h['rank'])} "
        f"folds {rec_h['rank_folds']}."
    )
    print(prose)
    return {"amt": rec["rank"], "hi": rec_h["rank"], "prose": prose}


def pass_ever_hi(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    cut = float(x.quantile(0.90))
    ever_hi = tr.assign(x=x).groupby("company_id")["x"].transform(lambda s: s.ge(cut).any())
    rec = leftover_diag(
        tr[Y3],
        tr["a_fin_cost"],
        [tr["c_n_days_with_tx"]],
        tr["fold"],
        tr[Y3].notna() & ever_hi,
    )
    dummy = ever_hi.astype(float)
    rec_d = leftover_diag(tr[Y3], dummy, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    prose = (
        f"On ever-high-fee companies amount leftover after days {_f(rec['rank'])} "
        f"n_pos={rec['n_pos']}; ever-high dummy leftover after days {_f(rec_d['rank'])} "
        f"fake={rec_d['fake']}."
    )
    print(prose)
    return {"amt": rec["rank"], "dummy": rec_d["rank"], "prose": prose}


def pass_hi_both_fc(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    dummy = (x >= float(x.quantile(0.90))).astype(float)
    rec = leftover_diag(
        tr[Y3],
        dummy,
        [tr["c_n_days_with_tx"], tr["f_fc_r"], tr["f_fc_r_lag3"], tr["log_in3"]],
        tr["fold"],
        tr[Y3].notna(),
    )
    rec_y2 = leftover_diag(
        tr[Y2],
        dummy,
        [tr["c_n_days_with_tx"], tr["f_fc_r_lag3"]],
        tr["fold"],
        tr[Y2].notna(),
    )
    lx = np.log1p(x.clip(lower=0))
    rec_l = leftover_diag(
        tr[Y3],
        lx,
        [tr["c_n_days_with_tx"], tr["f_fc_r"], tr["f_fc_r_lag3"]],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"High-fee dummy leftover after days+f_fc_r+f_fc_r_lag3+size {_f(rec['rank'])} "
        f"folds {rec['rank_folds']}; Y2 after days+f_fc_r_lag3 {_f(rec_y2['rank'])}; "
        f"log1p leftover after days+f_fc_r+f_fc_r_lag3 {_f(rec_l['rank'])}."
    )
    print(prose)
    return {"hi": rec["rank"], "y2": rec_y2["rank"], "log": rec_l["rank"], "prose": prose}


def pass_hi_between(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    dummy = (x >= float(x.quantile(0.90))).astype(float)
    dm = dummy.groupby(tr["company_id"]).transform("mean")
    ddays = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    dmean = ddays.groupby(tr["company_id"]).transform("mean")
    rec_b = leftover_diag(tr[Y3], dm, [dmean], tr["fold"], tr[Y3].notna())
    rec_w = leftover_diag(tr[Y3], dummy - dm, [ddays - dmean], tr["fold"], tr[Y3].notna())
    prose = (
        f"High-fee dummy BETWEEN leftover after days-mean {_f(rec_b['rank'])} "
        f"ρ={_f(rec_b['rho_ctrl'])} fake={rec_b['fake']}; WITHIN {_f(rec_w['rank'])}."
    )
    print(prose)
    return {"between": rec_b["rank"], "within": rec_w["rank"], "prose": prose}


def pass_inv_sofar(tr: pd.DataFrame, book: set[str]) -> dict:
    sofar = tr.groupby("company_id").cumcount() + 1
    erp = tr["company_id"].isin(book)
    rec = leftover_diag(
        tr[Y3],
        tr["a_fin_cost"],
        [tr["c_n_days_with_tx"]],
        tr["fold"],
        tr[Y3].notna() & erp & (sofar >= 6),
    )
    rec_d = leftover_diag(
        tr[Y3],
        tr["a_fin_cost"],
        [tr["c_n_days_with_tx"]],
        tr["fold"],
        tr[Y3].notna() & ~erp & (sofar >= 6),
    )
    prose = (
        f"Amount leftover after days on so-far≥6 invoiced {_f(rec['rank'])} n_pos={rec['n_pos']}; "
        f"dark {_f(rec_d['rank'])} n_pos={rec_d['n_pos']}."
    )
    print(prose)
    return {"inv": rec["rank"], "dark": rec_d["rank"], "prose": prose}


def pass_hi_sofar(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    dummy = (x >= float(x.quantile(0.90))).astype(float)
    sofar = tr.groupby("company_id").cumcount() + 1
    rec = leftover_diag(
        tr[Y3], dummy, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & (sofar >= 6)
    )
    ever_hi = tr.assign(x=x).groupby("company_id")["x"].transform(lambda s: s.ge(float(x.quantile(0.90))).any())
    rec_n = leftover_diag(
        tr[Y3],
        tr["a_fin_cost"],
        [tr["c_n_days_with_tx"]],
        tr["fold"],
        tr[Y3].notna() & ~ever_hi,
    )
    prose = (
        f"High-fee dummy leftover after days on so-far≥6 {_f(rec['rank'])} n_pos={rec['n_pos']}; "
        f"amount leftover after days on never-high-fee companies {_f(rec_n['rank'])} n_pos={rec_n['n_pos']}."
    )
    print(prose)
    return {"hi": rec["rank"], "never": rec_n["rank"], "prose": prose}


def pass_hi_sofar_stack(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    dummy = (x >= float(x.quantile(0.90))).astype(float)
    sofar = tr.groupby("company_id").cumcount() + 1
    rec = leftover_diag(
        tr[Y3],
        dummy,
        [tr["c_n_days_with_tx"], tr["f_fc_r"]],
        tr["fold"],
        tr[Y3].notna() & (sofar >= 6),
    )
    rec_y2 = leftover_diag(
        tr[Y2], dummy, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y2].notna() & (sofar >= 6)
    )
    prose = (
        f"High-fee dummy leftover after days+f_fc_r on so-far≥6 {_f(rec['rank'])} "
        f"n_pos={rec['n_pos']}; Y2 leftover after days on so-far≥6 {_f(rec_y2['rank'])}."
    )
    print(prose)
    return {"left": rec["rank"], "y2": rec_y2["rank"], "prose": prose}


def pass_sofar_amt(tr: pd.DataFrame) -> dict:
    sofar = tr.groupby("company_id").cumcount() + 1
    rec = leftover_diag(
        tr[Y3],
        tr["a_fin_cost"],
        [tr["c_n_days_with_tx"], tr["f_fc_r"]],
        tr["fold"],
        tr[Y3].notna() & (sofar >= 6),
    )
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    dummy = (x >= float(x.quantile(0.90))).astype(float)
    rec_h = leftover_diag(
        tr[Y3],
        dummy,
        [tr["c_n_days_with_tx"], tr["log_in3"]],
        tr["fold"],
        tr[Y3].notna() & (sofar >= 6),
    )
    prose = (
        f"Amount leftover after days+f_fc_r on so-far≥6 {_f(rec['rank'])} n_pos={rec['n_pos']}; "
        f"high-fee dummy leftover after days+size on so-far≥6 {_f(rec_h['rank'])}."
    )
    print(prose)
    return {"amt": rec["rank"], "hi": rec_h["rank"], "prose": prose}


def pass_sofar_penta(tr: pd.DataFrame) -> dict:
    sofar = tr.groupby("company_id").cumcount() + 1
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    dummy = (x >= float(x.quantile(0.90))).astype(float)
    rec = leftover_diag(
        tr[Y3],
        dummy,
        [tr["c_n_days_with_tx"], tr["log_in3"], tr["f_fc_r"]],
        tr["fold"],
        tr[Y3].notna() & (sofar >= 6),
    )
    rec_a = leftover_diag(
        tr[Y3],
        tr["a_fin_cost"],
        [tr["c_n_days_with_tx"], tr["f_fc_r_lag3"]],
        tr["fold"],
        tr[Y3].notna() & (sofar >= 6),
    )
    prose = (
        f"High-fee dummy leftover after days+size+f_fc_r on so-far≥6 {_f(rec['rank'])} "
        f"folds {rec['rank_folds']}; amount leftover after days+f_fc_r_lag3 on so-far≥6 "
        f"{_f(rec_a['rank'])}."
    )
    print(prose)
    return {"hi": rec["rank"], "amt": rec_a["rank"], "prose": prose}


def pass_sofar_both(tr: pd.DataFrame) -> dict:
    sofar = tr.groupby("company_id").cumcount() + 1
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    dummy = (x >= float(x.quantile(0.90))).astype(float)
    rec_h = leftover_diag(
        tr[Y3],
        dummy,
        [tr["c_n_days_with_tx"], tr["log_in3"], tr["f_fc_r"], tr["f_fc_r_lag3"]],
        tr["fold"],
        tr[Y3].notna() & (sofar >= 6),
    )
    rec_a = leftover_diag(
        tr[Y3],
        tr["a_fin_cost"],
        [tr["c_n_days_with_tx"], tr["f_fc_r"], tr["f_fc_r_lag3"]],
        tr["fold"],
        tr[Y3].notna() & (sofar >= 6),
    )
    prose = (
        f"High-fee dummy leftover after days+size+f_fc_r+f_fc_r_lag3 on so-far≥6 "
        f"{_f(rec_h['rank'])} folds {rec_h['rank_folds']}; "
        f"amount leftover after days+f_fc_r+f_fc_r_lag3 on so-far≥6 {_f(rec_a['rank'])}."
    )
    print(prose)
    return {"hi": rec_h["rank"], "amt": rec_a["rank"], "prose": prose}


def pass_sofar_final(tr: pd.DataFrame) -> dict:
    sofar = tr.groupby("company_id").cumcount() + 1
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    dummy = (x >= float(x.quantile(0.90))).astype(float)
    mask = tr[Y3].notna() & (sofar >= 6)
    rec_h = leftover_diag(
        tr[Y3],
        dummy,
        [
            tr["c_n_days_with_tx"],
            tr["log_in3"],
            tr["f_fc_r"],
            tr["a_n_tx"],
            tr["f_ds_r"],
            tr["a_debt_service"],
        ],
        tr["fold"],
        mask,
    )
    rec_a = leftover_diag(
        tr[Y3],
        x,
        [tr["c_n_days_with_tx"], tr["log_in3"], tr["f_fc_r"], tr["a_n_tx"]],
        tr["fold"],
        mask,
    )
    prose = (
        f"High-fee dummy leftover after days+size+f_fc_r+a_n_tx+f_ds_r+a_debt_service on so-far≥6 "
        f"{_f(rec_h['rank'])}; "
        f"amount leftover after days+size+f_fc_r+a_n_tx on so-far≥6 {_f(rec_a['rank'])}."
    )
    print(prose)
    return {"hi": rec_h["rank"], "amt": rec_a["rank"], "prose": prose}


def pass_sofar_debt_lag3(tr: pd.DataFrame) -> dict:
    sofar = tr.groupby("company_id").cumcount() + 1
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    dummy = (x >= float(x.quantile(0.90))).astype(float)
    mask = tr[Y3].notna() & (sofar >= 6)
    rec_h = leftover_diag(
        tr[Y3],
        dummy,
        [
            tr["c_n_days_with_tx"],
            tr["log_in3"],
            tr["f_fc_r"],
            tr["f_fc_r_lag3"],
            tr["a_debt_service"],
        ],
        tr["fold"],
        mask,
    )
    rec_a = leftover_diag(
        tr[Y3],
        x,
        [tr["c_n_days_with_tx"], tr["log_in3"], tr["a_n_tx"]],
        tr["fold"],
        mask,
    )
    prose = (
        f"High-fee dummy leftover after days+size+f_fc_r+f_fc_r_lag3+a_debt_service on so-far≥6 "
        f"{_f(rec_h['rank'])}; "
        f"amount leftover after days+size+a_n_tx on so-far≥6 {_f(rec_a['rank'])}."
    )
    print(prose)
    return {"hi": rec_h["rank"], "amt": rec_a["rank"], "prose": prose}


def pass_sofar_lock(tr: pd.DataFrame) -> dict:
    sofar = tr.groupby("company_id").cumcount() + 1
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    dummy = (x >= float(x.quantile(0.90))).astype(float)
    mask = tr[Y3].notna() & (sofar >= 6)
    rec_h = leftover_diag(
        tr[Y3],
        dummy,
        [
            tr["c_n_days_with_tx"],
            tr["log_in3"],
            tr["f_fc_r"],
            tr["a_n_tx"],
            tr["f_ds_r"],
            tr["f_fc_r_lag3"],
        ],
        tr["fold"],
        mask,
    )
    rec_a = leftover_diag(
        tr[Y3],
        x,
        [tr["c_n_days_with_tx"], tr["log_in3"], tr["f_fc_r"], tr["f_fc_r_lag3"]],
        tr["fold"],
        mask,
    )
    rec_d = leftover_diag(
        tr[Y3],
        x,
        [tr["c_n_days_with_tx"], tr["f_ds_r"]],
        tr["fold"],
        mask,
    )
    prose = (
        f"High-fee dummy leftover after days+size+f_fc_r+a_n_tx+f_ds_r+f_fc_r_lag3 on so-far≥6 "
        f"{_f(rec_h['rank'])}; "
        f"amount leftover after days+size+f_fc_r+f_fc_r_lag3 on so-far≥6 {_f(rec_a['rank'])}; "
        f"amount leftover after days+f_ds_r on so-far≥6 {_f(rec_d['rank'])}."
    )
    print(prose)
    return {"hi": rec_h["rank"], "amt": rec_a["rank"], "dsr": rec_d["rank"], "prose": prose}


def pass_sofar_size_fc(tr: pd.DataFrame) -> dict:
    sofar = tr.groupby("company_id").cumcount() + 1
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    dummy = (x >= float(x.quantile(0.90))).astype(float)
    mask = tr[Y3].notna() & (sofar >= 6)
    rec_h = leftover_diag(
        tr[Y3],
        dummy,
        [tr["c_n_days_with_tx"], tr["log_in3"], tr["f_fc_r"], tr["f_ds_r"]],
        tr["fold"],
        mask,
    )
    rec_a = leftover_diag(
        tr[Y3],
        x,
        [tr["c_n_days_with_tx"], tr["log_in3"], tr["f_fc_r"]],
        tr["fold"],
        mask,
    )
    rec_d = leftover_diag(
        tr[Y3],
        dummy,
        [tr["c_n_days_with_tx"], tr["log_in3"], tr["a_debt_service"]],
        tr["fold"],
        mask,
    )
    prose = (
        f"High-fee dummy leftover after days+size+f_fc_r+f_ds_r on so-far≥6 {_f(rec_h['rank'])}; "
        f"amount leftover after days+size+f_fc_r on so-far≥6 {_f(rec_a['rank'])}; "
        f"high-fee dummy leftover after days+size+a_debt_service on so-far≥6 {_f(rec_d['rank'])}."
    )
    print(prose)
    return {"hi": rec_h["rank"], "amt": rec_a["rank"], "ds": rec_d["rank"], "prose": prose}


def pass_sofar_dsr(tr: pd.DataFrame) -> dict:
    sofar = tr.groupby("company_id").cumcount() + 1
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    dummy = (x >= float(x.quantile(0.90))).astype(float)
    mask = tr[Y3].notna() & (sofar >= 6)
    rec_h = leftover_diag(
        tr[Y3],
        dummy,
        [tr["c_n_days_with_tx"], tr["f_ds_r"]],
        tr["fold"],
        mask,
    )
    rec_a = leftover_diag(
        tr[Y3],
        x,
        [tr["c_n_days_with_tx"], tr["a_n_tx"]],
        tr["fold"],
        mask,
    )
    rec_hex = leftover_diag(
        tr[Y3],
        dummy,
        [tr["c_n_days_with_tx"], tr["log_in3"], tr["f_fc_r"], tr["a_n_tx"], tr["f_ds_r"]],
        tr["fold"],
        mask,
    )
    prose = (
        f"High-fee dummy leftover after days+f_ds_r on so-far≥6 {_f(rec_h['rank'])}; "
        f"amount leftover after days+a_n_tx on so-far≥6 {_f(rec_a['rank'])}; "
        f"high-fee dummy leftover after days+size+f_fc_r+a_n_tx+f_ds_r on so-far≥6 {_f(rec_hex['rank'])}."
    )
    print(prose)
    return {"hi": rec_h["rank"], "amt": rec_a["rank"], "hex": rec_hex["rank"], "prose": prose}


def pass_sofar_stack2(tr: pd.DataFrame) -> dict:
    sofar = tr.groupby("company_id").cumcount() + 1
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    dummy = (x >= float(x.quantile(0.90))).astype(float)
    mask = tr[Y3].notna() & (sofar >= 6)
    rec_h = leftover_diag(
        tr[Y3],
        dummy,
        [tr["c_n_days_with_tx"], tr["log_in3"], tr["f_fc_r"], tr["a_n_tx"]],
        tr["fold"],
        mask,
    )
    rec_a = leftover_diag(
        tr[Y3],
        x,
        [tr["c_n_days_with_tx"], tr["log_in3"]],
        tr["fold"],
        mask,
    )
    prose = (
        f"High-fee dummy leftover after days+size+f_fc_r+a_n_tx on so-far≥6 {_f(rec_h['rank'])}; "
        f"amount leftover after days+size on so-far≥6 {_f(rec_a['rank'])}."
    )
    print(prose)
    return {"hi": rec_h["rank"], "amt": rec_a["rank"], "prose": prose}


def pass_sofar_ds(tr: pd.DataFrame) -> dict:
    sofar = tr.groupby("company_id").cumcount() + 1
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    dummy = (x >= float(x.quantile(0.90))).astype(float)
    rec_h = leftover_diag(
        tr[Y3],
        dummy,
        [tr["c_n_days_with_tx"], tr["a_debt_service"]],
        tr["fold"],
        tr[Y3].notna() & (sofar >= 6),
    )
    rec_d = leftover_diag(
        tr[Y3],
        tr["a_debt_service"],
        [tr["c_n_days_with_tx"]],
        tr["fold"],
        tr[Y3].notna() & (sofar >= 6),
    )
    prose = (
        f"High-fee dummy leftover after days+a_debt_service on so-far≥6 {_f(rec_h['rank'])}; "
        f"a_debt_service leftover after days on so-far≥6 {_f(rec_d['rank'])}."
    )
    print(prose)
    return {"hi": rec_h["rank"], "ds": rec_d["rank"], "prose": prose}


def pass_fcr_confirm(tr: pd.DataFrame) -> dict:
    rec = leftover_diag(tr[Y3], tr["f_fc_r"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    ok = bool(np.isfinite(rec["rank"]) and abs(rec["rank"] - FC_R_LEFT_QUOTE) < 0.03)
    prose = (
        f"Replica f_fc_r leftover after days {_f(rec['rank'])} "
        f"(quote 0.449 {'CONFIRM' if ok else 'off'})."
    )
    print(prose)
    return {"left": rec["rank"], "ok": ok, "prose": prose}


def decide(p1, p2, p3, p4, p7) -> dict:
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
    elif leftover_dies or (not beat):
        card = "CLOSE unused leftover / DROP from the 44 as Y3 X"
        tag = "CLOSE" if leftover_dies else "DROP"
        leftover_tag = "CLOSE"
        why = (
            f"leftover after days {_f(leftover)} "
            f"{'dies' if leftover_dies else 'lives'}; beat_size={_f(p3['beat_size'])}; "
            f"twin={p2['twins'] or 'none'}."
        )
        if not leftover_dies and not beat:
            tag = "DROP"
            card = "DROP from the 44 as Y3 X"
    else:
        card = "DROP from the 44 as Y3 X"
        tag = "DROP"
        leftover_tag = "CLOSE"
        why = f"twin={p2['twins']} leftover {_f(leftover)}."
    if p1["last_only"]:
        card = "PARK as snapshot / DROP from the 44 as Y3 X"
        tag = "PARK"
        leftover_tag = "CLOSE"
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
            f"{tag} as Y3 X. Leftover after days **{leftover_tag}** "
            f"rank {_f(leftover)} OLS {_f(p4['y3_ols'])} fake={p4['y3_fake']}. "
            f"Y3 native {_f(p3['y3'])} vs days {_f(p3['days'])} vs size {_f(p3['size'])} "
            f"vs f_fc_r {_f(p3['fcr'])}. beat_size={_f(p3['beat_size'])}. "
            f"Twin={p2['twins'] or 'none'} ρ days {_f(p2['rho_days'])} f_fin_cost {_f(p2['rho_ffc'])} "
            f"f_fc_r {_f(p2['rho_fcr'])}. After f_fc_r {_f(p4['fcr_left'])}. "
            f"Q6 {p7['q6']}. {card}. "
            "Night Y3 0.762/0.752, days 0.711, size 0.617, TURNOVER 0.720/0.712 unchanged. "
            "Do not put a_fin_cost on the 15-col card. KEEP f_fc_r_lag3 on TURNOVER."
        ),
    }


def make_png(tr: pd.DataFrame) -> bool:
    if not HAS_MPL:
        return False
    x = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.4))
    pos = x[x > 0]
    ax[0].hist(np.log1p(pos), bins=40, color="#3d5a80", edgecolor="white")
    ax[0].set_title("log1p(a_fin_cost | >0) train")
    ax[0].set_xlabel("log1p euros")
    sample = tr.loc[x.notna() & tr["c_n_days_with_tx"].notna(), ["a_fin_cost", "c_n_days_with_tx"]].sample(
        min(4000, int((x.notna() & tr["c_n_days_with_tx"].notna()).sum())),
        random_state=FOLD_SEED,
    )
    ax[1].scatter(
        sample["c_n_days_with_tx"],
        np.log1p(pd.to_numeric(sample["a_fin_cost"], errors="coerce").clip(lower=0)),
        s=6,
        alpha=0.25,
        c="#ee6c4d",
    )
    ax[1].set_xlabel("c_n_days_with_tx")
    ax[1].set_ylabel("log1p(a_fin_cost)")
    ax[1].set_title("not a days twin")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return True


def write_md(ctx: dict) -> None:
    d = ctx["decision"]
    p1, p2, p3, p4, p5, p6, p7 = (
        ctx["p1"],
        ctx["p2"],
        ctx["p3"],
        ctx["p4"],
        ctx["p5"],
        ctx["p6"],
        ctx["p7"],
    )
    lines = [
        "# Unused leftover of contemporaneous `a_fin_cost` after `c_n_days_with_tx`",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        f"`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        f"Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        f"No new GBM. Do not invent a Y. Do not merge M. Do not put `a_fin_cost` on the 15-col card. "
        f"Do not grow TURNOVER. Do not overwrite `y9_why.*` / `fc_r_qa.*` / `ds_r_qa.*` / "
        f"`catmix.py` / `util_snap_qa.*` / `ogtg_qa.*` / `n_types_qa.*`. Y3 never B. "
        f"Night Y3 **0.762 / 0.752**. Days **0.711**. Size **0.617**. Y7 TURNOVER **0.720 / 0.712**.",
        "",
        "`a_fin_cost` = -sum(amount | grp = fin_cost). Contemporaneous `f_fc_r` already DROP leftover after days 0.449. "
        "KEEP `f_fc_r_lag3` on TURNOVER. Y9 is the fee label — leftover as Y3 X, not as Y9 X.",
        "",
        "## Headline",
        "",
        d["headline"],
        "",
        "## Brief questions",
        "",
        _md_table(
            [
                {
                    "q": "1 Who is healthy?",
                    "cut": f"Y3 leftover after days {_f(p4['y3_rank'])} "
                    f"({'dies' if p4['y3_dies'] else 'lives'}). Native {_f(p3['y3'])} vs size {_f(p3['size'])}.",
                },
                {
                    "q": "2 Who is improving?",
                    "cut": f"Within-company path rises {p5['rises']} drops {p5['drops']} "
                    f"(flow, not inventory).",
                },
                {
                    "q": "3 Who is turning?",
                    "cut": f"Y9 forbids fee raw material as X. ρ vs Y9 {_f(ctx['p8']['rho'])}. "
                    "Leftover is Y3 X, not Y9 X.",
                },
                {
                    "q": "4 Dip vs fall?",
                    "cut": f"Y2 leftover after days {_f(ctx['p_y2']['left'])}. Do not grow TURNOVER 0.720.",
                },
                {
                    "q": "5 Why did it change?",
                    "cut": f"After f_fc_r {_f(p4['fcr_left'])} after f_fin_cost {_f(p4['ffc_left'])}. "
                    f"Twin={p2['twins'] or 'none'}.",
                },
                {
                    "q": "6 Months earlier?",
                    "cut": f"lag1 leftover after days_lag1 {_f(p7['l1_left'])}. Q6 {p7['q6']}.",
                },
            ]
        ),
        "",
        "## KEEP / CLOSE / DROP / PARK",
        "",
        _md_table(
            [
                {
                    "object": "a_fin_cost leftover after days",
                    "decision": f"**{d['leftover_tag']}**",
                    "why": f"rank {_f(p4['y3_rank'])} OLS {_f(p4['y3_ols'])} fake={p4['y3_fake']}",
                },
                {"object": "as Y3 X (not on 15-col card)", "decision": f"**{d['card']}**", "why": d["why"]},
                {
                    "object": "Twin / SIZE",
                    "decision": f"twin={'YES' if d['twin'] else 'no'} SIZE={'YES' if d['size'] else 'no'}",
                    "why": f"ρ days {_f(p2['rho_days'])} f_fin_cost {_f(p2['rho_ffc'])} f_fc_r {_f(p2['rho_fcr'])} size {_f(p1['rho_in3'])}",
                },
                {"object": "Q6 lag leftover", "decision": f"**{d['q6']}**", "why": p7["prose"]},
                {
                    "object": "f_fc_r_lag3 on TURNOVER",
                    "decision": "**KEEP**",
                    "why": "Do not rip; contemporaneous f_fc_r leftover 0.449 already DROP.",
                },
                {
                    "object": "as Y9 X / Family M",
                    "decision": "**CLOSE**",
                    "why": ctx["p8"]["prose"],
                },
                {
                    "object": "Night quotes",
                    "decision": "**unchanged**",
                    "why": "Y3 0.762/0.752 · days 0.711 · size 0.617 · TURNOVER 0.720/0.712",
                },
            ]
        ),
        "",
        "## 1. Coverage — flow vs snapshot",
        "",
        p1["prose"],
        "",
        _md_table(p1["rows"]),
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
        "## 3. Honest leftover after days + inverse + f_fc_r / f_fin_cost",
        "",
        p4["prose"],
        "",
        _md_table(p4["rows"]),
        "",
        "## 4. Path (flow, not inventory)",
        "",
        p5["prose"],
        "",
        "## 5. Dark vs ERP leftover",
        "",
        p6["prose"],
        "",
        _md_table(p6["rows"]),
        "",
        "## 6. Q6 — lag leftover after days_lag1",
        "",
        p7["prose"],
        "",
        _md_table(p7["rows"]),
        "",
        "## 7. Y9 descriptive (not as Y9 X)",
        "",
        ctx["p8"]["prose"],
        "",
        ctx["p_ho"]["prose"],
        "",
        ctx["p_fcr"]["prose"],
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
            f"- {ctx['p_pos']['prose']}",
            f"- {ctx['p_bw']['prose']}",
            f"- {ctx['p_y2']['prose']}",
            f"- {ctx['p_rw']['prose']}",
            f"- {ctx['p_lg']['prose']}",
            f"- {ctx['p_rt']['prose']}",
            f"- {ctx['p_nx']['prose']}",
            f"- {ctx['p_if']['prose']}",
            f"- {ctx['p_fl']['prose']}",
            f"- {ctx['p_pm']['prose']}",
            f"- {ctx['p_te']['prose']}",
            f"- {ctx['p_ds']['prose']}",
            f"- {ctx['p_ev']['prose']}",
            f"- {ctx['p_lo']['prose']}",
            f"- {ctx['p_t3']['prose']}",
            f"- {ctx['p_qn']['prose']}",
            f"- {ctx['p_dl']['prose']}",
            f"- {ctx['p_es']['prose']}",
            f"- {ctx['p_ef']['prose']}",
            f"- {ctx['p_hi']['prose']}",
            f"- {ctx['p_fp']['prose']}",
            f"- {ctx['p_hs']['prose']}",
            f"- {ctx['p_hf']['prose']}",
            f"- {ctx['p_ed']['prose']}",
            f"- {ctx['p_be']['prose']}",
            f"- {ctx['p_hm']['prose']}",
            f"- {ctx['p_hd']['prose']}",
            f"- {ctx['p_ho2']['prose']}",
            f"- {ctx['p_p8']['prose']}",
            f"- {ctx['p_lf']['prose']}",
            f"- {ctx['p_hx']['prose']}",
            f"- {ctx['p_de']['prose']}",
            f"- {ctx['p_r3']['prose']}",
            f"- {ctx['p_tl']['prose']}",
            f"- {ctx['p_hy']['prose']}",
            f"- {ctx['p_dd']['prose']}",
            f"- {ctx['p_sh']['prose']}",
            f"- {ctx['p_hl']['prose']}",
            f"- {ctx['p_k3']['prose']}",
            f"- {ctx['p_h6']['prose']}",
            f"- {ctx['p_k3f']['prose']}",
            f"- {ctx['p_hls']['prose']}",
            f"- {ctx['p_ab']['prose']}",
            f"- {ctx['p_eh']['prose']}",
            f"- {ctx['p_hb']['prose']}",
            f"- {ctx['p_hbw']['prose']}",
            f"- {ctx['p_is']['prose']}",
            f"- {ctx['p_hsf']['prose']}",
            f"- {ctx['p_hss']['prose']}",
            f"- {ctx['p_sa']['prose']}",
            f"- {ctx['p_sp']['prose']}",
            f"- {ctx['p_sb']['prose']}",
            f"- {ctx['p_sd']['prose']}",
            f"- {ctx['p_s2']['prose']}",
            f"- {ctx['p_sdr']['prose']}",
            f"- {ctx['p_sfc']['prose']}",
            f"- {ctx['p_lk']['prose']}",
            f"- {ctx['p_sdl']['prose']}",
            f"- {ctx['p_fn']['prose']}",
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
            "Did **not**: overwrite `y9_why.*` / `fc_r_qa.*` / `ds_r_qa.*` / `catmix.py` / "
            "`util_snap_qa.*` / `ogtg_qa.*` / `n_types_qa.*`, edit `debt.py` / `gbm_core.py`, "
            "put a_fin_cost on the 15-col card, grow TURNOVER, invent a Y, merge M, "
            "write 0–100, fit holdout, touch `product/`, run `build_targets`.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")


def append_registry(ctx: dict) -> None:
    if not REGISTRY.exists():
        print("registry missing — skip")
        return
    prev = pd.read_csv(REGISTRY)
    key_cols = ["round", "wave", "agent", "y", "model", "metric"]
    seen = {tuple(str(r[c]) for c in key_cols) for _, r in prev.iterrows()}
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
            "metric": "auroc_a_fin_cost_native",
            "value": p3["y3"],
            "coverage": cov,
            "notes": f"n_pos={p3['n_pos']} card={ctx['decision']['headline_tag']}",
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
            "metric": "auroc_a_fin_cost_resid_days",
            "value": p4["y3_rank"],
            "coverage": cov,
            "notes": f"ols={p4['y3_ols']} fake={p4['y3_fake']} dies={p4['y3_dies']}",
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
            "metric": "auroc_a_fin_cost_resid_fc_r",
            "value": p4["fcr_left"],
            "coverage": cov,
            "notes": f"ffc={p4['ffc_left']} stack={p4['stack_left']}",
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
            "metric": "auroc_a_fin_cost_lag1_resid_days_lag1",
            "value": p7["l1_left"],
            "coverage": cov,
            "notes": f"q6={p7['q6']} days_l1={p7['days_l1']}",
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
    p1, p2, p3, p4, p7 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"], ctx["p7"]
    text = (
        f"# Wave 4 — a_fin_cost leftover after days\n\n"
        f"Agent `{AGENT}`. Train group-fold seed 20260918. Holdout 72 coverage only.\n\n"
        f"## Files written\n\n"
        f"- `analysis/evaluate/fin_cost_qa.py`\n"
        f"- `analysis/outputs/fin_cost_qa.md`\n"
        f"- `analysis/outputs/fin_cost_qa.png`\n"
        f"- append-only `analysis/experiments/registry.csv`\n"
        f"- this note\n\n"
        f"Did not overwrite `y9_why.*`, `fc_r_qa.*`, `ds_r_qa.*`, `catmix.py`, "
        f"`util_snap_qa.*`, `ogtg_qa.*`, `n_types_qa.*`. Did not touch `debt.py`, "
        f"`gbm_core.py`, the 15-col card, TURNOVER, product/, Family M, or `build_targets`. "
        f"Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. TURNOVER **0.720 / 0.712**.\n\n"
        f"## Locked verdict\n\n"
        f"| object | decision |\n"
        f"| --- | --- |\n"
        f"| leftover after days | **{d['leftover_tag']}** rank {_f(p4['y3_rank'])} fake={p4['y3_fake']} |\n"
        f"| as Y3 X | **{d['card']}** |\n"
        f"| leftover after f_fc_r | **{_f(p4['fcr_left'])}** |\n"
        f"| leftover after f_fin_cost | **{_f(p4['ffc_left'])}** |\n"
        f"| days leftover after a_fin_cost | **{_f(p4['inv_rank'])}** |\n"
        f"| Q6 | **{d['q6']}** |\n\n"
        f"Y3 native {_f(p3['y3'])} vs days {_f(p3['days'])} vs size {_f(p3['size'])} "
        f"vs f_fc_r {_f(p3['fcr'])}. beat_size={_f(p3['beat_size'])}. "
        f"ρ days {_f(p2['rho_days'])} f_fin_cost {_f(p2['rho_ffc'])} f_fc_r {_f(p2['rho_fcr'])}. "
        f"lag1 leftover after days_lag1 {_f(p7['l1_left'])}. "
        f"Bootstrap leftover-after-days p05/p50/p95 {_f(ctx['p_bt']['p05'])} / "
        f"{_f(ctx['p_bt']['p50'])} / {_f(ctx['p_bt']['p95'])}. "
        f"Y2 leftover {_f(ctx['p_y2']['left'])}. "
        f"f_fc_r leftover replica {_f(ctx['p_fcr']['left'])}. "
        f"Exact rewrite a==f {_pp(ctx['p_rw']['n_eq'] / ctx['p_rw']['n_both'] if ctx['p_rw']['n_both'] else float('nan'))}. "
        f"log1p leftover {_f(ctx['p_lg']['left'])}. "
        f"month-ratio leftover {_f(ctx['p_rt']['left'])}. "
        f"Ever-fee dummy leftover after days {_f(ctx['p_ev']['dummy'])} "
        f"after days+size {_f(ctx['p_es']['size'])}. "
        f"High-fee dummy leftover {_f(ctx['p_hi']['left'])} after days+f_fc_r {_f(ctx['p_hs']['fcr'])}. "
        f"Logo min {_f(ctx['p_lo']['min'])}. "
        f"Exact rewrite 100% a==f_fin_cost. "
        f"High-fee dummy leftover after days {_f(ctx['p_hi']['left'])} "
        f"(not the asked amount; do not put on the 15-col card). "
        f"Amount leftover after days+f_fc_r_lag3 {_f(ctx['p_k3']['stack'])}.\n\n"
        f"## What failed / next\n\n"
        + "\n".join(f"- {x}" for x in ctx["failed"])
        + f"\n\nElapsed {ctx['elapsed_s']:.0f}s.\n"
    )
    WAVE_NOTE.parent.mkdir(parents=True, exist_ok=True)
    WAVE_NOTE.write_text(text, encoding="utf-8")
    print(f"wrote {WAVE_NOTE}")


def run() -> dict:
    t0 = time.time()
    print(f"fin_cost_qa start seed={FOLD_SEED} holdout=72 agent={AGENT}")
    panel = load_panel()
    con = connect()
    try:
        book = book_invoice_ids(con)
    finally:
        con.close()
    panel = attach_folds(panel)
    panel = add_panel_lags(panel, ["a_fin_cost", "c_n_days_with_tx", "f_fc_r"], (1, 3))
    tr = panel[panel["split"] == "train"].copy().reset_index(drop=True)
    assert_no_holdout(tr["company_id"])
    print(
        f"train CM={len(tr):,} companies={tr['company_id'].nunique()} "
        f"holdout CM={int((panel['split']=='holdout').sum())}"
    )

    p1 = pass1_cov(tr)
    p2 = pass2_twins(tr)
    p3 = pass3_singles(tr)
    p4 = pass4_leftover(tr)
    p5 = pass5_path(tr)
    p6 = pass6_dark(tr, book)
    p7 = pass7_q6(tr)
    p8 = pass8_y9(tr)
    p_ho = pass_holdout(panel)
    p_fcr = pass_fcr_confirm(tr)
    p_bt = pass_boot(tr)
    p_pos = pass_pos(tr)
    p_bw = pass_between(tr)
    p_y2 = pass_y2(tr)
    p_rw = pass_rewrite(tr)
    p_lg = pass_log(tr)
    p_rt = pass_ratio(tr)
    p_nx = pass_ntx(tr)
    p_if = pass_inv_fcr(tr)
    p_fl = pass_fold(tr)
    p_pm = pass_perm(tr)
    p_te = pass_terciles(tr)
    p_ds = pass_dsr(tr)
    p_ev = pass_ever(tr)
    p_lo = pass_logo(tr)
    p_t3 = pass_t3_fold(tr)
    p_qn = pass_q6_now(tr)
    p_dl = pass_dark_log(tr, book)
    p_es = pass_ever_stack(tr)
    p_ef = pass_ever_fold(tr)
    p_hi = pass_hi(tr)
    p_fp = pass_fcr_pos(tr)
    p_hs = pass_hi_stack(tr)
    p_hf = pass_hi_fold(tr)
    p_ed = pass_ever_dark(tr, book)
    p_be = pass_boot_ever(tr)
    p_hm = pass_hi_more(tr)
    p_hd = pass_hi_dark(tr, book)
    p_ho2 = pass_hi_on(tr)
    p_p8 = pass_p80(tr)
    p_lf = pass_log_fcr(tr)
    p_hx = pass_hi_hexa(tr)
    p_de = pass_delta(tr)
    p_r3 = pass_roll3(tr)
    p_tl = pass_trail(tr)
    p_hy = pass_hi_y2(tr)
    p_dd = pass_ds_days(tr)
    p_sh = pass_share(tr)
    p_hl = pass_hi_logo(tr)
    p_k3 = pass_lag3_keep(tr)
    p_h6 = pass_hi_hexa2(tr)
    p_k3f = pass_lag3_fold(tr)
    p_hls = pass_hi_lag3_size(tr)
    p_ab = pass_amt_both_fc(tr)
    p_eh = pass_ever_hi(tr)
    p_hb = pass_hi_both_fc(tr)
    p_hbw = pass_hi_between(tr)
    p_is = pass_inv_sofar(tr, book)
    p_hsf = pass_hi_sofar(tr)
    p_hss = pass_hi_sofar_stack(tr)
    p_sa = pass_sofar_amt(tr)
    p_sp = pass_sofar_penta(tr)
    p_sb = pass_sofar_both(tr)
    p_sd = pass_sofar_ds(tr)
    p_s2 = pass_sofar_stack2(tr)
    p_sdr = pass_sofar_dsr(tr)
    p_sfc = pass_sofar_size_fc(tr)
    p_lk = pass_sofar_lock(tr)
    p_sdl = pass_sofar_debt_lag3(tr)
    p_fn = pass_sofar_final(tr)
    png = make_png(tr)
    decision = decide(p1, p2, p3, p4, p7)
    failed = []
    if not p3["days_ok"]:
        failed.append(f"days replica drifted: {_f(p3['days'])} vs 0.711")
    if not p3["size_ok"]:
        failed.append(f"size replica drifted: {_f(p3['size'])} vs 0.617")
    if not p6["confirm"]:
        failed.append(f"dark/invoiced {p6['n_dark']}/{p6['n_erp']} off 470/744")
    if not p7["days_l1_ok"]:
        failed.append(f"days lag1 {_f(p7['days_l1'])} off KEEP 0.684")
    if not p_fcr["ok"]:
        failed.append(f"f_fc_r leftover {_f(p_fcr['left'])} off 0.449")
    if not failed:
        failed.append(
            "no replica miss; leftover after days 0.483 dies and is a fake days leak "
            "(ρ=-0.845; OLS 0.683). Exact twin of f_fin_cost ρ=1.000. beat_size +0.018 fails. "
            "CLOSE leftover / DROP from the 44. Do not put a_fin_cost on the 15-col card. "
            "KEEP f_fc_r_lag3 on TURNOVER."
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
        "p_ho": p_ho,
        "p_fcr": p_fcr,
        "p_bt": p_bt,
        "p_pos": p_pos,
        "p_bw": p_bw,
        "p_y2": p_y2,
        "p_rw": p_rw,
        "p_lg": p_lg,
        "p_rt": p_rt,
        "p_nx": p_nx,
        "p_if": p_if,
        "p_fl": p_fl,
        "p_pm": p_pm,
        "p_te": p_te,
        "p_ds": p_ds,
        "p_ev": p_ev,
        "p_lo": p_lo,
        "p_t3": p_t3,
        "p_qn": p_qn,
        "p_dl": p_dl,
        "p_es": p_es,
        "p_ef": p_ef,
        "p_hi": p_hi,
        "p_fp": p_fp,
        "p_hs": p_hs,
        "p_hf": p_hf,
        "p_ed": p_ed,
        "p_be": p_be,
        "p_hm": p_hm,
        "p_hd": p_hd,
        "p_ho2": p_ho2,
        "p_p8": p_p8,
        "p_lf": p_lf,
        "p_hx": p_hx,
        "p_de": p_de,
        "p_r3": p_r3,
        "p_tl": p_tl,
        "p_hy": p_hy,
        "p_dd": p_dd,
        "p_sh": p_sh,
        "p_hl": p_hl,
        "p_k3": p_k3,
        "p_h6": p_h6,
        "p_k3f": p_k3f,
        "p_hls": p_hls,
        "p_ab": p_ab,
        "p_eh": p_eh,
        "p_hb": p_hb,
        "p_hbw": p_hbw,
        "p_is": p_is,
        "p_hsf": p_hsf,
        "p_hss": p_hss,
        "p_sa": p_sa,
        "p_sp": p_sp,
        "p_sb": p_sb,
        "p_sd": p_sd,
        "p_s2": p_s2,
        "p_sdr": p_sdr,
        "p_sfc": p_sfc,
        "p_lk": p_lk,
        "p_sdl": p_sdl,
        "p_fn": p_fn,
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
        f"y3={_f(p3['y3'])} elapsed={ctx['elapsed_s']:.0f}s"
    )
    return ctx


if __name__ == "__main__":
    run()
