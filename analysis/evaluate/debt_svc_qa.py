"""Unused leftover of contemporaneous ``a_debt_service`` after days as Y3 X.

``a_debt_service`` = -sum(amount | grp = debt_service) this month (Family A).
Store also has ``f_debt_service`` / ``f_ds_r`` / ``a_fin_cost`` / ``f_fc_r``.
``f_ds_r`` leftover after days already DROP 0.528. Contemporaneous
``f_fc_r`` DROP leftover 0.449. KEEP ``f_fc_r_lag3`` on TURNOVER.
Do **not** overwrite ``ds_r_qa.*`` / ``fc_r_qa.*`` / ``op_out_qa.*``.
Do **not** put ``a_debt_service`` on the 15-col card. Do not grow TURNOVER.

KEEP-as-X: beat size ≥0.02 **and** leftover after days **and** not SIZE
(|ρ| vs log1p(a_in3) ≥0.50) **and** not a twin (|ρ|≥0.80 vs days /
a_n_tx / f_ds_r / f_debt_service / a_fin_cost / a_op_out). Leftover
<0.55 dies. Rank leftover is honest; OLS can fake a days leak.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.debt_svc_qa

Owned: analysis/evaluate/debt_svc_qa.py, analysis/outputs/debt_svc_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_debt_svc.md (end).
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
OUT_MD = ANALYSIS / "outputs" / "debt_svc_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "debt_svc_qa.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_debt_svc.md"
AGENT = "572fb928"
WAVE = "4"
ROUND = "R4"
MODEL = "debt_svc_qa"
X_FAM = "A"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
Y7 = "y7_top1_lost"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_QUOTE = 0.617
DAYS_LAG1_QUOTE = 0.684
DS_R_LEFT = 0.528
FC_R_LEFT = 0.449
Y3_NIGHT = (0.762, 0.752)
Y7_TURNOVER = 0.720
Y7_B_SHALLOW = 0.712
KEEP_DELTA = 0.02
TWIN_RHO = 0.80
SIZE_RHO = 0.50
CHANCE = 0.55
FAKE_DAYS_RHO = 0.40
MIN_POS = 50
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
    "a_debt_service",
    "f_debt_service",
    "f_ds_r",
    "a_fin_cost",
    "f_fc_r",
    "a_op_out",
    "a_op_in",
)

Y_KEEP = (Y2, Y3, Y7)
FLAG = "a_debt_service"
TWIN_COLS = (
    "c_n_days_with_tx",
    "a_n_tx",
    "f_ds_r",
    "f_debt_service",
    "a_fin_cost",
    "a_op_out",
)


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
            "train_sign": 0, "train_auc": float("nan"),
            "n_defined": n, "n_pos": n_pos, "n_neg": n_neg, "low_power": True,
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
    panel["log_ds"] = np.log1p(pd.to_numeric(panel[FLAG], errors="coerce").clip(lower=0))
    leak3 = leakage_check(
        [FLAG, "f_ds_r", "f_debt_service", "c_n_days_with_tx", "log_in3"],
        Y3, forbidden_prefixes=["b"],
    )
    if not leak3["ok"]:
        raise RuntimeError(f"Y3 never-B leak: {leak3['issues']}")
    leak7 = leakage_check([FLAG, "f_ds_r"], Y7, forbidden_prefixes=["d"])
    if not leak7["ok"]:
        raise RuntimeError(f"Y7 never-D leak: {leak7['issues']}")
    return panel


def attach_folds(panel: pd.DataFrame) -> pd.DataFrame:
    train = panel[panel["split"] == "train"][["company_id", "group_id"]].drop_duplicates()
    assert_no_holdout(train["company_id"])
    folds = group_folds(train, n=N_FOLDS, seed=FOLD_SEED)
    return panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")


def pass1_cov(tr: pd.DataFrame, book: set[str]) -> dict:
    print("\n" + "=" * 72)
    print("CUT 1 — coverage; twin / SIZE")
    print("=" * 72)
    n_cm, n_co = len(tr), int(tr["company_id"].nunique())
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    nn = int(x.notna().sum())
    eq0 = _pct(int((x == 0).sum()), nn)
    p50 = float(x.median()) if nn else float("nan")
    dark = ~tr["company_id"].isin(book)
    p50_d = float(x[dark].median()) if dark.any() else float("nan")
    p50_e = float(x[~dark].median()) if (~dark).any() else float("nan")
    same = float((x - pd.to_numeric(tr["f_debt_service"], errors="coerce")).abs().max())
    rhos = {c: spearman(x, tr[c]) for c in TWIN_COLS}
    rhos["log_in3"] = spearman(x, tr["log_in3"])
    rhos["f_fc_r"] = spearman(x, tr["f_fc_r"])
    gate_twins = [c for c in TWIN_COLS if np.isfinite(rhos[c]) and abs(rhos[c]) >= TWIN_RHO]
    is_size = bool(np.isfinite(rhos["log_in3"]) and abs(rhos["log_in3"]) >= SIZE_RHO)
    rows = [{"col": FLAG, "n_nn": f"{nn:,}", "cov": _pp(_pct(nn, n_cm)), "eq0": _pp(eq0), "p50": _f(p50)}]
    rho_rows = [
        {
            "vs": k,
            "rho": _f(v),
            "flag": "SIZE" if k == "log_in3" and is_size else ("TWIN" if abs(v) >= TWIN_RHO else "no"),
        }
        for k, v in rhos.items()
    ]
    prose = (
        f"Train {n_co:,} co / {n_cm:,} CM. {FLAG} nn={nn:,} cov {_pp(_pct(nn, n_cm))} "
        f"eq0 {_pp(eq0)} p50={_f(p50)}. Dark panel p50={_f(p50_d)} ERP p50={_f(p50_e)}. "
        f"max |a_debt_service - f_debt_service|={_f(same)}. "
        f"ρ vs days {_f(rhos['c_n_days_with_tx'])} vs a_n_tx {_f(rhos['a_n_tx'])} "
        f"vs f_ds_r {_f(rhos['f_ds_r'])} vs f_debt_service {_f(rhos['f_debt_service'])} "
        f"vs a_fin_cost {_f(rhos['a_fin_cost'])} vs a_op_out {_f(rhos['a_op_out'])} "
        f"vs size {_f(rhos['log_in3'])} vs f_fc_r {_f(rhos['f_fc_r'])}. "
        f"SIZE={is_size} gate_twins={gate_twins or 'none'}."
    )
    print(prose)
    return {
        "rows": rows, "rho_rows": rho_rows, "rhos": rhos, "is_size": is_size,
        "gate_twins": gate_twins, "twin_gate": bool(gate_twins),
        "n_cm": n_cm, "n_co": n_co, "cov": _pct(nn, n_cm), "p50": p50,
        "p50_d": p50_d, "p50_e": p50_e, "same": same, "prose": prose,
    }


def pass2_singles(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 2 — single-feature group-fold Y3")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    feats = {
        FLAG: tr[FLAG],
        "log1p(a_debt_service)": tr["log_ds"],
        "f_debt_service": tr["f_debt_service"],
        "f_ds_r": tr["f_ds_r"],
        "a_fin_cost": tr["a_fin_cost"],
        "f_fc_r": tr["f_fc_r"],
        "a_op_out": tr["a_op_out"],
        "a_n_tx": tr["a_n_tx"],
        "log1p(a_in3)": tr["log_in3"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
    }
    recs, rows = {}, []
    for name, x in feats.items():
        rec = signed_oof_auroc(y, x, tr["fold"], lab)
        recs[name] = rec
        rows.append(_auc_row(Y3, name, rec))
        print(f"  {name} CV={_f(_cv(rec))} n={rec['n_defined']} pos={rec['n_pos']} folds={fold_bits(rec)}")
    open_cv = _cv(recs[FLAG])
    days = _cv(recs["c_n_days_with_tx"])
    size = _cv(recs["log1p(a_in3)"])
    fds = _cv(recs["f_ds_r"])
    beat = bool(np.isfinite(open_cv) and (open_cv - SIZE_QUOTE) >= KEEP_DELTA)
    days_ok = bool(np.isfinite(days) and abs(days - DAYS_BENCH) < 0.008)
    size_ok = bool(np.isfinite(size) and abs(size - SIZE_QUOTE) < 0.008)
    prose = (
        f"Y3 {FLAG} {_f(open_cv)} n={recs[FLAG]['n_defined']:,} pos={recs[FLAG]['n_pos']:,}. "
        f"vs size {_f(size)} vs days {_f(days)} vs f_ds_r {_f(fds)} "
        f"vs f_debt_service {_f(_cv(recs['f_debt_service']))} "
        f"vs a_fin_cost {_f(_cv(recs['a_fin_cost']))} vs f_fc_r {_f(_cv(recs['f_fc_r']))} "
        f"vs a_op_out {_f(_cv(recs['a_op_out']))}. Replica days 0.711 "
        f"{'CONFIRM' if days_ok else 'DRIFT'} / size 0.617 {'CONFIRM' if size_ok else 'DRIFT'}. "
        f"Beat-size Δ={_f(open_cv - SIZE_QUOTE) if np.isfinite(open_cv) else '—'} "
        f"{'PASS' if beat else 'FAIL'}. f_ds_r leftover 0.528 DROP stays. Do not grow TURNOVER."
    )
    print(prose)
    return {
        "rows": rows, "open": open_cv, "days": days, "size": size, "fds": fds,
        "fdebt": _cv(recs["f_debt_service"]), "fcost": _cv(recs["a_fin_cost"]),
        "fcr": _cv(recs["f_fc_r"]), "opout": _cv(recs["a_op_out"]),
        "log": _cv(recs["log1p(a_debt_service)"]),
        "beat_size": beat, "days_ok": days_ok, "size_ok": size_ok, "prose": prose,
    }


def pass3_days(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 3 — honest leftover after days; inverse")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    inv = leftover_diag(y, tr["c_n_days_with_tx"], (tr[FLAG],), tr["fold"], lab)
    prose = (
        f"{FLAG} leftover after days OLS {_f(after['ols'])} rank {_f(after['rank'])} "
        f"fake={after['fake']} almost={after['almost']} ρ(resid,days)={_f(after['rho_ctrl'])} "
        f"R²={_f(after['r2'])} honest_dies={after['honest_dies']} n={after['n']:,} pos={after['n_pos']:,}. "
        f"Inverse: days leftover after {FLAG} OLS {_f(inv['ols'])} rank {_f(inv['rank'])} "
        f"dies={inv['honest_dies']}."
    )
    print(prose)
    return {
        "after": after, "inv": inv, "ols": after["ols"], "rank": after["rank"],
        "dies": after["honest_dies"], "fake": after["fake"], "r2": after["r2"],
        "inv_rank": inv["rank"], "inv_dies": inv["honest_dies"], "prose": prose,
    }


def pass4_dsr(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 4 — leftover after f_ds_r (ratio twin?)")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after = leftover_diag(y, tr[FLAG], (tr["f_ds_r"],), tr["fold"], lab)
    after_d = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["f_ds_r"]), tr["fold"], lab)
    inv = leftover_diag(y, tr["f_ds_r"], (tr[FLAG],), tr["fold"], lab)
    rewrite = bool(np.isfinite(after["rank"]) and after["rank"] < CHANCE)
    prose = (
        f"leftover after f_ds_r rank {_f(after['rank'])} OLS {_f(after['ols'])} "
        f"dies={after['honest_dies']} fake={after['fake']} rewrite={rewrite}. "
        f"after days+f_ds_r {_f(after_d['rank'])}. f_ds_r leftover after {FLAG} {_f(inv['rank'])}. "
        f"f_ds_r leftover after days 0.528 DROP stays."
    )
    print(prose)
    return {
        "rank": after["rank"], "ols": after["ols"], "dies": after["honest_dies"],
        "fake": after["fake"], "rewrite": rewrite, "both": after_d["rank"],
        "inv": inv["rank"], "prose": prose,
        "rows": [
            {"cut": "after f_ds_r", "rank": _f(after["rank"]), "ols": _f(after["ols"]), "fake": after["fake"]},
            {"cut": "after days+f_ds_r", "rank": _f(after_d["rank"]), "ols": _f(after_d["ols"]), "fake": after_d["fake"]},
            {"cut": "f_ds_r after a_debt_service", "rank": _f(inv["rank"]), "ols": _f(inv["ols"]), "fake": inv["fake"]},
        ],
    }


def pass5_size(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 5 — leftover after size / days+size")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after_s = leftover_diag(y, tr[FLAG], (tr["log_in3"],), tr["fold"], lab)
    after_b = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], lab)
    prose = (
        f"leftover after size {_f(after_s['rank'])} dies={after_s['honest_dies']} fake={after_s['fake']} "
        f"ρ={_f(after_s['rho_ctrl'])}. after days+size {_f(after_b['rank'])} "
        f"dies={after_b['honest_dies']} fake={after_b['fake']}. Size bar 0.617 stays."
    )
    print(prose)
    return {"after_s": after_s["rank"], "after_b": after_b["rank"], "dies_s": after_s["honest_dies"], "dies_b": after_b["honest_dies"], "prose": prose}


def pass6_q6(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 6 — Q6 lag1 leftover after days_lag1")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    rec = signed_oof_auroc(y, tr[f"{FLAG}_lag1"], tr["fold"], lab)
    rec_d = signed_oof_auroc(y, tr["c_n_days_with_tx_lag1"], tr["fold"], lab)
    after = leftover_diag(y, tr[f"{FLAG}_lag1"], (tr["c_n_days_with_tx_lag1"],), tr["fold"], lab)
    days_ok = bool(np.isfinite(_cv(rec_d)) and abs(_cv(rec_d) - DAYS_LAG1_QUOTE) < 0.012)
    print(f"  {FLAG}_lag1 CV={_f(_cv(rec))} n={rec['n_defined']}")
    print(f"  c_n_days_with_tx_lag1 CV={_f(_cv(rec_d))}")
    prose = (
        f"Y3 {FLAG}_lag1 {_f(_cv(rec))} leftover after days_lag1 rank {_f(after['rank'])} "
        f"dies={after['honest_dies']}. Days lag1 {_f(_cv(rec_d))} "
        f"(quote 0.684 {'CONFIRM' if days_ok else 'DRIFT'})."
    )
    print(prose)
    return {
        "l1": _cv(rec), "l1_rank": after["rank"], "dies": after["honest_dies"],
        "days_l1": _cv(rec_d), "days_ok": days_ok, "prose": prose,
        "rows": [
            {"feat": f"{FLAG}_lag1", "Y3": _f(_cv(rec)), "leftover": _f(after["rank"])},
            {"feat": "days_lag1", "Y3": _f(_cv(rec_d)), "leftover": "—"},
        ],
    }


def pass7_dark(tr: pd.DataFrame, book: set[str]) -> dict:
    print("\n" + "=" * 72)
    print("CUT 7 — Dark vs ERP leftover")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    last = tr.groupby("company_id")["period"].transform("max") == pd.to_datetime(tr["period"])
    dark_co = set(tr.loc[~tr["company_id"].isin(book), "company_id"].unique())
    n_dark = len(dark_co)
    last_d = x[last & tr["company_id"].isin(dark_co)]
    last_e = x[last & tr["company_id"].isin(book)]
    p50_d = float(last_d.median()) if last_d.notna().any() else float("nan")
    p50_e = float(last_e.median()) if last_e.notna().any() else float("nan")
    dark = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], y.notna() & tr["company_id"].isin(dark_co))
    erp = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], y.notna() & tr["company_id"].isin(book))
    prose = (
        f"Dark {n_dark} (want {N_DARK_WANT}) last-month p50={_f(p50_d)} ERP last p50={_f(p50_e)}. "
        f"Dark leftover {_f(dark['rank'])} ERP leftover {_f(erp['rank'])}."
    )
    print(prose)
    return {
        "n_dark": n_dark, "p50_d": p50_d, "p50_e": p50_e,
        "dark": dark["rank"], "erp": erp["rank"], "prose": prose,
        "rows": [
            {"slice": "Dark last-month", "p50": _f(p50_d), "leftover": _f(dark["rank"]), "n": dark["n"]},
            {"slice": "ERP last-month", "p50": _f(p50_e), "leftover": _f(erp["rank"]), "n": erp["n"]},
        ],
    }


def pass8_hold(hold: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 8 — holdout coverage only (no AUROC)")
    print("=" * 72)
    x = pd.to_numeric(hold[FLAG], errors="coerce")
    n_co = int(hold["company_id"].nunique())
    prose = (
        f"Holdout {n_co} co / {len(hold)} CM nn={int(x.notna().sum())} "
        f"cov={_pp(_pct(int(x.notna().sum()), len(hold)))} p50={_f(float(x.median()) if x.notna().any() else float('nan'))} "
        f"(no fit, no AUROC)."
    )
    print(prose)
    return {"n_co": n_co, "prose": prose}


def extra_bootstrap(tr: pd.DataFrame, n_boot: int = 40) -> dict:
    print("\n" + "=" * 72)
    print(f"EXTRA — company bootstrap leftover after days (n={n_boot})")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    cos = tr.loc[y.notna(), "company_id"].drop_duplicates().to_numpy()
    by_co = {c: tr.index[tr["company_id"] == c].to_numpy() for c in cos}
    rng = np.random.default_rng(20260918)
    ranks = []
    for _ in range(n_boot):
        draw = rng.choice(cos, size=len(cos), replace=True)
        idx = np.concatenate([by_co[c] for c in draw])
        sl = tr.loc[idx]
        after = leftover_diag(
            pd.to_numeric(sl[Y3], errors="coerce"), sl[FLAG],
            (sl["c_n_days_with_tx"],), sl["fold"],
            pd.to_numeric(sl[Y3], errors="coerce").notna(),
        )
        if np.isfinite(after["rank"]):
            ranks.append(after["rank"])
    arr = np.array(ranks, dtype=float)
    p05, p50, p95 = (float(np.quantile(arr, q)) for q in (0.05, 0.50, 0.95)) if len(arr) else (float("nan"),) * 3
    share = float((arr < CHANCE).mean()) if len(arr) else float("nan")
    prose = (
        f"Bootstrap leftover-after-days rank p05={_f(p05)} p50={_f(p50)} p95={_f(p95)} "
        f"share<0.55={_pp(share)} n={len(arr)}."
    )
    print(prose)
    return {"p05": p05, "p50": p50, "p95": p95, "share": share, "prose": prose}


def extra_icc(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — ICC / median company Pearson acf1")
    print("=" * 72)
    d = tr[["company_id", "period", FLAG]].copy()
    d[FLAG] = pd.to_numeric(d[FLAG], errors="coerce")
    d = d.dropna(subset=[FLAG]).sort_values(["company_id", "period"])
    g = d.groupby("company_id")[FLAG]
    means = g.mean()
    grand = float(d[FLAG].mean())
    n_i = g.size()
    k = int(means.shape[0])
    n = int(len(d))
    ss_b = float(((means - grand) ** 2 * n_i).sum())
    ss_w = float(((d[FLAG] - d["company_id"].map(means)) ** 2).sum())
    var_b = ss_b / max(k - 1, 1)
    var_w = ss_w / max(n - k, 1)
    icc = var_b / (var_b + var_w) if (var_b + var_w) > 0 else float("nan")
    rs = []
    for _, gg in d.groupby("company_id"):
        if len(gg) < 4:
            continue
        a = gg[FLAG].to_numpy(dtype=float)
        b = gg[FLAG].shift(1).to_numpy(dtype=float)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
            continue
        rs.append(float(np.corrcoef(a[m], b[m])[0, 1]))
    med = float(np.median(rs)) if rs else float("nan")
    prose = (
        f"ICC={_f(icc)} median company Pearson acf1={_f(med)} n_co={len(rs)}. "
        f"{'BETWEEN trait' if np.isfinite(icc) and icc >= 0.80 else 'month shock / WITHIN'}."
    )
    print(prose)
    return {"icc": icc, "acf1": med, "prose": prose}


def extra_twins(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after f_debt_service / a_fin_cost / a_op_out / a_n_tx")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    rows = []
    store = {}
    for col in ("f_debt_service", "a_fin_cost", "a_op_out", "a_n_tx", "f_fc_r"):
        after = leftover_diag(y, tr[FLAG], (tr[col],), tr["fold"], lab)
        store[col] = after["rank"]
        rows.append({"control": col, "leftover": _f(after["rank"]), "ols": _f(after["ols"]), "fake": after["fake"], "dies": after["honest_dies"]})
        print(f"  leftover after {col} {_f(after['rank'])} dies={after['honest_dies']} fake={after['fake']}")
    prose = (
        f"leftover after f_debt_service {_f(store['f_debt_service'])} "
        f"a_fin_cost {_f(store['a_fin_cost'])} a_op_out {_f(store['a_op_out'])} "
        f"a_n_tx {_f(store['a_n_tx'])} f_fc_r {_f(store['f_fc_r'])}."
    )
    print(prose)
    return {"rows": rows, "store": store, "prose": prose}


def extra_demean(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of company-demeaned a_debt_service after days")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    demean = x - x.groupby(tr["company_id"]).transform("mean")
    rec = signed_oof_auroc(y, demean, tr["fold"], y.notna())
    after = leftover_diag(y, demean, (tr["c_n_days_with_tx"],), tr["fold"], y.notna())
    prose = (
        f"demeaned a_debt_service Y3 {_f(_cv(rec))} leftover after days {_f(after['rank'])} "
        f"dies={after['honest_dies']} fake={after['fake']}."
    )
    print(prose)
    return {"cv": _cv(rec), "rank": after["rank"], "dies": after["honest_dies"], "prose": prose}


def extra_zero_log(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — zero-ds dummy leftover; leftover of log1p(a_debt_service)")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    lab = y.notna()
    dum = (x == 0).astype(float).where(x.notna())
    rec_z = signed_oof_auroc(y, dum, tr["fold"], lab)
    after_z = leftover_diag(y, dum, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    rec_l = signed_oof_auroc(y, tr["log_ds"], tr["fold"], lab)
    after_l = leftover_diag(y, tr["log_ds"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    prose = (
        f"zero-ds dummy Y3 {_f(_cv(rec_z))} leftover after days {_f(after_z['rank'])} "
        f"fake={after_z['fake']}. log1p(a_debt_service) Y3 {_f(_cv(rec_l))} leftover {_f(after_l['rank'])} "
        f"dies={after_l['honest_dies']}."
    )
    print(prose)
    return {"z": after_z["rank"], "log": after_l["rank"], "prose": prose}


def extra_y7_y2(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Y7 leftover after days (no TURNOVER seat); Y2 report-only")
    print("=" * 72)
    y7 = pd.to_numeric(tr[Y7], errors="coerce")
    rec7 = signed_oof_auroc(y7, tr[FLAG], tr["fold"], y7.notna())
    after7 = leftover_diag(y7, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], y7.notna())
    y2 = pd.to_numeric(tr[Y2], errors="coerce")
    rec2 = signed_oof_auroc(y2, tr[FLAG], tr["fold"], y2.notna())
    after2 = leftover_diag(y2, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], y2.notna())
    prose = (
        f"Y7 {FLAG} {_f(_cv(rec7))} leftover after days {_f(after7['rank'])} "
        f"dies={after7['honest_dies']}. Do not grow TURNOVER 0.720. "
        f"Y2 {_f(_cv(rec2))} leftover {_f(after2['rank'])} dies={after2['honest_dies']}."
    )
    print(prose)
    return {"y7": _cv(rec7), "y7_left": after7["rank"], "y2": _cv(rec2), "y2_left": after2["rank"], "prose": prose}


def extra_pos_size(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover on a_debt_service>0; leftover by size tercile")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    med = tr.groupby("company_id")["log_in3"].median()
    cuts = med.quantile([1 / 3, 2 / 3])
    terc = pd.Series("T2", index=tr.index)
    mapped = tr["company_id"].map(med)
    terc[mapped <= cuts.iloc[0]] = "T1"
    terc[mapped > cuts.iloc[1]] = "T3"
    rows = []
    store = {}
    slices = (
        ("ds>0", y.notna() & (x > 0)),
        ("T1_small", y.notna() & (terc == "T1")),
        ("T2", y.notna() & (terc == "T2")),
        ("T3_large", y.notna() & (terc == "T3")),
    )
    for name, mask in slices:
        rec = signed_oof_auroc(y, tr[FLAG], tr["fold"], mask)
        after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], mask)
        store[name] = after["rank"]
        rows.append({"slice": name, "Y3": _f(_cv(rec)), "leftover": _f(after["rank"]), "fake": after["fake"], "n": rec["n_defined"], "n_pos": rec["n_pos"]})
        print(f"  {name} Y3={_f(_cv(rec))} leftover={_f(after['rank'])} fake={after['fake']} n={rec['n_defined']}")
    prose = (
        f"ds>0 leftover {_f(store['ds>0'])} T1 {_f(store['T1_small'])} "
        f"T2 {_f(store['T2'])} T3 {_f(store['T3_large'])}."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def extra_ratio_stack(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of a_debt_service / a_in3; leftover after days+f_ds_r+size")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    inn = pd.to_numeric(tr["a_in3"], errors="coerce")
    intensity = (x / inn.replace(0, np.nan)).where(x.notna() & inn.notna())
    rec_i = signed_oof_auroc(y, intensity, tr["fold"], lab)
    after_i = leftover_diag(y, intensity, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    stacked = leftover_diag(
        y, tr[FLAG],
        (tr["c_n_days_with_tx"], tr["log_in3"], tr["f_ds_r"]),
        tr["fold"], lab,
    )
    after_fdebt = leftover_diag(
        y, tr[FLAG], (tr["c_n_days_with_tx"], tr["f_debt_service"]), tr["fold"], lab,
    )
    prose = (
        f"a_debt_service/a_in3 Y3 {_f(_cv(rec_i))} leftover after days {_f(after_i['rank'])}. "
        f"after days+size+f_ds_r {_f(stacked['rank'])} fake={stacked['fake']}. "
        f"after days+f_debt_service {_f(after_fdebt['rank'])}."
    )
    print(prose)
    return {"int": after_i["rank"], "stack": stacked["rank"], "fdebt": after_fdebt["rank"], "prose": prose}


def extra_pos_stack(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover on ds>0 after days+f_ds_r / days+size")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    pos = y.notna() & (x > 0)
    after_r = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["f_ds_r"]), tr["fold"], pos)
    after_s = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], pos)
    after_o = leftover_diag(y, tr[FLAG], (tr["f_ds_r"],), tr["fold"], pos)
    rec = signed_oof_auroc(y, tr[FLAG], tr["fold"], pos)
    rho = spearman(tr.loc[pos, FLAG], tr.loc[pos, "f_ds_r"]) if pos.any() else float("nan")
    prose = (
        f"ds>0 Y3 {_f(_cv(rec))} leftover after days+f_ds_r {_f(after_r['rank'])} "
        f"after days+size {_f(after_s['rank'])} after f_ds_r {_f(after_o['rank'])} "
        f"ρ vs f_ds_r {_f(rho)} n={after_r['n']}."
    )
    print(prose)
    return {"both": after_r["rank"], "size": after_s["rank"], "dsr": after_o["rank"], "prose": prose}


def extra_ever_year(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover on ever-ds vs never-ds; leftover by year")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    lab = y.notna()
    ever = (x > 0).groupby(tr["company_id"]).transform("any")
    e = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & ever)
    n = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & ~ever)
    yr = pd.to_datetime(tr["period"]).dt.year
    y25 = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & (yr == 2025))
    y26 = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & (yr == 2026))
    prose = (
        f"ever-ds leftover {_f(e['rank'])} n={e['n']}. never-ds leftover {_f(n['rank'])}. "
        f"2025 leftover {_f(y25['rank'])} 2026 leftover {_f(y26['rank'])}."
    )
    print(prose)
    return {"ever": e["rank"], "never": n["rank"], "y25": y25["rank"], "y26": y26["rank"], "prose": prose}


def extra_dsr_replica(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of f_ds_r after days (replica 0.528); leftover of f_debt_service")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    dsr = leftover_diag(y, tr["f_ds_r"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    fdebt = leftover_diag(y, tr["f_debt_service"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    fcr = leftover_diag(y, tr["f_fc_r"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    ok = bool(np.isfinite(dsr["rank"]) and abs(dsr["rank"] - DS_R_LEFT) < 0.015)
    prose = (
        f"f_ds_r leftover after days {_f(dsr['rank'])} (quote 0.528 {'CONFIRM' if ok else 'DRIFT'}). "
        f"f_debt_service leftover {_f(fdebt['rank'])}. f_fc_r leftover {_f(fcr['rank'])} "
        f"(quote 0.449)."
    )
    print(prose)
    return {"dsr": dsr["rank"], "fdebt": fdebt["rank"], "fcr": fcr["rank"], "ok": ok, "prose": prose}


def extra_last_mom(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — last-labeled leftover; MoM Δ leftover; leftover after days+a_fin_cost")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    last_lab = tr.loc[lab].groupby("company_id")["period"].transform("max")
    last_m = pd.Series(False, index=tr.index)
    last_m.loc[lab] = pd.to_datetime(tr.loc[lab, "period"]) == last_lab
    rec_l = signed_oof_auroc(y, tr[FLAG], tr["fold"], last_m)
    after_l = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], last_m)
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    mom = x - x.groupby(tr["company_id"]).shift(1)
    rec_m = signed_oof_auroc(y, mom, tr["fold"], lab)
    after_m = leftover_diag(y, mom, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_fc = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["a_fin_cost"]), tr["fold"], lab)
    t3_med = tr.groupby("company_id")["log_in3"].median()
    cut = t3_med.quantile(2 / 3)
    t3 = leftover_diag(
        y, tr[FLAG], (tr["c_n_days_with_tx"], tr["f_ds_r"]),
        tr["fold"], lab & (tr["company_id"].map(t3_med) > cut),
    )
    prose = (
        f"last-labeled Y3 {_f(_cv(rec_l))} leftover {_f(after_l['rank'])} n={after_l['n']}. "
        f"MoM Δ Y3 {_f(_cv(rec_m))} leftover {_f(after_m['rank'])} dies={after_m['honest_dies']}. "
        f"after days+a_fin_cost {_f(after_fc['rank'])}. T3 leftover after days+f_ds_r {_f(t3['rank'])}."
    )
    print(prose)
    return {"last": after_l["rank"], "mom": after_m["rank"], "fc": after_fc["rank"], "t3": t3["rank"], "prose": prose}


def extra_has_ds(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of has-ds dummy after days+size; leftover of a_debt_service/a_op_out")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    has = (x > 0).astype(float).where(x.notna())
    rec = signed_oof_auroc(y, has, tr["fold"], lab)
    after = leftover_diag(y, has, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_s = leftover_diag(y, has, (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], lab)
    opin = pd.to_numeric(tr["a_op_out"], errors="coerce")
    ratio = (x / opin.replace(0, np.nan)).where(x.notna() & opin.notna())
    rec_r = signed_oof_auroc(y, ratio, tr["fold"], lab)
    after_r = leftover_diag(y, ratio, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    rnk = tr.loc[lab].groupby("company_id")["period"].rank(method="first", ascending=False)
    last3 = pd.Series(False, index=tr.index)
    last3.loc[lab] = rnk <= 3
    l3 = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], last3)
    prose = (
        f"has-ds dummy Y3 {_f(_cv(rec))} leftover after days {_f(after['rank'])} "
        f"after days+size {_f(after_s['rank'])}. a_debt_service/a_op_out Y3 {_f(_cv(rec_r))} "
        f"leftover {_f(after_r['rank'])}. last-3 leftover {_f(l3['rank'])} n={l3['n']}."
    )
    print(prose)
    return {"has": after["rank"], "ratio": after_r["rank"], "l3": l3["rank"], "prose": prose}


def extra_mom_stack(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — MoM leftover after days+f_ds_r; T3 leftover after days+size+f_ds_r")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    mom = x - x.groupby(tr["company_id"]).shift(1)
    after = leftover_diag(y, mom, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_d = leftover_diag(y, mom, (tr["c_n_days_with_tx"], tr["f_ds_r"]), tr["fold"], lab)
    after_s = leftover_diag(y, mom, (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], lab)
    t3_med = tr.groupby("company_id")["log_in3"].median()
    cut = t3_med.quantile(2 / 3)
    t3m = lab & (tr["company_id"].map(t3_med) > cut)
    t3 = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["log_in3"], tr["f_ds_r"]), tr["fold"], t3m)
    t3_days = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], t3m)
    prose = (
        f"MoM leftover after days {_f(after['rank'])} fake={after['fake']} "
        f"ρ(resid,days)={_f(after['rho_ctrl'])}. after days+f_ds_r {_f(after_d['rank'])} "
        f"after days+size {_f(after_s['rank'])}. T3 leftover after days {_f(t3_days['rank'])} "
        f"after days+size+f_ds_r {_f(t3['rank'])} n={t3['n']}."
    )
    print(prose)
    return {"mom": after["rank"], "t3": t3["rank"], "prose": prose}


def extra_roll_share(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of 3m rolling mean; leftover of ever-ds share; last-labeled ever-ds")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    roll = x.groupby(tr["company_id"]).transform(lambda s: s.rolling(3, min_periods=1).mean())
    rec_r = signed_oof_auroc(y, roll, tr["fold"], lab)
    after_r = leftover_diag(y, roll, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    has = (x > 0).astype(float).where(x.notna())
    share = has.groupby(tr["company_id"]).transform(lambda s: s.expanding().mean())
    rec_s = signed_oof_auroc(y, share, tr["fold"], lab)
    after_s = leftover_diag(y, share, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    ever = (x.fillna(0).groupby(tr["company_id"]).transform("max") > 0)
    last_lab = tr.loc[lab].groupby("company_id")["period"].transform("max")
    last_m = pd.Series(False, index=tr.index)
    last_m.loc[lab] = pd.to_datetime(tr.loc[lab, "period"]) == last_lab
    last_e = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], last_m & ever)
    rec_e = signed_oof_auroc(y, tr[FLAG], tr["fold"], last_m & ever)
    prose = (
        f"3m rolling Y3 {_f(_cv(rec_r))} leftover {_f(after_r['rank'])} dies={after_r['honest_dies']}. "
        f"ever-ds share Y3 {_f(_cv(rec_s))} leftover {_f(after_s['rank'])}. "
        f"last-labeled ever-ds Y3 {_f(_cv(rec_e))} leftover {_f(last_e['rank'])} n={last_e['n']}."
    )
    print(prose)
    return {"roll": after_r["rank"], "share": after_s["rank"], "last_e": last_e["rank"], "prose": prose}


def extra_t3_stack(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — T3 leftover after days+size / a_fin_cost; never-ds leftover fake?")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    t3_med = tr.groupby("company_id")["log_in3"].median()
    cut = t3_med.quantile(2 / 3)
    t3m = lab & (tr["company_id"].map(t3_med) > cut)
    after_s = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], t3m)
    after_fc = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["a_fin_cost"]), tr["fold"], t3m)
    after_all = leftover_diag(
        y, tr[FLAG],
        (tr["c_n_days_with_tx"], tr["log_in3"], tr["f_ds_r"], tr["a_fin_cost"]),
        tr["fold"], t3m,
    )
    rec_t = signed_oof_auroc(y, tr[FLAG], tr["fold"], t3m)
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    never = x.fillna(0).groupby(tr["company_id"]).transform("max") <= 0
    nv = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & never)
    prose = (
        f"T3 Y3 {_f(_cv(rec_t))} leftover after days+size {_f(after_s['rank'])} "
        f"after days+a_fin_cost {_f(after_fc['rank'])} after days+size+f_ds_r+a_fin_cost "
        f"{_f(after_all['rank'])} fake={after_all['fake']} n={after_all['n']}. "
        f"never-ds leftover {_f(nv['rank'])} fake={nv['fake']} ρ={_f(nv['rho_ctrl'])}."
    )
    print(prose)
    return {"t3s": after_s["rank"], "t3all": after_all["rank"], "never": nv["rank"], "prose": prose}


def extra_per_tx(tr: pd.DataFrame, book: set[str]) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of a_debt_service/a_n_tx; leftover after days+f_fc_r / days+a_n_tx+f_ds_r")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    ntx = pd.to_numeric(tr["a_n_tx"], errors="coerce")
    per = (x / ntx.replace(0, np.nan)).where(x.notna() & ntx.notna())
    rec = signed_oof_auroc(y, per, tr["fold"], lab)
    after = leftover_diag(y, per, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_fc = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["f_fc_r"]), tr["fold"], lab)
    after_n = leftover_diag(
        y, tr[FLAG], (tr["c_n_days_with_tx"], tr["a_n_tx"], tr["f_ds_r"]), tr["fold"], lab,
    )
    last = pd.to_datetime(tr["period"]) == tr.groupby("company_id")["period"].transform("max")
    dark_last = leftover_diag(
        y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"],
        lab & last & ~tr["company_id"].isin(book),
    )
    prose = (
        f"a_debt_service/a_n_tx Y3 {_f(_cv(rec))} leftover {_f(after['rank'])}. "
        f"after days+f_fc_r {_f(after_fc['rank'])}. after days+a_n_tx+f_ds_r {_f(after_n['rank'])}. "
        f"Dark last-labeled leftover {_f(dark_last['rank'])} n={dark_last['n']}."
    )
    print(prose)
    return {"per": after["rank"], "fc": after_fc["rank"], "n": after_n["rank"], "prose": prose}


def extra_t3_eat(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — T3 leftover after days+a_n_tx / a_op_out; T3 ds>0 leftover")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    t3_med = tr.groupby("company_id")["log_in3"].median()
    cut = t3_med.quantile(2 / 3)
    t3m = lab & (tr["company_id"].map(t3_med) > cut)
    after_n = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["a_n_tx"]), tr["fold"], t3m)
    after_o = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["a_op_out"]), tr["fold"], t3m)
    after_all = leftover_diag(
        y, tr[FLAG],
        (tr["c_n_days_with_tx"], tr["log_in3"], tr["f_ds_r"], tr["a_fin_cost"], tr["a_n_tx"]),
        tr["fold"], t3m,
    )
    pos = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], t3m & (x > 0))
    rec_p = signed_oof_auroc(y, tr[FLAG], tr["fold"], t3m & (x > 0))
    last_lab = tr.loc[lab].groupby("company_id")["period"].transform("max")
    last_m = pd.Series(False, index=tr.index)
    last_m.loc[lab] = pd.to_datetime(tr.loc[lab, "period"]) == last_lab
    last_t = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], last_m & t3m)
    prose = (
        f"T3 leftover after days+a_n_tx {_f(after_n['rank'])} after days+a_op_out {_f(after_o['rank'])} "
        f"after days+size+f_ds_r+a_fin_cost+a_n_tx {_f(after_all['rank'])} fake={after_all['fake']}. "
        f"T3 ds>0 Y3 {_f(_cv(rec_p))} leftover {_f(pos['rank'])} n={pos['n']}. "
        f"T3 last-labeled leftover {_f(last_t['rank'])} n={last_t['n']}."
    )
    print(prose)
    return {"ntx": after_n["rank"], "all": after_all["rank"], "pos": pos["rank"], "prose": prose}


def extra_within(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of within-company rank of a_debt_service; leftover after days+f_ds_r+a_n_tx")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    rnk = x.groupby(tr["company_id"]).rank(method="average", pct=True)
    rec = signed_oof_auroc(y, rnk, tr["fold"], lab)
    after = leftover_diag(y, rnk, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_d = leftover_diag(
        y, tr[FLAG], (tr["c_n_days_with_tx"], tr["f_ds_r"], tr["a_n_tx"]), tr["fold"], lab,
    )
    days_med = tr.groupby("company_id")["c_n_days_with_tx"].median()
    hi = leftover_diag(
        y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"],
        lab & (tr["company_id"].map(days_med) >= days_med.median()),
    )
    lo = leftover_diag(
        y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"],
        lab & (tr["company_id"].map(days_med) < days_med.median()),
    )
    prose = (
        f"within-co rank Y3 {_f(_cv(rec))} leftover {_f(after['rank'])} dies={after['honest_dies']}. "
        f"after days+f_ds_r+a_n_tx {_f(after_d['rank'])}. "
        f"high-days leftover {_f(hi['rank'])} low-days leftover {_f(lo['rank'])}."
    )
    print(prose)
    return {"rnk": after["rank"], "stack": after_d["rank"], "hi": hi["rank"], "prose": prose}


def extra_t3_twins(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of f_ds_r / a_fin_cost after days on T3; leftover of high-days after f_ds_r")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    t3_med = tr.groupby("company_id")["log_in3"].median()
    cut = t3_med.quantile(2 / 3)
    t3m = lab & (tr["company_id"].map(t3_med) > cut)
    dsr = leftover_diag(y, tr["f_ds_r"], (tr["c_n_days_with_tx"],), tr["fold"], t3m)
    fc = leftover_diag(y, tr["a_fin_cost"], (tr["c_n_days_with_tx"],), tr["fold"], t3m)
    rec_d = signed_oof_auroc(y, tr["f_ds_r"], tr["fold"], t3m)
    rec_f = signed_oof_auroc(y, tr["a_fin_cost"], tr["fold"], t3m)
    days_med = tr.groupby("company_id")["c_n_days_with_tx"].median()
    him = lab & (tr["company_id"].map(days_med) >= days_med.median())
    hi_d = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["f_ds_r"]), tr["fold"], him)
    hi_s = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["log_in3"], tr["f_ds_r"]), tr["fold"], him)
    y25 = leftover_diag(
        y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"],
        t3m & (pd.to_datetime(tr["period"]).dt.year == 2025),
    )
    y26 = leftover_diag(
        y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"],
        t3m & (pd.to_datetime(tr["period"]).dt.year == 2026),
    )
    prose = (
        f"T3 f_ds_r leftover after days {_f(dsr['rank'])} Y3 {_f(_cv(rec_d))}. "
        f"T3 a_fin_cost leftover {_f(fc['rank'])} Y3 {_f(_cv(rec_f))}. "
        f"high-days leftover after days+f_ds_r {_f(hi_d['rank'])} after days+size+f_ds_r {_f(hi_s['rank'])}. "
        f"T3 2025 leftover {_f(y25['rank'])} 2026 leftover {_f(y26['rank'])}."
    )
    print(prose)
    return {"dsr": dsr["rank"], "fc": fc["rank"], "hi": hi_d["rank"], "prose": prose}


def extra_lag_days(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of a_debt_service after contemporaneous days on lag1; leftover of lag1 after days")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    lag = tr.get(f"{FLAG}_lag1")
    after_l = leftover_diag(y, lag, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_ll = leftover_diag(y, lag, (tr["c_n_days_with_tx_lag1"],), tr["fold"], lab)
    rec = signed_oof_auroc(y, lag, tr["fold"], lab)
    rnk = pd.Series(np.nan, index=tr.index)
    rnk.loc[lab] = tr.loc[lab].groupby("company_id")["period"].rank(method="first", ascending=False)
    last3 = leftover_diag(
        y, tr[FLAG], (tr["c_n_days_with_tx"], tr["f_ds_r"]), tr["fold"], lab & (rnk <= 3),
    )
    prose = (
        f"a_debt_service_lag1 leftover after days {_f(after_l['rank'])} "
        f"after days_lag1 {_f(after_ll['rank'])} Y3 {_f(_cv(rec))}. "
        f"last-3 leftover after days+f_ds_r {_f(last3['rank'])} n={last3['n']}."
    )
    print(prose)
    return {"l_days": after_l["rank"], "l_l1": after_ll["rank"], "l3": last3["rank"], "prose": prose}


def extra_t3_dsr_eat(tr: pd.DataFrame, book: set[str]) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — T3 leftover of f_ds_r after days+size / a_n_tx; ERP leftover; leftover after days+f_ds_r+a_op_out")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    t3_med = tr.groupby("company_id")["log_in3"].median()
    cut = t3_med.quantile(2 / 3)
    t3m = lab & (tr["company_id"].map(t3_med) > cut)
    dsr_s = leftover_diag(y, tr["f_ds_r"], (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], t3m)
    dsr_n = leftover_diag(y, tr["f_ds_r"], (tr["c_n_days_with_tx"], tr["a_n_tx"]), tr["fold"], t3m)
    erp = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & tr["company_id"].isin(book))
    after_o = leftover_diag(
        y, tr[FLAG], (tr["c_n_days_with_tx"], tr["f_ds_r"], tr["a_op_out"]), tr["fold"], lab,
    )
    dsr_nn = leftover_diag(
        y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"],
        lab & pd.to_numeric(tr["f_ds_r"], errors="coerce").notna(),
    )
    prose = (
        f"T3 f_ds_r leftover after days+size {_f(dsr_s['rank'])} after days+a_n_tx {_f(dsr_n['rank'])}. "
        f"ERP leftover {_f(erp['rank'])}. after days+f_ds_r+a_op_out {_f(after_o['rank'])}. "
        f"f_ds_r-notna leftover {_f(dsr_nn['rank'])}."
    )
    print(prose)
    return {"dsr_s": dsr_s["rank"], "erp": erp["rank"], "oo": after_o["rank"], "prose": prose}


def extra_winsor_long(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of winsorized a_debt_service; leftover on long-labeled cos; leftover after days+f_ds_r+a_fin_cost+a_op_out")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    hi = x.quantile(0.99)
    win = x.clip(upper=hi)
    rec = signed_oof_auroc(y, win, tr["fold"], lab)
    after_w = leftover_diag(y, win, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    nlab = lab.groupby(tr["company_id"]).transform("sum")
    long = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & (nlab >= 6))
    rec_l = signed_oof_auroc(y, tr[FLAG], tr["fold"], lab & (nlab >= 6))
    after_all = leftover_diag(
        y, tr[FLAG],
        (tr["c_n_days_with_tx"], tr["f_ds_r"], tr["a_fin_cost"], tr["a_op_out"]),
        tr["fold"], lab,
    )
    t3_med = tr.groupby("company_id")["log_in3"].median()
    cut = t3_med.quantile(2 / 3)
    t3m = lab & (tr["company_id"].map(t3_med) > cut)
    t3_o = leftover_diag(
        y, tr[FLAG], (tr["c_n_days_with_tx"], tr["f_ds_r"], tr["a_op_out"]), tr["fold"], t3m,
    )
    prose = (
        f"winsor99 Y3 {_f(_cv(rec))} leftover {_f(after_w['rank'])}. "
        f"≥6 labeled leftover {_f(long['rank'])} Y3 {_f(_cv(rec_l))} n={long['n']}. "
        f"after days+f_ds_r+a_fin_cost+a_op_out {_f(after_all['rank'])}. "
        f"T3 leftover after days+f_ds_r+a_op_out {_f(t3_o['rank'])}."
    )
    print(prose)
    return {"win": after_w["rank"], "long": long["rank"], "all": after_all["rank"], "prose": prose}


def extra_fc_ratio(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of a_debt_service/a_fin_cost; leftover on T3∩high-days; leftover after days+f_ds_r+f_fc_r")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    fc = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    ratio = (x / fc.replace(0, np.nan)).where(x.notna() & fc.notna())
    rec = signed_oof_auroc(y, ratio, tr["fold"], lab)
    after = leftover_diag(y, ratio, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    t3_med = tr.groupby("company_id")["log_in3"].median()
    days_med = tr.groupby("company_id")["c_n_days_with_tx"].median()
    both = leftover_diag(
        y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"],
        lab
        & (tr["company_id"].map(t3_med) > t3_med.quantile(2 / 3))
        & (tr["company_id"].map(days_med) >= days_med.median()),
    )
    after_f = leftover_diag(
        y, tr[FLAG], (tr["c_n_days_with_tx"], tr["f_ds_r"], tr["f_fc_r"]), tr["fold"], lab,
    )
    t3m = lab & (tr["company_id"].map(t3_med) > t3_med.quantile(2 / 3))
    t3_f = leftover_diag(
        y, tr[FLAG], (tr["c_n_days_with_tx"], tr["f_ds_r"], tr["f_fc_r"]), tr["fold"], t3m,
    )
    prose = (
        f"a_debt_service/a_fin_cost Y3 {_f(_cv(rec))} leftover {_f(after['rank'])}. "
        f"T3∩high-days leftover {_f(both['rank'])} n={both['n']}. "
        f"after days+f_ds_r+f_fc_r {_f(after_f['rank'])}. "
        f"T3 leftover after days+f_ds_r+f_fc_r {_f(t3_f['rank'])}."
    )
    print(prose)
    return {"ratio": after["rank"], "both": both["rank"], "ff": after_f["rank"], "prose": prose}


def extra_t3_full_q4(tr: pd.DataFrame, book: set[str]) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — T3 leftover after days+f_ds_r+a_n_tx+a_op_out; leftover on Q4 days; leftover of T3 ERP")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    t3_med = tr.groupby("company_id")["log_in3"].median()
    t3m = lab & (tr["company_id"].map(t3_med) > t3_med.quantile(2 / 3))
    t3_all = leftover_diag(
        y, tr[FLAG],
        (tr["c_n_days_with_tx"], tr["f_ds_r"], tr["a_n_tx"], tr["a_op_out"]),
        tr["fold"], t3m,
    )
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    q4 = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & (days >= days.quantile(0.75)))
    rec_q = signed_oof_auroc(y, tr[FLAG], tr["fold"], lab & (days >= days.quantile(0.75)))
    t3_erp = leftover_diag(
        y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], t3m & tr["company_id"].isin(book),
    )
    fc = pd.to_numeric(tr["a_fin_cost"], errors="coerce")
    ever_fc = fc.fillna(0).groupby(tr["company_id"]).transform("max") > 0
    after_fc = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & ever_fc)
    prose = (
        f"T3 leftover after days+f_ds_r+a_n_tx+a_op_out {_f(t3_all['rank'])} fake={t3_all['fake']}. "
        f"Q4-days leftover {_f(q4['rank'])} Y3 {_f(_cv(rec_q))} n={q4['n']}. "
        f"T3 ERP leftover {_f(t3_erp['rank'])} n={t3_erp['n']}. "
        f"ever-fin-cost leftover {_f(after_fc['rank'])}."
    )
    print(prose)
    return {"t3": t3_all["rank"], "q4": q4["rank"], "erp": t3_erp["rank"], "prose": prose}


def extra_q4_eat(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Q4-days leftover after f_ds_r / size; leftover of f_ds_r on Q4-days")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    q4m = lab & (days >= days.quantile(0.75))
    after_r = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["f_ds_r"]), tr["fold"], q4m)
    after_s = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], q4m)
    after_b = leftover_diag(
        y, tr[FLAG], (tr["c_n_days_with_tx"], tr["log_in3"], tr["f_ds_r"]), tr["fold"], q4m,
    )
    dsr = leftover_diag(y, tr["f_ds_r"], (tr["c_n_days_with_tx"],), tr["fold"], q4m)
    rec_d = signed_oof_auroc(y, tr["f_ds_r"], tr["fold"], q4m)
    rec_x = signed_oof_auroc(y, tr[FLAG], tr["fold"], q4m)
    prose = (
        f"Q4-days leftover after days+f_ds_r {_f(after_r['rank'])} after days+size {_f(after_s['rank'])} "
        f"after days+size+f_ds_r {_f(after_b['rank'])}. "
        f"Q4 f_ds_r leftover {_f(dsr['rank'])} Y3 {_f(_cv(rec_d))}. Q4 a_debt_service Y3 {_f(_cv(rec_x))}."
    )
    print(prose)
    return {"r": after_r["rank"], "s": after_s["rank"], "dsr": dsr["rank"], "prose": prose}


def extra_q4_dsr_q3(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of Q4 f_ds_r after days+size; leftover on Q3 days; leftover of Q4 after days+a_n_tx")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    q4m = lab & (days >= days.quantile(0.75))
    q3m = lab & (days >= days.quantile(0.50)) & (days < days.quantile(0.75))
    dsr_s = leftover_diag(y, tr["f_ds_r"], (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], q4m)
    dsr_n = leftover_diag(y, tr["f_ds_r"], (tr["c_n_days_with_tx"], tr["a_n_tx"]), tr["fold"], q4m)
    q3 = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], q3m)
    rec_q3 = signed_oof_auroc(y, tr[FLAG], tr["fold"], q3m)
    q4_n = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["a_n_tx"]), tr["fold"], q4m)
    prose = (
        f"Q4 f_ds_r leftover after days+size {_f(dsr_s['rank'])} after days+a_n_tx {_f(dsr_n['rank'])}. "
        f"Q3-days leftover {_f(q3['rank'])} Y3 {_f(_cv(rec_q3))} n={q3['n']}. "
        f"Q4 leftover after days+a_n_tx {_f(q4_n['rank'])}."
    )
    print(prose)
    return {"dsr_s": dsr_s["rank"], "q3": q3["rank"], "q4n": q4_n["rank"], "prose": prose}


def extra_q12_q4stack(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover on Q1/Q2 days; leftover of Q4 after days+f_ds_r+a_n_tx")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    q1m = lab & (days < days.quantile(0.25))
    q2m = lab & (days >= days.quantile(0.25)) & (days < days.quantile(0.50))
    q4m = lab & (days >= days.quantile(0.75))
    q1 = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], q1m)
    q2 = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], q2m)
    rec1 = signed_oof_auroc(y, tr[FLAG], tr["fold"], q1m)
    rec2 = signed_oof_auroc(y, tr[FLAG], tr["fold"], q2m)
    q4s = leftover_diag(
        y, tr[FLAG], (tr["c_n_days_with_tx"], tr["f_ds_r"], tr["a_n_tx"]), tr["fold"], q4m,
    )
    q4ss = leftover_diag(
        y, tr[FLAG],
        (tr["c_n_days_with_tx"], tr["f_ds_r"], tr["log_in3"], tr["a_n_tx"]),
        tr["fold"], q4m,
    )
    prose = (
        f"Q1-days leftover {_f(q1['rank'])} Y3 {_f(_cv(rec1))} n={q1['n']}. "
        f"Q2-days leftover {_f(q2['rank'])} Y3 {_f(_cv(rec2))} n={q2['n']}. "
        f"Q4 leftover after days+f_ds_r+a_n_tx {_f(q4s['rank'])} after days+size+f_ds_r+a_n_tx {_f(q4ss['rank'])}."
    )
    print(prose)
    return {"q1": q1["rank"], "q2": q2["rank"], "q4s": q4s["rank"], "prose": prose}


def extra_q4_t3_int(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover on Q4∩T3; leftover of Q4 after days+f_ds_r+a_n_tx+a_op_out")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    t3_med = tr.groupby("company_id")["log_in3"].median()
    q4m = lab & (days >= days.quantile(0.75))
    t3m = lab & (tr["company_id"].map(t3_med) > t3_med.quantile(2 / 3))
    both = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], q4m & t3m)
    rec = signed_oof_auroc(y, tr[FLAG], tr["fold"], q4m & t3m)
    both_r = leftover_diag(
        y, tr[FLAG], (tr["c_n_days_with_tx"], tr["f_ds_r"]), tr["fold"], q4m & t3m,
    )
    q4f = leftover_diag(
        y, tr[FLAG],
        (tr["c_n_days_with_tx"], tr["f_ds_r"], tr["a_n_tx"], tr["a_op_out"]),
        tr["fold"], q4m,
    )
    dsr_b = leftover_diag(y, tr["f_ds_r"], (tr["c_n_days_with_tx"],), tr["fold"], q4m & t3m)
    prose = (
        f"Q4∩T3 leftover {_f(both['rank'])} Y3 {_f(_cv(rec))} n={both['n']} "
        f"after days+f_ds_r {_f(both_r['rank'])}. "
        f"Q4 leftover after days+f_ds_r+a_n_tx+a_op_out {_f(q4f['rank'])}. "
        f"Q4∩T3 f_ds_r leftover {_f(dsr_b['rank'])}."
    )
    print(prose)
    return {"both": both["rank"], "q4f": q4f["rank"], "dsr": dsr_b["rank"], "prose": prose}


def extra_q4_fc_ntx(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of Q4 after days+f_ds_r+a_fin_cost; leftover on Q4 a_n_tx")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    ntx = pd.to_numeric(tr["a_n_tx"], errors="coerce")
    q4m = lab & (days >= days.quantile(0.75))
    q4n = lab & (ntx >= ntx.quantile(0.75))
    after_fc = leftover_diag(
        y, tr[FLAG], (tr["c_n_days_with_tx"], tr["f_ds_r"], tr["a_fin_cost"]), tr["fold"], q4m,
    )
    after_all = leftover_diag(
        y, tr[FLAG],
        (tr["c_n_days_with_tx"], tr["f_ds_r"], tr["a_n_tx"], tr["a_op_out"], tr["a_fin_cost"]),
        tr["fold"], q4m,
    )
    ntx_l = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], q4n)
    rec_n = signed_oof_auroc(y, tr[FLAG], tr["fold"], q4n)
    ntx_r = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["f_ds_r"]), tr["fold"], q4n)
    prose = (
        f"Q4 leftover after days+f_ds_r+a_fin_cost {_f(after_fc['rank'])} "
        f"after days+f_ds_r+a_n_tx+a_op_out+a_fin_cost {_f(after_all['rank'])}. "
        f"Q4 a_n_tx leftover {_f(ntx_l['rank'])} Y3 {_f(_cv(rec_n))} n={ntx_l['n']} "
        f"after days+f_ds_r {_f(ntx_r['rank'])}."
    )
    print(prose)
    return {"fc": after_fc["rank"], "all": after_all["rank"], "ntx": ntx_l["rank"], "prose": prose}


def extra_q4_last3(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of Q4 last-3; leftover of Q4 after days+f_ds_r+size+a_fin_cost")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    q4m = lab & (days >= days.quantile(0.75))
    rnk = pd.Series(np.nan, index=tr.index)
    rnk.loc[lab] = tr.loc[lab].groupby("company_id")["period"].rank(method="first", ascending=False)
    last3 = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], q4m & (rnk <= 3))
    rec = signed_oof_auroc(y, tr[FLAG], tr["fold"], q4m & (rnk <= 3))
    after = leftover_diag(
        y, tr[FLAG],
        (tr["c_n_days_with_tx"], tr["f_ds_r"], tr["log_in3"], tr["a_fin_cost"]),
        tr["fold"], q4m,
    )
    last3_r = leftover_diag(
        y, tr[FLAG], (tr["c_n_days_with_tx"], tr["f_ds_r"]), tr["fold"], q4m & (rnk <= 3),
    )
    prose = (
        f"Q4 last-3 leftover {_f(last3['rank'])} Y3 {_f(_cv(rec))} n={last3['n']} "
        f"after days+f_ds_r {_f(last3_r['rank'])}. "
        f"Q4 leftover after days+f_ds_r+size+a_fin_cost {_f(after['rank'])}."
    )
    print(prose)
    return {"l3": last3["rank"], "stack": after["rank"], "prose": prose}


def extra_q4_last_full(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of Q4 last-labeled; leftover of Q4 after days+f_ds_r+size+a_fin_cost+a_n_tx")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    q4m = lab & (days >= days.quantile(0.75))
    last_lab = tr.loc[lab].groupby("company_id")["period"].transform("max")
    last_m = pd.Series(False, index=tr.index)
    last_m.loc[lab] = pd.to_datetime(tr.loc[lab, "period"]) == last_lab
    last = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], q4m & last_m)
    rec = signed_oof_auroc(y, tr[FLAG], tr["fold"], q4m & last_m)
    after = leftover_diag(
        y, tr[FLAG],
        (tr["c_n_days_with_tx"], tr["f_ds_r"], tr["log_in3"], tr["a_fin_cost"], tr["a_n_tx"]),
        tr["fold"], q4m,
    )
    dsr = leftover_diag(y, tr["f_ds_r"], (tr["c_n_days_with_tx"], tr["log_in3"], tr["a_n_tx"]), tr["fold"], q4m)
    prose = (
        f"Q4 last-labeled leftover {_f(last['rank'])} Y3 {_f(_cv(rec))} n={last['n']}. "
        f"Q4 leftover after days+f_ds_r+size+a_fin_cost+a_n_tx {_f(after['rank'])}. "
        f"Q4 f_ds_r leftover after days+size+a_n_tx {_f(dsr['rank'])}."
    )
    print(prose)
    return {"last": last["rank"], "stack": after["rank"], "dsr": dsr["rank"], "prose": prose}


def extra_q4_year(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of Q4 f_ds_r after days+size+a_n_tx+a_fin_cost; leftover of Q4 in 2025")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    q4m = lab & (days >= days.quantile(0.75))
    yr = pd.to_datetime(tr["period"]).dt.year
    dsr = leftover_diag(
        y, tr["f_ds_r"],
        (tr["c_n_days_with_tx"], tr["log_in3"], tr["a_n_tx"], tr["a_fin_cost"]),
        tr["fold"], q4m,
    )
    y25 = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], q4m & (yr == 2025))
    y26 = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], q4m & (yr == 2026))
    rec25 = signed_oof_auroc(y, tr[FLAG], tr["fold"], q4m & (yr == 2025))
    rec26 = signed_oof_auroc(y, tr[FLAG], tr["fold"], q4m & (yr == 2026))
    after = leftover_diag(
        y, tr[FLAG],
        (tr["c_n_days_with_tx"], tr["f_ds_r"], tr["log_in3"], tr["a_n_tx"], tr["a_fin_cost"], tr["a_op_out"]),
        tr["fold"], q4m,
    )
    prose = (
        f"Q4 f_ds_r leftover after days+size+a_n_tx+a_fin_cost {_f(dsr['rank'])}. "
        f"Q4 2025 leftover {_f(y25['rank'])} Y3 {_f(_cv(rec25))} n={y25['n']}. "
        f"Q4 2026 leftover {_f(y26['rank'])} Y3 {_f(_cv(rec26))} n={y26['n']}. "
        f"Q4 leftover after days+f_ds_r+size+a_n_tx+a_fin_cost+a_op_out {_f(after['rank'])}."
    )
    print(prose)
    return {"dsr": dsr["rank"], "y25": y25["rank"], "y26": y26["rank"], "prose": prose}


def extra_q4_dsr_full(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of Q4 f_ds_r after full stack; leftover of Q4 ever-ds")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    q4m = lab & (days >= days.quantile(0.75))
    ever = x.fillna(0).groupby(tr["company_id"]).transform("max") > 0
    dsr = leftover_diag(
        y, tr["f_ds_r"],
        (tr["c_n_days_with_tx"], tr["log_in3"], tr["a_n_tx"], tr["a_fin_cost"], tr["a_op_out"]),
        tr["fold"], q4m,
    )
    ev = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], q4m & ever)
    rec = signed_oof_auroc(y, tr[FLAG], tr["fold"], q4m & ever)
    ev_r = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["f_ds_r"]), tr["fold"], q4m & ever)
    prose = (
        f"Q4 f_ds_r leftover after days+size+a_n_tx+a_fin_cost+a_op_out {_f(dsr['rank'])}. "
        f"Q4 ever-ds leftover {_f(ev['rank'])} Y3 {_f(_cv(rec))} n={ev['n']} "
        f"after days+f_ds_r {_f(ev_r['rank'])}."
    )
    print(prose)
    return {"dsr": dsr["rank"], "ev": ev["rank"], "prose": prose}


def extra_t3_dsr_full(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of T3 f_ds_r after full stack; leftover of T3 after days+f_ds_r+size+a_n_tx+a_op_out")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    t3_med = tr.groupby("company_id")["log_in3"].median()
    t3m = lab & (tr["company_id"].map(t3_med) > t3_med.quantile(2 / 3))
    dsr = leftover_diag(
        y, tr["f_ds_r"],
        (tr["c_n_days_with_tx"], tr["log_in3"], tr["a_n_tx"], tr["a_fin_cost"], tr["a_op_out"]),
        tr["fold"], t3m,
    )
    after = leftover_diag(
        y, tr[FLAG],
        (tr["c_n_days_with_tx"], tr["f_ds_r"], tr["log_in3"], tr["a_n_tx"], tr["a_op_out"]),
        tr["fold"], t3m,
    )
    rec = signed_oof_auroc(y, tr["f_ds_r"], tr["fold"], t3m)
    prose = (
        f"T3 f_ds_r leftover after days+size+a_n_tx+a_fin_cost+a_op_out {_f(dsr['rank'])} "
        f"Y3 {_f(_cv(rec))}. T3 leftover of a_debt_service after days+f_ds_r+size+a_n_tx+a_op_out "
        f"{_f(after['rank'])}."
    )
    print(prose)
    return {"dsr": dsr["rank"], "after": after["rank"], "prose": prose}


def decide(p1, p2, p3) -> dict:
    leftover_lives = bool(np.isfinite(p3["rank"]) and p3["rank"] >= CHANCE)
    is_size = bool(p1["is_size"])
    twin = bool(p1["twin_gate"])
    engine = leftover_lives and bool(p2["beat_size"]) and not is_size and not twin
    if engine:
        role = "KEEP as unused leftover"
        why = (
            f"KEEP-as-X passed leftover after days rank {p3['rank']:.3f}, "
            f"beat-size {_f(p2['open'])} vs 0.617. Off the 15-col card."
        )
        park = "KEEP leftover — still do not invent y_debt_service; it is X not Y"
    elif leftover_lives and (twin or is_size or not p2["beat_size"]):
        role = "CLOSE unused leftover"
        why = (
            f"honest leftover after days rank {p3['rank']:.3f} lives "
            f"(OLS {_f(p3['ols'])} fake={p3['fake']} ρ(resid,days)={_f(p3['after']['rho_ctrl'])}). "
            f"{'TWIN of ' + str(p1['gate_twins']) + '. ' if twin else ''}"
            f"{'SIZE (|ρ| vs log1p(a_in3) ≥0.50). ' if is_size else ''}"
            f"{'beat-size FAIL (' + _f(p2['open']) + ' vs 0.617). ' if not p2['beat_size'] else ''}"
            f"DROP from the 44. Off the 15-col card."
        )
        park = "PARK as Y — do not invent y_debt_service"
    else:
        role = "CLOSE unused leftover"
        why = (
            f"unused leftover after days: honest rank {_f(p3['rank'])} dies "
            f"(OLS {_f(p3['ols'])} fake={p3['fake']}). "
            f"DROP from the 44 as Y3 X. Off the 15-col card. Do not grow TURNOVER."
        )
        park = "PARK as Y — do not invent y_debt_service"
    return {
        "role": role, "why": why, "leftover_lives": leftover_lives,
        "engine": engine, "is_size": is_size, "twin": twin,
        "park_y": park,
        "card": "no — do not put a_debt_service on the 15-col card",
    }


def make_plot(tr: pd.DataFrame, p3: dict) -> bool:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return False
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    lab = y3.notna() & x.notna() & (x > 0)
    if lab.sum() > 20:
        q = pd.qcut(x[lab], 5, duplicates="drop")
        rates = y3[lab].groupby(q, observed=False).mean()
        ax.plot(range(len(rates)), rates.to_numpy(), marker="o", color="#1f4e79")
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels([str(i + 1) for i in range(len(rates))])
    ax.set_xlabel("a_debt_service quintile (positive)")
    ax.set_ylabel("Y3 rate")
    ax.set_title("debt_service vs recover")
    ax = axes[1]
    rec = p3["after"]["rrec"]
    ax.bar([r["fold"] for r in rec["folds"]], [r["auroc"] for r in rec["folds"]], color="#1f4e79")
    ax.axhline(CHANCE, color="#333", ls="--", lw=1, label="0.55 dies")
    ax.axhline(DAYS_BENCH, color="#c45c26", ls=":", lw=1, label="days 0.711")
    ax.set_xlabel("fold")
    ax.set_ylabel("rank leftover AUROC")
    ax.set_title("a_debt_service leftover after days")
    ax.set_ylim(0.35, 0.85)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return True


def write_md(ctx: dict) -> None:
    d = ctx["decision"]
    p1, p2, p3 = ctx["p1"], ctx["p2"], ctx["p3"]
    p4, p5, p6 = ctx["p4"], ctx["p5"], ctx["p6"]
    p7, p8 = ctx["p7"], ctx["p8"]
    lines = [
        "# Unused leftover of contemporaneous `a_debt_service` after days as Y3 X",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. No `build_targets`. Do not invent `y_debt_service`. "
        "Do not put `a_debt_service` on the 15-col card. "
        "Do not overwrite `ds_r_qa.*`, `fc_r_qa.*`, `op_out_qa.*`, `in3_qa.*`. "
        "Do not grow TURNOVER. `f_ds_r` leftover 0.528 DROP stays. `f_fc_r` leftover 0.449 DROP stays.",
        "",
        "`a_debt_service` = -sum(amount | grp = debt_service) this month. "
        "Store twins: `f_debt_service` / `f_ds_r` / `a_fin_cost` / `f_fc_r`.",
        "",
        "## Headline",
        "",
        (
            f"{d['role']} leftover-after-days rank {_f(p3['rank'])} (OLS {_f(p3['ols'])}, fake={p3['fake']}). "
            f"Y3 a_debt_service {_f(p2['open'])} vs days {_f(p2['days'])} vs size {_f(p2['size'])} "
            f"vs f_ds_r {_f(p2['fds'])}. SIZE={p1['is_size']} twin_gate={p1['twin_gate']} "
            f"twins={p1['gate_twins'] or 'none'}. Inverse days-after-a_debt_service {_f(p3['inv_rank'])}. "
            f"Leftover after f_ds_r {_f(p4['rank'])}. after days+size {_f(p5['after_b'])}. "
            f"Q6 lag1 leftover {_f(p6['l1_rank'])}. Demean leftover {_f(ctx['xd']['rank'])}. "
            f"Card: **{d['role']}** / KEEP off the 15-col card. "
            f"Night Y3 0.762/0.752, days 0.711, size 0.617, TURNOVER 0.720/0.712 unchanged."
        ),
        "",
        "## Brief questions",
        "",
        "| # | question | what this cut says |",
        "| --- | --- | --- |",
        f"| 1 | Who is healthy? | {d['park_y']}. Size bar 0.617 stays. |",
        f"| 2 | Who is improving? | leftover after days {_f(p3['rank'])}. Month debt service ≠ 45→65. |",
        f"| 3 | Who is turning? | **{d['role']}** vs days 0.711. |",
        f"| 4 | Dip vs fall? | leftover after f_ds_r {_f(p4['rank'])} — ratio rewrite? {p4['rewrite']}. |",
        f"| 5 | Why did it change? | twins={p1['gate_twins'] or 'none'}; SIZE={p1['is_size']}. |",
        f"| 6 | Months earlier? | lag1 leftover {_f(p6['l1_rank'])}; days_lag1 {_f(p6['days_l1'])}. |",
        "",
        "## PARK / CLOSE / KEEP / DROP-from-44",
        "",
        "| object | decision | why |",
        "| --- | --- | --- |",
        f"| `a_debt_service` as Y3 X / 15-col card | **{d['role']}** | {d['why']} |",
        f"| `a_debt_service` as engine X on the 44 | **{'KEEP leftover' if d['engine'] else 'DROP'}** | leftover lives={d['leftover_lives']} twin={d['twin']} SIZE={d['is_size']} beat-size={p2['beat_size']} |",
        f"| rewrite of f_ds_r | **{'YES' if p4['rewrite'] else 'NO'}** | leftover after f_ds_r {_f(p4['rank'])} |",
        f"| `y_debt_service` | **PARK** | do not invent unless KEEP-as-X |",
        f"| Q6 lag1 / TURNOVER | **CLOSE** | leftover {_f(p6['l1_rank'])}; do not grow 0.720 |",
        f"| `f_ds_r` leftover | **DROP quote 0.528** | store flow stays |",
        f"| `f_fc_r` leftover | **DROP quote 0.449** | KEEP f_fc_r_lag3 on TURNOVER |",
        f"| size bar `log1p(a_in3)` | **KEEP quote** | 0.617 stays |",
        "",
        "## 1 — Coverage; twin / SIZE",
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
        "## 4 — Leftover after f_ds_r",
        "",
        p4["prose"], "", _md_table(p4["rows"]), "",
        "## 5 — Leftover after size / days+size",
        "",
        p5["prose"], "",
        "## 6 — Q6 lag1 leftover after days_lag1",
        "",
        p6["prose"], "", _md_table(p6["rows"]), "",
        "## 7 — Dark vs ERP",
        "",
        p7["prose"], "", _md_table(p7["rows"]), "",
        "## 8 — Holdout coverage only",
        "",
        p8["prose"], "",
        "## Extras",
        "",
        "### Bootstrap leftover after days", "", ctx["xb"]["prose"], "",
        "### ICC / acf1", "", ctx["xa"]["prose"], "",
        "### leftover after twins / fin_cost / op_out", "", ctx["xt"]["prose"], "", _md_table(ctx["xt"]["rows"]), "",
        "### leftover of company-demeaned a_debt_service", "", ctx["xd"]["prose"], "",
        "### zero dummy / log1p leftover", "", ctx["xz"]["prose"], "",
        "### Y7 / Y2 leftover after days", "", ctx["xy"]["prose"], "",
        "### leftover on ds>0 / size terciles", "", ctx["xp"]["prose"], "", _md_table(ctx["xp"]["rows"]), "",
        "### leftover of intensity / stacked controls", "", ctx["xr"]["prose"], "",
        "### leftover on ds>0 after days+f_ds_r / size", "", ctx["xps"]["prose"], "",
        "### leftover on ever-ds / year", "", ctx["xe"]["prose"], "",
        "### leftover of f_ds_r / f_debt_service / f_fc_r after days", "", ctx["xrep"]["prose"], "",
        "### last-labeled / MoM / leftover after days+a_fin_cost", "", ctx["xl"]["prose"], "",
        "### leftover of has-ds dummy / a_debt_service/a_op_out / last-3", "", ctx["xh"]["prose"], "",
        "### MoM leftover after days+f_ds_r; T3 leftover after days+size+f_ds_r", "", ctx["xm"]["prose"], "",
        "### leftover of 3m rolling / ever-ds share / last-labeled ever-ds", "", ctx["xrs"]["prose"], "",
        "### T3 leftover after days+size / a_fin_cost; never-ds leftover fake?", "", ctx["xt3"]["prose"], "",
        "### leftover of a_debt_service/a_n_tx; leftover after days+f_fc_r", "", ctx["xpt"]["prose"], "",
        "### T3 leftover after days+a_n_tx / a_op_out; T3 ds>0 leftover", "", ctx["xte"]["prose"], "",
        "### leftover of within-company rank / leftover after days+f_ds_r+a_n_tx", "", ctx["xw"]["prose"], "",
        "### leftover of f_ds_r / a_fin_cost after days on T3; leftover of high-days after f_ds_r", "", ctx["xtt"]["prose"], "",
        "### leftover of a_debt_service_lag1 after days; last-3 leftover after days+f_ds_r", "", ctx["xld"]["prose"], "",
        "### T3 leftover of f_ds_r after days+size; ERP leftover", "", ctx["xtd"]["prose"], "",
        "### leftover of winsorized / long-labeled / leftover after days+f_ds_r+a_fin_cost+a_op_out", "", ctx["xwn"]["prose"], "",
        "### leftover of a_debt_service/a_fin_cost; leftover on T3∩high-days", "", ctx["xfr"]["prose"], "",
        "### T3 leftover after days+f_ds_r+a_n_tx+a_op_out; leftover on Q4 days", "", ctx["xq"]["prose"], "",
        "### Q4-days leftover after f_ds_r / size; leftover of f_ds_r on Q4-days", "", ctx["xq4"]["prose"], "",
        "### leftover of Q4 f_ds_r after days+size; leftover on Q3 days", "", ctx["xq3"]["prose"], "",
        "### leftover on Q1/Q2 days; leftover of Q4 after days+f_ds_r+a_n_tx", "", ctx["xqq"]["prose"], "",
        "### leftover on Q4∩T3; leftover of Q4 after days+f_ds_r+a_n_tx+a_op_out", "", ctx["xqi"]["prose"], "",
        "### leftover of Q4 after days+f_ds_r+a_fin_cost; leftover on Q4 a_n_tx", "", ctx["xqf"]["prose"], "",
        "### leftover of Q4 last-3; leftover of Q4 after days+f_ds_r+size+a_fin_cost", "", ctx["xl3"]["prose"], "",
        "### leftover of Q4 last-labeled; leftover of Q4 after days+f_ds_r+size+a_fin_cost+a_n_tx", "", ctx["xlf"]["prose"], "",
        "### leftover of Q4 f_ds_r after days+size+a_n_tx+a_fin_cost; leftover of Q4 in 2025", "", ctx["xyr"]["prose"], "",
        "### leftover of Q4 f_ds_r after full stack; leftover of Q4 ever-ds", "", ctx["xqf2"]["prose"], "",
        "### leftover of T3 f_ds_r after full stack", "", ctx["xtf"]["prose"], "",
        "## Night quotes (unchanged)",
        "",
        "| quote | locked |",
        "| --- | --- |",
        f"| Y3 | {Y3_NIGHT[0]:.3f} / {Y3_NIGHT[1]:.3f} |",
        f"| days | {DAYS_BENCH:.3f} |",
        f"| size | {SIZE_QUOTE:.3f} |",
        f"| Y7 TURNOVER | {Y7_TURNOVER:.3f} / {Y7_B_SHALLOW:.3f} |",
        f"| f_ds_r leftover | {DS_R_LEFT:.3f} DROP |",
        f"| f_fc_r leftover | {FC_R_LEFT:.3f} DROP |",
        f"| q6_keep | issued_lag1 / days_lag1 / ss_lag1 |",
        "",
        "Do not grow TURNOVER. Do not put `a_debt_service` on the 15-col card. "
        "KEEP `f_fc_r_lag3` on TURNOVER. Do not quote a_out_vol 0.722 as the engine.",
        "",
        "## Files written",
        "",
        "- `analysis/evaluate/debt_svc_qa.py`",
        "- `analysis/outputs/debt_svc_qa.md`",
        "- `analysis/outputs/debt_svc_qa.png`" if ctx.get("png") else "- (no PNG)",
        "- append-only `analysis/experiments/registry.csv`",
        "- `overnight/waves/wave4_debt_svc.md` (end, if WRITE_WAVE)",
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
    p1, p2, p3, p4 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"]
    d = ctx["decision"]
    rows = [
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_a_debt_service", "value": p2["open"], "coverage": f"{p1['cov']:.4f}", "notes": f"days={p2['days']:.4f} size={p2['size']:.4f} beat={p2['beat_size']}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_a_debt_service_resid_days", "value": p3["rank"], "coverage": f"{p1['cov']:.4f}", "notes": f"ols={p3['ols']:.4f} dies={p3['dies']} fake={p3['fake']}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_a_debt_service_resid_dsr", "value": p4["rank"], "coverage": f"{p1['cov']:.4f}", "notes": f"rewrite={p4['rewrite']}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train", "metric": "debt_svc_leftover", "value": 1 if d["leftover_lives"] else 0, "coverage": f"{p1['cov']:.4f}", "notes": d["role"] + " " + d["why"][:160]},
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
    p1, p2, p3, p4, p5, p6, p7 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"], ctx["p5"], ctx["p6"], ctx["p7"]
    text = (
        f"# Wave 4 — a_debt_service leftover after days as Y3 X\n\n"
        f"Agent `{AGENT}`. Train group-fold seed 20260918. Holdout 72 coverage only.\n\n"
        f"## Files written\n\n"
        f"- `analysis/evaluate/debt_svc_qa.py`\n"
        f"- `analysis/outputs/debt_svc_qa.md`\n"
        f"- `analysis/outputs/debt_svc_qa.png`\n"
        f"- append-only `analysis/experiments/registry.csv`\n"
        f"- this note\n\n"
        f"Did not touch `ds_r_qa.*`, `fc_r_qa.*`, `op_out_qa.*`, `in3_qa.*`, "
        f"`a_vol_qa.*`, `transfer_qa.*`, `growth_qa.*`, `n_accounts_qa.*`, "
        f"`gbm_core.py`, `cashflow.py`, `debt.py`, parquet / duckdb, `build_targets`, "
        f"`product/`, the 15-col card, TURNOVER, or the parent journal. "
        f"Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. "
        f"Y7 TURNOVER **0.720 / 0.712** unchanged. `f_ds_r` leftover 0.528 DROP stays. "
        f"`f_fc_r` leftover 0.449 DROP stays. KEEP `f_fc_r_lag3` on TURNOVER.\n\n"
        f"## Locked verdict\n\n"
        f"| object | decision |\n"
        f"| --- | --- |\n"
        f"| `a_debt_service` as Y3 X / 15-col card | **{d['role']}** |\n"
        f"| `a_debt_service` as engine X on the 44 | **{'KEEP leftover' if d['engine'] else 'DROP'}** |\n"
        f"| `y_debt_service` | **PARK** |\n"
        f"| TURNOVER | **CLOSE** — do not grow 0.720 |\n"
        f"| f_ds_r leftover | **DROP quote 0.528** |\n"
        f"| size bar | **KEEP quote 0.617** |\n\n"
        f"## Locked extras\n\n"
        f"- Honest leftover after days rank {_f(p3['rank'])} (lives={d['leftover_lives']}, OLS fake={p3['fake']}); inverse {_f(p3['inv_rank'])}.\n"
        f"- Single {_f(p2['open'])} vs days {_f(p2['days'])} vs size {_f(p2['size'])} vs f_ds_r {_f(p2['fds'])}.\n"
        f"- SIZE={p1['is_size']} twins={p1['gate_twins'] or 'none'}.\n"
        f"- leftover after f_ds_r {_f(p4['rank'])} rewrite={p4['rewrite']}. after days+size {_f(p5['after_b'])}.\n"
        f"- Q6 lag1 leftover {_f(p6['l1_rank'])}. Dark leftover {_f(p7['dark'])} ERP {_f(p7['erp'])}.\n"
        f"- {ctx['xa']['prose']}\n"
        f"- {ctx['xt']['prose']}\n"
        f"- {ctx['xd']['prose']}\n"
        f"- {ctx['xp']['prose']}\n"
        f"- {ctx['xr']['prose']}\n"
        f"- {ctx['xps']['prose']}\n"
        f"- {ctx['xe']['prose']}\n"
        f"- {ctx['xrep']['prose']}\n"
        f"- {ctx['xl']['prose']}\n"
        f"- {ctx['xh']['prose']}\n"
        f"- {ctx['xm']['prose']}\n"
        f"- {ctx['xrs']['prose']}\n"
        f"- {ctx['xt3']['prose']}\n"
        f"- {ctx['xpt']['prose']}\n"
        f"- {ctx['xte']['prose']}\n"
        f"- {ctx['xw']['prose']}\n"
        f"- {ctx['xtt']['prose']}\n"
        f"- {ctx['xld']['prose']}\n"
        f"- {ctx['xtd']['prose']}\n"
        f"- {ctx['xwn']['prose']}\n"
        f"- {ctx['xfr']['prose']}\n"
        f"- {ctx['xq']['prose']}\n"
        f"- {ctx['xq4']['prose']}\n"
        f"- {ctx['xq3']['prose']}\n"
        f"- {ctx['xqq']['prose']}\n"
        f"- {ctx['xqi']['prose']}\n"
        f"- {ctx['xqf']['prose']}\n"
        f"- {ctx['xl3']['prose']}\n"
        f"- {ctx['xlf']['prose']}\n"
        f"- {ctx['xyr']['prose']}\n"
        f"- {ctx['xqf2']['prose']}\n"
        f"- {ctx['xtf']['prose']}\n"
        f"- {ctx['xb']['prose']}\n\n"
        f"{d['why']}\n\n"
        f"## What failed / next\n\n"
        + ("\n".join(f"- {x}" for x in ctx["failed"]) if ctx["failed"] else "- none")
        + f"\n\nElapsed {ctx['elapsed_s']:.0f}s.\n"
    )
    WAVE_NOTE.write_text(text)
    print(f"wrote {WAVE_NOTE}")


def main() -> None:
    t0 = time.time()
    failed: list[str] = []
    print("debt_svc leftover QA — unused leftover of a_debt_service after days as Y3 X")
    panel = load_panel()
    panel = attach_folds(panel)
    panel = add_panel_lags(panel, [FLAG, "c_n_days_with_tx"], (1,))
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
    p4 = pass4_dsr(tr)
    p5 = pass5_size(tr)
    p6 = pass6_q6(tr)
    p7 = pass7_dark(tr, book)
    p8 = pass8_hold(hold)
    xb = extra_bootstrap(tr, n_boot=40)
    xa = extra_icc(tr)
    xt = extra_twins(tr)
    xd = extra_demean(tr)
    xz = extra_zero_log(tr)
    xy = extra_y7_y2(tr)
    xp = extra_pos_size(tr)
    xr = extra_ratio_stack(tr)
    xps = extra_pos_stack(tr)
    xe = extra_ever_year(tr)
    xrep = extra_dsr_replica(tr)
    xl = extra_last_mom(tr)
    xh = extra_has_ds(tr)
    xm = extra_mom_stack(tr)
    xrs = extra_roll_share(tr)
    xt3 = extra_t3_stack(tr)
    xpt = extra_per_tx(tr, book)
    xte = extra_t3_eat(tr)
    xw = extra_within(tr)
    xtt = extra_t3_twins(tr)
    xld = extra_lag_days(tr)
    xtd = extra_t3_dsr_eat(tr, book)
    xwn = extra_winsor_long(tr)
    xfr = extra_fc_ratio(tr)
    xq = extra_t3_full_q4(tr, book)
    xq4 = extra_q4_eat(tr)
    xq3 = extra_q4_dsr_q3(tr)
    xqq = extra_q12_q4stack(tr)
    xqi = extra_q4_t3_int(tr)
    xqf = extra_q4_fc_ntx(tr)
    xl3 = extra_q4_last3(tr)
    xlf = extra_q4_last_full(tr)
    xyr = extra_q4_year(tr)
    xqf2 = extra_q4_dsr_full(tr)
    xtf = extra_t3_dsr_full(tr)
    decision = decide(p1, p2, p3)
    print("\n" + "=" * 72)
    print(f"VERDICT: {decision['role']}")
    print(decision["why"])
    print("=" * 72)
    if not p2["days_ok"]:
        failed.append(f"days replica {p2['days']:.3f} ≠ 0.711")
    if not p2["size_ok"]:
        failed.append(f"size replica {p2['size']:.3f} ≠ 0.617")
    png = make_plot(tr, p3)
    elapsed = time.time() - t0
    ctx = {
        "p1": p1, "p2": p2, "p3": p3, "p4": p4, "p5": p5, "p6": p6, "p7": p7, "p8": p8,
        "xb": xb, "xa": xa, "xt": xt, "xd": xd, "xz": xz, "xy": xy, "xp": xp, "xr": xr,
        "xps": xps, "xe": xe, "xrep": xrep, "xl": xl, "xh": xh,
        "xm": xm, "xrs": xrs, "xt3": xt3, "xpt": xpt, "xte": xte, "xw": xw,
        "xtt": xtt, "xld": xld, "xtd": xtd, "xwn": xwn, "xfr": xfr, "xq": xq, "xq4": xq4, "xq3": xq3, "xqq": xqq, "xqi": xqi, "xqf": xqf, "xl3": xl3, "xlf": xlf, "xyr": xyr, "xqf2": xqf2, "xtf": xtf,
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
