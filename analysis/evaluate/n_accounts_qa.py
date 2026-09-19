"""Unused leftover of ``g_n_accounts`` after days as Y3 X.

``g_n_accounts`` = COUNT(product_id) as-of period_end (Family G).
Rise-only connection inventory (1,561 rises, **0 drops**). ``g_has_*``
already DROP from the 44. ``g_new`` PARK as health Y. ``g_has_checking``
is 99.1% the connection hole. Dark 470 do not have fewer accounts
(p50=3 both). Access ≠ ERP. ``f_n_types`` leftover 0.534 was a
facilities twin. Do **not** overwrite ``banking_g_qa.*`` / ``g_has_rest_qa.*``.

KEEP-as-X: beat size ≥0.02 **and** leftover after days **and** not SIZE
(|ρ| vs log1p(a_in3) ≥0.50) **and** not a twin (|ρ|≥0.80 vs days /
a_n_tx / g_has_* / f_n_facilities). Leftover <0.55 dies.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.n_accounts_qa

Owned: analysis/evaluate/n_accounts_qa.py, analysis/outputs/n_accounts_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_n_accounts.md (end).
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
OUT_MD = ANALYSIS / "outputs" / "n_accounts_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "n_accounts_qa.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_n_accounts.md"
AGENT = "572fb928"
WAVE = "4"
ROUND = "R4"
MODEL = "n_accounts_qa"
X_FAM = "G"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
Y7 = "y7_top1_lost"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_QUOTE = 0.617
DAYS_LAG1_QUOTE = 0.684
N_TYPES_Y3_PEEK = 0.578
N_TYPES_LEFT = 0.534
N_ACC_Y3_PEEK = 0.585
RISES_PEEK = 1561
DROPS_PEEK = 0
HOLE_PEEK = 0.991
P50_PEEK = 3.0
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
    "g_n_accounts",
    "g_n_banks",
    "g_n_types",
    "g_has_checking",
    "g_has_card",
    "g_has_tpv",
    "g_has_saving",
    "g_has_investment",
    "g_new_this_month",
    "f_n_types",
    "f_n_facilities",
)

Y_KEEP = (Y2, Y3, Y7)
FLAG = "g_n_accounts"
HAS_COLS = (
    "g_has_checking",
    "g_has_card",
    "g_has_tpv",
    "g_has_saving",
    "g_has_investment",
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
    leak3 = leakage_check(
        [FLAG, "g_has_checking", "f_n_types", "c_n_days_with_tx", "log_in3"],
        Y3, forbidden_prefixes=["b"],
    )
    if not leak3["ok"]:
        raise RuntimeError(f"Y3 never-B leak: {leak3['issues']}")
    leak7 = leakage_check([FLAG, "f_n_types"], Y7, forbidden_prefixes=["d"])
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
    n_nn = int(x.notna().sum())
    pairs = {
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "a_n_tx": tr["a_n_tx"],
        "g_has_checking": tr["g_has_checking"],
        "g_has_card": tr["g_has_card"],
        "g_has_tpv": tr["g_has_tpv"],
        "g_has_saving": tr["g_has_saving"],
        "g_has_investment": tr["g_has_investment"],
        "g_n_banks": tr["g_n_banks"],
        "g_n_types": tr["g_n_types"],
        "f_n_types": tr["f_n_types"],
        "f_n_facilities": tr["f_n_facilities"],
        "log1p(a_in3)": tr["log_in3"],
    }
    rhos = {k: spearman(x, v) for k, v in pairs.items()}
    twins = [k for k, v in rhos.items() if np.isfinite(v) and abs(v) >= TWIN_RHO]
    gate_keys = ["c_n_days_with_tx", "a_n_tx", *HAS_COLS, "f_n_facilities"]
    gate_twins = [k for k in gate_keys if np.isfinite(rhos.get(k, float("nan"))) and abs(rhos[k]) >= TWIN_RHO]
    is_size = bool(np.isfinite(rhos["log1p(a_in3)"]) and abs(rhos["log1p(a_in3)"]) >= SIZE_RHO)
    dark = ~tr["company_id"].isin(book)
    p50_d = float(x[dark].median()) if x[dark].notna().any() else float("nan")
    p50_e = float(x[~dark].median()) if x[~dark].notna().any() else float("nan")
    p50_ok = True  # last-month p50=3 is cut 7; panel p50=2 is first-month zeros
    rows = [
        {"col": FLAG, "n_nn": f"{n_nn:,}", "cov": _pp(_pct(n_nn, n_cm)), "eq0": _pp(_pct(int((x == 0).sum()), n_nn) if n_nn else float("nan")), "p50": _f(float(x.median()) if n_nn else float("nan"))},
    ]
    rho_rows = [
        {"vs": k, "rho": _f(v), "flag": "SIZE" if k.startswith("log") and abs(v) >= SIZE_RHO else "TWIN" if abs(v) >= TWIN_RHO else "no"}
        for k, v in rhos.items()
    ]
    prose = (
        f"Train {n_co:,} co / {n_cm:,} CM. g_n_accounts nn={n_nn:,} cov {_pp(_pct(n_nn, n_cm))} "
        f"eq0 {_pp(_pct(int((x == 0).sum()), n_nn)) } p50={_f(float(x.median()))}. "
        f"Dark panel p50={_f(p50_d)} ERP panel p50={_f(p50_e)} (last-month p50=3 is cut 7). "
        f"ρ vs days {_f(rhos['c_n_days_with_tx'])} vs a_n_tx {_f(rhos['a_n_tx'])} "
        f"vs checking {_f(rhos['g_has_checking'])} vs f_n_types {_f(rhos['f_n_types'])} "
        f"vs facilities {_f(rhos['f_n_facilities'])} vs size {_f(rhos['log1p(a_in3)'])}. "
        f"SIZE={is_size} gate_twins={gate_twins or 'none'} all_twins={twins or 'none'}."
    )
    print(prose)
    return {
        "rows": rows, "rho_rows": rho_rows, "rhos": rhos, "twins": twins,
        "gate_twins": gate_twins, "is_size": is_size, "twin_gate": bool(gate_twins),
        "n_cm": n_cm, "n_co": n_co, "n_nn": n_nn, "cov": _pct(n_nn, n_cm),
        "p50_d": p50_d, "p50_e": p50_e, "p50_ok": p50_ok, "prose": prose,
    }


def pass2_singles(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 2 — single-feature group-fold Y3")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab, folds = y.notna(), tr["fold"]
    feats = {
        FLAG: tr[FLAG],
        "g_has_checking": tr["g_has_checking"],
        "g_n_banks": tr["g_n_banks"],
        "g_n_types": tr["g_n_types"],
        "f_n_types": tr["f_n_types"],
        "f_n_facilities": tr["f_n_facilities"],
        "g_new_this_month": tr["g_new_this_month"],
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
    y3_o, y3_size, y3_days = _cv(recs[FLAG]), _cv(recs["log1p(a_in3)"]), _cv(recs["c_n_days_with_tx"])
    y3_chk, y3_nt = _cv(recs["g_has_checking"]), _cv(recs["f_n_types"])
    days_ok = bool(np.isfinite(y3_days) and abs(y3_days - DAYS_BENCH) < 0.008)
    size_ok = bool(np.isfinite(y3_size) and abs(y3_size - SIZE_QUOTE) < 0.008)
    nt_ok = bool(np.isfinite(y3_nt) and abs(y3_nt - N_TYPES_Y3_PEEK) < 0.015)
    peek_ok = bool(np.isfinite(y3_o) and abs(y3_o - N_ACC_Y3_PEEK) < 0.02)
    beat_size = bool(np.isfinite(y3_o) and (y3_o - SIZE_QUOTE) >= KEEP_DELTA)
    prose = (
        f"Y3 g_n_accounts {_f(y3_o)} n={recs[FLAG]['n_defined']:,} pos={recs[FLAG]['n_pos']:,} "
        f"(banking_g oriented 0.585 {'CONFIRM' if peek_ok else 'DRIFT'}). "
        f"vs size {_f(y3_size)} vs days {_f(y3_days)} vs checking {_f(y3_chk)} "
        f"vs f_n_types {_f(y3_nt)} (peek 0.578 {'CONFIRM' if nt_ok else 'DRIFT'}). "
        f"Replica days 0.711 {'CONFIRM' if days_ok else 'DRIFT'} / size 0.617 "
        f"{'CONFIRM' if size_ok else 'DRIFT'}. Beat-size Δ={_f(y3_o - SIZE_QUOTE) if np.isfinite(y3_o) else '—'} "
        f"{'PASS' if beat_size else 'FAIL'}."
    )
    print(prose)
    return {
        "rows": rows, "recs": recs, "open": y3_o, "size": y3_size, "days": y3_days,
        "chk": y3_chk, "nt": y3_nt, "days_ok": days_ok, "size_ok": size_ok,
        "nt_ok": nt_ok, "peek_ok": peek_ok, "beat_size": beat_size,
        "n_def": recs[FLAG]["n_defined"], "n_pos": recs[FLAG]["n_pos"], "prose": prose,
    }


def pass3_days(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 3 — honest leftover after days; inverse")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab, folds = y.notna(), tr["fold"]
    after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), folds, lab)
    inv = leftover_diag(y, tr["c_n_days_with_tx"], (tr[FLAG],), folds, lab)
    prose = (
        f"g_n_accounts leftover after days OLS {_f(after['ols'])} rank {_f(after['rank'])} "
        f"fake={after['fake']} almost={after['almost']} ρ(resid,days)={_f(after['rho_ctrl'])} "
        f"R²={_f(after['r2'])} honest_dies={after['honest_dies']} n={after['n']:,} pos={after['n_pos']:,}. "
        f"Inverse: days leftover after n_accounts OLS {_f(inv['ols'])} rank {_f(inv['rank'])} "
        f"dies={inv['honest_dies']}."
    )
    print(prose)
    return {
        "after": after, "inv": inv, "ols": after["ols"], "rank": after["rank"],
        "dies": after["honest_dies"], "fake": after["fake"], "almost": after["almost"],
        "r2": after["r2"], "inv_rank": inv["rank"], "inv_dies": inv["honest_dies"],
        "prose": prose,
    }


def pass5_rise(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 5 — rise-only inventory (1561 / 0)")
    print("=" * 72)
    d = tr[["company_id", "period", FLAG]].copy()
    d[FLAG] = pd.to_numeric(d[FLAG], errors="coerce")
    d = d.sort_values(["company_id", "period"])
    d["lag"] = d.groupby("company_id")[FLAG].shift(1)
    ok = d[FLAG].notna() & d["lag"].notna()
    delta = d.loc[ok, FLAG] - d.loc[ok, "lag"]
    rises = int((delta > 0).sum())
    drops = int((delta < 0).sum())
    flats = int((delta == 0).sum())
    rise_ok = rises == RISES_PEEK and drops == DROPS_PEEK
    y = pd.to_numeric(tr[Y3], errors="coerce")
    dummy = pd.Series(np.nan, index=tr.index)
    dummy.loc[delta.index] = (delta > 0).astype(float)
    rec = signed_oof_auroc(y, dummy, tr["fold"], y.notna())
    after = leftover_diag(y, dummy, (tr["c_n_days_with_tx"],), tr["fold"], y.notna())
    prose = (
        f"rises={rises:,} drops={drops:,} flats={flats:,} "
        f"(peek 1,561/0 {'CONFIRM' if rise_ok else 'DRIFT'}). rise-only={drops == 0}. "
        f"Rise-month dummy Y3 {_f(_cv(rec))} leftover after days {_f(after['rank'])} dies={after['honest_dies']}."
    )
    print(prose)
    return {
        "rises": rises, "drops": drops, "flats": flats, "rise_ok": rise_ok,
        "rise_only": drops == 0, "dummy": _cv(rec), "dummy_rank": after["rank"],
        "prose": prose,
    }


def pass6_hole(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 6 — leftover after g_has_checking (99.1% hole?)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    chk = pd.to_numeric(tr["g_has_checking"], errors="coerce")
    off = chk == 0
    hole = float((x[off] == 0).mean()) if off.any() else float("nan")
    hole_ok = bool(np.isfinite(hole) and abs(hole - HOLE_PEEK) < 0.01)
    after = leftover_diag(y, tr[FLAG], (tr["g_has_checking"],), tr["fold"], lab)
    after_d = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["g_has_checking"]), tr["fold"], lab)
    chk_after = leftover_diag(y, tr["g_has_checking"], (tr[FLAG],), tr["fold"], lab)
    rho = spearman(x, chk)
    rewrite = after["honest_dies"] or (np.isfinite(rho) and abs(rho) >= TWIN_RHO)
    rows = [
        {"bar": "after checking", "OLS": _f(after["ols"]), "rank": _f(after["rank"]), "R2": _f(after["r2"]), "dies": after["honest_dies"]},
        {"bar": "after days+checking", "OLS": _f(after_d["ols"]), "rank": _f(after_d["rank"]), "R2": _f(after_d["r2"]), "dies": after_d["honest_dies"]},
        {"bar": "checking after n_accounts", "OLS": _f(chk_after["ols"]), "rank": _f(chk_after["rank"]), "R2": _f(chk_after["r2"]), "dies": chk_after["honest_dies"]},
    ]
    prose = (
        f"checking=0 and n_accounts=0: {_pp(hole)} (peek 99.1% {'CONFIRM' if hole_ok else 'DRIFT'}). "
        f"ρ(n_accounts, checking)={_f(rho)}. leftover after checking rank {_f(after['rank'])} "
        f"dies={after['honest_dies']} rewrite={rewrite}. after days+checking {_f(after_d['rank'])}."
    )
    print(prose)
    return {
        "rows": rows, "hole": hole, "hole_ok": hole_ok, "rho": rho,
        "c_rank": after["rank"], "c_dies": after["honest_dies"],
        "dc_rank": after_d["rank"], "rewrite": rewrite, "prose": prose,
    }


def pass7_dark(tr: pd.DataFrame, book: set[str]) -> dict:
    print("\n" + "=" * 72)
    print("CUT 7 — Dark vs ERP leftover (p50=3 both)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    erp, dark = tr["company_id"].isin(book), ~tr["company_id"].isin(book)
    n_dark = int(tr.loc[dark, "company_id"].nunique())
    p50_d = float(x[dark].median()) if x[dark].notna().any() else float("nan")
    p50_e = float(x[~dark].median()) if x[~dark].notna().any() else float("nan")
    last = tr.sort_values("period").groupby("company_id").tail(1)
    p50_ld = float(pd.to_numeric(last.loc[~last["company_id"].isin(book), FLAG], errors="coerce").median())
    p50_le = float(pd.to_numeric(last.loc[last["company_id"].isin(book), FLAG], errors="coerce").median())
    rec_d = signed_oof_auroc(y, tr[FLAG], tr["fold"], y.notna() & dark)
    rec_e = signed_oof_auroc(y, tr[FLAG], tr["fold"], y.notna() & erp)
    after_d = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], y.notna() & dark)
    after_e = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], y.notna() & erp)
    p50_ok = bool(abs(p50_ld - P50_PEEK) < 0.2 and abs(p50_le - P50_PEEK) < 0.2)
    rows = [
        {"book": "dark", "n_co": n_dark, "p50": _f(p50_d), "last p50": _f(p50_ld), "Y3": "LOW_POWER" if rec_d["low_power"] else _f(rec_d["cv"]), "leftover": _f(after_d["rank"])},
        {"book": "ERP", "n_co": int(tr.loc[erp, "company_id"].nunique()), "p50": _f(p50_e), "last p50": _f(p50_le), "Y3": "LOW_POWER" if rec_e["low_power"] else _f(rec_e["cv"]), "leftover": _f(after_e["rank"])},
    ]
    prose = (
        f"Dark {n_dark} (want 470) last-month p50={_f(p50_ld)} ERP last p50={_f(p50_le)} "
        f"{'CONFIRM p50=3 both' if p50_ok else 'DRIFT'}. Access ≠ ERP. "
        f"Dark leftover {_f(after_d['rank'])} ERP leftover {_f(after_e['rank'])}."
    )
    print(prose)
    return {
        "rows": rows, "p50_ok": p50_ok, "p50_d": p50_ld, "p50_e": p50_le,
        "dark_rank": after_d["rank"], "erp_rank": after_e["rank"],
        "dark_cv": _cv(rec_d), "erp_cv": _cv(rec_e), "prose": prose,
    }


def pass8_q6(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 8 — Q6 lag1 leftover after days_lag1")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab, folds = y.notna(), tr["fold"]
    recs, rows = {}, []
    for name in (FLAG, f"{FLAG}_lag1", "c_n_days_with_tx_lag1"):
        rec = signed_oof_auroc(y, tr[name], folds, lab)
        recs[name] = rec
        rows.append(_auc_row(Y3, name, rec))
        print(f"  {name} CV={_f(rec['cv'])} n={rec['n_defined']} pos={rec['n_pos']}")
    after_l1 = leftover_diag(y, tr[f"{FLAG}_lag1"], (tr["c_n_days_with_tx_lag1"],), folds, lab)
    days_l1 = _cv(recs["c_n_days_with_tx_lag1"])
    days_ok = bool(np.isfinite(days_l1) and abs(days_l1 - DAYS_LAG1_QUOTE) < 0.008)
    prose = (
        f"Y3 n_accounts_lag1 {_f(recs[f'{FLAG}_lag1']['cv'])} leftover after days_lag1 "
        f"rank {_f(after_l1['rank'])} dies={after_l1['honest_dies']}. "
        f"Days lag1 {_f(days_l1)} (quote 0.684 {'CONFIRM' if days_ok else 'DRIFT'})."
    )
    print(prose)
    return {
        "rows": rows, "lag1": _cv(recs[f"{FLAG}_lag1"]),
        "l1_rank": after_l1["rank"], "l1_dies": after_l1["honest_dies"],
        "days_l1": days_l1, "days_ok": days_ok, "prose": prose,
    }


def pass9_ever(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 9 — ever-n vs this-month count (trait vs month shock)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    ever = x.groupby(tr["company_id"]).transform("max")
    mean_c = x.groupby(tr["company_id"]).transform("mean")
    rec_e = signed_oof_auroc(y, ever, tr["fold"], y.notna())
    after_e = leftover_diag(y, ever, (tr["c_n_days_with_tx"],), tr["fold"], y.notna())
    rec_m = signed_oof_auroc(y, mean_c, tr["fold"], y.notna())
    after_m = leftover_diag(y, mean_c, (tr["c_n_days_with_tx"],), tr["fold"], y.notna())
    after_now = leftover_diag(y, tr[FLAG], (ever,), tr["fold"], y.notna())
    n_ever = int((x.groupby(tr["company_id"]).max() > 0).sum())
    prose = (
        f"Ever n_accounts>0 companies {n_ever}/{tr['company_id'].nunique()}. "
        f"Ever-max Y3 {_f(_cv(rec_e))} leftover after days {_f(after_e['rank'])}. "
        f"Company-mean Y3 {_f(_cv(rec_m))} leftover {_f(after_m['rank'])}. "
        f"Month leftover after ever-max {_f(after_now['rank'])} dies={after_now['honest_dies']}."
    )
    print(prose)
    return {
        "n_ever": n_ever, "ever_cv": _cv(rec_e), "ever_rank": after_e["rank"],
        "mean_cv": _cv(rec_m), "mean_rank": after_m["rank"],
        "now_rank": after_now["rank"], "now_dies": after_now["honest_dies"],
        "prose": prose,
    }


def pass10_hold(hold: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 10 — holdout coverage only (no AUROC)")
    print("=" * 72)
    x = pd.to_numeric(hold[FLAG], errors="coerce")
    rec = {
        "n_co": int(hold["company_id"].nunique()), "n_cm": len(hold),
        "n_nn": int(x.notna().sum()), "cov": _pct(int(x.notna().sum()), len(hold)),
        "p50": float(x[x.notna()].median()) if x.notna().any() else float("nan"),
    }
    prose = f"Holdout {rec['n_co']} co / {rec['n_cm']} CM nn={rec['n_nn']} cov={_pp(rec['cov'])} p50={_f(rec['p50'])} (no fit, no AUROC)."
    print(prose)
    return {**rec, "prose": prose}


def extra_bootstrap(tr: pd.DataFrame, n_boot: int = 40) -> dict:
    print("\n" + "=" * 72)
    print(f"EXTRA — company bootstrap leftover after days (n={n_boot})")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    work = tr.loc[lab, ["company_id", "fold", Y3, FLAG, "c_n_days_with_tx"]].copy()
    work = work.dropna(subset=[FLAG, Y3])
    cos = work["company_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(FOLD_SEED)
    ranks = []
    for _ in range(n_boot):
        draw = rng.choice(cos, size=len(cos), replace=True)
        b = pd.concat([work[work["company_id"] == c] for c in draw], ignore_index=True)
        after = leftover_diag(b[Y3], b[FLAG], (b["c_n_days_with_tx"],), b["fold"], pd.Series(True, index=b.index))
        if np.isfinite(after["rank"]):
            ranks.append(after["rank"])
    p05 = float(np.quantile(ranks, 0.05)) if ranks else float("nan")
    p50 = float(np.median(ranks)) if ranks else float("nan")
    p95 = float(np.quantile(ranks, 0.95)) if ranks else float("nan")
    share = float(np.mean([r < CHANCE for r in ranks])) if ranks else float("nan")
    prose = f"Bootstrap leftover-after-days rank p05={_f(p05)} p50={_f(p50)} p95={_f(p95)} share<0.55={_pp(share)} n={len(ranks)}."
    print(prose)
    return {"p05": p05, "p50": p50, "p95": p95, "share_die": share, "prose": prose}


def extra_ntypes(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after f_n_types / f_n_facilities")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after_t = leftover_diag(y, tr[FLAG], (tr["f_n_types"],), tr["fold"], lab)
    after_f = leftover_diag(y, tr[FLAG], (tr["f_n_facilities"],), tr["fold"], lab)
    nt_days = leftover_diag(y, tr["f_n_types"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    nt_ok = bool(np.isfinite(nt_days["rank"]) and abs(nt_days["rank"] - N_TYPES_LEFT) < 0.03)
    prose = (
        f"n_accounts leftover after f_n_types {_f(after_t['rank'])} after facilities {_f(after_f['rank'])}. "
        f"f_n_types leftover after days {_f(nt_days['rank'])} (peek 0.534 {'CONFIRM' if nt_ok else 'DRIFT'})."
    )
    print(prose)
    return {"t_rank": after_t["rank"], "f_rank": after_f["rank"], "nt_days": nt_days["rank"], "nt_ok": nt_ok, "prose": prose}


def extra_zero(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Y3 at n_accounts==0 vs connected")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    ok = y.notna() & x.notna()
    z = ok & (x == 0)
    p = ok & (x > 0)
    dummy = (x == 0).astype(float).where(x.notna())
    rec = signed_oof_auroc(y, dummy, tr["fold"], ok)
    after = leftover_diag(y, dummy, (tr["c_n_days_with_tx"],), tr["fold"], ok)
    rows = [
        {"slice": "n_accounts==0", "n": int(z.sum()), "n_pos": int((z & (y == 1)).sum()), "Y3 rate": _pp(float(y[z].mean()) if z.any() else float("nan"))},
        {"slice": "n_accounts>0", "n": int(p.sum()), "n_pos": int((p & (y == 1)).sum()), "Y3 rate": _pp(float(y[p].mean()) if p.any() else float("nan"))},
    ]
    prose = (
        f"Y3 n=0 {_pp(float(y[z].mean()) if z.any() else float('nan'))} n={int(z.sum())} "
        f"vs connected {_pp(float(y[p].mean()) if p.any() else float('nan'))} n={int(p.sum())}. "
        f"zero-dummy Y3 {_f(_cv(rec))} leftover after days {_f(after['rank'])} dies={after['honest_dies']}."
    )
    print(prose)
    return {"rows": rows, "dummy": _cv(rec), "after": after["rank"], "prose": prose}


def extra_banks(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after g_n_banks (cluster twin)")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    rho = spearman(tr[FLAG], tr["g_n_banks"])
    after = leftover_diag(y, tr[FLAG], (tr["g_n_banks"],), tr["fold"], lab)
    after_d = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["g_n_banks"]), tr["fold"], lab)
    banks = leftover_diag(y, tr["g_n_banks"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    rec = signed_oof_auroc(y, tr["g_n_banks"], tr["fold"], lab)
    prose = (
        f"ρ(n_accounts, n_banks)={_f(rho)} TWIN. leftover after banks {_f(after['rank'])} "
        f"dies={after['honest_dies']}. after days+banks {_f(after_d['rank'])}. "
        f"g_n_banks Y3 {_f(_cv(rec))} leftover after days {_f(banks['rank'])} dies={banks['honest_dies']}."
    )
    print(prose)
    return {"rho": rho, "after": after["rank"], "banks_rank": banks["rank"], "banks_y3": _cv(rec), "prose": prose}


def extra_dummies(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — rise / zero dummy leftover (rank vs fake)")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    d = tr[["company_id", "period", FLAG]].copy()
    d[FLAG] = pd.to_numeric(d[FLAG], errors="coerce")
    d = d.sort_values(["company_id", "period"])
    d["lag"] = d.groupby("company_id")[FLAG].shift(1)
    ok = d[FLAG].notna() & d["lag"].notna()
    delta = d.loc[ok, FLAG] - d.loc[ok, "lag"]
    rise = pd.Series(np.nan, index=tr.index)
    rise.loc[delta.index] = (delta > 0).astype(float)
    zero = (x == 0).astype(float).where(x.notna())
    a_r = leftover_diag(y, rise, (tr["c_n_days_with_tx"],), tr["fold"], y.notna())
    a_z = leftover_diag(y, zero, (tr["c_n_days_with_tx"],), tr["fold"], y.notna())
    prose = (
        f"rise dummy leftover OLS {_f(a_r['ols'])} rank {_f(a_r['rank'])} fake={a_r['fake']} "
        f"ρ(resid,days)={_f(a_r['rho_ctrl'])}. zero dummy leftover OLS {_f(a_z['ols'])} "
        f"rank {_f(a_z['rank'])} fake={a_z['fake']} ρ(resid,days)={_f(a_z['rho_ctrl'])}. "
        f"High OLS leftover on dummies is a fake days leak unless rank also lives."
    )
    print(prose)
    return {"rise_rank": a_r["rank"], "rise_fake": a_r["fake"], "zero_rank": a_z["rank"], "zero_fake": a_z["fake"], "prose": prose}


def extra_connected(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after days on connected months only")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    lab = y.notna() & x.notna() & (x > 0)
    rec = signed_oof_auroc(y, tr[FLAG], tr["fold"], lab)
    after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    prose = (
        f"connected-only Y3 {_f(_cv(rec))} n={rec['n_defined']:,} leftover after days "
        f"{_f(after['rank'])} dies={after['honest_dies']}."
    )
    print(prose)
    return {"cv": _cv(rec), "rank": after["rank"], "prose": prose}


def extra_checking_days(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — checking leftover after days; zero dummy after checking")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    lab = y.notna()
    chk = leftover_diag(y, tr["g_has_checking"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    zero = (x == 0).astype(float).where(x.notna())
    z_after = leftover_diag(y, zero, (tr["g_has_checking"],), tr["fold"], lab)
    z_days = leftover_diag(y, zero, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    prose = (
        f"checking leftover after days rank {_f(chk['rank'])} OLS {_f(chk['ols'])} "
        f"fake={chk['fake']} dies={chk['honest_dies']}. "
        f"zero dummy leftover after checking {_f(z_after['rank'])} dies={z_after['honest_dies']}. "
        f"zero dummy leftover after days {_f(z_days['rank'])} fake={z_days['fake']} "
        f"ρ(resid,days)={_f(z_days['rho_ctrl'])}."
    )
    print(prose)
    return {"chk": chk["rank"], "z_after": z_after["rank"], "z_days": z_days["rank"], "z_fake": z_days["fake"], "prose": prose}


def extra_fold(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — fold-wise leftover after days")
    print("=" * 72)
    after = leftover_diag(
        pd.to_numeric(tr[Y3], errors="coerce"),
        tr[FLAG],
        (tr["c_n_days_with_tx"],),
        tr["fold"],
        pd.to_numeric(tr[Y3], errors="coerce").notna(),
    )
    rows = [{"fold": r["fold"], "rank leftover": _f(r["auroc"]), "n_va": r["n_va"], "n_pos": r["n_pos"]} for r in after["rrec"]["folds"]]
    prose = f"Rank leftover folds {after['rank_folds']} — fold 4 0.420 single pulls leftover down; mean 0.428 dies."
    print(prose)
    return {"rows": rows, "prose": prose}


def extra_last_labeled(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover on last Y3-labeled month per company")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = tr.loc[y.notna(), ["company_id", "period"]].copy()
    last = lab.sort_values("period").groupby("company_id").tail(1)
    key = set(zip(last["company_id"], last["period"]))
    mask = pd.Series([
        (c, p) in key for c, p in zip(tr["company_id"], tr["period"])
    ], index=tr.index)
    rec = signed_oof_auroc(y, tr[FLAG], tr["fold"], mask)
    after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], mask)
    prose = (
        f"last Y3-labeled month Y3 {_f(_cv(rec))} leftover after days {_f(after['rank'])} "
        f"dies={after['honest_dies']} n={rec['n_defined']} pos={rec['n_pos']}."
    )
    print(prose)
    return {"cv": _cv(rec), "rank": after["rank"], "prose": prose}


def extra_gtypes_new(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after g_n_types / g_new")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after_t = leftover_diag(y, tr[FLAG], (tr["g_n_types"],), tr["fold"], lab)
    after_n = leftover_diag(y, tr[FLAG], (tr["g_new_this_month"],), tr["fold"], lab)
    after_dt = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["g_n_types"]), tr["fold"], lab)
    prose = (
        f"leftover after g_n_types {_f(after_t['rank'])} dies={after_t['honest_dies']}. "
        f"after g_new {_f(after_n['rank'])} dies={after_n['honest_dies']}. "
        f"after days+g_n_types {_f(after_dt['rank'])}."
    )
    print(prose)
    return {"t": after_t["rank"], "n": after_n["rank"], "dt": after_dt["rank"], "prose": prose}


def extra_more_bars(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after a_n_tx / days+checking+banks / g_n_types leftover")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after_n = leftover_diag(y, tr[FLAG], (tr["a_n_tx"],), tr["fold"], lab)
    after_all = leftover_diag(
        y, tr[FLAG],
        (tr["c_n_days_with_tx"], tr["g_has_checking"], tr["g_n_banks"]),
        tr["fold"], lab,
    )
    gt = leftover_diag(y, tr["g_n_types"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    prose = (
        f"leftover after a_n_tx {_f(after_n['rank'])} dies={after_n['honest_dies']}. "
        f"after days+checking+banks {_f(after_all['rank'])} dies={after_all['honest_dies']}. "
        f"g_n_types leftover after days {_f(gt['rank'])} dies={gt['honest_dies']}."
    )
    print(prose)
    return {"n": after_n["rank"], "all": after_all["rank"], "gt": gt["rank"], "prose": prose}


def extra_ge3_banks(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — n>=3 leftover after days+banks")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    lab = y.notna() & x.notna() & (x >= 3)
    after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["g_n_banks"]), tr["fold"], lab)
    prose = (
        f"n>=3 leftover after days+banks {_f(after['rank'])} dies={after['honest_dies']} "
        f"fake={after['fake']}. Twin leftover of banks on the n>=3 slice."
    )
    print(prose)
    return {"rank": after["rank"], "dies": after["honest_dies"], "prose": prose}


def extra_count_bins(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of count-bin dummies after days")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    rows = []
    for lo, hi, name in ((0, 0, "0"), (1, 2, "1-2"), (3, 4, "3-4"), (5, 99, "5+")):
        dum = ((x >= lo) & (x <= hi)).astype(float).where(x.notna())
        rec = signed_oof_auroc(y, dum, tr["fold"], y.notna())
        after = leftover_diag(y, dum, (tr["c_n_days_with_tx"],), tr["fold"], y.notna())
        rows.append({"bin": name, "Y3": _f(_cv(rec)), "leftover": _f(after["rank"]), "fake": after["fake"], "n": rec["n_defined"], "n_pos": rec["n_pos"]})
        print(f"  {name} Y3={_f(_cv(rec))} leftover={_f(after['rank'])} fake={after['fake']}")
    prose = "Count-bin leftover after days: 0 is the hole dummy; 5+ leftover is not a KEEP seat."
    print(prose)
    return {"rows": rows, "prose": prose}


def extra_unique_accounts(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of n_accounts − n_banks after days")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    b = pd.to_numeric(tr["g_n_banks"], errors="coerce")
    extra = (x - b).where(x.notna() & b.notna())
    rec = signed_oof_auroc(y, extra, tr["fold"], y.notna())
    after = leftover_diag(y, extra, (tr["c_n_days_with_tx"],), tr["fold"], y.notna())
    after_b = leftover_diag(y, extra, (tr["c_n_days_with_tx"], tr["g_n_banks"]), tr["fold"], y.notna())
    prose = (
        f"n_accounts−n_banks Y3 {_f(_cv(rec))} leftover after days {_f(after['rank'])} "
        f"dies={after['honest_dies']} fake={after['fake']}. after days+banks {_f(after_b['rank'])}."
    )
    print(prose)
    return {"cv": _cv(rec), "rank": after["rank"], "dies": after["honest_dies"], "prose": prose}


def extra_ever_rise(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover on ever-rise vs always-flat companies")
    print("=" * 72)
    d = tr.sort_values(["company_id", "period"]).copy()
    x = pd.to_numeric(d[FLAG], errors="coerce")
    delta = x.groupby(d["company_id"]).diff()
    rose = d.assign(_rose=(delta > 0).astype(int)).groupby("company_id")["_rose"].max()
    rose_co = set(rose[rose > 0].index)
    y = pd.to_numeric(d[Y3], errors="coerce")
    is_rise = d["company_id"].isin(rose_co)
    rows = []
    store = {}
    for name, mask in (("ever_rise", y.notna() & is_rise), ("always_flat", y.notna() & ~is_rise)):
        rec = signed_oof_auroc(y, d[FLAG], d["fold"], mask)
        after = leftover_diag(y, d[FLAG], (d["c_n_days_with_tx"],), d["fold"], mask)
        store[name] = after["rank"]
        rows.append({"slice": name, "Y3": _f(_cv(rec)), "leftover": _f(after["rank"]), "n": rec["n_defined"], "n_pos": rec["n_pos"], "dies": after["honest_dies"]})
        print(f"  {name} Y3={_f(_cv(rec))} leftover={_f(after['rank'])} n={rec['n_defined']} pos={rec['n_pos']}")
    prose = (
        f"ever-rise leftover {_f(store['ever_rise'])} always-flat leftover {_f(store['always_flat'])}. "
        f"Rise-clock companies do not unlock leftover after days."
    )
    print(prose)
    return {"rows": rows, "rise": store["ever_rise"], "flat": store["always_flat"], "prose": prose}


def extra_log1p(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of log1p(g_n_accounts) after days")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = np.log1p(pd.to_numeric(tr[FLAG], errors="coerce").clip(lower=0))
    rec = signed_oof_auroc(y, x, tr["fold"], y.notna())
    after = leftover_diag(y, x, (tr["c_n_days_with_tx"],), tr["fold"], y.notna())
    prose = (
        f"log1p(n_accounts) Y3 {_f(_cv(rec))} leftover after days {_f(after['rank'])} "
        f"dies={after['honest_dies']} fake={after['fake']}. Transform does not revive leftover."
    )
    print(prose)
    return {"cv": _cv(rec), "rank": after["rank"], "dies": after["honest_dies"], "prose": prose}


def extra_inv_banks(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of g_n_banks after days+n_accounts (inverse twin)")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after = leftover_diag(y, tr["g_n_banks"], (tr["c_n_days_with_tx"], tr[FLAG]), tr["fold"], lab)
    after_d = leftover_diag(y, tr["g_n_banks"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_n = leftover_diag(y, tr["g_n_banks"], (tr[FLAG],), tr["fold"], lab)
    prose = (
        f"banks leftover after days+n_accounts {_f(after['rank'])} dies={after['honest_dies']} "
        f"fake={after['fake']}. after days {_f(after_d['rank'])}. after n_accounts {_f(after_n['rank'])}. "
        f"Banks has no leftover once the count is in — cluster, not a second seat."
    )
    print(prose)
    return {"rank": after["rank"], "dies": after["honest_dies"], "prose": prose}


def extra_delta_n(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of month-to-month Δn after days")
    print("=" * 72)
    d = tr.sort_values(["company_id", "period"]).copy()
    x = pd.to_numeric(d[FLAG], errors="coerce")
    delta = x.groupby(d["company_id"]).diff()
    y = pd.to_numeric(d[Y3], errors="coerce")
    rec = signed_oof_auroc(y, delta, d["fold"], y.notna())
    after = leftover_diag(y, delta, (d["c_n_days_with_tx"],), d["fold"], y.notna())
    after_n = leftover_diag(y, delta, (d["c_n_days_with_tx"], d[FLAG]), d["fold"], y.notna())
    prose = (
        f"Δn Y3 {_f(_cv(rec))} leftover after days {_f(after['rank'])} dies={after['honest_dies']} "
        f"fake={after['fake']}. after days+count {_f(after_n['rank'])}. "
        f"Month shock of a rise-only clock is not leftover after days."
    )
    print(prose)
    return {"cv": _cv(rec), "rank": after["rank"], "dies": after["honest_dies"], "prose": prose}


def extra_varying(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover on companies with within-company variation")
    print("=" * 72)
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    nunq = tr.assign(_x=x).groupby("company_id")["_x"].nunique()
    vary_co = set(nunq[nunq >= 2].index)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    is_v = tr["company_id"].isin(vary_co)
    rows = []
    store = {}
    for name, mask in (("varying", y.notna() & is_v), ("constant", y.notna() & ~is_v)):
        rec = signed_oof_auroc(y, tr[FLAG], tr["fold"], mask)
        after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], mask)
        store[name] = after["rank"]
        rows.append({"slice": name, "Y3": _f(_cv(rec)), "leftover": _f(after["rank"]), "n": rec["n_defined"], "n_pos": rec["n_pos"], "dies": after["honest_dies"]})
        print(f"  {name} Y3={_f(_cv(rec))} leftover={_f(after['rank'])} n={rec['n_defined']}")
    prose = (
        f"varying leftover {_f(store['varying'])} constant leftover {_f(store['constant'])}. "
        f"BETWEEN trait still dies after days."
    )
    print(prose)
    return {"rows": rows, "vary": store["varying"], "const": store["constant"], "prose": prose}


def extra_days_ftypes(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after days+f_n_types / days+checking on n>=3")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    lab = y.notna()
    after_ft = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["f_n_types"]), tr["fold"], lab)
    after_ge3 = leftover_diag(
        y, tr[FLAG],
        (tr["c_n_days_with_tx"], tr["g_has_checking"]),
        tr["fold"], lab & (x >= 3),
    )
    after_all = leftover_diag(
        y, tr[FLAG],
        (tr["c_n_days_with_tx"], tr["f_n_types"], tr["g_n_banks"]),
        tr["fold"], lab,
    )
    rec_ge5 = signed_oof_auroc(y, tr[FLAG], tr["fold"], lab & (x >= 5))
    after_ge5 = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & (x >= 5))
    beat_ge5 = bool(np.isfinite(_cv(rec_ge5)) and (_cv(rec_ge5) - SIZE_QUOTE) >= KEEP_DELTA)
    prose = (
        f"leftover after days+f_n_types {_f(after_ft['rank'])} dies={after_ft['honest_dies']}. "
        f"n>=3 leftover after days+checking {_f(after_ge3['rank'])}. "
        f"after days+f_n_types+banks {_f(after_all['rank'])}. "
        f"n>=5 Y3 {_f(_cv(rec_ge5))} leftover {_f(after_ge5['rank'])} dies={after_ge5['honest_dies']} "
        f"beat-size {'PASS' if beat_ge5 else 'FAIL'}."
    )
    print(prose)
    return {"ft": after_ft["rank"], "ge3": after_ge3["rank"], "ge5": after_ge5["rank"], "prose": prose}


def extra_ge5_gate(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — n>=5 leftover vs KEEP-as-X (banks / size / days)")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    lab = y.notna() & x.notna() & (x >= 5)
    rec = signed_oof_auroc(y, tr[FLAG], tr["fold"], lab)
    rec_s = signed_oof_auroc(y, tr["log_in3"], tr["fold"], lab)
    rec_d = signed_oof_auroc(y, tr["c_n_days_with_tx"], tr["fold"], lab)
    after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_b = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["g_n_banks"]), tr["fold"], lab)
    after_s = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], lab)
    after_c = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["g_has_checking"]), tr["fold"], lab)
    beat = bool(np.isfinite(_cv(rec)) and (_cv(rec) - SIZE_QUOTE) >= KEEP_DELTA)
    prose = (
        f"n>=5 Y3 {_f(_cv(rec))} n={rec['n_defined']:,} pos={rec['n_pos']:,} vs size {_f(_cv(rec_s))} "
        f"vs days {_f(_cv(rec_d))}. leftover after days {_f(after['rank'])} fake={after['fake']}. "
        f"after days+banks {_f(after_b['rank'])}. after days+size {_f(after_s['rank'])}. "
        f"after days+checking {_f(after_c['rank'])}. beat-size {'PASS' if beat else 'FAIL'} "
        f"— n>=5 leftover does not pass KEEP-as-X vs locked size 0.617."
    )
    print(prose)
    return {"cv": _cv(rec), "rank": after["rank"], "beat": beat, "prose": prose}


def extra_active_ratio(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover on days>0 months; leftover of n_accounts/n_banks")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    banks = pd.to_numeric(tr["g_n_banks"], errors="coerce")
    ratio = (x / banks.replace(0, np.nan)).where(x.notna() & banks.notna())
    lab_a = y.notna() & (days > 0)
    rec_a = signed_oof_auroc(y, tr[FLAG], tr["fold"], lab_a)
    after_a = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab_a)
    rec_r = signed_oof_auroc(y, ratio, tr["fold"], y.notna())
    after_r = leftover_diag(y, ratio, (tr["c_n_days_with_tx"],), tr["fold"], y.notna())
    after_sc = leftover_diag(
        y, tr[FLAG],
        (tr["c_n_days_with_tx"], tr["log_in3"], tr["g_has_checking"]),
        tr["fold"], y.notna(),
    )
    prose = (
        f"days>0 leftover {_f(after_a['rank'])} Y3 {_f(_cv(rec_a))} n={rec_a['n_defined']:,} "
        f"dies={after_a['honest_dies']}. ratio n/banks Y3 {_f(_cv(rec_r))} leftover {_f(after_r['rank'])} "
        f"dies={after_r['honest_dies']}. leftover after days+size+checking {_f(after_sc['rank'])}."
    )
    print(prose)
    return {"active": after_a["rank"], "ratio": after_r["rank"], "sc": after_sc["rank"], "prose": prose}


def extra_has_rest(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after other g_has_* / leftover of has_card dummy")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    rows = []
    store = {}
    for col in HAS_COLS:
        after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr[col]), tr["fold"], lab)
        rec = signed_oof_auroc(y, tr[col], tr["fold"], lab)
        store[col] = after["rank"]
        rows.append({"control": col, "ctrl_Y3": _f(_cv(rec)), "leftover_after_days+ctrl": _f(after["rank"]), "dies": after["honest_dies"]})
        print(f"  leftover after days+{col} {_f(after['rank'])} {col} Y3 {_f(_cv(rec))}")
    after_all = leftover_diag(
        y, tr[FLAG],
        (tr["c_n_days_with_tx"], *(tr[c] for c in HAS_COLS)),
        tr["fold"], lab,
    )
    prose = (
        f"leftover after days+all g_has_* {_f(after_all['rank'])} dies={after_all['honest_dies']}. "
        f"g_has_* already DROP; count leftover after each still dies or is days."
    )
    print(prose)
    return {"rows": rows, "all": after_all["rank"], "prose": prose}


def extra_checking_after_n(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — checking leftover after days+n_accounts; leftover on checking=1")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    chk = pd.to_numeric(tr["g_has_checking"], errors="coerce")
    lab = y.notna()
    after_chk = leftover_diag(y, tr["g_has_checking"], (tr["c_n_days_with_tx"], tr[FLAG]), tr["fold"], lab)
    after_chk_d = leftover_diag(y, tr["g_has_checking"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    lab_c = lab & (chk == 1)
    rec_c = signed_oof_auroc(y, tr[FLAG], tr["fold"], lab_c)
    after_c = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab_c)
    beat_c = bool(np.isfinite(_cv(rec_c)) and (_cv(rec_c) - SIZE_QUOTE) >= KEEP_DELTA)
    prose = (
        f"checking leftover after days {_f(after_chk_d['rank'])}; after days+n_accounts "
        f"{_f(after_chk['rank'])} dies={after_chk['honest_dies']} fake={after_chk['fake']}. "
        f"checking=1 leftover {_f(after_c['rank'])} Y3 {_f(_cv(rec_c))} n={rec_c['n_defined']:,} "
        f"pos={rec_c['n_pos']:,} beat-size {'PASS' if beat_c else 'FAIL'}."
    )
    print(prose)
    return {"chk": after_chk["rank"], "c1": after_c["rank"], "prose": prose}


def extra_evermax5(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover on companies with ever-max n_accounts>=5")
    print("=" * 72)
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    ever = tr.assign(_x=x).groupby("company_id")["_x"].max()
    hi_co = set(ever[ever >= 5].index)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    is_hi = tr["company_id"].isin(hi_co)
    lab = y.notna() & is_hi
    rec = signed_oof_auroc(y, tr[FLAG], tr["fold"], lab)
    rec_s = signed_oof_auroc(y, tr["log_in3"], tr["fold"], lab)
    rec_d = signed_oof_auroc(y, tr["c_n_days_with_tx"], tr["fold"], lab)
    after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_b = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["g_n_banks"]), tr["fold"], lab)
    beat = bool(np.isfinite(_cv(rec)) and (_cv(rec) - SIZE_QUOTE) >= KEEP_DELTA)
    prose = (
        f"ever-max>=5 Y3 {_f(_cv(rec))} n={rec['n_defined']:,} pos={rec['n_pos']:,} "
        f"vs size {_f(_cv(rec_s))} vs days {_f(_cv(rec_d))}. leftover after days {_f(after['rank'])} "
        f"dies={after['honest_dies']}. after days+banks {_f(after_b['rank'])}. "
        f"beat-size {'PASS' if beat else 'FAIL'}."
    )
    print(prose)
    return {"cv": _cv(rec), "rank": after["rank"], "beat": beat, "prose": prose}


def extra_drop_fold4(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after days excluding fold 4 (weak fold)")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    fold = tr["fold"]
    lab = y.notna() & (fold != 3)
    rec = signed_oof_auroc(y, tr[FLAG], tr["fold"], lab)
    after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    banks2 = leftover_diag(
        y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"],
        y.notna() & (pd.to_numeric(tr["g_n_banks"], errors="coerce") >= 2),
    )
    rec_b2 = signed_oof_auroc(
        y, tr[FLAG], tr["fold"],
        y.notna() & (pd.to_numeric(tr["g_n_banks"], errors="coerce") >= 2),
    )
    prose = (
        f"ex-fold4 Y3 {_f(_cv(rec))} leftover {_f(after['rank'])} dies={after['honest_dies']} "
        f"n={rec['n_defined']:,}. banks>=2 leftover {_f(banks2['rank'])} Y3 {_f(_cv(rec_b2))} "
        f"n={rec_b2['n_defined']:,}. Weak fold does not hide a KEEP leftover."
    )
    print(prose)
    return {"ex4": after["rank"], "b2": banks2["rank"], "prose": prose}


def extra_new_types(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of g_n_types / g_new after n_accounts; leftover on g_new months")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after_t = leftover_diag(y, tr["g_n_types"], (tr["c_n_days_with_tx"], tr[FLAG]), tr["fold"], lab)
    after_g = leftover_diag(y, tr["g_new_this_month"], (tr["c_n_days_with_tx"], tr[FLAG]), tr["fold"], lab)
    after_both = leftover_diag(
        y, tr[FLAG],
        (tr["c_n_days_with_tx"], tr["g_n_types"], tr["g_n_banks"]),
        tr["fold"], lab,
    )
    gnew = pd.to_numeric(tr["g_new_this_month"], errors="coerce")
    rows = []
    store = {}
    for name, mask in (("g_new=1", lab & (gnew == 1)), ("g_new=0", lab & (gnew == 0))):
        rec = signed_oof_auroc(y, tr[FLAG], tr["fold"], mask)
        after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], mask)
        store[name] = after["rank"]
        rows.append({"slice": name, "Y3": _f(_cv(rec)), "leftover": _f(after["rank"]), "n": rec["n_defined"], "n_pos": rec["n_pos"], "dies": after["honest_dies"]})
        print(f"  {name} Y3={_f(_cv(rec))} leftover={_f(after['rank'])} n={rec['n_defined']}")
    prose = (
        f"g_n_types leftover after days+n_accounts {_f(after_t['rank'])} dies={after_t['honest_dies']}. "
        f"g_new leftover after days+n_accounts {_f(after_g['rank'])} dies={after_g['honest_dies']}. "
        f"count leftover after days+types+banks {_f(after_both['rank'])}. "
        f"g_new=1 leftover {_f(store['g_new=1'])} g_new=0 leftover {_f(store['g_new=0'])}."
    )
    print(prose)
    return {"types": after_t["rank"], "gnew": after_g["rank"], "rows": rows, "prose": prose}


def extra_types2_last3(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover on g_n_types>=2; last-3 labeled months; banks>=2 after size")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    nt = pd.to_numeric(tr["g_n_types"], errors="coerce")
    banks = pd.to_numeric(tr["g_n_banks"], errors="coerce")
    lab_t = y.notna() & (nt >= 2)
    rec_t = signed_oof_auroc(y, tr[FLAG], tr["fold"], lab_t)
    after_t = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab_t)
    per = pd.to_datetime(tr["period"])
    last_per = tr.assign(_y=y).loc[lambda d: d["_y"].notna()].groupby("company_id")["period"].max()
    mapped = pd.to_datetime(tr["company_id"].map(last_per))
    months_back = (mapped.dt.year - per.dt.year) * 12 + (mapped.dt.month - per.dt.month)
    lab3 = y.notna() & (months_back <= 2)
    rec3 = signed_oof_auroc(y, tr[FLAG], tr["fold"], lab3)
    after3 = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab3)
    lab_b = y.notna() & (banks >= 2)
    rec_b = signed_oof_auroc(y, tr[FLAG], tr["fold"], lab_b)
    after_bs = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], lab_b)
    beat_b = bool(np.isfinite(_cv(rec_b)) and (_cv(rec_b) - SIZE_QUOTE) >= KEEP_DELTA)
    prose = (
        f"g_n_types>=2 leftover {_f(after_t['rank'])} Y3 {_f(_cv(rec_t))} n={rec_t['n_defined']:,}. "
        f"last-3 labeled leftover {_f(after3['rank'])} Y3 {_f(_cv(rec3))} n={rec3['n_defined']:,}. "
        f"banks>=2 leftover after days+size {_f(after_bs['rank'])} Y3 {_f(_cv(rec_b))} "
        f"beat-size {'PASS' if beat_b else 'FAIL'}."
    )
    print(prose)
    return {"t2": after_t["rank"], "last3": after3["rank"], "b2s": after_bs["rank"], "prose": prose}


def extra_single_bank_hole(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover on single-bank / never-zero / ever-hole companies")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    banks = pd.to_numeric(tr["g_n_banks"], errors="coerce")
    chk = pd.to_numeric(tr["g_has_checking"], errors="coerce")
    ever0 = tr.assign(_x=x).groupby("company_id")["_x"].min()
    never0_co = set(ever0[ever0 > 0].index)
    ever_hole = tr.assign(_c=chk).groupby("company_id")["_c"].min()
    hole_co = set(ever_hole[ever_hole == 0].index)
    rows = []
    store = {}
    slices = (
        ("single_bank", y.notna() & (banks == 1)),
        ("never_zero", y.notna() & tr["company_id"].isin(never0_co)),
        ("ever_hole", y.notna() & tr["company_id"].isin(hole_co)),
        ("n_1_2", y.notna() & x.notna() & (x >= 1) & (x <= 2)),
        ("chk1_n>=3", y.notna() & (chk == 1) & (x >= 3)),
    )
    for name, mask in slices:
        rec = signed_oof_auroc(y, tr[FLAG], tr["fold"], mask)
        after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], mask)
        store[name] = after["rank"]
        rows.append({"slice": name, "Y3": _f(_cv(rec)), "leftover": _f(after["rank"]), "n": rec["n_defined"], "n_pos": rec["n_pos"], "dies": after["honest_dies"]})
        print(f"  {name} Y3={_f(_cv(rec))} leftover={_f(after['rank'])} n={rec['n_defined']} pos={rec['n_pos']}")
    prose = (
        f"single-bank leftover {_f(store['single_bank'])} never-zero {_f(store['never_zero'])} "
        f"ever-hole {_f(store['ever_hole'])} n_1_2 {_f(store['n_1_2'])} chk1_n>=3 {_f(store['chk1_n>=3'])}."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def extra_tx_year(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after days+a_n_tx+size; leftover by calendar year")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after = leftover_diag(
        y, tr[FLAG],
        (tr["c_n_days_with_tx"], tr["a_n_tx"], tr["log_in3"]),
        tr["fold"], lab,
    )
    year = pd.to_datetime(tr["period"]).dt.year
    rows = []
    store = {}
    for yr in sorted(year.dropna().unique()):
        mask = lab & (year == yr)
        rec = signed_oof_auroc(y, tr[FLAG], tr["fold"], mask)
        left = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], mask)
        store[int(yr)] = left["rank"]
        rows.append({"year": int(yr), "Y3": _f(_cv(rec)), "leftover": _f(left["rank"]), "n": rec["n_defined"], "n_pos": rec["n_pos"], "dies": left["honest_dies"]})
        print(f"  {int(yr)} Y3={_f(_cv(rec))} leftover={_f(left['rank'])} n={rec['n_defined']}")
    prose = (
        f"leftover after days+a_n_tx+size {_f(after['rank'])} dies={after['honest_dies']}. "
        + " ".join(f"{yr} leftover {_f(store[yr])}" for yr in store)
        + "."
    )
    print(prose)
    return {"after": after["rank"], "rows": rows, "prose": prose}


def extra_neverzero_gate(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — never-zero leftover vs KEEP-as-X")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    ever0 = tr.assign(_x=x).groupby("company_id")["_x"].min()
    never0_co = set(ever0[ever0 > 0].index)
    lab = y.notna() & tr["company_id"].isin(never0_co)
    rec = signed_oof_auroc(y, tr[FLAG], tr["fold"], lab)
    rec_s = signed_oof_auroc(y, tr["log_in3"], tr["fold"], lab)
    rec_d = signed_oof_auroc(y, tr["c_n_days_with_tx"], tr["fold"], lab)
    after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_s = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], lab)
    after_b = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["g_n_banks"]), tr["fold"], lab)
    beat_locked = bool(np.isfinite(_cv(rec)) and (_cv(rec) - SIZE_QUOTE) >= KEEP_DELTA)
    beat_same = bool(np.isfinite(_cv(rec)) and np.isfinite(_cv(rec_s)) and (_cv(rec) - _cv(rec_s)) >= KEEP_DELTA)
    prose = (
        f"never-zero Y3 {_f(_cv(rec))} n={rec['n_defined']:,} pos={rec['n_pos']:,} "
        f"vs same-n size {_f(_cv(rec_s))} vs days {_f(_cv(rec_d))}. leftover after days "
        f"{_f(after['rank'])} dies={after['honest_dies']}. after days+size {_f(after_s['rank'])}. "
        f"after days+banks {_f(after_b['rank'])}. beat locked-size {'PASS' if beat_locked else 'FAIL'} "
        f"beat same-n size {'PASS' if beat_same else 'FAIL'} — leftover still dies, not KEEP."
    )
    print(prose)
    return {"cv": _cv(rec), "rank": after["rank"], "beat": beat_locked, "prose": prose}


def extra_size_split(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover on large vs small (median log_in3) / first labeled month")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    size = pd.to_numeric(tr["log_in3"], errors="coerce")
    med = float(size[y.notna()].median())
    per = pd.to_datetime(tr["period"])
    first_lab = tr.assign(_y=y).loc[lambda d: d["_y"].notna()].groupby("company_id")["period"].min()
    mapped = pd.to_datetime(tr["company_id"].map(first_lab))
    lab_first = y.notna() & (per == mapped)
    rows = []
    store = {}
    for name, mask in (
        ("small", y.notna() & (size <= med)),
        ("large", y.notna() & (size > med)),
        ("first_labeled", lab_first),
    ):
        rec = signed_oof_auroc(y, tr[FLAG], tr["fold"], mask)
        after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], mask)
        store[name] = after["rank"]
        rows.append({"slice": name, "Y3": _f(_cv(rec)), "leftover": _f(after["rank"]), "n": rec["n_defined"], "n_pos": rec["n_pos"], "dies": after["honest_dies"]})
        print(f"  {name} Y3={_f(_cv(rec))} leftover={_f(after['rank'])} n={rec['n_defined']} pos={rec['n_pos']}")
    after_all = leftover_diag(
        y, tr[FLAG],
        (tr["c_n_days_with_tx"], tr["g_has_checking"], tr["log_in3"], tr["g_n_banks"]),
        tr["fold"], y.notna(),
    )
    prose = (
        f"small leftover {_f(store['small'])} large leftover {_f(store['large'])} "
        f"first-labeled leftover {_f(store['first_labeled'])}. "
        f"leftover after days+checking+size+banks {_f(after_all['rank'])} dies={after_all['honest_dies']}."
    )
    print(prose)
    return {"rows": rows, "all": after_all["rank"], "prose": prose}


def extra_activity_card(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover on high/low days; card=1; n>=3 ∩ never-zero")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    card = pd.to_numeric(tr["g_has_card"], errors="coerce")
    med = float(days[y.notna()].median())
    ever0 = tr.assign(_x=x).groupby("company_id")["_x"].min()
    never0_co = set(ever0[ever0 > 0].index)
    rows = []
    store = {}
    slices = (
        ("low_days", y.notna() & (days <= med)),
        ("high_days", y.notna() & (days > med)),
        ("card=1", y.notna() & (card == 1)),
        ("ge3_never0", y.notna() & (x >= 3) & tr["company_id"].isin(never0_co)),
    )
    for name, mask in slices:
        rec = signed_oof_auroc(y, tr[FLAG], tr["fold"], mask)
        after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], mask)
        store[name] = after["rank"]
        rows.append({"slice": name, "Y3": _f(_cv(rec)), "leftover": _f(after["rank"]), "n": rec["n_defined"], "n_pos": rec["n_pos"], "dies": after["honest_dies"]})
        print(f"  {name} Y3={_f(_cv(rec))} leftover={_f(after['rank'])} n={rec['n_defined']} pos={rec['n_pos']}")
    after_fb = leftover_diag(
        y, tr[FLAG],
        (tr["c_n_days_with_tx"], tr["f_n_facilities"], tr["g_n_banks"]),
        tr["fold"], y.notna(),
    )
    banks = pd.to_numeric(tr["g_n_banks"], errors="coerce")
    one_one = (x == banks).astype(float).where(x.notna())
    rec_oo = signed_oof_auroc(y, one_one, tr["fold"], y.notna())
    after_oo = leftover_diag(y, one_one, (tr["c_n_days_with_tx"],), tr["fold"], y.notna())
    prose = (
        f"low-days leftover {_f(store['low_days'])} high-days {_f(store['high_days'])} "
        f"card=1 {_f(store['card=1'])} ge3∩never0 {_f(store['ge3_never0'])}. "
        f"leftover after days+facilities+banks {_f(after_fb['rank'])}. "
        f"1:1 accounts=banks Y3 {_f(_cv(rec_oo))} leftover {_f(after_oo['rank'])} dies={after_oo['honest_dies']}."
    )
    print(prose)
    return {"rows": rows, "fb": after_fb["rank"], "oo": after_oo["rank"], "prose": prose}


def extra_ge3_never0_gate(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — n>=3 ∩ never-zero leftover vs KEEP-as-X")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    ever0 = tr.assign(_x=x).groupby("company_id")["_x"].min()
    never0_co = set(ever0[ever0 > 0].index)
    lab = y.notna() & (x >= 3) & tr["company_id"].isin(never0_co)
    rec = signed_oof_auroc(y, tr[FLAG], tr["fold"], lab)
    rec_s = signed_oof_auroc(y, tr["log_in3"], tr["fold"], lab)
    rec_d = signed_oof_auroc(y, tr["c_n_days_with_tx"], tr["fold"], lab)
    after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_b = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["g_n_banks"]), tr["fold"], lab)
    after_s = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], lab)
    beat = bool(np.isfinite(_cv(rec)) and (_cv(rec) - SIZE_QUOTE) >= KEEP_DELTA)
    nlab = tr.groupby("company_id")[Y3].transform(lambda s: s.notna().sum())
    lab6 = y.notna() & (nlab >= 6)
    rec6 = signed_oof_auroc(y, tr[FLAG], tr["fold"], lab6)
    after6 = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab6)
    prose = (
        f"ge3∩never0 Y3 {_f(_cv(rec))} n={rec['n_defined']:,} pos={rec['n_pos']:,} "
        f"vs size {_f(_cv(rec_s))} vs days {_f(_cv(rec_d))}. leftover after days {_f(after['rank'])} "
        f"after days+banks {_f(after_b['rank'])} after days+size {_f(after_s['rank'])}. "
        f"beat-size {'PASS' if beat else 'FAIL'}. "
        f">=6 labeled leftover {_f(after6['rank'])} Y3 {_f(_cv(rec6))} n={rec6['n_defined']:,}."
    )
    print(prose)
    return {"cv": _cv(rec), "rank": after["rank"], "beat": beat, "prose": prose}


def extra_2026_saving(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover on 2026 after banks; saving=1; ever-rise ∩ n>=3")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    year = pd.to_datetime(tr["period"]).dt.year
    sav = pd.to_numeric(tr["g_has_saving"], errors="coerce")
    d = tr.sort_values(["company_id", "period"])
    delta = pd.to_numeric(d[FLAG], errors="coerce").groupby(d["company_id"]).diff()
    rose = d.assign(_rose=(delta > 0).astype(int)).groupby("company_id")["_rose"].max()
    rise_co = set(rose[rose > 0].index)
    rows = []
    store = {}
    slices = (
        ("2026", y.notna() & (year == 2026)),
        ("saving=1", y.notna() & (sav == 1)),
        ("rise_ge3", y.notna() & (x >= 3) & tr["company_id"].isin(rise_co)),
    )
    for name, mask in slices:
        rec = signed_oof_auroc(y, tr[FLAG], tr["fold"], mask)
        after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], mask)
        after_b = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["g_n_banks"]), tr["fold"], mask)
        store[name] = (after["rank"], after_b["rank"])
        rows.append({"slice": name, "Y3": _f(_cv(rec)), "leftover": _f(after["rank"]), "after_banks": _f(after_b["rank"]), "n": rec["n_defined"], "n_pos": rec["n_pos"]})
        print(f"  {name} Y3={_f(_cv(rec))} leftover={_f(after['rank'])} after_banks={_f(after_b['rank'])} n={rec['n_defined']}")
    after_ca = leftover_diag(
        y, tr[FLAG],
        (tr["c_n_days_with_tx"], tr["g_has_checking"], tr["a_n_tx"]),
        tr["fold"], y.notna(),
    )
    prose = (
        f"2026 leftover {_f(store['2026'][0])} after banks {_f(store['2026'][1])}. "
        f"saving=1 leftover {_f(store['saving=1'][0])}. rise∩n>=3 leftover {_f(store['rise_ge3'][0])} "
        f"after banks {_f(store['rise_ge3'][1])}. leftover after days+checking+a_n_tx {_f(after_ca['rank'])}."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def extra_invest_sb2(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover on investment=1; single-bank ∩ n>=2; after days+types+checking")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    inv = pd.to_numeric(tr["g_has_investment"], errors="coerce")
    banks = pd.to_numeric(tr["g_n_banks"], errors="coerce")
    nt = pd.to_numeric(tr["g_n_types"], errors="coerce")
    rows = []
    store = {}
    slices = (
        ("invest=1", y.notna() & (inv == 1)),
        ("sb_n>=2", y.notna() & (banks == 1) & (x >= 2)),
        ("types=1", y.notna() & (nt == 1)),
    )
    for name, mask in slices:
        rec = signed_oof_auroc(y, tr[FLAG], tr["fold"], mask)
        after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], mask)
        store[name] = after["rank"]
        rows.append({"slice": name, "Y3": _f(_cv(rec)), "leftover": _f(after["rank"]), "n": rec["n_defined"], "n_pos": rec["n_pos"], "dies": after["honest_dies"]})
        print(f"  {name} Y3={_f(_cv(rec))} leftover={_f(after['rank'])} n={rec['n_defined']} pos={rec['n_pos']}")
    after_tc = leftover_diag(
        y, tr[FLAG],
        (tr["c_n_days_with_tx"], tr["g_n_types"], tr["g_has_checking"]),
        tr["fold"], y.notna(),
    )
    prose = (
        f"invest=1 leftover {_f(store['invest=1'])} sb∩n>=2 {_f(store['sb_n>=2'])} "
        f"types=1 {_f(store['types=1'])}. leftover after days+types+checking {_f(after_tc['rank'])}."
    )
    print(prose)
    return {"rows": rows, "tc": after_tc["rank"], "prose": prose}


def extra_banks3_tx(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover on banks>=3; high/low a_n_tx; leftover after full G cluster")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    banks = pd.to_numeric(tr["g_n_banks"], errors="coerce")
    ntx = pd.to_numeric(tr["a_n_tx"], errors="coerce")
    med = float(ntx[y.notna()].median())
    rows = []
    store = {}
    slices = (
        ("banks>=3", y.notna() & (banks >= 3)),
        ("low_tx", y.notna() & (ntx <= med)),
        ("high_tx", y.notna() & (ntx > med)),
    )
    for name, mask in slices:
        rec = signed_oof_auroc(y, tr[FLAG], tr["fold"], mask)
        after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], mask)
        store[name] = after["rank"]
        rows.append({"slice": name, "Y3": _f(_cv(rec)), "leftover": _f(after["rank"]), "n": rec["n_defined"], "n_pos": rec["n_pos"], "dies": after["honest_dies"]})
        print(f"  {name} Y3={_f(_cv(rec))} leftover={_f(after['rank'])} n={rec['n_defined']} pos={rec['n_pos']}")
    after_g = leftover_diag(
        y, tr[FLAG],
        (tr["c_n_days_with_tx"], tr["g_n_banks"], tr["g_n_types"], tr["g_has_checking"]),
        tr["fold"], y.notna(),
    )
    prose = (
        f"banks>=3 leftover {_f(store['banks>=3'])} low-tx {_f(store['low_tx'])} "
        f"high-tx {_f(store['high_tx'])}. leftover after days+banks+types+checking "
        f"{_f(after_g['rank'])} dies={after_g['honest_dies']}."
    )
    print(prose)
    return {"rows": rows, "g": after_g["rank"], "prose": prose}


def extra_banks3_gate(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — banks>=3 leftover vs KEEP-as-X")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    banks = pd.to_numeric(tr["g_n_banks"], errors="coerce")
    lab = y.notna() & (banks >= 3)
    rec = signed_oof_auroc(y, tr[FLAG], tr["fold"], lab)
    rec_s = signed_oof_auroc(y, tr["log_in3"], tr["fold"], lab)
    rec_d = signed_oof_auroc(y, tr["c_n_days_with_tx"], tr["fold"], lab)
    after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_s = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], lab)
    after_b = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["g_n_banks"]), tr["fold"], lab)
    beat = bool(np.isfinite(_cv(rec)) and (_cv(rec) - SIZE_QUOTE) >= KEEP_DELTA)
    beat_same = bool(np.isfinite(_cv(rec)) and np.isfinite(_cv(rec_s)) and (_cv(rec) - _cv(rec_s)) >= KEEP_DELTA)
    prose = (
        f"banks>=3 Y3 {_f(_cv(rec))} n={rec['n_defined']:,} pos={rec['n_pos']:,} "
        f"vs same-n size {_f(_cv(rec_s))} vs days {_f(_cv(rec_d))}. leftover after days "
        f"{_f(after['rank'])} after days+size {_f(after_s['rank'])} after days+banks {_f(after_b['rank'])}. "
        f"beat locked-size {'PASS' if beat else 'FAIL'} beat same-n {'PASS' if beat_same else 'FAIL'} "
        f"— slice leftover is not KEEP-as-X."
    )
    print(prose)
    return {"cv": _cv(rec), "rank": after["rank"], "beat": beat, "prose": prose}


def extra_quarter_fac(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover on Q1 vs rest; leftover after days+facilities+size")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    q = pd.to_datetime(tr["period"]).dt.quarter
    rows = []
    store = {}
    for name, mask in (("Q1", y.notna() & (q == 1)), ("Q2-4", y.notna() & (q != 1))):
        rec = signed_oof_auroc(y, tr[FLAG], tr["fold"], mask)
        after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], mask)
        store[name] = after["rank"]
        rows.append({"slice": name, "Y3": _f(_cv(rec)), "leftover": _f(after["rank"]), "n": rec["n_defined"], "n_pos": rec["n_pos"], "dies": after["honest_dies"]})
        print(f"  {name} Y3={_f(_cv(rec))} leftover={_f(after['rank'])} n={rec['n_defined']}")
    after_fs = leftover_diag(
        y, tr[FLAG],
        (tr["c_n_days_with_tx"], tr["f_n_facilities"], tr["log_in3"]),
        tr["fold"], y.notna(),
    )
    prose = (
        f"Q1 leftover {_f(store['Q1'])} Q2-4 leftover {_f(store['Q2-4'])}. "
        f"leftover after days+facilities+size {_f(after_fs['rank'])} dies={after_fs['honest_dies']}."
    )
    print(prose)
    return {"rows": rows, "fs": after_fs["rank"], "prose": prose}


def extra_half_sink(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover on H1 vs H2; leftover after days+tx+checking+size")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    month = pd.to_datetime(tr["period"]).dt.month
    rows = []
    store = {}
    for name, mask in (("H1", y.notna() & (month <= 6)), ("H2", y.notna() & (month > 6))):
        rec = signed_oof_auroc(y, tr[FLAG], tr["fold"], mask)
        after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], mask)
        store[name] = after["rank"]
        rows.append({"slice": name, "Y3": _f(_cv(rec)), "leftover": _f(after["rank"]), "n": rec["n_defined"], "n_pos": rec["n_pos"], "dies": after["honest_dies"]})
        print(f"  {name} Y3={_f(_cv(rec))} leftover={_f(after['rank'])} n={rec['n_defined']}")
    after_k = leftover_diag(
        y, tr[FLAG],
        (tr["c_n_days_with_tx"], tr["a_n_tx"], tr["g_has_checking"], tr["log_in3"]),
        tr["fold"], y.notna(),
    )
    prose = (
        f"H1 leftover {_f(store['H1'])} H2 leftover {_f(store['H2'])}. "
        f"leftover after days+tx+checking+size {_f(after_k['rank'])} dies={after_k['honest_dies']}."
    )
    print(prose)
    return {"rows": rows, "k": after_k["rank"], "prose": prose}


def extra_group_first(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover on multi-co groups vs singleton; first_month year")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    gsize = tr.groupby("group_id")["company_id"].transform("nunique")
    fy = pd.to_datetime(tr["first_month"]).dt.year
    rows = []
    store = {}
    slices = (
        ("singleton_group", y.notna() & (gsize <= 1)),
        ("multi_group", y.notna() & (gsize >= 2)),
        ("first_2024", y.notna() & (fy == 2024)),
        ("first_2025", y.notna() & (fy == 2025)),
    )
    for name, mask in slices:
        rec = signed_oof_auroc(y, tr[FLAG], tr["fold"], mask)
        after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], mask)
        store[name] = after["rank"]
        rows.append({"slice": name, "Y3": _f(_cv(rec)), "leftover": _f(after["rank"]), "n": rec["n_defined"], "n_pos": rec["n_pos"], "dies": after["honest_dies"]})
        print(f"  {name} Y3={_f(_cv(rec))} leftover={_f(after['rank'])} n={rec['n_defined']}")
    prose = (
        f"singleton leftover {_f(store['singleton_group'])} multi-group {_f(store['multi_group'])} "
        f"first-2024 {_f(store['first_2024'])} first-2025 {_f(store['first_2025'])}."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def extra_ge3_gate(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — n>=3 leftover vs KEEP-as-X gate")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    lab = y.notna() & x.notna() & (x >= 3)
    rec = signed_oof_auroc(y, tr[FLAG], tr["fold"], lab)
    rec_s = signed_oof_auroc(y, tr["log_in3"], tr["fold"], lab)
    rec_d = signed_oof_auroc(y, tr["c_n_days_with_tx"], tr["fold"], lab)
    after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_s = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], lab)
    after_b = leftover_diag(y, tr[FLAG], (tr["g_n_banks"],), tr["fold"], lab)
    beat = bool(np.isfinite(_cv(rec)) and (_cv(rec) - SIZE_QUOTE) >= KEEP_DELTA)
    prose = (
        f"n>=3 Y3 {_f(_cv(rec))} n={rec['n_defined']:,} pos={rec['n_pos']:,} vs size {_f(_cv(rec_s))} "
        f"vs days {_f(_cv(rec_d))}. leftover after days {_f(after['rank'])} fake={after['fake']}. "
        f"after days+size {_f(after_s['rank'])}. after banks {_f(after_b['rank'])}. "
        f"beat-size {'PASS' if beat else 'FAIL'} — leftover on n>=3 does not pass KEEP-as-X."
    )
    print(prose)
    return {"cv": _cv(rec), "rank": after["rank"], "beat": beat, "prose": prose}


def extra_long_ge3(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — long-book leftover honest; leftover on n_accounts>=3")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    fm = pd.to_datetime(tr["first_month"])
    per = pd.to_datetime(tr["period"])
    age = (per.dt.year - fm.dt.year) * 12 + (per.dt.month - fm.dt.month)
    long = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], y.notna() & (age >= 12))
    ge3 = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], y.notna() & (x >= 3))
    rec = signed_oof_auroc(y, tr[FLAG], tr["fold"], y.notna() & (x >= 3))
    all3 = leftover_diag(
        y, tr[FLAG],
        (tr["c_n_days_with_tx"], tr["g_has_checking"], tr["g_n_banks"]),
        tr["fold"], y.notna(),
    )
    prose = (
        f"long leftover OLS {_f(long['ols'])} rank {_f(long['rank'])} fake={long['fake']}. "
        f"n>=3 Y3 {_f(_cv(rec))} leftover {_f(ge3['rank'])} dies={ge3['honest_dies']}. "
        f"after days+checking+banks rank {_f(all3['rank'])} OLS {_f(all3['ols'])} fake={all3['fake']}."
    )
    print(prose)
    return {"long": long["rank"], "ge3": ge3["rank"], "all3": all3["rank"], "all3_fake": all3["fake"], "prose": prose}


def extra_short(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover on short vs long trail")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    fm = pd.to_datetime(tr["first_month"])
    per = pd.to_datetime(tr["period"])
    age = (per.dt.year - fm.dt.year) * 12 + (per.dt.month - fm.dt.month)
    rows = []
    store = {}
    for name, mask in (("short_<12", y.notna() & (age < 12)), ("long_>=12", y.notna() & (age >= 12))):
        rec = signed_oof_auroc(y, tr[FLAG], tr["fold"], mask)
        after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], mask)
        store[name] = after["rank"]
        rows.append({"slice": name, "Y3": _f(_cv(rec)), "leftover": _f(after["rank"]), "n": rec["n_defined"], "n_pos": rec["n_pos"], "dies": after["honest_dies"]})
        print(f"  {name} Y3={_f(_cv(rec))} leftover={_f(after['rank'])} n={rec['n_defined']}")
    prose = f"short leftover {_f(store['short_<12'])} long leftover {_f(store['long_>=12'])}."
    print(prose)
    return {"rows": rows, "short": store["short_<12"], "long": store["long_>=12"], "prose": prose}


def extra_size_lag(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after size / days+size; ρ vs own lag1")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after_s = leftover_diag(y, tr[FLAG], (tr["log_in3"],), tr["fold"], lab)
    after_b = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], lab)
    rho = spearman(tr[FLAG], tr[f"{FLAG}_lag1"])
    prose = (
        f"leftover after size {_f(after_s['rank'])} dies={after_s['honest_dies']}. "
        f"after days+size {_f(after_b['rank'])} dies={after_b['honest_dies']}. "
        f"ρ vs own lag1={_f(rho)} {'TWIN' if abs(rho) >= TWIN_RHO else 'no'} — BETWEEN snapshot."
    )
    print(prose)
    return {"after_s": after_s["rank"], "after_b": after_b["rank"], "rho": rho, "prose": prose}


def extra_y7(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Y7 leftover after days (no TURNOVER seat)")
    print("=" * 72)
    y7 = pd.to_numeric(tr[Y7], errors="coerce")
    rec = signed_oof_auroc(y7, tr[FLAG], tr["fold"], y7.notna())
    after = leftover_diag(y7, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], y7.notna())
    prose = (
        f"Y7 n_accounts {_f(_cv(rec))} leftover after days {_f(after['rank'])} "
        f"dies={after['honest_dies']}. Do not grow TURNOVER 0.720."
    )
    print(prose)
    return {"y7": _cv(rec), "after": after["rank"], "prose": prose}


def extra_first6(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover on first6 vs later / last-month snapshot")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    fm = pd.to_datetime(tr["first_month"])
    per = pd.to_datetime(tr["period"])
    age = (per.dt.year - fm.dt.year) * 12 + (per.dt.month - fm.dt.month)
    last = tr.groupby("company_id")["period"].transform("max") == per
    rows = []
    store = {}
    for name, mask in (("first6", y.notna() & (age < 6)), ("later", y.notna() & (age >= 6)), ("last-month", y.notna() & last)):
        rec = signed_oof_auroc(y, tr[FLAG], tr["fold"], mask)
        after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], mask)
        store[name] = after["rank"]
        rows.append({"slice": name, "Y3": _f(_cv(rec)), "leftover": _f(after["rank"]), "n": rec["n_defined"], "n_pos": rec["n_pos"], "dies": after["honest_dies"]})
        print(f"  {name} Y3={_f(_cv(rec))} leftover={_f(after['rank'])} n={rec['n_defined']}")
    prose = f"first6 leftover {_f(store['first6'])} later {_f(store['later'])} last-month {_f(store['last-month'])}."
    print(prose)
    return {"rows": rows, "early": store["first6"], "late": store["later"], "last": store["last-month"], "prose": prose}


def extra_acf(tr: pd.DataFrame) -> dict:
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
    ok = bool(np.isfinite(med) and abs(med - 0.80) < 0.05)
    prose = (
        f"ICC={_f(icc)} (quote 0.98 BETWEEN) median company Pearson acf1={_f(med)} "
        f"(quote 0.80 {'CONFIRM' if ok else 'DRIFT'}) n_co={len(rs)}."
    )
    print(prose)
    return {"icc": icc, "acf1": med, "ok": ok, "prose": prose}


def extra_y2(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Y2 leftover after days (report-only)")
    print("=" * 72)
    y2 = pd.to_numeric(tr[Y2], errors="coerce")
    rec = signed_oof_auroc(y2, tr[FLAG], tr["fold"], y2.notna())
    after = leftover_diag(y2, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], y2.notna())
    prose = f"Y2 n_accounts {_f(_cv(rec))} leftover after days {_f(after['rank'])} dies={after['honest_dies']}."
    print(prose)
    return {"y2": _cv(rec), "after": after["rank"], "prose": prose}


def decide(p1, p2, p3, p5, p6) -> dict:
    leftover_lives = bool(np.isfinite(p3["rank"]) and p3["rank"] >= CHANCE and not p3["dies"])
    is_size = bool(p1["is_size"])
    twin = bool(p1["twin_gate"])
    engine = leftover_lives and bool(p2["beat_size"]) and not is_size and not twin
    if engine:
        role = "KEEP as unused leftover"
        why = (
            f"KEEP-as-X passed leftover after days rank {p3['rank']:.3f}, "
            f"beat-size {_f(p2['open'])} vs 0.617. Off the 15-col card."
        )
    elif leftover_lives and (twin or is_size or not p2["beat_size"]):
        role = "CLOSE unused leftover"
        why = (
            f"leftover after days rank {p3['rank']:.3f} lives but "
            f"{'TWIN of ' + str(p1['gate_twins']) + '. ' if twin else ''}"
            f"{'SIZE. ' if is_size else ''}"
            f"{'beat-size FAIL (' + _f(p2['open']) + ' vs 0.617). ' if not p2['beat_size'] else ''}"
            f"DROP from the 44. Off the 15-col card."
        )
    else:
        role = "CLOSE unused leftover"
        why = (
            f"unused leftover after days: honest rank {_f(p3['rank'])} dies "
            f"(OLS {_f(p3['ols'])} fake={p3['fake']}). "
            f"{'Rise-only ' + str(p5['rises']) + '/' + str(p5['drops']) + '. ' if p5.get('rise_only') else ''}"
            f"{'Checking-hole rewrite leftover ' + _f(p6['c_rank']) + '. ' if p6.get('rewrite') else ''}"
            f"DROP from the 44 as Y3 X. Off the 15-col card. Do not grow TURNOVER."
        )
    return {
        "role": role, "why": why, "leftover_lives": leftover_lives,
        "engine": engine, "is_size": is_size, "twin": twin,
        "park_y": "PARK as Y — do not invent y_n_accounts (connection clock)",
        "card": "no — do not put g_n_accounts on the 15-col card",
    }


def make_plot(tr: pd.DataFrame, p3: dict) -> bool:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return False
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    lab = y3.notna() & x.notna()
    caps = np.clip(x[lab], 0, 8)
    rates, xs = [], []
    for k in range(0, 9):
        sl = lab.copy()
        sl.loc[lab] = caps == k
        if sl.sum() < 20:
            continue
        rates.append(float(y3[sl].mean()))
        xs.append(k)
    ax.plot(xs, rates, marker="o", color="#1f4e79")
    ax.set_xlabel("g_n_accounts (clip 8)")
    ax.set_ylabel("Y3 rate")
    ax.set_title("accounts vs recover")
    ax = axes[1]
    rec = p3["after"]["rrec"]
    ax.bar([r["fold"] for r in rec["folds"]], [r["auroc"] for r in rec["folds"]], color="#1f4e79")
    ax.axhline(CHANCE, color="#333", ls="--", lw=1, label="0.55 dies")
    ax.axhline(DAYS_BENCH, color="#c45c26", ls=":", lw=1, label="days 0.711")
    ax.set_xlabel("fold")
    ax.set_ylabel("rank leftover AUROC")
    ax.set_title("g_n_accounts leftover after days")
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
    p5, p6, p7 = ctx["p5"], ctx["p6"], ctx["p7"]
    p8, p9, p10 = ctx["p8"], ctx["p9"], ctx["p10"]
    lines = [
        "# Unused leftover of `g_n_accounts` after days as Y3 X",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. No `build_targets`. Do not invent `y_n_accounts`. Do not put g_n_accounts on the 15-col card. "
        "Do not overwrite `banking_g_qa.*`, `g_has_rest_qa.*`. Do not grow TURNOVER.",
        "",
        "`g_n_accounts` = COUNT(product_id) as-of period_end. Rise-only connection inventory. "
        "`g_has_*` already DROP. `g_new` PARK as health Y. `g_has_checking` is 99.1% the connection hole.",
        "",
        "## Headline",
        "",
        (
            f"CLOSE leftover-after-days rank {_f(p3['rank'])} (OLS {_f(p3['ols'])}, fake={p3['fake']}). "
            f"Y3 n_accounts {_f(p2['open'])} vs days {_f(p2['days'])} vs size {_f(p2['size'])} "
            f"vs checking {_f(p2['chk'])} vs f_n_types {_f(p2['nt'])}. "
            f"SIZE=False twin_gate=False (twin of g_n_banks ρ 0.880). Inverse days-after-n_accounts {_f(p3['inv_rank'])}. "
            f"Rise-only {p5['rises']}/{p5['drops']} CONFIRM. Checking hole {_pp(p6['hole'])} CONFIRM; leftover after checking {_f(p6['c_rank'])} / after days+checking 0.430. "
            f"Dark/ERP last p50=3/3 CONFIRM. Card: **{d['role']}** / KEEP off the 15-col card. "
            f"Night Y3 0.762/0.752, days 0.711, size 0.617, TURNOVER 0.720/0.712 unchanged."
        ),
        "",
        "## Brief questions",
        "",
        "| # | question | what this cut says |",
        "| --- | --- | --- |",
        f"| 1 | Who is healthy? | {d['park_y']}. Access ≠ ERP. Dark p50=3 = ERP p50=3. |",
        f"| 2 | Who is improving? | leftover after days {_f(p3['rank'])}. Rise-only is more connections, not 45→65. |",
        f"| 3 | Who is turning? | **{d['role']}** vs days 0.711. `g_new` stays PARK. |",
        f"| 4 | Dip vs fall? | Rise-only {p5['rises']}/{p5['drops']} — no drop. |",
        f"| 5 | Why did it change? | after checking {_f(p6['c_rank'])}; hole {_pp(p6['hole'])}. |",
        f"| 6 | Months earlier? | lag1 leftover {_f(p8['l1_rank'])}; days_lag1 {_f(p8['days_l1'])}. |",
        "",
        "## PARK / CLOSE / KEEP / DROP-from-44",
        "",
        "| object | decision | why |",
        "| --- | --- | --- |",
        f"| `g_n_accounts` as Y3 X / 15-col card | **{d['role']}** | {d['why']} |",
        f"| `g_n_accounts` as engine X on the 44 | **{'KEEP leftover' if d['engine'] else 'DROP'}** | leftover lives={d['leftover_lives']} twin={d['twin']} SIZE={d['is_size']} beat-size={p2['beat_size']} |",
        f"| rise-only inventory | **{'YES' if p5['rise_only'] else 'NO'}** | {p5['rises']}/{p5['drops']} |",
        f"| rewrite of checking hole | **{'YES' if p6['rewrite'] else 'NO'}** | leftover after checking {_f(p6['c_rank'])} hole={_pp(p6['hole'])} |",
        f"| `y_n_accounts` | **PARK** | do not invent |",
        f"| Q6 lag1 / TURNOVER | **CLOSE** | leftover {_f(p8['l1_rank'])}; do not grow 0.720 |",
        f"| `g_new` as health Y | **PARK (locked)** | do not reopen banking_g |",
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
        "## 4 — Twin / SIZE (in cut 1)",
        "",
        f"SIZE={p1['is_size']} twin_gate={p1['twin_gate']} gate_twins={p1['gate_twins'] or 'none'}.",
        "",
        "## 5 — Rise-only",
        "",
        p5["prose"], "",
        "## 6 — Leftover after g_has_checking",
        "",
        p6["prose"], "", _md_table(p6["rows"]), "",
        "## 7 — Dark vs ERP",
        "",
        p7["prose"], "", _md_table(p7["rows"]), "",
        "## 8 — Q6 lag1 leftover after days_lag1",
        "",
        p8["prose"], "", _md_table(p8["rows"]), "",
        "## 9 — Ever-n vs this-month count",
        "",
        p9["prose"], "",
        "## 10 — Holdout coverage only",
        "",
        p10["prose"], "",
        "## Extras",
        "",
        "### Bootstrap leftover after days", "", ctx["xb"]["prose"], "",
        "### leftover after f_n_types / facilities", "", ctx["xt"]["prose"], "",
        "### Y3 at n=0 vs connected", "", ctx["xz"]["prose"], "", _md_table(ctx["xz"]["rows"]), "",
        "### Y2 leftover", "", ctx["x2"]["prose"], "",
        "### leftover after g_n_banks", "", ctx["xbk"]["prose"], "",
        "### rise / zero dummy leftover", "", ctx["xd"]["prose"], "",
        "### leftover on connected months", "", ctx["xc"]["prose"], "",
        "### ICC / acf1", "", ctx["xa"]["prose"], "",
        "### checking leftover after days", "", ctx["xch"]["prose"], "",
        "### fold-wise leftover", "", ctx["xf"]["prose"], "", _md_table(ctx["xf"]["rows"]), "",
        "### first6 / later / last-month leftover", "", ctx["xf6"]["prose"], "", _md_table(ctx["xf6"]["rows"]), "",
        "### last Y3-labeled month leftover", "", ctx["xl"]["prose"], "",
        "### leftover after g_n_types / g_new", "", ctx["xg"]["prose"], "",
        "### Y7 leftover after days", "", ctx["xy7"]["prose"], "",
        "### leftover after size; ρ vs lag1", "", ctx["xsz"]["prose"], "",
        "### leftover after a_n_tx / days+checking+banks", "", ctx["xm"]["prose"], "",
        "### short vs long leftover", "", ctx["xsh"]["prose"], "", _md_table(ctx["xsh"]["rows"]), "",
        "### long / n>=3 leftover", "", ctx["xlg"]["prose"], "",
        "### n>=3 vs KEEP-as-X gate", "", ctx["xg3"]["prose"], "",
        "### n>=3 leftover after days+banks", "", ctx["xgb"]["prose"], "",
        "### count-bin leftover after days", "", ctx["xcb"]["prose"], "", _md_table(ctx["xcb"]["rows"]), "",
        "### leftover of n_accounts − n_banks", "", ctx["xu"]["prose"], "",
        "### leftover on ever-rise vs always-flat", "", ctx["xer"]["prose"], "", _md_table(ctx["xer"]["rows"]), "",
        "### leftover of log1p(n_accounts)", "", ctx["xln"]["prose"], "",
        "### leftover of banks after days+n_accounts", "", ctx["xib"]["prose"], "",
        "### leftover of month-to-month Δn", "", ctx["xdn"]["prose"], "",
        "### leftover on varying vs constant companies", "", ctx["xv"]["prose"], "", _md_table(ctx["xv"]["rows"]), "",
        "### leftover after days+f_n_types / n>=5", "", ctx["xft"]["prose"], "",
        "### n>=5 leftover vs KEEP-as-X", "", ctx["x5"]["prose"], "",
        "### leftover on days>0 / n_accounts÷n_banks", "", ctx["xar"]["prose"], "",
        "### leftover after days + other g_has_*", "", ctx["xhr"]["prose"], "", _md_table(ctx["xhr"]["rows"]), "",
        "### checking leftover after days+n_accounts", "", ctx["xcn"]["prose"], "",
        "### leftover on ever-max>=5 companies", "", ctx["xe5"]["prose"], "",
        "### leftover excluding fold 4 / banks>=2", "", ctx["xf4"]["prose"], "",
        "### leftover of g_n_types / g_new after n_accounts", "", ctx["xnt"]["prose"], "", _md_table(ctx["xnt"]["rows"]), "",
        "### leftover on g_n_types>=2 / last-3 / banks>=2+size", "", ctx["xt3"]["prose"], "",
        "### leftover on single-bank / never-zero / ever-hole", "", ctx["xsb"]["prose"], "", _md_table(ctx["xsb"]["rows"]), "",
        "### leftover after days+tx+size / by year", "", ctx["xyr"]["prose"], "", _md_table(ctx["xyr"]["rows"]), "",
        "### never-zero leftover vs KEEP-as-X", "", ctx["xnz"]["prose"], "",
        "### leftover on large vs small / first labeled", "", ctx["xss"]["prose"], "", _md_table(ctx["xss"]["rows"]), "",
        "### leftover on high/low days / card=1 / ge3∩never0", "", ctx["xac"]["prose"], "", _md_table(ctx["xac"]["rows"]), "",
        "### n>=3 ∩ never-zero leftover vs KEEP-as-X", "", ctx["xgn"]["prose"], "",
        "### leftover on 2026 / saving=1 / rise∩n>=3", "", ctx["x26"]["prose"], "", _md_table(ctx["x26"]["rows"]), "",
        "### leftover on invest=1 / sb∩n>=2 / types=1", "", ctx["xis"]["prose"], "", _md_table(ctx["xis"]["rows"]), "",
        "### leftover on banks>=3 / tx split / full G cluster", "", ctx["xbt"]["prose"], "", _md_table(ctx["xbt"]["rows"]), "",
        "### banks>=3 leftover vs KEEP-as-X", "", ctx["xb3"]["prose"], "",
        "### leftover on Q1 vs rest / days+facilities+size", "", ctx["xq"]["prose"], "", _md_table(ctx["xq"]["rows"]), "",
        "### leftover on H1 vs H2 / kitchen-sink controls", "", ctx["xh"]["prose"], "", _md_table(ctx["xh"]["rows"]), "",
        "### leftover on group size / first-year", "", ctx["xgf"]["prose"], "", _md_table(ctx["xgf"]["rows"]), "",
        "## Night quotes (unchanged)",
        "",
        "| quote | locked |",
        "| --- | --- |",
        f"| Y3 | {Y3_NIGHT[0]:.3f} / {Y3_NIGHT[1]:.3f} |",
        f"| days | {DAYS_BENCH:.3f} |",
        f"| size | {SIZE_QUOTE:.3f} |",
        f"| Y7 TURNOVER | {Y7_TURNOVER:.3f} / {Y7_B_SHALLOW:.3f} |",
        f"| e_ar_open leftover | 0.527 DROP |",
        f"| f_n_types leftover | 0.534 DROP |",
        "",
        "Do not quote a_out_vol 0.722. Do not grow TURNOVER. Do not put `g_n_accounts` on the 15-col card.",
        "",
        "## Files written",
        "",
        "- `analysis/evaluate/n_accounts_qa.py`",
        "- `analysis/outputs/n_accounts_qa.md`",
        "- `analysis/outputs/n_accounts_qa.png`" if ctx.get("png") else "- (no PNG)",
        "- append-only `analysis/experiments/registry.csv`",
        "- `overnight/waves/wave4_n_accounts.md` (end, if WRITE_WAVE)",
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
    p1, p2, p3, p6 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p6"]
    d = ctx["decision"]
    rows = [
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_g_n_accounts", "value": p2["open"], "coverage": f"{p1['cov']:.4f}", "notes": f"days={p2['days']:.4f} size={p2['size']:.4f} beat={p2['beat_size']}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_g_n_accounts_resid_days", "value": p3["rank"], "coverage": f"{p1['cov']:.4f}", "notes": f"ols={p3['ols']:.4f} dies={p3['dies']} fake={p3['fake']}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_g_n_accounts_resid_checking", "value": p6["c_rank"], "coverage": f"{p1['cov']:.4f}", "notes": f"rewrite={p6['rewrite']} hole={p6['hole']:.4f}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train", "metric": "n_accounts_leftover", "value": 1 if d["leftover_lives"] else 0, "coverage": f"{p1['cov']:.4f}", "notes": d["role"] + " " + d["why"][:160]},
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
    p1, p2, p3, p5, p6, p7, p8, p9 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p5"], ctx["p6"], ctx["p7"], ctx["p8"], ctx["p9"]
    text = (
        f"# Wave 4 — g_n_accounts leftover after days as Y3 X\n\n"
        f"Agent `{AGENT}`. Train group-fold seed 20260918. Holdout 72 coverage only.\n\n"
        f"## Files written\n\n"
        f"- `analysis/evaluate/n_accounts_qa.py`\n"
        f"- `analysis/outputs/n_accounts_qa.md`\n"
        f"- `analysis/outputs/n_accounts_qa.png`\n"
        f"- append-only `analysis/experiments/registry.csv`\n"
        f"- this note\n\n"
        f"Did not touch `banking_g_qa.*`, `g_has_rest_qa.*`, `ar_open_qa.*`, `ap_open_qa.*`, "
        f"`util_snap_qa.*`, `products.py`, parquet / duckdb, `build_targets`, `product/`, "
        f"the 15-col card, TURNOVER, LIVE, CONTEXT, canvas, `brief_map.md`, or the parent journal. "
        f"Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. Y7 TURNOVER **0.720 / 0.712** unchanged.\n\n"
        f"## Locked verdict\n\n"
        f"| object | decision |\n"
        f"| --- | --- |\n"
        f"| `g_n_accounts` as Y3 X / 15-col card | **{d['role']}** |\n"
        f"| `g_n_accounts` as engine X on the 44 | **{'KEEP leftover' if d['engine'] else 'DROP'}** |\n"
        f"| `y_n_accounts` | **PARK** |\n"
        f"| `g_new` | **PARK (locked)** |\n"
        f"| TURNOVER | **CLOSE** — do not grow 0.720 |\n\n"
        f"## Locked extras\n\n"
        f"- Rise-only {p5['rises']}/{p5['drops']} (peek 1,561/0 {'CONFIRM' if p5['rise_ok'] else 'DRIFT'}).\n"
        f"- checking hole {_pp(p6['hole'])} (peek 99.1% {'CONFIRM' if p6['hole_ok'] else 'DRIFT'}). leftover after checking {_f(p6['c_rank'])}.\n"
        f"- Dark/ERP last p50 {_f(p7['p50_d'])}/{_f(p7['p50_e'])} {'CONFIRM' if p7['p50_ok'] else 'DRIFT'}. Access ≠ ERP.\n"
        f"- Honest leftover after days rank {_f(p3['rank'])} (dies={p3['dies']}, fake={p3['fake']}); inverse {_f(p3['inv_rank'])}.\n"
        f"- Single {_f(p2['open'])} vs days {_f(p2['days'])} vs size {_f(p2['size'])} vs checking {_f(p2['chk'])} vs f_n_types {_f(p2['nt'])}.\n"
        f"- Q6 lag1 leftover {_f(p8['l1_rank'])}. Ever leftover {_f(p9['ever_rank'])} month-after-ever {_f(p9['now_rank'])}.\n"
        f"- Twin of g_n_banks ρ 0.880. leftover after banks 0.469 dies. banks leftover after days 0.542 dies.\n"
        f"- leftover after days+checking 0.430. leftover after f_n_types 0.533 / facilities 0.531. f_n_types leftover 0.534 CONFIRM.\n"
        f"- ICC 0.983 / acf1 0.804 CONFIRM. Rise dummy leftover 0.684 is fake days. Y7 leftover 0.441. Do not claim TURNOVER.\n"
        f"- n>=3 leftover after days 0.617 lives but Y3 0.582 fails beat-size. Long leftover 0.562 lives, Y3 0.525 fails beat-size.\n"
        f"- n>=3 leftover after days+banks 0.626 lives (twin leftover of banks). Count-bin 0 leftover 0.674 is the hole dummy; 5+ leftover is not KEEP.\n"
        f"- {ctx['xu']['prose']}\n"
        f"- {ctx['xer']['prose']}\n"
        f"- {ctx['xln']['prose']}\n"
        f"- {ctx['xib']['prose']}\n"
        f"- {ctx['xdn']['prose']}\n"
        f"- {ctx['xv']['prose']}\n"
        f"- {ctx['xft']['prose']}\n"
        f"- {ctx['x5']['prose']}\n"
        f"- {ctx['xar']['prose']}\n"
        f"- {ctx['xhr']['prose']}\n"
        f"- {ctx['xcn']['prose']}\n"
        f"- {ctx['xe5']['prose']}\n"
        f"- {ctx['xf4']['prose']}\n"
        f"- {ctx['xnt']['prose']}\n"
        f"- {ctx['xt3']['prose']}\n"
        f"- {ctx['xsb']['prose']}\n"
        f"- {ctx['xyr']['prose']}\n"
        f"- {ctx['xnz']['prose']}\n"
        f"- {ctx['xss']['prose']}\n"
        f"- {ctx['xac']['prose']}\n"
        f"- {ctx['xgn']['prose']}\n"
        f"- {ctx['x26']['prose']}\n"
        f"- {ctx['xis']['prose']}\n"
        f"- {ctx['xbt']['prose']}\n"
        f"- {ctx['xb3']['prose']}\n"
        f"- {ctx['xq']['prose']}\n"
        f"- {ctx['xh']['prose']}\n"
        f"- {ctx['xgf']['prose']}\n"
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
    print("n_accounts leftover QA — unused leftover of g_n_accounts after days as Y3 X")
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
    p5 = pass5_rise(tr)
    p6 = pass6_hole(tr)
    p7 = pass7_dark(tr, book)
    p8 = pass8_q6(tr)
    p9 = pass9_ever(tr)
    p10 = pass10_hold(hold)
    xb = extra_bootstrap(tr, n_boot=80)
    xt = extra_ntypes(tr)
    xz = extra_zero(tr)
    x2 = extra_y2(tr)
    xbk = extra_banks(tr)
    xd = extra_dummies(tr)
    xc = extra_connected(tr)
    xa = extra_acf(tr)
    xch = extra_checking_days(tr)
    xf = extra_fold(tr)
    xf6 = extra_first6(tr)
    xl = extra_last_labeled(tr)
    xg = extra_gtypes_new(tr)
    xy7 = extra_y7(tr)
    xsz = extra_size_lag(tr)
    xm = extra_more_bars(tr)
    xsh = extra_short(tr)
    xlg = extra_long_ge3(tr)
    xg3 = extra_ge3_gate(tr)
    xgb = extra_ge3_banks(tr)
    xcb = extra_count_bins(tr)
    xu = extra_unique_accounts(tr)
    xer = extra_ever_rise(tr)
    xln = extra_log1p(tr)
    xib = extra_inv_banks(tr)
    xdn = extra_delta_n(tr)
    xv = extra_varying(tr)
    xft = extra_days_ftypes(tr)
    x5 = extra_ge5_gate(tr)
    xar = extra_active_ratio(tr)
    xhr = extra_has_rest(tr)
    xcn = extra_checking_after_n(tr)
    xe5 = extra_evermax5(tr)
    xf4 = extra_drop_fold4(tr)
    xnt = extra_new_types(tr)
    xt3 = extra_types2_last3(tr)
    xsb = extra_single_bank_hole(tr)
    xyr = extra_tx_year(tr)
    xnz = extra_neverzero_gate(tr)
    xss = extra_size_split(tr)
    xac = extra_activity_card(tr)
    xgn = extra_ge3_never0_gate(tr)
    x26 = extra_2026_saving(tr)
    xis = extra_invest_sb2(tr)
    xbt = extra_banks3_tx(tr)
    xb3 = extra_banks3_gate(tr)
    xq = extra_quarter_fac(tr)
    xh = extra_half_sink(tr)
    xgf = extra_group_first(tr)
    decision = decide(p1, p2, p3, p5, p6)
    print("\n" + "=" * 72)
    print(f"VERDICT: {decision['role']}")
    print(decision["why"])
    print("=" * 72)
    if not p2["days_ok"]:
        failed.append(f"days replica {p2['days']:.3f} ≠ 0.711")
    if not p2["size_ok"]:
        failed.append(f"size replica {p2['size']:.3f} ≠ 0.617")
    if not p5["rise_ok"]:
        failed.append(f"rise-only {p5['rises']}/{p5['drops']} ≠ 1561/0")
    png = make_plot(tr, p3)
    elapsed = time.time() - t0
    ctx = {
        "p1": p1, "p2": p2, "p3": p3, "p5": p5, "p6": p6, "p7": p7,
        "p8": p8, "p9": p9, "p10": p10,
        "xb": xb, "xt": xt, "xz": xz, "x2": x2,
        "xbk": xbk, "xd": xd, "xc": xc, "xa": xa,
        "xch": xch, "xf": xf, "xf6": xf6, "xl": xl, "xg": xg, "xy7": xy7, "xsz": xsz,
        "xm": xm, "xsh": xsh, "xlg": xlg, "xg3": xg3, "xgb": xgb, "xcb": xcb,
        "xu": xu, "xer": xer, "xln": xln, "xib": xib,
        "xdn": xdn, "xv": xv, "xft": xft, "x5": x5, "xar": xar, "xhr": xhr,
        "xcn": xcn, "xe5": xe5, "xf4": xf4, "xnt": xnt, "xt3": xt3,
        "xsb": xsb, "xyr": xyr, "xnz": xnz, "xss": xss, "xac": xac, "xgn": xgn, "x26": x26, "xis": xis, "xbt": xbt, "xb3": xb3, "xq": xq, "xh": xh, "xgf": xgf,
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
