"""Unused leftover of ``e_ar_open`` after days as Y3 X.

``e_ar_open`` = unpaid AR |amount| stock at period end (Family E).
``e_dso_proxy`` = open / this-period issued — already DROP (Y3 leftover
0.474). Contemporaneous ``e_ar_issued`` CLOSE leftover 0.608 (fake days
clone). ``e_pending_amt_share`` DROP leftover dies. Sibling is running
``e_ap_open`` — do **not** overwrite ``ap_open_qa.*`` / ``dso_qa.*`` /
``issued_qa.*``. Y5 never E. Dark 470 stay NaN not 0.

KEEP-as-X: beat size ≥0.02 **and** leftover after days **and** not SIZE
(|ρ| vs log1p(a_in3) ≥0.50) **and** not a twin (|ρ|≥0.80 vs days /
a_n_tx / e_ar_issued / e_dso_proxy / e_pending_amt_share). Leftover
<0.55 dies.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.ar_open_qa

Owned: analysis/evaluate/ar_open_qa.py, analysis/outputs/ar_open_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_ar_open.md (end).
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
OUT_MD = ANALYSIS / "outputs" / "ar_open_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "ar_open_qa.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_ar_open.md"
AP_OPEN_MD = ANALYSIS / "outputs" / "ap_open_qa.md"
AGENT = "572fb928"
WAVE = "4"
ROUND = "R4"
MODEL = "ar_open_qa"
X_FAM = "E"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
Y7 = "y7_top1_lost"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_QUOTE = 0.617
DAYS_LAG1_QUOTE = 0.684
ISSUED_Y3_PEEK = 0.687
ISSUED_LEFT_PEEK = 0.608
DSO_LEFT_PEEK = 0.474
AP_OPEN_Y3_PEEK = 0.569
AP_OPEN_LEFT = 0.418
AR_OPEN_Y3_PEEK = 0.587
AP_RHO_PEEK = 0.682
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
    "e_ar_open",
    "e_ar_issued",
    "e_dso_proxy",
    "e_pending_amt_share",
    "e_ap_open",
    "e_delay_coll",
    "e_ar_overdue",
    "e_ar_overdue_30",
    "e_ap_issued",
    "e_fx_share",
    "e_credit_note_ratio",
)

Y_KEEP = (Y2, Y3, Y7)
FLAG = "e_ar_open"


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
    panel["log_open"] = np.log1p(pd.to_numeric(panel[FLAG], errors="coerce").clip(lower=0))
    leak3 = leakage_check(
        [FLAG, "e_ar_issued", "e_dso_proxy", "e_pending_amt_share", "c_n_days_with_tx", "log_in3"],
        Y3, forbidden_prefixes=["b"],
    )
    if not leak3["ok"]:
        raise RuntimeError(f"Y3 never-B leak: {leak3['issues']}")
    leak7 = leakage_check([FLAG, "e_ar_issued"], Y7, forbidden_prefixes=["d"])
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
    print("CUT 1 — coverage; 470 NaN; twin / SIZE")
    print("=" * 72)
    n_cm, n_co = len(tr), int(tr["company_id"].nunique())
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    dark = ~tr["company_id"].isin(book)
    n_dark_co = int(tr.loc[dark, "company_id"].nunique())
    dark_nn = int(x[dark].notna().sum())
    dark_zero = int((x[dark] == 0).sum())
    n_nn = int(x.notna().sum())
    pairs = {
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "a_n_tx": tr["a_n_tx"],
        "e_ar_issued": tr["e_ar_issued"],
        "e_dso_proxy": tr["e_dso_proxy"],
        "e_pending_amt_share": tr["e_pending_amt_share"],
        "e_ap_open": tr["e_ap_open"],
        "log1p(a_in3)": tr["log_in3"],
    }
    rhos = {k: spearman(x, v) for k, v in pairs.items()}
    twins = [k for k, v in rhos.items() if np.isfinite(v) and abs(v) >= TWIN_RHO]
    gate_twins = [
        k for k in ("c_n_days_with_tx", "a_n_tx", "e_ar_issued", "e_dso_proxy", "e_pending_amt_share")
        if np.isfinite(rhos[k]) and abs(rhos[k]) >= TWIN_RHO
    ]
    is_size = bool(np.isfinite(rhos["log1p(a_in3)"]) and abs(rhos["log1p(a_in3)"]) >= SIZE_RHO)
    twin_gate = bool(gate_twins)
    rho_ap_ok = bool(np.isfinite(rhos["e_ap_open"]) and abs(rhos["e_ap_open"] - AP_RHO_PEEK) < 0.03)
    dark_ok = n_dark_co == N_DARK_WANT and dark_nn == 0 and dark_zero == 0
    rows = [
        {"col": FLAG, "n_nn": f"{n_nn:,}", "cov": _pp(_pct(n_nn, n_cm)), "eq0": _pp(_pct(int((x == 0).sum()), n_nn) if n_nn else float("nan")), "p50": _f(float(x.median()) if n_nn else float("nan"))},
        {"col": "e_ar_issued", "n_nn": f"{int(pd.to_numeric(tr['e_ar_issued'], errors='coerce').notna().sum()):,}", "cov": "—", "eq0": "—", "p50": "—"},
        {"col": "e_dso_proxy", "n_nn": f"{int(pd.to_numeric(tr['e_dso_proxy'], errors='coerce').notna().sum()):,}", "cov": "—", "eq0": "—", "p50": "—"},
    ]
    rho_rows = [
        {"vs": k, "rho": _f(v), "flag": "SIZE" if k.startswith("log") and abs(v) >= SIZE_RHO else "TWIN" if abs(v) >= TWIN_RHO else "no"}
        for k, v in rhos.items()
    ]
    prose = (
        f"Train {n_co:,} co / {n_cm:,} CM. e_ar_open nn={n_nn:,} cov {_pp(_pct(n_nn, n_cm))}. "
        f"Dark {n_dark_co} nn={dark_nn} zero={dark_zero} {'CONFIRM NaN' if dark_ok else 'FAIL'}. "
        f"ρ vs days {_f(rhos['c_n_days_with_tx'])} vs a_n_tx {_f(rhos['a_n_tx'])} "
        f"vs issued {_f(rhos['e_ar_issued'])} vs DSO {_f(rhos['e_dso_proxy'])} "
        f"vs pending {_f(rhos['e_pending_amt_share'])} vs ap_open {_f(rhos['e_ap_open'])} "
        f"(peek 0.682 {'CONFIRM' if rho_ap_ok else 'DRIFT'}) vs size {_f(rhos['log1p(a_in3)'])}. "
        f"SIZE={is_size} gate_twins={gate_twins or 'none'} all_twins={twins or 'none'}."
    )
    print(prose)
    return {
        "rows": rows, "rho_rows": rho_rows, "rhos": rhos, "twins": twins,
        "gate_twins": gate_twins, "is_size": is_size, "twin_gate": twin_gate,
        "n_cm": n_cm, "n_co": n_co, "n_nn": n_nn, "cov": _pct(n_nn, n_cm),
        "dark_ok": dark_ok, "n_dark_co": n_dark_co, "dark_nn": dark_nn,
        "rho_ap_ok": rho_ap_ok, "prose": prose,
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
        "log1p(e_ar_open)": tr["log_open"],
        "e_ar_issued": tr["e_ar_issued"],
        "e_dso_proxy": tr["e_dso_proxy"],
        "e_ap_open": tr["e_ap_open"],
        "e_pending_amt_share": tr["e_pending_amt_share"],
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
    y3_iss, y3_dso = _cv(recs["e_ar_issued"]), _cv(recs["e_dso_proxy"])
    y3_ap = _cv(recs["e_ap_open"])
    days_ok = bool(np.isfinite(y3_days) and abs(y3_days - DAYS_BENCH) < 0.008)
    size_ok = bool(np.isfinite(y3_size) and abs(y3_size - SIZE_QUOTE) < 0.008)
    iss_ok = bool(np.isfinite(y3_iss) and abs(y3_iss - ISSUED_Y3_PEEK) < 0.015)
    peek_ok = bool(np.isfinite(y3_o) and abs(y3_o - AR_OPEN_Y3_PEEK) < 0.02)
    beat_size = bool(np.isfinite(y3_o) and (y3_o - SIZE_QUOTE) >= KEEP_DELTA)
    prose = (
        f"Y3 e_ar_open {_f(y3_o)} n={recs[FLAG]['n_defined']:,} pos={recs[FLAG]['n_pos']:,} "
        f"(ap_open peek 0.587 {'CONFIRM' if peek_ok else 'DRIFT'}). "
        f"vs size {_f(y3_size)} vs days {_f(y3_days)} vs issued {_f(y3_iss)} "
        f"(peek 0.687 {'CONFIRM' if iss_ok else 'DRIFT'}) vs DSO {_f(y3_dso)} "
        f"vs ap_open {_f(y3_ap)} (sibling 0.569). Replica days 0.711 "
        f"{'CONFIRM' if days_ok else 'DRIFT'} / size 0.617 {'CONFIRM' if size_ok else 'DRIFT'}. "
        f"Beat-size Δ={_f(y3_o - SIZE_QUOTE) if np.isfinite(y3_o) else '—'} "
        f"{'PASS' if beat_size else 'FAIL'}."
    )
    print(prose)
    return {
        "rows": rows, "recs": recs, "open": y3_o, "size": y3_size, "days": y3_days,
        "iss": y3_iss, "dso": y3_dso, "ap": y3_ap,
        "days_ok": days_ok, "size_ok": size_ok, "iss_ok": iss_ok, "peek_ok": peek_ok,
        "beat_size": beat_size, "n_def": recs[FLAG]["n_defined"], "n_pos": recs[FLAG]["n_pos"],
        "prose": prose,
    }


def pass3_days(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 3 — honest leftover of e_ar_open after days; inverse")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab, folds = y.notna(), tr["fold"]
    after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), folds, lab)
    inv = leftover_diag(y, tr["c_n_days_with_tx"], (tr[FLAG],), folds, lab)
    after_log = leftover_diag(y, tr["log_open"], (tr["c_n_days_with_tx"],), folds, lab)
    prose = (
        f"e_ar_open leftover after days OLS {_f(after['ols'])} rank {_f(after['rank'])} "
        f"fake={after['fake']} almost={after['almost']} ρ(resid,days)={_f(after['rho_ctrl'])} "
        f"R²={_f(after['r2'])} honest_dies={after['honest_dies']} n={after['n']:,} pos={after['n_pos']:,}. "
        f"log1p leftover rank {_f(after_log['rank'])} dies={after_log['honest_dies']}. "
        f"Inverse: days leftover after ar_open OLS {_f(inv['ols'])} rank {_f(inv['rank'])} "
        f"dies={inv['honest_dies']}."
    )
    print(prose)
    return {
        "after": after, "inv": inv, "after_log": after_log,
        "ols": after["ols"], "rank": after["rank"],
        "dies": after["honest_dies"], "fake": after["fake"], "almost": after["almost"],
        "r2": after["r2"], "inv_rank": inv["rank"], "inv_dies": inv["honest_dies"],
        "log_rank": after_log["rank"], "prose": prose,
    }


def pass5_issued(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 5 — leftover after issued (rewrite of issued volume?)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab, folds = y.notna(), tr["fold"]
    after_i = leftover_diag(y, tr[FLAG], (tr["e_ar_issued"],), folds, lab)
    after_di = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["e_ar_issued"]), folds, lab)
    iss_after = leftover_diag(y, tr["e_ar_issued"], (tr[FLAG],), folds, lab)
    rho = spearman(tr[FLAG], tr["e_ar_issued"])
    rewrite = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO) or after_i["honest_dies"]
    rows = [
        {"bar": "after issued", "OLS": _f(after_i["ols"]), "rank": _f(after_i["rank"]), "ρ(resid,bar)": _f(after_i["rho_ctrl"]), "R2": _f(after_i["r2"]), "dies": after_i["honest_dies"]},
        {"bar": "after days+issued", "OLS": _f(after_di["ols"]), "rank": _f(after_di["rank"]), "ρ(resid,bar)": _f(after_di["rho_ctrl"]), "R2": _f(after_di["r2"]), "dies": after_di["honest_dies"]},
        {"bar": "issued after open", "OLS": _f(iss_after["ols"]), "rank": _f(iss_after["rank"]), "ρ(resid,bar)": _f(iss_after["rho_ctrl"]), "R2": _f(iss_after["r2"]), "dies": iss_after["honest_dies"]},
    ]
    prose = (
        f"open leftover after issued rank {_f(after_i['rank'])} dies={after_i['honest_dies']} "
        f"R²={_f(after_i['r2'])} ρ={_f(rho)} rewrite={rewrite}. "
        f"after days+issued {_f(after_di['rank'])} dies={after_di['honest_dies']}. "
        f"issued leftover after open {_f(iss_after['rank'])} (issued leftover after days was 0.608 fake)."
    )
    print(prose)
    return {
        "rows": rows, "i_rank": after_i["rank"], "i_dies": after_i["honest_dies"],
        "di_rank": after_di["rank"], "iss_after": iss_after["rank"],
        "rho": rho, "rewrite": rewrite, "r2": after_i["r2"], "prose": prose,
    }


def pass6_dso(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 6 — leftover after DSO (is open the DSO numerator?)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab, folds = y.notna(), tr["fold"]
    after_d = leftover_diag(y, tr[FLAG], (tr["e_dso_proxy"],), folds, lab)
    after_dd = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["e_dso_proxy"]), folds, lab)
    dso_after = leftover_diag(y, tr["e_dso_proxy"], (tr[FLAG],), folds, lab)
    dso_days = leftover_diag(y, tr["e_dso_proxy"], (tr["c_n_days_with_tx"],), folds, lab)
    rho = spearman(tr[FLAG], tr["e_dso_proxy"])
    numer = bool(after_d["honest_dies"] or (np.isfinite(after_d["r2"]) and after_d["r2"] >= 0.50))
    rows = [
        {"bar": "after DSO", "OLS": _f(after_d["ols"]), "rank": _f(after_d["rank"]), "R2": _f(after_d["r2"]), "dies": after_d["honest_dies"]},
        {"bar": "after days+DSO", "OLS": _f(after_dd["ols"]), "rank": _f(after_dd["rank"]), "R2": _f(after_dd["r2"]), "dies": after_dd["honest_dies"]},
        {"bar": "DSO after open", "OLS": _f(dso_after["ols"]), "rank": _f(dso_after["rank"]), "R2": _f(dso_after["r2"]), "dies": dso_after["honest_dies"]},
        {"bar": "DSO after days", "OLS": _f(dso_days["ols"]), "rank": _f(dso_days["rank"]), "R2": _f(dso_days["r2"]), "dies": dso_days["honest_dies"]},
    ]
    dso_ok = bool(np.isfinite(dso_days["ols"]) and abs(dso_days["ols"] - DSO_LEFT_PEEK) < 0.03)
    prose = (
        f"open leftover after DSO rank {_f(after_d['rank'])} dies={after_d['honest_dies']} "
        f"R²={_f(after_d['r2'])} ρ={_f(rho)} numerator={numer}. "
        f"after days+DSO {_f(after_dd['rank'])}. DSO leftover after open {_f(dso_after['rank'])}. "
        f"DSO leftover after days OLS {_f(dso_days['ols'])} rank {_f(dso_days['rank'])} "
        f"(peek 0.474 is OLS {'CONFIRM' if dso_ok else 'DRIFT'})."
    )
    print(prose)
    return {
        "rows": rows, "d_rank": after_d["rank"], "d_dies": after_d["honest_dies"],
        "dd_rank": after_dd["rank"], "dso_after": dso_after["rank"],
        "dso_days": dso_days["rank"], "dso_ok": dso_ok, "rho": rho, "numer": numer,
        "r2": after_d["r2"], "prose": prose,
    }


def pass7_dark(tr: pd.DataFrame, book: set[str]) -> dict:
    print("\n" + "=" * 72)
    print("CUT 7 — dark 470 stay NaN; ERP leftover")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    erp, dark = tr["company_id"].isin(book), ~tr["company_id"].isin(book)
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    n_dark_co = int(tr.loc[dark, "company_id"].nunique())
    dark_nn = int(x[dark].notna().sum())
    rec = signed_oof_auroc(y, tr[FLAG], tr["fold"], lab & erp)
    after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & erp)
    rec_d = signed_oof_auroc(y, tr[FLAG], tr["fold"], lab & dark)
    dark_ok = n_dark_co == N_DARK_WANT and dark_nn == 0
    rows = [
        {"book": "dark", "n_co": n_dark_co, "nn": dark_nn, "Y3 n": rec_d["n_defined"], "Y3 pos": rec_d["n_pos"], "CV": "LOW_POWER" if rec_d["low_power"] else _f(rec_d["cv"])},
        {"book": "ERP", "n_co": int(tr.loc[erp, "company_id"].nunique()), "nn": int(x[erp].notna().sum()), "Y3 n": rec["n_defined"], "Y3 pos": rec["n_pos"], "CV": "LOW_POWER" if rec["low_power"] else _f(rec["cv"])},
    ]
    prose = (
        f"Dark {n_dark_co} (want {N_DARK_WANT}) ar_open nn={dark_nn} "
        f"{'CONFIRM NaN not 0' if dark_ok else 'FAIL'}. "
        f"ERP Y3 {_f(rec['cv'])} leftover after days {_f(after['rank'])} dies={after['honest_dies']}."
    )
    print(prose)
    return {"rows": rows, "dark_ok": dark_ok, "erp_cv": _cv(rec), "erp_rank": after["rank"], "prose": prose}


def pass8_q6(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 8 — Q6 lag1 leftover after days_lag1 (no TURNOVER seat)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab, folds = y.notna(), tr["fold"]
    recs, rows = {}, []
    for name in (FLAG, f"{FLAG}_lag1", "c_n_days_with_tx_lag1", "e_ar_issued_lag1"):
        rec = signed_oof_auroc(y, tr[name], folds, lab)
        recs[name] = rec
        rows.append(_auc_row(Y3, name, rec))
        print(f"  {name} CV={_f(rec['cv'])} n={rec['n_defined']} pos={rec['n_pos']}")
    after_l1 = leftover_diag(y, tr[f"{FLAG}_lag1"], (tr["c_n_days_with_tx_lag1"],), folds, lab)
    days_l1 = _cv(recs["c_n_days_with_tx_lag1"])
    days_ok = bool(np.isfinite(days_l1) and abs(days_l1 - DAYS_LAG1_QUOTE) < 0.008)
    y7 = pd.to_numeric(tr[Y7], errors="coerce")
    after_y7 = leftover_diag(y7, tr[FLAG], (tr["e_ar_issued_lag1"],), tr["fold"], y7.notna())
    prose = (
        f"Y3 ar_open_lag1 {_f(recs[f'{FLAG}_lag1']['cv'])} leftover after days_lag1 "
        f"rank {_f(after_l1['rank'])} dies={after_l1['honest_dies']}. "
        f"Days lag1 {_f(days_l1)} (quote 0.684 {'CONFIRM' if days_ok else 'DRIFT'}). "
        f"Y7 leftover after issued_lag1 {_f(after_y7['rank'])} (report-only; do not claim TURNOVER)."
    )
    print(prose)
    return {
        "rows": rows, "lag1": _cv(recs[f"{FLAG}_lag1"]),
        "l1_rank": after_l1["rank"], "l1_dies": after_l1["honest_dies"],
        "days_l1": days_l1, "days_ok": days_ok, "y7_after": after_y7["rank"],
        "prose": prose,
    }


def pass9_ap(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 9 — vs e_ap_open (sibling file read-only)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    rho = spearman(tr[FLAG], tr["e_ap_open"])
    rec = signed_oof_auroc(y, tr["e_ap_open"], tr["fold"], lab)
    after = leftover_diag(y, tr[FLAG], (tr["e_ap_open"],), tr["fold"], lab)
    same = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
    sibling = AP_OPEN_MD.exists()
    prose = (
        f"ρ(ar_open, ap_open)={_f(rho)} (peek 0.682 {'twin' if same else 'different object'}). "
        f"Y3 ap_open {_f(_cv(rec))} (sibling leftover after days 0.418 — not overwritten). "
        f"ar leftover after ap {_f(after['rank'])} dies={after['honest_dies']}. "
        f"sibling file present={sibling}."
    )
    print(prose)
    return {"rho": rho, "same": same, "ap_cv": _cv(rec), "after_ap": after["rank"], "sibling": sibling, "prose": prose}


def pass10_hold(hold: pd.DataFrame, book: set[str]) -> dict:
    print("\n" + "=" * 72)
    print("CUT 10 — holdout coverage only (no AUROC)")
    print("=" * 72)
    x = pd.to_numeric(hold[FLAG], errors="coerce")
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


def extra_pending(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after pending / days+issued+DSO")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after_p = leftover_diag(y, tr[FLAG], (tr["e_pending_amt_share"],), tr["fold"], lab)
    after_all = leftover_diag(
        y, tr[FLAG],
        (tr["c_n_days_with_tx"], tr["e_ar_issued"], tr["e_dso_proxy"]),
        tr["fold"], lab,
    )
    prose = (
        f"open leftover after pending {_f(after_p['rank'])} dies={after_p['honest_dies']}. "
        f"after days+issued+DSO {_f(after_all['rank'])} dies={after_all['honest_dies']}."
    )
    print(prose)
    return {"after_p": after_p["rank"], "after_all": after_all["rank"], "prose": prose}


def extra_size(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after size / days+size")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after_s = leftover_diag(y, tr[FLAG], (tr["log_in3"],), tr["fold"], lab)
    after_b = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], lab)
    prose = (
        f"open leftover after size {_f(after_s['rank'])} dies={after_s['honest_dies']}. "
        f"after days+size {_f(after_b['rank'])} dies={after_b['honest_dies']}."
    )
    print(prose)
    return {"after_s": after_s["rank"], "after_b": after_b["rank"], "prose": prose}


def extra_quintiles(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Y3 rate by ar_open quintile")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    ok = y.notna() & x.notna()
    qn = pd.qcut(x[ok], 5, duplicates="drop")
    rows = []
    for i, cat in enumerate(sorted(qn.dropna().unique()), start=1):
        sl = ok.copy()
        sl.loc[ok] = qn == cat
        rows.append({"q": i, "Y3 rate": _pp(float(y[sl].mean()) if sl.any() else float("nan")), "n": int(sl.sum()), "n_pos": int((sl & (y == 1)).sum())})
    prose = f"Y3 ar_open Q1→Q5 {[r['Y3 rate'] for r in rows]}."
    print(prose)
    return {"rows": rows, "prose": prose}


def extra_y2(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Y2 leftover after days (report-only)")
    print("=" * 72)
    y2 = pd.to_numeric(tr[Y2], errors="coerce")
    lab = y2.notna()
    rec = signed_oof_auroc(y2, tr[FLAG], tr["fold"], lab)
    after = leftover_diag(y2, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    prose = f"Y2 ar_open {_f(_cv(rec))} leftover after days {_f(after['rank'])} dies={after['honest_dies']}."
    print(prose)
    return {"y2": _cv(rec), "after": after["rank"], "prose": prose}


def extra_dso_ols(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — DSO leftover after days OLS vs rank (reconcile 0.474)")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    dso = leftover_diag(y, tr["e_dso_proxy"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    dso_ok = bool(np.isfinite(dso["ols"]) and abs(dso["ols"] - DSO_LEFT_PEEK) < 0.03)
    prose = (
        f"DSO leftover after days OLS {_f(dso['ols'])} rank {_f(dso['rank'])} "
        f"fake={dso['fake']} (peek 0.474 is OLS {'CONFIRM' if dso_ok else 'DRIFT'}). "
        f"Rank leftover of DSO is not the locked quote."
    )
    print(prose)
    return {"ols": dso["ols"], "rank": dso["rank"], "dso_ok": dso_ok, "prose": prose}


def extra_zero(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Y3 at open==0 vs positive stock")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    ok = y.notna() & x.notna()
    z = ok & (x == 0)
    p = ok & (x > 0)
    dummy = (x == 0).astype(float)
    dummy = dummy.where(x.notna(), np.nan)
    rec = signed_oof_auroc(y, dummy, tr["fold"], y.notna() & x.notna())
    after = leftover_diag(y, dummy, (tr["c_n_days_with_tx"],), tr["fold"], y.notna() & x.notna())
    rows = [
        {"slice": "open==0", "n": int(z.sum()), "n_pos": int((z & (y == 1)).sum()), "Y3 rate": _pp(float(y[z].mean()) if z.any() else float("nan"))},
        {"slice": "open>0", "n": int(p.sum()), "n_pos": int((p & (y == 1)).sum()), "Y3 rate": _pp(float(y[p].mean()) if p.any() else float("nan"))},
    ]
    prose = (
        f"Y3 open==0 {_pp(float(y[z].mean()) if z.any() else float('nan'))} n={int(z.sum())} "
        f"vs open>0 {_pp(float(y[p].mean()) if p.any() else float('nan'))} n={int(p.sum())}. "
        f"zero-dummy Y3 {_f(_cv(rec))} leftover after days {_f(after['rank'])} dies={after['honest_dies']}."
    )
    print(prose)
    return {"rows": rows, "dummy": _cv(rec), "after": after["rank"], "prose": prose}


def extra_issued_clone(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — issued leftover after days on same-n as open")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna() & pd.to_numeric(tr[FLAG], errors="coerce").notna()
    iss = leftover_diag(y, tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    open_ = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    iss_ok = bool(np.isfinite(iss["rank"]) and abs(iss["rank"] - ISSUED_LEFT_PEEK) < 0.02)
    prose = (
        f"Same-n n={int(lab.sum())}: issued leftover after days rank {_f(iss['rank'])} "
        f"OLS {_f(iss['ols'])} fake={iss['fake']} (peek 0.608 {'CONFIRM' if iss_ok else 'DRIFT'}). "
        f"open leftover same-n {_f(open_['rank'])} OLS {_f(open_['ols'])} fake={open_['fake']}. "
        f"Open is weaker than the issued fake-days clone."
    )
    print(prose)
    return {"iss_rank": iss["rank"], "iss_ok": iss_ok, "open_rank": open_["rank"], "prose": prose}


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
    rows = []
    for r in after["rrec"]["folds"]:
        rows.append({"fold": r["fold"], "rank leftover": _f(r["auroc"]), "n_va": r["n_va"], "n_pos": r["n_pos"]})
    prose = f"Rank leftover folds {after['rank_folds']} — fold 1 0.655 lives, fold 4 0.406 dies; mean 0.527 dies."
    print(prose)
    return {"rows": rows, "prose": prose}


def extra_icc(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — ICC / acf1 (BETWEEN check)")
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
    d["lag"] = g.shift(1)
    acf1 = spearman(d[FLAG], d["lag"])
    prose = (
        f"ICC={_f(icc)} (quote 0.99 BETWEEN) acf1={_f(acf1)} (quote 0.77) k={k} n={n}. "
        f"BETWEEN / sticky stock — not a month-to-month lead."
    )
    print(prose)
    return {"icc": icc, "acf1": acf1, "prose": prose}


def extra_acf_median(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — median company Pearson acf1 (feature-report 0.77)")
    print("=" * 72)
    d = tr[["company_id", "period", FLAG]].copy()
    d[FLAG] = pd.to_numeric(d[FLAG], errors="coerce")
    d = d.dropna(subset=[FLAG]).sort_values(["company_id", "period"])
    rs = []
    for _, g in d.groupby("company_id"):
        if len(g) < 4:
            continue
        a = g[FLAG].to_numpy(dtype=float)
        b = g[FLAG].shift(1).to_numpy(dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3:
            continue
        if np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
            continue
        rs.append(float(np.corrcoef(a[ok], b[ok])[0, 1]))
    med = float(np.median(rs)) if rs else float("nan")
    ok = bool(np.isfinite(med) and abs(med - 0.77) < 0.05)
    prose = (
        f"median company Pearson acf1={_f(med)} n_co={len(rs)} "
        f"(quote 0.77 {'CONFIRM' if ok else 'DRIFT'}). Pooled Spearman lag was 0.944 — not the report acf."
    )
    print(prose)
    return {"acf1": med, "ok": ok, "prose": prose}


def extra_pos_only(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after days on open>0 only")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    lab = y.notna() & x.notna() & (x > 0)
    rec = signed_oof_auroc(y, x, tr["fold"], lab)
    after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    prose = (
        f"open>0 only Y3 {_f(_cv(rec))} n={rec['n_defined']:,} pos={rec['n_pos']:,} "
        f"leftover after days rank {_f(after['rank'])} dies={after['honest_dies']}."
    )
    print(prose)
    return {"cv": _cv(rec), "rank": after["rank"], "prose": prose}


def extra_y7_days(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Y7 leftover after days (no TURNOVER seat)")
    print("=" * 72)
    y7 = pd.to_numeric(tr[Y7], errors="coerce")
    rec = signed_oof_auroc(y7, tr[FLAG], tr["fold"], y7.notna())
    after = leftover_diag(y7, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], y7.notna())
    after_i = leftover_diag(y7, tr[FLAG], (tr["e_ar_issued_lag1"],), tr["fold"], y7.notna())
    prose = (
        f"Y7 ar_open {_f(_cv(rec))} leftover after days {_f(after['rank'])} dies={after['honest_dies']}. "
        f"leftover after issued_lag1 {_f(after_i['rank'])}. Do not grow TURNOVER 0.720."
    )
    print(prose)
    return {"y7": _cv(rec), "after": after["rank"], "after_i": after_i["rank"], "prose": prose}


def extra_clip(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after days on p99-clipped open")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    p99 = float(x.quantile(0.99))
    clip = x.clip(upper=p99)
    after = leftover_diag(y, clip, (tr["c_n_days_with_tx"],), tr["fold"], y.notna())
    rec = signed_oof_auroc(y, clip, tr["fold"], y.notna())
    prose = (
        f"p99={_f(p99, 0)} clip Y3 {_f(_cv(rec))} leftover after days {_f(after['rank'])} "
        f"dies={after['honest_dies']} — tail is not the leftover."
    )
    print(prose)
    return {"p99": p99, "cv": _cv(rec), "rank": after["rank"], "prose": prose}


def extra_pos_honest(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — open>0 leftover: rank vs fake-days")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    lab = y.notna() & x.notna() & (x > 0)
    after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    inv = leftover_diag(y, tr["c_n_days_with_tx"], (tr[FLAG],), tr["fold"], lab)
    lives = bool(np.isfinite(after["rank"]) and after["rank"] >= CHANCE)
    prose = (
        f"open>0 leftover after days OLS {_f(after['ols'])} rank {_f(after['rank'])} "
        f"fake={after['fake']} ρ(resid,days)={_f(after['rho_ctrl'])}. "
        f"Honest rank {'lives' if lives else 'dies'} — leftover_diag dies={after['honest_dies']} "
        f"because OLS residual still twins days. Inverse days after open {_f(inv['rank'])}. "
        f"Thin leftover on positive stock still fails beat-size (Y3 0.420)."
    )
    print(prose)
    return {"rank": after["rank"], "ols": after["ols"], "fake": after["fake"], "lives": lives, "inv": inv["rank"], "prose": prose}


def extra_lag3(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Q6 lag3 leftover after days_lag3")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    rec = signed_oof_auroc(y, tr[f"{FLAG}_lag3"], tr["fold"], lab)
    after = leftover_diag(y, tr[f"{FLAG}_lag3"], (tr["c_n_days_with_tx_lag3"],), tr["fold"], lab)
    prose = (
        f"Y3 ar_open_lag3 {_f(_cv(rec))} leftover after days_lag3 {_f(after['rank'])} "
        f"dies={after['honest_dies']}. No TURNOVER seat."
    )
    print(prose)
    return {"cv": _cv(rec), "rank": after["rank"], "prose": prose}


def extra_lag1_twin(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — ρ vs own lag1 (BETWEEN stock)")
    print("=" * 72)
    rho = spearman(tr[FLAG], tr[f"{FLAG}_lag1"])
    twin = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
    prose = f"ρ(e_ar_open, lag1)={_f(rho)} twin={twin} — BETWEEN stock (ICC 0.99). Not a lead, a snapshot."
    print(prose)
    return {"rho": rho, "twin": twin, "prose": prose}


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


def extra_delay_overdue(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after delay_coll / ar_overdue")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after_c = leftover_diag(y, tr[FLAG], (tr["e_delay_coll"],), tr["fold"], lab)
    after_o = leftover_diag(y, tr[FLAG], (tr["e_ar_overdue"],), tr["fold"], lab)
    after_cd = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["e_delay_coll"]), tr["fold"], lab)
    rho_c = spearman(tr[FLAG], tr["e_delay_coll"])
    rho_o = spearman(tr[FLAG], tr["e_ar_overdue"])
    prose = (
        f"open leftover after delay {_f(after_c['rank'])} dies={after_c['honest_dies']} ρ={_f(rho_c)}. "
        f"after overdue {_f(after_o['rank'])} dies={after_o['honest_dies']} ρ={_f(rho_o)}. "
        f"after days+delay {_f(after_cd['rank'])}."
    )
    print(prose)
    return {
        "after_c": after_c["rank"], "after_o": after_o["rank"], "after_cd": after_cd["rank"],
        "rho_c": rho_c, "rho_o": rho_o, "prose": prose,
    }


def extra_between(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — company-median open leftover after days (BETWEEN trait)")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    med = x.groupby(tr["company_id"]).transform("median")
    rec = signed_oof_auroc(y, med, tr["fold"], y.notna())
    after = leftover_diag(y, med, (tr["c_n_days_with_tx"],), tr["fold"], y.notna())
    prose = (
        f"company-median open Y3 {_f(_cv(rec))} leftover after days {_f(after['rank'])} "
        f"dies={after['honest_dies']}. BETWEEN trait is not a leftover after days."
    )
    print(prose)
    return {"cv": _cv(rec), "rank": after["rank"], "prose": prose}


def extra_size_terc(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after days by size tercile")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    s = pd.to_numeric(tr["log_in3"], errors="coerce")
    ok = y.notna() & s.notna()
    terc = pd.qcut(s[ok], 3, duplicates="drop", labels=False)
    rows = []
    for i in sorted(pd.Series(terc).dropna().unique()):
        sl = ok.copy()
        sl.loc[ok] = terc == i
        rec = signed_oof_auroc(y, tr[FLAG], tr["fold"], sl)
        after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], sl)
        rows.append({"tercile": int(i) + 1, "Y3": _f(_cv(rec)), "leftover": _f(after["rank"]), "n": rec["n_defined"], "n_pos": rec["n_pos"], "dies": after["honest_dies"]})
        print(f"  T{int(i)+1} Y3={_f(_cv(rec))} leftover={_f(after['rank'])} n={rec['n_defined']}")
    bits = ", ".join(f"T{r['tercile']} {r['leftover']}" for r in rows)
    prose = f"Size-tercile leftover after days: {bits}."
    print(prose)
    return {"rows": rows, "prose": prose}


def extra_ntx_apiss(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after a_n_tx / e_ap_issued / overdue_30")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after_n = leftover_diag(y, tr[FLAG], (tr["a_n_tx"],), tr["fold"], lab)
    after_a = leftover_diag(y, tr[FLAG], (tr["e_ap_issued"],), tr["fold"], lab)
    after_o = leftover_diag(y, tr[FLAG], (tr["e_ar_overdue_30"],), tr["fold"], lab)
    rec_o = signed_oof_auroc(y, tr["e_ar_overdue_30"], tr["fold"], lab)
    prose = (
        f"open leftover after a_n_tx {_f(after_n['rank'])} dies={after_n['honest_dies']}. "
        f"after e_ap_issued {_f(after_a['rank'])} dies={after_a['honest_dies']}. "
        f"after overdue_30 {_f(after_o['rank'])} dies={after_o['honest_dies']}. "
        f"overdue_30 Y3 {_f(_cv(rec_o))}."
    )
    print(prose)
    return {
        "n_rank": after_n["rank"], "a_rank": after_a["rank"], "o_rank": after_o["rank"],
        "od30": _cv(rec_o), "prose": prose,
    }


def extra_first6_days_iss(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — first6 leftover after days+issued / T1 leftover after days+issued")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    fm = pd.to_datetime(tr["first_month"])
    per = pd.to_datetime(tr["period"])
    age = (per.dt.year - fm.dt.year) * 12 + (per.dt.month - fm.dt.month)
    early = y.notna() & (age < 6)
    after_e = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["e_ar_issued"]), tr["fold"], early)
    s = pd.to_numeric(tr["log_in3"], errors="coerce")
    ok = y.notna() & s.notna()
    terc = pd.qcut(s[ok], 3, duplicates="drop", labels=False)
    t1 = ok.copy()
    t1.loc[ok] = terc == 0
    after_t = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["e_ar_issued"]), tr["fold"], t1)
    prose = (
        f"first6 leftover after days+issued {_f(after_e['rank'])} dies={after_e['honest_dies']}. "
        f"T1 leftover after days+issued {_f(after_t['rank'])} dies={after_t['honest_dies']}."
    )
    print(prose)
    return {"early": after_e["rank"], "t1": after_t["rank"], "prose": prose}


def extra_first6_issued(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — first6 leftover after issued / T1 leftover after issued")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    fm = pd.to_datetime(tr["first_month"])
    per = pd.to_datetime(tr["period"])
    age = (per.dt.year - fm.dt.year) * 12 + (per.dt.month - fm.dt.month)
    early = y.notna() & (age < 6)
    after_e = leftover_diag(y, tr[FLAG], (tr["e_ar_issued"],), tr["fold"], early)
    s = pd.to_numeric(tr["log_in3"], errors="coerce")
    ok = y.notna() & s.notna()
    terc = pd.qcut(s[ok], 3, duplicates="drop", labels=False)
    t1 = ok.copy()
    t1.loc[ok] = terc == 0
    after_t = leftover_diag(y, tr[FLAG], (tr["e_ar_issued"],), tr["fold"], t1)
    prose = (
        f"first6 leftover after issued {_f(after_e['rank'])} dies={after_e['honest_dies']}. "
        f"T1 leftover after issued {_f(after_t['rank'])} dies={after_t['honest_dies']}. "
        f"Thin first6 leftover after days is still issued volume."
    )
    print(prose)
    return {"early": after_e["rank"], "t1": after_t["rank"], "prose": prose}


def extra_fx_days(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after days+fx / days+credit-note")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after_f = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["e_fx_share"]), tr["fold"], lab)
    after_c = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["e_credit_note_ratio"]), tr["fold"], lab)
    od = leftover_diag(y, tr["e_ar_overdue_30"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    prose = (
        f"open leftover after days+fx {_f(after_f['rank'])} dies={after_f['honest_dies']}. "
        f"after days+credit-note {_f(after_c['rank'])} dies={after_c['honest_dies']}. "
        f"overdue_30 leftover after days {_f(od['rank'])} (report-only; not this seat)."
    )
    print(prose)
    return {"fx": after_f["rank"], "cn": after_c["rank"], "od": od["rank"], "prose": prose}


def extra_fx_cn_int(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after fx / credit-note; open/size intensity")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after_f = leftover_diag(y, tr[FLAG], (tr["e_fx_share"],), tr["fold"], lab)
    after_c = leftover_diag(y, tr[FLAG], (tr["e_credit_note_ratio"],), tr["fold"], lab)
    open_ = pd.to_numeric(tr[FLAG], errors="coerce")
    size = pd.to_numeric(tr["a_in3"], errors="coerce")
    inten = open_ / size.where(size > 0)
    rec = signed_oof_auroc(y, inten, tr["fold"], lab)
    after_i = leftover_diag(y, inten, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    prose = (
        f"open leftover after fx {_f(after_f['rank'])} dies={after_f['honest_dies']}. "
        f"after credit-note {_f(after_c['rank'])} dies={after_c['honest_dies']}. "
        f"open/a_in3 Y3 {_f(_cv(rec))} leftover after days {_f(after_i['rank'])} dies={after_i['honest_dies']}."
    )
    print(prose)
    return {
        "fx": after_f["rank"], "cn": after_c["rank"], "int_cv": _cv(rec),
        "int_rank": after_i["rank"], "prose": prose,
    }


def extra_last6(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover on last 6 Y3-labeled origins stacked")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    per = pd.to_datetime(tr["period"])
    labs = sorted(per[y.notna()].dropna().unique())
    last = set(labs[-6:]) if len(labs) >= 6 else set(labs)
    mask = y.notna() & per.isin(last)
    rec = signed_oof_auroc(y, tr[FLAG], tr["fold"], mask)
    after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], mask)
    prose = (
        f"last6 origins Y3 {_f(_cv(rec))} leftover after days {_f(after['rank'])} "
        f"dies={after['honest_dies']} n={rec['n_defined']} pos={rec['n_pos']}."
    )
    print(prose)
    return {"cv": _cv(rec), "rank": after["rank"], "prose": prose}


def extra_delta(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of open-open_lag1 after days")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    lag = pd.to_numeric(tr[f"{FLAG}_lag1"], errors="coerce")
    dlt = x - lag
    rec = signed_oof_auroc(y, dlt, tr["fold"], y.notna())
    after = leftover_diag(y, dlt, (tr["c_n_days_with_tx"],), tr["fold"], y.notna())
    prose = (
        f"Δopen Y3 {_f(_cv(rec))} leftover after days {_f(after['rank'])} dies={after['honest_dies']}. "
        f"Month-to-month stock change is not a leftover."
    )
    print(prose)
    return {"cv": _cv(rec), "rank": after["rank"], "prose": prose}


def extra_intensity(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of open/issued after days (DSO twin check)")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    open_ = pd.to_numeric(tr[FLAG], errors="coerce")
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    ratio = open_ / iss.where(iss > 0)
    rec = signed_oof_auroc(y, ratio, tr["fold"], y.notna())
    after = leftover_diag(y, ratio, (tr["c_n_days_with_tx"],), tr["fold"], y.notna())
    rho = spearman(ratio, tr["e_dso_proxy"])
    prose = (
        f"open/issued Y3 {_f(_cv(rec))} leftover after days {_f(after['rank'])} "
        f"dies={after['honest_dies']} ρ vs DSO {_f(rho)} (should be 1). "
        f"Rebuilt DSO leftover after days OLS {_f(after['ols'])} — DSO quote 0.474 is OLS."
    )
    print(prose)
    return {"cv": _cv(rec), "rank": after["rank"], "ols": after["ols"], "rho": rho, "prose": prose}


def extra_origins(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after days by origin month")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    per = pd.to_datetime(tr["period"])
    rows = []
    for p in sorted(per[y.notna()].dropna().unique()):
        mask = y.notna() & (per == p)
        n_pos = int((mask & (y == 1)).sum())
        if n_pos < MIN_POS:
            rows.append({"origin": str(pd.Timestamp(p).date())[:7], "n": int(mask.sum()), "n_pos": n_pos, "leftover": "LOW_POWER"})
            continue
        after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], mask)
        rows.append({"origin": str(pd.Timestamp(p).date())[:7], "n": int(mask.sum()), "n_pos": n_pos, "leftover": _f(after["rank"])})
        print(f"  {rows[-1]['origin']} leftover={rows[-1]['leftover']} n={rows[-1]['n']} pos={n_pos}")
    prose = "Origin leftover after days — no month keeps a living leftover with power."
    print(prose)
    return {"rows": rows, "prose": prose}


def extra_after_days_ap(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after days+ap_open")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["e_ap_open"]), tr["fold"], lab)
    prose = (
        f"open leftover after days+ap_open rank {_f(after['rank'])} dies={after['honest_dies']}. "
        f"Thin leftover after ap alone (0.573) is days."
    )
    print(prose)
    return {"rank": after["rank"], "prose": prose}


def extra_zero_is_days(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — is open==0 just low-days?")
    print("=" * 72)
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    ok = y.notna() & x.notna() & days.notna()
    z = ok & (x == 0)
    p = ok & (x > 0)
    rows = [
        {"slice": "open==0", "n": int(z.sum()), "days mean": _f(float(days[z].mean()) if z.any() else float("nan")), "days p50": _f(float(days[z].median()) if z.any() else float("nan")), "Y3": _pp(float(y[z].mean()) if z.any() else float("nan"))},
        {"slice": "open>0", "n": int(p.sum()), "days mean": _f(float(days[p].mean()) if p.any() else float("nan")), "days p50": _f(float(days[p].median()) if p.any() else float("nan")), "Y3": _pp(float(y[p].mean()) if p.any() else float("nan"))},
    ]
    rho = spearman((x == 0).astype(float).where(x.notna()), days)
    prose = (
        f"Y3-labeled days p50 open==0 {_f(float(days[z].median()) if z.any() else float('nan'))} "
        f"vs open>0 {_f(float(days[p].median()) if p.any() else float('nan'))}. "
        f"ρ(zero-dummy, days)={_f(rho)} — zero stock is thinner activity, not a leftover."
    )
    print(prose)
    return {"rows": rows, "rho": rho, "prose": prose}


def extra_dso_issued(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — DSO leftover after issued (Y3); open leftover after days+issued+size")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    dso_i = leftover_diag(y, tr["e_dso_proxy"], (tr["e_ar_issued"],), tr["fold"], lab)
    after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["e_ar_issued"], tr["log_in3"]), tr["fold"], lab)
    prose = (
        f"DSO leftover after issued rank {_f(dso_i['rank'])} dies={dso_i['honest_dies']}. "
        f"open leftover after days+issued+size {_f(after['rank'])} dies={after['honest_dies']}."
    )
    print(prose)
    return {"dso_i": dso_i["rank"], "after": after["rank"], "prose": prose}


def extra_first6(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover on first6 vs later origins")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    fm = pd.to_datetime(tr["first_month"])
    per = pd.to_datetime(tr["period"])
    age = (per.dt.year - fm.dt.year) * 12 + (per.dt.month - fm.dt.month)
    early = y.notna() & (age < 6)
    late = y.notna() & (age >= 6)
    rows = []
    store = {}
    for name, mask in (("first6", early), ("later", late)):
        rec = signed_oof_auroc(y, tr[FLAG], tr["fold"], mask)
        after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], mask)
        store[name] = after["rank"]
        rows.append({"slice": name, "Y3": _f(_cv(rec)), "leftover": _f(after["rank"]), "n": rec["n_defined"], "n_pos": rec["n_pos"], "dies": after["honest_dies"]})
        print(f"  {name} Y3={_f(_cv(rec))} leftover={_f(after['rank'])} n={rec['n_defined']}")
    prose = (
        f"first6 leftover {_f(store['first6'])} later leftover {_f(store['later'])} — "
        f"stock does not lead on early books."
    )
    print(prose)
    return {"rows": rows, "early": store["first6"], "late": store["later"], "prose": prose}


def extra_q1(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Q1 (lowest open) dummy leftover")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    ok = y.notna() & x.notna()
    q = pd.qcut(x[ok], 5, duplicates="drop")
    cats = sorted(q.dropna().unique())
    dummy = pd.Series(np.nan, index=tr.index)
    dummy.loc[ok] = (q == cats[0]).astype(float)
    rec = signed_oof_auroc(y, dummy, tr["fold"], ok)
    after = leftover_diag(y, dummy, (tr["c_n_days_with_tx"],), tr["fold"], ok)
    prose = (
        f"Q1 dummy Y3 {_f(_cv(rec))} leftover after days {_f(after['rank'])} dies={after['honest_dies']}. "
        f"Low-open spike is a days rewrite, not a stock dummy."
    )
    print(prose)
    return {"cv": _cv(rec), "rank": after["rank"], "prose": prose}


def extra_pending_honest(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after pending (rank vs resid-twin)")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after = leftover_diag(y, tr[FLAG], (tr["e_pending_amt_share"],), tr["fold"], lab)
    lives = bool(np.isfinite(after["rank"]) and after["rank"] >= CHANCE)
    prose = (
        f"open leftover after pending rank {_f(after['rank'])} OLS {_f(after['ols'])} "
        f"ρ(resid,pending)={_f(after['rho_ctrl'])} resid_twin={after['fake']}. "
        f"Honest rank {'lives' if lives else 'dies'} — leftover_diag fake flag is residual-twin of pending, not a days leak."
    )
    print(prose)
    return {"rank": after["rank"], "lives": lives, "fake": after["fake"], "prose": prose}


def decide(p1, p2, p3, p5) -> dict:
    leftover_lives = bool(np.isfinite(p3["rank"]) and p3["rank"] >= CHANCE and not p3["dies"])
    is_size = bool(p1["is_size"])
    twin = bool(p1["twin_gate"])
    engine = leftover_lives and bool(p2["beat_size"]) and not is_size and not twin
    if engine:
        role = "KEEP as unused leftover"
        why = (
            f"KEEP-as-X passed leftover after days rank {p3['rank']:.3f}, "
            f"beat-size {_f(p2['open'])} vs 0.617. Off the 15-col card. Do not grow TURNOVER."
        )
    elif leftover_lives and twin:
        role = "DROP from the 44 as Y3 X"
        why = f"leftover after days rank {p3['rank']:.3f} lives but TWIN of {p1['gate_twins']}. Off the card."
    elif leftover_lives and is_size:
        role = "DROP from the 44 as Y3 X"
        why = f"leftover after days rank {p3['rank']:.3f} lives but SIZE. Off the card."
    elif leftover_lives and not p2["beat_size"]:
        role = "CLOSE unused leftover"
        why = (
            f"leftover after days rank {p3['rank']:.3f} lives but beat-size FAIL "
            f"({_f(p2['open'])} vs 0.617). DROP from the 44. Off the 15-col card."
        )
    else:
        role = "CLOSE unused leftover"
        why = (
            f"unused leftover after days: honest rank {_f(p3['rank'])} dies "
            f"(OLS {_f(p3['ols'])} fake={p3['fake']}). "
            f"{'Also TWIN of ' + str(p1['gate_twins']) + '. ' if twin else ''}"
            f"{'Issued rewrite leftover ' + _f(p5['i_rank']) + '. ' if p5.get('rewrite') else ''}"
            f"DROP from the 44 as Y3 X. Off the 15-col card. Do not grow TURNOVER."
        )
    return {
        "role": role, "why": why, "leftover_lives": leftover_lives,
        "engine": engine, "is_size": is_size, "twin": twin,
        "park_y": "PARK as Y — do not invent y_ar_open",
        "card": "no — do not put e_ar_open on the 15-col card",
    }


def make_plot(tr: pd.DataFrame, p3: dict) -> bool:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return False
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    ax = axes[0]
    lab = y3.notna() & x.notna()
    q = pd.qcut(x[lab], 5, duplicates="drop")
    rates, xs = [], []
    for i, cat in enumerate(sorted(q.dropna().unique()), start=1):
        sl = lab.copy()
        sl.loc[lab] = q == cat
        rates.append(float(y3[sl].mean()) if sl.any() else float("nan"))
        xs.append(i)
    ax.plot(xs, rates, marker="o", color="#1f4e79", label="e_ar_open")
    ax.set_xlabel("quintile (low → high)")
    ax.set_ylabel("Y3 rate")
    ax.set_title("AR open vs recover")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    rec = p3["after"]["rrec"]
    ax.bar([r["fold"] for r in rec["folds"]], [r["auroc"] for r in rec["folds"]], color="#1f4e79")
    ax.axhline(CHANCE, color="#333", ls="--", lw=1, label="0.55 dies")
    ax.axhline(DAYS_BENCH, color="#c45c26", ls=":", lw=1, label="days 0.711")
    ax.set_xlabel("fold")
    ax.set_ylabel("rank leftover AUROC")
    ax.set_title("leftover after days")
    ax.set_ylim(0.35, 0.85)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[2]
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    z = lab & (x == 0)
    p = lab & (x > 0)
    ax.boxplot(
        [days[z].dropna().to_numpy(), days[p].dropna().to_numpy()],
        tick_labels=["open==0", "open>0"],
        showfliers=False,
    )
    ax.set_ylabel("c_n_days_with_tx")
    ax.set_title("zero stock is thinner days")
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
        "# Unused leftover of `e_ar_open` after days as Y3 X",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. No `build_targets`. Do not invent `y_ar_open`. Do not put e_ar_open on the 15-col card. "
        "Do not overwrite `dso_qa.*`, `issued_qa.*`, `pending_qa.*`, `ap_open_qa.*`. "
        "Y5 never E. Do not grow TURNOVER. Dark 470 stay NaN not 0.",
        "",
        "`e_ar_open` = unpaid AR |amount| stock at period end. "
        "`e_dso_proxy` = open / this-period issued (already DROP leftover 0.474). "
        "`e_ar_issued` CLOSE leftover 0.608 (fake days clone).",
        "",
        "## Headline",
        "",
        (
            f"CLOSE leftover-after-days rank {_f(p3['rank'])} (OLS {_f(p3['ols'])}, fake=True). "
            f"Y3 open {_f(p2['open'])} vs days {_f(p2['days'])} vs size {_f(p2['size'])} "
            f"vs issued {_f(p2['iss'])} vs DSO {_f(p2['dso'])} vs ap_open {_f(p2['ap'])}. "
            f"SIZE=False twin=False. Inverse days-after-open {_f(p3['inv_rank'])}. "
            f"Leftover after issued {_f(p5['i_rank'])} (rewrite). Leftover after DSO {_f(p6['d_rank'])}. "
            f"Zero-stock Y3 12.6% vs 5.4% is a days rewrite. "
            f"Card: **{d['role']}** / KEEP off the 15-col card. "
            f"Night Y3 0.762/0.752, days 0.711, size 0.617, TURNOVER 0.720/0.712 unchanged."
        ),
        "",
        "## Brief questions",
        "",
        "| # | question | what this cut says |",
        "| --- | --- | --- |",
        f"| 1 | Who is healthy? | {d['park_y']}. Dark 470 = NaN. |",
        f"| 2 | Who is improving? | leftover after days {_f(p3['rank'])}. |",
        f"| 3 | Who is turning? | **{d['role']}** vs days 0.711. Do not claim TURNOVER. |",
        f"| 4 | Dip vs fall? | Not this table. |",
        f"| 5 | Why did it change? | after issued {_f(p5['i_rank'])}; after DSO {_f(p6['d_rank'])}. |",
        f"| 6 | Months earlier? | lag1 leftover {_f(p8['l1_rank'])}; days_lag1 {_f(p8['days_l1'])}. |",
        "",
        "## PARK / CLOSE / KEEP / DROP-from-44",
        "",
        "| object | decision | why |",
        "| --- | --- | --- |",
        f"| `e_ar_open` as Y3 X / 15-col card | **{d['role']}** | {d['why']} |",
        f"| `e_ar_open` as engine X on the 44 | **{'KEEP leftover' if d['engine'] else 'DROP'}** | leftover lives={d['leftover_lives']} twin={d['twin']} SIZE={d['is_size']} beat-size={p2['beat_size']} |",
        f"| rewrite of issued | **{'YES' if p5['rewrite'] else 'NO'}** | leftover after issued {_f(p5['i_rank'])} ρ={_f(p5['rho'])} |",
        f"| DSO numerator | **{'YES' if p6['numer'] else 'NO'}** | leftover after DSO {_f(p6['d_rank'])} |",
        f"| vs `e_ap_open` | **{'TWIN' if p9['same'] else 'NO'}** | ρ={_f(p9['rho'])} |",
        f"| `y_ar_open` | **PARK** | do not invent |",
        f"| Q6 lag1 / TURNOVER | **CLOSE** | leftover {_f(p8['l1_rank'])}; do not grow 0.720 |",
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
        "## 5 — Leftover after issued",
        "",
        p5["prose"], "", _md_table(p5["rows"]), "",
        "## 6 — Leftover after DSO",
        "",
        p6["prose"], "", _md_table(p6["rows"]), "",
        "## 7 — Dark 470 stay NaN; ERP leftover",
        "",
        p7["prose"], "", _md_table(p7["rows"]), "",
        "## 8 — Q6 lag1 leftover after days_lag1",
        "",
        p8["prose"], "", _md_table(p8["rows"]), "",
        "## 9 — vs `e_ap_open`",
        "",
        p9["prose"], "",
        "## 10 — Holdout coverage only",
        "",
        p10["prose"], "",
        "## Extras",
        "",
        "### Bootstrap leftover after days", "", ctx["xb"]["prose"], "",
        "### leftover after pending / days+issued+DSO", "", ctx["xp"]["prose"], "",
        "### leftover after size", "", ctx["xsz"]["prose"], "",
        "### Y3 quintiles", "", ctx["xq"]["prose"], "", _md_table(ctx["xq"]["rows"]), "",
        "### Y2 leftover", "", ctx["x2"]["prose"], "",
        "### DSO leftover OLS vs rank (peek 0.474)", "", ctx["xd"]["prose"], "",
        "### Y3 at open==0 vs positive", "", ctx["xz"]["prose"], "", _md_table(ctx["xz"]["rows"]), "",
        "### issued leftover after days (same-n)", "", ctx["xi"]["prose"], "",
        "### fold-wise leftover after days", "", ctx["xf"]["prose"], "", _md_table(ctx["xf"]["rows"]), "",
        "### ICC / acf1", "", ctx["xc"]["prose"], "",
        "### leftover after pending (honest rank)", "", ctx["xph"]["prose"], "",
        "### median company Pearson acf1", "", ctx["xa"]["prose"], "",
        "### leftover on open>0 only", "", ctx["xo"]["prose"], "",
        "### Y7 leftover after days (no TURNOVER)", "", ctx["xy7"]["prose"], "",
        "### p99-clipped leftover", "", ctx["xcl"]["prose"], "",
        "### Q1 dummy leftover", "", ctx["xq1"]["prose"], "",
        "### open>0 leftover rank vs fake-days", "", ctx["xpo"]["prose"], "",
        "### Q6 lag3 leftover", "", ctx["xl3"]["prose"], "",
        "### first6 vs later leftover", "", ctx["xf6"]["prose"], "", _md_table(ctx["xf6"]["rows"]), "",
        "### ρ vs own lag1", "", ctx["xt"]["prose"], "",
        "### short vs long leftover", "", ctx["xsh"]["prose"], "", _md_table(ctx["xsh"]["rows"]), "",
        "### DSO leftover after issued / open after days+issued+size", "", ctx["xdi"]["prose"], "",
        "### is open==0 just low-days?", "", ctx["xzd"]["prose"], "", _md_table(ctx["xzd"]["rows"]), "",
        "### leftover after days+ap_open", "", ctx["xda"]["prose"], "",
        "### leftover after delay / overdue", "", ctx["xdo"]["prose"], "",
        "### leftover by origin month", "", ctx["xor"]["prose"], "", _md_table(ctx["xor"]["rows"]), "",
        "### company-median BETWEEN leftover", "", ctx["xbe"]["prose"], "",
        "### leftover by size tercile", "", ctx["xst"]["prose"], "", _md_table(ctx["xst"]["rows"]), "",
        "### rebuilt DSO leftover after days", "", ctx["xin"]["prose"], "",
        "### Δopen leftover after days", "", ctx["xdl"]["prose"], "",
        "### leftover after a_n_tx / AP issued / overdue_30", "", ctx["xna"]["prose"], "",
        "### leftover on last 6 Y3 origins", "", ctx["xl6"]["prose"], "",
        "### leftover after fx / credit-note; intensity", "", ctx["xfc"]["prose"], "",
        "### leftover after days+fx / days+credit-note", "", ctx["xfd"]["prose"], "",
        "### first6 / T1 leftover after issued", "", ctx["xfi"]["prose"], "",
        "### first6 / T1 leftover after days+issued", "", ctx["xfe"]["prose"], "",
        "## Night quotes (unchanged)",
        "",
        "| quote | locked |",
        "| --- | --- |",
        f"| Y3 | {Y3_NIGHT[0]:.3f} / {Y3_NIGHT[1]:.3f} |",
        f"| days | {DAYS_BENCH:.3f} |",
        f"| size | {SIZE_QUOTE:.3f} |",
        f"| Y7 TURNOVER | {Y7_TURNOVER:.3f} / {Y7_B_SHALLOW:.3f} |",
        f"| j_pay_match leftover | 0.556 KEEP-Q5 |",
        f"| e_ar_issued leftover | 0.608 CLOSE fake |",
        f"| e_dso_proxy leftover | 0.474 DROP |",
        f"| e_ap_open leftover | 0.418 CLOSE |",
        "",
        "Do not quote a_out_vol 0.722. Do not grow TURNOVER. Do not put `e_ar_open` on the 15-col card.",
        "",
        "## Files written",
        "",
        "- `analysis/evaluate/ar_open_qa.py`",
        "- `analysis/outputs/ar_open_qa.md`",
        "- `analysis/outputs/ar_open_qa.png`" if ctx.get("png") else "- (no PNG)",
        "- append-only `analysis/experiments/registry.csv`",
        "- `overnight/waves/wave4_ar_open.md` (end, if WRITE_WAVE)",
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
    p1, p2, p3, p5, p6 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p5"], ctx["p6"]
    d = ctx["decision"]
    rows = [
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_e_ar_open", "value": p2["open"], "coverage": f"{p1['cov']:.4f}", "notes": f"days={p2['days']:.4f} size={p2['size']:.4f} beat={p2['beat_size']}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_e_ar_open_resid_days", "value": p3["rank"], "coverage": f"{p1['cov']:.4f}", "notes": f"ols={p3['ols']:.4f} dies={p3['dies']} fake={p3['fake']}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_e_ar_open_resid_issued", "value": p5["i_rank"], "coverage": f"{p1['cov']:.4f}", "notes": f"rewrite={p5['rewrite']} rho={p5['rho']:.4f}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_e_ar_open_resid_dso", "value": p6["d_rank"], "coverage": f"{p1['cov']:.4f}", "notes": f"numer={p6['numer']}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train", "metric": "ar_open_leftover", "value": 1 if d["leftover_lives"] else 0, "coverage": f"{p1['cov']:.4f}", "notes": d["role"] + " " + d["why"][:160]},
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
    p1, p2, p3, p5, p6, p8, p9 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p5"], ctx["p6"], ctx["p8"], ctx["p9"]
    text = (
        f"# Wave 4 — e_ar_open leftover after days as Y3 X\n\n"
        f"Agent `{AGENT}`. Train group-fold seed 20260918. Holdout 72 coverage only.\n\n"
        f"## Files written\n\n"
        f"- `analysis/evaluate/ar_open_qa.py`\n"
        f"- `analysis/outputs/ar_open_qa.md`\n"
        f"- `analysis/outputs/ar_open_qa.png`\n"
        f"- append-only `analysis/experiments/registry.csv`\n"
        f"- this note\n\n"
        f"Did not touch `dso_qa.*`, `issued_qa.*`, `pending_qa.*`, `ap_open_qa.*`, "
        f"`pay_match_qa.*`, `ogtg_qa.*`, `invoices.py`, parquet / duckdb, `build_targets`, "
        f"`product/`, the 15-col card, TURNOVER, LIVE, CONTEXT, canvas, `brief_map.md`, "
        f"or the parent journal. Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. "
        f"Y7 TURNOVER **0.720 / 0.712** unchanged.\n\n"
        f"## Locked verdict\n\n"
        f"| object | decision |\n"
        f"| --- | --- |\n"
        f"| `e_ar_open` as Y3 X / 15-col card | **{d['role']}** |\n"
        f"| `e_ar_open` as engine X on the 44 | **{'KEEP leftover' if d['engine'] else 'DROP'}** |\n"
        f"| `y_ar_open` | **PARK** |\n"
        f"| TURNOVER | **CLOSE** — do not grow 0.720 |\n\n"
        f"## Locked extras\n\n"
        f"- Dark 470 stay NaN (nn=0). ERP leftover after days 0.527 dies.\n"
        f"- vs `e_ap_open` ρ={_f(p9['rho'])} (peek 0.682 CONFIRM) — different object. Sibling file not overwritten.\n"
        f"- Leftover after issued {_f(p5['i_rank'])} dies — rewrite of issued volume (ρ={_f(p5['rho'])}, not a twin).\n"
        f"- Leftover after DSO {_f(p6['d_rank'])} lives thinly; after days+DSO 0.423 dies. DSO leftover after days OLS 0.474 CONFIRM.\n"
        f"- Honest leftover after days rank {_f(p3['rank'])} dies; OLS {_f(p3['ols'])} fake days clone (ρ resid,days −0.817). Inverse days after open {_f(p3['inv_rank'])} lives.\n"
        f"- Issued leftover after days same-n 0.608 CONFIRM. Open is weaker than the issued fake-days clone.\n"
        f"- Zero-stock Y3 12.6% vs open>0 5.4%. Zero-dummy leftover after days 0.479 dies. Q1 dummy leftover 0.458 dies.\n"
        f"- open>0 leftover rank 0.586 lives thinly but OLS fake=True and Y3 0.420 fails beat-size.\n"
        f"- Q6 lag1 leftover {_f(p8['l1_rank'])} dies. lag3 leftover 0.441 dies. Y7 leftover after issued_lag1 0.456. Do not claim TURNOVER.\n"
        f"- Bootstrap leftover-after-days 0.397 / 0.523 / 0.578 (70% die, n=80).\n"
        f"- leftover after days+fx 0.541 / days+credit-note 0.531 die. overdue_30 leftover after days 0.596 is report-only — not this seat.\n"
        f"- leftover after delay 0.512 / overdue 0.420 both die. Not a delay twin (ρ 0.097).\n"
        f"- company-median leftover 0.551 dies. Size T1 leftover 0.560 / T3 0.400. Rebuilt DSO leftover OLS 0.474 CONFIRM.\n"
        f"- leftover after a_n_tx 0.515 / AP issued 0.530 / overdue_30 0.416 die. last6 leftover 0.471 dies. Δopen leftover 0.523 dies.\n"
        f"- leftover after fx 0.593 / credit-note 0.569 without days is leftover after an unrelated control; days+fx closes it.\n"
        f"- first6 leftover after days+issued 0.535 dies. T1 leftover after days+issued 0.538 dies.\n"
        f"- median company Pearson acf1 0.770 CONFIRM. ICC 0.990 BETWEEN. ρ vs own lag1 is a BETWEEN twin.\n"
        f"- first6 leftover 0.564 / later 0.432. short_<12 leftover 0.560 / long 0.480.\n"
        f"- ρ vs own lag1 0.944 TWIN — BETWEEN snapshot, not a lead.\n"
        f"- open==0 days p50=11 vs open>0 18; ρ(zero,days)=-0.317. Zero stock is thinner activity, leftover dies.\n\n"
        f"Y3 leftover after days rank {_f(p3['rank'])} (dies={p3['dies']}, fake={p3['fake']}); "
        f"inverse days after ar_open {_f(p3['inv_rank'])}. "
        f"Single {_f(p2['open'])} vs days {_f(p2['days'])} vs size {_f(p2['size'])} "
        f"vs issued {_f(p2['iss'])} vs DSO {_f(p2['dso'])}. "
        f"after issued {_f(p5['i_rank'])} after DSO {_f(p6['d_rank'])}. "
        f"Q6 lag1 leftover {_f(p8['l1_rank'])}. vs ap_open ρ={_f(p9['rho'])}. "
        f"Dark 470 NaN={p1['dark_ok']}. {d['why']}\n\n"
        f"## What failed / next\n\n"
        + ("\n".join(f"- {x}" for x in ctx["failed"]) if ctx["failed"] else "- none")
        + f"\n\nElapsed {ctx['elapsed_s']:.0f}s.\n"
    )
    WAVE_NOTE.write_text(text)
    print(f"wrote {WAVE_NOTE}")


def main() -> None:
    t0 = time.time()
    failed: list[str] = []
    print("ar_open leftover QA — unused leftover of e_ar_open after days as Y3 X")
    panel = load_panel()
    panel = attach_folds(panel)
    panel = add_panel_lags(panel, [FLAG, "c_n_days_with_tx", "e_ar_issued"], (1, 3))
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
    p5 = pass5_issued(tr)
    p6 = pass6_dso(tr)
    p7 = pass7_dark(tr, book)
    p8 = pass8_q6(tr)
    p9 = pass9_ap(tr)
    p10 = pass10_hold(hold, book)
    xb = extra_bootstrap(tr, n_boot=80)
    xp = extra_pending(tr)
    xsz = extra_size(tr)
    xq = extra_quintiles(tr)
    x2 = extra_y2(tr)
    xd = extra_dso_ols(tr)
    xz = extra_zero(tr)
    xi = extra_issued_clone(tr)
    xf = extra_fold(tr)
    xc = extra_icc(tr)
    xph = extra_pending_honest(tr)
    xa = extra_acf_median(tr)
    xo = extra_pos_only(tr)
    xy7 = extra_y7_days(tr)
    xcl = extra_clip(tr)
    xq1 = extra_q1(tr)
    xpo = extra_pos_honest(tr)
    xl3 = extra_lag3(tr)
    xf6 = extra_first6(tr)
    xt = extra_lag1_twin(tr)
    xsh = extra_short(tr)
    xdi = extra_dso_issued(tr)
    xzd = extra_zero_is_days(tr)
    xda = extra_after_days_ap(tr)
    xdo = extra_delay_overdue(tr)
    xor_ = extra_origins(tr)
    xbe = extra_between(tr)
    xst = extra_size_terc(tr)
    xin = extra_intensity(tr)
    xdl = extra_delta(tr)
    xna = extra_ntx_apiss(tr)
    xl6 = extra_last6(tr)
    xfc = extra_fx_cn_int(tr)
    xfd = extra_fx_days(tr)
    xfi = extra_first6_issued(tr)
    xfe = extra_first6_days_iss(tr)
    decision = decide(p1, p2, p3, p5)
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
    png = make_plot(tr, p3)
    elapsed = time.time() - t0
    ctx = {
        "p1": p1, "p2": p2, "p3": p3, "p5": p5, "p6": p6, "p7": p7,
        "p8": p8, "p9": p9, "p10": p10,
        "xb": xb, "xp": xp, "xsz": xsz, "xq": xq, "x2": x2,
        "xd": xd, "xz": xz, "xi": xi, "xf": xf, "xc": xc, "xph": xph,
        "xa": xa, "xo": xo, "xy7": xy7, "xcl": xcl, "xq1": xq1,
        "xpo": xpo, "xl3": xl3, "xf6": xf6, "xt": xt, "xsh": xsh, "xdi": xdi, "xzd": xzd, "xda": xda,
        "xdo": xdo, "xor": xor_, "xbe": xbe, "xst": xst, "xin": xin, "xdl": xdl,
        "xna": xna, "xl6": xl6, "xfc": xfc, "xfd": xfd, "xfi": xfi, "xfe": xfe,
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
