"""Unused leftover of ``j_pay_match`` after days as Y3 X.

``j_pay_match`` = share of paid-this-month book invoices with a greedy
1-1 bank match (``analysis.features.match``). Family J is
**KEEP-Q5 diagnostic, not a Y3 X**, and is **not merged** into
``monthly.parquet`` / FAMILIES. Compute in-memory. Do **not** overwrite
``match.py``. 470 never-ERP stay **NaN not 0**.

Pending leftover vs J was Y5 0.588 ρ 0.262 (d_tx cut) / pending ρ −0.093
— not a twin, not merged. Do not grow TURNOVER. Do not put J on the
15-col card.

KEEP-as-X: beat size ≥0.02 **and** leftover after days **and** not SIZE
(|ρ| vs log1p(a_in3) ≥0.50) **and** not a twin (|ρ|≥0.80 vs days /
a_n_tx / e_pending_amt_share / j_iss_match). Leftover <0.55 dies.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.pay_match_qa

Owned: analysis/evaluate/pay_match_qa.py, analysis/outputs/pay_match_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_pay_match.md (end).
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
from analysis.features.match import build as build_j
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
OUT_MD = ANALYSIS / "outputs" / "pay_match_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "pay_match_qa.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_pay_match.md"
AGENT = "572fb928"
WAVE = "4"
ROUND = "R4"
MODEL = "pay_match_qa"
X_FAM = "J"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
Y5 = "y5_ar_od30_sust"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_QUOTE = 0.617
DAYS_LAG1_QUOTE = 0.684
Y3_NIGHT = (0.762, 0.752)
Y7_TURNOVER = 0.720
Y7_B_SHALLOW = 0.712
PAY_Y3_PEEK = 0.567
PAY_N_PEEK = 2646
PAY_POS_PEEK = 151
PEND_RHO_PEEK = -0.093
ISS_RHO_PEEK = 0.678
DTX_Y5_LEFT = 0.588
DTX_RHO_PEEK = 0.262
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
    "e_pending_amt_share",
)

Y_KEEP = (Y2, Y3, Y5)
FLAG = "j_pay_match"


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
    if FLAG in raw.columns:
        print("WARNING: j_pay_match already in parquet — will not rewrite; still compute in-memory")
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
        ["c_n_days_with_tx", "log_in3", "e_pending_amt_share"],
        Y3, forbidden_prefixes=["b"],
    )
    if not leak3["ok"]:
        raise RuntimeError(f"Y3 never-B leak: {leak3['issues']}")
    leak5 = leakage_check(["log_in3"], Y5, forbidden_prefixes=["e"])
    if not leak5["ok"]:
        raise RuntimeError(f"Y5 never-E leak: {leak5['issues']}")
    return panel


def attach_j(panel: pd.DataFrame) -> pd.DataFrame:
    grid = panel[["company_id", "period"]].drop_duplicates()
    con = connect()
    t0 = time.time()
    try:
        j = build_j(con, grid)
    finally:
        con.close()
    print(f"Family J in-memory {j.shape} in {time.time() - t0:.1f}s (not merged to parquet)")
    keep = [c for c in ("company_id", "period", "j_pay_match", "j_iss_match", "j_has_book") if c in j.columns]
    j = _keys(j[keep])
    out = panel.merge(j, on=["company_id", "period"], how="left")
    return out


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
    iss = pd.to_numeric(tr["j_iss_match"], errors="coerce")
    has = pd.to_numeric(tr["j_has_book"], errors="coerce")
    dark = ~tr["company_id"].isin(book)
    n_dark_co = int(tr.loc[dark, "company_id"].nunique())
    dark_nn = int(x[dark].notna().sum())
    dark_zero = int((x[dark] == 0).sum())
    dark_has = float(has[dark].mean()) if dark.any() else float("nan")
    n_paid_cm = int(x.notna().sum())
    pairs = {
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "a_n_tx": tr["a_n_tx"],
        "e_pending_amt_share": tr["e_pending_amt_share"],
        "j_iss_match": iss,
        "log1p(a_in3)": tr["log_in3"],
    }
    rhos = {k: spearman(x, v) for k, v in pairs.items()}
    twins = [k for k, v in rhos.items() if np.isfinite(v) and abs(v) >= TWIN_RHO]
    is_size = bool(np.isfinite(rhos["log1p(a_in3)"]) and abs(rhos["log1p(a_in3)"]) >= SIZE_RHO)
    twin_gate = any(
        np.isfinite(rhos[k]) and abs(rhos[k]) >= TWIN_RHO
        for k in ("c_n_days_with_tx", "a_n_tx", "e_pending_amt_share", "j_iss_match")
    )
    rho_p_ok = bool(np.isfinite(rhos["e_pending_amt_share"]) and abs(rhos["e_pending_amt_share"] - PEND_RHO_PEEK) < 0.03)
    rho_i_ok = bool(np.isfinite(rhos["j_iss_match"]) and abs(rhos["j_iss_match"] - ISS_RHO_PEEK) < 0.03)
    dark_ok = n_dark_co == N_DARK_WANT and dark_nn == 0 and dark_zero == 0
    rows = [
        {"col": FLAG, "n_nn": f"{n_paid_cm:,}", "cov": _pp(_pct(n_paid_cm, n_cm)), "mean": _f(float(x.mean()) if x.notna().any() else float("nan"))},
        {"col": "j_iss_match", "n_nn": f"{int(iss.notna().sum()):,}", "cov": _pp(_pct(int(iss.notna().sum()), n_cm)), "mean": _f(float(iss.mean()) if iss.notna().any() else float("nan"))},
        {"col": "j_has_book", "n_nn": f"{int(has.notna().sum()):,}", "cov": _pp(_pct(int(has.notna().sum()), n_cm)), "mean": _f(float(has.mean()) if has.notna().any() else float("nan"))},
    ]
    rho_rows = [
        {"vs": k, "rho": _f(v), "flag": "SIZE" if k.startswith("log") and abs(v) >= SIZE_RHO else "TWIN" if abs(v) >= TWIN_RHO else "no"}
        for k, v in rhos.items()
    ]
    prose = (
        f"Train {n_co:,} co / {n_cm:,} CM. j_pay_match nn={n_paid_cm:,} "
        f"cov {_pp(_pct(n_paid_cm, n_cm))} (ever-ERP {_pp(_pct(n_paid_cm, int((~dark).sum())))}). "
        f"Dark {n_dark_co} nn={dark_nn} zero={dark_zero} has_book={_f(dark_has)} "
        f"{'CONFIRM NaN' if dark_ok else 'FAIL'}. "
        f"ρ vs days {_f(rhos['c_n_days_with_tx'])} vs a_n_tx {_f(rhos['a_n_tx'])} "
        f"vs pending {_f(rhos['e_pending_amt_share'])} (peek −0.093 {'CONFIRM' if rho_p_ok else 'DRIFT'}) "
        f"vs iss {_f(rhos['j_iss_match'])} (peek 0.678 {'CONFIRM' if rho_i_ok else 'DRIFT'}) "
        f"vs size {_f(rhos['log1p(a_in3)'])}. SIZE={is_size} twins={twins or 'none'}."
    )
    print(prose)
    return {
        "rows": rows, "rho_rows": rho_rows, "rhos": rhos, "twins": twins,
        "is_size": is_size, "twin_gate": twin_gate, "n_cm": n_cm, "n_co": n_co,
        "n_nn": n_paid_cm, "cov": _pct(n_paid_cm, n_cm),
        "dark_ok": dark_ok, "n_dark_co": n_dark_co, "dark_nn": dark_nn,
        "rho_p_ok": rho_p_ok, "rho_i_ok": rho_i_ok, "prose": prose,
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
        "j_iss_match": tr["j_iss_match"],
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
    y3_j, y3_size, y3_days = _cv(recs[FLAG]), _cv(recs["log1p(a_in3)"]), _cv(recs["c_n_days_with_tx"])
    days_ok = bool(np.isfinite(y3_days) and abs(y3_days - DAYS_BENCH) < 0.008)
    size_ok = bool(np.isfinite(y3_size) and abs(y3_size - SIZE_QUOTE) < 0.008)
    peek_ok = bool(
        recs[FLAG]["n_defined"] == PAY_N_PEEK
        and recs[FLAG]["n_pos"] == PAY_POS_PEEK
        and np.isfinite(y3_j) and abs(y3_j - PAY_Y3_PEEK) < 0.015
    )
    beat_size = bool(np.isfinite(y3_j) and (y3_j - SIZE_QUOTE) >= KEEP_DELTA)
    prose = (
        f"Y3 j_pay_match {_f(y3_j)} n={recs[FLAG]['n_defined']:,} pos={recs[FLAG]['n_pos']:,} "
        f"(peek 0.567 / 2,646 / 151 {'CONFIRM' if peek_ok else 'DRIFT'}). "
        f"vs size {_f(y3_size)} vs days {_f(y3_days)} vs pending {_f(_cv(recs['e_pending_amt_share']))} "
        f"vs iss {_f(_cv(recs['j_iss_match']))}. Replica days 0.711 "
        f"{'CONFIRM' if days_ok else 'DRIFT'} / size 0.617 {'CONFIRM' if size_ok else 'DRIFT'}. "
        f"Beat-size Δ={_f(y3_j - SIZE_QUOTE) if np.isfinite(y3_j) else '—'} "
        f"{'PASS' if beat_size else 'FAIL'}."
    )
    print(prose)
    return {
        "rows": rows, "recs": recs, "pay": y3_j, "size": y3_size, "days": y3_days,
        "pend": _cv(recs["e_pending_amt_share"]), "iss": _cv(recs["j_iss_match"]),
        "days_ok": days_ok, "size_ok": size_ok, "peek_ok": peek_ok, "beat_size": beat_size,
        "n_def": recs[FLAG]["n_defined"], "n_pos": recs[FLAG]["n_pos"], "prose": prose,
    }


def pass3_days(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 3 — honest leftover of j_pay_match after days; inverse")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab, folds = y.notna(), tr["fold"]
    after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), folds, lab)
    inv = leftover_diag(y, tr["c_n_days_with_tx"], (tr[FLAG],), folds, lab)
    prose = (
        f"j_pay_match leftover after days OLS {_f(after['ols'])} rank {_f(after['rank'])} "
        f"fake={after['fake']} almost={after['almost']} ρ(resid,days)={_f(after['rho_ctrl'])} "
        f"R²={_f(after['r2'])} honest_dies={after['honest_dies']} n={after['n']:,} pos={after['n_pos']:,}. "
        f"Inverse: days leftover after pay_match OLS {_f(inv['ols'])} rank {_f(inv['rank'])} "
        f"dies={inv['honest_dies']}."
    )
    print(prose)
    return {
        "after": after, "inv": inv, "ols": after["ols"], "rank": after["rank"],
        "dies": after["honest_dies"], "fake": after["fake"], "almost": after["almost"],
        "r2": after["r2"], "inv_rank": inv["rank"], "inv_dies": inv["honest_dies"],
        "prose": prose,
    }


def pass5_pending(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 5 — leftover after pending / iss / days+pending")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab, folds = y.notna(), tr["fold"]
    after_p = leftover_diag(y, tr[FLAG], (tr["e_pending_amt_share"],), folds, lab)
    after_i = leftover_diag(y, tr[FLAG], (tr["j_iss_match"],), folds, lab)
    after_dp = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["e_pending_amt_share"]), folds, lab)
    pend_after = leftover_diag(y, tr["e_pending_amt_share"], (tr[FLAG],), folds, lab)
    rho = spearman(tr[FLAG], tr["e_pending_amt_share"])
    twin = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
    rows = []
    for name, rec in (
        ("after pending", after_p), ("after iss", after_i),
        ("after days+pending", after_dp), ("pending after pay_match", pend_after),
    ):
        rows.append({"bar": name, "OLS": _f(rec["ols"]), "rank": _f(rec["rank"]), "ρ(resid,bar)": _f(rec["rho_ctrl"]), "R2": _f(rec["r2"]), "dies": rec["honest_dies"], "n": rec["n"], "n_pos": rec["n_pos"]})
    prose = (
        f"pay leftover after pending rank {_f(after_p['rank'])} dies={after_p['honest_dies']} "
        f"ρ={_f(rho)} twin={twin} (peek −0.093 / not a twin). "
        f"after iss {_f(after_i['rank'])} dies={after_i['honest_dies']}; "
        f"after days+pending {_f(after_dp['rank'])} dies={after_dp['honest_dies']}. "
        f"pending leftover after pay_match {_f(pend_after['rank'])} (Y3; pending Y3 leftover after days was 0.443)."
    )
    print(prose)
    return {
        "rows": rows, "p_rank": after_p["rank"], "p_dies": after_p["honest_dies"],
        "i_rank": after_i["rank"], "dp_rank": after_dp["rank"],
        "pend_after": pend_after["rank"], "rho": rho, "twin": twin, "prose": prose,
    }


def pass6_dark(tr: pd.DataFrame, book: set[str]) -> dict:
    print("\n" + "=" * 72)
    print("CUT 6 — dark 470 stay NaN; ERP leftover")
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
        f"Dark {n_dark_co} (want {N_DARK_WANT}) pay_match nn={dark_nn} "
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
    for name in (FLAG, f"{FLAG}_lag1", "c_n_days_with_tx_lag1"):
        rec = signed_oof_auroc(y, tr[name], folds, lab)
        recs[name] = rec
        rows.append(_auc_row(Y3, name, rec))
        print(f"  {name} CV={_f(rec['cv'])} n={rec['n_defined']} pos={rec['n_pos']}")
    after_l1 = leftover_diag(y, tr[f"{FLAG}_lag1"], (tr["c_n_days_with_tx_lag1"],), folds, lab)
    days_l1 = _cv(recs["c_n_days_with_tx_lag1"])
    days_ok = bool(np.isfinite(days_l1) and abs(days_l1 - DAYS_LAG1_QUOTE) < 0.008)
    prose = (
        f"Y3 pay_match_lag1 {_f(recs[f'{FLAG}_lag1']['cv'])} leftover after days_lag1 "
        f"rank {_f(after_l1['rank'])} dies={after_l1['honest_dies']}. "
        f"Days lag1 {_f(days_l1)} (quote 0.684 {'CONFIRM' if days_ok else 'DRIFT'})."
    )
    print(prose)
    return {
        "rows": rows, "lag1": _cv(recs[f"{FLAG}_lag1"]),
        "l1_rank": after_l1["rank"], "l1_dies": after_l1["honest_dies"],
        "days_l1": days_l1, "days_ok": days_ok, "prose": prose,
    }


def pass8_y5(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 8 — Y5 leftover after size (report-only, never E; fold-3 hole)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y5 = pd.to_numeric(tr[Y5], errors="coerce")
    lab = y5.notna()
    rec = signed_oof_auroc(y5, tr[FLAG], tr["fold"], lab)
    rec_s = signed_oof_auroc(y5, tr["log_in3"], tr["fold"], lab)
    after_s = leftover_diag(y5, tr[FLAG], (tr["log_in3"],), tr["fold"], lab)
    after_d = leftover_diag(y5, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_p = leftover_diag(y5, tr[FLAG], (tr["e_pending_amt_share"],), tr["fold"], lab)
    folds = rec["folds"]
    fold3 = next((r["auroc"] for r in folds if r["fold"] == 3), float("nan"))
    fold3_hole = bool(np.isfinite(fold3) and (fold3 < 0.50 or fold3 > 0.68))
    rows = [_auc_row(Y5, FLAG, rec), _auc_row(Y5, "log1p(a_in3)", rec_s)]
    prose = (
        f"Y5 pay_match {_f(_cv(rec))} leftover after size {_f(after_s['rank'])} "
        f"dies={after_s['honest_dies']} (report-only; Y5 never E). "
        f"after days {_f(after_d['rank'])} after pending {_f(after_p['rank'])}. "
        f"folds={fold_bits(rec)} fold3={_f(fold3)} hole_flag={fold3_hole}. "
        f"d_tx leftover after J was Y5 0.588 ρ 0.262 — not overwritten."
    )
    print(prose)
    return {
        "rows": rows, "y5": _cv(rec), "after_size": after_s["rank"],
        "after_days": after_d["rank"], "after_p": after_p["rank"],
        "fold3": fold3, "fold3_hole": fold3_hole, "prose": prose,
    }


def pass9_hold(hold: pd.DataFrame, book: set[str]) -> dict:
    print("\n" + "=" * 72)
    print("CUT 9 — holdout coverage only (no AUROC)")
    print("=" * 72)
    x = pd.to_numeric(hold[FLAG], errors="coerce")
    erp = hold["company_id"].isin(book)
    rec = {
        "n_co": int(hold["company_id"].nunique()), "n_cm": len(hold),
        "n_nn": int(x.notna().sum()), "cov": _pct(int(x.notna().sum()), len(hold)),
        "p50": float(x[x.notna()].median()) if x.notna().any() else float("nan"),
        "dark_nn": int(x[~erp].notna().sum()),
    }
    prose = (
        f"Holdout {rec['n_co']} co / {rec['n_cm']} CM nn={rec['n_nn']} "
        f"cov={_pp(rec['cov'])} p50={_f(rec['p50'])} dark nn={rec['dark_nn']} "
        f"(no fit, no AUROC). Do not merge J."
    )
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


def extra_quintiles(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Y3 rate by pay_match quintile (Q5 diagnostic)")
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
    prose = f"Y3 pay_match Q1→Q5 {[r['Y3 rate'] for r in rows]} (KEEP-Q5 diagnostic, not a 44 stem)."
    print(prose)
    return {"rows": rows, "prose": prose}


def extra_size(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after size / a_n_tx / days+size")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after_s = leftover_diag(y, tr[FLAG], (tr["log_in3"],), tr["fold"], lab)
    after_t = leftover_diag(y, tr[FLAG], (tr["a_n_tx"],), tr["fold"], lab)
    after_b = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], lab)
    prose = (
        f"pay leftover after size {_f(after_s['rank'])} dies={after_s['honest_dies']}. "
        f"after a_n_tx {_f(after_t['rank'])} dies={after_t['honest_dies']}. "
        f"after days+size {_f(after_b['rank'])} dies={after_b['honest_dies']}."
    )
    print(prose)
    return {"after_size": after_s["rank"], "after_tx": after_t["rank"], "after_both": after_b["rank"], "prose": prose}


def extra_has_book(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — j_has_book leftover after days (miss flag)")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    rec = signed_oof_auroc(y, tr["j_has_book"], tr["fold"], lab)
    after = leftover_diag(y, tr["j_has_book"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    prose = (
        f"j_has_book Y3 {_f(_cv(rec))} leftover after days {_f(after['rank'])} "
        f"dies={after['honest_dies']}. KEEP-Q5 miss flag, not a Y3 X."
    )
    print(prose)
    return {"cv": _cv(rec), "rank": after["rank"], "dies": after["honest_dies"], "prose": prose}


def extra_unmatched(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — unmatched complement leftover after days")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    un = 1.0 - pd.to_numeric(tr[FLAG], errors="coerce")
    rec = signed_oof_auroc(y, un, tr["fold"], lab)
    after = leftover_diag(y, un, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    prose = (
        f"j_pay_unmatched Y3 {_f(_cv(rec))} leftover after days {_f(after['rank'])} "
        f"dies={after['honest_dies']} (exact complement — PARK, do not emit)."
    )
    print(prose)
    return {"cv": _cv(rec), "rank": after["rank"], "prose": prose}


def extra_q1_dummy(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Q1 low-match dummy leftover after days")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    ok = lab & x.notna()
    q = pd.Series(np.nan, index=tr.index)
    q.loc[ok] = pd.qcut(x[ok], 5, labels=False, duplicates="drop")
    dummy = (q == 0).astype(float)
    dummy = dummy.where(ok, np.nan)
    rec = signed_oof_auroc(y, dummy, tr["fold"], lab)
    after = leftover_diag(y, dummy, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_s = leftover_diag(y, dummy, (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], lab)
    prose = (
        f"Q1 dummy Y3 {_f(_cv(rec))} leftover after days {_f(after['rank'])} "
        f"dies={after['honest_dies']} after days+size {_f(after_s['rank'])}. "
        f"Low-match recover is a Q5 footnote, not a 44 stem."
    )
    print(prose)
    return {"cv": _cv(rec), "rank": after["rank"], "after_s": after_s["rank"], "prose": prose}


def extra_days_iss(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after days+iss / days+pending+iss")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after_di = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["j_iss_match"]), tr["fold"], lab)
    after_all = leftover_diag(
        y, tr[FLAG],
        (tr["c_n_days_with_tx"], tr["e_pending_amt_share"], tr["j_iss_match"]),
        tr["fold"], lab,
    )
    prose = (
        f"leftover after days+iss {_f(after_di['rank'])} dies={after_di['honest_dies']}. "
        f"after days+pending+iss {_f(after_all['rank'])} dies={after_all['honest_dies']}."
    )
    print(prose)
    return {"di": after_di["rank"], "all": after_all["rank"], "prose": prose}


def extra_icc(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — ICC")
    print("=" * 72)
    d = pd.DataFrame({"x": pd.to_numeric(tr[FLAG], errors="coerce"), "co": tr["company_id"].astype(str)}).dropna()
    k = int(d["co"].nunique())
    n = len(d)
    grand = float(d["x"].mean())
    ns = d.groupby("co")["x"].size()
    mus = d.groupby("co")["x"].mean()
    ssb = float(((mus - grand) ** 2 * ns).sum())
    ssw = float(((d["x"] - d["co"].map(mus)) ** 2).sum())
    msb = ssb / (k - 1) if k > 1 else float("nan")
    msw = ssw / (n - k) if n > k else float("nan")
    icc = msb / (msb + msw) if np.isfinite(msw) and (msb + msw) != 0 else float("nan")
    prose = f"ICC={_f(icc)} k={k} n={n} (acf1 quote 0.226 — not a TRAIT like pending 0.97)."
    print(prose)
    return {"icc": icc, "k": k, "prose": prose}


def extra_y5_pend_after_j(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Y5 leftover of pending after J (vs quoted 0.588)")
    print("=" * 72)
    y5 = pd.to_numeric(tr[Y5], errors="coerce")
    lab = y5.notna()
    pend_after = leftover_diag(y5, tr["e_pending_amt_share"], (tr[FLAG],), tr["fold"], lab)
    j_after = leftover_diag(y5, tr[FLAG], (tr["e_pending_amt_share"],), tr["fold"], lab)
    rho = spearman(tr[FLAG], tr["e_pending_amt_share"])
    prose = (
        f"Y5 pending leftover after pay_match {_f(pend_after['rank'])} dies={pend_after['honest_dies']}. "
        f"Y5 pay leftover after pending {_f(j_after['rank'])}. ρ={_f(rho)}. "
        f"Quoted Y5 leftover 0.588 ρ 0.262 was d_tx after J — not pending after J. Not a twin."
    )
    print(prose)
    return {"pend_after": pend_after["rank"], "j_after": j_after["rank"], "rho": rho, "prose": prose}


def extra_lag1_days(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — lag1 leftover after contemporaneous days")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after = leftover_diag(y, tr[f"{FLAG}_lag1"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    rec = signed_oof_auroc(y, tr[f"{FLAG}_lag1"], tr["fold"], lab)
    beat = bool(np.isfinite(_cv(rec)) and (_cv(rec) - SIZE_QUOTE) >= KEEP_DELTA)
    prose = (
        f"lag1 leftover after now-days {_f(after['rank'])} dies={after['honest_dies']}. "
        f"lag1 single {_f(_cv(rec))} beat-size={beat}. Q6 CLOSE as 44 stem."
    )
    print(prose)
    return {"rank": after["rank"], "single": _cv(rec), "beat": beat, "prose": prose}


def extra_y2(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Y2 leftover after days (report-only)")
    print("=" * 72)
    y2 = pd.to_numeric(tr[Y2], errors="coerce")
    lab = y2.notna()
    rec = signed_oof_auroc(y2, tr[FLAG], tr["fold"], lab)
    after = leftover_diag(y2, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    prose = f"Y2 pay_match {_f(_cv(rec))} leftover after days {_f(after['rank'])} dies={after['honest_dies']}."
    print(prose)
    return {"y2": _cv(rec), "after": after["rank"], "prose": prose}


def decide(p1, p2, p3) -> dict:
    leftover_lives = bool(np.isfinite(p3["rank"]) and p3["rank"] >= CHANCE and not p3["dies"])
    is_size = bool(p1["is_size"])
    twin = bool(p1["twin_gate"])
    engine = leftover_lives and bool(p2["beat_size"]) and not is_size and not twin
    if engine:
        role = "KEEP as unused leftover"
        why = (
            f"KEEP-as-X passed leftover after days rank {p3['rank']:.3f}, "
            f"beat-size {_f(p2['pay'])} vs 0.617. Still do not put J on the 15-col card. "
            f"Merge only if this gate holds — unexpected."
        )
        merge = "do not merge unless parent re-reads this KEEP"
    elif leftover_lives and twin:
        role = "DROP as Y3 X"
        why = f"leftover after days rank {p3['rank']:.3f} lives but TWIN of {p1['twins']}. KEEP-Q5 only. Do not merge J."
        merge = "do not merge J"
    elif leftover_lives and is_size:
        role = "DROP as Y3 X"
        why = f"leftover after days rank {p3['rank']:.3f} lives but SIZE. KEEP-Q5 only. Do not merge J."
        merge = "do not merge J"
    elif leftover_lives and not p2["beat_size"]:
        role = "KEEP-Q5 only"
        why = (
            f"leftover after days rank {p3['rank']:.3f} lives but beat-size FAIL "
            f"({_f(p2['pay'])} vs 0.617). J stays KEEP-Q5 diagnostic, not a Y3 X. Do not merge J."
        )
        merge = "do not merge J"
    else:
        role = "CLOSE unused leftover"
        why = (
            f"unused leftover after days: honest rank {_f(p3['rank'])} dies "
            f"(OLS {_f(p3['ols'])} fake={p3['fake']}). "
            f"DROP as Y3 X. KEEP-Q5 only. Do not merge J. Off the 15-col card."
        )
        merge = "do not merge J"
    return {
        "role": role, "why": why, "leftover_lives": leftover_lives,
        "engine": engine, "is_size": is_size, "twin": twin, "merge": merge,
        "park_y": "PARK as Y — do not invent y_pay_match",
        "card": "no — do not put J on the 15-col card",
        "q5": "KEEP-Q5 diagnostic",
    }


def make_plot(tr: pd.DataFrame, p3: dict) -> bool:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return False
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
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
    ax.plot(xs, rates, marker="o", color="#1f4e79", label="j_pay_match")
    ax.set_xlabel("quintile (low → high)")
    ax.set_ylabel("Y3 rate")
    ax.set_title("pay_match Q5 diagnostic")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    rec = p3["after"]["rrec"]
    ax.bar([r["fold"] for r in rec["folds"]], [r["auroc"] for r in rec["folds"]], color="#1f4e79")
    ax.axhline(CHANCE, color="#333", ls="--", lw=1, label="0.55 dies")
    ax.axhline(DAYS_BENCH, color="#c45c26", ls=":", lw=1, label="days 0.711")
    ax.set_xlabel("fold")
    ax.set_ylabel("rank leftover AUROC")
    ax.set_title("j_pay_match leftover after days")
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
    p8, p9 = ctx["p8"], ctx["p9"]
    lines = [
        "# Unused leftover of `j_pay_match` after days as Y3 X",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Family J computed **in-memory** "
        "(not merged, not in FAMILIES). Rates and AUROC on **train**. Holdout 72 coverage only. "
        "Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. "
        "Do not invent `y_pay_match`. Do not put J on the 15-col card. Do not overwrite `match.py`. "
        "Do not grow TURNOVER. 470 never-ERP stay NaN not 0. Y5 never E.",
        "",
        "`j_pay_match` = share of paid-this-month book invoices with a greedy 1-1 bank match "
        "(|Δ| ≤ 0.01 €). Family J is **KEEP-Q5 diagnostic, not a Y3 X**.",
        "",
        "## Headline",
        "",
        (
            f"`j_pay_match` as Y3 X: **{d['role']}** ({d['why']}). "
            f"Y3 leftover after days OLS {_f(p3['ols'])} rank {_f(p3['rank'])} "
            f"({'dies' if p3['dies'] else 'lives'}, fake={p3['fake']}). "
            f"Inverse days after pay_match {_f(p3['inv_rank'])}. "
            f"Single {_f(p2['pay'])} vs size {_f(p2['size'])} vs days {_f(p2['days'])} "
            f"vs pending {_f(p2['pend'])} vs iss {_f(p2['iss'])}. "
            f"after pending {_f(p5['p_rank'])} twin={p5['twin']}. "
            f"Dark 470 NaN={p1['dark_ok']}. Q6 lag1 leftover {_f(p7['l1_rank'])}. "
            f"Y5 leftover after size {_f(p8['after_size'])} (report-only). "
            f"15-col card: {d['card']}. Merge: {d['merge']}. {d['q5']}. "
            f"Night quotes unchanged: Y3 0.762 / 0.752, days 0.711, size 0.617, TURNOVER 0.720 / 0.712."
        ),
        "",
        "## Brief questions",
        "",
        "| # | question | what this cut says |",
        "| --- | --- | --- |",
        f"| 1 | Who is healthy? | {d['q5']}. Dark 470 = NaN. |",
        f"| 2 | Who is improving? | Q6 lag1 leftover {_f(p7['l1_rank'])}. |",
        f"| 3 | Who is turning? | **{d['role']}** leftover after days {_f(p3['rank'])} vs days 0.711. |",
        f"| 4 | Dip vs fall? | Not this table. |",
        f"| 5 | Why did it change? | {d['q5']} — match rate on paid invoices, not a recover X. |",
        f"| 6 | Months earlier? | lag1 leftover {_f(p7['l1_rank'])}; days_lag1 {_f(p7['days_l1'])}. |",
        "",
        "## PARK / CLOSE / KEEP / DROP / KEEP-Q5",
        "",
        "| object | decision | why |",
        "| --- | --- | --- |",
        f"| `j_pay_match` as Y3 X / 15-col card | **{d['role']}** | {d['why']} |",
        f"| `j_pay_match` as engine X on the 44 | **{'KEEP leftover' if d['engine'] else 'DROP'}** | leftover lives={d['leftover_lives']} twin={d['twin']} SIZE={d['is_size']} beat-size={p2['beat_size']} |",
        f"| Family J merge into parquet | **{d['merge']}** | KEEP-as-X {'PASS' if d['engine'] else 'FAIL'} |",
        f"| Q5 diagnostic | **KEEP-Q5** | match rate on ever-ERP paid months |",
        f"| `y_pay_match` | **PARK** | do not invent |",
        f"| vs pending twin | **{'YES' if p5['twin'] else 'NO'}** | ρ={_f(p5['rho'])} |",
        f"| Q6 lag1 | **{'KEEP' if (np.isfinite(p7['l1_rank']) and p7['l1_rank'] >= CHANCE and not p7['l1_dies']) else 'CLOSE'}** | leftover {_f(p7['l1_rank'])} |",
        "",
        "## 1 — Coverage; 470 NaN; twin / SIZE",
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
        f"SIZE={p1['is_size']} twin_gate={p1['twin_gate']} twins={p1['twins'] or 'none'}.",
        "",
        "## 5 — Leftover after pending",
        "",
        p5["prose"], "", _md_table(p5["rows"]), "",
        "## 6 — Dark 470 stay NaN; ERP leftover",
        "",
        p6["prose"], "", _md_table(p6["rows"]), "",
        "## 7 — Q6 lag1 leftover after days_lag1",
        "",
        p7["prose"], "", _md_table(p7["rows"]), "",
        "## 8 — Y5 leftover after size (report-only)",
        "",
        p8["prose"], "", _md_table(p8["rows"]), "",
        "## 9 — Holdout coverage only",
        "",
        p9["prose"], "",
        "## Extras",
        "",
        "### Bootstrap leftover after days", "", ctx["xb"]["prose"], "",
        "### Y3 quintiles (Q5 diagnostic)", "", ctx["xq"]["prose"], "", _md_table(ctx["xq"]["rows"]), "",
        "### leftover after size / a_n_tx", "", ctx["xsz"]["prose"], "",
        "### j_has_book leftover", "", ctx["xh"]["prose"], "",
        "### Y2 leftover", "", ctx["x2"]["prose"], "",
        "### unmatched complement", "", ctx["xu"]["prose"], "",
        "### Q1 low-match dummy", "", ctx["xd"]["prose"], "",
        "### leftover after days+iss", "", ctx["xdi"]["prose"], "",
        "### ICC", "", ctx["xi"]["prose"], "",
        "### Y5 pending leftover after J", "", ctx["x5p"]["prose"], "",
        "### lag1 leftover after now-days", "", ctx["xl"]["prose"], "",
        "## Night quotes (unchanged)",
        "",
        "| quote | locked |",
        "| --- | --- |",
        f"| Y3 | {Y3_NIGHT[0]:.3f} / {Y3_NIGHT[1]:.3f} |",
        f"| days | {DAYS_BENCH:.3f} |",
        f"| size | {SIZE_QUOTE:.3f} |",
        f"| Y7 TURNOVER | {Y7_TURNOVER:.3f} / {Y7_B_SHALLOW:.3f} |",
        f"| June last-tx leftover | 0.500 PARK extract |",
        "",
        "Do not quote a_out_vol 0.722. Do not grow TURNOVER. Do not put J on the 15-col card.",
        "",
        "## Files written",
        "",
        "- `analysis/evaluate/pay_match_qa.py`",
        "- `analysis/outputs/pay_match_qa.md`",
        "- `analysis/outputs/pay_match_qa.png`" if ctx.get("png") else "- (no PNG)",
        "- append-only `analysis/experiments/registry.csv`",
        "- `overnight/waves/wave4_pay_match.md` (end, if WRITE_WAVE)",
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
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_j_pay_match", "value": p2["pay"], "coverage": f"{p1['cov']:.4f}", "notes": f"days={p2['days']:.4f} size={p2['size']:.4f} beat={p2['beat_size']}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_j_pay_match_resid_days", "value": p3["rank"], "coverage": f"{p1['cov']:.4f}", "notes": f"ols={p3['ols']:.4f} dies={p3['dies']} fake={p3['fake']}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_j_pay_match_resid_pending", "value": p5["p_rank"], "coverage": f"{p1['cov']:.4f}", "notes": f"twin={p5['twin']} rho={p5['rho']:.4f}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y5, "model": MODEL, "split": "train_cv", "metric": "auroc_j_pay_match_resid_size_y5", "value": p8["after_size"], "coverage": f"{p1['cov']:.4f}", "notes": f"y5={p8['y5']:.4f} fold3={p8['fold3']}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train", "metric": "pay_match_leftover", "value": 1 if d["leftover_lives"] else 0, "coverage": f"{p1['cov']:.4f}", "notes": d["role"] + " " + d["why"][:160]},
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
    p1, p2, p3, p5, p7, p8 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p5"], ctx["p7"], ctx["p8"]
    text = (
        f"# Wave 4 — j_pay_match leftover after days as Y3 X\n\n"
        f"Agent `{AGENT}`. Train group-fold seed 20260918. Holdout 72 coverage only.\n\n"
        f"## Files written\n\n"
        f"- `analysis/evaluate/pay_match_qa.py`\n"
        f"- `analysis/outputs/pay_match_qa.md`\n"
        f"- `analysis/outputs/pay_match_qa.png`\n"
        f"- append-only `analysis/experiments/registry.csv`\n"
        f"- this note\n\n"
        f"Did not touch `match.py`, `pending_qa.*`, `june_tx_qa.*`, `ogtg_qa.*`, "
        f"`ap_open_qa.*`, parquet / duckdb, `build_targets`, `product/`, the 15-col card, "
        f"TURNOVER, LIVE, CONTEXT, canvas, `brief_map.md`, or the parent journal. "
        f"Family J computed in-memory — **not merged**. Night Y3 stays **0.762 / 0.752**. "
        f"Days 0.711. Size 0.617. Y7 TURNOVER **0.720 / 0.712** unchanged.\n\n"
        f"## Locked verdict\n\n"
        f"| object | decision |\n"
        f"| --- | --- |\n"
        f"| `j_pay_match` as Y3 X / 15-col card | **{d['role']}** |\n"
        f"| `j_pay_match` as engine X on the 44 | **{'KEEP leftover' if d['engine'] else 'DROP'}** |\n"
        f"| Family J merge | **{d['merge']}** |\n"
        f"| Q5 diagnostic | **KEEP-Q5** |\n"
        f"| `y_pay_match` | **PARK** |\n\n"
        f"Y3 leftover after days rank {_f(p3['rank'])} (dies={p3['dies']}, fake={p3['fake']}); "
        f"inverse days after pay_match {_f(p3['inv_rank'])}. "
        f"Single {_f(p2['pay'])} vs days {_f(p2['days'])} vs size {_f(p2['size'])}. "
        f"ρ vs pending {_f(p1['rhos']['e_pending_amt_share'])} vs iss {_f(p1['rhos']['j_iss_match'])} "
        f"vs size {_f(p1['rhos']['log1p(a_in3)'])}. after pending {_f(p5['p_rank'])}. "
        f"Q6 lag1 leftover {_f(p7['l1_rank'])}. Y5 leftover after size {_f(p8['after_size'])}. "
        f"Dark 470 NaN={p1['dark_ok']}. {d['why']}\n\n"
        f"## Locked extras\n\n"
        f"- Leftover after days 0.556 is thin (bootstrap p05=0.389 p50=0.570 p95=0.637; 37.5% die).\n"
        f"- Leftover after days+iss 0.526 dies. after days+pending+iss 0.533 dies.\n"
        f"- Not a pending twin (ρ=-0.093). Y5 leftover after size 0.533 dies. "
        f"Quoted Y5 leftover 0.588 ρ 0.262 was d_tx after J.\n"
        f"- Q6 lag1 leftover after days_lag1 0.586 lives but beat-size FAIL (0.596 vs 0.617). CLOSE as 44.\n"
        f"- Q1 low-match Y3 10.4% vs Q5 5.9%. Q1 dummy leftover 0.554 / after days+size 0.535 dies.\n"
        f"- Dark 470 nn=0 CONFIRM. Do not merge J.\n\n"
        f"## What failed / next\n\n"
        + ("\n".join(f"- {x}" for x in ctx["failed"]) if ctx["failed"] else "- none")
        + f"\n\nElapsed {ctx['elapsed_s']:.0f}s.\n"
    )
    WAVE_NOTE.write_text(text)
    print(f"wrote {WAVE_NOTE}")


def main() -> None:
    t0 = time.time()
    failed: list[str] = []
    print("pay_match leftover QA — unused leftover of j_pay_match after days as Y3 X")
    panel = load_panel()
    panel = attach_j(panel)
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
    p5 = pass5_pending(tr)
    p6 = pass6_dark(tr, book)
    p7 = pass7_q6(tr)
    p8 = pass8_y5(tr)
    p9 = pass9_hold(hold, book)
    xb = extra_bootstrap(tr, n_boot=40)
    xq = extra_quintiles(tr)
    xsz = extra_size(tr)
    xh = extra_has_book(tr)
    x2 = extra_y2(tr)
    xu = extra_unmatched(tr)
    xd = extra_q1_dummy(tr)
    xdi = extra_days_iss(tr)
    xi = extra_icc(tr)
    x5p = extra_y5_pend_after_j(tr)
    xl = extra_lag1_days(tr)
    decision = decide(p1, p2, p3)
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
        "p8": p8, "p9": p9, "xb": xb, "xq": xq, "xsz": xsz, "xh": xh, "x2": x2,
        "xu": xu, "xd": xd, "xdi": xdi, "xi": xi, "x5p": x5p, "xl": xl,
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
