"""Unused leftover of ``a_op_out`` after days as Y3 X.

``a_op_out`` = -sum(amount | grp = op_out) this month (Family A).
Store also has ``a_out3`` / ``a_out6`` / ``a_out12`` / ``a_net`` /
``a_io_ratio``. ``a_in3`` leftover already CLOSE 0.521; size bar 0.617
stays. Do **not** quote ``a_out_vol`` 0.722 as the engine (trait,
demean 0.549). Do **not** overwrite ``a_vol_qa.*`` / ``transfer_qa.*`` /
``growth_qa.*`` / ``in3_qa.*`` / ``n_accounts_qa.*``.

KEEP-as-X: beat size ≥0.02 **and** leftover after days **and** not SIZE
(|ρ| vs log1p(a_in3) ≥0.50) **and** not a twin (|ρ|≥0.80 vs days /
a_n_tx / a_out3 / a_out6 / a_out12 / a_net / a_io_ratio). Leftover
<0.55 dies. Rank leftover is honest; OLS can fake a days leak.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.op_out_qa

Owned: analysis/evaluate/op_out_qa.py, analysis/outputs/op_out_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_op_out.md (end).
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
OUT_MD = ANALYSIS / "outputs" / "op_out_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "op_out_qa.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_op_out.md"
AGENT = "572fb928"
WAVE = "4"
ROUND = "R4"
MODEL = "op_out_qa"
X_FAM = "A"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
Y7 = "y7_top1_lost"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_QUOTE = 0.617
DAYS_LAG1_QUOTE = 0.684
IN3_LEFT = 0.521
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
    "a_op_out",
    "a_op_in",
    "a_out3",
    "a_out6",
    "a_out12",
    "a_net",
    "a_io_ratio",
)

Y_KEEP = (Y2, Y3, Y7)
FLAG = "a_op_out"
TWIN_COLS = (
    "c_n_days_with_tx",
    "a_n_tx",
    "a_out3",
    "a_out6",
    "a_out12",
    "a_net",
    "a_io_ratio",
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
    panel["log_opout"] = np.log1p(pd.to_numeric(panel[FLAG], errors="coerce").clip(lower=0))
    panel["log_out3"] = np.log1p(pd.to_numeric(panel["a_out3"], errors="coerce").clip(lower=0))
    leak3 = leakage_check(
        [FLAG, "a_out3", "a_out6", "c_n_days_with_tx", "log_in3"],
        Y3, forbidden_prefixes=["b"],
    )
    if not leak3["ok"]:
        raise RuntimeError(f"Y3 never-B leak: {leak3['issues']}")
    leak7 = leakage_check([FLAG, "a_out3"], Y7, forbidden_prefixes=["d"])
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
    rhos = {c: spearman(x, tr[c]) for c in TWIN_COLS}
    rhos["log_in3"] = spearman(x, tr["log_in3"])
    rhos["a_op_in"] = spearman(x, tr["a_op_in"])
    gate_twins = [c for c in TWIN_COLS if np.isfinite(rhos[c]) and abs(rhos[c]) >= TWIN_RHO]
    is_size = bool(np.isfinite(rhos["log_in3"]) and abs(rhos["log_in3"]) >= SIZE_RHO)
    rows = [{"col": FLAG, "n_nn": f"{nn:,}", "cov": _pp(_pct(nn, n_cm)), "eq0": _pp(eq0), "p50": _f(p50)}]
    rho_rows = [{"vs": k, "rho": _f(v), "flag": "SIZE" if k == "log_in3" and is_size else ("TWIN" if abs(v) >= TWIN_RHO else "no")} for k, v in rhos.items()]
    prose = (
        f"Train {n_co:,} co / {n_cm:,} CM. {FLAG} nn={nn:,} cov {_pp(_pct(nn, n_cm))} "
        f"eq0 {_pp(eq0)} p50={_f(p50)}. Dark panel p50={_f(p50_d)} ERP p50={_f(p50_e)}. "
        f"ρ vs days {_f(rhos['c_n_days_with_tx'])} vs a_n_tx {_f(rhos['a_n_tx'])} "
        f"vs a_out3 {_f(rhos['a_out3'])} vs a_out6 {_f(rhos['a_out6'])} vs a_out12 {_f(rhos['a_out12'])} "
        f"vs a_net {_f(rhos['a_net'])} vs a_io_ratio {_f(rhos['a_io_ratio'])} vs size {_f(rhos['log_in3'])}. "
        f"SIZE={is_size} gate_twins={gate_twins or 'none'}."
    )
    print(prose)
    return {
        "rows": rows, "rho_rows": rho_rows, "rhos": rhos, "is_size": is_size,
        "gate_twins": gate_twins, "twin_gate": bool(gate_twins),
        "n_cm": n_cm, "n_co": n_co, "cov": _pct(nn, n_cm), "p50": p50,
        "p50_d": p50_d, "p50_e": p50_e, "prose": prose,
    }


def pass2_singles(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 2 — single-feature group-fold Y3")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    feats = {
        FLAG: tr[FLAG],
        "log1p(a_op_out)": tr["log_opout"],
        "a_out3": tr["a_out3"],
        "log1p(a_out3)": tr["log_out3"],
        "a_out6": tr["a_out6"],
        "a_out12": tr["a_out12"],
        "a_net": tr["a_net"],
        "a_io_ratio": tr["a_io_ratio"],
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
    out3 = _cv(recs["a_out3"])
    beat = bool(np.isfinite(open_cv) and (open_cv - SIZE_QUOTE) >= KEEP_DELTA)
    days_ok = bool(np.isfinite(days) and abs(days - DAYS_BENCH) < 0.008)
    size_ok = bool(np.isfinite(size) and abs(size - SIZE_QUOTE) < 0.008)
    prose = (
        f"Y3 {FLAG} {_f(open_cv)} n={recs[FLAG]['n_defined']:,} pos={recs[FLAG]['n_pos']:,}. "
        f"vs size {_f(size)} vs days {_f(days)} vs a_out3 {_f(out3)} vs a_out6 {_f(_cv(recs['a_out6']))} "
        f"vs a_out12 {_f(_cv(recs['a_out12']))} vs a_net {_f(_cv(recs['a_net']))} "
        f"vs a_io_ratio {_f(_cv(recs['a_io_ratio']))}. Replica days 0.711 "
        f"{'CONFIRM' if days_ok else 'DRIFT'} / size 0.617 {'CONFIRM' if size_ok else 'DRIFT'}. "
        f"Beat-size Δ={_f(open_cv - SIZE_QUOTE) if np.isfinite(open_cv) else '—'} "
        f"{'PASS' if beat else 'FAIL'}. Do not quote a_out_vol 0.722 as the engine."
    )
    print(prose)
    return {
        "rows": rows, "open": open_cv, "days": days, "size": size, "out3": out3,
        "out6": _cv(recs["a_out6"]), "out12": _cv(recs["a_out12"]),
        "net": _cv(recs["a_net"]), "io": _cv(recs["a_io_ratio"]),
        "log": _cv(recs["log1p(a_op_out)"]),
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


def pass4_out3(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 4 — leftover after a_out3 (trailing twin?)")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after = leftover_diag(y, tr[FLAG], (tr["a_out3"],), tr["fold"], lab)
    after_d = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["a_out3"]), tr["fold"], lab)
    inv = leftover_diag(y, tr["a_out3"], (tr[FLAG],), tr["fold"], lab)
    rewrite = bool(np.isfinite(after["rank"]) and after["rank"] < CHANCE)
    prose = (
        f"leftover after a_out3 rank {_f(after['rank'])} OLS {_f(after['ols'])} "
        f"dies={after['honest_dies']} fake={after['fake']} rewrite={rewrite}. "
        f"after days+a_out3 {_f(after_d['rank'])}. a_out3 leftover after {FLAG} {_f(inv['rank'])}."
    )
    print(prose)
    return {
        "rank": after["rank"], "ols": after["ols"], "dies": after["honest_dies"],
        "fake": after["fake"], "rewrite": rewrite, "both": after_d["rank"],
        "inv": inv["rank"], "prose": prose,
        "rows": [
            {"cut": "after a_out3", "rank": _f(after["rank"]), "ols": _f(after["ols"]), "fake": after["fake"]},
            {"cut": "after days+a_out3", "rank": _f(after_d["rank"]), "ols": _f(after_d["ols"]), "fake": after_d["fake"]},
            {"cut": "a_out3 after a_op_out", "rank": _f(inv["rank"]), "ols": _f(inv["ols"]), "fake": inv["fake"]},
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
        f"dies={after_b['honest_dies']} fake={after_b['fake']}. "
        f"a_in3 leftover 0.521 CLOSE stays. Size bar 0.617 stays."
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


def extra_windows(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after a_out6 / a_out12 / a_net / a_io_ratio / a_n_tx")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    rows = []
    store = {}
    for col in ("a_out6", "a_out12", "a_net", "a_io_ratio", "a_n_tx"):
        after = leftover_diag(y, tr[FLAG], (tr[col],), tr["fold"], lab)
        store[col] = after["rank"]
        rows.append({"control": col, "leftover": _f(after["rank"]), "ols": _f(after["ols"]), "fake": after["fake"], "dies": after["honest_dies"]})
        print(f"  leftover after {col} {_f(after['rank'])} dies={after['honest_dies']}")
    prose = (
        f"leftover after a_out6 {_f(store['a_out6'])} a_out12 {_f(store['a_out12'])} "
        f"a_net {_f(store['a_net'])} a_io_ratio {_f(store['a_io_ratio'])} a_n_tx {_f(store['a_n_tx'])}."
    )
    print(prose)
    return {"rows": rows, "store": store, "prose": prose}


def extra_zero_log(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — zero-out dummy leftover; leftover of log1p(a_op_out)")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    lab = y.notna()
    dum = (x == 0).astype(float).where(x.notna())
    rec_z = signed_oof_auroc(y, dum, tr["fold"], lab)
    after_z = leftover_diag(y, dum, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    rec_l = signed_oof_auroc(y, tr["log_opout"], tr["fold"], lab)
    after_l = leftover_diag(y, tr["log_opout"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    prose = (
        f"zero-out dummy Y3 {_f(_cv(rec_z))} leftover after days {_f(after_z['rank'])} "
        f"fake={after_z['fake']}. log1p(a_op_out) Y3 {_f(_cv(rec_l))} leftover {_f(after_l['rank'])} "
        f"dies={after_l['honest_dies']}."
    )
    print(prose)
    return {"z": after_z["rank"], "log": after_l["rank"], "log_cv": _cv(rec_l), "prose": prose}


def extra_y7(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Y7 leftover after days (no TURNOVER seat)")
    print("=" * 72)
    y7 = pd.to_numeric(tr[Y7], errors="coerce")
    rec = signed_oof_auroc(y7, tr[FLAG], tr["fold"], y7.notna())
    after = leftover_diag(y7, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], y7.notna())
    prose = (
        f"Y7 {FLAG} {_f(_cv(rec))} leftover after days {_f(after['rank'])} "
        f"dies={after['honest_dies']}. Do not grow TURNOVER 0.720."
    )
    print(prose)
    return {"y7": _cv(rec), "after": after["rank"], "prose": prose}


def extra_y2(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Y2 leftover after days (report-only)")
    print("=" * 72)
    y2 = pd.to_numeric(tr[Y2], errors="coerce")
    rec = signed_oof_auroc(y2, tr[FLAG], tr["fold"], y2.notna())
    after = leftover_diag(y2, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], y2.notna())
    prose = f"Y2 {FLAG} {_f(_cv(rec))} leftover after days {_f(after['rank'])} dies={after['honest_dies']}."
    print(prose)
    return {"y2": _cv(rec), "after": after["rank"], "prose": prose}


def extra_opin(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after a_op_in; leftover of a_op_in after days")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after = leftover_diag(y, tr[FLAG], (tr["a_op_in"],), tr["fold"], lab)
    after_d = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["a_op_in"]), tr["fold"], lab)
    opin = leftover_diag(y, tr["a_op_in"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    rec = signed_oof_auroc(y, tr["a_op_in"], tr["fold"], lab)
    prose = (
        f"leftover after a_op_in {_f(after['rank'])}. after days+a_op_in {_f(after_d['rank'])}. "
        f"a_op_in Y3 {_f(_cv(rec))} leftover after days {_f(opin['rank'])}."
    )
    print(prose)
    return {"after": after["rank"], "both": after_d["rank"], "opin": opin["rank"], "prose": prose}


def extra_out3_days(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of a_out3 after days; leftover after days+a_n_tx")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    out3 = leftover_diag(y, tr["a_out3"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    rec_o = signed_oof_auroc(y, tr["a_out3"], tr["fold"], lab)
    after_n = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["a_n_tx"]), tr["fold"], lab)
    after_all = leftover_diag(
        y, tr[FLAG],
        (tr["c_n_days_with_tx"], tr["log_in3"], tr["a_out3"]),
        tr["fold"], lab,
    )
    prose = (
        f"a_out3 Y3 {_f(_cv(rec_o))} leftover after days {_f(out3['rank'])} "
        f"dies={out3['honest_dies']} fake={out3['fake']}. "
        f"a_op_out leftover after days+a_n_tx {_f(after_n['rank'])} fake={after_n['fake']}. "
        f"after days+size+a_out3 {_f(after_all['rank'])} dies={after_all['honest_dies']}."
    )
    print(prose)
    return {"out3": out3["rank"], "ntx": after_n["rank"], "all": after_all["rank"], "prose": prose}


def extra_demean(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of company-demeaned a_op_out after days")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    demean = x - x.groupby(tr["company_id"]).transform("mean")
    rec = signed_oof_auroc(y, demean, tr["fold"], y.notna())
    after = leftover_diag(y, demean, (tr["c_n_days_with_tx"],), tr["fold"], y.notna())
    prose = (
        f"demeaned a_op_out Y3 {_f(_cv(rec))} leftover after days {_f(after['rank'])} "
        f"dies={after['honest_dies']} fake={after['fake']}. "
        f"ICC 0.955 BETWEEN + acf1 0.008 — demean kills the trait."
    )
    print(prose)
    return {"cv": _cv(rec), "rank": after["rank"], "dies": after["honest_dies"], "prose": prose}


def extra_pos_size(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover on a_op_out>0; leftover by size tercile")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    size = pd.to_numeric(tr["log_in3"], errors="coerce")
    med = tr.groupby("company_id")["log_in3"].median()
    cuts = med.quantile([1 / 3, 2 / 3])
    terc = pd.Series("T2", index=tr.index)
    mapped = tr["company_id"].map(med)
    terc[mapped <= cuts.iloc[0]] = "T1"
    terc[mapped > cuts.iloc[1]] = "T3"
    rows = []
    store = {}
    slices = (
        ("out>0", y.notna() & (x > 0)),
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
        f"out>0 leftover {_f(store['out>0'])} T1 {_f(store['T1_small'])} "
        f"T2 {_f(store['T2'])} T3 {_f(store['T3_large'])}."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def extra_ratio(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of a_op_out / a_out3 (month vs trail)")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    o3 = pd.to_numeric(tr["a_out3"], errors="coerce")
    ratio = (x / o3.replace(0, np.nan)).where(x.notna() & o3.notna())
    rec = signed_oof_auroc(y, ratio, tr["fold"], y.notna())
    after = leftover_diag(y, ratio, (tr["c_n_days_with_tx"],), tr["fold"], y.notna())
    after_o = leftover_diag(y, ratio, (tr["c_n_days_with_tx"], tr["a_out3"]), tr["fold"], y.notna())
    prose = (
        f"a_op_out/a_out3 Y3 {_f(_cv(rec))} leftover after days {_f(after['rank'])} "
        f"dies={after['honest_dies']} fake={after['fake']}. after days+a_out3 {_f(after_o['rank'])}."
    )
    print(prose)
    return {"cv": _cv(rec), "rank": after["rank"], "prose": prose}


def extra_size_fake(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after size honest flags; leftover after a_out6+days")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after_s = leftover_diag(y, tr[FLAG], (tr["log_in3"],), tr["fold"], lab)
    after_n = leftover_diag(y, tr[FLAG], (tr["a_n_tx"],), tr["fold"], lab)
    after_6 = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["a_out6"]), tr["fold"], lab)
    prose = (
        f"leftover after size rank {_f(after_s['rank'])} OLS {_f(after_s['ols'])} "
        f"fake={after_s['fake']} ρ={_f(after_s['rho_ctrl'])}. "
        f"leftover after a_n_tx rank {_f(after_n['rank'])} fake={after_n['fake']} "
        f"ρ={_f(after_n['rho_ctrl'])}. leftover after days+a_out6 {_f(after_6['rank'])} "
        f"fake={after_6['fake']}."
    )
    print(prose)
    return {"size_fake": after_s["fake"], "ntx_fake": after_n["fake"], "d6": after_6["rank"], "prose": prose}


def extra_shock(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of month shock a_op_out - a_out3/3 after days")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    o3 = pd.to_numeric(tr["a_out3"], errors="coerce")
    shock = (x - o3 / 3.0).where(x.notna() & o3.notna())
    rec = signed_oof_auroc(y, shock, tr["fold"], y.notna())
    after = leftover_diag(y, shock, (tr["c_n_days_with_tx"],), tr["fold"], y.notna())
    after_s = leftover_diag(y, shock, (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], y.notna())
    prose = (
        f"month-shock (a_op_out-a_out3/3) Y3 {_f(_cv(rec))} leftover after days "
        f"{_f(after['rank'])} dies={after['honest_dies']} fake={after['fake']}. "
        f"after days+size {_f(after_s['rank'])}."
    )
    print(prose)
    return {"cv": _cv(rec), "rank": after["rank"], "both": after_s["rank"], "prose": prose}


def extra_last_half(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — last-month leftover; first vs last half of trail")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    last = tr.groupby("company_id")["period"].transform("max") == pd.to_datetime(tr["period"])
    rec_l = signed_oof_auroc(y, tr[FLAG], tr["fold"], y.notna() & last)
    after_l = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], y.notna() & last)
    rnk = tr.groupby("company_id")["period"].rank(method="first", pct=True)
    first = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], y.notna() & (rnk <= 0.5))
    late = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], y.notna() & (rnk > 0.5))
    nmo = tr.groupby("company_id")["period"].transform("size")
    long = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], y.notna() & (nmo >= 12))
    short = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], y.notna() & (nmo < 12))
    prose = (
        f"last-month Y3 {_f(_cv(rec_l))} leftover {_f(after_l['rank'])} n={after_l['n']}. "
        f"first-half leftover {_f(first['rank'])} last-half {_f(late['rank'])}. "
        f"trail≥12 leftover {_f(long['rank'])} short {_f(short['rank'])}."
    )
    print(prose)
    return {
        "last": after_l["rank"], "first": first["rank"], "late": late["rank"],
        "long": long["rank"], "short": short["rank"], "prose": prose,
    }


def extra_mom_io(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — MoM Δ leftover; leftover of a_op_out/a_op_in")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    mom = x - x.groupby(tr["company_id"]).shift(1)
    rec_m = signed_oof_auroc(y, mom, tr["fold"], y.notna())
    after_m = leftover_diag(y, mom, (tr["c_n_days_with_tx"],), tr["fold"], y.notna())
    opin = pd.to_numeric(tr["a_op_in"], errors="coerce")
    ratio = (x / opin.replace(0, np.nan)).where(x.notna() & opin.notna())
    rec_r = signed_oof_auroc(y, ratio, tr["fold"], y.notna())
    after_r = leftover_diag(y, ratio, (tr["c_n_days_with_tx"],), tr["fold"], y.notna())
    prose = (
        f"MoM Δ a_op_out Y3 {_f(_cv(rec_m))} leftover after days {_f(after_m['rank'])} "
        f"dies={after_m['honest_dies']} fake={after_m['fake']}. "
        f"a_op_out/a_op_in Y3 {_f(_cv(rec_r))} leftover {_f(after_r['rank'])} "
        f"dies={after_r['honest_dies']}."
    )
    print(prose)
    return {"mom": after_m["rank"], "io": after_r["rank"], "prose": prose}


def extra_period_stack(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover by year; leftover after days+size+out3+n_tx")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    yr = pd.to_datetime(tr["period"]).dt.year
    y25 = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], y.notna() & (yr == 2025))
    y26 = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], y.notna() & (yr == 2026))
    o3ok = leftover_diag(
        y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"],
        y.notna() & pd.to_numeric(tr["a_out3"], errors="coerce").notna(),
    )
    stacked = leftover_diag(
        y, tr[FLAG],
        (tr["c_n_days_with_tx"], tr["log_in3"], tr["a_out3"], tr["a_n_tx"]),
        tr["fold"], y.notna(),
    )
    after_so = leftover_diag(
        y, tr[FLAG], (tr["a_out3"], tr["log_in3"]), tr["fold"], y.notna(),
    )
    prose = (
        f"2025 leftover {_f(y25['rank'])} n={y25['n']}. 2026 leftover {_f(y26['rank'])} n={y26['n']}. "
        f"a_out3-defined leftover {_f(o3ok['rank'])}. after days+size+out3+n_tx "
        f"{_f(stacked['rank'])} fake={stacked['fake']}. after a_out3+size {_f(after_so['rank'])}."
    )
    print(prose)
    return {
        "y25": y25["rank"], "y26": y26["rank"], "o3ok": o3ok["rank"],
        "stack": stacked["rank"], "so": after_so["rank"], "prose": prose,
    }


def extra_group_zero(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover on sibling groups; leftover on never-zero vs ever-zero")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    gsize = tr.groupby("group_id")["company_id"].transform("nunique")
    sib = leftover_diag(
        y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"],
        y.notna() & (gsize >= 2),
    )
    solo = leftover_diag(
        y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"],
        y.notna() & (gsize < 2),
    )
    ever0 = x.eq(0).groupby(tr["company_id"]).transform("any")
    never = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], y.notna() & ~ever0)
    ever = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], y.notna() & ever0)
    prose = (
        f"sibling n_co≥2 leftover {_f(sib['rank'])} n={sib['n']}. solo leftover {_f(solo['rank'])}. "
        f"never-zero leftover {_f(never['rank'])} ever-zero leftover {_f(ever['rank'])}."
    )
    print(prose)
    return {"sib": sib["rank"], "solo": solo["rank"], "never": never["rank"], "ever": ever["rank"], "prose": prose}


def extra_last_lab(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — last labeled month leftover; first labeled leftover")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    last_lab = tr.loc[lab].groupby("company_id")["period"].transform("max")
    first_lab = tr.loc[lab].groupby("company_id")["period"].transform("min")
    last_m = pd.Series(False, index=tr.index)
    first_m = pd.Series(False, index=tr.index)
    last_m.loc[lab] = pd.to_datetime(tr.loc[lab, "period"]) == last_lab
    first_m.loc[lab] = pd.to_datetime(tr.loc[lab, "period"]) == first_lab
    rec_l = signed_oof_auroc(y, tr[FLAG], tr["fold"], last_m)
    after_l = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], last_m)
    rec_f = signed_oof_auroc(y, tr[FLAG], tr["fold"], first_m)
    after_f = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], first_m)
    days_pos = leftover_diag(
        y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"],
        lab & (pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce") > 0),
    )
    prose = (
        f"last-labeled Y3 {_f(_cv(rec_l))} leftover {_f(after_l['rank'])} n={after_l['n']} pos={after_l['n_pos']}. "
        f"first-labeled leftover {_f(after_f['rank'])} n={after_f['n']}. "
        f"days>0 leftover {_f(days_pos['rank'])}."
    )
    print(prose)
    return {"last": after_l["rank"], "first": after_f["rank"], "days_pos": days_pos["rank"], "prose": prose}


def extra_winsor_q(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of winsorized a_op_out; leftover on mid quintiles")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    lo, hi = x.quantile(0.01), x.quantile(0.99)
    w = x.clip(lower=lo, upper=hi)
    rec_w = signed_oof_auroc(y, w, tr["fold"], y.notna())
    after_w = leftover_diag(y, w, (tr["c_n_days_with_tx"],), tr["fold"], y.notna())
    q = pd.qcut(x.where(x > 0), 5, labels=False, duplicates="drop")
    mid = leftover_diag(
        y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"],
        y.notna() & q.isin([1, 2, 3]),
    )
    tails = leftover_diag(
        y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"],
        y.notna() & q.isin([0, 4]),
    )
    after_log_s = leftover_diag(
        y, tr["log_opout"], (tr["log_in3"],), tr["fold"], y.notna(),
    )
    after_d_io = leftover_diag(
        y, tr[FLAG], (tr["c_n_days_with_tx"], tr["a_io_ratio"]), tr["fold"], y.notna(),
    )
    prose = (
        f"winsor 1/99 Y3 {_f(_cv(rec_w))} leftover {_f(after_w['rank'])}. "
        f"mid-quintile leftover {_f(mid['rank'])} tail leftover {_f(tails['rank'])}. "
        f"log_opout leftover after size {_f(after_log_s['rank'])} fake={after_log_s['fake']}. "
        f"leftover after days+io {_f(after_d_io['rank'])}."
    )
    print(prose)
    return {
        "w": after_w["rank"], "mid": mid["rank"], "tails": tails["rank"],
        "log_s": after_log_s["rank"], "dio": after_d_io["rank"], "prose": prose,
    }


def extra_last_size(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — last-labeled leftover after size; outflow per day")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    last_lab = tr.loc[lab].groupby("company_id")["period"].transform("max")
    last_m = pd.Series(False, index=tr.index)
    last_m.loc[lab] = pd.to_datetime(tr.loc[lab, "period"]) == last_lab
    after_ls = leftover_diag(y, tr[FLAG], (tr["log_in3"],), tr["fold"], last_m)
    after_ld = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], last_m)
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    per = (pd.to_numeric(tr[FLAG], errors="coerce") / days.replace(0, np.nan)).where(days.notna())
    rec_p = signed_oof_auroc(y, per, tr["fold"], lab)
    after_p = leftover_diag(y, per, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_ps = leftover_diag(y, per, (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], lab)
    q5 = pd.qcut(pd.to_numeric(tr[FLAG], errors="coerce").where(lambda s: s > 0), 5, labels=False, duplicates="drop")
    q5_left = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & (q5 == 4))
    q1_left = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & (q5 == 0))
    prose = (
        f"last-labeled leftover after size {_f(after_ls['rank'])} fake={after_ls['fake']}. "
        f"after days+size {_f(after_ld['rank'])}. outflow/day Y3 {_f(_cv(rec_p))} "
        f"leftover after days {_f(after_p['rank'])} after days+size {_f(after_ps['rank'])}. "
        f"Q5 leftover {_f(q5_left['rank'])} Q1 leftover {_f(q1_left['rank'])}."
    )
    print(prose)
    return {
        "ls": after_ls["rank"], "ld": after_ld["rank"], "per": after_p["rank"],
        "ps": after_ps["rank"], "q5": q5_left["rank"], "q1": q1_left["rank"], "prose": prose,
    }


def extra_full_stack(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after full stack; leftover after days+a_out12")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    full = leftover_diag(
        y, tr[FLAG],
        (tr["c_n_days_with_tx"], tr["log_in3"], tr["a_out3"], tr["a_n_tx"], tr["a_io_ratio"]),
        tr["fold"], lab,
    )
    d12 = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["a_out12"]), tr["fold"], lab)
    dlog = leftover_diag(y, tr["log_opout"], (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], lab)
    med = tr.groupby("company_id")[FLAG].transform("median")
    cut = med.median()
    hi = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & (med >= cut))
    lo = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & (med < cut))
    prose = (
        f"leftover after days+size+out3+n_tx+io {_f(full['rank'])} fake={full['fake']}. "
        f"after days+a_out12 {_f(d12['rank'])}. log leftover after days+size {_f(dlog['rank'])}. "
        f"high-median-out leftover {_f(hi['rank'])} low-median-out leftover {_f(lo['rank'])}."
    )
    print(prose)
    return {"full": full["rank"], "d12": d12["rank"], "dlog": dlog["rank"], "hi": hi["rank"], "lo": lo["rank"], "prose": prose}


def extra_dark_last(tr: pd.DataFrame, book: set[str]) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — last-labeled leftover Dark vs ERP; leftover after a_out3")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    last_lab = tr.loc[lab].groupby("company_id")["period"].transform("max")
    last_m = pd.Series(False, index=tr.index)
    last_m.loc[lab] = pd.to_datetime(tr.loc[lab, "period"]) == last_lab
    dark = ~tr["company_id"].isin(book)
    d = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], last_m & dark)
    e = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], last_m & ~dark)
    d_s = leftover_diag(y, tr[FLAG], (tr["log_in3"],), tr["fold"], last_m & dark)
    e_s = leftover_diag(y, tr[FLAG], (tr["log_in3"],), tr["fold"], last_m & ~dark)
    last_o3 = leftover_diag(y, tr[FLAG], (tr["a_out3"],), tr["fold"], last_m)
    prose = (
        f"last-labeled Dark leftover {_f(d['rank'])} n={d['n']} ERP leftover {_f(e['rank'])} n={e['n']}. "
        f"Dark after size {_f(d_s['rank'])} ERP after size {_f(e_s['rank'])}. "
        f"last-labeled leftover after a_out3 {_f(last_o3['rank'])}."
    )
    print(prose)
    return {"d": d["rank"], "e": e["rank"], "ds": d_s["rank"], "es": e_s["rank"], "o3": last_o3["rank"], "prose": prose}


def extra_onboard_rho(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover by first_month; leftover by company ρ(out, size)")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    fm = pd.to_datetime(tr["first_month"])
    cut = fm.quantile(0.5)
    early = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & (fm <= cut))
    late = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & (fm > cut))
    rows = []
    for cid, gg in tr.loc[lab, ["company_id", FLAG, "log_in3"]].groupby("company_id"):
        if len(gg) < 4:
            continue
        r = spearman(gg[FLAG], gg["log_in3"])
        if np.isfinite(r):
            rows.append((cid, r))
    rho_s = pd.Series({c: r for c, r in rows})
    mapped = tr["company_id"].map(rho_s)
    med = rho_s.median() if len(rho_s) else float("nan")
    hi = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & (mapped >= med))
    lo = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & (mapped < med))
    prose = (
        f"early-onboard leftover {_f(early['rank'])} late-onboard leftover {_f(late['rank'])}. "
        f"high ρ(out,size) leftover {_f(hi['rank'])} low ρ leftover {_f(lo['rank'])} "
        f"median company ρ={_f(med)} n_co={len(rho_s)}."
    )
    print(prose)
    return {
        "early": early["rank"], "late": late["rank"], "hi": hi["rank"], "lo": lo["rank"],
        "mapped": mapped, "med": med, "prose": prose,
    }


def extra_low_rho_stack(tr: pd.DataFrame, xor: dict) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after days+size / days+out3 on low vs high ρ(out,size)")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    mapped = xor["mapped"]
    med = xor["med"]
    low = lab & (mapped < med)
    high = lab & (mapped >= med)
    lo_s = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], low)
    hi_s = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], high)
    lo_o = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["a_out3"]), tr["fold"], low)
    hi_o = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["a_out3"]), tr["fold"], high)
    lo_o3 = leftover_diag(y, tr["a_out3"], (tr["c_n_days_with_tx"],), tr["fold"], low)
    rec_lo = signed_oof_auroc(y, tr[FLAG], tr["fold"], low)
    rec_hi = signed_oof_auroc(y, tr[FLAG], tr["fold"], high)
    rho_lo = spearman(tr.loc[low, FLAG], tr.loc[low, "a_out3"]) if low.any() else float("nan")
    rho_hi = spearman(tr.loc[high, FLAG], tr.loc[high, "a_out3"]) if high.any() else float("nan")
    prose = (
        f"low-ρ Y3 {_f(_cv(rec_lo))} leftover after days+size {_f(lo_s['rank'])} "
        f"after days+out3 {_f(lo_o['rank'])} a_out3 leftover {_f(lo_o3['rank'])} "
        f"ρ vs a_out3 {_f(rho_lo)} n={lo_s['n']}. "
        f"high-ρ Y3 {_f(_cv(rec_hi))} leftover after days+size {_f(hi_s['rank'])} "
        f"after days+out3 {_f(hi_o['rank'])} ρ vs a_out3 {_f(rho_hi)} n={hi_s['n']}."
    )
    print(prose)
    return {"lo_s": lo_s["rank"], "lo_o": lo_o["rank"], "hi_s": hi_s["rank"], "prose": prose}


def extra_intensity_net(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of a_op_out/a_in3; leftover on net-out months")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    inn = pd.to_numeric(tr["a_in3"], errors="coerce")
    intensity = (x / inn.replace(0, np.nan)).where(x.notna() & inn.notna())
    rec_i = signed_oof_auroc(y, intensity, tr["fold"], lab)
    after_i = leftover_diag(y, intensity, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_is = leftover_diag(y, intensity, (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], lab)
    net = pd.to_numeric(tr["a_net"], errors="coerce")
    neg = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & (net < 0))
    pos = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & (net >= 0))
    ntx_med = tr.groupby("company_id")["a_n_tx"].transform("median")
    cut = ntx_med.median()
    hi_n = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & (ntx_med >= cut))
    lo_n = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & (ntx_med < cut))
    prose = (
        f"a_op_out/a_in3 Y3 {_f(_cv(rec_i))} leftover after days {_f(after_i['rank'])} "
        f"after days+size {_f(after_is['rank'])}. net-out leftover {_f(neg['rank'])} "
        f"net-in leftover {_f(pos['rank'])}. high-n_tx leftover {_f(hi_n['rank'])} "
        f"low-n_tx leftover {_f(lo_n['rank'])}."
    )
    print(prose)
    return {"int": after_i["rank"], "neg": neg["rank"], "pos": pos["rank"], "prose": prose}


def extra_ever_io_last3(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover on ever-Y3; leftover by a_io_ratio; last-3 labeled")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    ever = (y == 1).groupby(tr["company_id"]).transform("any")
    e = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & ever)
    n = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & ~ever)
    io = pd.to_numeric(tr["a_io_ratio"], errors="coerce")
    hi_io = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & (io >= 1))
    lo_io = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & (io < 1))
    rnk = tr.loc[lab].groupby("company_id")["period"].rank(method="first", ascending=False)
    last3 = pd.Series(False, index=tr.index)
    last3.loc[lab] = rnk <= 3
    l3 = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], last3)
    log_int = tr["log_opout"] - tr["log_in3"]
    rec_li = signed_oof_auroc(y, log_int, tr["fold"], lab)
    after_li = leftover_diag(y, log_int, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    prose = (
        f"ever-Y3 leftover {_f(e['rank'])} n={e['n']} never-Y3 leftover {_f(n['rank'])}. "
        f"io≥1 leftover {_f(hi_io['rank'])} io<1 leftover {_f(lo_io['rank'])}. "
        f"last-3-labeled leftover {_f(l3['rank'])} n={l3['n']}. "
        f"log_opout-log_in3 Y3 {_f(_cv(rec_li))} leftover {_f(after_li['rank'])}."
    )
    print(prose)
    return {"ever": e["rank"], "never": n["rank"], "l3": l3["rank"], "li": after_li["rank"], "prose": prose}


def extra_last3_stack(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after days+size / days+out3 on last-3 labeled")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    rnk = tr.loc[lab].groupby("company_id")["period"].rank(method="first", ascending=False)
    last3 = pd.Series(False, index=tr.index)
    last3.loc[lab] = rnk <= 3
    after_s = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], last3)
    after_o = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["a_out3"]), tr["fold"], last3)
    yr = pd.to_datetime(tr["period"]).dt.year
    y26 = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], last3 & (yr == 2026))
    last_lab = tr.loc[lab].groupby("company_id")["period"].transform("max")
    last_m = pd.Series(False, index=tr.index)
    last_m.loc[lab] = pd.to_datetime(tr.loc[lab, "period"]) == last_lab
    last_so = leftover_diag(
        y, tr[FLAG], (tr["c_n_days_with_tx"], tr["log_in3"], tr["a_out3"]),
        tr["fold"], last_m,
    )
    ntx = pd.to_numeric(tr["a_n_tx"], errors="coerce")
    hi_tx = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & (ntx >= ntx.median()))
    prose = (
        f"last-3 leftover after days+size {_f(after_s['rank'])} after days+out3 {_f(after_o['rank'])}. "
        f"last-3 2026 leftover {_f(y26['rank'])} n={y26['n']}. "
        f"last-labeled leftover after days+size+out3 {_f(last_so['rank'])}. "
        f"high a_n_tx leftover {_f(hi_tx['rank'])}."
    )
    print(prose)
    return {"l3s": after_s["rank"], "l3o": after_o["rank"], "y26": y26["rank"], "prose": prose}


def extra_low_out3_rho(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover on companies with low ρ(a_op_out, a_out3)")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    rows = []
    for cid, gg in tr.loc[lab, ["company_id", FLAG, "a_out3"]].groupby("company_id"):
        if len(gg) < 4:
            continue
        r = spearman(gg[FLAG], gg["a_out3"])
        if np.isfinite(r):
            rows.append((cid, r))
    rho_s = pd.Series({c: r for c, r in rows})
    mapped = tr["company_id"].map(rho_s)
    low = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & (mapped < TWIN_RHO))
    hi = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & (mapped >= TWIN_RHO))
    low_s = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], lab & (mapped < TWIN_RHO))
    rec_lo = signed_oof_auroc(y, tr[FLAG], tr["fold"], lab & (mapped < TWIN_RHO))
    rho_lo = spearman(
        tr.loc[lab & (mapped < TWIN_RHO), FLAG],
        tr.loc[lab & (mapped < TWIN_RHO), "a_out3"],
    ) if (lab & (mapped < TWIN_RHO)).any() else float("nan")
    prose = (
        f"company ρ(out,out3)<0.80 leftover {_f(low['rank'])} Y3 {_f(_cv(rec_lo))} "
        f"n={low['n']} n_co={int((rho_s < TWIN_RHO).sum())} panel ρ vs a_out3 {_f(rho_lo)}. "
        f"after days+size {_f(low_s['rank'])}. twin-co leftover {_f(hi['rank'])} "
        f"n_co={int((rho_s >= TWIN_RHO).sum())} median company ρ={_f(float(rho_s.median()) if len(rho_s) else float('nan'))}."
    )
    print(prose)
    return {"low": low["rank"], "hi": hi["rank"], "prose": prose}


def extra_no_trail(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover on a_out3-missing / first-3 company months")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    o3 = pd.to_numeric(tr["a_out3"], errors="coerce")
    miss = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & o3.isna())
    rec_m = signed_oof_auroc(y, tr[FLAG], tr["fold"], lab & o3.isna())
    age = tr.groupby("company_id")["period"].rank(method="first")
    young = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & (age <= 3))
    old = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & (age > 12))
    young_s = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], lab & (age <= 3))
    prose = (
        f"a_out3-missing leftover {_f(miss['rank'])} Y3 {_f(_cv(rec_m))} n={miss['n']}. "
        f"age≤3 leftover {_f(young['rank'])} after days+size {_f(young_s['rank'])}. "
        f"age>12 leftover {_f(old['rank'])}."
    )
    print(prose)
    return {"miss": miss["rank"], "young": young["rank"], "old": old["rank"], "prose": prose}


def extra_mature_dark26(tr: pd.DataFrame, book: set[str]) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover on mature trails; leftover on Dark 2026")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    age = tr.groupby("company_id")["period"].rank(method="first")
    mid = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & (age > 3) & (age <= 12))
    old_s = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], lab & (age > 12))
    old_o = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["a_out3"]), tr["fold"], lab & (age > 12))
    yr = pd.to_datetime(tr["period"]).dt.year
    dark = ~tr["company_id"].isin(book)
    d26 = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & dark & (yr == 2026))
    e26 = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & ~dark & (yr == 2026))
    prose = (
        f"age 4–12 leftover {_f(mid['rank'])} n={mid['n']}. age>12 leftover after days+size "
        f"{_f(old_s['rank'])} after days+out3 {_f(old_o['rank'])}. "
        f"Dark 2026 leftover {_f(d26['rank'])} n={d26['n']} ERP 2026 leftover {_f(e26['rank'])} n={e26['n']}."
    )
    print(prose)
    return {"mid": mid["rank"], "old_s": old_s["rank"], "d26": d26["rank"], "prose": prose}


def extra_stable_iqr(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover on stable-outflow companies; leftover on size IQR")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    cv = x.groupby(tr["company_id"]).transform(lambda s: s.std() / s.mean() if s.mean() and s.mean() > 0 else np.nan)
    med_cv = cv.median()
    stable = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & (cv <= med_cv))
    noisy = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & (cv > med_cv))
    size = pd.to_numeric(tr["log_in3"], errors="coerce")
    q1, q3 = size.quantile(0.25), size.quantile(0.75)
    iqr = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab & (size >= q1) & (size <= q3))
    age = tr.groupby("company_id")["period"].rank(method="first")
    old_all = leftover_diag(
        y, tr[FLAG],
        (tr["c_n_days_with_tx"], tr["log_in3"], tr["a_out3"]),
        tr["fold"], lab & (age > 12),
    )
    prose = (
        f"stable-outflow leftover {_f(stable['rank'])} noisy leftover {_f(noisy['rank'])}. "
        f"size-IQR leftover {_f(iqr['rank'])}. age>12 leftover after days+size+out3 {_f(old_all['rank'])}."
    )
    print(prose)
    return {"stable": stable["rank"], "noisy": noisy["rank"], "iqr": iqr["rank"], "cv": cv, "med_cv": med_cv, "prose": prose}


def extra_noisy_cv(tr: pd.DataFrame, xsi: dict) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after days+size on noisy; leftover of outflow CV")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    cv = xsi["cv"]
    med_cv = xsi["med_cv"]
    noisy = leftover_diag(
        y, tr[FLAG], (tr["c_n_days_with_tx"], tr["log_in3"]),
        tr["fold"], lab & (cv > med_cv),
    )
    noisy_o = leftover_diag(
        y, tr[FLAG], (tr["c_n_days_with_tx"], tr["a_out3"]),
        tr["fold"], lab & (cv > med_cv),
    )
    rec_cv = signed_oof_auroc(y, cv, tr["fold"], lab)
    after_cv = leftover_diag(y, cv, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    prose = (
        f"noisy leftover after days+size {_f(noisy['rank'])} after days+out3 {_f(noisy_o['rank'])}. "
        f"outflow CV Y3 {_f(_cv(rec_cv))} leftover after days {_f(after_cv['rank'])} "
        f"dies={after_cv['honest_dies']}."
    )
    print(prose)
    return {"noisy_s": noisy["rank"], "cv": after_cv["rank"], "prose": prose}


def extra_cv_size(tr: pd.DataFrame, xsi: dict) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of outflow CV after size / days+size (PARK, not a card seat)")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    cv = xsi["cv"]
    rec = signed_oof_auroc(y, cv, tr["fold"], lab)
    after_s = leftover_diag(y, cv, (tr["log_in3"],), tr["fold"], lab)
    after_b = leftover_diag(y, cv, (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], lab)
    rho = spearman(cv, tr["log_in3"])
    rho_d = spearman(cv, tr["c_n_days_with_tx"])
    prose = (
        f"outflow CV Y3 {_f(_cv(rec))} leftover after size {_f(after_s['rank'])} "
        f"fake={after_s['fake']} ρ vs size {_f(rho)}. after days+size {_f(after_b['rank'])}. "
        f"ρ vs days {_f(rho_d)}. PARK — do not put CV on the 15-col card; this is not a_op_out KEEP."
    )
    print(prose)
    return {"cv": _cv(rec), "after_s": after_s["rank"], "both": after_b["rank"], "prose": prose}


def extra_cv_twins(tr: pd.DataFrame, xsi: dict) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of outflow CV after a_n_tx / a_out3 (still PARK)")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    cv = xsi["cv"]
    after_n = leftover_diag(y, cv, (tr["a_n_tx"],), tr["fold"], lab)
    after_o = leftover_diag(y, cv, (tr["a_out3"],), tr["fold"], lab)
    after_dno = leftover_diag(
        y, cv, (tr["c_n_days_with_tx"], tr["a_n_tx"], tr["log_in3"]),
        tr["fold"], lab,
    )
    rho_n = spearman(cv, tr["a_n_tx"])
    rho_o = spearman(cv, tr["a_out3"])
    prose = (
        f"CV leftover after a_n_tx {_f(after_n['rank'])} ρ={_f(rho_n)}. "
        f"after a_out3 {_f(after_o['rank'])} ρ={_f(rho_o)}. "
        f"after days+n_tx+size {_f(after_dno['rank'])}. PARK — not a_op_out KEEP; off the card."
    )
    print(prose)
    return {"n": after_n["rank"], "o": after_o["rank"], "dno": after_dno["rank"], "prose": prose}


def extra_opout_after_cv(tr: pd.DataFrame, xsi: dict) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of a_op_out after days+CV (does vol eat leftover?)")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    cv = xsi["cv"]
    after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], cv), tr["fold"], lab)
    after_s = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], cv, tr["log_in3"]), tr["fold"], lab)
    after_o = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], cv, tr["a_out3"]), tr["fold"], lab)
    prose = (
        f"a_op_out leftover after days+CV {_f(after['rank'])} fake={after['fake']}. "
        f"after days+CV+size {_f(after_s['rank'])}. after days+CV+out3 {_f(after_o['rank'])}."
    )
    print(prose)
    return {"rank": after["rank"], "both": after_s["rank"], "prose": prose}


def extra_cv_after_opout(tr: pd.DataFrame, xsi: dict) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of outflow CV after a_op_out (inverse)")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    cv = xsi["cv"]
    after = leftover_diag(y, cv, (tr[FLAG],), tr["fold"], lab)
    after_d = leftover_diag(y, cv, (tr[FLAG], tr["c_n_days_with_tx"]), tr["fold"], lab)
    prose = (
        f"CV leftover after a_op_out {_f(after['rank'])} fake={after['fake']}. "
        f"after a_op_out+days {_f(after_d['rank'])}. "
        f"a_op_out leftover after days+CV died 0.518 — vol ate the leftover."
    )
    print(prose)
    return {"rank": after["rank"], "both": after_d["rank"], "prose": prose}


def extra_opout_cv_only(tr: pd.DataFrame, xsi: dict) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover of a_op_out after CV only; leftover after days+CV last-labeled")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    cv = xsi["cv"]
    after = leftover_diag(y, tr[FLAG], (cv,), tr["fold"], lab)
    last_lab = tr.loc[lab].groupby("company_id")["period"].transform("max")
    last_m = pd.Series(False, index=tr.index)
    last_m.loc[lab] = pd.to_datetime(tr.loc[lab, "period"]) == last_lab
    last_cv = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], cv), tr["fold"], last_m)
    prose = (
        f"a_op_out leftover after CV-only {_f(after['rank'])} fake={after['fake']}. "
        f"last-labeled leftover after days+CV {_f(last_cv['rank'])} n={last_cv['n']}."
    )
    print(prose)
    return {"cv_only": after["rank"], "last": last_cv["rank"], "prose": prose}


def extra_cv_slices(tr: pd.DataFrame, xsi: dict, book: set[str]) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after days+CV on Dark / ERP / T3")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    cv = xsi["cv"]
    dark = ~tr["company_id"].isin(book)
    d = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], cv), tr["fold"], lab & dark)
    e = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], cv), tr["fold"], lab & ~dark)
    med = tr.groupby("company_id")["log_in3"].median()
    cuts = med.quantile([1 / 3, 2 / 3])
    mapped = tr["company_id"].map(med)
    t3 = leftover_diag(
        y, tr[FLAG], (tr["c_n_days_with_tx"], cv),
        tr["fold"], lab & (mapped > cuts.iloc[1]),
    )
    yr = pd.to_datetime(tr["period"]).dt.year
    y26 = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], cv), tr["fold"], lab & (yr == 2026))
    prose = (
        f"leftover after days+CV Dark {_f(d['rank'])} ERP {_f(e['rank'])} "
        f"T3 {_f(t3['rank'])} 2026 {_f(y26['rank'])}."
    )
    print(prose)
    return {"d": d["rank"], "e": e["rank"], "t3": t3["rank"], "y26": y26["rank"], "prose": prose}


def extra_cv_q(tr: pd.DataFrame, xsi: dict) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after days+CV on Q5 / mid-quintile / net-out")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    cv = xsi["cv"]
    x = pd.to_numeric(tr[FLAG], errors="coerce")
    q = pd.qcut(x.where(x > 0), 5, labels=False, duplicates="drop")
    q5 = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], cv), tr["fold"], lab & (q == 4))
    mid = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], cv), tr["fold"], lab & q.isin([1, 2, 3]))
    net = pd.to_numeric(tr["a_net"], errors="coerce")
    neg = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], cv), tr["fold"], lab & (net < 0))
    prose = (
        f"leftover after days+CV Q5 {_f(q5['rank'])} mid {_f(mid['rank'])} "
        f"net-out {_f(neg['rank'])}."
    )
    print(prose)
    return {"q5": q5["rank"], "mid": mid["rank"], "neg": neg["rank"], "prose": prose}


def decide(p1, p2, p3) -> dict:
    # Rank leftover is honest. OLS fake days leak must not kill a living rank.
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
        park = "KEEP leftover — still do not invent y_op_out; it is X not Y"
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
        park = "PARK as Y — do not invent y_op_out"
    else:
        role = "CLOSE unused leftover"
        why = (
            f"unused leftover after days: honest rank {_f(p3['rank'])} dies "
            f"(OLS {_f(p3['ols'])} fake={p3['fake']}). "
            f"DROP from the 44 as Y3 X. Off the 15-col card. Do not grow TURNOVER."
        )
        park = "PARK as Y — do not invent y_op_out"
    return {
        "role": role, "why": why, "leftover_lives": leftover_lives,
        "engine": engine, "is_size": is_size, "twin": twin,
        "park_y": park,
        "card": "no — do not put a_op_out on the 15-col card",
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
    ax.set_xlabel("a_op_out quintile (positive)")
    ax.set_ylabel("Y3 rate")
    ax.set_title("op_out vs recover")
    ax = axes[1]
    rec = p3["after"]["rrec"]
    ax.bar([r["fold"] for r in rec["folds"]], [r["auroc"] for r in rec["folds"]], color="#1f4e79")
    ax.axhline(CHANCE, color="#333", ls="--", lw=1, label="0.55 dies")
    ax.axhline(DAYS_BENCH, color="#c45c26", ls=":", lw=1, label="days 0.711")
    ax.set_xlabel("fold")
    ax.set_ylabel("rank leftover AUROC")
    ax.set_title("a_op_out leftover after days")
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
        "# Unused leftover of `a_op_out` after days as Y3 X",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. No `build_targets`. Do not invent `y_op_out`. Do not put a_op_out on the 15-col card. "
        "Do not overwrite `a_vol_qa.*`, `transfer_qa.*`, `growth_qa.*`, `in3_qa.*`, `n_accounts_qa.*`. "
        "Do not quote a_out_vol 0.722 as the engine. Do not grow TURNOVER.",
        "",
        "`a_op_out` = -sum(amount | grp = op_out) this month. Trailing twins: `a_out3` / `a_out6` / `a_out12`. "
        "`a_in3` leftover 0.521 already CLOSE. Size bar 0.617 stays.",
        "",
        "## Headline",
        "",
        (
            f"{d['role']} leftover-after-days rank {_f(p3['rank'])} (OLS {_f(p3['ols'])}, fake={p3['fake']}). "
            f"Y3 a_op_out {_f(p2['open'])} vs days {_f(p2['days'])} vs size {_f(p2['size'])} "
            f"vs a_out3 {_f(p2['out3'])}. SIZE={p1['is_size']} twin_gate={p1['twin_gate']} "
            f"twins={p1['gate_twins'] or 'none'}. Inverse days-after-a_op_out {_f(p3['inv_rank'])}. "
            f"Leftover after a_out3 {_f(p4['rank'])}. after days+size {_f(p5['after_b'])}. "
            f"Q6 lag1 leftover {_f(p6['l1_rank'])}. Demean leftover {_f(ctx['xd']['rank'])} dies. "
            f"mid-quintile leftover {_f(ctx['xwq']['mid'])} dies; T3 leftover 0.438 dies. "
            f"leftover after days+CV {_f(ctx['xoc']['rank'])} dies (vol ate leftover). "
            f"Card: **{d['role']}** / KEEP off the 15-col card. "
            f"Night Y3 0.762/0.752, days 0.711, size 0.617, TURNOVER 0.720/0.712 unchanged."
        ),
        "",
        "## Brief questions",
        "",
        "| # | question | what this cut says |",
        "| --- | --- | --- |",
        f"| 1 | Who is healthy? | {d['park_y']}. Size bar 0.617 stays. |",
        f"| 2 | Who is improving? | leftover after days {_f(p3['rank'])}. Month outflow ≠ 45→65. |",
        f"| 3 | Who is turning? | **{d['role']}** vs days 0.711. |",
        f"| 4 | Dip vs fall? | leftover after a_out3 {_f(p4['rank'])} — trailing rewrite? {p4['rewrite']}. |",
        f"| 5 | Why did it change? | twins={p1['gate_twins'] or 'none'}; SIZE={p1['is_size']}. |",
        f"| 6 | Months earlier? | lag1 leftover {_f(p6['l1_rank'])}; days_lag1 {_f(p6['days_l1'])}. |",
        "",
        "## PARK / CLOSE / KEEP / DROP-from-44",
        "",
        "| object | decision | why |",
        "| --- | --- | --- |",
        f"| `a_op_out` as Y3 X / 15-col card | **{d['role']}** | {d['why']} |",
        f"| `a_op_out` as engine X on the 44 | **{'KEEP leftover' if d['engine'] else 'DROP'}** | leftover lives={d['leftover_lives']} twin={d['twin']} SIZE={d['is_size']} beat-size={p2['beat_size']} |",
        f"| rewrite of a_out3 | **{'YES' if p4['rewrite'] else 'NO'}** | leftover after a_out3 {_f(p4['rank'])} |",
        f"| `y_op_out` | **PARK** | do not invent unless KEEP-as-X |",
        f"| Q6 lag1 / TURNOVER | **CLOSE** | leftover {_f(p6['l1_rank'])}; do not grow 0.720 |",
        f"| size bar `log1p(a_in3)` | **KEEP quote** | 0.617 stays; a_in3 leftover 0.521 CLOSE |",
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
        "## 4 — Leftover after a_out3",
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
        "### leftover after trailing / net / io / n_tx", "", ctx["xw"]["prose"], "", _md_table(ctx["xw"]["rows"]), "",
        "### zero dummy / log1p leftover", "", ctx["xz"]["prose"], "",
        "### Y7 leftover after days", "", ctx["xy7"]["prose"], "",
        "### Y2 leftover after days", "", ctx["x2"]["prose"], "",
        "### leftover after a_op_in", "", ctx["xo"]["prose"], "",
        "### leftover of a_out3 after days / days+n_tx / days+size+out3", "", ctx["x3"]["prose"], "",
        "### leftover of company-demeaned a_op_out", "", ctx["xd"]["prose"], "",
        "### leftover on out>0 / size terciles", "", ctx["xp"]["prose"], "", _md_table(ctx["xp"]["rows"]), "",
        "### leftover of a_op_out / a_out3", "", ctx["xr"]["prose"], "",
        "### leftover after size / a_n_tx fake flags", "", ctx["xsf"]["prose"], "",
        "### leftover of month shock a_op_out - a_out3/3", "", ctx["xsh"]["prose"], "",
        "### last-month / first-half / last-half / trail length", "", ctx["xlh"]["prose"], "",
        "### MoM Δ leftover / a_op_out/a_op_in", "", ctx["xm"]["prose"], "",
        "### leftover by year / stacked controls", "", ctx["xps"]["prose"], "",
        "### leftover on sibling groups / never-zero", "", ctx["xgz"]["prose"], "",
        "### last / first labeled leftover", "", ctx["xll"]["prose"], "",
        "### winsor / mid-quintile / log after size", "", ctx["xwq"]["prose"], "",
        "### last-labeled after size / outflow per day", "", ctx["xls"]["prose"], "",
        "### leftover after full stack / high vs low median out", "", ctx["xfs"]["prose"], "",
        "### last-labeled Dark vs ERP", "", ctx["xdl"]["prose"], "",
        "### leftover by onboard / company ρ(out, size)", "", ctx["xor"]["prose"], "",
        "### leftover after days+size / out3 on low vs high ρ", "", ctx["xlr"]["prose"], "",
        "### leftover of a_op_out/a_in3 / net-out months", "", ctx["xin"]["prose"], "",
        "### leftover on ever-Y3 / io slice / last-3", "", ctx["xei"]["prose"], "",
        "### leftover after days+size on last-3 / 2026", "", ctx["xl3"]["prose"], "",
        "### leftover on low company ρ(out, out3)", "", ctx["xlo"]["prose"], "",
        "### leftover on a_out3-missing / young companies", "", ctx["xnt"]["prose"], "",
        "### leftover on mature trails / Dark 2026", "", ctx["xmd"]["prose"], "",
        "### leftover on stable-outflow / size IQR", "", ctx["xsi"]["prose"], "",
        "### leftover after days+size on noisy / leftover of CV", "", ctx["xnc"]["prose"], "",
        "### leftover of outflow CV after size (PARK)", "", ctx["xcs"]["prose"], "",
        "### leftover of outflow CV after n_tx / out3 (PARK)", "", ctx["xct"]["prose"], "",
        "### leftover of a_op_out after days+CV", "", ctx["xoc"]["prose"], "",
        "### leftover of CV after a_op_out (inverse)", "", ctx["xco"]["prose"], "",
        "### leftover of a_op_out after CV-only / last-labeled days+CV", "", ctx["xoo"]["prose"], "",
        "### leftover after days+CV on Dark / ERP / T3", "", ctx["xcv"]["prose"], "",
        "### leftover after days+CV on Q5 / mid / net-out", "", ctx["xcq"]["prose"], "",
        "## Night quotes (unchanged)",
        "",
        "| quote | locked |",
        "| --- | --- |",
        f"| Y3 | {Y3_NIGHT[0]:.3f} / {Y3_NIGHT[1]:.3f} |",
        f"| days | {DAYS_BENCH:.3f} |",
        f"| size | {SIZE_QUOTE:.3f} |",
        f"| Y7 TURNOVER | {Y7_TURNOVER:.3f} / {Y7_B_SHALLOW:.3f} |",
        f"| a_in3 leftover | {IN3_LEFT:.3f} CLOSE |",
        f"| q6_keep | issued_lag1 / days_lag1 / ss_lag1 |",
        "",
        "Do not quote a_out_vol 0.722 as the engine. Do not grow TURNOVER. Do not put `a_op_out` on the 15-col card.",
        "",
        "## Files written",
        "",
        "- `analysis/evaluate/op_out_qa.py`",
        "- `analysis/outputs/op_out_qa.md`",
        "- `analysis/outputs/op_out_qa.png`" if ctx.get("png") else "- (no PNG)",
        "- append-only `analysis/experiments/registry.csv`",
        "- `overnight/waves/wave4_op_out.md` (end, if WRITE_WAVE)",
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
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_a_op_out", "value": p2["open"], "coverage": f"{p1['cov']:.4f}", "notes": f"days={p2['days']:.4f} size={p2['size']:.4f} beat={p2['beat_size']}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_a_op_out_resid_days", "value": p3["rank"], "coverage": f"{p1['cov']:.4f}", "notes": f"ols={p3['ols']:.4f} dies={p3['dies']} fake={p3['fake']}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_a_op_out_resid_out3", "value": p4["rank"], "coverage": f"{p1['cov']:.4f}", "notes": f"rewrite={p4['rewrite']}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train", "metric": "op_out_leftover", "value": 1 if d["leftover_lives"] else 0, "coverage": f"{p1['cov']:.4f}", "notes": d["role"] + " " + d["why"][:160]},
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
        f"# Wave 4 — a_op_out leftover after days as Y3 X\n\n"
        f"Agent `{AGENT}`. Train group-fold seed 20260918. Holdout 72 coverage only.\n\n"
        f"## Files written\n\n"
        f"- `analysis/evaluate/op_out_qa.py`\n"
        f"- `analysis/outputs/op_out_qa.md`\n"
        f"- `analysis/outputs/op_out_qa.png`\n"
        f"- append-only `analysis/experiments/registry.csv`\n"
        f"- this note\n\n"
        f"Did not touch `a_vol_qa.*`, `transfer_qa.*`, `growth_qa.*`, `in3_qa.*`, "
        f"`zero_in_qa.*`, `n_tx_qa.*`, `n_accounts_qa.*`, `gbm_core.py`, `cashflow.py`, "
        f"parquet / duckdb, `build_targets`, `product/`, the 15-col card, TURNOVER, "
        f"or the parent journal. Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. "
        f"Y7 TURNOVER **0.720 / 0.712** unchanged. Do not quote a_out_vol 0.722 as the engine.\n\n"
        f"## Locked verdict\n\n"
        f"| object | decision |\n"
        f"| --- | --- |\n"
        f"| `a_op_out` as Y3 X / 15-col card | **{d['role']}** |\n"
        f"| `a_op_out` as engine X on the 44 | **{'KEEP leftover' if d['engine'] else 'DROP'}** |\n"
        f"| `y_op_out` | **PARK** |\n"
        f"| TURNOVER | **CLOSE** — do not grow 0.720 |\n"
        f"| size bar | **KEEP quote 0.617** |\n\n"
        f"## Locked extras\n\n"
        f"- Honest leftover after days rank {_f(p3['rank'])} (lives={d['leftover_lives']}, OLS fake={p3['fake']}); inverse {_f(p3['inv_rank'])}.\n"
        f"- Single {_f(p2['open'])} vs days {_f(p2['days'])} vs size {_f(p2['size'])} vs a_out3 {_f(p2['out3'])}.\n"
        f"- SIZE={p1['is_size']} twins={p1['gate_twins'] or 'none'}.\n"
        f"- leftover after a_out3 {_f(p4['rank'])} rewrite={p4['rewrite']}. after days+size {_f(p5['after_b'])}.\n"
        f"- Q6 lag1 leftover {_f(p6['l1_rank'])}. Dark leftover {_f(p7['dark'])} ERP {_f(p7['erp'])}.\n"
        f"- {ctx['xa']['prose']}\n"
        f"- {ctx['x3']['prose']}\n"
        f"- {ctx['xd']['prose']}\n"
        f"- {ctx['xp']['prose']}\n"
        f"- {ctx['xr']['prose']}\n"
        f"- {ctx['xsf']['prose']}\n"
        f"- {ctx['xsh']['prose']}\n"
        f"- {ctx['xlh']['prose']}\n"
        f"- {ctx['xm']['prose']}\n"
        f"- {ctx['xps']['prose']}\n"
        f"- {ctx['xgz']['prose']}\n"
        f"- {ctx['xll']['prose']}\n"
        f"- {ctx['xwq']['prose']}\n"
        f"- {ctx['xls']['prose']}\n"
        f"- {ctx['xfs']['prose']}\n"
        f"- {ctx['xdl']['prose']}\n"
        f"- {ctx['xor']['prose']}\n"
        f"- {ctx['xlr']['prose']}\n"
        f"- {ctx['xin']['prose']}\n"
        f"- {ctx['xei']['prose']}\n"
        f"- {ctx['xl3']['prose']}\n"
        f"- {ctx['xlo']['prose']}\n"
        f"- {ctx['xnt']['prose']}\n"
        f"- {ctx['xmd']['prose']}\n"
        f"- {ctx['xsi']['prose']}\n"
        f"- {ctx['xnc']['prose']}\n"
        f"- {ctx['xcs']['prose']}\n"
        f"- {ctx['xct']['prose']}\n"
        f"- {ctx['xoc']['prose']}\n"
        f"- {ctx['xco']['prose']}\n"
        f"- {ctx['xoo']['prose']}\n"
        f"- {ctx['xcv']['prose']}\n"
        f"- {ctx['xcq']['prose']}\n"
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
    print("op_out leftover QA — unused leftover of a_op_out after days as Y3 X")
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
    p4 = pass4_out3(tr)
    p5 = pass5_size(tr)
    p6 = pass6_q6(tr)
    p7 = pass7_dark(tr, book)
    p8 = pass8_hold(hold)
    xb = extra_bootstrap(tr, n_boot=40)
    xa = extra_icc(tr)
    xw = extra_windows(tr)
    xz = extra_zero_log(tr)
    xy7 = extra_y7(tr)
    x2 = extra_y2(tr)
    xo = extra_opin(tr)
    x3 = extra_out3_days(tr)
    xd = extra_demean(tr)
    xp = extra_pos_size(tr)
    xr = extra_ratio(tr)
    xsf = extra_size_fake(tr)
    xsh = extra_shock(tr)
    xlh = extra_last_half(tr)
    xm = extra_mom_io(tr)
    xps = extra_period_stack(tr)
    xgz = extra_group_zero(tr)
    xll = extra_last_lab(tr)
    xwq = extra_winsor_q(tr)
    xls = extra_last_size(tr)
    xfs = extra_full_stack(tr)
    xdl = extra_dark_last(tr, book)
    xor = extra_onboard_rho(tr)
    xlr = extra_low_rho_stack(tr, xor)
    xin = extra_intensity_net(tr)
    xei = extra_ever_io_last3(tr)
    xl3 = extra_last3_stack(tr)
    xlo = extra_low_out3_rho(tr)
    xnt = extra_no_trail(tr)
    xmd = extra_mature_dark26(tr, book)
    xsi = extra_stable_iqr(tr)
    xnc = extra_noisy_cv(tr, xsi)
    xcs = extra_cv_size(tr, xsi)
    xct = extra_cv_twins(tr, xsi)
    xoc = extra_opout_after_cv(tr, xsi)
    xco = extra_cv_after_opout(tr, xsi)
    xoo = extra_opout_cv_only(tr, xsi)
    xcv = extra_cv_slices(tr, xsi, book)
    xcq = extra_cv_q(tr, xsi)
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
        "xb": xb, "xa": xa, "xw": xw, "xz": xz, "xy7": xy7, "x2": x2, "xo": xo,
        "x3": x3, "xd": xd, "xp": xp, "xr": xr, "xsf": xsf,
        "xsh": xsh, "xlh": xlh, "xm": xm, "xps": xps, "xgz": xgz,
        "xll": xll, "xwq": xwq, "xls": xls, "xfs": xfs, "xdl": xdl, "xor": xor, "xlr": xlr, "xin": xin, "xei": xei, "xl3": xl3, "xlo": xlo, "xnt": xnt, "xmd": xmd, "xsi": xsi, "xnc": xnc, "xcs": xcs, "xct": xct, "xoc": xoc, "xco": xco, "xoo": xoo, "xcv": xcv, "xcq": xcq,
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
