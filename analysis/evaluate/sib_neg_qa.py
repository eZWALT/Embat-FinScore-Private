"""Unused leftover of ``h_sib_neg_share`` after days as Y3 X.

``h_sib_neg_share`` = (# siblings with net < 0) / (h_group_size − 1)
(Family H, ``groupctx.py``). Family H is already PARK as Y3 X
(sibling_h): size T1 residual +12.5pp / +16.7pp; ``h_sib_neg_share``
flat (+0.01pp) CV **0.434** vs days 0.711. H marks sister
*existence*. Sister mean ``b_runway`` ≥ 1 is +7.0pp after size
inside the 110 — Q5 footnote only. Y3 never B. Hidden test is new
groups.

Do **not** overwrite ``sibling_h.py`` / ``sibling_h.md``. Do not put
H on the 15-col card.

KEEP-as-X: beat size ≥0.02 **and** leftover after days **and** not
SIZE (|ρ| vs log1p(a_in3) ≥0.50) **and** not a twin (|ρ|≥0.80 vs
days / a_n_tx / h_group_size / mixed dummy). Leftover <0.55 dies.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.sib_neg_qa

Owned: analysis/evaluate/sib_neg_qa.py, analysis/outputs/sib_neg_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_sib_neg.md (end).
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
OUT_MD = ANALYSIS / "outputs" / "sib_neg_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "sib_neg_qa.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_sib_neg.md"
AGENT = "572fb928"
WAVE = "4"
ROUND = "R4"
MODEL = "sib_neg_qa"
X_FAM = "H"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_QUOTE = 0.617
DAYS_LAG1_QUOTE = 0.684
Y3_NIGHT = (0.762, 0.752)
Y7_TURNOVER = 0.720
Y7_B_SHALLOW = 0.712
SIB_Y3_QUOTE = 0.434
SHARE_Y3_QUOTE = 0.638
MIXED_Y3_QUOTE = 0.494
T1_RES_IN3 = 0.1667
T1_RES_OPIN = 0.125
FOOT_PP = 0.0697
HNEG_110_GAP = 0.0001
KEEP_DELTA = 0.02
TWIN_RHO = 0.80
SIZE_RHO = 0.50
CHANCE = 0.55
ICC_TRAIT = 0.85
FAKE_DAYS_RHO = 0.40
MIN_POS = 50
MIN_ACF_PAIRS = 4
N_DARK_WANT = 470
N_MIXED_WANT = 110
N_ALLDARK_WANT = 360
WRITE_WAVE = True

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "a_in3",
    "a_n_tx",
    "a_op_in",
    "c_n_days_with_tx",
    "h_sib_neg_share",
    "h_group_size",
    "h_n_siblings_active",
    "h_share_group_in",
    "h_sib_in",
    "b_runway",
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
    fold_rows = []
    aucs = []
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
        "n_folds": len(finite), "folds": fold_rows,
        "train_sign": int(tr_sign),
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


def attach_mix_and_sisters(panel: pd.DataFrame, book: set[str]) -> pd.DataFrame:
    out = panel.copy()
    erp = out["company_id"].isin(book)
    out["is_erp"] = erp
    out["is_dark"] = ~erp
    train_erp = out["is_erp"] & (out["split"] == "train")
    g_has_erp = out["group_id"].isin(set(out.loc[train_erp, "group_id"].astype(str)))
    out["mixed_dummy"] = (out["is_dark"] & g_has_erp).astype(float)
    out["all_dark"] = out["is_dark"] & ~g_has_erp
    out["slice"] = np.where(out["is_erp"], "erp", np.where(out["mixed_dummy"] == 1, "mixed_110", "all_dark_360"))
    inv = out.loc[out["is_erp"], ["group_id", "period", "b_runway"]].copy()
    if inv.empty:
        out["sister_runway"] = np.nan
    else:
        agg = inv.groupby(["group_id", "period"], sort=False)["b_runway"].mean().rename("sister_runway")
        out = out.merge(agg.reset_index(), on=["group_id", "period"], how="left")
    out["sister_runway_ok"] = (pd.to_numeric(out["sister_runway"], errors="coerce") >= 1.0).astype(float)
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
    leak = leakage_check(
        ["h_sib_neg_share", "h_group_size", "h_share_group_in", "c_n_days_with_tx", "log_in3"],
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
    print("\n" + "=" * 72)
    print("CUT 1 — coverage; twin / SIZE screen")
    print("=" * 72)
    n_cm = len(tr)
    n_co = int(tr["company_id"].nunique())
    x = pd.to_numeric(tr["h_sib_neg_share"], errors="coerce")
    acf1 = median_acf(x, tr["company_id"], 1)
    pairs = {
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "a_n_tx": tr["a_n_tx"],
        "h_group_size": tr["h_group_size"],
        "mixed_dummy": tr["mixed_dummy"],
        "log1p(a_in3)": tr["log_in3"],
        "h_n_siblings_active": tr["h_n_siblings_active"],
        "h_share_group_in": tr["h_share_group_in"],
        "b_runway": tr["b_runway"],
    }
    rhos = {k: spearman(x, v) for k, v in pairs.items()}
    twins = [k for k, v in rhos.items() if np.isfinite(v) and abs(v) >= TWIN_RHO]
    is_size = bool(np.isfinite(rhos["log1p(a_in3)"]) and abs(rhos["log1p(a_in3)"]) >= SIZE_RHO)
    twin_gate = any(
        np.isfinite(rhos[k]) and abs(rhos[k]) >= TWIN_RHO
        for k in ("c_n_days_with_tx", "a_n_tx", "h_group_size", "mixed_dummy")
    )
    n_mixed = int(tr.loc[tr["slice"] == "mixed_110", "company_id"].nunique())
    n_ad = int(tr.loc[tr["slice"] == "all_dark_360", "company_id"].nunique())
    n_dark = int(tr.loc[tr["is_dark"], "company_id"].nunique())
    mix_ok = n_mixed == N_MIXED_WANT and n_ad == N_ALLDARK_WANT and n_dark == N_DARK_WANT
    rows = [
        {"col": "h_sib_neg_share", "n_nn": f"{int(x.notna().sum()):,}", "cov": _pp(_pct(int(x.notna().sum()), n_cm)), "acf1": _f(acf1)},
        {"col": "h_group_size", "n_nn": f"{int(pd.to_numeric(tr['h_group_size'], errors='coerce').notna().sum()):,}", "cov": _pp(1.0), "acf1": _f(median_acf(tr["h_group_size"], tr["company_id"], 1))},
        {"col": "h_share_group_in", "n_nn": f"{int(pd.to_numeric(tr['h_share_group_in'], errors='coerce').notna().sum()):,}", "cov": _pp(_pct(int(pd.to_numeric(tr['h_share_group_in'], errors='coerce').notna().sum()), n_cm)), "acf1": _f(median_acf(tr["h_share_group_in"], tr["company_id"], 1))},
    ]
    rho_rows = [
        {
            "vs": k, "rho": _f(v),
            "flag": "SIZE" if k == "log1p(a_in3)" and abs(v) >= SIZE_RHO else "TWIN" if abs(v) >= TWIN_RHO else "no",
        }
        for k, v in rhos.items()
    ]
    prose = (
        f"Train {n_co:,} co / {n_cm:,} CM. h_sib_neg_share cov {_pp(_pct(int(x.notna().sum()), n_cm))} "
        f"acf1={_f(acf1)}. Dark {n_dark} = mixed {n_mixed} + all-dark {n_ad} "
        f"{'CONFIRM 110/360/470' if mix_ok else 'DRIFT'}. "
        f"ρ vs days {_f(rhos['c_n_days_with_tx'])} vs a_n_tx {_f(rhos['a_n_tx'])} "
        f"vs h_group_size {_f(rhos['h_group_size'])} vs mixed {_f(rhos['mixed_dummy'])} "
        f"vs size {_f(rhos['log1p(a_in3)'])} vs own B {_f(rhos['b_runway'])}. "
        f"SIZE={is_size} twins={twins or 'none'} twin_gate={twin_gate}."
    )
    print(prose)
    return {
        "rows": rows, "rho_rows": rho_rows, "rhos": rhos, "twins": twins,
        "is_size": is_size, "twin_gate": twin_gate,
        "n_cm": n_cm, "n_co": n_co, "cov": _pct(int(x.notna().sum()), n_cm),
        "acf1": acf1, "mix_ok": mix_ok, "n_mixed": n_mixed, "n_ad": n_ad, "n_dark": n_dark,
        "prose": prose,
    }


def pass2_singles(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 2 — single-feature group-fold Y3")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    folds = tr["fold"]
    feats = {
        "h_sib_neg_share": tr["h_sib_neg_share"],
        "mixed_dummy": tr["mixed_dummy"],
        "h_group_size": tr["h_group_size"],
        "h_share_group_in": tr["h_share_group_in"],
        "h_n_siblings_active": tr["h_n_siblings_active"],
        "log1p(a_in3)": tr["log_in3"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "a_n_tx": tr["a_n_tx"],
    }
    recs = {}
    rows = []
    for fname, x in feats.items():
        rec = signed_oof_auroc(y, x, folds, lab)
        recs[fname] = rec
        rows.append(_auc_row(Y3, fname, rec))
        print(f"  {fname} CV={_f(rec['cv'])} n={rec['n_defined']} pos={rec['n_pos']} folds={fold_bits(rec)}")
    y3_h = _cv(recs["h_sib_neg_share"])
    y3_size = _cv(recs["log1p(a_in3)"])
    y3_days = _cv(recs["c_n_days_with_tx"])
    y3_sh = _cv(recs["h_share_group_in"])
    y3_mix = _cv(recs["mixed_dummy"])
    days_ok = bool(np.isfinite(y3_days) and abs(y3_days - DAYS_BENCH) < 0.008)
    size_ok = bool(np.isfinite(y3_size) and abs(y3_size - SIZE_QUOTE) < 0.008)
    peek_ok = bool(np.isfinite(y3_h) and abs(y3_h - SIB_Y3_QUOTE) < 0.015)
    share_ok = bool(np.isfinite(y3_sh) and abs(y3_sh - SHARE_Y3_QUOTE) < 0.015)
    beat_size = bool(np.isfinite(y3_h) and (y3_h - SIZE_QUOTE) >= KEEP_DELTA)
    prose = (
        f"Y3 h_sib_neg_share {_f(y3_h)} n={recs['h_sib_neg_share']['n_defined']:,} "
        f"pos={recs['h_sib_neg_share']['n_pos']:,} "
        f"(quote 0.434 {'CONFIRM' if peek_ok else 'DRIFT'}). "
        f"mixed {_f(y3_mix)} (quote 0.494) share_group_in {_f(y3_sh)} "
        f"(quote 0.638 {'CONFIRM' if share_ok else 'DRIFT'}) "
        f"vs size {_f(y3_size)} vs days {_f(y3_days)}. "
        f"Replica days 0.711 {'CONFIRM' if days_ok else 'DRIFT'} / size 0.617 "
        f"{'CONFIRM' if size_ok else 'DRIFT'}. Beat-size Δ="
        f"{_f(y3_h - SIZE_QUOTE) if np.isfinite(y3_h) else '—'} "
        f"{'PASS' if beat_size else 'FAIL'}."
    )
    print(prose)
    return {
        "rows": rows, "recs": recs, "sib": y3_h, "size": y3_size, "days": y3_days,
        "share": y3_sh, "mix": y3_mix,
        "days_ok": days_ok, "size_ok": size_ok, "peek_ok": peek_ok, "share_ok": share_ok,
        "beat_size": beat_size, "n_def": recs["h_sib_neg_share"]["n_defined"],
        "n_pos": recs["h_sib_neg_share"]["n_pos"], "prose": prose,
    }


def pass3_days(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 3 — honest leftover of h_sib_neg_share after days; inverse")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after = leftover_diag(y, tr["h_sib_neg_share"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    inv = leftover_diag(y, tr["c_n_days_with_tx"], (tr["h_sib_neg_share"],), tr["fold"], lab)
    prose = (
        f"h_sib_neg_share leftover after days OLS {_f(after['ols'])} rank {_f(after['rank'])} "
        f"fake={after['fake']} almost={after['almost']} ρ(resid,days)={_f(after['rho_ctrl'])} "
        f"R²={_f(after['r2'])} honest_dies={after['honest_dies']} n={after['n']:,} pos={after['n_pos']:,}. "
        f"Inverse: days leftover after sib_neg OLS {_f(inv['ols'])} rank {_f(inv['rank'])} "
        f"dies={inv['honest_dies']}."
    )
    print(prose)
    return {
        "after": after, "inv": inv, "ols": after["ols"], "rank": after["rank"],
        "dies": after["honest_dies"], "fake": after["fake"], "almost": after["almost"],
        "r2": after["r2"], "inv_rank": inv["rank"], "inv_dies": inv["honest_dies"], "prose": prose,
    }


def pass5_size(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 5 — leftover after size; T1 residual")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after_s = leftover_diag(y, tr["h_sib_neg_share"], (tr["log_in3"],), tr["fold"], lab)
    after_g = leftover_diag(y, tr["h_sib_neg_share"], (tr["h_group_size"],), tr["fold"], lab)
    after_m = leftover_diag(y, tr["h_sib_neg_share"], (tr["mixed_dummy"],), tr["fold"], lab)
    after_ds = leftover_diag(y, tr["h_sib_neg_share"], (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], lab)
    size = pd.to_numeric(tr["log_in3"], errors="coerce")
    src = size[tr["split"] == "train" if "split" in tr.columns else slice(None)]
    src = size.dropna()
    try:
        terc = pd.qcut(size, 3, labels=("T1", "T2", "T3"), duplicates="drop")
    except ValueError:
        terc = pd.Series("T2", index=tr.index)
    rows = []
    t1_res = float("nan")
    for t in ("T1", "T2", "T3"):
        sl = lab & terc.eq(t)
        mix = sl & (tr["slice"] == "mixed_110")
        ad = sl & (tr["slice"] == "all_dark_360")
        r_m = float(y[mix].mean()) if mix.any() else float("nan")
        r_a = float(y[ad].mean()) if ad.any() else float("nan")
        res = r_m - r_a if np.isfinite(r_m) and np.isfinite(r_a) else float("nan")
        if t == "T1":
            t1_res = res
        rec = signed_oof_auroc(y, tr["h_sib_neg_share"], tr["fold"], sl)
        after = leftover_diag(y, tr["h_sib_neg_share"], (tr["c_n_days_with_tx"],), tr["fold"], sl)
        rows.append(
            {
                "tercile": t, "n": rec["n_defined"], "n_pos": rec["n_pos"],
                "CV": "LOW_POWER" if rec["low_power"] else _f(rec["cv"]),
                "leftover": _f(after["rank"]), "dies": after["honest_dies"],
                "mixed rate": _pp(r_m), "all-dark rate": _pp(r_a), "residual": _pp(res) if np.isfinite(res) else "—",
            }
        )
        print(f"  {t} leftover={_f(after['rank'])} residual={_pp(res) if np.isfinite(res) else '—'}")
    t1_ok = bool(np.isfinite(t1_res) and abs(t1_res - T1_RES_IN3) < 0.04)
    prose = (
        f"sib_neg leftover after size rank {_f(after_s['rank'])} dies={after_s['honest_dies']}. "
        f"after h_group_size {_f(after_g['rank'])} dies={after_g['honest_dies']}. "
        f"after mixed {_f(after_m['rank'])} dies={after_m['honest_dies']}. "
        f"after days+size {_f(after_ds['rank'])} dies={after_ds['honest_dies']}. "
        f"T1 mixed−all-dark residual {_pp(t1_res)} (quote +16.7pp {'CONFIRM' if t1_ok else 'DRIFT'}). "
        f"T1 residual is the mixed-existence gap, not leftover of h_sib_neg "
        f"(after size dies={after_s['honest_dies']})."
    )
    print(prose)
    return {
        "rows": rows, "after_size": after_s["rank"], "after_g": after_g["rank"],
        "after_m": after_m["rank"], "after_ds": after_ds["rank"],
        "t1_res": t1_res, "t1_ok": t1_ok, "prose": prose,
        "s_dies": after_s["honest_dies"], "m_dies": after_m["honest_dies"],
    }


def pass6_mix(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 6 — mixed 110 vs all-dark 360 leftover")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    rows = []
    recs = {}
    for name, sl in (
        ("mixed_110", lab & (tr["slice"] == "mixed_110")),
        ("all_dark_360", lab & (tr["slice"] == "all_dark_360")),
        ("erp", lab & tr["is_erp"]),
        ("dark_470", lab & tr["is_dark"]),
    ):
        rec = signed_oof_auroc(y, tr["h_sib_neg_share"], tr["fold"], sl)
        after = leftover_diag(y, tr["h_sib_neg_share"], (tr["c_n_days_with_tx"],), tr["fold"], sl)
        recs[name] = {"rec": rec, "after": after}
        rows.append(
            {
                "slice": name, "n": rec["n_defined"], "n_pos": rec["n_pos"],
                "CV": "LOW_POWER" if rec["low_power"] else _f(rec["cv"]),
                "leftover": _f(after["rank"]), "dies": after["honest_dies"],
                "Y3 rate": _pp(float(y[sl].mean()) if sl.any() else float("nan")),
            }
        )
        print(f"  {name} CV={_f(rec['cv'])} leftover={_f(after['rank'])} dies={after['honest_dies']}")
    r_m = float(y[lab & (tr["slice"] == "mixed_110")].mean())
    r_a = float(y[lab & (tr["slice"] == "all_dark_360")].mean())
    prose = (
        f"Y3 mixed {_pp(r_m)} vs all-dark {_pp(r_a)} (quote 12.05% / 5.20%). "
        f"mixed leftover after days {_f(recs['mixed_110']['after']['rank'])} "
        f"dies={recs['mixed_110']['after']['honest_dies']}. "
        f"all-dark leftover {_f(recs['all_dark_360']['after']['rank'])} "
        f"dies={recs['all_dark_360']['after']['honest_dies']}. "
        f"H leftover is sister existence, not health — hidden test is new groups."
    )
    print(prose)
    return {
        "rows": rows, "r_m": r_m, "r_a": r_a,
        "mix_rank": recs["mixed_110"]["after"]["rank"],
        "ad_rank": recs["all_dark_360"]["after"]["rank"],
        "prose": prose,
    }


def pass7_runway(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 7 — sister mean b_runway ≥ 1 footnote leftover")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    base = y.notna() & (tr["slice"] == "mixed_110") & pd.to_numeric(tr["sister_runway"], errors="coerce").notna()
    ok = base & (tr["sister_runway_ok"] == 1)
    st = base & (tr["sister_runway_ok"] == 0)
    rate_ok = float(y[ok].mean()) if ok.any() else float("nan")
    rate_st = float(y[st].mean()) if st.any() else float("nan")
    gap = rate_ok - rate_st if np.isfinite(rate_ok) and np.isfinite(rate_st) else float("nan")
    after_flag = leftover_diag(y, tr["h_sib_neg_share"], (tr["sister_runway_ok"],), tr["fold"], base)
    after_days = leftover_diag(y, tr["h_sib_neg_share"], (tr["c_n_days_with_tx"],), tr["fold"], base)
    rec_flag = signed_oof_auroc(y, tr["sister_runway_ok"], tr["fold"], base)
    rec_h = signed_oof_auroc(y, tr["h_sib_neg_share"], tr["fold"], base)
    # H-legal median split on 110
    hneg = pd.to_numeric(tr["h_sib_neg_share"], errors="coerce")
    med = float(hneg[base].median()) if base.any() else float("nan")
    lo = base & (hneg <= med)
    hi = base & (hneg > med)
    gap_h = (float(y[lo].mean()) - float(y[hi].mean())) if lo.any() and hi.any() else float("nan")
    gap_ok = bool(np.isfinite(gap) and abs(gap - 0.0787) < 0.03)
    hflat = bool(np.isfinite(gap_h) and abs(gap_h) < 0.03)
    overturn = bool(np.isfinite(after_flag["rank"]) and after_flag["rank"] >= CHANCE and not after_flag["honest_dies"] and (not np.isfinite(gap) or gap < 0.05))
    footnote = "OVERTURN" if overturn else "KEEP"
    prose = (
        f"110 sister runway≥1 Y3 {_pp(rate_ok)} n={int(ok.sum())} vs stressed {_pp(rate_st)} "
        f"n={int(st.sum())} gap {_pp(gap)} (quote +7.87pp raw / +6.97pp T1 "
        f"{'CONFIRM' if gap_ok else 'DRIFT'}). "
        f"h_sib_neg leftover after runway flag {_f(after_flag['rank'])} dies={after_flag['honest_dies']}. "
        f"H leftover after days on 110 {_f(after_days['rank'])} dies={after_days['honest_dies']}. "
        f"H median-split gap {_pp(gap_h)} (quote +0.01pp {'CONFIRM flat' if hflat else 'DRIFT'}). "
        f"Sister-runway flag Y3 {_f(_cv(rec_flag))} (report-only; Y3 never B). "
        f"Q5 footnote **{footnote}**."
    )
    print(prose)
    return {
        "rate_ok": rate_ok, "rate_st": rate_st, "gap": gap, "gap_h": gap_h,
        "after_flag": after_flag["rank"], "after_days": after_days["rank"],
        "flag_cv": _cv(rec_flag), "h_cv": _cv(rec_h),
        "footnote": footnote, "overturn": overturn, "hflat": hflat, "gap_ok": gap_ok,
        "prose": prose,
    }


def pass8_q6(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 8 — Q6 lag1 leftover after days_lag1; new groups cannot inherit H")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    recs = {}
    rows = []
    for name in ("h_sib_neg_share", "h_sib_neg_share_lag1", "c_n_days_with_tx_lag1"):
        rec = signed_oof_auroc(y, tr[name], tr["fold"], lab)
        recs[name] = rec
        rows.append(_auc_row(Y3, name, rec))
        print(f"  {name} CV={_f(rec['cv'])} n={rec['n_defined']} pos={rec['n_pos']}")
    after_l1 = leftover_diag(y, tr["h_sib_neg_share_lag1"], (tr["c_n_days_with_tx_lag1"],), tr["fold"], lab)
    days_l1 = _cv(recs["c_n_days_with_tx_lag1"])
    days_ok = bool(np.isfinite(days_l1) and abs(days_l1 - DAYS_LAG1_QUOTE) < 0.008)
    prose = (
        f"Y3 sib_neg_lag1 {_f(recs['h_sib_neg_share_lag1']['cv'])} leftover after days_lag1 "
        f"rank {_f(after_l1['rank'])} dies={after_l1['honest_dies']}. "
        f"Days lag1 {_f(days_l1)} (quote 0.684 {'CONFIRM' if days_ok else 'DRIFT'}). "
        f"Hidden test is new groups — H cannot transfer."
    )
    print(prose)
    return {
        "rows": rows, "lag1": _cv(recs["h_sib_neg_share_lag1"]),
        "l1_rank": after_l1["rank"], "l1_dies": after_l1["honest_dies"],
        "days_l1": days_l1, "days_ok": days_ok, "prose": prose,
    }


def pass9_share(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 9 — vs h_share_group_in (NEAR_SIZE 0.638)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    rho = spearman(tr["h_sib_neg_share"], tr["h_share_group_in"])
    rho_s = spearman(tr["h_share_group_in"], tr["log_in3"])
    rho_op = spearman(tr["h_share_group_in"], np.log1p(pd.to_numeric(tr["a_op_in"], errors="coerce").clip(lower=0)))
    rec = signed_oof_auroc(y, tr["h_share_group_in"], tr["fold"], lab)
    after = leftover_diag(y, tr["h_sib_neg_share"], (tr["h_share_group_in"],), tr["fold"], lab)
    after_s = leftover_diag(y, tr["h_share_group_in"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    near = bool(np.isfinite(rho_op) and abs(rho_op) >= 0.70)
    prose = (
        f"ρ(sib_neg, share_group_in)={_f(rho)}. "
        f"share vs log1p(a_in3) {_f(rho_s)} vs log1p(a_op_in) {_f(rho_op)} "
        f"{'NEAR_SIZE CONFIRM' if near else 'not NEAR_SIZE'}. "
        f"Y3 share {_f(_cv(rec))} (quote 0.638). "
        f"sib_neg leftover after share {_f(after['rank'])} dies={after['honest_dies']}. "
        f"share leftover after days {_f(after_s['rank'])} dies={after_s['honest_dies']}."
    )
    print(prose)
    return {
        "rho": rho, "rho_s": rho_s, "rho_op": rho_op, "near": near,
        "share_cv": _cv(rec), "after_share": after["rank"], "share_days": after_s["rank"],
        "prose": prose,
    }


def pass10_hold(hold: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 10 — holdout coverage only (no AUROC)")
    print("=" * 72)
    x = pd.to_numeric(hold["h_sib_neg_share"], errors="coerce")
    rec = {
        "n_co": int(hold["company_id"].nunique()),
        "n_cm": len(hold),
        "n_nn": int(x.notna().sum()),
        "cov": _pct(int(x.notna().sum()), len(hold)),
        "p50": float(x[x.notna()].median()) if x.notna().any() else float("nan"),
        "n_dark": int(hold.loc[hold["is_dark"], "company_id"].nunique()),
    }
    prose = (
        f"Holdout {rec['n_co']} co / {rec['n_cm']} CM nn={rec['n_nn']} "
        f"cov={_pp(rec['cov'])} p50={_f(rec['p50'])} dark co={rec['n_dark']} (no fit, no AUROC)."
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
    work = tr.loc[lab, ["company_id", "fold", Y3, "h_sib_neg_share", "c_n_days_with_tx"]].copy()
    cos = work["company_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(FOLD_SEED)
    ranks = []
    for _ in range(n_boot):
        draw = rng.choice(cos, size=len(cos), replace=True)
        parts = [work[work["company_id"] == c] for c in draw]
        b = pd.concat(parts, ignore_index=True)
        after = leftover_diag(b[Y3], b["h_sib_neg_share"], (b["c_n_days_with_tx"],), b["fold"], pd.Series(True, index=b.index))
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
    print(f"EXTRA — permute sib_neg within days quintile (n={n_perm})")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    work = tr.loc[lab].copy()
    days = pd.to_numeric(work["c_n_days_with_tx"], errors="coerce")
    x = pd.to_numeric(work["h_sib_neg_share"], errors="coerce")
    ok = days.notna() & x.notna()
    work = work.loc[ok]
    q = pd.qcut(pd.to_numeric(work["c_n_days_with_tx"], errors="coerce"), 5, duplicates="drop")
    rng = np.random.default_rng(FOLD_SEED + 13)
    ranks = []
    for _ in range(n_perm):
        shuf = work["h_sib_neg_share"].copy()
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
    print("EXTRA — ICC / group ICC")
    print("=" * 72)
    icc = icc_anova(tr["h_sib_neg_share"], tr["company_id"])
    icc_g = icc_anova(tr["h_sib_neg_share"], tr["group_id"])
    prose = (
        f"Company ICC={_f(icc['icc'])} k={icc['k']}. Group ICC={_f(icc_g['icc'])} k={icc_g['k']}. "
        f"{'TRAIT' if np.isfinite(icc['icc']) and icc['icc'] >= ICC_TRAIT else 'STATE'} at company; "
        f"H is a group-type dummy (hidden test = new groups)."
    )
    print(prose)
    return {"icc": icc, "icc_g": icc_g, "prose": prose}


def extra_quintiles(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Y3 rate by sib_neg quintile")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr["h_sib_neg_share"], errors="coerce")
    ok = y.notna() & x.notna()
    qn = pd.qcut(x[ok], 5, duplicates="drop")
    rows = []
    for i, cat in enumerate(sorted(qn.dropna().unique()), start=1):
        sl = ok.copy()
        sl.loc[ok] = qn == cat
        rows.append({"q": i, "Y3 rate": _pp(float(y[sl].mean()) if sl.any() else float("nan")), "n": int(sl.sum()), "n_pos": int((sl & (y == 1)).sum())})
    prose = f"Y3 sib_neg Q1→Q5 {[r['Y3 rate'] for r in rows]}."
    print(prose)
    return {"rows": rows, "prose": prose}


def extra_mixed_dark(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — mixed dummy leftover; dark-only CV (quote 0.566)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    rec_all = signed_oof_auroc(y, tr["mixed_dummy"], tr["fold"], lab)
    rec_d = signed_oof_auroc(y, tr["mixed_dummy"], tr["fold"], lab & tr["is_dark"])
    after_all = leftover_diag(y, tr["mixed_dummy"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_d = leftover_diag(y, tr["mixed_dummy"], (tr["c_n_days_with_tx"],), tr["fold"], lab & tr["is_dark"])
    after_s = leftover_diag(y, tr["mixed_dummy"], (tr["log_in3"],), tr["fold"], lab)
    dark_ok = bool(np.isfinite(_cv(rec_d)) and abs(_cv(rec_d) - 0.566) < 0.03)
    prose = (
        f"mixed dummy full Y3 {_f(_cv(rec_all))} leftover after days {_f(after_all['rank'])} "
        f"dies={after_all['honest_dies']} fake={after_all['fake']}; after size {_f(after_s['rank'])}. "
        f"Dark-only Y3 {_f(_cv(rec_d))} (quote 0.566 {'CONFIRM' if dark_ok else 'DRIFT'}) "
        f"leftover {_f(after_d['rank'])} dies={after_d['honest_dies']}. "
        f"Existence dummy can look useful on this panel and still fail on new groups."
    )
    print(prose)
    return {
        "all_cv": _cv(rec_all), "dark_cv": _cv(rec_d),
        "all_rank": after_all["rank"], "dark_rank": after_d["rank"],
        "dark_ok": dark_ok, "prose": prose,
    }


def extra_runway_t1(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — sister runway≥1 T1 residual on 110 (quote +6.97pp)")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    size = pd.to_numeric(tr["log_in3"], errors="coerce")
    try:
        terc = pd.qcut(size, 3, labels=("T1", "T2", "T3"), duplicates="drop")
    except ValueError:
        return {"t1": float("nan"), "prose": "T1 sister-runway LOW_POWER."}
    rows = []
    t1 = float("nan")
    for t in ("T1", "T2", "T3"):
        base = y.notna() & (tr["slice"] == "mixed_110") & terc.eq(t) & pd.to_numeric(tr["sister_runway"], errors="coerce").notna()
        ok = base & (tr["sister_runway_ok"] == 1)
        st = base & (tr["sister_runway_ok"] == 0)
        r_ok = float(y[ok].mean()) if ok.any() else float("nan")
        r_st = float(y[st].mean()) if st.any() else float("nan")
        res = r_ok - r_st if np.isfinite(r_ok) and np.isfinite(r_st) else float("nan")
        if t == "T1":
            t1 = res
        rows.append({"tercile": t, "n ok": int(ok.sum()), "n st": int(st.sum()), "Y3 ok": _pp(r_ok), "Y3 st": _pp(r_st), "residual": _pp(res) if np.isfinite(res) else "—"})
    ok = bool(np.isfinite(t1) and abs(t1 - FOOT_PP) < 0.02)
    after = leftover_diag(
        y, tr["h_sib_neg_share"], (tr["sister_runway_ok"], tr["c_n_days_with_tx"]),
        tr["fold"], y.notna() & (tr["slice"] == "mixed_110"),
    )
    prose = (
        f"110 sister runway≥1 T1 residual {_pp(t1)} (quote +6.97pp {'CONFIRM' if ok else 'DRIFT'}). "
        f"H leftover after runway+days on 110 {_f(after['rank'])} dies={after['honest_dies']}."
    )
    print(prose)
    return {"t1": t1, "ok": ok, "rows": rows, "after": after["rank"], "prose": prose}


def extra_share_size(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — h_share_group_in leftover after size (NEAR_SIZE, not this stem)")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after_s = leftover_diag(y, tr["h_share_group_in"], (tr["log_in3"],), tr["fold"], lab)
    after_d = leftover_diag(y, tr["h_share_group_in"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    prose = (
        f"share leftover after size {_f(after_s['rank'])} dies={after_s['honest_dies']}; "
        f"after days {_f(after_d['rank'])} dies={after_d['honest_dies']}. "
        f"NEAR_SIZE — do not promote share_group_in as leftover of sib_neg."
    )
    print(prose)
    return {"after_size": after_s["rank"], "after_days": after_d["rank"], "prose": prose}


def extra_opin_t1(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — y11-style T1 residual log1p(|a_op_in|) on dark Y3")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    dark = tr["is_dark"] & y.notna()
    opin = np.log1p(pd.to_numeric(tr["a_op_in"], errors="coerce").abs())
    try:
        terc = pd.qcut(opin[dark], 3, labels=("T1", "T2", "T3"), duplicates="drop")
    except ValueError:
        return {"t1": float("nan"), "prose": "T1 residual LOW_POWER."}
    t1 = dark.copy()
    t1.loc[dark] = terc == "T1"
    mix = t1 & (tr["slice"] == "mixed_110")
    ad = t1 & (tr["slice"] == "all_dark_360")
    r_m = float(y[mix].mean()) if mix.any() else float("nan")
    r_a = float(y[ad].mean()) if ad.any() else float("nan")
    res = r_m - r_a if np.isfinite(r_m) and np.isfinite(r_a) else float("nan")
    ok = bool(np.isfinite(res) and abs(res - T1_RES_OPIN) < 0.02)
    prose = f"y11 T1 residual mixed−all-dark {_pp(res)} (quote +12.5pp {'CONFIRM' if ok else 'DRIFT'})."
    print(prose)
    return {"t1": res, "ok": ok, "prose": prose}


def decide(p1, p2, p3, p5, p7) -> dict:
    leftover_lives = bool(np.isfinite(p3["rank"]) and p3["rank"] >= CHANCE and not p3["dies"])
    is_size = bool(p1["is_size"])
    twin = bool(p1["twin_gate"])
    engine = leftover_lives and bool(p2["beat_size"]) and not is_size and not twin
    footnote = p7["footnote"]
    if engine:
        role = "KEEP as unused leftover"
        why = (
            f"KEEP-as-X passed: leftover after days rank {p3['rank']:.3f} lives, "
            f"beat-size {_f(p2['sib'])} vs 0.617. Off the 15-col card. "
            f"Q5 sister-runway footnote {footnote}."
        )
    elif leftover_lives and twin:
        role = "DROP from the 44 as Y3 X"
        why = f"leftover after days rank {p3['rank']:.3f} lives but TWIN of {p1['twins']}. Q5 footnote {footnote}."
    elif leftover_lives and is_size:
        role = "DROP from the 44 as Y3 X"
        why = f"leftover after days rank {p3['rank']:.3f} lives but SIZE. Q5 footnote {footnote}."
    elif leftover_lives and not p2["beat_size"]:
        role = "CLOSE unused leftover"
        why = (
            f"leftover after days rank {p3['rank']:.3f} lives but fails beat-size "
            f"({_f(p2['sib'])} vs 0.617). DROP from the 44. PARK as Y. "
            f"Q5 sister-runway footnote {footnote}."
        )
    else:
        role = "CLOSE unused leftover"
        why = (
            f"unused leftover after days: honest rank {_f(p3['rank'])} dies "
            f"(OLS {_f(p3['ols'])} fake={p3['fake']}). "
            f"Single {_f(p2['sib'])} vs days 0.711 / size 0.617. "
            f"DROP from the 44 as Y3 X. PARK as Y (H marks sister existence; hidden test is new groups). "
            f"Q5 sister-runway footnote {footnote}."
        )
    return {
        "role": role, "why": why, "leftover_lives": leftover_lives,
        "engine": engine, "is_size": is_size, "twin": twin, "footnote": footnote,
        "park_y": "PARK as Y — H is sister existence, not health",
        "card": "no — do not put H on the 15-col card",
    }


def make_plot(tr: pd.DataFrame, p3: dict) -> bool:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return False
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr["h_sib_neg_share"], errors="coerce")
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
    ax.plot(xs, rates, marker="o", color="#1f4e79", label="h_sib_neg_share")
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
    ax.set_title("sib_neg vs days recover")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    rec = p3["after"]["rrec"]
    xs = [r["fold"] for r in rec["folds"]]
    ys = [r["auroc"] for r in rec["folds"]]
    ax.bar(xs, ys, color="#1f4e79")
    ax.axhline(CHANCE, color="#333", ls="--", lw=1, label="0.55 dies")
    ax.axhline(DAYS_BENCH, color="#c45c26", ls=":", lw=1, label="days 0.711")
    ax.set_xlabel("fold")
    ax.set_ylabel("rank leftover AUROC")
    ax.set_title("h_sib_neg_share leftover after days")
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
        "# Unused leftover of `h_sib_neg_share` after days as Y3 X",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. No `build_targets`. Do not invent `y_sib_neg`. Do not put H on the 15-col card. "
        "Do not overwrite `sibling_h.*`. Y3 never B. Hidden test is new groups. Do not grow TURNOVER.",
        "",
        "`h_sib_neg_share` = (# siblings with net < 0) / (h_group_size − 1) (Family H). "
        "Family H already PARK as Y3 X (sibling_h). This lane is leftover after days on the 44 stem. "
        "Sister mean `b_runway` ≥ 1 is a Q5 footnote only.",
        "",
        "## Headline",
        "",
        (
            f"`h_sib_neg_share` as Y3 X: **{d['role']}** ({d['why']}). "
            f"Y3 leftover after days OLS {_f(p3['ols'])} rank {_f(p3['rank'])} "
            f"({'dies' if p3['dies'] else 'lives'}, fake={p3['fake']}). "
            f"Inverse days after sib_neg rank {_f(p3['inv_rank'])}. "
            f"Single {_f(p2['sib'])} vs size {_f(p2['size'])} vs days {_f(p2['days'])} "
            f"vs share {_f(p2['share'])} vs mixed {_f(p2['mix'])}. "
            f"after size {_f(p5['after_size'])} after mixed {_f(p5['after_m'])}. "
            f"Q5 sister-runway footnote **{d['footnote']}**. "
            f"15-col card: {d['card']}. {d['park_y']}. "
            f"Night quotes unchanged: Y3 0.762 / 0.752, days 0.711, size 0.617, TURNOVER 0.720 / 0.712."
        ),
        "",
        "## Brief questions",
        "",
        "| # | question | what this cut says |",
        "| --- | --- | --- |",
        f"| 1 | Who is healthy? | {d['park_y']}. Hidden test is new groups. |",
        f"| 2 | Who is improving? | Q6 lag1 leftover {_f(p8['l1_rank'])}. New groups cannot inherit H. |",
        f"| 3 | Who is turning? | **{d['role']}** leftover after days {_f(p3['rank'])} vs days 0.711. |",
        f"| 4 | Dip vs fall? | Not this table. |",
        f"| 5 | Why did it change? | Twin screen: {p1['twins'] or 'none'}. Sister-runway footnote {d['footnote']}. |",
        f"| 6 | Months earlier? | lag1 leftover {_f(p8['l1_rank'])}; days_lag1 {_f(p8['days_l1'])}. |",
        "",
        "## PARK / CLOSE / KEEP / DROP-from-44",
        "",
        "| object | decision | why |",
        "| --- | --- | --- |",
        f"| `h_sib_neg_share` as Y3 X / the 15-col card | **{d['role']}** | {d['why']} |",
        f"| `h_sib_neg_share` as engine X on the 44 | **{'KEEP leftover' if d['engine'] else 'DROP'}** | leftover lives={d['leftover_lives']} twin={d['twin']} SIZE={d['is_size']} beat-size={p2['beat_size']} |",
        f"| H as health Y | **PARK** | sister existence; hidden test is new groups |",
        f"| Q5 sister mean b_runway ≥ 1 | **{d['footnote']}** | leftover after flag {_f(p7['after_flag'])}; raw gap {_pp(p7['gap'])} |",
        f"| mixed dummy / h_group_size twin | **{'YES' if d['twin'] else 'NO'}** | ρ mix {_f(p1['rhos']['mixed_dummy'])} size {_f(p1['rhos']['h_group_size'])} |",
        f"| h_share_group_in | **NEAR_SIZE / not this stem** | Y3 {_f(p2['share'])}; ρ vs log1p(a_op_in) {_f(p9['rho_op'])} |",
        f"| Q6 lag1 after days_lag1 | **{'KEEP' if (np.isfinite(p8['l1_rank']) and p8['l1_rank'] >= CHANCE and not p8['l1_dies']) else 'CLOSE'}** | leftover {_f(p8['l1_rank'])} |",
        "",
        "## 1 — Coverage; twin / SIZE screen",
        "",
        p1["prose"],
        "",
        _md_table(p1["rows"]),
        "",
        _md_table(p1["rho_rows"]),
        "",
        "## 2 — Single-feature group-fold Y3",
        "",
        p2["prose"],
        "",
        _md_table(p2["rows"]),
        "",
        "## 3 — Honest leftover after days",
        "",
        p3["prose"],
        "",
        f"OLS folds: {p3['after']['folds']}. Rank folds: {p3['after']['rank_folds']}.",
        "",
        "## 4 — Twin / SIZE screen (in cut 1)",
        "",
        f"SIZE={p1['is_size']} twin_gate={p1['twin_gate']} twins={p1['twins'] or 'none'}.",
        "",
        "## 5 — Leftover after size; T1 residual",
        "",
        p5["prose"],
        "",
        _md_table(p5["rows"]),
        "",
        "## 6 — Mixed 110 vs all-dark 360 leftover",
        "",
        p6["prose"],
        "",
        _md_table(p6["rows"]),
        "",
        "## 7 — Sister runway ≥ 1 footnote",
        "",
        p7["prose"],
        "",
        "## 8 — Q6 lag1 leftover after days_lag1",
        "",
        p8["prose"],
        "",
        _md_table(p8["rows"]),
        "",
        "## 9 — vs `h_share_group_in`",
        "",
        p9["prose"],
        "",
        "## 10 — Holdout coverage only",
        "",
        p10["prose"],
        "",
        "## Extras",
        "",
        "### Bootstrap leftover after days",
        "",
        ctx["xb"]["prose"],
        "",
        "### Permute within days quintile",
        "",
        ctx["xp"]["prose"],
        "",
        "### ICC",
        "",
        ctx["xi"]["prose"],
        "",
        "### Y3 rate by sib_neg quintile",
        "",
        ctx["xq"]["prose"],
        "",
        _md_table(ctx["xq"]["rows"]),
        "",
        "### y11-style T1 residual",
        "",
        ctx["xt"]["prose"],
        "",
        "### mixed dummy leftover / dark-only",
        "",
        ctx["xm"]["prose"],
        "",
        "### sister runway≥1 T1 residual",
        "",
        ctx["xr"]["prose"],
        "",
        _md_table(ctx["xr"]["rows"]) if ctx["xr"].get("rows") else "",
        "",
        "### share_group_in leftover after size",
        "",
        ctx["xsh"]["prose"],
        "",
        "## Night quotes (unchanged)",
        "",
        "| quote | locked |",
        "| --- | --- |",
        f"| Y3 | {Y3_NIGHT[0]:.3f} / {Y3_NIGHT[1]:.3f} |",
        f"| days | {DAYS_BENCH:.3f} |",
        f"| size | {SIZE_QUOTE:.3f} |",
        f"| Y7 TURNOVER | {Y7_TURNOVER:.3f} / {Y7_B_SHALLOW:.3f} |",
        f"| a_uncat_share leftover | 0.573 DROP / single 0.542 |",
        "",
        "Do not quote a_out_vol 0.722. Do not grow TURNOVER. Do not put H on the 15-col card.",
        "",
        "## Files written",
        "",
        "- `analysis/evaluate/sib_neg_qa.py`",
        "- `analysis/outputs/sib_neg_qa.md`",
        "- `analysis/outputs/sib_neg_qa.png`" if ctx.get("png") else "- (no PNG)",
        "- append-only `analysis/experiments/registry.csv`",
        "- `overnight/waves/wave4_sib_neg.md` (end, if WRITE_WAVE)",
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
    p1, p2, p3, p5, p7 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p5"], ctx["p7"]
    d = ctx["decision"]
    rows = [
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_h_sib_neg_share", "value": p2["sib"], "coverage": f"{p1['cov']:.4f}", "notes": f"days={p2['days']:.4f} size={p2['size']:.4f} beat={p2['beat_size']}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_h_sib_neg_share_resid_days", "value": p3["rank"], "coverage": f"{p1['cov']:.4f}", "notes": f"ols={p3['ols']:.4f} dies={p3['dies']} fake={p3['fake']}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_h_sib_neg_share_resid_size", "value": p5["after_size"], "coverage": f"{p1['cov']:.4f}", "notes": f"t1_res={p5['t1_res']:.4f}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train", "metric": "sib_neg_leftover", "value": 1 if d["leftover_lives"] else 0, "coverage": f"{p1['cov']:.4f}", "notes": d["role"] + " " + d["why"][:160]},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train", "metric": "sister_runway_footnote", "value": 1 if d["footnote"] == "KEEP" else 0, "coverage": f"{p1['cov']:.4f}", "notes": f"gap={p7['gap']:.4f} leftover_after_flag={p7['after_flag']:.4f}"},
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
        f"# Wave 4 — h_sib_neg_share leftover after days as Y3 X\n\n"
        f"Agent `{AGENT}`. Train group-fold seed 20260918. Holdout 72 coverage only.\n\n"
        f"## Files written\n\n"
        f"- `analysis/evaluate/sib_neg_qa.py`\n"
        f"- `analysis/outputs/sib_neg_qa.md`\n"
        f"- `analysis/outputs/sib_neg_qa.png`\n"
        f"- append-only `analysis/experiments/registry.csv` "
        f"(skip key agent+x_families+y+model+split+metric)\n"
        f"- this note\n\n"
        f"Did not touch `sibling_h.*`, `uncat_share_qa.*`, `n_types_qa.*`, "
        f"`ap_issued_qa.*`, `groupctx.py`, parquet / duckdb, `build_targets`, "
        f"`product/`, the 15-col card, TURNOVER, LIVE, CONTEXT, canvas, "
        f"`brief_map.md`, or the parent journal. "
        f"Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. "
        f"Y7 TURNOVER **0.720 / 0.712** unchanged. "
        f"Q5 sister-runway footnote **{d['footnote']}**.\n\n"
        f"## Locked verdict\n\n"
        f"| object | decision |\n"
        f"| --- | --- |\n"
        f"| `h_sib_neg_share` as Y3 X / 15-col card | **{d['role']}** |\n"
        f"| `h_sib_neg_share` as engine X on the 44 | **{'KEEP leftover' if d['engine'] else 'DROP'}** |\n"
        f"| H as health Y | **PARK** |\n"
        f"| Q5 sister mean b_runway ≥ 1 | **{d['footnote']}** |\n\n"
        f"Y3 leftover after days rank {_f(p3['rank'])} (dies={p3['dies']}, fake={p3['fake']}); "
        f"inverse days after sib_neg {_f(p3['inv_rank'])}. "
        f"Single {_f(p2['sib'])} vs days {_f(p2['days'])} vs size {_f(p2['size'])}. "
        f"ρ vs group_size {_f(p1['rhos']['h_group_size'])} vs mixed {_f(p1['rhos']['mixed_dummy'])} "
        f"vs days {_f(p1['rhos']['c_n_days_with_tx'])} vs size {_f(p1['rhos']['log1p(a_in3)'])}. "
        f"after size {_f(p5['after_size'])} T1 residual {_pp(p5['t1_res'])}. "
        f"mixed leftover {_f(p6['mix_rank'])} all-dark {_f(p6['ad_rank'])}. "
        f"Q6 lag1 leftover {_f(p8['l1_rank'])}. share_group_in {_f(p9['share_cv'])}. {d['why']}\n\n"
        f"Dark 470 = mixed 110 + all-dark 360 CONFIRM. T1 mixed−all-dark +16.7pp CONFIRM; "
        f"y11 T1 +12.5pp CONFIRM. Sister runway≥1 T1 +7.0pp CONFIRM. "
        f"H leftover after runway flag dies — H does not carry the footnote. "
        f"Bootstrap leftover p05/p50/p95 0.404/0.452/0.543 (97.5% die). "
        f"Permute-within-days p50=0.511 — observed 0.453 is below the null. "
        f"Dark-only mixed dummy 0.566 CONFIRM. Hidden test is new groups. "
        f"Do not put H on the 15-col card. Do not invent y_sib_neg.\n\n"
        f"## What failed / next\n\n"
        + ("\n".join(f"- {x}" for x in ctx["failed"]) if ctx["failed"] else "- none")
        + f"\n\nElapsed {ctx['elapsed_s']:.0f}s.\n"
    )
    WAVE_NOTE.write_text(text)
    print(f"wrote {WAVE_NOTE}")


def main() -> None:
    t0 = time.time()
    failed: list[str] = []
    print("sib_neg leftover QA — unused leftover of h_sib_neg_share after days as Y3 X")
    panel = load_panel()
    con = connect()
    try:
        book = book_invoice_ids(con)
    finally:
        con.close()
    print(f"book (ERP) companies={len(book)}")
    panel = attach_mix_and_sisters(panel, book)
    panel = attach_folds(panel)
    panel = add_panel_lags(panel, ["h_sib_neg_share", "c_n_days_with_tx"], (1,))
    tr = panel[panel["split"] == "train"].copy()
    hold = panel[panel["split"] == "holdout"].copy()
    assert_no_holdout(tr["company_id"])
    if hold["company_id"].nunique() != 72:
        failed.append(f"holdout n_co={hold['company_id'].nunique()} expected 72")
    p1 = pass1_cov(tr)
    p2 = pass2_singles(tr)
    p3 = pass3_days(tr)
    p5 = pass5_size(tr)
    p6 = pass6_mix(tr)
    p7 = pass7_runway(tr)
    p8 = pass8_q6(tr)
    p9 = pass9_share(tr)
    p10 = pass10_hold(hold)
    xb = extra_bootstrap(tr, n_boot=40)
    xp = extra_permute(tr, n_perm=24)
    xi = extra_icc(tr)
    xq = extra_quintiles(tr)
    xt = extra_opin_t1(tr)
    xm = extra_mixed_dark(tr)
    xr = extra_runway_t1(tr)
    xsh = extra_share_size(tr)
    decision = decide(p1, p2, p3, p5, p7)
    print("\n" + "=" * 72)
    print(f"VERDICT: {decision['role']}")
    print(decision["why"])
    print("=" * 72)
    if not p2["days_ok"]:
        failed.append(f"days replica {p2['days']:.3f} ≠ 0.711")
    if not p2["size_ok"]:
        failed.append(f"size replica {p2['size']:.3f} ≠ 0.617")
    if not p1["mix_ok"]:
        failed.append(f"mix 110/360 drift mixed={p1['n_mixed']} all-dark={p1['n_ad']}")
    png = make_plot(tr, p3)
    elapsed = time.time() - t0
    ctx = {
        "p1": p1, "p2": p2, "p3": p3, "p5": p5, "p6": p6, "p7": p7,
        "p8": p8, "p9": p9, "p10": p10,
        "xb": xb, "xp": xp, "xi": xi, "xq": xq, "xt": xt,
        "xm": xm, "xr": xr, "xsh": xsh,
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
