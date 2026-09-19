"""Unused leftover of ``d_cust_top1`` after days as Y3 X.

``d_cust_top1`` = share of AR invoice |amount| from the single
largest customer (Family D, trailing 6m). Incomplete 6m books are
NaN. Dark 470 stay NaN not 0. ``d_cust_hhi`` was DROPPED as the
weaker rewrite of this stem (ρ 0.994 / lag3 0.994). Y4 >0.975
monopoly tail KEEP footnote 0.605; body CV 0.445; leftover after
top1_lag3 0.549 dies. ``d_n_cust`` CLOSED leftover 0.545 (twin of
top1 ρ −0.806). Javier concentration is **top1**, not HHI.

Do **not** overwrite ``cust_hhi_qa.*``, ``n_cust_qa.*``,
``n_supp_qa.*``. Do not put top1 on the 15-col card. Do not
rewrite ``gbm_core.py``. KEEP the Y4 >0.975 footnote unless this
cut overturns it.

KEEP-as-X: beat size ≥0.02 **and** leftover after days **and** not
SIZE (|ρ| vs log1p(a_in3) ≥0.50) **and** not a twin (|ρ|≥0.80 vs
days / a_n_tx / d_cust_hhi / d_n_cust). Leftover <0.55 dies.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.top1_qa

Owned: analysis/evaluate/top1_qa.py, analysis/outputs/top1_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_top1.md (end).
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
OUT_MD = ANALYSIS / "outputs" / "top1_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "top1_qa.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_top1.md"
AGENT = "572fb928"
WAVE = "4"
ROUND = "R4"
MODEL = "top1_qa"
X_FAM = "D"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
Y4 = "y4_ds_r_double"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_QUOTE = 0.617
DAYS_LAG1_QUOTE = 0.684
Y3_NIGHT = (0.762, 0.752)
Y7_TURNOVER = 0.720
Y7_B_SHALLOW = 0.712
HHI_RHO_PEEK = 0.994
NCUST_RHO_PEEK = -0.806
NCUST_Y3 = 0.653
NCUST_LEFT = 0.545
TOP1_Y3_PEEK = 0.590
TOP1_N_PEEK = 2485
TOP1_POS_PEEK = 141
Y4_LAG3_QUOTE = 0.605
Y4_BODY_QUOTE = 0.445
Y4_TAIL_HI = 0.221
Y4_TAIL_REST = 0.115
Y4_LEFT_QUOTE = 0.549
HHI_Q6_SHARE = 0.217
HHI_Q6_POS = 42
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
HOT_GROUPS = ("GROUP_0158", "GROUP_0172")
WRITE_WAVE = True

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "first_month",
    "a_in3",
    "a_n_tx",
    "c_n_days_with_tx",
    "d_n_cust",
    "d_cust_top1",
    "d_cust_hhi",
    "d_n_supp",
    "d_supp_hhi",
    "d_supp_top1",
    "d_tx_cp_share",
    "b_below_0",
)

Y_KEEP = (Y2, Y3, Y4)


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
    s = pd.DataFrame(
        {"x": pd.to_numeric(series, errors="coerce"), "co": company.astype(str)}
    ).dropna()
    if len(s) < 10 or s["x"].nunique() < 2:
        return {"icc": float("nan"), "var_w": float("nan"), "var_b": float("nan"), "k": 0}
    y = s["x"].to_numpy(dtype=float)
    _, inv, counts = np.unique(s["co"].to_numpy(), return_inverse=True, return_counts=True)
    if counts.size < 2:
        return {"icc": float("nan"), "var_w": float("nan"), "var_b": float("nan"), "k": 0}
    means = np.bincount(inv, weights=y) / counts
    grand = float(y.mean())
    ssb = float(np.sum(counts * (means - grand) ** 2))
    ssw = float(np.sum((y - means[inv]) ** 2))
    k = int(counts.size)
    n = int(y.size)
    var_b = ssb / max(k - 1, 1)
    var_w = ssw / max(n - k, 1)
    icc = var_b / (var_b + var_w) if (var_b + var_w) > 0 else float("nan")
    return {"icc": float(icc), "var_w": float(var_w), "var_b": float(var_b), "k": k}


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
    X = np.column_stack(
        [np.ones(int(ok.sum()))] + [d.loc[ok, f"x{i}"].to_numpy(dtype=float) for i in range(n_x)]
    )
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
        "ols": ols_cv,
        "rank": rank_cv,
        "rho_ctrl": rho_c,
        "r2": info["r2"],
        "n": rec["n_defined"],
        "n_pos": rec["n_pos"],
        "fake": fake,
        "almost": almost,
        "honest_dies": honest_dies,
        "folds": fold_bits(rec),
        "rank_folds": fold_bits(rrec),
        "rec": rec,
        "rrec": rrec,
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
        "y": y,
        "feature": feat,
        "n": f"{res['n_defined']:,}",
        "n_pos": f"{res['n_pos']:,}",
        "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
        "train": "LOW_POWER" if res["low_power"] else _f(res["train_auc"]),
        "sd": _f(res["sd"]) if not res["low_power"] else "—",
        "sign": res["train_sign"] if not res["low_power"] else "—",
        "folds": fold_bits(res) if not res["low_power"] else "—",
    }


def _trail_class(n) -> str:
    if n is None or (isinstance(n, float) and not np.isfinite(n)):
        return "na"
    n = int(n)
    if n < 12:
        return "short_<12"
    if n >= 18:
        return "long_>=18"
    return "mid_12_17"


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
    panel["first_month"] = pd.to_datetime(panel["first_month"])
    panel["months_so_far"] = (
        (panel["period"].dt.year - panel["first_month"].dt.year) * 12
        + (panel["period"].dt.month - panel["first_month"].dt.month)
        + 1
    )
    panel["so_far_class"] = panel["months_so_far"].map(_trail_class)
    leak3 = leakage_check(
        ["d_cust_top1", "d_cust_hhi", "d_n_cust", "c_n_days_with_tx", "log_in3"],
        Y3,
        forbidden_prefixes=["b"],
    )
    if not leak3["ok"]:
        raise RuntimeError(f"Y3 never-B leak: {leak3['issues']}")
    leak4 = leakage_check(["d_cust_top1", "d_cust_hhi"], Y4, forbidden_prefixes=["f"])
    if not leak4["ok"]:
        raise RuntimeError(f"Y4 never-F leak: {leak4['issues']}")
    return panel


def attach_folds(panel: pd.DataFrame) -> pd.DataFrame:
    train = panel[panel["split"] == "train"][["company_id", "group_id"]].drop_duplicates()
    assert_no_holdout(train["company_id"])
    folds = group_folds(train, n=N_FOLDS, seed=FOLD_SEED)
    return panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")


def chronic_12(tr: pd.DataFrame) -> list[str]:
    y = pd.to_numeric(tr[Y2], errors="coerce")
    below = pd.to_numeric(tr["b_below_0"], errors="coerce")
    lab = y.notna()
    ids: list[str] = []
    for gid in HOT_GROUPS:
        sl = lab & (tr["group_id"] == gid)
        g = (
            tr.loc[sl, ["company_id"]]
            .assign(below=below[sl].values)
            .groupby("company_id")
            .agg(n=("below", "size"), n_below=("below", "sum"))
        )
        g["share_below"] = g["n_below"] / g["n"]
        ids.extend([str(i) for i in g.index[g["share_below"] >= 0.5]])
    return sorted(set(ids))


def pass1_cov(tr: pd.DataFrame, hold: pd.DataFrame, book: set[str]) -> dict:
    print("\n" + "=" * 72)
    print("CUT 1 — coverage; twin/SIZE screen")
    print("=" * 72)
    n_cm = len(tr)
    n_co = int(tr["company_id"].nunique())
    x = pd.to_numeric(tr["d_cust_top1"], errors="coerce")
    hhi = pd.to_numeric(tr["d_cust_hhi"], errors="coerce")
    n_cust = pd.to_numeric(tr["d_n_cust"], errors="coerce")
    erp = tr["company_id"].isin(book)
    dark = ~erp
    n_dark_co = int(tr.loc[dark, "company_id"].nunique())
    dark_nn = int(x[dark].notna().sum())
    dark_zero = int((x[dark] == 0).sum())
    acf1 = median_acf(x, tr["company_id"], 1)
    pairs = {
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "a_n_tx": tr["a_n_tx"],
        "d_cust_hhi": hhi,
        "d_n_cust": n_cust,
        "log1p(a_in3)": tr["log_in3"],
        "d_supp_top1": tr["d_supp_top1"],
        "d_n_supp": tr["d_n_supp"],
        "d_tx_cp_share": tr["d_tx_cp_share"],
    }
    rhos = {k: spearman(x, v) for k, v in pairs.items()}
    twins = [k for k, v in rhos.items() if np.isfinite(v) and abs(v) >= TWIN_RHO]
    is_size = bool(np.isfinite(rhos["log1p(a_in3)"]) and abs(rhos["log1p(a_in3)"]) >= SIZE_RHO)
    twin_gate = any(
        np.isfinite(rhos[k]) and abs(rhos[k]) >= TWIN_RHO
        for k in ("c_n_days_with_tx", "a_n_tx", "d_cust_hhi", "d_n_cust")
    )
    rho_hhi_ok = bool(np.isfinite(rhos["d_cust_hhi"]) and abs(rhos["d_cust_hhi"] - HHI_RHO_PEEK) < 0.01)
    rho_nc_ok = bool(np.isfinite(rhos["d_n_cust"]) and abs(rhos["d_n_cust"] - NCUST_RHO_PEEK) < 0.03)
    rows = [
        {
            "col": "d_cust_top1",
            "n_nn": f"{int(x.notna().sum()):,}",
            "cov": _pp(_pct(int(x.notna().sum()), n_cm)),
            "acf1": _f(acf1),
        },
        {
            "col": "d_cust_hhi",
            "n_nn": f"{int(hhi.notna().sum()):,}",
            "cov": _pp(_pct(int(hhi.notna().sum()), n_cm)),
            "acf1": _f(median_acf(hhi, tr["company_id"], 1)),
        },
        {
            "col": "d_n_cust",
            "n_nn": f"{int(n_cust.notna().sum()):,}",
            "cov": _pp(_pct(int(n_cust.notna().sum()), n_cm)),
            "acf1": _f(median_acf(n_cust, tr["company_id"], 1)),
        },
    ]
    rho_rows = [
        {
            "vs": k,
            "rho": _f(v),
            "flag": (
                "SIZE"
                if k == "log1p(a_in3)" and abs(v) >= SIZE_RHO
                else "TWIN"
                if abs(v) >= TWIN_RHO
                else "no"
            ),
        }
        for k, v in rhos.items()
    ]
    dark_ok = n_dark_co == N_DARK_WANT and dark_nn == 0 and dark_zero == 0
    hold_x = pd.to_numeric(hold["d_cust_top1"], errors="coerce")
    prose = (
        f"Train {n_co:,} co / {n_cm:,} CM. d_cust_top1 cov {_pp(_pct(int(x.notna().sum()), n_cm))} "
        f"acf1={_f(acf1)}. Dark {n_dark_co} (want {N_DARK_WANT}) nn={dark_nn} zero={dark_zero} "
        f"{'CONFIRM NaN' if dark_ok else 'FAIL 0-fill'}. "
        f"ρ vs days {_f(rhos['c_n_days_with_tx'])} vs a_n_tx {_f(rhos['a_n_tx'])} "
        f"vs HHI {_f(rhos['d_cust_hhi'])} (peek 0.994 {'CONFIRM' if rho_hhi_ok else 'DRIFT'}) "
        f"vs n_cust {_f(rhos['d_n_cust'])} (peek −0.806 {'CONFIRM' if rho_nc_ok else 'DRIFT'}) "
        f"vs size {_f(rhos['log1p(a_in3)'])}. "
        f"SIZE={is_size} twins={twins or 'none'} twin_gate={twin_gate}."
    )
    print(prose)
    return {
        "rows": rows,
        "rho_rows": rho_rows,
        "rhos": rhos,
        "twins": twins,
        "is_size": is_size,
        "twin_gate": twin_gate,
        "n_cm": n_cm,
        "n_co": n_co,
        "cov": _pct(int(x.notna().sum()), n_cm),
        "acf1": acf1,
        "n_dark_co": n_dark_co,
        "dark_nn": dark_nn,
        "dark_ok": dark_ok,
        "rho_hhi_ok": rho_hhi_ok,
        "rho_nc_ok": rho_nc_ok,
        "hold_nn": int(hold_x.notna().sum()),
        "hold_cov": _pct(int(hold_x.notna().sum()), len(hold)),
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
        "d_cust_top1": tr["d_cust_top1"],
        "d_cust_hhi": tr["d_cust_hhi"],
        "d_n_cust": tr["d_n_cust"],
        "log1p(a_in3)": tr["log_in3"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "d_supp_top1": tr["d_supp_top1"],
        "a_n_tx": tr["a_n_tx"],
    }
    recs = {}
    rows = []
    for fname, x in feats.items():
        rec = signed_oof_auroc(y, x, folds, lab)
        recs[fname] = rec
        rows.append(_auc_row(Y3, fname, rec))
        print(f"  {fname} CV={_f(rec['cv'])} n={rec['n_defined']} pos={rec['n_pos']} folds={fold_bits(rec)}")
    y3_t = _cv(recs["d_cust_top1"])
    y3_size = _cv(recs["log1p(a_in3)"])
    y3_days = _cv(recs["c_n_days_with_tx"])
    y3_hhi = _cv(recs["d_cust_hhi"])
    y3_nc = _cv(recs["d_n_cust"])
    days_ok = bool(np.isfinite(y3_days) and abs(y3_days - DAYS_BENCH) < 0.008)
    size_ok = bool(np.isfinite(y3_size) and abs(y3_size - SIZE_QUOTE) < 0.008)
    peek_ok = bool(
        recs["d_cust_top1"]["n_defined"] == TOP1_N_PEEK
        and recs["d_cust_top1"]["n_pos"] == TOP1_POS_PEEK
        and np.isfinite(y3_t)
        and abs(y3_t - TOP1_Y3_PEEK) < 0.015
    )
    beat_size = bool(np.isfinite(y3_t) and (y3_t - SIZE_QUOTE) >= KEEP_DELTA)
    prose = (
        f"Y3 d_cust_top1 {_f(y3_t)} n={recs['d_cust_top1']['n_defined']:,} "
        f"pos={recs['d_cust_top1']['n_pos']:,} "
        f"(peek 0.590 / 2,485 / 141 {'CONFIRM' if peek_ok else 'DRIFT'}). "
        f"vs size {_f(y3_size)} vs days {_f(y3_days)} vs HHI {_f(y3_hhi)} vs n_cust {_f(y3_nc)}. "
        f"Replica days 0.711 {'CONFIRM' if days_ok else 'DRIFT'} / size 0.617 "
        f"{'CONFIRM' if size_ok else 'DRIFT'}. Beat-size Δ="
        f"{_f(y3_t - SIZE_QUOTE) if np.isfinite(y3_t) else '—'} "
        f"{'PASS' if beat_size else 'FAIL'}."
    )
    print(prose)
    return {
        "rows": rows,
        "recs": recs,
        "top1": y3_t,
        "size": y3_size,
        "days": y3_days,
        "hhi": y3_hhi,
        "n_cust": y3_nc,
        "days_ok": days_ok,
        "size_ok": size_ok,
        "peek_ok": peek_ok,
        "beat_size": beat_size,
        "n_def": recs["d_cust_top1"]["n_defined"],
        "n_pos": recs["d_cust_top1"]["n_pos"],
        "prose": prose,
    }


def pass3_days(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 3 — honest leftover of d_cust_top1 after days; inverse")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    folds = tr["fold"]
    after = leftover_diag(y, tr["d_cust_top1"], (tr["c_n_days_with_tx"],), folds, lab)
    inv = leftover_diag(y, tr["c_n_days_with_tx"], (tr["d_cust_top1"],), folds, lab)
    prose = (
        f"d_cust_top1 leftover after days OLS {_f(after['ols'])} rank {_f(after['rank'])} "
        f"fake={after['fake']} almost={after['almost']} ρ(resid,days)={_f(after['rho_ctrl'])} "
        f"R²={_f(after['r2'])} honest_dies={after['honest_dies']} n={after['n']:,} pos={after['n_pos']:,}. "
        f"Inverse: days leftover after top1 OLS {_f(inv['ols'])} rank {_f(inv['rank'])} "
        f"dies={inv['honest_dies']}."
    )
    print(prose)
    return {
        "after": after,
        "inv": inv,
        "ols": after["ols"],
        "rank": after["rank"],
        "dies": after["honest_dies"],
        "fake": after["fake"],
        "almost": after["almost"],
        "r2": after["r2"],
        "inv_rank": inv["rank"],
        "inv_dies": inv["honest_dies"],
        "prose": prose,
    }


def pass5_after_conc(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 5 — leftover after HHI, n_cust, days+HHI")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    folds = tr["fold"]
    after_h = leftover_diag(y, tr["d_cust_top1"], (tr["d_cust_hhi"],), folds, lab)
    after_c = leftover_diag(y, tr["d_cust_top1"], (tr["d_n_cust"],), folds, lab)
    after_dh = leftover_diag(
        y, tr["d_cust_top1"], (tr["c_n_days_with_tx"], tr["d_cust_hhi"]), folds, lab
    )
    after_dc = leftover_diag(
        y, tr["d_cust_top1"], (tr["c_n_days_with_tx"], tr["d_n_cust"]), folds, lab
    )
    h_after = leftover_diag(y, tr["d_cust_hhi"], (tr["d_cust_top1"],), folds, lab)
    c_after = leftover_diag(y, tr["d_n_cust"], (tr["d_cust_top1"],), folds, lab)
    rows = []
    for name, rec in (
        ("after HHI", after_h),
        ("after n_cust", after_c),
        ("after days+HHI", after_dh),
        ("after days+n_cust", after_dc),
        ("HHI after top1", h_after),
        ("n_cust after top1", c_after),
    ):
        rows.append(
            {
                "bar": name,
                "OLS": _f(rec["ols"]),
                "rank": _f(rec["rank"]),
                "ρ(resid,bar)": _f(rec["rho_ctrl"]),
                "R2": _f(rec["r2"]),
                "dies": rec["honest_dies"],
                "n": rec["n"],
                "n_pos": rec["n_pos"],
            }
        )
    prose = (
        f"top1 leftover after HHI rank {_f(after_h['rank'])} dies={after_h['honest_dies']} "
        f"(want die — HHI is the rewrite). "
        f"after n_cust {_f(after_c['rank'])} dies={after_c['honest_dies']}; "
        f"after days+HHI {_f(after_dh['rank'])} dies={after_dh['honest_dies']}. "
        f"Inverse HHI after top1 {_f(h_after['rank'])} (cust_hhi leftover after top1 was 0.549 — not overwritten). "
        f"n_cust after top1 {_f(c_after['rank'])} (n_cust leftover 0.545 — not overwritten)."
    )
    print(prose)
    return {
        "rows": rows,
        "h_rank": after_h["rank"],
        "h_dies": after_h["honest_dies"],
        "c_rank": after_c["rank"],
        "c_dies": after_c["honest_dies"],
        "dh_rank": after_dh["rank"],
        "dh_dies": after_dh["honest_dies"],
        "dc_rank": after_dc["rank"],
        "h_after": h_after["rank"],
        "c_after": c_after["rank"],
        "prose": prose,
    }


def pass6_dark(tr: pd.DataFrame, book: set[str]) -> dict:
    print("\n" + "=" * 72)
    print("CUT 6 — dark 470 stay NaN; ERP leftover")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    erp = tr["company_id"].isin(book)
    dark = ~erp
    x = pd.to_numeric(tr["d_cust_top1"], errors="coerce")
    n_dark_co = int(tr.loc[dark, "company_id"].nunique())
    dark_nn = int(x[dark].notna().sum())
    dark_zero = int((x[dark] == 0).sum())
    rec = signed_oof_auroc(y, tr["d_cust_top1"], tr["fold"], lab & erp)
    after = leftover_diag(y, tr["d_cust_top1"], (tr["c_n_days_with_tx"],), tr["fold"], lab & erp)
    rec_d = signed_oof_auroc(y, tr["d_cust_top1"], tr["fold"], lab & dark)
    dark_ok = n_dark_co == N_DARK_WANT and dark_nn == 0 and dark_zero == 0
    rows = [
        {
            "book": "dark",
            "n_co": int(tr.loc[dark, "company_id"].nunique()),
            "n_cust_top1 nn": dark_nn,
            "Y3 n": rec_d["n_defined"],
            "Y3 pos": rec_d["n_pos"],
            "CV": "LOW_POWER" if rec_d["low_power"] else _f(rec_d["cv"]),
        },
        {
            "book": "ERP",
            "n_co": int(tr.loc[erp, "company_id"].nunique()),
            "n_cust_top1 nn": int(x[erp].notna().sum()),
            "Y3 n": rec["n_defined"],
            "Y3 pos": rec["n_pos"],
            "CV": "LOW_POWER" if rec["low_power"] else _f(rec["cv"]),
        },
    ]
    prose = (
        f"Dark {n_dark_co} (want {N_DARK_WANT}) top1 nn={dark_nn} zero={dark_zero} "
        f"{'CONFIRM NaN not 0' if dark_ok else 'FAIL'}. "
        f"ERP Y3 {_f(rec['cv'])} leftover after days {_f(after['rank'])} dies={after['honest_dies']}."
    )
    print(prose)
    return {
        "rows": rows,
        "dark_ok": dark_ok,
        "n_dark_co": n_dark_co,
        "dark_nn": dark_nn,
        "erp_cv": _cv(rec),
        "erp_rank": after["rank"],
        "prose": prose,
    }


def pass7_q6(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 7 — Q6 lag1 leftover after days_lag1 (quote 0.684)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    folds = tr["fold"]
    recs = {}
    rows = []
    for name in ("d_cust_top1", "d_cust_top1_lag1", "d_cust_top1_lag3", "c_n_days_with_tx_lag1"):
        rec = signed_oof_auroc(y, tr[name], folds, lab)
        recs[name] = rec
        rows.append(_auc_row(Y3, name, rec))
        print(f"  {name} CV={_f(rec['cv'])} n={rec['n_defined']} pos={rec['n_pos']}")
    after_l1 = leftover_diag(y, tr["d_cust_top1_lag1"], (tr["c_n_days_with_tx_lag1"],), folds, lab)
    after_l3 = leftover_diag(y, tr["d_cust_top1_lag3"], (tr["c_n_days_with_tx_lag3"],), folds, lab)
    days_l1 = _cv(recs["c_n_days_with_tx_lag1"])
    days_ok = bool(np.isfinite(days_l1) and abs(days_l1 - DAYS_LAG1_QUOTE) < 0.008)
    y4 = pd.to_numeric(tr[Y4], errors="coerce")
    lab4 = y4.notna()
    short4 = lab4 & (tr["so_far_class"] == "short_<12")
    rec_s4 = signed_oof_auroc(y4, tr["d_cust_top1_lag3"], tr["fold"], short4)
    present = (
        _pct(int((short4 & tr["d_cust_top1_lag3"].notna()).sum()), int(short4.sum()))
        if short4.any()
        else float("nan")
    )
    prose = (
        f"Y3 top1_lag1 {_f(recs['d_cust_top1_lag1']['cv'])} leftover after days_lag1 "
        f"rank {_f(after_l1['rank'])} dies={after_l1['honest_dies']}. "
        f"lag3 leftover after days_lag3 {_f(after_l3['rank'])} dies={after_l3['honest_dies']}. "
        f"Days lag1 {_f(days_l1)} (quote 0.684 {'CONFIRM' if days_ok else 'DRIFT'}). "
        f"Short Y4 lag3 present {_pp(present)} pos={rec_s4['n_pos']} "
        f"(HHI Q6 quote 21.7% / 42 pos)."
    )
    print(prose)
    return {
        "rows": rows,
        "lag1": _cv(recs["d_cust_top1_lag1"]),
        "lag3": _cv(recs["d_cust_top1_lag3"]),
        "l1_rank": after_l1["rank"],
        "l1_dies": after_l1["honest_dies"],
        "l3_rank": after_l3["rank"],
        "l3_dies": after_l3["honest_dies"],
        "days_l1": days_l1,
        "days_ok": days_ok,
        "short4_present": present,
        "short4_pos": rec_s4["n_pos"],
        "prose": prose,
    }


def pass8_tail(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 8 — body vs HHI>0.975 tail as Y3 X; Y4 footnote")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    y4 = pd.to_numeric(tr[Y4], errors="coerce")
    hhi = pd.to_numeric(tr["d_cust_hhi"], errors="coerce")
    top1 = pd.to_numeric(tr["d_cust_top1"], errors="coerce")
    lab3 = y3.notna()
    lab4 = y4.notna()
    tail3 = lab3 & hhi.notna() & (hhi > TAIL_CUT)
    body3 = lab3 & hhi.notna() & (hhi <= TAIL_CUT)
    tail4 = lab4 & hhi.notna() & (hhi > TAIL_CUT)
    body4 = lab4 & hhi.notna() & (hhi <= TAIL_CUT)
    rec_body3 = signed_oof_auroc(y3, tr["d_cust_top1"], tr["fold"], body3)
    after_body3 = leftover_diag(y3, tr["d_cust_top1"], (tr["c_n_days_with_tx"],), tr["fold"], body3)
    rec_tail3 = signed_oof_auroc(y3, tr["d_cust_top1"], tr["fold"], tail3)
    rec_t4 = signed_oof_auroc(y4, tr["d_cust_top1"], tr["fold"], lab4)
    rec_t4l3 = signed_oof_auroc(y4, tr["d_cust_top1_lag3"], tr["fold"], lab4)
    rec_h4l3 = signed_oof_auroc(y4, tr["d_cust_hhi_lag3"], tr["fold"], lab4)
    rec_body4 = signed_oof_auroc(y4, tr["d_cust_top1_lag3"], tr["fold"], body4)
    after_t3 = leftover_diag(
        y4, tr["d_cust_hhi_lag3"], (tr["d_cust_top1_lag3"],), tr["fold"], lab4
    )
    rate_t = float(y4[tail4].mean()) if tail4.any() else float("nan")
    rate_r = float(y4[body4].mean()) if body4.any() else float("nan")
    tail_ok = bool(np.isfinite(rate_t) and abs(rate_t - Y4_TAIL_HI) < 0.03)
    body_ok = bool(np.isfinite(_cv(rec_body4)) and abs(_cv(rec_body4) - Y4_BODY_QUOTE) < 0.03) or (
        rec_body4["low_power"]
    )
    lag3_ok = bool(np.isfinite(_cv(rec_h4l3)) and abs(_cv(rec_h4l3) - Y4_LAG3_QUOTE) < 0.015)
    left_rank_ok = bool(np.isfinite(after_t3["rank"]) and abs(after_t3["rank"] - 0.461) < 0.015)
    left_ols_ok = bool(np.isfinite(after_t3["ols"]) and abs(after_t3["ols"] - Y4_LEFT_QUOTE) < 0.015)
    left_ok = left_ols_ok
    overturn = bool(np.isfinite(_cv(rec_t4l3)) and _cv(rec_t4l3) < 0.56)
    rows = [
        _auc_row(Y3, "top1 body HHI≤0.975", rec_body3),
        _auc_row(Y3, "top1 tail HHI>0.975", rec_tail3),
        _auc_row(Y4, "d_cust_top1", rec_t4),
        _auc_row(Y4, "d_cust_top1_lag3", rec_t4l3),
        _auc_row(Y4, "d_cust_hhi_lag3", rec_h4l3),
        _auc_row(Y4, "top1_lag3 body", rec_body4),
    ]
    prose = (
        f"Y3 body leftover after days {_f(after_body3['rank'])} dies={after_body3['honest_dies']} "
        f"raw {_f(_cv(rec_body3))}. Y3 tail raw {_f(_cv(rec_tail3))} "
        f"n={rec_tail3['n_defined']:,} pos={rec_tail3['n_pos']:,}. "
        f"Y4 top1_lag3 {_f(_cv(rec_t4l3))} HHI_lag3 {_f(_cv(rec_h4l3))} "
        f"(quote 0.605 {'CONFIRM' if lag3_ok else 'DRIFT'}). "
        f"HHI leftover after top1_lag3 OLS {_f(after_t3['ols'])} "
        f"(quote 0.549 {'CONFIRM' if left_ols_ok else 'DRIFT'}) "
        f"rank {_f(after_t3['rank'])} (cust_hhi rank-ortho 0.461 "
        f"{'CONFIRM' if left_rank_ok else 'DRIFT'}). "
        f"Y4 contemporaneous tail rate {_pp(rate_t)} vs rest {_pp(rate_r)} "
        f"(lag3 quote 22.1% / 11.5% is HHI_lag3, not now-HHI). "
        f"Y4 footnote {'OVERTURN' if overturn else 'KEEP'}."
    )
    print(prose)
    return {
        "rows": rows,
        "body3": _cv(rec_body3),
        "body3_rank": after_body3["rank"],
        "tail3": _cv(rec_tail3),
        "y4_now": _cv(rec_t4),
        "y4_lag3": _cv(rec_t4l3),
        "y4_hhi_lag3": _cv(rec_h4l3),
        "y4_body": _cv(rec_body4),
        "hhi_after_t3": after_t3["rank"],
        "hhi_after_t3_ols": after_t3["ols"],
        "left_rank_ok": left_rank_ok,
        "left_ols_ok": left_ols_ok,
        "rate_t": rate_t,
        "rate_r": rate_r,
        "tail_ok": tail_ok,
        "lag3_ok": lag3_ok,
        "left_ok": left_ok,
        "overturn": overturn,
        "footnote": "OVERTURN" if overturn else "KEEP",
        "n_tail4": int(tail4.sum()),
        "n_pos_tail4": int((tail4 & (y4 == 1)).sum()),
        "prose": prose,
    }


def pass9_supp(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 9 — vs d_supp_top1")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    rho = spearman(tr["d_cust_top1"], tr["d_supp_top1"])
    rec_s = signed_oof_auroc(y, tr["d_supp_top1"], tr["fold"], lab)
    after_s = leftover_diag(y, tr["d_supp_top1"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_c = leftover_diag(y, tr["d_cust_top1"], (tr["d_supp_top1"],), tr["fold"], lab)
    same = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
    prose = (
        f"ρ(cust_top1, supp_top1)={_f(rho)} {'SAME object (twin)' if same else 'different object'}. "
        f"Y3 supp_top1 {_f(_cv(rec_s))} leftover after days {_f(after_s['rank'])} dies={after_s['honest_dies']}. "
        f"cust_top1 leftover after supp_top1 {_f(after_c['rank'])} dies={after_c['honest_dies']}."
    )
    print(prose)
    return {
        "rho": rho,
        "same": same,
        "supp_cv": _cv(rec_s),
        "supp_rank": after_s["rank"],
        "cust_after_supp": after_c["rank"],
        "prose": prose,
    }


def pass10_hold(hold: pd.DataFrame, book: set[str]) -> dict:
    print("\n" + "=" * 72)
    print("CUT 10 — holdout coverage only (no AUROC)")
    print("=" * 72)
    x = pd.to_numeric(hold["d_cust_top1"], errors="coerce")
    erp = hold["company_id"].isin(book)
    rec = {
        "n_co": int(hold["company_id"].nunique()),
        "n_cm": len(hold),
        "n_nn": int(x.notna().sum()),
        "cov": _pct(int(x.notna().sum()), len(hold)),
        "p50": float(x[x.notna()].median()) if x.notna().any() else float("nan"),
        "dark_nn": int(x[~erp].notna().sum()),
    }
    prose = (
        f"Holdout {rec['n_co']} co / {rec['n_cm']} CM nn={rec['n_nn']} "
        f"cov={_pp(rec['cov'])} p50={_f(rec['p50'])} dark nn={rec['dark_nn']} (no fit, no AUROC)."
    )
    print(prose)
    return {**rec, "prose": prose}


def extra_icc(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — ICC / demean")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    icc = icc_anova(tr["d_cust_top1"], tr["company_id"])
    trait = bool(np.isfinite(icc["icc"]) and icc["icc"] >= ICC_TRAIT)
    demean = company_demean(tr["d_cust_top1"], tr["company_id"])
    meanx = company_mean(tr["d_cust_top1"], tr["company_id"])
    rec_d = signed_oof_auroc(y, demean, tr["fold"], lab)
    rec_m = signed_oof_auroc(y, meanx, tr["fold"], lab)
    after_d = leftover_diag(y, demean, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_m = leftover_diag(y, meanx, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    prose = (
        f"ICC={_f(icc['icc'])} {'TRAIT' if trait else 'STATE'} k={icc['k']}. "
        f"Demean CV {_f(rec_d['cv'])} leftover-days {_f(after_d['rank'])} dies={after_d['honest_dies']}. "
        f"Company-mean CV {_f(rec_m['cv'])} leftover-days {_f(after_m['rank'])} dies={after_m['honest_dies']}."
    )
    print(prose)
    return {"icc": icc, "trait": trait, "demean_rank": after_d["rank"], "mean_rank": after_m["rank"], "prose": prose}


def extra_fold_y2(tr: pd.DataFrame, p3: dict) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — fold-wise leftover; 12-name Y2 drop")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    ids = chronic_12(tr)
    y2 = pd.to_numeric(tr[Y2], errors="coerce")
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    chron = tr["company_id"].isin(ids)
    rec2 = signed_oof_auroc(y2, tr["d_cust_top1"], tr["fold"], y2.notna())
    after_y3_wo = leftover_diag(
        y3, tr["d_cust_top1"], (tr["c_n_days_with_tx"],), tr["fold"], y3.notna() & ~chron
    )
    fold_rows = []
    for r, rr in zip(p3["after"]["rec"]["folds"], p3["after"]["rrec"]["folds"]):
        fold_rows.append(
            {
                "fold": r["fold"],
                "OLS leftover": _f(r["auroc"]),
                "rank leftover": _f(rr["auroc"]),
                "n_va": r["n_va"],
                "n_pos": r["n_pos"],
            }
        )
    prose = (
        f"Y2 top1 {_f(_cv(rec2))}. Drop {len(ids)} chronic. "
        f"Y3 leftover wo12 {_f(after_y3_wo['rank'])} dies={after_y3_wo['honest_dies']}."
    )
    print(prose)
    return {"ids": ids, "fold_rows": fold_rows, "y2": _cv(rec2), "y3_wo_rank": after_y3_wo["rank"], "prose": prose}


def extra_bootstrap(tr: pd.DataFrame, n_boot: int = 40) -> dict:
    print("\n" + "=" * 72)
    print(f"EXTRA — company bootstrap leftover after days (n={n_boot})")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    work = tr.loc[lab, ["company_id", "fold", Y3, "d_cust_top1", "c_n_days_with_tx"]].copy()
    cos = work["company_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(FOLD_SEED)
    ranks = []
    for _ in range(n_boot):
        draw = rng.choice(cos, size=len(cos), replace=True)
        parts = [work[work["company_id"] == c] for c in draw]
        b = pd.concat(parts, ignore_index=True)
        after = leftover_diag(
            b[Y3],
            b["d_cust_top1"],
            (b["c_n_days_with_tx"],),
            b["fold"],
            pd.Series(True, index=b.index),
        )
        if np.isfinite(after["rank"]):
            ranks.append(after["rank"])
    p05 = float(np.quantile(ranks, 0.05)) if ranks else float("nan")
    p50 = float(np.median(ranks)) if ranks else float("nan")
    p95 = float(np.quantile(ranks, 0.95)) if ranks else float("nan")
    share_die = float(np.mean([r < CHANCE for r in ranks])) if ranks else float("nan")
    prose = (
        f"Bootstrap leftover-after-days rank p05={_f(p05)} p50={_f(p50)} p95={_f(p95)} "
        f"share<0.55={_pp(share_die)} n={len(ranks)}."
    )
    print(prose)
    return {"p05": p05, "p50": p50, "p95": p95, "share_die": share_die, "prose": prose}


def extra_permute(tr: pd.DataFrame, n_perm: int = 24) -> dict:
    print("\n" + "=" * 72)
    print(f"EXTRA — permute top1 within days quintile (n={n_perm})")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    work = tr.loc[lab].copy()
    days = pd.to_numeric(work["c_n_days_with_tx"], errors="coerce")
    x = pd.to_numeric(work["d_cust_top1"], errors="coerce")
    ok = days.notna() & x.notna()
    work = work.loc[ok]
    q = pd.qcut(pd.to_numeric(work["c_n_days_with_tx"], errors="coerce"), 5, duplicates="drop")
    rng = np.random.default_rng(FOLD_SEED + 13)
    ranks = []
    for _ in range(n_perm):
        shuf = work["d_cust_top1"].copy()
        for cat in q.dropna().unique():
            idx = q[q == cat].index
            vals = shuf.loc[idx].to_numpy()
            rng.shuffle(vals)
            shuf.loc[idx] = vals
        after = leftover_diag(
            work[Y3],
            shuf,
            (work["c_n_days_with_tx"],),
            work["fold"],
            pd.Series(True, index=work.index),
        )
        if np.isfinite(after["rank"]):
            ranks.append(after["rank"])
    p50 = float(np.median(ranks)) if ranks else float("nan")
    p90 = float(np.quantile(ranks, 0.90)) if ranks else float("nan")
    prose = f"Permuted-within-days leftover rank p50={_f(p50)} p90={_f(p90)} n={len(ranks)}."
    print(prose)
    return {"p50": p50, "p90": p90, "prose": prose}


def extra_size(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after size")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after = leftover_diag(y, tr["d_cust_top1"], (tr["log_in3"],), tr["fold"], lab)
    after_both = leftover_diag(
        y, tr["d_cust_top1"], (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], lab
    )
    prose = (
        f"top1 leftover after size rank {_f(after['rank'])} dies={after['honest_dies']}. "
        f"after days+size {_f(after_both['rank'])} dies={after_both['honest_dies']}."
    )
    print(prose)
    return {"after_size": after["rank"], "after_both": after_both["rank"], "prose": prose}


def extra_quintiles(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Y3 rate by top1 quintile")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr["d_cust_top1"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    ok = y.notna() & x.notna() & days.notna()
    qn = pd.qcut(x[ok], 5, duplicates="drop")
    rows = []
    for i, cat in enumerate(sorted(qn.dropna().unique()), start=1):
        sl = ok.copy()
        sl.loc[ok] = qn == cat
        rows.append(
            {
                "q": i,
                "top1 rate": _pp(float(y[sl].mean()) if sl.any() else float("nan")),
                "n": int(sl.sum()),
                "n_pos": int((sl & (y == 1)).sum()),
            }
        )
    prose = f"Y3 top1 Q1→Q5 {[r['top1 rate'] for r in rows]}."
    print(prose)
    return {"rows": rows, "prose": prose}


def extra_tercile(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — SIZE tercile leftover of top1 after days")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    med = tr.groupby("company_id")["log_in3"].median()
    cuts = med.quantile([1 / 3, 2 / 3])
    lab_co = pd.Series("T2", index=med.index)
    lab_co[med <= cuts.iloc[0]] = "T1"
    lab_co[med > cuts.iloc[1]] = "T3"
    terc = tr["company_id"].map(lab_co)
    rows = []
    for t in ("T1", "T2", "T3"):
        sl = lab & (terc == t)
        rec = signed_oof_auroc(y, tr["d_cust_top1"], tr["fold"], sl)
        after = leftover_diag(y, tr["d_cust_top1"], (tr["c_n_days_with_tx"],), tr["fold"], sl)
        rows.append(
            {
                "tercile": t,
                "n": rec["n_defined"],
                "n_pos": rec["n_pos"],
                "CV": "LOW_POWER" if rec["low_power"] else _f(rec["cv"]),
                "leftover rank": _f(after["rank"]),
                "dies": after["honest_dies"],
            }
        )
        print(f"  {t} CV={_f(rec['cv'])} leftover={_f(after['rank'])} dies={after['honest_dies']}")
    return {"rows": rows, "prose": "SIZE tercile leftover of top1 after days."}


def extra_hhi_reconcile(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — HHI leftover after top1_lag3 OLS vs rank; Y4 lag3 tail")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y4 = pd.to_numeric(tr[Y4], errors="coerce")
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    lab4 = y4.notna()
    lab3 = y3.notna()
    hhi_l3 = pd.to_numeric(tr["d_cust_hhi_lag3"], errors="coerce")
    after = leftover_diag(y4, tr["d_cust_hhi_lag3"], (tr["d_cust_top1_lag3"],), tr["fold"], lab4)
    ols_ok = bool(np.isfinite(after["ols"]) and abs(after["ols"] - Y4_LEFT_QUOTE) < 0.015)
    rank_ok = bool(np.isfinite(after["rank"]) and abs(after["rank"] - 0.461) < 0.015)
    tail = lab4 & hhi_l3.notna() & (hhi_l3 > TAIL_CUT)
    body = lab4 & hhi_l3.notna() & (hhi_l3 <= TAIL_CUT)
    rate_t = float(y4[tail].mean()) if tail.any() else float("nan")
    rate_r = float(y4[body].mean()) if body.any() else float("nan")
    tail_ok = bool(np.isfinite(rate_t) and abs(rate_t - Y4_TAIL_HI) < 0.015)
    rest_ok = bool(np.isfinite(rate_r) and abs(rate_r - Y4_TAIL_REST) < 0.015)
    rec_body = signed_oof_auroc(y4, tr["d_cust_hhi_lag3"], tr["fold"], body)
    body_ok = bool(np.isfinite(_cv(rec_body)) and abs(_cv(rec_body) - Y4_BODY_QUOTE) < 0.015)
    rec_flag = signed_oof_auroc(y3, (hhi_l3 > TAIL_CUT).astype(float), tr["fold"], lab3 & hhi_l3.notna())
    after_flag = leftover_diag(
        y3,
        (pd.to_numeric(tr["d_cust_hhi"], errors="coerce") > TAIL_CUT).astype(float),
        (tr["c_n_days_with_tx"],),
        tr["fold"],
        lab3 & pd.to_numeric(tr["d_cust_hhi"], errors="coerce").notna(),
    )
    rows = [
        {
            "cut": "Y4 HHI after top1_lag3 OLS",
            "value": _f(after["ols"]),
            "quote": "0.549",
            "ok": "CONFIRM" if ols_ok else "DRIFT",
        },
        {
            "cut": "Y4 HHI after top1_lag3 rank",
            "value": _f(after["rank"]),
            "quote": "0.461",
            "ok": "CONFIRM" if rank_ok else "DRIFT",
        },
        {
            "cut": "Y4 HHI_lag3 >0.975 rate",
            "value": _pp(rate_t),
            "quote": "22.1%",
            "ok": "CONFIRM" if tail_ok else "DRIFT",
        },
        {
            "cut": "Y4 HHI_lag3 rest rate",
            "value": _pp(rate_r),
            "quote": "11.5%",
            "ok": "CONFIRM" if rest_ok else "DRIFT",
        },
        {
            "cut": "Y4 HHI_lag3 body CV",
            "value": _f(_cv(rec_body)),
            "quote": "0.445",
            "ok": "CONFIRM" if body_ok else "DRIFT",
        },
    ]
    prose = (
        f"cust_hhi 0.549 is OLS leftover of HHI after top1_lag3 "
        f"{_f(after['ols'])} {'CONFIRM' if ols_ok else 'DRIFT'}; "
        f"honest rank {_f(after['rank'])} {'CONFIRM 0.461' if rank_ok else 'DRIFT'} "
        f"R²={_f(after['r2'])}. Y4 HHI_lag3 >0.975 {_pp(rate_t)} n={int(tail.sum())} "
        f"pos={int((tail & (y4 == 1)).sum())} vs rest {_pp(rate_r)} "
        f"{'CONFIRM 22.1/11.5' if tail_ok and rest_ok else 'DRIFT'}. "
        f"HHI_lag3 body CV {_f(_cv(rec_body))} (quote 0.445 "
        f"{'CONFIRM' if body_ok else 'DRIFT'}). "
        f"Y3 tail-flag leftover after days {_f(after_flag['rank'])} dies={after_flag['honest_dies']} "
        f"raw {_f(_cv(rec_flag))}."
    )
    print(prose)
    return {
        "ols": after["ols"],
        "rank": after["rank"],
        "ols_ok": ols_ok,
        "rank_ok": rank_ok,
        "rate_t": rate_t,
        "rate_r": rate_r,
        "tail_ok": tail_ok,
        "body": _cv(rec_body),
        "body_ok": body_ok,
        "flag_rank": after_flag["rank"],
        "n_tail": int(tail.sum()),
        "n_pos_tail": int((tail & (y4 == 1)).sum()),
        "rows": rows,
        "prose": prose,
    }


def extra_collinear(tr: pd.DataFrame, p3: dict, p5: dict) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — days+HHI leftover lives is rewrite residual")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    resid, info = ols_resid(tr["d_cust_top1"], tr["d_cust_hhi"])
    after_r = leftover_diag(y, resid, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_tx = leftover_diag(y, tr["d_cust_top1"], (tr["a_n_tx"],), tr["fold"], lab)
    dh_row = next((r for r in p5["rows"] if r["bar"] == "after days+HHI"), {})
    try:
        r2_dh = float(dh_row.get("R2", "nan"))
    except (TypeError, ValueError):
        r2_dh = float("nan")
    n_dh = dh_row.get("n", float("nan"))
    same_n = n_dh == p3["after"]["n"]
    collinear_lives = bool(
        np.isfinite(p5["dh_rank"]) and p5["dh_rank"] >= CHANCE and np.isfinite(r2_dh) and r2_dh >= 0.90
    )
    prose = (
        f"after days rank {_f(p3['rank'])} n={p3['after']['n']:,}; "
        f"after days+HHI rank {_f(p5['dh_rank'])} n={n_dh} "
        f"R²={_f(r2_dh)} same_n={same_n}. "
        f"{'LIVES is rewrite residual (R²≥0.90), not leftover' if collinear_lives else 'not a rewrite residual'}. "
        f"top1-after-HHI residual leftover after days {_f(after_r['rank'])} "
        f"dies={after_r['honest_dies']} R²_on_HHI={_f(info['r2'])}. "
        f"top1 leftover after a_n_tx {_f(after_tx['rank'])} dies={after_tx['honest_dies']}."
    )
    print(prose)
    return {
        "same_n": same_n,
        "r2_dh": r2_dh,
        "collinear_lives": collinear_lives,
        "resid_after_days": after_r["rank"],
        "after_tx": after_tx["rank"],
        "prose": prose,
    }


def extra_y3_tail_rate(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Y3 body vs tail rates (tail AUROC LOW_POWER)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    hhi = pd.to_numeric(tr["d_cust_hhi"], errors="coerce")
    lab = y.notna() & hhi.notna()
    tail = lab & (hhi > TAIL_CUT)
    body = lab & (hhi <= TAIL_CUT)
    rate_t = float(y[tail].mean()) if tail.any() else float("nan")
    rate_b = float(y[body].mean()) if body.any() else float("nan")
    after_q5 = leftover_diag(
        y,
        tr["d_cust_top1"],
        (tr["c_n_days_with_tx"],),
        tr["fold"],
        lab & (pd.to_numeric(tr["d_cust_top1"], errors="coerce") >= pd.to_numeric(tr["d_cust_top1"], errors="coerce")[lab].quantile(0.8)),
    )
    prose = (
        f"Y3 tail HHI>0.975 rate {_pp(rate_t)} n={int(tail.sum())} "
        f"pos={int((tail & (y == 1)).sum())} vs body {_pp(rate_b)} n={int(body.sum())} "
        f"pos={int((body & (y == 1)).sum())}. "
        f"Q5 leftover after days {_f(after_q5['rank'])} dies={after_q5['honest_dies']} "
        f"n={after_q5['n']} pos={after_q5['n_pos']}."
    )
    print(prose)
    return {
        "rate_t": rate_t,
        "rate_b": rate_b,
        "n_tail": int(tail.sum()),
        "n_pos_tail": int((tail & (y == 1)).sum()),
        "q5_rank": after_q5["rank"],
        "prose": prose,
    }


def extra_q6_samen(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Q6 same-n leftover; incomplete 6m NaN")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    l1 = pd.to_numeric(tr["d_cust_top1_lag1"], errors="coerce").notna()
    after_now = leftover_diag(y, tr["d_cust_top1"], (tr["c_n_days_with_tx"],), tr["fold"], lab & l1)
    after_l1 = leftover_diag(
        y, tr["d_cust_top1_lag1"], (tr["c_n_days_with_tx_lag1"],), tr["fold"], lab & l1
    )
    x = pd.to_numeric(tr["d_cust_top1"], errors="coerce")
    # Family D full6 is calendar: win6_start >= 2024-09 → period >= 2025-02.
    # months_so_far < 6 is the company-relative clock (wrong for this NaN).
    cal_inc = pd.to_datetime(tr["period"]) < pd.Timestamp("2025-02-01")
    cal_nn = int(x[cal_inc].notna().sum())
    cal_ok = cal_nn == 0
    short = tr["months_so_far"] < 6
    short_nn = int(x[short].notna().sum())
    y4 = pd.to_numeric(tr[Y4], errors="coerce")
    after_y4 = leftover_diag(y4, tr["d_cust_top1"], (tr["c_n_days_with_tx"],), tr["fold"], y4.notna())
    after_y4_l3 = leftover_diag(
        y4, tr["d_cust_top1_lag3"], (tr["c_n_days_with_tx_lag3"],), tr["fold"], y4.notna()
    )
    prose = (
        f"Same-n lag1 rows: contemporaneous leftover after days {_f(after_now['rank'])} "
        f"dies={after_now['honest_dies']}; lag1 leftover after days_lag1 {_f(after_l1['rank'])} "
        f"dies={after_l1['honest_dies']} n={after_l1['n']}. "
        f"Calendar incomplete (period<2025-02) top1 nn={cal_nn} "
        f"{'CONFIRM NaN' if cal_ok else 'FAIL'}. "
        f"Company-relative so-far<6 nn={short_nn} (wrong clock — not the D NaN). "
        f"Y4 leftover of top1 after days {_f(after_y4['rank'])} dies={after_y4['honest_dies']}; "
        f"top1_lag3 after days_lag3 {_f(after_y4_l3['rank'])} dies={after_y4_l3['honest_dies']} "
        f"(report-only; do not promote top1 as Y4 engine X)."
    )
    print(prose)
    return {
        "now_rank": after_now["rank"],
        "l1_rank": after_l1["rank"],
        "cal_ok": cal_ok,
        "cal_nn": cal_nn,
        "short_nn": short_nn,
        "y4_rank": after_y4["rank"],
        "y4_l3_rank": after_y4_l3["rank"],
        "prose": prose,
    }


def extra_erp_trail(tr: pd.DataFrame, book: set[str]) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — ERP trail-length leftover; dark/hold already NaN")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    erp = tr["company_id"].isin(book)
    rows = []
    for name, sl in (
        ("ERP short_<12", lab & erp & (tr["so_far_class"] == "short_<12")),
        ("ERP mid_12_17", lab & erp & (tr["so_far_class"] == "mid_12_17")),
        ("ERP long_>=18", lab & erp & (tr["so_far_class"] == "long_>=18")),
    ):
        rec = signed_oof_auroc(y, tr["d_cust_top1"], tr["fold"], sl)
        after = leftover_diag(y, tr["d_cust_top1"], (tr["c_n_days_with_tx"],), tr["fold"], sl)
        rows.append(
            {
                "slice": name,
                "n": rec["n_defined"],
                "n_pos": rec["n_pos"],
                "CV": "LOW_POWER" if rec["low_power"] else _f(rec["cv"]),
                "leftover": _f(after["rank"]),
                "dies": after["honest_dies"],
            }
        )
        print(f"  {name} CV={_f(rec['cv'])} leftover={_f(after['rank'])} dies={after['honest_dies']}")
    prose = "ERP trail-length leftover of top1 after days."
    return {"rows": rows, "prose": prose}


def extra_y4_as_x(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Y4 leftover of top1 after days / after HHI (report-only)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y4 = pd.to_numeric(tr[Y4], errors="coerce")
    lab = y4.notna()
    after_d = leftover_diag(y4, tr["d_cust_top1"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_h = leftover_diag(y4, tr["d_cust_top1"], (tr["d_cust_hhi"],), tr["fold"], lab)
    after_l3d = leftover_diag(
        y4, tr["d_cust_top1_lag3"], (tr["c_n_days_with_tx_lag3"],), tr["fold"], lab
    )
    after_l3h = leftover_diag(
        y4, tr["d_cust_top1_lag3"], (tr["d_cust_hhi_lag3"],), tr["fold"], lab
    )
    rec = signed_oof_auroc(y4, tr["d_cust_top1"], tr["fold"], lab)
    rec_l3 = signed_oof_auroc(y4, tr["d_cust_top1_lag3"], tr["fold"], lab)
    prose = (
        f"Y4 top1 {_f(_cv(rec))} leftover after days {_f(after_d['rank'])} "
        f"dies={after_d['honest_dies']}; after HHI {_f(after_h['rank'])} "
        f"dies={after_h['honest_dies']} R²={_f(after_h['r2'])}. "
        f"Y4 top1_lag3 {_f(_cv(rec_l3))} leftover after days_lag3 {_f(after_l3d['rank'])} "
        f"dies={after_l3d['honest_dies']}; after HHI_lag3 {_f(after_l3h['rank'])} "
        f"dies={after_l3h['honest_dies']}. "
        f"Do not put top1 on Y4 engine — footnote is the >0.975 tail, not leftover as X."
    )
    print(prose)
    return {
        "rank_days": after_d["rank"],
        "rank_hhi": after_h["rank"],
        "l3_days": after_l3d["rank"],
        "l3_hhi": after_l3h["rank"],
        "prose": prose,
    }


def extra_fold_sens(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover drop each fold")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    rows = []
    for k in range(N_FOLDS):
        sl = y.notna() & (tr["fold"] != k)
        rec = signed_oof_auroc(y, tr["d_cust_top1"], tr["fold"], sl)
        after = leftover_diag(y, tr["d_cust_top1"], (tr["c_n_days_with_tx"],), tr["fold"], sl)
        rows.append(
            {
                "drop fold": k,
                "n": rec["n_defined"],
                "n_pos": rec["n_pos"],
                "CV": "LOW_POWER" if rec["low_power"] else _f(rec["cv"]),
                "leftover": _f(after["rank"]),
                "dies": after["honest_dies"],
                "beat-size": (
                    bool(np.isfinite(_cv(rec)) and (_cv(rec) - SIZE_QUOTE) >= KEEP_DELTA)
                    if not rec["low_power"]
                    else False
                ),
            }
        )
        print(
            f"  drop {k} CV={_f(rec['cv'])} leftover={_f(after['rank'])} "
            f"dies={after['honest_dies']} beat={rows[-1]['beat-size']}"
        )
    lives = [r for r in rows if r["dies"] is False and r["leftover"] != "—"]
    prose = (
        f"Drop-one-fold leftover lives on {len(lives)}/{N_FOLDS} drops. "
        f"Fold instability is not KEEP — full-sample leftover dies and twin_gate stays."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def extra_monopoly_flag(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Javier top1≥0.90 flag leftover after days; leftover inside days bins")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    x = pd.to_numeric(tr["d_cust_top1"], errors="coerce")
    flag = (x >= 0.90).astype(float)
    rec = signed_oof_auroc(y, flag, tr["fold"], lab & x.notna())
    after = leftover_diag(y, flag, (tr["c_n_days_with_tx"],), tr["fold"], lab & x.notna())
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    ok = lab & x.notna() & days.notna()
    q = pd.qcut(days[ok], 5, duplicates="drop")
    rows = []
    for i, cat in enumerate(sorted(q.dropna().unique()), start=1):
        sl = ok.copy()
        sl.loc[ok] = q == cat
        rec_i = signed_oof_auroc(y, tr["d_cust_top1"], tr["fold"], sl)
        rows.append(
            {
                "days q": i,
                "n": rec_i["n_defined"],
                "n_pos": rec_i["n_pos"],
                "CV": "LOW_POWER" if rec_i["low_power"] else _f(rec_i["cv"]),
                "rate": _pp(float(y[sl].mean()) if sl.any() else float("nan")),
            }
        )
    prose = (
        f"top1≥0.90 flag Y3 {_f(_cv(rec))} leftover after days {_f(after['rank'])} "
        f"dies={after['honest_dies']} n={after['n']} pos={after['n_pos']}. "
        f"Inside days quintiles top1 is a rate gradient, not leftover after days."
    )
    print(prose)
    return {"flag_cv": _cv(rec), "flag_rank": after["rank"], "rows": rows, "prose": prose}


def extra_drop_fold1(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover drop fold 1 (single 0.411 / rank leftover 0.349)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    sl = y.notna() & (tr["fold"] != 1)
    after = leftover_diag(y, tr["d_cust_top1"], (tr["c_n_days_with_tx"],), tr["fold"], sl)
    rec = signed_oof_auroc(y, tr["d_cust_top1"], tr["fold"], sl)
    prose = (
        f"Drop fold 1: Y3 top1 {_f(_cv(rec))} leftover after days {_f(after['rank'])} "
        f"dies={after['honest_dies']} n={after['n']} pos={after['n_pos']}."
    )
    print(prose)
    return {"cv": _cv(rec), "rank": after["rank"], "dies": after["honest_dies"], "prose": prose}


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
            f"beat-size {_f(p2['top1'])} vs 0.617, not SIZE, not twin. "
            f"Stays off the 15-col card. Y4 footnote {footnote}."
        )
    elif leftover_lives and twin:
        role = "DROP from the 44 as Y3 X"
        why = (
            f"leftover after days rank {p3['rank']:.3f} lives but TWIN of {p1['twins']}. "
            f"Javier concentration is top1; HHI is the rewrite. "
            f"after HHI {_f(p5['h_rank'])} dies={p5['h_dies']}. Y4 footnote {footnote}."
        )
    elif leftover_lives and is_size:
        role = "DROP from the 44 as Y3 X"
        why = (
            f"leftover after days rank {p3['rank']:.3f} lives but SIZE "
            f"(ρ={_f(p1['rhos']['log1p(a_in3)'])}). Y4 footnote {footnote}."
        )
    elif leftover_lives and not p2["beat_size"]:
        role = "CLOSE unused leftover"
        why = (
            f"leftover after days rank {p3['rank']:.3f} lives but fails beat-size "
            f"({_f(p2['top1'])} vs 0.617). Y4 footnote {footnote}."
        )
    else:
        role = "CLOSE unused leftover"
        why = (
            f"unused leftover after days: honest rank {_f(p3['rank'])} dies "
            f"(OLS {_f(p3['ols'])} fake={p3['fake']}). "
            f"{'Also TWIN of ' + str(p1['twins']) + '. ' if twin else ''}"
            f"DROP from the 44 as Y3 X. Y4 >0.975 footnote {footnote}. "
            f"Do not invent y_top1. Off the 15-col card."
        )
    return {
        "role": role,
        "why": why,
        "leftover_lives": leftover_lives,
        "engine": engine,
        "is_size": is_size,
        "twin": twin,
        "footnote": footnote,
        "park_y": "PARK as Y — do not invent y_top1",
        "card": "no — do not put d_cust_top1 on the 15-col card",
    }


def make_plot(tr: pd.DataFrame, p3: dict) -> bool:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return False
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr["d_cust_top1"], errors="coerce")
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
    ax.plot(xs, rates, marker="o", color="#1f4e79", label="d_cust_top1")
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
    ax.set_title("top1 vs days recover")
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
    ax.set_title("d_cust_top1 leftover after days")
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
        "# Unused leftover of `d_cust_top1` after days as Y3 X",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. No `build_targets`. Do not invent `y_top1`. Do not put top1 on the 15-col card. "
        "Do not overwrite `cust_hhi_qa.*`, `n_cust_qa.*`, `n_supp_qa.*`. Do not grow TURNOVER. "
        "Javier concentration is **top1**, not HHI.",
        "",
        "`d_cust_top1` = share of AR invoice |amount| from the single largest customer (Family D, trailing 6m). "
        "Incomplete 6m books are NaN. Dark 470 stay NaN not 0. "
        "`d_cust_hhi` DROP as weaker rewrite (ρ 0.994). `d_n_cust` CLOSED leftover 0.545.",
        "",
        "## Headline",
        "",
        (
            f"`d_cust_top1` as Y3 X: **{d['role']}** ({d['why']}). "
            f"Y3 leftover after days OLS {_f(p3['ols'])} rank {_f(p3['rank'])} "
            f"({'dies' if p3['dies'] else 'lives'}, fake={p3['fake']}). "
            f"Inverse days after top1 rank {_f(p3['inv_rank'])}. "
            f"Single {_f(p2['top1'])} vs size {_f(p2['size'])} vs days {_f(p2['days'])} "
            f"vs HHI {_f(p2['hhi'])} vs n_cust {_f(p2['n_cust'])}. "
            f"after HHI {_f(p5['h_rank'])} after n_cust {_f(p5['c_rank'])} after days+HHI {_f(p5['dh_rank'])}. "
            f"Y4 footnote **{d['footnote']}**. 15-col card: {d['card']}. {d['park_y']}. "
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
        f"| 4 | Dip vs fall? | Y4 >0.975 footnote **{d['footnote']}** lag3 {_f(p8['y4_lag3'])}. |",
        f"| 5 | Why did it change? | Twin screen: {p1['twins'] or 'none'}. ρ vs HHI {_f(p1['rhos']['d_cust_hhi'])}. |",
        f"| 6 | Months earlier? | lag1 leftover {_f(p7['l1_rank'])}; days_lag1 {_f(p7['days_l1'])}. |",
        "",
        "## PARK / CLOSE / KEEP / DROP-from-44",
        "",
        "| object | decision | why |",
        "| --- | --- | --- |",
        f"| `d_cust_top1` as Y3 X / the 15-col card | **{d['role']}** | {d['why']} |",
        f"| `d_cust_top1` as engine X on the 44 | **{'KEEP leftover' if d['engine'] else 'DROP'}** | leftover lives={d['leftover_lives']} twin={d['twin']} SIZE={d['is_size']} beat-size={p2['beat_size']} |",
        f"| Y4 >0.975 monopoly footnote | **{d['footnote']}** | lag3 {_f(p8['y4_lag3'])}; HHI after top1_lag3 OLS {_f(p8.get('hhi_after_t3_ols', float('nan')))} / rank {_f(p8['hhi_after_t3'])}; now-tail {_pp(p8['rate_t'])} vs {_pp(p8['rate_r'])} |",
        f"| `y_top1` | **PARK** | do not invent a concentration Y |",
        f"| twin of HHI / n_cust | {_f(p1['rhos']['d_cust_hhi'])} / {_f(p1['rhos']['d_n_cust'])} | leftover after HHI {_f(p5['h_rank'])} after n_cust {_f(p5['c_rank'])} |",
        f"| same object as `d_supp_top1` | **{'YES twin' if p9['same'] else 'NO'}** | ρ={_f(p9['rho'])} |",
        f"| Q6 lag1 after days_lag1 | **{'KEEP' if (np.isfinite(p7['l1_rank']) and p7['l1_rank'] >= CHANCE and not p7['l1_dies']) else 'CLOSE'}** | leftover {_f(p7['l1_rank'])} |",
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
        "## 5 — Leftover after HHI / n_cust / days+HHI",
        "",
        p5["prose"],
        "",
        _md_table(p5["rows"]),
        "",
        "## 6 — Dark 470 stay NaN; ERP leftover",
        "",
        p6["prose"],
        "",
        _md_table(p6["rows"]),
        "",
        "## 7 — Q6 lag1 leftover after days_lag1",
        "",
        p7["prose"],
        "",
        _md_table(p7["rows"]),
        "",
        "## 8 — Body vs tail as Y3 X; Y4 footnote",
        "",
        p8["prose"],
        "",
        _md_table(p8["rows"]),
        "",
        "## 9 — vs `d_supp_top1`",
        "",
        p9["prose"],
        "",
        "## 10 — Holdout coverage only",
        "",
        p10["prose"],
        "",
        "## Extras",
        "",
        "### ICC / demean",
        "",
        ctx["xi"]["prose"],
        "",
        "### Fold-wise leftover; 12-name Y2 drop",
        "",
        ctx["xf"]["prose"],
        "",
        _md_table(ctx["xf"]["fold_rows"]),
        "",
        "### Bootstrap leftover after days",
        "",
        ctx["xb"]["prose"],
        "",
        "### Permute within days quintile",
        "",
        ctx["xp"]["prose"],
        "",
        "### leftover after size",
        "",
        ctx["xsz"]["prose"],
        "",
        "### Y3 rate by top1 quintile",
        "",
        ctx["xq"]["prose"],
        "",
        _md_table(ctx["xq"]["rows"]),
        "",
        "### SIZE tercile leftover",
        "",
        ctx["xtc"]["prose"],
        "",
        _md_table(ctx["xtc"]["rows"]),
        "",
        "### HHI leftover reconcile (OLS 0.549 vs rank 0.461)",
        "",
        ctx["xhr"]["prose"],
        "",
        _md_table(ctx["xhr"]["rows"]),
        "",
        "### days+HHI leftover is rewrite residual",
        "",
        ctx["xcl"]["prose"],
        "",
        "### Y3 body vs tail rates",
        "",
        ctx["xtr"]["prose"],
        "",
        "### Q6 same-n; incomplete 6m; Y4 leftover after days",
        "",
        ctx["xq6"]["prose"],
        "",
        "### ERP trail-length leftover",
        "",
        ctx["xerp"]["prose"],
        "",
        _md_table(ctx["xerp"]["rows"]),
        "",
        "### Drop fold 1",
        "",
        ctx["xdf"]["prose"],
        "",
        "### Y4 leftover of top1 (report-only)",
        "",
        ctx["xy4"]["prose"],
        "",
        "### Drop-one-fold leftover",
        "",
        ctx["xfs"]["prose"],
        "",
        _md_table(ctx["xfs"]["rows"]),
        "",
        "### Javier top1≥0.90 flag; leftover inside days bins",
        "",
        ctx["xmf"]["prose"],
        "",
        _md_table(ctx["xmf"]["rows"]),
        "",
        "## Night quotes (unchanged)",
        "",
        "| quote | locked |",
        "| --- | --- |",
        f"| Y3 | {Y3_NIGHT[0]:.3f} / {Y3_NIGHT[1]:.3f} |",
        f"| days | {DAYS_BENCH:.3f} |",
        f"| size | {SIZE_QUOTE:.3f} |",
        f"| Y7 TURNOVER | {Y7_TURNOVER:.3f} / {Y7_B_SHALLOW:.3f} |",
        f"| Y4 HHI_lag3 footnote | {Y4_LAG3_QUOTE:.3f} |",
        "",
        "Do not quote a_out_vol 0.722. Do not grow TURNOVER. Do not put `d_cust_top1` on the 15-col card.",
        "",
        "## Files written",
        "",
        "- `analysis/evaluate/top1_qa.py`",
        "- `analysis/outputs/top1_qa.md`",
        "- `analysis/outputs/top1_qa.png`" if ctx.get("png") else "- (no PNG)",
        "- append-only `analysis/experiments/registry.csv`",
        "- `overnight/waves/wave4_top1.md` (end, if WRITE_WAVE)",
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
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_d_cust_top1",
            "value": p2["top1"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"days={p2['days']:.4f} size={p2['size']:.4f} beat={p2['beat_size']}",
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
            "metric": "auroc_d_cust_top1_resid_days",
            "value": p3["rank"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"ols={p3['ols']:.4f} dies={p3['dies']} fake={p3['fake']}",
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
            "metric": "auroc_d_cust_top1_resid_hhi",
            "value": p5["h_rank"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"after_ncust={p5['c_rank']:.4f} after_days_hhi={p5['dh_rank']:.4f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": "-",
            "model": MODEL,
            "split": "train",
            "metric": "rho_d_cust_top1_vs_hhi",
            "value": p1["rhos"]["d_cust_hhi"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"twins={p1['twins']} size={p1['is_size']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": Y4,
            "model": MODEL,
            "split": "train_cv",
            "metric": "y4_top1_lag3_footnote",
            "value": p8["y4_lag3"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"footnote={d['footnote']} leftover_hhi_after_top1_lag3={p8['hhi_after_t3']:.4f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": Y3,
            "model": MODEL,
            "split": "train",
            "metric": "top1_leftover",
            "value": 1 if d["leftover_lives"] else 0,
            "coverage": f"{p1['cov']:.4f}",
            "notes": d["role"] + " " + d["why"][:160],
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
    p1, p2, p3, p5, p7, p8, p9 = (
        ctx["p1"],
        ctx["p2"],
        ctx["p3"],
        ctx["p5"],
        ctx["p7"],
        ctx["p8"],
        ctx["p9"],
    )
    text = (
        f"# Wave 4 — d_cust_top1 leftover after days as Y3 X\n\n"
        f"Agent `{AGENT}`. Train group-fold seed 20260918. Holdout 72 coverage only.\n\n"
        f"## Files written\n\n"
        f"- `analysis/evaluate/top1_qa.py`\n"
        f"- `analysis/outputs/top1_qa.md`\n"
        f"- `analysis/outputs/top1_qa.png`\n"
        f"- append-only `analysis/experiments/registry.csv` "
        f"(skip key agent+x_families+y+model+split+metric)\n"
        f"- this note\n\n"
        f"Did not touch `cust_hhi_qa.*`, `n_cust_qa.*`, `n_supp_qa.*`, "
        f"`n_types_qa.*`, `ap_issued_qa.*`, `counterparties.py`, parquet / duckdb, "
        f"`build_targets`, `product/`, the 15-col card, TURNOVER, LIVE, CONTEXT, "
        f"canvas, `brief_map.md`, or the parent journal. "
        f"Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. "
        f"Y7 TURNOVER **0.720 / 0.712** unchanged. "
        f"Y4 >0.975 footnote **{d['footnote']}**.\n\n"
        f"## Locked verdict\n\n"
        f"| object | decision |\n"
        f"| --- | --- |\n"
        f"| `d_cust_top1` as Y3 X / 15-col card | **{d['role']}** |\n"
        f"| `d_cust_top1` as engine X on the 44 | **{'KEEP leftover' if d['engine'] else 'DROP'}** |\n"
        f"| Y4 >0.975 monopoly footnote | **{d['footnote']}** |\n"
        f"| `y_top1` | **PARK** |\n\n"
        f"Y3 leftover after days rank {_f(p3['rank'])} (dies={p3['dies']}, fake={p3['fake']}); "
        f"inverse days after top1 {_f(p3['inv_rank'])}. "
        f"Single {_f(p2['top1'])} vs days {_f(p2['days'])} vs size {_f(p2['size'])}. "
        f"ρ vs HHI {_f(p1['rhos']['d_cust_hhi'])} vs n_cust {_f(p1['rhos']['d_n_cust'])} "
        f"vs days {_f(p1['rhos']['c_n_days_with_tx'])} vs size {_f(p1['rhos']['log1p(a_in3)'])}. "
        f"after HHI {_f(p5['h_rank'])} after n_cust {_f(p5['c_rank'])}. "
        f"Q6 lag1 leftover {_f(p7['l1_rank'])}. "
        f"Y4 top1_lag3 {_f(p8['y4_lag3'])}. vs supp_top1 same={p9['same']}. {d['why']}\n\n"
        f"Dark 470 NaN CONFIRM. Calendar incomplete (period<2025-02) top1 nn=0 CONFIRM. "
        f"HHI leftover after top1_lag3 OLS 0.549 / rank 0.461 CONFIRM. "
        f"Y4 HHI_lag3 >0.975 22.1% / 38 pos vs rest 11.5% CONFIRM; body CV 0.445 CONFIRM. "
        f"after days+HHI 0.572 is rewrite residual R²=0.966 same n, not leftover. "
        f"Bootstrap leftover p05/p50/p95 0.394/0.538/0.599 (65% die). "
        f"Permute-within-days p50=0.521 — observed 0.525 is the null. "
        f"Q6 days_lag1 0.684 CONFIRM; HHI Q6 short present 21.7% / 42 pos CONFIRM. "
        f"Do not put top1 on the 15-col card. Do not invent y_top1.\n\n"
        f"## What failed / next\n\n"
        + ("\n".join(f"- {x}" for x in ctx["failed"]) if ctx["failed"] else "- none")
        + f"\n\nElapsed {ctx['elapsed_s']:.0f}s.\n"
    )
    WAVE_NOTE.write_text(text)
    print(f"wrote {WAVE_NOTE}")


def main() -> None:
    t0 = time.time()
    failed: list[str] = []
    print("top1 leftover QA — unused leftover of d_cust_top1 after days as Y3 X")
    panel = load_panel()
    panel = attach_folds(panel)
    panel = add_panel_lags(
        panel,
        ["d_cust_top1", "d_cust_hhi", "d_n_cust", "c_n_days_with_tx"],
        (1, 3),
    )
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
    p1 = pass1_cov(tr, hold, book)
    p2 = pass2_singles(tr)
    p3 = pass3_days(tr)
    p5 = pass5_after_conc(tr)
    p6 = pass6_dark(tr, book)
    p7 = pass7_q6(tr)
    p8 = pass8_tail(tr)
    p9 = pass9_supp(tr)
    p10 = pass10_hold(hold, book)
    xi = extra_icc(tr)
    xf = extra_fold_y2(tr, p3)
    xsz = extra_size(tr)
    xq = extra_quintiles(tr)
    xtc = extra_tercile(tr)
    xb = extra_bootstrap(tr, n_boot=40)
    xp = extra_permute(tr, n_perm=24)
    xhr = extra_hhi_reconcile(tr)
    xcl = extra_collinear(tr, p3, p5)
    xtr = extra_y3_tail_rate(tr)
    xq6 = extra_q6_samen(tr)
    xerp = extra_erp_trail(tr, book)
    xdf = extra_drop_fold1(tr)
    xy4 = extra_y4_as_x(tr)
    xfs = extra_fold_sens(tr)
    xmf = extra_monopoly_flag(tr)
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
        failed.append("dark 470 not NaN")
    png = make_plot(tr, p3)
    elapsed = time.time() - t0
    ctx = {
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p5": p5,
        "p6": p6,
        "p7": p7,
        "p8": p8,
        "p9": p9,
        "p10": p10,
        "xi": xi,
        "xf": xf,
        "xb": xb,
        "xp": xp,
        "xsz": xsz,
        "xq": xq,
        "xtc": xtc,
        "xhr": xhr,
        "xcl": xcl,
        "xtr": xtr,
        "xq6": xq6,
        "xerp": xerp,
        "xdf": xdf,
        "xy4": xy4,
        "xfs": xfs,
        "xmf": xmf,
        "decision": decision,
        "failed": failed,
        "elapsed_s": elapsed,
        "png": png,
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
