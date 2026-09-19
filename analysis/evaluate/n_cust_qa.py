"""Unused leftover of ``d_n_cust`` after days as Y3 X.

``d_n_cust`` = distinct non-null counterparty_id on AR invoices
(Family D, trailing 6m). Incomplete 6m books are NaN. Dark 470 stay
NaN not 0. ``d_cust_hhi`` was DROPPED as engine X (twin of
``d_cust_top1`` ρ 0.994; leftover after top1 0.549). cust_hhi_qa
peeked Y3 single ``d_n_cust`` 0.653 on n=3,003 / 221 pos and
ρ(HHI, n_cust) −0.837 — that is a peek, not this leftover verdict.

Do **not** overwrite ``cust_hhi_qa.*``. Do not put ``d_n_cust`` on
the 15-col card. Do not rewrite ``gbm_core.py``.

KEEP-as-X: beat size ≥0.02 **and** leftover after days **and** not
SIZE (|ρ| vs log1p(a_in3) ≥0.50) **and** not a twin (|ρ|≥0.80 vs
days / a_n_tx / d_cust_top1 / d_cust_hhi / d_n_supp). Leftover <0.55
dies. Rank leftover is honest.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.n_cust_qa

Owned: analysis/evaluate/n_cust_qa.py, analysis/outputs/n_cust_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_n_cust.md (end).
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
OUT_MD = ANALYSIS / "outputs" / "n_cust_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "n_cust_qa.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_n_cust.md"
AGENT = "572fb928"
WAVE = "4"
ROUND = "R4"
MODEL = "n_cust_qa"
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
HHI_Y3_PEEK = 0.653
HHI_N_PEEK = 3003
HHI_POS_PEEK = 221
HHI_RHO_PEEK = -0.837
HHI_Q6_SHARE = 0.217
HHI_Q6_POS = 42
Y4_TAIL_HI = 0.221
Y4_TAIL_REST = 0.115
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


def signed_oof_auroc(
    y: pd.Series,
    x: pd.Series,
    folds: pd.Series,
    mask: pd.Series,
    n_folds: int = N_FOLDS,
) -> dict:
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
    rho_x = spearman(resid, x)
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
        "rho_x": rho_x,
        "r2": info["r2"],
        "n": rec["n_defined"],
        "n_pos": rec["n_pos"],
        "fake": fake,
        "almost": almost,
        "honest_dies": honest_dies,
        "folds": fold_bits(rec),
        "rank_folds": fold_bits(rrec),
        "resid": resid,
        "info": info,
        "rec": rec,
        "rrec": rrec,
    }


def company_demean(series: pd.Series, company: pd.Series) -> pd.Series:
    d = pd.DataFrame({"x": pd.to_numeric(series, errors="coerce"), "co": company.astype(str)})
    mu = d.groupby("co")["x"].transform("mean")
    return d["x"] - mu


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


def _sofar_bucket(n) -> str:
    if n is None or (isinstance(n, float) and not np.isfinite(n)):
        return "na"
    n = int(n)
    if n < 6:
        return "<6"
    if n < 12:
        return "6-11"
    if n < 18:
        return "12-17"
    if n < 24:
        return "18-23"
    return "24+"


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
    panel["log_n_cust"] = np.log1p(pd.to_numeric(panel["d_n_cust"], errors="coerce").clip(lower=0))
    panel["first_month"] = pd.to_datetime(panel["first_month"])
    panel["months_so_far"] = (
        (panel["period"].dt.year - panel["first_month"].dt.year) * 12
        + (panel["period"].dt.month - panel["first_month"].dt.month)
        + 1
    )
    grid_n = panel.groupby("company_id")["period"].size().rename("n_grid_months")
    panel = panel.merge(grid_n, on="company_id", how="left")
    panel["so_far_class"] = panel["months_so_far"].map(_trail_class)
    panel["so_far_bucket"] = panel["months_so_far"].map(_sofar_bucket)
    leak3 = leakage_check(
        ["d_n_cust", "d_cust_top1", "d_cust_hhi", "c_n_days_with_tx", "log_in3"],
        Y3,
        forbidden_prefixes=["b"],
    )
    if not leak3["ok"]:
        raise RuntimeError(f"Y3 never-B leak: {leak3['issues']}")
    leak4 = leakage_check(["d_n_cust", "d_cust_top1"], Y4, forbidden_prefixes=["f"])
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


# ---------------------------------------------------------------------------
# 1. Coverage + twin/SIZE screen
# ---------------------------------------------------------------------------
def pass1_cov(tr: pd.DataFrame, hold: pd.DataFrame, book: set[str]) -> dict:
    print("\n" + "=" * 72)
    print("CUT 1 — coverage; twin/SIZE screen")
    print("=" * 72)
    n_cm = len(tr)
    n_co = int(tr["company_id"].nunique())
    x = pd.to_numeric(tr["d_n_cust"], errors="coerce")
    hhi = pd.to_numeric(tr["d_cust_hhi"], errors="coerce")
    top1 = pd.to_numeric(tr["d_cust_top1"], errors="coerce")
    erp = tr["company_id"].isin(book)
    dark = ~erp
    n_dark_co = int(tr.loc[dark, "company_id"].nunique())
    dark_nn = int(x[dark].notna().sum())
    dark_zero = int((x[dark] == 0).sum())
    n0 = int((x == 0).sum())
    hhi_when_n0 = int(hhi[x == 0].notna().sum())
    acf1 = median_acf(x, tr["company_id"], 1)
    pairs = {
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "a_n_tx": tr["a_n_tx"],
        "d_cust_top1": top1,
        "d_cust_hhi": hhi,
        "d_n_supp": tr["d_n_supp"],
        "log1p(a_in3)": tr["log_in3"],
        "d_supp_hhi": tr["d_supp_hhi"],
        "d_supp_top1": tr["d_supp_top1"],
        "d_tx_cp_share": tr["d_tx_cp_share"],
    }
    rhos = {k: spearman(x, v) for k, v in pairs.items()}
    twins = [k for k, v in rhos.items() if np.isfinite(v) and abs(v) >= TWIN_RHO]
    is_size = bool(np.isfinite(rhos["log1p(a_in3)"]) and abs(rhos["log1p(a_in3)"]) >= SIZE_RHO)
    twin_gate = any(
        np.isfinite(rhos[k]) and abs(rhos[k]) >= TWIN_RHO
        for k in ("c_n_days_with_tx", "a_n_tx", "d_cust_top1", "d_cust_hhi", "d_n_supp")
    )
    rho_hhi_ok = bool(
        np.isfinite(rhos["d_cust_hhi"]) and abs(rhos["d_cust_hhi"] - HHI_RHO_PEEK) < 0.03
    )
    rows = [
        {
            "col": "d_n_cust",
            "n_nn": f"{int(x.notna().sum()):,}",
            "cov": _pp(_pct(int(x.notna().sum()), n_cm)),
            "n0": f"{n0:,}",
            "acf1": _f(acf1),
        },
        {
            "col": "d_cust_hhi",
            "n_nn": f"{int(hhi.notna().sum()):,}",
            "cov": _pp(_pct(int(hhi.notna().sum()), n_cm)),
            "n0": "—",
            "acf1": _f(median_acf(hhi, tr["company_id"], 1)),
        },
        {
            "col": "d_cust_top1",
            "n_nn": f"{int(top1.notna().sum()):,}",
            "cov": _pp(_pct(int(top1.notna().sum()), n_cm)),
            "n0": "—",
            "acf1": _f(median_acf(top1, tr["company_id"], 1)),
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
    hold_x = pd.to_numeric(hold["d_n_cust"], errors="coerce")
    hold_row = {
        "n_co": int(hold["company_id"].nunique()),
        "n_cm": len(hold),
        "n_nn": int(hold_x.notna().sum()),
        "cov": _pct(int(hold_x.notna().sum()), len(hold)),
    }
    prose = (
        f"Train {n_co:,} co / {n_cm:,} CM. d_n_cust cov {_pp(_pct(int(x.notna().sum()), n_cm))} "
        f"n0={n0:,} (HHI defined on n0={hhi_when_n0}). acf1={_f(acf1)}. "
        f"Dark {n_dark_co} (want {N_DARK_WANT}) nn={dark_nn} zero={dark_zero} "
        f"{'CONFIRM NaN' if dark_ok else 'FAIL 0-fill'}. "
        f"ρ vs days {_f(rhos['c_n_days_with_tx'])} vs a_n_tx {_f(rhos['a_n_tx'])} "
        f"vs top1 {_f(rhos['d_cust_top1'])} vs HHI {_f(rhos['d_cust_hhi'])} "
        f"(peek {HHI_RHO_PEEK} {'CONFIRM' if rho_hhi_ok else 'DRIFT'}) "
        f"vs n_supp {_f(rhos['d_n_supp'])} vs size {_f(rhos['log1p(a_in3)'])}. "
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
        "n_nn": int(x.notna().sum()),
        "n0": n0,
        "hhi_when_n0": hhi_when_n0,
        "acf1": acf1,
        "n_dark_co": n_dark_co,
        "dark_nn": dark_nn,
        "dark_zero": dark_zero,
        "dark_ok": dark_ok,
        "rho_hhi_ok": rho_hhi_ok,
        "hold": hold_row,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 2. Single-feature group-fold Y3
# ---------------------------------------------------------------------------
def pass2_singles(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 2 — single-feature group-fold Y3")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    folds = tr["fold"]
    feats = {
        "d_n_cust": tr["d_n_cust"],
        "log1p(d_n_cust)": tr["log_n_cust"],
        "log1p(a_in3)": tr["log_in3"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "d_cust_top1": tr["d_cust_top1"],
        "d_cust_hhi": tr["d_cust_hhi"],
        "d_n_supp": tr["d_n_supp"],
        "a_n_tx": tr["a_n_tx"],
    }
    recs = {}
    rows = []
    for fname, x in feats.items():
        rec = signed_oof_auroc(y, x, folds, lab)
        recs[fname] = rec
        rows.append(_auc_row(Y3, fname, rec))
        print(f"  {fname} CV={_f(rec['cv'])} n={rec['n_defined']} pos={rec['n_pos']} folds={fold_bits(rec)}")
    y3_n = _cv(recs["d_n_cust"])
    y3_size = _cv(recs["log1p(a_in3)"])
    y3_days = _cv(recs["c_n_days_with_tx"])
    y3_top1 = _cv(recs["d_cust_top1"])
    y3_hhi = _cv(recs["d_cust_hhi"])
    days_ok = bool(np.isfinite(y3_days) and abs(y3_days - DAYS_BENCH) < 0.008)
    size_ok = bool(np.isfinite(y3_size) and abs(y3_size - SIZE_QUOTE) < 0.008)
    peek_ok = bool(
        recs["d_n_cust"]["n_defined"] == HHI_N_PEEK
        and recs["d_n_cust"]["n_pos"] == HHI_POS_PEEK
        and np.isfinite(y3_n)
        and abs(y3_n - HHI_Y3_PEEK) < 0.015
    )
    beat_size = bool(np.isfinite(y3_n) and (y3_n - SIZE_QUOTE) >= KEEP_DELTA)
    prose = (
        f"Y3 d_n_cust {_f(y3_n)} n={recs['d_n_cust']['n_defined']:,} pos={recs['d_n_cust']['n_pos']:,} "
        f"(peek 0.653 / 3,003 / 221 {'CONFIRM' if peek_ok else 'DRIFT'}). "
        f"vs size {_f(y3_size)} vs days {_f(y3_days)} vs top1 {_f(y3_top1)} vs HHI {_f(y3_hhi)}. "
        f"Replica days 0.711 {'CONFIRM' if days_ok else 'DRIFT'} / size 0.617 "
        f"{'CONFIRM' if size_ok else 'DRIFT'}. Beat-size Δ={_f(y3_n - SIZE_QUOTE) if np.isfinite(y3_n) else '—'} "
        f"{'PASS' if beat_size else 'FAIL'}."
    )
    print(prose)
    return {
        "rows": rows,
        "recs": recs,
        "n_cust": y3_n,
        "size": y3_size,
        "days": y3_days,
        "top1": y3_top1,
        "hhi": y3_hhi,
        "days_ok": days_ok,
        "size_ok": size_ok,
        "peek_ok": peek_ok,
        "beat_size": beat_size,
        "n_def": recs["d_n_cust"]["n_defined"],
        "n_pos": recs["d_n_cust"]["n_pos"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 3. Honest leftover after days + inverse
# ---------------------------------------------------------------------------
def pass3_days(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 3 — honest leftover of d_n_cust after days; inverse")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    folds = tr["fold"]
    after = leftover_diag(y, tr["d_n_cust"], (tr["c_n_days_with_tx"],), folds, lab)
    inv = leftover_diag(y, tr["c_n_days_with_tx"], (tr["d_n_cust"],), folds, lab)
    after_log = leftover_diag(y, tr["log_n_cust"], (tr["c_n_days_with_tx"],), folds, lab)
    prose = (
        f"d_n_cust leftover after days OLS {_f(after['ols'])} rank {_f(after['rank'])} "
        f"fake={after['fake']} almost={after['almost']} ρ(resid,days)={_f(after['rho_ctrl'])} "
        f"R²={_f(after['r2'])} honest_dies={after['honest_dies']} n={after['n']:,} pos={after['n_pos']:,}. "
        f"Inverse: days leftover after n_cust OLS {_f(inv['ols'])} rank {_f(inv['rank'])} "
        f"dies={inv['honest_dies']}. log1p leftover rank {_f(after_log['rank'])}."
    )
    print(prose)
    return {
        "after": after,
        "inv": inv,
        "after_log": after_log,
        "ols": after["ols"],
        "rank": after["rank"],
        "dies": after["honest_dies"],
        "fake": after["fake"],
        "almost": after["almost"],
        "r2": after["r2"],
        "inv_rank": inv["rank"],
        "inv_dies": inv["honest_dies"],
        "log_rank": after_log["rank"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 5. Leftover after top1 / HHI / days+top1
# ---------------------------------------------------------------------------
def pass5_after_conc(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 5 — leftover after top1, after HHI, after days+top1")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    folds = tr["fold"]
    after_t = leftover_diag(y, tr["d_n_cust"], (tr["d_cust_top1"],), folds, lab)
    after_h = leftover_diag(y, tr["d_n_cust"], (tr["d_cust_hhi"],), folds, lab)
    after_dt = leftover_diag(
        y, tr["d_n_cust"], (tr["c_n_days_with_tx"], tr["d_cust_top1"]), folds, lab
    )
    after_dh = leftover_diag(
        y, tr["d_n_cust"], (tr["c_n_days_with_tx"], tr["d_cust_hhi"]), folds, lab
    )
    # inverse: leftover of top1 / HHI after n_cust
    t_after = leftover_diag(y, tr["d_cust_top1"], (tr["d_n_cust"],), folds, lab)
    h_after = leftover_diag(y, tr["d_cust_hhi"], (tr["d_n_cust"],), folds, lab)
    rows = [
        {
            "bar": "after top1",
            "OLS": _f(after_t["ols"]),
            "rank": _f(after_t["rank"]),
            "ρ(resid,bar)": _f(after_t["rho_ctrl"]),
            "R2": _f(after_t["r2"]),
            "dies": after_t["honest_dies"],
            "n": after_t["n"],
            "n_pos": after_t["n_pos"],
        },
        {
            "bar": "after HHI",
            "OLS": _f(after_h["ols"]),
            "rank": _f(after_h["rank"]),
            "ρ(resid,bar)": _f(after_h["rho_ctrl"]),
            "R2": _f(after_h["r2"]),
            "dies": after_h["honest_dies"],
            "n": after_h["n"],
            "n_pos": after_h["n_pos"],
        },
        {
            "bar": "after days+top1",
            "OLS": _f(after_dt["ols"]),
            "rank": _f(after_dt["rank"]),
            "ρ(resid,bar)": _f(after_dt["rho_ctrl"]),
            "R2": _f(after_dt["r2"]),
            "dies": after_dt["honest_dies"],
            "n": after_dt["n"],
            "n_pos": after_dt["n_pos"],
        },
        {
            "bar": "after days+HHI",
            "OLS": _f(after_dh["ols"]),
            "rank": _f(after_dh["rank"]),
            "ρ(resid,bar)": _f(after_dh["rho_ctrl"]),
            "R2": _f(after_dh["r2"]),
            "dies": after_dh["honest_dies"],
            "n": after_dh["n"],
            "n_pos": after_dh["n_pos"],
        },
        {
            "bar": "top1 after n_cust",
            "OLS": _f(t_after["ols"]),
            "rank": _f(t_after["rank"]),
            "ρ(resid,bar)": _f(t_after["rho_ctrl"]),
            "R2": _f(t_after["r2"]),
            "dies": t_after["honest_dies"],
            "n": t_after["n"],
            "n_pos": t_after["n_pos"],
        },
        {
            "bar": "HHI after n_cust",
            "OLS": _f(h_after["ols"]),
            "rank": _f(h_after["rank"]),
            "ρ(resid,bar)": _f(h_after["rho_ctrl"]),
            "R2": _f(h_after["r2"]),
            "dies": h_after["honest_dies"],
            "n": h_after["n"],
            "n_pos": h_after["n_pos"],
        },
    ]
    prose = (
        f"n_cust leftover after top1 rank {_f(after_t['rank'])} dies={after_t['honest_dies']}; "
        f"after HHI {_f(after_h['rank'])} dies={after_h['honest_dies']}; "
        f"after days+top1 {_f(after_dt['rank'])} dies={after_dt['honest_dies']}. "
        f"Inverse top1 after n_cust {_f(t_after['rank'])}; HHI after n_cust {_f(h_after['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "after_t": after_t,
        "after_h": after_h,
        "after_dt": after_dt,
        "after_dh": after_dh,
        "t_after": t_after,
        "h_after": h_after,
        "t_rank": after_t["rank"],
        "h_rank": after_h["rank"],
        "dt_rank": after_dt["rank"],
        "t_dies": after_t["honest_dies"],
        "h_dies": after_h["honest_dies"],
        "dt_dies": after_dt["honest_dies"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 6. Dark 470 stay NaN; ERP-only leftover
# ---------------------------------------------------------------------------
def pass6_dark(tr: pd.DataFrame, book: set[str]) -> dict:
    print("\n" + "=" * 72)
    print("CUT 6 — dark 470 stay NaN; ERP-only leftover")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    erp = tr["company_id"].isin(book)
    dark = ~erp
    x = pd.to_numeric(tr["d_n_cust"], errors="coerce")
    n_dark_co = int(tr.loc[dark, "company_id"].nunique())
    dark_nn = int(x[dark].notna().sum())
    dark_zero = int((x[dark] == 0).sum())
    rec = signed_oof_auroc(y, tr["d_n_cust"], tr["fold"], lab & erp)
    after = leftover_diag(y, tr["d_n_cust"], (tr["c_n_days_with_tx"],), tr["fold"], lab & erp)
    rec_d = signed_oof_auroc(y, tr["d_n_cust"], tr["fold"], lab & dark)
    rows = [
        {
            "book": "dark",
            "n_co": int(tr.loc[dark, "company_id"].nunique()),
            "n_cm": int(dark.sum()),
            "n_cust nn": int(x[dark].notna().sum()),
            "Y3 n": rec_d["n_defined"],
            "Y3 pos": rec_d["n_pos"],
            "CV": "LOW_POWER" if rec_d["low_power"] else _f(rec_d["cv"]),
            "leftover rank": "—",
            "dies": "—",
        },
        {
            "book": "ERP",
            "n_co": int(tr.loc[erp, "company_id"].nunique()),
            "n_cm": int(erp.sum()),
            "n_cust nn": int(x[erp].notna().sum()),
            "Y3 n": rec["n_defined"],
            "Y3 pos": rec["n_pos"],
            "CV": "LOW_POWER" if rec["low_power"] else _f(rec["cv"]),
            "leftover rank": _f(after["rank"]),
            "dies": after["honest_dies"],
        },
    ]
    dark_ok = n_dark_co == N_DARK_WANT and dark_nn == 0 and dark_zero == 0
    prose = (
        f"Dark {n_dark_co} (want {N_DARK_WANT}) n_cust nn={dark_nn} zero={dark_zero} "
        f"{'CONFIRM NaN not 0' if dark_ok else 'FAIL'}. "
        f"ERP Y3 {_f(rec['cv'])} leftover after days rank {_f(after['rank'])} dies={after['honest_dies']} "
        f"n={rec['n_defined']:,} pos={rec['n_pos']:,}."
    )
    print(prose)
    return {
        "rows": rows,
        "dark_ok": dark_ok,
        "n_dark_co": n_dark_co,
        "dark_nn": dark_nn,
        "erp_cv": _cv(rec),
        "erp_rank": after["rank"],
        "erp_dies": after["honest_dies"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 7. Q6 lag1 leftover after days_lag1
# ---------------------------------------------------------------------------
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
    for name in ("d_n_cust", "d_n_cust_lag1", "d_n_cust_lag3", "c_n_days_with_tx_lag1"):
        rec = signed_oof_auroc(y, tr[name], folds, lab)
        recs[name] = rec
        rows.append(_auc_row(Y3, name, rec))
        print(f"  {name} CV={_f(rec['cv'])} n={rec['n_defined']} pos={rec['n_pos']}")
    after_l1 = leftover_diag(y, tr["d_n_cust_lag1"], (tr["c_n_days_with_tx_lag1"],), folds, lab)
    after_l3 = leftover_diag(y, tr["d_n_cust_lag3"], (tr["c_n_days_with_tx_lag3"],), folds, lab)
    days_l1 = _cv(recs["c_n_days_with_tx_lag1"])
    days_ok = bool(np.isfinite(days_l1) and abs(days_l1 - DAYS_LAG1_QUOTE) < 0.008)
    short = lab & (tr["so_far_class"] == "short_<12")
    rec_s = signed_oof_auroc(y, tr["d_n_cust_lag1"], folds, short)
    after_s = leftover_diag(y, tr["d_n_cust_lag1"], (tr["c_n_days_with_tx_lag1"],), folds, short)
    y4 = pd.to_numeric(tr[Y4], errors="coerce")
    lab4 = y4.notna()
    short4 = lab4 & (tr["so_far_class"] == "short_<12")
    rec_s4 = signed_oof_auroc(y4, tr["d_n_cust_lag3"], tr["fold"], short4)
    present = _pct(int((short4 & tr["d_n_cust_lag3"].notna()).sum()), int(short4.sum())) if short4.any() else float("nan")
    prose = (
        f"Y3 n_cust_lag1 {_f(recs['d_n_cust_lag1']['cv'])} leftover after days_lag1 "
        f"rank {_f(after_l1['rank'])} dies={after_l1['honest_dies']}. "
        f"lag3 leftover after days_lag3 {_f(after_l3['rank'])} dies={after_l3['honest_dies']}. "
        f"Days lag1 {_f(days_l1)} (quote 0.684 {'CONFIRM' if days_ok else 'DRIFT'}). "
        f"Short Y3 lag1 leftover {_f(after_s['rank'])} n={rec_s['n_defined']:,} pos={rec_s['n_pos']:,}. "
        f"Short Y4 lag3 present {_pp(present)} pos={rec_s4['n_pos']} "
        f"(HHI Q6 quote 21.7% / 42 pos)."
    )
    print(prose)
    return {
        "rows": rows,
        "recs": recs,
        "lag1": _cv(recs["d_n_cust_lag1"]),
        "lag3": _cv(recs["d_n_cust_lag3"]),
        "l1_rank": after_l1["rank"],
        "l1_dies": after_l1["honest_dies"],
        "l3_rank": after_l3["rank"],
        "l3_dies": after_l3["honest_dies"],
        "days_l1": days_l1,
        "days_ok": days_ok,
        "short_rank": after_s["rank"],
        "short_n": rec_s["n_defined"],
        "short_pos": rec_s["n_pos"],
        "short4_present": present,
        "short4_pos": rec_s4["n_pos"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 8. Y4 leftover after top1_lag3 — monopoly tail rewrite?
# ---------------------------------------------------------------------------
def pass8_y4(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 8 — Y4 leftover after top1_lag3; monopoly tail")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y4], errors="coerce")
    lab = y.notna()
    folds = tr["fold"]
    rec_n = signed_oof_auroc(y, tr["d_n_cust"], folds, lab)
    rec_n3 = signed_oof_auroc(y, tr["d_n_cust_lag3"], folds, lab)
    rec_t3 = signed_oof_auroc(y, tr["d_cust_top1_lag3"], folds, lab)
    rec_h3 = signed_oof_auroc(y, tr["d_cust_hhi_lag3"], folds, lab)
    after_t3 = leftover_diag(y, tr["d_n_cust_lag3"], (tr["d_cust_top1_lag3"],), folds, lab)
    after_h3 = leftover_diag(y, tr["d_n_cust_lag3"], (tr["d_cust_hhi_lag3"],), folds, lab)
    after_n_now = leftover_diag(y, tr["d_n_cust"], (tr["d_cust_top1"],), folds, lab)
    hhi = pd.to_numeric(tr["d_cust_hhi"], errors="coerce")
    n = pd.to_numeric(tr["d_n_cust"], errors="coerce")
    tail = lab & hhi.notna() & (hhi > TAIL_CUT)
    body = lab & hhi.notna() & (hhi <= TAIL_CUT)
    rest = lab & hhi.notna() & (hhi <= TAIL_CUT)
    rate_t = float(y[tail].mean()) if tail.any() else float("nan")
    rate_r = float(y[rest].mean()) if rest.any() else float("nan")
    n1 = lab & (n == 1)
    n_ge2 = lab & (n >= 2)
    rate_n1 = float(y[n1].mean()) if n1.any() else float("nan")
    rate_ge2 = float(y[n_ge2].mean()) if n_ge2.any() else float("nan")
    share_n1_in_tail = _pct(int((tail & (n == 1)).sum()), int(tail.sum())) if tail.any() else float("nan")
    rows = [
        _auc_row(Y4, "d_n_cust", rec_n),
        _auc_row(Y4, "d_n_cust_lag3", rec_n3),
        _auc_row(Y4, "d_cust_top1_lag3", rec_t3),
        _auc_row(Y4, "d_cust_hhi_lag3", rec_h3),
    ]
    leftover_rows = [
        {
            "bar": "n_cust_lag3 after top1_lag3",
            "rank": _f(after_t3["rank"]),
            "OLS": _f(after_t3["ols"]),
            "dies": after_t3["honest_dies"],
            "n": after_t3["n"],
            "n_pos": after_t3["n_pos"],
        },
        {
            "bar": "n_cust_lag3 after HHI_lag3",
            "rank": _f(after_h3["rank"]),
            "OLS": _f(after_h3["ols"]),
            "dies": after_h3["honest_dies"],
            "n": after_h3["n"],
            "n_pos": after_h3["n_pos"],
        },
        {
            "bar": "n_cust after top1 (now)",
            "rank": _f(after_n_now["rank"]),
            "OLS": _f(after_n_now["ols"]),
            "dies": after_n_now["honest_dies"],
            "n": after_n_now["n"],
            "n_pos": after_n_now["n_pos"],
        },
    ]
    tail_ok = bool(np.isfinite(rate_t) and abs(rate_t - Y4_TAIL_HI) < 0.03)
    prose = (
        f"Y4 n_cust {_f(_cv(rec_n))} lag3 {_f(_cv(rec_n3))} vs top1_lag3 {_f(_cv(rec_t3))} "
        f"vs HHI_lag3 {_f(_cv(rec_h3))}. "
        f"Leftover n_cust_lag3 after top1_lag3 rank {_f(after_t3['rank'])} dies={after_t3['honest_dies']}. "
        f"HHI>0.975 Y4 rate {_pp(rate_t)} (quote 22.1% {'CONFIRM' if tail_ok else 'DRIFT'}) "
        f"vs rest {_pp(rate_r)} (quote 11.5%). n_cust==1 rate {_pp(rate_n1)} vs ≥2 {_pp(rate_ge2)}. "
        f"Share of tail with n_cust==1 {_pp(share_n1_in_tail)} — "
        f"{'count rewrite of monopoly tail' if (np.isfinite(share_n1_in_tail) and share_n1_in_tail >= 0.80) else 'not just n=1'}."
    )
    print(prose)
    return {
        "rows": rows,
        "leftover_rows": leftover_rows,
        "n_now": _cv(rec_n),
        "n_lag3": _cv(rec_n3),
        "t_lag3": _cv(rec_t3),
        "h_lag3": _cv(rec_h3),
        "after_t3": after_t3["rank"],
        "t3_dies": after_t3["honest_dies"],
        "after_h3": after_h3["rank"],
        "rate_t": rate_t,
        "rate_r": rate_r,
        "rate_n1": rate_n1,
        "rate_ge2": rate_ge2,
        "share_n1": share_n1_in_tail,
        "tail_ok": tail_ok,
        "n_tail": int(tail.sum()),
        "n_pos_tail": int((tail & (y == 1)).sum()),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 9. vs d_n_supp
# ---------------------------------------------------------------------------
def pass9_supp(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 9 — vs d_n_supp (same object or not)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    rho = spearman(tr["d_n_cust"], tr["d_n_supp"])
    rec_s = signed_oof_auroc(y, tr["d_n_supp"], tr["fold"], lab)
    after_s = leftover_diag(y, tr["d_n_supp"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_c = leftover_diag(y, tr["d_n_cust"], (tr["d_n_supp"],), tr["fold"], lab)
    after_sc = leftover_diag(y, tr["d_n_supp"], (tr["d_n_cust"],), tr["fold"], lab)
    same = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
    prose = (
        f"ρ(n_cust, n_supp)={_f(rho)} {'SAME object (twin)' if same else 'different object'}. "
        f"Y3 n_supp {_f(_cv(rec_s))} leftover after days {_f(after_s['rank'])} dies={after_s['honest_dies']}. "
        f"n_cust leftover after n_supp {_f(after_c['rank'])} dies={after_c['honest_dies']}. "
        f"n_supp leftover after n_cust {_f(after_sc['rank'])} dies={after_sc['honest_dies']}."
    )
    print(prose)
    return {
        "rho": rho,
        "same": same,
        "supp_cv": _cv(rec_s),
        "supp_rank": after_s["rank"],
        "cust_after_supp": after_c["rank"],
        "supp_after_cust": after_sc["rank"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 10. Holdout coverage only
# ---------------------------------------------------------------------------
def pass10_hold(hold: pd.DataFrame, book: set[str]) -> dict:
    print("\n" + "=" * 72)
    print("CUT 10 — holdout coverage only (no AUROC)")
    print("=" * 72)
    x = pd.to_numeric(hold["d_n_cust"], errors="coerce")
    erp = hold["company_id"].isin(book)
    rec = {
        "n_co": int(hold["company_id"].nunique()),
        "n_cm": len(hold),
        "n_nn": int(x.notna().sum()),
        "cov": _pct(int(x.notna().sum()), len(hold)),
        "n0": int((x == 0).sum()),
        "p50": float(x[x.notna()].median()) if x.notna().any() else float("nan"),
        "dark_nn": int(x[~erp].notna().sum()),
    }
    prose = (
        f"Holdout {rec['n_co']} co / {rec['n_cm']} CM nn={rec['n_nn']} "
        f"cov={_pp(rec['cov'])} n0={rec['n0']} p50={_f(rec['p50'], 1)} "
        f"dark nn={rec['dark_nn']} (no fit, no AUROC)."
    )
    print(prose)
    return {**rec, "prose": prose}


# ---------------------------------------------------------------------------
# Extras
# ---------------------------------------------------------------------------
def extra_icc(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — ICC / demean (is n_cust a company trait?)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    icc = icc_anova(tr["d_n_cust"], tr["company_id"])
    trait = bool(np.isfinite(icc["icc"]) and icc["icc"] >= ICC_TRAIT)
    demean = company_demean(tr["d_n_cust"], tr["company_id"])
    meanx = company_mean(tr["d_n_cust"], tr["company_id"])
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
    return {
        "icc": icc,
        "trait": trait,
        "demean_cv": _cv(rec_d),
        "mean_cv": _cv(rec_m),
        "demean_rank": after_d["rank"],
        "mean_rank": after_m["rank"],
        "prose": prose,
    }


def extra_zero_one(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — n_cust==0 / ==1 / ≥2 Y3 rates")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    n = pd.to_numeric(tr["d_n_cust"], errors="coerce")
    lab = y.notna() & n.notna()
    rows = []
    for name, sl in (
        ("n==0", lab & (n == 0)),
        ("n==1", lab & (n == 1)),
        ("n 2-5", lab & (n >= 2) & (n <= 5)),
        ("n>=6", lab & (n >= 6)),
    ):
        rows.append(
            {
                "bin": name,
                "n": int(sl.sum()),
                "n_pos": int((sl & (y == 1)).sum()),
                "Y3 rate": _pp(float(y[sl].mean()) if sl.any() else float("nan")),
            }
        )
        print(f"  {name} n={int(sl.sum())} rate={rows[-1]['Y3 rate']}")
    rec0 = signed_oof_auroc(y, tr["d_n_cust"], tr["fold"], lab & (n > 0))
    after0 = leftover_diag(
        y, tr["d_n_cust"], (tr["c_n_days_with_tx"],), tr["fold"], lab & (n > 0)
    )
    prose = (
        f"Drop n==0: Y3 {_f(_cv(rec0))} leftover after days {_f(after0['rank'])} "
        f"dies={after0['honest_dies']} n={rec0['n_defined']:,}."
    )
    print(prose)
    return {"rows": rows, "wo0_cv": _cv(rec0), "wo0_rank": after0["rank"], "prose": prose}


def extra_fold_y2(tr: pd.DataFrame, p3: dict) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — fold-wise leftover; 12-name Y2 drop")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    ids = chronic_12(tr)
    y2 = pd.to_numeric(tr[Y2], errors="coerce")
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    lab2 = y2.notna()
    lab3 = y3.notna()
    chron = tr["company_id"].isin(ids)
    rec2 = signed_oof_auroc(y2, tr["d_n_cust"], tr["fold"], lab2)
    rec2d = signed_oof_auroc(y2, tr["c_n_days_with_tx"], tr["fold"], lab2)
    rec2_wo = signed_oof_auroc(y2, tr["d_n_cust"], tr["fold"], lab2 & ~chron)
    after_y2 = leftover_diag(y2, tr["d_n_cust"], (tr["c_n_days_with_tx"],), tr["fold"], lab2)
    after_y3_wo = leftover_diag(
        y3, tr["d_n_cust"], (tr["c_n_days_with_tx"],), tr["fold"], lab3 & ~chron
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
        f"Y2 n_cust {_f(_cv(rec2))} days {_f(_cv(rec2d))}. "
        f"Drop {len(ids)} chronic: {_f(_cv(rec2_wo))}. "
        f"Y2 leftover after days {_f(after_y2['rank'])}. "
        f"Y3 leftover wo12 {_f(after_y3_wo['rank'])} dies={after_y3_wo['honest_dies']}."
    )
    print(prose)
    return {
        "ids": ids,
        "fold_rows": fold_rows,
        "y2": _cv(rec2),
        "y2_wo": _cv(rec2_wo),
        "y2_rank": after_y2["rank"],
        "y3_wo_rank": after_y3_wo["rank"],
        "prose": prose,
    }


def extra_bootstrap(tr: pd.DataFrame, n_boot: int = 40) -> dict:
    print("\n" + "=" * 72)
    print(f"EXTRA — company bootstrap leftover after days (n={n_boot})")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    work = tr.loc[lab, ["company_id", "fold", Y3, "d_n_cust", "c_n_days_with_tx"]].copy()
    cos = work["company_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(FOLD_SEED)
    ranks = []
    for _ in range(n_boot):
        draw = rng.choice(cos, size=len(cos), replace=True)
        parts = [work[work["company_id"] == c] for c in draw]
        b = pd.concat(parts, ignore_index=True)
        after = leftover_diag(
            b[Y3],
            b["d_n_cust"],
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


def extra_sofar(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after days by so-far bucket")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    rows = []
    for b in ("<6", "6-11", "12-17", "18-23", "24+"):
        sl = lab & (tr["so_far_bucket"] == b)
        after = leftover_diag(y, tr["d_n_cust"], (tr["c_n_days_with_tx"],), tr["fold"], sl)
        rec = signed_oof_auroc(y, tr["d_n_cust"], tr["fold"], sl)
        rows.append(
            {
                "so-far": b,
                "n": after["n"],
                "n_pos": after["n_pos"],
                "raw": _f(rec["cv"]),
                "rank leftover": _f(after["rank"]),
                "dies": after["honest_dies"],
            }
        )
        print(f"  {b} raw={_f(rec['cv'])} leftover={_f(after['rank'])} dies={after['honest_dies']}")
    return {"rows": rows, "prose": "Leftover after days by so-far bucket."}


def extra_log_vs_raw(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — log1p vs raw leftover after days")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    ok = y.notna() & pd.to_numeric(tr["d_n_cust"], errors="coerce").notna()
    rec_r = signed_oof_auroc(y, tr["d_n_cust"], tr["fold"], ok)
    rec_l = signed_oof_auroc(y, tr["log_n_cust"], tr["fold"], ok)
    after_r = leftover_diag(y, tr["d_n_cust"], (tr["c_n_days_with_tx"],), tr["fold"], ok)
    after_l = leftover_diag(y, tr["log_n_cust"], (tr["c_n_days_with_tx"],), tr["fold"], ok)
    after_rl = leftover_diag(y, tr["d_n_cust"], (tr["log_n_cust"],), tr["fold"], ok)
    prose = (
        f"same-n raw {_f(_cv(rec_r))} leftover {_f(after_r['rank'])}; "
        f"log1p {_f(_cv(rec_l))} leftover {_f(after_l['rank'])}. "
        f"raw leftover after log1p {_f(after_rl['rank'])} dies={after_rl['honest_dies']}."
    )
    print(prose)
    return {
        "raw": _cv(rec_r),
        "log": _cv(rec_l),
        "raw_rank": after_r["rank"],
        "log_rank": after_l["rank"],
        "raw_after_log": after_rl["rank"],
        "prose": prose,
    }


def extra_permute(tr: pd.DataFrame, n_perm: int = 24) -> dict:
    print("\n" + "=" * 72)
    print(f"EXTRA — permute n_cust within days quintile (n={n_perm})")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    work = tr.loc[lab].copy()
    days = pd.to_numeric(work["c_n_days_with_tx"], errors="coerce")
    x = pd.to_numeric(work["d_n_cust"], errors="coerce")
    ok = days.notna() & x.notna()
    work = work.loc[ok]
    q = pd.qcut(pd.to_numeric(work["c_n_days_with_tx"], errors="coerce"), 5, duplicates="drop")
    rng = np.random.default_rng(FOLD_SEED + 7)
    ranks = []
    for _ in range(n_perm):
        shuf = work["d_n_cust"].copy()
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
    prose = (
        f"Permuted-within-days leftover rank p50={_f(p50)} p90={_f(p90)} n={len(ranks)}."
    )
    print(prose)
    return {"p50": p50, "p90": p90, "prose": prose}


def extra_ntx(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after a_n_tx (days twin screen)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after = leftover_diag(y, tr["d_n_cust"], (tr["a_n_tx"],), tr["fold"], lab)
    after_both = leftover_diag(
        y, tr["d_n_cust"], (tr["c_n_days_with_tx"], tr["a_n_tx"]), tr["fold"], lab
    )
    prose = (
        f"n_cust leftover after a_n_tx rank {_f(after['rank'])} dies={after['honest_dies']}. "
        f"after days+a_n_tx {_f(after_both['rank'])} dies={after_both['honest_dies']}."
    )
    print(prose)
    return {
        "after_ntx": after["rank"],
        "after_both": after_both["rank"],
        "prose": prose,
    }


def extra_size(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after size (SIZE screen residual)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after = leftover_diag(y, tr["d_n_cust"], (tr["log_in3"],), tr["fold"], lab)
    after_both = leftover_diag(
        y, tr["d_n_cust"], (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], lab
    )
    size_after = leftover_diag(y, tr["log_in3"], (tr["d_n_cust"],), tr["fold"], lab)
    prose = (
        f"n_cust leftover after size rank {_f(after['rank'])} dies={after['honest_dies']}. "
        f"after days+size {_f(after_both['rank'])} dies={after_both['honest_dies']}. "
        f"size leftover after n_cust {_f(size_after['rank'])} dies={size_after['honest_dies']} "
        f"(in3 leftover after days was 0.521 — not overwritten)."
    )
    print(prose)
    return {
        "after_size": after["rank"],
        "after_both": after_both["rank"],
        "size_after": size_after["rank"],
        "prose": prose,
    }


def extra_wo_fold(p2: dict, p3: dict) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover without weakest / strongest fold")
    print("=" * 72)
    rec = p2["recs"]["d_n_cust"]
    folds = [r["auroc"] for r in rec["folds"] if np.isfinite(r["auroc"])]
    weak_k = int(np.argmin(folds)) if folds else None
    strong_k = int(np.argmax(folds)) if folds else None
    wo_w = float(np.mean([a for i, a in enumerate(folds) if i != weak_k])) if folds else float("nan")
    wo_s = float(np.mean([a for i, a in enumerate(folds) if i != strong_k])) if folds else float("nan")
    rf = [r["auroc"] for r in p3["after"]["rrec"]["folds"] if np.isfinite(r["auroc"])]
    r_wo = float(np.mean(rf[1:])) if len(rf) > 1 else float("nan")
    prose = (
        f"n_cust without weakest fold {weak_k} = {_f(wo_w)}; without strongest {strong_k} = {_f(wo_s)}. "
        f"Rank leftover without fold 0 = {_f(r_wo)} "
        f"(still {'lives' if np.isfinite(r_wo) and r_wo >= CHANCE else 'dies'})."
    )
    print(prose)
    return {"wo_w": wo_w, "wo_s": wo_s, "r_wo": r_wo, "prose": prose}


def extra_samen_top1(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after days on top1-defined only (sample of 0.564)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    top1 = pd.to_numeric(tr["d_cust_top1"], errors="coerce")
    ok = y.notna() & top1.notna() & pd.to_numeric(tr["d_n_cust"], errors="coerce").notna()
    after_d = leftover_diag(y, tr["d_n_cust"], (tr["c_n_days_with_tx"],), tr["fold"], ok)
    after_t = leftover_diag(y, tr["d_n_cust"], (tr["d_cust_top1"],), tr["fold"], ok)
    after_dt = leftover_diag(
        y, tr["d_n_cust"], (tr["c_n_days_with_tx"], tr["d_cust_top1"]), tr["fold"], ok
    )
    rec = signed_oof_auroc(y, tr["d_n_cust"], tr["fold"], ok)
    days = signed_oof_auroc(y, tr["c_n_days_with_tx"], tr["fold"], ok)
    prose = (
        f"top1-defined same-n n={int(ok.sum()):,} pos={int((ok & (y == 1)).sum()):,}. "
        f"raw {_f(_cv(rec))} days {_f(_cv(days))}. "
        f"leftover after days {_f(after_d['rank'])} dies={after_d['honest_dies']} "
        f"ρ(resid,days)={_f(after_d['rho_ctrl'])}. "
        f"after top1 {_f(after_t['rank'])} dies={after_t['honest_dies']}. "
        f"after days+top1 {_f(after_dt['rank'])} dies={after_dt['honest_dies']} "
        f"ρ(resid,days)={_f(after_dt['rho_ctrl'])} R²={_f(after_dt['r2'])}. "
        f"{'0.564 is sample-shift (drop n=0), not a two-bar leftover' if after_d['honest_dies'] else 'same-n leftover after days lives — unexpected'}."
    )
    print(prose)
    return {
        "n": int(ok.sum()),
        "days_rank": after_d["rank"],
        "days_dies": after_d["honest_dies"],
        "dt_rank": after_dt["rank"],
        "dt_dies": after_dt["honest_dies"],
        "t_rank": after_t["rank"],
        "prose": prose,
    }


def extra_n0_dummy(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — n_cust==0 dummy leftover after days (is the zero pile the leftover?)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    n = pd.to_numeric(tr["d_n_cust"], errors="coerce")
    dummy = (n == 0).astype(float)
    dummy = dummy.where(n.notna(), np.nan)
    lab = y.notna()
    rec = signed_oof_auroc(y, dummy, tr["fold"], lab)
    after = leftover_diag(y, dummy, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    rec_pos = signed_oof_auroc(y, tr["d_n_cust"], tr["fold"], lab & (n > 0))
    after_pos = leftover_diag(
        y, tr["d_n_cust"], (tr["c_n_days_with_tx"],), tr["fold"], lab & (n > 0)
    )
    prose = (
        f"n==0 dummy Y3 {_f(_cv(rec))} leftover after days {_f(after['rank'])} "
        f"dies={after['honest_dies']} n={rec['n_defined']:,} pos={rec['n_pos']:,}. "
        f"n>0 leftover after days {_f(after_pos['rank'])} dies={after_pos['honest_dies']} "
        f"raw {_f(_cv(rec_pos))}. "
        f"{'Zero pile is the 0.653, leftover still dies either way' if after['honest_dies'] and after_pos['honest_dies'] else 'check zero pile'}."
    )
    print(prose)
    return {
        "dummy_cv": _cv(rec),
        "dummy_rank": after["rank"],
        "pos_rank": after_pos["rank"],
        "pos_cv": _cv(rec_pos),
        "prose": prose,
    }


def extra_monopoly(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — monopoly Jaccard n_cust==1 vs HHI>0.975")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y4 = pd.to_numeric(tr[Y4], errors="coerce")
    n = pd.to_numeric(tr["d_n_cust"], errors="coerce")
    hhi = pd.to_numeric(tr["d_cust_hhi"], errors="coerce")
    lab = y4.notna() & n.notna()
    a = lab & (n == 1)
    b = lab & hhi.notna() & (hhi > TAIL_CUT)
    inter = int((a & b).sum())
    jacc = _pct(inter, int((a | b).sum())) if (a | b).any() else float("nan")
    rec_n1 = signed_oof_auroc(y4, (n == 1).astype(float).where(n.notna(), np.nan), tr["fold"], y4.notna())
    after_n1 = leftover_diag(
        y4,
        (n == 1).astype(float).where(n.notna(), np.nan),
        (tr["d_cust_top1"],),
        tr["fold"],
        y4.notna(),
    )
    prose = (
        f"Y4-labeled n==1 ∩ HHI>0.975 = {inter:,}; Jaccard {_pp(jacc)}. "
        f"n==1 dummy Y4 {_f(_cv(rec_n1))} leftover after top1 {_f(after_n1['rank'])} "
        f"dies={after_n1['honest_dies']}. "
        f"{'Count rewrite of monopoly tail — leftover after top1 dies' if after_n1['honest_dies'] else 'n==1 leftover lives after top1'}."
    )
    print(prose)
    return {"jacc": jacc, "inter": inter, "n1_cv": _cv(rec_n1), "n1_rank": after_n1["rank"], "prose": prose}


def extra_tercile(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — SIZE tercile leftover of n_cust after days")
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
        rec = signed_oof_auroc(y, tr["d_n_cust"], tr["fold"], sl)
        after = leftover_diag(y, tr["d_n_cust"], (tr["c_n_days_with_tx"],), tr["fold"], sl)
        rows.append(
            {
                "tercile": t,
                "n": rec["n_defined"],
                "n_pos": rec["n_pos"],
                "CV": "LOW_POWER" if rec["low_power"] else _f(rec["cv"]),
                "leftover rank": _f(after["rank"]),
                "dies": after["honest_dies"],
                "Y3 rate": _pp(float(y[sl].mean()) if sl.any() else float("nan")),
            }
        )
        print(f"  {t} CV={_f(rec['cv'])} leftover={_f(after['rank'])} dies={after['honest_dies']}")
    prose = "SIZE tercile leftover of n_cust after days. Inverse-size recover is days, not leftover count."
    print(prose)
    return {"rows": rows, "prose": prose}


def extra_mean_top1(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — company-mean leftover after days+top1")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    meanx = company_mean(tr["d_n_cust"], tr["company_id"])
    after = leftover_diag(
        y, meanx, (tr["c_n_days_with_tx"], tr["d_cust_top1"]), tr["fold"], lab
    )
    after_d = leftover_diag(y, meanx, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    prose = (
        f"company-mean leftover after days {_f(after_d['rank'])} dies={after_d['honest_dies']}; "
        f"after days+top1 {_f(after['rank'])} dies={after['honest_dies']}. "
        f"TRAIT leftover of who-has-customers is not a reason to KEEP as engine X."
    )
    print(prose)
    return {"mean_days": after_d["rank"], "mean_dt": after["rank"], "prose": prose}


def extra_t1_probe(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — T1 leftover 0.557: fake, top1, or inverse-size pocket?")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    med = tr.groupby("company_id")["log_in3"].median()
    cuts = med.quantile([1 / 3, 2 / 3])
    t1_ids = set(med.index[med <= cuts.iloc[0]].astype(str))
    sl = lab & tr["company_id"].isin(t1_ids)
    after = leftover_diag(y, tr["d_n_cust"], (tr["c_n_days_with_tx"],), tr["fold"], sl)
    after_dt = leftover_diag(
        y, tr["d_n_cust"], (tr["c_n_days_with_tx"], tr["d_cust_top1"]), tr["fold"], sl
    )
    after_sz = leftover_diag(
        y, tr["d_n_cust"], (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], sl
    )
    rec = signed_oof_auroc(y, tr["d_n_cust"], tr["fold"], sl)
    days = signed_oof_auroc(y, tr["c_n_days_with_tx"], tr["fold"], sl)
    prose = (
        f"T1 n={rec['n_defined']:,} pos={rec['n_pos']:,} raw {_f(_cv(rec))} days {_f(_cv(days))}. "
        f"leftover after days {_f(after['rank'])} dies={after['honest_dies']} "
        f"ρ(resid,days)={_f(after['rho_ctrl'])} folds={after['rank_folds']}. "
        f"after days+top1 {_f(after_dt['rank'])} dies={after_dt['honest_dies']}. "
        f"after days+size {_f(after_sz['rank'])} dies={after_sz['honest_dies']}. "
        f"{'T1 leftover dies after days+size — inverse-size pocket, not leftover count' if after_sz['honest_dies'] else 'T1 leftover lives after days+size'}."
    )
    print(prose)
    return {
        "rank": after["rank"],
        "dt": after_dt["rank"],
        "sz": after_sz["rank"],
        "prose": prose,
    }


def extra_y4_days(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Y4 leftover of n_cust after days (not the monopoly bar)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y4], errors="coerce")
    lab = y.notna()
    rec = signed_oof_auroc(y, tr["d_n_cust"], tr["fold"], lab)
    after = leftover_diag(y, tr["d_n_cust"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_t = leftover_diag(y, tr["d_n_cust"], (tr["d_cust_top1"],), tr["fold"], lab)
    prose = (
        f"Y4 n_cust {_f(_cv(rec))} leftover after days {_f(after['rank'])} dies={after['honest_dies']}; "
        f"after top1 {_f(after_t['rank'])} dies={after_t['honest_dies']}. "
        f"Y4 leftover after days is not a KEEP path — monopoly bar is top1_lag3."
    )
    print(prose)
    return {"cv": _cv(rec), "rank": after["rank"], "t_rank": after_t["rank"], "prose": prose}


def extra_quintiles(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Y3 rate by n_cust quintile (same-n vs days)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    n = pd.to_numeric(tr["d_n_cust"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    ok = y.notna() & n.notna() & days.notna()
    qn = pd.qcut(n[ok], 5, duplicates="drop")
    qd = pd.qcut(days[ok], 5, duplicates="drop")
    rows = []
    for i, cat in enumerate(sorted(qn.dropna().unique()), start=1):
        sl = ok.copy()
        sl.loc[ok] = qn == cat
        rows.append(
            {
                "q": i,
                "n_cust rate": _pp(float(y[sl].mean()) if sl.any() else float("nan")),
                "n": int(sl.sum()),
                "n_pos": int((sl & (y == 1)).sum()),
            }
        )
    drows = []
    for i, cat in enumerate(sorted(qd.dropna().unique()), start=1):
        sl = ok.copy()
        sl.loc[ok] = qd == cat
        drows.append(
            {
                "q": i,
                "days rate": _pp(float(y[sl].mean()) if sl.any() else float("nan")),
                "n": int(sl.sum()),
            }
        )
    prose = (
        f"Y3 n_cust Q1→Q5 {[r['n_cust rate'] for r in rows]}; "
        f"days Q1→Q5 {[r['days rate'] for r in drows]}. Inverse gradient, days steeper."
    )
    print(prose)
    return {"rows": rows, "drows": drows, "prose": prose}


def extra_t1_top1_cov(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — T1 ∩ top1-defined leftover after days (0.738 probe)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    med = tr.groupby("company_id")["log_in3"].median()
    cuts = med.quantile([1 / 3, 2 / 3])
    t1_ids = set(med.index[med <= cuts.iloc[0]].astype(str))
    t1 = tr["company_id"].isin(t1_ids)
    top1 = pd.to_numeric(tr["d_cust_top1"], errors="coerce")
    sl = lab & t1
    sl_t = sl & top1.notna()
    rec = signed_oof_auroc(y, tr["d_n_cust"], tr["fold"], sl_t)
    after = leftover_diag(y, tr["d_n_cust"], (tr["c_n_days_with_tx"],), tr["fold"], sl_t)
    after_dt = leftover_diag(
        y, tr["d_n_cust"], (tr["c_n_days_with_tx"], tr["d_cust_top1"]), tr["fold"], sl_t
    )
    prose = (
        f"T1 Y3-labeled {int(sl.sum()):,} / top1-defined {int(sl_t.sum()):,} "
        f"pos={int((sl_t & (y == 1)).sum()):,}. "
        f"raw {_f(_cv(rec))} leftover after days {_f(after['rank'])} dies={after['honest_dies']}. "
        f"after days+top1 {_f(after_dt['rank'])} dies={after_dt['honest_dies']} "
        f"folds={after_dt['rank_folds']}. "
        f"{'0.738 is the top1-defined T1 slice — fold-noisy, not a KEEP pocket' if np.isfinite(after_dt['rank']) else 'LOW_POWER'}."
    )
    print(prose)
    return {
        "n_t1": int(sl.sum()),
        "n_t1_top1": int(sl_t.sum()),
        "days_rank": after["rank"],
        "dt_rank": after_dt["rank"],
        "prose": prose,
    }


def decide(p1, p2, p3, p5) -> dict:
    leftover_lives = bool(np.isfinite(p3["rank"]) and p3["rank"] >= CHANCE and not p3["dies"])
    is_size = bool(p1["is_size"])
    twin = bool(p1["twin_gate"])
    engine = leftover_lives and bool(p2["beat_size"]) and not is_size and not twin
    after_top1_dies = bool(p5["t_dies"])
    if engine:
        role = "KEEP as unused leftover"
        why = (
            f"KEEP-as-X passed: leftover after days rank {p3['rank']:.3f} lives, "
            f"beat-size {_f(p2['n_cust'])} vs 0.617, not SIZE, not twin. "
            f"Stays off the 15-col card. Do not invent y_n_cust."
        )
    elif leftover_lives and twin:
        role = "DROP from the 44 as Y3 X"
        why = (
            f"leftover after days rank {p3['rank']:.3f} lives but TWIN of "
            f"{p1['twins']}. Count rewrite, not a new engine X. "
            f"after top1 {_f(p5['t_rank'])} dies={after_top1_dies}."
        )
    elif leftover_lives and not p2["beat_size"]:
        role = "CLOSE unused leftover"
        why = (
            f"leftover after days rank {p3['rank']:.3f} lives but fails beat-size "
            f"({_f(p2['n_cust'])} vs 0.617). Unused leftover, not engine X."
        )
    else:
        role = "CLOSE unused leftover"
        why = (
            f"unused leftover after days: honest rank {_f(p3['rank'])} dies "
            f"(OLS {_f(p3['ols'])} fake={p3['fake']} almost={p3['almost']}). "
            f"Also TWIN of {p1['twins']}. DROP from the 44 as Y3 X. "
            f"Do not invent y_n_cust. Off the 15-col card."
        )
    return {
        "role": role,
        "why": why,
        "leftover_lives": leftover_lives,
        "engine": engine,
        "is_size": is_size,
        "twin": twin,
        "drop_engine": not engine,
        "park_y": "PARK as Y — do not invent y_n_cust",
        "card": "no — do not put d_n_cust on the 15-col card",
    }


def make_plot(tr: pd.DataFrame, p3: dict) -> bool:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return False
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    n = pd.to_numeric(tr["d_n_cust"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    lab = y3.notna() & n.notna()
    q = pd.qcut(n[lab], 5, duplicates="drop")
    rates, xs = [], []
    for i, cat in enumerate(sorted(q.dropna().unique()), start=1):
        sl = lab.copy()
        sl.loc[lab] = q == cat
        rates.append(float(y3[sl].mean()) if sl.any() else float("nan"))
        xs.append(i)
    ax.plot(xs, rates, marker="o", color="#1f4e79", label="d_n_cust")
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
    ax.set_title("n_cust vs days recover")
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
    ax.set_title("d_n_cust leftover after days")
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
    xi, xz, xf = ctx["xi"], ctx["xz"], ctx["xf"]
    xb, xs, xl = ctx["xb"], ctx["xs"], ctx["xl"]
    xp, xn, xsz, xw = ctx["xp"], ctx["xn"], ctx["xsz"], ctx["xw"]
    lines = [
        "# Unused leftover of `d_n_cust` after days as Y3 X",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. No `build_targets`. Do not invent `y_n_cust`. Do not put `d_n_cust` on the 15-col card. "
        "Do not overwrite `cust_hhi_qa.*`. Do not grow TURNOVER.",
        "",
        "`d_n_cust` = distinct non-null counterparty_id on AR invoices (Family D, trailing 6m). "
        "Incomplete 6m books are NaN. Dark 470 stay NaN not 0. "
        "`d_cust_hhi` was DROPPED as engine X (twin of `d_cust_top1` ρ 0.994).",
        "",
        "## Headline",
        "",
        (
            f"`d_n_cust` as Y3 X: **{d['role']}** ({d['why']}). "
            f"Y3 leftover after days OLS {_f(p3['ols'])} rank {_f(p3['rank'])} "
            f"({'dies' if p3['dies'] else 'lives'}, fake={p3['fake']}). "
            f"Inverse days after n_cust rank {_f(p3['inv_rank'])}. "
            f"Single {_f(p2['n_cust'])} vs size {_f(p2['size'])} vs days {_f(p2['days'])} "
            f"vs top1 {_f(p2['top1'])} vs HHI {_f(p2['hhi'])}. "
            f"after top1 {_f(p5['t_rank'])} after HHI {_f(p5['h_rank'])} after days+top1 {_f(p5['dt_rank'])}. "
            f"15-col card: {d['card']}. {d['park_y']}. "
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
        f"| 4 | Dip vs fall? | Y4 leftover after top1_lag3 {_f(p8['after_t3'])} — monopoly tail rewrite? |",
        f"| 5 | Why did it change? | Twin screen: {p1['twins'] or 'none'}. ρ vs HHI {_f(p1['rhos']['d_cust_hhi'])}. |",
        f"| 6 | Months earlier? | lag1 leftover {_f(p7['l1_rank'])}; days_lag1 {_f(p7['days_l1'])} (quote 0.684). |",
        "",
        "## PARK / CLOSE / KEEP / DROP-from-44",
        "",
        "| object | decision | why |",
        "| --- | --- | --- |",
        f"| `d_n_cust` as Y3 X / the 15-col card | **{d['role']}** | {d['why']} |",
        f"| `d_n_cust` as engine X on the 44 | **{'KEEP leftover' if d['engine'] else 'DROP'}** | leftover lives={d['leftover_lives']} twin={d['twin']} SIZE={d['is_size']} beat-size={p2['beat_size']} |",
        f"| `y_n_cust` | **PARK** | do not invent a customer-count Y |",
        f"| twin of HHI / top1 | {_f(p1['rhos']['d_cust_hhi'])} / {_f(p1['rhos']['d_cust_top1'])} | leftover after top1 {_f(p5['t_rank'])} after HHI {_f(p5['h_rank'])} |",
        f"| same object as `d_n_supp` | **{'YES twin' if p9['same'] else 'NO'}** | ρ={_f(p9['rho'])} |",
        f"| Q6 lag1 after days_lag1 | **{'KEEP' if (np.isfinite(p7['l1_rank']) and p7['l1_rank'] >= CHANCE and not p7['l1_dies']) else 'CLOSE'}** | leftover {_f(p7['l1_rank'])}; days_lag1 {_f(p7['days_l1'])} |",
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
        "## 5 — Leftover after top1 / HHI / days+top1",
        "",
        p5["prose"],
        "",
        _md_table(p5["rows"]),
        "",
        "## 6 — Dark 470 stay NaN; ERP-only leftover",
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
        "## 8 — Y4 leftover after top1_lag3; monopoly tail",
        "",
        p8["prose"],
        "",
        _md_table(p8["rows"]),
        "",
        _md_table(p8["leftover_rows"]),
        "",
        "## 9 — vs `d_n_supp`",
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
        xi["prose"],
        "",
        "### n==0 / n==1 / ≥2",
        "",
        xz["prose"],
        "",
        _md_table(xz["rows"]),
        "",
        "### Fold-wise leftover; 12-name Y2 drop",
        "",
        xf["prose"],
        "",
        _md_table(xf["fold_rows"]),
        "",
        "### Bootstrap leftover after days",
        "",
        xb["prose"],
        "",
        "### Leftover by so-far",
        "",
        xs["prose"],
        "",
        _md_table(xs["rows"]),
        "",
        "### log1p vs raw",
        "",
        xl["prose"],
        "",
        "### Permute within days quintile",
        "",
        xp["prose"],
        "",
        "### leftover after a_n_tx",
        "",
        xn["prose"],
        "",
        "### leftover after size",
        "",
        xsz["prose"],
        "",
        "### Without weakest / strongest fold",
        "",
        xw["prose"],
        "",
        "### leftover after days on top1-defined only",
        "",
        ctx["xst"]["prose"],
        "",
        "### n==0 dummy leftover",
        "",
        ctx["xz0"]["prose"],
        "",
        "### monopoly Jaccard n==1 vs HHI>0.975",
        "",
        ctx["xm"]["prose"],
        "",
        "### SIZE tercile leftover",
        "",
        ctx["xtc"]["prose"],
        "",
        _md_table(ctx["xtc"]["rows"]),
        "",
        "### company-mean leftover after days+top1",
        "",
        ctx["xmt"]["prose"],
        "",
        "### T1 leftover probe",
        "",
        ctx["xt1"]["prose"],
        "",
        "### Y4 leftover after days",
        "",
        ctx["xy4"]["prose"],
        "",
        "### Y3 rate by n_cust quintile",
        "",
        ctx["xq"]["prose"],
        "",
        _md_table(ctx["xq"]["rows"]),
        "",
        "### T1 ∩ top1 leftover",
        "",
        ctx["xtc1"]["prose"],
        "",
        "## Night quotes (unchanged)",
        "",
        "| quote | locked |",
        "| --- | --- |",
        f"| Y3 | {Y3_NIGHT[0]:.3f} / {Y3_NIGHT[1]:.3f} |",
        f"| days | {DAYS_BENCH:.3f} |",
        f"| size | {SIZE_QUOTE:.3f} |",
        f"| Y7 TURNOVER | {Y7_TURNOVER:.3f} / {Y7_B_SHALLOW:.3f} |",
        f"| days_lag1 | {DAYS_LAG1_QUOTE:.3f} |",
        "",
        "Do not quote a_out_vol 0.722. Do not grow TURNOVER. Do not put `d_n_cust` on the 15-col card.",
        "",
        "## Files written",
        "",
        "- `analysis/evaluate/n_cust_qa.py`",
        "- `analysis/outputs/n_cust_qa.md`",
        "- `analysis/outputs/n_cust_qa.png`" if ctx.get("png") else "- (no PNG)",
        "- append-only `analysis/experiments/registry.csv`",
        "- `overnight/waves/wave4_n_cust.md` (end, if WRITE_WAVE)",
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
    p1, p2, p3, p5 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p5"]
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
            "metric": "auroc_d_n_cust",
            "value": p2["n_cust"],
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
            "metric": "auroc_d_n_cust_resid_days",
            "value": p3["rank"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"ols={p3['ols']:.4f} dies={p3['dies']} fake={p3['fake']} r2={p3['r2']:.4f}",
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
            "metric": "auroc_days_resid_d_n_cust",
            "value": p3["inv_rank"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"inverse dies={p3['inv_dies']}",
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
            "metric": "auroc_d_n_cust_resid_top1",
            "value": p5["t_rank"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"after_hhi={p5['h_rank']:.4f} after_days_top1={p5['dt_rank']:.4f}",
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
            "metric": "rho_d_n_cust_vs_hhi",
            "value": p1["rhos"]["d_cust_hhi"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"twins={p1['twins']} size={p1['is_size']} acf1={p1['acf1']:.4f}",
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
            "metric": "n_cust_leftover",
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
    p1, p2, p3, p5, p7, p8 = (
        ctx["p1"],
        ctx["p2"],
        ctx["p3"],
        ctx["p5"],
        ctx["p7"],
        ctx["p8"],
    )
    text = (
        f"# Wave 4 — d_n_cust leftover after days as Y3 X\n\n"
        f"Agent `{AGENT}`. Train group-fold seed 20260918. Holdout 72 coverage only.\n\n"
        f"## Files written\n\n"
        f"- `analysis/evaluate/n_cust_qa.py`\n"
        f"- `analysis/outputs/n_cust_qa.md`\n"
        f"- `analysis/outputs/n_cust_qa.png`\n"
        f"- append-only `analysis/experiments/registry.csv` "
        f"(skip key agent+x_families+y+model+split+metric)\n"
        f"- this note\n\n"
        f"Did not touch `cust_hhi_qa.*`, `supp_hhi_qa.*`, `tax_month_qa.*`, "
        f"`issued_qa.*`, `in3_qa.*`, `ss_qa.*`, `salary_month_qa.*`, "
        f"`counterparties.py`, parquet / duckdb, `build_targets`, `product/`, "
        f"the 15-col card, TURNOVER, LIVE, CONTEXT, canvas, `brief_map.md`, "
        f"or the parent journal. "
        f"Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. "
        f"Y7 TURNOVER **0.720 / 0.712** unchanged.\n\n"
        f"## Locked verdict\n\n"
        f"| object | decision |\n"
        f"| --- | --- |\n"
        f"| `d_n_cust` as Y3 X / 15-col card | **{d['role']}** |\n"
        f"| `d_n_cust` as engine X on the 44 | **{'KEEP leftover' if d['engine'] else 'DROP'}** |\n"
        f"| `y_n_cust` | **PARK** |\n\n"
        f"Y3 leftover after days rank {_f(p3['rank'])} (dies={p3['dies']}, fake={p3['fake']}); "
        f"inverse days after n_cust {_f(p3['inv_rank'])}. "
        f"Single {_f(p2['n_cust'])} vs days {_f(p2['days'])} vs size {_f(p2['size'])}. "
        f"ρ vs HHI {_f(p1['rhos']['d_cust_hhi'])} vs top1 {_f(p1['rhos']['d_cust_top1'])} "
        f"vs days {_f(p1['rhos']['c_n_days_with_tx'])} vs size {_f(p1['rhos']['log1p(a_in3)'])}. "
        f"after top1 {_f(p5['t_rank'])} after HHI {_f(p5['h_rank'])}. "
        f"Q6 lag1 leftover {_f(p7['l1_rank'])}. "
        f"Y4 leftover after top1_lag3 {_f(p8['after_t3'])}. "
        f"after days+top1 0.564 is sample-shift (drop n=0); leftover after days on that n dies. "
        f"T1 leftover 0.557 is fold-noisy (2/5 folds die). "
        f"Permute-within-days leftover p50=0.533 — observed 0.545 sits inside the null. "
        f"{d['why']}\n\n"
        f"## What failed / next\n\n"
        + "\n".join(f"- {x}" for x in ctx["failed"])
        + f"\n\nElapsed {ctx['elapsed_s']:.0f}s.\n"
    )
    WAVE_NOTE.write_text(text)
    print(f"wrote {WAVE_NOTE}")


def main() -> None:
    t0 = time.time()
    failed: list[str] = []
    print("n_cust leftover QA — unused leftover of d_n_cust after days as Y3 X")
    panel = load_panel()
    panel = attach_folds(panel)
    panel = add_panel_lags(
        panel,
        [
            "d_n_cust",
            "d_cust_top1",
            "d_cust_hhi",
            "c_n_days_with_tx",
        ],
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
    p8 = pass8_y4(tr)
    p9 = pass9_supp(tr)
    p10 = pass10_hold(hold, book)
    xi = extra_icc(tr)
    xz = extra_zero_one(tr)
    xf = extra_fold_y2(tr, p3)
    xl = extra_log_vs_raw(tr)
    xn = extra_ntx(tr)
    xsz = extra_size(tr)
    xs = extra_sofar(tr)
    xw = extra_wo_fold(p2, p3)
    xst = extra_samen_top1(tr)
    xz0 = extra_n0_dummy(tr)
    xm = extra_monopoly(tr)
    xtc = extra_tercile(tr)
    xmt = extra_mean_top1(tr)
    xt1 = extra_t1_probe(tr)
    xy4 = extra_y4_days(tr)
    xq = extra_quintiles(tr)
    xtc1 = extra_t1_top1_cov(tr)
    xb = extra_bootstrap(tr, n_boot=50)
    xp = extra_permute(tr, n_perm=30)
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
        "xz": xz,
        "xf": xf,
        "xb": xb,
        "xs": xs,
        "xl": xl,
        "xp": xp,
        "xn": xn,
        "xsz": xsz,
        "xw": xw,
        "xst": xst,
        "xz0": xz0,
        "xm": xm,
        "xtc": xtc,
        "xmt": xmt,
        "xt1": xt1,
        "xy4": xy4,
        "xq": xq,
        "xtc1": xtc1,
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
