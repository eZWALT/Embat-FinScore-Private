"""Unused leftover of `d_tx_cp_share` on the 44.

NORTH_STAR: missing-CP is the fill twin (ρ −0.947) and CLOSE as Y3 X.
Night Y5 AR `d_tx_cp_share` **0.611** stays a number, PARK as X (fold-3
one-group). This lane asks leftover after days (Y3 X), leftover after
the missing-CP twin, leftover after size (Y5), and whether the 44
should lose `d_tx_cp_share`.

KEEP-as-X: beat size ≥0.02 AND leftover after the honest bar AND not
SIZE AND not a twin (|ρ|≥0.80 vs miss_cp / uncat / d_cust_hhi).
Honest bars: Y3 leftover after days; Y5 leftover after size.
Leftover <0.55 dies. Do not invent `y_d_tx`. Do not merge Family D.
Do not put d_tx on the 15-col Y3 card. Y7 never D. Y5 never E.
Y3 never B. Trees stay PARK. Night Y3 0.762/0.752 unchanged.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.d_tx_qa

Owned: analysis/evaluate/d_tx_qa.py, analysis/outputs/d_tx_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_d_tx.md (end).
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
from analysis.features.common import ANALYSIS, CAT_MAP, DATA, connect
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
OUT_MD = ANALYSIS / "outputs" / "d_tx_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "d_tx_leftover.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_d_tx.md"
AGENT = "7f2e91c4"
WAVE = "4"
ROUND = "R4"
MODEL = "d_tx_qa"
X_FAM = "D"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
Y5_AR = "y5_ar_od30_sust"
Y5_AP = "y5_ap_od30_ownp80"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_Y3_QUOTE = 0.617
Y5_DTX_QUOTE = 0.611
Y5_CV_QUOTE = 0.576
Y3_NIGHT = (0.762, 0.752)
KEEP_DELTA = 0.02
SIZE_PARK = 0.60
TWIN_RHO = 0.80
SIZE_RHO = 0.50
CHANCE = 0.55
ICC_STYLE = 0.85
ACF_STYLE = 0.40
MIN_POS = 50
MIN_ACF_PAIRS = 4
HOLE_SHARE = 0.01
MISS_TWIN_QUOTE = -0.947
INV_FILL_QUOTE = 0.999
TX_NAMED_QUOTE = 0.252
WRITE_WAVE = True

BOOK = (
    "document_type = 'invoice' AND status <> 'cancel' "
    "AND amount <> 0 AND issuance_date IS NOT NULL"
)

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "first_month",
    "a_uncat_share",
    "a_n_tx",
    "a_in3",
    "c_n_days_with_tx",
    "d_tx_cp_share",
    "d_n_cust",
    "d_n_supp",
    "d_cust_hhi",
    "d_supp_hhi",
    "d_interco_share",
)

Y_KEEP = (Y2, Y3, Y5_AR, Y5_AP)


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


def rank_resid(y: pd.Series, *xs: pd.Series) -> pd.Series:
    yr = pd.to_numeric(y, errors="coerce").rank(method="average")
    xr = [pd.to_numeric(x, errors="coerce").rank(method="average") for x in xs]
    resid, _ = ols_resid(yr, *xr)
    return resid


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


def _cat_sql_list() -> str:
    return ", ".join("'" + k.replace("'", "''") + "'" for k in CAT_MAP)


def _trail_class(n) -> str:
    if n is None or (isinstance(n, float) and not np.isfinite(n)):
        return "na"
    n = int(n)
    if n < 12:
        return "short_<12"
    if n >= 18:
        return "long_>=18"
    return "mid_12_17"


def load_monthly_miss(con) -> pd.DataFrame:
    """Calendar-month missing-CP share. In-memory only — never written."""
    cats = _cat_sql_list()
    df = con.execute(
        f"""
        SELECT
          CAST(t.company_id AS VARCHAR) AS company_id,
          CAST(date_trunc('month', t."date") AS DATE) AS month,
          COUNT(*) AS n_tx,
          SUM(CASE WHEN t.counterparty_id IS NULL
                     OR length(trim(CAST(t.counterparty_id AS VARCHAR))) = 0
                    THEN 1 ELSE 0 END) AS n_miss
        FROM transactions t
        WHERE t."date" IS NOT NULL
          AND t."date" >= DATE '2024-09-01'
          AND t."date" < DATE '2026-09-01'
        GROUP BY 1, 2
        """
    ).df()
    df["company_id"] = df["company_id"].astype(str)
    df["period"] = pd.to_datetime(df["month"])
    n_tx = pd.to_numeric(df["n_tx"], errors="coerce")
    df["miss_cp_share"] = np.where(n_tx > 0, df["n_miss"] / n_tx, np.nan)
    df["named_share"] = np.where(n_tx > 0, 1.0 - df["miss_cp_share"], np.nan)
    return df[["company_id", "period", "n_tx", "n_miss", "miss_cp_share", "named_share"]]


def load_monthly_inv_cp(con) -> pd.DataFrame:
    df = con.execute(
        f"""
        SELECT
          CAST(company_id AS VARCHAR) AS company_id,
          CAST(date_trunc('month', issuance_date) AS DATE) AS month,
          COUNT(*) AS n_inv,
          SUM(CASE WHEN counterparty_id IS NULL
                     OR length(trim(CAST(counterparty_id AS VARCHAR))) = 0
                    THEN 1 ELSE 0 END) AS n_inv_miss
        FROM invoices
        WHERE {BOOK}
        GROUP BY 1, 2
        """
    ).df()
    df["company_id"] = df["company_id"].astype(str)
    df["period"] = pd.to_datetime(df["month"])
    n = pd.to_numeric(df["n_inv"], errors="coerce")
    df["inv_miss_share"] = np.where(n > 0, df["n_inv_miss"] / n, np.nan)
    df["inv_cp_share"] = np.where(n > 0, 1.0 - df["inv_miss_share"], np.nan)
    return df[["company_id", "period", "n_inv", "inv_cp_share"]]


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
    extra = [c for c in ("j_pay_match",) if c in raw.columns]
    panel = _keys(raw[list(STORE_COLS) + extra])
    y = _keys(yraw[["company_id", "period", *Y_KEEP]])
    panel = panel.merge(y, on=["company_id", "period"], how="left")
    hold = load_holdout()
    panel["split"] = np.where(panel["company_id"].isin(hold), "holdout", "train")
    panel["log_in3"] = np.log1p(pd.to_numeric(panel["a_in3"], errors="coerce").clip(lower=0))
    if "first_month" in panel.columns:
        panel["first_month"] = pd.to_datetime(panel["first_month"])
        panel["months_so_far"] = (
            (panel["period"].dt.year - panel["first_month"].dt.year) * 12
            + (panel["period"].dt.month - panel["first_month"].dt.month)
            + 1
        )
    else:
        panel["months_so_far"] = np.nan
    panel["so_far_class"] = panel["months_so_far"].map(_trail_class)
    leak3 = leakage_check(["d_tx_cp_share", "c_n_days_with_tx", "log_in3"], Y3, forbidden_prefixes=["b"])
    leak5 = leakage_check(["d_tx_cp_share", "log_in3"], Y5_AR, forbidden_prefixes=["e"])
    leak7 = leakage_check(["c_n_days_with_tx", "log_in3"], "y7_top1_lost", forbidden_prefixes=["d"])
    if not leak3["ok"]:
        raise RuntimeError(f"Y3 never-B leak: {leak3['issues']}")
    if not leak5["ok"]:
        raise RuntimeError(f"Y5 never-E leak: {leak5['issues']}")
    if not leak7["ok"]:
        raise RuntimeError(f"Y7 never-D leak: {leak7['issues']}")
    return panel


def attach_folds(panel: pd.DataFrame) -> pd.DataFrame:
    train = panel[panel["split"] == "train"][["company_id", "group_id"]].drop_duplicates()
    assert_no_holdout(train["company_id"])
    folds = group_folds(train, n=N_FOLDS, seed=FOLD_SEED)
    return panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")


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


def attach_miss(panel: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    return panel.merge(monthly, on=["company_id", "period"], how="left")


def attach_inv(panel: pd.DataFrame, inv: pd.DataFrame) -> pd.DataFrame:
    return panel.merge(inv, on=["company_id", "period"], how="left")


# ---------------------------------------------------------------------------
# 1. Coverage; 470 vs 744; invoice fill vs tx named
# ---------------------------------------------------------------------------
def pass1_cov(tr: pd.DataFrame, book: set[str]) -> dict:
    dtx = pd.to_numeric(tr["d_tx_cp_share"], errors="coerce")
    miss = pd.to_numeric(tr["miss_cp_share"], errors="coerce")
    named = pd.to_numeric(tr["named_share"], errors="coerce")
    inv = pd.to_numeric(tr["inv_cp_share"], errors="coerce")
    n_cm = int(len(tr))
    n_nn = int(dtx.notna().sum())
    n_zero = int((dtx == 0).sum())
    n_nan = int(dtx.isna().sum())
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    n_erp = int(last["ever_erp"].sum())
    n_dark = int((~last["ever_erp"]).sum())
    confirm_744 = n_erp == 744 and n_dark == 470
    dark = ~tr["ever_erp"]
    erp = tr["ever_erp"]
    dark_nn = int(dtx[dark].notna().sum())
    dark_zero = int((dtx[dark] == 0).sum())
    dark_nan = int(dtx[dark].isna().sum())
    dark_mean = float(dtx[dark].mean())
    erp_mean = float(dtx[erp].mean())
    both = inv.notna() & named.notna()
    inv_mean = float(inv[both].mean()) if both.any() else float("nan")
    tx_mean = float(named[both].mean()) if both.any() else float("nan")
    inv_ok = bool(np.isfinite(inv_mean) and abs(inv_mean - INV_FILL_QUOTE) < 0.005)
    tx_ok = bool(np.isfinite(tx_mean) and abs(tx_mean - TX_NAMED_QUOTE) < 0.01)
    rows = [
        {
            "slice": "train all",
            "n_cm": f"{n_cm:,}",
            "n_co": int(tr["company_id"].nunique()),
            "dtx_nn": f"{n_nn:,}",
            "cov": _pp(_pct(n_nn, n_cm)),
            "eq0": _pp(_pct(n_zero, n_cm)),
            "nan": _pp(_pct(n_nan, n_cm)),
            "mean": _f(float(dtx.mean())),
            "p50": _f(float(dtx.median())),
        },
        {
            "slice": "ever_erp_744",
            "n_cm": f"{int(erp.sum()):,}",
            "n_co": n_erp,
            "dtx_nn": f"{int(dtx[erp].notna().sum()):,}",
            "cov": _pp(_pct(int(dtx[erp].notna().sum()), int(erp.sum()))),
            "eq0": _pp(_pct(int((dtx[erp] == 0).sum()), int(erp.sum()))),
            "nan": _pp(_pct(int(dtx[erp].isna().sum()), int(erp.sum()))),
            "mean": _f(erp_mean),
            "p50": _f(float(dtx[erp].median())),
        },
        {
            "slice": "never_erp_470",
            "n_cm": f"{int(dark.sum()):,}",
            "n_co": n_dark,
            "dtx_nn": f"{dark_nn:,}",
            "cov": _pp(_pct(dark_nn, int(dark.sum()))),
            "eq0": _pp(_pct(dark_zero, int(dark.sum()))),
            "nan": _pp(_pct(dark_nan, int(dark.sum()))),
            "mean": _f(dark_mean),
            "p50": _f(float(dtx[dark].median())),
        },
    ]
    inv_rows = [
        {
            "slice": "same-month both",
            "n_cm": int(both.sum()),
            "inv_named_mean": _f(inv_mean),
            "tx_named_mean": _f(tx_mean),
            "inv_named_p50": _f(float(inv[both].median()) if both.any() else float("nan")),
            "tx_named_p50": _f(float(named[both].median()) if both.any() else float("nan")),
        }
    ]
    dark_is_zero = bool(dark_zero / max(int(dark.sum()), 1) >= 0.80 and dark_nan / max(int(dark.sum()), 1) < 0.05)
    prose = (
        f"Train d_tx_cp_share cov {_pp(_pct(n_nn, n_cm))} mean {_f(float(dtx.mean()))} "
        f"p50 {_f(float(dtx.median()))} eq0 {_pp(_pct(n_zero, n_cm))}. "
        f"Last-month companies ever-ERP {n_erp} / never-ERP {n_dark} "
        f"({'CONFIRM 744/470' if confirm_744 else 'off 744/470'}). "
        f"Dark d_tx mean {_f(dark_mean)} eq0 {_pp(_pct(dark_zero, int(dark.sum())))} "
        f"NaN {_pp(_pct(dark_nan, int(dark.sum())))} "
        f"({'defined as 0, not NaN' if dark_is_zero else 'not a clean 0-fill'}). "
        f"Same-month invoice CP fill {_f(inv_mean)} vs tx named {_f(tx_mean)} "
        f"({'CONFIRM 0.999 vs 0.252' if inv_ok and tx_ok else 'drift vs 0.999/0.252'})."
    )
    print(prose)
    return {
        "rows": rows,
        "inv_rows": inv_rows,
        "cov": _pct(n_nn, n_cm),
        "n_nn": n_nn,
        "n_cm": n_cm,
        "n_zero": n_zero,
        "n_nan": n_nan,
        "mean": float(dtx.mean()),
        "p50": float(dtx.median()),
        "n_erp": n_erp,
        "n_dark": n_dark,
        "confirm_744": confirm_744,
        "dark_mean": dark_mean,
        "erp_mean": erp_mean,
        "dark_zero": dark_zero,
        "dark_nan": dark_nan,
        "dark_nn": dark_nn,
        "dark_is_zero": dark_is_zero,
        "inv_mean": inv_mean,
        "tx_mean": tx_mean,
        "inv_ok": inv_ok,
        "tx_ok": tx_ok,
        "n_both": int(both.sum()),
        "miss_mean": float(miss.mean()),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 2–3. Spearman twins; reproduce ρ vs miss_cp −0.947
# ---------------------------------------------------------------------------
def pass2_twins(tr: pd.DataFrame) -> dict:
    dtx = tr["d_tx_cp_share"]
    pairs = [
        ("miss_cp_share", tr["miss_cp_share"]),
        ("1-miss_cp_share", 1.0 - pd.to_numeric(tr["miss_cp_share"], errors="coerce")),
        ("named_share", tr["named_share"]),
        ("a_uncat_share", tr["a_uncat_share"]),
        ("d_cust_hhi", tr["d_cust_hhi"]),
        ("d_supp_hhi", tr["d_supp_hhi"]),
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ("log1p(a_in3)", tr["log_in3"]),
        ("d_n_cust", tr["d_n_cust"]),
        ("ever_erp", tr["ever_erp"].astype(float)),
    ]
    rows = []
    rhos = {}
    twins = []
    size_flag = False
    for name, s in pairs:
        rho = spearman(dtx, s)
        rhos[name] = rho
        twin = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
        sizeish = bool(name == "log1p(a_in3)" and np.isfinite(rho) and abs(rho) >= SIZE_RHO)
        if twin:
            twins.append(name)
        if sizeish:
            size_flag = True
        rows.append(
            {
                "a": "d_tx_cp_share",
                "b": name,
                "ρ": _f(rho),
                "twin_|ρ|≥0.80": "YES" if twin else "no",
                "SIZE_|ρ|≥0.50": "YES" if sizeish else ("—" if name != "log1p(a_in3)" else "no"),
            }
        )
    miss_rho = rhos["miss_cp_share"]
    miss_ok = bool(np.isfinite(miss_rho) and abs(miss_rho - MISS_TWIN_QUOTE) < 0.01)
    weaker = ""
    if "miss_cp_share" in twins:
        weaker = "TWIN of miss_cp — miss is in-memory only; d_tx is the store column (DROP weaker leftover, keep the quote)"
    prose = (
        f"d_tx vs miss_cp ρ={_f(miss_rho)} "
        f"({'CONFIRM −0.947' if miss_ok else 'DRIFT vs −0.947'}). "
        f"vs uncat {_f(rhos['a_uncat_share'])} vs d_cust_hhi {_f(rhos['d_cust_hhi'])} "
        f"vs d_supp_hhi {_f(rhos['d_supp_hhi'])} vs days {_f(rhos['c_n_days_with_tx'])} "
        f"vs size {_f(rhos['log1p(a_in3)'])}. "
        f"Twins: {twins or 'none'}. SIZE={size_flag}."
    )
    print(prose)
    return {
        "rows": rows,
        "rhos": rhos,
        "twins": twins,
        "size_flag": size_flag,
        "miss_rho": miss_rho,
        "miss_ok": miss_ok,
        "gate_twins": bool(twins),
        "weaker": weaker,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 4. Single-feature group-fold AUROC Y3 / Y5 AR
# ---------------------------------------------------------------------------
def pass4_singles(tr: pd.DataFrame) -> dict:
    feats = [
        ("d_tx_cp_share", tr["d_tx_cp_share"]),
        ("miss_cp_share", tr["miss_cp_share"]),
        ("named_share", tr["named_share"]),
        ("a_uncat_share", tr["a_uncat_share"]),
        ("log1p_a_in3", tr["log_in3"]),
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ("d_cust_hhi", tr["d_cust_hhi"]),
    ]
    rows = []
    store = {}
    for ycol in (Y3, Y5_AR, Y2):
        lab = tr[ycol].notna()
        for name, x in feats:
            res = signed_oof_auroc(tr[ycol], x, tr["fold"], lab)
            store[(ycol, name)] = res
            rows.append(_auc_row(ycol, name, res))
            print(
                f"{ycol} {name}: CV={_f(res['cv'])} train={_f(res['train_auc'])} "
                f"sign={res['train_sign']} folds={fold_bits(res)}"
            )
    y3 = _cv(store[(Y3, "d_tx_cp_share")])
    y5 = _cv(store[(Y5_AR, "d_tx_cp_share")])
    y5_tr = store[(Y5_AR, "d_tx_cp_share")]["train_auc"]
    days = _cv(store[(Y3, "c_n_days_with_tx")])
    size3 = _cv(store[(Y3, "log1p_a_in3")])
    size5 = _cv(store[(Y5_AR, "log1p_a_in3")])
    y5_ok = bool(np.isfinite(y5_tr) and abs(y5_tr - Y5_DTX_QUOTE) < 0.005)
    y5_cv_ok = bool(np.isfinite(y5) and abs(y5 - Y5_CV_QUOTE) < 0.005)
    days_ok = bool(np.isfinite(days) and abs(days - DAYS_BENCH) < 0.005)
    size_ok = bool(np.isfinite(size3) and abs(size3 - SIZE_Y3_QUOTE) < 0.005)
    beat3 = bool(np.isfinite(y3) and np.isfinite(size3) and (y3 - size3) >= KEEP_DELTA)
    beat5 = bool(np.isfinite(y5) and np.isfinite(size5) and (y5 - size5) >= KEEP_DELTA)
    prose = (
        f"Y3 d_tx CV {_f(y3)} vs size {_f(size3)} "
        f"({'CONFIRM 0.617' if size_ok else 'size drifted'}) "
        f"vs days {_f(days)} ({'CONFIRM 0.711' if days_ok else 'days drifted'}). "
        f"Y5 AR d_tx train {_f(y5_tr)} "
        f"({'CONFIRM night 0.611' if y5_ok else 'DRIFT vs 0.611'}) "
        f"CV {_f(y5)} ({'CONFIRM 0.576' if y5_cv_ok else 'CV drifted'}). "
        f"Y5 size {_f(size5)}. beat_size Y3={beat3} Y5={beat5}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "y3": y3,
        "y5": y5,
        "y5_train": y5_tr,
        "y2": _cv(store[(Y2, "d_tx_cp_share")]),
        "days": days,
        "size3": size3,
        "size5": size5,
        "y5_ok": y5_ok,
        "y5_cv_ok": y5_cv_ok,
        "days_ok": days_ok,
        "size_ok": size_ok,
        "beat3": beat3,
        "beat5": beat5,
        "y3_sign": store[(Y3, "d_tx_cp_share")]["train_sign"],
        "y5_sign": store[(Y5_AR, "d_tx_cp_share")]["train_sign"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 5–6. Honest leftover after days (Y3), size (Y5), miss_cp (both)
# ---------------------------------------------------------------------------
def _leftover_one(tr: pd.DataFrame, ycol: str, x: pd.Series, *ctrls: pd.Series) -> tuple[dict, dict]:
    resid, info = ols_resid(x, *ctrls)
    res = signed_oof_auroc(tr[ycol], resid, tr["fold"], tr[ycol].notna())
    return res, info


def pass5_leftover(tr: pd.DataFrame) -> dict:
    dtx = tr["d_tx_cp_share"]
    days = tr["c_n_days_with_tx"]
    size = tr["log_in3"]
    miss = tr["miss_cp_share"]
    uncat = tr["a_uncat_share"]
    hhi = tr["d_cust_hhi"]
    blocks = [
        (Y3, "after days", (days,)),
        (Y3, "after size", (size,)),
        (Y3, "after days+size", (days, size)),
        (Y3, "after miss_cp", (miss,)),
        (Y3, "after miss+days", (miss, days)),
        (Y3, "after uncat", (uncat,)),
        (Y3, "after d_cust_hhi", (hhi,)),
        (Y5_AR, "after size", (size,)),
        (Y5_AR, "after days", (days,)),
        (Y5_AR, "after miss_cp", (miss,)),
        (Y5_AR, "after miss+size", (miss, size)),
        (Y5_AR, "after d_cust_hhi", (hhi,)),
        (Y2, "after size", (size,)),
        (Y2, "after miss_cp", (miss,)),
    ]
    rows = []
    store = {}
    infos = {}
    for ycol, name, xs in blocks:
        res, info = _leftover_one(tr, ycol, dtx, *xs)
        key = (ycol, name)
        store[key] = res
        infos[key] = info
        rows.append(
            {
                "y": ycol,
                "residual": name,
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                "train": "LOW_POWER" if res["low_power"] else _f(res["train_auc"]),
                "R²": _f(info["r2"]),
                "folds": fold_bits(res) if not res["low_power"] else "—",
            }
        )
        print(f"{ycol} leftover {name}: {_f(res['cv'])} R2={_f(info['r2'])}")
    y3_days = _cv(store[(Y3, "after days")])
    y3_miss = _cv(store[(Y3, "after miss_cp")])
    y3_both = _cv(store[(Y3, "after miss+days")])
    y5_size = _cv(store[(Y5_AR, "after size")])
    y5_miss = _cv(store[(Y5_AR, "after miss_cp")])
    y3_dies = not (np.isfinite(y3_days) and y3_days >= CHANCE)
    y5_dies = not (np.isfinite(y5_size) and y5_size >= CHANCE)
    twin_dies = not (np.isfinite(y3_miss) and y3_miss >= CHANCE) and not (
        np.isfinite(y5_miss) and y5_miss >= CHANCE
    )
    prose = (
        f"Y3 leftover after days {_f(y3_days)} "
        f"({'dies <0.55' if y3_dies else 'lives'}). "
        f"Y5 leftover after size {_f(y5_size)} "
        f"({'dies <0.55' if y5_dies else 'lives'}). "
        f"Leftover after miss_cp Y3 {_f(y3_miss)} Y5 {_f(y5_miss)} "
        f"({'dies — twins' if twin_dies else 'still leftover after twin'}). "
        f"Y3 after miss+days {_f(y3_both)}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "infos": infos,
        "y3_days": y3_days,
        "y3_size": _cv(store[(Y3, "after size")]),
        "y3_miss": y3_miss,
        "y3_both": y3_both,
        "y3_uncat": _cv(store[(Y3, "after uncat")]),
        "y5_size": y5_size,
        "y5_days": _cv(store[(Y5_AR, "after days")]),
        "y5_miss": y5_miss,
        "y5_miss_size": _cv(store[(Y5_AR, "after miss+size")]),
        "y2_miss": _cv(store[(Y2, "after miss_cp")]),
        "y3_dies": y3_dies,
        "y5_dies": y5_dies,
        "twin_dies": twin_dies,
        "r2_days": infos[(Y3, "after days")]["r2"],
        "r2_miss": infos[(Y3, "after miss_cp")]["r2"],
        "r2_size5": infos[(Y5_AR, "after size")]["r2"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 7. Fold 3 / one-group hole; drop fold 3 — does 0.611 collapse?
# ---------------------------------------------------------------------------
def pass7_fold3(tr: pd.DataFrame) -> dict:
    y = pd.to_numeric(tr[Y5_AR], errors="coerce")
    lab = y.notna()
    share = pd.to_numeric(tr["d_tx_cp_share"], errors="coerce")
    nc = pd.to_numeric(tr["d_n_cust"], errors="coerce")
    folds = pd.to_numeric(tr["fold"], errors="coerce")
    med_n = float(nc[lab & nc.notna()].median()) if (lab & nc.notna()).any() else float("nan")
    hi_cust = nc > med_n
    hole = lab & share.notna() & nc.notna() & (share <= HOLE_SHARE) & hi_cust
    named_hi = lab & share.notna() & nc.notna() & (share > HOLE_SHARE) & hi_cust
    hole_pos = hole & (y == 1)
    n_hole_pos = int(hole_pos.sum())
    fold_rows = []
    fold3_share = float("nan")
    for k in range(N_FOLDS):
        h = hole & (folds == k)
        n_h = int(h.sum())
        n_hp = int((h & (y == 1)).sum())
        n_n = int((named_hi & (folds == k)).sum())
        n_np = int((named_hi & (folds == k) & (y == 1)).sum())
        fold_rows.append(
            {
                "fold": k,
                "n_hole": n_h,
                "hole_pos": n_hp,
                "P(Y5=1) hole": _pp(_pct(n_hp, n_h)),
                "P(Y5=1) named": _pp(_pct(n_np, n_n)),
                "n_named": n_n,
            }
        )
        if k == 3 and n_hole_pos:
            fold3_share = _pct(n_hp, n_hole_pos)
    wo = lab & (folds != 3)
    hole_wo = _pct(int((hole & wo & (y == 1)).sum()), int((hole & wo).sum()))
    named_wo = _pct(int((named_hi & wo & (y == 1)).sum()), int((named_hi & wo).sum()))
    survives = bool(
        np.isfinite(hole_wo) and np.isfinite(named_wo) and (hole_wo - named_wo) >= 0.04
    )
    one_group = bool(np.isfinite(fold3_share) and fold3_share >= 0.70)
    rec_all = signed_oof_auroc(tr[Y5_AR], tr["d_tx_cp_share"], tr["fold"], lab)
    rec_wo = signed_oof_auroc(tr[Y5_AR], tr["d_tx_cp_share"], tr["fold"], wo)
    collapse = bool(
        np.isfinite(rec_all["train_auc"])
        and np.isfinite(rec_wo["train_auc"])
        and (rec_all["train_auc"] - rec_wo["train_auc"]) >= 0.03
    )
    # groups that own fold-3 hole positives
    g = (
        tr.loc[hole_pos & (folds == 3), "group_id"]
        .value_counts()
        .rename_axis("group_id")
        .reset_index(name="n")
    )
    n_g_f3 = int(g.shape[0])
    top_g = str(g.iloc[0]["group_id"]) if len(g) else "—"
    top_n = int(g.iloc[0]["n"]) if len(g) else 0
    prose = (
        f"Y5 AR hole pos in fold 3: {_pp(fold3_share)} of hole positives "
        f"({'CONFIRM one-group' if one_group else 'not ≥70%'}). "
        f"Without fold 3: hole {_pp(hole_wo)} vs named {_pp(named_wo)} "
        f"(survives={survives}). "
        f"Y5 d_tx train all {_f(rec_all['train_auc'])} / drop-fold3 {_f(rec_wo['train_auc'])} "
        f"CV all {_f(rec_all['cv'])} / drop {_f(rec_wo['cv'])} "
        f"({'0.611 collapses without fold 3' if collapse else 'train quote does not collapse ≥0.03'}). "
        f"Fold-3 hole-pos groups={n_g_f3} top={top_g} n={top_n}."
    )
    print(prose)
    return {
        "rows": fold_rows,
        "fold3_share": fold3_share,
        "one_group": one_group,
        "survives": survives,
        "hole_wo": hole_wo,
        "named_wo": named_wo,
        "n_hole_pos": n_hole_pos,
        "med_n_cust": med_n,
        "train_all": rec_all["train_auc"],
        "train_wo": rec_wo["train_auc"],
        "cv_all": rec_all["cv"],
        "cv_wo": rec_wo["cv"],
        "collapse": collapse,
        "n_g_f3": n_g_f3,
        "top_g": top_g,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 8. SIZE terciles — does d_tx survive inside T1?
# ---------------------------------------------------------------------------
def pass8_terciles(tr: pd.DataFrame) -> dict:
    size = pd.to_numeric(tr["log_in3"], errors="coerce")
    defined = size.notna()
    terc = pd.Series(np.nan, index=tr.index, dtype=object)
    terc.loc[defined] = pd.qcut(size[defined].rank(method="first"), 3, labels=["T1", "T2", "T3"])
    rows = []
    store = {}
    for ycol in (Y3, Y5_AR):
        for t in ("T1", "T2", "T3"):
            m = terc == t
            res = signed_oof_auroc(tr[ycol], tr["d_tx_cp_share"], tr["fold"], m & tr[ycol].notna())
            store[(ycol, t)] = res
            sl = tr.loc[m]
            rows.append(
                {
                    "y": ycol,
                    "size_tercile": t,
                    "p50_log_in3": _f(float(size[m].median())),
                    "n": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                    "mean_dtx": _f(float(pd.to_numeric(sl["d_tx_cp_share"], errors="coerce").mean())),
                }
            )
    t1_y3 = _cv(store[(Y3, "T1")])
    t1_y5 = _cv(store[(Y5_AR, "T1")])
    survive_t1 = bool(
        (np.isfinite(t1_y3) and t1_y3 >= CHANCE) or (np.isfinite(t1_y5) and t1_y5 >= CHANCE)
    )
    prose = (
        f"Y3 d_tx inside T1 {_f(t1_y3)} T2 {_f(_cv(store[(Y3, 'T2')]))} "
        f"T3 {_f(_cv(store[(Y3, 'T3')]))}. "
        f"Y5 T1 {_f(t1_y5)} T2 {_f(_cv(store[(Y5_AR, 'T2')]))} "
        f"T3 {_f(_cv(store[(Y5_AR, 'T3')]))}. "
        f"{'survives in T1' if survive_t1 else 'dies inside T1 — not a small-firm leftover'}."
    )
    print(prose)
    return {
        "rows": rows,
        "t1_y3": t1_y3,
        "t1_y5": t1_y5,
        "t2_y5": _cv(store[(Y5_AR, "T2")]),
        "t3_y5": _cv(store[(Y5_AR, "T3")]),
        "survive_t1": survive_t1,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 9. Q6 lag1/lag3 on short books
# ---------------------------------------------------------------------------
def pass9_q6(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    short = tr["so_far_class"] == "short_<12"
    long = tr["so_far_class"] == "long_>=18"
    for ycol in (Y3, Y5_AR):
        for sl_name, mask in (("all", pd.Series(True, index=tr.index)), ("short_<12", short), ("long_>=18", long)):
            for col, lag in (("d_tx_cp_share", 0), ("d_tx_cp_share_lag1", 1), ("d_tx_cp_share_lag3", 3)):
                if col not in tr.columns:
                    continue
                res = signed_oof_auroc(tr[ycol], tr[col], tr["fold"], mask & tr[ycol].notna())
                store[(ycol, sl_name, col)] = res
                rows.append(
                    {
                        "y": ycol,
                        "slice": sl_name,
                        "col": col,
                        "lag": lag,
                        "n": f"{res['n_defined']:,}",
                        "n_pos": f"{res['n_pos']:,}",
                        "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                        "sign": res["train_sign"] if not res["low_power"] else "—",
                    }
                )
    y5_now = _cv(store.get((Y5_AR, "all", "d_tx_cp_share"), {"low_power": True}))
    y5_lag1 = _cv(store.get((Y5_AR, "all", "d_tx_cp_share_lag1"), {"low_power": True}))
    y5_short = _cv(store.get((Y5_AR, "short_<12", "d_tx_cp_share"), {"low_power": True}))
    y5_short_l1 = _cv(store.get((Y5_AR, "short_<12", "d_tx_cp_share_lag1"), {"low_power": True}))
    y3_lag1 = _cv(store.get((Y3, "all", "d_tx_cp_share_lag1"), {"low_power": True}))
    q6_close = True
    if np.isfinite(y5_short_l1) and y5_short_l1 >= CHANCE and np.isfinite(y5_now) and y5_short_l1 >= y5_now - 0.02:
        q6_close = False
    prose = (
        f"Y5 now {_f(y5_now)} lag1 {_f(y5_lag1)}; short now {_f(y5_short)} "
        f"lag1 {_f(y5_short_l1)}. Y3 lag1 {_f(y3_lag1)}. "
        f"{'Q6 CLOSE — short books die / lag does not hold' if q6_close else 'Q6 leftover lives on short lag1'}."
    )
    print(prose)
    return {
        "rows": rows,
        "y5_now": y5_now,
        "y5_lag1": y5_lag1,
        "y5_short": y5_short,
        "y5_short_l1": y5_short_l1,
        "y3_lag1": y3_lag1,
        "q6_close": q6_close,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 10. ICC / company-demean
# ---------------------------------------------------------------------------
def pass10_icc(tr: pd.DataFrame) -> dict:
    dtx = tr["d_tx_cp_share"]
    icc = icc_anova(dtx, tr["company_id"])
    acf1 = median_acf(dtx, tr["company_id"], 1)
    acf3 = median_acf(dtx, tr["company_id"], 3)
    acf6 = median_acf(dtx, tr["company_id"], 6)
    mu = company_mean(dtx, tr["company_id"])
    shock = company_demean(dtx, tr["company_id"])
    rows = []
    store = {}
    for ycol in (Y3, Y5_AR, Y2):
        for name, x in (("co_mean", mu), ("demean", shock), ("now", dtx)):
            res = signed_oof_auroc(tr[ycol], x, tr["fold"], tr[ycol].notna())
            store[(ycol, name)] = res
            rows.append(_auc_row(ycol, name, res))
    style = bool(np.isfinite(icc["icc"]) and icc["icc"] >= ICC_STYLE)
    miss_icc_note = "missing-CP ICC 0.967"
    prose = (
        f"d_tx acf1={_f(acf1)} acf3={_f(acf3)} acf6={_f(acf6)}; "
        f"ICC={_f(icc['icc'])} k={icc['k']} "
        f"({'sticky fill habit (BETWEEN)' if style else 'more month shock'}). "
        f"Y3 mean {_f(_cv(store[(Y3, 'co_mean')]))} vs demean {_f(_cv(store[(Y3, 'demean')]))}; "
        f"Y5 mean {_f(_cv(store[(Y5_AR, 'co_mean')]))} vs demean {_f(_cv(store[(Y5_AR, 'demean')]))}. "
        f"Compare {miss_icc_note}."
    )
    print(prose)
    return {
        "rows": rows,
        "icc": icc["icc"],
        "acf1": acf1,
        "acf3": acf3,
        "acf6": acf6,
        "k": icc["k"],
        "style": style,
        "y3_mean": _cv(store[(Y3, "co_mean")]),
        "y3_shock": _cv(store[(Y3, "demean")]),
        "y5_mean": _cv(store[(Y5_AR, "co_mean")]),
        "y5_shock": _cv(store[(Y5_AR, "demean")]),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 11. Dark 470: d_tx defined? NaN vs 0
# ---------------------------------------------------------------------------
def pass11_dark_def(tr: pd.DataFrame) -> dict:
    dtx = pd.to_numeric(tr["d_tx_cp_share"], errors="coerce")
    dark = ~tr["ever_erp"]
    erp = tr["ever_erp"]
    rows = []
    store = {}
    for name, m in (("invoiced_744", erp), ("dark_470", dark)):
        sl_mask = m
        res3 = signed_oof_auroc(tr[Y3], dtx, tr["fold"], sl_mask & tr[Y3].notna())
        res5 = signed_oof_auroc(tr[Y5_AR], dtx, tr["fold"], sl_mask & tr[Y5_AR].notna())
        store[(name, "Y3")] = res3
        store[(name, "Y5")] = res5
        sl = tr.loc[m]
        x = dtx[m]
        rows.append(
            {
                "slice": name,
                "n_cm": int(m.sum()),
                "n_co": int(sl["company_id"].nunique()),
                "dtx_nn": int(x.notna().sum()),
                "dtx_eq0": int((x == 0).sum()),
                "dtx_nan": int(x.isna().sum()),
                "mean": _f(float(x.mean())),
                "Y3_CV": "LOW_POWER" if res3["low_power"] else _f(res3["cv"]),
                "Y5_CV": "LOW_POWER" if res5["low_power"] else _f(res5["cv"]),
                "Y5_n_pos": res5["n_pos"],
            }
        )
    # Y5 on dark should be empty / LOW_POWER (no invoice book)
    y5_dark = store[("dark_470", "Y5")]
    y5_erp = _cv(store[("invoiced_744", "Y5")])
    dark_skill_dead = bool(y5_dark["low_power"] or (np.isfinite(y5_dark["cv"]) and y5_dark["cv"] < CHANCE))
    prose = (
        f"Dark 470 d_tx defined {_pp(_pct(int(dtx[dark].notna().sum()), int(dark.sum())))} "
        f"eq0 {_pp(_pct(int((dtx[dark] == 0).sum()), int(dark.sum())))} "
        f"NaN {_pp(_pct(int(dtx[dark].isna().sum()), int(dark.sum())))}. "
        f"Y5 on 744 {_f(y5_erp)}; dark Y5 "
        f"{'LOW_POWER / dead — needs named bank CPs on an invoice book' if dark_skill_dead else _f(y5_dark['cv'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "y5_erp": y5_erp,
        "y3_erp": _cv(store[("invoiced_744", "Y3")]),
        "y3_dark": _cv(store[("dark_470", "Y3")]),
        "y5_dark_lp": y5_dark["low_power"],
        "dark_skill_dead": dark_skill_dead,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra — holdout coverage only
# ---------------------------------------------------------------------------
def pass_holdout(panel: pd.DataFrame) -> dict:
    ho = panel[panel["split"] == "holdout"].copy()
    assert set(ho["company_id"]) <= load_holdout()
    rows = []
    for col in ("d_tx_cp_share", "miss_cp_share", "named_share", "inv_cp_share", "a_uncat_share"):
        if col not in ho.columns:
            continue
        s = pd.to_numeric(ho[col], errors="coerce")
        rows.append(
            {
                "col": col,
                "n_cm": int(len(ho)),
                "n_co": int(ho["company_id"].nunique()),
                "defined": _pp(_pct(int(s.notna().sum()), len(ho))),
                "mean": _f(float(s.mean())),
                "p50": _f(float(s.median())),
                "eq0": _pp(_pct(int((s == 0).sum()), len(ho))),
            }
        )
    prose = f"Holdout coverage only (no AUROC): {ho['company_id'].nunique()} companies / {len(ho)} CM."
    print(prose)
    return {"rows": rows, "n_co": int(ho["company_id"].nunique()), "n_cm": int(len(ho)), "prose": prose}


# ---------------------------------------------------------------------------
# Extra — quintiles (descriptive)
# ---------------------------------------------------------------------------
def pass_quintiles(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["d_tx_cp_share"], errors="coerce")
    ok = x.notna()
    q = pd.Series(np.nan, index=tr.index, dtype=float)
    try:
        q.loc[ok] = pd.qcut(x[ok].rank(method="first"), 5, labels=False) + 1
        n_bins = 5
    except ValueError:
        q.loc[ok] = pd.qcut(x[ok].rank(method="first"), 2, labels=False) + 1
        n_bins = 2
    rows = []
    rates = {Y2: [], Y3: [], Y5_AR: []}
    for i in range(1, n_bins + 1):
        m = q == i
        sl = tr.loc[m]
        row = {
            "q": i,
            "n_cm": int(m.sum()),
            "p50": _f(float(x[m].median())),
            "Y2": _pp(float(pd.to_numeric(sl[Y2], errors="coerce").mean())),
            "Y3": _pp(float(pd.to_numeric(sl[Y3], errors="coerce").mean())),
            "Y5_AR": _pp(float(pd.to_numeric(sl[Y5_AR], errors="coerce").mean())),
        }
        rows.append(row)
        for yc in (Y2, Y3, Y5_AR):
            rates[yc].append(float(pd.to_numeric(sl[yc], errors="coerce").mean()))
    y5 = rates[Y5_AR]
    head_only = bool(len(y5) >= 5 and y5[0] == max(y5) and y5[0] - y5[-1] >= 0.04)
    prose = (
        f"d_tx quintiles bins={n_bins}. Y5 Q1 {_pp(y5[0]) if y5 else '—'} "
        f"Q{n_bins} {_pp(y5[-1]) if y5 else '—'}"
        f"{' head_only (night Q5 shape)' if head_only else ''}."
    )
    print(prose)
    return {"rows": rows, "n_bins": n_bins, "head_only": head_only, "y5_q1": y5[0] if y5 else float("nan"), "prose": prose}


# ---------------------------------------------------------------------------
# Extra — monthly named vs 6m d_tx
# ---------------------------------------------------------------------------
def pass_monthly_vs_6m(tr: pd.DataFrame) -> dict:
    named = pd.to_numeric(tr["named_share"], errors="coerce")
    dtx = pd.to_numeric(tr["d_tx_cp_share"], errors="coerce")
    both = named.notna() & dtx.notna()
    rho = spearman(named, dtx)
    delta = (named - dtx).abs()
    prose = (
        f"Monthly named_share vs 6m d_tx ρ={_f(rho)} "
        f"max|Δ|={_f(float(delta[both].max()) if both.any() else float('nan'))} "
        f"mean|Δ|={_f(float(delta[both].mean()) if both.any() else float('nan'))}. "
        f"miss vs 1−d_tx ρ={_f(spearman(tr['miss_cp_share'], 1.0 - dtx))}."
    )
    print(prose)
    return {
        "rho": rho,
        "max_d": float(delta[both].max()) if both.any() else float("nan"),
        "mean_d": float(delta[both].mean()) if both.any() else float("nan"),
        "n": int(both.sum()),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra — rank leftover + zero dummy
# ---------------------------------------------------------------------------
def pass_rank_zero(tr: pd.DataFrame) -> dict:
    dtx = pd.to_numeric(tr["d_tx_cp_share"], errors="coerce")
    days = tr["c_n_days_with_tx"]
    size = tr["log_in3"]
    miss = tr["miss_cp_share"]
    zero = (dtx == 0).astype(float).where(dtx.notna())
    gt0 = dtx.where(dtx > 0)
    r_days = rank_resid(dtx, days)
    r_size = rank_resid(dtx, size)
    r_miss = rank_resid(dtx, miss)
    rows = []
    store = {}
    checks = [
        (Y3, "rank leftover days", r_days),
        (Y3, "OLS leftover days", ols_resid(dtx, days)[0]),
        (Y3, "zero dummy", zero),
        (Y3, "d_tx | >0", gt0),
        (Y5_AR, "rank leftover size", r_size),
        (Y5_AR, "OLS leftover size", ols_resid(dtx, size)[0]),
        (Y5_AR, "rank leftover miss", r_miss),
        (Y5_AR, "zero dummy", zero),
        (Y5_AR, "d_tx | >0", gt0),
    ]
    for ycol, name, x in checks:
        res = signed_oof_auroc(tr[ycol], x, tr["fold"], tr[ycol].notna())
        store[(ycol, name)] = res
        rows.append(_auc_row(ycol, name, res))
    y3_rank = _cv(store[(Y3, "rank leftover days")])
    y5_rank = _cv(store[(Y5_AR, "rank leftover size")])
    y5_gt0 = _cv(store[(Y5_AR, "d_tx | >0")])
    y5_zero = _cv(store[(Y5_AR, "zero dummy")])
    presence = bool(np.isfinite(y5_zero) and np.isfinite(y5_gt0) and y5_gt0 < CHANCE and y5_zero >= CHANCE)
    prose = (
        f"Y3 rank leftover after days {_f(y3_rank)}. "
        f"Y5 rank leftover after size {_f(y5_rank)}; "
        f"zero dummy {_f(y5_zero)}; intensity on >0 {_f(y5_gt0)}. "
        f"{'presence rewrite' if presence else 'not only a zero dummy (intensity still ranks or zero dies)'}."
    )
    print(prose)
    return {
        "rows": rows,
        "y3_rank": y3_rank,
        "y5_rank": y5_rank,
        "y5_gt0": y5_gt0,
        "y5_zero": y5_zero,
        "presence": presence,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra — leftover after days on invoiced-744 only
# ---------------------------------------------------------------------------
def pass_erp_leftover(tr: pd.DataFrame) -> dict:
    sl = tr.loc[tr["ever_erp"]].copy()
    dtx = sl["d_tx_cp_share"]
    rows = []
    store = {}
    for ycol, name, xs in (
        (Y3, "raw", None),
        (Y3, "after days", (sl["c_n_days_with_tx"],)),
        (Y3, "after miss_cp", (sl["miss_cp_share"],)),
        (Y5_AR, "raw", None),
        (Y5_AR, "after size", (sl["log_in3"],)),
        (Y5_AR, "after miss_cp", (sl["miss_cp_share"],)),
    ):
        if xs is None:
            res = signed_oof_auroc(sl[ycol], dtx, sl["fold"], sl[ycol].notna())
            info = {"r2": float("nan")}
        else:
            res, info = _leftover_one(sl, ycol, dtx, *xs)
        store[(ycol, name)] = res
        rows.append(
            {
                "y": ycol,
                "residual": name,
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                "train": "LOW_POWER" if res["low_power"] else _f(res["train_auc"]),
                "R²": _f(info["r2"]),
            }
        )
    rho744 = spearman(sl["d_tx_cp_share"], sl["miss_cp_share"])
    prose = (
        f"Invoiced-744: miss↔d_tx ρ={_f(rho744)}. "
        f"Y3 raw {_f(_cv(store[(Y3, 'raw')]))} leftover-days {_f(_cv(store[(Y3, 'after days')]))} "
        f"leftover-miss {_f(_cv(store[(Y3, 'after miss_cp')]))}. "
        f"Y5 raw {_f(_cv(store[(Y5_AR, 'raw')]))} leftover-size {_f(_cv(store[(Y5_AR, 'after size')]))} "
        f"leftover-miss {_f(_cv(store[(Y5_AR, 'after miss_cp')]))}."
    )
    print(prose)
    return {
        "rows": rows,
        "rho744": rho744,
        "y3_days": _cv(store[(Y3, "after days")]),
        "y3_miss": _cv(store[(Y3, "after miss_cp")]),
        "y5_size": _cv(store[(Y5_AR, "after size")]),
        "y5_miss": _cv(store[(Y5_AR, "after miss_cp")]),
        "y5_raw": _cv(store[(Y5_AR, "raw")]),
        "y5_train": store[(Y5_AR, "raw")]["train_auc"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra — leftover after days fold-wise
# ---------------------------------------------------------------------------
def pass_fold_leftover(tr: pd.DataFrame) -> dict:
    resid, info = ols_resid(tr["d_tx_cp_share"], tr["c_n_days_with_tx"])
    res = signed_oof_auroc(tr[Y3], resid, tr["fold"], tr[Y3].notna())
    rows = []
    for r in res.get("folds", []):
        rows.append(
            {
                "fold": r["fold"],
                "leftover_days": _f(r["auroc"]),
                "n_va": r["n_va"],
                "n_pos": r["n_pos"],
            }
        )
    spread = float("nan")
    finite = [r["auroc"] for r in res.get("folds", []) if np.isfinite(r["auroc"])]
    if len(finite) >= 2:
        spread = float(max(finite) - min(finite))
    prose = (
        f"Y3 leftover-after-days fold spread {_f(spread)} R2={_f(info['r2'])}. "
        f"{'one-fold leftover' if np.isfinite(spread) and spread >= 0.15 else 'fold leftover not a dummy'}."
    )
    print(prose)
    return {"rows": rows, "spread": spread, "cv": _cv(res), "r2": info["r2"], "prose": prose}


# ---------------------------------------------------------------------------
# Extra — group ICC of company-median
# ---------------------------------------------------------------------------
def pass_group_icc(tr: pd.DataFrame) -> dict:
    g = (
        tr.groupby("company_id", as_index=False)
        .agg(med=("d_tx_cp_share", "median"), group_id=("group_id", "first"))
    )
    icc = icc_anova(g["med"], g["group_id"])
    prose = f"Company-median d_tx ICC across group_id {_f(icc['icc'])} (k={icc['k']})."
    print(prose)
    return {"icc": icc["icc"], "k": icc["k"], "prose": prose}


def leftover_diag(y, x, controls, folds, mask) -> dict:
    resid, info = ols_resid(x, *controls)
    rec = signed_oof_auroc(y, resid, folds, mask)
    rho_c = spearman(resid, controls[0]) if controls else float("nan")
    rho_x = spearman(resid, x)
    xr = pd.to_numeric(x, errors="coerce").rank(method="average")
    cr = [pd.to_numeric(c, errors="coerce").rank(method="average") for c in controls]
    rresid, _ = ols_resid(xr, *cr)
    rrec = signed_oof_auroc(y, rresid, folds, mask)
    # labeled-only OLS
    lab = mask.fillna(False)
    xs_l = [c.where(lab) for c in controls]
    lresid, linfo = ols_resid(x.where(lab), *xs_l)
    lrec = signed_oof_auroc(y, lresid, folds, mask)
    fake = bool(np.isfinite(rho_c) and abs(rho_c) >= TWIN_RHO)
    rank_cv = _cv(rrec)
    ols_cv = _cv(rec)
    honest_dies = bool(fake or (np.isfinite(rank_cv) and rank_cv < CHANCE))
    return {
        "ols": ols_cv,
        "rank": rank_cv,
        "labeled": _cv(lrec),
        "rho_ctrl": rho_c,
        "rho_x": rho_x,
        "r2": info["r2"],
        "r2_lab": linfo["r2"],
        "fake": fake,
        "honest_dies": honest_dies,
        "folds": fold_bits(rec),
        "rank_folds": fold_bits(rrec),
        "n": rec["n_defined"],
        "n_pos": rec["n_pos"],
    }


def pass_honest(tr: pd.DataFrame) -> dict:
    dtx = tr["d_tx_cp_share"]
    days = tr["c_n_days_with_tx"]
    size = tr["log_in3"]
    miss = tr["miss_cp_share"]
    erp = tr["ever_erp"].astype(float)
    rows = []
    store = {}
    specs = [
        (Y3, "after days", (days,)),
        (Y3, "after size", (size,)),
        (Y3, "after miss_cp", (miss,)),
        (Y3, "after ever_erp", (erp,)),
        (Y3, "after days+erp", (days, erp)),
        (Y5_AR, "after size", (size,)),
        (Y5_AR, "after miss_cp", (miss,)),
        (Y5_AR, "after ever_erp", (erp,)),
    ]
    for ycol, name, xs in specs:
        rec = leftover_diag(tr[ycol], dtx, list(xs), tr["fold"], tr[ycol].notna())
        store[(ycol, name)] = rec
        rows.append(
            {
                "y": ycol,
                "control": name,
                "OLS": _f(rec["ols"]),
                "rank": _f(rec["rank"]),
                "labeled-OLS": _f(rec["labeled"]),
                "ρ(resid,ctrl)": _f(rec["rho_ctrl"]),
                "ρ(resid,dtx)": _f(rec["rho_x"]),
                "R²": _f(rec["r2"]),
                "honest_dies": "YES" if rec["honest_dies"] else "no",
            }
        )
        print(
            f"honest {ycol} {name}: OLS={_f(rec['ols'])} rank={_f(rec['rank'])} "
            f"lab={_f(rec['labeled'])} ρctrl={_f(rec['rho_ctrl'])} ρx={_f(rec['rho_x'])}"
        )
    y3 = store[(Y3, "after days")]
    y5 = store[(Y5_AR, "after size")]
    y3_erp = store[(Y3, "after ever_erp")]
    # 0.600 vs raw 0.534: if resid≈dtx (ρ~1) leftover should match raw
    clone = bool(np.isfinite(y3["rho_x"]) and abs(y3["rho_x"]) >= 0.95)
    prose = (
        f"Y3 leftover-days OLS {_f(y3['ols'])} rank {_f(y3['rank'])} "
        f"labeled-OLS {_f(y3['labeled'])} ρ(resid,days)={_f(y3['rho_ctrl'])} "
        f"ρ(resid,dtx)={_f(y3['rho_x'])} "
        f"({'resid≈d_tx clone — OLS 0.600 is not new leftover' if clone else 'resid is not a d_tx clone'}). "
        f"Y5 leftover-size OLS {_f(y5['ols'])} rank {_f(y5['rank'])}. "
        f"Y3 leftover after ever_erp {_f(y3_erp['ols'])} rank {_f(y3_erp['rank'])} "
        f"(dark is a 0-fill). Honest leftover "
        f"{'DIES' if y3['honest_dies'] else 'lives'}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "y3_rank": y3["rank"],
        "y3_ols": y3["ols"],
        "y3_lab": y3["labeled"],
        "y3_rho_x": y3["rho_x"],
        "y3_clone": clone,
        "y3_honest_dies": y3["honest_dies"],
        "y5_rank": y5["rank"],
        "y5_ols": y5["ols"],
        "y3_erp": y3_erp["ols"],
        "y3_erp_rank": y3_erp["rank"],
        "prose": prose,
    }


def pass_j_match(tr: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """In-memory Family J leftover. Do not merge J. Do not write parquet."""
    from analysis.features.match import build as build_j

    con = connect()
    try:
        grid = panel[["company_id", "period"]].drop_duplicates()
        j = build_j(con, grid)
    finally:
        con.close()
    j = _keys(j[["company_id", "period", "j_pay_match", "j_has_book"]])
    sl = tr.merge(j, on=["company_id", "period"], how="left")
    assert_no_holdout(sl["company_id"])
    rho = spearman(sl["d_tx_cp_share"], sl["j_pay_match"])
    twin = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
    rows = []
    store = {}
    for ycol in (Y3, Y5_AR):
        raw = signed_oof_auroc(sl[ycol], sl["j_pay_match"], sl["fold"], sl[ycol].notna())
        resid, info = ols_resid(sl["d_tx_cp_share"], sl["j_pay_match"])
        left = signed_oof_auroc(sl[ycol], resid, sl["fold"], sl[ycol].notna())
        store[(ycol, "j")] = raw
        store[(ycol, "left")] = left
        rows.append(
            {
                "y": ycol,
                "feature": "j_pay_match",
                "n": f"{raw['n_defined']:,}",
                "n_pos": f"{raw['n_pos']:,}",
                "CV": "LOW_POWER" if raw["low_power"] else _f(raw["cv"]),
                "d_tx leftover after J": "LOW_POWER" if left["low_power"] else _f(left["cv"]),
                "R²": _f(info["r2"]),
            }
        )
    y3_left = _cv(store[(Y3, "left")])
    y5_left = _cv(store[(Y5_AR, "left")])
    nn = int(pd.to_numeric(sl["j_pay_match"], errors="coerce").notna().sum())
    dark_nn = int(
        pd.to_numeric(sl.loc[~sl["ever_erp"], "j_pay_match"], errors="coerce").notna().sum()
    )
    prose = (
        f"In-memory j_pay_match (not merged). ρ vs d_tx={_f(rho)} "
        f"({'TWIN' if twin else 'not a twin'}). "
        f"Defined {nn:,} train CM; dark finite {dark_nn} (must be 0). "
        f"Y3 leftover after J {_f(y3_left)}; Y5 leftover after J {_f(y5_left)}. "
        "Do not merge Family J."
    )
    print(prose)
    return {
        "rows": rows,
        "rho": rho,
        "twin": twin,
        "y3_left": y3_left,
        "y5_left": y5_left,
        "y3_j": _cv(store[(Y3, "j")]),
        "y5_j": _cv(store[(Y5_AR, "j")]),
        "n": nn,
        "dark_nn": dark_nn,
        "prose": prose,
    }


def pass_dropf3_leftover(tr: pd.DataFrame) -> dict:
    wo = tr["fold"] != 3
    dtx = tr["d_tx_cp_share"]
    rows = []
    store = {}
    specs = [
        (Y5_AR, "raw drop-f3", dtx, wo & tr[Y5_AR].notna()),
        (Y5_AR, "leftover size drop-f3", ols_resid(dtx, tr["log_in3"])[0], wo & tr[Y5_AR].notna()),
        (Y5_AR, "leftover miss drop-f3", ols_resid(dtx, tr["miss_cp_share"])[0], wo & tr[Y5_AR].notna()),
        (Y3, "raw drop-f3", dtx, wo & tr[Y3].notna()),
        (Y3, "leftover days drop-f3", ols_resid(dtx, tr["c_n_days_with_tx"])[0], wo & tr[Y3].notna()),
        (Y3, "rank leftover days drop-f3", rank_resid(dtx, tr["c_n_days_with_tx"]), wo & tr[Y3].notna()),
    ]
    for ycol, name, x, mask in specs:
        res = signed_oof_auroc(tr[ycol], x, tr["fold"], mask)
        store[name] = res
        rows.append(_auc_row(ycol, name, res))
    y5_raw = store["raw drop-f3"]["train_auc"]
    y5_cv = _cv(store["raw drop-f3"])
    y5_size = _cv(store["leftover size drop-f3"])
    y3_rank = _cv(store["rank leftover days drop-f3"])
    prose = (
        f"Drop fold 3: Y5 train {_f(y5_raw)} CV {_f(y5_cv)} leftover-size {_f(y5_size)}; "
        f"Y3 rank leftover-days {_f(y3_rank)}. "
        f"{'0.611 / leftover-size die without the one group' if (np.isfinite(y5_raw) and y5_raw < 0.56) else 'still lives without fold 3'}."
    )
    print(prose)
    return {
        "rows": rows,
        "y5_train": y5_raw,
        "y5_cv": y5_cv,
        "y5_size": y5_size,
        "y3_rank": y3_rank,
        "prose": prose,
    }


def pass_sofar(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    bins = [
        ("1-3", (tr["months_so_far"] >= 1) & (tr["months_so_far"] <= 3)),
        ("4-6", (tr["months_so_far"] >= 4) & (tr["months_so_far"] <= 6)),
        ("7-12", (tr["months_so_far"] >= 7) & (tr["months_so_far"] <= 12)),
        ("13-18", (tr["months_so_far"] >= 13) & (tr["months_so_far"] <= 18)),
        ("19-24", (tr["months_so_far"] >= 19) & (tr["months_so_far"] <= 24)),
    ]
    dtx = pd.to_numeric(tr["d_tx_cp_share"], errors="coerce")
    for name, m in bins:
        res3 = signed_oof_auroc(tr[Y3], dtx, tr["fold"], m & tr[Y3].notna())
        res5 = signed_oof_auroc(tr[Y5_AR], dtx, tr["fold"], m & tr[Y5_AR].notna())
        store[(name, "Y3")] = res3
        store[(name, "Y5")] = res5
        rows.append(
            {
                "so_far": name,
                "n_cm": int(m.sum()),
                "dtx_mean": _f(float(dtx[m].mean())),
                "Y3_CV": "LOW_POWER" if res3["low_power"] else _f(res3["cv"]),
                "Y3_n_pos": res3["n_pos"],
                "Y5_CV": "LOW_POWER" if res5["low_power"] else _f(res5["cv"]),
                "Y5_n_pos": res5["n_pos"],
            }
        )
    prose = (
        f"Months 1–3 d_tx mean {_f(store and float(dtx[(tr['months_so_far']>=1)&(tr['months_so_far']<=3)].mean()))} "
        f"vs 13–18 {_f(float(dtx[(tr['months_so_far']>=13)&(tr['months_so_far']<=18)].mean()))}."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def pass_q6_leftover(tr: pd.DataFrame) -> dict:
    dtx1 = tr["d_tx_cp_share_lag1"]
    days = tr["c_n_days_with_tx"]
    size = tr["log_in3"]
    rows = []
    store = {}
    short = tr["so_far_class"] == "short_<12"
    for ycol, ctrl, cname in ((Y3, days, "days"), (Y5_AR, size, "size")):
        resid, _ = ols_resid(dtx1, ctrl)
        for sl_name, mask in (("all", tr[ycol].notna()), ("short_<12", short & tr[ycol].notna())):
            res = signed_oof_auroc(tr[ycol], resid, tr["fold"], mask)
            store[(ycol, sl_name)] = res
            rows.append(_auc_row(ycol, f"lag1 leftover {cname} {sl_name}", res))
    prose = (
        f"Y3 lag1 leftover after days {_f(_cv(store[(Y3, 'all')]))} "
        f"short {_f(_cv(store[(Y3, 'short_<12')]))}. "
        f"Y5 lag1 leftover after size {_f(_cv(store[(Y5_AR, 'all')]))} "
        f"short {_f(_cv(store[(Y5_AR, 'short_<12')]))}."
    )
    print(prose)
    return {
        "rows": rows,
        "y3": _cv(store[(Y3, "all")]),
        "y3_short": _cv(store[(Y3, "short_<12")]),
        "y5": _cv(store[(Y5_AR, "all")]),
        "y5_short": _cv(store[(Y5_AR, "short_<12")]),
        "prose": prose,
    }


def pass_f3_groups(tr: pd.DataFrame) -> dict:
    y = pd.to_numeric(tr[Y5_AR], errors="coerce")
    lab = y.notna()
    share = pd.to_numeric(tr["d_tx_cp_share"], errors="coerce")
    nc = pd.to_numeric(tr["d_n_cust"], errors="coerce")
    med_n = float(nc[lab & nc.notna()].median()) if (lab & nc.notna()).any() else float("nan")
    hole = lab & (share <= HOLE_SHARE) & (nc > med_n)
    hole_pos = hole & (y == 1) & (tr["fold"] == 3)
    g = (
        tr.loc[hole_pos]
        .groupby("group_id")
        .agg(n_pos=("company_id", "size"), n_co=("company_id", "nunique"))
        .reset_index()
        .sort_values("n_pos", ascending=False)
    )
    rows = []
    for _, r in g.iterrows():
        rows.append(
            {
                "group_id": r["group_id"],
                "hole_pos": int(r["n_pos"]),
                "n_co": int(r["n_co"]),
                "share_of_f3_hole": _pp(_pct(int(r["n_pos"]), int(hole_pos.sum()))),
            }
        )
    top = str(g.iloc[0]["group_id"]) if len(g) else ""
    drop_top = tr["group_id"].astype(str) != top
    rec_drop = signed_oof_auroc(tr[Y5_AR], tr["d_tx_cp_share"], tr["fold"], drop_top & lab)
    rec_all = signed_oof_auroc(tr[Y5_AR], tr["d_tx_cp_share"], tr["fold"], lab)
    one_name = bool(len(g) == 1)
    prose = (
        f"Fold-3 hole-pos groups={len(g)} top={top} "
        f"{int(g.iloc[0]['n_pos']) if len(g) else 0}/{int(hole_pos.sum())}. "
        f"Drop only {top}: Y5 train {_f(rec_drop['train_auc'])} vs all {_f(rec_all['train_auc'])}. "
        f"{'one named group' if one_name else 'fold-3 cluster (several groups), not a single name'}."
    )
    print(prose)
    return {
        "rows": rows,
        "n_g": int(len(g)),
        "top": top,
        "train_drop_top": rec_drop["train_auc"],
        "cv_drop_top": _cv(rec_drop),
        "one_name": one_name,
        "prose": prose,
    }


def pass_744_rank(tr: pd.DataFrame) -> dict:
    sl = tr.loc[tr["ever_erp"]].copy()
    rec = leftover_diag(
        sl[Y3], sl["d_tx_cp_share"], [sl["c_n_days_with_tx"]], sl["fold"], sl[Y3].notna()
    )
    rec5 = leftover_diag(
        sl[Y5_AR], sl["d_tx_cp_share"], [sl["log_in3"]], sl["fold"], sl[Y5_AR].notna()
    )
    rec_m = leftover_diag(
        sl[Y3], sl["d_tx_cp_share"], [sl["miss_cp_share"]], sl["fold"], sl[Y3].notna()
    )
    rows = [
        {
            "y": Y3,
            "control": "days on 744",
            "OLS": _f(rec["ols"]),
            "rank": _f(rec["rank"]),
            "labeled": _f(rec["labeled"]),
            "honest_dies": "YES" if rec["honest_dies"] else "no",
        },
        {
            "y": Y5_AR,
            "control": "size on 744",
            "OLS": _f(rec5["ols"]),
            "rank": _f(rec5["rank"]),
            "labeled": _f(rec5["labeled"]),
            "honest_dies": "YES" if rec5["honest_dies"] else "no",
        },
        {
            "y": Y3,
            "control": "miss_cp on 744",
            "OLS": _f(rec_m["ols"]),
            "rank": _f(rec_m["rank"]),
            "labeled": _f(rec_m["labeled"]),
            "honest_dies": "YES" if rec_m["honest_dies"] else "no",
        },
    ]
    prose = (
        f"Invoiced-744 honest leftover: Y3-days OLS {_f(rec['ols'])} rank {_f(rec['rank'])} "
        f"labeled {_f(rec['labeled'])}; Y5-size rank {_f(rec5['rank'])}; "
        f"Y3-miss rank {_f(rec_m['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "y3_rank": rec["rank"],
        "y3_lab": rec["labeled"],
        "y5_rank": rec5["rank"],
        "y3_miss": rec_m["rank"],
        "prose": prose,
    }


def pass_partial(tr: pd.DataFrame) -> dict:
    dtx = pd.to_numeric(tr["d_tx_cp_share"], errors="coerce")
    zero = dtx == 0
    partial = dtx > 0
    rows = []
    for name, m in (("eq0", zero), (">0", partial), ("all_nn", dtx.notna())):
        res3 = signed_oof_auroc(tr[Y3], dtx, tr["fold"], m & tr[Y3].notna())
        res5 = signed_oof_auroc(tr[Y5_AR], dtx, tr["fold"], m & tr[Y5_AR].notna())
        rows.append(
            {
                "slice": name,
                "n_cm": int(m.sum()),
                "Y3_CV": "LOW_POWER" if res3["low_power"] else _f(res3["cv"]),
                "Y3_n_pos": res3["n_pos"],
                "Y5_CV": "LOW_POWER" if res5["low_power"] else _f(res5["cv"]),
                "Y5_train": "LOW_POWER" if res5["low_power"] else _f(res5["train_auc"]),
                "Y5_n_pos": res5["n_pos"],
            }
        )
    rec5 = signed_oof_auroc(tr[Y5_AR], dtx, tr["fold"], partial & tr[Y5_AR].notna())
    prose = (
        f"Y5 on d_tx>0 CV {_f(rec5['cv'])} train {_f(rec5['train_auc'])} "
        f"(night intensity 0.580). Zero dummy is not the KEEP quote."
    )
    print(prose)
    return {"rows": rows, "y5_gt0": _cv(rec5), "y5_gt0_tr": rec5["train_auc"], "prose": prose}


def pass_dn_cust(tr: pd.DataFrame) -> dict:
    rec = leftover_diag(
        tr[Y5_AR],
        tr["d_tx_cp_share"],
        [tr["d_n_cust"]],
        tr["fold"],
        tr[Y5_AR].notna(),
    )
    rec3 = leftover_diag(
        tr[Y3],
        tr["d_tx_cp_share"],
        [tr["d_n_cust"]],
        tr["fold"],
        tr[Y3].notna(),
    )
    rho = spearman(tr["d_tx_cp_share"], tr["d_n_cust"])
    rows = [
        {
            "y": Y5_AR,
            "OLS": _f(rec["ols"]),
            "rank": _f(rec["rank"]),
            "ρ(dtx,n_cust)": _f(rho),
            "honest_dies": "YES" if rec["honest_dies"] else "no",
        },
        {
            "y": Y3,
            "OLS": _f(rec3["ols"]),
            "rank": _f(rec3["rank"]),
            "ρ(dtx,n_cust)": _f(rho),
            "honest_dies": "YES" if rec3["honest_dies"] else "no",
        },
    ]
    prose = (
        f"Leftover after d_n_cust: Y5 OLS {_f(rec['ols'])} rank {_f(rec['rank'])} "
        f"Y3 rank {_f(rec3['rank'])} ρ={_f(rho)} (not a thickness twin)."
    )
    print(prose)
    return {"rows": rows, "y5_rank": rec["rank"], "y3_rank": rec3["rank"], "rho": rho, "prose": prose}


def pass_drop4(tr: pd.DataFrame, groups: list[str]) -> dict:
    drop = ~tr["group_id"].astype(str).isin(set(groups))
    rec = signed_oof_auroc(tr[Y5_AR], tr["d_tx_cp_share"], tr["fold"], drop & tr[Y5_AR].notna())
    rec3 = signed_oof_auroc(tr[Y3], tr["d_tx_cp_share"], tr["fold"], drop & tr[Y3].notna())
    rec5l = leftover_diag(
        tr[Y5_AR],
        tr["d_tx_cp_share"],
        [tr["log_in3"]],
        tr["fold"],
        drop & tr[Y5_AR].notna(),
    )
    rows = [
        _auc_row(Y5_AR, "raw drop-4-groups", rec),
        _auc_row(Y3, "raw drop-4-groups", rec3),
        {
            "y": Y5_AR,
            "feature": "leftover size drop-4",
            "n": f"{rec5l['n']:,}",
            "n_pos": f"{rec5l['n_pos']:,}",
            "CV": _f(rec5l["ols"]),
            "train": "—",
            "sd": "—",
            "sign": "—",
            "folds": rec5l["folds"],
        },
    ]
    prose = (
        f"Drop 4 fold-3 hole groups {groups}: Y5 train {_f(rec['train_auc'])} "
        f"CV {_f(rec['cv'])} leftover-size {_f(rec5l['ols'])} / rank {_f(rec5l['rank'])}. "
        f"Y3 CV {_f(rec3['cv'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "y5_train": rec["train_auc"],
        "y5_cv": rec["cv"],
        "y5_size": rec5l["ols"],
        "y5_rank": rec5l["rank"],
        "groups": groups,
        "prose": prose,
    }


def pass_delta_lag_rank(tr: pd.DataFrame) -> dict:
    dtx = pd.to_numeric(tr["d_tx_cp_share"], errors="coerce")
    lag1 = pd.to_numeric(tr["d_tx_cp_share_lag1"], errors="coerce")
    delta = dtx - lag1
    days = tr["c_n_days_with_tx"]
    size = tr["log_in3"]
    rows = []
    store = {}
    for ycol, xname, x, ctrl in (
        (Y3, "delta", delta, days),
        (Y5_AR, "delta", delta, size),
        (Y3, "lag1", lag1, days),
        (Y5_AR, "lag1", lag1, size),
    ):
        rec = leftover_diag(tr[ycol], x, [ctrl], tr["fold"], tr[ycol].notna())
        store[(ycol, xname)] = rec
        rows.append(
            {
                "y": ycol,
                "stem": xname,
                "OLS leftover": _f(rec["ols"]),
                "rank leftover": _f(rec["rank"]),
                "labeled": _f(rec["labeled"]),
                "honest_dies": "YES" if rec["honest_dies"] else "no",
            }
        )
    short = tr["so_far_class"] == "short_<12"
    rec_s = leftover_diag(
        tr[Y3], lag1, [days], tr["fold"], short & tr[Y3].notna()
    )
    prose = (
        f"Δ d_tx leftover Y5 rank {_f(store[(Y5_AR, 'delta')]['rank'])} "
        f"Y3 rank {_f(store[(Y3, 'delta')]['rank'])}. "
        f"lag1 rank leftover Y5 {_f(store[(Y5_AR, 'lag1')]['rank'])} "
        f"Y3 {_f(store[(Y3, 'lag1')]['rank'])}; short Y3 lag1 rank {_f(rec_s['rank'])} "
        f"(OLS short was 0.640 — "
        f"{'rank dies' if rec_s['honest_dies'] else 'rank lives'})."
    )
    print(prose)
    return {
        "rows": rows + [
            {
                "y": Y3,
                "stem": "lag1 short rank",
                "OLS leftover": _f(rec_s["ols"]),
                "rank leftover": _f(rec_s["rank"]),
                "labeled": _f(rec_s["labeled"]),
                "honest_dies": "YES" if rec_s["honest_dies"] else "no",
            }
        ],
        "y5_delta": store[(Y5_AR, "delta")]["rank"],
        "y3_lag1": store[(Y3, "lag1")]["rank"],
        "y3_short_rank": rec_s["rank"],
        "prose": prose,
    }


def pass_activity(tr: pd.DataFrame) -> dict:
    ntx = tr["a_n_tx"] if "a_n_tx" in tr.columns else tr["c_n_days_with_tx"]
    rec3 = leftover_diag(tr[Y3], tr["d_tx_cp_share"], [ntx], tr["fold"], tr[Y3].notna())
    rec5 = leftover_diag(tr[Y5_AR], tr["d_tx_cp_share"], [ntx], tr["fold"], tr[Y5_AR].notna())
    rho = spearman(tr["d_tx_cp_share"], ntx)
    rows = [
        {
            "y": Y3,
            "OLS": _f(rec3["ols"]),
            "rank": _f(rec3["rank"]),
            "ρ": _f(rho),
            "honest_dies": "YES" if rec3["honest_dies"] else "no",
        },
        {
            "y": Y5_AR,
            "OLS": _f(rec5["ols"]),
            "rank": _f(rec5["rank"]),
            "ρ": _f(rho),
            "honest_dies": "YES" if rec5["honest_dies"] else "no",
        },
    ]
    prose = (
        f"Leftover after a_n_tx: Y3 rank {_f(rec3['rank'])} Y5 rank {_f(rec5['rank'])} "
        f"ρ={_f(rho)}."
    )
    print(prose)
    return {"rows": rows, "y3_rank": rec3["rank"], "y5_rank": rec5["rank"], "rho": rho, "prose": prose}


def decide(p1, p2, p4, p5, p7, p8, p9, p10, p_h=None) -> dict:
    twin = bool(p2["gate_twins"])
    size = bool(p2["size_flag"])
    beat3 = bool(p4["beat3"])
    beat5 = bool(p4["beat5"])
    y3_honest_dies = bool(p5["y3_dies"] or (p_h is not None and p_h.get("y3_honest_dies")))
    y5_keep_x = bool(beat5 and (not p5["y5_dies"]) and (not size) and (not twin))
    y3_keep_x = bool(beat3 and (not y3_honest_dies) and (not size) and (not twin))
    if y3_keep_x:
        y3_x = "KEEP"
    elif p5["y3_dies"] or twin or (np.isfinite(p4["y3"]) and np.isfinite(p4["days"]) and p4["y3"] < p4["days"]):
        y3_x = "CLOSE / DROP from the 44"
    else:
        y3_x = "CLOSE"
    if y5_keep_x:
        y5_x = "KEEP"
    else:
        y5_x = "PARK as X"
    q6 = "CLOSE" if p9["q6_close"] else "KEEP"
    drop44 = (not y3_keep_x) and (not y5_keep_x)
    quote_stands = bool(p4["y5_ok"])
    y5_quote = "KEEP the 0.611 quote" if quote_stands else "document drift — cannot reproduce 0.611"
    headline = (
        f"d_tx vs miss_cp ρ={_f(p2['miss_rho'])} "
        f"({'CONFIRM TWIN −0.947' if p2['miss_ok'] else 'twin drifted'}). "
        f"Y3 leftover after days OLS {_f(p5['y3_days'])} "
        f"rank {_f(p_h['y3_rank'] if p_h else float('nan'))} vs days {_f(p4['days'])} — {y3_x}. "
        f"Y5 leftover after size {_f(p5['y5_size'])} leftover after miss {_f(p5['y5_miss'])} — {y5_x}. "
        f"Y5 train {_f(p4['y5_train'])} ({y5_quote}). "
        f"Fold-3 hole {_pp(p7['fold3_share'])} one-group={p7['one_group']} "
        f"drop-fold3 train {_f(p7['train_wo'])} collapse={p7['collapse']}. "
        f"Q6 {q6}. "
        f"{'DROP d_tx_cp_share from the 44' if drop44 else 'keep-list stay pending leftover'}. "
        f"Do not invent y_d_tx. Night Y3 0.762/0.752 and Y5 0.611 quotes unchanged."
    )
    return {
        "y3": y3_x,
        "y5": y5_x,
        "q6": q6,
        "park_y": "PARK",
        "drop44": drop44,
        "twin": twin,
        "size": size,
        "beat3": beat3,
        "beat5": beat5,
        "y3_keep_x": y3_keep_x,
        "y5_keep_x": y5_keep_x,
        "quote_stands": quote_stands,
        "y5_quote": y5_quote,
        "y3_honest_dies": y3_honest_dies,
        "headline": headline,
    }


def make_png(tr: pd.DataFrame, p7: dict) -> bool:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return False
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3))
    ax = axes[0]
    dtx = pd.to_numeric(tr["d_tx_cp_share"], errors="coerce")
    miss = pd.to_numeric(tr["miss_cp_share"], errors="coerce")
    both = dtx.notna() & miss.notna()
    ax.scatter(miss[both], dtx[both], s=5, alpha=0.18, c="#3d5a80")
    ax.set_xlabel("miss_cp_share (in-memory month)")
    ax.set_ylabel("d_tx_cp_share (6m store)")
    ax.set_title(f"twin ρ={spearman(dtx, miss):.3f}")
    ax = axes[1]
    y = pd.to_numeric(tr[Y5_AR], errors="coerce")
    lab = y.notna() & dtx.notna()
    sl = tr.loc[lab].copy()
    sl["_x"] = dtx[lab]
    sl["_y"] = y[lab]
    sl["_f3"] = sl["fold"] == 3
    try:
        sl["_q"] = pd.qcut(sl["_x"].rank(method="first"), 5, labels=False) + 1
    except ValueError:
        sl["_q"] = 1
    for flag, label, color in ((False, "folds 0–2,4", "#98c1d9"), (True, "fold 3", "#ee6c4d")):
        sub = sl.loc[sl["_f3"] == flag]
        if sub.empty:
            continue
        g = sub.groupby("_q")["_y"].mean()
        ax.plot(g.index, g.values, marker="o", label=label, color=color)
    ax.set_xlabel("d_tx_cp_share quintile")
    ax.set_ylabel("P(Y5 AR=1)")
    ax.set_title("Y5 head is fold-3 (one group)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return True


def brief_map(d: dict, p5: dict, p7: dict, p9: dict) -> list[dict]:
    return [
        {
            "#": "1",
            "question": "Who is healthy?",
            "what this cut says": "d_tx is PARK as a health Y. Do not invent `y_d_tx`.",
        },
        {
            "#": "2",
            "question": "Who is improving?",
            "what this cut says": "Not this 6m named-fill share.",
        },
        {
            "#": "3",
            "question": "Who is turning?",
            "what this cut says": (
                f"Y3 leftover after days {_f(p5['y3_days'])} — {d['y3']}. "
                "A fill twin is not a turning X."
            ),
        },
        {
            "#": "4",
            "question": "Dip vs fall?",
            "what this cut says": "Not this table.",
        },
        {
            "#": "5",
            "question": "Why did it change?",
            "what this cut says": (
                f"{d['y5_quote']}; {d['y5']}. Fold-3 hole {_pp(p7['fold3_share'])} "
                f"one-group={p7['one_group']}. Q5 sentence stays a number, not a Y3 X "
                "and not a new Family D column."
            ),
        },
        {
            "#": "6",
            "question": "Months earlier?",
            "what this cut says": (
                f"Y5 short lag1 {_f(p9['y5_short_l1'])} — {d['q6']}."
            ),
        },
    ]


def write_md(ctx: dict) -> None:
    d = ctx["decision"]
    p1, p2, p4, p5 = ctx["p1"], ctx["p2"], ctx["p4"], ctx["p5"]
    p7, p8, p9, p10 = ctx["p7"], ctx["p8"], ctx["p9"], ctx["p10"]
    p11, ph, pq = ctx["p11"], ctx["ph"], ctx["pq"]
    lines = [
        "# Unused leftover of `d_tx_cp_share` on the 44",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. No `build_targets`. Do not invent `y_d_tx`. Do not merge a Family D column. "
        "Do not put this on the 15-col Y3 card. Y7 never uses D. Y5 never E. Y3 never B. "
        f"Night Y3 **{Y3_NIGHT[0]}/{Y3_NIGHT[1]}**. Days **{DAYS_BENCH}**. Size **{SIZE_Y3_QUOTE}**. "
        f"Y7 TURNOVER **0.720 / 0.712**. Night Y5 `d_tx_cp_share` **{Y5_DTX_QUOTE}** stays a number, "
        "PARK as X. Trees stay PARK.",
        "",
        "`d_tx_cp_share` = 6-month *named* bank-CP fill (store). `miss_cp_share` = calendar-month "
        "share of txs with null/blank `counterparty_id` (in-memory; not written).",
        "",
        "## Headline",
        "",
        d["headline"],
        "",
        "## Brief questions",
        "",
        _md_table(brief_map(d, p5, p7, p9)),
        "",
        "## PARK / CLOSE / KEEP",
        "",
        _md_table(
            [
                {
                    "object": "d_tx_cp_share as Y3 X",
                    "decision": f"**{d['y3']}**",
                    "why": (
                        f"leftover after days OLS {_f(p5['y3_days'])} "
                        f"rank {_f(ctx['p_h']['y3_rank'])} vs days {_f(p4['days'])} "
                        f"size {_f(p4['size3'])}; twin ρ={_f(p2['miss_rho'])}"
                    ),
                },
                {
                    "object": "d_tx_cp_share as Y5 X",
                    "decision": f"**{d['y5']}**",
                    "why": (
                        f"train {_f(p4['y5_train'])} leftover after size {_f(p5['y5_size'])} "
                        f"after miss {_f(p5['y5_miss'])}; fold-3 one-group={p7['one_group']}"
                    ),
                },
                {
                    "object": "Night Y5 0.611 quote",
                    "decision": f"**{d['y5_quote']}**",
                    "why": f"this-run train {_f(p4['y5_train'])} CV {_f(p4['y5'])}",
                },
                {
                    "object": "d_tx as a health Y",
                    "decision": "**PARK**",
                    "why": "do not invent `y_d_tx`",
                },
                {
                    "object": "Q6 lag1/lag3",
                    "decision": f"**{d['q6']}**",
                    "why": p9["prose"],
                },
                {
                    "object": "miss_cp twin",
                    "decision": "**YES — CLOSE leftover**" if d["twin"] else "**no**",
                    "why": f"ρ={_f(p2['miss_rho'])}; leftover after miss Y3 {_f(p5['y3_miss'])} Y5 {_f(p5['y5_miss'])}",
                },
                {
                    "object": "d_tx_cp_share on the 44",
                    "decision": "**DROP from the 44**" if d["drop44"] else "**pending leftover**",
                    "why": (
                        f"Y3 {d['y3']}; Y5 {d['y5']}; KEEP-as-X gate "
                        f"beat≥0.02 / leftover / not SIZE / not twin"
                    ),
                },
                {
                    "object": "15-col Y3 card",
                    "decision": "**not added**",
                    "why": "night quote stays 0.762 / 0.752",
                },
                {
                    "object": "parquet merge / new D column",
                    "decision": "**not done**",
                    "why": "parent decides; do not invent y_d_tx",
                },
            ]
        ),
        "",
        "## 1. Coverage; 470 vs 744; invoice vs tx named",
        "",
        p1["prose"],
        "",
        _md_table(p1["rows"]),
        "",
        _md_table(p1["inv_rows"]),
        "",
        "## 2–3. Spearman twins — reproduce ρ −0.947",
        "",
        p2["prose"],
        "",
        _md_table(p2["rows"]),
        "",
        "## 4. Single-feature train group-fold AUROC",
        "",
        p4["prose"],
        "",
        "Sign from the train side of each fold. Days bar 0.711. Size `log1p(a_in3)` 0.617. Never Y7.",
        "",
        _md_table(p4["rows"]),
        "",
        "## 5–6. Honest leftover after days / size / miss_cp",
        "",
        p5["prose"],
        "",
        (
            f"Honest screen: rank leftover after days {_f(ctx['p_h']['y3_rank'])} "
            f"labeled-OLS {_f(ctx['p_h']['y3_lab'])} (panel-slope OLS {_f(p5['y3_days'])} "
            f"is not leftover). After miss_cp Y3 {_f(p5['y3_miss'])} Y5 {_f(p5['y5_miss'])} dies — twins."
        ),
        "",
        _md_table(p5["rows"]),
        "",
        "## 7. Fold 3 / one-group hole",
        "",
        p7["prose"],
        "",
        _md_table(p7["rows"]),
        "",
        "## 8. SIZE terciles",
        "",
        p8["prose"],
        "",
        _md_table(p8["rows"]),
        "",
        "## 9. Q6 lag1 / lag3 on short vs long books",
        "",
        p9["prose"],
        "",
        _md_table(p9["rows"]),
        "",
        "## 10. ICC / company-demean",
        "",
        p10["prose"],
        "",
        _md_table(p10["rows"]),
        "",
        "## 11. Dark 470: defined as 0 vs NaN",
        "",
        p11["prose"],
        "",
        _md_table(p11["rows"]),
        "",
        "## Extra — holdout coverage only",
        "",
        ph["prose"],
        "",
        _md_table(ph["rows"]),
        "",
        "## Extra — quintiles (no Y7 X)",
        "",
        pq["prose"],
        "",
        _md_table(pq["rows"]),
        "",
        "## Extra — monthly named vs 6m `d_tx_cp_share`",
        "",
        ctx["p_m6"]["prose"],
        "",
        "## Extra — rank leftover + zero dummy",
        "",
        ctx["p_rz"]["prose"],
        "",
        _md_table(ctx["p_rz"]["rows"]),
        "",
        "## Extra — invoiced-744 leftover",
        "",
        ctx["p_erp"]["prose"],
        "",
        _md_table(ctx["p_erp"]["rows"]),
        "",
        "## Extra — Y3 fold-wise leftover after days",
        "",
        ctx["p_fl"]["prose"],
        "",
        _md_table(ctx["p_fl"]["rows"]),
        "",
        "## Extra — group ICC of company-median",
        "",
        ctx["p_gicc"]["prose"],
        "",
        "## Extra — honest leftover (rank / labeled-OLS / clone)",
        "",
        ctx["p_h"]["prose"],
        "",
        _md_table(ctx["p_h"]["rows"]),
        "",
        "## Extra — leftover after Family J `j_pay_match` (in-memory)",
        "",
        ctx["p_j"]["prose"],
        "",
        _md_table(ctx["p_j"]["rows"]),
        "",
        "## Extra — drop fold 3 leftover",
        "",
        ctx["p_df3"]["prose"],
        "",
        _md_table(ctx["p_df3"]["rows"]),
        "",
        "## Extra — months-on-book",
        "",
        ctx["p_sf"]["prose"],
        "",
        _md_table(ctx["p_sf"]["rows"]),
        "",
        "## Extra — Q6 leftover after days / size",
        "",
        ctx["p_q6l"]["prose"],
        "",
        _md_table(ctx["p_q6l"]["rows"]),
        "",
        "## Extra — fold-3 hole groups (drop top name)",
        "",
        ctx["p_g3"]["prose"],
        "",
        _md_table(ctx["p_g3"]["rows"]),
        "",
        "## Extra — invoiced-744 honest leftover",
        "",
        ctx["p_744r"]["prose"],
        "",
        _md_table(ctx["p_744r"]["rows"]),
        "",
        "## Extra — zero vs intensity",
        "",
        ctx["p_part"]["prose"],
        "",
        _md_table(ctx["p_part"]["rows"]),
        "",
        "## Extra — leftover after `d_n_cust`",
        "",
        ctx["p_nc"]["prose"],
        "",
        _md_table(ctx["p_nc"]["rows"]),
        "",
        "## Extra — drop the 4 fold-3 hole groups",
        "",
        ctx["p_d4"]["prose"],
        "",
        _md_table(ctx["p_d4"]["rows"]),
        "",
        "## Extra — Δ / lag1 honest leftover",
        "",
        ctx["p_dl"]["prose"],
        "",
        _md_table(ctx["p_dl"]["rows"]),
        "",
        "## Extra — leftover after `a_n_tx`",
        "",
        ctx["p_act"]["prose"],
        "",
        _md_table(ctx["p_act"]["rows"]),
        "",
        f"Plot: `{OUT_PNG.name}`." if ctx.get("png") else "Plot: skipped.",
        "",
        "## What failed / next",
        "",
    ]
    for x in ctx["failed"]:
        lines.append(f"- {x}")
    lines.extend(
        [
            "",
            f"Elapsed {ctx['elapsed_s']:.0f}s. Cuts: coverage/470/invoice-fill, Spearman twin, "
            "singles, leftover days/size/miss, fold-3, size terciles, Q6, ICC, dark NaN-vs-0, "
            "holdout, quintiles, monthly-vs-6m, rank/zero, 744 leftover, fold leftover, group ICC, "
            "honest leftover, J leftover, drop-f3 leftover, so-far, Q6 leftover, fold-3 groups, "
            "744 rank, zero vs intensity, leftover after d_n_cust.",
            "",
            "Did **not**: merge parquet, invent `y_d_tx`, score Y7, edit counterparties.py / "
            "missing_cp_qa.py / y5_why.py, rewrite duckdb, run `build_targets`, touch `product/`, "
            "write 0–100, change night Y3 0.762/0.752 or Y5 0.611 quotes, write the parent journal / "
            "LIVE / canvas, put d_tx on the 15-col card.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD}")


def append_registry(ctx: dict) -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    prev = pd.read_csv(REGISTRY)
    key_cols = ["agent", "x_families", "y", "model", "split", "metric"]
    seen = set()
    if not prev.empty and all(c in prev.columns for c in key_cols):
        seen = set(tuple(str(r[c]) for c in key_cols) for _, r in prev.iterrows())
    d = ctx["decision"]
    p1, p2, p4, p5, p7 = ctx["p1"], ctx["p2"], ctx["p4"], ctx["p5"], ctx["p7"]
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
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
            "metric": "auroc_d_tx_cp_share",
            "value": p4["y3"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"days={p4['days']:.4f} leftover_days={p5['y3_days']:.4f} y3={d['y3']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": Y5_AR,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_d_tx_cp_share",
            "value": p4["y5"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"train={p4['y5_train']:.4f} leftover_size={p5['y5_size']:.4f} quote={d['y5_quote']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": Y5_AR,
            "model": MODEL,
            "split": "train",
            "metric": "auroc_d_tx_cp_share_train",
            "value": p4["y5_train"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"CONFIRM_0.611={p4['y5_ok']} cv={p4['y5']:.4f} drop44={d['drop44']}",
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
            "metric": "rho_d_tx_vs_miss_cp",
            "value": p2["miss_rho"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"CONFIRM_-0.947={p2['miss_ok']} uncat={p2['rhos']['a_uncat_share']:.4f} size={p2['rhos']['log1p(a_in3)']:.4f}",
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
            "metric": "auroc_d_tx_resid_days",
            "value": p5["y3_days"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"dies={p5['y3_dies']} after_miss={p5['y3_miss']:.4f} r2={p5['r2_days']:.4f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": Y5_AR,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_d_tx_resid_size",
            "value": p5["y5_size"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"dies={p5['y5_dies']} after_miss={p5['y5_miss']:.4f} fold3={p7['fold3_share']:.4f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": Y5_AR,
            "model": MODEL,
            "split": "train",
            "metric": "d_tx_fold3_hole_share",
            "value": p7["fold3_share"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"one_group={p7['one_group']} collapse={p7['collapse']} train_wo={p7['train_wo']:.4f}",
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
            "metric": "auroc_d_tx_rank_resid_days",
            "value": ctx["p_h"]["y3_rank"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"ols={ctx['p_h']['y3_ols']:.4f} clone={ctx['p_h']['y3_clone']} honest_dies={ctx['p_h']['y3_honest_dies']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": Y5_AR,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_d_tx_resid_j_pay_match",
            "value": ctx["p_j"]["y5_left"],
            "coverage": f"{ctx['p_j']['n'] / max(p1['n_cm'], 1):.4f}",
            "notes": f"rho={ctx['p_j']['rho']:.4f} y3_left={ctx['p_j']['y3_left']:.4f} dark_nn={ctx['p_j']['dark_nn']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": Y5_AR,
            "model": MODEL,
            "split": "train",
            "metric": "auroc_d_tx_drop_top_f3_group",
            "value": ctx["p_g3"]["train_drop_top"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"top={ctx['p_g3']['top']} n_g={ctx['p_g3']['n_g']} one_name={ctx['p_g3']['one_name']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": Y5_AR,
            "model": MODEL,
            "split": "train",
            "metric": "auroc_d_tx_drop_4_hole_groups",
            "value": ctx["p_d4"]["y5_train"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"groups={ctx['p_d4']['groups']} cv={ctx['p_d4']['y5_cv']:.4f} leftover_size={ctx['p_d4']['y5_size']:.4f}",
        },
    ]
    new = []
    for r in rows:
        key = tuple(str(r[c]) for c in key_cols)
        if key in seen:
            continue
        new.append(r)
        seen.add(key)
    if not new:
        print("registry: no new rows (skip-key hit)")
        return
    with REGISTRY.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prev.columns))
        for r in new:
            w.writerow({c: r.get(c, "") for c in prev.columns})
    print(f"registry appended {len(new)} rows")


def write_wave(ctx: dict) -> None:
    d = ctx["decision"]
    p2, p4, p5, p7 = ctx["p2"], ctx["p4"], ctx["p5"], ctx["p7"]
    text = (
        f"# Wave 4 — d_tx leftover\n\n"
        f"Agent `{AGENT}`. Train group-fold seed 20260918. Holdout 72 coverage only.\n\n"
        f"## Files written\n\n"
        f"- `analysis/evaluate/d_tx_qa.py`\n"
        f"- `analysis/outputs/d_tx_qa.md`\n"
        f"- `analysis/outputs/d_tx_leftover.png`\n"
        f"- append-only `analysis/experiments/registry.csv` "
        f"(skip key agent+x_families+y+model+split+metric)\n"
        f"- this note\n\n"
        f"Did not touch `counterparties.py`, `missing_cp_qa.py`, `y5_why.py`, `dso_qa.py`, "
        f"`brief_map.py`, parquet / duckdb, `build_targets`, `product/`, the 15-col card, "
        f"LIVE, CONTEXT, canvas, or the parent journal. Night Y3 stays **0.762 / 0.752**. "
        f"Days 0.711. Size 0.617. Y7 TURNOVER 0.720 / 0.712. Y5 0.611 quote "
        f"{'stands' if d['quote_stands'] else 'drifted'}.\n\n"
        f"## Locked verdict\n\n"
        f"| object | decision |\n"
        f"| --- | --- |\n"
        f"| d_tx as Y3 X | **{d['y3']}** |\n"
        f"| d_tx as Y5 X | **{d['y5']}** |\n"
        f"| Night Y5 0.611 quote | **{d['y5_quote']}** |\n"
        f"| d_tx as a health Y | **PARK** |\n"
        f"| Q6 | **{d['q6']}** |\n"
        f"| d_tx_cp_share on the 44 | **{'DROP' if d['drop44'] else 'pending'}** |\n\n"
        f"ρ vs miss_cp {_f(p2['miss_rho'])} CONFIRM twin. "
        f"Y3 leftover-days OLS {_f(p5['y3_days'])} rank {_f(ctx['p_h']['y3_rank'])} "
        f"labeled {_f(ctx['p_h']['y3_lab'])} vs days {_f(p4['days'])}. "
        f"Y5 leftover-size {_f(p5['y5_size'])} after miss {_f(p5['y5_miss'])}. "
        f"Fold-3 hole {_pp(p7['fold3_share'])} drop-fold3 train {_f(p7['train_wo'])} "
        f"drop-4-groups train {_f(ctx['p_d4']['y5_train'])} collapse={p7['collapse']}. "
        f"J leftover Y5 {_f(ctx['p_j']['y5_left'])} ρ={_f(ctx['p_j']['rho'])} (not merged).\n\n"
        f"## What failed / next\n\n"
        + "\n".join(f"- {x}" for x in ctx["failed"])
        + f"\n\nElapsed {ctx['elapsed_s']:.0f}s.\n"
    )
    WAVE_NOTE.parent.mkdir(parents=True, exist_ok=True)
    WAVE_NOTE.write_text(text, encoding="utf-8")
    print(f"wrote {WAVE_NOTE}")


def run() -> dict:
    t0 = time.time()
    print(f"d_tx_qa start seed={FOLD_SEED} holdout=72 agent={AGENT} wave={WRITE_WAVE}")
    panel = load_panel()
    con = connect()
    try:
        book = book_invoice_ids(con)
        print("in-memory miss_cp_share + invoice fill (not written)")
        monthly = load_monthly_miss(con)
        inv = load_monthly_inv_cp(con)
    finally:
        con.close()
    panel = attach_miss(panel, monthly)
    panel = attach_inv(panel, inv)
    panel["ever_erp"] = panel["company_id"].isin(book)
    panel = attach_folds(panel)
    panel = add_panel_lags(panel, ["d_tx_cp_share", "miss_cp_share"], (1, 3))
    tr = panel[panel["split"] == "train"].copy().reset_index(drop=True)
    assert_no_holdout(tr["company_id"])
    print(
        f"train CM={len(tr):,} companies={tr['company_id'].nunique()} "
        f"holdout CM={int((panel['split']=='holdout').sum())}"
    )

    print("pass 1 coverage / 470 / invoice vs tx")
    p1 = pass1_cov(tr, book)
    print("pass 2–3 twins")
    p2 = pass2_twins(tr)
    print("pass 4 singles")
    p4 = pass4_singles(tr)
    print("pass 5 leftover")
    p5 = pass5_leftover(tr)
    print("pass 7 fold 3")
    p7 = pass7_fold3(tr)
    print("pass 8 terciles")
    p8 = pass8_terciles(tr)
    print("pass 9 Q6")
    p9 = pass9_q6(tr)
    print("pass 10 ICC")
    p10 = pass10_icc(tr)
    print("pass 11 dark defined")
    p11 = pass11_dark_def(tr)
    print("extra holdout")
    ph = pass_holdout(panel)
    print("extra quintiles")
    pq = pass_quintiles(tr)
    print("extra monthly vs 6m")
    p_m6 = pass_monthly_vs_6m(tr)
    print("extra rank / zero")
    p_rz = pass_rank_zero(tr)
    print("extra 744 leftover")
    p_erp = pass_erp_leftover(tr)
    print("extra fold leftover")
    p_fl = pass_fold_leftover(tr)
    print("extra group ICC")
    p_gicc = pass_group_icc(tr)
    print("extra honest leftover")
    p_h = pass_honest(tr)
    print("extra Family J leftover (in-memory)")
    p_j = pass_j_match(tr, panel)
    print("extra drop-fold3 leftover")
    p_df3 = pass_dropf3_leftover(tr)
    print("extra so-far")
    p_sf = pass_sofar(tr)
    print("extra Q6 leftover")
    p_q6l = pass_q6_leftover(tr)
    print("extra fold-3 groups")
    p_g3 = pass_f3_groups(tr)
    print("extra 744 rank leftover")
    p_744r = pass_744_rank(tr)
    print("extra zero vs intensity")
    p_part = pass_partial(tr)
    print("extra leftover after d_n_cust")
    p_nc = pass_dn_cust(tr)
    print("extra drop 4 hole groups")
    hole_groups = [str(r["group_id"]) for r in p_g3["rows"]]
    p_d4 = pass_drop4(tr, hole_groups)
    print("extra delta / lag1 rank leftover")
    p_dl = pass_delta_lag_rank(tr)
    print("extra leftover after a_n_tx")
    p_act = pass_activity(tr)

    decision = decide(p1, p2, p4, p5, p7, p8, p9, p10, p_h)
    print(decision["headline"])
    png = make_png(tr, p7)
    failed = []
    if not p2["miss_ok"]:
        failed.append(f"ρ vs miss_cp {_f(p2['miss_rho'])} drifted from −0.947")
    if not p4["y5_ok"]:
        failed.append(f"Y5 train {_f(p4['y5_train'])} drifted from 0.611")
    if p5["y3_dies"] or p_h["y3_honest_dies"]:
        failed.append(
            f"Y3 leftover after days OLS {_f(p5['y3_days'])} rank {_f(p_h['y3_rank'])} — {decision['y3']}"
        )
    if p_h["y3_clone"]:
        failed.append(f"OLS leftover-days is a d_tx clone ρ(resid,dtx)={_f(p_h['y3_rho_x'])}")
    if p5["twin_dies"]:
        failed.append(f"leftover after miss_cp dies Y3 {_f(p5['y3_miss'])} Y5 {_f(p5['y5_miss'])}")
    if p7["one_group"] and not p7["survives"]:
        failed.append("fold-3 one-group hole CONFIRM — CLOSE as leave-one-group law")
    if decision["drop44"]:
        failed.append("DROP d_tx_cp_share from the 44")
    if p9["q6_close"]:
        failed.append(f"Q6 CLOSE short lag1 {_f(p9['y5_short_l1'])} rank leftover {_f(p_dl['y3_short_rank'])}")
    failed.append(
        f"drop 4 hole groups {p_d4['groups']}: Y5 train {_f(p_d4['y5_train'])} "
        f"(same collapse as drop fold 3)"
    )
    failed.append(
        f"Y3 leftover after J {_f(p_j['y3_left'])} Y5 {_f(p_j['y5_left'])} ρ={_f(p_j['rho'])} — do not merge J"
    )
    ctx = {
        "p1": p1,
        "p2": p2,
        "p4": p4,
        "p5": p5,
        "p7": p7,
        "p8": p8,
        "p9": p9,
        "p10": p10,
        "p11": p11,
        "ph": ph,
        "pq": pq,
        "p_m6": p_m6,
        "p_rz": p_rz,
        "p_erp": p_erp,
        "p_fl": p_fl,
        "p_gicc": p_gicc,
        "p_h": p_h,
        "p_j": p_j,
        "p_df3": p_df3,
        "p_sf": p_sf,
        "p_q6l": p_q6l,
        "p_g3": p_g3,
        "p_744r": p_744r,
        "p_part": p_part,
        "p_nc": p_nc,
        "p_d4": p_d4,
        "p_dl": p_dl,
        "p_act": p_act,
        "decision": decision,
        "png": png,
        "failed": failed,
        "elapsed_s": time.time() - t0,
        "tr": tr,
        "panel": panel,
        "book": book,
    }
    write_md(ctx)
    append_registry(ctx)
    if WRITE_WAVE:
        write_wave(ctx)
    print(f"done elapsed={ctx['elapsed_s']:.0f}s")
    return ctx


if __name__ == "__main__":
    run()
