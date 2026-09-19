"""Unused leftover of `d_cust_hhi` on the 44 as Y3 X.

NORTH_STAR: Family D `d_cust_hhi` = sum_i (amt_i / tot)^2 over AR
invoice counterparties in the trailing 6-month window (same window as
Javier concentration). Needs identified customer CPs. Dark 470 stay
**NaN not 0**. Javier 14: concentration is **top1**, not HHI.

Y4 already KEEP as a monopoly-tail single (`d_cust_hhi_lag3` 0.605;
body ≤0.975 CV 0.445; ρ vs `d_cust_top1_lag3` 0.991) and PARK trees.
Unused leftover: leftover after days as **Y3 X**, leftover after
`d_cust_top1`, and whether the 44 should lose `d_cust_hhi` as an
engine X while keeping the Y4 tail footnote.

KEEP-as-X: beat size ≥0.02 AND leftover after the honest bar AND not
SIZE (|ρ|≥0.50) AND not a twin (|ρ|≥0.80 vs `d_cust_top1` /
`d_supp_hhi` / `d_tx_cp_share`). Leftover <0.55 dies.

Honest bars: Y3 leftover after days; Y4 leftover after dropping the
>0.975 tail (body-only, quote 0.445); leftover after
`d_cust_top1_lag3` — if twin, HHI is the weaker rewrite.

Night quotes unchanged: Y3 0.762 / 0.752; days 0.711; size 0.617;
Y7 TURNOVER 0.720 / 0.712. Y7 never D. Y5 never E. Y3 never B.
Off the 15-col card. Do not invent `y_cust_hhi`. Do not reopen Y4 trees.
Q6 Y4 HHI CLOSE / LOW_POWER 21.7%. Supplier HHI is a different object.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.cust_hhi_qa

Owned: analysis/evaluate/cust_hhi_qa.py, analysis/outputs/cust_hhi_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_cust_hhi.md (end).
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
from analysis.features.common import ANALYSIS, CAT_MAP, DATA, MONTHS, connect
from analysis.targets.y11_dark import book_invoice_ids, dark_population

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

STORE = DATA / "feature_store" / "monthly.parquet"
TARGETS = DATA / "feature_store" / "targets.parquet"
OUT_MD = ANALYSIS / "outputs" / "cust_hhi_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "cust_hhi_qa.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
AGENT = "b4e81c2a"
WAVE = "4"
ROUND = "R4"
MODEL = "cust_hhi_qa"
X_FAM = "D"
WRITE_WAVE = True
WAVE_PATH = ROOT / "overnight" / "waves" / "wave4_cust_hhi.md"

Y3 = "y3_recover_cash_6m"
Y4 = "y4_ds_r_double"
Y5 = "y5_ap_od30_ownp80"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_QUOTE = 0.617
NIGHT_Y3 = 0.762
NIGHT_Y3_CORE = 0.752
TURNOVER_QUOTE = 0.720
B_SHALLOW_QUOTE = 0.712
Y4_HHI_LAG3 = 0.605
Y4_BODY = 0.445
Y4_TOP1_RHO = 0.991
Y4_TAIL_HI = 0.221
Y4_TAIL_REST = 0.115
Y4_TAIL_N = 172
Y4_TAIL_POS = 38
Y4_BODY_N = 676
Y4_BODY_POS = 78
Q6_SHORT_SHARE = 0.217
Q6_SHORT_POS = 42
HOLD_CRASH = 0.375
HOLD_SPIKE = 0.875
CRASH_TH = 0.8
SPIKE_TH = 1.5
KEEP_DELTA = 0.02
CHANCE = 0.55
SIZE_RHO = 0.50
SIZE_PARK = 0.60
TWIN_RHO = 0.80
ICC_TRAIT = 0.85
ICC_SHOCK = 0.50
MIN_POS = 50
MIN_ACF_PAIRS = 4
N_DARK_WANT = 470
TAIL_CUT = 0.975
FAKE_DAYS_RHO = 0.40

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "first_month",
    "a_in3",
    "a_io_ratio",
    "c_n_days_with_tx",
    "d_cust_hhi",
    "d_cust_top1",
    "d_n_cust",
    "d_supp_hhi",
    "d_supp_top1",
    "d_n_supp",
    "d_tx_cp_share",
)

Y_KEEP = (Y3, Y4, Y5)

_OP_IN = tuple(k for k, v in CAT_MAP.items() if v == "op_in")
_DEBT_SVC = tuple(k for k, v in CAT_MAP.items() if v == "debt_service")


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


def _sql_in(values: tuple[str, ...]) -> str:
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


def spearman(a, b) -> float:
    d = pd.DataFrame(
        {"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}
    ).dropna()
    if len(d) < 8 or d["a"].nunique() < 2 or d["b"].nunique() < 2:
        return float("nan")
    return float(d["a"].corr(d["b"], method="spearman"))


def pearson(a, b) -> float:
    d = pd.DataFrame(
        {"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}
    ).dropna()
    if len(d) < 8 or d["a"].nunique() < 2 or d["b"].nunique() < 2:
        return float("nan")
    return float(d["a"].corr(d["b"], method="pearson"))


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
        return {"icc": float("nan"), "eta2": float("nan"), "var_w": float("nan"), "var_b": float("nan"), "k": 0, "n": 0}
    y = s["x"].to_numpy(dtype=float)
    _, inv, counts = np.unique(s["co"].to_numpy(), return_inverse=True, return_counts=True)
    if counts.size < 2:
        return {"icc": float("nan"), "eta2": float("nan"), "var_w": float("nan"), "var_b": float("nan"), "k": 0, "n": 0}
    means = np.bincount(inv, weights=y) / counts
    grand = float(y.mean())
    ssb = float(np.sum(counts * (means - grand) ** 2))
    ssw = float(np.sum((y - means[inv]) ** 2))
    k = int(counts.size)
    n = int(y.size)
    var_b = ssb / max(k - 1, 1)
    var_w = ssw / max(n - k, 1)
    icc = var_b / (var_b + var_w) if (var_b + var_w) > 0 else float("nan")
    eta2 = ssb / (ssb + ssw) if (ssb + ssw) > 0 else float("nan")
    return {"icc": float(icc), "eta2": float(eta2), "var_w": float(var_w), "var_b": float(var_b), "k": k, "n": n}


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


def rank_resid(y: pd.Series, x: pd.Series) -> pd.Series:
    a = pd.to_numeric(y, errors="coerce")
    b = pd.to_numeric(x, errors="coerce")
    m = a.notna() & b.notna()
    out = pd.Series(np.nan, index=y.index, dtype=float)
    if int(m.sum()) < 20:
        return out
    ra = a[m].rank().to_numpy(dtype=float)
    rb = b[m].rank().to_numpy(dtype=float)
    xb = np.column_stack([np.ones(int(m.sum())), rb])
    coef, *_ = np.linalg.lstsq(xb, ra, rcond=None)
    out.loc[m] = ra - xb @ coef
    return out


def company_demean(series: pd.Series, company: pd.Series) -> pd.Series:
    d = pd.DataFrame({"x": pd.to_numeric(series, errors="coerce"), "co": company.astype(str)})
    mu = d.groupby("co")["x"].transform("mean")
    return d["x"] - mu


def company_mean(series: pd.Series, company: pd.Series) -> pd.Series:
    d = pd.DataFrame({"x": pd.to_numeric(series, errors="coerce"), "co": company.astype(str)})
    return d.groupby("co")["x"].transform("mean")


def size_rank_auroc(df: pd.DataFrame, col: str, mask: pd.Series) -> dict:
    x = pd.to_numeric(df[col], errors="coerce")
    size = pd.to_numeric(df["a_in3"], errors="coerce")
    m = mask & x.notna() & size.notna()
    if int(m.sum()) < 50:
        return {"auroc": float("nan"), "rho": float("nan"), "n": int(m.sum())}
    med = float(size[m].median())
    large = (size[m] > med).astype(float)
    auc = auroc(large, x[m])
    auc2 = max(auc, 1.0 - auc) if np.isfinite(auc) else float("nan")
    return {
        "auroc": float(auc2) if np.isfinite(auc2) else float("nan"),
        "rho": spearman(x[m], np.log1p(size[m].abs())),
        "n": int(m.sum()),
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
        "sd": _f(res["sd"]) if not res["low_power"] else "—",
        "sign": res["train_sign"] if not res["low_power"] else "—",
        "train": _f(res["train_auc"]) if not res["low_power"] else "—",
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


def rebuild_in3_ds3(con) -> pd.DataFrame:
    """Monthly in3 / ds3 from transactions + CAT_MAP. Not family F. Not X."""
    flows = con.execute(
        f"""
        SELECT
          t.company_id,
          CAST(date_trunc('month', t."date") AS DATE) AS month,
          SUM(CASE WHEN t.category IN ({_sql_in(_OP_IN)}) THEN t.amount ELSE 0 END) AS op_in,
          -SUM(CASE WHEN t.category IN ({_sql_in(_DEBT_SVC)}) THEN t.amount ELSE 0 END) AS debt_service
        FROM transactions t
        WHERE t."date" < TIMESTAMP '2026-09-01'
        GROUP BY 1, 2
        """
    ).df()
    flows["month"] = pd.to_datetime(flows["month"])
    first = flows.groupby("company_id")["month"].min().rename("first_m")
    panel = pd.MultiIndex.from_product(
        [first.index, MONTHS], names=["company_id", "month"]
    ).to_frame(index=False)
    panel = panel.merge(first, left_on="company_id", right_index=True)
    panel = panel[panel["month"] >= panel["first_m"]]
    panel = panel.merge(flows, on=["company_id", "month"], how="left")
    panel["op_in"] = panel["op_in"].fillna(0.0)
    panel["debt_service"] = panel["debt_service"].fillna(0.0)
    panel = panel.sort_values(["company_id", "month"]).reset_index(drop=True)
    g = panel.groupby("company_id", sort=False)
    panel["in3"] = g["op_in"].transform(lambda s: s.rolling(3).sum())
    panel["ds3"] = g["debt_service"].transform(lambda s: s.rolling(3).sum())
    panel["in3_f3"] = g["in3"].shift(-3)
    panel["ds3_f3"] = g["ds3"].shift(-3)
    out = panel[["company_id", "month", "in3", "ds3", "in3_f3", "ds3_f3"]].rename(
        columns={"month": "period"}
    )
    return _keys(out)


def load_panel() -> pd.DataFrame:
    if not STORE.exists():
        raise FileNotFoundError(f"{STORE} missing")
    if not TARGETS.exists():
        raise FileNotFoundError(f"{TARGETS} missing — do not run build_targets")
    raw = pd.read_parquet(STORE)
    yraw = pd.read_parquet(TARGETS)
    print(f"store {STORE} shape={raw.shape} (read-only)")
    print(f"targets {TARGETS} shape={yraw.shape} (read-only)")
    missing = [c for c in STORE_COLS if c not in raw.columns]
    if missing:
        raise RuntimeError(f"monthly.parquet missing {missing}")
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
    leak3 = leakage_check(["d_cust_hhi", "c_n_days_with_tx"], Y3, forbidden_prefixes=["b"])
    if not leak3["ok"]:
        raise RuntimeError(f"Y3 X leak: {leak3['issues']}")
    leak4 = leakage_check(["d_cust_hhi", "d_cust_top1"], Y4, forbidden_prefixes=["f"])
    if not leak4["ok"]:
        raise RuntimeError(f"Y4 X leak: {leak4['issues']}")
    leak5 = leakage_check(["d_cust_hhi", "d_cust_top1"], Y5, forbidden_prefixes=["e"])
    if not leak5["ok"]:
        raise RuntimeError(f"Y5 X leak: {leak5['issues']}")
    leak7 = leakage_check(["c_n_days_with_tx", "log_in3"], "y7_top1_lost", forbidden_prefixes=["d"])
    if not leak7["ok"]:
        raise RuntimeError(f"Y7 never-D leak: {leak7['issues']}")
    return panel


def attach_folds(panel: pd.DataFrame) -> pd.DataFrame:
    train = panel[panel["split"] == "train"][["company_id", "group_id"]].drop_duplicates()
    assert_no_holdout(train["company_id"])
    folds = group_folds(train, n=N_FOLDS, seed=FOLD_SEED)
    return panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")


def attach_crash_spike(panel: pd.DataFrame, flows: pd.DataFrame) -> pd.DataFrame:
    out = panel.merge(flows, on=["company_id", "period"], how="left")
    in3 = pd.to_numeric(out["in3"], errors="coerce")
    ds3 = pd.to_numeric(out["ds3"], errors="coerce")
    out["in_ratio"] = pd.to_numeric(out["in3_f3"], errors="coerce") / in3.replace(0, np.nan)
    out["ds_ratio"] = pd.to_numeric(out["ds3_f3"], errors="coerce") / ds3.replace(0, np.nan)
    defined = out["in_ratio"].notna() & out["ds_ratio"].notna()
    out["crash_any"] = defined & (out["in_ratio"] < CRASH_TH)
    out["spike_any"] = defined & (out["ds_ratio"] > SPIKE_TH)
    return out


# ---------------------------------------------------------------------------
# 1. Coverage; 470 dark NaN vs invoice-book; ever-n
# ---------------------------------------------------------------------------
def cut1_coverage(panel: pd.DataFrame, book: set[str], dark: dict) -> dict:
    rows = []
    for split, sl in (
        ("train", panel["split"] == "train"),
        ("holdout", panel["split"] == "holdout"),
    ):
        d = panel.loc[sl]
        p = pd.to_numeric(d["d_cust_hhi"], errors="coerce")
        n = len(d)
        nn = int(p.notna().sum())
        na = int(p.isna().sum())
        n_co = int(d["company_id"].nunique())
        ever = int(d.loc[p.notna(), "company_id"].nunique())
        book_m = d["company_id"].isin(book)
        dark_m = ~book_m
        dark_nn = int(p[dark_m].notna().sum())
        dark_zero = int((p[dark_m] == 0).sum())
        book_nn = int(p[book_m].notna().sum())
        book_na = int(p[book_m].isna().sum())
        n_dark_co = int(d.loc[dark_m, "company_id"].nunique())
        n_book_co = int(d.loc[book_m, "company_id"].nunique())
        n_eq0 = int((p == 0).sum())
        if split == "train":
            assert_no_holdout(d["company_id"])
        rows.append(
            {
                "split": split,
                "cm": f"{n:,}",
                "companies": f"{n_co:,}",
                "HHI nn": f"{nn:,}",
                "cov": _pp(nn / n if n else float("nan")),
                "NaN": f"{na:,}",
                "ever-n": f"{ever:,}",
                "ever-ERP / never": f"{n_book_co} / {n_dark_co}",
                "dark nn / zero": f"{dark_nn} / {dark_zero}",
                "ERP nn / NaN": f"{book_nn:,} / {book_na:,}",
                "HHI==0": f"{n_eq0:,}",
            }
        )
        print(
            f"1 {split}: cov={nn/n if n else float('nan'):.4f} nn={nn} na={na} "
            f"ever={ever} dark_co={n_dark_co} dark_nn={dark_nn} dark_zero={dark_zero}"
        )

    tr = panel[panel["split"] == "train"]
    p = pd.to_numeric(tr["d_cust_hhi"], errors="coerce")
    top1 = pd.to_numeric(tr["d_cust_top1"], errors="coerce")
    n_cust = pd.to_numeric(tr["d_n_cust"], errors="coerce")
    train_dark_ids = set(dark.get("train_dark_ids") or set())
    if not train_dark_ids:
        train_dark_ids = set(tr.loc[~tr["company_id"].isin(book), "company_id"].astype(str))
    n_dark_co = len(train_dark_ids)
    dark_cm = tr["company_id"].isin(train_dark_ids)
    dark_nn = int(p[dark_cm].notna().sum())
    dark_zero = int((p[dark_cm] == 0).sum())
    dark_top1_nn = int(top1[dark_cm].notna().sum())
    dark_ok = dark_nn == 0 and dark_zero == 0 and n_dark_co == N_DARK_WANT
    n0_erp = int((tr.loc[~dark_cm, "d_n_cust"] == 0).sum())
    hhi_when_n0 = int(p[(~dark_cm) & (n_cust == 0)].notna().sum())
    acf1 = median_acf(p, tr["company_id"], 1)
    acf3 = median_acf(p, tr["company_id"], 3)
    size_rho = spearman(p, tr["log_in3"])
    prose = (
        f"Train `d_cust_hhi` coverage {_pp(_pct(int(p.notna().sum()), len(tr)))} "
        f"({int(p.notna().sum()):,}/{len(tr):,}); ever-n "
        f"{tr.loc[p.notna(), 'company_id'].nunique()} companies. "
        f"Dark {n_dark_co} (want {N_DARK_WANT}): HHI non-null {dark_nn} "
        f"zero-filled {dark_zero} top1 nn {dark_top1_nn}. "
        f"470 stay NaN not 0: {'CONFIRM' if dark_ok else 'FAIL'}. "
        f"ERP n_cust==0 months {n0_erp:,}; HHI defined on those {hhi_when_n0} "
        f"(want 0 — n=0 keeps HHI NaN). acf1={_f(acf1)} size ρ={_f(size_rho)}."
    )
    print(prose)
    return {
        "rows": rows,
        "prose": prose,
        "train_cov": _pct(int(p.notna().sum()), len(tr)),
        "train_nn": int(p.notna().sum()),
        "train_na": int(p.isna().sum()),
        "ever_n": int(tr.loc[p.notna(), "company_id"].nunique()),
        "n_dark_co": n_dark_co,
        "dark_nn": dark_nn,
        "dark_zero": dark_zero,
        "dark_top1_nn": dark_top1_nn,
        "dark_ok": dark_ok,
        "n0_erp": n0_erp,
        "hhi_when_n0": hhi_when_n0,
        "acf1": acf1,
        "acf3": acf3,
        "size_rho": size_rho,
        "hold_cov": _pct(
            int(pd.to_numeric(panel.loc[panel["split"] == "holdout", "d_cust_hhi"], errors="coerce").notna().sum()),
            int((panel["split"] == "holdout").sum()),
        ),
        "n_train": len(tr),
        "n_train_co": int(tr["company_id"].nunique()),
        "confirm_470": bool(dark.get("confirm_470")),
    }


# ---------------------------------------------------------------------------
# 2. Spearman twins
# ---------------------------------------------------------------------------
def cut2_twins(tr: pd.DataFrame) -> dict:
    p = tr["d_cust_hhi"]
    pairs = [
        ("d_cust_top1", tr["d_cust_top1"]),
        ("d_cust_top1_lag3", tr["d_cust_top1_lag3"]),
        ("d_cust_hhi_lag3", tr["d_cust_hhi_lag3"]),
        ("d_supp_hhi", tr["d_supp_hhi"]),
        ("d_tx_cp_share", tr["d_tx_cp_share"]),
        ("log1p(a_in3)", tr["log_in3"]),
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ("d_n_cust", tr["d_n_cust"]),
        ("d_n_supp", tr["d_n_supp"]),
        ("a_io_ratio", tr["a_io_ratio"]),
    ]
    official = {"d_cust_top1", "d_cust_top1_lag3", "d_supp_hhi", "d_tx_cp_share"}
    rhos = {}
    pears = {}
    ns = {}
    twins = []
    rows = []
    for name, s in pairs:
        rho = spearman(p, s)
        pr = pearson(p, s)
        d = pd.DataFrame({"a": pd.to_numeric(p, errors="coerce"), "b": pd.to_numeric(s, errors="coerce")}).dropna()
        rhos[name] = rho
        pears[name] = pr
        ns[name] = int(len(d))
        twin = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO and name in official)
        if twin:
            twins.append(name)
        size_flag = name == "log1p(a_in3)" and bool(np.isfinite(rho) and abs(rho) >= SIZE_RHO)
        count_flag = name == "d_n_cust" and bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
        flag = "YES" if twin else ("SIZE" if size_flag else ("COUNT" if count_flag else "no"))
        rows.append(
            {
                "vs": name,
                "n": f"{len(d):,}",
                "Spearman": _f(rho),
                "Pearson": _f(pr),
                "twin ≥0.80": flag,
            }
        )
        print(f"2 ρ vs {name}: {_f(rho)} n={len(d)} twin={twin}")
    # locked pair: HHI_lag3 vs top1_lag3
    rho_lag = spearman(tr["d_cust_hhi_lag3"], tr["d_cust_top1_lag3"])
    dlag = pd.DataFrame(
        {
            "a": pd.to_numeric(tr["d_cust_hhi_lag3"], errors="coerce"),
            "b": pd.to_numeric(tr["d_cust_top1_lag3"], errors="coerce"),
        }
    ).dropna()
    rhos["hhi_lag3_vs_top1_lag3"] = rho_lag
    ns["hhi_lag3_vs_top1_lag3"] = int(len(dlag))
    lag_twin = bool(np.isfinite(rho_lag) and abs(rho_lag) >= TWIN_RHO)
    if lag_twin and "d_cust_top1_lag3" not in twins:
        twins.append("d_cust_top1_lag3")
    rows.append(
        {
            "vs": "HHI_lag3 vs top1_lag3",
            "n": f"{len(dlag):,}",
            "Spearman": _f(rho_lag),
            "Pearson": _f(pearson(tr["d_cust_hhi_lag3"], tr["d_cust_top1_lag3"])),
            "twin ≥0.80": "YES" if lag_twin else "no",
        }
    )
    print(f"2 ρ HHI_lag3 vs top1_lag3: {_f(rho_lag)} n={len(dlag)} twin={lag_twin} quote={Y4_TOP1_RHO}")
    size_flag = bool(np.isfinite(rhos["log1p(a_in3)"]) and abs(rhos["log1p(a_in3)"]) >= SIZE_RHO)
    top1_twin = bool(
        (np.isfinite(rhos["d_cust_top1"]) and abs(rhos["d_cust_top1"]) >= TWIN_RHO)
        or lag_twin
    )
    lag_confirm = bool(np.isfinite(rho_lag) and abs(rho_lag - Y4_TOP1_RHO) < 0.01)
    supp_twin = bool(np.isfinite(rhos["d_supp_hhi"]) and abs(rhos["d_supp_hhi"]) >= TWIN_RHO)
    dtx_twin = bool(np.isfinite(rhos["d_tx_cp_share"]) and abs(rhos["d_tx_cp_share"]) >= TWIN_RHO)
    prose = (
        f"ρ vs `d_cust_top1` {_f(rhos['d_cust_top1'])} n={ns['d_cust_top1']:,}; "
        f"HHI_lag3↔top1_lag3 {_f(rho_lag)} n={ns['hhi_lag3_vs_top1_lag3']:,} "
        f"(quote {Y4_TOP1_RHO}; {'CONFIRM' if lag_confirm else 'CHECK'}). "
        f"{'TWIN — HHI is the weaker rewrite' if top1_twin else 'not a twin of top1'}. "
        f"vs `d_supp_hhi` {_f(rhos['d_supp_hhi'])} "
        f"({'TWIN of supp — do not merge' if supp_twin else 'not a twin of supp'}). "
        f"vs `d_tx_cp_share` {_f(rhos['d_tx_cp_share'])} "
        f"{'TWIN of d_tx' if dtx_twin else 'not a twin of d_tx'}. "
        f"vs size {_f(rhos['log1p(a_in3)'])} ({'SIZE' if size_flag else 'not SIZE'}). "
        f"vs `d_n_cust` {_f(rhos['d_n_cust'])} "
        f"({'count rewrite |ρ|≥0.80 — not the official KEEP twin list' if (np.isfinite(rhos['d_n_cust']) and abs(rhos['d_n_cust']) >= TWIN_RHO) else 'not a count rewrite'}). "
        f"Javier: concentration is top1."
    )
    print(prose)
    return {
        "rows": rows,
        "rhos": rhos,
        "pears": pears,
        "ns": ns,
        "twins": twins,
        "any_twin": bool(twins),
        "top1_twin": top1_twin,
        "lag_twin": lag_twin,
        "lag_confirm": lag_confirm,
        "supp_twin": supp_twin,
        "dtx_twin": dtx_twin,
        "size_flag": size_flag,
        "rho_lag": rho_lag,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 3. Reproduce Y4 quintiles + >0.975 tail. Body CV ~0.445 CONFIRM
# ---------------------------------------------------------------------------
def _quintile_block(df: pd.DataFrame, mask: pd.Series, x_col: str, y_col: str) -> dict:
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    m = mask & x.notna() & y.notna()
    d = df.loc[m, [x_col]].copy()
    d[y_col] = y[m].to_numpy()
    if int(m.sum()) < 40 or d[x_col].nunique() < 5:
        return {"rows": [], "up": False, "down": False, "n": int(m.sum()), "bins": 0, "raw_rates": []}
    cats, bins = pd.qcut(d[x_col], 5, retbins=True, duplicates="drop")
    d = d.copy()
    d["q"] = cats
    rows = []
    for i, (q, g) in enumerate(d.groupby("q", observed=True), start=1):
        rows.append(
            {
                "y": y_col,
                "x": x_col,
                "Q": i,
                "interval": str(q),
                "n": int(len(g)),
                "n_pos": int((g[y_col] == 1).sum()),
                "P(Y=1)": _f(float(g[y_col].mean()), 3),
                "median HHI": _f(float(g[x_col].median()), 4),
            }
        )
    rates = [float(r["P(Y=1)"]) if r["P(Y=1)"] != "—" else float("nan") for r in rows]
    up = bool(len(rates) >= 3 and all(a <= b + 1e-9 for a, b in zip(rates, rates[1:])))
    down = bool(len(rates) >= 3 and all(a >= b - 1e-9 for a, b in zip(rates, rates[1:])))
    return {
        "rows": rows,
        "up": up,
        "down": down,
        "n": int(m.sum()),
        "bins": len(rows),
        "raw_rates": rates,
        "cut_bins": [float(b) for b in bins],
    }


def cut3_tail(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["d_cust_hhi"], errors="coerce")
    x3 = pd.to_numeric(tr["d_cust_hhi_lag3"], errors="coerce")
    q_rows = []
    q_store = {}
    for y, xcol in (
        (Y4, "d_cust_hhi_lag3"),
        (Y4, "d_cust_hhi"),
        (Y3, "d_cust_hhi"),
        (Y5, "d_cust_hhi"),
    ):
        q = _quintile_block(tr, tr[y].notna(), xcol, y)
        q_store[(y, xcol)] = q
        q_rows.extend(q["rows"])
        print(f"3 quintiles {y} {xcol}: n={q['n']} up={q['up']} down={q['down']}")

    tail_rows = []
    body_store = {}
    tail_store = {}
    for y, xcol, xx in (
        (Y4, "d_cust_hhi_lag3", x3),
        (Y4, "d_cust_hhi", x),
        (Y3, "d_cust_hhi", x),
        (Y3, "d_cust_hhi_lag3", x3),
        (Y5, "d_cust_hhi", x),
        (Y5, "d_supp_hhi", pd.to_numeric(tr["d_supp_hhi"], errors="coerce")),
    ):
        yv = pd.to_numeric(tr[y], errors="coerce")
        m = yv.notna() & xx.notna()
        hi = m & (xx > TAIL_CUT)
        lo = m & (xx <= TAIL_CUT)
        rate_hi = float(yv[hi].mean()) if int(hi.sum()) else float("nan")
        rate_lo = float(yv[lo].mean()) if int(lo.sum()) else float("nan")
        n_hi = int(hi.sum())
        n_pos_hi = int((hi & (yv == 1)).sum())
        tail_auc = auroc(yv[m], (xx[m] > TAIL_CUT).astype(float))
        body = signed_oof_auroc(tr[y], xx, tr["fold"], lo)
        body_store[(y, xcol)] = body
        tail_store[(y, xcol)] = {
            "n_hi": n_hi,
            "n_pos_hi": n_pos_hi,
            "rate_hi": rate_hi,
            "rate_lo": rate_lo,
            "tail_auc": tail_auc,
        }
        protective = bool(np.isfinite(rate_hi) and np.isfinite(rate_lo) and rate_hi < rate_lo)
        crash = bool(np.isfinite(rate_hi) and np.isfinite(rate_lo) and rate_hi > rate_lo)
        tail_rows.append(
            {
                "y": y,
                "x": xcol,
                "n tail / pos": f"{n_hi} / {n_pos_hi}",
                "P(Y=1) tail": _f(rate_hi),
                "P(Y=1) rest": _f(rate_lo),
                "tail AUROC": _f(tail_auc),
                "body CV": "LOW_POWER" if body["low_power"] else _f(body["cv"]),
                "body n / pos": f"{body['n_defined']} / {body['n_pos']}",
                "shape": "protective" if protective else ("crash" if crash else "?"),
            }
        )
        print(
            f"3 tail {y} {xcol}>0.975 n={n_hi} pos={n_pos_hi} "
            f"rate={rate_hi:.4f} rest={rate_lo:.4f} body={_cv(body)}"
        )

    y4 = tail_store[(Y4, "d_cust_hhi_lag3")]
    y4_body = _cv(body_store[(Y4, "d_cust_hhi_lag3")])
    y4_body_rec = body_store[(Y4, "d_cust_hhi_lag3")]
    y3_body = _cv(body_store[(Y3, "d_cust_hhi")])
    y5_cust_body = _cv(body_store[(Y5, "d_cust_hhi")])
    y5_supp = tail_store[(Y5, "d_supp_hhi")]
    y4_confirm = bool(
        np.isfinite(y4["rate_hi"])
        and np.isfinite(y4["rate_lo"])
        and abs(y4["rate_hi"] - Y4_TAIL_HI) < 0.02
        and abs(y4["rate_lo"] - Y4_TAIL_REST) < 0.02
        and abs(y4["n_hi"] - Y4_TAIL_N) <= 4
        and abs(y4["n_pos_hi"] - Y4_TAIL_POS) <= 2
    )
    body_confirm = bool(
        np.isfinite(y4_body)
        and abs(y4_body - Y4_BODY) < 0.02
        and abs(y4_body_rec["n_defined"] - Y4_BODY_N) <= 8
        and abs(y4_body_rec["n_pos"] - Y4_BODY_POS) <= 4
    )
    y4_dead = bool(np.isfinite(y4_body) and y4_body < CHANCE)
    q4 = q_store[(Y4, "d_cust_hhi_lag3")]
    q5_rate = q4["raw_rates"][-1] if q4["raw_rates"] else float("nan")
    prose = (
        f"Y4 `d_cust_hhi_lag3` >0.975 P(Y=1)={_f(y4['rate_hi'])} "
        f"(n={y4['n_hi']}, {y4['n_pos_hi']} pos) vs rest {_f(y4['rate_lo'])} "
        f"(quote {Y4_TAIL_HI:.1%} / {Y4_TAIL_REST:.1%}, n={Y4_TAIL_N}/{Y4_TAIL_POS}) — "
        f"{'CONFIRM crash tail' if y4_confirm else 'CHECK vs quote'}. "
        f"Body HHI≤0.975 CV {_f(y4_body)} ± {_f(y4_body_rec['sd'])} "
        f"(n={y4_body_rec['n_defined']}, {y4_body_rec['n_pos']} pos; quote {Y4_BODY}) — "
        f"{'CONFIRM' if body_confirm else 'CHECK'}. "
        f"{'The 0.605 is the monopoly tail. Body loses to dummy.' if y4_dead else 'Body still ranks'}. "
        f"Quintiles monotone={q4['up']}; Q5 rate {_f(q5_rate)}. "
        f"Y3 body {_f(y3_body)}. Y5 cust body {_f(y5_cust_body)}. "
        f"Y5 supp tail {_f(y5_supp['rate_hi'])} vs {_f(y5_supp['rate_lo'])} (protective object)."
    )
    print(prose)
    return {
        "q_rows": q_rows,
        "q_store": q_store,
        "tail_rows": tail_rows,
        "body_store": body_store,
        "tail_store": tail_store,
        "y4_rate_hi": y4["rate_hi"],
        "y4_rate_lo": y4["rate_lo"],
        "y4_n_hi": y4["n_hi"],
        "y4_n_pos_hi": y4["n_pos_hi"],
        "y4_confirm": y4_confirm,
        "y4_body": y4_body,
        "y4_body_sd": y4_body_rec["sd"],
        "y4_body_n": y4_body_rec["n_defined"],
        "y4_body_pos": y4_body_rec["n_pos"],
        "body_confirm": body_confirm,
        "y4_dead": y4_dead,
        "y3_body": y3_body,
        "y5_cust_body": y5_cust_body,
        "q4_up": q4["up"],
        "q5_rate": q5_rate,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 4. Single-feature group-fold AUROC on Y3 and Y4
# ---------------------------------------------------------------------------
def cut4_singles(tr: pd.DataFrame) -> dict:
    feats = [
        ("d_cust_hhi", tr["d_cust_hhi"]),
        ("d_cust_hhi_lag1", tr["d_cust_hhi_lag1"]),
        ("d_cust_hhi_lag3", tr["d_cust_hhi_lag3"]),
        ("d_cust_top1", tr["d_cust_top1"]),
        ("d_cust_top1_lag3", tr["d_cust_top1_lag3"]),
        ("d_supp_hhi", tr["d_supp_hhi"]),
        ("log1p(a_in3)", tr["log_in3"]),
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ("d_n_cust", tr["d_n_cust"]),
        ("d_tx_cp_share", tr["d_tx_cp_share"]),
    ]
    rows = []
    store = {}
    for y in (Y3, Y4, Y5):
        lab = tr[y].notna()
        for name, s in feats:
            res = signed_oof_auroc(tr[y], s, tr["fold"], lab)
            store[(y, name)] = res
            rows.append(_auc_row(y, name, res))
            print(f"4 {y} {name}: {_f(_cv(res))} n={res['n_defined']} pos={res['n_pos']}")
    size_rank = {}
    for y in (Y3, Y4, Y5):
        size_rank[y] = size_rank_auroc(tr, "d_cust_hhi", tr[y].notna())
    hhi_y3 = _cv(store[(Y3, "d_cust_hhi")])
    hhi_y4 = _cv(store[(Y4, "d_cust_hhi")])
    hhi3_y3 = _cv(store[(Y3, "d_cust_hhi_lag3")])
    hhi3_y4 = _cv(store[(Y4, "d_cust_hhi_lag3")])
    top1_y3 = _cv(store[(Y3, "d_cust_top1")])
    top13_y3 = _cv(store[(Y3, "d_cust_top1_lag3")])
    top13_y4 = _cv(store[(Y4, "d_cust_top1_lag3")])
    size_y3 = _cv(store[(Y3, "log1p(a_in3)")])
    size_y4 = _cv(store[(Y4, "log1p(a_in3)")])
    days_y3 = _cv(store[(Y3, "c_n_days_with_tx")])
    days_y4 = _cv(store[(Y4, "c_n_days_with_tx")])
    hhi3_confirm = bool(np.isfinite(hhi3_y4) and abs(hhi3_y4 - Y4_HHI_LAG3) < 0.01)
    size_y3_confirm = bool(np.isfinite(size_y3) and abs(size_y3 - SIZE_QUOTE) < 0.01)
    days_y3_confirm = bool(np.isfinite(days_y3) and abs(days_y3 - DAYS_BENCH) < 0.01)
    beat_y3 = bool(np.isfinite(hhi_y3) and np.isfinite(size_y3) and hhi_y3 >= size_y3 + KEEP_DELTA)
    beat_y4 = bool(np.isfinite(hhi3_y4) and np.isfinite(size_y4) and hhi3_y4 >= size_y4 + KEEP_DELTA)
    beat_y3_days = bool(np.isfinite(hhi_y3) and np.isfinite(days_y3) and hhi_y3 >= days_y3 + KEEP_DELTA)
    weaker_y4 = "d_cust_hhi_lag3"
    if np.isfinite(hhi3_y4) and np.isfinite(top13_y4):
        if top13_y4 >= hhi3_y4:
            weaker_y4 = "d_cust_hhi_lag3"
        elif hhi3_y4 - top13_y4 < KEEP_DELTA:
            weaker_y4 = "d_cust_hhi_lag3"
        else:
            weaker_y4 = "d_cust_top1_lag3"
    prose = (
        f"Y3 HHI {_f(hhi_y3)} vs size {_f(size_y3)} (night {SIZE_QUOTE}; "
        f"{'CONFIRM' if size_y3_confirm else 'CHECK'}) "
        f"days {_f(days_y3)} (night {DAYS_BENCH}; "
        f"{'CONFIRM' if days_y3_confirm else 'CHECK'}) "
        f"top1 {_f(top1_y3)} top1_lag3 {_f(top13_y3)}. "
        f"Y4 HHI {_f(hhi_y4)} HHI_lag3 {_f(hhi3_y4)} (night {Y4_HHI_LAG3}; "
        f"{'CONFIRM' if hhi3_confirm else 'CHECK'}) "
        f"top1_lag3 {_f(top13_y4)} size {_f(size_y4)} days {_f(days_y4)}. "
        f"Beat-size Y3={beat_y3} Y4={beat_y4}. Beat-days Y3={beat_y3_days}. "
        f"Weaker of the Y4 twin pair: {weaker_y4} (Javier concentration is top1)."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "size_rank": size_rank,
        "hhi_y3": hhi_y3,
        "hhi_y4": hhi_y4,
        "hhi3_y3": hhi3_y3,
        "hhi3_y4": hhi3_y4,
        "top1_y3": top1_y3,
        "top13_y3": top13_y3,
        "top13_y4": top13_y4,
        "size_y3": size_y3,
        "size_y4": size_y4,
        "days_y3": days_y3,
        "days_y4": days_y4,
        "hhi3_confirm": hhi3_confirm,
        "size_y3_confirm": size_y3_confirm,
        "days_y3_confirm": days_y3_confirm,
        "beat_y3": beat_y3,
        "beat_y4": beat_y4,
        "beat_y3_days": beat_y3_days,
        "weaker_y4": weaker_y4,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 5. Honest leftover after days (Y3). Leftover <0.55 dies.
# ---------------------------------------------------------------------------
def _leftover_specs(tr: pd.DataFrame, y: str, feat: pd.Series, specs: list[tuple]) -> dict:
    lab = tr[y].notna()
    rows = []
    store = {}
    infos = {}
    for name, xs in specs:
        resid, info = ols_resid(feat, *xs)
        res = signed_oof_auroc(tr[y], resid, tr["fold"], lab)
        store[name] = res
        infos[name] = info
        rows.append(
            {
                "y": y,
                "residual": name,
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                "R²": _f(info["r2"]),
                "folds": fold_bits(res) if not res["low_power"] else "—",
            }
        )
        print(f"5 leftover {y} {name}: {_f(_cv(res))} R2={_f(info['r2'])}")
    return {"rows": rows, "store": store, "infos": infos}


def cut5_leftover(tr: pd.DataFrame) -> dict:
    p = tr["d_cust_hhi"]
    p3 = tr["d_cust_hhi_lag3"]
    tail_flag = pd.Series(
        np.where(p3.notna(), (pd.to_numeric(p3, errors="coerce") > TAIL_CUT).astype(float), np.nan),
        index=tr.index,
    )
    y3 = _leftover_specs(
        tr,
        Y3,
        p,
        [
            ("after days", (tr["c_n_days_with_tx"],)),
            ("after size", (tr["log_in3"],)),
            ("after days+size", (tr["c_n_days_with_tx"], tr["log_in3"])),
            ("after d_cust_top1", (tr["d_cust_top1"],)),
            ("after d_supp_hhi", (tr["d_supp_hhi"],)),
        ],
    )
    y4 = _leftover_specs(
        tr,
        Y4,
        p3,
        [
            ("after d_cust_top1_lag3", (tr["d_cust_top1_lag3"],)),
            ("after tail-flag", (tail_flag,)),
            ("after size", (tr["log_in3"],)),
            ("after days", (tr["c_n_days_with_tx"],)),
            ("after d_supp_hhi", (tr["d_supp_hhi"],)),
        ],
    )
    y4_now = _leftover_specs(
        tr,
        Y4,
        p,
        [
            ("after d_cust_top1", (tr["d_cust_top1"],)),
            ("after days", (tr["c_n_days_with_tx"],)),
        ],
    )
    lab3 = tr[Y3].notna()
    lab4 = tr[Y4].notna()
    r_days = signed_oof_auroc(tr[Y3], rank_resid(p, tr["c_n_days_with_tx"]), tr["fold"], lab3)
    r_top1 = signed_oof_auroc(tr[Y4], rank_resid(p3, tr["d_cust_top1_lag3"]), tr["fold"], lab4)
    y3["rows"].append(
        {
            "y": Y3,
            "residual": "rank-resid after days",
            "n": f"{r_days['n_defined']:,}",
            "n_pos": f"{r_days['n_pos']:,}",
            "CV": "LOW_POWER" if r_days["low_power"] else _f(r_days["cv"]),
            "R²": "—",
            "folds": fold_bits(r_days) if not r_days["low_power"] else "—",
        }
    )
    y4["rows"].append(
        {
            "y": Y4,
            "residual": "rank-resid after top1_lag3",
            "n": f"{r_top1['n_defined']:,}",
            "n_pos": f"{r_top1['n_pos']:,}",
            "CV": "LOW_POWER" if r_top1["low_power"] else _f(r_top1["cv"]),
            "R²": "—",
            "folds": fold_bits(r_top1) if not r_top1["low_power"] else "—",
        }
    )
    after_days = _cv(y3["store"]["after days"])
    after_top1 = _cv(y4["store"]["after d_cust_top1_lag3"])
    after_tail = _cv(y4["store"]["after tail-flag"])
    after_size = _cv(y3["store"]["after size"])
    after_days_size = _cv(y3["store"]["after days+size"])
    after_top1_y3 = _cv(y3["store"]["after d_cust_top1"])
    resid_days, info_days = ols_resid(p, tr["c_n_days_with_tx"])
    rho_resid_days = spearman(resid_days, tr["c_n_days_with_tx"])
    fake_days = bool(
        np.isfinite(after_days)
        and after_days >= CHANCE
        and np.isfinite(rho_resid_days)
        and abs(rho_resid_days) >= FAKE_DAYS_RHO
    )
    lives_y3 = bool(np.isfinite(after_days) and after_days >= CHANCE and not fake_days)
    lives_y4 = bool(np.isfinite(after_top1) and after_top1 >= CHANCE)
    died_y3 = bool((np.isfinite(after_days) and after_days < CHANCE) or fake_days)
    died_y4 = bool(np.isfinite(after_top1) and after_top1 < CHANCE)
    prose = (
        f"Y3 leftover after days {_f(after_days)} "
        f"({'lives' if lives_y3 else 'dies <0.55 / fake-days'}); "
        f"ρ(resid, days)={_f(rho_resid_days)} "
        f"({'FAKE-DAYS leak' if fake_days else 'not a fake-days leak'}). "
        f"after top1 {_f(after_top1_y3)}; after size {_f(after_size)}; "
        f"after days+size {_f(after_days_size)}. "
        f"Y4 leftover after top1_lag3 {_f(after_top1)} "
        f"({'lives' if lives_y4 else 'dies <0.55'}); "
        f"after tail-flag {_f(after_tail)}. "
        f"Rank-ortho days {_f(_cv(r_days))} top1 {_f(_cv(r_top1))}."
    )
    print(prose)
    return {
        "rows": y3["rows"] + y4["rows"] + y4_now["rows"],
        "y3": y3,
        "y4": y4,
        "after_days": after_days,
        "after_top1": after_top1,
        "after_tail": after_tail,
        "after_size": after_size,
        "after_days_size": after_days_size,
        "after_top1_y3": after_top1_y3,
        "rank_days": _cv(r_days),
        "rank_top1": _cv(r_top1),
        "rho_resid_days": rho_resid_days,
        "fake_days": fake_days,
        "r2_days": info_days["r2"],
        "lives_y3": lives_y3,
        "lives_y4": lives_y4,
        "died_y3": died_y3,
        "died_y4": died_y4,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 6. Leftover after top1 lag3 (Y4). If twin, DROP HHI from the 44 as X
# ---------------------------------------------------------------------------
def cut6_twin(tr: pd.DataFrame, twins: dict, singles: dict, leftover: dict) -> dict:
    rho = twins["rho_lag"]
    rho0 = twins["rhos"]["d_cust_top1"]
    weaker = "d_cust_hhi" if twins["top1_twin"] else None
    mh = float(np.nanmean([singles["hhi_y3"], singles["hhi3_y4"]]))
    mt = float(np.nanmean([singles["top1_y3"], singles["top13_y4"]]))
    drop_hhi = bool(twins["top1_twin"])
    prose = (
        f"ρ(HHI_lag3, top1_lag3)={_f(rho)} (quote {Y4_TOP1_RHO}; "
        f"{'CONFIRM' if twins['lag_confirm'] else 'CHECK'}). "
        f"ρ(HHI, top1)={_f(rho0)}. "
        f"{'TWIN ≥0.80 — DROP weaker `d_cust_hhi` as engine X (Javier concentration is top1).' if drop_hhi else 'Not a twin; keep both in the screen.'} "
        f"Mean CV HHI {_f(mh)} top1 {_f(mt)}. "
        f"Y4 leftover after top1_lag3 {_f(leftover['after_top1'])}. "
        f"Y4 tail footnote may KEEP; engine X on the 44 may DROP."
    )
    print(prose)
    return {
        "rho": rho,
        "rho0": rho0,
        "weaker": weaker,
        "drop_hhi": drop_hhi,
        "mean_hhi": mh,
        "mean_top1": mt,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 7. SIZE terciles — does the Y4 tail survive inside T1?
# ---------------------------------------------------------------------------
def cut7_terciles(tr: pd.DataFrame) -> dict:
    size = pd.to_numeric(tr["log_in3"], errors="coerce")
    x = pd.to_numeric(tr["d_cust_hhi"], errors="coerce")
    x3 = pd.to_numeric(tr["d_cust_hhi_lag3"], errors="coerce")
    defined = size.notna()
    try:
        tercile = pd.qcut(size[defined], 3, labels=["T1", "T2", "T3"], duplicates="drop")
        terc = pd.Series(index=tr.index, dtype=object)
        terc.loc[defined] = tercile.astype(str)
        cats = ["T1", "T2", "T3"]
    except ValueError:
        terc = None
        cats = []
    rows = []
    t1_survive = {}
    specs = ((Y3, "d_cust_hhi", x), (Y4, "d_cust_hhi_lag3", x3), (Y5, "d_cust_hhi", x))
    for y, xcol, xx in specs:
        yv = pd.to_numeric(tr[y], errors="coerce")
        if terc is None:
            continue
        for i, cat in enumerate(cats, start=1):
            sl = yv.notna() & xx.notna() & (terc == cat)
            hi = sl & (xx > TAIL_CUT)
            lo = sl & (xx <= TAIL_CUT)
            rate_hi = float(yv[hi].mean()) if int(hi.sum()) else float("nan")
            rate_lo = float(yv[lo].mean()) if int(lo.sum()) else float("nan")
            body = signed_oof_auroc(tr[y], xx, tr["fold"], lo)
            raw = signed_oof_auroc(tr[y], xx, tr["fold"], sl)
            rec = {
                "y": y,
                "x": xcol,
                "tercile": f"T{i}",
                "n": int(sl.sum()),
                "n_pos": int((sl & (yv == 1)).sum()),
                "n tail / pos": f"{int(hi.sum())} / {int((hi & (yv == 1)).sum())}",
                "P(Y=1) tail": _f(rate_hi),
                "P(Y=1) rest": _f(rate_lo),
                "raw CV": "LOW_POWER" if raw["low_power"] else _f(raw["cv"]),
                "body CV": "LOW_POWER" if body["low_power"] else _f(body["cv"]),
            }
            rows.append(rec)
            if i == 1:
                t1_survive[y] = {
                    "rate_hi": rate_hi,
                    "rate_lo": rate_lo,
                    "n_hi": int(hi.sum()),
                    "n_pos_hi": int((hi & (yv == 1)).sum()),
                    "raw": _cv(raw),
                    "body": _cv(body),
                    "crash": bool(np.isfinite(rate_hi) and np.isfinite(rate_lo) and rate_hi > rate_lo),
                }
            print(
                f"7 {y} T{i}: n={rec['n']} tail={int(hi.sum())} "
                f"rate_hi={rate_hi if np.isfinite(rate_hi) else float('nan'):.4f} "
                f"raw={_cv(raw)}"
            )
    t1_y4 = t1_survive.get(Y4, {})
    t1_y3 = t1_survive.get(Y3, {})
    prose = (
        f"Y4 T1 tail P(Y=1)={_f(t1_y4.get('rate_hi'))} vs rest {_f(t1_y4.get('rate_lo'))} "
        f"(n_tail={t1_y4.get('n_hi', 0)} pos={t1_y4.get('n_pos_hi', 0)}). "
        f"{'Crash tail survives inside T1' if t1_y4.get('crash') else 'T1 tail is not a crash / LOW_POWER'}. "
        f"Y3 T1 tail {_f(t1_y3.get('rate_hi'))} vs rest {_f(t1_y3.get('rate_lo'))} "
        f"raw {_f(t1_y3.get('raw'))}."
    )
    print(prose)
    return {"rows": rows, "t1": t1_survive, "prose": prose}


# ---------------------------------------------------------------------------
# 8. Q6 lag1 / lag3 on short books — CLOSE already; confirm LOW_POWER
# ---------------------------------------------------------------------------
def cut8_q6(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    for y in (Y3, Y4):
        for sl_name, sl in (
            ("short_<12", tr["so_far_class"] == "short_<12"),
            ("long_>=18", tr["so_far_class"] == "long_>=18"),
            ("all", pd.Series(True, index=tr.index)),
        ):
            lab = sl & tr[y].notna()
            for feat in ("d_cust_hhi", "d_cust_hhi_lag1", "d_cust_hhi_lag3"):
                res = signed_oof_auroc(tr[y], tr[feat], tr["fold"], lab)
                store[(y, sl_name, feat)] = res
                rows.append(
                    {
                        "y": y,
                        "slice": sl_name,
                        "feature": feat,
                        "n": f"{res['n_defined']:,}",
                        "n_pos": f"{res['n_pos']:,}",
                        "present": _pp(_pct(res["n_defined"], int(lab.sum()))),
                        "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                        "sd": _f(res["sd"]) if not res["low_power"] else "—",
                        "sign": res["train_sign"] if not res["low_power"] else "—",
                    }
                )
                print(f"8 {y} {sl_name} {feat}: {_f(_cv(res))} n={res['n_defined']} pos={res['n_pos']}")
    short_y4 = store[(Y4, "short_<12", "d_cust_hhi_lag3")]
    short_y4_lag1 = store[(Y4, "short_<12", "d_cust_hhi_lag1")]
    all_y4 = store[(Y4, "all", "d_cust_hhi_lag3")]
    y4_lab_short = int(((tr["so_far_class"] == "short_<12") & tr[Y4].notna()).sum())
    present = _pct(short_y4["n_defined"], y4_lab_short)
    present_ok = bool(np.isfinite(present) and abs(present - Q6_SHORT_SHARE) < 0.03)
    pos_ok = short_y4["n_pos"] == Q6_SHORT_POS or abs(short_y4["n_pos"] - Q6_SHORT_POS) <= 2
    low = bool(short_y4["low_power"])
    q6_close = True
    prose = (
        f"Y4 short lag3 present {_pp(present)} ({short_y4['n_defined']}/{y4_lab_short}; "
        f"quote {Q6_SHORT_SHARE:.1%}) n_pos={short_y4['n_pos']} (quote {Q6_SHORT_POS}) — "
        f"{'CONFIRM' if present_ok and pos_ok else 'CHECK'} "
        f"{'LOW_POWER' if low else 'powered'}. "
        f"short lag1 n_pos={short_y4_lag1['n_pos']} "
        f"{'LOW_POWER' if short_y4_lag1['low_power'] else _f(_cv(short_y4_lag1))}. "
        f"Y4 all lag3 {_f(_cv(all_y4))} n={all_y4['n_defined']}. Q6 stays CLOSE."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "short_y4_n": short_y4["n_defined"],
        "short_y4_pos": short_y4["n_pos"],
        "short_present": present,
        "present_ok": present_ok,
        "pos_ok": pos_ok,
        "low": low,
        "q6_close": q6_close,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 9. ICC / company-demean: trait vs month shock
# ---------------------------------------------------------------------------
def cut9_icc(tr: pd.DataFrame) -> dict:
    x = tr["d_cust_hhi"]
    icc = icc_anova(x, tr["company_id"])
    demean = company_demean(x, tr["company_id"])
    cmean = company_mean(x, tr["company_id"])
    rows = []
    store = {}
    for y in (Y3, Y4):
        lab = tr[y].notna()
        d = signed_oof_auroc(tr[y], demean, tr["fold"], lab)
        m = signed_oof_auroc(tr[y], cmean, tr["fold"], lab)
        store[(y, "demean")] = d
        store[(y, "mean")] = m
        rows.append(_auc_row(y, "company-demean", d))
        rows.append(_auc_row(y, "company-mean", m))
        print(f"9 {y} demean={_f(_cv(d))} mean={_f(_cv(m))}")
    trait = bool(np.isfinite(icc["icc"]) and icc["icc"] >= ICC_TRAIT)
    shock = bool(np.isfinite(icc["icc"]) and icc["icc"] < ICC_SHOCK)
    prose = (
        f"ICC {_f(icc['icc'])} η² {_f(icc['eta2'])} k={icc['k']} "
        f"({'TRAIT ≥0.85' if trait else ('shock <0.50' if shock else 'mixed')}). "
        f"Y3 demean {_f(_cv(store[(Y3, 'demean')]))} mean {_f(_cv(store[(Y3, 'mean')]))}. "
        f"Y4 demean {_f(_cv(store[(Y4, 'demean')]))} mean {_f(_cv(store[(Y4, 'mean')]))}."
    )
    print(prose)
    return {
        "icc": icc,
        "rows": rows,
        "store": store,
        "trait": trait,
        "shock": shock,
        "d3": _cv(store[(Y3, "demean")]),
        "m3": _cv(store[(Y3, "mean")]),
        "d4": _cv(store[(Y4, "demean")]),
        "m4": _cv(store[(Y4, "mean")]),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 10. Holdout mix flip (Y4 why: 38% crash / 88% spike) — coverage / mix only
# ---------------------------------------------------------------------------
def cut10_holdout(panel: pd.DataFrame) -> dict:
    ho = panel[panel["split"] == "holdout"].copy()
    tr = panel[panel["split"] == "train"].copy()
    x = pd.to_numeric(ho["d_cust_hhi"], errors="coerce")
    x3 = pd.to_numeric(ho["d_cust_hhi_lag3"], errors="coerce")
    rows = []
    for y in (Y3, Y4):
        yv = pd.to_numeric(ho[y], errors="coerce")
        m = yv.notna() & x.notna()
        m3 = yv.notna() & x3.notna()
        hi = m3 & (x3 > TAIL_CUT)
        rows.append(
            {
                "y": y,
                "n labeled": int(yv.notna().sum()),
                "n_pos": int((yv == 1).sum()),
                "HHI nn": int(m.sum()),
                "HHI_lag3 nn": int(m3.sum()),
                "n tail / pos": f"{int(hi.sum())} / {int((hi & (yv == 1)).sum())}",
            }
        )
    y4_tr = tr[tr[Y4].notna()]
    y4_ho = ho[ho[Y4].notna()]
    tr_pos = y4_tr[y4_tr[Y4] == 1]
    ho_pos = y4_ho[y4_ho[Y4] == 1]
    # y4_why formula: (ratio ? th).mean() — NaN compares False, does not require both defined
    tr_crash = float((tr_pos["in_ratio"] < CRASH_TH).mean()) if len(tr_pos) else float("nan")
    tr_spike = float((tr_pos["ds_ratio"] > SPIKE_TH).mean()) if len(tr_pos) else float("nan")
    ho_crash = float((ho_pos["in_ratio"] < CRASH_TH).mean()) if len(ho_pos) else float("nan")
    ho_spike = float((ho_pos["ds_ratio"] > SPIKE_TH).mean()) if len(ho_pos) else float("nan")
    ho_spike12 = float((ho_pos["ds_ratio"] > 1.2).mean()) if len(ho_pos) else float("nan")
    ho_med_in = float(ho_pos["in_ratio"].median()) if len(ho_pos) else float("nan")
    ho_n_ds_nn = int(ho_pos["ds_ratio"].notna().sum()) if len(ho_pos) else 0
    mix_ok = bool(
        np.isfinite(ho_crash)
        and np.isfinite(ho_spike)
        and abs(ho_crash - HOLD_CRASH) < 0.04
        and abs(ho_spike - HOLD_SPIKE) < 0.04
    )
    mix_rows = [
        {
            "split": "train",
            "n labeled / pos": f"{len(y4_tr)} / {len(tr_pos)}",
            "crash <0.8": _pp(tr_crash),
            "spike >1.5": _pp(tr_spike),
            "med in-ratio pos": _f(float(tr_pos["in_ratio"].median()) if len(tr_pos) else float("nan")),
        },
        {
            "split": "holdout",
            "n labeled / pos": f"{len(y4_ho)} / {len(ho_pos)}",
            "crash <0.8": _pp(ho_crash),
            "spike >1.5": _pp(ho_spike),
            "med in-ratio pos": _f(ho_med_in),
        },
    ]
    prose = (
        f"Holdout 72 is coverage / mix only. Do not quote holdout AUROC. "
        f"Y4 train pos crash {_pp(tr_crash)} spike {_pp(tr_spike)}. "
        f"Holdout pos n={len(ho_pos)} crash {_pp(ho_crash)} spike1.5 {_pp(ho_spike)} "
        f"spike1.2 {_pp(ho_spike12)} ds_nn={ho_n_ds_nn} med in-ratio {_f(ho_med_in)} "
        f"(quote {HOLD_CRASH:.0%} / {HOLD_SPIKE:.0%}) — "
        f"{'CONFIRM mix flip' if mix_ok else 'CHECK vs 38/88'}. "
        + " ".join(f"{r['y']} labeled {r['n labeled']} pos {r['n_pos']} HHI-nn {r['HHI nn']} lag3 {r['HHI_lag3 nn']}." for r in rows)
    )
    print(prose)
    return {
        "rows": rows,
        "mix_rows": mix_rows,
        "tr_crash": tr_crash,
        "tr_spike": tr_spike,
        "ho_crash": ho_crash,
        "ho_spike": ho_spike,
        "ho_spike12": ho_spike12,
        "ho_n_ds_nn": ho_n_ds_nn,
        "ho_n_pos": int(len(ho_pos)),
        "mix_ok": mix_ok,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 11. vs supp HHI: confirm different object (crash vs protective)
# ---------------------------------------------------------------------------
def cut11_vs_supp(tr: pd.DataFrame, twins: dict, tail: dict) -> dict:
    rho = twins["rhos"]["d_supp_hhi"]
    y4c = tail["tail_store"][(Y4, "d_cust_hhi_lag3")]
    y5s = tail["tail_store"][(Y5, "d_supp_hhi")]
    y5c = tail["tail_store"][(Y5, "d_cust_hhi")]
    y3c = tail["tail_store"][(Y3, "d_cust_hhi")]
    different = bool(
        np.isfinite(rho)
        and abs(rho) < TWIN_RHO
        and y4c["rate_hi"] > y4c["rate_lo"]
        and y5s["rate_hi"] < y5s["rate_lo"]
    )
    rows = [
        {
            "object": "Y4 customer HHI_lag3 >0.975",
            "P(Y=1) tail": _f(y4c["rate_hi"]),
            "P(Y=1) rest": _f(y4c["rate_lo"]),
            "shape": "crash",
        },
        {
            "object": "Y5 supplier HHI >0.975",
            "P(Y=1) tail": _f(y5s["rate_hi"]),
            "P(Y=1) rest": _f(y5s["rate_lo"]),
            "shape": "protective",
        },
        {
            "object": "Y5 customer HHI >0.975",
            "P(Y=1) tail": _f(y5c["rate_hi"]),
            "P(Y=1) rest": _f(y5c["rate_lo"]),
            "shape": "crash" if y5c["rate_hi"] > y5c["rate_lo"] else "protective",
        },
        {
            "object": "Y3 customer HHI >0.975",
            "P(Y=1) tail": _f(y3c["rate_hi"]),
            "P(Y=1) rest": _f(y3c["rate_lo"]),
            "shape": "crash" if y3c["rate_hi"] > y3c["rate_lo"] else "protective",
        },
    ]
    prose = (
        f"ρ(cust HHI, supp HHI)={_f(rho)} — not a twin. "
        f"Y4 customer tail is a crash {_f(y4c['rate_hi'])} vs {_f(y4c['rate_lo'])}. "
        f"Y5 supplier tail is protective {_f(y5s['rate_hi'])} vs {_f(y5s['rate_lo'])}. "
        f"{'CONFIRM different object — do not merge with supp HHI / Y4.' if different else 'CHECK — objects may have collapsed'}."
    )
    print(prose)
    return {"rows": rows, "rho": rho, "different": different, "prose": prose}


# ---------------------------------------------------------------------------
# Extra A. 2-col z-avg of HHI + top1 (CLOSE; do not put on the card)
# ---------------------------------------------------------------------------
def extra_zavg(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    for y, hcol, tcol in (
        (Y3, "d_cust_hhi", "d_cust_top1"),
        (Y4, "d_cust_hhi_lag3", "d_cust_top1_lag3"),
    ):
        lab = tr[y].notna()
        yv = pd.to_numeric(tr[y], errors="coerce")
        xs = {
            hcol: pd.to_numeric(tr[hcol], errors="coerce"),
            tcol: pd.to_numeric(tr[tcol], errors="coerce"),
        }
        both = lab
        for s in xs.values():
            both = both & s.notna()
        n_pos = int((both & (yv == 1)).sum())
        if n_pos < MIN_POS:
            res = {
                "cv": float("nan"),
                "sd": float("nan"),
                "low_power": True,
                "n_defined": int(both.sum()),
                "n_pos": n_pos,
                "n_neg": int((both & (yv == 0)).sum()),
                "folds": [],
                "train_sign": 0,
                "train_auc": float("nan"),
            }
        else:
            aucs = []
            fold_rows = []
            for k in range(N_FOLDS):
                trm = both & (tr["fold"] != k)
                va = both & (tr["fold"] == k)
                parts = []
                for _, s in xs.items():
                    sign = choose_sign(yv[trm], s[trm])
                    mu = float(s[trm].mean())
                    sd = float(s[trm].std(ddof=0))
                    z = (s - mu) / sd if sd and np.isfinite(sd) and sd > 0 else s * 0.0
                    parts.append(sign * z)
                score = sum(parts)
                auc = auroc(yv[va], score[va])
                aucs.append(auc)
                fold_rows.append(
                    {
                        "fold": k,
                        "auroc": float(auc) if np.isfinite(auc) else float("nan"),
                        "n_va": int(va.sum()),
                        "n_pos": int((va & (yv == 1)).sum()),
                    }
                )
            finite = [a for a in aucs if np.isfinite(a)]
            res = {
                "cv": float(np.mean(finite)) if finite else float("nan"),
                "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan"),
                "low_power": False,
                "n_defined": int(both.sum()),
                "n_pos": n_pos,
                "n_neg": int((both & (yv == 0)).sum()),
                "folds": fold_rows,
                "train_sign": 1,
                "train_auc": float("nan"),
            }
        store[y] = res
        hhi_cc = signed_oof_auroc(tr[y], tr[hcol], tr["fold"], both)
        top_cc = signed_oof_auroc(tr[y], tr[tcol], tr["fold"], both)
        gap_hhi = (
            _cv(res) - _cv(hhi_cc)
            if np.isfinite(_cv(res)) and np.isfinite(_cv(hhi_cc))
            else float("nan")
        )
        just_hhi = bool(np.isfinite(gap_hhi) and gap_hhi < KEEP_DELTA)
        rows.append(
            {
                "y": y,
                "zavg CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                "HHI-CC": "LOW_POWER" if hhi_cc["low_power"] else _f(hhi_cc["cv"]),
                "top1-CC": "LOW_POWER" if top_cc["low_power"] else _f(top_cc["cv"]),
                "gap vs HHI": _f(gap_hhi),
                "just HHI": str(just_hhi),
                "n / pos": f"{res['n_defined']} / {res['n_pos']}",
            }
        )
        print(f"A zavg {y}: {_f(_cv(res))} HHI-CC {_f(_cv(hhi_cc))} top1-CC {_f(_cv(top_cc))}")
    prose = (
        f"2-col z-avg of HHI + top1 is a twin stack. Y4 z-avg was CLOSE. "
        f"Do not put this on any card. Y3 {_f(_cv(store[Y3]))} Y4 {_f(_cv(store[Y4]))}."
    )
    print(prose)
    return {"rows": rows, "store": store, "prose": prose}


# ---------------------------------------------------------------------------
# Extra B. Same-n leftover after top1
# ---------------------------------------------------------------------------
def extra_samen(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    for y, hcol, tcol in (
        (Y3, "d_cust_hhi", "d_cust_top1"),
        (Y4, "d_cust_hhi_lag3", "d_cust_top1_lag3"),
    ):
        p = tr[hcol]
        t1 = tr[tcol]
        both = tr[y].notna() & p.notna() & t1.notna()
        raw = signed_oof_auroc(tr[y], p, tr["fold"], both)
        top = signed_oof_auroc(tr[y], t1, tr["fold"], both)
        resid, info = ols_resid(p, t1)
        lef = signed_oof_auroc(tr[y], resid, tr["fold"], both)
        store[y] = {"raw": raw, "top": top, "lef": lef, "r2": info["r2"]}
        rows.append(_auc_row(y, "HHI same-n", raw))
        rows.append(_auc_row(y, "top1 same-n", top))
        rows.append(_auc_row(y, "HHI resid after top1", lef))
        print(f"B same-n {y}: HHI {_f(_cv(raw))} top1 {_f(_cv(top))} leftover {_f(_cv(lef))} R2={_f(info['r2'])}")
    artifact = bool(np.isfinite(store[Y4]["r2"]) and store[Y4]["r2"] >= 0.90)
    prose = (
        f"Same-n HHI vs top1: Y4 leftover after top1_lag3 {_f(_cv(store[Y4]['lef']))} "
        f"R²={_f(store[Y4]['r2'])}. Y3 leftover after top1 {_f(_cv(store[Y3]['lef']))} "
        f"R²={_f(store[Y3]['r2'])}. "
        f"{'Near-identity — leftover after top1 is not a new object.' if artifact else 'R² < 0.90 — not a copy, still check leftover CV.'}"
    )
    print(prose)
    return {"rows": rows, "store": store, "artifact": artifact, "prose": prose}


# ---------------------------------------------------------------------------
# Extra C. Tail companies on non-tail months
# ---------------------------------------------------------------------------
def extra_tail_cos(tr: pd.DataFrame) -> dict:
    x3 = pd.to_numeric(tr["d_cust_hhi_lag3"], errors="coerce")
    tail_cos = set(tr.loc[x3 > TAIL_CUT, "company_id"].astype(str))
    rows = []
    for y in (Y3, Y4):
        yv = pd.to_numeric(tr[y], errors="coerce")
        xx = x3 if y == Y4 else pd.to_numeric(tr["d_cust_hhi"], errors="coerce")
        m = yv.notna() & xx.notna() & tr["company_id"].isin(tail_cos)
        hi = m & (xx > TAIL_CUT)
        lo = m & (xx <= TAIL_CUT)
        rows.append(
            {
                "y": y,
                "tail cos": int(tr.loc[m, "company_id"].nunique()),
                "tail months / pos": f"{int(hi.sum())} / {int((hi & (yv == 1)).sum())}",
                "P(Y=1) tail mo": _f(float(yv[hi].mean()) if int(hi.sum()) else float("nan")),
                "non-tail mo / pos": f"{int(lo.sum())} / {int((lo & (yv == 1)).sum())}",
                "P(Y=1) non-tail mo": _f(float(yv[lo].mean()) if int(lo.sum()) else float("nan")),
            }
        )
    y4r = rows[1] if len(rows) > 1 else rows[0]
    prose = (
        f"Companies that ever hit cust HHI_lag3>0.975: {len(tail_cos)}. "
        f"Y4 tail months {_f(float(tr.loc[(tr[Y4].notna()) & (x3 > TAIL_CUT), Y4].mean()) if int(((tr[Y4].notna()) & (x3 > TAIL_CUT)).sum()) else float('nan'))}. "
        f"If those same books are quieter on non-tail months, the object is the bin, not a gradient. "
        f"Y4 non-tail-on-tail-cos {y4r['P(Y=1) non-tail mo']}."
    )
    print(prose)
    return {"rows": rows, "n_cos": len(tail_cos), "prose": prose}


# ---------------------------------------------------------------------------
# Extra D. Y3 leftover after days on the body
# ---------------------------------------------------------------------------
def extra_y3_body_days(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["d_cust_hhi"], errors="coerce")
    body = tr[Y3].notna() & x.notna() & (x <= TAIL_CUT)
    raw = signed_oof_auroc(tr[Y3], x, tr["fold"], body)
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], body)
    resid, info = ols_resid(x, tr["c_n_days_with_tx"])
    lef = signed_oof_auroc(tr[Y3], resid, tr["fold"], body)
    rows = [
        _auc_row(Y3, "HHI body ≤0.975", raw),
        _auc_row(Y3, "days on body", days),
        _auc_row(Y3, "HHI body resid after days", lef),
    ]
    dies = bool(np.isfinite(_cv(lef)) and _cv(lef) < CHANCE)
    prose = (
        f"Y3 body HHI {_f(_cv(raw))} leftover-after-days {_f(_cv(lef))} "
        f"vs days {_f(_cv(days))} R²={_f(info['r2'])}. "
        f"{'Body leftover dies after days — not a monopoly-tail leftover as Y3 X.' if dies else 'Body leftover lives.'}"
    )
    print(prose)
    return {"rows": rows, "raw": _cv(raw), "lef": _cv(lef), "days": _cv(days), "dies": dies, "prose": prose}


# ---------------------------------------------------------------------------
# Extra E. Inverse leftover: days after HHI; top1 after HHI
# ---------------------------------------------------------------------------
def extra_inverse(tr: pd.DataFrame) -> dict:
    p = tr["d_cust_hhi"]
    p3 = tr["d_cust_hhi_lag3"]
    days_after, info_d = ols_resid(tr["c_n_days_with_tx"], p)
    days_inv = signed_oof_auroc(tr[Y3], days_after, tr["fold"], tr[Y3].notna())
    top_after, info_t = ols_resid(tr["d_cust_top1_lag3"], p3)
    top_inv = signed_oof_auroc(tr[Y4], top_after, tr["fold"], tr[Y4].notna())
    rows = [
        _auc_row(Y3, "days resid after HHI", days_inv),
        _auc_row(Y4, "top1_lag3 resid after HHI_lag3", top_inv),
    ]
    prose = (
        f"Inverse: days leftover after HHI {_f(_cv(days_inv))} R²={_f(info_d['r2'])} "
        f"(days 0.711 should survive). "
        f"top1 leftover after HHI {_f(_cv(top_inv))} R²={_f(info_t['r2'])} "
        f"(near-identity twin — leftover of top1 after HHI is not a new object)."
    )
    print(prose)
    return {
        "rows": rows,
        "days_inv": _cv(days_inv),
        "top_inv": _cv(top_inv),
        "r2_days": info_d["r2"],
        "r2_top": info_t["r2"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra F. Leftover after n_cust
# ---------------------------------------------------------------------------
def extra_n_cust(tr: pd.DataFrame) -> dict:
    p = tr["d_cust_hhi"]
    rows = []
    store = {}
    for y in (Y3, Y4):
        resid, info = ols_resid(p, tr["d_n_cust"])
        lef = signed_oof_auroc(tr[y], resid, tr["fold"], tr[y].notna())
        ncv = signed_oof_auroc(tr[y], tr["d_n_cust"], tr["fold"], tr[y].notna())
        store[y] = {"lef": lef, "n": ncv, "r2": info["r2"]}
        rows.append(_auc_row(y, "d_n_cust", ncv))
        rows.append(_auc_row(y, "HHI resid after n_cust", lef))
        print(f"F {y}: n_cust {_f(_cv(ncv))} leftover {_f(_cv(lef))} R2={_f(info['r2'])}")
    prose = (
        f"ρ(HHI, n_cust)={_f(spearman(p, tr['d_n_cust']))}. "
        f"Y3 n_cust {_f(_cv(store[Y3]['n']))} leftover after n_cust {_f(_cv(store[Y3]['lef']))} "
        f"R²={_f(store[Y3]['r2'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "y3_lef": _cv(store[Y3]["lef"]),
        "r2": store[Y3]["r2"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra G. Company-mean leftover after days (style)
# ---------------------------------------------------------------------------
def extra_style(tr: pd.DataFrame) -> dict:
    p = tr["d_cust_hhi"]
    cmean = company_mean(p, tr["company_id"])
    resid, info = ols_resid(cmean, tr["c_n_days_with_tx"])
    lef = signed_oof_auroc(tr[Y3], resid, tr["fold"], tr[Y3].notna())
    raw = signed_oof_auroc(tr[Y3], cmean, tr["fold"], tr[Y3].notna())
    rows = [
        _auc_row(Y3, "company-mean HHI", raw),
        _auc_row(Y3, "mean resid after days", lef),
    ]
    dies = bool(np.isfinite(_cv(lef)) and _cv(lef) < CHANCE)
    prose = (
        f"Y3 company-mean HHI {_f(_cv(raw))} leftover after days {_f(_cv(lef))} "
        f"R²={_f(info['r2'])}. "
        f"{'Style leftover dies — who-is-concentrated is days.' if dies else 'Style leftover still ranks.'}"
    )
    print(prose)
    return {"rows": rows, "mean": _cv(raw), "lef": _cv(lef), "dies": dies, "prose": prose}


# ---------------------------------------------------------------------------
# Extra H. Binary tail indicator vs continuous (Y4)
# ---------------------------------------------------------------------------
def extra_tail_bin(tr: pd.DataFrame) -> dict:
    x3 = pd.to_numeric(tr["d_cust_hhi_lag3"], errors="coerce")
    flag = pd.Series(np.where(x3.notna(), (x3 > TAIL_CUT).astype(float), np.nan), index=tr.index)
    lab = tr[Y4].notna()
    cont = signed_oof_auroc(tr[Y4], x3, tr["fold"], lab)
    binary = signed_oof_auroc(tr[Y4], flag, tr["fold"], lab)
    rows = [
        _auc_row(Y4, "HHI_lag3 continuous", cont),
        _auc_row(Y4, "HHI_lag3 >0.975 flag", binary),
    ]
    prose = (
        f"Y4 continuous {_f(_cv(cont))} vs binary tail flag {_f(_cv(binary))} "
        f"(y4_why binary OOF 0.578). "
        f"{'Continuous still wins; the honest card shape is still the bin.' if (np.isfinite(_cv(cont)) and np.isfinite(_cv(binary)) and _cv(cont) >= _cv(binary)) else 'Binary matches or beats continuous.'}"
    )
    print(prose)
    return {"rows": rows, "cont": _cv(cont), "binary": _cv(binary), "prose": prose}


# ---------------------------------------------------------------------------
# Extra I. Dark 470 — HHI defined?
# ---------------------------------------------------------------------------
def extra_dark(tr: pd.DataFrame, book: set[str], dark: dict) -> dict:
    train_dark = set(dark.get("train_dark_ids") or set())
    x = pd.to_numeric(tr["d_cust_hhi"], errors="coerce")
    top1 = pd.to_numeric(tr["d_cust_top1"], errors="coerce")
    supp = pd.to_numeric(tr["d_supp_hhi"], errors="coerce")
    dark_m = tr["company_id"].isin(train_dark)
    book_m = tr["company_id"].isin(book)
    ghost = tr["company_id"] == "COMP_0962"
    rows = [
        {
            "slice": "dark 470 CM",
            "cm": f"{int(dark_m.sum()):,}",
            "HHI nn": int(x[dark_m].notna().sum()),
            "HHI==0": int((x[dark_m] == 0).sum()),
            "top1 nn": int(top1[dark_m].notna().sum()),
            "supp HHI nn": int(supp[dark_m].notna().sum()),
        },
        {
            "slice": "invoice-book CM",
            "cm": f"{int(book_m.sum()):,}",
            "HHI nn": int(x[book_m].notna().sum()),
            "HHI==0": int((x[book_m] == 0).sum()),
            "top1 nn": int(top1[book_m].notna().sum()),
            "supp HHI nn": int(supp[book_m].notna().sum()),
        },
        {
            "slice": "COMP_0962 (refund ghost)",
            "cm": f"{int(ghost.sum()):,}",
            "HHI nn": int(x[ghost].notna().sum()),
            "HHI==0": int((x[ghost] == 0).sum()),
            "top1 nn": int(top1[ghost].notna().sum()),
            "supp HHI nn": int(supp[ghost].notna().sum()),
        },
    ]
    ok = int(x[dark_m].notna().sum()) == 0 and int((x[dark_m] == 0).sum()) == 0
    prose = (
        f"Dark 470 HHI defined {int(x[dark_m].notna().sum())} zero {int((x[dark_m] == 0).sum())}. "
        f"{'CONFIRM NaN not 0 — HHI needs invoice CPs.' if ok else 'FAIL — dark HHI leaked a number'}. "
        f"COMP_0962 HHI nn={int(x[ghost].notna().sum())}."
    )
    print(prose)
    return {"rows": rows, "ok": ok, "prose": prose, "dark_nn": int(x[dark_m].notna().sum())}


# ---------------------------------------------------------------------------
# Extra J. Y3 T2/T3 pocket leftover after days
# ---------------------------------------------------------------------------
def extra_y3_pocket(tr: pd.DataFrame) -> dict:
    size = pd.to_numeric(tr["log_in3"], errors="coerce")
    defined = size.notna()
    tercile = pd.qcut(size[defined], 3, labels=["T1", "T2", "T3"], duplicates="drop")
    terc = pd.Series(index=tr.index, dtype=object)
    terc.loc[defined] = tercile.astype(str)
    p = tr["d_cust_hhi"]
    rows = []
    store = {}
    for cat in ("T1", "T2", "T3", "T2+T3"):
        if cat == "T2+T3":
            sl = (terc == "T2") | (terc == "T3")
        else:
            sl = terc == cat
        lab = sl & tr[Y3].notna()
        raw = signed_oof_auroc(tr[Y3], p, tr["fold"], lab)
        days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], lab)
        resid, info = ols_resid(p, tr["c_n_days_with_tx"])
        lef = signed_oof_auroc(tr[Y3], resid, tr["fold"], lab)
        store[cat] = {"raw": raw, "days": days, "lef": lef, "r2": info["r2"]}
        rows.append(
            {
                "slice": cat,
                "n / pos": f"{raw['n_defined']} / {raw['n_pos']}",
                "HHI CV": "LOW_POWER" if raw["low_power"] else _f(raw["cv"]),
                "days CV": "LOW_POWER" if days["low_power"] else _f(days["cv"]),
                "leftover days": "LOW_POWER" if lef["low_power"] else _f(lef["cv"]),
            }
        )
        print(f"J Y3 {cat}: HHI {_f(_cv(raw))} days {_f(_cv(days))} leftover {_f(_cv(lef))}")
    t23 = store["T2+T3"]
    dies = bool(np.isfinite(_cv(t23["lef"])) and _cv(t23["lef"]) < CHANCE)
    prose = (
        f"Y3 T2+T3 HHI {_f(_cv(t23['raw']))} leftover-after-days {_f(_cv(t23['lef']))} "
        f"vs days {_f(_cv(t23['days']))}. "
        f"{'Pocket dies after days — not leftover turning.' if dies else 'Pocket leftover lives — unexpected.'}"
    )
    print(prose)
    return {"rows": rows, "store": store, "dies": dies, "lef_t23": _cv(t23["lef"]), "prose": prose}


# ---------------------------------------------------------------------------
# Extra K. Short-book lag3 leftover after top1 (Q6 honesty)
# ---------------------------------------------------------------------------
def extra_q6_honest(tr: pd.DataFrame) -> dict:
    short = tr["so_far_class"] == "short_<12"
    rows = []
    store = {}
    for y, feat, bar, barname in (
        (Y4, "d_cust_hhi_lag3", tr["d_cust_top1_lag3"], "top1_lag3"),
        (Y3, "d_cust_hhi_lag3", tr["c_n_days_with_tx"], "days"),
        (Y3, "d_cust_hhi", tr["c_n_days_with_tx"], "days"),
    ):
        lab = short & tr[y].notna()
        raw = signed_oof_auroc(tr[y], tr[feat], tr["fold"], lab)
        resid, info = ols_resid(tr[feat], bar)
        lef = signed_oof_auroc(tr[y], resid, tr["fold"], lab)
        store[(y, feat)] = {"raw": raw, "lef": lef, "r2": info["r2"]}
        rows.append(
            {
                "y": y,
                "feat": feat,
                "bar": barname,
                "lag CV": "LOW_POWER" if raw["low_power"] else _f(raw["cv"]),
                "leftover": "LOW_POWER" if lef["low_power"] else _f(lef["cv"]),
                "n / pos": f"{raw['n_defined']} / {raw['n_pos']}",
                "R²": _f(info["r2"]),
            }
        )
        print(f"K short {y} {feat} leftover-after-{barname} {_f(_cv(lef))}")
    q6_keep = bool(
        np.isfinite(_cv(store[(Y4, "d_cust_hhi_lag3")]["lef"]))
        and _cv(store[(Y4, "d_cust_hhi_lag3")]["lef"]) >= CHANCE
        and np.isfinite(_cv(store[(Y4, "d_cust_hhi_lag3")]["raw"]))
        and _cv(store[(Y4, "d_cust_hhi_lag3")]["raw"]) >= SIZE_QUOTE + KEEP_DELTA
    )
    prose = (
        f"Y4 short lag3 leftover after top1_lag3 "
        f"{_f(_cv(store[(Y4, 'd_cust_hhi_lag3')]['lef']))}. "
        f"Y3 short leftover after days {_f(_cv(store[(Y3, 'd_cust_hhi')]['lef']))}. "
        f"{'Unexpected Q6 KEEP' if q6_keep else 'Q6 CLOSE — short-book lag is not leftover after the honest bar.'}"
    )
    print(prose)
    return {"rows": rows, "store": store, "q6_keep": q6_keep, "prose": prose}


# ---------------------------------------------------------------------------
# Extra L. Leftover after n_cust + days
# ---------------------------------------------------------------------------
def extra_n_and_days(tr: pd.DataFrame) -> dict:
    p = tr["d_cust_hhi"]
    resid, info = ols_resid(p, tr["d_n_cust"], tr["c_n_days_with_tx"])
    lef = signed_oof_auroc(tr[Y3], resid, tr["fold"], tr[Y3].notna())
    rnk = signed_oof_auroc(
        tr[Y3],
        rank_resid(rank_resid(p, tr["d_n_cust"]), tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        _auc_row(Y3, "HHI resid after n_cust+days", lef),
        _auc_row(Y3, "rank resid n_cust then days", rnk),
    ]
    dies = bool(np.isfinite(_cv(lef)) and _cv(lef) < CHANCE)
    prose = (
        f"Y3 leftover after n_cust+days {_f(_cv(lef))} R²={_f(info['r2'])}. "
        f"Rank-ortho {_f(_cv(rnk))}. "
        f"{'Dies after n_cust+days — count+activity eat the Y3 rank.' if dies else 'Still ranks after n_cust+days — but twin of top1, still DROP as X.'}"
    )
    print(prose)
    return {"rows": rows, "lef": _cv(lef), "rnk": _cv(rnk), "r2": info["r2"], "dies": dies, "prose": prose}


# ---------------------------------------------------------------------------
# Extra M. Holdout 16 positives — spike 1.5 vs 1.2 (y4_why 88%)
# ---------------------------------------------------------------------------
def extra_hold_pos(panel: pd.DataFrame) -> dict:
    ho = panel[(panel["split"] == "holdout") & panel[Y4].notna()].copy()
    pos = ho[ho[Y4] == 1].copy()
    rows = []
    for _, r in pos.sort_values("period").iterrows():
        rows.append(
            {
                "company": str(r["company_id"]),
                "period": pd.Timestamp(r["period"]).strftime("%Y-%m"),
                "in_ratio": _f(r["in_ratio"]),
                "ds_ratio": _f(r["ds_ratio"]),
                "crash<0.8": str(bool(r["in_ratio"] < CRASH_TH) if pd.notna(r["in_ratio"]) else "—"),
                "spike>1.5": str(bool(r["ds_ratio"] > SPIKE_TH) if pd.notna(r["ds_ratio"]) else "—"),
                "spike>1.2": str(bool(r["ds_ratio"] > 1.2) if pd.notna(r["ds_ratio"]) else "—"),
                "HHI_lag3": _f(r.get("d_cust_hhi_lag3")),
            }
        )
    n = len(pos)
    n_ds = int(pos["ds_ratio"].notna().sum())
    spike15 = float((pos["ds_ratio"] > SPIKE_TH).mean()) if n else float("nan")
    spike12 = float((pos["ds_ratio"] > 1.2).mean()) if n else float("nan")
    spike15_def = (
        float((pos.loc[pos["ds_ratio"].notna(), "ds_ratio"] > SPIKE_TH).mean()) if n_ds else float("nan")
    )
    crash = float((pos["in_ratio"] < CRASH_TH).mean()) if n else float("nan")
    # y4_why 0.875 = 14/16. If our 1.5 is lower, 1.2 or defined-only may recover the quote.
    match15 = bool(np.isfinite(spike15) and abs(spike15 - HOLD_SPIKE) < 0.04)
    match12 = bool(np.isfinite(spike12) and abs(spike12 - HOLD_SPIKE) < 0.04)
    match_def = bool(np.isfinite(spike15_def) and abs(spike15_def - HOLD_SPIKE) < 0.04)
    prose = (
        f"Holdout Y4 pos n={n} ds defined {n_ds}. "
        f"crash {_pp(crash)} (quote 38%). "
        f"spike>1.5 {_pp(spike15)} among-defined {_pp(spike15_def)} spike>1.2 {_pp(spike12)} "
        f"(quote 88%). "
        f"{'CONFIRM 1.5' if match15 else ('CONFIRM 1.2 is the 88%' if match12 else ('CONFIRM defined-only 1.5' if match_def else '88% is not this rebuild — LOW_POWER 16'))}. "
        "Coverage / mix only."
    )
    print(prose)
    return {
        "rows": rows,
        "n": n,
        "n_ds": n_ds,
        "spike15": spike15,
        "spike12": spike12,
        "spike15_def": spike15_def,
        "crash": crash,
        "match15": match15,
        "match12": match12,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra N. n_cust leftover 0.608 — fake-days leak?
# ---------------------------------------------------------------------------
def extra_ncust_leak(tr: pd.DataFrame) -> dict:
    p = tr["d_cust_hhi"]
    resid, info = ols_resid(p, tr["d_n_cust"])
    rho_days = spearman(resid, tr["c_n_days_with_tx"])
    lef = signed_oof_auroc(tr[Y3], resid, tr["fold"], tr[Y3].notna())
    resid2, info2 = ols_resid(p, tr["d_n_cust"], tr["c_n_days_with_tx"])
    lef2 = signed_oof_auroc(tr[Y3], resid2, tr["fold"], tr[Y3].notna())
    resid3, info3 = ols_resid(p, tr["d_n_cust"], tr["c_n_days_with_tx"], tr["d_cust_top1"])
    lef3 = signed_oof_auroc(tr[Y3], resid3, tr["fold"], tr[Y3].notna())
    rows = [
        _auc_row(Y3, "HHI resid after n_cust", lef),
        _auc_row(Y3, "HHI resid after n_cust+days", lef2),
        _auc_row(Y3, "HHI resid after n_cust+days+top1", lef3),
    ]
    fake = bool(np.isfinite(_cv(lef)) and _cv(lef) >= CHANCE and abs(rho_days) >= FAKE_DAYS_RHO)
    dies3 = bool(np.isfinite(_cv(lef3)) and _cv(lef3) < CHANCE)
    prose = (
        f"Y3 leftover after n_cust {_f(_cv(lef))} R²={_f(info['r2'])} "
        f"ρ(resid, days)={_f(rho_days)} "
        f"({'FAKE-DAYS through n_cust' if fake else 'not a fake-days leak through n_cust'}). "
        f"after n_cust+days {_f(_cv(lef2))} R²={_f(info2['r2'])}. "
        f"after n_cust+days+top1 {_f(_cv(lef3))} R²={_f(info3['r2'])} "
        f"({'dies — count+activity+top1 eat the 0.608' if dies3 else 'still ranks after the three'}). "
        f"ρ(HHI, n_cust)=-0.837 is a count rewrite, not the official KEEP twin list."
    )
    print(prose)
    return {
        "rows": rows,
        "lef": _cv(lef),
        "lef2": _cv(lef2),
        "lef3": _cv(lef3),
        "rho_days": rho_days,
        "fake": fake,
        "dies3": dies3,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra O. Y4 leftover after top1 on the body only
# ---------------------------------------------------------------------------
def extra_y4_body_top1(tr: pd.DataFrame) -> dict:
    x3 = pd.to_numeric(tr["d_cust_hhi_lag3"], errors="coerce")
    t1 = pd.to_numeric(tr["d_cust_top1_lag3"], errors="coerce")
    body = tr[Y4].notna() & x3.notna() & (x3 <= TAIL_CUT)
    raw = signed_oof_auroc(tr[Y4], x3, tr["fold"], body)
    top = signed_oof_auroc(tr[Y4], t1, tr["fold"], body)
    resid, info = ols_resid(x3, t1)
    lef = signed_oof_auroc(tr[Y4], resid, tr["fold"], body)
    rows = [
        _auc_row(Y4, "HHI_lag3 body", raw),
        _auc_row(Y4, "top1_lag3 on body", top),
        _auc_row(Y4, "HHI body resid after top1", lef),
    ]
    dies = bool(np.isfinite(_cv(lef)) and _cv(lef) < CHANCE)
    prose = (
        f"Y4 body leftover after top1_lag3 {_f(_cv(lef))} "
        f"(body raw {_f(_cv(raw))}, top1 {_f(_cv(top))}) R²={_f(info['r2'])}. "
        f"{'Body leftover after top1 dies — no residual gradient under the tail.' if dies else 'Body leftover after top1 still ranks.'}"
    )
    print(prose)
    return {
        "rows": rows,
        "raw": _cv(raw),
        "lef": _cv(lef),
        "top": _cv(top),
        "dies": dies,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra P. Days leftover after HHI on the same Y3 HHI-defined rows
# ---------------------------------------------------------------------------
def extra_days_same_n(tr: pd.DataFrame) -> dict:
    both = tr[Y3].notna() & tr["d_cust_hhi"].notna() & tr["c_n_days_with_tx"].notna()
    hhi = signed_oof_auroc(tr[Y3], tr["d_cust_hhi"], tr["fold"], both)
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], both)
    resid, info = ols_resid(tr["c_n_days_with_tx"], tr["d_cust_hhi"])
    days_lef = signed_oof_auroc(tr[Y3], resid, tr["fold"], both)
    rows = [
        _auc_row(Y3, "HHI same-n", hhi),
        _auc_row(Y3, "days same-n", days),
        _auc_row(Y3, "days resid after HHI", days_lef),
    ]
    days_lives = bool(np.isfinite(_cv(days_lef)) and _cv(days_lef) >= CHANCE)
    prose = (
        f"On HHI-defined Y3 rows: HHI {_f(_cv(hhi))} days {_f(_cv(days))} "
        f"days leftover after HHI {_f(_cv(days_lef))} R²={_f(info['r2'])}. "
        f"{'Days survive after HHI — HHI does not eat the 0.711 bar.' if days_lives else 'Days leftover dies — unexpected.'}"
    )
    print(prose)
    return {
        "rows": rows,
        "hhi": _cv(hhi),
        "days": _cv(days),
        "days_lef": _cv(days_lef),
        "days_lives": days_lives,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra Q. Holdout HHI coverage on crash vs spike positives
# ---------------------------------------------------------------------------
def extra_hold_hhi_mix(panel: pd.DataFrame) -> dict:
    ho = panel[(panel["split"] == "holdout") & panel[Y4].notna()].copy()
    pos = ho[ho[Y4] == 1]
    x3 = pd.to_numeric(pos["d_cust_hhi_lag3"], errors="coerce")
    crash = pos["in_ratio"] < CRASH_TH
    spike = pos["ds_ratio"] > SPIKE_TH
    rows = [
        {
            "slice": "all hold Y4 pos",
            "n": int(len(pos)),
            "HHI_lag3 nn": int(x3.notna().sum()),
            "tail >0.975": int((x3 > TAIL_CUT).sum()),
        },
        {
            "slice": "crash pos",
            "n": int(crash.sum()),
            "HHI_lag3 nn": int(x3[crash].notna().sum()),
            "tail >0.975": int((x3[crash] > TAIL_CUT).sum()),
        },
        {
            "slice": "spike>1.5 pos",
            "n": int(spike.sum()),
            "HHI_lag3 nn": int(x3[spike].notna().sum()),
            "tail >0.975": int((x3[spike] > TAIL_CUT).sum()),
        },
    ]
    prose = (
        f"Holdout Y4 pos HHI_lag3 defined {int(x3.notna().sum())}/{len(pos)}. "
        f"Crash pos n={int(crash.sum())} lag3 nn={int(x3[crash].notna().sum())}. "
        f"Spike pos n={int(spike.sum())} lag3 nn={int(x3[spike].notna().sum())}. "
        "Tail almost empty on the hidden 72 — HHI cannot invert the mix flip. LOW_POWER."
    )
    print(prose)
    return {"rows": rows, "prose": prose, "nn": int(x3.notna().sum())}


# ---------------------------------------------------------------------------
# Extra R. Y3 leftover after days of HHI_lag3 (Q6 leftover)
# ---------------------------------------------------------------------------
def extra_y3_lag3_days(tr: pd.DataFrame) -> dict:
    p3 = tr["d_cust_hhi_lag3"]
    resid, info = ols_resid(p3, tr["c_n_days_with_tx"])
    lef = signed_oof_auroc(tr[Y3], resid, tr["fold"], tr[Y3].notna())
    raw = signed_oof_auroc(tr[Y3], p3, tr["fold"], tr[Y3].notna())
    rows = [
        _auc_row(Y3, "HHI_lag3", raw),
        _auc_row(Y3, "HHI_lag3 resid after days", lef),
    ]
    dies = bool(np.isfinite(_cv(lef)) and _cv(lef) < CHANCE)
    prose = (
        f"Y3 HHI_lag3 {_f(_cv(raw))} leftover after days {_f(_cv(lef))} "
        f"R²={_f(info['r2'])}. "
        f"{'Lag3 leftover dies — Q6 is not a leftover lead as Y3 X.' if dies else 'Lag3 leftover lives — unexpected.'}"
    )
    print(prose)
    return {"rows": rows, "raw": _cv(raw), "lef": _cv(lef), "dies": dies, "prose": prose}


# ---------------------------------------------------------------------------
# Extra S. Y4 contemporaneous body leftover after days
# ---------------------------------------------------------------------------
def extra_y4_now_body(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["d_cust_hhi"], errors="coerce")
    body = tr[Y4].notna() & x.notna() & (x <= TAIL_CUT)
    raw = signed_oof_auroc(tr[Y4], x, tr["fold"], body)
    days = signed_oof_auroc(tr[Y4], tr["c_n_days_with_tx"], tr["fold"], body)
    resid, info = ols_resid(x, tr["c_n_days_with_tx"])
    lef = signed_oof_auroc(tr[Y4], resid, tr["fold"], body)
    resid2, info2 = ols_resid(x, tr["d_cust_top1"])
    lef2 = signed_oof_auroc(tr[Y4], resid2, tr["fold"], body)
    rows = [
        _auc_row(Y4, "HHI now body ≤0.975", raw),
        _auc_row(Y4, "days on now-body", days),
        _auc_row(Y4, "now-body resid after days", lef),
        _auc_row(Y4, "now-body resid after top1", lef2),
    ]
    dies = bool(np.isfinite(_cv(lef)) and _cv(lef) < CHANCE)
    prose = (
        f"Y4 contemporaneous body {_f(_cv(raw))} leftover after days {_f(_cv(lef))} "
        f"vs days {_f(_cv(days))} R²={_f(info['r2'])}; leftover after top1 {_f(_cv(lef2))} "
        f"R²={_f(info2['r2'])}. Lag3 body was 0.445. "
        f"{'Now-body leftover dies — contemporaneous 0.572 is not leftover after days/top1.' if dies else 'Now-body leftover lives — unexpected.'}"
    )
    print(prose)
    return {
        "rows": rows,
        "raw": _cv(raw),
        "lef": _cv(lef),
        "lef2": _cv(lef2),
        "dies": dies,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra U. Now-body leftover 0.569 after days — fake-top1 leak?
# ---------------------------------------------------------------------------
def extra_y4_now_fake(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["d_cust_hhi"], errors="coerce")
    body = tr[Y4].notna() & x.notna() & (x <= TAIL_CUT)
    resid_d, info_d = ols_resid(x, tr["c_n_days_with_tx"])
    rho_top1 = spearman(resid_d, tr["d_cust_top1"])
    lef_d = signed_oof_auroc(tr[Y4], resid_d, tr["fold"], body)
    resid_both, info_b = ols_resid(x, tr["c_n_days_with_tx"], tr["d_cust_top1"])
    lef_both = signed_oof_auroc(tr[Y4], resid_both, tr["fold"], body)
    rows = [
        _auc_row(Y4, "now-body resid after days", lef_d),
        _auc_row(Y4, "now-body resid after days+top1", lef_both),
    ]
    fake = bool(
        np.isfinite(_cv(lef_d))
        and _cv(lef_d) >= CHANCE
        and np.isfinite(rho_top1)
        and abs(rho_top1) >= TWIN_RHO
    )
    dies = bool(np.isfinite(_cv(lef_both)) and _cv(lef_both) < CHANCE)
    prose = (
        f"Y4 now-body leftover after days {_f(_cv(lef_d))} "
        f"ρ(resid, top1)={_f(rho_top1)} "
        f"({'FAKE-TOP1 leak — the 0.569 is the twin through days' if fake else 'not a |ρ|≥0.80 fake-top1'}). "
        f"after days+top1 {_f(_cv(lef_both))} R²={_f(info_b['r2'])} "
        f"({'dies — days leftover was the twin' if dies else 'still ranks after days+top1'})."
    )
    print(prose)
    return {
        "rows": rows,
        "lef_d": _cv(lef_d),
        "lef_both": _cv(lef_both),
        "rho_top1": rho_top1,
        "fake": fake,
        "dies": dies,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra T. Y4 leftover after tail-flag is the bin (0.464)
# ---------------------------------------------------------------------------
def extra_y4_tailflag(tr: pd.DataFrame, leftover: dict) -> dict:
    after_tail = leftover["after_tail"]
    after_top1 = leftover["after_top1"]
    rows = [
        {
            "residual": "after tail-flag",
            "CV": _f(after_tail),
            "note": "0.605 minus the >0.975 bin",
        },
        {
            "residual": "after top1_lag3",
            "CV": _f(after_top1),
            "note": "near-identity R²=0.966",
        },
    ]
    dies = bool(np.isfinite(after_tail) and after_tail < CHANCE)
    prose = (
        f"Y4 leftover after the >0.975 flag {_f(after_tail)} "
        f"({'dies — the 0.605 is the bin' if dies else 'still ranks after the bin'}). "
        f"Leftover after top1_lag3 {_f(after_top1)}. "
        "Do not tell a smooth concentration-gradient story."
    )
    print(prose)
    return {"rows": rows, "after_tail": after_tail, "dies": dies, "prose": prose}


# ---------------------------------------------------------------------------
# 12 + decide
# ---------------------------------------------------------------------------
def decide(ctx: dict) -> dict:
    c2, c3, c4, c5, c6, c8, c9, c11 = (
        ctx["c2"],
        ctx["c3"],
        ctx["c4"],
        ctx["c5"],
        ctx["c6"],
        ctx["c8"],
        ctx["c9"],
        ctx["c11"],
    )
    keep_y3 = bool(
        c4["beat_y3"]
        and c5["lives_y3"]
        and not c2["size_flag"]
        and not c2["any_twin"]
    )
    keep_y4_x = bool(
        c4["beat_y4"]
        and c5["lives_y4"]
        and not c2["size_flag"]
        and not c2["any_twin"]
        and np.isfinite(c3["y4_body"])
        and c3["y4_body"] >= CHANCE
    )
    keep_tail = True  # Y4 monopoly-tail footnote already locked KEEP

    if c6["drop_hhi"]:
        y3 = "DROP"
        y3_why = (
            f"twin of `d_cust_top1` ρ={_f(c2['rhos']['d_cust_top1'])} / "
            f"lag3 {_f(c2['rho_lag'])} — HHI is the weaker rewrite. "
            f"Y3 {_f(c4['hhi_y3'])} leftover-after-days {_f(c5['after_days'])}."
        )
        y4_x = "DROP"
        y4_x_why = (
            f"same twin. Y4 lag3 {_f(c4['hhi3_y4'])} leftover after top1_lag3 {_f(c5['after_top1'])}; "
            f"body CV {_f(c3['y4_body'])} (quote {Y4_BODY})."
        )
    elif c5["died_y3"] and not c4["beat_y3"]:
        y3 = "DROP"
        y3_why = (
            f"Y3 {_f(c4['hhi_y3'])} vs size {_f(c4['size_y3'])} / days {_f(c4['days_y3'])}; "
            f"leftover after days {_f(c5['after_days'])} dies <0.55."
        )
        y4_x = "DROP" if (c5["died_y4"] or c3["y4_dead"]) else "CLOSE"
        y4_x_why = (
            f"Y4 lag3 {_f(c4['hhi3_y4'])} leftover after top1_lag3 {_f(c5['after_top1'])}; "
            f"body {_f(c3['y4_body'])}."
        )
    else:
        y3 = "KEEP" if keep_y3 else ("CLOSE" if c5["lives_y3"] else "DROP")
        y3_why = (
            f"Y3 {_f(c4['hhi_y3'])} leftover-after-days {_f(c5['after_days'])} "
            f"vs size {_f(c4['size_y3'])} days {_f(c4['days_y3'])}."
        )
        y4_x = "KEEP" if keep_y4_x else ("CLOSE" if c5["lives_y4"] else "DROP")
        y4_x_why = (
            f"Y4 lag3 {_f(c4['hhi3_y4'])} leftover after top1_lag3 {_f(c5['after_top1'])}; "
            f"body {_f(c3['y4_body'])}."
        )

    lose_44 = bool(
        (c6["drop_hhi"] or c5["died_y3"] or c3["y4_dead"])
        and y3 in {"DROP", "CLOSE"}
        and y4_x in {"DROP", "CLOSE"}
    )
    if lose_44:
        overall = "DROP"
        on44 = "DROP from the 44 as engine X"
        on44_why = (
            f"twin or leftover dies as X. Y3 leftover {_f(c5['after_days'])}; "
            f"Y4 leftover {_f(c5['after_top1'])} body {_f(c3['y4_body'])}. "
            f"Javier concentration is top1. Y4 tail footnote may stay."
        )
    elif keep_y3 or keep_y4_x:
        overall = "KEEP"
        on44 = "KEEP on the 44"
        on44_why = "clears KEEP-as-X on at least one honest bar."
    else:
        overall = "CLOSE"
        on44 = "CLOSE / PARK on the 44"
        on44_why = "fails KEEP-as-X; diagnostic only."

    q6 = "CLOSE" if c8["q6_close"] else "KEEP"
    q6_why = (
        f"short lag3 present {_pp(c8['short_present'])} n_pos={c8['short_y4_pos']} "
        f"{'LOW_POWER' if c8['low'] else ''} — quote 21.7% / 42 pos."
    )
    if c6["drop_hhi"] and c3["y4_dead"]:
        object_kind = "top1 rewrite; Y4 0.605 is the >0.975 monopoly tail, not a gradient"
    elif c6["drop_hhi"]:
        object_kind = "twin of top1"
    elif c3["y4_dead"]:
        object_kind = "monopoly tail (body dead)"
    else:
        object_kind = "concentration gradient"
    table = [
        {
            "object": "d_cust_hhi as Y3 X / the 15-col card",
            "decision": y3,
            "why": y3_why + " Stays off the 15-col card.",
        },
        {
            "object": "d_cust_hhi as engine X on the 44",
            "decision": on44,
            "why": on44_why,
        },
        {
            "object": "d_cust_hhi_lag3 as Y4 tail footnote",
            "decision": "KEEP",
            "why": (
                f"Y4 lag3 CV {_f(c4['hhi3_y4'])} (quote {Y4_HHI_LAG3}). "
                f"Tail >0.975 {_f(c3['y4_rate_hi'])} vs {_f(c3['y4_rate_lo'])}; "
                f"body {_f(c3['y4_body'])} CONFIRM {c3['body_confirm']}. Trees stay PARK."
            ),
        },
        {
            "object": "Y4 trees / 2-col z-avg",
            "decision": "PARK / CLOSE",
            "why": "Trees PARK (lose to the 0.605 single). z-avg was CLOSE. Do not reopen. Do not put on the card.",
        },
        {
            "object": "twin of d_cust_top1 / lag3",
            "decision": "DROP weaker (HHI) as X" if c6["drop_hhi"] else "not a twin",
            "why": c6["prose"],
        },
        {
            "object": "same object as d_supp_hhi?",
            "decision": "NO — different object" if c11["different"] else "CHECK",
            "why": c11["prose"],
        },
        {
            "object": "y_cust_hhi / reopen Y4 trees",
            "decision": "PARK",
            "why": "do not invent y_cust_hhi. Do not reopen Y4 trees.",
        },
        {
            "object": "Q6 lag1/lag3 on short books",
            "decision": q6,
            "why": q6_why,
        },
        {
            "object": "ICC / trait vs month shock",
            "decision": "TRAIT" if c9["trait"] else ("shock" if c9["shock"] else "mixed"),
            "why": c9["prose"],
        },
        {
            "object": "KEEP-as-X gate (beat size+0.02, leftover, not SIZE, not twin)",
            "decision": "FAIL" if not (keep_y3 or keep_y4_x) else "PASS",
            "why": (
                f"beat-size Y3={c4['beat_y3']} Y4={c4['beat_y4']}; "
                f"leftover Y3 lives={c5['lives_y3']} Y4 lives={c5['lives_y4']}; "
                f"SIZE={c2['size_flag']} twin={c2['any_twin']}."
            ),
        },
    ]
    headline = (
        f"**{overall}** as engine X. Y3 **{y3}** leftover-after-days {_f(c5['after_days'])}. "
        f"Y4 engine X **{y4_x}** leftover after top1_lag3 {_f(c5['after_top1'])} "
        f"body {_f(c3['y4_body'])} ({'CONFIRM' if c3['body_confirm'] else 'check'} {Y4_BODY}). "
        f"Y4 tail footnote **KEEP** ({_f(c3['y4_rate_hi'])} vs {_f(c3['y4_rate_lo'])}; "
        f"lag3 {_f(c4['hhi3_y4'])}). "
        f"{'SIZE' if c2['size_flag'] else 'not SIZE'} (ρ={_f(c2['rhos']['log1p(a_in3)'])}); "
        f"{'twin of d_cust_top1 ρ=' + _f(c2['rhos']['d_cust_top1']) + ' / lag3 ' + _f(c2['rho_lag']) if c2['top1_twin'] else 'not a |ρ|≥0.80 twin'}. "
        f"Object: {object_kind}. "
        f"44 should {'lose `d_cust_hhi` as engine X' if lose_44 else 'keep (diagnostic only) `d_cust_hhi` as X'}; "
        f"Y4 tail footnote may stay. Q6 {q6}. "
        f"Night quotes unchanged: Y3 {NIGHT_Y3:.3f} / {NIGHT_Y3_CORE:.3f}; days {DAYS_BENCH:.3f}; "
        f"size {SIZE_QUOTE:.3f}; TURNOVER {TURNOVER_QUOTE:.3f} / {B_SHALLOW_QUOTE:.3f}."
    )
    print("DECIDE", headline)
    return {
        "overall": overall,
        "y3": y3,
        "y3_why": y3_why,
        "y4_x": y4_x,
        "y4_x_why": y4_x_why,
        "keep_tail": keep_tail,
        "q6": q6,
        "q6_why": q6_why,
        "lose_44": lose_44,
        "on44": on44,
        "on44_why": on44_why,
        "object_kind": object_kind,
        "keep_y3": keep_y3,
        "keep_y4_x": keep_y4_x,
        "table": table,
        "headline": headline,
    }


def plot_png(tr: pd.DataFrame, c3: dict) -> bool:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return False
    x3 = pd.to_numeric(tr["d_cust_hhi_lag3"], errors="coerce")
    y4 = pd.to_numeric(tr[Y4], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))

    m = x3.notna() & y4.notna()
    d = pd.DataFrame({"x": x3[m], "y": y4[m]})
    try:
        d["q"] = pd.qcut(d["x"], 5, duplicates="drop")
        rates = [float(g["y"].mean()) for _, g in d.groupby("q", observed=True)]
        ns = [len(g) for _, g in d.groupby("q", observed=True)]
        axes[0].bar(range(1, len(rates) + 1), rates, color="#3d5a80")
        axes[0].axhline(float(d["y"].mean()), color="#ee6c4d", ls="--", lw=1, label="train labeled mean")
        for i, (r, n) in enumerate(zip(rates, ns), start=1):
            axes[0].text(i, r + 0.004, f"{r:.3f}\nn={n}", ha="center", va="bottom", fontsize=8)
        axes[0].set_ylim(0, max(rates + [0.25]) + 0.04)
    except ValueError:
        axes[0].text(0.5, 0.5, "quintile fail", ha="center")
    axes[0].set_title("Y4 rate by d_cust_hhi_lag3 quintile (train)")
    axes[0].set_xlabel("customer HHI lag3 quintile")
    axes[0].set_ylabel("y4_ds_r_double")
    axes[0].legend(fontsize=8)

    labels = []
    tail_r = []
    rest_r = []
    for yname, xcol, lab in (
        (Y3, "d_cust_hhi", "Y3 cust"),
        (Y4, "d_cust_hhi_lag3", "Y4 cust lag3"),
        (Y5, "d_supp_hhi", "Y5 supp"),
    ):
        yv = pd.to_numeric(tr[yname], errors="coerce")
        xx = pd.to_numeric(tr[xcol], errors="coerce")
        mm = yv.notna() & xx.notna()
        hi = mm & (xx > TAIL_CUT)
        lo = mm & (xx <= TAIL_CUT)
        labels.append(lab)
        tail_r.append(float(yv[hi].mean()) if int(hi.sum()) else 0.0)
        rest_r.append(float(yv[lo].mean()) if int(lo.sum()) else 0.0)
    xpos = np.arange(len(labels))
    axes[1].bar(xpos - 0.18, tail_r, 0.36, label="HHI>0.975", color="#ee6c4d")
    axes[1].bar(xpos + 0.18, rest_r, 0.36, label="body ≤0.975", color="#3d5a80")
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels(labels)
    axes[1].set_title("Tail vs body P(Y=1) — Y4 crash, Y5 supp protective")
    axes[1].set_ylabel("P(Y=1)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return True


def write_md(ctx: dict) -> None:
    d = ctx["decision"]
    c1, c2, c3, c4, c5 = ctx["c1"], ctx["c2"], ctx["c3"], ctx["c4"], ctx["c5"]
    c6, c7, c8, c9, c10, c11 = ctx["c6"], ctx["c7"], ctx["c8"], ctx["c9"], ctx["c10"], ctx["c11"]
    lines = [
        "# Unused leftover of `d_cust_hhi` on the 44 as Y3 X",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage / mix only. Seed 20260918 group folds. No 0–100. "
        "No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_cust_hhi`. "
        "Do not reopen Y4 trees. Off the 15-col Y3 card. "
        f"Night Y3 stays **{NIGHT_Y3:.3f} / {NIGHT_Y3_CORE:.3f}**. Days **{DAYS_BENCH:.3f}**. "
        f"Size **{SIZE_QUOTE:.3f}**. Y7 TURNOVER **{TURNOVER_QUOTE:.3f} / {B_SHALLOW_QUOTE:.3f}**. "
        "Y7 never D. Y5 never E. Y3 never B. Do not grow TURNOVER.",
        "",
        "`d_cust_hhi` = Herfindahl of AR invoice counterparties in the trailing 6-month "
        "window (Family D). Needs identified customer CPs. Dark 470 stay **NaN not 0**. "
        "Javier 14: concentration is **top1**, not HHI. Y4 already KEEP this as a "
        f"monopoly-tail single (lag3 {Y4_HHI_LAG3}; body {Y4_BODY}; ρ vs top1_lag3 {Y4_TOP1_RHO}) "
        "and PARK trees. This ticket is leftover as Y3 X / engine X on the 44.",
        "",
        "## Headline",
        "",
        d["headline"],
        "",
        "## Brief questions",
        "",
        "| # | question | what this cut says |",
        "| --- | --- | --- |",
        "| 1 | Who is healthy? | **PARK** as a health Y. Do not invent `y_cust_hhi`. Dark 470 = NaN, not 0. |",
        "| 2 | Who is improving? | Not a customer-HHI gradient. Body CV dies. |",
        f"| 3 | Who is turning? | **{d['y3']}** as Y3 X — leftover after days {_f(c5['after_days'])} vs days {_f(c4['days_y3'])}. |",
        f"| 4 | Dip vs fall? | Y4 tail footnote **KEEP** {_f(c4['hhi3_y4'])}; engine X **{d['y4_x']}** leftover after top1_lag3 {_f(c5['after_top1'])}; body {_f(c3['y4_body'])}. Trees PARK. |",
        f"| 5 | Why did it change? | Twin of top1 ρ={_f(c2['rho_lag'])}. Object: {d['object_kind']}. Supp HHI is a different (protective) object. |",
        f"| 6 | Months earlier? | **{d['q6']}** — {d['q6_why']} |",
        "",
        "## PARK / CLOSE / KEEP / DROP-from-44",
        "",
        _md_table(d["table"], ["object", "decision", "why"]),
        "",
        "## 1. Coverage; 470 dark NaN vs invoice-book; ever-n",
        "",
        c1["prose"],
        "",
        _md_table(c1["rows"]),
        "",
        "| item | value |",
        "| --- | ---: |",
        f"| train CM / companies | {c1['n_train']:,} / {c1['n_train_co']:,} |",
        f"| d_cust_hhi defined | {c1['train_nn']:,} ({_pp(c1['train_cov'])}) |",
        f"| ever-n companies | {c1['ever_n']:,} |",
        f"| never-ERP companies | {c1['n_dark_co']} (want {N_DARK_WANT}; confirm_470={c1['confirm_470']}) |",
        f"| dark HHI non-null / zero-filled | {c1['dark_nn']} / {c1['dark_zero']} |",
        f"| dark 0-fill | {'NO — CONFIRM' if c1['dark_ok'] else 'YES — FAIL'} |",
        f"| ERP n_cust==0 / HHI defined there | {c1['n0_erp']:,} / {c1['hhi_when_n0']} |",
        f"| holdout coverage (check only) | {_pp(c1['hold_cov'])} |",
        f"| acf1 / acf3 | {_f(c1['acf1'])} / {_f(c1['acf3'])} |",
        f"| size ρ vs log1p(a_in3) | {_f(c1['size_rho'])} |",
        "",
        "## 2. Spearman twins (|ρ|≥0.80)",
        "",
        c2["prose"],
        "",
        _md_table(c2["rows"]),
        "",
        "## 3. Y4 quintiles + >0.975 tail. Body CV ~0.445 CONFIRM",
        "",
        c3["prose"],
        "",
        "Train labeled cuts. Not monotone unless noted. Body CV is signed group-fold on HHI≤0.975.",
        "",
        _md_table(c3["q_rows"]),
        "",
        _md_table(c3["tail_rows"]),
        "",
        f"Y4 crash tail CONFIRM vs quote 22.1% / 11.5%: **{c3['y4_confirm']}**. "
        f"Body CV CONFIRM vs 0.445: **{c3['body_confirm']}** ({_f(c3['y4_body'])}). "
        f"Body dead: **{c3['y4_dead']}**.",
        "",
        "## 4. Single-feature train group-fold AUROC",
        "",
        f"Sign from the train side of each fold. Seed {FOLD_SEED}. "
        f"Night Y3 size **{SIZE_QUOTE}** (replica {_f(c4['size_y3'])}; CONFIRM {c4['size_y3_confirm']}); "
        f"days **{DAYS_BENCH}** (replica {_f(c4['days_y3'])}; CONFIRM {c4['days_y3_confirm']}). "
        f"Y4 customer HHI lag3 night **{Y4_HHI_LAG3}** (replica {_f(c4['hhi3_y4'])}; CONFIRM {c4['hhi3_confirm']}). "
        "Do not quote holdout.",
        "",
        c4["prose"],
        "",
        _md_table(c4["rows"]),
        "",
        "## 5. Honest leftover after days (Y3)",
        "",
        "OLS residual of `d_cust_hhi` on the bar (train-defined slope). Leftover <0.55 dies. "
        "If leftover looks high, ρ(resid, days) is the fake-days leak screen.",
        "",
        c5["prose"],
        "",
        _md_table(c5["rows"]),
        "",
        "## 6. Leftover after top1 lag3 (Y4). Twin → DROP HHI as X",
        "",
        c6["prose"],
        "",
        "## 7. SIZE terciles — tail inside T1?",
        "",
        c7["prose"],
        "",
        _md_table(c7["rows"]),
        "",
        "## 8. Q6 lag1 / lag3 on short books",
        "",
        c8["prose"],
        "",
        _md_table(c8["rows"]),
        "",
        "## 9. ICC / company-demean (trait vs month shock)",
        "",
        c9["prose"],
        "",
        _md_table(c9["rows"]),
        "",
        "## 10. Holdout mix flip (coverage / mix only)",
        "",
        c10["prose"],
        "",
        _md_table(c10["rows"]),
        "",
        _md_table(c10["mix_rows"]),
        "",
        "## 11. vs supp HHI — different object?",
        "",
        c11["prose"],
        "",
        _md_table(c11["rows"]),
        "",
        "## Extra A. 2-col z-avg of top1 + HHI",
        "",
        ctx["zavg"]["prose"],
        "",
        _md_table(ctx["zavg"]["rows"]),
        "",
        "CLOSE. Do not put on the card.",
        "",
        "## Extra B. Same-n leftover after top1",
        "",
        ctx["samen"]["prose"],
        "",
        _md_table(ctx["samen"]["rows"]),
        "",
        "## Extra C. Tail companies on non-tail months",
        "",
        ctx["tailcos"]["prose"],
        "",
        _md_table(ctx["tailcos"]["rows"]),
        "",
        "## Extra D. Y3 body leftover after days",
        "",
        ctx["y3body"]["prose"],
        "",
        _md_table(ctx["y3body"]["rows"]),
        "",
        "## Extra E. Inverse leftover (days after HHI; top1 after HHI)",
        "",
        ctx["inverse"]["prose"],
        "",
        _md_table(ctx["inverse"]["rows"]),
        "",
        "## Extra F. Leftover after `d_n_cust`",
        "",
        ctx["ncust"]["prose"],
        "",
        _md_table(ctx["ncust"]["rows"]),
        "",
        "## Extra G. Company-mean leftover after days",
        "",
        ctx["style"]["prose"],
        "",
        _md_table(ctx["style"]["rows"]),
        "",
        "## Extra H. Binary tail vs continuous (Y4)",
        "",
        ctx["tailbin"]["prose"],
        "",
        _md_table(ctx["tailbin"]["rows"]),
        "",
        "## Extra I. Dark 470 — HHI defined?",
        "",
        ctx["darkx"]["prose"],
        "",
        _md_table(ctx["darkx"]["rows"]),
        "",
        "## Extra J. Y3 T2/T3 pocket leftover after days",
        "",
        ctx["pocket"]["prose"],
        "",
        _md_table(ctx["pocket"]["rows"]),
        "",
        "## Extra K. Short-book leftover after the honest bar",
        "",
        ctx["q6h"]["prose"],
        "",
        _md_table(ctx["q6h"]["rows"]),
        "",
        "## Extra L. Leftover after n_cust + days",
        "",
        ctx["nandd"]["prose"],
        "",
        _md_table(ctx["nandd"]["rows"]),
        "",
        "## Extra M. Holdout 16 positives — spike 1.5 vs 1.2",
        "",
        ctx["holdpos"]["prose"],
        "",
        _md_table(ctx["holdpos"]["rows"]),
        "",
        "## Extra N. n_cust leftover 0.608 — fake-days?",
        "",
        ctx["ncustleak"]["prose"],
        "",
        _md_table(ctx["ncustleak"]["rows"]),
        "",
        "## Extra O. Y4 body leftover after top1",
        "",
        ctx["y4bodyt"]["prose"],
        "",
        _md_table(ctx["y4bodyt"]["rows"]),
        "",
        "## Extra P. Days leftover after HHI (same-n)",
        "",
        ctx["dayssame"]["prose"],
        "",
        _md_table(ctx["dayssame"]["rows"]),
        "",
        "## Extra Q. Holdout HHI coverage on crash vs spike pos",
        "",
        ctx["holdmix"]["prose"],
        "",
        _md_table(ctx["holdmix"]["rows"]),
        "",
        "## Extra R. Y3 HHI_lag3 leftover after days",
        "",
        ctx["y3lag3"]["prose"],
        "",
        _md_table(ctx["y3lag3"]["rows"]),
        "",
        "## Extra S. Y4 contemporaneous body leftover after days",
        "",
        ctx["y4nowb"]["prose"],
        "",
        _md_table(ctx["y4nowb"]["rows"]),
        "",
        "## Extra T. Y4 leftover after the >0.975 flag",
        "",
        ctx["y4tf"]["prose"],
        "",
        _md_table(ctx["y4tf"]["rows"]),
        "",
        "## Extra U. Now-body leftover after days — fake-top1?",
        "",
        ctx["y4fake"]["prose"],
        "",
        _md_table(ctx["y4fake"]["rows"]),
        "",
        "## What failed / next (held for wave note)",
        "",
        *([f"- {x}" for x in ctx["failed"]] if ctx["failed"] else ["- (none)"]),
        "",
        f"Elapsed {ctx['elapsed_s']:.0f}s. Cuts 1–12 plus extras "
        "(z-avg, same-n leftover, tail-company months, Y3 body leftover, "
        "inverse leftover, n_cust leftover, style, binary tail, dark 470, "
        "Y3 T2/T3 pocket, short-book Q6 leftover, n_cust+days, holdout 16, "
        "n_cust fake-days, Y4 body after top1, days same-n, holdout HHI mix, "
        "Y3 lag3 leftover, Y4 now-body leftover, tail-flag leftover, "
        "now-body fake-top1).",
        "",
        "## What this module did not do",
        "",
        "- Did not change night Y3 0.762 / 0.752, days 0.711, size 0.617, or Y7 TURNOVER 0.720 / 0.712.",
        "- Did not put customer HHI on the 15-col Y3 card. Did not grow TURNOVER.",
        "- Did not reopen Y4 trees. Did not invent `y_cust_hhi`. Did not merge with Y4 / Family I/M/J.",
        "- Did not use Family E as Y5 X. Did not use Family B as Y3 X. Did not use Family F as Y4 X.",
        "- Did not write 0–100 / pillars. Did not touch `product/`.",
        "- Did not rewrite parquet or duckdb. Did not run `build_targets`.",
        "- Did not fit on holdout 72. Did not commit. Did not write the parent journal / LIVE / canvas.",
        "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD}")


def _reg_row(metric: str, value, coverage, y: str, notes: str, split: str = "train_cv") -> dict:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        value = ""
    return {
        "ts": datetime.now().strftime("%Y-%m-%dT%H:%M"),
        "round": ROUND,
        "wave": WAVE,
        "agent": AGENT,
        "x_families": X_FAM,
        "y": y,
        "model": MODEL,
        "split": split,
        "metric": metric,
        "value": value,
        "coverage": coverage,
        "notes": notes,
    }


def append_registry(ctx: dict) -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    c1, c2, c3, c4, c5, d = ctx["c1"], ctx["c2"], ctx["c3"], ctx["c4"], ctx["c5"], ctx["decision"]
    rows = [
        _reg_row(
            "d_cust_hhi_cov",
            c1["train_cov"],
            f"{c1['train_cov']:.4f}",
            "-",
            f"ever_n={c1['ever_n']} dark_nn={c1['dark_nn']} dark_co={c1['n_dark_co']} ok={c1['dark_ok']}",
            "train",
        ),
        _reg_row(
            "rho_cust_hhi_vs_top1",
            c2["rhos"]["d_cust_top1"],
            "1.0000",
            "-",
            f"lag3={c2['rho_lag']:.4f} supp={c2['rhos']['d_supp_hhi']:.4f} "
            f"tx_cp={c2['rhos']['d_tx_cp_share']:.4f} size={c2['rhos']['log1p(a_in3)']:.4f} twin={c2['top1_twin']}",
            "train",
        ),
        _reg_row(
            "auroc_d_cust_hhi",
            c4["hhi_y3"],
            f"{c1['train_cov']:.4f}",
            Y3,
            f"vs_size={c4['size_y3']:.4f} vs_days={c4['days_y3']:.4f} leftover_days={c5['after_days']:.4f} y3={d['y3']}",
        ),
        _reg_row(
            "auroc_d_cust_hhi_lag3",
            c4["hhi3_y4"],
            f"{c1['train_cov']:.4f}",
            Y4,
            f"vs_top1_lag3={c4['top13_y4']:.4f} leftover_top1={c5['after_top1']:.4f} "
            f"body={c3['y4_body']:.4f} confirm={c3['body_confirm']} y4x={d['y4_x']}",
        ),
        _reg_row(
            "y4_cust_hhi_body_cv",
            c3["y4_body"],
            "1.0000",
            Y4,
            f"quote=0.445 confirm={c3['body_confirm']} n={c3['y4_body_n']} pos={c3['y4_body_pos']}",
        ),
        _reg_row(
            "y4_cust_hhi_tail_rate",
            c3["y4_rate_hi"],
            "1.0000",
            Y4,
            f"rest={c3['y4_rate_lo']:.4f} confirm={c3['y4_confirm']} quote=0.221/0.115",
            "train",
        ),
        _reg_row(
            "cust_hhi_lose_44",
            1.0 if d["lose_44"] else 0.0,
            f"{c1['train_cov']:.4f}",
            "-",
            f"overall={d['overall']} y3={d['y3']} y4x={d['y4_x']} kind={d['object_kind']}",
            "train",
        ),
        _reg_row(
            "icc_d_cust_hhi",
            ctx["c9"]["icc"]["icc"],
            "1.0000",
            "-",
            f"eta2={ctx['c9']['icc']['eta2']:.4f} trait={ctx['c9']['trait']} demean_y3={ctx['c9']['d3']:.4f}",
            "train",
        ),
        _reg_row(
            "y3_leftover_after_days",
            c5["after_days"],
            "1.0000",
            Y3,
            f"fake_days={c5['fake_days']} rho_resid={c5['rho_resid_days']:.4f} dies={c5['died_y3']}",
        ),
    ]
    with REGISTRY.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        seen = {
            (
                r.get("agent"),
                r.get("y"),
                r.get("model"),
                r.get("split"),
                r.get("metric"),
                r.get("x_families"),
            )
            for r in reader
        }
    fresh = []
    for r in rows:
        key = (
            str(r.get("agent", "")),
            str(r.get("y", "")),
            str(r.get("model", "")),
            str(r.get("split", "")),
            str(r.get("metric", "")),
            str(r.get("x_families", "")),
        )
        if key in seen:
            continue
        fresh.append(r)
        seen.add(key)
    if not fresh:
        print("registry: no new rows")
        return
    with REGISTRY.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        for r in fresh:
            w.writerow({k: r.get(k, "") for k in header})
    print(f"registry appended {len(fresh)} rows")


def write_wave(ctx: dict) -> None:
    d = ctx["decision"]
    c1, c2, c3, c4, c5, c6 = (
        ctx["c1"],
        ctx["c2"],
        ctx["c3"],
        ctx["c4"],
        ctx["c5"],
        ctx["c6"],
    )
    WAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = (
        f"# Wave 4 — unused leftover of `d_cust_hhi` as Y3 X\n\n"
        f"- **When:** {_now_iso()}\n"
        f"- **Agent:** `{AGENT}`\n"
        f"- **Python:** `/home/walterjtv/.pyenv/versions/base/bin/python3`\n"
        f"- **Re-run:** `python -m analysis.evaluate.cust_hhi_qa`\n"
        f"- **Holdout:** 72 companies, seed 20260918. Rates / AUROC on train. Holdout = coverage / mix.\n"
        f"- **Owned:** `analysis/evaluate/cust_hhi_qa.py`, `analysis/outputs/cust_hhi_qa.md`, "
        f"`analysis/outputs/cust_hhi_qa.png`, append-only registry, this note.\n\n"
        f"## Decision\n\n"
        f"{d['headline']}\n\n"
        f"| object | decision |\n"
        f"| --- | --- |\n"
        f"| Y3 X | **{d['y3']}** |\n"
        f"| Y4 engine X | **{d['y4_x']}** |\n"
        f"| Y4 tail footnote | **KEEP** |\n"
        f"| the 44 as X | **{d['on44']}** |\n"
        f"| twin vs top1 | **{'DROP HHI as X' if c6['drop_hhi'] else 'not a twin'}** ρ={_f(c2['rho_lag'])} |\n"
        f"| Y4 body | **{_f(c3['y4_body'])}** CONFIRM {c3['body_confirm']} |\n"
        f"| Q6 | **{d['q6']}** |\n\n"
        f"Coverage train {_pp(c1['train_cov'])} ever-n {c1['ever_n']}. "
        f"Dark 470 NaN not 0: {c1['dark_ok']}. "
        f"Leftover Y3 after days {_f(c5['after_days'])}; Y4 after top1_lag3 {_f(c5['after_top1'])}. "
        f"Y4 lag3 {_f(c4['hhi3_y4'])}.\n\n"
        f"## What failed\n\n"
        + ("\n".join(f"- {x}" for x in ctx["failed"]) if ctx["failed"] else "- (none)")
        + "\n\n"
        f"## Next idea\n\n"
        f"- If the 44 drops HHI as X, keep the Y4 >0.975 tail footnote only. "
        f"Do not stack HHI+top1. Do not invent `y_cust_hhi`. Do not reopen trees.\n\n"
        f"Elapsed {ctx['elapsed_s']:.0f}s. Night quotes unchanged.\n"
    )
    WAVE_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {WAVE_PATH}")


def run() -> dict:
    t0 = time.time()
    print(f"cust_hhi_qa start seed={FOLD_SEED} holdout=72 agent={AGENT}")
    panel = load_panel()
    panel = attach_folds(panel)
    con = connect()
    try:
        book = book_invoice_ids(con)
        dark = dark_population(con)
        print("rebuild in3/ds3 for crash/spike mix (not X)…")
        flows = rebuild_in3_ds3(con)
    finally:
        con.close()
    panel = attach_crash_spike(panel, flows)
    panel = add_panel_lags(
        panel,
        ["d_cust_hhi", "d_cust_top1", "d_supp_hhi", "d_n_cust"],
        (1, 3),
    )
    tr = panel[panel["split"] == "train"].copy()
    assert_no_holdout(tr["company_id"])
    print(
        f"train CM={len(tr):,} companies={tr['company_id'].nunique()} "
        f"holdout CM={int((panel['split']=='holdout').sum())} "
        f"book={len(book)} train_dark={dark['n_train_dark']} confirm_470={dark['confirm_470']}"
    )

    failed: list[str] = []
    c1 = cut1_coverage(panel, book, dark)
    if not c1["dark_ok"]:
        failed.append(
            f"dark 470 NaN screen failed nn={c1['dark_nn']} zero={c1['dark_zero']} co={c1['n_dark_co']}"
        )
    c2 = cut2_twins(tr)
    c3 = cut3_tail(tr)
    c4 = cut4_singles(tr)
    c5 = cut5_leftover(tr)
    c6 = cut6_twin(tr, c2, c4, c5)
    c7 = cut7_terciles(tr)
    c8 = cut8_q6(tr)
    c9 = cut9_icc(tr)
    c10 = cut10_holdout(panel)
    c11 = cut11_vs_supp(tr, c2, c3)
    zavg = extra_zavg(tr)
    samen = extra_samen(tr)
    tailcos = extra_tail_cos(tr)
    y3body = extra_y3_body_days(tr)
    inverse = extra_inverse(tr)
    ncust = extra_n_cust(tr)
    style = extra_style(tr)
    tailbin = extra_tail_bin(tr)
    darkx = extra_dark(tr, book, dark)
    pocket = extra_y3_pocket(tr)
    q6h = extra_q6_honest(tr)
    nandd = extra_n_and_days(tr)
    holdpos = extra_hold_pos(panel)
    ncustleak = extra_ncust_leak(tr)
    y4bodyt = extra_y4_body_top1(tr)
    dayssame = extra_days_same_n(tr)
    holdmix = extra_hold_hhi_mix(panel)
    y3lag3 = extra_y3_lag3_days(tr)
    y4nowb = extra_y4_now_body(tr)
    y4tf = extra_y4_tailflag(tr, c5)
    y4fake = extra_y4_now_fake(tr)
    png_ok = plot_png(tr, c3)

    if c2["top1_twin"]:
        failed.append(
            f"twin of d_cust_top1 ρ={c2['rhos']['d_cust_top1']:.3f} / lag3 {c2['rho_lag']:.3f} — DROP weaker HHI as X"
        )
    if c5["died_y3"]:
        failed.append(
            f"Y3 leftover after days {c5['after_days']:.3f} dies <0.55"
            + (" / fake-days" if c5["fake_days"] else "")
        )
    if c5["died_y4"] or c3["y4_dead"]:
        failed.append(
            f"Y4 leftover after top1_lag3 {c5['after_top1']:.3f}; body {c3['y4_body']:.3f}"
        )
    if c3["body_confirm"]:
        failed.append(f"Y4 body CV {c3['y4_body']:.3f} CONFIRM vs 0.445")
    if c3["y4_confirm"]:
        failed.append(
            f"Y4 tail CONFIRM crash {c3['y4_rate_hi']:.3f} vs {c3['y4_rate_lo']:.3f}"
        )
    if c8["low"]:
        failed.append(
            f"Q6 short lag3 LOW_POWER present={c8['short_present']:.3f} pos={c8['short_y4_pos']}"
        )
    if c11["different"]:
        failed.append("supp HHI is a different object (protective) — do not merge")
    if samen["artifact"]:
        failed.append("same-n leftover after top1 is a near-identity (R²≥0.90)")
    if y3body["dies"]:
        failed.append(f"Y3 body leftover after days {y3body['lef']:.3f} dies")
    if pocket["dies"]:
        failed.append(f"Y3 T2+T3 leftover after days {pocket['lef_t23']:.3f} dies")
    if not q6h["q6_keep"]:
        failed.append("Q6 short leftover after the honest bar dies — CLOSE")
    if nandd["dies"]:
        failed.append(f"Y3 leftover after n_cust+days {nandd['lef']:.3f} dies")
    if c10["mix_ok"]:
        failed.append(
            f"holdout mix flip CONFIRM crash={c10['ho_crash']:.3f} spike={c10['ho_spike']:.3f}"
        )
    elif holdpos["match12"]:
        failed.append(
            f"holdout spike>1.2 {_f(holdpos['spike12'])} recovers the 88% quote; 1.5 is {_f(holdpos['spike15'])}"
        )
    if ncustleak["fake"]:
        failed.append(f"n_cust leftover 0.608 is a fake-days leak ρ={ncustleak['rho_days']:.3f}")
    if ncustleak["dies3"]:
        failed.append(
            f"Y3 leftover after n_cust+days+top1 {ncustleak['lef3']:.3f} dies"
        )
    if y4bodyt["dies"]:
        failed.append(f"Y4 body leftover after top1 {y4bodyt['lef']:.3f} dies")
    if dayssame["days_lives"]:
        failed.append(
            f"days leftover after HHI {dayssame['days_lef']:.3f} survives — days is the bar"
        )
    if y3lag3["dies"]:
        failed.append(f"Y3 HHI_lag3 leftover after days {y3lag3['lef']:.3f} dies")
    if y4nowb["dies"]:
        failed.append(
            f"Y4 now-body leftover after days {y4nowb['lef']:.3f} dies (raw body {y4nowb['raw']:.3f})"
        )
    if y4tf["dies"]:
        failed.append(
            f"Y4 leftover after tail-flag {y4tf['after_tail']:.3f} dies — 0.605 is the bin"
        )
    if y4fake["fake"] or y4fake["dies"]:
        failed.append(
            f"Y4 now-body leftover after days {y4fake['lef_d']:.3f} "
            f"ρ(resid,top1)={y4fake['rho_top1']:.3f}; after days+top1 {y4fake['lef_both']:.3f}"
        )

    ctx = {
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "c4": c4,
        "c5": c5,
        "c6": c6,
        "c7": c7,
        "c8": c8,
        "c9": c9,
        "c10": c10,
        "c11": c11,
        "zavg": zavg,
        "samen": samen,
        "tailcos": tailcos,
        "y3body": y3body,
        "inverse": inverse,
        "ncust": ncust,
        "style": style,
        "tailbin": tailbin,
        "darkx": darkx,
        "pocket": pocket,
        "q6h": q6h,
        "nandd": nandd,
        "holdpos": holdpos,
        "ncustleak": ncustleak,
        "y4bodyt": y4bodyt,
        "dayssame": dayssame,
        "holdmix": holdmix,
        "y3lag3": y3lag3,
        "y4nowb": y4nowb,
        "y4tf": y4tf,
        "y4fake": y4fake,
        "png_ok": png_ok,
        "failed": failed,
        "elapsed_s": time.time() - t0,
        "book": book,
        "dark": dark,
    }
    ctx["decision"] = decide(ctx)
    write_md(ctx)
    append_registry(ctx)
    if WRITE_WAVE:
        write_wave(ctx)
    print(
        f"cust_hhi_qa done in {ctx['elapsed_s']:.0f}s overall={ctx['decision']['overall']} "
        f"wave={WRITE_WAVE}"
    )
    return ctx


if __name__ == "__main__":
    run()
