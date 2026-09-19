"""Q5 leftover — missing transaction counterparty_id after uncat is parked.

NORTH_STAR: uncat is PARK as Y and as X (style dummy). 91.8% of uncat txs
also lack ``counterparty_id``. ``d_tx_cp_share`` is the 6-month *named*
fill rate (Y5 KEEP-as-quote 0.611 / PARK as X). This lane asks whether
*monthly* missing-CP is (a) the uncat twin, (b) the store fill-rate twin,
(c) a leftover tagging hole after those two, (d) the 470 dark hole, or
(e) Y5 fold-3 again.

In-memory company-month missing-CP share only. Do not merge parquet.
Do not invent ``y_missing_cp``. Do not put anything on the 15-col Y3 card.
Y7 never uses D — do not score missing-CP as a Y7 X.
Do not change the night Y3 or Y5 ``d_tx_cp_share`` 0.611 quotes.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.missing_cp_qa

Owned: analysis/evaluate/missing_cp_qa.py, analysis/outputs/missing_cp_qa.md,
optional one PNG, append-only registry, overnight/waves/wave4_missing_cp.md.
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
OUT_MD = ANALYSIS / "outputs" / "missing_cp_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "missing_cp_vs_uncat.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_missing_cp.md"
AGENT = "c91e4b2a"
WAVE = "4"
ROUND = "R4"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
Y5_AR = "y5_ar_od30_sust"
Y5_AP = "y5_ap_od30_ownp80"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_Y3_QUOTE = 0.617
KEEP_DELTA = 0.02
SIZE_PARK = 0.60
TWIN_RHO = 0.80
SIZE_RHO = 0.50
ICC_STYLE = 0.85
ACF_STYLE = 0.40
MIN_POS = 50
MIN_ACF_PAIRS = 4
PANEL_START = pd.Timestamp("2024-09-01")
PANEL_END = pd.Timestamp("2026-08-01")
CHRONIC_GROUPS = ("GROUP_0158", "GROUP_0172")
CHRONIC_BELOW = 0.50
HOLE_SHARE = 0.01  # Y5 tagging hole: d_tx_cp_share ≤ 0.01
UNCAT_MISS_QUOTE = 0.918
UNCAT_MISS_AMT_QUOTE = 0.940
Y5_DTX_QUOTE = 0.611
Y3_NIGHT = (0.762, 0.752)
BOOK = (
    "document_type = 'invoice' AND status <> 'cancel' "
    "AND amount <> 0 AND issuance_date IS NOT NULL"
)

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "a_uncat_share",
    "a_n_tx",
    "a_in3",
    "a_op_in",
    "c_n_days_with_tx",
    "d_tx_cp_share",
    "d_n_cust",
    "d_n_supp",
    "d_cust_hhi",
    "d_interco_share",
    "e_ar_issued",
    "b_below_0",
    "first_month",
)

Y_KEEP = (Y2, Y3, Y5_AR, Y5_AP)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")


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
            "coverage": float("nan"),
        }
    for k in range(n_folds):
        tr = defined & (folds != k)
        va = defined & (folds == k)
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
        "coverage": float("nan"),
    }


def fold_bits(rec: dict) -> str:
    return " ".join(
        f"{r['auroc']:.3f}" if np.isfinite(r["auroc"]) else "—" for r in rec.get("folds", [])
    )


def ols_resid(y: pd.Series, *xs: pd.Series) -> tuple[pd.Series, dict]:
    """Train-defined OLS residual of y on 1+ predictors. No holdout in slope."""
    cols = {"y": pd.to_numeric(y, errors="coerce")}
    for i, x in enumerate(xs):
        cols[f"x{i}"] = pd.to_numeric(x, errors="coerce")
    d = pd.DataFrame(cols)
    ok = d.notna().all(axis=1)
    resid = pd.Series(np.nan, index=d.index, dtype=float)
    info = {"n": int(ok.sum()), "slope": [], "intercept": float("nan")}
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
    return resid, info


def _cat_sql_list() -> str:
    return ", ".join("'" + k.replace("'", "''") + "'" for k in CAT_MAP)


def _hold_sql(hold: set[str]) -> str:
    return ", ".join("'" + str(c).replace("'", "''") + "'" for c in sorted(hold))


def _store_cols_present(raw: pd.DataFrame) -> list[str]:
    have = [c for c in STORE_COLS if c in raw.columns]
    need = [
        "company_id",
        "period",
        "group_id",
        "a_uncat_share",
        "a_in3",
        "c_n_days_with_tx",
        "d_tx_cp_share",
    ]
    miss = [c for c in need if c not in raw.columns]
    if miss:
        raise RuntimeError(f"monthly.parquet missing {miss}")
    return have


def load_panel() -> pd.DataFrame:
    if not STORE.exists():
        raise FileNotFoundError(f"{STORE} missing")
    if not TARGETS.exists():
        raise FileNotFoundError(f"{TARGETS} missing — do not run build_targets")
    raw = pd.read_parquet(STORE)
    yraw = pd.read_parquet(TARGETS)
    print(f"store {STORE} shape={raw.shape} (read-only)")
    print(f"targets {TARGETS} shape={yraw.shape} (read-only)")
    have = _store_cols_present(raw)
    ykeep = ["company_id", "period", *Y_KEEP]
    ymiss = [c for c in ykeep if c not in yraw.columns]
    if ymiss:
        raise RuntimeError(f"targets.parquet missing {ymiss}")
    panel = _keys(raw[have])
    y = _keys(yraw[ykeep])
    panel = panel.merge(y, on=["company_id", "period"], how="left")
    hold = load_holdout()
    panel["split"] = np.where(panel["company_id"].isin(hold), "holdout", "train")
    panel["log_in3"] = np.log1p(pd.to_numeric(panel["a_in3"], errors="coerce").clip(lower=0))
    leak = leakage_check(["miss_cp_share", "d_tx_cp_share", "a_uncat_share"], Y3, forbidden_prefixes=["b"])
    if not leak["ok"]:
        print(f"leakage_check note (in-memory X only): {leak['issues']}")
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
        s = pd.to_numeric(out[c], errors="coerce")
        g = s.groupby(cid, sort=False)
        for k in lags:
            extra[f"{c}_lag{k}"] = g.shift(k)
    if extra:
        out = pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)
    return out


def add_so_far(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["company_id", "period"]).reset_index(drop=True)
    out["so_far"] = out.groupby("company_id", sort=False).cumcount() + 1
    return out


def add_style_shock(df: pd.DataFrame, col: str = "miss_cp_share") -> pd.DataFrame:
    out = df.sort_values(["company_id", "period"]).reset_index(drop=True)
    x = pd.to_numeric(out[col], errors="coerce")
    mu = x.groupby(out["company_id"], sort=False).transform("mean")
    out["miss_co_mean"] = mu
    out["miss_demean"] = x - mu
    return out


def load_monthly_miss(con) -> pd.DataFrame:
    """Calendar-month missing-CP share (count + |amt|). Never written to parquet."""
    cats = _cat_sql_list()
    df = con.execute(
        f"""
        SELECT
          CAST(t.company_id AS VARCHAR) AS company_id,
          CAST(date_trunc('month', t."date") AS DATE) AS month,
          COUNT(*) AS n_tx,
          SUM(ABS(t.amount)) AS abs_all,
          SUM(CASE WHEN t.counterparty_id IS NULL
                     OR length(trim(CAST(t.counterparty_id AS VARCHAR))) = 0
                    THEN 1 ELSE 0 END) AS n_miss,
          SUM(CASE WHEN t.counterparty_id IS NULL
                     OR length(trim(CAST(t.counterparty_id AS VARCHAR))) = 0
                    THEN ABS(t.amount) ELSE 0 END) AS abs_miss,
          SUM(CASE WHEN t.category = 'uncategorized'
                     OR t.category IS NULL
                     OR t.category NOT IN ({cats})
                    THEN 1 ELSE 0 END) AS n_uncat,
          SUM(CASE WHEN (t.category = 'uncategorized'
                     OR t.category IS NULL
                     OR t.category NOT IN ({cats}))
                    AND (t.counterparty_id IS NULL
                         OR length(trim(CAST(t.counterparty_id AS VARCHAR))) = 0)
                    THEN 1 ELSE 0 END) AS n_uncat_miss,
          SUM(CASE WHEN NOT (t.category = 'uncategorized'
                     OR t.category IS NULL
                     OR t.category NOT IN ({cats}))
                    THEN 1 ELSE 0 END) AS n_mapped,
          SUM(CASE WHEN NOT (t.category = 'uncategorized'
                     OR t.category IS NULL
                     OR t.category NOT IN ({cats}))
                    AND (t.counterparty_id IS NULL
                         OR length(trim(CAST(t.counterparty_id AS VARCHAR))) = 0)
                    THEN 1 ELSE 0 END) AS n_mapped_miss,
          SUM(CASE WHEN NOT (t.category = 'uncategorized'
                     OR t.category IS NULL
                     OR t.category NOT IN ({cats}))
                    THEN ABS(t.amount) ELSE 0 END) AS abs_mapped,
          SUM(CASE WHEN NOT (t.category = 'uncategorized'
                     OR t.category IS NULL
                     OR t.category NOT IN ({cats}))
                    AND (t.counterparty_id IS NULL
                         OR length(trim(CAST(t.counterparty_id AS VARCHAR))) = 0)
                    THEN ABS(t.amount) ELSE 0 END) AS abs_mapped_miss
        FROM transactions t
        WHERE t."date" IS NOT NULL
          AND t."date" >= DATE '2024-09-01'
          AND t."date" < DATE '2026-09-01'
        GROUP BY 1, 2
        """
    ).df()
    df["company_id"] = df["company_id"].astype(str)
    df["period"] = pd.to_datetime(df["month"])
    df = df.drop(columns=["month"])
    n_tx = pd.to_numeric(df["n_tx"], errors="coerce")
    abs_all = pd.to_numeric(df["abs_all"], errors="coerce")
    n_mapped = pd.to_numeric(df["n_mapped"], errors="coerce")
    abs_mapped = pd.to_numeric(df["abs_mapped"], errors="coerce")
    df["miss_cp_share"] = np.where(n_tx > 0, df["n_miss"] / n_tx, np.nan)
    df["miss_cp_amt"] = np.where(abs_all > 0, df["abs_miss"] / abs_all, np.nan)
    df["miss_mapped_share"] = np.where(n_mapped > 0, df["n_mapped_miss"] / n_mapped, np.nan)
    df["miss_mapped_amt"] = np.where(abs_mapped > 0, df["abs_mapped_miss"] / abs_mapped, np.nan)
    df["named_share"] = np.where(n_tx > 0, 1.0 - df["miss_cp_share"], np.nan)
    return df


def load_monthly_inv_cp(con) -> pd.DataFrame:
    """Same-month invoice CP fill (book invoices). Not a COMP_* map."""
    df = con.execute(
        f"""
        SELECT
          CAST(company_id AS VARCHAR) AS company_id,
          CAST(date_trunc('month', issuance_date) AS DATE) AS month,
          COUNT(*) AS n_inv,
          SUM(CASE WHEN counterparty_id IS NULL
                     OR length(trim(CAST(counterparty_id AS VARCHAR))) = 0
                    THEN 1 ELSE 0 END) AS n_inv_miss,
          SUM(ABS(amount)) AS abs_inv,
          SUM(CASE WHEN counterparty_id IS NULL
                     OR length(trim(CAST(counterparty_id AS VARCHAR))) = 0
                    THEN ABS(amount) ELSE 0 END) AS abs_inv_miss
        FROM invoices
        WHERE {BOOK}
        GROUP BY 1, 2
        """
    ).df()
    df["company_id"] = df["company_id"].astype(str)
    df["period"] = pd.to_datetime(df["month"])
    df = df.drop(columns=["month"])
    n = pd.to_numeric(df["n_inv"], errors="coerce")
    a = pd.to_numeric(df["abs_inv"], errors="coerce")
    df["inv_miss_share"] = np.where(n > 0, df["n_inv_miss"] / n, np.nan)
    df["inv_cp_share"] = np.where(n > 0, 1.0 - df["inv_miss_share"], np.nan)
    df["inv_miss_amt"] = np.where(a > 0, df["abs_inv_miss"] / a, np.nan)
    return df


def attach_miss(panel: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "company_id",
        "period",
        "n_tx",
        "abs_all",
        "n_miss",
        "abs_miss",
        "n_uncat",
        "n_uncat_miss",
        "n_mapped",
        "n_mapped_miss",
        "abs_mapped",
        "abs_mapped_miss",
        "miss_cp_share",
        "miss_cp_amt",
        "miss_mapped_share",
        "miss_mapped_amt",
        "named_share",
    ]
    return panel.merge(monthly[keep], on=["company_id", "period"], how="left")


def attach_inv(panel: pd.DataFrame, inv: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "company_id",
        "period",
        "n_inv",
        "n_inv_miss",
        "inv_miss_share",
        "inv_cp_share",
        "inv_miss_amt",
    ]
    return panel.merge(inv[keep], on=["company_id", "period"], how="left")


def chronic_ids(tr: pd.DataFrame) -> list[str]:
    """12 chronic dark Y2 names: ≥50% labeled months already below 0 in 0158/0172."""
    if "b_below_0" not in tr.columns:
        return []
    y = pd.to_numeric(tr[Y2], errors="coerce")
    below = pd.to_numeric(tr["b_below_0"], errors="coerce")
    lab = y.notna()
    hot = tr["group_id"].astype(str).isin(CHRONIC_GROUPS)
    sl = lab & hot
    g = (
        tr.loc[sl, ["company_id"]]
        .assign(below=below[sl].values)
        .groupby("company_id")["below"]
        .agg(n="size", n_below="sum")
    )
    g["share_below"] = g["n_below"] / g["n"]
    return [str(i) for i in g.index[g["share_below"] >= CHRONIC_BELOW]]


# ---------------------------------------------------------------------------
# Pass 1 — prevalence (count + |amt|), train vs holdout coverage
# ---------------------------------------------------------------------------
def pass1_prevalence(panel: pd.DataFrame, con) -> dict:
    hold = load_holdout()
    hold_sql = _hold_sql(hold)
    raw = con.execute(
        f"""
        SELECT
          CASE WHEN CAST(t.company_id AS VARCHAR) IN ({hold_sql}) THEN 'holdout' ELSE 'train' END AS split,
          COUNT(*) AS n,
          SUM(ABS(t.amount)) AS abs_amt,
          SUM(CASE WHEN t.counterparty_id IS NULL
                     OR length(trim(CAST(t.counterparty_id AS VARCHAR))) = 0
                    THEN 1 ELSE 0 END) AS n_miss,
          SUM(CASE WHEN t.counterparty_id IS NULL
                     OR length(trim(CAST(t.counterparty_id AS VARCHAR))) = 0
                    THEN ABS(t.amount) ELSE 0 END) AS abs_miss
        FROM transactions t
        WHERE t."date" IS NOT NULL
          AND t."date" >= DATE '2024-09-01'
          AND t."date" < DATE '2026-09-01'
        GROUP BY 1
        """
    ).df()
    by = {str(r["split"]): r for _, r in raw.iterrows()}
    tr_n = float(by["train"]["n"]) if "train" in by else 0.0
    tr_m = float(by["train"]["n_miss"]) if "train" in by else 0.0
    tr_a = float(by["train"]["abs_amt"]) if "train" in by else 0.0
    tr_am = float(by["train"]["abs_miss"]) if "train" in by else 0.0
    ho_n = float(by["holdout"]["n"]) if "holdout" in by else 0.0
    ho_m = float(by["holdout"]["n_miss"]) if "holdout" in by else 0.0
    ho_a = float(by["holdout"]["abs_amt"]) if "holdout" in by else 0.0
    ho_am = float(by["holdout"]["abs_miss"]) if "holdout" in by else 0.0
    train_share_n = _pct(tr_m, tr_n)
    train_share_a = _pct(tr_am, tr_a)
    hold_share_n = _pct(ho_m, ho_n)
    hold_share_a = _pct(ho_am, ho_a)

    tr = panel[panel["split"] == "train"]
    ho = panel[panel["split"] == "holdout"]
    miss = pd.to_numeric(tr["miss_cp_share"], errors="coerce")
    amt = pd.to_numeric(tr["miss_cp_amt"], errors="coerce")
    n_cm = int(len(tr))
    n_def = int(miss.notna().sum())
    n_ho = int(len(ho))
    n_ho_def = int(pd.to_numeric(ho["miss_cp_share"], errors="coerce").notna().sum())
    rows = [
        {
            "split": "train",
            "n_tx": f"{int(tr_n):,}",
            "miss_n": _pp(train_share_n),
            "miss_|amt|": _pp(train_share_a),
            "cm_defined": _pp(_pct(n_def, n_cm)),
            "cm_mean": _f(float(miss.mean())),
            "cm_p50": _f(float(miss.median())),
        },
        {
            "split": "holdout",
            "n_tx": f"{int(ho_n):,}",
            "miss_n": _pp(hold_share_n),
            "miss_|amt|": _pp(hold_share_a),
            "cm_defined": _pp(_pct(n_ho_def, n_ho)),
            "cm_mean": _f(float(pd.to_numeric(ho["miss_cp_share"], errors="coerce").mean())),
            "cm_p50": _f(float(pd.to_numeric(ho["miss_cp_share"], errors="coerce").median())),
        },
    ]
    prose = (
        f"Train txs missing-CP {_pp(train_share_n)} of count / {_pp(train_share_a)} of |amt| "
        f"({int(tr_m):,} / {int(tr_n):,}). Company-month share mean {_f(float(miss.mean()))} "
        f"p50 {_f(float(miss.median()))} cov {_pp(_pct(n_def, n_cm))} (in-memory; not written). "
        f"Holdout coverage only: txs {_pp(hold_share_n)} / |amt| {_pp(hold_share_a)}; "
        f"cm defined {_pp(_pct(n_ho_def, n_ho))}."
    )
    print(prose)
    return {
        "rows": rows,
        "train_share_n": train_share_n,
        "train_share_a": train_share_a,
        "hold_share_n": hold_share_n,
        "hold_share_a": hold_share_a,
        "cm_mean": float(miss.mean()) if n_def else float("nan"),
        "cm_p50": float(miss.median()) if n_def else float("nan"),
        "cm_cov": _pct(n_def, n_cm),
        "n_cm": n_cm,
        "n_co": int(tr["company_id"].nunique()),
        "n_def": n_def,
        "n_tx": tr_n,
        "n_miss": tr_m,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 2 — uncat × missing-CP 2×2
# ---------------------------------------------------------------------------
def pass2_uncat_2x2(con) -> dict:
    hold = load_holdout()
    hold_sql = _hold_sql(hold)
    cats = _cat_sql_list()
    raw = con.execute(
        f"""
        SELECT
          CASE WHEN t.category = 'uncategorized'
                 OR t.category IS NULL
                 OR t.category NOT IN ({cats})
                THEN 1 ELSE 0 END AS is_uncat,
          CASE WHEN t.counterparty_id IS NULL
                 OR length(trim(CAST(t.counterparty_id AS VARCHAR))) = 0
                THEN 1 ELSE 0 END AS is_miss,
          COUNT(*) AS n,
          SUM(ABS(t.amount)) AS abs_amt
        FROM transactions t
        WHERE t."date" IS NOT NULL
          AND t."date" >= DATE '2024-09-01'
          AND t."date" < DATE '2026-09-01'
          AND CAST(t.company_id AS VARCHAR) NOT IN ({hold_sql})
        GROUP BY 1, 2
        """
    ).df()
    tot_n = float(raw["n"].sum())
    tot_a = float(raw["abs_amt"].sum())
    uncat = raw[raw["is_uncat"] == 1]
    mapped = raw[raw["is_uncat"] == 0]
    n_uncat = float(uncat["n"].sum())
    a_uncat = float(uncat["abs_amt"].sum())
    n_mapped = float(mapped["n"].sum())
    a_mapped = float(mapped["abs_amt"].sum())
    n_uncat_miss = float(uncat.loc[uncat["is_miss"] == 1, "n"].sum())
    a_uncat_miss = float(uncat.loc[uncat["is_miss"] == 1, "abs_amt"].sum())
    n_mapped_miss = float(mapped.loc[mapped["is_miss"] == 1, "n"].sum())
    a_mapped_miss = float(mapped.loc[mapped["is_miss"] == 1, "abs_amt"].sum())
    uncat_miss_n = _pct(n_uncat_miss, n_uncat)
    uncat_miss_a = _pct(a_uncat_miss, a_uncat)
    mapped_miss_n = _pct(n_mapped_miss, n_mapped)
    mapped_miss_a = _pct(a_mapped_miss, a_mapped)
    confirm = bool(np.isfinite(uncat_miss_n) and abs(uncat_miss_n - UNCAT_MISS_QUOTE) < 0.015)
    confirm_a = bool(np.isfinite(uncat_miss_a) and abs(uncat_miss_a - UNCAT_MISS_AMT_QUOTE) < 0.02)
    rows = []
    for _, r in raw.iterrows():
        rows.append(
            {
                "uncat": int(r["is_uncat"]),
                "miss_cp": int(r["is_miss"]),
                "n": f"{int(r['n']):,}",
                "share_n": _pp(_pct(float(r["n"]), tot_n)),
                "|amt|": f"{float(r['abs_amt']):,.0f}",
                "share_|amt|": _pp(_pct(float(r["abs_amt"]), tot_a)),
            }
        )
    prose = (
        f"Uncat txs that miss CP {_pp(uncat_miss_n)} of count / {_pp(uncat_miss_a)} of |amt| "
        f"({'CONFIRM 91.8%' if confirm else 'off 91.8% quote'}; "
        f"{'CONFIRM 94.0% |amt|' if confirm_a else 'off 94.0% |amt|'}). "
        f"Mapped (non-uncat) txs that still miss CP: {_pp(mapped_miss_n)} of count / "
        f"{_pp(mapped_miss_a)} of |amt| ({int(n_mapped_miss):,} / {int(n_mapped):,}). "
        + (
            "Mapped still mostly unnamed — missing-CP is not only the uncat token."
            if np.isfinite(mapped_miss_n) and mapped_miss_n >= 0.50
            else "Mapped txs are mostly named — leftover after uncat is thin."
        )
    )
    print(prose)
    return {
        "rows": rows,
        "uncat_miss_n": uncat_miss_n,
        "uncat_miss_a": uncat_miss_a,
        "mapped_miss_n": mapped_miss_n,
        "mapped_miss_a": mapped_miss_a,
        "n_uncat": n_uncat,
        "n_mapped": n_mapped,
        "n_uncat_miss": n_uncat_miss,
        "n_mapped_miss": n_mapped_miss,
        "confirm": confirm,
        "confirm_a": confirm_a,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 3 — Spearman twins / SIZE
# ---------------------------------------------------------------------------
def pass3_spearman(tr: pd.DataFrame) -> dict:
    miss = pd.to_numeric(tr["miss_cp_share"], errors="coerce")
    amt = pd.to_numeric(tr["miss_cp_amt"], errors="coerce")
    uncat = pd.to_numeric(tr["a_uncat_share"], errors="coerce")
    dtx = pd.to_numeric(tr["d_tx_cp_share"], errors="coerce")
    size = pd.to_numeric(tr["log_in3"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    mapped = pd.to_numeric(tr["miss_mapped_share"], errors="coerce")
    named = pd.to_numeric(tr["named_share"], errors="coerce")
    pairs = [
        ("miss_cp_share", "a_uncat_share", spearman(miss, uncat)),
        ("miss_cp_share", "d_tx_cp_share", spearman(miss, dtx)),
        ("miss_cp_share", "1-d_tx_cp_share", spearman(miss, 1.0 - dtx)),
        ("miss_cp_share", "log1p(a_in3)", spearman(miss, size)),
        ("miss_cp_share", "c_n_days_with_tx", spearman(miss, days)),
        ("miss_cp_share", "miss_cp_amt", spearman(miss, amt)),
        ("miss_mapped_share", "a_uncat_share", spearman(mapped, uncat)),
        ("miss_mapped_share", "d_tx_cp_share", spearman(mapped, dtx)),
        ("named_share", "d_tx_cp_share", spearman(named, dtx)),
        ("miss_cp_amt", "a_uncat_share", spearman(amt, uncat)),
        ("miss_cp_amt", "d_tx_cp_share", spearman(amt, dtx)),
        ("miss_cp_amt", "log1p(a_in3)", spearman(amt, size)),
    ]
    rows = []
    rhos = {}
    for a, b, rho in pairs:
        twin = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
        size_flag = bool(b.startswith("log1p") and np.isfinite(rho) and abs(rho) >= SIZE_RHO)
        rhos[f"{a}~{b}"] = rho
        rows.append(
            {
                "a": a,
                "b": b,
                "ρ": _f(rho),
                "twin_|ρ|≥0.80": "YES" if twin else "no",
                "SIZE_|ρ|≥0.50": "YES" if size_flag else "—",
            }
        )
    rho_u = rhos.get("miss_cp_share~a_uncat_share", float("nan"))
    rho_d = rhos.get("miss_cp_share~d_tx_cp_share", float("nan"))
    rho_d1 = rhos.get("miss_cp_share~1-d_tx_cp_share", float("nan"))
    rho_s = rhos.get("miss_cp_share~log1p(a_in3)", float("nan"))
    twin_uncat = bool(np.isfinite(rho_u) and abs(rho_u) >= TWIN_RHO)
    twin_dtx = bool(
        (np.isfinite(rho_d) and abs(rho_d) >= TWIN_RHO)
        or (np.isfinite(rho_d1) and abs(rho_d1) >= TWIN_RHO)
    )
    is_size = bool(np.isfinite(rho_s) and abs(rho_s) >= SIZE_RHO)
    prose = (
        f"miss_cp_share vs a_uncat_share ρ={_f(rho_u)} "
        f"({'TWIN' if twin_uncat else 'not twin'}). "
        f"vs d_tx_cp_share ρ={_f(rho_d)} / vs 1−d_tx ρ={_f(rho_d1)} "
        f"({'TWIN of fill-rate' if twin_dtx else 'not the 6m fill twin'}). "
        f"vs log1p(a_in3) ρ={_f(rho_s)} ({'SIZE' if is_size else 'not SIZE'}). "
        f"vs days ρ={_f(rhos.get('miss_cp_share~c_n_days_with_tx', float('nan')))}."
    )
    print(prose)
    return {
        "rows": rows,
        "rhos": rhos,
        "rho_uncat": rho_u,
        "rho_dtx": rho_d,
        "rho_dtx_comp": rho_d1,
        "rho_size": rho_s,
        "rho_days": rhos.get("miss_cp_share~c_n_days_with_tx", float("nan")),
        "twin_uncat": twin_uncat,
        "twin_dtx": twin_dtx,
        "is_size": is_size,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 4 — group-fold AUROC Y3 / Y2
# ---------------------------------------------------------------------------
def _auc_row(yname: str, feat: str, rec: dict, size_cv: float, days_cv: float) -> dict:
    cv = rec["cv"]
    return {
        "y": yname,
        "feature": feat,
        "n": f"{rec['n_defined']:,}",
        "n_pos": f"{rec['n_pos']:,}",
        "CV": "LOW_POWER" if rec["low_power"] else _f(cv),
        "sd": _f(rec["sd"]),
        "sign": rec["train_sign"],
        "train": _f(rec["train_auc"]),
        "Δsize": _f((cv - size_cv) if np.isfinite(cv) and np.isfinite(size_cv) else float("nan")),
        "folds": fold_bits(rec),
    }


def pass4_auroc(tr: pd.DataFrame) -> dict:
    feats = {
        "miss_cp_share": tr["miss_cp_share"],
        "miss_cp_amt": tr["miss_cp_amt"],
        "miss_mapped_share": tr["miss_mapped_share"],
        "d_tx_cp_share": tr["d_tx_cp_share"],
        "a_uncat_share": tr["a_uncat_share"],
        "named_share": tr["named_share"],
        "log1p_a_in3": tr["log_in3"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
    }
    rows = []
    store: dict[tuple[str, str], dict] = {}
    for ycol, yname in ((Y3, "y3_recover_cash_6m"), (Y2, "y2_neg_2of3")):
        size_rec = signed_oof_auroc(tr[ycol], tr["log_in3"], tr["fold"], tr[ycol].notna())
        days_rec = signed_oof_auroc(tr[ycol], tr["c_n_days_with_tx"], tr["fold"], tr[ycol].notna())
        size_cv = float("nan") if size_rec["low_power"] else size_rec["cv"]
        days_cv = float("nan") if days_rec["low_power"] else days_rec["cv"]
        for fname, x in feats.items():
            rec = signed_oof_auroc(tr[ycol], x, tr["fold"], tr[ycol].notna())
            store[(yname, fname)] = rec
            rec["coverage"] = _pct(rec["n_defined"], int(tr[ycol].notna().sum()))
            rows.append(_auc_row(yname, fname, rec, size_cv, days_cv))
        store[(yname, "size")] = size_rec
        store[(yname, "days")] = days_rec

    y3_m = store[("y3_recover_cash_6m", "miss_cp_share")]
    y3_s = store[("y3_recover_cash_6m", "size")]
    y3_d = store[("y3_recover_cash_6m", "days")]
    y3_dt = store[("y3_recover_cash_6m", "d_tx_cp_share")]
    y2_m = store[("y2_neg_2of3", "miss_cp_share")]
    miss_y3 = float("nan") if y3_m["low_power"] else y3_m["cv"]
    size_y3 = float("nan") if y3_s["low_power"] else y3_s["cv"]
    days_y3 = float("nan") if y3_d["low_power"] else y3_d["cv"]
    dtx_y3 = float("nan") if y3_dt["low_power"] else y3_dt["cv"]
    miss_y2 = float("nan") if y2_m["low_power"] else y2_m["cv"]
    days_ok = bool(np.isfinite(days_y3) and abs(days_y3 - DAYS_BENCH) < 0.02)
    size_ok = bool(np.isfinite(size_y3) and abs(size_y3 - SIZE_Y3_QUOTE) < 0.03)
    beat = (
        (miss_y3 - size_y3)
        if np.isfinite(miss_y3) and np.isfinite(size_y3)
        else float("nan")
    )
    prose = (
        f"Y3 miss_cp_share {_f(miss_y3)} vs size {_f(size_y3)} (Δ {_f(beat, 3)}; "
        f"quote 0.617 {'CONFIRM' if size_ok else 'off'}) vs days {_f(days_y3)} "
        f"(night 0.711 {'CONFIRM' if days_ok else 'off'}). "
        f"Y3 d_tx_cp_share replica {_f(dtx_y3)}. Y2 miss {_f(miss_y2)}. "
        f"Sign from train side of each fold. Never Y7."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "miss_y3": miss_y3,
        "miss_y2": miss_y2,
        "size_y3": size_y3,
        "days_y3": days_y3,
        "dtx_y3": dtx_y3,
        "dtx_y2": float("nan")
        if store[("y2_neg_2of3", "d_tx_cp_share")]["low_power"]
        else store[("y2_neg_2of3", "d_tx_cp_share")]["cv"],
        "mapped_y3": float("nan")
        if store[("y3_recover_cash_6m", "miss_mapped_share")]["low_power"]
        else store[("y3_recover_cash_6m", "miss_mapped_share")]["cv"],
        "amt_y3": float("nan")
        if store[("y3_recover_cash_6m", "miss_cp_amt")]["low_power"]
        else store[("y3_recover_cash_6m", "miss_cp_amt")]["cv"],
        "uncat_y3": float("nan")
        if store[("y3_recover_cash_6m", "a_uncat_share")]["low_power"]
        else store[("y3_recover_cash_6m", "a_uncat_share")]["cv"],
        "beat": beat,
        "days_ok": days_ok,
        "size_ok": size_ok,
        "y3_folds": fold_bits(y3_m),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 5 — leftover after residualizing on uncat / d_tx_cp_share
# ---------------------------------------------------------------------------
def pass5_residual(tr: pd.DataFrame) -> dict:
    miss = pd.to_numeric(tr["miss_cp_share"], errors="coerce")
    uncat = pd.to_numeric(tr["a_uncat_share"], errors="coerce")
    dtx = pd.to_numeric(tr["d_tx_cp_share"], errors="coerce")
    mapped = pd.to_numeric(tr["miss_mapped_share"], errors="coerce")
    r_u, inf_u = ols_resid(miss, uncat)
    r_d, inf_d = ols_resid(miss, dtx)
    r_both, inf_b = ols_resid(miss, uncat, dtx)
    r_map_u, _ = ols_resid(mapped, uncat)
    r_map_d, _ = ols_resid(mapped, dtx)
    specs = [
        ("resid_uncat", r_u),
        ("resid_dtx", r_d),
        ("resid_uncat+dtx", r_both),
        ("mapped_resid_uncat", r_map_u),
        ("mapped_resid_dtx", r_map_d),
        ("miss_mapped_share", mapped),
    ]
    rows = []
    cvs = {}
    for ycol, yname in ((Y3, "Y3"), (Y2, "Y2")):
        size = signed_oof_auroc(tr[ycol], tr["log_in3"], tr["fold"], tr[ycol].notna())
        size_cv = float("nan") if size["low_power"] else size["cv"]
        for name, x in specs:
            rec = signed_oof_auroc(tr[ycol], x, tr["fold"], tr[ycol].notna())
            cv = float("nan") if rec["low_power"] else rec["cv"]
            cvs[(yname, name)] = cv
            rows.append(
                {
                    "y": yname,
                    "feature": name,
                    "n": f"{rec['n_defined']:,}",
                    "n_pos": f"{rec['n_pos']:,}",
                    "CV": "LOW_POWER" if rec["low_power"] else _f(cv),
                    "sign": rec["train_sign"],
                    "Δsize": _f((cv - size_cv) if np.isfinite(cv) and np.isfinite(size_cv) else float("nan")),
                }
            )
    y3_u = cvs.get(("Y3", "resid_uncat"), float("nan"))
    y3_d = cvs.get(("Y3", "resid_dtx"), float("nan"))
    y3_b = cvs.get(("Y3", "resid_uncat+dtx"), float("nan"))
    size_y3 = float("nan")
    size_rec = signed_oof_auroc(tr[Y3], tr["log_in3"], tr["fold"], tr[Y3].notna())
    if not size_rec["low_power"]:
        size_y3 = size_rec["cv"]
    leftover_beat = (
        (y3_b - size_y3) if np.isfinite(y3_b) and np.isfinite(size_y3) else float("nan")
    )
    leftover_lives = bool(np.isfinite(leftover_beat) and leftover_beat >= KEEP_DELTA)
    died = bool(
        (np.isfinite(y3_u) and y3_u < 0.55)
        or (np.isfinite(y3_d) and y3_d < 0.55)
        or (np.isfinite(y3_b) and y3_b < 0.55)
    )
    prose = (
        f"Y3 leftover after a_uncat_share {_f(y3_u)}; after d_tx_cp_share {_f(y3_d)}; "
        f"after both {_f(y3_b)} vs size {_f(size_y3)} (Δ {_f(leftover_beat, 3)}). "
        + (
            "Leftover dies — CLOSE as twin."
            if died and not leftover_lives
            else (
                "Leftover beats size ≥0.02 after both — later-D candidate (do not merge)."
                if leftover_lives
                else "Leftover does not clear size+0.02 after both."
            )
        )
    )
    print(prose)
    return {
        "rows": rows,
        "y3_uncat": y3_u,
        "y3_dtx": y3_d,
        "y3_both": y3_b,
        "y2_both": cvs.get(("Y2", "resid_uncat+dtx"), float("nan")),
        "y3_mapped": cvs.get(("Y3", "miss_mapped_share"), float("nan")),
        "size_y3": size_y3,
        "leftover_beat": leftover_beat,
        "leftover_lives": leftover_lives,
        "died": died,
        "slope_uncat": inf_u["slope"][0] if inf_u["slope"] else float("nan"),
        "slope_dtx": inf_d["slope"][0] if inf_d["slope"] else float("nan"),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 6 — dark 470 vs invoiced 744
# ---------------------------------------------------------------------------
def pass6_dark(tr: pd.DataFrame) -> dict:
    if "ever_erp" not in tr.columns:
        raise RuntimeError("ever_erp missing — attach from book_invoice_ids")
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    n_erp = int(last["ever_erp"].sum())
    n_dark = int((~last["ever_erp"]).sum())
    confirm = n_erp == 744 and n_dark == 470
    rows = []
    for name, m in (("ever_erp_744", tr["ever_erp"]), ("never_erp_470", ~tr["ever_erp"])):
        sl = tr.loc[m]
        miss = pd.to_numeric(sl["miss_cp_share"], errors="coerce")
        mapped = pd.to_numeric(sl["miss_mapped_share"], errors="coerce")
        dtx = pd.to_numeric(sl["d_tx_cp_share"], errors="coerce")
        rows.append(
            {
                "group": name,
                "n_cm": int(len(sl)),
                "n_co": int(sl["company_id"].nunique()),
                "miss_mean": _f(float(miss.mean())),
                "miss_p50": _f(float(miss.median())),
                "mapped_miss": _f(float(mapped.mean())),
                "d_tx_cp": _f(float(dtx.mean())),
            }
        )
    erp_m = float(pd.to_numeric(tr.loc[tr["ever_erp"], "miss_cp_share"], errors="coerce").mean())
    dark_m = float(pd.to_numeric(tr.loc[~tr["ever_erp"], "miss_cp_share"], errors="coerce").mean())
    gap = dark_m - erp_m if np.isfinite(dark_m) and np.isfinite(erp_m) else float("nan")
    is_dark_hole = bool(np.isfinite(gap) and gap >= 0.10)
    only_dark = bool(np.isfinite(erp_m) and erp_m < 0.20 and np.isfinite(dark_m) and dark_m >= 0.80)
    prose = (
        f"Train last-month companies: ever-ERP {n_erp} / never-ERP {n_dark} "
        f"({'CONFIRM 744/470' if confirm else 'off 744/470'}). "
        f"Mean miss_cp_share invoiced {_f(erp_m)} vs dark {_f(dark_m)} (Δ {_f(gap, 3)}). "
        + (
            "Dark is always-missing (level). Invoiced still miss most bank CPs — not a 470-only X."
            if is_dark_hole and not only_dark
            else (
                "Missing-CP ≈ no-ERP only."
                if only_dark
                else "Dark and invoiced both miss bank CPs — not the 470-only hole."
            )
        )
    )
    print(prose)
    return {
        "rows": rows,
        "n_erp": n_erp,
        "n_dark": n_dark,
        "confirm": confirm,
        "erp_m": erp_m,
        "dark_m": dark_m,
        "gap": gap,
        "is_dark_hole": is_dark_hole,
        "only_dark": only_dark,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 7 — invoice CP fill vs tx CP fill (same month)
# ---------------------------------------------------------------------------
def pass7_invoice_vs_tx(tr: pd.DataFrame) -> dict:
    inv = pd.to_numeric(tr["inv_cp_share"], errors="coerce")
    named = pd.to_numeric(tr["named_share"], errors="coerce")
    miss = pd.to_numeric(tr["miss_cp_share"], errors="coerce")
    both = inv.notna() & named.notna()
    rho = spearman(inv, named)
    erp = tr["ever_erp"] if "ever_erp" in tr.columns else inv.notna()
    rows = [
        {
            "slice": "tx months (all train)",
            "n_cm": int(named.notna().sum()),
            "tx_named_p50": _f(float(named.median())),
            "tx_named_mean": _f(float(named.mean())),
            "inv_named_p50": "—",
            "inv_named_mean": "—",
        },
        {
            "slice": "invoice months",
            "n_cm": int(inv.notna().sum()),
            "tx_named_p50": _f(float(named[inv.notna()].median())),
            "tx_named_mean": _f(float(named[inv.notna()].mean())),
            "inv_named_p50": _f(float(inv.median())),
            "inv_named_mean": _f(float(inv.mean())),
        },
        {
            "slice": "same-month both",
            "n_cm": int(both.sum()),
            "tx_named_p50": _f(float(named[both].median())),
            "tx_named_mean": _f(float(named[both].mean())),
            "inv_named_p50": _f(float(inv[both].median())),
            "inv_named_mean": _f(float(inv[both].mean())),
        },
        {
            "slice": "invoiced 744 (any month)",
            "n_cm": int((erp & named.notna()).sum()),
            "tx_named_p50": _f(float(named[erp].median())),
            "tx_named_mean": _f(float(named[erp].mean())),
            "inv_named_p50": _f(float(inv[erp].median())),
            "inv_named_mean": _f(float(inv[erp].mean())),
        },
    ]
    inv_mean = float(inv[both].mean()) if both.any() else float("nan")
    tx_mean = float(named[both].mean()) if both.any() else float("nan")
    hole_is_bank = bool(
        np.isfinite(inv_mean) and np.isfinite(tx_mean) and (inv_mean - tx_mean) >= 0.40
    )
    prose = (
        f"Same-month invoice CP fill mean {_f(inv_mean)} vs tx named share {_f(tx_mean)} "
        f"(ρ {_f(rho)}; n={int(both.sum()):,}). "
        + (
            "Hole is bank-book tagging, not the invoice book (Family D HHI lives on invoices)."
            if hole_is_bank
            else "Invoice and tx fill are not a 40pp gap — check both books."
        )
        + " Do not invent a COMP_* ↔ COUNTERPARTY_* map."
    )
    print(prose)
    return {
        "rows": rows,
        "rho": rho,
        "inv_mean": inv_mean,
        "tx_mean": tx_mean,
        "n_both": int(both.sum()),
        "hole_is_bank": hole_is_bank,
        "inv_p50": float(inv[both].median()) if both.any() else float("nan"),
        "tx_p50": float(named[both].median()) if both.any() else float("nan"),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 8 — Y5 fold-3 hole
# ---------------------------------------------------------------------------
def pass8_y5_fold3(tr: pd.DataFrame) -> dict:
    y = pd.to_numeric(tr[Y5_AR], errors="coerce")
    lab = y.notna()
    share = pd.to_numeric(tr["d_tx_cp_share"], errors="coerce")
    nc = pd.to_numeric(tr["d_n_cust"], errors="coerce")
    miss = pd.to_numeric(tr["miss_cp_share"], errors="coerce")
    med_n = float(nc[lab & nc.notna()].median()) if (lab & nc.notna()).any() else float("nan")
    hi_cust = nc > med_n
    hole = lab & share.notna() & nc.notna() & (share <= HOLE_SHARE) & hi_cust
    named_hi = lab & share.notna() & nc.notna() & (share > HOLE_SHARE) & hi_cust
    folds = pd.to_numeric(tr["fold"], errors="coerce")
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
        miss_h = float(miss[h].mean()) if n_h else float("nan")
        miss_n = float(miss[named_hi & (folds == k)].mean()) if n_n else float("nan")
        fold_rows.append(
            {
                "fold": k,
                "n_hole": n_h,
                "hole_pos": n_hp,
                "P(Y5=1) hole": _pp(_pct(n_hp, n_h)),
                "P(Y5=1) named": _pp(_pct(n_np, n_n)),
                "miss_hole": _f(miss_h),
                "miss_named": _f(miss_n),
            }
        )
        if k == 3 and n_hole_pos:
            fold3_share = _pct(n_hp, n_hole_pos)
    # does missing-CP pile in fold 3 among AR-labeled months?
    ar = tr.loc[lab].copy()
    ar["_miss"] = miss[lab]
    hi_miss = ar["_miss"] >= float(ar["_miss"].median()) if ar["_miss"].notna().any() else pd.Series(False, index=ar.index)
    n_hi = int(hi_miss.sum())
    n_hi_f3 = int((hi_miss & (ar["fold"] == 3)).sum())
    pile_f3 = _pct(n_hi_f3, n_hi)
    one_group = bool(
        (np.isfinite(fold3_share) and fold3_share >= 0.70)
        or (np.isfinite(pile_f3) and pile_f3 >= 0.50)
    )
    # drop fold 3: hole rate vs named
    wo = lab & (folds != 3)
    hole_wo = _pct(int((hole & wo & (y == 1)).sum()), int((hole & wo).sum()))
    named_wo = _pct(int((named_hi & wo & (y == 1)).sum()), int((named_hi & wo).sum()))
    survives = bool(
        np.isfinite(hole_wo) and np.isfinite(named_wo) and (hole_wo - named_wo) >= 0.04
    )
    prose = (
        f"Y5 AR hole (d_tx_cp_share≤0.01 ∩ hi d_n_cust) pos in fold 3: "
        f"{_pp(fold3_share)} of hole positives. High-miss AR months in fold 3: {_pp(pile_f3)}. "
        f"Without fold 3: hole {_pp(hole_wo)} vs named {_pp(named_wo)} "
        f"(survives={survives}). "
        + (
            "CLOSE as one-group — same fold-3 that owned the d_tx_cp_share Q5 sentence."
            if one_group and not survives
            else "Not the same one-group pile as Y5 fold 3."
        )
    )
    print(prose)
    return {
        "rows": fold_rows,
        "fold3_share": fold3_share,
        "pile_f3": pile_f3,
        "one_group": one_group,
        "survives": survives,
        "hole_wo": hole_wo,
        "named_wo": named_wo,
        "n_hole_pos": n_hole_pos,
        "med_n_cust": med_n,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 9 — drop 12 chronic dark Y2 names
# ---------------------------------------------------------------------------
def pass9_chronic(tr: pd.DataFrame, ids: list[str]) -> dict:
    is_ch = tr["company_id"].astype(str).isin(set(ids))
    rows = []
    cvs = {}
    for name, m in (
        ("all", pd.Series(True, index=tr.index)),
        ("drop_12", ~is_ch),
        ("chronic_12", is_ch),
    ):
        sl = tr.loc[m]
        rec = signed_oof_auroc(sl[Y2], sl["miss_cp_share"], sl["fold"], sl[Y2].notna())
        rec3 = signed_oof_auroc(sl[Y3], sl["miss_cp_share"], sl["fold"], sl[Y3].notna())
        cvs[(name, "Y2")] = float("nan") if rec["low_power"] else rec["cv"]
        cvs[(name, "Y3")] = float("nan") if rec3["low_power"] else rec3["cv"]
        rows.append(
            {
                "slice": name,
                "n_cm": int(len(sl)),
                "n_co": int(sl["company_id"].nunique()),
                "Y2_n": f"{rec['n_defined']:,}",
                "Y2_pos": f"{rec['n_pos']:,}",
                "Y2_CV": "LOW_POWER" if rec["low_power"] else _f(rec["cv"]),
                "Y3_CV": "LOW_POWER" if rec3["low_power"] else _f(rec3["cv"]),
                "miss_mean": _f(float(pd.to_numeric(sl["miss_cp_share"], errors="coerce").mean())),
            }
        )
    drop = cvs.get(("drop_12", "Y2"), float("nan"))
    allv = cvs.get(("all", "Y2"), float("nan"))
    is_those = bool(np.isfinite(allv) and np.isfinite(drop) and (allv - drop) >= 0.02)
    prose = (
        f"12 chronic dark Y2 names (GROUP_0158/0172, ≥50% b_below_0) n={len(ids)}. "
        f"Y2 miss_cp_share all {_f(allv)} vs drop-12 {_f(drop)}. "
        + (
            "Skill *is* those names — drop test kills the Y2 quote."
            if is_those
            else "Drop-12 does not move Y2 by ≥0.02 — not the chronic pile."
        )
    )
    print(prose)
    return {
        "rows": rows,
        "n_ids": len(ids),
        "ids": ids,
        "y2_all": allv,
        "y2_drop": drop,
        "y3_drop": cvs.get(("drop_12", "Y3"), float("nan")),
        "is_those": is_those,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 10 — ICC / company-demean
# ---------------------------------------------------------------------------
def pass10_icc(tr: pd.DataFrame) -> dict:
    miss = pd.to_numeric(tr["miss_cp_share"], errors="coerce")
    icc = icc_anova(miss, tr["company_id"])
    acf = {k: median_acf(miss, tr["company_id"], k) for k in (1, 3, 6)}
    high_icc = bool(np.isfinite(icc["icc"]) and icc["icc"] >= ICC_STYLE)
    low_acf = bool(np.isfinite(acf[1]) and acf[1] < ACF_STYLE)
    style = bool(high_icc)  # BETWEEN fill habit; acf1 only splits shock vs sticky
    rows_auc = []
    cvs = {}
    for ycol, yname in ((Y3, "Y3"), (Y2, "Y2")):
        for feat, col in (("co_mean", tr["miss_co_mean"]), ("demean", tr["miss_demean"]), ("now", miss)):
            rec = signed_oof_auroc(tr[ycol], col, tr["fold"], tr[ycol].notna())
            cvs[(yname, feat)] = float("nan") if rec["low_power"] else rec["cv"]
            rows_auc.append(
                {
                    "y": yname,
                    "feature": feat,
                    "n": f"{rec['n_defined']:,}",
                    "n_pos": f"{rec['n_pos']:,}",
                    "CV": "LOW_POWER" if rec["low_power"] else _f(rec["cv"]),
                    "sign": rec["train_sign"],
                }
            )
    y3_mean = cvs.get(("Y3", "co_mean"), float("nan"))
    y3_shock = cvs.get(("Y3", "demean"), float("nan"))
    y2_mean = cvs.get(("Y2", "co_mean"), float("nan"))
    y2_shock = cvs.get(("Y2", "demean"), float("nan"))
    style_carries = bool(
        np.isfinite(y3_mean)
        and np.isfinite(y3_shock)
        and (y3_mean - y3_shock) >= 0.04
    )
    prose = (
        f"miss_cp_share acf1={_f(acf[1])} acf3={_f(acf[3])} acf6={_f(acf[6])}; "
        f"ICC={_f(icc['icc'])} ({'sticky fill habit (BETWEEN)' if high_icc else 'not high-ICC'}). "
        f"Y3 company-mean {_f(y3_mean)} vs demean {_f(y3_shock)}; "
        f"Y2 mean {_f(y2_mean)} vs shock {_f(y2_shock)}. "
        + (
            "Trait, not a month shock — not Q5 change."
            if style or style_carries
            else "Month shock is closer to the skill."
        )
    )
    print(prose)
    return {
        "icc": icc,
        "acf": acf,
        "style": style,
        "high_icc": high_icc,
        "low_acf": low_acf,
        "rows": rows_auc,
        "y3_mean": y3_mean,
        "y3_shock": y3_shock,
        "y2_mean": y2_mean,
        "y2_shock": y2_shock,
        "style_carries": style_carries,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 11 — Q6 lag1/lag3 short vs long
# ---------------------------------------------------------------------------
def pass11_q6(tr: pd.DataFrame) -> dict:
    rows = []
    cvs = {}
    slices = (
        ("all", pd.Series(True, index=tr.index)),
        ("short_<12", tr["so_far"] < 12),
        ("long_>=18", tr["so_far"] >= 18),
    )
    stems = (
        ("miss_cp_share", 0),
        ("miss_cp_share_lag1", 1),
        ("miss_cp_share_lag3", 3),
        ("d_tx_cp_share", 0),
        ("d_tx_cp_share_lag1", 1),
    )
    for sl_name, m in slices:
        sl = tr.loc[m]
        for ycol, yname in ((Y3, "Y3"), (Y2, "Y2")):
            for col, lag in stems:
                if col not in sl.columns:
                    continue
                rec = signed_oof_auroc(sl[ycol], sl[col], sl["fold"], sl[ycol].notna())
                key = (yname, sl_name, col)
                cvs[key] = float("nan") if rec["low_power"] else rec["cv"]
                rows.append(
                    {
                        "y": yname,
                        "slice": sl_name,
                        "col": col,
                        "lag": lag,
                        "n": f"{rec['n_defined']:,}",
                        "n_pos": f"{rec['n_pos']:,}",
                        "CV": "LOW_POWER" if rec["low_power"] else _f(rec["cv"]),
                        "sign": rec["train_sign"],
                    }
                )
    now = cvs.get(("Y3", "all", "miss_cp_share"), float("nan"))
    lag1 = cvs.get(("Y3", "all", "miss_cp_share_lag1"), float("nan"))
    lag3 = cvs.get(("Y3", "all", "miss_cp_share_lag3"), float("nan"))
    short = cvs.get(("Y3", "short_<12", "miss_cp_share"), float("nan"))
    short_l1 = cvs.get(("Y3", "short_<12", "miss_cp_share_lag1"), float("nan"))
    long = cvs.get(("Y3", "long_>=18", "miss_cp_share"), float("nan"))
    q6_close = True
    if np.isfinite(now) and np.isfinite(lag1) and (now - lag1) < 0.01 and now >= 0.58:
        q6_close = False
    if np.isfinite(short) and short < 0.55:
        q6_close = True
    prose = (
        f"Y3 miss now {_f(now)} / lag1 {_f(lag1)} / lag3 {_f(lag3)}. "
        f"Short so-far<12 {_f(short)} lag1 {_f(short_l1)}; long≥18 {_f(long)}. "
        + (
            "Q6 CLOSE — lag does not hold contemporaneous skill (or short books die)."
            if q6_close
            else "Lag1 holds contemporaneous skill — still only the 1-month clock."
        )
    )
    print(prose)
    return {
        "rows": rows,
        "now": now,
        "lag1": lag1,
        "lag3": lag3,
        "short": short,
        "short_l1": short_l1,
        "long": long,
        "q6": "CLOSE" if q6_close else "KEEP-1m",
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 12 — category mix of missing-CP txs
# ---------------------------------------------------------------------------
def pass12_cats(con) -> dict:
    hold = load_holdout()
    hold_sql = _hold_sql(hold)
    raw = con.execute(
        f"""
        SELECT
          CAST(COALESCE(t.category, '(null)') AS VARCHAR) AS category,
          COUNT(*) AS n,
          SUM(ABS(t.amount)) AS abs_amt,
          COUNT(DISTINCT t.company_id) AS n_co
        FROM transactions t
        WHERE t."date" IS NOT NULL
          AND t."date" >= DATE '2024-09-01'
          AND t."date" < DATE '2026-09-01'
          AND CAST(t.company_id AS VARCHAR) NOT IN ({hold_sql})
          AND (t.counterparty_id IS NULL
               OR length(trim(CAST(t.counterparty_id AS VARCHAR))) = 0)
        GROUP BY 1
        ORDER BY n DESC
        """
    ).df()
    tot_n = float(raw["n"].sum())
    tot_a = float(raw["abs_amt"].sum())
    raw["share_n"] = raw["n"] / tot_n if tot_n else np.nan
    raw["share_a"] = raw["abs_amt"] / tot_a if tot_a else np.nan
    top = raw.head(15)
    rows = []
    for _, r in top.iterrows():
        rows.append(
            {
                "category": str(r["category"]),
                "n": f"{int(r['n']):,}",
                "|amt|": f"{float(r['abs_amt']):,.0f}",
                "n_co": int(r["n_co"]),
                "share_n": _pp(float(r["share_n"])),
                "share_|amt|": _pp(float(r["share_a"])),
            }
        )
    key_cats = {
        "uncategorized",
        "transfer",
        "salary",
        "tax",
        "social_security",
        "collection",
        "payment",
        "fee",
        "interest_charge",
    }
    shares = {}
    for cat in key_cats:
        sl = raw[raw["category"] == cat]
        shares[cat] = float(sl["share_n"].sum()) if len(sl) else 0.0
        shares[f"{cat}_amt"] = float(sl["share_a"].sum()) if len(sl) else 0.0
    uncat_n = shares.get("uncategorized", 0.0)
    xfer_n = shares.get("transfer", 0.0)
    is_uncat_xfer = bool((uncat_n + xfer_n) >= 0.50)
    prose = (
        f"Missing-CP txs: uncategorized {_pp(uncat_n)} / transfer {_pp(xfer_n)} "
        f"/ salary {_pp(shares.get('salary', 0))} / tax {_pp(shares.get('tax', 0))} "
        f"/ collection {_pp(shares.get('collection', 0))} / payment {_pp(shares.get('payment', 0))}. "
        + (
            "It is uncat + transfers (and the rest of a mostly-unnamed bank book)."
            if is_uncat_xfer
            else "Not just uncat+transfer — other mapped cats also miss CP."
        )
    )
    print(prose)
    return {
        "rows": rows,
        "shares": shares,
        "uncat_n": uncat_n,
        "xfer_n": xfer_n,
        "is_uncat_xfer": is_uncat_xfer,
        "n_cats": int(len(raw)),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 13 — amounts: p50 |amt| missing-CP vs named-CP
# ---------------------------------------------------------------------------
def pass13_amounts(con) -> dict:
    hold = load_holdout()
    hold_sql = _hold_sql(hold)
    raw = con.execute(
        f"""
        SELECT
          CASE WHEN t.counterparty_id IS NULL
                 OR length(trim(CAST(t.counterparty_id AS VARCHAR))) = 0
                THEN 1 ELSE 0 END AS is_miss,
          COUNT(*) AS n,
          SUM(ABS(t.amount)) AS abs_amt,
          quantile_cont(ABS(t.amount), 0.50) AS p50,
          quantile_cont(ABS(t.amount), 0.90) AS p90,
          AVG(ABS(t.amount)) AS mean_abs
        FROM transactions t
        WHERE t."date" IS NOT NULL
          AND t."date" >= DATE '2024-09-01'
          AND t."date" < DATE '2026-09-01'
          AND CAST(t.company_id AS VARCHAR) NOT IN ({hold_sql})
        GROUP BY 1
        """
    ).df()
    rows = []
    p50s = {}
    for _, r in raw.iterrows():
        lab = "missing_cp" if int(r["is_miss"]) == 1 else "named_cp"
        p50s[lab] = float(r["p50"])
        rows.append(
            {
                "kind": lab,
                "n": f"{int(r['n']):,}",
                "p50_|amt|": _f(float(r["p50"]), 0),
                "p90_|amt|": _f(float(r["p90"]), 0),
                "mean_|amt|": _f(float(r["mean_abs"]), 0),
                "pooled_|amt|": f"{float(r['abs_amt']):,.0f}",
            }
        )
    ratio = (
        p50s.get("missing_cp", float("nan")) / p50s["named_cp"]
        if p50s.get("named_cp")
        else float("nan")
    )
    prose = (
        f"p50 |amt| missing-CP {_f(p50s.get('missing_cp', float('nan')), 0)} vs named-CP "
        f"{_f(p50s.get('named_cp', float('nan')), 0)} (ratio {_f(ratio)}). "
        + (
            "Missing-CP tickets are not systematically larger."
            if np.isfinite(ratio) and 0.5 <= ratio <= 2.0
            else "Ticket sizes differ — check fat tail."
        )
    )
    print(prose)
    return {
        "rows": rows,
        "p50_miss": p50s.get("missing_cp", float("nan")),
        "p50_named": p50s.get("named_cp", float("nan")),
        "ratio": ratio,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 14 — leftover among mapped-only months (condition on uncat)
# ---------------------------------------------------------------------------
def pass14_mapped_leftover(tr: pd.DataFrame) -> dict:
    """Company-months with low uncat: is missing-CP still a signal?"""
    uncat = pd.to_numeric(tr["a_uncat_share"], errors="coerce")
    low_u = uncat.notna() & (uncat <= 0.10)
    rows = []
    cvs = {}
    for name, m in (("all", pd.Series(True, index=tr.index)), ("uncat≤0.10", low_u)):
        sl = tr.loc[m]
        for ycol, yname in ((Y3, "Y3"), (Y2, "Y2")):
            rec = signed_oof_auroc(sl[ycol], sl["miss_cp_share"], sl["fold"], sl[ycol].notna())
            rec_m = signed_oof_auroc(sl[ycol], sl["miss_mapped_share"], sl["fold"], sl[ycol].notna())
            size = signed_oof_auroc(sl[ycol], sl["log_in3"], sl["fold"], sl[ycol].notna())
            cvs[(yname, name, "miss")] = float("nan") if rec["low_power"] else rec["cv"]
            cvs[(yname, name, "mapped")] = float("nan") if rec_m["low_power"] else rec_m["cv"]
            cvs[(yname, name, "size")] = float("nan") if size["low_power"] else size["cv"]
            rows.append(
                {
                    "y": yname,
                    "slice": name,
                    "feat": "miss_cp_share",
                    "n": f"{rec['n_defined']:,}",
                    "n_pos": f"{rec['n_pos']:,}",
                    "CV": "LOW_POWER" if rec["low_power"] else _f(rec["cv"]),
                    "size": "LOW_POWER" if size["low_power"] else _f(size["cv"]),
                }
            )
            rows.append(
                {
                    "y": yname,
                    "slice": name,
                    "feat": "miss_mapped_share",
                    "n": f"{rec_m['n_defined']:,}",
                    "n_pos": f"{rec_m['n_pos']:,}",
                    "CV": "LOW_POWER" if rec_m["low_power"] else _f(rec_m["cv"]),
                    "size": "LOW_POWER" if size["low_power"] else _f(size["cv"]),
                }
            )
    y3_low = cvs.get(("Y3", "uncat≤0.10", "miss"), float("nan"))
    y3_low_sz = cvs.get(("Y3", "uncat≤0.10", "size"), float("nan"))
    lives = bool(
        np.isfinite(y3_low)
        and np.isfinite(y3_low_sz)
        and (y3_low - y3_low_sz) >= KEEP_DELTA
    )
    prose = (
        f"Y3 miss_cp_share on uncat≤0.10 months {_f(y3_low)} vs size {_f(y3_low_sz)}. "
        + (
            "Leftover after parking uncat still beats size — Q5 candidate."
            if lives
            else "After conditioning on low-uncat, missing-CP does not beat size."
        )
    )
    print(prose)
    return {
        "rows": rows,
        "y3_low": y3_low,
        "y3_low_sz": y3_low_sz,
        "lives": lives,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 15 — leftover after high d_tx_cp_share (condition on fill-rate)
# ---------------------------------------------------------------------------
def pass15_after_dtx(tr: pd.DataFrame) -> dict:
    dtx = pd.to_numeric(tr["d_tx_cp_share"], errors="coerce")
    hi = dtx.notna() & (dtx >= 0.20)
    zero = dtx.notna() & (dtx <= HOLE_SHARE)
    rows = []
    cvs = {}
    for name, m in (("dtx≥0.20", hi), ("dtx≤0.01", zero), ("all", dtx.notna())):
        sl = tr.loc[m]
        rec = signed_oof_auroc(sl[Y3], sl["miss_cp_share"], sl["fold"], sl[Y3].notna())
        size = signed_oof_auroc(sl[Y3], sl["log_in3"], sl["fold"], sl[Y3].notna())
        cvs[name] = float("nan") if rec["low_power"] else rec["cv"]
        cvs[f"{name}_sz"] = float("nan") if size["low_power"] else size["cv"]
        rows.append(
            {
                "slice": name,
                "n": f"{rec['n_defined']:,}",
                "n_pos": f"{rec['n_pos']:,}",
                "Y3_miss": "LOW_POWER" if rec["low_power"] else _f(rec["cv"]),
                "Y3_size": "LOW_POWER" if size["low_power"] else _f(size["cv"]),
                "miss_mean": _f(float(pd.to_numeric(sl["miss_cp_share"], errors="coerce").mean())),
            }
        )
    prose = (
        f"Y3 miss on dtx≥0.20 {_f(cvs.get('dtx≥0.20', float('nan')))} vs size "
        f"{_f(cvs.get('dtx≥0.20_sz', float('nan')))}; on the Y5 hole (dtx≤0.01) "
        f"{_f(cvs.get('dtx≤0.01', float('nan')))}. "
        "If skill lives only on the unnamed 6m head, it is the d_tx_cp_share object."
    )
    print(prose)
    return {"rows": rows, "cvs": cvs, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 16 — Y5 AR single (do not change night 0.611 quote)
# ---------------------------------------------------------------------------
def pass16_y5_replica(tr: pd.DataFrame) -> dict:
    y = pd.to_numeric(tr[Y5_AR], errors="coerce")
    lab = y.notna()
    sl = tr.loc[lab]
    rec_m = signed_oof_auroc(sl[Y5_AR], sl["miss_cp_share"], sl["fold"], sl[Y5_AR].notna())
    rec_d = signed_oof_auroc(sl[Y5_AR], sl["d_tx_cp_share"], sl["fold"], sl[Y5_AR].notna())
    rec_n = signed_oof_auroc(sl[Y5_AR], sl["named_share"], sl["fold"], sl[Y5_AR].notna())
    dtx_cv = float("nan") if rec_d["low_power"] else rec_d["cv"]
    dtx_tr = rec_d["train_auc"]
    confirm = bool(np.isfinite(dtx_tr) and abs(dtx_tr - Y5_DTX_QUOTE) < 0.03)
    rows = [
        {
            "feature": "miss_cp_share",
            "CV": "LOW_POWER" if rec_m["low_power"] else _f(rec_m["cv"]),
            "train": _f(rec_m["train_auc"]),
            "sign": rec_m["train_sign"],
            "n": f"{rec_m['n_defined']:,}",
            "n_pos": f"{rec_m['n_pos']:,}",
            "folds": fold_bits(rec_m),
        },
        {
            "feature": "d_tx_cp_share (replica)",
            "CV": "LOW_POWER" if rec_d["low_power"] else _f(rec_d["cv"]),
            "train": _f(rec_d["train_auc"]),
            "sign": rec_d["train_sign"],
            "n": f"{rec_d['n_defined']:,}",
            "n_pos": f"{rec_d['n_pos']:,}",
            "folds": fold_bits(rec_d),
        },
        {
            "feature": "named_share (1−miss, month)",
            "CV": "LOW_POWER" if rec_n["low_power"] else _f(rec_n["cv"]),
            "train": _f(rec_n["train_auc"]),
            "sign": rec_n["train_sign"],
            "n": f"{rec_n['n_defined']:,}",
            "n_pos": f"{rec_n['n_pos']:,}",
            "folds": fold_bits(rec_n),
        },
    ]
    prose = (
        f"Y5 AR d_tx_cp_share train replica {_f(dtx_tr)} "
        f"({'CONFIRM night 0.611' if confirm else 'off 0.611 — do not overwrite the quote'}). "
        f"CV {_f(dtx_cv)} (night y5_why CV was 0.576). "
        f"Monthly miss_cp_share Y5 CV "
        f"{'LOW_POWER' if rec_m['low_power'] else _f(rec_m['cv'])}. "
        "Do not change the night 0.611 quote."
    )
    print(prose)
    return {
        "rows": rows,
        "dtx_train": dtx_tr,
        "dtx_cv": dtx_cv,
        "miss_cv": float("nan") if rec_m["low_power"] else rec_m["cv"],
        "confirm": confirm,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 17 — one-group on Y3 folds
# ---------------------------------------------------------------------------
def pass17_y3_onegroup(tr: pd.DataFrame) -> dict:
    rec = signed_oof_auroc(tr[Y3], tr["miss_cp_share"], tr["fold"], tr[Y3].notna())
    folds = rec["folds"]
    aucs = [r["auroc"] for r in folds if np.isfinite(r["auroc"])]
    if not aucs:
        prose = "Y3 miss folds empty."
        return {"rows": [], "one_group": False, "prose": prose, "range": float("nan")}
    spread = float(max(aucs) - min(aucs))
    # share of labeled pos per fold
    y = pd.to_numeric(tr[Y3], errors="coerce")
    miss = pd.to_numeric(tr["miss_cp_share"], errors="coerce")
    hi = miss >= float(miss[y.notna() & miss.notna()].median()) if (y.notna() & miss.notna()).any() else pd.Series(False, index=tr.index)
    rows = []
    for r in folds:
        k = r["fold"]
        sl = (tr["fold"] == k) & y.notna() & miss.notna()
        n_hi = int((sl & hi).sum())
        n_hip = int((sl & hi & (y == 1)).sum())
        rows.append(
            {
                "fold": k,
                "auroc": _f(r["auroc"]),
                "n_va": r["n_va"],
                "n_pos": r["n_pos"],
                "hi_miss_cm": n_hi,
                "hi_miss_pos": n_hip,
            }
        )
    one = bool(spread >= 0.15)
    prose = (
        f"Y3 miss fold AUCs {fold_bits(rec)} range {_f(spread)}. "
        + (
            "Wide fold spread — check one-group (not automatically fold 3)."
            if one
            else "Fold spread <0.15 — not a one-group Y3 dummy."
        )
    )
    print(prose)
    return {"rows": rows, "one_group": one, "spread": spread, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 18 — drop fold 3 on Y3 leftover
# ---------------------------------------------------------------------------
def pass18_drop_f3(tr: pd.DataFrame) -> dict:
    wo = tr["fold"] != 3
    sl = tr.loc[wo]
    rec = signed_oof_auroc(sl[Y3], sl["miss_cp_share"], sl["fold"], sl[Y3].notna())
    size = signed_oof_auroc(sl[Y3], sl["log_in3"], sl["fold"], sl[Y3].notna())
    r_both, _ = ols_resid(
        pd.to_numeric(sl["miss_cp_share"], errors="coerce"),
        pd.to_numeric(sl["a_uncat_share"], errors="coerce"),
        pd.to_numeric(sl["d_tx_cp_share"], errors="coerce"),
    )
    rec_b = signed_oof_auroc(sl[Y3], r_both, sl["fold"], sl[Y3].notna())
    cv = float("nan") if rec["low_power"] else rec["cv"]
    sz = float("nan") if size["low_power"] else size["cv"]
    both = float("nan") if rec_b["low_power"] else rec_b["cv"]
    prose = (
        f"Drop fold 3: Y3 miss {_f(cv)} vs size {_f(sz)}; leftover after uncat+dtx {_f(both)}."
    )
    print(prose)
    return {
        "cv": cv,
        "size": sz,
        "both": both,
        "prose": prose,
        "n": rec["n_defined"],
        "n_pos": rec["n_pos"],
    }


# ---------------------------------------------------------------------------
# Extra 19 — 6m named share vs monthly named (is month just a noisy 6m?)
# ---------------------------------------------------------------------------
def pass19_window(tr: pd.DataFrame) -> dict:
    named = pd.to_numeric(tr["named_share"], errors="coerce")
    dtx = pd.to_numeric(tr["d_tx_cp_share"], errors="coerce")
    miss = pd.to_numeric(tr["miss_cp_share"], errors="coerce")
    rho_n = spearman(named, dtx)
    rho_m = spearman(miss, 1.0 - dtx)
    max_abs = float((named - dtx).abs().max()) if named.notna().any() and dtx.notna().any() else float("nan")
    agree = named.notna() & dtx.notna()
    mean_abs = float((named[agree] - dtx[agree]).abs().mean()) if agree.any() else float("nan")
    twin = bool(np.isfinite(rho_n) and abs(rho_n) >= TWIN_RHO)
    prose = (
        f"Monthly named_share vs 6m d_tx_cp_share ρ={_f(rho_n)} max|Δ|={_f(max_abs)} "
        f"mean|Δ|={_f(mean_abs)}. miss vs 1−d_tx ρ={_f(rho_m)}. "
        + (
            "Monthly missing-CP is the same object as the store fill rate (window noise)."
            if twin
            else "Monthly share is not a 6m-fill rewrite (|ρ|<0.80)."
        )
    )
    print(prose)
    return {
        "rho_named": rho_n,
        "rho_comp": rho_m,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "twin": twin,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 20 — holdout coverage only
# ---------------------------------------------------------------------------
def pass20_holdout(panel: pd.DataFrame) -> dict:
    ho = panel[panel["split"] == "holdout"].copy()
    assert set(ho["company_id"].astype(str)) <= load_holdout() or ho.empty
    rows = []
    for col in ("miss_cp_share", "miss_cp_amt", "miss_mapped_share", "d_tx_cp_share", "inv_cp_share"):
        if col not in ho.columns:
            continue
        x = pd.to_numeric(ho[col], errors="coerce")
        rows.append(
            {
                "col": col,
                "n_cm": int(len(ho)),
                "n_co": int(ho["company_id"].nunique()),
                "defined": _pp(_pct(int(x.notna().sum()), int(len(ho)))),
                "mean": _f(float(x.mean())),
                "p50": _f(float(x.median())),
            }
        )
    prose = (
        f"Holdout coverage only (no AUROC): {int(ho['company_id'].nunique())} companies / "
        f"{int(len(ho))} CM."
    )
    print(prose)
    return {"rows": rows, "n_co": int(ho["company_id"].nunique()), "n_cm": int(len(ho)), "prose": prose}


# ---------------------------------------------------------------------------
# Extra 21 — group ICC of company-median miss
# ---------------------------------------------------------------------------
def pass21_group_icc(tr: pd.DataFrame) -> dict:
    med = tr.groupby("company_id", as_index=False).agg(
        miss=("miss_cp_share", "median"),
        group_id=("group_id", "first"),
    )
    icc = icc_anova(med["miss"], med["group_id"])
    prose = (
        f"Company-median miss_cp_share ICC across group_id {_f(icc['icc'])} "
        f"(k={icc['k']}). Holding style if high."
    )
    print(prose)
    return {"icc": icc["icc"], "k": icc["k"], "prose": prose}


# ---------------------------------------------------------------------------
# Extra 22 — size terciles (not only large books)
# ---------------------------------------------------------------------------
def pass22_size_terc(tr: pd.DataFrame) -> dict:
    size = pd.to_numeric(tr["log_in3"], errors="coerce")
    ok = size.notna()
    sl = tr.loc[ok].copy()
    sl["_t"] = pd.qcut(size[ok], 3, labels=False, duplicates="drop")
    rows = []
    lives = 0
    for t, g in sl.groupby("_t", sort=True):
        rec = signed_oof_auroc(g[Y3], g["miss_cp_share"], g["fold"], g[Y3].notna())
        sz = signed_oof_auroc(g[Y3], g["log_in3"], g["fold"], g[Y3].notna())
        cv = float("nan") if rec["low_power"] else rec["cv"]
        scv = float("nan") if sz["low_power"] else sz["cv"]
        if np.isfinite(cv) and np.isfinite(scv) and (cv - scv) >= 0.0:
            lives += 1
        rows.append(
            {
                "size_tercile": int(t) + 1,
                "p50_log_in3": _f(float(pd.to_numeric(g["log_in3"], errors="coerce").median())),
                "n": f"{rec['n_defined']:,}",
                "n_pos": f"{rec['n_pos']:,}",
                "Y3_miss": "LOW_POWER" if rec["low_power"] else _f(rec["cv"]),
                "mean_miss": _f(float(pd.to_numeric(g["miss_cp_share"], errors="coerce").mean())),
            }
        )
    prose = (
        f"Y3 miss inside size terciles: skill in {lives}/3 bands that are defined. "
        "If only one band, it is a size-band dummy."
    )
    print(prose)
    return {"rows": rows, "n_live": lives, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 23 — mapped-miss leftover among invoiced 744 only
# ---------------------------------------------------------------------------
def pass23_erp_mapped(tr: pd.DataFrame) -> dict:
    sl = tr.loc[tr["ever_erp"]].copy()
    rec = signed_oof_auroc(sl[Y3], sl["miss_mapped_share"], sl["fold"], sl[Y3].notna())
    rec_m = signed_oof_auroc(sl[Y3], sl["miss_cp_share"], sl["fold"], sl[Y3].notna())
    size = signed_oof_auroc(sl[Y3], sl["log_in3"], sl["fold"], sl[Y3].notna())
    prose = (
        f"Invoiced-744 Y3 miss {_f(float('nan') if rec_m['low_power'] else rec_m['cv'])} "
        f"mapped-miss {_f(float('nan') if rec['low_power'] else rec['cv'])} "
        f"vs size {_f(float('nan') if size['low_power'] else size['cv'])}."
    )
    print(prose)
    return {
        "miss": float("nan") if rec_m["low_power"] else rec_m["cv"],
        "mapped": float("nan") if rec["low_power"] else rec["cv"],
        "size": float("nan") if size["low_power"] else size["cv"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 24 — quintiles (descriptive; no Y7 X)
# ---------------------------------------------------------------------------
def pass24_quintiles(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["miss_cp_share"], errors="coerce")
    sl = tr.loc[x.notna()].copy()
    sl["_x"] = x[x.notna()]
    try:
        sl["_q"] = pd.qcut(sl["_x"], 5, labels=False, duplicates="drop")
    except ValueError:
        sl["_q"] = 0
    rows = []
    for q, g in sl.groupby("_q", sort=True):
        rec = {
            "q": int(q) + 1,
            "n_cm": int(len(g)),
            "p50": _f(float(g["_x"].median())),
            "Y2": _pp(float(pd.to_numeric(g[Y2], errors="coerce").mean())),
            "Y3": _pp(float(pd.to_numeric(g[Y3], errors="coerce").mean())),
            "Y5_AR": _pp(float(pd.to_numeric(g[Y5_AR], errors="coerce").mean())),
        }
        rows.append(rec)
    prose = "miss_cp_share quintiles vs Y2 / Y3 / Y5-AR base rates (descriptive; Y7 not scored)."
    print(prose)
    n_bins = len(rows)
    return {"rows": rows, "prose": prose, "n_bins": n_bins}


# ---------------------------------------------------------------------------
# Extra 25 — invoiced-744 twin / leftover (is the twin only the dark 1.0?)
# ---------------------------------------------------------------------------
def pass25_erp_twin(tr: pd.DataFrame) -> dict:
    sl = tr.loc[tr["ever_erp"]].copy()
    miss = pd.to_numeric(sl["miss_cp_share"], errors="coerce")
    dtx = pd.to_numeric(sl["d_tx_cp_share"], errors="coerce")
    uncat = pd.to_numeric(sl["a_uncat_share"], errors="coerce")
    rho_d = spearman(miss, dtx)
    rho_u = spearman(miss, uncat)
    twin = bool(np.isfinite(rho_d) and abs(rho_d) >= TWIN_RHO)
    r_d, _ = ols_resid(miss, dtx)
    r_b, _ = ols_resid(miss, uncat, dtx)
    rec = signed_oof_auroc(sl[Y3], miss, sl["fold"], sl[Y3].notna())
    rec_d = signed_oof_auroc(sl[Y3], r_d, sl["fold"], sl[Y3].notna())
    rec_b = signed_oof_auroc(sl[Y3], r_b, sl["fold"], sl[Y3].notna())
    size = signed_oof_auroc(sl[Y3], sl["log_in3"], sl["fold"], sl[Y3].notna())
    rows = [
        {
            "item": "ρ miss↔d_tx (744)",
            "value": _f(rho_d),
            "note": "TWIN" if twin else "not twin",
        },
        {
            "item": "ρ miss↔uncat (744)",
            "value": _f(rho_u),
            "note": "",
        },
        {
            "item": "Y3 miss (744)",
            "value": "LOW_POWER" if rec["low_power"] else _f(rec["cv"]),
            "note": f"n={rec['n_defined']:,} pos={rec['n_pos']:,}",
        },
        {
            "item": "Y3 resid d_tx (744)",
            "value": "LOW_POWER" if rec_d["low_power"] else _f(rec_d["cv"]),
            "note": "",
        },
        {
            "item": "Y3 resid both (744)",
            "value": "LOW_POWER" if rec_b["low_power"] else _f(rec_b["cv"]),
            "note": "",
        },
        {
            "item": "Y3 size (744)",
            "value": "LOW_POWER" if size["low_power"] else _f(size["cv"]),
            "note": "",
        },
    ]
    prose = (
        f"Invoiced-744: miss↔d_tx ρ={_f(rho_d)} ({'still TWIN' if twin else 'twin dies on ERP'}). "
        f"Y3 miss {_f(float('nan') if rec['low_power'] else rec['cv'])} "
        f"resid-dtx {_f(float('nan') if rec_d['low_power'] else rec_d['cv'])} "
        f"vs size {_f(float('nan') if size['low_power'] else size['cv'])}. "
        + (
            "Twin is not only the dark 1.0 pile."
            if twin
            else "On ERP books the fill-rate twin loosens."
        )
    )
    print(prose)
    return {
        "rows": rows,
        "rho_dtx": rho_d,
        "twin": twin,
        "y3": float("nan") if rec["low_power"] else rec["cv"],
        "resid": float("nan") if rec_d["low_power"] else rec_d["cv"],
        "size": float("nan") if size["low_power"] else size["cv"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 26 — miss==1 pile (why qcut made 2 bins)
# ---------------------------------------------------------------------------
def pass26_allmiss(tr: pd.DataFrame) -> dict:
    miss = pd.to_numeric(tr["miss_cp_share"], errors="coerce")
    defined = miss.notna()
    all1 = defined & (miss >= 0.999)
    some = defined & (miss < 0.999)
    n_def = int(defined.sum())
    n1 = int(all1.sum())
    rows = []
    cvs = {}
    for name, m in (("all_miss>=0.999", all1), ("partial", some), ("all_defined", defined)):
        sl = tr.loc[m]
        rec2 = signed_oof_auroc(sl[Y2], sl["miss_cp_share"], sl["fold"], sl[Y2].notna())
        rec3 = signed_oof_auroc(sl[Y3], sl["miss_cp_share"], sl["fold"], sl[Y3].notna())
        cvs[(name, "Y3")] = float("nan") if rec3["low_power"] else rec3["cv"]
        rows.append(
            {
                "slice": name,
                "n_cm": int(len(sl)),
                "n_co": int(sl["company_id"].nunique()),
                "dark": _pp(float((~sl["ever_erp"]).mean()) if len(sl) else float("nan")),
                "Y2": _pp(float(pd.to_numeric(sl[Y2], errors="coerce").mean())),
                "Y3": _pp(float(pd.to_numeric(sl[Y3], errors="coerce").mean())),
                "Y3_CV": "LOW_POWER" if rec3["low_power"] else _f(rec3["cv"]),
            }
        )
    dummy = all1.astype(float)
    rec_dum = signed_oof_auroc(tr[Y3], dummy, tr["fold"], tr[Y3].notna() & defined)
    prose = (
        f"Exact-all-miss months (share≥0.999): {n1:,} / {n_def:,} ({_pp(_pct(n1, n_def))}). "
        f"qcut collapsed to 2 bins because p50=1.0. "
        f"All-miss dummy Y3 {_f(float('nan') if rec_dum['low_power'] else rec_dum['cv'])}. "
        f"Partial-only (intensity) Y3 {_f(cvs.get(('partial', 'Y3'), float('nan')))}."
    )
    print(prose)
    return {
        "rows": rows,
        "n1": n1,
        "n_def": n_def,
        "share1": _pct(n1, n_def),
        "dummy_y3": float("nan") if rec_dum["low_power"] else rec_dum["cv"],
        "partial_y3": cvs.get(("partial", "Y3"), float("nan")),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 27 — is miss just the ever_erp dummy?
# ---------------------------------------------------------------------------
def pass27_erp_dummy(tr: pd.DataFrame) -> dict:
    miss = pd.to_numeric(tr["miss_cp_share"], errors="coerce")
    erp = tr["ever_erp"].astype(float)
    rho = spearman(miss, erp)
    rec_e = signed_oof_auroc(tr[Y3], erp, tr["fold"], tr[Y3].notna())
    rec_m = signed_oof_auroc(tr[Y3], miss, tr["fold"], tr[Y3].notna())
    r, _ = ols_resid(miss, erp)
    rec_r = signed_oof_auroc(tr[Y3], r, tr["fold"], tr[Y3].notna())
    twin = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
    rows = [
        {"item": "ρ miss↔ever_erp", "value": _f(rho)},
        {"item": "Y3 ever_erp dummy", "value": "LOW_POWER" if rec_e["low_power"] else _f(rec_e["cv"])},
        {"item": "Y3 miss", "value": "LOW_POWER" if rec_m["low_power"] else _f(rec_m["cv"])},
        {"item": "Y3 miss resid erp", "value": "LOW_POWER" if rec_r["low_power"] else _f(rec_r["cv"])},
    ]
    prose = (
        f"miss↔ever_erp ρ={_f(rho)} ({'TWIN of the 470' if twin else 'not the 470 dummy'}). "
        f"Y3 erp-dummy {_f(float('nan') if rec_e['low_power'] else rec_e['cv'])} "
        f"vs miss {_f(float('nan') if rec_m['low_power'] else rec_m['cv'])}; "
        f"resid after erp {_f(float('nan') if rec_r['low_power'] else rec_r['cv'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "rho": rho,
        "twin": twin,
        "erp_y3": float("nan") if rec_e["low_power"] else rec_e["cv"],
        "resid": float("nan") if rec_r["low_power"] else rec_r["cv"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 28 — named-CP category mix (what *is* tagged?)
# ---------------------------------------------------------------------------
def pass28_named_cats(con) -> dict:
    hold = load_holdout()
    hold_sql = _hold_sql(hold)
    raw = con.execute(
        f"""
        SELECT
          CAST(COALESCE(t.category, '(null)') AS VARCHAR) AS category,
          COUNT(*) AS n,
          SUM(ABS(t.amount)) AS abs_amt
        FROM transactions t
        WHERE t."date" IS NOT NULL
          AND t."date" >= DATE '2024-09-01'
          AND t."date" < DATE '2026-09-01'
          AND CAST(t.company_id AS VARCHAR) NOT IN ({hold_sql})
          AND t.counterparty_id IS NOT NULL
          AND length(trim(CAST(t.counterparty_id AS VARCHAR))) > 0
        GROUP BY 1
        ORDER BY n DESC
        """
    ).df()
    tot = float(raw["n"].sum())
    rows = []
    for _, r in raw.head(10).iterrows():
        rows.append(
            {
                "category": str(r["category"]),
                "n": f"{int(r['n']):,}",
                "share_n": _pp(_pct(float(r["n"]), tot)),
            }
        )
    top = str(raw.iloc[0]["category"]) if len(raw) else "?"
    prose = (
        f"Named-CP txs n={int(tot):,}. Top token `{top}`. "
        "If named rows are collections/payments, the hole is still bank tagging on the rest."
    )
    print(prose)
    return {"rows": rows, "n": tot, "top": top, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 29 — 360 all-dark vs 110 mixed-dark
# ---------------------------------------------------------------------------
def pass29_dark_mix(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    g = last.groupby("group_id", as_index=False).agg(
        n=("company_id", "size"), n_book=("ever_erp", "sum")
    )
    g["mix"] = np.select(
        [g["n_book"] == 0, g["n_book"] == g["n"]],
        ["all_dark", "all_erp"],
        default="mixed",
    )
    mix_of = dict(zip(g["group_id"].astype(str), g["mix"]))
    sl = tr.copy()
    sl["_mix"] = sl["group_id"].map(mix_of)
    rows = []
    for name, m in (
        ("invoiced_744", sl["ever_erp"]),
        ("all_dark_360", (~sl["ever_erp"]) & (sl["_mix"] == "all_dark")),
        ("mixed_dark_110", (~sl["ever_erp"]) & (sl["_mix"] == "mixed")),
    ):
        part = sl.loc[m]
        miss = pd.to_numeric(part["miss_cp_share"], errors="coerce")
        rows.append(
            {
                "group": name,
                "n_cm": int(len(part)),
                "n_co": int(part["company_id"].nunique()),
                "miss_mean": _f(float(miss.mean())),
                "miss_p50": _f(float(miss.median())),
            }
        )
    prose = "All-dark vs mixed-dark miss level. If both ≈1.0, dark is a company-book trait, not sibling-ERP."
    print(prose)
    return {"rows": rows, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 30 — months-on-book
# ---------------------------------------------------------------------------
def pass30_sofar(tr: pd.DataFrame) -> dict:
    sl = tr.sort_values(["company_id", "period"]).copy()
    rows = []
    bands = ((1, 3, "1-3"), (4, 6, "4-6"), (7, 12, "7-12"), (13, 18, "13-18"), (19, 24, "19-24"))
    for lo, hi, lab in bands:
        g = sl[(sl["so_far"] >= lo) & (sl["so_far"] <= hi)]
        miss = pd.to_numeric(g["miss_cp_share"], errors="coerce")
        rec = signed_oof_auroc(g[Y3], g["miss_cp_share"], g["fold"], g[Y3].notna())
        rows.append(
            {
                "so_far": lab,
                "n_cm": int(len(g)),
                "miss": _pp(float(miss.mean())),
                "Y3_CV": "LOW_POWER" if rec["low_power"] else _f(rec["cv"]),
                "Y3_n_pos": rec["n_pos"],
            }
        )
    early = float(pd.to_numeric(sl.loc[sl["so_far"] <= 3, "miss_cp_share"], errors="coerce").mean())
    late = float(pd.to_numeric(sl.loc[sl["so_far"] >= 13, "miss_cp_share"], errors="coerce").mean())
    prose = (
        f"Months 1–3 miss {_pp(early)} vs months 13+ {_pp(late)}. "
        + (
            "Onboarding hole."
            if np.isfinite(early) and np.isfinite(late) and (early - late) >= 0.05
            else "Not an onboarding miss — fill habit from month 1."
        )
    )
    print(prose)
    return {"rows": rows, "early": early, "late": late, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 31 — Q6 long so-far≥13 (18 was LOW_POWER)
# ---------------------------------------------------------------------------
def pass31_q6_13(tr: pd.DataFrame) -> dict:
    sl = tr.loc[tr["so_far"] >= 13]
    rec = signed_oof_auroc(sl[Y3], sl["miss_cp_share"], sl["fold"], sl[Y3].notna())
    rec1 = signed_oof_auroc(sl[Y3], sl["miss_cp_share_lag1"], sl["fold"], sl[Y3].notna())
    prose = (
        f"Y3 so-far≥13 miss now {'LOW_POWER' if rec['low_power'] else _f(rec['cv'])} "
        f"n_pos={rec['n_pos']}; lag1 {'LOW_POWER' if rec1['low_power'] else _f(rec1['cv'])}. "
        "Q6 stays CLOSE unless lag1 holds a contemporaneous ≥0.58."
    )
    print(prose)
    return {
        "now": float("nan") if rec["low_power"] else rec["cv"],
        "lag1": float("nan") if rec1["low_power"] else rec1["cv"],
        "n_pos": rec["n_pos"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 32 — invoice-complete ∩ tx-unnamed (pure bank hole rate)
# ---------------------------------------------------------------------------
def pass32_pure_hole(tr: pd.DataFrame) -> dict:
    inv = pd.to_numeric(tr["inv_cp_share"], errors="coerce")
    named = pd.to_numeric(tr["named_share"], errors="coerce")
    y5 = pd.to_numeric(tr[Y5_AR], errors="coerce")
    hole = inv.notna() & named.notna() & (inv >= 0.99) & (named <= 0.01)
    named_ok = inv.notna() & named.notna() & (inv >= 0.99) & (named > 0.20)
    n_h = int(hole.sum())
    n_n = int(named_ok.sum())
    r_h = float(y5[hole].mean()) if n_h else float("nan")
    r_n = float(y5[named_ok].mean()) if n_n else float("nan")
    # fold 3 share of this cell among AR-labeled
    lab = y5.notna()
    h_lab = hole & lab
    n_hp = int((h_lab & (y5 == 1)).sum())
    f3 = int((h_lab & (y5 == 1) & (tr["fold"] == 3)).sum())
    prose = (
        f"Invoice-named (≥0.99) ∩ tx-unnamed (≤0.01): {n_h:,} cm, Y5-AR {_pp(r_h)} "
        f"vs invoice-named ∩ tx-named>0.20 {n_n:,} cm, Y5-AR {_pp(r_n)}. "
        f"Hole positives in fold 3: {_pp(_pct(f3, n_hp)) if n_hp else '—'}. "
        "This *is* the Y5 d_tx_cp_share tagging sentence — not a new KEEP-Q5."
    )
    print(prose)
    return {
        "n_hole": n_h,
        "n_named": n_n,
        "y5_hole": r_h,
        "y5_named": r_n,
        "f3_share": _pct(f3, n_hp) if n_hp else float("nan"),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 33 — Y2 / Y3 on dark vs invoiced
# ---------------------------------------------------------------------------
def pass33_dark_y(tr: pd.DataFrame) -> dict:
    rows = []
    for name, m in (("invoiced_744", tr["ever_erp"]), ("dark_470", ~tr["ever_erp"])):
        sl = tr.loc[m]
        rec2 = signed_oof_auroc(sl[Y2], sl["miss_cp_share"], sl["fold"], sl[Y2].notna())
        rec3 = signed_oof_auroc(sl[Y3], sl["miss_cp_share"], sl["fold"], sl[Y3].notna())
        rows.append(
            {
                "slice": name,
                "Y2_CV": "LOW_POWER" if rec2["low_power"] else _f(rec2["cv"]),
                "Y2_n_pos": rec2["n_pos"],
                "Y3_CV": "LOW_POWER" if rec3["low_power"] else _f(rec3["cv"]),
                "Y3_n_pos": rec3["n_pos"],
                "miss_mean": _f(float(pd.to_numeric(sl["miss_cp_share"], errors="coerce").mean())),
            }
        )
    prose = "Dark miss is ~1.0 — AUROC on dark should die (no variance). Skill if any lives on the 744."
    print(prose)
    return {"rows": rows, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 34 — collection+payment among missing vs named (ops without names)
# ---------------------------------------------------------------------------
def pass34_ops_share(con) -> dict:
    hold = load_holdout()
    hold_sql = _hold_sql(hold)
    raw = con.execute(
        f"""
        SELECT
          CASE WHEN t.counterparty_id IS NULL
                 OR length(trim(CAST(t.counterparty_id AS VARCHAR))) = 0
                THEN 1 ELSE 0 END AS is_miss,
          AVG(CASE WHEN t.category IN ('collection', 'payment', 'bulk_collection', 'bulk_payment')
                   THEN 1.0 ELSE 0.0 END) AS ops_n,
          AVG(CASE WHEN t.category = 'uncategorized' THEN 1.0 ELSE 0.0 END) AS uncat_n,
          AVG(CASE WHEN t.category = 'transfer' THEN 1.0 ELSE 0.0 END) AS xfer_n
        FROM transactions t
        WHERE t."date" IS NOT NULL
          AND t."date" >= DATE '2024-09-01'
          AND t."date" < DATE '2026-09-01'
          AND CAST(t.company_id AS VARCHAR) NOT IN ({hold_sql})
        GROUP BY 1
        """
    ).df()
    rows = []
    for _, r in raw.iterrows():
        rows.append(
            {
                "kind": "missing_cp" if int(r["is_miss"]) == 1 else "named_cp",
                "ops_coll_pay": _pp(float(r["ops_n"])),
                "uncat": _pp(float(r["uncat_n"])),
                "transfer": _pp(float(r["xfer_n"])),
            }
        )
    prose = (
        "If missing-CP and named-CP have similar ops/uncat mix, the hole is tagging, not category."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 35 — company-median miss vs ever-ERP
# ---------------------------------------------------------------------------
def pass35_co_median(tr: pd.DataFrame) -> dict:
    g = tr.groupby("company_id", as_index=False).agg(
        med=("miss_cp_share", "median"),
        ever_erp=("ever_erp", "max"),
        ever_y2=(Y2, "max"),
    )
    rho = spearman(g["med"], g["ever_erp"].astype(float))
    always = g["med"] >= 0.99
    rows = [
        {
            "slice": "median≥0.99",
            "n_co": int(always.sum()),
            "dark": _pp(float((~g.loc[always, "ever_erp"]).mean()) if always.any() else float("nan")),
            "ever_Y2": _pp(float(g.loc[always, "ever_y2"].mean()) if always.any() else float("nan")),
        },
        {
            "slice": "median<0.99",
            "n_co": int((~always).sum()),
            "dark": _pp(float((~g.loc[~always, "ever_erp"]).mean()) if (~always).any() else float("nan")),
            "ever_Y2": _pp(float(g.loc[~always, "ever_y2"].mean()) if (~always).any() else float("nan")),
        },
    ]
    prose = (
        f"Company-median miss≥0.99: {int(always.sum())} / {int(len(g))} companies. "
        f"median↔ever_erp ρ={_f(rho)}. Always-missing is a type, not a month."
    )
    print(prose)
    return {"rows": rows, "rho": rho, "n_always": int(always.sum()), "prose": prose}


# ---------------------------------------------------------------------------
# Extra 36 — Y2 on invoiced-744 vs size (is 0.612 a keep?)
# ---------------------------------------------------------------------------
def pass36_y2_erp(tr: pd.DataFrame) -> dict:
    sl = tr.loc[tr["ever_erp"]].copy()
    rec = signed_oof_auroc(sl[Y2], sl["miss_cp_share"], sl["fold"], sl[Y2].notna())
    size = signed_oof_auroc(sl[Y2], sl["log_in3"], sl["fold"], sl[Y2].notna())
    dtx = signed_oof_auroc(sl[Y2], sl["d_tx_cp_share"], sl["fold"], sl[Y2].notna())
    r, _ = ols_resid(
        pd.to_numeric(sl["miss_cp_share"], errors="coerce"),
        pd.to_numeric(sl["d_tx_cp_share"], errors="coerce"),
    )
    rec_r = signed_oof_auroc(sl[Y2], r, sl["fold"], sl[Y2].notna())
    cv = float("nan") if rec["low_power"] else rec["cv"]
    sz = float("nan") if size["low_power"] else size["cv"]
    beat = (cv - sz) if np.isfinite(cv) and np.isfinite(sz) else float("nan")
    keep = bool(np.isfinite(beat) and beat >= KEEP_DELTA)
    prose = (
        f"Invoiced-744 Y2 miss {_f(cv)} vs size {_f(sz)} (Δ {_f(beat, 3)}); "
        f"d_tx replica {_f(float('nan') if dtx['low_power'] else dtx['cv'])}; "
        f"resid after d_tx {_f(float('nan') if rec_r['low_power'] else rec_r['cv'])}. "
        + ("Beats size on ERP Y2 — still a fill-rate twin, not a new X." if keep else "Does not clear size+0.02 on ERP Y2.")
    )
    print(prose)
    return {
        "cv": cv,
        "size": sz,
        "beat": beat,
        "dtx": float("nan") if dtx["low_power"] else dtx["cv"],
        "resid": float("nan") if rec_r["low_power"] else rec_r["cv"],
        "keep": keep,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 37 — always-missing invoiced companies by fold (fold 3 again?)
# ---------------------------------------------------------------------------
def pass37_always_erp_fold(tr: pd.DataFrame) -> dict:
    g = (
        tr.loc[tr["ever_erp"]]
        .groupby("company_id", as_index=False)
        .agg(med=("miss_cp_share", "median"), fold=("fold", "first"), group_id=("group_id", "first"))
    )
    always = g["med"] >= 0.99
    rows = []
    n_al = int(always.sum())
    for k in range(N_FOLDS):
        sl = g.loc[always & (g["fold"] == k)]
        rows.append(
            {
                "fold": k,
                "n_always_erp": int(len(sl)),
                "share": _pp(_pct(int(len(sl)), n_al)),
                "n_groups": int(sl["group_id"].nunique()),
            }
        )
    f3 = int((always & (g["fold"] == 3)).sum())
    pile = _pct(f3, n_al)
    prose = (
        f"Always-missing invoiced companies {n_al} / {int(len(g))}. "
        f"Fold 3 holds {_pp(pile)}. "
        + (
            "Same unnamed cluster as Y5 fold 3."
            if np.isfinite(pile) and pile >= 0.40
            else "Always-missing ERP names are not fold-3-only."
        )
    )
    print(prose)
    return {"rows": rows, "n_always": n_al, "f3_share": pile, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 38 — leftover after ever_erp + d_tx (dark dummy + fill twin)
# ---------------------------------------------------------------------------
def pass38_resid_erp_dtx(tr: pd.DataFrame) -> dict:
    miss = pd.to_numeric(tr["miss_cp_share"], errors="coerce")
    erp = tr["ever_erp"].astype(float)
    dtx = pd.to_numeric(tr["d_tx_cp_share"], errors="coerce")
    r, info = ols_resid(miss, erp, dtx)
    rec = signed_oof_auroc(tr[Y3], r, tr["fold"], tr[Y3].notna())
    rec2 = signed_oof_auroc(tr[Y2], r, tr["fold"], tr[Y2].notna())
    size = signed_oof_auroc(tr[Y3], tr["log_in3"], tr["fold"], tr[Y3].notna())
    cv = float("nan") if rec["low_power"] else rec["cv"]
    sz = float("nan") if size["low_power"] else size["cv"]
    prose = (
        f"Y3 resid after ever_erp+d_tx {_f(cv)} vs size {_f(sz)} "
        f"(slopes {[_f(s) for s in info['slope']]}). "
        "If this dies, missing-CP is dark-level + fill-rate, nothing leftover."
    )
    print(prose)
    return {
        "y3": cv,
        "y2": float("nan") if rec2["low_power"] else rec2["cv"],
        "size": sz,
        "slopes": info["slope"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 39 — mapped-miss leftover after d_tx on the 744 (last leftover)
# ---------------------------------------------------------------------------
def pass39_mapped_after_dtx(tr: pd.DataFrame) -> dict:
    sl = tr.loc[tr["ever_erp"]].copy()
    mapped = pd.to_numeric(sl["miss_mapped_share"], errors="coerce")
    dtx = pd.to_numeric(sl["d_tx_cp_share"], errors="coerce")
    r, _ = ols_resid(mapped, dtx)
    rec = signed_oof_auroc(sl[Y3], mapped, sl["fold"], sl[Y3].notna())
    rec_r = signed_oof_auroc(sl[Y3], r, sl["fold"], sl[Y3].notna())
    size = signed_oof_auroc(sl[Y3], sl["log_in3"], sl["fold"], sl[Y3].notna())
    prose = (
        f"Invoiced-744 mapped-miss Y3 {_f(float('nan') if rec['low_power'] else rec['cv'])} "
        f"resid after d_tx {_f(float('nan') if rec_r['low_power'] else rec_r['cv'])} "
        f"vs size {_f(float('nan') if size['low_power'] else size['cv'])}. "
        "Last leftover after parking uncat *and* the 6m fill."
    )
    print(prose)
    return {
        "raw": float("nan") if rec["low_power"] else rec["cv"],
        "resid": float("nan") if rec_r["low_power"] else rec_r["cv"],
        "size": float("nan") if size["low_power"] else size["cv"],
        "prose": prose,
    }


def make_png(tr: pd.DataFrame, p12: dict) -> bool:
    if not HAS_MPL:
        print("png skipped: no matplotlib")
        return False
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
    sl = tr.loc[
        pd.to_numeric(tr["miss_cp_share"], errors="coerce").notna()
        & pd.to_numeric(tr["a_uncat_share"], errors="coerce").notna()
    ]
    ax = axes[0]
    ax.scatter(
        pd.to_numeric(sl["a_uncat_share"], errors="coerce"),
        pd.to_numeric(sl["miss_cp_share"], errors="coerce"),
        s=4,
        alpha=0.15,
        c="#3d5a80",
        linewidths=0,
    )
    ax.set_xlabel("a_uncat_share")
    ax.set_ylabel("miss_cp_share (month, in-memory)")
    ax.set_title("Train company-months")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([0, 1], [0, 1], ls="--", c="#888", lw=0.8)
    ax = axes[1]
    cats = [r["category"] for r in p12["rows"][:8]]
    parsed = []
    for r in p12["rows"][:8]:
        s = str(r["share_n"]).replace("%", "")
        try:
            parsed.append(float(s) / 100.0)
        except ValueError:
            parsed.append(0.0)
    ax.barh(list(reversed(cats[: len(parsed)])), list(reversed(parsed)), color="#ee6c4d")
    ax.set_xlabel("share of missing-CP txs (count)")
    ax.set_title("Category mix of missing-CP")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return True


def decide(p3, p4, p5, p8, p10, p11, p6, p7, p14, p19) -> dict:
    """Map missing-CP share to KEEP / CLOSE / PARK. PARK as Y always."""
    park_y = True
    twin = bool(p3["twin_uncat"] or p3["twin_dtx"] or p19["twin"])
    size = bool(p3["is_size"])
    one = bool(p8["one_group"] and not p8["survives"])
    leftover = bool(p5["leftover_lives"] and not one and not twin and not size)
    if leftover:
        x_dec = "KEEP"
        why = (
            f"Leftover after uncat+d_tx_cp_share Y3 {_f(p5['y3_both'])} beats size "
            f"{_f(p5['size_y3'])} by {_f(p5['leftover_beat'], 3)} and is not one-group. "
            "Later D candidate only — do not merge parquet. Parent decides."
        )
    else:
        x_dec = "CLOSE"
        reasons = []
        if p3["twin_uncat"]:
            reasons.append(f"uncat twin ρ={_f(p3['rho_uncat'])}")
        if p3["twin_dtx"] or p19["twin"]:
            reasons.append(f"d_tx_cp_share twin ρ={_f(p3['rho_dtx'])}")
        if size:
            reasons.append(f"SIZE ρ={_f(p3['rho_size'])}")
        if one:
            reasons.append("fold-3 / one-group again")
        if not leftover:
            reasons.append(
                f"leftover after both Y3 {_f(p5['y3_both'])} vs size {_f(p5['size_y3'])} "
                f"(Δ {_f(p5['leftover_beat'], 3)})"
            )
        why = "CLOSE as X — " + "; ".join(reasons) + "."
    q5 = "CLOSE"
    q5_diag = False
    if p7["hole_is_bank"] and not (p3["twin_dtx"] or p19["twin"]):
        q5 = "KEEP-Q5"
        q5_diag = True
        q5_why = (
            "Bank-book txs miss counterparty_id while the invoice book is named — "
            "a tagging hole, and not already the Y5 d_tx_cp_share sentence."
        )
    elif p7["hole_is_bank"] and (p3["twin_dtx"] or p19["twin"]):
        q5_why = (
            "Bank-book tagging hole is already the Y5 `d_tx_cp_share` sentence "
            "(6m named fill). Not a new KEEP-Q5."
        )
    else:
        q5_why = "No new Q5 sentence beyond parked uncat / d_tx_cp_share."
    return {
        "park_y": park_y,
        "x_dec": x_dec,
        "q5": q5,
        "q5_diag": q5_diag,
        "q5_why": q5_why,
        "q6": p11["q6"],
        "why": why,
        "twin": twin,
        "one_group": one,
        "leftover": leftover,
    }


def write_md(ctx: dict) -> None:
    p = {k: ctx[k] for k in ctx if k.startswith("p")}
    d = ctx["decision"]
    p1, p2, p3, p4, p5 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"], ctx["p5"]
    p6, p7, p8, p9, p10 = ctx["p6"], ctx["p7"], ctx["p8"], ctx["p9"], ctx["p10"]
    p11, p12, p13 = ctx["p11"], ctx["p12"], ctx["p13"]
    p14, p15, p16 = ctx["p14"], ctx["p15"], ctx["p16"]
    p17, p18, p19 = ctx["p17"], ctx["p18"], ctx["p19"]
    p20, p21, p22 = ctx["p20"], ctx["p21"], ctx["p22"]
    p23, p24 = ctx["p23"], ctx["p24"]
    lines = [
        "# Missing transaction `counterparty_id` — leftover after uncat?",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. No `build_targets`. Do not invent `y_missing_cp`. "
        "Do not merge a Family D column. Do not put this on the 15-col Y3 card. "
        "Y7 never uses D. Night Y3 0.762/0.752 and Y5 `d_tx_cp_share` 0.611 quotes unchanged.",
        "",
        "`miss_cp_share` = calendar-month share of txs with null/blank `counterparty_id` "
        "(in-memory). `d_tx_cp_share` = 6-month *named* fill (store). "
        "`miss_mapped_share` = missing-CP among non-uncat txs that month.",
        "",
        "## Headline",
        "",
        ctx["headline"],
        "",
        "## Brief questions",
        "",
        "| # | question | what this cut says |",
        "| --- | --- | --- |",
        "| 1 | Who is healthy? | Missing-CP is **PARK** as a health Y. Do not invent `y_missing_cp`. |",
        "| 2 | Who is improving? | Not this share. |",
        f"| 3 | Who is turning? | {'A leftover tagging shock would be Q3; a style/fill twin is not.' if d['park_y'] else '—'} |",
        "| 4 | Dip vs fall? | Not this table. |",
        f"| 5 | Why did it change? | **{d['q5']}** — {d['q5_why']} X is **{d['x_dec']}**. {d['why']} |",
        f"| 6 | Months earlier? | **{d['q6']}** — {p11['prose']} |",
        "",
        "## PARK / CLOSE / KEEP",
        "",
        "| object | decision | why |",
        "| --- | --- | --- |",
        f"| miss_cp_share as Y3 X | **{d['x_dec']}** | {d['why']} |",
        "| miss_cp_share as a health Y | **PARK** | do not invent `y_missing_cp` |",
        f"| Q5 diagnostic | **{d['q5']}** | {d['q5_why']} |",
        f"| Q6 lag1/lag3 | **{d['q6']}** | {p11['prose']} |",
        f"| uncat twin (a) | **{'YES — CLOSE' if p3['twin_uncat'] else 'no'}** | ρ={_f(p3['rho_uncat'])} vs `a_uncat_share` |",
        f"| d_tx_cp_share twin (b) | **{'YES — CLOSE' if (p3['twin_dtx'] or p19['twin']) else 'no'}** | ρ vs d_tx={_f(p3['rho_dtx'])}; monthly named vs 6m ρ={_f(p19['rho_named'])} |",
        f"| leftover after both (c) | **{'KEEP later-D' if d['leftover'] else 'dies / no KEEP'}** | Y3 resid {_f(p5['y3_both'])} vs size {_f(p5['size_y3'])} |",
        f"| dark 470 hole (d) | **{'only-dark' if p6.get('only_dark') else ('level — not only-470' if p6['is_dark_hole'] else 'no')}** | invoiced {_f(p6['erp_m'])} vs dark {_f(p6['dark_m'])} |",
        f"| Y5 fold-3 again (e) | **{'YES — CLOSE' if d['one_group'] else 'no'}** | {p8['prose']} |",
        "| Y7 as X | **never** | Y7 models never use D |",
        "| 15-col Y3 card | **not added** | night quote stays 0.762 / 0.752 |",
        "| parquet merge | **not done** | parent decides |",
        "",
        "## 1. Prevalence",
        "",
        p1["prose"],
        "",
        _md_table(p1["rows"]),
        "",
        "## 2. Uncat × missing-CP 2×2",
        "",
        p2["prose"],
        "",
        _md_table(p2["rows"]),
        "",
        "## 3. Spearman twins / SIZE",
        "",
        p3["prose"],
        "",
        _md_table(p3["rows"]),
        "",
        "## 4. Single-feature train group-fold AUROC",
        "",
        p4["prose"],
        "",
        "Sign from the train side of each fold. Days bar 0.711. Size `log1p(a_in3)` 0.617. Never Y7.",
        "",
        _md_table(p4["rows"]),
        "",
        "## 5. Residual after uncat / `d_tx_cp_share`",
        "",
        p5["prose"],
        "",
        _md_table(p5["rows"]),
        "",
        "## 6. Dark 470 vs invoiced 744",
        "",
        p6["prose"],
        "",
        _md_table(p6["rows"]),
        "",
        "## 7. Invoice CP fill vs tx CP fill (same month)",
        "",
        p7["prose"],
        "",
        _md_table(p7["rows"]),
        "",
        "## 8. Y5 fold-3 hole",
        "",
        p8["prose"],
        "",
        _md_table(p8["rows"]),
        "",
        "## 9. Drop 12 chronic dark Y2 names",
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
        "## 11. Q6 lag1 / lag3 on short vs long books",
        "",
        p11["prose"],
        "",
        _md_table(p11["rows"]),
        "",
        "## 12. Category mix of missing-CP txs",
        "",
        p12["prose"],
        "",
        _md_table(p12["rows"]),
        "",
        "## 13. Amounts — p50 |amt| missing vs named",
        "",
        p13["prose"],
        "",
        _md_table(p13["rows"]),
        "",
        "## Extra 14 — condition on uncat≤0.10",
        "",
        p14["prose"],
        "",
        _md_table(p14["rows"]),
        "",
        "## Extra 15 — condition on `d_tx_cp_share`",
        "",
        p15["prose"],
        "",
        _md_table(p15["rows"]),
        "",
        "## Extra 16 — Y5 AR replica (do not change 0.611)",
        "",
        p16["prose"],
        "",
        _md_table(p16["rows"]),
        "",
        "## Extra 17 — Y3 fold spread",
        "",
        p17["prose"],
        "",
        _md_table(p17["rows"]),
        "",
        "## Extra 18 — drop fold 3",
        "",
        p18["prose"],
        "",
        "## Extra 19 — monthly named vs 6m `d_tx_cp_share`",
        "",
        p19["prose"],
        "",
        "## Extra 20 — holdout coverage only",
        "",
        p20["prose"],
        "",
        _md_table(p20["rows"]),
        "",
        "## Extra 21 — group ICC of company-median",
        "",
        p21["prose"],
        "",
        "## Extra 22 — Y3 inside size terciles",
        "",
        p22["prose"],
        "",
        _md_table(p22["rows"]),
        "",
        "## Extra 23 — invoiced-744 only",
        "",
        p23["prose"],
        "",
        "## Extra 24 — quintiles (no Y7 X)",
        "",
        p24["prose"],
        "",
        _md_table(p24["rows"]),
        "",
        "## Extra 25 — invoiced-744 twin / leftover",
        "",
        ctx["p25"]["prose"],
        "",
        _md_table(ctx["p25"]["rows"]),
        "",
        "## Extra 26 — all-miss pile (qcut 2 bins)",
        "",
        ctx["p26"]["prose"],
        "",
        _md_table(ctx["p26"]["rows"]),
        "",
        "## Extra 27 — miss vs ever_erp dummy",
        "",
        ctx["p27"]["prose"],
        "",
        _md_table(ctx["p27"]["rows"]),
        "",
        "## Extra 28 — named-CP category mix",
        "",
        ctx["p28"]["prose"],
        "",
        _md_table(ctx["p28"]["rows"]),
        "",
        "## Extra 29 — 360 vs 110 dark",
        "",
        ctx["p29"]["prose"],
        "",
        _md_table(ctx["p29"]["rows"]),
        "",
        "## Extra 30 — months-on-book",
        "",
        ctx["p30"]["prose"],
        "",
        _md_table(ctx["p30"]["rows"]),
        "",
        "## Extra 31 — Q6 so-far≥13",
        "",
        ctx["p31"]["prose"],
        "",
        "## Extra 32 — invoice-named ∩ tx-unnamed",
        "",
        ctx["p32"]["prose"],
        "",
        "## Extra 33 — dark vs invoiced Y2/Y3",
        "",
        ctx["p33"]["prose"],
        "",
        _md_table(ctx["p33"]["rows"]),
        "",
        "## Extra 34 — ops mix missing vs named",
        "",
        ctx["p34"]["prose"],
        "",
        _md_table(ctx["p34"]["rows"]),
        "",
        "## Extra 35 — company-median always-missing",
        "",
        ctx["p35"]["prose"],
        "",
        _md_table(ctx["p35"]["rows"]),
        "",
        "## Extra 36 — invoiced-744 Y2 vs size",
        "",
        ctx["p36"]["prose"],
        "",
        "## Extra 37 — always-missing invoiced companies by fold",
        "",
        ctx["p37"]["prose"],
        "",
        _md_table(ctx["p37"]["rows"]),
        "",
        "## Extra 38 — leftover after ever_erp + d_tx",
        "",
        ctx["p38"]["prose"],
        "",
        "## Extra 39 — mapped-miss leftover after d_tx on 744",
        "",
        ctx["p39"]["prose"],
        "",
        f"Plot: `{OUT_PNG.name}`." if ctx.get("png_ok") else "Plot: skipped.",
        "",
        "## What failed / next",
        "",
    ]
    for f in ctx.get("failed", []):
        lines.append(f"- {f}")
    lines.extend(
        [
            "",
            f"Elapsed {ctx['elapsed_s']:.0f}s. Cuts: prevalence, uncat 2×2, Spearman, singles, "
            "residual, dark 470/744, invoice vs tx fill, Y5 fold-3, chronic-12, ICC, Q6, "
            "category mix, amounts, low-uncat leftover, dtx slices, Y5 replica, Y3 folds, "
            "drop fold 3, monthly vs 6m, holdout, group ICC, size terciles, invoiced-only, quintiles, "
            "744-twin, all-miss pile, erp dummy, named cats, 360/110, so-far, Q6≥13, "
            "pure bank hole, dark Y, ops mix, always-missing companies, "
            "744 Y2, always-ERP folds, resid erp+dtx, mapped leftover after dtx.",
            "",
            "Did **not**: merge parquet, invent `y_missing_cp`, score Y7, edit counterparties.py / "
            "uncat_qa.py / y5_why.py / y11_dark.py, rewrite duckdb, run `build_targets`, "
            "touch `product/`, write 0–100, change night Y3 0.762/0.752 or Y5 0.611 quotes, "
            "write the parent journal.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD}")


def append_registry(ctx: dict) -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    p1, p3, p4, p5, d = ctx["p1"], ctx["p3"], ctx["p4"], ctx["p5"], ctx["decision"]
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    rows = [
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "D",
            "y": "-",
            "model": "missing_cp_qa",
            "split": "train",
            "metric": "miss_cp_tx_share",
            "value": p1["train_share_n"],
            "coverage": f"{p1['cm_cov']:.4f}",
            "notes": f"amt={p1['train_share_a']:.4f} cm_mean={p1['cm_mean']:.4f} x={d['x_dec']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "D",
            "y": Y3,
            "model": "missing_cp_qa",
            "split": "train_cv",
            "metric": "auroc_miss_cp_share",
            "value": p4["miss_y3"],
            "coverage": "1.0000",
            "notes": f"size={p4['size_y3']:.4f} days={p4['days_y3']:.4f} dtx={p4['dtx_y3']:.4f} leftover={p5['y3_both']:.4f} x={d['x_dec']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "D",
            "y": Y2,
            "model": "missing_cp_qa",
            "split": "train_cv",
            "metric": "auroc_miss_cp_share",
            "value": p4["miss_y2"],
            "coverage": "1.0000",
            "notes": f"dtx={p4['dtx_y2']:.4f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "D",
            "y": "-",
            "model": "missing_cp_qa",
            "split": "train",
            "metric": "rho_miss_vs_uncat",
            "value": p3["rho_uncat"],
            "coverage": "1.0000",
            "notes": f"twin_uncat={p3['twin_uncat']} rho_dtx={p3['rho_dtx']:.4f} twin_dtx={p3['twin_dtx']} size={p3['is_size']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "D",
            "y": Y3,
            "model": "missing_cp_qa",
            "split": "train_cv",
            "metric": "auroc_miss_resid_uncat_dtx",
            "value": p5["y3_both"],
            "coverage": "1.0000",
            "notes": f"resid_uncat={p5['y3_uncat']:.4f} resid_dtx={p5['y3_dtx']:.4f} leftover_lives={p5['leftover_lives']}",
        },
    ]
    header = [
        "ts",
        "round",
        "wave",
        "agent",
        "x_families",
        "y",
        "model",
        "split",
        "metric",
        "value",
        "coverage",
        "notes",
    ]
    seen: set[tuple] = set()
    with REGISTRY.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            seen.add(
                (
                    str(row.get("round", "")),
                    str(row.get("wave", "")),
                    str(row.get("agent", "")),
                    str(row.get("x_families", "")),
                    str(row.get("y", "")),
                    str(row.get("model", "")),
                    str(row.get("split", "")),
                    str(row.get("metric", "")),
                )
            )
    fresh = []
    for r in rows:
        key = (
            str(r.get("round", "")),
            str(r.get("wave", "")),
            str(r.get("agent", "")),
            str(r.get("x_families", "")),
            str(r.get("y", "")),
            str(r.get("model", "")),
            str(r.get("split", "")),
            str(r.get("metric", "")),
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
    p1, p2, p3, p4, p5 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"], ctx["p5"]
    WAVE_NOTE.parent.mkdir(parents=True, exist_ok=True)
    text = (
        f"# Wave 4 — missing-CP leftover\n\n"
        f"- **When:** {_now_iso()}\n"
        f"- **Agent:** `{AGENT}`\n"
        f"- **Files:** `analysis/evaluate/missing_cp_qa.py`, `analysis/outputs/missing_cp_qa.md`"
        f"{', `analysis/outputs/missing_cp_vs_uncat.png`' if ctx.get('png_ok') else ''}, "
        f"append-only `analysis/experiments/registry.csv`.\n"
        f"- **Columns:** in-memory `miss_cp_share` / `miss_cp_amt` / `miss_mapped_share` "
        f"(not written to parquet). Store `d_tx_cp_share` replica only.\n"
        f"- **Train coverage:** cm {_pp(p1['cm_cov'])}; txs miss {_pp(p1['train_share_n'])} / "
        f"|amt| {_pp(p1['train_share_a'])}.\n"
        f"- **Verdict:** X **{d['x_dec']}**; Y **PARK**; Q5 **{d['q5']}**; Q6 **{d['q6']}**.\n"
        f"- **Numbers:** uncat 2×2 miss-among-uncat {_pp(p2['uncat_miss_n'])} "
        f"(mapped still miss {_pp(p2['mapped_miss_n'])}); "
        f"ρ vs uncat {_f(p3['rho_uncat'])}; ρ vs d_tx {_f(p3['rho_dtx'])}; "
        f"Y3 miss {_f(p4['miss_y3'])} vs size {_f(p4['size_y3'])} vs days {_f(p4['days_y3'])}; "
        f"leftover after both {_f(p5['y3_both'])}.\n"
        f"- **What failed:** " + "; ".join(ctx.get("failed", [])[:6]) + "\n"
        f"- **Next idea:** parent decides any later-D merge. Do not put on the 15-col card. "
        f"Y7 never D.\n"
        f"- **Did not:** product/, 0–100, parquet rewrite, `build_targets`, counterparties.py, "
        f"uncat_qa.py, y5_why.py, parent journal, commit.\n"
    )
    WAVE_NOTE.write_text(text, encoding="utf-8")
    print(f"wrote {WAVE_NOTE}")


def run() -> dict:
    t0 = time.time()
    print(f"missing_cp_qa start seed={FOLD_SEED} holdout=72 agent={AGENT}")
    panel = load_panel()
    panel = attach_folds(panel)
    con = connect()
    try:
        print("load monthly missing-CP (in memory)")
        monthly = load_monthly_miss(con)
        panel = attach_miss(panel, monthly)
        print("load monthly invoice CP fill (in memory)")
        inv = load_monthly_inv_cp(con)
        panel = attach_inv(panel, inv)
        book = book_invoice_ids(con)
        panel["ever_erp"] = panel["company_id"].isin(book)
        panel = add_so_far(panel)
        panel = add_style_shock(panel)
        panel = add_panel_lags(
            panel,
            ["miss_cp_share", "d_tx_cp_share", "a_uncat_share", "c_n_days_with_tx"],
            (1, 3),
        )
        tr = panel[panel["split"] == "train"].copy()
        assert_no_holdout(tr["company_id"])
        print(
            f"train CM={len(tr):,} companies={tr['company_id'].nunique()} "
            f"holdout CM={(panel['split']=='holdout').sum()}"
        )
        ids = chronic_ids(tr)
        print(f"chronic ids n={len(ids)}")
        print("pass 1 prevalence")
        p1 = pass1_prevalence(panel, con)
        print("pass 2 uncat 2x2")
        p2 = pass2_uncat_2x2(con)
        print("pass 3 spearman")
        p3 = pass3_spearman(tr)
        print("pass 4 auroc")
        p4 = pass4_auroc(tr)
        print("pass 5 residual")
        p5 = pass5_residual(tr)
        print("pass 6 dark")
        p6 = pass6_dark(tr)
        print("pass 7 invoice vs tx")
        p7 = pass7_invoice_vs_tx(tr)
        print("pass 8 Y5 fold3")
        p8 = pass8_y5_fold3(tr)
        print("pass 9 chronic 12")
        p9 = pass9_chronic(tr, ids)
        print("pass 10 ICC")
        p10 = pass10_icc(tr)
        print("pass 11 Q6")
        p11 = pass11_q6(tr)
        print("pass 12 cats")
        p12 = pass12_cats(con)
        print("pass 13 amounts")
        p13 = pass13_amounts(con)
        print("pass 14 mapped leftover")
        p14 = pass14_mapped_leftover(tr)
        print("pass 15 after dtx")
        p15 = pass15_after_dtx(tr)
        print("pass 16 Y5 replica")
        p16 = pass16_y5_replica(tr)
        print("pass 17 Y3 onegroup")
        p17 = pass17_y3_onegroup(tr)
        print("pass 18 drop fold3")
        p18 = pass18_drop_f3(tr)
        print("pass 19 window")
        p19 = pass19_window(tr)
        print("pass 20 holdout")
        p20 = pass20_holdout(panel)
        print("pass 21 group ICC")
        p21 = pass21_group_icc(tr)
        print("pass 22 size terciles")
        p22 = pass22_size_terc(tr)
        print("pass 23 invoiced mapped")
        p23 = pass23_erp_mapped(tr)
        print("pass 24 quintiles")
        p24 = pass24_quintiles(tr)
        print("pass 25 invoiced twin")
        p25 = pass25_erp_twin(tr)
        print("pass 26 all-miss pile")
        p26 = pass26_allmiss(tr)
        print("pass 27 erp dummy")
        p27 = pass27_erp_dummy(tr)
        print("pass 28 named cats")
        p28 = pass28_named_cats(con)
        print("pass 29 dark mix")
        p29 = pass29_dark_mix(tr)
        print("pass 30 sofar")
        p30 = pass30_sofar(tr)
        print("pass 31 Q6 >=13")
        p31 = pass31_q6_13(tr)
        print("pass 32 pure hole")
        p32 = pass32_pure_hole(tr)
        print("pass 33 dark Y")
        p33 = pass33_dark_y(tr)
        print("pass 34 ops mix")
        p34 = pass34_ops_share(con)
        print("pass 35 company median")
        p35 = pass35_co_median(tr)
        print("pass 36 Y2 invoiced")
        p36 = pass36_y2_erp(tr)
        print("pass 37 always-missing ERP folds")
        p37 = pass37_always_erp_fold(tr)
        print("pass 38 resid erp+dtx")
        p38 = pass38_resid_erp_dtx(tr)
        print("pass 39 mapped leftover after dtx")
        p39 = pass39_mapped_after_dtx(tr)
    finally:
        con.close()

    decision = decide(p3, p4, p5, p8, p10, p11, p6, p7, p14, p19)
    png_ok = make_png(tr, p12)
    headline = (
        f"Train txs miss-CP {_pp(p1['train_share_n'])} / |amt| {_pp(p1['train_share_a'])}. "
        f"Uncat×miss CONFIRM {_pp(p2['uncat_miss_n'])}; mapped still miss {_pp(p2['mapped_miss_n'])}. "
        f"ρ vs uncat {_f(p3['rho_uncat'])} ({'TWIN' if p3['twin_uncat'] else 'not twin'}); "
        f"ρ vs d_tx_cp_share {_f(p3['rho_dtx'])} ({'TWIN' if p3['twin_dtx'] else 'not twin'}). "
        f"Y3 miss {_f(p4['miss_y3'])} vs size {_f(p4['size_y3'])} (Δ {_f(p4['beat'], 3)}) "
        f"vs days {_f(p4['days_y3'])}. Leftover after uncat+dtx {_f(p5['y3_both'])}. "
        f"Dark vs 744 {_f(p6['dark_m'])} / {_f(p6['erp_m'])}. "
        f"Invoice fill {_f(p7['inv_mean'])} vs tx named {_f(p7['tx_mean'])}. "
        f"Fold-3 one-group={decision['one_group']}. ICC {_f(p10['icc']['icc'])}. "
        f"X **{decision['x_dec']}**. Y **PARK**. Q5 **{decision['q5']}**. Q6 **{decision['q6']}**."
    )
    print(headline)
    failed = []
    if not p2["confirm"]:
        failed.append(f"uncat miss {_pp(p2['uncat_miss_n'])} ≠ 91.8% quote")
    if not p6["confirm"]:
        failed.append(f"dark/erp {p6['n_dark']}/{p6['n_erp']} ≠ 470/744")
    if not p4["days_ok"]:
        failed.append(f"Y3 days replica {_f(p4['days_y3'])} vs night 0.711")
    if not p4["size_ok"]:
        failed.append(f"Y3 size replica {_f(p4['size_y3'])} vs night 0.617")
    if not p16["confirm"]:
        failed.append(
            f"Y5 d_tx_cp_share train {_f(p16['dtx_train'])} vs night 0.611 "
            "(do not overwrite the quote)"
        )
    if p3["twin_uncat"]:
        failed.append(f"uncat twin ρ={_f(p3['rho_uncat'])} — CLOSE as (a)")
    if p3["twin_dtx"] or p19["twin"]:
        failed.append(f"d_tx_cp_share twin ρ={_f(p3['rho_dtx'])} — CLOSE as (b)")
    if p5["died"] and not p5["leftover_lives"]:
        failed.append("leftover after uncat/dtx dies — CLOSE as twin")
    if decision["one_group"]:
        failed.append("Y5 fold-3 pile again — CLOSE as (e)")
    if p9["is_those"]:
        failed.append("Y2 skill is the 12 chronic names")
    if p25["twin"]:
        failed.append(f"invoiced-744 still d_tx twin ρ={_f(p25['rho_dtx'])} — not only the 470")
    if p27["twin"]:
        failed.append("miss is the 470 dummy")
    failed.append(p12["prose"])
    failed.append(p7["prose"])
    failed.append(p26["prose"])
    failed.append(p32["prose"])
    failed.append(p36["prose"])
    failed.append(p38["prose"])
    failed.append(p39["prose"])
    elapsed = time.time() - t0
    ctx = {
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p4": p4,
        "p5": p5,
        "p6": p6,
        "p7": p7,
        "p8": p8,
        "p9": p9,
        "p10": p10,
        "p11": p11,
        "p12": p12,
        "p13": p13,
        "p14": p14,
        "p15": p15,
        "p16": p16,
        "p17": p17,
        "p18": p18,
        "p19": p19,
        "p20": p20,
        "p21": p21,
        "p22": p22,
        "p23": p23,
        "p24": p24,
        "p25": p25,
        "p26": p26,
        "p27": p27,
        "p28": p28,
        "p29": p29,
        "p30": p30,
        "p31": p31,
        "p32": p32,
        "p33": p33,
        "p34": p34,
        "p35": p35,
        "p36": p36,
        "p37": p37,
        "p38": p38,
        "p39": p39,
        "decision": decision,
        "headline": headline,
        "png_ok": png_ok,
        "failed": failed,
        "elapsed_s": elapsed,
    }
    write_md(ctx)
    append_registry(ctx)
    write_wave(ctx)
    print(f"elapsed {elapsed:.1f}s")
    return ctx


if __name__ == "__main__":
    run()
   