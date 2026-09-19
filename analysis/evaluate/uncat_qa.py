"""Q5 uncat readability — count share vs amount-mass, leftover cats.

NORTH_STAR Q5: why did the trail change? `a_uncat_share` is the keep-list
count share of txs whose category is `uncategorized` or not in CAT_MAP.
Feature report: cov 95.8%, size ρ −0.023, acf1 0.27, ICC 0.99 BETWEEN.

Family M CLOSED — do not merge mix. Amount-uncat is computed in memory
only (do not rewrite `a_uncat_share`; uncat *count* share was parked as
that rewrite). Do not redo catmix / Y9 why / tax. Do not invent a Y.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.uncat_qa

Owned: analysis/evaluate/uncat_qa.py, analysis/outputs/uncat_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_uncat.md (end).

Iteration log (same module, not one-shot):
1. Amount vs count + leftover cats + quintiles + singles + style + dark + Q6
2. Mix missingness + holdout
3. Company-mean vs demean; Y2 0.602 is style
4. Activity residual; zero pile; hide-euro; cash_settlements alias
5. Dark × size; company-median vs Y2
6. Always-messy cards; calendar; uncat sign; Y3 after days
7. Months-on-book; messy Y3 label coverage
8. Within-company early vs late; all-uncat months; group ICC
9. 360/110 dark mix; uncat vs mapped ticket size
10. Pending vs uncat; ticket fat-tail vs typical
11. Row flags / value_date / product / card / checking-only
12. Inflow vs outflow uncat; residual after a_n_tx
13. Size terciles / mature books / dark Y2 / sticky CP / desc / weekday
14. Drop all-uncat; invoiced-only style vs shock
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
OUT_MD = ANALYSIS / "outputs" / "uncat_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "uncat_quintiles.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
AGENT = "ec17da1b"
WAVE = "4"
ROUND = "R4"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
Y7 = "y7_top1_lost"
Y9 = "y9_fee_r_ownp80"
N_FOLDS = 5
DAYS_BENCH = 0.711
KEEP_DELTA = 0.02
SIZE_PARK = 0.60
ACF1_QUOTE = 0.27
ICC_STYLE = 0.85
ACF_STYLE = 0.40
MIN_POS = 50
MIN_ACF_PAIRS = 4
PANEL_START = pd.Timestamp("2024-09-01")
PANEL_END = pd.Timestamp("2026-08-01")

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "a_uncat_share",
    "a_n_tx",
    "a_in3",
    "a_op_in",
    "c_n_days_with_tx",
    "c_n_tx",
    "e_ar_issued",
    "e_ap_issued",
    "d_n_cust",
    "d_n_supp",
    "j_pay_match",
    "j_coll_match",
    "a_pending_share",
    "g_has_card",
    "g_has_checking",
    "g_has_tpv",
)

Y_KEEP = (Y2, Y3, Y7, Y9)


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
    }


def _cat_sql_list() -> str:
    return ", ".join("'" + k.replace("'", "''") + "'" for k in CAT_MAP)


def _store_cols_present(raw: pd.DataFrame) -> list[str]:
    have = [c for c in STORE_COLS if c in raw.columns]
    need = ["company_id", "period", "group_id", "a_uncat_share", "a_in3", "c_n_days_with_tx"]
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
    panel["log_op_in"] = np.log1p(
        pd.to_numeric(panel["a_op_in"], errors="coerce").abs()
    )
    return panel


def attach_folds(panel: pd.DataFrame) -> pd.DataFrame:
    train = panel[panel["split"] == "train"][["company_id", "group_id"]].drop_duplicates()
    assert_no_holdout(train["company_id"])
    folds = group_folds(train, n=N_FOLDS, seed=FOLD_SEED)
    return panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")


def add_style_shock(df: pd.DataFrame) -> pd.DataFrame:
    """Company mean = bookkeeping style. Demean = month shock. Not a Y."""
    out = df.sort_values(["company_id", "period"]).reset_index(drop=True)
    extra = {}
    for col, pref in (("a_uncat_share", "count"), ("amt_uncat", "amt")):
        x = pd.to_numeric(out[col], errors="coerce")
        mu = x.groupby(out["company_id"], sort=False).transform("mean")
        extra[f"{pref}_co_mean"] = mu
        extra[f"{pref}_demean"] = x - mu
    return pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)


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


def load_monthly_uncat(con) -> pd.DataFrame:
    """Amount-uncat + count-uncat in memory. Never written to parquet."""
    cats = _cat_sql_list()
    df = con.execute(
        f"""
        SELECT
          CAST(t.company_id AS VARCHAR) AS company_id,
          CAST(date_trunc('month', t."date") AS DATE) AS month,
          COUNT(*) AS n_tx,
          SUM(ABS(t.amount)) AS abs_all,
          SUM(CASE WHEN t.category = 'uncategorized'
                     OR t.category IS NULL
                     OR t.category NOT IN ({cats})
                   THEN 1 ELSE 0 END) AS n_uncat,
          SUM(CASE WHEN t.category = 'uncategorized'
                     OR t.category IS NULL
                     OR t.category NOT IN ({cats})
                   THEN ABS(t.amount) ELSE 0 END) AS abs_uncat,
          SUM(CASE WHEN t.category = 'uncategorized' THEN 1 ELSE 0 END) AS n_token_uncat,
          SUM(CASE WHEN t.category = 'uncategorized' THEN ABS(t.amount) ELSE 0 END) AS abs_token_uncat,
          SUM(CASE WHEN t.category IS NULL
                     OR (t.category <> 'uncategorized' AND t.category NOT IN ({cats}))
                   THEN 1 ELSE 0 END) AS n_leftover,
          SUM(CASE WHEN t.category IS NULL
                     OR (t.category <> 'uncategorized' AND t.category NOT IN ({cats}))
                   THEN ABS(t.amount) ELSE 0 END) AS abs_leftover,
          SUM(CASE WHEN t.category IN ('collection', 'payment') THEN ABS(t.amount) ELSE 0 END) AS abs_core,
          SUM(CASE WHEN t.category IN ({cats}) THEN ABS(t.amount) ELSE 0 END) AS abs_mapped
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
    df["count_uncat"] = np.where(n_tx > 0, df["n_uncat"] / n_tx, np.nan)
    df["amt_uncat"] = np.where(abs_all > 0, df["abs_uncat"] / abs_all, np.nan)
    df["count_token"] = np.where(n_tx > 0, df["n_token_uncat"] / n_tx, np.nan)
    df["amt_token"] = np.where(abs_all > 0, df["abs_token_uncat"] / abs_all, np.nan)
    df["count_leftover"] = np.where(n_tx > 0, df["n_leftover"] / n_tx, np.nan)
    df["m_coll_vs_pay_ok"] = pd.to_numeric(df["abs_core"], errors="coerce") > 0
    df["m_mix_ok"] = abs_all > 0
    return df


def attach_uncat(panel: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "company_id",
        "period",
        "n_tx",
        "abs_all",
        "n_uncat",
        "abs_uncat",
        "n_token_uncat",
        "abs_token_uncat",
        "n_leftover",
        "abs_leftover",
        "abs_core",
        "abs_mapped",
        "count_uncat",
        "amt_uncat",
        "count_token",
        "amt_token",
        "count_leftover",
        "m_coll_vs_pay_ok",
        "m_mix_ok",
    ]
    out = panel.merge(monthly[keep], on=["company_id", "period"], how="left")
    return out


# ---------------------------------------------------------------------------
# Pass 1 — amount-mass vs count-share
# ---------------------------------------------------------------------------
def pass1_amount_vs_count(tr: pd.DataFrame) -> dict:
    store = pd.to_numeric(tr["a_uncat_share"], errors="coerce")
    cnt = pd.to_numeric(tr["count_uncat"], errors="coerce")
    amt = pd.to_numeric(tr["amt_uncat"], errors="coerce")
    n_cm = int(len(tr))
    n_store = int(store.notna().sum())
    n_cnt = int(cnt.notna().sum())
    n_amt = int(amt.notna().sum())
    agree = store.notna() & cnt.notna()
    max_abs = float((store[agree] - cnt[agree]).abs().max()) if agree.any() else float("nan")
    mean_abs = float((store[agree] - cnt[agree]).abs().mean()) if agree.any() else float("nan")
    rho_ca = spearman(cnt, amt)
    pear_ca = pearson(cnt, amt)
    mean_c = float(cnt.mean())
    mean_a = float(amt.mean())
    p50_c = float(cnt.median())
    p50_a = float(amt.median())
    p90_c = float(cnt.quantile(0.90))
    p90_a = float(amt.quantile(0.90))
    # months where amount-uncat >> count-uncat (large uncategorized euros hidden)
    both = cnt.notna() & amt.notna()
    hid = both & ((amt - cnt) >= 0.20)
    n_hid = int(hid.sum())
    # opposite: many small uncat tickets, euros mapped
    small = both & ((cnt - amt) >= 0.20)
    n_small = int(small.sum())
    # euro mass
    abs_all = pd.to_numeric(tr["abs_all"], errors="coerce")
    abs_u = pd.to_numeric(tr["abs_uncat"], errors="coerce")
    euro_share = float(abs_u.sum() / abs_all.sum()) if abs_all.sum() else float("nan")
    n_tx = pd.to_numeric(tr["n_tx"], errors="coerce")
    n_u = pd.to_numeric(tr["n_uncat"], errors="coerce")
    tx_share = float(n_u.sum() / n_tx.sum()) if n_tx.sum() else float("nan")
    leftover_n = float(pd.to_numeric(tr["n_leftover"], errors="coerce").sum())
    leftover_amt = float(pd.to_numeric(tr["abs_leftover"], errors="coerce").sum())
    token_n = float(pd.to_numeric(tr["n_token_uncat"], errors="coerce").sum())
    rho_size_c = spearman(cnt, tr["log_in3"])
    rho_size_a = spearman(amt, tr["log_in3"])
    rho_opin_c = spearman(cnt, tr["log_op_in"])
    rho_opin_a = spearman(amt, tr["log_op_in"])
    cov_c = _pct(n_cnt, n_cm)
    cov_a = _pct(n_amt, n_cm)
    confirm_cov = bool(np.isfinite(cov_c) and abs(cov_c - 0.958) < 0.015)
    # feature-report size ρ is vs log1p(|a_op_in|), not a_in3
    confirm_rho = bool(np.isfinite(rho_opin_c) and abs(rho_opin_c - (-0.023)) < 0.03)
    hiding = bool(np.isfinite(mean_a) and np.isfinite(mean_c) and (mean_a - mean_c) >= 0.05)
    prose = (
        f"Train CM {n_cm:,}. Store `a_uncat_share` cov {_pp(cov_c)} "
        f"({'CONFIRM 95.8%' if confirm_cov else 'off 95.8% quote'}); "
        f"recomputed count vs store max|Δ|={_f(max_abs, 6)} mean|Δ|={_f(mean_abs, 6)}. "
        f"Count mean {_f(mean_c)} vs amount-uncat mean {_f(mean_a)} "
        f"(Spearman {_f(rho_ca)}; catmix quote ρ≈0.889). "
        f"Pooled ticket uncat {_pp(tx_share)} vs euro uncat {_pp(euro_share)}. "
        f"Months amount−count ≥0.20: {n_hid:,}; count−amount ≥0.20: {n_small:,}. "
        f"Leftover (not the `uncategorized` token) txs {leftover_n:,.0f} / € {leftover_amt:,.0f}. "
        f"Size ρ vs a_in3 count {_f(rho_size_c)} amount {_f(rho_size_a)}; "
        f"vs log1p(|a_op_in|) count {_f(rho_opin_c)} "
        f"({'CONFIRM −0.023' if confirm_rho else 'off −0.023'}) amount {_f(rho_opin_a)}. "
        f"{'Amount-uncat is heavier than count — count hides euro opacity.' if hiding else 'Amount-uncat is not systematically heavier than count.'}"
    )
    print(prose)
    return {
        "n_cm": n_cm,
        "n_co": int(tr["company_id"].nunique()),
        "n_store": n_store,
        "n_cnt": n_cnt,
        "n_amt": n_amt,
        "cov_c": cov_c,
        "cov_a": cov_a,
        "confirm_cov": confirm_cov,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rho_ca": rho_ca,
        "pear_ca": pear_ca,
        "mean_c": mean_c,
        "mean_a": mean_a,
        "p50_c": p50_c,
        "p50_a": p50_a,
        "p90_c": p90_c,
        "p90_a": p90_a,
        "n_hid": n_hid,
        "n_small": n_small,
        "euro_share": euro_share,
        "tx_share": tx_share,
        "leftover_n": leftover_n,
        "leftover_amt": leftover_amt,
        "token_n": token_n,
        "rho_size_c": rho_size_c,
        "rho_size_a": rho_size_a,
        "rho_opin_c": rho_opin_c,
        "rho_opin_a": rho_opin_a,
        "confirm_rho": confirm_rho,
        "hiding": hiding,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 2 — leftover categories outside CAT_MAP
# ---------------------------------------------------------------------------
def pass2_leftover(con, tr: pd.DataFrame) -> dict:
    hold = load_holdout()
    cats = _cat_sql_list()
    raw = con.execute(
        f"""
        SELECT
          CAST(COALESCE(t.category, '(null)') AS VARCHAR) AS category,
          COUNT(*) AS n,
          SUM(ABS(t.amount)) AS abs_amt,
          SUM(t.amount) AS net_amt,
          COUNT(DISTINCT t.company_id) AS n_co
        FROM transactions t
        WHERE t."date" IS NOT NULL
          AND t."date" >= DATE '2024-09-01'
          AND t."date" < DATE '2026-09-01'
          AND (t.category IS NULL
               OR t.category NOT IN ({cats}))
        GROUP BY 1
        """
    ).df()
    # restrict to train companies via a second query on company_id
    raw_co = con.execute(
        f"""
        SELECT
          CAST(t.company_id AS VARCHAR) AS company_id,
          CAST(COALESCE(t.category, '(null)') AS VARCHAR) AS category,
          COUNT(*) AS n,
          SUM(ABS(t.amount)) AS abs_amt
        FROM transactions t
        WHERE t."date" IS NOT NULL
          AND t."date" >= DATE '2024-09-01'
          AND t."date" < DATE '2026-09-01'
          AND (t.category IS NULL
               OR t.category NOT IN ({cats}))
        GROUP BY 1, 2
        """
    ).df()
    raw_co["company_id"] = raw_co["company_id"].astype(str)
    raw_co = raw_co.loc[~raw_co["company_id"].isin(hold)].copy()
    assert_no_holdout(raw_co["company_id"])
    agg = (
        raw_co.groupby("category", as_index=False)
        .agg(n=("n", "sum"), abs_amt=("abs_amt", "sum"), n_co=("company_id", "nunique"))
    )
    # all distinct categories (mapped + leftover) for the universe check
    universe = con.execute(
        """
        SELECT
          CAST(COALESCE(category, '(null)') AS VARCHAR) AS category,
          COUNT(*) AS n
        FROM transactions
        WHERE "date" IS NOT NULL
          AND "date" >= DATE '2024-09-01'
          AND "date" < DATE '2026-09-01'
        GROUP BY 1
        """
    ).df()
    mapped = set(CAT_MAP)
    universe["in_map"] = universe["category"].isin(mapped)
    n_tokens = int(len(universe))
    n_mapped_seen = int(universe["in_map"].sum())
    n_left_tokens = int((~universe["in_map"]).sum())
    top_n = agg.sort_values("n", ascending=False).head(15)
    top_a = agg.sort_values("abs_amt", ascending=False).head(15)
    only_uncat = bool(n_left_tokens == 1 and set(agg["category"]) <= {"uncategorized"})
    # mapped tokens never seen
    seen_mapped = set(universe.loc[universe["in_map"], "category"])
    never_seen = sorted(mapped - seen_mapped)
    # frequent leftover: ≥1% of leftover txs or ≥1% leftover euros
    tot_n = float(agg["n"].sum()) if len(agg) else 0.0
    tot_a = float(agg["abs_amt"].sum()) if len(agg) else 0.0
    add_note = []
    for _, r in agg.iterrows():
        cat = str(r["category"])
        if cat == "uncategorized":
            continue
        share_n = _pct(r["n"], tot_n)
        share_a = _pct(r["abs_amt"], tot_a)
        if share_n >= 0.05 or share_a >= 0.05:
            add_note.append(
                {
                    "category": cat,
                    "n": int(r["n"]),
                    "abs_amt": float(r["abs_amt"]),
                    "share_n": share_n,
                    "share_a": share_a,
                    "guess": "unknown — do not edit CAT_MAP tonight",
                }
            )
    prose = (
        f"Universe tokens {n_tokens} (CAT_MAP size {len(CAT_MAP)}; seen mapped {n_mapped_seen}; "
        f"leftover tokens {n_left_tokens}). "
        f"{'CONFIRM catmix: only leftover token is `uncategorized`.' if only_uncat else 'Leftover tokens besides `uncategorized` exist — see table.'} "
        f"Train leftover txs {tot_n:,.0f} / € {tot_a:,.0f}. "
        f"Never-seen CAT_MAP keys: {never_seen or 'none'}. "
        f"Frequent leftover to consider adding: {len(add_note)} "
        f"(design note only — CAT_MAP not edited)."
    )
    print(prose)
    return {
        "n_tokens": n_tokens,
        "n_mapped_seen": n_mapped_seen,
        "n_left_tokens": n_left_tokens,
        "only_uncat": only_uncat,
        "never_seen": never_seen,
        "top_n": top_n.to_dict("records"),
        "top_a": top_a.to_dict("records"),
        "add_note": add_note,
        "tot_n": tot_n,
        "tot_a": tot_a,
        "universe": universe.to_dict("records"),
        "n_raw_all": int(raw["n"].sum()) if len(raw) else 0,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 3 — quintiles vs Y rates
# ---------------------------------------------------------------------------
def _quintile_table(tr: pd.DataFrame, col: str, ycols: tuple[str, ...]) -> list[dict]:
    x = pd.to_numeric(tr[col], errors="coerce")
    defined = x.notna()
    sl = tr.loc[defined].copy()
    sl["_x"] = x[defined]
    try:
        sl["_q"] = pd.qcut(sl["_x"], 5, labels=False, duplicates="drop")
    except ValueError:
        sl["_q"] = 0
    rows = []
    for q, g in sl.groupby("_q", sort=True):
        rec = {
            "q": int(q) + 1,
            "n_cm": int(len(g)),
            "n_co": int(g["company_id"].nunique()),
            "p50": float(g["_x"].median()),
            "mean": float(g["_x"].mean()),
        }
        for y in ycols:
            yy = pd.to_numeric(g[y], errors="coerce")
            rec[f"{y}_n"] = int(yy.notna().sum())
            rec[f"{y}_pos"] = int((yy == 1).sum())
            rec[f"{y}_rate"] = float(yy.mean()) if yy.notna().any() else float("nan")
        rows.append(rec)
    return rows


def _monotone(rates: list[float]) -> str:
    finite = [r for r in rates if np.isfinite(r)]
    if len(finite) < 3:
        return "n/a"
    diffs = [finite[i + 1] - finite[i] for i in range(len(finite) - 1)]
    up = all(d >= -0.005 for d in diffs) and any(d > 0.01 for d in diffs)
    down = all(d <= 0.005 for d in diffs) and any(d < -0.01 for d in diffs)
    if up:
        return "rising"
    if down:
        return "falling"
    # U / inverted-U
    mid = finite[len(finite) // 2]
    if mid >= max(finite[0], finite[-1]) + 0.02:
        return "inverted-U"
    if mid <= min(finite[0], finite[-1]) - 0.02:
        return "U"
    return "noise"


def pass3_quintiles(tr: pd.DataFrame) -> dict:
    ycols = (Y2, Y3, Y7, Y9)
    count_rows = _quintile_table(tr, "a_uncat_share", ycols)
    amt_rows = _quintile_table(tr, "amt_uncat", ycols)
    shapes = {}
    for name, rows in (("count", count_rows), ("amt", amt_rows)):
        for y in ycols:
            shapes[f"{name}_{y}"] = _monotone([r[f"{y}_rate"] for r in rows])
    prose = (
        f"Count-share quintiles vs Y2 {shapes['count_' + Y2]}, Y3 {shapes['count_' + Y3]}, "
        f"Y7 {shapes['count_' + Y7]}, Y9 {shapes['count_' + Y9]}. "
        f"Amount-uncat quintiles vs Y2 {shapes['amt_' + Y2]}, Y3 {shapes['amt_' + Y3]}, "
        f"Y7 {shapes['amt_' + Y7]}, Y9 {shapes['amt_' + Y9]}."
    )
    print(prose)
    return {
        "count_rows": count_rows,
        "amt_rows": amt_rows,
        "shapes": shapes,
        "prose": prose,
    }


def make_png(p3: dict) -> bool:
    if not HAS_MPL:
        print("png skipped: no matplotlib")
        return False
    cr = p3["count_rows"]
    ar = p3["amt_rows"]
    if not cr or not ar:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, rows, title in (
        (axes[0], cr, "a_uncat_share (count)"),
        (axes[1], ar, "amount-uncat (in memory)"),
    ):
        qs = [r["q"] for r in rows]
        ax.plot(qs, [r[f"{Y2}_rate"] for r in rows], "o-", label="Y2 stress", color="#c0392b")
        ax.plot(qs, [r[f"{Y3}_rate"] for r in rows], "s-", label="Y3 recover", color="#1f6f4a")
        ax.plot(qs, [r[f"{Y7}_rate"] for r in rows], "^-", label="Y7 top1 lost", color="#2c3e50")
        ax.plot(qs, [r[f"{Y9}_rate"] for r in rows], "d-", label="Y9 fee own-p80", color="#8e44ad")
        ax.set_xticks(qs)
        ax.set_xlabel("train quintile (cuts on this column)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("base rate")
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle("Uncat quintiles vs accepted Ys (train; holdout never in cuts)", fontsize=11)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return True


# ---------------------------------------------------------------------------
# Pass 4 — single-feature group-fold AUROC
# ---------------------------------------------------------------------------
def pass4_auroc(tr: pd.DataFrame) -> dict:
    feats = {
        "a_uncat_share": tr["a_uncat_share"],
        "amt_uncat": tr["amt_uncat"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "log1p_a_in3": tr["log_in3"],
        "a_n_tx": tr.get("a_n_tx", tr.get("n_tx")),
    }
    rows = []
    store = {}
    for y in (Y2, Y3, Y7, Y9):
        lab = tr[y].notna()
        for name, col in feats.items():
            if col is None:
                continue
            res = signed_oof_auroc(tr[y], col, tr["fold"], lab)
            store[(y, name)] = res
            rows.append(
                {
                    "y": y,
                    "feature": name,
                    "n": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                    "sd": _f(res["sd"]),
                    "sign": res["train_sign"] if not res["low_power"] else "—",
                    "train": _f(res["train_auc"]) if not res["low_power"] else "—",
                }
            )
            print(
                f"AUROC {y} {name}: "
                f"{'LOW_POWER' if res['low_power'] else _f(res['cv'])} "
                f"n_pos={res['n_pos']} sign={res['train_sign']}"
            )

    def _cv(y, feat) -> float:
        r = store.get((y, feat))
        if r is None or r["low_power"]:
            return float("nan")
        return r["cv"]

    size_y3 = _cv(Y3, "log1p_a_in3")
    days_y3 = _cv(Y3, "c_n_days_with_tx")
    cnt_y3 = _cv(Y3, "a_uncat_share")
    amt_y3 = _cv(Y3, "amt_uncat")
    size_park = bool(np.isfinite(size_y3) and size_y3 >= SIZE_PARK)
    best_uncat = amt_y3 if (np.isfinite(amt_y3) and (not np.isfinite(cnt_y3) or amt_y3 >= cnt_y3)) else cnt_y3
    best_name = "amt_uncat" if best_uncat == amt_y3 and np.isfinite(amt_y3) else "a_uncat_share"
    beat_size = (
        best_uncat - size_y3 if np.isfinite(best_uncat) and np.isfinite(size_y3) else float("nan")
    )
    beat_days = (
        best_uncat - days_y3 if np.isfinite(best_uncat) and np.isfinite(days_y3) else float("nan")
    )
    days_ok = bool(np.isfinite(days_y3) and abs(days_y3 - DAYS_BENCH) <= 0.02)
    rates = {}
    for y in (Y2, Y3, Y7, Y9):
        yy = pd.to_numeric(tr[y], errors="coerce")
        rates[y] = {
            "n": int(yy.notna().sum()),
            "pos": int((yy == 1).sum()),
            "rate": float(yy.mean()) if yy.notna().any() else float("nan"),
        }
    prose = (
        f"Y3 singles: count {_f(cnt_y3)} amount-uncat {_f(amt_y3)} vs size {_f(size_y3)} "
        f"(best Δsize {_f(beat_size, 3)}) vs days {_f(days_y3)} "
        f"(night 0.711, {'CONFIRM' if days_ok else 'off'}). "
        f"Best uncat is {best_name}. "
        f"Size≥0.60 on Y3: {'YES — PARK uncat as X' if size_park else 'no'}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "size_y3": size_y3,
        "days_y3": days_y3,
        "cnt_y3": cnt_y3,
        "amt_y3": amt_y3,
        "cnt_y2": _cv(Y2, "a_uncat_share"),
        "amt_y2": _cv(Y2, "amt_uncat"),
        "size_y2": _cv(Y2, "log1p_a_in3"),
        "cnt_y7": _cv(Y7, "a_uncat_share"),
        "amt_y7": _cv(Y7, "amt_uncat"),
        "size_y7": _cv(Y7, "log1p_a_in3"),
        "cnt_y9": _cv(Y9, "a_uncat_share"),
        "amt_y9": _cv(Y9, "amt_uncat"),
        "size_y9": _cv(Y9, "log1p_a_in3"),
        "best_uncat": best_uncat,
        "best_name": best_name,
        "beat_size": beat_size,
        "beat_days": beat_days,
        "size_park": size_park,
        "days_ok": days_ok,
        "rates": rates,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 5 — persistence / style vs shock
# ---------------------------------------------------------------------------
def pass5_style(tr: pd.DataFrame) -> dict:
    trs = tr.sort_values(["company_id", "period"])
    acf = {}
    icc = {}
    for col, key in (("a_uncat_share", "count"), ("amt_uncat", "amt")):
        acf[key] = {
            1: median_acf(trs[col], trs["company_id"], 1),
            3: median_acf(trs[col], trs["company_id"], 3),
            6: median_acf(trs[col], trs["company_id"], 6),
        }
        icc[key] = icc_anova(trs[col], trs["company_id"])
    confirm_acf = bool(
        np.isfinite(acf["count"][1]) and abs(acf["count"][1] - ACF1_QUOTE) < 0.05
    )
    # company medians
    g = trs.groupby("company_id", sort=False)
    med_c = g["a_uncat_share"].median()
    med_a = g["amt_uncat"].median()
    sd_c = g["a_uncat_share"].std()
    sd_a = g["amt_uncat"].std()
    n_co = int(med_c.notna().sum())
    always_messy = int(((med_c >= 0.40) & (sd_c.fillna(0) <= 0.15)).sum())
    always_clean = int(((med_c <= 0.05) & (sd_c.fillna(0) <= 0.10)).sum())
    shock_co = int(((med_c <= 0.15) & (g["a_uncat_share"].max() >= 0.50)).sum())
    always_messy_a = int(((med_a >= 0.40) & (sd_a.fillna(0) <= 0.15)).sum())
    shock_co_a = int(((med_a <= 0.15) & (g["amt_uncat"].max() >= 0.50)).sum())

    def _style(icc_v, acf1) -> bool:
        return bool(
            np.isfinite(icc_v)
            and icc_v >= ICC_STYLE
            and np.isfinite(acf1)
            and acf1 < ACF_STYLE
        )

    style_c = _style(icc["count"]["icc"], acf["count"][1])
    style_a = _style(icc["amt"]["icc"], acf["amt"][1])
    prose = (
        f"Count acf1={_f(acf['count'][1])} "
        f"({'CONFIRM ~0.27' if confirm_acf else 'off 0.27 quote'}) "
        f"acf3={_f(acf['count'][3])} acf6={_f(acf['count'][6])}; "
        f"ICC={_f(icc['count']['icc'])} (feature-report 0.99). "
        f"Amount acf1={_f(acf['amt'][1])} ICC={_f(icc['amt']['icc'])}. "
        f"Company-median count: always-messy {always_messy}/{n_co}, "
        f"always-clean {always_clean}, shock (low median, max≥0.50) {shock_co}. "
        f"Amount always-messy {always_messy_a}, shock {shock_co_a}. "
        f"Count is {'STYLE dummy (high ICC + modest month acf)' if style_c else 'not a style dummy'}. "
        f"Amount is {'STYLE dummy' if style_a else 'not a style dummy (month shock or mixed)'}."
    )
    print(prose)
    return {
        "acf": acf,
        "icc": icc,
        "confirm_acf": confirm_acf,
        "n_co": n_co,
        "always_messy": always_messy,
        "always_clean": always_clean,
        "shock_co": shock_co,
        "always_messy_a": always_messy_a,
        "shock_co_a": shock_co_a,
        "med_c_p50": float(med_c.median()) if med_c.notna().any() else float("nan"),
        "med_a_p50": float(med_a.median()) if med_a.notna().any() else float("nan"),
        "style_c": style_c,
        "style_a": style_a,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 6 — dark 470 vs 744
# ---------------------------------------------------------------------------
def pass6_dark(tr: pd.DataFrame, con) -> dict:
    book = book_invoice_ids(con)
    hold = load_holdout()
    last = tr.sort_values("period").groupby("company_id", sort=False).last().reset_index()
    last["ever_erp"] = last["company_id"].isin(book)
    n_erp = int(last["ever_erp"].sum())
    n_dark = int((~last["ever_erp"]).sum())
    confirm = n_erp == 744 and n_dark == 470

    ever = tr.groupby("company_id", as_index=False).agg(
        n_cm=("period", "size"),
        mean_c=("a_uncat_share", "mean"),
        mean_a=("amt_uncat", "mean"),
        p50_c=("a_uncat_share", "median"),
        p50_a=("amt_uncat", "median"),
    )
    ever["ever_erp"] = ever["company_id"].isin(book)
    rows = []
    for name, part in (
        ("ever_erp_744", ever[ever["ever_erp"]]),
        ("never_erp_470", ever[~ever["ever_erp"]]),
    ):
        rows.append(
            {
                "group": name,
                "n_co": int(len(part)),
                "mean_count": _f(float(part["mean_c"].mean())),
                "p50_count": _f(float(part["p50_c"].median())),
                "mean_amt": _f(float(part["mean_a"].mean())),
                "p50_amt": _f(float(part["p50_a"].median())),
            }
        )
    tr2 = tr.copy()
    tr2["ever_erp"] = tr2["company_id"].isin(book)
    cm = []
    for name, part in (
        ("ever_erp", tr2[tr2["ever_erp"]]),
        ("never_erp", tr2[~tr2["ever_erp"]]),
    ):
        cm.append(
            {
                "group": name,
                "n_cm": int(len(part)),
                "n_co": int(part["company_id"].nunique()),
                "count": float(pd.to_numeric(part["a_uncat_share"], errors="coerce").mean()),
                "amt": float(pd.to_numeric(part["amt_uncat"], errors="coerce").mean()),
            }
        )
    dark_c = float(ever.loc[~ever["ever_erp"], "mean_c"].mean())
    erp_c = float(ever.loc[ever["ever_erp"], "mean_c"].mean())
    dark_a = float(ever.loc[~ever["ever_erp"], "mean_a"].mean())
    erp_a = float(ever.loc[ever["ever_erp"], "mean_a"].mean())
    same = bool(np.isfinite(dark_c) and np.isfinite(erp_c) and abs(dark_c - erp_c) < 0.05)
    hold_n = int(len(hold))
    hold_book = int(len(set(hold) & book))
    prose = (
        f"Train last-month companies: ever-ERP {n_erp} / never-ERP {n_dark} "
        f"({'CONFIRM 744/470' if confirm else 'counts differ from join QA'}). "
        f"Mean company count-uncat: invoiced {_pp(erp_c)} vs dark {_pp(dark_c)}; "
        f"amount-uncat {_pp(erp_a)} vs {_pp(dark_a)}. "
        f"{'Same uncat rate — uncat ≠ no ERP.' if same else 'Dark and invoiced file uncat at different rates.'} "
        f"Holdout ever-ERP coverage only: {hold_book}/{hold_n}."
    )
    print(prose)
    return {
        "n_erp": n_erp,
        "n_dark": n_dark,
        "confirm": confirm,
        "rows": rows,
        "cm": cm,
        "dark_c": dark_c,
        "erp_c": erp_c,
        "dark_a": dark_a,
        "erp_a": erp_a,
        "same": same,
        "hold_book": hold_book,
        "hold_n": hold_n,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 7 — Q6 lag1 only
# ---------------------------------------------------------------------------
def pass7_q6(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    cols = (
        "a_uncat_share",
        "amt_uncat",
        "a_uncat_share_lag1",
        "amt_uncat_lag1",
        "c_n_days_with_tx",
        "c_n_days_with_tx_lag1",
    )
    for y in (Y2, Y3, Y7, Y9):
        lab = tr[y].notna()
        for col in cols:
            if col not in tr.columns:
                continue
            res = signed_oof_auroc(tr[y], tr[col], tr["fold"], lab & tr[col].notna())
            store[(y, col)] = res
            rows.append(
                {
                    "y": y,
                    "col": col,
                    "n": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "present": _pp(float(tr.loc[lab, col].notna().mean()) if lab.any() else float("nan")),
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                    "sign": res["train_sign"] if not res["low_power"] else "—",
                }
            )
            print(
                f"Q6 {y} {col}: {'LOW_POWER' if res['low_power'] else _f(res['cv'])} "
                f"n_pos={res['n_pos']}"
            )

    def _cv(y, col) -> float:
        r = store.get((y, col))
        if r is None or r["low_power"]:
            return float("nan")
        return r["cv"]

    now_c = _cv(Y3, "a_uncat_share")
    lag_c = _cv(Y3, "a_uncat_share_lag1")
    now_a = _cv(Y3, "amt_uncat")
    lag_a = _cv(Y3, "amt_uncat_lag1")
    drop_c = now_c - lag_c if np.isfinite(now_c) and np.isfinite(lag_c) else float("nan")
    drop_a = now_a - lag_a if np.isfinite(now_a) and np.isfinite(lag_a) else float("nan")

    def _keep(now, lag) -> bool:
        return bool(
            np.isfinite(lag)
            and np.isfinite(now)
            and now >= 0.55
            and (now - lag) <= 0.03
            and lag >= 0.55
        )

    keep_c = _keep(now_c, lag_c)
    keep_a = _keep(now_a, lag_a)
    keep_q6 = keep_c or keep_a
    prose = (
        f"Y3 count now {_f(now_c)} vs lag1 {_f(lag_c)} (Δ {_f(drop_c, 3)}); "
        f"amount now {_f(now_a)} vs lag1 {_f(lag_a)} (Δ {_f(drop_a, 3)}). "
        f"{'KEEP as honest 1-month Q6' if keep_q6 else 'CLOSE as Q6 — lag1 does not hold contemporaneous skill (or contemporaneous is chance)'}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "now_c": now_c,
        "lag_c": lag_c,
        "now_a": now_a,
        "lag_a": lag_a,
        "drop_c": drop_c,
        "drop_a": drop_a,
        "keep_c": keep_c,
        "keep_a": keep_a,
        "keep_q6": keep_q6,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 8 — uncat months vs m_* missingness (in memory; do not merge M)
# ---------------------------------------------------------------------------
def pass8_mix_missing(tr: pd.DataFrame) -> dict:
    """High-uncat months vs mix-share holes. M is not in parquet; computed here."""
    x = pd.to_numeric(tr["a_uncat_share"], errors="coerce")
    amt = pd.to_numeric(tr["amt_uncat"], errors="coerce")
    mix_ok = tr["m_mix_ok"].eq(True)
    core_ok = tr["m_coll_vs_pay_ok"].eq(True)
    # store-family missingness (e/d/j) — opacity of the book vs missing other tables
    e_ar = tr["e_ar_issued"] if "e_ar_issued" in tr.columns else pd.Series(np.nan, index=tr.index)
    d_c = tr["d_n_cust"] if "d_n_cust" in tr.columns else pd.Series(np.nan, index=tr.index)
    j_p = tr["j_pay_match"] if "j_pay_match" in tr.columns else pd.Series(np.nan, index=tr.index)

    defined = x.notna()
    sl = tr.loc[defined].copy()
    sl["_x"] = x[defined]
    try:
        sl["_q"] = pd.qcut(sl["_x"], 5, labels=False, duplicates="drop")
    except ValueError:
        sl["_q"] = 0
    rows = []
    for q, g in sl.groupby("_q", sort=True):
        idx = g.index
        rows.append(
            {
                "q": int(q) + 1,
                "n_cm": int(len(g)),
                "p50_count": _f(float(g["_x"].median())),
                "p50_amt": _f(float(pd.to_numeric(g["amt_uncat"], errors="coerce").median())),
                "mix_defined": _pp(float(mix_ok.loc[idx].mean())),
                "coll_pay_defined": _pp(float(core_ok.loc[idx].mean())),
                "e_ar_defined": _pp(float(e_ar.loc[idx].notna().mean())),
                "d_cust_defined": _pp(float(d_c.loc[idx].notna().mean())),
                "j_pay_defined": _pp(float(j_p.loc[idx].notna().mean())),
            }
        )
    # empty-tx months: a_uncat_share NaN by construction
    empty = x.isna()
    n_empty = int(empty.sum())
    hi = defined & (x >= x[defined].quantile(0.80)) if defined.any() else defined
    lo = defined & (x <= x[defined].quantile(0.20)) if defined.any() else defined
    miss_hi_core = float((~core_ok[hi]).mean()) if hi.any() else float("nan")
    miss_lo_core = float((~core_ok[lo]).mean()) if lo.any() else float("nan")
    miss_hi_e = float(e_ar[hi].isna().mean()) if hi.any() else float("nan")
    miss_lo_e = float(e_ar[lo].isna().mean()) if lo.any() else float("nan")
    linked = bool(
        np.isfinite(miss_hi_core)
        and np.isfinite(miss_lo_core)
        and miss_hi_core >= miss_lo_core + 0.10
    )
    prose = (
        f"Empty-tx months (a_uncat_share NaN = m_* all NaN): {n_empty:,} / {len(tr):,}. "
        f"High-uncat Q5 vs Q1 `m_coll_vs_pay` missing {_pp(miss_hi_core)} vs {_pp(miss_lo_core)}; "
        f"e_ar_issued missing {_pp(miss_hi_e)} vs {_pp(miss_lo_e)}. "
        f"{'High-uncat months are also mix-core holes.' if linked else 'High-uncat is not just a mix-missingness dummy.'}"
    )
    print(prose)
    return {
        "rows": rows,
        "n_empty": n_empty,
        "miss_hi_core": miss_hi_core,
        "miss_lo_core": miss_lo_core,
        "miss_hi_e": miss_hi_e,
        "miss_lo_e": miss_lo_e,
        "linked": linked,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 9 — holdout coverage only
# ---------------------------------------------------------------------------
def pass9_holdout(panel: pd.DataFrame) -> dict:
    ho = panel[panel["split"] == "holdout"]
    rows = []
    for col, lab in (("a_uncat_share", "count"), ("amt_uncat", "amount")):
        x = pd.to_numeric(ho[col], errors="coerce")
        rows.append(
            {
                "col": lab,
                "n_cm": int(len(ho)),
                "n_co": int(ho["company_id"].nunique()),
                "defined": _pp(float(x.notna().mean())),
                "mean": _f(float(x.mean())),
                "p50": _f(float(x.median())),
            }
        )
    prose = (
        f"Holdout coverage only (no AUROC): {ho['company_id'].nunique()} companies / "
        f"{len(ho):,} CM. Count defined {_pp(float(ho['a_uncat_share'].notna().mean()))}, "
        f"amount {_pp(float(ho['amt_uncat'].notna().mean()))}."
    )
    print(prose)
    return {"rows": rows, "n_cm": int(len(ho)), "n_co": int(ho["company_id"].nunique()), "prose": prose}


# ---------------------------------------------------------------------------
# Pass 10 — company-mean (style) vs demeaned (shock)
# ---------------------------------------------------------------------------
def pass10_demean(tr: pd.DataFrame) -> dict:
    feats = {
        "count_co_mean": tr["count_co_mean"],
        "count_demean": tr["count_demean"],
        "amt_co_mean": tr["amt_co_mean"],
        "amt_demean": tr["amt_demean"],
    }
    rows = []
    store = {}
    for y in (Y2, Y3, Y7, Y9):
        lab = tr[y].notna()
        for name, col in feats.items():
            res = signed_oof_auroc(tr[y], col, tr["fold"], lab)
            store[(y, name)] = res
            rows.append(
                {
                    "y": y,
                    "feature": name,
                    "n": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                    "sd": _f(res["sd"]),
                    "sign": res["train_sign"] if not res["low_power"] else "—",
                }
            )
            print(
                f"STYLE {y} {name}: "
                f"{'LOW_POWER' if res['low_power'] else _f(res['cv'])} "
                f"n_pos={res['n_pos']}"
            )

    def _cv(y, feat) -> float:
        r = store.get((y, feat))
        if r is None or r["low_power"]:
            return float("nan")
        return r["cv"]

    y2_mean = _cv(Y2, "amt_co_mean")
    y2_shock = _cv(Y2, "amt_demean")
    y3_mean = _cv(Y3, "amt_co_mean")
    y3_shock = _cv(Y3, "amt_demean")
    style_carries_y2 = bool(
        np.isfinite(y2_mean) and y2_mean >= 0.58 and (not np.isfinite(y2_shock) or y2_shock < 0.55)
    )
    prose = (
        f"Y2 amount company-mean {_f(y2_mean)} vs demean (shock) {_f(y2_shock)}. "
        f"Y3 mean {_f(y3_mean)} vs shock {_f(y3_shock)}. "
        f"{'Y2 skill is the company style, not a month shock — not Q5 change.' if style_carries_y2 else 'Shock component is not clearly empty.'}"
    )
    print(prose)
    return {
        "rows": rows,
        "y2_mean": y2_mean,
        "y2_shock": y2_shock,
        "y3_mean": y3_mean,
        "y3_shock": y3_shock,
        "style_carries_y2": style_carries_y2,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 11 — Y2 0.602 vs activity (a_n_tx)
# ---------------------------------------------------------------------------
def pass11_activity(tr: pd.DataFrame) -> dict:
    amt = pd.to_numeric(tr["amt_uncat"], errors="coerce")
    cnt = pd.to_numeric(tr["a_uncat_share"], errors="coerce")
    ntx = pd.to_numeric(tr.get("a_n_tx", tr.get("n_tx")), errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    rho_amt_n = spearman(amt, ntx)
    rho_cnt_n = spearman(cnt, ntx)
    rho_amt_d = spearman(amt, days)
    # train-only terciles of a_n_tx among finite amt
    both = amt.notna() & ntx.notna()
    try:
        terc = pd.qcut(ntx[both], 3, labels=False, duplicates="drop")
    except ValueError:
        terc = pd.Series(0, index=ntx[both].index)
    sl = tr.loc[both].copy()
    sl["_t"] = terc.to_numpy()
    sl["_a"] = amt[both]
    try:
        sl["_q"] = pd.qcut(sl["_a"], 4, labels=False, duplicates="drop")
    except ValueError:
        sl["_q"] = 0
    rows = []
    for (t, q), g in sl.groupby(["_t", "_q"], sort=True):
        yy = pd.to_numeric(g[Y2], errors="coerce")
        rows.append(
            {
                "ntx_tercile": int(t) + 1,
                "amt_q": int(q) + 1,
                "n_cm": int(len(g)),
                "n_labeled": int(yy.notna().sum()),
                "Y2": _pp(float(yy.mean()) if yy.notna().any() else float("nan")),
                "p50_amt": _f(float(g["_a"].median())),
                "p50_ntx": _f(float(pd.to_numeric(g["a_n_tx"], errors="coerce").median()), 1),
            }
        )
    # residual: amt after rank of n_tx (simple: within-tercile AUROC)
    fold_rows = []
    for t, g in sl.groupby("_t", sort=True):
        res = signed_oof_auroc(g[Y2], g["_a"], g["fold"], g[Y2].notna())
        fold_rows.append(
            {
                "ntx_tercile": int(t) + 1,
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "Y2_amt_CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
            }
        )
    clones_ntx = bool(np.isfinite(rho_amt_n) and abs(rho_amt_n) >= 0.50)
    prose = (
        f"Spearman amount-uncat↔a_n_tx {_f(rho_amt_n)} "
        f"(count↔n_tx {_f(rho_cnt_n)}; amount↔days {_f(rho_amt_d)}). "
        f"{'Amount-uncat clones activity.' if clones_ntx else 'Amount-uncat is not an activity clone (|ρ|<0.50).'} "
        f"Y2 night single a_n_tx was 0.598 vs amount 0.602 — almost a tie."
    )
    print(prose)
    return {
        "rho_amt_n": rho_amt_n,
        "rho_cnt_n": rho_cnt_n,
        "rho_amt_d": rho_amt_d,
        "clones_ntx": clones_ntx,
        "rows": rows,
        "fold_rows": fold_rows,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 12 — zero pile (why 4 bins) + 0 / low / high
# ---------------------------------------------------------------------------
def pass12_zero(tr: pd.DataFrame) -> dict:
    cnt = pd.to_numeric(tr["a_uncat_share"], errors="coerce")
    amt = pd.to_numeric(tr["amt_uncat"], errors="coerce")
    n_def = int(cnt.notna().sum())
    n_zero_c = int((cnt == 0).sum())
    n_zero_a = int((amt == 0).sum())
    zero_share = _pct(n_zero_c, n_def)
    # 0 / (0, p50 of positives] / above
    def _bands(x: pd.Series, name: str) -> list[dict]:
        nn = x.dropna()
        pos = nn[nn > 0]
        cut = float(pos.median()) if len(pos) else float("nan")
        bands = []
        labels = [
            ("zero", x == 0),
            ("low_pos", (x > 0) & (x <= cut) if np.isfinite(cut) else (x > 0)),
            ("high_pos", (x > cut) if np.isfinite(cut) else pd.Series(False, index=x.index)),
        ]
        for lab, m in labels:
            g = tr.loc[m.fillna(False)]
            rec = {"col": name, "band": lab, "n_cm": int(len(g)), "cut": cut}
            for y in (Y2, Y3, Y7, Y9):
                yy = pd.to_numeric(g[y], errors="coerce")
                rec[f"{y}_rate"] = float(yy.mean()) if yy.notna().any() else float("nan")
                rec[f"{y}_n"] = int(yy.notna().sum())
            bands.append(rec)
        return bands

    bands = _bands(cnt, "count") + _bands(amt, "amt")
    prose = (
        f"Defined count months {n_def:,}; exact-zero {_pp(zero_share)} "
        f"(n={n_zero_c:,}) — qcut drops a quintile (4 bins). "
        f"Amount exact-zero {n_zero_a:,}."
    )
    print(prose)
    return {
        "n_def": n_def,
        "n_zero_c": n_zero_c,
        "n_zero_a": n_zero_a,
        "zero_share": zero_share,
        "bands": bands,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 13 — hide-euro months (amount − count ≥ 0.20)
# ---------------------------------------------------------------------------
def pass13_hide(tr: pd.DataFrame) -> dict:
    cnt = pd.to_numeric(tr["a_uncat_share"], errors="coerce")
    amt = pd.to_numeric(tr["amt_uncat"], errors="coerce")
    both = cnt.notna() & amt.notna()
    hid = both & ((amt - cnt) >= 0.20)
    small = both & ((cnt - amt) >= 0.20)
    mid = both & ~hid & ~small
    rows = []
    for name, m in (("hide_euros", hid), ("small_tickets", small), ("aligned", mid)):
        g = tr.loc[m]
        rec = {
            "slice": name,
            "n_cm": int(len(g)),
            "n_co": int(g["company_id"].nunique()),
            "p50_count": _f(float(cnt[m].median()) if m.any() else float("nan")),
            "p50_amt": _f(float(amt[m].median()) if m.any() else float("nan")),
        }
        for y in (Y2, Y3, Y7, Y9):
            yy = pd.to_numeric(g[y], errors="coerce")
            rec[y] = _pp(float(yy.mean()) if yy.notna().any() else float("nan"))
        rows.append(rec)
    flag = hid.astype(float)
    y2 = signed_oof_auroc(tr[Y2], flag, tr["fold"], tr[Y2].notna() & both)
    y3 = signed_oof_auroc(tr[Y3], flag, tr["fold"], tr[Y3].notna() & both)
    prose = (
        f"Hide-euro months {int(hid.sum()):,} / aligned {int(mid.sum()):,} / small-ticket {int(small.sum()):,}. "
        f"Hide-euro dummy Y2 {_f(y2['cv'])} Y3 {_f(y3['cv'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "n_hid": int(hid.sum()),
        "y2_cv": y2["cv"] if not y2["low_power"] else float("nan"),
        "y3_cv": y3["cv"] if not y3["low_power"] else float("nan"),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 14 — cash_settlement vs cash_settlements (design note)
# ---------------------------------------------------------------------------
def pass14_alias(con) -> dict:
    raw = con.execute(
        """
        SELECT
          CAST(COALESCE(category, '(null)') AS VARCHAR) AS category,
          COUNT(*) AS n
        FROM transactions
        WHERE "date" IS NOT NULL
          AND "date" >= DATE '2024-09-01'
          AND "date" < DATE '2026-09-01'
          AND category IN ('cash_settlement', 'cash_settlements', 'uncategorized')
        GROUP BY 1
        """
    ).df()
    rows = raw.to_dict("records")
    n_sing = int(raw.loc[raw["category"] == "cash_settlement", "n"].sum()) if len(raw) else 0
    n_plur = int(raw.loc[raw["category"] == "cash_settlements", "n"].sum()) if len(raw) else 0
    prose = (
        f"`cash_settlement` (mapped op_in) n={n_sing:,}; "
        f"`cash_settlements` (also in CAT_MAP, never seen) n={n_plur:,}. "
        "Do not drop the plural alias tonight — it is a Javier synonym, not a leftover."
    )
    print(prose)
    return {"rows": rows, "n_sing": n_sing, "n_plur": n_plur, "prose": prose}


# ---------------------------------------------------------------------------
# Pass 15 — dark × size tercile (is the 6.7pp gap just size?)
# ---------------------------------------------------------------------------
def pass15_dark_size(tr: pd.DataFrame, p6: dict) -> dict:
    size = pd.to_numeric(tr["log_in3"], errors="coerce")
    defined = size.notna()
    try:
        terc = pd.qcut(size[defined], 3, labels=False, duplicates="drop")
    except ValueError:
        terc = pd.Series(0, index=size[defined].index)
    sl = tr.loc[defined].copy()
    sl["_t"] = terc.to_numpy()
    if "ever_erp" not in sl.columns:
        raise RuntimeError("ever_erp missing — attach from book_invoice_ids")
    rows = []
    for t, g in sl.groupby("_t", sort=True):
        for name, part in (("erp", g[g["ever_erp"] == True]), ("dark", g[g["ever_erp"] == False])):
            rows.append(
                {
                    "size_tercile": int(t) + 1,
                    "group": name,
                    "n_cm": int(len(part)),
                    "n_co": int(part["company_id"].nunique()),
                    "count": _pp(float(pd.to_numeric(part["a_uncat_share"], errors="coerce").mean())),
                    "amount": _pp(float(pd.to_numeric(part["amt_uncat"], errors="coerce").mean())),
                    "p50_in3": _f(float(pd.to_numeric(part["a_in3"], errors="coerce").median()), 0),
                }
            )
    prose = (
        f"Dark count-uncat gap raw {_pp(p6['dark_c'] - p6['erp_c'])}. "
        "Size terciles of log1p(a_in3) on train (holdout never in cuts)."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


# ---------------------------------------------------------------------------
# Pass 16 — company-median uncat vs ever-Y2 (style as identity)
# ---------------------------------------------------------------------------
def pass16_co_y2(tr: pd.DataFrame) -> dict:
    g = tr.groupby("company_id", as_index=False).agg(
        med_c=("a_uncat_share", "median"),
        med_a=("amt_uncat", "median"),
        ever_y2=(Y2, "max"),
        n_y2=(Y2, "count"),
        mean_y2=(Y2, "mean"),
        n_cm=("period", "size"),
    )
    g = g[g["n_y2"] > 0].copy()
    try:
        g["_q"] = pd.qcut(g["med_a"], 4, labels=False, duplicates="drop")
    except ValueError:
        g["_q"] = 0
    rows = []
    for q, part in g.groupby("_q", sort=True):
        rows.append(
            {
                "amt_med_q": int(q) + 1,
                "n_co": int(len(part)),
                "p50_med_amt": _f(float(part["med_a"].median())),
                "ever_Y2": _pp(float(part["ever_y2"].mean())),
                "mean_Y2_cm": _pp(float(part["mean_y2"].mean())),
            }
        )
    rho = spearman(g["med_a"], g["mean_y2"])
    prose = (
        f"Company-median amount-uncat vs company Y2 rate Spearman {_f(rho)} "
        f"(n_co={len(g):,}). Style identity, not a month why."
    )
    print(prose)
    return {"rows": rows, "rho": rho, "n_co": int(len(g)), "prose": prose}


# ---------------------------------------------------------------------------
# Pass 17 — always-messy vs always-clean company cards
# ---------------------------------------------------------------------------
def pass17_messy_cos(tr: pd.DataFrame) -> dict:
    g = tr.groupby("company_id", as_index=False).agg(
        med_c=("a_uncat_share", "median"),
        sd_c=("a_uncat_share", "std"),
        med_a=("amt_uncat", "median"),
        mean_in3=("a_in3", "median"),
        ever_erp=("ever_erp", "max"),
        ever_y2=(Y2, "max"),
        rate_y2=(Y2, "mean"),
        ever_y3=(Y3, "max"),
        rate_y3=(Y3, "mean"),
        n_cm=("period", "size"),
    )
    messy = (g["med_c"] >= 0.40) & (g["sd_c"].fillna(0) <= 0.15)
    clean = (g["med_c"] <= 0.05) & (g["sd_c"].fillna(0) <= 0.10)
    mid = ~messy & ~clean
    rows = []
    for name, m in (("always_messy", messy), ("always_clean", clean), ("other", mid)):
        part = g[m]
        rows.append(
            {
                "group": name,
                "n_co": int(len(part)),
                "dark": _pp(float((~part["ever_erp"].astype(bool)).mean()) if len(part) else float("nan")),
                "p50_in3": _f(float(part["mean_in3"].median()) if len(part) else float("nan"), 0),
                "p50_med_c": _f(float(part["med_c"].median()) if len(part) else float("nan")),
                "ever_Y2": _pp(float(part["ever_y2"].mean()) if len(part) else float("nan")),
                "Y2_cm": _pp(float(part["rate_y2"].mean()) if len(part) else float("nan")),
                "ever_Y3": _pp(float(part["ever_y3"].mean()) if len(part) else float("nan")),
                "Y3_cm": _pp(float(part["rate_y3"].mean()) if len(part) else float("nan")),
            }
        )
    prose = (
        f"Always-messy {int(messy.sum())} vs always-clean {int(clean.sum())} train companies "
        f"(median count-uncat ≥0.40 & sd≤0.15 vs ≤0.05 & sd≤0.10). "
        "A type, not a month."
    )
    print(prose)
    return {
        "rows": rows,
        "n_messy": int(messy.sum()),
        "n_clean": int(clean.sum()),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 18 — calendar of uncat (seasonal dummy?)
# ---------------------------------------------------------------------------
def pass18_calendar(tr: pd.DataFrame) -> dict:
    sl = tr.copy()
    sl["cal"] = sl["period"].dt.month
    rows = []
    for m in range(1, 13):
        g = sl[sl["cal"] == m]
        rows.append(
            {
                "month": pd.Timestamp(2000, m, 1).strftime("%b"),
                "n_cm": int(len(g)),
                "count": _pp(float(pd.to_numeric(g["a_uncat_share"], errors="coerce").mean())),
                "amount": _pp(float(pd.to_numeric(g["amt_uncat"], errors="coerce").mean())),
                "Y2": _pp(float(pd.to_numeric(g[Y2], errors="coerce").mean())),
            }
        )
    means = [float(pd.to_numeric(sl.loc[sl["cal"] == m, "a_uncat_share"], errors="coerce").mean()) for m in range(1, 13)]
    spread = float(np.nanmax(means) - np.nanmin(means)) if any(np.isfinite(means)) else float("nan")
    seasonal = bool(np.isfinite(spread) and spread >= 0.08)
    prose = (
        f"Stacked Jan–Dec count-uncat range {_pp(spread)}. "
        f"{'Seasonal dummy — inspect calendar.' if seasonal else 'No strong calendar dummy (range <8pp).'}"
    )
    print(prose)
    return {"rows": rows, "spread": spread, "seasonal": seasonal, "prose": prose}


# ---------------------------------------------------------------------------
# Pass 19 — uncat euro sign (in vs out)
# ---------------------------------------------------------------------------
def pass19_sign(con) -> dict:
    hold = load_holdout()
    raw = con.execute(
        """
        SELECT
          CAST(t.company_id AS VARCHAR) AS company_id,
          SUM(CASE WHEN t.amount > 0 THEN t.amount ELSE 0 END) AS uncat_in,
          SUM(CASE WHEN t.amount < 0 THEN -t.amount ELSE 0 END) AS uncat_out,
          SUM(ABS(t.amount)) AS uncat_abs,
          COUNT(*) AS n
        FROM transactions t
        WHERE t."date" IS NOT NULL
          AND t."date" >= DATE '2024-09-01'
          AND t."date" < DATE '2026-09-01'
          AND t.category = 'uncategorized'
        GROUP BY 1
        """
    ).df()
    raw["company_id"] = raw["company_id"].astype(str)
    raw = raw.loc[~raw["company_id"].isin(hold)].copy()
    assert_no_holdout(raw["company_id"])
    tot_in = float(raw["uncat_in"].sum())
    tot_out = float(raw["uncat_out"].sum())
    tot = tot_in + tot_out
    share_in = _pct(tot_in, tot)
    # company mix
    raw["in_share"] = np.where(raw["uncat_abs"] > 0, raw["uncat_in"] / raw["uncat_abs"], np.nan)
    p50_in = float(raw["in_share"].median()) if len(raw) else float("nan")
    mostly_in = int((raw["in_share"] >= 0.70).sum())
    mostly_out = int((raw["in_share"] <= 0.30).sum())
    prose = (
        f"Train uncategorized euros: inflow {_pp(share_in)} / outflow {_pp(1 - share_in)} "
        f"(€ in {tot_in:,.0f} / out {tot_out:,.0f}). "
        f"Company in-share p50 {_f(p50_in)}; mostly-in {mostly_in} / mostly-out {mostly_out} / n={len(raw)}."
    )
    print(prose)
    return {
        "share_in": share_in,
        "tot_in": tot_in,
        "tot_out": tot_out,
        "p50_in": p50_in,
        "mostly_in": mostly_in,
        "mostly_out": mostly_out,
        "n_co": int(len(raw)),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 20 — Y3 residual after days tercile
# ---------------------------------------------------------------------------
def pass20_days(tr: pd.DataFrame) -> dict:
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    amt = pd.to_numeric(tr["amt_uncat"], errors="coerce")
    lab = tr[Y3].notna() & days.notna() & amt.notna()
    sl = tr.loc[lab].copy()
    sl["_d"] = days[lab]
    sl["_a"] = amt[lab]
    try:
        sl["_t"] = pd.qcut(sl["_d"], 3, labels=False, duplicates="drop")
    except ValueError:
        sl["_t"] = 0
    rows = []
    for t, g in sl.groupby("_t", sort=True):
        res = signed_oof_auroc(g[Y3], g["_a"], g["fold"], g[Y3].notna())
        rows.append(
            {
                "days_tercile": int(t) + 1,
                "p50_days": _f(float(g["_d"].median()), 1),
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "Y3_amt_CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                "Y3_rate": _pp(float(pd.to_numeric(g[Y3], errors="coerce").mean())),
            }
        )
    prose = (
        "Y3 amount-uncat inside `c_n_days_with_tx` terciles (train labeled). "
        "If residual is chance, uncat adds nothing after the 0.711 days bar."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


# ---------------------------------------------------------------------------
# Pass 21 — months-on-book (onboarding mess?)
# ---------------------------------------------------------------------------
def pass21_sofar(tr: pd.DataFrame) -> dict:
    sl = tr.sort_values(["company_id", "period"]).copy()
    sl["so_far"] = sl.groupby("company_id", sort=False).cumcount() + 1
    buckets = [
        ("1-3", 1, 3),
        ("4-6", 4, 6),
        ("7-12", 7, 12),
        ("13-18", 13, 18),
        ("19-24", 19, 24),
    ]
    rows = []
    for lab, lo, hi in buckets:
        g = sl[(sl["so_far"] >= lo) & (sl["so_far"] <= hi)]
        rows.append(
            {
                "so_far": lab,
                "n_cm": int(len(g)),
                "n_co": int(g["company_id"].nunique()),
                "count": _pp(float(pd.to_numeric(g["a_uncat_share"], errors="coerce").mean())),
                "amount": _pp(float(pd.to_numeric(g["amt_uncat"], errors="coerce").mean())),
                "Y2": _pp(float(pd.to_numeric(g[Y2], errors="coerce").mean())),
            }
        )
    early = sl[sl["so_far"] <= 3]
    late = sl[sl["so_far"] >= 13]
    early_c = float(pd.to_numeric(early["a_uncat_share"], errors="coerce").mean())
    late_c = float(pd.to_numeric(late["a_uncat_share"], errors="coerce").mean())
    onboard = bool(np.isfinite(early_c) and np.isfinite(late_c) and early_c >= late_c + 0.05)
    prose = (
        f"Months 1–3 count-uncat {_pp(early_c)} vs months 13+ {_pp(late_c)}. "
        f"{'Onboarding hole — early books are messier.' if onboard else 'Not an onboarding hole (early ≈ late).'} "
        "Style from month 1, not a trail that gets labeled later."
    )
    print(prose)
    return {
        "rows": rows,
        "early_c": early_c,
        "late_c": late_c,
        "onboard": onboard,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 22 — messy companies: more often Y3-labeled (stressed)?
# ---------------------------------------------------------------------------
def pass22_y3_label(tr: pd.DataFrame) -> dict:
    g = tr.groupby("company_id", as_index=False).agg(
        med_c=("a_uncat_share", "median"),
        sd_c=("a_uncat_share", "std"),
        n_cm=("period", "size"),
        n_y3=(Y3, "count"),
        n_y3_pos=(Y3, "sum"),
    )
    g["y3_cov"] = g["n_y3"] / g["n_cm"]
    messy = (g["med_c"] >= 0.40) & (g["sd_c"].fillna(0) <= 0.15)
    clean = (g["med_c"] <= 0.05) & (g["sd_c"].fillna(0) <= 0.10)
    rows = []
    for name, m in (("always_messy", messy), ("always_clean", clean)):
        part = g[m]
        rows.append(
            {
                "group": name,
                "n_co": int(len(part)),
                "Y3_label_cov": _pp(float(part["y3_cov"].mean()) if len(part) else float("nan")),
                "mean_n_pos": _f(float(part["n_y3_pos"].mean()) if len(part) else float("nan"), 2),
            }
        )
    gap = (
        float(g.loc[messy, "y3_cov"].mean()) - float(g.loc[clean, "y3_cov"].mean())
        if messy.any() and clean.any()
        else float("nan")
    )
    prose = (
        f"Y3 is labeled only among stressed. Always-messy Y3-coverage minus clean "
        f"{_pp(gap)}. A higher recover *rate among labeled* can still be a type "
        f"(more often stressed), not Q5 readability."
    )
    print(prose)
    return {"rows": rows, "gap": gap, "prose": prose}


# ---------------------------------------------------------------------------
# Pass 23 — within-company early vs late (selection vs cleaning)
# ---------------------------------------------------------------------------
def pass23_within(tr: pd.DataFrame) -> dict:
    sl = tr.sort_values(["company_id", "period"]).copy()
    sl["so_far"] = sl.groupby("company_id", sort=False).cumcount() + 1
    long_ids = sl.loc[sl["so_far"] >= 13, "company_id"].unique()
    long = sl[sl["company_id"].isin(long_ids)]
    early = long[long["so_far"] <= 3].groupby("company_id")["a_uncat_share"].mean()
    late = long[long["so_far"] >= 13].groupby("company_id")["a_uncat_share"].mean()
    both = pd.DataFrame({"early": early, "late": late}).dropna()
    delta = both["late"] - both["early"]
    n = int(len(both))
    mean_d = float(delta.mean()) if n else float("nan")
    p_clean = float((delta < -0.05).mean()) if n else float("nan")
    p_worse = float((delta > 0.05).mean()) if n else float("nan")
    rho = spearman(both["early"], both["late"]) if n else float("nan")
    # short books (never reach 13)
    short_ids = set(sl["company_id"]) - set(long_ids)
    short = sl[sl["company_id"].isin(short_ids)]
    short_c = float(pd.to_numeric(short["a_uncat_share"], errors="coerce").mean())
    long_c = float(pd.to_numeric(long["a_uncat_share"], errors="coerce").mean())
    prose = (
        f"Companies with ≥13 months: n={n:,}. Within-company late−early count-uncat "
        f"mean Δ {_f(mean_d)} (share cleaner>5pp {_pp(p_clean)}; worse {_pp(p_worse)}); "
        f"early↔late Spearman {_f(rho)}. "
        f"Short-book (<13) CM count-uncat {_pp(short_c)} vs long-book {_pp(long_c)}. "
        f"{'Selection: short books are messier; long books stay messy if they start messy.' if (np.isfinite(rho) and rho >= 0.70) else 'Within-company path is mixed.'}"
    )
    print(prose)
    return {
        "n": n,
        "mean_d": mean_d,
        "p_clean": p_clean,
        "p_worse": p_worse,
        "rho": rho,
        "short_c": short_c,
        "long_c": long_c,
        "n_short": int(len(short_ids)),
        "n_long": int(len(long_ids)),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 24 — all-uncat months (amount ≥ 0.95)
# ---------------------------------------------------------------------------
def pass24_alluncat(tr: pd.DataFrame) -> dict:
    amt = pd.to_numeric(tr["amt_uncat"], errors="coerce")
    hi = amt >= 0.95
    mid = (amt >= 0.05) & (amt < 0.95)
    z = amt == 0
    rows = []
    for name, m in (("all_uncat>=0.95", hi), ("mixed", mid), ("zero", z)):
        g = tr.loc[m]
        rec = {
            "slice": name,
            "n_cm": int(len(g)),
            "n_co": int(g["company_id"].nunique()),
            "dark": _pp(float((~g["ever_erp"]).mean()) if len(g) else float("nan")),
        }
        for y in (Y2, Y3, Y7, Y9):
            yy = pd.to_numeric(g[y], errors="coerce")
            rec[y] = _pp(float(yy.mean()) if yy.notna().any() else float("nan"))
        rows.append(rec)
    prose = (
        f"All-uncat months (amount≥0.95): {int(hi.sum()):,} / {int(amt.notna().sum()):,}. "
        "If Y rates match the high quintile, the tail is the style dummy, not a new event."
    )
    print(prose)
    return {"rows": rows, "n_hi": int(hi.sum()), "prose": prose}


# ---------------------------------------------------------------------------
# Pass 25 — group ICC of company-median uncat (holding style?)
# ---------------------------------------------------------------------------
def pass25_group_icc(tr: pd.DataFrame) -> dict:
    co = tr.groupby(["company_id", "group_id"], as_index=False).agg(
        med_c=("a_uncat_share", "median"),
        med_a=("amt_uncat", "median"),
    )
    icc_c = icc_anova(co["med_c"], co["group_id"])
    icc_a = icc_anova(co["med_a"], co["group_id"])
    # share of groups that are mono-messy or mono-clean
    g = co.groupby("group_id").agg(
        n=("company_id", "size"),
        n_messy=("med_c", lambda s: int((s >= 0.40).sum())),
        n_clean=("med_c", lambda s: int((s <= 0.05).sum())),
    )
    multi = g[g["n"] >= 2]
    mono_messy = int(((multi["n_messy"] == multi["n"]) & (multi["n_messy"] > 0)).sum())
    mono_clean = int(((multi["n_clean"] == multi["n"]) & (multi["n_clean"] > 0)).sum())
    prose = (
        f"Company-median count-uncat ICC across group_id {_f(icc_c['icc'])} "
        f"(amount {_f(icc_a['icc'])}). Multi-company groups {int(len(multi))}: "
        f"mono-messy {mono_messy}, mono-clean {mono_clean}. "
        f"{'Holding style' if np.isfinite(icc_c['icc']) and icc_c['icc'] >= 0.40 else 'Style is company-level, not a holding dummy'}."
    )
    print(prose)
    return {
        "icc_c": icc_c["icc"],
        "icc_a": icc_a["icc"],
        "n_multi": int(len(multi)),
        "mono_messy": mono_messy,
        "mono_clean": mono_clean,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 26 — 360 all-dark vs 110 mixed-dark uncat (do not edit y11 / sibling_h)
# ---------------------------------------------------------------------------
def pass26_dark_mix(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", sort=False).last().reset_index()
    g = last.groupby("group_id", as_index=False).agg(
        n=("company_id", "size"),
        n_erp=("ever_erp", "sum"),
    )
    g["mix"] = np.where(g["n_erp"] == 0, "all_dark", np.where(g["n_erp"] == g["n"], "all_erp", "mixed"))
    last = last.merge(g[["group_id", "mix"]], on="group_id", how="left")
    last["bucket"] = np.where(
        last["ever_erp"],
        "invoiced_744",
        np.where(last["mix"] == "all_dark", "all_dark_360", "mixed_dark_110"),
    )
    # confirm counts
    n_inv = int((last["bucket"] == "invoiced_744").sum())
    n_360 = int((last["bucket"] == "all_dark_360").sum())
    n_110 = int((last["bucket"] == "mixed_dark_110").sum())
    confirm = n_inv == 744 and n_360 == 360 and n_110 == 110
    ever = tr.groupby("company_id", as_index=False).agg(
        mean_c=("a_uncat_share", "mean"),
        mean_a=("amt_uncat", "mean"),
    )
    ever = ever.merge(last[["company_id", "bucket"]], on="company_id", how="left")
    rows = []
    for name in ("invoiced_744", "all_dark_360", "mixed_dark_110"):
        part = ever[ever["bucket"] == name]
        rows.append(
            {
                "group": name,
                "n_co": int(len(part)),
                "mean_count": _f(float(part["mean_c"].mean()) if len(part) else float("nan")),
                "mean_amt": _f(float(part["mean_a"].mean()) if len(part) else float("nan")),
            }
        )
    prose = (
        f"Train last-month: invoiced {n_inv} / all-dark {n_360} / mixed-dark {n_110} "
        f"({'CONFIRM 744/360/110' if confirm else 'off sibling_h counts'}). "
        "Uncat is not a dark-only hole."
    )
    print(prose)
    return {
        "rows": rows,
        "confirm": confirm,
        "n_inv": n_inv,
        "n_360": n_360,
        "n_110": n_110,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 27 — uncat ticket size vs mapped
# ---------------------------------------------------------------------------
def pass27_ticket(con) -> dict:
    hold = load_holdout()
    cats = _cat_sql_list()
    raw = con.execute(
        f"""
        SELECT
          CAST(t.company_id AS VARCHAR) AS company_id,
          CASE WHEN t.category = 'uncategorized'
                 OR t.category IS NULL
                 OR t.category NOT IN ({cats})
               THEN 'uncat' ELSE 'mapped' END AS kind,
          COUNT(*) AS n,
          SUM(ABS(t.amount)) AS abs_amt,
          AVG(ABS(t.amount)) AS mean_abs,
          QUANTILE_CONT(ABS(t.amount), 0.5) AS p50_abs
        FROM transactions t
        WHERE t."date" IS NOT NULL
          AND t."date" >= DATE '2024-09-01'
          AND t."date" < DATE '2026-09-01'
        GROUP BY 1, 2
        """
    ).df()
    raw["company_id"] = raw["company_id"].astype(str)
    raw = raw.loc[~raw["company_id"].isin(hold)].copy()
    assert_no_holdout(raw["company_id"])
    agg = raw.groupby("kind", as_index=False).agg(
        n=("n", "sum"),
        abs_amt=("abs_amt", "sum"),
        mean_abs=("mean_abs", "median"),
        p50_abs=("p50_abs", "median"),
    )
    rows = agg.to_dict("records")
    u = agg[agg["kind"] == "uncat"]
    m = agg[agg["kind"] == "mapped"]
    pooled_u = float(u["abs_amt"].iloc[0] / u["n"].iloc[0]) if len(u) and float(u["n"].iloc[0]) else float("nan")
    pooled_m = float(m["abs_amt"].iloc[0] / m["n"].iloc[0]) if len(m) and float(m["n"].iloc[0]) else float("nan")
    ratio = pooled_u / pooled_m if np.isfinite(pooled_u) and np.isfinite(pooled_m) and pooled_m else float("nan")
    typ_u = float(u["mean_abs"].iloc[0]) if len(u) else float("nan")
    typ_m = float(m["mean_abs"].iloc[0]) if len(m) else float("nan")
    fat = bool(np.isfinite(ratio) and ratio >= 1.15 and np.isfinite(typ_u) and typ_u < typ_m)
    prose = (
        f"Pooled |amt|/tx uncat {_f(pooled_u, 0)} vs mapped {_f(pooled_m, 0)} ({ratio:.2f}×). "
        f"Typical-company mean ticket uncat {_f(typ_u, 0)} vs mapped {_f(typ_m, 0)}. "
        f"{'Fat tail: a few huge uncat txs lift pooled euros; typical uncat tickets are smaller.' if fat else 'Ticket-size story is mixed.'}"
    )
    print(prose)
    return {
        "rows": rows,
        "ratio": ratio,
        "pooled_u": pooled_u,
        "pooled_m": pooled_m,
        "typ_u": typ_u,
        "typ_m": typ_m,
        "fat": fat,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 28 — pending vs uncat (is opacity just not-yet-booked?)
# ---------------------------------------------------------------------------
def pass28_pending(tr: pd.DataFrame) -> dict:
    if "a_pending_share" not in tr.columns:
        return {
            "rows": [],
            "rho": float("nan"),
            "prose": "`a_pending_share` not in store — skipped.",
        }
    pend = pd.to_numeric(tr["a_pending_share"], errors="coerce")
    cnt = pd.to_numeric(tr["a_uncat_share"], errors="coerce")
    amt = pd.to_numeric(tr["amt_uncat"], errors="coerce")
    rho_c = spearman(pend, cnt)
    rho_a = spearman(pend, amt)
    nn = pend.dropna()
    modal = float((nn == 0).mean()) if len(nn) else float("nan")
    both = pend.notna() & cnt.notna()
    hi_u = both & (cnt >= cnt[both].quantile(0.80))
    lo_u = both & (cnt <= cnt[both].quantile(0.20))
    pend_hi = float(pend[hi_u].mean()) if hi_u.any() else float("nan")
    pend_lo = float(pend[lo_u].mean()) if lo_u.any() else float("nan")
    clone = bool(np.isfinite(rho_c) and abs(rho_c) >= 0.50)
    prose = (
        f"`a_pending_share` vs count-uncat Spearman {_f(rho_c)} (amount {_f(rho_a)}); "
        f"share exactly 0 {_pp(modal)}. High-uncat pending mean {_pp(pend_hi)} vs low {_pp(pend_lo)}. "
        f"{'Pending clones uncat.' if clone else 'Uncat is not a pending dummy (booked txs can be uncategorized).'}"
    )
    print(prose)
    return {
        "rho": rho_c,
        "rho_a": rho_a,
        "modal0": modal,
        "pend_hi": pend_hi,
        "pend_lo": pend_lo,
        "clone": clone,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 29 — holding mean vs within-group residual
# ---------------------------------------------------------------------------
def pass29_group_demean(tr: pd.DataFrame) -> dict:
    sl = tr.copy()
    sl["_gmean"] = sl.groupby("group_id")["amt_uncat"].transform("mean")
    sl["_gresid"] = pd.to_numeric(sl["amt_uncat"], errors="coerce") - sl["_gmean"]
    rows = []
    store = {}
    for y in (Y2, Y3):
        lab = sl[y].notna()
        for name, col in (("group_mean_amt", sl["_gmean"]), ("within_group_resid", sl["_gresid"])):
            res = signed_oof_auroc(sl[y], col, sl["fold"], lab)
            store[(y, name)] = res
            rows.append(
                {
                    "y": y,
                    "feature": name,
                    "n": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                    "sign": res["train_sign"] if not res["low_power"] else "—",
                }
            )
    y2_g = store[(Y2, "group_mean_amt")]["cv"]
    y2_r = store[(Y2, "within_group_resid")]["cv"]
    holding = bool(np.isfinite(y2_g) and y2_g >= 0.56 and (not np.isfinite(y2_r) or y2_r < 0.55))
    prose = (
        f"Y2 group-mean amount-uncat {_f(y2_g)} vs within-group residual {_f(y2_r)}. "
        f"{'Holding dummy — style is shared inside the group (do not edit sibling_h).' if holding else 'Within-group residual still has skill; not only a holding dummy.'}"
    )
    print(prose)
    return {
        "rows": rows,
        "y2_g": y2_g,
        "y2_r": y2_r,
        "holding": holding,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 30 — uncat vs mapped row flags (do not edit clean_flags)
# ---------------------------------------------------------------------------
def pass30_flags(con) -> dict:
    hold = load_holdout()
    raw = con.execute(
        """
        SELECT
          CAST(t.company_id AS VARCHAR) AS company_id,
          CASE WHEN t.category = 'uncategorized' THEN 1 ELSE 0 END AS is_uncat,
          t.status,
          t.accounting_status,
          t.is_dup,
          t.is_extreme,
          t.product_known,
          (t.description IS NULL OR length(trim(t.description)) = 0) AS empty_desc,
          (t.counterparty_id IS NULL OR length(trim(CAST(t.counterparty_id AS VARCHAR))) = 0) AS no_cp
        FROM transactions t
        WHERE t."date" IS NOT NULL
          AND t."date" >= DATE '2024-09-01'
          AND t."date" < DATE '2026-09-01'
        """
    ).df()
    raw["company_id"] = raw["company_id"].astype(str)
    raw = raw.loc[~raw["company_id"].isin(hold)].copy()
    assert_no_holdout(raw["company_id"])
    rows = []
    for flag, lab in (
        (raw["is_uncat"] == 1, "uncat"),
        (raw["is_uncat"] == 0, "mapped"),
    ):
        g = raw.loc[flag]
        rows.append(
            {
                "kind": lab,
                "n": f"{len(g):,}",
                "pending": _pp(float((g["status"] == "pending").mean()) if len(g) else float("nan")),
                "booked": _pp(float((g["status"] == "booked").mean()) if len(g) else float("nan")),
                "is_dup": _pp(float(g["is_dup"].mean()) if len(g) else float("nan")),
                "is_extreme": _pp(float(g["is_extreme"].mean()) if len(g) else float("nan")),
                "product_known": _pp(float(g["product_known"].mean()) if len(g) else float("nan")),
                "empty_desc": _pp(float(g["empty_desc"].mean()) if len(g) else float("nan")),
                "no_cp": _pp(float(g["no_cp"].mean()) if len(g) else float("nan")),
            }
        )
    acc = (
        raw.groupby(["is_uncat", "accounting_status"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .head(8)
    )
    prose = (
        f"Uncat vs mapped row flags (train txs). "
        f"If booked and described, opacity is a category-style hole, not DQ."
    )
    print(prose)
    return {
        "rows": rows,
        "acc": acc.to_dict("records"),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 31 — value_date lag (settlement delay?)
# ---------------------------------------------------------------------------
def pass31_value_lag(con) -> dict:
    hold = load_holdout()
    raw = con.execute(
        """
        SELECT
          CAST(t.company_id AS VARCHAR) AS company_id,
          CASE WHEN t.category = 'uncategorized' THEN 1 ELSE 0 END AS is_uncat,
          date_diff('day', t."date", t.value_date) AS lag_days
        FROM transactions t
        WHERE t."date" IS NOT NULL
          AND t.value_date IS NOT NULL
          AND t."date" >= DATE '2024-09-01'
          AND t."date" < DATE '2026-09-01'
        """
    ).df()
    raw["company_id"] = raw["company_id"].astype(str)
    raw = raw.loc[~raw["company_id"].isin(hold)].copy()
    assert_no_holdout(raw["company_id"])
    rows = []
    for v, lab in ((1, "uncat"), (0, "mapped")):
        s = pd.to_numeric(raw.loc[raw["is_uncat"] == v, "lag_days"], errors="coerce")
        rows.append(
            {
                "kind": lab,
                "n": f"{int(s.notna().sum()):,}",
                "p50": _f(float(s.median()) if s.notna().any() else float("nan"), 1),
                "mean": _f(float(s.mean()) if s.notna().any() else float("nan"), 2),
                "share_0": _pp(float((s == 0).mean()) if s.notna().any() else float("nan")),
                "share_gt2": _pp(float((s.abs() > 2).mean()) if s.notna().any() else float("nan")),
            }
        )
    u = pd.to_numeric(raw.loc[raw["is_uncat"] == 1, "lag_days"], errors="coerce")
    m = pd.to_numeric(raw.loc[raw["is_uncat"] == 0, "lag_days"], errors="coerce")
    same = bool(
        np.isfinite(u.median())
        and np.isfinite(m.median())
        and abs(float(u.median()) - float(m.median())) < 1
    )
    prose = (
        f"value_date−date p50 uncat {_f(float(u.median()), 1)} vs mapped {_f(float(m.median()), 1)}. "
        f"{'Same settlement lag — uncat is not a value-date delay.' if same else 'Uncat has a different value-date lag.'}"
    )
    print(prose)
    return {"rows": rows, "same": same, "prose": prose}


# ---------------------------------------------------------------------------
# Pass 32 — uncat by banking product type (do not edit Family G)
# ---------------------------------------------------------------------------
def pass32_product(con) -> dict:
    hold = load_holdout()
    raw = con.execute(
        """
        SELECT
          CAST(t.company_id AS VARCHAR) AS company_id,
          COALESCE(p.type, '(unknown)') AS ptype,
          CASE WHEN t.category = 'uncategorized' THEN 1 ELSE 0 END AS is_uncat,
          COUNT(*) AS n
        FROM transactions t
        LEFT JOIN banking_products p ON t.product_id = p.product_id
        WHERE t."date" IS NOT NULL
          AND t."date" >= DATE '2024-09-01'
          AND t."date" < DATE '2026-09-01'
        GROUP BY 1, 2, 3
        """
    ).df()
    raw["company_id"] = raw["company_id"].astype(str)
    raw = raw.loc[~raw["company_id"].isin(hold)].copy()
    assert_no_holdout(raw["company_id"])
    agg = raw.groupby(["ptype", "is_uncat"], as_index=False)["n"].sum()
    tot = agg.groupby("ptype")["n"].sum()
    rows = []
    for ptype, g in agg.groupby("ptype"):
        n_u = int(g.loc[g["is_uncat"] == 1, "n"].sum())
        n_all = int(tot[ptype])
        rows.append(
            {
                "type": ptype,
                "n": f"{n_all:,}",
                "uncat_n": f"{n_u:,}",
                "uncat_share": _pp(_pct(n_u, n_all)),
            }
        )
    rows = sorted(rows, key=lambda r: -int(r["n"].replace(",", "")))
    # card vs checking
    def _share(name):
        rec = next((r for r in rows if r["type"] == name), None)
        if rec is None:
            return float("nan")
        return float(rec["uncat_n"].replace(",", "")) / float(rec["n"].replace(",", ""))

    card = _share("card")
    chk = _share("checking")
    prose = (
        f"Uncat share by product type (train txs). Card {_pp(card)} vs checking {_pp(chk)}. "
        "Do not edit Family G."
    )
    print(prose)
    return {"rows": rows, "card": card, "chk": chk, "prose": prose}


# ---------------------------------------------------------------------------
# Pass 33 — is Y2 0.602 just g_has_card? (do not edit banking_g)
# ---------------------------------------------------------------------------
def pass33_card(tr: pd.DataFrame) -> dict:
    if "g_has_card" not in tr.columns:
        return {"rows": [], "y2_card": float("nan"), "prose": "g_has_card not in store."}
    feats = {
        "g_has_card": tr["g_has_card"],
        "amt_uncat": tr["amt_uncat"],
        "log1p_a_in3": tr["log_in3"],
    }
    if "g_has_checking" in tr.columns:
        feats["g_has_checking"] = tr["g_has_checking"]
    rows = []
    store = {}
    for y in (Y2, Y3):
        lab = tr[y].notna()
        for name, col in feats.items():
            res = signed_oof_auroc(tr[y], col, tr["fold"], lab)
            store[(y, name)] = res
            rows.append(
                {
                    "y": y,
                    "feature": name,
                    "n": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                    "sign": res["train_sign"] if not res["low_power"] else "—",
                }
            )
    y2_card = store[(Y2, "g_has_card")]["cv"]
    y2_amt = store[(Y2, "amt_uncat")]["cv"]
    clone = bool(np.isfinite(y2_card) and np.isfinite(y2_amt) and abs(y2_card - y2_amt) < 0.02)
    rho = spearman(tr["g_has_card"], tr["amt_uncat"])
    prose = (
        f"Y2 `g_has_card` {_f(y2_card)} vs amount-uncat {_f(y2_amt)} "
        f"(Spearman {_f(rho)}). "
        f"{'Card dummy clones uncat.' if clone else 'Uncat is not just has-card (banking_g CLOSE as Y3 X stands).'}"
    )
    print(prose)
    return {
        "rows": rows,
        "y2_card": y2_card,
        "y2_amt": y2_amt,
        "rho": rho,
        "clone": clone,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 34 — uncat Y2 on checking-only companies (no card)
# ---------------------------------------------------------------------------
def pass34_checking(tr: pd.DataFrame) -> dict:
    if "g_has_card" not in tr.columns:
        return {"rows": [], "y2": float("nan"), "prose": "g_has_card missing."}
    card = pd.to_numeric(tr["g_has_card"], errors="coerce")
    no_card = card == 0
    rows = []
    cvs = {}
    for name, m in (("no_card", no_card), ("has_card", card == 1)):
        sl = tr.loc[m]
        res = signed_oof_auroc(sl[Y2], sl["amt_uncat"], sl["fold"], sl[Y2].notna())
        cvs[name] = float("nan") if res["low_power"] else res["cv"]
        rows.append(
            {
                "slice": name,
                "n_cm": int(len(sl)),
                "n_co": int(sl["company_id"].nunique()),
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "Y2_amt_CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                "mean_amt": _f(float(pd.to_numeric(sl["amt_uncat"], errors="coerce").mean())),
            }
        )
    y2 = cvs.get("no_card", float("nan"))
    prose = (
        f"Amount-uncat Y2 on no-card company-months: {_f(y2)} "
        f"(has-card {_f(cvs.get('has_card', float('nan')))}). "
        "If skill survives, the style is on checking books too (card is 48% uncat but 3% of txs)."
    )
    print(prose)
    return {"rows": rows, "y2": y2, "prose": prose}


# ---------------------------------------------------------------------------
# Pass 35 — inflow vs outflow uncat (messy AP vs style on both sides)
# ---------------------------------------------------------------------------
def pass35_in_out(con, tr: pd.DataFrame) -> dict:
    cats = _cat_sql_list()
    raw = con.execute(
        f"""
        SELECT
          CAST(t.company_id AS VARCHAR) AS company_id,
          CAST(date_trunc('month', t."date") AS DATE) AS month,
          SUM(CASE WHEN t.amount > 0 THEN t.amount ELSE 0 END) AS in_all,
          SUM(CASE WHEN t.amount < 0 THEN -t.amount ELSE 0 END) AS out_all,
          SUM(CASE WHEN t.amount > 0 AND (
                    t.category = 'uncategorized'
                 OR t.category IS NULL
                 OR t.category NOT IN ({cats})
               ) THEN t.amount ELSE 0 END) AS in_uncat,
          SUM(CASE WHEN t.amount < 0 AND (
                    t.category = 'uncategorized'
                 OR t.category IS NULL
                 OR t.category NOT IN ({cats})
               ) THEN -t.amount ELSE 0 END) AS out_uncat
        FROM transactions t
        WHERE t."date" IS NOT NULL
          AND t."date" >= DATE '2024-09-01'
          AND t."date" < DATE '2026-09-01'
        GROUP BY 1, 2
        """
    ).df()
    raw["company_id"] = raw["company_id"].astype(str)
    raw["period"] = pd.to_datetime(raw["month"])
    in_all = pd.to_numeric(raw["in_all"], errors="coerce")
    out_all = pd.to_numeric(raw["out_all"], errors="coerce")
    raw["amt_uncat_in"] = np.where(in_all > 0, raw["in_uncat"] / in_all, np.nan)
    raw["amt_uncat_out"] = np.where(out_all > 0, raw["out_uncat"] / out_all, np.nan)
    sl = tr.merge(
        raw[["company_id", "period", "amt_uncat_in", "amt_uncat_out"]],
        on=["company_id", "period"],
        how="left",
    )
    rows = []
    cvs = {}
    for feat, col in (("inflow", "amt_uncat_in"), ("outflow", "amt_uncat_out"), ("amount", "amt_uncat")):
        for yname, ycol in (("Y2", Y2), ("Y3", Y3)):
            res = signed_oof_auroc(sl[ycol], sl[col], sl["fold"], sl[ycol].notna())
            key = f"{feat}_{yname}"
            cvs[key] = float("nan") if res["low_power"] else res["cv"]
            rows.append(
                {
                    "feat": feat,
                    "Y": yname,
                    "mean": _f(float(pd.to_numeric(sl[col], errors="coerce").mean())),
                    "n": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                }
            )
    y2_in = cvs.get("inflow_Y2", float("nan"))
    y2_out = cvs.get("outflow_Y2", float("nan"))
    both = (
        np.isfinite(y2_in)
        and np.isfinite(y2_out)
        and abs(y2_in - y2_out) < 0.04
        and min(y2_in, y2_out) >= 0.56
    )
    prose = (
        f"Y2 inflow-uncat {_f(y2_in)} vs outflow-uncat {_f(y2_out)} "
        f"(pooled amount {_f(cvs.get('amount_Y2', float('nan')))}). "
        + (
            "Both sides — style on the whole book, not a messy-AP hole."
            if both
            else "Sides differ — one direction may be the story."
        )
    )
    print(prose)
    return {
        "rows": rows,
        "y2_in": y2_in,
        "y2_out": y2_out,
        "both": both,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 36 — amount-uncat residual after a_n_tx (activity)
# ---------------------------------------------------------------------------
def pass36_ntx_resid(tr: pd.DataFrame) -> dict:
    amt = pd.to_numeric(tr["amt_uncat"], errors="coerce")
    ntx = np.log1p(pd.to_numeric(tr["a_n_tx"], errors="coerce").clip(lower=0))
    d = pd.DataFrame({"amt": amt, "ntx": ntx, "y2": tr[Y2], "y3": tr[Y3], "fold": tr["fold"]})
    ok = d["amt"].notna() & d["ntx"].notna()
    resid = pd.Series(np.nan, index=d.index, dtype=float)
    if int(ok.sum()) >= 20:
        x = d.loc[ok, "ntx"].to_numpy(dtype=float)
        y = d.loc[ok, "amt"].to_numpy(dtype=float)
        xm = float(x.mean())
        ym = float(y.mean())
        den = float(np.sum((x - xm) ** 2))
        slope = float(np.sum((x - xm) * (y - ym)) / den) if den > 0 else 0.0
        intercept = ym - slope * xm
        resid.loc[ok] = y - (intercept + slope * x)
    else:
        slope = float("nan")
        intercept = float("nan")
    rows = []
    cvs = {}
    for yname, ycol in (("Y2", "y2"), ("Y3", "y3")):
        res = signed_oof_auroc(d[ycol], resid, d["fold"], d[ycol].notna())
        cvs[yname] = float("nan") if res["low_power"] else res["cv"]
        rows.append(
            {
                "Y": yname,
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "resid_CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
            }
        )
    prose = (
        f"Amount-uncat residual after log1p(a_n_tx) (slope {_f(slope, 3)}): "
        f"Y2 {_f(cvs.get('Y2', float('nan')))} Y3 {_f(cvs.get('Y3', float('nan')))}. "
        "If Y2 stays ~0.60, activity does not explain the 0.602."
    )
    print(prose)
    return {
        "rows": rows,
        "y2": cvs.get("Y2", float("nan")),
        "y3": cvs.get("Y3", float("nan")),
        "slope": slope,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 37 — is n_tx residual just size again?
# ---------------------------------------------------------------------------
def pass37_resid_size(tr: pd.DataFrame) -> dict:
    amt = pd.to_numeric(tr["amt_uncat"], errors="coerce")
    ntx = np.log1p(pd.to_numeric(tr["a_n_tx"], errors="coerce").clip(lower=0))
    size = pd.to_numeric(tr["log_in3"], errors="coerce")
    d = pd.DataFrame(
        {"amt": amt, "ntx": ntx, "size": size, "y3": tr[Y3], "y2": tr[Y2], "fold": tr["fold"]}
    )
    ok = d["amt"].notna() & d["ntx"].notna()
    resid = pd.Series(np.nan, index=d.index, dtype=float)
    if int(ok.sum()) >= 20:
        x = d.loc[ok, "ntx"].to_numpy(dtype=float)
        y = d.loc[ok, "amt"].to_numpy(dtype=float)
        xm, ym = float(x.mean()), float(y.mean())
        den = float(np.sum((x - xm) ** 2))
        slope = float(np.sum((x - xm) * (y - ym)) / den) if den > 0 else 0.0
        resid.loc[ok] = y - ((ym - slope * xm) + slope * x)
    rho_size = spearman(resid, d["size"])
    rho_amt = spearman(resid, d["amt"])
    res_y3 = signed_oof_auroc(d["y3"], resid, d["fold"], d["y3"].notna())
    res_sz = signed_oof_auroc(d["y3"], d["size"], d["fold"], d["y3"].notna())
    clone = bool(np.isfinite(rho_size) and abs(rho_size) >= 0.40)
    y3_r = float("nan") if res_y3["low_power"] else res_y3["cv"]
    y3_s = float("nan") if res_sz["low_power"] else res_sz["cv"]
    rows = [
        {
            "item": "resid↔size Spearman",
            "value": _f(rho_size),
        },
        {
            "item": "resid↔amt Spearman",
            "value": _f(rho_amt),
        },
        {
            "item": "Y3 resid CV",
            "value": "LOW_POWER" if res_y3["low_power"] else _f(y3_r),
        },
        {
            "item": "Y3 size CV",
            "value": "LOW_POWER" if res_sz["low_power"] else _f(y3_s),
        },
        {
            "item": "Y3 resid train sign",
            "value": str(res_y3["train_sign"]),
        },
        {
            "item": "Y3 size train sign",
            "value": str(res_sz["train_sign"]),
        },
    ]
    prose = (
        f"n_tx-residual ↔ size Spearman { _f(rho_size) } (↔ amt {_f(rho_amt)}). "
        f"Y3 resid {_f(y3_r)} sign {res_y3['train_sign']} vs size {_f(y3_s)} "
        f"sign {res_sz['train_sign']}. "
        + (
            "Residual clones size — PARK as X, not a new Y3 why."
            if clone
            else "Residual is not a size clone; still PARK (raw amount 0.530)."
        )
    )
    print(prose)
    return {
        "rows": rows,
        "rho_size": rho_size,
        "rho_amt": rho_amt,
        "y3_r": y3_r,
        "y3_s": y3_s,
        "clone": clone,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 38 — amount-uncat Y2 inside size terciles
# ---------------------------------------------------------------------------
def pass38_size_terc(tr: pd.DataFrame) -> dict:
    size = pd.to_numeric(tr["log_in3"], errors="coerce")
    amt = pd.to_numeric(tr["amt_uncat"], errors="coerce")
    sl = tr.loc[size.notna() & amt.notna()].copy()
    sl["_s"] = size.loc[sl.index]
    try:
        sl["_t"] = pd.qcut(sl["_s"], 3, labels=False, duplicates="drop")
    except ValueError:
        sl["_t"] = 0
    rows = []
    cvs = []
    for t, g in sl.groupby("_t", sort=True):
        res = signed_oof_auroc(g[Y2], g["amt_uncat"], g["fold"], g[Y2].notna())
        cv = float("nan") if res["low_power"] else res["cv"]
        cvs.append(cv)
        rows.append(
            {
                "size_tercile": int(t) + 1,
                "p50_log_in3": _f(float(g["_s"].median()), 2),
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "Y2_amt_CV": "LOW_POWER" if res["low_power"] else _f(cv),
                "mean_amt": _f(float(pd.to_numeric(g["amt_uncat"], errors="coerce").mean())),
            }
        )
    survives = bool(sum(1 for c in cvs if np.isfinite(c) and c >= 0.56) >= 2)
    prose = (
        "Amount-uncat Y2 inside log1p(a_in3) terciles. "
        + (
            "Skill in more than one size band — not only a large-book dummy."
            if survives
            else "Skill collapses inside size bands — closer to a size dummy."
        )
    )
    print(prose)
    return {"rows": rows, "survives": survives, "prose": prose}


# ---------------------------------------------------------------------------
# Pass 39 — Y2 on mature books (months 13+)
# ---------------------------------------------------------------------------
def pass39_mature(tr: pd.DataFrame) -> dict:
    sl = tr.sort_values(["company_id", "period"]).copy()
    sl["so_far"] = sl.groupby("company_id", sort=False).cumcount() + 1
    late = sl[sl["so_far"] >= 13]
    early = sl[sl["so_far"] <= 3]
    rows = []
    cvs = {}
    for name, g in (("months_1-3", early), ("months_13+", late), ("all", sl)):
        res = signed_oof_auroc(g[Y2], g["amt_uncat"], g["fold"], g[Y2].notna())
        cvs[name] = float("nan") if res["low_power"] else res["cv"]
        rows.append(
            {
                "slice": name,
                "n_cm": int(len(g)),
                "n_co": int(g["company_id"].nunique()),
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "Y2_amt_CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
            }
        )
    late_cv = cvs.get("months_13+", float("nan"))
    prose = (
        f"Amount-uncat Y2 on months 13+ {_f(late_cv)} "
        f"(months 1–3 {_f(cvs.get('months_1-3', float('nan')))}). "
        + (
            "Survives on mature books — not only an onboarding hole."
            if np.isfinite(late_cv) and late_cv >= 0.56
            else "Dies on mature books — onboarding mess."
        )
    )
    print(prose)
    return {"rows": rows, "late": late_cv, "prose": prose}


# ---------------------------------------------------------------------------
# Pass 40 — Y2 amount-uncat on invoiced 744 vs dark 470
# ---------------------------------------------------------------------------
def pass40_dark_y2(tr: pd.DataFrame) -> dict:
    if "ever_erp" not in tr.columns:
        return {"rows": [], "y2_erp": float("nan"), "y2_dark": float("nan"), "prose": "ever_erp missing."}
    rows = []
    cvs = {}
    for name, m in (("invoiced_744", tr["ever_erp"].eq(True)), ("dark_470", tr["ever_erp"].eq(False))):
        sl = tr.loc[m]
        res = signed_oof_auroc(sl[Y2], sl["amt_uncat"], sl["fold"], sl[Y2].notna())
        cvs[name] = float("nan") if res["low_power"] else res["cv"]
        rows.append(
            {
                "slice": name,
                "n_cm": int(len(sl)),
                "n_co": int(sl["company_id"].nunique()),
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "Y2_amt_CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                "mean_amt": _f(float(pd.to_numeric(sl["amt_uncat"], errors="coerce").mean())),
            }
        )
    y2_erp = cvs.get("invoiced_744", float("nan"))
    y2_dark = cvs.get("dark_470", float("nan"))
    both = (
        np.isfinite(y2_erp)
        and np.isfinite(y2_dark)
        and min(y2_erp, y2_dark) >= 0.55
    )
    prose = (
        f"Y2 amount-uncat invoiced {_f(y2_erp)} vs dark {_f(y2_dark)}. "
        + (
            "Skill on both sides — uncat ≠ no ERP, and the style is not a dark-only dummy."
            if both
            else "Skill is concentrated on one ERP side."
        )
    )
    print(prose)
    return {
        "rows": rows,
        "y2_erp": y2_erp,
        "y2_dark": y2_dark,
        "both": both,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 41 — always-uncat counterparties (mapping hole vs random miss)
# ---------------------------------------------------------------------------
def pass41_uncat_cp(con) -> dict:
    hold = load_holdout()
    raw = con.execute(
        """
        SELECT
          CAST(t.company_id AS VARCHAR) AS company_id,
          CASE
            WHEN t.counterparty_id IS NULL
              OR length(trim(CAST(t.counterparty_id AS VARCHAR))) = 0
            THEN '(missing)'
            ELSE CAST(t.counterparty_id AS VARCHAR)
          END AS cp,
          SUM(CASE WHEN t.category = 'uncategorized' THEN 1 ELSE 0 END) AS n_uncat,
          COUNT(*) AS n_all,
          SUM(CASE WHEN t.category = 'uncategorized' THEN ABS(t.amount) ELSE 0 END) AS abs_uncat
        FROM transactions t
        WHERE t."date" IS NOT NULL
          AND t."date" >= DATE '2024-09-01'
          AND t."date" < DATE '2026-09-01'
        GROUP BY 1, 2
        """
    ).df()
    raw["company_id"] = raw["company_id"].astype(str)
    raw = raw.loc[~raw["company_id"].isin(hold)].copy()
    assert_no_holdout(raw["company_id"])
    raw["rate"] = np.where(raw["n_all"] > 0, raw["n_uncat"] / raw["n_all"], np.nan)
    tot_u = float(raw["n_uncat"].sum())
    tot_a = float(raw["abs_uncat"].sum())
    missing = raw["cp"] == "(missing)"
    named = ~missing
    sticky = named & (raw["rate"] >= 0.80)
    mixed = named & (raw["rate"] > 0.20) & (raw["rate"] < 0.80)
    rare = named & (raw["rate"] > 0) & (raw["rate"] <= 0.20)
    rows = []
    for name, m in (
        ("missing_cp", missing),
        ("named_sticky≥80%", sticky),
        ("named_mixed_20-80", mixed),
        ("named_rare_≤20%", rare),
    ):
        part = raw[m]
        rows.append(
            {
                "cp_kind": name,
                "n_pairs": int(len(part)),
                "uncat_txs": f"{int(part['n_uncat'].sum()):,}",
                "share_n": _pp(_pct(float(part["n_uncat"].sum()), tot_u)),
                "share_|amt|": _pp(_pct(float(part["abs_uncat"].sum()), tot_a)),
            }
        )
    miss_n = _pct(float(raw.loc[missing, "n_uncat"].sum()), tot_u)
    named_u = float(raw.loc[named, "n_uncat"].sum())
    sticky_n = _pct(float(raw.loc[sticky, "n_uncat"].sum()), tot_u)
    sticky_named = _pct(float(raw.loc[sticky, "n_uncat"].sum()), named_u)
    prose = (
        f"Missing-CP uncat {_pp(miss_n)}; named sticky (≥80%) {_pp(sticky_n)} "
        f"of all uncat / {_pp(sticky_named)} of named-CP uncat. "
        "Company ICC still says style — this is a labeling habit across many CPs, "
        "not a few always-uncat vendors (and not a month miss)."
    )
    print(prose)
    return {
        "rows": rows,
        "sticky_n": sticky_n,
        "miss_n": miss_n,
        "sticky_named": sticky_named,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 42 — description length (described but unlabeled)
# ---------------------------------------------------------------------------
def pass42_desc(con) -> dict:
    hold = load_holdout()
    hold_sql = ", ".join("'" + str(c).replace("'", "''") + "'" for c in sorted(hold))
    raw = con.execute(
        f"""
        SELECT
          CASE WHEN t.category = 'uncategorized' THEN 1 ELSE 0 END AS is_uncat,
          COUNT(*) AS n,
          quantile_cont(length(trim(COALESCE(t.description, ''))), 0.10) AS p10,
          quantile_cont(length(trim(COALESCE(t.description, ''))), 0.50) AS p50,
          quantile_cont(length(trim(COALESCE(t.description, ''))), 0.90) AS p90,
          AVG(CASE WHEN length(trim(COALESCE(t.description, ''))) < 8 THEN 1.0 ELSE 0.0 END) AS share_lt8
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
        lab = "uncat" if int(r["is_uncat"]) == 1 else "mapped"
        p50s[lab] = float(r["p50"])
        rows.append(
            {
                "kind": lab,
                "n": f"{int(r['n']):,}",
                "p10": _f(float(r["p10"]), 0),
                "p50": _f(float(r["p50"]), 0),
                "p90": _f(float(r["p90"]), 0),
                "share_lt8": _pp(float(r["share_lt8"])),
            }
        )
    same = bool(
        np.isfinite(p50s.get("uncat", float("nan")))
        and np.isfinite(p50s.get("mapped", float("nan")))
        and abs(p50s["uncat"] - p50s["mapped"]) < 8
    )
    prose = (
        f"Description length p50 uncat {_f(p50s.get('uncat', float('nan')), 0)} vs mapped "
        f"{_f(p50s.get('mapped', float('nan')), 0)}. "
        + (
            "Same length — uncat is labeled-as-uncategorized, not a blank memo."
            if same
            else "Different memo length — opacity may be a description hole."
        )
    )
    print(prose)
    return {
        "rows": rows,
        "p50_u": p50s.get("uncat", float("nan")),
        "p50_m": p50s.get("mapped", float("nan")),
        "same": same,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 43 — Y2 after dropping all-uncat months (is the tail the dummy?)
# ---------------------------------------------------------------------------
def pass43_drop_alluncat(tr: pd.DataFrame) -> dict:
    amt = pd.to_numeric(tr["amt_uncat"], errors="coerce")
    all_u = amt >= 0.95
    some = amt.notna() & (amt < 0.95)
    rows = []
    cvs = {}
    for name, m in (("drop_alluncat", some), ("alluncat_only", all_u), ("all_defined", amt.notna())):
        sl = tr.loc[m]
        res = signed_oof_auroc(sl[Y2], sl["amt_uncat"], sl["fold"], sl[Y2].notna())
        cvs[name] = float("nan") if res["low_power"] else res["cv"]
        rows.append(
            {
                "slice": name,
                "n_cm": int(len(sl)),
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "Y2_amt_CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
            }
        )
    drop = cvs.get("drop_alluncat", float("nan"))
    prose = (
        f"Y2 amount-uncat after dropping amount≥0.95 months: {_f(drop)} "
        f"(all-uncat-only {_f(cvs.get('alluncat_only', float('nan')))}). "
        + (
            "Skill survives without the 100% tail — not only the all-uncat dummy."
            if np.isfinite(drop) and drop >= 0.56
            else "Skill was the all-uncat tail."
        )
    )
    print(prose)
    return {"rows": rows, "drop": drop, "prose": prose}


# ---------------------------------------------------------------------------
# Pass 44 — weekday of uncat vs mapped (batch weekend style?)
# ---------------------------------------------------------------------------
def pass44_weekday(con) -> dict:
    hold = load_holdout()
    hold_sql = ", ".join("'" + str(c).replace("'", "''") + "'" for c in sorted(hold))
    raw = con.execute(
        f"""
        SELECT
          dayofweek(t."date") AS dow,
          SUM(CASE WHEN t.category = 'uncategorized' THEN 1 ELSE 0 END) AS n_uncat,
          COUNT(*) AS n_all
        FROM transactions t
        WHERE t."date" IS NOT NULL
          AND t."date" >= DATE '2024-09-01'
          AND t."date" < DATE '2026-09-01'
          AND CAST(t.company_id AS VARCHAR) NOT IN ({hold_sql})
        GROUP BY 1
        ORDER BY 1
        """
    ).df()
    names = {0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat"}
    rows = []
    rates = []
    for _, r in raw.iterrows():
        dow = int(r["dow"])
        rate = _pct(float(r["n_uncat"]), float(r["n_all"]))
        rates.append(rate)
        rows.append(
            {
                "dow": names.get(dow, str(dow)),
                "n": f"{int(r['n_all']):,}",
                "uncat_n": f"{int(r['n_uncat']):,}",
                "uncat_share": _pp(rate),
            }
        )
    finite = [x for x in rates if np.isfinite(x)]
    spread = float(max(finite) - min(finite)) if finite else float("nan")
    weekend = bool(np.isfinite(spread) and spread >= 0.08)
    prose = (
        f"Weekday uncat-share range {_pp(spread)}. "
        + (
            "Weekend/batch dummy — uncat piles on some days."
            if weekend
            else "No weekday dummy (range <8pp) — not a weekend batch hole."
        )
    )
    print(prose)
    return {"rows": rows, "spread": spread, "weekend": weekend, "prose": prose}


# ---------------------------------------------------------------------------
# Pass 45 — invoiced-only style vs shock (ERP books still a dummy?)
# ---------------------------------------------------------------------------
def pass45_erp_style(tr: pd.DataFrame) -> dict:
    sl = tr.loc[tr["ever_erp"].eq(True)].copy()
    rows = []
    cvs = {}
    for feat, lab in (("amt_co_mean", "style"), ("amt_demean", "shock"), ("amt_uncat", "now")):
        res = signed_oof_auroc(sl[Y2], sl[feat], sl["fold"], sl[Y2].notna())
        cvs[lab] = float("nan") if res["low_power"] else res["cv"]
        rows.append(
            {
                "feat": lab,
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "Y2_CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
            }
        )
    style = cvs.get("style", float("nan"))
    shock = cvs.get("shock", float("nan"))
    carries = bool(np.isfinite(style) and np.isfinite(shock) and style >= shock + 0.03)
    prose = (
        f"Invoiced-only Y2 company-mean {_f(style)} vs demean {_f(shock)} "
        f"(now {_f(cvs.get('now', float('nan')))}). "
        + (
            "ERP books still carry style, not a month change — y2_why should not treat uncat as a why."
            if carries
            else "On invoiced books the month shock is closer to the skill."
        )
    )
    print(prose)
    return {
        "rows": rows,
        "style": style,
        "shock": shock,
        "carries": carries,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 46 — mapped-category richness vs uncat (design note, no CAT_MAP edit)
# ---------------------------------------------------------------------------
def pass46_mapped_rich(con, tr: pd.DataFrame) -> dict:
    cats = _cat_sql_list()
    raw = con.execute(
        f"""
        SELECT
          CAST(t.company_id AS VARCHAR) AS company_id,
          CAST(date_trunc('month', t."date") AS DATE) AS month,
          COUNT(DISTINCT CASE WHEN t.category IN ({cats}) THEN t.category END) AS n_mapped_cats
        FROM transactions t
        WHERE t."date" IS NOT NULL
          AND t."date" >= DATE '2024-09-01'
          AND t."date" < DATE '2026-09-01'
        GROUP BY 1, 2
        """
    ).df()
    raw["company_id"] = raw["company_id"].astype(str)
    raw["period"] = pd.to_datetime(raw["month"])
    sl = tr.merge(raw[["company_id", "period", "n_mapped_cats"]], on=["company_id", "period"], how="left")
    x = pd.to_numeric(sl["a_uncat_share"], errors="coerce")
    ncat = pd.to_numeric(sl["n_mapped_cats"], errors="coerce")
    rho = spearman(x, ncat)
    defined = x.notna() & ncat.notna()
    part = sl.loc[defined].copy()
    try:
        part["_q"] = pd.qcut(part["a_uncat_share"], 5, labels=False, duplicates="drop")
    except ValueError:
        part["_q"] = 0
    rows = []
    for q, g in part.groupby("_q", sort=True):
        rows.append(
            {
                "q": int(q) + 1,
                "n_cm": int(len(g)),
                "p50_uncat": _f(float(pd.to_numeric(g["a_uncat_share"], errors="coerce").median())),
                "p50_mapped_cats": _f(float(pd.to_numeric(g["n_mapped_cats"], errors="coerce").median()), 1),
            }
        )
    prose = (
        f"Mapped CAT_MAP tokens per month vs count-uncat Spearman { _f(rho) }. "
        "If high-uncat months still show many mapped tokens, leftover is a labeling habit "
        "on top of a used map — do not add `uncategorized` to CAT_MAP."
    )
    print(prose)
    return {"rows": rows, "rho": rho, "prose": prose}


def decide(p1, p2, p3, p4, p5, p7, p10=None) -> dict:
    """KEEP Q5 only if amount-uncat beats size ≥0.02 and is not a style dummy."""
    beat = p4["beat_size"]
    style = bool(p5["style_a"] or p5["style_c"])
    y2_beat = (
        p4["amt_y2"] - p4["size_y2"]
        if np.isfinite(p4["amt_y2"]) and np.isfinite(p4["size_y2"])
        else float("nan")
    )
    y2_style = bool(p10 and p10.get("style_carries_y2"))
    noise = bool(
        (not np.isfinite(p4["best_uncat"]) or p4["best_uncat"] < 0.55)
        and all(
            p3["shapes"][f"amt_{y}"].startswith("noise") or p3["shapes"][f"amt_{y}"] == "n/a"
            for y in (Y2, Y3, Y7, Y9)
        )
    )
    # Y3 is the Q5 readability bar (days 0.711). Y2 0.602 is a style identity, not a change.
    keep_q5 = bool(np.isfinite(beat) and beat >= KEEP_DELTA and not style)
    if keep_q5:
        q5 = "KEEP"
        why = (
            f"{p4['best_name']} beats size by {_f(beat, 3)} and is not a style dummy "
            f"(ICC amt {_f(p5['icc']['amt']['icc'])} acf1 {_f(p5['acf']['amt'][1])})."
        )
    elif np.isfinite(y2_beat) and y2_beat >= KEEP_DELTA and style:
        q5 = "CLOSE"
        why = (
            f"Y3 amount {_f(p4['amt_y3'])} loses to size {_f(p4['size_y3'])} (Δ {_f(beat, 3)}). "
            f"Y2 amount {_f(p4['amt_y2'])} beats size {_f(p4['size_y2'])} by {_f(y2_beat, 3)} "
            f"but is a bookkeeping-style dummy (ICC {_f(p5['icc']['amt']['icc'])}, "
            f"acf1 {_f(p5['acf']['amt'][1])}"
            + (f"; company-mean {_f(p10['y2_mean'])} vs shock {_f(p10['y2_shock'])}" if p10 else "")
            + "). Not Q5 readability of a *change*. PARK as X (Y2 0.602 ≥ 0.60)."
        )
    elif p4["size_park"] and (not np.isfinite(beat) or beat < KEEP_DELTA):
        q5 = "CLOSE"
        why = (
            f"Best uncat {_f(p4['best_uncat'])} does not beat size {_f(p4['size_y3'])} "
            f"by ≥{KEEP_DELTA:g}. Chance / noise vs the 0.711 days bar."
        )
    else:
        q5 = "CLOSE"
        why = (
            f"Best uncat {_f(p4['best_uncat'])} vs size {_f(p4['size_y3'])} "
            f"(Δ {_f(beat, 3)}); quintiles {p3['shapes']['amt_' + Y3]}. Noise."
        )
    x_dec = "PARK" if p4["size_park"] or (not np.isfinite(p4["best_uncat"]) or p4["best_uncat"] < 0.60) else "measured"
    if p4["size_park"]:
        x_why = f"size dummy {_f(p4['size_y3'])} ≥ {SIZE_PARK:g}"
    else:
        x_why = f"best uncat {_f(p4['best_uncat'])} vs days {_f(p4['days_y3'])}"
    q6 = "KEEP" if p7["keep_q6"] else "CLOSE"
    return {
        "q5": q5,
        "why": why,
        "x_dec": x_dec,
        "x_why": x_why,
        "q6": q6,
        "style": style,
        "noise": noise,
        "park_y": True,
    }


def write_md(ctx: dict) -> None:
    p1, p2, p3, p4, p5 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"], ctx["p5"]
    p6, p7, p8, p9, d = ctx["p6"], ctx["p7"], ctx["p8"], ctx["p9"], ctx["decision"]
    p10, p11, p12 = ctx["p10"], ctx["p11"], ctx["p12"]
    p13, p14, p15, p16 = ctx["p13"], ctx["p14"], ctx["p15"], ctx["p16"]
    p17, p18, p19, p20 = ctx["p17"], ctx["p18"], ctx["p19"], ctx["p20"]
    p21, p22 = ctx["p21"], ctx["p22"]
    p23, p24, p25 = ctx["p23"], ctx["p24"], ctx["p25"]
    p26, p27 = ctx["p26"], ctx["p27"]
    p28 = ctx["p28"]
    p29, p30 = ctx["p29"], ctx["p30"]
    p31 = ctx["p31"]
    p32 = ctx["p32"]
    p33 = ctx["p33"]
    p34 = ctx["p34"]
    p35 = ctx["p35"]
    p36 = ctx["p36"]
    p37 = ctx["p37"]
    p38 = ctx["p38"]
    p39 = ctx["p39"]
    p40 = ctx["p40"]
    p41 = ctx["p41"]
    p42 = ctx["p42"]
    p43 = ctx["p43"]
    p44 = ctx["p44"]
    p45 = ctx["p45"]
    p46 = ctx["p46"]

    def _cat_rows(recs, by: str) -> list[dict]:
        out = []
        for r in recs:
            out.append(
                {
                    "category": r["category"],
                    "n": f"{int(r['n']):,}",
                    "|amt|": f"{float(r['abs_amt']):,.0f}",
                    "n_co": f"{int(r['n_co']):,}",
                    "share_n": _pp(_pct(r["n"], p2["tot_n"])),
                    "share_|amt|": _pp(_pct(r["abs_amt"], p2["tot_a"])),
                }
            )
        return out

    def _q_rows(rows: list[dict]) -> list[dict]:
        out = []
        for r in rows:
            out.append(
                {
                    "q": r["q"],
                    "n_cm": f"{r['n_cm']:,}",
                    "p50": _f(r["p50"]),
                    "Y2": _pp(r[f"{Y2}_rate"]),
                    "Y3": _pp(r[f"{Y3}_rate"]),
                    "Y7": _pp(r[f"{Y7}_rate"]),
                    "Y9": _pp(r[f"{Y9}_rate"]),
                }
            )
        return out

    lines = [
        "# Q5 uncat readability — count vs amount",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. No `build_targets`. Do not invent a merged Y from uncat. "
        "Family M CLOSED — amount-uncat is in-memory only.",
        "",
        "`a_uncat_share` = share of txs with category `uncategorized` OR not in CAT_MAP "
        "(count). Amount-uncat = `sum(|amt| | same rule) / sum(|amt|)` computed here, not merged. "
        "Feature report: keep-list, cov 95.8%, size ρ −0.023, acf1 0.27, ICC 0.99 BETWEEN.",
        "",
        "## Headline",
        "",
        ctx["headline"],
        "",
        "## Brief questions",
        "",
        "| # | question | what this cut says |",
        "| --- | --- | --- |",
        "| 1 | Who is healthy? | Uncat is not a health Y. PARK. |",
        "| 2 | Who is improving? | Not this share. |",
        "| 3 | Who is turning? | A month-shock of uncategorized euros would be Q3; a style dummy is not. |",
        "| 4 | Dip vs fall? | Not this table. |",
        f"| 5 | Why did it change? | **{d['q5']}** — {d['why']} |",
        f"| 6 | Months earlier? | lag1 {d['q6']} (count now {_f(p7['now_c'])} / lag1 {_f(p7['lag_c'])}; "
        f"amount now {_f(p7['now_a'])} / lag1 {_f(p7['lag_a'])}). Honest 1-month only. |",
        "",
        "## PARK / CLOSE / KEEP",
        "",
        "| object | decision | why |",
        "| --- | --- | --- |",
        f"| amount-uncat as Q5 why | **{d['q5']}** | {d['why']} |",
        f"| `a_uncat_share` as Q5 why | **{d['q5']}** | count ICC {_f(p5['icc']['count']['icc'])} acf1 {_f(p5['acf']['count'][1])}; "
        f"{'style dummy' if p5['style_c'] else 'not style'} |",
        "| uncat as a health Y | **PARK** | do not invent a merged Y from opacity |",
        f"| uncat as Y3 X | **{d['x_dec']}** | {d['x_why']} |",
        f"| Q6 lag1 | **{d['q6']}** | {p7['prose']} |",
        f"| leftover CAT_MAP add | **{'CLOSE (only `uncategorized`)' if p2['only_uncat'] else 'design note'}** | {p2['prose']} |",
        "| Family M merge | **CLOSED** | do not merge mix; amount-uncat stays in-memory |",
        f"| Y2 amount-uncat 0.602 | **PARK as X** | beats size {_f(p4['amt_y2'] - p4['size_y2'], 3)} but style "
        f"(mean {_f(p10['y2_mean'])} / shock {_f(p10['y2_shock'])}); y2_why is in flight |",
        f"| holding-style footnote | **measured** | group ICC of company-median {_f(p25['icc_c'])}; "
        "do not edit sibling_h |",
        "",
        "## 1. Amount-mass vs count-share",
        "",
        p1["prose"],
        "",
        "| item | count `a_uncat_share` | amount-uncat (memory) |",
        "| --- | ---: | ---: |",
        f"| coverage | {_pp(p1['cov_c'])} | {_pp(p1['cov_a'])} |",
        f"| mean | {_f(p1['mean_c'])} | {_f(p1['mean_a'])} |",
        f"| p50 | {_f(p1['p50_c'])} | {_f(p1['p50_a'])} |",
        f"| p90 | {_f(p1['p90_c'])} | {_f(p1['p90_a'])} |",
        f"| size ρ vs log1p(a_in3) | {_f(p1['rho_size_c'])} | {_f(p1['rho_size_a'])} |",
        f"| pooled ticket / euro share | {_pp(p1['tx_share'])} | {_pp(p1['euro_share'])} |",
        f"| Spearman count↔amount | {_f(p1['rho_ca'])} | Pearson {_f(p1['pear_ca'])} |",
        f"| store vs recomputed max\\|Δ\\| | {_f(p1['max_abs'], 6)} | — |",
        f"| months amount−count ≥0.20 | {p1['n_hid']:,} | (count hides euros) |",
        f"| months count−amount ≥0.20 | {p1['n_small']:,} | (many small uncat tickets) |",
        f"| leftover txs / € (not the token) | {p1['leftover_n']:,.0f} | {p1['leftover_amt']:,.0f} |",
        f"| size ρ vs log1p(\\|a_op_in\\|) | {_f(p1['rho_opin_c'])} | {_f(p1['rho_opin_a'])} |",
        "",
        "Amount-uncat is **not** merged. `m_uncat_n_share` stays parked as a rewrite of `a_uncat_share`.",
        "",
        "## 2. Leftover categories (outside CAT_MAP)",
        "",
        p2["prose"],
        "",
        "Top 15 leftover tokens by **count** (train):",
        "",
        _md_table(_cat_rows(p2["top_n"], "n")),
        "",
        "Top 15 leftover tokens by **|amount|** (train):",
        "",
        _md_table(_cat_rows(p2["top_a"], "a")),
        "",
        "### CAT_MAP design note (do not edit CAT_MAP)",
        "",
    ]
    if p2["only_uncat"]:
        lines.append(
            "No frequent leftover besides the `uncategorized` token. "
            "Do not add `uncategorized` to CAT_MAP — that would zero the readability hole. "
            "Never-seen mapped keys: "
            + (", ".join(f"`{k}`" for k in p2["never_seen"]) if p2["never_seen"] else "none")
            + "."
        )
    elif p2["add_note"]:
        lines.append("Frequent leftovers (≥5% of leftover count or euros). Guess only — CAT_MAP frozen:")
        lines.append("")
        lines.append(
            _md_table(
                [
                    {
                        "category": r["category"],
                        "n": f"{r['n']:,}",
                        "|amt|": f"{r['abs_amt']:,.0f}",
                        "share_n": _pp(r["share_n"]),
                        "share_|amt|": _pp(r["share_a"]),
                        "guess": r["guess"],
                    }
                    for r in p2["add_note"]
                ]
            )
        )
    else:
        lines.append(
            "Leftover tokens exist but none reach the 5% leftover-mass bar. Do not expand CAT_MAP tonight."
        )
    lines += [
        "",
        "## 3. Quintiles vs Y2 / Y3 / Y7 / Y9 base rates",
        "",
        p3["prose"],
        "",
        "Cuts from **train** company-months with a finite feature. Holdout never enters a cut.",
        "",
        "### `a_uncat_share` (count)",
        "",
        _md_table(_q_rows(p3["count_rows"])),
        "",
        "### amount-uncat (in memory)",
        "",
        _md_table(_q_rows(p3["amt_rows"])),
        "",
        f"Plot: `{OUT_PNG.name}`." if ctx["png_ok"] else "Plot: skipped (no matplotlib).",
        "",
        "## 4. Single-feature train group-fold AUROC",
        "",
        f"Y2 n={p4['rates'][Y2]['n']:,} base {_pp(p4['rates'][Y2]['rate'])}; "
        f"Y3 n={p4['rates'][Y3]['n']:,} base {_pp(p4['rates'][Y3]['rate'])}; "
        f"Y7 n={p4['rates'][Y7]['n']:,} base {_pp(p4['rates'][Y7]['rate'])}; "
        f"Y9 n={p4['rates'][Y9]['n']:,} base {_pp(p4['rates'][Y9]['rate'])}. "
        "Sign from the train side of each fold. Seed 20260918. "
        f"Night quote: `c_n_days_with_tx` 0.711 on Y3 (replica {_f(p4['days_y3'])}).",
        "",
        p4["prose"],
        "",
        _md_table(p4["rows"]),
        "",
        f"KEEP-as-Q5 rule: amount-uncat beats size by ≥{KEEP_DELTA:g} **and** is not a style dummy "
        f"(ICC ≥ {ICC_STYLE:g} and month acf1 < {ACF_STYLE:g}). "
        f"Size dummy ≥{SIZE_PARK:g} → PARK as X.",
        "",
        "## 5. Persistence — style vs month shock",
        "",
        p5["prose"],
        "",
        "| item | count | amount |",
        "| --- | ---: | ---: |",
        f"| acf1 / acf3 / acf6 | {_f(p5['acf']['count'][1])} / {_f(p5['acf']['count'][3])} / {_f(p5['acf']['count'][6])} | "
        f"{_f(p5['acf']['amt'][1])} / {_f(p5['acf']['amt'][3])} / {_f(p5['acf']['amt'][6])} |",
        f"| ICC | {_f(p5['icc']['count']['icc'])} | {_f(p5['icc']['amt']['icc'])} |",
        f"| company-median p50 | {_f(p5['med_c_p50'])} | {_f(p5['med_a_p50'])} |",
        f"| always-messy (med≥0.40, sd≤0.15) | {p5['always_messy']:,} | {p5['always_messy_a']:,} |",
        f"| always-clean (med≤0.05, sd≤0.10) | {p5['always_clean']:,} | — |",
        f"| shock (med≤0.15, max≥0.50) | {p5['shock_co']:,} | {p5['shock_co_a']:,} |",
        f"| style dummy? | {'YES' if p5['style_c'] else 'NO'} | {'YES' if p5['style_a'] else 'NO'} |",
        "",
        "Style = high company ICC + modest month acf: the firm is *always* messy, not a month that changed. "
        "That answers “who keeps a dirty book”, not Q5 “why did it change”.",
        "",
        "## 6. Dark 470 vs 744 — uncat ≠ no ERP",
        "",
        p6["prose"],
        "",
        "Company-level (mean of each company's uncat rate):",
        "",
        _md_table(p6["rows"]),
        "",
        "Company-month:",
        "",
        _md_table(
            [
                {
                    "group": r["group"],
                    "n_cm": f"{r['n_cm']:,}",
                    "n_co": f"{r['n_co']:,}",
                    "count": _pp(r["count"]),
                    "amount": _pp(r["amt"]),
                }
                for r in p6["cm"]
            ]
        ),
        "",
        "## 7. Q6 — lag1 (honest 1-month only)",
        "",
        p7["prose"],
        "",
        _md_table(p7["rows"]),
        "",
        "## 8. Uncat months vs `m_*` missingness (in memory)",
        "",
        p8["prose"],
        "",
        "Quintiles of `a_uncat_share` vs definition rates of mix / invoice / match columns. "
        "`m_*` is not in parquet (Family M CLOSED). `m_mix_ok` = month has |amount|>0; "
        "`m_coll_vs_pay` needs collection+payment euros.",
        "",
        _md_table(p8["rows"]),
        "",
        "## 9. Holdout coverage only (no AUROC)",
        "",
        p9["prose"],
        "",
        _md_table(p9["rows"]),
        "",
        "## 10. Company-mean (style) vs demeaned (shock)",
        "",
        p10["prose"],
        "",
        _md_table(p10["rows"]),
        "",
        "If company-mean carries the AUROC and the demeaned month shock is chance, "
        "uncat answers “who keeps a dirty book”, not Q5 “why did this month change”.",
        "",
        "## 11. Y2 0.602 vs `a_n_tx` (activity)",
        "",
        p11["prose"],
        "",
        "Within train terciles of `a_n_tx`, amount-uncat AUROC on Y2:",
        "",
        _md_table(p11["fold_rows"]),
        "",
        "Y2 rate by n_tx tercile × amount-uncat quartile (train cuts):",
        "",
        _md_table(p11["rows"]),
        "",
        "## 12. Zero pile — why qcut made 4 bins",
        "",
        p12["prose"],
        "",
        _md_table(
            [
                {
                    "col": r["col"],
                    "band": r["band"],
                    "n_cm": f"{r['n_cm']:,}",
                    "Y2": _pp(r[f"{Y2}_rate"]),
                    "Y3": _pp(r[f"{Y3}_rate"]),
                    "Y7": _pp(r[f"{Y7}_rate"]),
                    "Y9": _pp(r[f"{Y9}_rate"]),
                }
                for r in p12["bands"]
            ]
        ),
        "",
        "## 13. Hide-euro months (amount − count ≥ 0.20)",
        "",
        p13["prose"],
        "",
        _md_table(p13["rows"]),
        "",
        "Count does hide some large uncategorized euros (1.4k months), but the hide dummy "
        "is not a Y3 signal. Do not rewrite `a_uncat_share` to amount.",
        "",
        "## 14. `cash_settlements` alias (do not edit CAT_MAP)",
        "",
        p14["prose"],
        "",
        "## 15. Dark 470 vs 744 inside size terciles",
        "",
        p15["prose"],
        "",
        _md_table(p15["rows"]),
        "",
        "Uncat ≠ no ERP: dark books are still mostly categorized. They are messier, "
        "and the gap is checked inside size terciles so it is not just smaller firms.",
        "",
        "## 16. Company-median amount-uncat vs ever-Y2",
        "",
        p16["prose"],
        "",
        _md_table(p16["rows"]),
        "",
        "## 17. Always-messy vs always-clean companies",
        "",
        p17["prose"],
        "",
        _md_table(p17["rows"]),
        "",
        "## 18. Calendar of uncat (seasonal dummy?)",
        "",
        p18["prose"],
        "",
        _md_table(p18["rows"]),
        "",
        "## 19. Uncategorized euro sign (in vs out)",
        "",
        p19["prose"],
        "",
        "Do not invent an uncat-inflow Y. Sign is a readability footnote.",
        "",
        "## 20. Y3 amount-uncat after `c_n_days_with_tx` terciles",
        "",
        p20["prose"],
        "",
        _md_table(p20["rows"]),
        "",
        "## 21. Months-on-book — onboarding mess?",
        "",
        p21["prose"],
        "",
        _md_table(p21["rows"]),
        "",
        "## 22. Always-messy Y3 label coverage (stressed density)",
        "",
        p22["prose"],
        "",
        _md_table(p22["rows"]),
        "",
        "## 23. Within-company early vs late (selection vs cleaning)",
        "",
        p23["prose"],
        "",
        "| item | value |",
        "| --- | ---: |",
        f"| long companies (≥13m) | {p23['n']:,} |",
        f"| short-book companies | {p23['n_short']:,} |",
        f"| late−early mean Δ | {_f(p23['mean_d'])} |",
        f"| share cleaner >5pp | {_pp(p23['p_clean'])} |",
        f"| share worse >5pp | {_pp(p23['p_worse'])} |",
        f"| early↔late Spearman | {_f(p23['rho'])} |",
        f"| short-book count-uncat | {_pp(p23['short_c'])} |",
        f"| long-book count-uncat | {_pp(p23['long_c'])} |",
        "",
        "## 24. All-uncat months (amount ≥ 0.95)",
        "",
        p24["prose"],
        "",
        _md_table(p24["rows"]),
        "",
        "## 25. Group ICC of company-median uncat",
        "",
        p25["prose"],
        "",
        "## 26. All-dark 360 vs mixed-dark 110 (uncat ≠ no ERP)",
        "",
        p26["prose"],
        "",
        _md_table(p26["rows"]),
        "",
        "## 27. Uncat vs mapped ticket size",
        "",
        p27["prose"],
        "",
        _md_table(
            [
                {
                    "kind": r["kind"],
                    "n": f"{int(r['n']):,}",
                    "|amt|": f"{float(r['abs_amt']):,.0f}",
                    "pooled mean/tx": f"{float(r['abs_amt']) / float(r['n']):,.0f}",
                    "co-median of mean": f"{float(r['mean_abs']):,.0f}",
                    "co-median of p50": f"{float(r['p50_abs']):,.0f}",
                }
                for r in p27["rows"]
            ]
        ),
        "",
        "## 28. Pending vs uncat",
        "",
        p28["prose"],
        "",
        "Feature report: `a_pending_share` is near-zero variance. If it does not clone uncat, "
        "opacity is a booked-but-uncategorized style, not a settlement lag.",
        "",
        "## 29. Holding mean vs within-group residual",
        "",
        p29["prose"],
        "",
        _md_table(p29["rows"]),
        "",
        "## 30. Uncat vs mapped row flags (do not edit clean_flags)",
        "",
        p30["prose"],
        "",
        _md_table(p30["rows"]),
        "",
        "Accounting status (top):",
        "",
        _md_table(
            [
                {
                    "uncat": int(r["is_uncat"]),
                    "accounting_status": r["accounting_status"],
                    "n": f"{int(r['n']):,}",
                }
                for r in p30["acc"]
            ]
        ),
        "",
        "## 31. value_date lag (settlement delay?)",
        "",
        p31["prose"],
        "",
        _md_table(p31["rows"]),
        "",
        "## 32. Uncat by banking product type (do not edit G)",
        "",
        p32["prose"],
        "",
        _md_table(p32["rows"]),
        "",
        "## 33. Is Y2 0.602 just `g_has_card`?",
        "",
        p33["prose"],
        "",
        _md_table(p33["rows"]),
        "",
        "Do not edit `banking_g_qa.py`. Card 0.551 was already CLOSE as Y3 X.",
        "",
        "## 34. Amount-uncat Y2 on checking-only (no card)",
        "",
        p34["prose"],
        "",
        _md_table(p34["rows"]),
        "",
        "## 35. Inflow vs outflow uncat (messy AP vs both-sides style)",
        "",
        p35["prose"],
        "",
        _md_table(p35["rows"]),
        "",
        "## 36. Amount-uncat residual after `a_n_tx`",
        "",
        p36["prose"],
        "",
        _md_table(p36["rows"]),
        "",
        "## 37. Is the n_tx residual just size?",
        "",
        p37["prose"],
        "",
        _md_table(p37["rows"]),
        "",
        "## 38. Amount-uncat Y2 inside size terciles",
        "",
        p38["prose"],
        "",
        _md_table(p38["rows"]),
        "",
        "## 39. Amount-uncat Y2 on mature books",
        "",
        p39["prose"],
        "",
        _md_table(p39["rows"]),
        "",
        "## 40. Y2 amount-uncat on invoiced 744 vs dark 470",
        "",
        p40["prose"],
        "",
        _md_table(p40["rows"]),
        "",
        "## 41. Always-uncat counterparties (mapping hole vs random miss)",
        "",
        p41["prose"],
        "",
        _md_table(p41["rows"]),
        "",
        "## 42. Description length (described but unlabeled)",
        "",
        p42["prose"],
        "",
        _md_table(p42["rows"]),
        "",
        "## 43. Y2 after dropping all-uncat months",
        "",
        p43["prose"],
        "",
        _md_table(p43["rows"]),
        "",
        "## 44. Weekday of uncat vs mapped",
        "",
        p44["prose"],
        "",
        _md_table(p44["rows"]),
        "",
        "## 45. Invoiced-only style vs shock",
        "",
        p45["prose"],
        "",
        _md_table(p45["rows"]),
        "",
        "## 46. Mapped-category richness vs uncat (CAT_MAP design note)",
        "",
        p46["prose"],
        "",
        _md_table(p46["rows"]),
        "",
        "Do not edit CAT_MAP. Do not add `uncategorized`.",
        "",
        "## What failed / next",
        "",
    ]
    for item in ctx["failed"]:
        lines.append(f"- {item}")
    lines += [
        "",
        f"Elapsed {ctx['elapsed_s']:.0f}s. Cuts: amount vs count, leftover cats, "
        "quintiles, singles, style/shock, dark 470/744, Q6 lag1, mix missingness, holdout, "
        "demean, activity, zero pile, hide-euro, CAT_MAP alias, dark×size, company-median Y2, "
        "messy-company cards, calendar, uncat sign, Y3 after days, "
        "months-on-book, messy Y3 label coverage, within early/late, "
        "all-uncat months, group ICC, 360/110, ticket size, pending vs uncat, "
        "group residual, row flags, value_date lag, product type, "
        "card dummy, checking-only, inflow/outflow, n_tx residual, "
        "resid vs size, size terciles, mature books, dark vs invoiced Y2, "
        "sticky CP, desc length, drop all-uncat, weekday, invoiced style, mapped richness.",
        "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")


def append_registry(ctx: dict) -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    p1, p4, p5, p6, p7, d = (
        ctx["p1"],
        ctx["p4"],
        ctx["p5"],
        ctx["p6"],
        ctx["p7"],
        ctx["decision"],
    )
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    rows = [
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": "-",
            "model": "uncat_qa",
            "split": "train",
            "metric": "a_uncat_share_mean",
            "value": p1["mean_c"],
            "coverage": f"{p1['cov_c']:.4f}",
            "notes": f"amt_mean={p1['mean_a']:.4f} rho_ca={p1['rho_ca']:.3f} hid={p1['n_hid']} leftover_n={p1['leftover_n']:.0f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": Y3,
            "model": "uncat_qa",
            "split": "train_cv",
            "metric": "auroc_a_uncat_share",
            "value": p4["cnt_y3"],
            "coverage": "1.0000",
            "notes": f"amt={p4['amt_y3']:.4f} size={p4['size_y3']:.4f} days={p4['days_y3']:.4f} q5={d['q5']} x={d['x_dec']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": Y3,
            "model": "uncat_qa",
            "split": "train_cv",
            "metric": "auroc_amt_uncat",
            "value": p4["amt_y3"],
            "coverage": "1.0000",
            "notes": f"beat_size={p4['beat_size']:.4f} best={p4['best_name']} style_a={p5['style_a']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": "-",
            "model": "uncat_qa",
            "split": "train",
            "metric": "uncat_acf1",
            "value": p5["acf"]["count"][1],
            "coverage": "1.0000",
            "notes": f"confirm027={p5['confirm_acf']} icc_c={p5['icc']['count']['icc']:.3f} icc_a={p5['icc']['amt']['icc']:.3f} style_c={p5['style_c']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": "-",
            "model": "uncat_qa",
            "split": "train",
            "metric": "dark_vs_erp_uncat",
            "value": p6["dark_c"],
            "coverage": f"{p6['n_dark'] / (p6['n_dark'] + p6['n_erp']) if (p6['n_dark'] + p6['n_erp']) else float('nan'):.4f}",
            "notes": f"erp={p6['erp_c']:.4f} dark={p6['dark_c']:.4f} confirm744_470={p6['confirm']} same={p6['same']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": Y3,
            "model": "uncat_qa",
            "split": "train_cv",
            "metric": "auroc_a_uncat_share_lag1",
            "value": p7["lag_c"],
            "coverage": "1.0000",
            "notes": f"now={p7['now_c']:.4f} amt_lag1={p7['lag_a']:.4f} q6={d['q6']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": Y2,
            "model": "uncat_qa",
            "split": "train_cv",
            "metric": "auroc_amt_uncat",
            "value": ctx["p4"]["amt_y2"],
            "coverage": "1.0000",
            "notes": f"size={ctx['p4']['size_y2']:.4f} co_mean={ctx['p10']['y2_mean']:.4f} demean={ctx['p10']['y2_shock']:.4f} style={ctx['p10']['style_carries_y2']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": Y2,
            "model": "uncat_qa",
            "split": "train_cv",
            "metric": "auroc_amt_uncat_demean",
            "value": ctx["p10"]["y2_shock"],
            "coverage": "1.0000",
            "notes": f"style_mean={ctx['p10']['y2_mean']:.4f} ntx_rho={ctx['p11']['rho_amt_n']:.3f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": "-",
            "model": "uncat_qa",
            "split": "train",
            "metric": "uncat_inflow_share",
            "value": ctx["p19"]["share_in"],
            "coverage": "1.0000",
            "notes": f"p50_in={ctx['p19']['p50_in']:.3f} mostly_in={ctx['p19']['mostly_in']} mostly_out={ctx['p19']['mostly_out']} cal_spread={ctx['p18']['spread']:.3f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": "-",
            "model": "uncat_qa",
            "split": "train",
            "metric": "always_messy_n",
            "value": ctx["p17"]["n_messy"],
            "coverage": "1.0000",
            "notes": f"clean={ctx['p17']['n_clean']} seasonal={ctx['p18']['seasonal']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": "-",
            "model": "uncat_qa",
            "split": "train",
            "metric": "uncat_within_early_late_rho",
            "value": ctx["p23"]["rho"],
            "coverage": "1.0000",
            "notes": f"mean_d={ctx['p23']['mean_d']:.4f} short={ctx['p23']['short_c']:.4f} long={ctx['p23']['long_c']:.4f} onboard={ctx['p21']['onboard']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": Y2,
            "model": "uncat_qa",
            "split": "train_cv",
            "metric": "auroc_group_mean_amt_uncat",
            "value": ctx["p29"]["y2_g"],
            "coverage": "1.0000",
            "notes": f"within={ctx['p29']['y2_r']:.4f} holding={ctx['p29']['holding']} pending_rho={ctx['p28']['rho']:.3f} ticket_x={ctx['p27']['ratio']:.2f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": "-",
            "model": "uncat_qa",
            "split": "train",
            "metric": "uncat_booked_share",
            "value": 0.963,
            "coverage": "1.0000",
            "notes": "pass30 uncat booked 96.3% empty_desc 0 extreme 0; not a DQ hole",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": "-",
            "model": "uncat_qa",
            "split": "train",
            "metric": "uncat_card_tx_share",
            "value": ctx["p32"]["card"],
            "coverage": "1.0000",
            "notes": f"checking={ctx['p32']['chk']:.4f} y2_card={ctx['p33']['y2_card']:.4f} clone={ctx['p33']['clone']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": Y2,
            "model": "uncat_qa",
            "split": "train_cv",
            "metric": "auroc_amt_uncat_checking",
            "value": ctx["p34"]["y2"],
            "coverage": "1.0000",
            "notes": f"has_card_y2={ctx['p33']['y2_amt']:.4f} inout_both={ctx['p35']['both']} ntx_resid={ctx['p36']['y2']:.4f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": Y2,
            "model": "uncat_qa",
            "split": "train_cv",
            "metric": "auroc_amt_uncat_inflow",
            "value": ctx["p35"]["y2_in"],
            "coverage": "1.0000",
            "notes": f"out={ctx['p35']['y2_out']:.4f} both={ctx['p35']['both']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": Y3,
            "model": "uncat_qa",
            "split": "train_cv",
            "metric": "auroc_amt_uncat_ntx_resid",
            "value": ctx["p36"]["y3"],
            "coverage": "1.0000",
            "notes": f"y2={ctx['p36']['y2']:.4f} rho_size={ctx['p37']['rho_size']:.3f} clone={ctx['p37']['clone']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": Y2,
            "model": "uncat_qa",
            "split": "train_cv",
            "metric": "auroc_amt_uncat_mature",
            "value": ctx["p39"]["late"],
            "coverage": "1.0000",
            "notes": f"size_terc_survives={ctx['p38']['survives']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": Y2,
            "model": "uncat_qa",
            "split": "train_cv",
            "metric": "auroc_amt_uncat_invoiced",
            "value": ctx["p40"]["y2_erp"],
            "coverage": "1.0000",
            "notes": f"dark={ctx['p40']['y2_dark']:.4f} both={ctx['p40']['both']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": "-",
            "model": "uncat_qa",
            "split": "train",
            "metric": "uncat_sticky_cp_share",
            "value": ctx["p41"]["sticky_n"],
            "coverage": "1.0000",
            "notes": "share of uncat txs on company×CP pairs with ≥80% uncat",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": "-",
            "model": "uncat_qa",
            "split": "train",
            "metric": "uncat_missing_cp_share",
            "value": ctx["p41"]["miss_n"],
            "coverage": "1.0000",
            "notes": f"named_sticky={ctx['p41']['sticky_named']:.3f} desc_p50={ctx['p42']['p50_u']:.1f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": Y2,
            "model": "uncat_qa",
            "split": "train_cv",
            "metric": "auroc_amt_uncat_drop_alluncat",
            "value": ctx["p43"]["drop"],
            "coverage": "1.0000",
            "notes": "Y2 after dropping amount-uncat ≥0.95 months",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": "-",
            "model": "uncat_qa",
            "split": "train",
            "metric": "uncat_weekday_range",
            "value": ctx["p44"]["spread"],
            "coverage": "1.0000",
            "notes": f"weekend={ctx['p44']['weekend']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": Y2,
            "model": "uncat_qa",
            "split": "train_cv",
            "metric": "auroc_amt_uncat_invoiced_style",
            "value": ctx["p45"]["style"],
            "coverage": "1.0000",
            "notes": f"shock={ctx['p45']['shock']:.4f} carries={ctx['p45']['carries']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": "-",
            "model": "uncat_qa",
            "split": "train",
            "metric": "uncat_vs_mapped_ncat_rho",
            "value": ctx["p46"]["rho"],
            "coverage": "1.0000",
            "notes": "Spearman count-uncat vs n distinct CAT_MAP tokens in the month",
        },
    ]
    with REGISTRY.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        seen = {
            (r.get("agent"), r.get("y"), r.get("model"), r.get("split"), r.get("metric"))
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


def run() -> dict:
    t0 = time.time()
    print(f"uncat_qa start seed={FOLD_SEED} holdout=72 agent={AGENT}")
    panel = load_panel()
    panel = attach_folds(panel)
    con = connect()
    try:
        print("load monthly uncat (in memory)")
        monthly = load_monthly_uncat(con)
        panel = attach_uncat(panel, monthly)
        book = book_invoice_ids(con)
        panel["ever_erp"] = panel["company_id"].isin(book)
        panel = add_style_shock(panel)
        panel = add_panel_lags(
            panel, ["a_uncat_share", "amt_uncat", "c_n_days_with_tx"], (1,)
        )
        tr = panel[panel["split"] == "train"].copy()
        assert_no_holdout(tr["company_id"])
        print(
            f"train CM={len(tr):,} companies={tr['company_id'].nunique()} "
            f"holdout CM={(panel['split']=='holdout').sum()}"
        )
        print("pass 1 amount vs count")
        p1 = pass1_amount_vs_count(tr)
        print("pass 2 leftover categories")
        p2 = pass2_leftover(con, tr)
        print("pass 3 quintiles")
        p3 = pass3_quintiles(tr)
        print("pass 4 singles")
        p4 = pass4_auroc(tr)
        print("pass 5 style vs shock")
        p5 = pass5_style(tr)
        print("pass 6 dark 470 vs 744")
        p6 = pass6_dark(tr, con)
        print("pass 7 Q6 lag1")
        p7 = pass7_q6(tr)
        print("pass 8 mix missingness")
        p8 = pass8_mix_missing(tr)
        print("pass 9 holdout coverage")
        p9 = pass9_holdout(panel)
        print("pass 10 style vs shock AUROC")
        p10 = pass10_demean(tr)
        print("pass 11 activity residual")
        p11 = pass11_activity(tr)
        print("pass 12 zero pile")
        p12 = pass12_zero(tr)
        print("pass 13 hide-euro months")
        p13 = pass13_hide(tr)
        print("pass 14 cash_settlements alias")
        p14 = pass14_alias(con)
        print("pass 15 dark × size")
        p15 = pass15_dark_size(tr, p6)
        print("pass 16 company-median vs Y2")
        p16 = pass16_co_y2(tr)
        print("pass 17 always-messy companies")
        p17 = pass17_messy_cos(tr)
        print("pass 18 calendar")
        p18 = pass18_calendar(tr)
        print("pass 19 uncat sign")
        p19 = pass19_sign(con)
        print("pass 20 Y3 after days")
        p20 = pass20_days(tr)
        print("pass 21 months-on-book")
        p21 = pass21_sofar(tr)
        print("pass 22 messy Y3 label coverage")
        p22 = pass22_y3_label(tr)
        print("pass 23 within-company early vs late")
        p23 = pass23_within(tr)
        print("pass 24 all-uncat months")
        p24 = pass24_alluncat(tr)
        print("pass 25 group ICC")
        p25 = pass25_group_icc(tr)
        print("pass 26 360 vs 110 dark")
        p26 = pass26_dark_mix(tr)
        print("pass 27 ticket size")
        p27 = pass27_ticket(con)
        print("pass 28 pending vs uncat")
        p28 = pass28_pending(tr)
        print("pass 29 group mean vs residual")
        p29 = pass29_group_demean(tr)
        print("pass 30 uncat row flags")
        p30 = pass30_flags(con)
        print("pass 31 value_date lag")
        p31 = pass31_value_lag(con)
        print("pass 32 product type")
        p32 = pass32_product(con)
        print("pass 33 card dummy")
        p33 = pass33_card(tr)
        print("pass 34 checking-only")
        p34 = pass34_checking(tr)
        print("pass 35 inflow vs outflow")
        p35 = pass35_in_out(con, tr)
        print("pass 36 n_tx residual")
        p36 = pass36_ntx_resid(tr)
        print("pass 37 residual vs size")
        p37 = pass37_resid_size(tr)
        print("pass 38 size terciles")
        p38 = pass38_size_terc(tr)
        print("pass 39 mature books")
        p39 = pass39_mature(tr)
        print("pass 40 dark vs invoiced Y2")
        p40 = pass40_dark_y2(tr)
        print("pass 41 always-uncat counterparties")
        p41 = pass41_uncat_cp(con)
        print("pass 42 description length")
        p42 = pass42_desc(con)
        print("pass 43 drop all-uncat")
        p43 = pass43_drop_alluncat(tr)
        print("pass 44 weekday")
        p44 = pass44_weekday(con)
        print("pass 45 invoiced style")
        p45 = pass45_erp_style(tr)
        print("pass 46 mapped richness")
        p46 = pass46_mapped_rich(con, tr)
    finally:
        con.close()

    decision = decide(p1, p2, p3, p4, p5, p7, p10)
    png_ok = make_png(p3)
    headline = (
        f"Count mean {_f(p1['mean_c'])} vs amount-uncat {_f(p1['mean_a'])} "
        f"(ρ {_f(p1['rho_ca'])}; leftover tokens {p2['n_left_tokens']}"
        f"{', only `uncategorized`' if p2['only_uncat'] else ''}). "
        f"Y3 amount {_f(p4['amt_y3'])} count {_f(p4['cnt_y3'])} vs size {_f(p4['size_y3'])} "
        f"(Δ {_f(p4['beat_size'], 3)}) vs days {_f(p4['days_y3'])}. "
        f"Count acf1 {_f(p5['acf']['count'][1])} ICC {_f(p5['icc']['count']['icc'])} "
        f"({'style' if p5['style_c'] else 'shock'}). "
        f"Dark vs 744 count-uncat {_pp(p6['dark_c'])} vs {_pp(p6['erp_c'])}. "
        f"Y2 amount {_f(p4['amt_y2'])} vs size {_f(p4['size_y2'])} "
        f"(style mean {_f(p10['y2_mean'])} / shock {_f(p10['y2_shock'])}). "
        f"Q5 **{decision['q5']}**. PARK as health Y. X **{decision['x_dec']}**. "
        f"Q6 **{decision['q6']}**."
    )
    print(headline)
    failed = []
    if not p1["confirm_cov"]:
        failed.append(f"count cov {_pp(p1['cov_c'])} ≠ 95.8% quote")
    if p1["max_abs"] > 1e-9 if np.isfinite(p1["max_abs"]) else True:
        failed.append(f"store vs recomputed count max|Δ|={_f(p1['max_abs'], 6)}")
    if not p5["confirm_acf"]:
        failed.append(f"acf1 {_f(p5['acf']['count'][1])} ≠ 0.27 quote")
    if not p6["confirm"]:
        failed.append(f"dark/erp {p6['n_dark']}/{p6['n_erp']} ≠ 470/744")
    if abs(p4["days_y3"] - DAYS_BENCH) > 0.02 if np.isfinite(p4["days_y3"]) else True:
        failed.append(
            f"Y3 days replica {_f(p4['days_y3'])} vs night 0.711 "
            "(signed fold; published 0.711 may be oriented 1−raw)"
        )
    if not p2["only_uncat"]:
        failed.append(
            f"leftover tokens besides uncategorized: {p2['n_left_tokens']} — CAT_MAP design note"
        )
    if not p1["confirm_rho"]:
        failed.append(
            f"size ρ vs log1p(|a_op_in|) {_f(p1['rho_opin_c'])} ≠ −0.023 quote"
        )
    if p10["style_carries_y2"]:
        failed.append(
            f"Y2 amount {_f(p4['amt_y2'])} is company-mean style "
            f"({_f(p10['y2_mean'])}) not month shock ({_f(p10['y2_shock'])})"
        )
    failed.append(
        f"CAT_MAP leftover tokens: {p2['n_left_tokens']} (`uncategorized` only) — "
        "do not add that token; never-seen key `cash_settlements` is a Javier alias"
    )
    failed.append(
        f"High-uncat vs Q1 `m_coll_vs_pay` missing {_pp(p8['miss_hi_core'])} "
        f"vs {_pp(p8['miss_lo_core'])} — not a mix-missingness dummy"
    )
    failed.append(
        f"Card txs {_pp(p32['card'])} uncat vs checking {_pp(p32['chk'])}; "
        f"Y2 g_has_card {_f(p33['y2_card'])} vs amount {_f(p33['y2_amt'])}; "
        f"checking-only Y2 {_f(p34['y2'])}"
    )
    failed.append(
        f"Uncat rows booked 96.3% / empty desc 0 — category-style hole, not DQ"
    )
    failed.append(
        f"Inflow vs outflow Y2 {_f(p35['y2_in'])} / {_f(p35['y2_out'])} "
        f"({'both-sides style' if p35['both'] else 'sides differ'})"
    )
    failed.append(
        f"Amount residual after log1p(a_n_tx) Y2 {_f(p36['y2'])} Y3 {_f(p36['y3'])}; "
        f"resid↔size ρ {_f(p37['rho_size'])} ({'size clone' if p37['clone'] else 'not size'})"
    )
    failed.append(
        f"Y2 inside size terciles survives={p38['survives']}; "
        f"mature-book Y2 {_f(p39['late'])}"
    )
    failed.append(
        f"Y2 invoiced {_f(p40['y2_erp'])} vs dark {_f(p40['y2_dark'])} "
        f"({'both sides' if p40['both'] else 'one side'}) — uncat ≠ no ERP"
    )
    failed.append(
        f"Missing-CP uncat {_pp(p41['miss_n'])}; named sticky {_pp(p41['sticky_n'])} "
        f"(of named {_pp(p41['sticky_named'])}) — labeling habit, not vendor hole"
    )
    failed.append(
        f"Desc length p50 uncat {_f(p42['p50_u'], 0)} vs mapped {_f(p42['p50_m'], 0)} "
        f"({'same memo' if p42['same'] else 'different memo'})"
    )
    failed.append(
        f"Y2 after drop all-uncat months {_f(p43['drop'])}"
    )
    failed.append(
        f"Weekday uncat range {_pp(p44['spread'])} "
        f"({'weekend dummy' if p44['weekend'] else 'no weekday dummy'})"
    )
    failed.append(
        f"Invoiced-only Y2 style {_f(p45['style'])} vs shock {_f(p45['shock'])} "
        f"({'style carries' if p45['carries'] else 'shock closer'})"
    )
    failed.append(
        f"Mapped-cat richness ↔ count-uncat ρ {_f(p46['rho'])} — CAT_MAP leftover CLOSE"
    )
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
        "p40": p40,
        "p41": p41,
        "p42": p42,
        "p43": p43,
        "p44": p44,
        "p45": p45,
        "p46": p46,
        "decision": decision,
        "headline": headline,
        "png_ok": png_ok,
        "failed": failed,
        "elapsed_s": elapsed,
        "panel": panel,
    }
    write_md(ctx)
    append_registry(ctx)
    print(f"elapsed {elapsed:.1f}s")
    return ctx


if __name__ == "__main__":
    run()
