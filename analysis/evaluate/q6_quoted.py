"""Quoted Q6 leads vs trail length — coverage and single-feature AUROC.

A lead is only as long as the trail (NORTH_STAR Q6). This module does **not**
redo the months-on-book histogram (`trail_length.md`). It measures whether the
*quoted* Y3 / Y7 / Y4 lead columns are even defined on short books, and whether
the night single-feature AUROCs still hold on short vs long labeled rows.

No 0–100. No product/. No parquet rewrite. No new GBM. No new Y.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.q6_quoted

Owned: analysis/evaluate/q6_quoted.py, analysis/outputs/q6_quoted.md,
optional PNG, registry appends, overnight/waves/wave4_q6_quoted.md (end).
Read-only: duckdb `clean`, monthly.parquet, targets.parquet.

Holdout 72 is coverage only. Seed 20260918 for group folds. Rates / AUROC on
train. Fixed trail cuts 6 / 12 / 18 / 24 — not quantiles.

Quoted engines (do not change):
- Y3 15-col shallow-A 0.752: stems + lags 1,3 of c_ss_month, c_salary_month,
  a_n_tx, f_ds_r, c_n_days_with_tx. Never B. Never a_op_in.
- Y7 TURNOVER 0.720 / n_x=5: issued lag1 + issued-lag CV + credit-note ±lag1
  + f_fc_r_lag3. No DSO. issued-lag CV only if already in the store.
- Y4 single d_cust_hhi_lag3 0.605 (monopoly tail, not a tree).

KEEP as Q6 only if the lead is present on the short half AND short
single-feature train group-fold AUROC is within 0.03 of the night quote
and n_pos ≥ 50. Else CLOSE (or LOW_POWER).
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
from analysis.features.common import ANALYSIS, DATA, MONTHS, connect

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

STORE = DATA / "feature_store" / "monthly.parquet"
TARGETS = DATA / "feature_store" / "targets.parquet"
OUT_MD = ANALYSIS / "outputs" / "q6_quoted.md"
OUT_PNG = ANALYSIS / "outputs" / "q6_quoted_lag_share.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
AGENT = "c0ddcae1"
WAVE = "4"
ROUND = "R4"

Y3 = "y3_recover_cash_6m"
Y7 = "y7_top1_lost"
Y4 = "y4_ds_r_double"
Y_LABEL = {
    Y3: "y3_recover",
    Y7: "y7_top1_lost",
    Y4: "y4_ds_r_double",
}

# Quoted 15-col Y3 stems. Lags 1,3 added in-module (panel shift). Never B / a_op_in.
Y3_STEMS = (
    "c_ss_month",
    "c_salary_month",
    "a_n_tx",
    "f_ds_r",
    "c_n_days_with_tx",
)
Y3_LAGS = (1, 3)

# Y7 TURNOVER quoted columns. CV skipped unless already in store.
Y7_LEADS = (
    "e_ar_issued_lag1",
    "e_ar_issued_lag_cv",
    "e_credit_note_ratio",
    "e_credit_note_ratio_lag1",
    "f_fc_r_lag3",
)
Y7_OPTIONAL = ("e_ar_issued_lag_cv",)

Y4_LEADS = ("d_cust_hhi_lag3",)

# Night single-feature quotes (train group-fold unless noted).
NIGHT = {
    ("y3", "c_n_days_with_tx"): 0.711,  # sibling_h / night bar
    ("y3", "c_ss_month"): 0.690,  # univ |AUROC| on y3_importances
    ("y7", "e_ar_issued_lag1"): 0.630,  # y7_core single on TURNOVER X
    ("y4", "d_cust_hhi_lag3"): 0.605,  # y4_why stable lag-3
}

SINGLES = (
    ("y3", Y3, "c_n_days_with_tx", 0),
    ("y3", Y3, "c_ss_month", 0),
    ("y7", Y7, "e_ar_issued_lag1", 1),
    ("y4", Y4, "d_cust_hhi_lag3", 3),
)

N_FOLDS = 5
KEEP_BAND = 0.03
MIN_POS = 50
SHORT_MAX = 11
LONG_MIN = 18
PRESENT_HALF = 0.50
PANEL_START = pd.Timestamp("2024-09-01")
# Family D full6: win6_start = (period month-end − 5m) month-start >= 2024-09
# → period >= 2025-02. Lag-3 of HHI needs full6 at t−3 → period >= 2025-05.
FULL6_PERIOD = pd.Timestamp("2025-02-01")
FULL6_AT_LAG3 = pd.Timestamp("2025-05-01")
# Y3 lag3 "full4": company-relative so-far >= 4 (panel shift), not a 2024-12 calendar gate.
FULL4_SOFAR = 4

BUCKETS = [
    ("<6", 1, 5),
    ("6-11", 6, 11),
    ("12-17", 12, 17),
    ("18-23", 18, 23),
    ("24", 24, 24),
]
BOOK = (
    "document_type = 'invoice' AND status <> 'cancel' "
    "AND amount <> 0 AND issuance_date IS NOT NULL"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["company_id"] = out["company_id"].astype(str)
    if "period" in out.columns:
        out["period"] = pd.to_datetime(out["period"]).astype("datetime64[ns]")
    if "group_id" in out.columns:
        out["group_id"] = out["group_id"].astype(str)
    return out


def _pct(n, d) -> float:
    if d is None or d == 0:
        return float("nan")
    return float(n) / float(d)


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


def _bucket_label(n) -> str:
    if n is None or (isinstance(n, float) and not np.isfinite(n)):
        return "na"
    n = int(n)
    for name, lo, hi in BUCKETS:
        if lo <= n <= hi:
            return name
    return "na"


def _trail_class(n) -> str:
    if n is None or (isinstance(n, float) and not np.isfinite(n)):
        return "na"
    n = int(n)
    if n < 12:
        return "short_<12"
    if n >= 18:
        return "long_>=18"
    return "mid_12_17"


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
    """Group-fold CV AUROC on `mask`. Sign from the train side of each fold."""
    y = pd.to_numeric(y, errors="coerce")
    x = pd.to_numeric(x, errors="coerce")
    defined = mask & y.notna() & x.notna()
    n = int(defined.sum())
    n_pos = int(((defined) & (y == 1)).sum())
    n_neg = int(((defined) & (y == 0)).sum())
    low = n_pos < MIN_POS
    fold_rows = []
    aucs = []
    if low or n_neg == 0:
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
            "coverage": float("nan"),
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
                "n_va": int((va).sum()),
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
        "coverage": 1.0,
        "low_power": False,
    }


def add_panel_lags(df: pd.DataFrame, stems: list[str], lags: tuple[int, ...]) -> pd.DataFrame:
    """Past-only company shift. Does not write parquet."""
    out = df.sort_values(["company_id", "period"]).reset_index(drop=True)
    cid = out["company_id"]
    extra = {}
    for c in stems:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        g = s.groupby(cid, sort=False)
        for k in lags:
            name = f"{c}_lag{k}"
            extra[name] = g.shift(k)
    if extra:
        out = pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)
    return out


def y3_lead_cols() -> list[str]:
    cols = []
    for s in Y3_STEMS:
        cols.append(s)
        for k in Y3_LAGS:
            cols.append(f"{s}_lag{k}")
    return cols


def load_panel() -> tuple[pd.DataFrame, list[str]]:
    if not STORE.exists():
        raise FileNotFoundError(f"{STORE} missing")
    if not TARGETS.exists():
        raise FileNotFoundError(f"{TARGETS} missing — do not run build_targets")
    raw = pd.read_parquet(STORE)
    yraw = pd.read_parquet(TARGETS)
    print(f"store {STORE} shape={raw.shape} (read-only)")
    print(f"targets {TARGETS} shape={yraw.shape} (read-only)")

    store_lags = [c for c in raw.columns if "lag" in c]
    has_issued_cv = "e_ar_issued_lag_cv" in raw.columns
    print(f"store lag-like cols: {len(store_lags)}; e_ar_issued_lag_cv in store={has_issued_cv}")

    need = [
        "company_id",
        "period",
        "group_id",
        "first_month",
        *Y3_STEMS,
        "e_ar_issued",
        "e_credit_note_ratio",
        "f_fc_r",
        "d_cust_hhi",
        "a_in3",
    ]
    extra_have = [c for c in ("e_ar_issued_lag_cv", "e_credit_note_ratio_lag1", "f_fc_r_lag3", "d_cust_hhi_lag3") if c in raw.columns]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise RuntimeError(f"monthly.parquet missing {missing}")

    panel = _keys(raw[need + extra_have])
    ykeep = ["company_id", "period", Y3, Y7, Y4]
    ymiss = [c for c in ykeep if c not in yraw.columns]
    if ymiss:
        raise RuntimeError(f"targets.parquet missing {ymiss}")
    y = _keys(yraw[ykeep])
    panel = panel.merge(y, on=["company_id", "period"], how="left")

    # Always shift quoted lags here so missingness is the panel clock, not a store accident.
    shift_stems = list(Y3_STEMS) + ["e_ar_issued", "e_credit_note_ratio", "f_fc_r", "d_cust_hhi"]
    panel = add_panel_lags(panel, shift_stems, (1, 3))

    skipped = [c for c in Y7_OPTIONAL if c not in panel.columns]
    print(f"Y7 optional skipped (not in store, not invented): {skipped}")
    return panel, skipped


def attach_trails(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["first_month"] = pd.to_datetime(out["first_month"])
    out["months_so_far"] = (
        (out["period"].dt.year - out["first_month"].dt.year) * 12
        + (out["period"].dt.month - out["first_month"].dt.month)
        + 1
    )
    grid_n = out.groupby("company_id")["period"].size().rename("n_grid_months")
    out = out.merge(grid_n, on="company_id", how="left")
    out["so_far_bucket"] = out["months_so_far"].map(_bucket_label)
    out["so_far_class"] = out["months_so_far"].map(_trail_class)
    out["co_bucket"] = out["n_grid_months"].map(_bucket_label)
    out["co_class"] = out["n_grid_months"].map(_trail_class)
    hold = load_holdout()
    out["split"] = np.where(out["company_id"].isin(hold), "holdout", "train")
    out["full6_now"] = out["period"] >= FULL6_PERIOD
    out["full6_at_lag3"] = out["period"] >= FULL6_AT_LAG3
    return out


def attach_folds(panel: pd.DataFrame) -> pd.DataFrame:
    train = panel[panel["split"] == "train"][["company_id", "group_id"]].drop_duplicates()
    assert_no_holdout(train["company_id"])
    folds = group_folds(train, n=N_FOLDS, seed=FOLD_SEED)
    out = panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")
    return out


def attach_source_months(panel: pd.DataFrame, con) -> pd.DataFrame:
    """Months of each source strictly before t. Not a Y. Not written to parquet."""
    bank = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               CAST(date_trunc('month', "date") AS DATE) AS month
        FROM transactions
        WHERE "date" IS NOT NULL
          AND "date" >= DATE '2024-09-01'
          AND "date" < DATE '2026-09-01'
        GROUP BY 1, 2
        """
    ).df()
    inv = con.execute(
        f"""
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               CAST(date_trunc('month', issuance_date) AS DATE) AS month
        FROM invoices
        WHERE {BOOK}
        GROUP BY 1, 2
        """
    ).df()
    cust = con.execute(
        f"""
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               CAST(date_trunc('month', issuance_date) AS DATE) AS month
        FROM invoices
        WHERE {BOOK}
          AND amount > 0
          AND counterparty_id IS NOT NULL
        GROUP BY 1, 2
        """
    ).df()
    print(
        f"source months: bank={len(bank):,} inv={len(inv):,} erp_cust={len(cust):,} "
        f"(strictly before t)"
    )

    def _before(src: pd.DataFrame, name: str) -> pd.DataFrame:
        s = src.copy()
        s["company_id"] = s["company_id"].astype(str)
        s["month"] = pd.to_datetime(s["month"]).astype("datetime64[ns]")
        left = panel[["company_id", "period"]].copy()
        left["period"] = pd.to_datetime(left["period"]).astype("datetime64[ns]")
        m = left.merge(s, on="company_id", how="left")
        hit = m["month"].notna() & (m["month"] < m["period"])
        cnt = (
            m.loc[hit]
            .groupby(["company_id", "period"], sort=False)
            .size()
            .rename(name)
            .reset_index()
        )
        out = left.merge(cnt, on=["company_id", "period"], how="left")
        out[name] = out[name].fillna(0).astype(int)
        return out[["company_id", "period", name]]

    out = panel.copy()
    for src, name in ((bank, "n_bank_before"), (inv, "n_inv_before"), (cust, "n_cust_before")):
        piece = _before(src, name)
        out = out.merge(piece, on=["company_id", "period"], how="left")
        out[name] = out[name].fillna(0).astype(int)
    return out


def coverage_rows(
    panel: pd.DataFrame,
    ycol: str,
    leads: list[str],
    clock: str,
    split: str = "train",
) -> list[dict]:
    """Share non-null of each lead on labeled rows, by trail bucket."""
    clock_col = "so_far_bucket" if clock == "sofar" else "co_bucket"
    class_col = "so_far_class" if clock == "sofar" else "co_class"
    df = panel[panel["split"] == split]
    slices = (
        [("<6", df[clock_col] == "<6"),
         ("6-11", df[clock_col] == "6-11"),
         ("12-17", df[clock_col] == "12-17"),
         ("18-23", df[clock_col] == "18-23"),
         ("24", df[clock_col] == "24"),
         ("short_<12", df[class_col] == "short_<12"),
         ("long_>=18", df[class_col] == "long_>=18"),
         ("all", pd.Series(True, index=df.index))]
    )
    rows = []
    for lead in leads:
        if lead not in df.columns:
            continue
        for sname, mask in slices:
            sl = df.loc[mask]
            lab = sl[ycol].notna()
            n = int(lab.sum())
            nn = int((lab & sl[lead].notna()).sum()) if n else 0
            rows.append(
                {
                    "y": Y_LABEL[ycol],
                    "lead": lead,
                    "clock": clock,
                    "split": split,
                    "bucket": sname,
                    "n_labeled": n,
                    "n_nn": nn,
                    "share_nn": _pct(nn, n),
                }
            )
    return rows


def decomp_missing(lab: pd.DataFrame, col: str, k: int, kind: str) -> dict:
    """Exclusive missing reasons on labeled rows.

    Y3 / bank stems: (1) shift so-far < k+1  (2) residual source-NaN at t−k.
    Calendar-absolute (period < 2024-09 + k months) is reported but not exclusive
    unless kind='calendar_abs'.
    Y4 HHI: shift / calendar full6 at t−k / source NaN (same exclusive order as
    trail_length: shift, then calendar, then source).
    """
    x = lab[col]
    miss = x.isna()
    n = int(len(lab))
    n_miss = int(miss.sum())
    sofar = lab["months_so_far"]
    r_shift = miss & (sofar < (k + 1))
    if kind == "hhi":
        r_cal = miss & ~r_shift & ~lab["full6_at_lag3"]
        src = lab["d_cust_hhi_at_lag"] if "d_cust_hhi_at_lag" in lab.columns else None
        if src is not None:
            r_src = miss & ~r_shift & ~r_cal & src.isna()
        else:
            r_src = miss & ~r_shift & ~r_cal
        r_res = miss & ~r_shift & ~r_cal & ~r_src
        return {
            "col": col,
            "n_lab": n,
            "n_miss": n_miss,
            "share_miss": _pct(n_miss, n),
            "shift": int(r_shift.sum()),
            "calendar": int(r_cal.sum()),
            "source": int(r_src.sum()),
            "residual": int(r_res.sum()),
        }
    # bank / invoice panel shift
    r_src = miss & ~r_shift
    cal_abs = miss & (lab["period"] < (PANEL_START + pd.DateOffset(months=k)))
    return {
        "col": col,
        "n_lab": n,
        "n_miss": n_miss,
        "share_miss": _pct(n_miss, n),
        "shift": int(r_shift.sum()),
        "calendar_abs": int(cal_abs.sum()),
        "source_or_residual": int(r_src.sum()),
        "residual": int((miss & ~r_shift).sum()),
        "n_sofar_ge_k1": int((sofar >= (k + 1)).sum()),
        "nn_given_sofar_ge_k1": int(((sofar >= (k + 1)) & x.notna()).sum()),
    }


def honest_source_rows(lab: pd.DataFrame, ycol: str, src_col: str, lead: str | None = None) -> list[dict]:
    rows = []
    for sname, mask in (
        ("all", pd.Series(True, index=lab.index)),
        ("short_<12", lab["so_far_class"] == "short_<12"),
        ("long_>=18", lab["so_far_class"] == "long_>=18"),
        ("short_<12_company", lab["co_class"] == "short_<12"),
        ("long_>=18_company", lab["co_class"] == "long_>=18"),
    ):
        sl = lab.loc[mask]
        n = int(len(sl))
        src = sl[src_col]
        rec = {
            "y": Y_LABEL[ycol],
            "slice": sname,
            "n_lab": n,
            "src": src_col,
            "p50_src_before": float(src.median()) if n else float("nan"),
            "ge3": _pct(int((src >= 3).sum()), n),
            "ge6": _pct(int((src >= 6).sum()), n),
        }
        if lead and lead in sl.columns:
            nn = sl[lead].notna()
            rec["lead"] = lead
            rec["n_nn"] = int(nn.sum())
            rec["ge3_given_nn"] = _pct(int((nn & (src >= 3)).sum()), int(nn.sum()))
            rec["ge6_given_nn"] = _pct(int((nn & (src >= 6)).sum()), int(nn.sum()))
        rows.append(rec)
    return rows


def fmt_cov_table(
    rows: list[dict],
    leads: list[str],
    clock: str,
    yname: str,
    split: str = "train",
) -> list[dict]:
    """One row per bucket; columns are lead nn shares. Default split=train."""
    buckets = ["<6", "6-11", "12-17", "18-23", "24", "short_<12", "long_>=18", "all"]
    out = []
    by = {
        (r["lead"], r["bucket"]): r
        for r in rows
        if r["clock"] == clock and r["y"] == yname and r["split"] == split
    }
    for b in buckets:
        rec = {"bucket": b}
        n = None
        for lead in leads:
            r = by.get((lead, b))
            if r is None:
                rec[lead] = "—"
                continue
            n = r["n_labeled"]
            rec[lead] = f"{r['n_nn']:,} ({_pp(r['share_nn'])})"
        rec["n_labeled"] = f"{n:,}" if n is not None else "—"
        out.append(rec)
    return out


def verdict_keep(present_short: float, cv_short: float, night: float, n_pos: int, low_power: bool) -> str:
    if low_power or n_pos < MIN_POS:
        return "LOW_POWER"
    if not np.isfinite(present_short) or present_short < PRESENT_HALF:
        return "CLOSE"
    if not np.isfinite(cv_short) or not np.isfinite(night):
        return "CLOSE"
    if abs(cv_short - night) <= KEEP_BAND:
        return "KEEP"
    return "CLOSE"


def plot_lag_share(cov: list[dict]) -> None:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return
    want = [
        (Y_LABEL[Y3], "c_n_days_with_tx_lag1"),
        (Y_LABEL[Y3], "c_n_days_with_tx_lag3"),
        (Y_LABEL[Y7], "e_ar_issued_lag1"),
        (Y_LABEL[Y4], "d_cust_hhi_lag3"),
    ]
    buckets = ["<6", "6-11", "12-17", "18-23"]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = np.arange(len(buckets))
    width = 0.2
    for i, (yname, lead) in enumerate(want):
        ys = []
        for b in buckets:
            hit = [
                r
                for r in cov
                if r["y"] == yname
                and r["lead"] == lead
                and r["clock"] == "sofar"
                and r["split"] == "train"
                and r["bucket"] == b
            ]
            ys.append(100.0 * hit[0]["share_nn"] if hit and np.isfinite(hit[0]["share_nn"]) else 0.0)
        short = {
            "c_n_days_with_tx_lag1": "Y3 days lag1",
            "c_n_days_with_tx_lag3": "Y3 days lag3",
            "e_ar_issued_lag1": "Y7 issued lag1",
            "d_cust_hhi_lag3": "Y4 HHI lag3",
        }.get(lead, lead)
        ax.bar(x + (i - 1.5) * width, ys, width, label=short)
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_ylabel("non-null share of labeled rows (%)")
    ax.set_xlabel("months-on-book so far (train labeled)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Quoted Q6 lead available share by trail bucket")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


def append_registry(rows: list[dict]) -> None:
    if not rows:
        return
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    with REGISTRY.open("r", encoding="utf-8") as f:
        header = next(csv.reader(f))
    existing = REGISTRY.read_text(encoding="utf-8")
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    written = 0
    with REGISTRY.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        for r in rows:
            rec = {
                "ts": ts,
                "round": ROUND,
                "wave": WAVE,
                "agent": AGENT,
                "x_families": r.get("x_families", "quoted"),
                "y": r.get("y", ""),
                "model": r.get("model", "single"),
                "split": r.get("split", "train"),
                "metric": r.get("metric", ""),
                "value": r.get("value", ""),
                "coverage": r.get("coverage", ""),
                "notes": r.get("notes", ""),
            }
            marker = (
                f"{AGENT},{rec['x_families']},{rec['y']},{rec['model']},"
                f"{rec['split']},{rec['metric']}"
            )
            if marker in existing:
                continue
            w.writerow({k: rec.get(k, "") for k in header})
            written += 1
    print(f"registry appended {written} rows (skipped {len(rows) - written} already present)")


def pass8_cuts(tr: pd.DataFrame, auc_store: dict) -> dict:
    """Company-short presence, f_ds_r residual, so-far 2–3 vs ≥4, Y7 mix."""
    y3 = tr[tr[Y3].notna()]
    y7 = tr[tr[Y7].notna()]
    y4 = tr[tr[Y4].notna()]

    co_present = {}
    for col, df in (
        ("c_n_days_with_tx", y3),
        ("c_ss_month", y3),
        ("c_n_days_with_tx_lag1", y3),
        ("c_n_days_with_tx_lag3", y3),
        ("f_ds_r_lag3", y3),
        ("e_ar_issued_lag1", y7),
        ("e_credit_note_ratio_lag1", y7),
        ("f_fc_r_lag3", y7),
        ("d_cust_hhi_lag3", y4),
    ):
        sl = df[df["co_class"] == "short_<12"]
        co_present[col] = float(sl[col].notna().mean()) if len(sl) and col in sl.columns else float("nan")

    # f_ds_r residual: so-far>=4 but lag3 null
    short3 = y3[y3["so_far_class"] == "short_<12"]
    stem_nn = int(short3["f_ds_r"].notna().sum())
    lag_miss = short3["f_ds_r_lag3"].isna()
    ge4_miss = lag_miss & (short3["months_so_far"] >= 4)
    n_ge4_miss = int(ge4_miss.sum())
    # of those, was stem at t-3 null? we have d-equivalent: f_ds_r_lag3 is the shift
    # count stem-null months on the panel for those companies at t-3 via the lag column
    # residual after shift = stem NaN at t-3
    stem_null_share = float(short3["f_ds_r"].isna().mean())
    ge4 = short3[short3["months_so_far"] >= 4]
    stem_null_ge4 = float(ge4["f_ds_r"].isna().mean()) if len(ge4) else float("nan")
    lag_nn_ge4 = float(ge4["f_ds_r_lag3"].notna().mean()) if len(ge4) else float("nan")

    # f_fc_r_lag3 on Y7 short
    short7 = y7[y7["so_far_class"] == "short_<12"]
    fc_shift = int((short7["f_fc_r_lag3"].isna() & (short7["months_so_far"] < 4)).sum())
    fc_res = int((short7["f_fc_r_lag3"].isna() & (short7["months_so_far"] >= 4)).sum())
    fc_stem_short = float(short7["f_fc_r"].notna().mean()) if "f_fc_r" in short7.columns else float("nan")

    rows = [
        {
            "cut": "Y3 company-short days_lag3 nn",
            "n": int((y3["co_class"] == "short_<12").sum()),
            "n_nn": int(
                ((y3["co_class"] == "short_<12") & y3["c_n_days_with_tx_lag3"].notna()).sum()
            ),
            "share": _pp(co_present["c_n_days_with_tx_lag3"]),
        },
        {
            "cut": "Y3 company-short days_lag1 nn",
            "n": int((y3["co_class"] == "short_<12").sum()),
            "n_nn": int(
                ((y3["co_class"] == "short_<12") & y3["c_n_days_with_tx_lag1"].notna()).sum()
            ),
            "share": _pp(co_present["c_n_days_with_tx_lag1"]),
        },
        {
            "cut": "Y7 company-short issued_lag1 nn",
            "n": int((y7["co_class"] == "short_<12").sum()),
            "n_nn": int(((y7["co_class"] == "short_<12") & y7["e_ar_issued_lag1"].notna()).sum()),
            "share": _pp(co_present["e_ar_issued_lag1"]),
        },
        {
            "cut": "Y7 company-short f_fc_r_lag3 nn",
            "n": int((y7["co_class"] == "short_<12").sum()),
            "n_nn": int(((y7["co_class"] == "short_<12") & y7["f_fc_r_lag3"].notna()).sum()),
            "share": _pp(co_present["f_fc_r_lag3"]),
        },
        {
            "cut": "Y4 company-short HHI_lag3 nn",
            "n": int((y4["co_class"] == "short_<12").sum()),
            "n_nn": int(((y4["co_class"] == "short_<12") & y4["d_cust_hhi_lag3"].notna()).sum()),
            "share": _pp(co_present["d_cust_hhi_lag3"]),
        },
        {
            "cut": "Y3 short f_ds_r stem nn",
            "n": int(len(short3)),
            "n_nn": stem_nn,
            "share": _pp(_pct(stem_nn, len(short3))),
        },
        {
            "cut": "Y3 short so-far≥4 f_ds_r_lag3 nn",
            "n": int(len(ge4)),
            "n_nn": int(ge4["f_ds_r_lag3"].notna().sum()) if len(ge4) else 0,
            "share": _pp(lag_nn_ge4),
        },
        {
            "cut": "Y3 short f_ds_r_lag3 miss after so-far≥4",
            "n": int(len(short3)),
            "n_nn": n_ge4_miss,
            "share": _pp(_pct(n_ge4_miss, len(short3))),
        },
        {
            "cut": "Y7 short f_fc_r_lag3 miss = shift so-far<4",
            "n": int(short7["f_fc_r_lag3"].isna().sum()),
            "n_nn": fc_shift,
            "share": _pp(_pct(fc_shift, int(short7["f_fc_r_lag3"].isna().sum()))),
        },
        {
            "cut": "Y7 short f_fc_r_lag3 miss residual so-far≥4",
            "n": int(short7["f_fc_r_lag3"].isna().sum()),
            "n_nn": fc_res,
            "share": _pp(_pct(fc_res, int(short7["f_fc_r_lag3"].isna().sum()))),
        },
    ]

    auc_extra = {}
    auc_rows = []
    slices = [
        ("y3", Y3, "c_n_days_with_tx", 0.711, "sofar_2_3", y3["months_so_far"].between(2, 3)),
        ("y3", Y3, "c_n_days_with_tx", 0.711, "sofar_ge4_short", (y3["months_so_far"] >= 4) & (y3["so_far_class"] == "short_<12")),
        ("y3", Y3, "c_n_days_with_tx_lag1", 0.711, "sofar_2_3", y3["months_so_far"].between(2, 3)),
        ("y3", Y3, "c_n_days_with_tx_lag1", 0.711, "sofar_ge4_short", (y3["months_so_far"] >= 4) & (y3["so_far_class"] == "short_<12")),
        ("y3", Y3, "c_n_days_with_tx_lag3", 0.711, "sofar_ge4_short", (y3["months_so_far"] >= 4) & (y3["so_far_class"] == "short_<12")),
        ("y3", Y3, "c_n_days_with_tx", 0.711, "mid_12_17_sofar", y3["so_far_class"] == "mid_12_17"),
        ("y7", Y7, "e_ar_issued_lag1", 0.630, "mid_12_17_sofar", y7["so_far_class"] == "mid_12_17"),
        ("y7", Y7, "e_ar_issued_lag1", 0.630, "sofar_short_and_co_long", (y7["so_far_class"] == "short_<12") & (y7["co_class"] == "long_>=18")),
        ("y7", Y7, "e_ar_issued_lag1", 0.630, "sofar_short_and_co_short", (y7["so_far_class"] == "short_<12") & (y7["co_class"] == "short_<12")),
        ("y4", Y4, "d_cust_hhi_lag3", 0.605, "mid_12_17_sofar", y4["so_far_class"] == "mid_12_17"),
        ("y7", Y7, "f_fc_r_lag3", 0.630, "sofar_ge4_short", (y7["months_so_far"] >= 4) & (y7["so_far_class"] == "short_<12")),
    ]
    for key, ycol, col, night, sl_name, sl in slices:
        if col not in tr.columns:
            continue
        # sl is on y3/y7/y4 subset index; align to tr
        mask_lab = tr[ycol].notna()
        sl_tr = pd.Series(False, index=tr.index)
        sl_tr.loc[sl.index] = sl.to_numpy()
        mask = mask_lab & sl_tr & tr[col].notna()
        res = signed_oof_auroc(tr[ycol], tr[col], tr["fold"], mask)
        present = float(tr.loc[mask_lab & sl_tr, col].notna().mean()) if (mask_lab & sl_tr).any() else float("nan")
        auc_extra[(key, col, sl_name)] = {**res, "present": present, "night": night}
        auc_rows.append(
            {
                "y": Y_LABEL[ycol],
                "col": col,
                "slice": sl_name,
                "n_nn": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "present": _pp(present),
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                "Δ night": (
                    "—"
                    if res["low_power"] or not np.isfinite(res["cv"])
                    else f"{res['cv'] - night:+.3f}"
                ),
            }
        )
        print(
            f"P8 AUROC {Y_LABEL[ycol]} {col} {sl_name}: "
            f"{'LOW_POWER' if res['low_power'] else _f(res['cv'])} n_pos={res['n_pos']}"
        )

    prose = (
        f"On <12-month *companies*, Y3 days_lag3 is present on only "
        f"**{_pp(co_present['c_n_days_with_tx_lag3'])}** of labeled rows (so-far-short was 84.7%). "
        f"Y3 days_lag1 stays **{_pp(co_present['c_n_days_with_tx_lag1'])}**. "
        f"Y7 issued_lag1 company-short nn **{_pp(co_present['e_ar_issued_lag1'])}**. "
        f"Y7 `f_fc_r_lag3` company-short nn **{_pp(co_present['f_fc_r_lag3'])}** — CLOSE as a 3-month "
        f"TURNOVER lead on short books. Y3 `f_ds_r` stem nn on short {_pp(_pct(stem_nn, len(short3)))}; "
        f"lag3 nn given so-far≥4 {_pp(lag_nn_ge4)} "
        f"(stem-null share on those rows {_pp(stem_null_ge4)}). "
        f"Y7 short `f_fc_r` stem {_pp(fc_stem_short)}; lag3 holes are "
        f"{fc_shift:,} shift + {fc_res:,} residual."
    )
    return {
        "rows": rows,
        "auc_rows": auc_rows,
        "auc_extra": auc_extra,
        "co_present": co_present,
        "prose": prose,
        "stem_null_ge4": stem_null_ge4,
    }


def pass9_f_window(tr: pd.DataFrame) -> dict:
    """f_ds_r / f_fc_r need rolling 3, so lag3 is empty until so-far>=6."""
    rows = []
    for ycol, stem, lag3, yname in (
        (Y3, "f_ds_r", "f_ds_r_lag3", Y_LABEL[Y3]),
        (Y7, "f_fc_r", "f_fc_r_lag3", Y_LABEL[Y7]),
    ):
        lab = tr[tr[ycol].notna()]
        for k in range(1, 9):
            sl = lab[lab["months_so_far"] == k]
            n = int(len(sl))
            rows.append(
                {
                    "y": yname,
                    "so-far": k,
                    "n_lab": n,
                    f"stem nn": (
                        f"{int(sl[stem].notna().sum()):,} ({_pp(float(sl[stem].notna().mean()) if n else float('nan'))})"
                    ),
                    "lag3 nn": (
                        f"{int(sl[lag3].notna().sum()):,} ({_pp(float(sl[lag3].notna().mean()) if n else float('nan'))})"
                    ),
                }
            )
    # two stems in one table — split display by using generic columns already
    y3 = tr[tr[Y3].notna()]
    y7 = tr[tr[Y7].notna()]
    ds_lt3 = float(y3.loc[y3["months_so_far"] < 3, "f_ds_r"].notna().mean()) if (y3["months_so_far"] < 3).any() else float("nan")
    ds_ge3 = float(y3.loc[y3["months_so_far"] >= 3, "f_ds_r"].notna().mean()) if (y3["months_so_far"] >= 3).any() else float("nan")
    ds_l3_lt6 = float(y3.loc[y3["months_so_far"] < 6, "f_ds_r_lag3"].notna().mean()) if (y3["months_so_far"] < 6).any() else float("nan")
    ds_l3_ge6 = float(y3.loc[y3["months_so_far"] >= 6, "f_ds_r_lag3"].notna().mean()) if (y3["months_so_far"] >= 6).any() else float("nan")
    fc_l3_lt6 = float(y7.loc[y7["months_so_far"] < 6, "f_fc_r_lag3"].notna().mean()) if (y7["months_so_far"] < 6).any() else float("nan")
    fc_l3_ge6 = float(y7.loc[y7["months_so_far"] >= 6, "f_fc_r_lag3"].notna().mean()) if (y7["months_so_far"] >= 6).any() else float("nan")
    prose = (
        f"Family F ratios need `in3`/`ds3` (rolling 3). Y3 `f_ds_r` nn is {_pp(ds_lt3)} "
        f"on so-far<3 and {_pp(ds_ge3)} on so-far≥3. Therefore `f_ds_r_lag3` nn is "
        f"**{_pp(ds_l3_lt6)}** on so-far<6 and {_pp(ds_l3_ge6)} on so-far≥6. "
        f"Y7 `f_fc_r_lag3` is the same clock: {_pp(fc_l3_lt6)} on so-far<6, "
        f"{_pp(fc_l3_ge6)} on so-far≥6. The quoted 15-col / TURNOVER 3-month F leads "
        "are a **company-relative full6**, not the C/A full4. Still not Y4 calendar `full6`."
    )
    # prettier per-Y tables
    pretty = []
    for ycol, stem, lag3, yname in (
        (Y3, "f_ds_r", "f_ds_r_lag3", Y_LABEL[Y3]),
        (Y7, "f_fc_r", "f_fc_r_lag3", Y_LABEL[Y7]),
    ):
        lab = tr[tr[ycol].notna()]
        for k in range(1, 9):
            sl = lab[lab["months_so_far"] == k]
            n = int(len(sl))
            pretty.append(
                {
                    "y": yname,
                    "so-far": k,
                    "n_lab": f"{n:,}",
                    "stem": stem,
                    "stem nn": _pp(float(sl[stem].notna().mean()) if n else float("nan")),
                    "lag3": lag3,
                    "lag3 nn": _pp(float(sl[lag3].notna().mean()) if n else float("nan")),
                }
            )
    return {"prose": prose, "rows": pretty}


def write_md(ctx: dict) -> None:
    p = ctx
    lines = [
        "# Quoted Q6 leads vs trail — coverage and single-feature AUROC",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. No new Y.",
        "",
        "Quote `trail_length.md` (do not redo the histogram): `created_at` 73.6% is the "
        "**connection** clock; late first tx 64.2% is the **bank trail**. Train months-on-book: "
        "<12 28.8%, ≥18 58.8%, =24 35.8%. Holdout =24 **4.2%**. Y4 `d_cust_hhi_lag3` missing on "
        "78% of short labeled rows. Y7 `e_ar_issued_lag1` present on 95% short; honest≥6 = 47%. "
        "**PARK** `created_at` as a health Y.",
        "",
        "## Headline",
        "",
        p["headline"],
        "",
        "### Quoted engines (unchanged)",
        "",
        "- Y3 15-col shallow-A **0.752**: stems + lags 1,3 of `c_ss_month`, `c_salary_month`, "
        "`a_n_tx`, `f_ds_r`, `c_n_days_with_tx`. Never B. Never `a_op_in`.",
        "- Y7 TURNOVER **0.720** / n_x=5: issued lag1 + issued-lag CV + credit-note ±lag1 + "
        "`f_fc_r_lag3`. No DSO.",
        "- Y4 single `d_cust_hhi_lag3` **0.605** (monopoly tail, not a tree).",
        "",
        p.get("skip_note", ""),
        "",
        "## Pass 1 — lead non-null share by months-so-far (train labeled)",
        "",
        "Clock is **months-on-book so far** at the company-month (Q6-honest). A 24-month "
        "company is still short in its first months. Holdout coverage is a later table.",
        "",
        "### Y3 `y3_recover_cash_6m`",
        "",
        _md_table(
            fmt_cov_table(p["cov"], p["y3_leads"], "sofar", Y_LABEL[Y3]),
            ["bucket", "n_labeled", *p["y3_leads_short"]],
        ),
        "",
        "Full 15-col nn shares (train labeled, so-far):",
        "",
        _md_table(
            fmt_cov_table(p["cov"], p["y3_leads"], "sofar", Y_LABEL[Y3]),
            ["bucket", "n_labeled", *p["y3_leads"]],
        ),
        "",
        "### Y7 `y7_top1_lost`",
        "",
        _md_table(
            fmt_cov_table(p["cov"], p["y7_leads"], "sofar", Y_LABEL[Y7]),
            ["bucket", "n_labeled", *p["y7_leads"]],
        ),
        "",
        "### Y4 `y4_ds_r_double`",
        "",
        _md_table(
            fmt_cov_table(p["cov"], list(Y4_LEADS), "sofar", Y_LABEL[Y4]),
            ["bucket", "n_labeled", *Y4_LEADS],
        ),
        "",
        f"Plot: `{OUT_PNG.name}`." if HAS_MPL else "Plot: skipped (no matplotlib).",
        "",
        "## Pass 2 — Y3 lag1 / lag3 missing on short vs Y7 issued_lag1",
        "",
        p["p2_prose"],
        "",
        _md_table(p["p2_rows"]),
        "",
        "## Pass 3 — single-feature train group-fold AUROC (short vs long so-far)",
        "",
        "Only rows where the quoted column is non-null. Sign from the train side of each "
        f"fold. Seed {FOLD_SEED}. If a slice has <{MIN_POS} positives, **LOW_POWER** — do not "
        "quote AUROC as a keep. Night quotes: `c_n_days_with_tx` 0.711, `c_ss_month` 0.690 "
        "(univ), `e_ar_issued_lag1` 0.630, `d_cust_hhi_lag3` 0.605. KEEP band ±0.03 vs night "
        "on the **short** slice, and present on the short half.",
        "",
        _md_table(p["p3_rows"]),
        "",
        "Lag-as-single (same rule; not a tree):",
        "",
        _md_table(p["p3_lag_rows"]),
        "",
        "## Pass 4 — honest source length before t (train labeled)",
        "",
        "Share of labeled rows with ≥3 / ≥6 months of *that source* strictly before t. "
        "Bank tx months for Y3, any book-invoice months for Y7, AR-customer invoice months "
        "for Y4. This is not `months_so_far − k` (quoted from trail_length: Y7 short "
        "honest≥6 = 47% given issued_lag1; Y4 short honest≥6 = 66% given HHI_lag3).",
        "",
        _md_table(p["p4_rows"]),
        "",
        "## Pass 5 — holdout coverage only (no AUROC claim)",
        "",
        p["p5_sentence"],
        "",
        _md_table(p["p5_rows"]),
        "",
        "## Pass 6 — same coverage on company-total trail (not so-far)",
        "",
        "Robustness clock: the company's official grid span. Prefer this for *who* has a "
        "long book; so-far remains the Q6-honest clock for *when* a lead is defined. "
        "Y3 so-far ≥18 is a Feb-2026 sliver of 2024-09 starters (trail_length pass 19) — "
        "do not read so-far-long AUROC as a long-book world.",
        "",
        "### Y3 company-total",
        "",
        _md_table(
            fmt_cov_table(p["cov"], p["y3_leads_short"], "company", Y_LABEL[Y3]),
            ["bucket", "n_labeled", *p["y3_leads_short"]],
        ),
        "",
        "### Y7 company-total",
        "",
        _md_table(
            fmt_cov_table(p["cov"], p["y7_leads"], "company", Y_LABEL[Y7]),
            ["bucket", "n_labeled", *p["y7_leads"]],
        ),
        "",
        "### Y4 company-total",
        "",
        _md_table(
            fmt_cov_table(p["cov"], list(Y4_LEADS), "company", Y_LABEL[Y4]),
            ["bucket", "n_labeled", *Y4_LEADS],
        ),
        "",
        "Single-feature AUROC on company-total short vs long (train labeled, feature non-null):",
        "",
        _md_table(p["p6_auc_rows"]),
        "",
        "## Pass 7 — is Y3 lag3 a calendar full4 like Y4 full6?",
        "",
        p["p7_prose"],
        "",
        _md_table(p["p7_rows"]),
        "",
        "## Pass 8 — company-short CLOSE, f_ds_r residual, so-far 2–3 vs ≥4",
        "",
        p["p8_prose"],
        "",
        _md_table(p["p8_rows"]),
        "",
        "Extra singles (train group-fold, feature non-null):",
        "",
        _md_table(p["p8_auc_rows"]),
        "",
        "## Pass 9 — `f_ds_r` / `f_fc_r` lag3 is a rolling-3 full6, not C/A full4",
        "",
        p["p9_prose"],
        "",
        _md_table(p["p9_rows"]),
        "",
        "## KEEP / CLOSE (honest Q6 sentence)",
        "",
        p["keep_prose"],
        "",
        _md_table(p["keep_rows"]),
        "",
        "## Six brief questions",
        "",
        "| # | question | what this cut says |",
        "| --- | --- | --- |",
        "| 1 | Who is healthy? | Not a health reading. Trail gates the lead, not the label. |",
        "| 2 | Who is improving? | A short book cannot show a 12-month improvement. |",
        "| 3 | Who is turning? | Y4 HHI_lag3 is missing on the short half — no 3-month turn clock there. |",
        "| 4 | Dip vs fall? | Y7 issued_lag1 is present on short; the cap is length, not a hole. |",
        "| 5 | Why did it change? | Contemporaneous Y3 singles can still rank on short books; that is not lead time. |",
        "| 6 | Months earlier? | "
        + p["q6_one"]
        + " |",
        "",
        f"Elapsed {p['elapsed_s']:.0f}s. Cuts: so-far coverage, Y3 vs Y7 missingness, "
        "short/long singles, source-honest length, holdout coverage, company-total, "
        "Y3 full4 vs Y4 full6, company-short CLOSE / f_ds_r / so-far 2–3, F full6, KEEP/CLOSE.",
        "",
        "## What failed / next",
        "",
        *([f"- {x}" for x in p["failed"]] if p["failed"] else ["- (none)"]),
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD}")


def run() -> dict:
    t0 = time.time()
    panel, skipped = load_panel()
    panel = attach_trails(panel)
    panel = attach_folds(panel)
    con = connect()
    try:
        panel = attach_source_months(panel, con)
        # source at t−3 for Y4 decomp (same exclusive order as trail_length)
        tmp = panel.sort_values(["company_id", "period"]).copy()
        tmp["d_cust_hhi_at_lag"] = tmp.groupby("company_id", sort=False)["d_cust_hhi"].shift(3)
        panel = panel.merge(
            tmp[["company_id", "period", "d_cust_hhi_at_lag"]],
            on=["company_id", "period"],
            how="left",
        )
    finally:
        con.close()

    hold = load_holdout()
    assert_no_holdout(panel.loc[panel["split"] == "train", "company_id"])
    leaked = set(panel.loc[panel["split"] == "holdout", "company_id"]) - hold
    if leaked:
        raise AssertionError(f"holdout split mismatch: {sorted(leaked)[:5]}")

    y3_leads = [c for c in y3_lead_cols() if c in panel.columns]
    y3_short = [
        c
        for c in (
            "c_n_days_with_tx",
            "c_n_days_with_tx_lag1",
            "c_n_days_with_tx_lag3",
            "c_ss_month",
            "c_ss_month_lag1",
            "c_ss_month_lag3",
        )
        if c in panel.columns
    ]
    y7_leads = [c for c in Y7_LEADS if c in panel.columns]
    print(f"Y3 leads ({len(y3_leads)}): {y3_leads}")
    print(f"Y7 leads: {y7_leads}  skipped={skipped}")

    cov = []
    for ycol, leads in ((Y3, y3_leads), (Y7, y7_leads), (Y4, list(Y4_LEADS))):
        cov.extend(coverage_rows(panel, ycol, leads, "sofar", "train"))
        cov.extend(coverage_rows(panel, ycol, leads, "company", "train"))
        cov.extend(coverage_rows(panel, ycol, leads, "sofar", "holdout"))

    def _share(yname, lead, bucket, clock="sofar", split="train") -> tuple[float, int, int]:
        hit = [
            r
            for r in cov
            if r["y"] == yname
            and r["lead"] == lead
            and r["bucket"] == bucket
            and r["clock"] == clock
            and r["split"] == split
        ]
        if not hit:
            return float("nan"), 0, 0
        r = hit[0]
        return r["share_nn"], r["n_nn"], r["n_labeled"]

    # --- pass 2 decomp ---
    tr = panel[panel["split"] == "train"]
    p2_rows = []
    for ycol, col, k, kind, sl_name, sl_mask in (
        (Y3, "c_n_days_with_tx_lag1", 1, "bank", "short", tr[Y3].notna() & (tr["so_far_class"] == "short_<12")),
        (Y3, "c_n_days_with_tx_lag3", 3, "bank", "short", tr[Y3].notna() & (tr["so_far_class"] == "short_<12")),
        (Y3, "c_ss_month_lag1", 1, "bank", "short", tr[Y3].notna() & (tr["so_far_class"] == "short_<12")),
        (Y3, "c_ss_month_lag3", 3, "bank", "short", tr[Y3].notna() & (tr["so_far_class"] == "short_<12")),
        (Y3, "a_n_tx_lag3", 3, "bank", "short", tr[Y3].notna() & (tr["so_far_class"] == "short_<12")),
        (Y3, "f_ds_r_lag3", 3, "bank", "short", tr[Y3].notna() & (tr["so_far_class"] == "short_<12")),
        (Y3, "c_n_days_with_tx_lag3", 3, "bank", "all", tr[Y3].notna()),
        (Y7, "e_ar_issued_lag1", 1, "inv", "short", tr[Y7].notna() & (tr["so_far_class"] == "short_<12")),
        (Y7, "e_ar_issued_lag1", 1, "inv", "all", tr[Y7].notna()),
        (Y4, "d_cust_hhi_lag3", 3, "hhi", "short", tr[Y4].notna() & (tr["so_far_class"] == "short_<12")),
        (Y4, "d_cust_hhi_lag3", 3, "hhi", "all", tr[Y4].notna()),
    ):
        lab = tr.loc[sl_mask]
        d = decomp_missing(lab, col, k, kind)
        d["y"] = Y_LABEL[ycol]
        d["slice"] = sl_name
        if kind == "hhi":
            p2_rows.append(
                {
                    "y": d["y"],
                    "col": col,
                    "slice": sl_name,
                    "n_lab": f"{d['n_lab']:,}",
                    "missing": f"{d['n_miss']:,} ({_pp(d['share_miss'])})",
                    "shift<k+1": f"{d['shift']:,}",
                    "calendar": f"{d['calendar']:,}",
                    "source/residual": f"{d['source']:,} / {d['residual']:,}",
                    "nn | history": "—",
                }
            )
        else:
            p2_rows.append(
                {
                    "y": d["y"],
                    "col": col,
                    "slice": sl_name,
                    "n_lab": f"{d['n_lab']:,}",
                    "missing": f"{d['n_miss']:,} ({_pp(d['share_miss'])})",
                    "shift<k+1": f"{d['shift']:,}",
                    "calendar": f"{d['calendar_abs']:,} (abs)",
                    "source/residual": f"{d['residual']:,}",
                    "nn | history": f"{d['nn_given_sofar_ge_k1']:,}/{d['n_sofar_ge_k1']:,}",
                }
            )

    y3s = tr[tr[Y3].notna() & (tr["so_far_class"] == "short_<12")]
    y7s = tr[tr[Y7].notna() & (tr["so_far_class"] == "short_<12")]
    days_l1 = float(y3s["c_n_days_with_tx_lag1"].notna().mean()) if len(y3s) else float("nan")
    days_l3 = float(y3s["c_n_days_with_tx_lag3"].notna().mean()) if len(y3s) else float("nan")
    ss_l1 = float(y3s["c_ss_month_lag1"].notna().mean()) if len(y3s) else float("nan")
    ss_l3 = float(y3s["c_ss_month_lag3"].notna().mean()) if len(y3s) else float("nan")
    iss_l1 = float(y7s["e_ar_issued_lag1"].notna().mean()) if len(y7s) else float("nan")
    y3_ge4 = int((y3s["months_so_far"] >= 4).sum())
    y3_lt4 = int((y3s["months_so_far"] < 4).sum())
    p2_prose = (
        f"Y3 short labeled n={len(y3s):,} (so-far<4 = {y3_lt4:,}; so-far≥4 = {y3_ge4:,}). "
        f"`c_n_days_with_tx` lag1 nn **{_pp(days_l1)}**, lag3 nn **{_pp(days_l3)}**. "
        f"`c_ss_month` lag1 {_pp(ss_l1)}, lag3 {_pp(ss_l3)}. "
        f"Y7 short `e_ar_issued_lag1` nn **{_pp(iss_l1)}** (trail_length 95.2% — confirm). "
        "C/A stems (`c_*`, `a_n_tx`) are 0-filled on the grid, so those lag holes are the "
        "panel shift (so-far < k+1). `f_ds_r` is **not** — it needs a rolling 3, so lag3 "
        "is empty on all so-far<6 (pass 9). C/A lag3 is a company-relative **full4**; "
        "F lag3 is a company-relative **full6**, still not Y4's calendar `full6`."
    )
    print(p2_prose)

    # --- pass 3 AUROC ---
    p3_rows = []
    auc_store = {}
    for key, ycol, col, _k in SINGLES:
        night = NIGHT[(key, col)]
        for sl_name, sl in (
            ("all", tr[ycol].notna()),
            ("short_<12_sofar", tr[ycol].notna() & (tr["so_far_class"] == "short_<12")),
            ("long_>=18_sofar", tr[ycol].notna() & (tr["so_far_class"] == "long_>=18")),
        ):
            mask = sl & tr[col].notna()
            res = signed_oof_auroc(tr[ycol], tr[col], tr["fold"], mask)
            present = float(tr.loc[sl, col].notna().mean()) if sl.any() else float("nan")
            auc_store[(key, col, sl_name)] = {**res, "present": present, "night": night}
            flag = "LOW_POWER" if res["low_power"] else _f(res["cv"])
            p3_rows.append(
                {
                    "y": Y_LABEL[ycol],
                    "col": col,
                    "slice": sl_name,
                    "n_nn": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "present": _pp(present),
                    "CV": flag,
                    "sd": _f(res["sd"]),
                    "sign": res["train_sign"] if not res["low_power"] else "—",
                    "night": f"{night:.3f}",
                    "Δ night": (
                        "—"
                        if res["low_power"] or not np.isfinite(res["cv"])
                        else f"{res['cv'] - night:+.3f}"
                    ),
                    "verdict": (
                        "LOW_POWER"
                        if res["low_power"]
                        else verdict_keep(present, res["cv"], night, res["n_pos"], False)
                        if sl_name.startswith("short")
                        else "quote-only"
                    ),
                }
            )
            print(
                f"AUROC {Y_LABEL[ycol]} {col} {sl_name}: "
                f"cv={flag} n_pos={res['n_pos']} present={present:.3f}"
            )

    extra_singles = (
        ("y3", Y3, "c_n_days_with_tx_lag1", 0.711),
        ("y3", Y3, "c_n_days_with_tx_lag3", 0.711),
        ("y3", Y3, "c_ss_month_lag1", 0.690),
        ("y3", Y3, "c_ss_month_lag3", 0.690),
        ("y3", Y3, "f_ds_r_lag3", 0.711),
        ("y7", Y7, "e_credit_note_ratio_lag1", 0.630),
        ("y7", Y7, "f_fc_r_lag3", 0.630),
    )
    p3_lag_rows = []
    for key, ycol, col, night in extra_singles:
        if col not in tr.columns:
            continue
        for sl_name, sl in (
            ("all", tr[ycol].notna()),
            ("short_<12_sofar", tr[ycol].notna() & (tr["so_far_class"] == "short_<12")),
            ("long_>=18_sofar", tr[ycol].notna() & (tr["so_far_class"] == "long_>=18")),
        ):
            mask = sl & tr[col].notna()
            res = signed_oof_auroc(tr[ycol], tr[col], tr["fold"], mask)
            present = float(tr.loc[sl, col].notna().mean()) if sl.any() else float("nan")
            auc_store[(key, col, sl_name)] = {**res, "present": present, "night": night}
            p3_lag_rows.append(
                {
                    "y": Y_LABEL[ycol],
                    "col": col,
                    "slice": sl_name,
                    "n_nn": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "present": _pp(present),
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                    "Δ vs stem-night": (
                        "—"
                        if res["low_power"] or not np.isfinite(res["cv"])
                        else f"{res['cv'] - night:+.3f}"
                    ),
                }
            )

    # --- pass 4 honest source ---
    p4 = []
    p4.extend(honest_source_rows(tr[tr[Y3].notna()], Y3, "n_bank_before", "c_n_days_with_tx_lag3"))
    p4.extend(honest_source_rows(tr[tr[Y7].notna()], Y7, "n_inv_before", "e_ar_issued_lag1"))
    p4.extend(honest_source_rows(tr[tr[Y4].notna()], Y4, "n_cust_before", "d_cust_hhi_lag3"))
    p4_rows = [
        {
            "y": r["y"],
            "slice": r["slice"],
            "n_lab": f"{r['n_lab']:,}",
            "source": r["src"],
            "p50 before t": _f(r["p50_src_before"], 1),
            "≥3": _pp(r["ge3"]),
            "≥6": _pp(r["ge6"]),
            "lead nn": f"{r.get('n_nn', 0):,}" if "n_nn" in r else "—",
            "ge6 given nn": _pp(r["ge6_given_nn"]) if "ge6_given_nn" in r else "—",
        }
        for r in p4
    ]

    # --- pass 5 holdout ---
    ho = panel[panel["split"] == "holdout"]
    p5_rows = []
    for ycol, lead in (
        (Y3, "c_n_days_with_tx_lag3"),
        (Y3, "c_n_days_with_tx_lag1"),
        (Y7, "e_ar_issued_lag1"),
        (Y4, "d_cust_hhi_lag3"),
    ):
        for sname, mask in (
            ("all", ho[ycol].notna()),
            ("short_<12", ho[ycol].notna() & (ho["so_far_class"] == "short_<12")),
            ("eq24_book", ho[ycol].notna() & (ho["n_grid_months"] == 24)),
        ):
            sl = ho.loc[mask]
            n = int(len(sl))
            nn = int(sl[lead].notna().sum()) if n else 0
            p5_rows.append(
                {
                    "y": Y_LABEL[ycol],
                    "lead": lead,
                    "slice": sname,
                    "n_labeled": n,
                    "n_nn": nn,
                    "share_nn": _pp(_pct(nn, n)),
                    "n_pos": int((sl[ycol] == 1).sum()) if n else 0,
                }
            )
    n24 = int((ho.drop_duplicates("company_id")["n_grid_months"] == 24).sum()) if len(ho) else 0
    s4, nn4, n4 = _share(Y_LABEL[Y4], "d_cust_hhi_lag3", "all", "sofar", "holdout")
    s7, nn7, n7 = _share(Y_LABEL[Y7], "e_ar_issued_lag1", "all", "sofar", "holdout")
    s3, nn3, n3 = _share(Y_LABEL[Y3], "c_n_days_with_tx_lag3", "all", "sofar", "holdout")
    p5_sentence = (
        f"On the hidden 72 (train quote only; =24 books = {n24} / 72 = 4.2% per trail_length) "
        f"Y4 `d_cust_hhi_lag3` is defined on {nn4}/{n4} labeled rows ({_pp(s4)}); "
        f"Y7 `e_ar_issued_lag1` on {nn7}/{n7} ({_pp(s7)}); "
        f"Y3 days_lag3 on {nn3}/{n3} ({_pp(s3)}). "
        "A 3-month HHI lead cannot be said on the hidden 72 (almost no 24-month books, "
        "and the lag is almost undefined). A 1-month issued lead *is* defined. "
        "Y3 lag3 is defined wherever so-far≥4 — that is most holdout Y3 labels, not a 24-month privilege. "
        "Do not transfer the Y7 0.630 night number onto the hidden 72: company-short issued_lag1 CV is 0.722 "
        "(+0.092); early months of long books are 0.601. Coverage transfers; the AUROC mix may not."
    )
    print(p5_sentence)

    # --- pass 6 company-total AUROC ---
    p6_auc_rows = []
    for key, ycol, col, _k in SINGLES:
        night = NIGHT[(key, col)]
        for sl_name, sl in (
            ("short_<12_company", tr[ycol].notna() & (tr["co_class"] == "short_<12")),
            ("long_>=18_company", tr[ycol].notna() & (tr["co_class"] == "long_>=18")),
        ):
            mask = sl & tr[col].notna()
            res = signed_oof_auroc(tr[ycol], tr[col], tr["fold"], mask)
            present = float(tr.loc[sl, col].notna().mean()) if sl.any() else float("nan")
            p6_auc_rows.append(
                {
                    "y": Y_LABEL[ycol],
                    "col": col,
                    "slice": sl_name,
                    "n_nn": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "present": _pp(present),
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                    "Δ night": (
                        "—"
                        if res["low_power"] or not np.isfinite(res["cv"])
                        else f"{res['cv'] - night:+.3f}"
                    ),
                }
            )
            print(
                f"company-clock AUROC {Y_LABEL[ycol]} {col} {sl_name}: "
                f"{'LOW_POWER' if res['low_power'] else _f(res['cv'])} n_pos={res['n_pos']}"
            )

    # --- pass 7 calendar full4 vs full6 ---
    y3_lab = tr[tr[Y3].notna()]
    y4_lab = tr[tr[Y4].notna()]
    # among short Y3 with so-far>=4, is lag3 present?
    ge4 = y3_lab["months_so_far"] >= 4
    short = y3_lab["so_far_class"] == "short_<12"
    late_start = y3_lab["first_month"] > PANEL_START
    p7_rows = [
        {
            "test": "Y3 short ∧ so-far≥4 → days_lag3 nn",
            "n": int((short & ge4).sum()),
            "n_nn": int((short & ge4 & y3_lab["c_n_days_with_tx_lag3"].notna()).sum()),
            "share": _pp(
                _pct(
                    int((short & ge4 & y3_lab["c_n_days_with_tx_lag3"].notna()).sum()),
                    int((short & ge4).sum()),
                )
            ),
        },
        {
            "test": "Y3 late-start ∧ so-far≥4 → days_lag3 nn",
            "n": int((late_start & ge4).sum()),
            "n_nn": int((late_start & ge4 & y3_lab["c_n_days_with_tx_lag3"].notna()).sum()),
            "share": _pp(
                _pct(
                    int((late_start & ge4 & y3_lab["c_n_days_with_tx_lag3"].notna()).sum()),
                    int((late_start & ge4).sum()),
                )
            ),
        },
        {
            "test": "Y3 period<2024-12 (abs calendar) among short miss lag3",
            "n": int((short & y3_lab["c_n_days_with_tx_lag3"].isna()).sum()),
            "n_nn": int(
                (
                    short
                    & y3_lab["c_n_days_with_tx_lag3"].isna()
                    & (y3_lab["period"] < pd.Timestamp("2024-12-01"))
                ).sum()
            ),
            "share": _pp(
                _pct(
                    int(
                        (
                            short
                            & y3_lab["c_n_days_with_tx_lag3"].isna()
                            & (y3_lab["period"] < pd.Timestamp("2024-12-01"))
                        ).sum()
                    ),
                    int((short & y3_lab["c_n_days_with_tx_lag3"].isna()).sum()),
                )
            ),
        },
        {
            "test": "Y4 short ∧ so-far≥4 → HHI_lag3 nn",
            "n": int(((y4_lab["so_far_class"] == "short_<12") & (y4_lab["months_so_far"] >= 4)).sum()),
            "n_nn": int(
                (
                    (y4_lab["so_far_class"] == "short_<12")
                    & (y4_lab["months_so_far"] >= 4)
                    & y4_lab["d_cust_hhi_lag3"].notna()
                ).sum()
            ),
            "share": _pp(
                _pct(
                    int(
                        (
                            (y4_lab["so_far_class"] == "short_<12")
                            & (y4_lab["months_so_far"] >= 4)
                            & y4_lab["d_cust_hhi_lag3"].notna()
                        ).sum()
                    ),
                    int(((y4_lab["so_far_class"] == "short_<12") & (y4_lab["months_so_far"] >= 4)).sum()),
                )
            ),
        },
        {
            "test": "Y4 short ∧ period≥2025-05 (full6 at t−3) → HHI_lag3 nn",
            "n": int(((y4_lab["so_far_class"] == "short_<12") & y4_lab["full6_at_lag3"]).sum()),
            "n_nn": int(
                (
                    (y4_lab["so_far_class"] == "short_<12")
                    & y4_lab["full6_at_lag3"]
                    & y4_lab["d_cust_hhi_lag3"].notna()
                ).sum()
            ),
            "share": _pp(
                _pct(
                    int(
                        (
                            (y4_lab["so_far_class"] == "short_<12")
                            & y4_lab["full6_at_lag3"]
                            & y4_lab["d_cust_hhi_lag3"].notna()
                        ).sum()
                    ),
                    int(((y4_lab["so_far_class"] == "short_<12") & y4_lab["full6_at_lag3"]).sum()),
                )
            ),
        },
    ]
    y3_ge4_nn = p7_rows[0]["share"]
    y4_ge4_nn = p7_rows[3]["share"]
    p7_prose = (
        f"Y3 `*_lag3` is present on **{y3_ge4_nn}** of short labeled rows with so-far≥4, "
        f"including late first-tx books. Missing lag3 is almost only so-far<4 (panel shift). "
        f"Y4 `d_cust_hhi_lag3` is still missing on short even after so-far≥4 "
        f"(nn share {y4_ge4_nn}) — calendar `full6` at t−3 plus no ERP HHI, same as "
        "trail_length pass 10. Y3 lag3 is **not** a 2024-12 calendar gate like Y4 `full6`."
    )
    print(p7_prose)

    # --- pass 8: next same-module cuts ---
    p8 = pass8_cuts(tr, auc_store)
    for k, res in p8["auc_extra"].items():
        auc_store[k] = res
    p8_rows = p8["rows"]
    p8_auc_rows = p8["auc_rows"]
    p8_prose = p8["prose"]
    print(p8_prose)

    p9 = pass9_f_window(tr)
    print(p9["prose"])

    # --- KEEP / CLOSE (so-far first, then company-short can CLOSE a lead) ---
    keep_rows = []
    keep_verdicts = {}
    q6_claims = (
        ("y3", Y3, "c_n_days_with_tx", 0.711, "signal", False),
        ("y3", Y3, "c_ss_month", 0.690, "signal", False),
        ("y3", Y3, "c_n_days_with_tx_lag1", 0.711, "q6", True),
        ("y3", Y3, "c_n_days_with_tx_lag3", 0.711, "q6", True),
        ("y3", Y3, "f_ds_r_lag3", 0.711, "q6", True),
        ("y7", Y7, "e_ar_issued_lag1", 0.630, "q6", True),
        ("y7", Y7, "e_credit_note_ratio_lag1", 0.630, "q6", True),
        ("y7", Y7, "f_fc_r_lag3", 0.630, "q6", True),
        ("y4", Y4, "d_cust_hhi_lag3", 0.605, "q6", True),
    )
    for key, ycol, col, night, kind, is_lead in q6_claims:
        rec = auc_store.get((key, col, "short_<12_sofar"))
        if rec is None:
            continue
        sofar_v = verdict_keep(rec["present"], rec["cv"], night, rec["n_pos"], rec["low_power"])
        co_present = p8["co_present"].get(col, float("nan"))
        # After company-total: missing on the short *book* half CLOSES a Q6 lead.
        if is_lead and np.isfinite(co_present) and co_present < PRESENT_HALF:
            q6_v = "CLOSE"
        elif kind == "signal":
            q6_v = "SIGNAL" if sofar_v == "KEEP" else sofar_v
        else:
            q6_v = sofar_v
        keep_verdicts[(key, col)] = q6_v
        keep_rows.append(
            {
                "claim": f"{Y_LABEL[ycol]} `{col}`",
                "kind": kind,
                "so-far nn": _pp(rec["present"]),
                "co-short nn": _pp(co_present),
                "so-far n_pos": rec["n_pos"],
                "so-far CV": "LOW_POWER" if rec["low_power"] else _f(rec["cv"]),
                "night": f"{night:.3f}",
                "Δ": (
                    "—"
                    if rec["low_power"] or not np.isfinite(rec["cv"])
                    else f"{rec['cv'] - night:+.3f}"
                ),
                "so-far rule": sofar_v,
                "Q6": q6_v,
            }
        )

    y3_days = keep_verdicts.get(("y3", "c_n_days_with_tx"), "—")
    y3_ss = keep_verdicts.get(("y3", "c_ss_month"), "—")
    y3_l1 = keep_verdicts.get(("y3", "c_n_days_with_tx_lag1"), "—")
    y3_l3 = keep_verdicts.get(("y3", "c_n_days_with_tx_lag3"), "—")
    y7_v = keep_verdicts.get(("y7", "e_ar_issued_lag1"), "—")
    y4_v = keep_verdicts.get(("y4", "d_cust_hhi_lag3"), "—")

    survivors = [r["claim"] for r in keep_rows if r["Q6"] == "KEEP"]
    if survivors:
        keep_prose = (
            "KEEP as an honest Q6 sentence only: "
            + "; ".join(survivors)
            + ". Contemporaneous Y3 days/ss are SIGNAL (rank on short books) not lead time. "
            "CLOSE if missing on the so-far short half *or* the company-short half, "
            "AUROC off the night quote by >0.03, or LOW_POWER n_pos<50."
        )
    else:
        keep_prose = (
            "No quoted lead KEEPs as an honest Q6 sentence on short books "
            "(present on so-far short ∧ present on company-short ∧ short CV within 0.03 of night "
            f"∧ n_pos≥50). Y3 days={y3_days}, Y3 ss={y3_ss}, Y3 days_lag1={y3_l1}, "
            f"Y3 days_lag3={y3_l3}, Y7 issued_lag1={y7_v}, Y4 HHI_lag3={y4_v}."
        )

    rec7 = auc_store[("y7", "e_ar_issued_lag1", "short_<12_sofar")]
    rec3 = auc_store[("y3", "c_n_days_with_tx", "short_<12_sofar")]
    rec4 = auc_store[("y4", "d_cust_hhi_lag3", "short_<12_sofar")]
    q6_one = (
        f"Y7 issued_lag1 {y7_v} (present {_pp(rec7['present'])} short"
        + (
            f", CV {_f(rec7['cv'])} vs 0.630"
            if not rec7["low_power"]
            else ", LOW_POWER"
        )
        + f"); Y4 HHI_lag3 {y4_v} (present {_pp(rec4['present'])} — missing on the short half); "
        f"Y3 contemporaneous days {y3_days} (signal, not a lead); Y3 days_lag3 {y3_l3} "
        f"(so-far nn {_pp(days_l3)}; company-short nn "
        f"{_pp(p8['co_present'].get('c_n_days_with_tx_lag3', float('nan')))} — "
        "full4 shift, not calendar full6)."
    )

    headline = (
        f"Y3 short lag1 nn {_pp(days_l1)} / lag3 nn {_pp(days_l3)} "
        f"(so-far<4 = {y3_lt4:,} of {len(y3s):,}); "
        f"Y7 issued_lag1 nn {_pp(iss_l1)}; "
        f"Y4 HHI_lag3 short present {_pp(rec4['present'])}. "
        f"Short singles: days {('LOW_POWER' if rec3['low_power'] else _f(rec3['cv']))} "
        f"vs 0.711; issued_lag1 {('LOW_POWER' if rec7['low_power'] else _f(rec7['cv']))} "
        f"vs 0.630; HHI_lag3 {('LOW_POWER' if rec4['low_power'] else _f(rec4['cv']))} "
        f"vs 0.605. Survivors: {', '.join(survivors) if survivors else 'none'}."
    )
    print(headline)
    print(keep_prose)

    failed = []
    if skipped:
        failed.append(
            f"`e_ar_issued_lag_cv` not in store — skipped (not invented). TURNOVER coverage "
            "uses issued_lag1 + credit-note ±lag1 + f_fc_r_lag3 only."
        )
    failed.append(
        "Y3 so-far≥18 labeled is a Feb-2026 sliver (trail_length) — long so-far AUROC is "
        "LOW_POWER or a calendar mix, not a long-book world. Prefer company-total for who."
    )
    if rec4["present"] < PRESENT_HALF:
        failed.append(
            "Y4 `d_cust_hhi_lag3` CLOSE as Q6 on short: missing on the short half "
            f"({_pp(rec4['present'])}); trail_length 21.7%."
        )
    if p8["co_present"].get("c_n_days_with_tx_lag3", 1.0) < PRESENT_HALF:
        failed.append(
            "Y3 days_lag3 CLOSE as Q6 on short *companies* "
            f"(nn {_pp(p8['co_present']['c_n_days_with_tx_lag3'])}) even though so-far-short is 84.7%."
        )
    if p8["co_present"].get("f_fc_r_lag3", 1.0) < PRESENT_HALF:
        failed.append(
            "Y7 `f_fc_r_lag3` CLOSE as a 3-month TURNOVER lead on short companies "
            f"(nn {_pp(p8['co_present']['f_fc_r_lag3'])})."
        )
    failed.append(
        "Y3 `f_ds_r_lag3` is 0% on so-far<6 (rolling-3 at t−3). The 15-col is two clocks: "
        "C/A full4 vs F full6. CLOSE the F 3-month lead on short books."
    )

    skip_note = (
        "`e_ar_issued_lag_cv` is **not** in `monthly.parquet` (it is derived inside "
        "`gbm_y7_core.py`). Skipped. Do not invent it here."
        if skipped
        else "`e_ar_issued_lag_cv` found in store — included in coverage."
    )

    plot_lag_share(cov)

    # registry: coverage + short AUROCs
    reg = []
    for yname, lead, bucket in (
        (Y_LABEL[Y3], "c_n_days_with_tx_lag1", "short_<12"),
        (Y_LABEL[Y3], "c_n_days_with_tx_lag3", "short_<12"),
        (Y_LABEL[Y7], "e_ar_issued_lag1", "short_<12"),
        (Y_LABEL[Y4], "d_cust_hhi_lag3", "short_<12"),
    ):
        sh, nn, nlab = _share(yname, lead, bucket)
        reg.append(
            {
                "x_families": "quoted",
                "y": yname,
                "model": "coverage",
                "split": "train",
                "metric": f"{lead}_nn_short_sofar",
                "value": sh,
                "coverage": sh,
                "notes": f"n_nn={nn}/{nlab}; so-far clock; not a GBM",
            }
        )
    for key, ycol, col, _k in SINGLES:
        rec = auc_store[(key, col, "short_<12_sofar")]
        reg.append(
            {
                "x_families": "quoted",
                "y": Y_LABEL[ycol],
                "model": "single_short",
                "split": "train",
                "metric": f"{col}_cv_short_sofar",
                "value": rec["cv"] if not rec["low_power"] else "",
                "coverage": rec["present"],
                "notes": (
                    f"n_pos={rec['n_pos']}; night={NIGHT[(key, col)]}; "
                    f"verdict={keep_verdicts[(key, col)]}; "
                    + ("LOW_POWER" if rec["low_power"] else f"cv={rec['cv']:.4f}")
                ),
            }
        )
    s7h, nn7h, n7h = _share(Y_LABEL[Y7], "e_ar_issued_lag1", "all", "sofar", "holdout")
    s4h, nn4h, n4h = _share(Y_LABEL[Y4], "d_cust_hhi_lag3", "all", "sofar", "holdout")
    reg.append(
        {
            "x_families": "quoted",
            "y": Y_LABEL[Y7],
            "model": "coverage",
            "split": "holdout",
            "metric": "e_ar_issued_lag1_nn_holdout",
            "value": s7h,
            "coverage": s7h,
            "notes": f"n_nn={nn7h}/{n7h}; LOW_POWER coverage only",
        }
    )
    reg.append(
        {
            "x_families": "quoted",
            "y": Y_LABEL[Y4],
            "model": "coverage",
            "split": "holdout",
            "metric": "d_cust_hhi_lag3_nn_holdout",
            "value": s4h,
            "coverage": s4h,
            "notes": f"n_nn={nn4h}/{n4h}; LOW_POWER coverage only; trail_length 21/135",
        }
    )
    for col, yname in (
        ("c_n_days_with_tx_lag3", Y_LABEL[Y3]),
        ("e_ar_issued_lag1", Y_LABEL[Y7]),
        ("f_fc_r_lag3", Y_LABEL[Y7]),
        ("d_cust_hhi_lag3", Y_LABEL[Y4]),
    ):
        sh = p8["co_present"].get(col, float("nan"))
        reg.append(
            {
                "x_families": "quoted",
                "y": yname,
                "model": "coverage",
                "split": "train",
                "metric": f"{col}_nn_short_company",
                "value": sh,
                "coverage": sh,
                "notes": "company-total trail <12; pass8 CLOSE gate",
            }
        )
    append_registry(reg)

    elapsed = time.time() - t0
    ctx = {
        "headline": headline,
        "skip_note": skip_note,
        "cov": cov,
        "y3_leads": y3_leads,
        "y3_leads_short": y3_short,
        "y7_leads": y7_leads,
        "p2_prose": p2_prose,
        "p2_rows": p2_rows,
        "p3_rows": p3_rows,
        "p3_lag_rows": p3_lag_rows,
        "p4_rows": p4_rows,
        "p5_sentence": p5_sentence,
        "p5_rows": p5_rows,
        "p6_auc_rows": p6_auc_rows,
        "p7_prose": p7_prose,
        "p7_rows": p7_rows,
        "p8_prose": p8_prose,
        "p8_rows": p8_rows,
        "p8_auc_rows": p8_auc_rows,
        "p9_prose": p9["prose"],
        "p9_rows": p9["rows"],
        "keep_prose": keep_prose,
        "keep_rows": keep_rows,
        "q6_one": q6_one,
        "elapsed_s": elapsed,
        "failed": failed,
        "auc_store": auc_store,
        "keep_verdicts": keep_verdicts,
        "panel": panel,
    }
    write_md(ctx)
    print(f"elapsed {elapsed:.1f}s")
    return ctx


if __name__ == "__main__":
    run()
