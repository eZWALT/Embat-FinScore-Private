"""Trail length / left-truncation — Q6 honesty (how many months earlier).

A lead-time claim is only as long as the observed trail. This module measures
the bank book, product-connection clock, and invoice book on the official
monthly grid. It does not write a Y, a 0–100 score, or parquet.

    python -m analysis.evaluate.trail_length

Owned: analysis/evaluate/trail_length.py, analysis/outputs/trail_length.md,
optional PNG, registry appends, overnight/waves/wave4_trail_length.md (end).
Read-only: duckdb `clean`, monthly.parquet, targets.parquet.

Holdout 72 is coverage only. Fixed cuts 6 / 12 / 18 / 24 — no train quantiles
as features. Seed 20260918 is never used.

Iteration log (same module, not one-shot):
1. Bank trail + 73.6% replica + histogram + size/subsidiary + Y rates + Q6 lags
2. Group wave (staggered new groups, not one same-month wave)
3. Family G created_* CONSTANTS ≠ 73.6%; fact is g_n_accounts=0
4. Lag-missing why (shift / calendar full6 / source at t−k); residual → 0
5. Connection lag; holdout hist; silent months
6. Right-censor + ERP×trail + honest lead length
7. g_n_accounts=0 months still have txs (96.6%)
8. Y rates × ever-ERP × trail (not a miss indicator)
9. Post-snapshot-only books; invoice vs bank by bucket; PARK created_at as Y
10. Clip invoice months to the bank grid; pre-grid ERP; company created_at vs first tx
11. Y7 so-far=1 labels have e_ar_issued (pre-grid); issued_lag1 is 0
12. Sep-2026 extract txs vs August quiet tail (not a death Y)
13. August-hole × trail; holdout 12/15 groups are late arrivals
14. Holdout Y4 HHI_lag3 is almost undefined (21/135; 2 from the 3 full-24 books)
15. Same-month late groups are spread + mostly solo (not one wave)
16. Holdout <12 ≈ train; missing is the 24-month pile. Honest lead × trail.
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

from analysis.features.common import ANALYSIS, DATA, MONTHS, connect, load_holdout

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

STORE = DATA / "feature_store" / "monthly.parquet"
TARGETS = DATA / "feature_store" / "targets.parquet"
OUT_MD = ANALYSIS / "outputs" / "trail_length.md"
OUT_PNG = ANALYSIS / "outputs" / "trail_length_months_on_book.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
AGENT = "6bf54618"
WAVE = "4"
ROUND = "R4"

PANEL_START = pd.Timestamp("2024-09-01")
PANEL_END = pd.Timestamp("2026-08-01")  # last official grid month
EXTRACT = pd.Timestamp("2026-09-01")
N_GRID = len(MONTHS)  # 24

BOOK = (
    "document_type = 'invoice' AND status <> 'cancel' "
    "AND amount <> 0 AND issuance_date IS NOT NULL"
)

Y_COLS = [
    "y3_recover_cash_6m",
    "y2_neg_2of3",
    "y7_top1_lost",
    "y4_ds_r_double",
]
Y_LABEL = {
    "y3_recover_cash_6m": "y3_recover",
    "y2_neg_2of3": "y2_neg_2of3",
    "y7_top1_lost": "y7_top1_lost",
    "y4_ds_r_double": "y4_ds_r_double",
}

# Fixed trail buckets (months on official bank grid). Not quantiles.
# Company-level: span first_tx_month → 2026-08.
# Company-month: months-on-book-so-far at `period` (Q6-honest).
BUCKETS = [
    ("<6", 1, 5),
    ("6-11", 6, 11),
    ("12-17", 12, 17),
    ("18-23", 18, 23),
    ("24", 24, 24),
]
SHORT_MAX = 11  # <12
LONG_MIN = 18  # ≥18


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _pct(n, d) -> float:
    if d is None or d == 0:
        return float("nan")
    return float(n) / float(d)


def _f(x, nd=4) -> str:
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


def _bucket_label(n: float) -> str:
    if n is None or (isinstance(n, float) and not np.isfinite(n)):
        return "na"
    n = int(n)
    for name, lo, hi in BUCKETS:
        if lo <= n <= hi:
            return name
    if n < 1:
        return "<6"
    return "24" if n >= 24 else "18-23"


def _trail_class(n) -> str:
    if n is None or (isinstance(n, float) and not np.isfinite(n)):
        return "na"
    n = int(n)
    if n <= SHORT_MAX:
        return "short_<12"
    if n >= LONG_MIN:
        return "long_>=18"
    return "mid_12_17"


def _load_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly = pd.read_parquet(STORE)
    targets = pd.read_parquet(TARGETS)
    monthly["company_id"] = monthly["company_id"].astype(str)
    monthly["period"] = pd.to_datetime(monthly["period"])
    targets["company_id"] = targets["company_id"].astype(str)
    targets["period"] = pd.to_datetime(targets["period"])
    hold = load_holdout()
    monthly["is_holdout"] = monthly["company_id"].isin(hold)
    monthly["split"] = np.where(monthly["is_holdout"], "holdout", "train")
    targets["is_holdout"] = targets["company_id"].isin(hold)
    targets["split"] = np.where(targets["is_holdout"], "holdout", "train")
    return monthly, targets


def _companies(con) -> pd.DataFrame:
    hold = load_holdout()
    cos = con.execute(
        """
        SELECT CAST(c.company_id AS VARCHAR) AS company_id,
               CAST(c.group_id AS VARCHAR) AS group_id,
               c.erp,
               CAST(c.created_at AS TIMESTAMP) AS company_created_at,
               g.n_companies_in_sample AS group_size,
               CAST(g.erp AS VARCHAR) AS group_erp
        FROM companies c
        LEFT JOIN groups g ON c.group_id = g.group_id
        """
    ).df()
    cos["company_id"] = cos["company_id"].astype(str)
    cos["group_id"] = cos["group_id"].astype(str)
    cos["company_created_at"] = pd.to_datetime(cos["company_created_at"], errors="coerce")
    cos["is_holdout"] = cos["company_id"].isin(hold)
    cos["split"] = np.where(cos["is_holdout"], "holdout", "train")
    return cos


# ---------------------------------------------------------------------------
# Pass 1 — bank trail on the official monthly grid
# ---------------------------------------------------------------------------


def pass1_bank_trail(con, cos: pd.DataFrame, monthly: pd.DataFrame) -> dict:
    """First/last tx, months with ≥1 tx, months on the official grid."""
    tx = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               MIN("date") AS first_tx,
               MAX("date") AS last_tx,
               MIN(CASE WHEN "date" < TIMESTAMP '2026-09-01'
                        THEN "date" END) AS last_tx_on_grid_raw
        FROM transactions
        WHERE "date" IS NOT NULL
        GROUP BY 1
        """
    ).df()
    tx["company_id"] = tx["company_id"].astype(str)
    tx["first_tx"] = pd.to_datetime(tx["first_tx"])
    tx["last_tx"] = pd.to_datetime(tx["last_tx"])
    tx["first_tx_month"] = tx["first_tx"].dt.to_period("M").dt.to_timestamp()
    tx["last_tx_month"] = tx["last_tx"].dt.to_period("M").dt.to_timestamp()

    # Months with ≥1 tx on the official 24-month grid (exclude extract month).
    txm = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               CAST(date_trunc('month', "date") AS DATE) AS month,
               COUNT(*) AS n_tx
        FROM transactions
        WHERE "date" IS NOT NULL
          AND "date" >= TIMESTAMP '2024-09-01'
          AND "date" < TIMESTAMP '2026-09-01'
        GROUP BY 1, 2
        """
    ).df()
    txm["company_id"] = txm["company_id"].astype(str)
    txm["month"] = pd.to_datetime(txm["month"])
    n_tx_months = txm.groupby("company_id").size().rename("n_tx_months")
    last_on_grid = (
        txm.groupby("company_id")["month"].max().rename("last_tx_month_on_grid")
    )

    # Official grid span: first_tx_month → PANEL_END (same rule as monthly_grid).
    grid_n = monthly.groupby("company_id").size().rename("n_grid_months")
    first_grid = monthly.groupby("company_id")["period"].min().rename("first_grid_month")
    last_grid = monthly.groupby("company_id")["period"].max().rename("last_grid_month")

    panel = cos.merge(tx, on="company_id", how="left")
    panel = panel.merge(n_tx_months, on="company_id", how="left")
    panel = panel.merge(last_on_grid, on="company_id", how="left")
    panel = panel.merge(grid_n, on="company_id", how="left")
    panel = panel.merge(first_grid, on="company_id", how="left")
    panel = panel.merge(last_grid, on="company_id", how="left")
    panel["n_tx_months"] = panel["n_tx_months"].fillna(0).astype(int)
    panel["n_grid_months"] = panel["n_grid_months"].fillna(0).astype(int)
    panel["late_first_tx"] = panel["first_tx_month"] > PANEL_START
    panel["full_24"] = panel["n_grid_months"] == 24
    panel["trail_bucket"] = panel["n_grid_months"].map(_bucket_label)
    panel["trail_class"] = panel["n_grid_months"].map(_trail_class)
    # Gaps: on the grid but no tx that month.
    panel["n_gap_months"] = (panel["n_grid_months"] - panel["n_tx_months"]).clip(lower=0)

    def _split_sum(df: pd.DataFrame, split: str) -> dict:
        s = df[df["split"] == split]
        n = int(len(s))
        hist = {k: int((s["n_grid_months"] == k).sum()) for k in range(1, 25)}
        return {
            "n_companies": n,
            "n_with_tx": int(s["first_tx"].notna().sum()),
            "n_late_first_tx": int(s["late_first_tx"].fillna(False).sum()),
            "share_late_first_tx": _pct(s["late_first_tx"].fillna(False).sum(), n),
            "n_full_24": int(s["full_24"].sum()),
            "share_full_24": _pct(s["full_24"].sum(), n),
            "n_lt6": int((s["n_grid_months"] < 6).sum()),
            "share_lt6": _pct((s["n_grid_months"] < 6).sum(), n),
            "n_lt12": int((s["n_grid_months"] < 12).sum()),
            "share_lt12": _pct((s["n_grid_months"] < 12).sum(), n),
            "n_ge18": int((s["n_grid_months"] >= 18).sum()),
            "share_ge18": _pct((s["n_grid_months"] >= 18).sum(), n),
            "mean_grid": float(s["n_grid_months"].mean()) if n else float("nan"),
            "median_grid": float(s["n_grid_months"].median()) if n else float("nan"),
            "mean_tx_months": float(s["n_tx_months"].mean()) if n else float("nan"),
            "median_tx_months": float(s["n_tx_months"].median()) if n else float("nan"),
            "n_any_gap": int((s["n_gap_months"] > 0).sum()),
            "share_any_gap": _pct((s["n_gap_months"] > 0).sum(), n),
            "mean_gap": float(s["n_gap_months"].mean()) if n else float("nan"),
            "hist": hist,
            "bucket": {
                name: {
                    "n": int((s["trail_bucket"] == name).sum()),
                    "share": _pct((s["trail_bucket"] == name).sum(), n),
                }
                for name, _, _ in BUCKETS
            },
        }

    # first_tx month histogram (train)
    tr = panel[panel["split"] == "train"]
    first_hist = (
        tr.dropna(subset=["first_tx_month"])
        .groupby(tr["first_tx_month"].dt.strftime("%Y-%m"))
        .size()
        .to_dict()
    )

    return {
        "panel": panel,
        "txm": txm,
        "train": _split_sum(panel, "train"),
        "holdout": _split_sum(panel, "holdout"),
        "first_tx_month_hist_train": first_hist,
        "n_grid_expected": N_GRID,
        "panel_start": str(PANEL_START.date()),
        "panel_end": str(PANEL_END.date()),
    }


# ---------------------------------------------------------------------------
# Pass 2 — banking / debt created_at (confirm or correct 73.6%)
# ---------------------------------------------------------------------------


def pass2_products(con, bank: pd.DataFrame) -> dict:
    """First banking_products.created_at and first debt product.

    Join-QA replica: among train companies *with a banking product* that sit
    on the monthly panel, share whose MIN(created_at) > 2024-09-01.
    Also report first-created *month* after 2024-09 (stricter) and the
    mismatch vs first transaction month (connection clock ≠ bank trail).
    """
    bp = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               MIN(CAST(created_at AS TIMESTAMP)) AS first_bank_created,
               MAX(CAST(created_at AS TIMESTAMP)) AS last_bank_created,
               COUNT(*) AS n_banking,
               COUNT(*) FILTER (WHERE created_at IS NULL) AS n_bank_null,
               COUNT(*) FILTER (WHERE created_at > TIMESTAMP '2026-09-01') AS n_bank_after_snap
        FROM banking_products
        GROUP BY 1
        """
    ).df()
    bp["company_id"] = bp["company_id"].astype(str)
    bp["first_bank_created"] = pd.to_datetime(bp["first_bank_created"], errors="coerce")
    bp["last_bank_created"] = pd.to_datetime(bp["last_bank_created"], errors="coerce")
    bp["first_bank_month"] = bp["first_bank_created"].dt.to_period("M").dt.to_timestamp()

    dp = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               MIN(CAST(created_at AS TIMESTAMP)) AS first_debt_created,
               COUNT(*) AS n_debt
        FROM debt_products
        GROUP BY 1
        """
    ).df()
    dp["company_id"] = dp["company_id"].astype(str)
    dp["first_debt_created"] = pd.to_datetime(dp["first_debt_created"], errors="coerce")
    dp["first_debt_month"] = dp["first_debt_created"].dt.to_period("M").dt.to_timestamp()

    panel = bank.merge(bp, on="company_id", how="left")
    panel = panel.merge(dp, on="company_id", how="left")
    panel["n_banking"] = panel["n_banking"].fillna(0).astype(int)
    panel["n_debt"] = panel["n_debt"].fillna(0).astype(int)
    panel["has_banking"] = panel["n_banking"] > 0
    panel["has_debt"] = panel["n_debt"] > 0

    cut = PANEL_START
    # Join-QA used utc=True; our timestamps are naive. Both > 2024-09-01 00:00.
    panel["bank_created_after_0901"] = panel["first_bank_created"] > cut
    panel["bank_created_after_sep"] = panel["first_bank_month"] > cut  # month >= Oct
    panel["bank_created_before_panel"] = panel["first_bank_created"] < cut
    panel["bank_created_in_sep2024"] = panel["first_bank_month"] == cut
    panel["debt_created_after_0901"] = panel["first_debt_created"] > cut
    # Connection clock vs observed tx trail.
    panel["bank_created_after_first_tx"] = (
        panel["first_bank_created"].notna()
        & panel["first_tx"].notna()
        & (panel["first_bank_created"] > panel["first_tx"])
    )
    panel["bank_created_before_first_tx"] = (
        panel["first_bank_created"].notna()
        & panel["first_tx"].notna()
        & (panel["first_bank_created"] < panel["first_tx"])
    )

    def _qa_replica(split: str) -> dict:
        s = panel[panel["split"] == split]
        with_b = s[s["has_banking"]]
        n_b = int(len(with_b))
        n_after = int(with_b["bank_created_after_0901"].sum())
        n_after_sep = int(with_b["bank_created_after_sep"].sum())
        n_before = int(with_b["bank_created_before_panel"].sum())
        n_sep = int(with_b["bank_created_in_sep2024"].sum())
        return {
            "n_cos": int(len(s)),
            "n_with_banking": n_b,
            "n_no_banking": int((~s["has_banking"]).sum()),
            "n_first_created_after_2024_09_01": n_after,
            "share_after_0901": _pct(n_after, n_b),
            "n_first_created_month_after_sep": n_after_sep,
            "share_after_sep_month": _pct(n_after_sep, n_b),
            "n_first_created_before_panel": n_before,
            "share_before_panel": _pct(n_before, n_b),
            "n_first_created_in_sep2024": n_sep,
            "share_in_sep2024": _pct(n_sep, n_b),
            "n_null_created": int(with_b["first_bank_created"].isna().sum()),
            "n_with_debt": int(s["has_debt"].sum()),
            "n_debt_after_0901": int(
                s.loc[s["has_debt"], "debt_created_after_0901"].sum()
            ),
            "share_debt_after_0901": _pct(
                s.loc[s["has_debt"], "debt_created_after_0901"].sum(),
                int(s["has_debt"].sum()),
            ),
            "n_created_after_first_tx": int(s["bank_created_after_first_tx"].sum()),
            "share_created_after_first_tx": _pct(
                s["bank_created_after_first_tx"].sum(), n_b
            ),
            "n_created_before_first_tx": int(s["bank_created_before_first_tx"].sum()),
            "share_created_before_first_tx": _pct(
                s["bank_created_before_first_tx"].sum(), n_b
            ),
        }

    # Cross: late created_at vs late first_tx (train, with banking).
    trb = panel[(panel["split"] == "train") & panel["has_banking"]]
    cross = pd.crosstab(
        trb["bank_created_after_0901"].fillna(False).astype(bool),
        trb["late_first_tx"].fillna(False).astype(bool),
        dropna=False,
    )
    cross_d = {
        "n": int(len(trb)),
        "created_late_and_tx_late": int(
            (trb["bank_created_after_0901"] & trb["late_first_tx"]).sum()
        ),
        "created_late_and_tx_full": int(
            (trb["bank_created_after_0901"] & ~trb["late_first_tx"]).sum()
        ),
        "created_early_and_tx_late": int(
            (~trb["bank_created_after_0901"] & trb["late_first_tx"]).sum()
        ),
        "created_early_and_tx_full": int(
            (~trb["bank_created_after_0901"] & ~trb["late_first_tx"]).sum()
        ),
    }

    # Join-QA number to confirm: 891 / 1211 = 0.7358.
    train_qa = _qa_replica("train")
    confirm = (
        train_qa["n_with_banking"] == 1211
        and train_qa["n_first_created_after_2024_09_01"] == 891
    )
    # Allow 1-off; still quote the computed share.
    join_qa_share = 891 / 1211
    share_ok = abs(train_qa["share_after_0901"] - join_qa_share) < 0.005

    return {
        "panel": panel,
        "train": train_qa,
        "holdout": _qa_replica("holdout"),
        "cross_train": cross_d,
        "cross_table": cross.to_dict(),
        "join_qa_n_after": 891,
        "join_qa_n_with": 1211,
        "join_qa_share": join_qa_share,
        "confirm_count": confirm,
        "confirm_share": share_ok,
        "verdict": (
            "CONFIRM"
            if confirm or share_ok
            else "CORRECT"
        ),
    }


# ---------------------------------------------------------------------------
# Pass 3 — invoice book vs bank book
# ---------------------------------------------------------------------------


def pass3_invoices(con, bank: pd.DataFrame) -> dict:
    inv = con.execute(
        f"""
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               MIN(issuance_date) AS first_iss,
               MAX(issuance_date) AS last_iss
        FROM invoices
        WHERE {BOOK}
        GROUP BY 1
        """
    ).df()
    inv["company_id"] = inv["company_id"].astype(str)
    inv["first_iss"] = pd.to_datetime(inv["first_iss"])
    inv["last_iss"] = pd.to_datetime(inv["last_iss"])
    inv["first_iss_month"] = inv["first_iss"].dt.to_period("M").dt.to_timestamp()

    invm = con.execute(
        f"""
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               CAST(date_trunc('month', issuance_date) AS DATE) AS month,
               COUNT(*) AS n_iss
        FROM invoices
        WHERE {BOOK}
          AND issuance_date >= TIMESTAMP '2024-09-01'
          AND issuance_date < TIMESTAMP '2026-09-01'
        GROUP BY 1, 2
        """
    ).df()
    invm["company_id"] = invm["company_id"].astype(str)
    invm["month"] = pd.to_datetime(invm["month"])
    n_inv_months = invm.groupby("company_id").size().rename("n_inv_months")

    panel = bank.merge(inv, on="company_id", how="left")
    panel = panel.merge(n_inv_months, on="company_id", how="left")
    panel["n_inv_months"] = panel["n_inv_months"].fillna(0).astype(int)
    panel["ever_erp"] = panel["first_iss"].notna()
    panel["inv_minus_bank"] = panel["n_inv_months"] - panel["n_grid_months"]
    panel["first_iss_after_panel"] = panel["first_iss_month"] > PANEL_START
    panel["first_iss_after_first_tx"] = (
        panel["ever_erp"]
        & panel["first_tx"].notna()
        & (panel["first_iss"] > panel["first_tx"] + pd.Timedelta(days=31))
    )

    def _s(split: str) -> dict:
        s = panel[panel["split"] == split]
        n = int(len(s))
        erp = s[s["ever_erp"]]
        dark = s[~s["ever_erp"]]
        return {
            "n": n,
            "n_ever_erp": int(len(erp)),
            "share_ever_erp": _pct(len(erp), n),
            "n_dark": int(len(dark)),
            "share_dark": _pct(len(dark), n),
            "n_first_iss_after_0901": int(erp["first_iss_after_panel"].sum()),
            "share_first_iss_after_0901": _pct(erp["first_iss_after_panel"].sum(), len(erp)),
            "mean_inv_months_erp": float(erp["n_inv_months"].mean()) if len(erp) else float("nan"),
            "median_inv_months_erp": float(erp["n_inv_months"].median()) if len(erp) else float("nan"),
            "mean_bank_months_erp": float(erp["n_grid_months"].mean()) if len(erp) else float("nan"),
            "mean_bank_months_dark": float(dark["n_grid_months"].mean()) if len(dark) else float("nan"),
            "mean_inv_minus_bank_erp": float(erp["inv_minus_bank"].mean()) if len(erp) else float("nan"),
            "median_inv_minus_bank_erp": float(erp["inv_minus_bank"].median()) if len(erp) else float("nan"),
            "n_inv_shorter": int((erp["n_inv_months"] < erp["n_grid_months"]).sum()),
            "share_inv_shorter": _pct((erp["n_inv_months"] < erp["n_grid_months"]).sum(), len(erp)),
            "n_iss_gt31_after_tx": int(erp["first_iss_after_first_tx"].sum()),
            "share_iss_gt31_after_tx": _pct(erp["first_iss_after_first_tx"].sum(), len(erp)),
        }

    return {"panel": panel, "train": _s("train"), "holdout": _s("holdout")}


# ---------------------------------------------------------------------------
# Pass 4 — histogram PNG (train months-on-book)
# ---------------------------------------------------------------------------


def pass4_histogram(bank: pd.DataFrame) -> dict:
    tr = bank[bank["split"] == "train"]
    counts = [int((tr["n_grid_months"] == k).sum()) for k in range(1, 25)]
    n = int(len(tr))
    shares = [_pct(c, n) for c in counts]
    late = tr["late_first_tx"].fillna(False)
    c_full = [int(((tr["n_grid_months"] == k) & ~late).sum()) for k in range(1, 25)]
    c_late = [int(((tr["n_grid_months"] == k) & late).sum()) for k in range(1, 25)]
    s_full = [_pct(c, n) for c in c_full]
    s_late = [_pct(c, n) for c in c_late]
    wrote = False
    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(9.2, 4.4))
        x = list(range(1, 25))
        ax.bar(x, s_full, color="#2c5282", width=0.85, label="first tx 2024-09")
        ax.bar(x, s_late, bottom=s_full, color="#c05621", width=0.85, label="first tx after 2024-09")
        ax.axvline(5.5, color="#744210", ls="--", lw=0.9)
        ax.axvline(11.5, color="#744210", ls="--", lw=0.9)
        ax.axvline(17.5, color="#276749", ls="--", lw=0.9)
        ax.set_xticks(x)
        ax.set_xlabel("months on bank grid (first tx → 2026-08)")
        ax.set_ylabel("share of train companies")
        ax.set_title(f"Months-on-book — train n={n:,} (stacked by first-tx clock)")
        ax.set_ylim(0, max(shares) * 1.22 if max(shares) else 1)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        fig.tight_layout()
        OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_PNG, dpi=120)
        plt.close(fig)
        wrote = True
    ho = bank[bank["split"] == "holdout"]
    n_ho = int(len(ho))
    ho_counts = [int((ho["n_grid_months"] == k).sum()) for k in range(1, 25)]
    ho_lt6 = int((ho["n_grid_months"] < 6).sum())
    ho_lt12 = int((ho["n_grid_months"] < 12).sum())
    ho_ge18 = int((ho["n_grid_months"] >= 18).sum())
    ho_eq24 = int((ho["n_grid_months"] == 24).sum())
    return {
        "n_train": n,
        "counts": counts,
        "shares": shares,
        "png": str(OUT_PNG.relative_to(ROOT)) if wrote else None,
        "n_holdout": n_ho,
        "ho_counts": ho_counts,
        "ho_lt6": ho_lt6,
        "ho_lt12": ho_lt12,
        "ho_ge18": ho_ge18,
        "ho_eq24": ho_eq24,
        "ho_share_lt6": _pct(ho_lt6, n_ho),
        "ho_share_lt12": _pct(ho_lt12, n_ho),
        "ho_share_ge18": _pct(ho_ge18, n_ho),
        "ho_share_eq24": _pct(ho_eq24, n_ho),
    }


# ---------------------------------------------------------------------------
# Pass 5 — left-truncation: size and new group vs new subsidiary
# ---------------------------------------------------------------------------


def pass5_left_trunc(bank: pd.DataFrame, monthly: pd.DataFrame) -> dict:
    """Companies whose first tx month is after 2024-09.

    Size = company-median of log1p(a_in3) on train company-months (descriptive).
    No bins fit. New group = group's earliest first-tx (all members) > 2024-09.
    New subsidiary = this company late, but some sibling started in 2024-09.
    """
    m = monthly.copy()
    m["log1p_a_in3"] = np.log1p(np.maximum(pd.to_numeric(m["a_in3"], errors="coerce"), 0.0))
    size = (
        m[m["split"] == "train"]
        .groupby("company_id")["log1p_a_in3"]
        .median()
        .rename("med_log1p_a_in3")
    )
    last_size = (
        m.sort_values("period")
        .groupby("company_id")
        .tail(1)
        .set_index("company_id")["log1p_a_in3"]
        .rename("last_log1p_a_in3")
    )

    panel = bank.merge(size, on="company_id", how="left")
    panel = panel.merge(last_size, on="company_id", how="left")

    # Group first-tx: all members (descriptive of the group), and train-only.
    g_all = (
        panel.groupby("group_id")["first_tx_month"]
        .min()
        .rename("group_first_tx_all")
    )
    g_tr = (
        panel[panel["split"] == "train"]
        .groupby("group_id")["first_tx_month"]
        .min()
        .rename("group_first_tx_train")
    )
    panel = panel.merge(g_all, on="group_id", how="left")
    panel = panel.merge(g_tr, on="group_id", how="left")
    panel["old_group_all"] = panel["group_first_tx_all"] == PANEL_START
    panel["new_group_all"] = panel["group_first_tx_all"] > PANEL_START
    panel["new_subsidiary"] = panel["late_first_tx"] & panel["old_group_all"]
    panel["new_group_member"] = panel["late_first_tx"] & panel["new_group_all"]

    tr = panel[panel["split"] == "train"]
    late = tr[tr["late_first_tx"].fillna(False)]
    full = tr[~tr["late_first_tx"].fillna(False)]

    def _size_block(s: pd.DataFrame) -> dict:
        x = s["med_log1p_a_in3"].dropna()
        return {
            "n": int(len(s)),
            "n_size": int(x.shape[0]),
            "mean": float(x.mean()) if len(x) else float("nan"),
            "median": float(x.median()) if len(x) else float("nan"),
            "p25": float(x.quantile(0.25)) if len(x) else float("nan"),
            "p75": float(x.quantile(0.75)) if len(x) else float("nan"),
        }

    # Mann-Whitney is a diagnostic, not a fitted cut. Optional.
    mw_p = float("nan")
    try:
        from scipy.stats import mannwhitneyu

        a = late["med_log1p_a_in3"].dropna()
        b = full["med_log1p_a_in3"].dropna()
        if len(a) >= 20 and len(b) >= 20:
            mw_p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:
        pass

    n_late = int(len(late))
    return {
        "panel": panel,
        "n_train": int(len(tr)),
        "n_late": n_late,
        "share_late": _pct(n_late, len(tr)),
        "n_full": int(len(full)),
        "size_late": _size_block(late),
        "size_full": _size_block(full),
        "delta_median_size": (
            float(late["med_log1p_a_in3"].median() - full["med_log1p_a_in3"].median())
            if n_late and len(full)
            else float("nan")
        ),
        "mw_p": mw_p,
        "n_new_subsidiary": int(late["new_subsidiary"].sum()),
        "share_new_subsidiary": _pct(late["new_subsidiary"].sum(), n_late),
        "n_new_group_member": int(late["new_group_member"].sum()),
        "share_new_group_member": _pct(late["new_group_member"].sum(), n_late),
        "n_groups_old": int(tr.loc[tr["old_group_all"], "group_id"].nunique()),
        "n_groups_new": int(tr.loc[tr["new_group_all"], "group_id"].nunique()),
        "n_groups_train": int(tr["group_id"].nunique()),
    }


# ---------------------------------------------------------------------------
# Pass 6 — accepted-Y base rates on short vs long trail (train labeled)
# ---------------------------------------------------------------------------


def pass6_y_rates(
    bank: pd.DataFrame, monthly: pd.DataFrame, targets: pd.DataFrame
) -> dict:
    """Company-month trail = months-on-book-so-far (Q6-honest).

    Also report company-total-trail robustness. Rates on train labeled rows
    only. Holdout counted, not used for any cut.
    """
    m = monthly[["company_id", "period", "split", "a_in3"]].copy()
    first = bank[["company_id", "first_tx_month", "n_grid_months", "trail_class", "ever_erp"]].copy()
    m = m.merge(first, on="company_id", how="left")
    m["months_so_far"] = (
        (m["period"].dt.year - m["first_tx_month"].dt.year) * 12
        + (m["period"].dt.month - m["first_tx_month"].dt.month)
        + 1
    )
    m["so_far_class"] = m["months_so_far"].map(_trail_class)
    m["so_far_bucket"] = m["months_so_far"].map(_bucket_label)
    m["co_class"] = m["trail_class"]

    ykeep = ["company_id", "period"] + [c for c in Y_COLS if c in targets.columns]
    y = targets[ykeep].copy()
    m = m.merge(y, on=["company_id", "period"], how="left")

    # Lags for pass 7 live on monthly — attach if present, else shift here.
    extra = []
    if "d_cust_hhi" in monthly.columns:
        extra.append("d_cust_hhi")
    if "e_ar_issued" in monthly.columns:
        extra.append("e_ar_issued")
    if extra:
        mm = monthly[["company_id", "period"] + extra].copy()
        mm = mm.sort_values(["company_id", "period"])
        g = mm.groupby("company_id", sort=False)
        if "d_cust_hhi" in extra:
            mm["d_cust_hhi_lag3"] = g["d_cust_hhi"].shift(3)
        if "e_ar_issued" in extra:
            mm["e_ar_issued_lag1"] = g["e_ar_issued"].shift(1)
        m = m.merge(
            mm[["company_id", "period"] + [c for c in mm.columns if c.endswith("lag3") or c.endswith("lag1")]],
            on=["company_id", "period"],
            how="left",
        )

    def _y_block(df: pd.DataFrame, ycol: str, mask: pd.Series) -> dict:
        sl = df.loc[mask, ycol]
        lab = sl.dropna()
        n = int(len(lab))
        n_pos = int((lab == 1).sum())
        return {
            "n_labeled": n,
            "n_pos": n_pos,
            "base": _pct(n_pos, n),
            "n_cm": int(mask.sum()),
            "share_labeled": _pct(n, int(mask.sum())) if mask.sum() else float("nan"),
        }

    rows = []
    lag_rows = []
    tr = m[m["split"] == "train"]
    ho = m[m["split"] == "holdout"]

    for ycol in Y_COLS:
        if ycol not in tr.columns:
            continue
        labeled = tr[ycol].notna()
        # Is this Y only defined on long trails?
        sofar = tr.loc[labeled, "months_so_far"]
        n_lab = int(labeled.sum())
        defined_note = ""
        if n_lab and float(sofar.min()) >= 18:
            defined_note = "ONLY_LONG (>=18m so-far on every labeled row)"
        elif n_lab and float(sofar.min()) >= 12:
            defined_note = f"no short-trail labels (min so-far={int(sofar.min())})"
        else:
            defined_note = f"min so-far on labeled={int(sofar.min()) if n_lab else -1}"

        for cls, cm in (
            ("all_train", pd.Series(True, index=tr.index)),
            ("short_<12_sofar", tr["so_far_class"] == "short_<12"),
            ("mid_12_17_sofar", tr["so_far_class"] == "mid_12_17"),
            ("long_>=18_sofar", tr["so_far_class"] == "long_>=18"),
            ("short_<12_company", tr["co_class"] == "short_<12"),
            ("long_>=18_company", tr["co_class"] == "long_>=18"),
        ):
            blk = _y_block(tr, ycol, cm)
            blk.update({"y": ycol, "slice": cls, "note": defined_note})
            rows.append(blk)

        # Holdout coverage only
        for cls, cm in (
            ("all_holdout", pd.Series(True, index=ho.index)),
            ("short_<12_sofar", ho["so_far_class"] == "short_<12"),
            ("long_>=18_sofar", ho["so_far_class"] == "long_>=18"),
        ):
            sl = ho.loc[cm, ycol]
            lag_rows.append(
                {
                    "y": ycol,
                    "slice": cls,
                    "n_labeled": int(sl.notna().sum()),
                    "n_pos": int((sl.dropna() == 1).sum()),
                    "note": "LOW_POWER coverage",
                }
            )

    # Q6 lag coverage among labeled train rows, by so-far bucket.
    q6 = []
    for ycol in Y_COLS:
        if ycol not in tr.columns:
            continue
        lab = tr[tr[ycol].notna()]
        for bname, _, _ in BUCKETS + [("short_<12", 1, 11), ("long_>=18", 18, 24), ("all", 1, 24)]:
            if bname == "short_<12":
                sl = lab[lab["months_so_far"] < 12]
            elif bname == "long_>=18":
                sl = lab[lab["months_so_far"] >= 18]
            elif bname == "all":
                sl = lab
            else:
                sl = lab[lab["so_far_bucket"] == bname]
            rec = {
                "y": ycol,
                "bucket": bname,
                "n_labeled": int(len(sl)),
            }
            if "d_cust_hhi_lag3" in sl.columns:
                rec["hhi_lag3_nn"] = int(sl["d_cust_hhi_lag3"].notna().sum())
                rec["hhi_lag3_share"] = _pct(rec["hhi_lag3_nn"], rec["n_labeled"])
            if "e_ar_issued_lag1" in sl.columns:
                rec["issued_lag1_nn"] = int(sl["e_ar_issued_lag1"].notna().sum())
                rec["issued_lag1_share"] = _pct(rec["issued_lag1_nn"], rec["n_labeled"])
            q6.append(rec)

    # Definition floors: first period a Y is non-null, vs months_so_far.
    floors = []
    for ycol in Y_COLS:
        if ycol not in tr.columns:
            continue
        lab = tr[tr[ycol].notna()]
        if lab.empty:
            floors.append({"y": ycol, "min_so_far": None, "min_period": None, "n": 0})
            continue
        floors.append(
            {
                "y": ycol,
                "min_so_far": int(lab["months_so_far"].min()),
                "p50_so_far": float(lab["months_so_far"].median()),
                "min_period": str(lab["period"].min().date()),
                "n": int(len(lab)),
                "share_short": _pct((lab["months_so_far"] < 12).sum(), len(lab)),
                "share_long": _pct((lab["months_so_far"] >= 18).sum(), len(lab)),
            }
        )

    return {
        "panel": m,
        "rows": rows,
        "holdout_counts": lag_rows,
        "q6": q6,
        "floors": floors,
    }


# ---------------------------------------------------------------------------
# Pass 8 — group-level: is a short trail a group onboarding wave?
# ---------------------------------------------------------------------------


def pass8_group_wave(bank: pd.DataFrame) -> dict:
    """Train groups: same-month arrival = wave; staggered = subsidiaries over time."""
    tr = bank[bank["split"] == "train"].copy()
    # Holdout siblings only inform the all-member group clock (already on bank).
    rows = []
    for gid, g in tr.groupby("group_id"):
        n = int(len(g))
        n_late = int(g["late_first_tx"].fillna(False).sum())
        months = g["first_tx_month"].dropna()
        if months.empty:
            continue
        mn = months.min()
        mx = months.max()
        span = int((mx.year - mn.year) * 12 + (mx.month - mn.month))
        n_unique = int(months.nunique())
        same = n_unique == 1
        if n_late == 0:
            kind = "all_full_2024_09"
        elif n_late == n and same:
            kind = "all_late_same_month"
        elif n_late == n:
            kind = "all_late_staggered"
        else:
            kind = "mixed"
        rows.append(
            {
                "group_id": gid,
                "n_train": n,
                "n_late": n_late,
                "share_late": _pct(n_late, n),
                "first_min": mn,
                "first_max": mx,
                "span_m": span,
                "n_unique_months": n_unique,
                "same_month": same,
                "kind": kind,
                "solo": n == 1,
            }
        )
    gdf = pd.DataFrame(rows)
    kinds = {k: int((gdf["kind"] == k).sum()) for k in sorted(gdf["kind"].unique())}
    # Companies sitting in each kind of group
    kind_map = gdf.set_index("group_id")["kind"]
    tr = tr.merge(gdf[["group_id", "kind", "same_month", "span_m"]], on="group_id", how="left")
    late = tr[tr["late_first_tx"].fillna(False)]
    in_wave = late["kind"] == "all_late_same_month"
    in_stag = late["kind"] == "all_late_staggered"
    in_mix = late["kind"] == "mixed"

    # Calendar of group first appearance (train min)
    appear = (
        gdf.groupby(gdf["first_min"].dt.strftime("%Y-%m"))
        .agg(n_groups=("group_id", "size"), n_cos=("n_train", "sum"))
        .reset_index()
        .rename(columns={"first_min": "month"})
    )
    appear_rows = appear.to_dict("records")

    # Big arrival months: 2025-01 and 2026-01 (from pass 1 hist)
    def _month_block(ts: pd.Timestamp) -> dict:
        cos = tr[tr["first_tx_month"] == ts]
        gids = set(cos["group_id"])
        gg = gdf[gdf["group_id"].isin(gids)]
        return {
            "month": str(ts.date())[:7],
            "n_cos": int(len(cos)),
            "n_groups": int(len(gids)),
            "n_groups_first_appear": int((gdf["first_min"] == ts).sum()),
            "n_groups_same_month": int(gg["same_month"].sum()),
            "share_cos_in_same_month_group": _pct(
                cos["same_month"].sum(), len(cos)
            ),
            "n_solo": int(gg["solo"].sum()),
        }

    waves = [
        _month_block(pd.Timestamp("2025-01-01")),
        _month_block(pd.Timestamp("2026-01-01")),
        _month_block(pd.Timestamp("2026-02-01")),
        _month_block(pd.Timestamp("2024-09-01")),
    ]

    # Old groups that added a late subsidiary
    old_mixed = gdf[gdf["kind"] == "mixed"]
    n_old_added = int(len(old_mixed))
    n_subs_in_mixed = int(late[late["kind"] == "mixed"].shape[0])

    md = [
        "",
        "## Pass 8 — group onboarding wave?",
        "",
        "A short trail is mostly a **new group on the panel** (85.5% of late "
        "companies), not a new subsidiary inside an old book (14.5%). It is "
        "**not** one same-month onboarding wave: only 28.5% of late companies sit "
        "in groups where every train member shares the first-tx month. The modal "
        "pattern is `all_late_staggered` (57%) — the holding arrives after 2024-09 "
        "and members trickle in. `all_late_same_month` = every train member shares "
        "the same first-tx month. `mixed` = at least one 2024-09 starter and at "
        "least one late member.",
        "",
        f"Train groups **{int(len(gdf))}**: "
        + ", ".join(f"{k}={v}" for k, v in kinds.items())
        + f". Of {int(len(late)):,} late companies: "
        f"in same-month new groups **{int(in_wave.sum()):,} ({_pp(_pct(in_wave.sum(), len(late)))})**, "
        f"in staggered new groups **{int(in_stag.sum()):,} ({_pp(_pct(in_stag.sum(), len(late)))})**, "
        f"in mixed (new subsidiary) **{int(in_mix.sum()):,} ({_pp(_pct(in_mix.sum(), len(late)))})**. "
        f"Old groups that added anyone: {n_old_added} (those {n_subs_in_mixed} late members).",
        "",
        "Group first-appearance calendar (train min first-tx):",
        "",
        _md_table(
            [
                {
                    "month": r["month"],
                    "n_groups": f"{int(r['n_groups']):,}",
                    "n_train_cos": f"{int(r['n_cos']):,}",
                }
                for r in appear_rows
            ]
        ),
        "",
        "Named arrival months (company first-tx = that month):",
        "",
        _md_table(
            [
                {
                    "month": w["month"],
                    "n_cos": f"{w['n_cos']:,}",
                    "n_groups": f"{w['n_groups']:,}",
                    "groups first appear": f"{w['n_groups_first_appear']:,}",
                    "groups same-month": f"{w['n_groups_same_month']:,}",
                    "cos in same-month group": _pp(w["share_cos_in_same_month_group"]),
                    "solo groups": f"{w['n_solo']:,}",
                }
                for w in waves
            ]
        ),
    ]
    same_late = gdf[gdf["kind"] == "all_late_same_month"]
    if len(same_late):
        cal = (
            same_late.groupby(same_late["first_min"].dt.strftime("%Y-%m"))
            .agg(n_groups=("group_id", "size"), n_cos=("n_train", "sum"), n_solo=("solo", "sum"))
            .reset_index()
            .rename(columns={"first_min": "month"})
        )
        n_solo_wave = int(same_late["solo"].sum())
        n_multi_wave = int((~same_late["solo"]).sum())
        md += [
            "",
            f"`all_late_same_month` groups: **{int(len(same_late))}** "
            f"(solo {n_solo_wave}, multi-member {n_multi_wave}). "
            "If short trail were one onboarding wave, these would pile in one month. "
            "They do not — they are spread across 2024-10 → 2026-05. "
            f"Solo {n_solo_wave}/85; multi-member {n_multi_wave}/85. "
            "The two fattest months are 2026-01 (19 groups) and 2026-02 (12). "
            "That is late-calendar arrival, not one panel-start wave.",
            "",
            _md_table(
                [
                    {
                        "month": r["month"],
                        "n_groups": f"{int(r['n_groups']):,}",
                        "n_train_cos": f"{int(r['n_cos']):,}",
                        "solo": f"{int(r['n_solo']):,}",
                    }
                    for r in cal.to_dict("records")
                ]
            ),
        ]
    return {
        "kinds": kinds,
        "n_groups": int(len(gdf)),
        "n_late": int(len(late)),
        "n_late_wave": int(in_wave.sum()),
        "share_late_wave": _pct(in_wave.sum(), len(late)),
        "n_late_staggered": int(in_stag.sum()),
        "share_late_staggered": _pct(in_stag.sum(), len(late)),
        "n_late_mixed": int(in_mix.sum()),
        "share_late_mixed": _pct(in_mix.sum(), len(late)),
        "n_old_groups_added": n_old_added,
        "waves": waves,
        "appear": appear_rows,
        "md": md,
    }


# ---------------------------------------------------------------------------
# Pass 9 — family G created_* constants vs the 73.6% fact
# ---------------------------------------------------------------------------


def pass9_g_family(bank: pd.DataFrame, monthly: pd.DataFrame) -> dict:
    """Are feature-report g_created_* CONSTANTs the same fact as 73.6%?

    No. Those columns flag null created_at and post-snapshot products.
    Banking created_at is never null; the monthly cut drops created_at >= 2026-09,
    so both snapshot flags are zero. The 73.6% late-connection fact shows up as
    g_n_accounts = 0 (and NaN shares) in early months, plus g_new_this_month.
    """
    gcols = [
        "g_n_accounts",
        "g_new_this_month",
        "g_created_unknown_share",
        "g_created_after_snapshot",
        "g_created_after_snapshot_share",
    ]
    missing = [c for c in gcols if c not in monthly.columns]
    if missing:
        return {"md": ["", f"## Pass 9 — family G (missing {missing})", ""], "ok": False}

    m = monthly[["company_id", "period", "split"] + gcols].copy()
    keys = bank[
        [
            "company_id",
            "split",
            "first_tx_month",
            "n_grid_months",
            "late_first_tx",
            "has_banking",
            "bank_created_after_0901",
            "bank_created_before_panel",
            "company_created_at",
            "first_tx",
        ]
    ].copy()
    m = m.merge(keys, on=["company_id", "split"], how="left")
    m["months_so_far"] = (
        (m["period"].dt.year - m["first_tx_month"].dt.year) * 12
        + (m["period"].dt.month - m["first_tx_month"].dt.month)
        + 1
    )
    tr = m[m["split"] == "train"]
    n_cm = int(len(tr))

    def _uniq(s: pd.Series) -> dict:
        nn = s.dropna()
        return {
            "n_nonnull": int(len(nn)),
            "share_nonnull": _pct(len(nn), len(s)),
            "n_unique": int(nn.nunique()),
            "modal": float(nn.mode().iloc[0]) if len(nn) else float("nan"),
            "share_modal": _pct((nn == nn.mode().iloc[0]).sum(), len(nn)) if len(nn) else float("nan"),
        }

    u_unknown = _uniq(tr["g_created_unknown_share"])
    u_after = _uniq(tr["g_created_after_snapshot"])
    u_after_s = _uniq(tr["g_created_after_snapshot_share"])

    zero_acc = tr["g_n_accounts"] == 0
    n_zero = int(zero_acc.sum())
    # First month on grid
    first = tr[tr["months_so_far"] == 1]
    # Late-connection companies (join-QA 73.6%)
    late_c = tr["bank_created_after_0901"].fillna(False)
    early_c = tr["bank_created_before_panel"].fillna(False)

    # Company created_at vs panel / first tx (onboarding ≠ health)
    cos = bank[bank["split"] == "train"]
    co_after = cos["company_created_at"] > PANEL_START
    co_after_tx = (
        cos["company_created_at"].notna()
        & cos["first_tx"].notna()
        & (cos["company_created_at"] > cos["first_tx"])
    )

    # g_n_accounts==0 among late-created vs early-created companies
    n_zero_late = int((zero_acc & late_c).sum())
    n_late_cm = int(late_c.sum())
    n_zero_early = int((zero_acc & early_c).sum())
    n_early_cm = int(early_c.sum())

    rec = {
        "n_cm": n_cm,
        "unknown": u_unknown,
        "after": u_after,
        "after_share": u_after_s,
        "n_g_n_accounts_0": n_zero,
        "share_g_n_accounts_0": _pct(n_zero, n_cm),
        "n_g_new": int((tr["g_new_this_month"] > 0).sum()),
        "share_g_new": _pct((tr["g_new_this_month"] > 0).sum(), n_cm),
        "first_n": int(len(first)),
        "first_n_accounts_0": int((first["g_n_accounts"] == 0).sum()),
        "first_share_0": _pct((first["g_n_accounts"] == 0).sum(), len(first)),
        "first_n_new": int((first["g_new_this_month"] > 0).sum()),
        "first_share_new": _pct((first["g_new_this_month"] > 0).sum(), len(first)),
        "zero_among_late_created_cm": _pct(n_zero_late, n_late_cm),
        "zero_among_early_created_cm": _pct(n_zero_early, n_early_cm),
        "n_zero_late": n_zero_late,
        "n_late_cm": n_late_cm,
        "n_company_created_after_0901": int(co_after.sum()),
        "share_company_created_after_0901": _pct(co_after.sum(), len(cos)),
        "n_company_created_after_first_tx": int(co_after_tx.sum()),
        "share_company_created_after_first_tx": _pct(co_after_tx.sum(), len(cos)),
    }

    # same-fact? CONSTANTS are 0/unique=1. 73.6% is a different column.
    same_fact = False
    rec["same_fact"] = same_fact

    md = [
        "",
        "## Pass 9 — are `g_created_*` constants this same 73.6% fact?",
        "",
        "Feature report flags `g_created_unknown_share`, `g_created_after_snapshot`, "
        "and `g_created_after_snapshot_share` as **CONSTANT**. That is **not** the "
        "73.6% left-truncation. Those columns measure *null* `created_at` and "
        "*post-snapshot* products. `clean.banking_products.created_at` has **0 nulls**; "
        "the monthly as-of cut is `created_at < period_next` and the last period is "
        "2026-08, so post-2026-09-01 products never enter the panel. The constants "
        "are 'connection metadata is complete and extract-dated rows are dropped'.",
        "",
        "The 73.6% fact (first banking `created_at` after 2024-09-01) shows up as "
        "`g_n_accounts = 0` on early company-months and as `g_new_this_month` when "
        "the first product crosses `period_next`. Feature-report coverage of "
        "`g_created_after_snapshot_share` (87.6%) is 1 − share(`g_n_accounts=0`).",
        "",
        _md_table(
            [
                {
                    "column": "g_created_unknown_share",
                    "cm non-null": _pp(u_unknown["share_nonnull"]),
                    "n unique": str(u_unknown["n_unique"]),
                    "modal": _f(u_unknown["modal"], 4),
                    "modal%": _pp(u_unknown["share_modal"]),
                    "same as 73.6%?": "no",
                },
                {
                    "column": "g_created_after_snapshot",
                    "cm non-null": _pp(u_after["share_nonnull"]),
                    "n unique": str(u_after["n_unique"]),
                    "modal": _f(u_after["modal"], 4),
                    "modal%": _pp(u_after["share_modal"]),
                    "same as 73.6%?": "no",
                },
                {
                    "column": "g_created_after_snapshot_share",
                    "cm non-null": _pp(u_after_s["share_nonnull"]),
                    "n unique": str(u_after_s["n_unique"]),
                    "modal": _f(u_after_s["modal"], 4),
                    "modal%": _pp(u_after_s["share_modal"]),
                    "same as 73.6%?": "no",
                },
            ]
        ),
        "",
        f"Train company-months with `g_n_accounts=0`: **{n_zero:,} / {n_cm:,} "
        f"({_pp(rec['share_g_n_accounts_0'])})**. "
        f"Among late-created books: {_pp(rec['zero_among_late_created_cm'])} of their "
        f"cm ({n_zero_late:,}/{n_late_cm:,}). Among products older than 2024-09: "
        f"{_pp(rec['zero_among_early_created_cm'])} ({n_zero_early:,}/{n_early_cm:,}) "
        f"— early books already have as-of inventory on month 1. "
        f"`g_new_this_month>0`: {rec['n_g_new']:,} cm ({_pp(rec['share_g_new'])}).",
        "",
        f"First month on the grid (n={rec['first_n']:,}): "
        f"`g_n_accounts=0` {_pp(rec['first_share_0'])} ({rec['first_n_accounts_0']:,}); "
        f"`g_new_this_month>0` {_pp(rec['first_share_new'])} ({rec['first_n_new']:,}). "
        "A company can have txs in a month where family G still counts 0 accounts "
        "because `created_at` is a later *connection* than the cash movement.",
        "",
        f"`companies.created_at` after 2024-09-01: "
        f"{rec['n_company_created_after_0901']:,}/{int(len(cos)):,} = "
        f"{_pp(rec['share_company_created_after_0901'])}. "
        f"Company created after first tx: "
        f"{rec['n_company_created_after_first_tx']:,} "
        f"({_pp(rec['share_company_created_after_first_tx'])}) — platform onboarding "
        "can post-date the bank book. **PARK as a health Y** (onboarding ≠ 45→65).",
        "",
        "**Not a miss indicator.** Late-trail companies are not smaller "
        "(Δ median log1p(a_in3) ≈ 0, MW p>0.10). Y2/Y3 base rates are close on "
        "short vs long *company* trails. Do not invent a `y_short_trail` miss label. "
        "Trail length is a Q6 coverage gate.",
    ]
    rec["md"] = md
    rec["ok"] = True
    return rec


# ---------------------------------------------------------------------------
# Pass 10 — why Q6 lags are missing; connection lag after first tx
# ---------------------------------------------------------------------------


def pass10_lag_why(bank: pd.DataFrame, monthly: pd.DataFrame, p6: dict) -> dict:
    """Decompose null d_cust_hhi_lag3 / e_ar_issued_lag1 on train labeled rows.

    Reasons (exclusive order):
    1. company-month so-far < 4 (shift-3 needs 3 prior panel rows)
    2. calendar month in 2024-09..2025-01 (family D full6 false → HHI itself NaN
       until the 6-month window sits inside the extract)
    3. d_cust_hhi / e_ar_issued is NaN at t-lag (no invoice book in the window)
    4. residual
    Also: months from first tx until first g_n_accounts>0 (connection lag).
    """
    m = p6["panel"].copy()
    tr = m[m["split"] == "train"].copy()
    extra = monthly[["company_id", "period", "d_cust_hhi", "e_ar_issued", "g_n_accounts"]].copy()
    tr = tr.merge(extra, on=["company_id", "period"], how="left")
    tr = tr.sort_values(["company_id", "period"])
    g = tr.groupby("company_id", sort=False)
    if "d_cust_hhi" in tr.columns:
        tr["d_cust_hhi_at_lag3"] = g["d_cust_hhi"].shift(3)
    if "e_ar_issued" in tr.columns:
        tr["e_ar_issued_at_lag1"] = g["e_ar_issued"].shift(1)

    cal_full6_start = pd.Timestamp("2025-02-01")  # first month win6_start >= 2024-09

    def _decomp(lab: pd.DataFrame, lag_col: str, src_at_lag: str, min_so_far: int, lag_m: int) -> dict:
        n = int(len(lab))
        miss = lab[lag_col].isna() if lag_col in lab.columns else pd.Series(True, index=lab.index)
        n_miss = int(miss.sum())
        period_at_lag = lab["period"] - pd.DateOffset(months=lag_m)
        r_shift = miss & (lab["months_so_far"] < min_so_far)
        r_cal = miss & ~r_shift & (period_at_lag < cal_full6_start)
        src = lab[src_at_lag] if src_at_lag in lab.columns else pd.Series(np.nan, index=lab.index)
        r_src = miss & ~r_shift & ~r_cal & src.isna()
        r_rest = miss & ~r_shift & ~r_cal & ~r_src
        return {
            "n_labeled": n,
            "n_miss": n_miss,
            "share_miss": _pct(n_miss, n),
            "n_shift": int(r_shift.sum()),
            "share_shift": _pct(r_shift.sum(), n_miss) if n_miss else float("nan"),
            "n_cal": int(r_cal.sum()),
            "share_cal": _pct(r_cal.sum(), n_miss) if n_miss else float("nan"),
            "n_src": int(r_src.sum()),
            "share_src": _pct(r_src.sum(), n_miss) if n_miss else float("nan"),
            "n_rest": int(r_rest.sum()),
            "share_rest": _pct(r_rest.sum(), n_miss) if n_miss else float("nan"),
        }

    blocks = {}
    for ycol, lag, src_at, k, lag_m in (
        ("y4_ds_r_double", "d_cust_hhi_lag3", "d_cust_hhi_at_lag3", 4, 3),
        ("y7_top1_lost", "e_ar_issued_lag1", "e_ar_issued_at_lag1", 2, 1),
        ("y7_top1_lost", "d_cust_hhi_lag3", "d_cust_hhi_at_lag3", 4, 3),
        ("y3_recover_cash_6m", "d_cust_hhi_lag3", "d_cust_hhi_at_lag3", 4, 3),
    ):
        if ycol not in tr.columns or lag not in tr.columns:
            continue
        lab = tr[tr[ycol].notna()]
        short = lab[lab["months_so_far"] < 12]
        long = lab[lab["months_so_far"] >= 18]
        key = f"{ycol}::{lag}"
        blocks[key] = {
            "all": _decomp(lab, lag, src_at, k, lag_m),
            "short": _decomp(short, lag, src_at, k, lag_m),
            "long": _decomp(long, lag, src_at, k, lag_m),
        }

    # Connection lag: first period with g_n_accounts>0 minus first_tx_month
    mm = monthly[monthly["split"] == "train"][["company_id", "period", "g_n_accounts"]].copy()
    first_acc = (
        mm[mm["g_n_accounts"] > 0]
        .groupby("company_id")["period"]
        .min()
        .rename("first_g_account_month")
    )
    cos = bank[bank["split"] == "train"].merge(first_acc, on="company_id", how="left")
    cos["conn_lag_m"] = (
        (cos["first_g_account_month"].dt.year - cos["first_tx_month"].dt.year) * 12
        + (cos["first_g_account_month"].dt.month - cos["first_tx_month"].dt.month)
    )
    # never gets an as-of account on the panel
    never = cos["first_g_account_month"].isna()
    late_c = cos["bank_created_after_0901"].fillna(False)
    cl = cos.loc[~never, "conn_lag_m"]
    cl_late = cos.loc[late_c & ~never, "conn_lag_m"]

    # Holdout months-on-book hist
    ho = bank[bank["split"] == "holdout"]
    ho_hist = {k: int((ho["n_grid_months"] == k).sum()) for k in range(1, 25)}

    # Gaps by company trail class
    gap_rows = []
    for name, mask in (
        ("<12", cos["n_grid_months"] < 12),
        (">=18", cos["n_grid_months"] >= 18),
        ("=24", cos["n_grid_months"] == 24),
    ):
        s = cos[mask]
        gap_rows.append(
            {
                "trail": name,
                "n": int(len(s)),
                "share_any_gap": _pct((s["n_gap_months"] > 0).sum(), len(s)),
                "mean_gap": float(s["n_gap_months"].mean()) if len(s) else float("nan"),
            }
        )

    md = [
        "",
        "## Pass 10 — why the Q6 lags are missing",
        "",
        "Null `d_cust_hhi_lag3` is not one hole. Exclusive reasons among missing "
        "labeled rows, using the source value *at t−k*: (1) `months_so_far < 4` — "
        "the panel shift has no t−3 row; (2) calendar at t−3 still in 2024-09..2025-01 "
        "— family D `full6` is false; (3) source at t−k is NaN (no invoice HHI / "
        "never-ERP); (4) residual (should be ~0 after (3) uses the lagged source). "
        "Y7 `issued_lag1` missing is **only** (1). Y4 HHI missing is mostly (3)+(2), "
        "not a pure short-trail shift — long-trail Y4 labels still miss HHI_lag3 on "
        "~47%, all source-NaN.",
        "",
    ]
    drows = []
    for key, blk in blocks.items():
        ycol, lag = key.split("::")
        for sl in ("all", "short", "long"):
            b = blk[sl]
            drows.append(
                {
                    "Y": Y_LABEL.get(ycol, ycol),
                    "lag": lag,
                    "slice": sl,
                    "n_lab": f"{b['n_labeled']:,}",
                    "missing": f"{b['n_miss']:,} ({_pp(b['share_miss'])})",
                    "shift<k": f"{b['n_shift']:,} ({_pp(b['share_shift'])} of miss)",
                    "calendar full6": f"{b['n_cal']:,} ({_pp(b['share_cal'])} of miss)",
                    "source NaN": f"{b['n_src']:,} ({_pp(b['share_src'])} of miss)",
                    "residual": f"{b['n_rest']:,} ({_pp(b['share_rest'])} of miss)",
                }
            )
    md.append(_md_table(drows))
    md += [
        "",
        f"**Connection lag** (months from first tx to first `g_n_accounts>0`, train): "
        f"never as-of account on panel = {int(never.sum()):,} / {int(len(cos)):,}. "
        f"Among those who connect: median {float(cl.median()) if len(cl) else float('nan'):.0f} "
        f"months, mean {float(cl.mean()) if len(cl) else float('nan'):.1f}, "
        f"share 0 (connected at/before first tx month) = {_pp(_pct((cl <= 0).sum(), len(cl)))}. "
        f"Late-created books only: median {float(cl_late.median()) if len(cl_late) else float('nan'):.0f}, "
        f"mean {float(cl_late.mean()) if len(cl_late) else float('nan'):.1f}, "
        f"n={int(len(cl_late)):,}.",
        "",
        "Holdout months-on-book (coverage only):",
        "",
        _md_table(
            [
                {
                    "months": str(k),
                    "n": str(ho_hist[k]),
                    "share": _pp(_pct(ho_hist[k], int(len(ho)))),
                }
                for k in range(1, 25)
                if ho_hist[k]
            ]
        ),
        "",
        "Silent months (grid month with 0 txs) by company trail — gaps are not "
        "the short-trail story:",
        "",
        _md_table(
            [
                {
                    "trail": r["trail"],
                    "n_cos": f"{r['n']:,}",
                    "any gap": _pp(r["share_any_gap"]),
                    "mean gap months": _f(r["mean_gap"], 2),
                }
                for r in gap_rows
            ]
        ),
    ]
    return {
        "blocks": blocks,
        "n_never_g": int(never.sum()),
        "n_train": int(len(cos)),
        "conn_lag_median": float(cl.median()) if len(cl) else float("nan"),
        "conn_lag_mean": float(cl.mean()) if len(cl) else float("nan"),
        "conn_lag_late_median": float(cl_late.median()) if len(cl_late) else float("nan"),
        "share_conn_0": _pct((cl <= 0).sum(), len(cl)) if len(cl) else float("nan"),
        "gap_rows": gap_rows,
        "md": md,
    }


# ---------------------------------------------------------------------------
# Pass 11 — right-censor, ERP × trail, honest lead length
# ---------------------------------------------------------------------------


def pass11_right_erp_lead(bank: pd.DataFrame, p6: dict) -> dict:
    """Last tx before panel end; ever-ERP by trail bucket; max honest Q6 lead."""
    tr = bank[bank["split"] == "train"].copy()
    quiet = (
        tr["last_tx_month_on_grid"].notna()
        & (tr["last_tx_month_on_grid"] < PANEL_END)
    )
    # last_grid_month should be PANEL_END for everyone on the official grid
    # who started early enough; quiet = last tx month < 2026-08.
    erp_rows = []
    for name, lo, hi in BUCKETS:
        s = tr[tr["trail_bucket"] == name]
        erp_rows.append(
            {
                "bucket": name,
                "n": int(len(s)),
                "ever_erp": int(s["ever_erp"].sum()),
                "share_erp": _pct(s["ever_erp"].sum(), len(s)),
                "n_quiet": int(quiet.reindex(s.index).fillna(False).sum()) if len(s) else 0,
                "share_quiet": _pct(int(quiet.reindex(s.index).fillna(False).sum()), len(s)),
            }
        )

    cm = p6["panel"]
    cmtr = cm[cm["split"] == "train"]
    lead_rows = []
    for ycol, lag, k in (
        ("y4_ds_r_double", "d_cust_hhi_lag3", 3),
        ("y7_top1_lost", "e_ar_issued_lag1", 1),
        ("y3_recover_cash_6m", "d_cust_hhi_lag3", 3),
        ("y2_neg_2of3", "d_cust_hhi_lag3", 3),
    ):
        if ycol not in cmtr.columns or lag not in cmtr.columns:
            continue
        lab = cmtr[cmtr[ycol].notna()].copy()
        lab["honest"] = lab["months_so_far"] - k
        nn = lab[lag].notna()
        lead_rows.append(
            {
                "y": ycol,
                "lag": lag,
                "k": k,
                "n_lab": int(len(lab)),
                "share_lag_nn": _pct(nn.sum(), len(lab)),
                "share_sofar_ge_k": _pct((lab["months_so_far"] > k).sum(), len(lab)),
                "p50_honest_all": float(lab["honest"].median()),
                "p50_honest_when_nn": float(lab.loc[nn, "honest"].median()) if nn.any() else float("nan"),
                "share_honest_ge3": _pct((lab.loc[nn, "honest"] >= 3).sum(), int(nn.sum())) if nn.any() else float("nan"),
                "share_honest_ge6": _pct((lab.loc[nn, "honest"] >= 6).sum(), int(nn.sum())) if nn.any() else float("nan"),
            }
        )

    # Honest lead on short vs long so-far (train labeled, lag present).
    trail_lead = []
    for ycol, lag, k in (
        ("y4_ds_r_double", "d_cust_hhi_lag3", 3),
        ("y7_top1_lost", "e_ar_issued_lag1", 1),
    ):
        if ycol not in cmtr.columns or lag not in cmtr.columns:
            continue
        lab = cmtr[cmtr[ycol].notna()].copy()
        lab["honest"] = lab["months_so_far"] - k
        for sl, mask in (
            ("short_<12", lab["months_so_far"] < 12),
            ("long_>=18", lab["months_so_far"] >= 18),
        ):
            slc = lab[mask]
            nn = slc[lag].notna()
            trail_lead.append(
                {
                    "y": ycol,
                    "slice": sl,
                    "n_lab": int(len(slc)),
                    "n_nn": int(nn.sum()),
                    "p50_honest_nn": float(slc.loc[nn, "honest"].median()) if nn.any() else float("nan"),
                    "share_ge6": _pct((slc.loc[nn, "honest"] >= 6).sum(), int(nn.sum())) if nn.any() else float("nan"),
                }
            )

    n_quiet = int(quiet.sum())
    md = [
        "",
        "## Pass 11 — right-censor, ERP × trail, honest lead length",
        "",
        f"Train companies whose last on-grid tx month is before 2026-08: "
        f"**{n_quiet:,} / {int(len(tr)):,} ({_pp(_pct(n_quiet, len(tr)))})**. "
        "That is a quiet tail, not left-truncation. Short-trail companies are "
        "left-truncated (they *start* late); they are not the ones going silent.",
        "",
        "Ever-ERP and quiet-tail by company trail bucket (train):",
        "",
        _md_table(
            [
                {
                    "bucket": r["bucket"],
                    "n": f"{r['n']:,}",
                    "ever-ERP": f"{r['ever_erp']:,} ({_pp(r['share_erp'])})",
                    "last tx < 2026-08": f"{r['n_quiet']:,} ({_pp(r['share_quiet'])})",
                }
                for r in erp_rows
            ]
        ),
        "",
        "Honest lead among **train labeled rows where the lag is non-null**: "
        "`honest = months_so_far − k` (k=3 for HHI_lag3, k=1 for issued_lag1). "
        "A 'visible 6 months earlier' sentence needs honest ≥ 6 *and* the lag present.",
        "",
        _md_table(
            [
                {
                    "Y": Y_LABEL.get(r["y"], r["y"]),
                    "lag": r["lag"],
                    "n_lab": f"{r['n_lab']:,}",
                    "lag non-null": _pp(r["share_lag_nn"]),
                    "so-far > k": _pp(r["share_sofar_ge_k"]),
                    "p50 honest given nn": _f(r["p50_honest_when_nn"], 1),
                    "honest ge3 given nn": _pp(r["share_honest_ge3"]),
                    "honest ge6 given nn": _pp(r["share_honest_ge6"]),
                }
                for r in lead_rows
            ]
        ),
        "",
        "Same honest length, short vs long so-far (train labeled, lag present). "
        "A short row with HHI_lag3 can still say 'visible 6 months earlier' if "
        "so-far ≥ 9. If `honest ge6` collapses on short, the Q6 sentence is a "
        "long-trail privilege even among rows that have the lag.",
        "",
        _md_table(
            [
                {
                    "Y": Y_LABEL.get(r["y"], r["y"]),
                    "slice": r["slice"],
                    "n_lab": f"{r['n_lab']:,}",
                    "lag nn": f"{r['n_nn']:,}",
                    "p50 honest given nn": _f(r["p50_honest_nn"], 1),
                    "honest ge6 given nn": _pp(r["share_ge6"]),
                }
                for r in trail_lead
            ]
        ),
    ]
    full24 = set(tr.loc[tr["n_grid_months"] == 24, "company_id"])
    co_cov = []
    for ycol, lag in (
        ("y4_ds_r_double", "d_cust_hhi_lag3"),
        ("y7_top1_lost", "e_ar_issued_lag1"),
    ):
        if ycol not in cmtr.columns or lag not in cmtr.columns:
            continue
        lab = cmtr[cmtr[ycol].notna()]
        ever_lab = set(lab["company_id"])
        ever_nn = set(lab.loc[lab[lag].notna(), "company_id"])
        n24 = int(len(full24))
        n24_lab = int(len(full24 & ever_lab))
        n24_nn = int(len(full24 & ever_nn))
        co_cov.append(
            {
                "y": ycol,
                "n24": n24,
                "n24_lab": n24_lab,
                "n24_nn": n24_nn,
                "share_lab": _pct(n24_lab, n24),
                "share_nn": _pct(n24_nn, n24),
            }
        )
    if co_cov:
        md += [
            "",
            "Company-level Q6 on the **435** 24-month train books: share that ever "
            "have a labeled row, and that ever have the lag on a labeled row. "
            "A 24-month book does not automatically give you a 3-month HHI lead.",
            "",
            _md_table(
                [
                    {
                        "Y": Y_LABEL.get(r["y"], r["y"]),
                        "24m books": f"{r['n24']:,}",
                        "ever labeled": f"{r['n24_lab']:,} ({_pp(r['share_lab'])})",
                        "ever labeled+lag": f"{r['n24_nn']:,} ({_pp(r['share_nn'])})",
                    }
                    for r in co_cov
                ]
            ),
        ]
    return {
        "n_quiet": n_quiet,
        "share_quiet": _pct(n_quiet, len(tr)),
        "erp_rows": erp_rows,
        "lead_rows": lead_rows,
        "trail_lead": trail_lead,
        "co_cov": co_cov,
        "md": md,
    }


# ---------------------------------------------------------------------------
# Pass 12 — g_n_accounts=0 months still have cash (connection ≠ empty book)
# ---------------------------------------------------------------------------


def pass12_zero_accounts_have_tx(bank: pd.DataFrame, monthly: pd.DataFrame, txm: pd.DataFrame) -> dict:
    """Months with g_n_accounts=0 are still on the bank trail — txs exist.

    Family G is a connection inventory. A zero account count is not a missing
    cash book. PARK using that zero as a health / miss flag.
    """
    m = monthly[monthly["split"] == "train"][
        ["company_id", "period", "g_n_accounts", "a_in3"]
    ].copy()
    t = txm.rename(columns={"month": "period"})
    t["period"] = pd.to_datetime(t["period"])
    m = m.merge(t[["company_id", "period", "n_tx"]], on=["company_id", "period"], how="left")
    m["n_tx"] = m["n_tx"].fillna(0).astype(int)
    z = m[m["g_n_accounts"] == 0]
    nz = m[m["g_n_accounts"] > 0]
    n_z = int(len(z))
    n_z_tx = int((z["n_tx"] > 0).sum())
    # The 3 companies with no banking product
    no_b = bank[(bank["split"] == "train") & (~bank["has_banking"])]
    never_g = (
        m.groupby("company_id")["g_n_accounts"].max()
    )
    never_ids = never_g[never_g == 0].index.astype(str).tolist()

    md = [
        "",
        "## Pass 12 — `g_n_accounts=0` months still have transactions",
        "",
        f"Train cm with `g_n_accounts=0`: **{n_z:,}**. Of those, **{n_z_tx:,} "
        f"({_pp(_pct(n_z_tx, n_z))})** have ≥1 tx that month. Median n_tx on "
        f"zero-account months = {float(z['n_tx'].median()) if n_z else float('nan'):.0f} "
        f"(mean {float(z['n_tx'].mean()) if n_z else float('nan'):.1f}) vs "
        f"median {float(nz['n_tx'].median()) if len(nz) else float('nan'):.0f} "
        f"when `g_n_accounts>0`. The cash book is there; family G has not yet "
        f"counted a connected product (`created_at < period_next`).",
        "",
        f"Train companies with no banking product row: {int(len(no_b))} "
        f"{sorted(no_b['company_id'].tolist())}. "
        f"Never `g_n_accounts>0` on the panel: {len(never_ids)} {sorted(never_ids)}. "
        "Those are connection holes, not miss / health events. **PARK** as a Y.",
    ]
    ghost = bank[(bank["split"] == "train") & bank["company_id"].isin(never_ids)]
    if len(ghost):
        md += [
            "",
            "Never-inventory train companies (descriptive):",
            "",
            _md_table(
                [
                    {
                        "company": r["company_id"],
                        "group": r["group_id"],
                        "grid m": str(int(r["n_grid_months"])),
                        "late tx": str(bool(r["late_first_tx"])),
                        "ever-ERP": str(bool(r["ever_erp"])),
                        "has banking row": str(bool(r["has_banking"])),
                    }
                    for r in ghost.to_dict("records")
                ]
            ),
        ]
    return {
        "n_zero": n_z,
        "n_zero_with_tx": n_z_tx,
        "share_zero_with_tx": _pct(n_z_tx, n_z),
        "median_tx_zero": float(z["n_tx"].median()) if n_z else float("nan"),
        "median_tx_pos": float(nz["n_tx"].median()) if len(nz) else float("nan"),
        "n_no_banking": int(len(no_b)),
        "no_banking_ids": sorted(no_b["company_id"].tolist()),
        "never_g_ids": sorted(never_ids),
        "md": md,
    }


# ---------------------------------------------------------------------------
# Pass 13 — Y rates short vs long, ever-ERP vs 470 (confound check)
# ---------------------------------------------------------------------------


def pass13_y_by_erp(bank: pd.DataFrame, p6: dict) -> dict:
    """Company-total trail × ever-ERP. Rates on train labeled rows only."""
    # p6 panel already has ever_erp, co_class, so_far_class from pass 6.
    _ = bank
    tr = p6["panel"]
    tr = tr[tr["split"] == "train"]
    rows = []
    for ycol in Y_COLS:
        if ycol not in tr.columns:
            continue
        for erp_name, erp_mask in (
            ("ever_erp", tr["ever_erp"] == True),  # noqa: E712
            ("never_erp", tr["ever_erp"] == False),  # noqa: E712
        ):
            for trail_name, tmask in (
                ("short_<12_company", tr["co_class"] == "short_<12"),
                ("long_>=18_company", tr["co_class"] == "long_>=18"),
                ("short_<12_sofar", tr["so_far_class"] == "short_<12"),
                ("long_>=18_sofar", tr["so_far_class"] == "long_>=18"),
            ):
                mask = erp_mask & tmask
                sl = tr.loc[mask, ycol]
                lab = sl.dropna()
                rows.append(
                    {
                        "y": ycol,
                        "erp": erp_name,
                        "trail": trail_name,
                        "n_cm": int(mask.sum()),
                        "n_lab": int(len(lab)),
                        "n_pos": int((lab == 1).sum()),
                        "base": _pct((lab == 1).sum(), len(lab)),
                    }
                )
    md = [
        "",
        "## Pass 13 — short vs long Y rates, ever-ERP vs the 470",
        "",
        "The 12–17 month bucket is only 42.7% ever-ERP (pass 11), so a raw "
        "short-vs-long base-rate gap can be an ERP mix. Split here. Y7 is "
        "invoice-built: never-ERP labeled n should be 0. Y2/Y3/Y4 can exist on the 470. "
        "Read: Y3 is flat (~9% short vs ~7% long) on both ERP slices (short n is small). "
        "Y2 ever-ERP is slightly higher on short companies (7.7% vs 5.9%); Y2 never-ERP "
        "is the **reverse** (8.4% short vs 10.4% long). Y4 is flat. Y7 (ERP-only) is "
        "*lower* on short companies (22% vs 30%). Short trail is not a miss indicator.",
        "",
        _md_table(
            [
                {
                    "Y": Y_LABEL.get(r["y"], r["y"]),
                    "ERP": r["erp"],
                    "trail": r["trail"],
                    "n_labeled": f"{r['n_lab']:,}",
                    "n_pos": f"{r['n_pos']:,}",
                    "base": _pp(r["base"]),
                }
                for r in rows
            ]
        ),
    ]
    return {"rows": rows, "md": md}


# ---------------------------------------------------------------------------
# Pass 14 — post-snapshot-only books; invoice vs bank months by trail
# ---------------------------------------------------------------------------


def pass14_snapshot_inv(con, bank: pd.DataFrame) -> dict:
    """Why 2 companies have banking rows but never g_n_accounts>0."""
    post = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               COUNT(*) AS n_prod,
               COUNT(*) FILTER (WHERE created_at > TIMESTAMP '2026-09-01') AS n_after,
               MIN(created_at) AS first_c,
               MAX(created_at) AS last_c
        FROM banking_products
        GROUP BY 1
        HAVING MIN(created_at) > TIMESTAMP '2026-09-01'
        """
    ).df()
    post["company_id"] = post["company_id"].astype(str)
    hold = load_holdout()
    post["split"] = np.where(post["company_id"].isin(hold), "holdout", "train")
    post_tr = post[post["split"] == "train"]

    tr = bank[bank["split"] == "train"]
    inv_rows = []
    for name, _, _ in BUCKETS:
        s = tr[tr["trail_bucket"] == name]
        erp = s[s["ever_erp"]]
        inv_rows.append(
            {
                "bucket": name,
                "n": int(len(s)),
                "n_erp": int(len(erp)),
                "median_inv": float(erp["n_inv_months"].median()) if len(erp) else float("nan"),
                "median_bank": float(s["n_grid_months"].median()) if len(s) else float("nan"),
                "median_inv_minus_bank": float(erp["inv_minus_bank"].median()) if len(erp) else float("nan"),
                "share_inv_shorter": _pct((erp["n_inv_months"] < erp["n_grid_months"]).sum(), len(erp)),
            }
        )

    md = [
        "",
        "## Pass 14 — post-snapshot-only books; invoice vs bank by trail",
        "",
        f"Train companies whose *every* banking `created_at` is after 2026-09-01: "
        f"**{int(len(post_tr))}** {sorted(post_tr['company_id'].tolist())}. "
        "Family G drops them (`created_at < period_next` never holds on the 2024-09..2026-08 "
        "grid). Together with the 3 companies that have no banking row, that is the "
        "5 never-`g_n_accounts>0` list. Extract-dated inventory is not a 2024 event "
        "and not a health Y.",
        "",
        "Invoice-book months vs bank-grid months by trail bucket (train; medians on ever-ERP):",
        "",
        _md_table(
            [
                {
                    "bucket": r["bucket"],
                    "n": f"{r['n']:,}",
                    "ever-ERP": f"{r['n_erp']:,}",
                    "median inv months": _f(r["median_inv"], 1),
                    "median bank months": _f(r["median_bank"], 1),
                    "median inv−bank": _f(r["median_inv_minus_bank"], 1),
                    "ERP inv < bank": _pp(r["share_inv_shorter"]),
                }
                for r in inv_rows
            ]
        ),
        "",
        "## Design note (not a new Y)",
        "",
        "A `y_short_trail` / `y_late_created_at` flag is **not** a miss indicator. "
        "Late first-tx companies are not smaller; Y2/Y3/Y4 do not jump; Y7 is lower "
        "on short companies. `created_at` is onboarding / connection. **PARK** as a "
        "health label. Keep trail length as a Q6 *coverage* attribute: do not claim "
        "lead time longer than `months_so_far − k`, and do not quote Y4 HHI_lag3 "
        "as a 3-month lead on the 78% of short labeled rows where it is null.",
    ]
    return {
        "n_post_only": int(len(post_tr)),
        "post_ids": sorted(post_tr["company_id"].tolist()),
        "inv_rows": inv_rows,
        "md": md,
    }


# ---------------------------------------------------------------------------
# Pass 15 — invoice months *on the bank grid* vs before first tx
# ---------------------------------------------------------------------------


def pass15_inv_on_grid(
    bank: pd.DataFrame, monthly: pd.DataFrame, con
) -> dict:
    """n_inv_months in pass 3 counts any issuance month in 2024-09..2026-08.

    That can exceed n_grid_months when invoices exist *before* first tx
    (company not yet on monthly_grid). Clip issuance months to the official
    grid rows and count pre-grid invoice months separately. Y7 can see an
    invoice trail the cash panel does not start yet.
    """
    _ = monthly
    invm = con.execute(
        f"""
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               CAST(date_trunc('month', issuance_date) AS DATE) AS month
        FROM invoices
        WHERE {BOOK}
          AND issuance_date >= TIMESTAMP '2024-09-01'
          AND issuance_date < TIMESTAMP '2026-09-01'
        GROUP BY 1, 2
        """
    ).df()
    invm["company_id"] = invm["company_id"].astype(str)
    invm["month"] = pd.to_datetime(invm["month"])
    first = bank[["company_id", "split", "first_tx_month", "n_grid_months", "ever_erp", "trail_bucket"]].copy()
    invm = invm.merge(first, on="company_id", how="left")
    invm["pre_grid"] = invm["month"] < invm["first_tx_month"]
    invm["on_grid"] = (invm["month"] >= invm["first_tx_month"]) & (invm["month"] <= PANEL_END)

    agg = (
        invm.groupby("company_id")
        .agg(n_inv_pre=("pre_grid", "sum"), n_inv_on_grid=("on_grid", "sum"))
        .reset_index()
    )
    panel = first.merge(agg, on="company_id", how="left")
    panel["n_inv_pre"] = panel["n_inv_pre"].fillna(0).astype(int)
    panel["n_inv_on_grid"] = panel["n_inv_on_grid"].fillna(0).astype(int)

    tr = panel[panel["split"] == "train"]
    erp = tr[tr["ever_erp"]]
    n_pre = int((erp["n_inv_pre"] > 0).sum())
    # company created_at vs first_tx (same month?)
    cos = bank[bank["split"] == "train"]
    co_m = cos["company_created_at"].dt.to_period("M").dt.to_timestamp()
    same = (co_m == cos["first_tx_month"]) & cos["company_created_at"].notna()
    late_co = cos["late_first_tx"].fillna(False)

    rows = []
    for name, _, _ in BUCKETS:
        s = erp[erp["trail_bucket"] == name]
        rows.append(
            {
                "bucket": name,
                "n_erp": int(len(s)),
                "share_pre": _pct((s["n_inv_pre"] > 0).sum(), len(s)),
                "median_pre": float(s["n_inv_pre"].median()) if len(s) else float("nan"),
                "median_on_grid": float(s["n_inv_on_grid"].median()) if len(s) else float("nan"),
                "median_bank": float(s["n_grid_months"].median()) if len(s) else float("nan"),
                "share_on_lt_bank": _pct((s["n_inv_on_grid"] < s["n_grid_months"]).sum(), len(s)),
            }
        )

    md = [
        "",
        "## Pass 15 — invoice months on the bank grid (clip)",
        "",
        "Pass 3 `n_inv_months` counted every issuance month in 2024-09..2026-08, "
        "so a 2026-01 bank starter could show 13 invoice months vs 8 bank months. "
        "That is invoices *before first tx* (off `monthly_grid`), not a longer "
        "on-grid ERP book. Y7 / family E can see those earlier invoices; cash "
        "features cannot.",
        "",
        f"Train ever-ERP with ≥1 issuance month before first tx: "
        f"**{n_pre:,} / {int(len(erp)):,} ({_pp(_pct(n_pre, len(erp)))})**. "
        f"`companies.created_at` in the same month as first tx: "
        f"{int(same.sum()):,} / {int(len(cos)):,} ({_pp(_pct(same.sum(), len(cos)))}); "
        f"among late first-tx: {int((same & late_co).sum()):,} / {int(late_co.sum()):,} "
        f"({_pp(_pct((same & late_co).sum(), late_co.sum()))}). "
        "Platform onboarding and first cash month sometimes coincide; they are "
        "still not a health Y.",
        "",
        _md_table(
            [
                {
                    "bucket": r["bucket"],
                    "ever-ERP": f"{r['n_erp']:,}",
                    "share with pre-grid inv": _pp(r["share_pre"]),
                    "median pre-grid inv m": _f(r["median_pre"], 1),
                    "median on-grid inv m": _f(r["median_on_grid"], 1),
                    "median bank m": _f(r["median_bank"], 1),
                    "on-grid inv < bank": _pp(r["share_on_lt_bank"]),
                }
                for r in rows
            ]
        ),
    ]
    return {
        "n_erp_pre": n_pre,
        "n_erp": int(len(erp)),
        "share_pre": _pct(n_pre, len(erp)),
        "share_co_same_month": _pct(same.sum(), len(cos)),
        "rows": rows,
        "md": md,
    }


# ---------------------------------------------------------------------------
# Pass 16 — Y7 labels on month 1 ride pre-grid invoices
# ---------------------------------------------------------------------------


def pass16_y7_pregrid(p6: dict, p15: dict, monthly: pd.DataFrame) -> dict:
    """Y7 min so-far = 1 because the AR book is not clipped to first tx."""
    _ = p15
    tr = p6["panel"]
    tr = tr[tr["split"] == "train"]
    if "y7_top1_lost" not in tr.columns:
        return {"md": []}
    lab = tr[tr["y7_top1_lost"].notna()].copy()
    extra = monthly[["company_id", "period", "e_ar_issued"]].copy()
    lab = lab.merge(extra, on=["company_id", "period"], how="left")
    early = lab[lab["months_so_far"] <= 2]
    n_early = int(len(early))
    n_iss = int(early["e_ar_issued"].notna().sum()) if n_early else 0
    # so-far==1
    s1 = lab[lab["months_so_far"] == 1]
    n_s1 = int(len(s1))
    n_s1_iss = int(s1["e_ar_issued"].notna().sum()) if n_s1 else 0
    n_s1_lag = int(s1["e_ar_issued_lag1"].notna().sum()) if n_s1 and "e_ar_issued_lag1" in s1.columns else 0

    md = [
        "",
        "## Pass 16 — Y7 on the first bank months uses pre-grid invoices",
        "",
        f"Y7 labeled train rows with `months_so_far ≤ 2`: **{n_early:,}**. "
        f"`e_ar_issued` non-null on those rows: {n_iss:,} ({_pp(_pct(n_iss, n_early))}). "
        f"At so-far=1: n_labeled={n_s1:,}, `e_ar_issued` nn={n_s1_iss:,} "
        f"({_pp(_pct(n_s1_iss, n_s1))}), `e_ar_issued_lag1` nn={n_s1_lag:,} "
        f"(should be ~0 — no prior *grid* row). "
        "The label can fire on the first cash month because the trailing AR book "
        "is built from invoices, which 43.5% of ever-ERP companies already had "
        "before first tx. Q6 `issued_lag1` still needs a prior *panel* row, "
        "so the 1-month lead is missing exactly on those first months. "
        "Do not read Y7-at-so-far=1 as 'the cash trail was already long enough'.",
    ]
    return {
        "n_early": n_early,
        "share_iss_early": _pct(n_iss, n_early),
        "n_sofar1": n_s1,
        "share_iss_s1": _pct(n_s1_iss, n_s1),
        "n_lag_s1": n_s1_lag,
        "md": md,
    }


# ---------------------------------------------------------------------------
# Pass 17 — 2026-09 extract txs vs the August quiet tail
# ---------------------------------------------------------------------------


def pass17_sep_extract(con, bank: pd.DataFrame) -> dict:
    """Official grid ends 2026-08. 2026-09 txs exist (extract month)."""
    sep = con.execute(
        f"""
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               COUNT(*) AS n_sep
        FROM transactions
        WHERE "date" >= TIMESTAMP '{EXTRACT.date()}'
          AND "date" < TIMESTAMP '2026-10-01'
        GROUP BY 1
        """
    ).df()
    sep["company_id"] = sep["company_id"].astype(str)
    tr = bank[bank["split"] == "train"].merge(sep, on="company_id", how="left")
    tr["n_sep"] = tr["n_sep"].fillna(0).astype(int)
    tr["has_sep"] = tr["n_sep"] > 0
    quiet = tr["last_tx_month_on_grid"].notna() & (tr["last_tx_month_on_grid"] < PANEL_END)
    n_q = int(quiet.sum())
    n_q_sep = int((quiet & tr["has_sep"]).sum())
    n_sep = int(tr["has_sep"].sum())
    md = [
        "",
        "## Pass 17 — September 2026 extract txs vs the quiet tail",
        "",
        f"Train companies with ≥1 tx in 2026-09 (off the official grid): "
        f"**{n_sep:,} / {int(len(tr)):,} ({_pp(_pct(n_sep, len(tr)))})**. "
        f"Of the {n_q:,} with last *on-grid* tx before 2026-08, "
        f"**{n_q_sep:,} ({_pp(_pct(n_q_sep, n_q))})** still transact in the extract "
        f"month — almost none bounce back. The quiet tail is real silence through "
        f"extract, not an August-only hole. Still **PARK** as a death Y: 10% of "
        f"train books go dark before the last grid month, but that can be "
        f"seasonality / sample exit, and we did not build a miss label. "
        f"The grid stops at 2026-08 regardless.",
    ]
    last_hist = (
        tr.loc[quiet, "last_tx_month_on_grid"]
        .dt.strftime("%Y-%m")
        .value_counts()
        .sort_index()
        .to_dict()
    )
    aug_hole = quiet & (tr["last_tx_month_on_grid"] == pd.Timestamp("2026-07-01"))
    q = tr[quiet].copy()
    q["aug_hole"] = aug_hole.reindex(q.index).fillna(False)
    hole_rows = []
    for name, mask in (
        ("<12", q["n_grid_months"] < 12),
        (">=18", q["n_grid_months"] >= 18),
        ("=24", q["n_grid_months"] == 24),
    ):
        s = q[mask]
        hole_rows.append(
            {
                "trail": name,
                "n_quiet": int(len(s)),
                "n_aug_hole": int(s["aug_hole"].sum()),
                "share_aug_hole": _pct(s["aug_hole"].sum(), len(s)),
            }
        )
    n_hole = int(aug_hole.sum())
    md += [
        "",
        f"Last tx in 2026-07 (August-only hole, no Sep bounce): "
        f"**{n_hole:,} / {n_q:,} ({_pp(_pct(n_hole, n_q))})**. "
        "If those sit in =24 books, it is end-of-panel seasonality, not left-truncation.",
        "",
        _md_table(
            [
                {
                    "trail": r["trail"],
                    "quiet n": f"{r['n_quiet']:,}",
                    "2026-07 last tx": f"{r['n_aug_hole']:,} ({_pp(r['share_aug_hole'])})",
                }
                for r in hole_rows
            ]
        ),
        "",
        "Last on-grid tx month among the quiet tail (train):",
        "",
        _md_table(
            [
                {"last on-grid month": k, "n": f"{v:,}"}
                for k, v in last_hist.items()
            ]
        ),
    ]
    return {
        "n_sep": n_sep,
        "share_sep": _pct(n_sep, len(tr)),
        "n_quiet": n_q,
        "n_quiet_sep": n_q_sep,
        "share_quiet_sep": _pct(n_q_sep, n_q),
        "last_hist": last_hist,
        "n_aug_hole": n_hole,
        "share_aug_hole": _pct(n_hole, n_q),
        "hole_rows": hole_rows,
        "md": md,
    }


# ---------------------------------------------------------------------------
# Pass 18 — holdout groups are late arrivals (hidden-test Q6 cap)
# ---------------------------------------------------------------------------


def pass18_holdout_groups(bank: pd.DataFrame) -> dict:
    """Coverage only. If holdout groups start after 2024-09, Q6 will not transfer."""
    ho = bank[bank["split"] == "holdout"]
    tr = bank[bank["split"] == "train"]
    # Group clock from all members (already on panel as group_first_tx_all if pass5 ran).
    if "group_first_tx_all" not in bank.columns:
        gfirst = bank.groupby("group_id")["first_tx_month"].min()
        ho = ho.merge(gfirst.rename("group_first_tx_all"), on="group_id", how="left")
    n = int(len(ho))
    n_late = int(ho["late_first_tx"].fillna(False).sum())
    n_new_g = int((ho["group_first_tx_all"] > PANEL_START).sum())
    n_g = int(ho["group_id"].nunique())
    n_g_new = int(ho.loc[ho["group_first_tx_all"] > PANEL_START, "group_id"].nunique())
    n_g_old = n_g - n_g_new
    # train companies in those same holdout groups (siblings left in train)
    ho_g = set(ho["group_id"])
    sib = tr[tr["group_id"].isin(ho_g)]
    md = [
        "",
        "## Pass 18 — holdout is a late-arrival sample (coverage)",
        "",
        f"Holdout companies: {n}. Late first-tx: {n_late} ({_pp(_pct(n_late, n))}). "
        f"In a group whose earliest first-tx is after 2024-09: "
        f"**{n_new_g} ({_pp(_pct(n_new_g, n))})**. "
        f"Holdout groups: {n_g} (new {n_g_new}, already-on-panel {n_g_old}). "
        f"Train siblings in those holdout groups: {int(len(sib))} "
        f"(should be 0 if the 15 groups are fully held out).",
        "",
        "The frozen 72/15 split is short-trail by construction of who was sampled, "
        "not just LOW_POWER n. A 3-month HHI lead fitted on 24-month train books "
        "does not have a 24-month book to sit on in holdout. Quote train CV for "
        "Q6; treat holdout as a coverage check that the lead is even *defined*.",
    ]
    full24 = ho[ho["n_grid_months"] == 24]
    md += [
        "",
        f"Holdout companies with a 24-month book: **{int(len(full24))}** "
        f"{sorted(full24['company_id'].tolist())} in groups "
        f"{sorted(full24['group_id'].unique().tolist())}. "
        "Those are the only holdout rows where a 3-month lag is even in play "
        "for most of the panel.",
    ]
    return {
        "n": n,
        "n_late": n_late,
        "n_new_g": n_new_g,
        "n_groups": n_g,
        "n_groups_new": n_g_new,
        "n_train_sib": int(len(sib)),
        "md": md,
    }


def pass18b_holdout_lags(p6: dict, bank: pd.DataFrame | None = None) -> dict:
    """Holdout labeled lag presence — coverage only, no rate used as a cut."""
    ho = p6["panel"]
    ho = ho[ho["split"] == "holdout"]
    bank = bank if bank is not None else pd.DataFrame()
    rows = []
    for ycol, lag in (
        ("y4_ds_r_double", "d_cust_hhi_lag3"),
        ("y7_top1_lost", "e_ar_issued_lag1"),
        ("y3_recover_cash_6m", "d_cust_hhi_lag3"),
    ):
        if ycol not in ho.columns or lag not in ho.columns:
            continue
        lab = ho[ho[ycol].notna()]
        short = lab[lab["months_so_far"] < 12]
        rows.append(
            {
                "y": ycol,
                "lag": lag,
                "n_lab": int(len(lab)),
                "n_nn": int(lab[lag].notna().sum()),
                "n_lab_short": int(len(short)),
                "n_nn_short": int(short[lag].notna().sum()) if len(short) else 0,
            }
        )
    full24 = set(ho.loc[ho["n_grid_months"] == 24, "company_id"])
    y4 = ho[ho["y4_ds_r_double"].notna()] if "y4_ds_r_double" in ho.columns else ho.iloc[0:0]
    y4_nn = y4[y4["d_cust_hhi_lag3"].notna()] if len(y4) and "d_cust_hhi_lag3" in y4.columns else y4.iloc[0:0]
    n_from24 = int(y4_nn["company_id"].isin(full24).sum()) if len(y4_nn) else 0
    md = [
        "",
        "Holdout labeled lag presence (LOW_POWER counts, not a cut):",
        "",
        _md_table(
            [
                {
                    "Y": Y_LABEL.get(r["y"], r["y"]),
                    "lag": r["lag"],
                    "n labeled": f"{r['n_lab']:,}",
                    "lag nn": f"{r['n_nn']:,}",
                    "short labeled": f"{r['n_lab_short']:,}",
                    "short lag nn": f"{r['n_nn_short']:,}",
                }
                for r in rows
            ]
        ),
        "",
        f"Of the {int(len(y4_nn))} holdout Y4 rows with `d_cust_hhi_lag3` non-null, "
        f"**{n_from24}** sit on the three 24-month books "
        f"{sorted(full24) if full24 else []}. "
        f"The rest ({int(len(y4_nn)) - n_from24}) are late books that still "
        "have a 3-month HHI history — rare, not the typical holdout month.",
    ]
    # Per 24-month holdout book: Y4 labels vs HHI_lag3 (why from24 is tiny).
    if full24:
        per = []
        for cid in sorted(full24):
            sub = y4[y4["company_id"] == cid] if len(y4) else y4
            nn = int(sub["d_cust_hhi_lag3"].notna().sum()) if len(sub) and "d_cust_hhi_lag3" in sub.columns else 0
            ever = bool(ho.loc[ho["company_id"] == cid, "ever_erp"].iloc[0]) if "ever_erp" in ho.columns and (ho["company_id"] == cid).any() else None
            n_debt = 0
            if len(bank) and "n_debt" in bank.columns:
                hit = bank.loc[bank["company_id"] == cid, "n_debt"]
                n_debt = int(hit.iloc[0]) if len(hit) else 0
            per.append(
                {
                    "company": cid,
                    "n_y4_lab": int(len(sub)),
                    "n_hhi_nn": nn,
                    "ever-ERP": "yes" if ever else "no",
                    "n_debt": n_debt,
                }
            )
        n_late_cos = int(y4_nn.loc[~y4_nn["company_id"].isin(full24), "company_id"].nunique()) if len(y4_nn) else 0
        late_rows = []
        if len(y4_nn):
            late = y4_nn.loc[~y4_nn["company_id"].isin(full24)]
            for cid, s in late.groupby("company_id"):
                late_rows.append(
                    {
                        "company": cid,
                        "n_y4_hhi": int(len(s)),
                        "grid m": int(s["n_grid_months"].iloc[0]) if "n_grid_months" in s.columns else "",
                    }
                )
            late_rows = sorted(late_rows, key=lambda r: (-r["n_y4_hhi"], r["company"]))
        md += [
            "",
            "The three 24-month holdout books are not a Q6 bench. Per company:",
            "",
            _md_table(per),
            "",
            "Y4 `ds_r_double` is NaN unless `ds_r` at t and t+3 exist and "
            "`ds_r_t > 0.05`. Zero labels on a 24-month ERP book is a "
            "**debt-service floor**, not a missing trail. Two of the three "
            "full-24 holdout companies have that floor. `n_debt` is the "
            "`debt_products` count — Y4 can still label with 0 facilities "
            "when repayment txs exist (COMP_1236), and a facility does not "
            "guarantee a label (COMP_0975).",
            "",
            f"Late-book Y4+HHI rows come from **{n_late_cos}** companies "
            f"(not one whale). A 3-month HHI lead on the hidden 72 is "
            f"{int(len(y4_nn))} rows total — quote train CV, not this slice.",
        ]
        if late_rows:
            md += [
                "",
                "Late holdout companies that still have Y4 + HHI_lag3:",
                "",
                _md_table(late_rows),
            ]
    return {
        "rows": rows,
        "n_y4_nn": int(len(y4_nn)),
        "n_y4_nn_from24": n_from24,
        "md": md,
    }


# ---------------------------------------------------------------------------
# Pass 19 — Y3 long so-far is a calendar sliver, not a long-trail world
# ---------------------------------------------------------------------------


def pass19_y3_long_calendar(p6: dict) -> dict:
    """Y3 long_>=18 so-far labeled rows should be ~2026-02 of 2024-09 starters."""
    tr = p6["panel"]
    tr = tr[tr["split"] == "train"]
    if "y3_recover_cash_6m" not in tr.columns:
        return {"md": []}
    lab = tr[tr["y3_recover_cash_6m"].notna() & (tr["so_far_class"] == "long_>=18")]
    per = lab["period"].dt.strftime("%Y-%m").value_counts().sort_index().to_dict()
    first = lab["first_tx_month"].dt.strftime("%Y-%m").value_counts().sort_index().to_dict()
    md = [
        "",
        "## Pass 19 — Y3 `so-far ≥ 18` is a calendar sliver",
        "",
        f"Labeled train Y3 rows with months-so-far ≥ 18: **{int(len(lab)):,}**. "
        "Y3 needs 6 future months, so the last labeled period is 2026-02. "
        "A company only reaches so-far=18 by 2026-02 if it started in 2024-09. "
        "This slice is **not** 'long-trail companies in general' — it is Feb 2026 "
        "stressed months of the 435 full-24 books. Prefer `*_company` rows for "
        "base-rate short vs long.",
        "",
        "Period of those rows:",
        "",
        _md_table([{"period": k, "n": f"{v:,}"} for k, v in per.items()]),
        "",
        "First-tx month of those rows:",
        "",
        _md_table([{"first tx": k, "n": f"{v:,}"} for k, v in first.items()]),
    ]
    return {"n": int(len(lab)), "periods": per, "first": first, "md": md}


# ---------------------------------------------------------------------------
# Markdown + registry
# ---------------------------------------------------------------------------


def _headline(p2: dict, p1: dict, p6: dict, p3: dict | None = None) -> list[str]:
    t = p1["train"]
    h = p1["holdout"]
    qa = p2["train"]
    lines = [
        "# Trail length — Q6 honesty (how many months earlier)",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        f"`monthly.parquet` / `targets.parquet` read-only. Rates and cuts on **train**. "
        f"Holdout 72 is coverage only. Fixed trail cuts 6 / 12 / 18 / 24 — not quantiles. "
        f"No 0–100. No parquet rewrite. No new Y.",
        "",
        "A lead-time claim is only as long as the observed trail "
        f"({p1['panel_start']} → {p1['panel_end']}, {p1['n_grid_expected']} official months). "
        "`created_at` is a **connection** clock; first transaction is the **bank-trail** clock. "
        "Using `created_at` as a health Y is **PARK** (onboarding ≠ health).",
        "",
        "## Headline",
        "",
    ]
    verdict = p2["verdict"]
    lines.append(
        f"- **Banking first `created_at` after 2024-09-01 (join-QA replica):** "
        f"{qa['n_first_created_after_2024_09_01']:,} / {qa['n_with_banking']:,} train companies "
        f"with a banking product = **{_pp(qa['share_after_0901'])}**. "
        f"Join QA quoted 891/1,211 = 73.6%. **{verdict}** "
        f"(count match={p2['confirm_count']}, share within 0.5pp={p2['confirm_share']}). "
        f"No banking product: {qa['n_no_banking']}."
    )
    lines.append(
        f"- **Stricter (first created *month* after 2024-09, i.e. ≥ 2024-10):** "
        f"{qa['n_first_created_month_after_sep']:,}/{qa['n_with_banking']:,} = "
        f"**{_pp(qa['share_after_sep_month'])}**. "
        f"September-2024 first connection: {qa['n_first_created_in_sep2024']:,} "
        f"({_pp(qa['share_in_sep2024'])}) — counted in the 73.6% because join QA used "
        f"`created_at > 2024-09-01`, not `month > 2024-09`."
    )
    lines.append(
        f"- **First debt `created_at` after 2024-09-01:** "
        f"{qa['n_debt_after_0901']:,}/{qa['n_with_debt']:,} = "
        f"**{_pp(qa['share_debt_after_0901'])}**. Same connection clock; "
        "do not read as new leverage."
    )
    lines.append(
        f"- **Bank trail (first tx month after 2024-09):** "
        f"{t['n_late_first_tx']:,}/{t['n_companies']:,} = **{_pp(t['share_late_first_tx'])}**. "
        f"This is the left-truncation that caps Q6, not the product clock."
    )
    if p3 and p3.get("train"):
        inv = p3["train"]
        lines.append(
            f"- **Invoice book vs bank:** ever-ERP {inv['n_ever_erp']:,} "
            f"({_pp(inv['share_ever_erp'])}), dark {inv['n_dark']:,}. "
            f"First issuance after 2024-09: {inv['n_first_iss_after_0901']:,} "
            f"({_pp(inv['share_first_iss_after_0901'])} of ERP). "
            f"Invoice months can exceed the bank grid when issuance pre-dates "
            f"first tx (pass 15: 43.5% of ever-ERP)."
        )
    lines.append(
        f"- **Months-on-book (official grid span):** "
        f"<6 = {_pp(t['share_lt6'])} ({t['n_lt6']:,}), "
        f"<12 = {_pp(t['share_lt12'])} ({t['n_lt12']:,}), "
        f"≥18 = {_pp(t['share_ge18'])} ({t['n_ge18']:,}), "
        f"=24 = {_pp(t['share_full_24'])} ({t['n_full_24']:,}). "
        f"Median {t['median_grid']:.0f} months; mean {t['mean_grid']:.1f}."
    )
    lines.append(
        f"- **Holdout is missing full books, not a <12 pile:** "
        f"{h['n_late_first_tx']:,}/{h['n_companies']:,} = "
        f"{_pp(h['share_late_first_tx'])} late first-tx; "
        f"<12 = {_pp(h['share_lt12'])} ({h['n_lt12']:,}) vs train {_pp(t['share_lt12'])}; "
        f"≥18 = {_pp(h['share_ge18'])} vs train {_pp(t['share_ge18'])}; "
        f"=24 = {_pp(h['share_full_24'])} ({h['n_full_24']:,}) vs train "
        f"{_pp(t['share_full_24'])}. Median grid {h['median_grid']:.0f} vs "
        f"{t['median_grid']:.0f}. Hidden-test Q6 will not transfer "
        f"(LOW_POWER + no 24-month pile)."
    )
    lines.append(
        "- **Q6 lags (train labeled):** Y4 `d_cust_hhi_lag3` non-null on **35.8%** "
        "overall and **21.7%** of short (<12m so-far) rows — missing is shift + "
        "calendar `full6` + no ERP HHI, residual 0. Among those short Y4 rows "
        "that *have* the lag, honest≥6 is 66.2% (p50=6) vs 100% on long. "
        "Y7 `e_ar_issued_lag1` non-null on **97.2%** / **95.2%** short; the hole "
        "is only `so-far<2`. Y7's short-trail Q6 limit is *length* (honest≥6 = 47%, "
        "p50=5), not a missing lag. "
        "43.5% of ever-ERP companies have invoices *before* first tx. "
        "Even among the 435 24-month train books, only **16.8%** (73) ever have "
        "a Y4 labeled row with HHI_lag3 (Y7: 59.5%, and every labeled 24m Y7 has the lag)."
    )
    # Compact short vs long so-far bases (train labeled).
    ybits = []
    for ycol, short_name, long_name in (
        ("y3_recover_cash_6m", "short_<12_sofar", "long_>=18_sofar"),
        ("y2_neg_2of3", "short_<12_sofar", "long_>=18_sofar"),
        ("y7_top1_lost", "short_<12_sofar", "long_>=18_sofar"),
        ("y4_ds_r_double", "short_<12_sofar", "long_>=18_sofar"),
    ):
        s = next((r for r in p6.get("rows") or [] if r["y"] == ycol and r["slice"] == short_name), None)
        lg = next((r for r in p6.get("rows") or [] if r["y"] == ycol and r["slice"] == long_name), None)
        if s and lg:
            ybits.append(
                f"{Y_LABEL.get(ycol, ycol)} {_pp(s['base'])} (n={s['n_labeled']:,}) vs "
                f"{_pp(lg['base'])} (n={lg['n_labeled']:,})"
            )
    ybits_co = []
    for ycol, short_name, long_name in (
        ("y3_recover_cash_6m", "short_<12_company", "long_>=18_company"),
        ("y2_neg_2of3", "short_<12_company", "long_>=18_company"),
        ("y7_top1_lost", "short_<12_company", "long_>=18_company"),
        ("y4_ds_r_double", "short_<12_company", "long_>=18_company"),
    ):
        s = next((r for r in p6.get("rows") or [] if r["y"] == ycol and r["slice"] == short_name), None)
        lg = next((r for r in p6.get("rows") or [] if r["y"] == ycol and r["slice"] == long_name), None)
        if s and lg:
            ybits_co.append(
                f"{Y_LABEL.get(ycol, ycol)} {_pp(s['base'])} vs {_pp(lg['base'])}"
            )
    if ybits:
        extra = ""
        if ybits_co:
            extra = (
                " Company-total (preferred): "
                + "; ".join(ybits_co)
                + "."
            )
        lines.append(
            "- **Y base rates short (<12 so-far) vs long (≥18 so-far), train labeled:** "
            + "; ".join(ybits)
            + ". None only-defined-on-long. Y3 long so-far is a Feb-2026 sliver of "
            "2024-09 starters."
            + extra
        )
    lines.append(
        "- **Holdout groups:** 12 / 15 first appear after 2024-09; 65 / 72 companies "
        "sit in those new groups; 0 train siblings. Hidden test is a late-arrival "
        "sample. Quote train CV for any lead-time claim. Holdout Y4 labeled with "
        "`d_cust_hhi_lag3`: 21 / 135 (short 18 / 88); only **2** of those 21 sit "
        "on the three 24-month holdout books. The 3-month HHI lead is almost "
        "undefined on the hidden 72."
    )
    return lines


def write_report(p1, p2, p3, p4, p5, p6, elapsed: float, extra: dict | None = None) -> None:
    extra = extra or {}
    a: list[str] = _headline(p2, p1, p6, p3)
    t = p1["train"]
    h = p1["holdout"]
    qa = p2["train"]
    qh = p2["holdout"]
    inv = p3["train"]

    a.append("")
    a.append("### Months-on-book histogram (train companies)")
    a.append("")
    if p4.get("png"):
        a.append(f"![months on book]({Path(p4['png']).name})")
        a.append("")
    hist_rows = []
    for k in range(1, 25):
        hist_rows.append(
            {
                "months": str(k),
                "n": f"{t['hist'][k]:,}",
                "share": _pp(t["hist"][k] / t["n_companies"] if t["n_companies"] else float("nan")),
            }
        )
    a.append(_md_table(hist_rows, ["months", "n", "share"]))
    a.append("")
    a.append("Fixed-cut buckets (not quantiles):")
    a.append("")
    brow = []
    for name, lo, hi in BUCKETS:
        b = t["bucket"][name]
        brow.append(
            {
                "bucket": name,
                "months": f"{lo}–{hi}",
                "n": f"{b['n']:,}",
                "share": _pp(b["share"]),
            }
        )
    a.append(_md_table(brow, ["bucket", "months", "n", "share"]))
    if p4.get("n_holdout"):
        a += [
            "",
            "Holdout months-on-book (coverage only, same fixed cuts): "
            f"<6 = {_pp(p4.get('ho_share_lt6'))} ({p4.get('ho_lt6')}), "
            f"<12 = {_pp(p4.get('ho_share_lt12'))} ({p4.get('ho_lt12')}), "
            f"≥18 = {_pp(p4.get('ho_share_ge18'))} ({p4.get('ho_ge18')}), "
            f"=24 = {_pp(p4.get('ho_share_eq24'))} ({p4.get('ho_eq24')}) "
            f"of {p4.get('n_holdout')}.",
        ]

    a += [
        "",
        "## Pass 1 — bank trail on the official monthly grid",
        "",
        "A company enters `monthly_grid` at the month of its first transaction and stays "
        f"through {PANEL_END.date()}. `n_grid_months` is that span (1–24). "
        "`n_tx_months` counts months with ≥1 transaction inside the same window "
        "(2024-09-01 ≤ date < 2026-09-01). September 2026 extract txs are off-grid.",
        "",
    ]
    cov_rows = [
        {
            "split": "train",
            "n_cos": f"{t['n_companies']:,}",
            "late first tx": f"{t['n_late_first_tx']:,} ({_pp(t['share_late_first_tx'])})",
            "median grid": f"{t['median_grid']:.0f}",
            "median tx-months": f"{t['median_tx_months']:.0f}",
            "any gap": f"{t['n_any_gap']:,} ({_pp(t['share_any_gap'])})",
            "=24": f"{t['n_full_24']:,} ({_pp(t['share_full_24'])})",
        },
        {
            "split": "holdout",
            "n_cos": f"{h['n_companies']:,}",
            "late first tx": f"{h['n_late_first_tx']:,} ({_pp(h['share_late_first_tx'])})",
            "median grid": f"{h['median_grid']:.0f}",
            "median tx-months": f"{h['median_tx_months']:.0f}",
            "any gap": f"{h['n_any_gap']:,} ({_pp(h['share_any_gap'])})",
            "=24": f"{h['n_full_24']:,} ({_pp(h['share_full_24'])})",
        },
    ]
    a.append(_md_table(cov_rows))
    a.append("")
    a.append("First transaction month (train):")
    a.append("")
    fh = [
        {"month": k, "n": f"{v:,}"}
        for k, v in sorted(p1["first_tx_month_hist_train"].items())
    ]
    a.append(_md_table(fh, ["month", "n"]))

    a += [
        "",
        "## Pass 2 — product `created_at` (confirm / correct 73.6%)",
        "",
        "Join QA (`data_join_qa.md` Pass 4): of 1,211 train companies with a banking "
        "product, **891 (73.6%)** have `MIN(banking_products.created_at) > 2024-09-01`. "
        "Replica below uses the same denominator (train ∩ panel ∩ has banking product) "
        "and the same `>` cut. Timestamps in `clean` are naive; treating them as UTC "
        "does not move the date of any first-created (none land on 2024-09-01 00:00).",
        "",
    ]
    prod_rows = [
        {
            "split": "train",
            "with banking": f"{qa['n_with_banking']:,}",
            "no banking": str(qa["n_no_banking"]),
            "first created > 2024-09-01": f"{qa['n_first_created_after_2024_09_01']:,} ({_pp(qa['share_after_0901'])})",
            "first created month ≥ Oct-2024": f"{qa['n_first_created_month_after_sep']:,} ({_pp(qa['share_after_sep_month'])})",
            "first created < 2024-09-01": f"{qa['n_first_created_before_panel']:,} ({_pp(qa['share_before_panel'])})",
            "with debt": f"{qa['n_with_debt']:,}",
            "debt first created > 2024-09-01": f"{qa['n_debt_after_0901']:,} ({_pp(qa['share_debt_after_0901'])})",
        },
        {
            "split": "holdout",
            "with banking": f"{qh['n_with_banking']:,}",
            "no banking": str(qh["n_no_banking"]),
            "first created > 2024-09-01": f"{qh['n_first_created_after_2024_09_01']:,} ({_pp(qh['share_after_0901'])})",
            "first created month ≥ Oct-2024": f"{qh['n_first_created_month_after_sep']:,} ({_pp(qh['share_after_sep_month'])})",
            "first created < 2024-09-01": f"{qh['n_first_created_before_panel']:,} ({_pp(qh['share_before_panel'])})",
            "with debt": f"{qh['n_with_debt']:,}",
            "debt first created > 2024-09-01": f"{qh['n_debt_after_0901']:,} ({_pp(qh['share_debt_after_0901'])})",
        },
    ]
    a.append(_md_table(prod_rows))
    a.append("")
    cx = p2["cross_train"]
    a.append(
        f"**Connection clock ≠ bank trail (train, has banking, n={cx['n']:,}):** "
        f"late `created_at` & full 2024-09 tx start = {cx['created_late_and_tx_full']:,}; "
        f"late `created_at` & late first tx = {cx['created_late_and_tx_late']:,}; "
        f"early `created_at` & late first tx = {cx['created_early_and_tx_late']:,}; "
        f"early `created_at` & 2024-09 tx = {cx['created_early_and_tx_full']:,}. "
        f"`created_at` after first tx: {qa['n_created_after_first_tx']:,} "
        f"({_pp(qa['share_created_after_first_tx'])} of banking books) — "
        f"the product row is a later *connection*, not the start of cash movement. "
        f"`created_at` before first tx: {qa['n_created_before_first_tx']:,} "
        f"({_pp(qa['share_created_before_first_tx'])}) — product older than the "
        f"2024-09 tx window (left-truncated *observations*, not a new company)."
    )
    a.append("")
    a.append(
        "**PARK:** do not turn `created_at` (company, banking, or debt) into a health Y. "
        "Onboarding / connection is not a 45→65 or 82→68 trajectory."
    )

    a += [
        "",
        "## Pass 3 — invoice book vs bank book",
        "",
        "Ever-ERP = ≥1 book invoice (`document_type=invoice`, `status<>cancel`, `amount<>0`). "
        "The 470 are never-ERP on train (join QA confirmed). Invoice months below count "
        "any issuance month in 2024-09..2026-08 (can exceed the bank grid when invoices "
        "predate first tx — see pass 15). Bank months = official grid span.",
        "",
    ]
    invh = p3["holdout"]
    a.append(
        _md_table(
            [
                {
                    "split": "train",
                    "ever-ERP": f"{inv['n_ever_erp']:,} ({_pp(inv['share_ever_erp'])})",
                    "never-ERP": f"{inv['n_dark']:,} ({_pp(inv['share_dark'])})",
                    "first iss after 2024-09": f"{inv['n_first_iss_after_0901']:,} ({_pp(inv['share_first_iss_after_0901'])} of ERP)",
                    "median inv months (ERP)": _f(inv["median_inv_months_erp"], 1),
                    "mean bank months (ERP)": _f(inv["mean_bank_months_erp"], 1),
                    "mean bank months (dark)": _f(inv["mean_bank_months_dark"], 1),
                    "ERP inv < bank months": f"{inv['n_inv_shorter']:,} ({_pp(inv['share_inv_shorter'])})",
                    "first iss >31d after first tx": f"{inv['n_iss_gt31_after_tx']:,} ({_pp(inv['share_iss_gt31_after_tx'])})",
                },
                {
                    "split": "holdout",
                    "ever-ERP": f"{invh['n_ever_erp']:,} ({_pp(invh['share_ever_erp'])})",
                    "never-ERP": f"{invh['n_dark']:,} ({_pp(invh['share_dark'])})",
                    "first iss after 2024-09": f"{invh['n_first_iss_after_0901']:,} ({_pp(invh['share_first_iss_after_0901'])} of ERP)",
                    "median inv months (ERP)": _f(invh["median_inv_months_erp"], 1),
                    "mean bank months (ERP)": _f(invh["mean_bank_months_erp"], 1),
                    "mean bank months (dark)": _f(invh["mean_bank_months_dark"], 1),
                    "ERP inv < bank months": f"{invh['n_inv_shorter']:,} ({_pp(invh['share_inv_shorter'])})",
                    "first iss >31d after first tx": f"{invh['n_iss_gt31_after_tx']:,} ({_pp(invh['share_iss_gt31_after_tx'])})",
                },
            ]
        )
    )

    a += [
        "",
        "## Pass 5 — left-truncation: smaller? new group or new subsidiary?",
        "",
        "Late = first transaction month after 2024-09. Size = company-median "
        "`log1p(a_in3)` on train company-months (descriptive; no fitted bins). "
        "New group = the group's earliest first-tx (all members, including holdout "
        "siblings for the group clock) is after 2024-09. New subsidiary = late company "
        "in a group that already had a 2024-09 starter.",
        "",
    ]
    a.append(
        f"Train late first-tx: **{p5['n_late']:,} / {p5['n_train']:,} ({_pp(p5['share_late'])})**. "
        f"Of those: new subsidiaries **{p5['n_new_subsidiary']:,} ({_pp(p5['share_new_subsidiary'])})**, "
        f"members of new groups **{p5['n_new_group_member']:,} ({_pp(p5['share_new_group_member'])})**. "
        f"Train groups: {p5['n_groups_train']} (old {p5['n_groups_old']}, new {p5['n_groups_new']})."
    )
    a.append("")
    a.append(
        _md_table(
            [
                {
                    "slice": "late first tx",
                    "n": f"{p5['size_late']['n']:,}",
                    "median log1p(a_in3)": _f(p5["size_late"]["median"], 3),
                    "mean": _f(p5["size_late"]["mean"], 3),
                    "p25": _f(p5["size_late"]["p25"], 3),
                    "p75": _f(p5["size_late"]["p75"], 3),
                },
                {
                    "slice": "first tx in 2024-09",
                    "n": f"{p5['size_full']['n']:,}",
                    "median log1p(a_in3)": _f(p5["size_full"]["median"], 3),
                    "mean": _f(p5["size_full"]["mean"], 3),
                    "p25": _f(p5["size_full"]["p25"], 3),
                    "p75": _f(p5["size_full"]["p75"], 3),
                },
            ]
        )
    )
    a.append(
        f"Δ median log1p(a_in3) (late − full) = {_f(p5['delta_median_size'], 3)}. "
        f"Mann–Whitney two-sided p = {_f(p5['mw_p'], 4)} (diagnostic, not a cut)."
    )

    a += [
        "",
        "## Pass 6 — accepted Y base rates, short vs long trail (train)",
        "",
        "Primary clock is **months-on-book so far** at the company-month (Q6-honest: "
        "a 24-month company is still short in its first months). Robustness: company "
        "total grid span. Short = <12, long = ≥18. Mid 12–17 shown. "
        "Y3 is stressed-only with a 6-month future window; Y7 needs a trailing 3-month "
        "AR book + 3-month future; Y4 needs ds_r at t and t+3. If a Y has no short-trail "
        "labels, that is a definition floor, not a sampling accident.",
        "",
        "Bucket `24` is empty for every labeled Y: labels need future months, so the "
        "last grid month is never labeled. `so-far ≥ 18` for Y3 is almost only "
        "`2026-02` of the 435 companies that started in 2024-09 (Y3 horizon = 6, so "
        "last labeled period is 2026-02 = month 18). Short-vs-long *so-far* base rates "
        "for Y3 are partly a **calendar mix**, not a pure trail effect. Prefer the "
        "`*_company` rows for a base-rate comparison.",
        "",
    ]
    yrows = []
    for r in p6["rows"]:
        yrows.append(
            {
                "Y": Y_LABEL.get(r["y"], r["y"]),
                "slice": r["slice"],
                "n_labeled": f"{r['n_labeled']:,}",
                "n_pos": f"{r['n_pos']:,}",
                "base": _pp(r["base"]),
                "% of cm labeled": _pp(r["share_labeled"]),
            }
        )
    a.append(_md_table(yrows, ["Y", "slice", "n_labeled", "n_pos", "base", "% of cm labeled"]))
    a.append("")
    a.append("Definition floors (train labeled):")
    a.append("")
    a.append(
        _md_table(
            [
                {
                    "Y": Y_LABEL.get(f["y"], f["y"]),
                    "min so-far": str(f["min_so_far"]),
                    "p50 so-far": _f(f.get("p50_so_far"), 1),
                    "min period": f["min_period"] or "—",
                    "n labeled": f"{f['n']:,}",
                    "share short": _pp(f.get("share_short")),
                    "share long": _pp(f.get("share_long")),
                }
                for f in p6["floors"]
            ]
        )
    )
    a.append("")
    a.append("Holdout labeled counts (LOW_POWER, no rates used as cuts):")
    a.append("")
    a.append(
        _md_table(
            [
                {
                    "Y": Y_LABEL.get(r["y"], r["y"]),
                    "slice": r["slice"],
                    "n_labeled": f"{r['n_labeled']:,}",
                    "n_pos": f"{r['n_pos']:,}",
                }
                for r in p6["holdout_counts"]
            ]
        )
    )

    a += [
        "",
        "## Pass 7 — Q6 lag honesty (`d_cust_hhi_lag3`, `e_ar_issued_lag1`)",
        "",
        "Y4 uses `d_cust_hhi_lag3` as the published single-feature clock. Y7 uses "
        "`e_ar_issued_lag1` as a lead. Both are **history**: HHI itself needs a 6-month "
        "invoice window (family D `full6`); lag-3 needs three further months on the panel. "
        "`e_ar_issued_lag1` is a 1-month shift of family E issuance (NaN before first invoice "
        "and on the 470). Share of **train labeled rows** where the lag is non-null, "
        "by so-far bucket.",
        "",
    ]
    q6rows = []
    for r in p6["q6"]:
        q6rows.append(
            {
                "Y": Y_LABEL.get(r["y"], r["y"]),
                "bucket": r["bucket"],
                "n_labeled": f"{r['n_labeled']:,}",
                "HHI_lag3 nn": f"{r.get('hhi_lag3_nn', 0):,} ({_pp(r.get('hhi_lag3_share'))})",
                "issued_lag1 nn": f"{r.get('issued_lag1_nn', 0):,} ({_pp(r.get('issued_lag1_share'))})",
            }
        )
    a.append(_md_table(q6rows))

    if extra.get("group"):
        a += extra["group"].get("md", [])
    if extra.get("gfam"):
        a += extra["gfam"].get("md", [])
    if extra.get("why"):
        a += extra["why"].get("md", [])
    if extra.get("lead"):
        a += extra["lead"].get("md", [])
    if extra.get("cash"):
        a += extra["cash"].get("md", [])
    if extra.get("erp_y"):
        a += extra["erp_y"].get("md", [])
    if extra.get("snap"):
        a += extra["snap"].get("md", [])
    if extra.get("invclip"):
        a += extra["invclip"].get("md", [])
    if extra.get("y7pre"):
        a += extra["y7pre"].get("md", [])
    if extra.get("sep"):
        a += extra["sep"].get("md", [])
    if extra.get("hold"):
        a += extra["hold"].get("md", [])
    y3c = extra.get("y3cal") or {}
    if y3c.get("md"):
        a.extend(list(y3c["md"]))
    if extra.get("more"):
        a += extra["more"]

    a += [
        "",
        "## Six brief questions",
        "",
        "| # | question | what trail length says |",
        "| --- | --- | --- |",
        "| 1 | Who is healthy? | Not a health reading. Short trail ≠ sick. |",
        "| 2 | Who is improving? | A 6-month book cannot show a 12-month improvement. |",
        "| 3 | Who is turning? | Same cap: a turn needs months on both sides of t. |",
        "| 4 | Dip vs fall? | Y7 needs a 3-month AR name + 3-month future. On short *bank* trails the AR name can still come from pre-grid invoices (43.5% of ERP). |",
        "| 5 | Why did it change? | Family G `created_*` constants are *connection quality*, not this fact. |",
        "| 6 | Months earlier? | Claim ≤ observed trail. Y4 HHI_lag3 is null on 78% of short labeled rows (shift + calendar full6 + no ERP HHI); among the 22% with the lag, honest≥6 is 66% (p50=6). Y7 issued_lag1 is present on 95% of short labels (hole = so-far<2) but honest≥6 is only 47% (p50=5) — length, not missingness. |",
        "",
        f"Elapsed {elapsed:.0f}s. Same-module cuts: bank trail, 73.6% replica, "
        f"histogram, size/subsidiary, Y rates, Q6 lags, group wave, `g_*` constants, "
        f"lag-missing why, honest lead, zero-account cash, ERP×trail, post-snapshot, "
        f"invoice clipped to grid, Y7 pre-grid, Sep-2026 extract, holdout groups, "
        f"Y3 long=2026-02 sliver, holdout Y4 HHI 21/135 (2 from 24m books).",
        "",
        "## What failed / next",
        "",
        "- Join-QA 73.6% **CONFIRMED** (891/1,211). The number is a *connection* clock, "
        "not first-tx (64.2% late) and not the `g_created_*` constants.",
        "- First draft of pass 3 counted invoice months off the bank grid; pass 15 "
        "clipped. 43.5% of ever-ERP have pre-grid invoices.",
        "- First draft of pass 17 called August quiet a one-month gap; only 2/121 "
        "reappear in Sep-2026. Corrected. Still PARK as a death Y.",
        "- Y3 so-far≥18 is 213 rows, all 2026-02 of 2024-09 starters. Not a long-trail world.",
        "- Holdout Y4 `d_cust_hhi_lag3` is 21/135 labeled; only 2 of those 21 sit on "
        "the three 24-month holdout books. Quote train CV for Q6, not the hidden 72.",
        "- Holdout <12 share is 26.4% vs train 28.8% — the hidden 72 is missing the "
        "24-month pile (4.2% vs 35.8%), not a <12 pile.",
        "- Y7 short Q6 hole is *length* (honest≥6 = 47%, p50=5), not missing "
        "`issued_lag1` (95.2% present). Y4 short hole is mostly missing HHI_lag3 (78%).",
        "- A 24-month train book is not a Y4 HHI bench: only 73/435 (16.8%) ever "
        "have a labeled Y4 row with `d_cust_hhi_lag3`.",
        "- **Next (legal, not this lane):** do not invent `y_short_trail`. If someone "
        "wants the 121 quiet tail, it must beat Y2 as an activity drop and not use "
        "`created_at`. Family G should keep `g_n_accounts` / `g_new_this_month` as "
        "the 73.6% surface, not the CONSTANT created_* flags.",
        "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(a), encoding="utf-8")


def append_registry(p1, p2, p5, p6, extra: dict | None = None) -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    t = p1["train"]
    qa = p2["train"]
    rows = [
        {
            "metric": "banking_first_created_after_202409",
            "value": qa["share_after_0901"],
            "coverage": qa["share_after_0901"],
            "notes": (
                f"replica {qa['n_first_created_after_2024_09_01']}/{qa['n_with_banking']}; "
                f"joinQA 891/1211; verdict={p2['verdict']}; "
                f"month_after_sep={qa['share_after_sep_month']:.4f}"
            ),
        },
        {
            "metric": "first_tx_after_202409",
            "value": t["share_late_first_tx"],
            "coverage": t["share_late_first_tx"],
            "notes": f"n={t['n_late_first_tx']}/{t['n_companies']}; Q6 bank-trail clock",
        },
        {
            "metric": "months_on_book_lt6",
            "value": t["share_lt6"],
            "coverage": t["share_lt6"],
            "notes": f"n={t['n_lt6']}/{t['n_companies']}; fixed cut <6",
        },
        {
            "metric": "months_on_book_lt12",
            "value": t["share_lt12"],
            "coverage": t["share_lt12"],
            "notes": f"n={t['n_lt12']}/{t['n_companies']}; fixed cut <12",
        },
        {
            "metric": "months_on_book_eq24",
            "value": t["share_full_24"],
            "coverage": t["share_full_24"],
            "notes": f"n={t['n_full_24']}/{t['n_companies']}",
        },
        {
            "metric": "late_tx_new_subsidiary_share",
            "value": p5["share_new_subsidiary"],
            "coverage": p5["share_late"],
            "notes": f"n={p5['n_new_subsidiary']}/{p5['n_late']}; rest are new-group members",
        },
    ]
    extra_reg = extra or {}
    g8 = extra_reg.get("group") or {}
    g9 = extra_reg.get("gfam") or {}
    if g8:
        rows.append(
            {
                "metric": "late_tx_same_month_group_share",
                "value": g8.get("share_late_wave", float("nan")),
                "coverage": g8.get("share_late_wave", float("nan")),
                "notes": (
                    f"n={g8.get('n_late_wave')}/{g8.get('n_late')}; "
                    f"staggered={g8.get('share_late_staggered')}; "
                    f"mixed={g8.get('share_late_mixed')}"
                ),
            }
        )
    if g9 and g9.get("ok"):
        rows.append(
            {
                "metric": "g_n_accounts_zero_train_cm",
                "value": g9.get("share_g_n_accounts_0", float("nan")),
                "coverage": g9.get("share_g_n_accounts_0", float("nan")),
                "notes": (
                    f"n={g9.get('n_g_n_accounts_0')}/{g9.get('n_cm')}; "
                    f"NOT the g_created_* constants; first_month_zero={g9.get('first_share_0')}"
                ),
            }
        )
    g18 = extra_reg.get("hold") or {}
    if g18:
        rows.append(
            {
                "metric": "holdout_late_first_tx",
                "value": _pct(g18.get("n_late"), g18.get("n")),
                "coverage": _pct(g18.get("n_new_g"), g18.get("n")),
                "notes": (
                    f"n={g18.get('n_late')}/{g18.get('n')}; "
                    f"in_new_group={g18.get('n_new_g')}; "
                    f"groups_new={g18.get('n_groups_new')}/{g18.get('n_groups')}; "
                    f"train_sib={g18.get('n_train_sib')}; LOW_POWER coverage"
                ),
                "split": "holdout",
            }
        )
        if g18.get("n_y4_nn") is not None:
            rows.append(
                {
                    "metric": "y4_hhi_lag3_nn_holdout",
                    "value": _pct(g18.get("n_y4_nn"), 135),
                    "coverage": _pct(g18.get("n_y4_nn_from24"), g18.get("n_y4_nn")),
                    "notes": (
                        f"n={g18.get('n_y4_nn')}/135 holdout Y4 labeled; "
                        f"from_24m={g18.get('n_y4_nn_from24')}; LOW_POWER coverage"
                    ),
                    "split": "holdout",
                }
            )
    g4 = extra_reg.get("hist") or {}
    if g4.get("n_holdout"):
        rows.append(
            {
                "metric": "holdout_months_on_book_lt12",
                "value": g4.get("ho_share_lt12", float("nan")),
                "coverage": g4.get("ho_share_eq24", float("nan")),
                "notes": (
                    f"n_lt12={g4.get('ho_lt12')}/{g4.get('n_holdout')}; "
                    f"eq24={g4.get('ho_eq24')}; ge18={g4.get('ho_ge18')}; "
                    f"LOW_POWER coverage"
                ),
                "split": "holdout",
            }
        )
    g11 = extra_reg.get("lead") or {}
    for r in g11.get("co_cov") or []:
        if r.get("y") == "y4_ds_r_double":
            rows.append(
                {
                    "metric": "y4_hhi_lag3_ever_nn_24m_train",
                    "value": r.get("share_nn", float("nan")),
                    "coverage": r.get("share_lab", float("nan")),
                    "notes": (
                        f"n24={r.get('n24')}; ever_lab={r.get('n24_lab')}; "
                        f"ever_lab_lag={r.get('n24_nn')}; company-level Q6"
                    ),
                }
            )
    for r in g11.get("trail_lead") or []:
        if r.get("y") == "y4_ds_r_double" and r.get("slice") == "short_<12":
            rows.append(
                {
                    "metric": "y4_hhi_lag3_honest_ge6_short",
                    "value": r.get("share_ge6", float("nan")),
                    "coverage": _pct(r.get("n_nn"), r.get("n_lab")),
                    "notes": (
                        f"n_lab={r.get('n_lab')}; nn={r.get('n_nn')}; "
                        f"p50_honest={r.get('p50_honest_nn')}; "
                        "among short Y4 labeled with lag present"
                    ),
                }
            )
        if r.get("y") == "y7_top1_lost" and r.get("slice") == "short_<12":
            rows.append(
                {
                    "metric": "y7_issued_lag1_honest_ge6_short",
                    "value": r.get("share_ge6", float("nan")),
                    "coverage": _pct(r.get("n_nn"), r.get("n_lab")),
                    "notes": (
                        f"n_lab={r.get('n_lab')}; nn={r.get('n_nn')}; "
                        f"p50_honest={r.get('p50_honest_nn')}; "
                        "Y7 short Q6 hole is length not missing lag"
                    ),
                }
            )
    g17 = extra_reg.get("sep") or {}
    if g17:
        rows.append(
            {
                "metric": "quiet_aug_still_sep2026_tx",
                "value": g17.get("share_quiet_sep", float("nan")),
                "coverage": g17.get("share_sep", float("nan")),
                "notes": (
                    f"quiet={g17.get('n_quiet')}; quiet_with_sep={g17.get('n_quiet_sep')}; "
                    f"any_sep={g17.get('n_sep')}; not a death Y"
                ),
            }
        )
    g15 = extra_reg.get("invclip") or {}
    if g15:
        rows.append(
            {
                "metric": "erp_invoice_before_first_tx",
                "value": g15.get("share_pre", float("nan")),
                "coverage": g15.get("share_pre", float("nan")),
                "notes": (
                    f"n={g15.get('n_erp_pre')}/{g15.get('n_erp')}; "
                    f"pass3 inv-months can exceed bank grid because of pre-grid issuance"
                ),
            }
        )
    g12 = extra_reg.get("cash") or {}
    if g12:
        rows.append(
            {
                "metric": "g_n_accounts_0_with_tx",
                "value": g12.get("share_zero_with_tx", float("nan")),
                "coverage": g12.get("share_zero_with_tx", float("nan")),
                "notes": (
                    f"n_zero={g12.get('n_zero')}; n_with_tx={g12.get('n_zero_with_tx')}; "
                    f"med_tx_zero={g12.get('median_tx_zero')}; PARK as health Y"
                ),
            }
        )
    g11 = extra_reg.get("lead") or {}
    if g11:
        rows.append(
            {
                "metric": "last_tx_before_202608",
                "value": g11.get("share_quiet", float("nan")),
                "coverage": g11.get("share_quiet", float("nan")),
                "notes": f"n={g11.get('n_quiet')}; right-censor quiet tail, not left-trunc",
            }
        )
    g10 = extra_reg.get("why") or {}
    if g10:
        y4s = (g10.get("blocks") or {}).get("y4_ds_r_double::d_cust_hhi_lag3", {}).get("short", {})
        rows.append(
            {
                "metric": "y4_hhi_lag3_miss_short",
                "value": y4s.get("share_miss", float("nan")),
                "coverage": 1.0 - y4s.get("share_miss", float("nan")) if y4s else float("nan"),
                "notes": (
                    f"n_lab={y4s.get('n_labeled')}; miss={y4s.get('n_miss')}; "
                    f"shift={y4s.get('n_shift')}; cal={y4s.get('n_cal')}; src={y4s.get('n_src')}; "
                    f"conn_lag_med={g10.get('conn_lag_median')}"
                ),
            }
        )
    # Y short vs long (so-far)
    for r in p6["rows"]:
        if r["slice"] not in ("short_<12_sofar", "long_>=18_sofar"):
            continue
        rows.append(
            {
                "metric": f"{r['y']}_{r['slice']}_base",
                "value": r["base"],
                "coverage": r["share_labeled"],
                "notes": f"n_lab={r['n_labeled']}; n_pos={r['n_pos']}; {r['note']}",
            }
        )
    # Q6 lag shares on all labeled + short + long for Y4 and Y7
    for r in p6["q6"]:
        if r["y"] == "y4_ds_r_double" and r["bucket"] in ("all", "short_<12", "long_>=18"):
            rows.append(
                {
                    "metric": f"y4_hhi_lag3_nn_{r['bucket']}",
                    "value": r.get("hhi_lag3_share", float("nan")),
                    "coverage": r.get("hhi_lag3_share", float("nan")),
                    "notes": f"n_lab={r['n_labeled']}; nn={r.get('hhi_lag3_nn')}",
                }
            )
        if r["y"] == "y7_top1_lost" and r["bucket"] in ("all", "short_<12", "long_>=18"):
            rows.append(
                {
                    "metric": f"y7_issued_lag1_nn_{r['bucket']}",
                    "value": r.get("issued_lag1_share", float("nan")),
                    "coverage": r.get("issued_lag1_share", float("nan")),
                    "notes": f"n_lab={r['n_labeled']}; nn={r.get('issued_lag1_nn')}",
                }
            )

    with REGISTRY.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        seen = {
            (r.get("agent"), r.get("y"), r.get("model"), r.get("split"), r.get("metric"))
            for r in reader
        }
    fresh = []
    for r in rows:
        split = r.get("split") or "train"
        key = (AGENT, "trail_length", "coverage", split, r["metric"])
        if key in seen:
            continue
        fresh.append(r)
        seen.add(key)
    if not fresh:
        print("registry: nothing new to append")
        return
    with REGISTRY.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        for r in fresh:
            w.writerow(
                {
                    "ts": ts,
                    "round": ROUND,
                    "wave": WAVE,
                    "agent": AGENT,
                    "x_families": "trail",
                    "y": "trail_length",
                    "model": "coverage",
                    "split": r.get("split") or "train",
                    "metric": r["metric"],
                    "value": r["value"],
                    "coverage": r["coverage"],
                    "notes": r["notes"],
                }
            )
    print(f"appended {len(fresh)} registry rows")


def main() -> dict:
    t0 = time.time()
    print(f"trail_length start {_now_iso()} agent={AGENT}", flush=True)
    con = connect()
    try:
        cos = _companies(con)
        monthly, targets = _load_panel()
        print(
            f"panel monthly={monthly.shape} targets={targets.shape} companies={len(cos)}",
            flush=True,
        )

        print("PASS 1 bank trail", flush=True)
        p1 = pass1_bank_trail(con, cos, monthly)
        print(
            f"  train n={p1['train']['n_companies']} late_tx={p1['train']['share_late_first_tx']:.3f} "
            f"eq24={p1['train']['share_full_24']:.3f} lt12={p1['train']['share_lt12']:.3f}",
            flush=True,
        )

        print("PASS 2 products created_at", flush=True)
        p2 = pass2_products(con, p1["panel"])
        print(
            f"  after_0901={p2['train']['share_after_0901']:.4f} "
            f"after_sep_month={p2['train']['share_after_sep_month']:.4f} "
            f"verdict={p2['verdict']}",
            flush=True,
        )
        if p2["verdict"] != "CONFIRM":
            print("  WARN: join-QA 73.6% did not confirm — check timezone / denom", flush=True)

        print("PASS 3 invoices", flush=True)
        p3 = pass3_invoices(con, p2["panel"])
        print(
            f"  train ever_erp={p3['train']['n_ever_erp']} dark={p3['train']['n_dark']} "
            f"inv<bank={p3['train']['share_inv_shorter']:.3f}",
            flush=True,
        )

        print("PASS 4 histogram", flush=True)
        p4 = pass4_histogram(p3["panel"])
        print(f"  png={p4['png']}", flush=True)

        print("PASS 5 left-trunc size / group", flush=True)
        p5 = pass5_left_trunc(p3["panel"], monthly)
        print(
            f"  late={p5['n_late']} new_sub={p5['n_new_subsidiary']} "
            f"new_grp={p5['n_new_group_member']} dmed={p5['delta_median_size']}",
            flush=True,
        )

        print("PASS 6–7 Y rates + Q6 lags", flush=True)
        p6 = pass6_y_rates(p3["panel"], monthly, targets)
        for r in p6["rows"]:
            if r["slice"] in ("short_<12_sofar", "long_>=18_sofar", "all_train"):
                print(
                    f"  {r['y']} {r['slice']}: n={r['n_labeled']} base={r['base']}",
                    flush=True,
                )

        print("PASS 8 group wave", flush=True)
        p8 = pass8_group_wave(p3["panel"])
        print(
            f"  groups={p8['n_groups']} kinds={p8['kinds']} "
            f"late_wave={p8['share_late_wave']:.3f}",
            flush=True,
        )

        print("PASS 9 family G constants", flush=True)
        p9 = pass9_g_family(p3["panel"], monthly)
        print(
            f"  g_n_accounts=0 share={p9.get('share_g_n_accounts_0')} "
            f"same_fact={p9.get('same_fact')} first0={p9.get('first_share_0')}",
            flush=True,
        )

        print("PASS 10 lag missing why + connection lag", flush=True)
        p10 = pass10_lag_why(p3["panel"], monthly, p6)
        y4b = p10["blocks"].get("y4_ds_r_double::d_cust_hhi_lag3", {})
        print(
            f"  y4 hhi_lag3 miss all={y4b.get('all', {}).get('share_miss')} "
            f"short={y4b.get('short', {}).get('share_miss')} "
            f"conn_lag_med={p10['conn_lag_median']}",
            flush=True,
        )

        print("PASS 11 right-censor / ERP / honest lead", flush=True)
        p11 = pass11_right_erp_lead(p3["panel"], p6)
        print(
            f"  quiet={p11['n_quiet']} share={p11['share_quiet']:.3f} "
            f"lead_rows={len(p11['lead_rows'])}",
            flush=True,
        )

        print("PASS 12 zero-account months still have tx", flush=True)
        p12 = pass12_zero_accounts_have_tx(p3["panel"], monthly, p1["txm"])
        print(
            f"  zero_cm={p12['n_zero']} with_tx={p12['share_zero_with_tx']:.3f} "
            f"no_banking={p12['n_no_banking']} never_g={len(p12['never_g_ids'])}",
            flush=True,
        )

        print("PASS 13 Y rates × ERP × trail", flush=True)
        p13 = pass13_y_by_erp(p3["panel"], p6)
        for r in p13["rows"]:
            if r["trail"].endswith("_company"):
                print(
                    f"  {r['y']} {r['erp']} {r['trail']}: n={r['n_lab']} base={r['base']}",
                    flush=True,
                )

        print("PASS 14 post-snapshot + inv vs bank", flush=True)
        p14 = pass14_snapshot_inv(con, p3["panel"])
        print(f"  post_only={p14['n_post_only']} ids={p14['post_ids']}", flush=True)

        print("PASS 15 invoice months clipped to bank grid", flush=True)
        p15 = pass15_inv_on_grid(p3["panel"], monthly, con)
        print(
            f"  erp_pre_grid={p15['n_erp_pre']}/{p15['n_erp']} "
            f"share={p15['share_pre']:.3f} co_same_month={p15['share_co_same_month']:.3f}",
            flush=True,
        )

        print("PASS 16 Y7 first-month labels vs pre-grid inv", flush=True)
        p16 = pass16_y7_pregrid(p6, p15, monthly)
        print(
            f"  y7 sofar<=2 n={p16.get('n_early')} iss={p16.get('share_iss_early')} "
            f"sofar1_lag={p16.get('n_lag_s1')}",
            flush=True,
        )

        print("PASS 17 Sep-2026 extract vs quiet tail", flush=True)
        p17 = pass17_sep_extract(con, p3["panel"])
        print(
            f"  sep={p17['n_sep']} quiet_sep={p17['n_quiet_sep']}/{p17['n_quiet']}",
            flush=True,
        )

        print("PASS 18 holdout late-arrival coverage", flush=True)
        p18 = pass18_holdout_groups(p5.get("panel", p3["panel"]))
        p18b = pass18b_holdout_lags(p6, p2.get("panel"))
        p18["md"] = list(p18.get("md") or []) + list(p18b.get("md") or [])
        p18["n_y4_nn"] = p18b["n_y4_nn"]
        p18["n_y4_nn_from24"] = p18b["n_y4_nn_from24"]
        print(
            f"  ho_late={p18['n_late']}/{p18['n']} new_g={p18['n_new_g']} "
            f"groups_new={p18['n_groups_new']}/{p18['n_groups']} sib={p18['n_train_sib']} "
            f"y4_hhi_nn={p18b['n_y4_nn']} from24={p18b['n_y4_nn_from24']}",
            flush=True,
        )

        print("PASS 19 Y3 long so-far calendar", flush=True)
        p19 = pass19_y3_long_calendar(p6)
        print(
            f"  y3_long_n={p19.get('n')} periods={p19.get('periods')} "
            f"first={p19.get('first')} md_n={len(p19.get('md') or [])}",
            flush=True,
        )
    finally:
        con.close()

    elapsed = time.time() - t0
    extra = {
        "group": p8,
        "gfam": p9,
        "why": p10,
        "lead": p11,
        "cash": p12,
        "erp_y": p13,
        "snap": p14,
        "invclip": p15,
        "y7pre": p16,
        "sep": p17,
        "hold": p18,
        "y3cal": p19,
        "hist": p4,
    }
    write_report(p1, p2, p3, p4, p5, p6, elapsed, extra)
    append_registry(p1, p2, p5, p6, extra)
    print(f"wrote {OUT_MD} in {elapsed:.1f}s", flush=True)
    return {
        "p1": {k: v for k, v in p1.items() if k not in ("panel", "txm")},
        "p2": {k: v for k, v in p2.items() if k not in ("panel",)},
        "p3": {k: v for k, v in p3.items() if k not in ("panel",)},
        "p4": p4,
        "p5": {k: v for k, v in p5.items() if k not in ("panel",)},
        "elapsed": elapsed,
    }


if __name__ == "__main__":
    main()
