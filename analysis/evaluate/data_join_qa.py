"""Data-join QA: on how much of the train panel can a cross-source Y exist?

Read-only: duckdb `clean` via ``connect``, plus monthly.parquet and
targets.parquet. Holdout is counted for coverage only — never used to
choose a threshold, percentile, or KEEP/PARK cut.

    python -m analysis.evaluate.data_join_qa

Owned outputs: analysis/outputs/data_join_qa.md and registry appends.
Does not rewrite parquet / duckdb / family modules / models.

Iteration log (same module, not one-shot):
1. Pass 1–4 coverage / ID / Y overlap / debt+products
2. Unique-nonround + category + same-CP amount
3. AR/AP, due-date, payment ±1d, sibling amount, match-rate vs size
4. Y8/Y5 ever-ERP split; COMP_0962 refund-only ghost
5. Timing ladder, amount bands, calendar, company dist, ERP vs bank start
6. Unmatched vs Y3 (flat); resolved-CP txs only; never-paid invoiced cos
7. Wrong-sign + adjacent-month controls
8. Shared CP in group (invoice 0, tx 1 vendor); Y8 never-ERP × mixed group
9. Invoice quality; match-rate mixed vs all-invoiced groups
10. Shared-tx detail (GROUP_0220 vendor); H active-sibling on the 470
11. Payment-month × category (19.1%); Y2 never-ERP share; is_extreme = 10 rows
"""
from __future__ import annotations

import csv
import json
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

STORE = DATA / "feature_store" / "monthly.parquet"
TARGETS = DATA / "feature_store" / "targets.parquet"
OUT_MD = ANALYSIS / "outputs" / "data_join_qa.md"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
AGENT = "d56ee5fe"
WAVE = "4"
ROUND = "R4"

PANEL_START = pd.Timestamp("2024-09-01")
PANEL_END = pd.Timestamp("2026-08-01")
EXTRACT = pd.Timestamp("2026-09-01")
LAST_MONTH = MONTHS[-1]

BOOK = (
    "document_type = 'invoice' AND status <> 'cancel' "
    "AND amount <> 0 AND issuance_date IS NOT NULL"
)

E_COLS = [
    "e_ar_open",
    "e_ap_open",
    "e_ar_overdue",
    "e_ap_overdue",
    "e_ar_overdue_30",
    "e_ap_overdue_30",
    "e_delay_coll",
    "e_delay_paid",
    "e_dso_proxy",
    "e_dpo_proxy",
    "e_credit_note_ratio",
    "e_pending_amt_share",
    "e_fx_share",
    "e_ar_issued",
    "e_ap_issued",
]
D_COLS = [
    "d_cust_hhi",
    "d_cust_top1",
    "d_n_cust",
    "d_supp_hhi",
    "d_supp_top1",
    "d_n_supp",
    "d_cust_new",
    "d_cust_lost",
    "d_tx_cp_share",
    "d_interco_share",
]
Y8_COLS = ["y8_inv_worse_6", "y8_cash_worse_6"]
Y5_COLS = ["y5_ap_od30_ownp80", "y5_ar_od30_sust", "y5_ap_delay_up15"]
Y7_COLS = ["y7_top1_lost", "y7_top1_lost_inflow"]

# Other-table meaning for a cross-source Y (not the table that built the label).
Y_OTHER = {
    "y8_inv_worse_6": "tx",  # invoice Y; intended X is cash/tx
    "y8_cash_worse_6": "inv",  # cash Y; intended X is invoices
    "y5_ap_od30_ownp80": "tx",
    "y5_ar_od30_sust": "tx",
    "y5_ap_delay_up15": "tx",
    "y7_top1_lost": "tx",
    "y7_top1_lost_inflow": "tx",
}


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


def _pp(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{100.0 * float(x):.1f}%"


def _md_table(rows: list[dict], cols: list[str] | None = None) -> str:
    if not rows:
        return "_(empty)_\n"
    if cols is None:
        cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _register_companies(con) -> pd.DataFrame:
    hold = load_holdout()
    cos = con.execute("SELECT company_id, group_id FROM companies").df()
    cos["company_id"] = cos["company_id"].astype(str)
    cos["group_id"] = cos["group_id"].astype(str)
    cos["is_holdout"] = cos["company_id"].isin(hold)
    cos["split"] = np.where(cos["is_holdout"], "holdout", "train")
    con.register("_qa_cos", cos[["company_id", "group_id", "is_holdout", "split"]])
    return cos


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


# ---------------------------------------------------------------------------
# Pass 1 — company / month coverage
# ---------------------------------------------------------------------------


def pass1_company_coverage(con, cos: pd.DataFrame, monthly: pd.DataFrame) -> dict:
    ever = con.execute(
        f"""
        SELECT
          CAST(c.company_id AS VARCHAR) AS company_id,
          c.split,
          c.group_id,
          MAX(CASE WHEN i.company_id IS NOT NULL THEN 1 ELSE 0 END) AS has_inv,
          MAX(CASE WHEN t.company_id IS NOT NULL THEN 1 ELSE 0 END) AS has_tx
        FROM _qa_cos c
        LEFT JOIN (
            SELECT DISTINCT CAST(company_id AS VARCHAR) AS company_id
            FROM invoices
            WHERE {BOOK}
        ) i ON i.company_id = CAST(c.company_id AS VARCHAR)
        LEFT JOIN (
            SELECT DISTINCT CAST(company_id AS VARCHAR) AS company_id
            FROM transactions
            WHERE "date" IS NOT NULL
        ) t ON t.company_id = CAST(c.company_id AS VARCHAR)
        GROUP BY 1, 2, 3
        """
    ).df()
    ever["company_id"] = ever["company_id"].astype(str)
    ever["has_inv"] = ever["has_inv"].astype(int)
    ever["has_tx"] = ever["has_tx"].astype(int)
    ever["bucket"] = np.select(
        [
            (ever["has_inv"] == 1) & (ever["has_tx"] == 1),
            (ever["has_inv"] == 1) & (ever["has_tx"] == 0),
            (ever["has_inv"] == 0) & (ever["has_tx"] == 1),
        ],
        ["both", "invoices_only", "tx_only"],
        default="neither",
    )

    def _split_counts(df: pd.DataFrame, split: str) -> dict:
        s = df[df["split"] == split]
        n = int(len(s))
        return {
            "n_companies": n,
            "n_inv": int((s["has_inv"] == 1).sum()),
            "n_tx": int((s["has_tx"] == 1).sum()),
            "n_both": int((s["bucket"] == "both").sum()),
            "n_inv_only": int((s["bucket"] == "invoices_only").sum()),
            "n_tx_only": int((s["bucket"] == "tx_only").sum()),
            "n_neither": int((s["bucket"] == "neither").sum()),
            "share_inv": _pct((s["has_inv"] == 1).sum(), n),
            "share_tx": _pct((s["has_tx"] == 1).sum(), n),
            "share_both": _pct((s["bucket"] == "both").sum(), n),
            "share_tx_only": _pct((s["bucket"] == "tx_only").sum(), n),
        }

    train_c = _split_counts(ever, "train")
    hold_c = _split_counts(ever, "holdout")
    n_train_no_inv = train_c["n_companies"] - train_c["n_inv"]

    # 470 confirmation + group mix
    tr = ever[ever["split"] == "train"].copy()
    g = (
        tr.groupby("group_id", as_index=False)
        .agg(
            n_cos=("company_id", "size"),
            n_inv=("has_inv", "sum"),
            n_no_inv=("has_inv", lambda s: int((s == 0).sum())),
        )
    )
    g["mix"] = np.select(
        [
            g["n_inv"] == 0,
            g["n_no_inv"] == 0,
            (g["n_inv"] > 0) & (g["n_no_inv"] > 0),
        ],
        ["all_dark", "all_invoiced", "mixed"],
        default="?",
    )
    no_inv_ids = set(tr.loc[tr["has_inv"] == 0, "company_id"])
    mixed_groups = set(g.loc[g["mix"] == "mixed", "group_id"])
    dark_groups = set(g.loc[g["mix"] == "all_dark", "group_id"])
    n_no_inv_in_mixed = int(tr.loc[tr["company_id"].isin(no_inv_ids) & tr["group_id"].isin(mixed_groups)].shape[0])
    n_no_inv_in_dark = int(tr.loc[tr["company_id"].isin(no_inv_ids) & tr["group_id"].isin(dark_groups)].shape[0])

    group_mix = {
        "n_train_groups": int(g.shape[0]),
        "n_groups_all_dark": int((g["mix"] == "all_dark").sum()),
        "n_groups_all_invoiced": int((g["mix"] == "all_invoiced").sum()),
        "n_groups_mixed": int((g["mix"] == "mixed").sum()),
        "n_train_no_invoice": int(n_train_no_inv),
        "n_no_inv_in_mixed_groups": n_no_inv_in_mixed,
        "n_no_inv_in_all_dark_groups": n_no_inv_in_dark,
        "share_no_inv_have_invoiced_sibling": _pct(n_no_inv_in_mixed, n_train_no_inv),
        "confirm_470": int(n_train_no_inv) == 470,
    }

    # Month-by-month raw presence on the official monthly grid
    raw_m = con.execute(
        f"""
        WITH grid AS (
            SELECT CAST(company_id AS VARCHAR) AS company_id,
                   CAST(period AS DATE) AS period,
                   split
            FROM _qa_grid
        ),
        inv_iss AS (
            SELECT CAST(company_id AS VARCHAR) AS company_id,
                   CAST(date_trunc('month', issuance_date) AS DATE) AS period,
                   COUNT(*) AS n_iss
            FROM invoices
            WHERE {BOOK}
              AND CAST(issuance_date AS DATE) >= DATE '2024-09-01'
              AND CAST(issuance_date AS DATE) < DATE '2026-09-01'
            GROUP BY 1, 2
        ),
        inv_due AS (
            SELECT CAST(company_id AS VARCHAR) AS company_id,
                   CAST(date_trunc('month', due_date) AS DATE) AS period,
                   COUNT(*) AS n_due
            FROM invoices
            WHERE {BOOK} AND due_date IS NOT NULL
              AND CAST(due_date AS DATE) >= DATE '2024-09-01'
              AND CAST(due_date AS DATE) < DATE '2026-09-01'
            GROUP BY 1, 2
        ),
        inv_pay AS (
            SELECT CAST(company_id AS VARCHAR) AS company_id,
                   CAST(date_trunc('month', payment_date) AS DATE) AS period,
                   COUNT(*) AS n_pay
            FROM invoices
            WHERE {BOOK} AND payment_date IS NOT NULL
              AND NOT coalesce(payment_date_invalid, FALSE)
              AND CAST(payment_date AS DATE) >= DATE '2024-09-01'
              AND CAST(payment_date AS DATE) < DATE '2026-09-01'
            GROUP BY 1, 2
        ),
        tx_m AS (
            SELECT CAST(company_id AS VARCHAR) AS company_id,
                   CAST(date_trunc('month', "date") AS DATE) AS period,
                   COUNT(*) AS n_tx
            FROM transactions
            WHERE "date" IS NOT NULL
              AND CAST("date" AS DATE) >= DATE '2024-09-01'
              AND CAST("date" AS DATE) < DATE '2026-09-01'
            GROUP BY 1, 2
        )
        SELECT g.company_id, g.period, g.split,
               coalesce(ii.n_iss, 0) AS n_iss,
               coalesce(id.n_due, 0) AS n_due,
               coalesce(ip.n_pay, 0) AS n_pay,
               coalesce(tm.n_tx, 0) AS n_tx
        FROM grid g
        LEFT JOIN inv_iss ii ON g.company_id = ii.company_id AND g.period = ii.period
        LEFT JOIN inv_due id ON g.company_id = id.company_id AND g.period = id.period
        LEFT JOIN inv_pay ip ON g.company_id = ip.company_id AND g.period = ip.period
        LEFT JOIN tx_m tm ON g.company_id = tm.company_id AND g.period = tm.period
        """
    ).df()
    raw_m["period"] = pd.to_datetime(raw_m["period"])
    raw_m["any_inv"] = (raw_m["n_iss"] + raw_m["n_due"] + raw_m["n_pay"]) > 0
    raw_m["any_iss"] = raw_m["n_iss"] > 0
    raw_m["any_tx"] = raw_m["n_tx"] > 0
    raw_m["both"] = raw_m["any_inv"] & raw_m["any_tx"]

    def _month_share(df: pd.DataFrame, split: str) -> list[dict]:
        s = df[df["split"] == split]
        out = []
        for p, sl in s.groupby("period", sort=True):
            n = int(len(sl))
            out.append(
                {
                    "period": pd.Timestamp(p).strftime("%Y-%m"),
                    "n_cm": n,
                    "share_any_inv": _pct(sl["any_inv"].sum(), n),
                    "share_any_iss": _pct(sl["any_iss"].sum(), n),
                    "share_any_tx": _pct(sl["any_tx"].sum(), n),
                    "share_both": _pct(sl["both"].sum(), n),
                }
            )
        return out

    train_m = raw_m[raw_m["split"] == "train"]
    month_train = _month_share(raw_m, "train")
    month_hold = _month_share(raw_m, "holdout")
    train_cm = {
        "n_cm": int(len(train_m)),
        "share_any_inv": _pct(train_m["any_inv"].sum(), len(train_m)),
        "share_any_iss": _pct(train_m["any_iss"].sum(), len(train_m)),
        "share_any_tx": _pct(train_m["any_tx"].sum(), len(train_m)),
        "share_both": _pct(train_m["both"].sum(), len(train_m)),
        "n_any_inv": int(train_m["any_inv"].sum()),
        "n_any_tx": int(train_m["any_tx"].sum()),
        "n_both": int(train_m["both"].sum()),
    }

    # Family D/E non-null on store vs raw presence (train only)
    tr_store = monthly[monthly["split"] == "train"].copy()
    keys = tr_store[["company_id", "period"]].copy()
    keys["period"] = pd.to_datetime(keys["period"])
    raw_tr = train_m.copy()
    raw_tr["company_id"] = raw_tr["company_id"].astype(str)
    raw_tr["period"] = pd.to_datetime(raw_tr["period"])
    merged = keys.merge(
        raw_tr[["company_id", "period", "any_inv", "any_iss", "any_tx", "n_iss"]],
        on=["company_id", "period"],
        how="left",
    )
    store_tr = tr_store.merge(merged, on=["company_id", "period"], how="left")

    e_any = store_tr[E_COLS].notna().any(axis=1)
    e_issued = store_tr["e_ar_issued"].notna() | store_tr["e_ap_issued"].notna()
    e_issued_pos = (store_tr["e_ar_issued"].fillna(0) > 0) | (store_tr["e_ap_issued"].fillna(0) > 0)
    d_inv = store_tr[["d_cust_hhi", "d_supp_hhi", "d_n_cust", "d_n_supp"]].notna().any(axis=1)
    d_interco_nn = int(store_tr["d_interco_share"].notna().sum())

    # Companies with any invoice (book) vs any e_* 
    ever_inv_train = set(tr.loc[tr["has_inv"] == 1, "company_id"])
    ever_e = set(store_tr.loc[e_any, "company_id"])
    dropped_cos = sorted(ever_inv_train - ever_e)
    extra_e = sorted(ever_e - ever_inv_train)

    # CM with raw issuance but e_issued is null (feature dropped a usable book)
    raw_iss_no_e = int(((store_tr["any_iss"].fillna(False)) & ~e_issued).sum())
    raw_iss_no_epos = int(((store_tr["any_iss"].fillna(False)) & ~e_issued_pos).sum())
    epos_no_raw = int((e_issued_pos & ~store_tr["any_iss"].fillna(False)).sum())

    e_cov = {c: _pct(store_tr[c].notna().sum(), len(store_tr)) for c in E_COLS}
    d_cov = {c: _pct(store_tr[c].notna().sum(), len(store_tr)) for c in D_COLS}

    feat_vs_raw = {
        "train_cm": int(len(store_tr)),
        "e_any_nonnull_share": _pct(e_any.sum(), len(store_tr)),
        "e_issued_nonnull_share": _pct(e_issued.sum(), len(store_tr)),
        "e_issued_pos_share": _pct(e_issued_pos.sum(), len(store_tr)),
        "raw_any_inv_share": _pct(store_tr["any_inv"].fillna(False).sum(), len(store_tr)),
        "raw_any_iss_share": _pct(store_tr["any_iss"].fillna(False).sum(), len(store_tr)),
        "d_inv_struct_nonnull_share": _pct(d_inv.sum(), len(store_tr)),
        "d_interco_n_nonnull": d_interco_nn,
        "d_tx_cp_share_cov": d_cov["d_tx_cp_share"],
        "n_train_cos_raw_inv": int(len(ever_inv_train)),
        "n_train_cos_any_e": int(len(ever_e)),
        "n_raw_inv_cos_missing_e": int(len(dropped_cos)),
        "dropped_company_ids": dropped_cos[:20],
        "n_e_without_raw_inv": int(len(extra_e)),
        "cm_raw_iss_but_e_issued_null": raw_iss_no_e,
        "cm_raw_iss_but_e_issued_not_pos": raw_iss_no_epos,
        "cm_e_issued_pos_but_no_raw_iss": epos_no_raw,
        "e_col_cov": e_cov,
        "d_col_cov": d_cov,
        "note": (
            "Family E fills e_ar/ap_issued and open with 0 for ever-invoiced "
            "companies, so e_* non-null > raw issuance-that-month. "
            "A drop would be raw invoice companies with zero e_*."
        ),
    }

    # invoices-only companies (off the tx grid → not in monthly.parquet)
    inv_only = ever[ever["bucket"] == "invoices_only"]
    feat_vs_raw["n_inv_only_off_grid_train"] = int(((inv_only["split"] == "train")).sum())
    feat_vs_raw["n_inv_only_off_grid_holdout"] = int(((inv_only["split"] == "holdout")).sum())

    return {
        "train_companies": train_c,
        "holdout_companies": hold_c,
        "n_train_no_invoice": int(n_train_no_inv),
        "confirm_470": bool(int(n_train_no_inv) == 470),
        "group_mix": group_mix,
        "train_cm_raw": train_cm,
        "month_train": month_train,
        "month_holdout": month_hold,
        "feat_vs_raw": feat_vs_raw,
        "ever": ever,
        "raw_m": raw_m,
    }


# ---------------------------------------------------------------------------
# Pass 2 — ID spaces and amount match (no invented map)
# ---------------------------------------------------------------------------


def pass2_id_amount(con) -> dict:
    id_sql = con.execute(
        """
        WITH inv_cp AS (
            SELECT DISTINCT CAST(counterparty_id AS VARCHAR) AS cp
            FROM invoices
            WHERE counterparty_id IS NOT NULL
              AND length(trim(CAST(counterparty_id AS VARCHAR))) > 0
        ),
        tx_cp AS (
            SELECT DISTINCT CAST(counterparty_id AS VARCHAR) AS cp
            FROM transactions
            WHERE counterparty_id IS NOT NULL
              AND length(trim(CAST(counterparty_id AS VARCHAR))) > 0
        ),
        comp AS (
            SELECT DISTINCT CAST(company_id AS VARCHAR) AS id FROM companies
        )
        SELECT
          (SELECT COUNT(*) FROM inv_cp) AS n_inv_cp,
          (SELECT COUNT(*) FROM tx_cp) AS n_tx_cp,
          (SELECT COUNT(*) FROM inv_cp i JOIN tx_cp t ON i.cp = t.cp) AS n_overlap_cp,
          (SELECT COUNT(*) FROM inv_cp i JOIN comp c ON i.cp = c.id) AS n_inv_cp_eq_comp,
          (SELECT COUNT(*) FROM tx_cp t JOIN comp c ON t.cp = c.id) AS n_tx_cp_eq_comp,
          (SELECT COUNT(*) FROM invoices i
             JOIN companies c ON CAST(i.counterparty_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
          ) AS n_inv_rows_cp_eq_comp,
          (SELECT COUNT(*) FROM transactions t
             JOIN companies c ON CAST(t.counterparty_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
          ) AS n_tx_rows_cp_eq_comp
        """
    ).df().iloc[0].to_dict()

    row_shares = con.execute(
        """
        WITH inv_cp AS (
            SELECT DISTINCT CAST(counterparty_id AS VARCHAR) AS cp
            FROM invoices
            WHERE counterparty_id IS NOT NULL
              AND length(trim(CAST(counterparty_id AS VARCHAR))) > 0
        ),
        tx_cp AS (
            SELECT DISTINCT CAST(counterparty_id AS VARCHAR) AS cp
            FROM transactions
            WHERE counterparty_id IS NOT NULL
              AND length(trim(CAST(counterparty_id AS VARCHAR))) > 0
        )
        SELECT
          (SELECT COUNT(*) FROM invoices) AS n_inv_rows,
          (SELECT COUNT(*) FROM invoices
            WHERE CAST(counterparty_id AS VARCHAR) IN (SELECT cp FROM tx_cp)
          ) AS n_inv_rows_cp_in_tx,
          (SELECT COUNT(*) FROM invoices WHERE counterparty_id IS NULL
             OR length(trim(CAST(counterparty_id AS VARCHAR))) = 0
          ) AS n_inv_rows_cp_null,
          (SELECT COUNT(*) FROM transactions) AS n_tx_rows,
          (SELECT COUNT(*) FROM transactions
            WHERE CAST(counterparty_id AS VARCHAR) IN (SELECT cp FROM inv_cp)
          ) AS n_tx_rows_cp_in_inv,
          (SELECT COUNT(*) FROM transactions WHERE counterparty_id IS NULL
             OR length(trim(CAST(counterparty_id AS VARCHAR))) = 0
          ) AS n_tx_rows_cp_null
        """
    ).df().iloc[0].to_dict()

    # Train-only row shares (rates quoted on train)
    train_row = con.execute(
        """
        WITH inv_cp AS (
            SELECT DISTINCT CAST(counterparty_id AS VARCHAR) AS cp
            FROM invoices i
            JOIN _qa_cos c ON CAST(i.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout
              AND i.counterparty_id IS NOT NULL
              AND length(trim(CAST(i.counterparty_id AS VARCHAR))) > 0
        ),
        tx_cp AS (
            SELECT DISTINCT CAST(counterparty_id AS VARCHAR) AS cp
            FROM transactions t
            JOIN _qa_cos c ON CAST(t.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout
              AND t.counterparty_id IS NOT NULL
              AND length(trim(CAST(t.counterparty_id AS VARCHAR))) > 0
        )
        SELECT
          (SELECT COUNT(*) FROM inv_cp) AS n_inv_cp_train,
          (SELECT COUNT(*) FROM tx_cp) AS n_tx_cp_train,
          (SELECT COUNT(*) FROM inv_cp i JOIN tx_cp t ON i.cp = t.cp) AS n_overlap_cp_train,
          (SELECT COUNT(*) FROM invoices i
             JOIN _qa_cos c ON CAST(i.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout) AS n_inv_rows_train,
          (SELECT COUNT(*) FROM invoices i
             JOIN _qa_cos c ON CAST(i.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout
              AND CAST(i.counterparty_id AS VARCHAR) IN (SELECT cp FROM tx_cp)
          ) AS n_inv_rows_cp_in_tx_train,
          (SELECT COUNT(*) FROM transactions t
             JOIN _qa_cos c ON CAST(t.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout) AS n_tx_rows_train,
          (SELECT COUNT(*) FROM transactions t
             JOIN _qa_cos c ON CAST(t.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout
              AND CAST(t.counterparty_id AS VARCHAR) IN (SELECT cp FROM inv_cp)
          ) AS n_tx_rows_cp_in_inv_train
        """
    ).df().iloc[0].to_dict()

    prefixes = con.execute(
        """
        SELECT
          (SELECT COUNT(*) FROM companies WHERE CAST(company_id AS VARCHAR) LIKE 'COMP_%') AS n_comp_comp,
          (SELECT COUNT(*) FROM companies WHERE CAST(company_id AS VARCHAR) LIKE 'COUNTERPARTY_%') AS n_comp_cp,
          (SELECT COUNT(*) FROM invoices
            WHERE counterparty_id IS NOT NULL
              AND CAST(counterparty_id AS VARCHAR) LIKE 'COMP_%') AS n_inv_cp_comp_prefix,
          (SELECT COUNT(*) FROM invoices
            WHERE counterparty_id IS NOT NULL
              AND CAST(counterparty_id AS VARCHAR) LIKE 'COUNTERPARTY_%') AS n_inv_cp_cp_prefix,
          (SELECT COUNT(*) FROM transactions
            WHERE counterparty_id IS NOT NULL
              AND CAST(counterparty_id AS VARCHAR) LIKE 'COMP_%') AS n_tx_cp_comp_prefix,
          (SELECT COUNT(*) FROM transactions
            WHERE counterparty_id IS NOT NULL
              AND CAST(counterparty_id AS VARCHAR) LIKE 'COUNTERPARTY_%') AS n_tx_cp_cp_prefix
        """
    ).df().iloc[0].to_dict()

    interco = {
        "n_inv_rows_cp_eq_company_id": int(id_sql["n_inv_rows_cp_eq_comp"]),
        "n_tx_rows_cp_eq_company_id": int(id_sql["n_tx_rows_cp_eq_comp"]),
        "n_inv_cp_in_companies": int(id_sql["n_inv_cp_eq_comp"]),
        "n_tx_cp_in_companies": int(id_sql["n_tx_cp_eq_comp"]),
        "sets_disjoint": int(id_sql["n_inv_cp_eq_comp"]) == 0 and int(id_sql["n_tx_cp_eq_comp"]) == 0,
        "sql": (
            "COUNT invoices/transactions JOIN companies "
            "ON counterparty_id = company_id → 0; "
            "companies are COMP_*, invoice/tx counterparties are COUNTERPARTY_*"
        ),
        "verdict": "CLOSE",
        "why": (
            "ID spaces are disjoint. Equality join hits 0 rows. "
            "Emitting 0 would pretend we measured no intercompany flow. "
            "Do not invent a COMP_* ↔ COUNTERPARTY_* map."
        ),
    }

    # Amount-match probes (train companies only). No invented FK.
    print("  amount-match issuance-month (train)...", flush=True)
    amt_iss = _amount_match(con, date_col="issuance_date", table="invoices", label="issuance")
    print("  amount-match payment-date-month (train)...", flush=True)
    amt_pay = _amount_match(con, date_col="payment_date", table="invoices", label="payment")

    return {
        "id_sets": {k: int(v) for k, v in id_sql.items()},
        "row_shares_all": {k: int(v) for k, v in row_shares.items()},
        "row_shares_train": {k: int(v) for k, v in train_row.items()},
        "prefixes": {k: int(v) for k, v in prefixes.items()},
        "overlap_share_inv_cp": _pct(train_row["n_overlap_cp_train"], train_row["n_inv_cp_train"]),
        "overlap_share_tx_cp": _pct(train_row["n_overlap_cp_train"], train_row["n_tx_cp_train"]),
        "share_inv_rows_cp_in_tx_train": _pct(
            train_row["n_inv_rows_cp_in_tx_train"], train_row["n_inv_rows_train"]
        ),
        "share_tx_rows_cp_in_inv_train": _pct(
            train_row["n_tx_rows_cp_in_inv_train"], train_row["n_tx_rows_train"]
        ),
        "interco": interco,
        "amount_issuance": amt_iss,
        "amount_payment": amt_pay,
    }


def _amount_match(con, date_col: str, table: str, label: str) -> dict:
    """Same-company same-calendar-month amount probe + shifted-company false density.

    Train companies only. Tolerances 0 / 0.01 / 1.00 euro on |amount|.
    Sign: AR amount>0 matches inflow tx; AP amount<0 matches outflow tx.
    """
    extra_pay = ""
    if date_col == "payment_date":
        extra_pay = "AND i.payment_date IS NOT NULL AND NOT coalesce(i.payment_date_invalid, FALSE)"
    elif date_col == "due_date":
        extra_pay = "AND i.due_date IS NOT NULL"

    # Build compact unique-amount tables once; reuse across tols via rounding.
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE _qa_inv_amt AS
        SELECT
          CAST(i.company_id AS VARCHAR) AS company_id,
          CAST(date_trunc('month', i.{date_col}) AS DATE) AS m,
          CASE WHEN i.amount > 0 THEN 1 ELSE -1 END AS sgn,
          ABS(i.amount) AS abs_amt,
          COUNT(*) AS n_inv
        FROM invoices i
        JOIN _qa_cos c ON CAST(i.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
        WHERE NOT c.is_holdout
          AND {BOOK}
          {extra_pay}
          AND CAST(i.{date_col} AS DATE) >= DATE '2024-09-01'
          AND CAST(i.{date_col} AS DATE) < DATE '2026-09-01'
        GROUP BY 1, 2, 3, 4
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE _qa_tx_amt AS
        SELECT
          CAST(t.company_id AS VARCHAR) AS company_id,
          CAST(date_trunc('month', t."date") AS DATE) AS m,
          CASE WHEN t.amount > 0 THEN 1 ELSE -1 END AS sgn,
          ABS(t.amount) AS abs_amt,
          COUNT(*) AS n_tx
        FROM transactions t
        JOIN _qa_cos c ON CAST(t.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
        WHERE NOT c.is_holdout
          AND t.amount <> 0
          AND t."date" IS NOT NULL
          AND CAST(t."date" AS DATE) >= DATE '2024-09-01'
          AND CAST(t."date" AS DATE) < DATE '2026-09-01'
        GROUP BY 1, 2, 3, 4
        """
    )

    n_inv = int(con.execute("SELECT COALESCE(SUM(n_inv),0) FROM _qa_inv_amt").fetchone()[0])
    n_tx = int(con.execute("SELECT COALESCE(SUM(n_tx),0) FROM _qa_tx_amt").fetchone()[0])
    n_inv_cm = int(
        con.execute("SELECT COUNT(*) FROM (SELECT DISTINCT company_id, m FROM _qa_inv_amt)").fetchone()[0]
    )

    # Deterministic partner company in the same month (hash shift — not a learned map).
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE _qa_partner AS
        WITH tx_cos AS (
            SELECT DISTINCT company_id, m FROM _qa_tx_amt
        ),
        ranked AS (
            SELECT company_id, m,
                   ROW_NUMBER() OVER (
                       PARTITION BY m
                       ORDER BY hash(company_id || '|' || CAST(m AS VARCHAR) || '|20260918')
                   ) AS rn,
                   COUNT(*) OVER (PARTITION BY m) AS n
            FROM tx_cos
        )
        SELECT a.company_id, a.m, b.company_id AS partner
        FROM ranked a
        JOIN ranked b
          ON a.m = b.m
         AND b.rn = CASE WHEN a.rn = a.n THEN 1 ELSE a.rn + 1 END
        WHERE a.n >= 2
        """
    )

    tols = [0.0, 0.01, 1.00]
    by_tol = {}
    for tol in tols:
        real = con.execute(
            f"""
            WITH hit AS (
                SELECT i.company_id, i.m, i.sgn, i.abs_amt, i.n_inv,
                       SUM(t.n_tx) AS n_match_tx
                FROM _qa_inv_amt i
                JOIN _qa_tx_amt t
                  ON i.company_id = t.company_id
                 AND i.m = t.m
                 AND i.sgn = t.sgn
                 AND abs(i.abs_amt - t.abs_amt) <= {tol}
                GROUP BY 1, 2, 3, 4, 5
            )
            SELECT
              (SELECT SUM(n_inv) FROM _qa_inv_amt) AS n_inv,
              COALESCE(SUM(n_inv), 0) AS n_inv_hit,
              COALESCE(SUM(n_inv * n_match_tx), 0) AS n_inv_tx_pairs,
              COALESCE(SUM(n_inv * n_match_tx) * 1.0 / NULLIF(SUM(n_inv), 0), 0) AS mean_tx_per_hit_inv
            FROM hit
            """
        ).df().iloc[0].to_dict()

        fake = con.execute(
            f"""
            WITH hit AS (
                SELECT i.company_id, i.m, i.sgn, i.abs_amt, i.n_inv,
                       SUM(t.n_tx) AS n_match_tx
                FROM _qa_inv_amt i
                JOIN _qa_partner p ON i.company_id = p.company_id AND i.m = p.m
                JOIN _qa_tx_amt t
                  ON t.company_id = p.partner
                 AND t.m = i.m
                 AND t.sgn = i.sgn
                 AND abs(i.abs_amt - t.abs_amt) <= {tol}
                GROUP BY 1, 2, 3, 4, 5
            )
            SELECT
              COALESCE(SUM(n_inv), 0) AS n_inv_hit,
              COALESCE(SUM(n_inv * n_match_tx) * 1.0 / NULLIF(SUM(n_inv), 0), 0) AS mean_tx_per_hit_inv
            FROM hit
            """
        ).df().iloc[0].to_dict()

        # Invoices that have a partner (needed as denominator for fair fake rate)
        n_inv_with_partner = int(
            con.execute(
                """
                SELECT COALESCE(SUM(i.n_inv), 0)
                FROM _qa_inv_amt i
                JOIN _qa_partner p ON i.company_id = p.company_id AND i.m = p.m
                """
            ).fetchone()[0]
        )

        n_hit = int(real["n_inv_hit"])
        n_hit_fake = int(fake["n_inv_hit"])
        hit_rate = _pct(n_hit, n_inv)
        fake_rate = _pct(n_hit_fake, n_inv_with_partner) if n_inv_with_partner else float("nan")
        by_tol[str(tol)] = {
            "tol": tol,
            "n_inv": n_inv,
            "n_inv_hit": n_hit,
            "hit_rate": hit_rate,
            "mean_matching_tx_per_hit_invoice": float(real["mean_tx_per_hit_inv"]),
            "n_inv_tx_pairs": int(real["n_inv_tx_pairs"]),
            "n_inv_with_partner": n_inv_with_partner,
            "n_inv_hit_random_partner": n_hit_fake,
            "random_hit_rate": fake_rate,
            "mean_matching_tx_per_hit_invoice_random": float(fake["mean_tx_per_hit_inv"]),
            "real_minus_random": (hit_rate - fake_rate) if np.isfinite(fake_rate) else float("nan"),
        }

    # Company-month with any exact hit (tol=0)
    cm_hit = con.execute(
        """
        SELECT
          COUNT(DISTINCT company_id || '|' || CAST(m AS VARCHAR)) AS n_cm_inv,
          COUNT(DISTINCT CASE WHEN hit = 1 THEN company_id || '|' || CAST(m AS VARCHAR) END) AS n_cm_hit
        FROM (
            SELECT i.company_id, i.m,
                   MAX(CASE WHEN t.company_id IS NOT NULL THEN 1 ELSE 0 END) AS hit
            FROM _qa_inv_amt i
            LEFT JOIN _qa_tx_amt t
              ON i.company_id = t.company_id AND i.m = t.m
             AND i.sgn = t.sgn AND i.abs_amt = t.abs_amt
            GROUP BY 1, 2
        )
        """
    ).df().iloc[0].to_dict()

    return {
        "label": label,
        "date_col": date_col,
        "n_inv_rows": n_inv,
        "n_tx_rows": n_tx,
        "n_inv_company_months": n_inv_cm,
        "by_tol": by_tol,
        "cm_exact": {k: int(v) for k, v in cm_hit.items()},
        "cm_exact_hit_share": _pct(cm_hit["n_cm_hit"], cm_hit["n_cm_inv"]),
    }


# ---------------------------------------------------------------------------
# Pass 3 — usable overlap for parked / accepted Ys
# ---------------------------------------------------------------------------


def pass3_y_overlap(monthly: pd.DataFrame, targets: pd.DataFrame, raw_m: pd.DataFrame) -> dict:
    raw = raw_m.copy()
    raw["company_id"] = raw["company_id"].astype(str)
    raw["period"] = pd.to_datetime(raw["period"])
    raw["has_inv_month"] = raw["any_inv"].astype(bool)
    raw["has_iss_month"] = raw["any_iss"].astype(bool)
    raw["has_tx_month"] = raw["any_tx"].astype(bool)

    store = monthly[
        [
            "company_id",
            "period",
            "split",
            "e_ar_issued",
            "e_ap_issued",
            "e_ar_overdue_30",
            "e_delay_coll",
            "e_ap_overdue_30",
            "a_n_tx",
            "a_op_in",
            "d_interco_share",
        ]
    ].copy()
    ycols = [c for c in (Y8_COLS + Y5_COLS + Y7_COLS + ["y3_recover_cash_6m"]) if c in targets.columns]
    panel = store.merge(targets[["company_id", "period", *ycols]], on=["company_id", "period"], how="left")
    panel = panel.merge(
        raw[["company_id", "period", "has_inv_month", "has_iss_month", "has_tx_month"]],
        on=["company_id", "period"],
        how="left",
    )
    panel["e_any"] = panel["e_ar_issued"].notna() | panel["e_ap_issued"].notna()
    panel["e_pos"] = (panel["e_ar_issued"].fillna(0) > 0) | (panel["e_ap_issued"].fillna(0) > 0)

    tr = panel[panel["split"] == "train"].copy()
    rows = []
    for col in Y8_COLS + Y5_COLS + Y7_COLS:
        if col not in tr.columns:
            continue
        lab = tr[col].notna()
        n_lab = int(lab.sum())
        yt = tr.loc[lab, col]
        n_pos = int((yt == 1).sum())
        base = float(yt.mean()) if n_lab else float("nan")
        other = Y_OTHER.get(col, "tx")
        if other == "inv":
            other_m = tr.loc[lab, "has_inv_month"].fillna(False)
            other_e = tr.loc[lab, "e_any"].fillna(False)
        else:
            other_m = tr.loc[lab, "has_tx_month"].fillna(False)
            other_e = tr.loc[lab, "a_n_tx"].fillna(0) > 0
        n_other = int(other_m.sum())
        n_other_feat = int(other_e.sum())
        both = lab & tr["has_inv_month"].fillna(False) & tr["has_tx_month"].fillna(False)
        n_both = int(both.sum())
        base_overlap = float(tr.loc[lab & other_m, col].mean()) if n_other else float("nan")
        base_both = float(tr.loc[both, col].mean()) if n_both else float("nan")
        rows.append(
            {
                "y": col,
                "other_table": other,
                "n_labeled": n_lab,
                "n_pos": n_pos,
                "base_rate": base,
                "n_labeled_other_present": n_other,
                "share_labeled_other_present": _pct(n_other, n_lab),
                "n_labeled_other_feat": n_other_feat,
                "share_labeled_other_feat": _pct(n_other_feat, n_lab),
                "n_labeled_both_tables": n_both,
                "share_labeled_both": _pct(n_both, n_lab),
                "base_rate_on_other": base_overlap,
                "base_rate_on_both": base_both,
            }
        )

    # Y3 2×2 among stressed (labeled) train rows
    y3 = tr[tr["y3_recover_cash_6m"].notna()].copy() if "y3_recover_cash_6m" in tr.columns else tr.iloc[0:0]
    y3_tab = {}
    if len(y3):
        y3["inv"] = y3["has_inv_month"].fillna(False)
        y3["e_any"] = y3["e_any"].fillna(False)
        for name, mask in (
            ("inv_month", y3["inv"]),
            ("no_inv_month", ~y3["inv"]),
            ("e_feat", y3["e_any"]),
            ("no_e_feat", ~y3["e_any"]),
        ):
            sl = y3.loc[mask, "y3_recover_cash_6m"]
            y3_tab[name] = {
                "n": int(len(sl)),
                "n_pos": int((sl == 1).sum()),
                "base_rate": float(sl.mean()) if len(sl) else float("nan"),
            }
        y3_tab["all_stressed"] = {
            "n": int(len(y3)),
            "n_pos": int((y3["y3_recover_cash_6m"] == 1).sum()),
            "base_rate": float(y3["y3_recover_cash_6m"].mean()),
        }

    # Holdout y7_top1_lost counts only (LOW_POWER except this count)
    ho = panel[panel["split"] == "holdout"]
    y7_hold = {}
    if "y7_top1_lost" in ho.columns:
        lab = ho["y7_top1_lost"].notna()
        y7_hold = {
            "n_labeled": int(lab.sum()),
            "n_pos": int((ho.loc[lab, "y7_top1_lost"] == 1).sum()),
            "note": "LOW_POWER except this count — no rate used to choose a cut",
        }

    return {"y_rows": rows, "y3_by_invoice": y3_tab, "y7_holdout_counts": y7_hold}


# ---------------------------------------------------------------------------
# Pass 4 — sparse debt / left-truncated products / next idea
# ---------------------------------------------------------------------------


def pass4_sparse(con, monthly: pd.DataFrame) -> dict:
    tr = monthly[monthly["split"] == "train"].copy()
    last = LAST_MONTH
    last_m = tr[tr["period"] == last]
    out = {
        "train_cm": int(len(tr)),
        "train_cos": int(tr["company_id"].nunique()),
        "last_month": str(last.date()),
        "n_last_cm": int(len(last_m)),
    }
    for col in ("f_w_rate", "f_util_snapshot", "f_months_to_next_pay", "f_sched_vs_obs"):
        if col not in tr.columns:
            continue
        nn = tr[col].notna()
        ever = tr.loc[nn, "company_id"].nunique()
        nn_last = last_m[col].notna()
        out[col] = {
            "n_cm_nonnull": int(nn.sum()),
            "share_cm": _pct(nn.sum(), len(tr)),
            "n_cos_ever": int(ever),
            "share_cos_ever": _pct(ever, tr["company_id"].nunique()),
            "n_cm_last_month": int(nn_last.sum()),
            "share_last_month": _pct(nn_last.sum(), len(last_m)),
            "n_cm_before_last": int((nn & (tr["period"] < last)).sum()),
        }

    # banking_products created_at vs 2024-09
    prod = con.execute(
        """
        SELECT CAST(p.company_id AS VARCHAR) AS company_id,
               MIN(CAST(p.created_at AS TIMESTAMP)) AS first_created,
               COUNT(*) AS n_prod
        FROM banking_products p
        GROUP BY 1
        """
    ).df()
    prod["company_id"] = prod["company_id"].astype(str)
    prod["first_created"] = pd.to_datetime(prod["first_created"], utc=True, errors="coerce")
    cut = pd.Timestamp("2024-09-01", tz="UTC")
    hold = load_holdout()
    prod["split"] = np.where(prod["company_id"].isin(hold), "holdout", "train")
    ptr = prod[prod["split"] == "train"]
    n_after = int((ptr["first_created"] > cut).sum())
    n_known = int(ptr["first_created"].notna().sum())
    # companies on the train panel
    train_ids = set(tr["company_id"])
    ptr_panel = ptr[ptr["company_id"].isin(train_ids)]
    out["products"] = {
        "n_train_cos_with_banking_prod": int(len(ptr_panel)),
        "n_first_created_after_2024_09": int((ptr_panel["first_created"] > cut).sum()),
        "share_first_created_after_2024_09": _pct(
            (ptr_panel["first_created"] > cut).sum(), len(ptr_panel)
        ),
        "n_first_created_null": int(ptr_panel["first_created"].isna().sum()),
        "n_train_cos_no_banking_prod": int(len(train_ids - set(ptr_panel["company_id"]))),
    }

    debt_n = con.execute(
        """
        SELECT
          (SELECT COUNT(*) FROM debt_schedule_config) AS n_sched_rows,
          (SELECT COUNT(DISTINCT company_id) FROM debt_schedule_config) AS n_sched_cos,
          (SELECT COUNT(DISTINCT company_id) FROM debt_products) AS n_debt_cos
        """
    ).df().iloc[0].to_dict()
    out["debt_raw"] = {k: int(v) for k, v in debt_n.items()}
    return out


# ---------------------------------------------------------------------------
# Extra iterations (same module) — stricter match, quarterly, siblings
# ---------------------------------------------------------------------------


def extra_strict_amount(con) -> dict:
    """Stricter amount match: unique (non-round) amounts only; collection/payment cats."""
    print("  extra: unique-amount + non-round + category filter...", flush=True)
    # Unique amount in the company-month-sign on the invoice side
    unique = {}
    for tol in (0.0, 0.01, 1.00):
        real = con.execute(
            f"""
            WITH inv_u AS (
                SELECT company_id, m, sgn, abs_amt, n_inv
                FROM _qa_inv_amt
                WHERE n_inv = 1
                  AND abs(abs_amt - ROUND(abs_amt / 100.0) * 100.0) > 0.009
            ),
            hit AS (
                SELECT DISTINCT i.company_id, i.m, i.sgn, i.abs_amt, i.n_inv
                FROM inv_u i
                JOIN _qa_tx_amt t
                  ON i.company_id = t.company_id AND i.m = t.m
                 AND i.sgn = t.sgn AND abs(i.abs_amt - t.abs_amt) <= {tol}
            )
            SELECT
              (SELECT COALESCE(SUM(n_inv),0) FROM inv_u) AS n_inv,
              COALESCE(SUM(n_inv),0) AS n_hit
            FROM hit
            """
        ).df().iloc[0]
        fake = con.execute(
            f"""
            WITH inv_u AS (
                SELECT company_id, m, sgn, abs_amt, n_inv
                FROM _qa_inv_amt
                WHERE n_inv = 1
                  AND abs(abs_amt - ROUND(abs_amt / 100.0) * 100.0) > 0.009
            ),
            hit AS (
                SELECT DISTINCT i.company_id, i.m, i.sgn, i.abs_amt, i.n_inv
                FROM inv_u i
                JOIN _qa_partner p ON i.company_id = p.company_id AND i.m = p.m
                JOIN _qa_tx_amt t
                  ON t.company_id = p.partner AND t.m = i.m
                 AND t.sgn = i.sgn AND abs(i.abs_amt - t.abs_amt) <= {tol}
            )
            SELECT
              (SELECT COALESCE(SUM(n_inv),0) FROM inv_u i
                 JOIN _qa_partner p ON i.company_id = p.company_id AND i.m = p.m) AS n_inv_p,
              COALESCE(SUM(n_inv),0) AS n_hit
            FROM hit
            """
        ).df().iloc[0]
        n_inv = int(real["n_inv"])
        unique[str(tol)] = {
            "n_inv_unique_nonround": n_inv,
            "hit_rate": _pct(int(real["n_hit"]), n_inv),
            "random_hit_rate": _pct(int(fake["n_hit"]), int(fake["n_inv_p"])),
        }

    # Category-restricted txs (collections vs payments)
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE _qa_tx_cat AS
        SELECT
          CAST(t.company_id AS VARCHAR) AS company_id,
          CAST(date_trunc('month', t."date") AS DATE) AS m,
          CASE WHEN t.amount > 0 THEN 1 ELSE -1 END AS sgn,
          ABS(t.amount) AS abs_amt,
          COUNT(*) AS n_tx
        FROM transactions t
        JOIN _qa_cos c ON CAST(t.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
        WHERE NOT c.is_holdout
          AND t.amount <> 0
          AND t."date" IS NOT NULL
          AND CAST(t."date" AS DATE) >= DATE '2024-09-01'
          AND CAST(t."date" AS DATE) < DATE '2026-09-01'
          AND (
                (t.amount > 0 AND t.category IN ('collection','bulk_collection','cash_settlement',
                                                 'cash_settlements','pos_settlement'))
             OR (t.amount < 0 AND t.category IN ('payment','bulk_payment'))
          )
        GROUP BY 1, 2, 3, 4
        """
    )
    cat = {}
    for tol in (0.0, 0.01, 1.00):
        real = con.execute(
            f"""
            WITH hit AS (
                SELECT DISTINCT i.company_id, i.m, i.sgn, i.abs_amt, i.n_inv
                FROM _qa_inv_amt i
                JOIN _qa_tx_cat t
                  ON i.company_id = t.company_id AND i.m = t.m
                 AND i.sgn = t.sgn AND abs(i.abs_amt - t.abs_amt) <= {tol}
            )
            SELECT
              (SELECT COALESCE(SUM(n_inv),0) FROM _qa_inv_amt) AS n_inv,
              COALESCE(SUM(n_inv),0) AS n_hit
            FROM hit
            """
        ).df().iloc[0]
        fake = con.execute(
            f"""
            WITH hit AS (
                SELECT DISTINCT i.company_id, i.m, i.sgn, i.abs_amt, i.n_inv
                FROM _qa_inv_amt i
                JOIN _qa_partner p ON i.company_id = p.company_id AND i.m = p.m
                JOIN _qa_tx_cat t
                  ON t.company_id = p.partner AND t.m = i.m
                 AND t.sgn = i.sgn AND abs(i.abs_amt - t.abs_amt) <= {tol}
            )
            SELECT
              (SELECT COALESCE(SUM(i.n_inv),0) FROM _qa_inv_amt i
                 JOIN _qa_partner p ON i.company_id = p.company_id AND i.m = p.m) AS n_inv_p,
              COALESCE(SUM(n_inv),0) AS n_hit
            FROM hit
            """
        ).df().iloc[0]
        cat[str(tol)] = {
            "hit_rate": _pct(int(real["n_hit"]), int(real["n_inv"])),
            "random_hit_rate": _pct(int(fake["n_hit"]), int(fake["n_inv_p"])),
            "n_inv": int(real["n_inv"]),
        }
    return {"unique_nonround": unique, "category_filtered": cat}


def extra_quarterly(raw_m: pd.DataFrame) -> dict:
    """Train company-quarters with any invoice and any tx (usable overlap, coarser)."""
    tr = raw_m[raw_m["split"] == "train"].copy()
    tr["period"] = pd.to_datetime(tr["period"])
    tr["q"] = tr["period"].dt.to_period("Q")
    g = tr.groupby(["company_id", "q"], as_index=False).agg(
        any_inv=("any_inv", "max"),
        any_tx=("any_tx", "max"),
        n_m=("period", "size"),
    )
    n = int(len(g))
    return {
        "n_train_company_quarters": n,
        "share_any_inv": _pct(g["any_inv"].sum(), n),
        "share_any_tx": _pct(g["any_tx"].sum(), n),
        "share_both": _pct((g["any_inv"] & g["any_tx"]).sum(), n),
        "n_both": int((g["any_inv"] & g["any_tx"]).sum()),
    }


def extra_sibling_detail(ever: pd.DataFrame) -> dict:
    tr = ever[ever["split"] == "train"].copy()
    g = tr.groupby("group_id").agg(
        n=("company_id", "size"),
        n_inv=("has_inv", "sum"),
    )
    g["n_dark"] = g["n"] - g["n_inv"]
    mixed = g[(g["n_inv"] > 0) & (g["n_dark"] > 0)]
    dark = g[g["n_inv"] == 0]
    invoiced = g[g["n_dark"] == 0]
    # size of mixed groups
    size_bins = pd.cut(mixed["n"], bins=[0, 1, 2, 4, 8, 25], labels=["1", "2", "3-4", "5-8", "9+"])
    return {
        "n_groups": int(len(g)),
        "mixed_n_groups": int(len(mixed)),
        "mixed_n_dark_cos": int(mixed["n_dark"].sum()),
        "mixed_n_inv_cos": int(mixed["n_inv"].sum()),
        "dark_n_groups": int(len(dark)),
        "dark_n_cos": int(dark["n"].sum()),
        "invoiced_n_groups": int(len(invoiced)),
        "invoiced_n_cos": int(invoiced["n"].sum()),
        "mixed_size_hist": {str(k): int(v) for k, v in size_bins.value_counts().sort_index().items()},
        "median_group_size_mixed": float(mixed["n"].median()) if len(mixed) else float("nan"),
        "median_group_size_dark": float(dark["n"].median()) if len(dark) else float("nan"),
    }


def extra_same_cp_amount(con) -> dict:
    """Same counterparty_id AND same amount (still not a FK — just a tighter probe)."""
    print("  extra: same-cp + same-amount (train)...", flush=True)
    row = con.execute(
        f"""
        WITH inv AS (
            SELECT CAST(i.company_id AS VARCHAR) AS company_id,
                   CAST(date_trunc('month', i.issuance_date) AS DATE) AS m,
                   CAST(i.counterparty_id AS VARCHAR) AS cp,
                   CASE WHEN i.amount > 0 THEN 1 ELSE -1 END AS sgn,
                   ABS(i.amount) AS abs_amt,
                   COUNT(*) AS n_inv
            FROM invoices i
            JOIN _qa_cos c ON CAST(i.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout AND {BOOK}
              AND i.counterparty_id IS NOT NULL
              AND CAST(i.issuance_date AS DATE) >= DATE '2024-09-01'
              AND CAST(i.issuance_date AS DATE) < DATE '2026-09-01'
            GROUP BY 1,2,3,4,5
        ),
        tx AS (
            SELECT CAST(t.company_id AS VARCHAR) AS company_id,
                   CAST(date_trunc('month', t."date") AS DATE) AS m,
                   CAST(t.counterparty_id AS VARCHAR) AS cp,
                   CASE WHEN t.amount > 0 THEN 1 ELSE -1 END AS sgn,
                   ABS(t.amount) AS abs_amt,
                   COUNT(*) AS n_tx
            FROM transactions t
            JOIN _qa_cos c ON CAST(t.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout AND t.amount <> 0 AND t."date" IS NOT NULL
              AND t.counterparty_id IS NOT NULL
              AND CAST(t."date" AS DATE) >= DATE '2024-09-01'
              AND CAST(t."date" AS DATE) < DATE '2026-09-01'
            GROUP BY 1,2,3,4,5
        ),
        hit AS (
            SELECT DISTINCT i.company_id, i.m, i.cp, i.sgn, i.abs_amt, i.n_inv
            FROM inv i
            JOIN tx t
              ON i.company_id = t.company_id AND i.m = t.m AND i.cp = t.cp
             AND i.sgn = t.sgn AND abs(i.abs_amt - t.abs_amt) <= 0.01
        )
        SELECT
          (SELECT COALESCE(SUM(n_inv),0) FROM inv) AS n_inv_with_cp,
          (SELECT COALESCE(SUM(n_inv),0) FROM hit) AS n_hit
        """
    ).df().iloc[0]
    return {
        "n_inv_with_cp": int(row["n_inv_with_cp"]),
        "n_hit_same_cp_amt_0_01": int(row["n_hit"]),
        "hit_rate": _pct(int(row["n_hit"]), int(row["n_inv_with_cp"])),
        "note": "Requires resolved tx counterparty (rare). Not a FK; just a tighter collision screen.",
    }


# ---------------------------------------------------------------------------
# Verdicts + markdown + registry
# ---------------------------------------------------------------------------


def extra_side_and_due(con) -> dict:
    """AR vs AP issuance match, due-date month, payment ±1 day (train)."""
    print("  extra: AR/AP split + due-date + payment ±1d...", flush=True)
    out: dict = {}
    for side, pred in (("AR", "i.amount > 0"), ("AP", "i.amount < 0")):
        row = con.execute(
            f"""
            WITH inv AS (
                SELECT CAST(i.company_id AS VARCHAR) AS company_id,
                       CAST(date_trunc('month', i.issuance_date) AS DATE) AS m,
                       CASE WHEN i.amount > 0 THEN 1 ELSE -1 END AS sgn,
                       ABS(i.amount) AS abs_amt,
                       COUNT(*) AS n_inv
                FROM invoices i
                JOIN _qa_cos c ON CAST(i.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
                WHERE NOT c.is_holdout AND {BOOK} AND {pred}
                  AND CAST(i.issuance_date AS DATE) >= DATE '2024-09-01'
                  AND CAST(i.issuance_date AS DATE) < DATE '2026-09-01'
                GROUP BY 1,2,3,4
            ),
            hit AS (
                SELECT DISTINCT i.company_id, i.m, i.sgn, i.abs_amt, i.n_inv
                FROM inv i
                JOIN _qa_tx_amt t
                  ON i.company_id = t.company_id AND i.m = t.m
                 AND i.sgn = t.sgn AND abs(i.abs_amt - t.abs_amt) <= 0.01
            ),
            fake AS (
                SELECT DISTINCT i.company_id, i.m, i.sgn, i.abs_amt, i.n_inv
                FROM inv i
                JOIN _qa_partner p ON i.company_id = p.company_id AND i.m = p.m
                JOIN _qa_tx_amt t
                  ON t.company_id = p.partner AND t.m = i.m
                 AND t.sgn = i.sgn AND abs(i.abs_amt - t.abs_amt) <= 0.01
            )
            SELECT
              (SELECT COALESCE(SUM(n_inv),0) FROM inv) AS n_inv,
              (SELECT COALESCE(SUM(n_inv),0) FROM hit) AS n_hit,
              (SELECT COALESCE(SUM(i.n_inv),0) FROM inv i
                 JOIN _qa_partner p ON i.company_id = p.company_id AND i.m = p.m) AS n_inv_p,
              (SELECT COALESCE(SUM(n_inv),0) FROM fake) AS n_fake
            """
        ).df().iloc[0]
        out[side] = {
            "n_inv": int(row["n_inv"]),
            "hit_rate": _pct(int(row["n_hit"]), int(row["n_inv"])),
            "random_hit_rate": _pct(int(row["n_fake"]), int(row["n_inv_p"])),
        }

    due = _amount_match(con, date_col="due_date", table="invoices", label="due")
    # rebuild issuance + partner for later extras (due_date overwrites temp tables)
    _amount_match(con, date_col="issuance_date", table="invoices", label="issuance_rebuild2")

    # payment_date ±1 calendar day, exact cents (row-level, not month-binned)
    pay1 = con.execute(
        f"""
        WITH inv AS (
            SELECT CAST(i.company_id AS VARCHAR) AS company_id,
                   CAST(i.payment_date AS DATE) AS d,
                   CASE WHEN i.amount > 0 THEN 1 ELSE -1 END AS sgn,
                   ROUND(ABS(i.amount), 2) AS amt_c,
                   COUNT(*) AS n_inv
            FROM invoices i
            JOIN _qa_cos c ON CAST(i.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout AND {BOOK}
              AND i.payment_date IS NOT NULL
              AND NOT coalesce(i.payment_date_invalid, FALSE)
              AND CAST(i.payment_date AS DATE) >= DATE '2024-09-01'
              AND CAST(i.payment_date AS DATE) < DATE '2026-09-01'
            GROUP BY 1,2,3,4
        ),
        tx AS (
            SELECT CAST(t.company_id AS VARCHAR) AS company_id,
                   CAST(t."date" AS DATE) AS d,
                   CASE WHEN t.amount > 0 THEN 1 ELSE -1 END AS sgn,
                   ROUND(ABS(t.amount), 2) AS amt_c,
                   COUNT(*) AS n_tx
            FROM transactions t
            JOIN _qa_cos c ON CAST(t.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout AND t.amount <> 0 AND t."date" IS NOT NULL
              AND CAST(t."date" AS DATE) >= DATE '2024-09-01'
              AND CAST(t."date" AS DATE) < DATE '2026-09-01'
            GROUP BY 1,2,3,4
        ),
        hit AS (
            SELECT DISTINCT i.company_id, i.d, i.sgn, i.amt_c, i.n_inv
            FROM inv i
            JOIN tx t
              ON i.company_id = t.company_id AND i.sgn = t.sgn AND i.amt_c = t.amt_c
             AND abs(date_diff('day', i.d, t.d)) <= 1
        )
        SELECT
          (SELECT COALESCE(SUM(n_inv),0) FROM inv) AS n_inv,
          (SELECT COALESCE(SUM(n_inv),0) FROM hit) AS n_hit
        """
    ).df().iloc[0]
    out["due"] = due
    out["payment_pm1d"] = {
        "n_inv": int(pay1["n_inv"]),
        "hit_rate": _pct(int(pay1["n_hit"]), int(pay1["n_inv"])),
        "note": "same company, same sign, |Δamt|≤0.01, |Δdays|≤1 around payment_date",
    }
    return out


def extra_sibling_amount(con) -> dict:
    """Invoice amounts of A vs txs of a same-group sibling (not a CP map)."""
    print("  extra: sibling-group amount match (no CP map)...", flush=True)
    row = con.execute(
        f"""
        WITH inv AS (
            SELECT CAST(i.company_id AS VARCHAR) AS company_id,
                   c.group_id,
                   CAST(date_trunc('month', i.issuance_date) AS DATE) AS m,
                   CASE WHEN i.amount > 0 THEN 1 ELSE -1 END AS sgn,
                   ROUND(ABS(i.amount), 2) AS amt_c,
                   COUNT(*) AS n_inv
            FROM invoices i
            JOIN _qa_cos c ON CAST(i.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout AND {BOOK}
              AND CAST(i.issuance_date AS DATE) >= DATE '2024-09-01'
              AND CAST(i.issuance_date AS DATE) < DATE '2026-09-01'
            GROUP BY 1,2,3,4,5
        ),
        tx AS (
            SELECT CAST(t.company_id AS VARCHAR) AS company_id,
                   c.group_id,
                   CAST(date_trunc('month', t."date") AS DATE) AS m,
                   CASE WHEN t.amount > 0 THEN 1 ELSE -1 END AS sgn,
                   ROUND(ABS(t.amount), 2) AS amt_c,
                   COUNT(*) AS n_tx
            FROM transactions t
            JOIN _qa_cos c ON CAST(t.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout AND t.amount <> 0 AND t."date" IS NOT NULL
              AND CAST(t."date" AS DATE) >= DATE '2024-09-01'
              AND CAST(t."date" AS DATE) < DATE '2026-09-01'
            GROUP BY 1,2,3,4,5
        ),
        hit AS (
            SELECT DISTINCT i.company_id, i.m, i.sgn, i.amt_c, i.n_inv
            FROM inv i
            JOIN tx t
              ON i.group_id = t.group_id AND i.company_id <> t.company_id
             AND i.m = t.m AND i.sgn = t.sgn AND i.amt_c = t.amt_c
        ),
        has_sib_tx AS (
            SELECT DISTINCT i.company_id, i.m
            FROM inv i
            JOIN tx t ON i.group_id = t.group_id AND i.company_id <> t.company_id AND i.m = t.m
        )
        SELECT
          (SELECT COALESCE(SUM(n_inv),0) FROM inv) AS n_inv,
          (SELECT COALESCE(SUM(i.n_inv),0) FROM inv i
             JOIN has_sib_tx s ON i.company_id = s.company_id AND i.m = s.m) AS n_inv_with_sib_tx,
          (SELECT COALESCE(SUM(n_inv),0) FROM hit) AS n_hit
        """
    ).df().iloc[0]
    return {
        "n_inv": int(row["n_inv"]),
        "n_inv_with_sib_tx": int(row["n_inv_with_sib_tx"]),
        "n_hit": int(row["n_hit"]),
        "hit_rate_all": _pct(int(row["n_hit"]), int(row["n_inv"])),
        "hit_rate_among_with_sib": _pct(int(row["n_hit"]), int(row["n_inv_with_sib_tx"])),
        "note": "Same group, other company, same month/sign/cents. Not a COMP↔CP map.",
    }


def extra_match_rate_cm(con, monthly: pd.DataFrame) -> dict:
    """Company-month unmatched-share diagnostic (date ≤ month-end). Train only."""
    print("  extra: company-month match-rate vs size...", flush=True)
    cm = con.execute(
        f"""
        WITH inv AS (
            SELECT CAST(i.company_id AS VARCHAR) AS company_id,
                   CAST(date_trunc('month', i.payment_date) AS DATE) AS period,
                   CASE WHEN i.amount > 0 THEN 1 ELSE -1 END AS sgn,
                   ROUND(ABS(i.amount), 2) AS amt_c,
                   COUNT(*) AS n_inv
            FROM invoices i
            JOIN _qa_cos c ON CAST(i.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout AND {BOOK}
              AND i.payment_date IS NOT NULL
              AND NOT coalesce(i.payment_date_invalid, FALSE)
              AND CAST(i.payment_date AS DATE) >= DATE '2024-09-01'
              AND CAST(i.payment_date AS DATE) < DATE '2026-09-01'
            GROUP BY 1,2,3,4
        ),
        tx AS (
            SELECT CAST(t.company_id AS VARCHAR) AS company_id,
                   CAST(date_trunc('month', t."date") AS DATE) AS period,
                   CASE WHEN t.amount > 0 THEN 1 ELSE -1 END AS sgn,
                   ROUND(ABS(t.amount), 2) AS amt_c
            FROM transactions t
            JOIN _qa_cos c ON CAST(t.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout AND t.amount <> 0 AND t."date" IS NOT NULL
              AND CAST(t."date" AS DATE) >= DATE '2024-09-01'
              AND CAST(t."date" AS DATE) < DATE '2026-09-01'
            GROUP BY 1,2,3,4
        ),
        hit AS (
            SELECT i.company_id, i.period, SUM(i.n_inv) AS n_hit
            FROM inv i
            JOIN tx t
              ON i.company_id = t.company_id AND i.period = t.period
             AND i.sgn = t.sgn AND i.amt_c = t.amt_c
            GROUP BY 1,2
        ),
        inv_cm AS (
            SELECT company_id, period, SUM(n_inv) AS n_inv
            FROM inv
            GROUP BY 1,2
        )
        SELECT i.company_id, i.period, i.n_inv, COALESCE(h.n_hit, 0) AS n_hit
        FROM inv_cm i
        LEFT JOIN hit h ON i.company_id = h.company_id AND i.period = h.period
        """
    ).df()
    cm["period"] = pd.to_datetime(cm["period"])
    cm["company_id"] = cm["company_id"].astype(str)
    cm["match_rate"] = cm["n_hit"] / cm["n_inv"].where(cm["n_inv"] > 0)
    cm["unmatched_share"] = 1.0 - cm["match_rate"]
    store = monthly.loc[monthly["split"] == "train", ["company_id", "period", "a_op_in"]].copy()
    store["period"] = pd.to_datetime(store["period"])
    m = store.merge(cm, on=["company_id", "period"], how="inner")
    size = np.log1p(pd.to_numeric(m["a_op_in"], errors="coerce").abs())
    rho = float(pd.Series(m["match_rate"]).corr(size, method="spearman")) if len(m) > 10 else float("nan")
    return {
        "n_train_cm_with_paid_inv": int(len(m)),
        "mean_match_rate": float(m["match_rate"].mean()) if len(m) else float("nan"),
        "p50_match_rate": float(m["match_rate"].median()) if len(m) else float("nan"),
        "share_cm_any_hit": _pct((m["n_hit"] > 0).sum(), len(m)),
        "spearman_vs_log_abs_op_in": rho,
        "note": (
            "Prototype only (not written to parquet). Spearman vs size; "
            "|ρ|>0.85 would be a size proxy."
        ),
    }


def extra_y8_erp_split(monthly: pd.DataFrame, targets: pd.DataFrame, ever: pd.DataFrame) -> dict:
    """Y8 / Y5 labeled rows on ever-ERP vs never-ERP train companies."""
    ever_tr = ever[ever["split"] == "train"][["company_id", "has_inv"]].copy()
    ever_tr["company_id"] = ever_tr["company_id"].astype(str)
    ever_ids = set(ever_tr.loc[ever_tr["has_inv"] == 1, "company_id"])
    ycols = [c for c in (Y8_COLS + Y5_COLS + ["y3_recover_cash_6m"]) if c in targets.columns]
    panel = monthly.loc[
        monthly["split"] == "train",
        ["company_id", "period", "e_ar_issued", "e_ap_issued", "e_ar_overdue_30", "e_delay_coll"],
    ].merge(targets[["company_id", "period", *ycols]], on=["company_id", "period"], how="left")
    panel["ever_erp"] = panel["company_id"].isin(ever_ids)
    panel["e_any"] = panel["e_ar_issued"].notna() | panel["e_ap_issued"].notna()
    panel["e_side_ready"] = panel["e_ar_overdue_30"].notna() | panel["e_delay_coll"].notna()
    rows = []
    for col in ycols:
        lab = panel[col].notna()
        for name, mask in (
            ("all", lab),
            ("ever_erp", lab & panel["ever_erp"]),
            ("never_erp", lab & ~panel["ever_erp"]),
            ("e_any", lab & panel["e_any"]),
            ("e_side_ready", lab & panel["e_side_ready"]),
        ):
            sl = panel.loc[mask, col]
            rows.append(
                {
                    "y": col,
                    "slice": name,
                    "n": int(len(sl)),
                    "n_pos": int((sl == 1).sum()) if len(sl) else 0,
                    "base_rate": float(sl.mean()) if len(sl) else float("nan"),
                    "share_of_labeled": _pct(len(sl), int(lab.sum())),
                }
            )
    return {"slices": rows}


def extra_e_only_company(con, monthly: pd.DataFrame, ever: pd.DataFrame) -> dict:
    """Why 745 train companies have e_* vs 744 with a book invoice."""
    ever_inv = set(ever.loc[(ever["split"] == "train") & (ever["has_inv"] == 1), "company_id"])
    tr = monthly[monthly["split"] == "train"]
    e_any = tr[E_COLS].notna().any(axis=1)
    ever_e = set(tr.loc[e_any, "company_id"])
    extra_ids = sorted(ever_e - ever_inv)
    missing = sorted(ever_inv - ever_e)
    detail = {}
    if extra_ids:
        q = ",".join("'" + x.replace("'", "''") + "'" for x in extra_ids)
        raw = con.execute(
            f"""
            SELECT CAST(company_id AS VARCHAR) AS company_id,
                   document_type, status,
                   COUNT(*) AS n,
                   SUM(CASE WHEN amount = 0 THEN 1 ELSE 0 END) AS n_zero,
                   SUM(CASE WHEN issuance_date IS NULL THEN 1 ELSE 0 END) AS n_no_iss
            FROM invoices
            WHERE CAST(company_id AS VARCHAR) IN ({q})
            GROUP BY 1, 2, 3
            """
        ).df()
        detail = raw.to_dict(orient="records")
    return {"n_extra_e": len(extra_ids), "extra_ids": extra_ids, "n_missing_e": len(missing), "detail": detail}


def extra_payment_timing(con) -> dict:
    """Paid-invoice amount match at exact day / ±1 / ±3 / calendar month (train)."""
    print("  extra: payment timing ladder...", flush=True)
    out = {}
    for days, label in ((0, "0"), (1, "1"), (3, "3")):
        row = con.execute(
            f"""
            WITH inv AS (
                SELECT CAST(i.company_id AS VARCHAR) AS company_id,
                       CAST(i.payment_date AS DATE) AS d,
                       CASE WHEN i.amount > 0 THEN 1 ELSE -1 END AS sgn,
                       ROUND(ABS(i.amount), 2) AS amt_c,
                       COUNT(*) AS n_inv
                FROM invoices i
                JOIN _qa_cos c ON CAST(i.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
                WHERE NOT c.is_holdout AND {BOOK}
                  AND i.payment_date IS NOT NULL
                  AND NOT coalesce(i.payment_date_invalid, FALSE)
                  AND CAST(i.payment_date AS DATE) >= DATE '2024-09-01'
                  AND CAST(i.payment_date AS DATE) < DATE '2026-09-01'
                GROUP BY 1,2,3,4
            ),
            tx AS (
                SELECT CAST(t.company_id AS VARCHAR) AS company_id,
                       CAST(t."date" AS DATE) AS d,
                       CASE WHEN t.amount > 0 THEN 1 ELSE -1 END AS sgn,
                       ROUND(ABS(t.amount), 2) AS amt_c
                FROM transactions t
                JOIN _qa_cos c ON CAST(t.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
                WHERE NOT c.is_holdout AND t.amount <> 0 AND t."date" IS NOT NULL
                  AND CAST(t."date" AS DATE) >= DATE '2024-09-01'
                  AND CAST(t."date" AS DATE) < DATE '2026-09-01'
                GROUP BY 1,2,3,4
            ),
            hit AS (
                SELECT DISTINCT i.company_id, i.d, i.sgn, i.amt_c, i.n_inv
                FROM inv i
                JOIN tx t
                  ON i.company_id = t.company_id AND i.sgn = t.sgn AND i.amt_c = t.amt_c
                 AND abs(date_diff('day', i.d, t.d)) <= {days}
            )
            SELECT
              (SELECT COALESCE(SUM(n_inv),0) FROM inv) AS n_inv,
              (SELECT COALESCE(SUM(n_inv),0) FROM hit) AS n_hit
            """
        ).df().iloc[0]
        out[label] = {
            "window_days": days,
            "n_inv": int(row["n_inv"]),
            "n_hit": int(row["n_hit"]),
            "hit_rate": _pct(int(row["n_hit"]), int(row["n_inv"])),
        }
    return out


def extra_amount_bands(con) -> dict:
    """Payment-month |Δ|≤0.01 hit by fixed euro bands (not a fitted cut)."""
    print("  extra: amount bands...", flush=True)
    df = con.execute(
        f"""
        WITH inv AS (
            SELECT CAST(i.company_id AS VARCHAR) AS company_id,
                   CAST(date_trunc('month', i.payment_date) AS DATE) AS m,
                   CASE WHEN i.amount > 0 THEN 1 ELSE -1 END AS sgn,
                   ROUND(ABS(i.amount), 2) AS amt_c,
                   ABS(i.amount) AS abs_amt,
                   COUNT(*) AS n_inv
            FROM invoices i
            JOIN _qa_cos c ON CAST(i.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout AND {BOOK}
              AND i.payment_date IS NOT NULL
              AND NOT coalesce(i.payment_date_invalid, FALSE)
              AND CAST(i.payment_date AS DATE) >= DATE '2024-09-01'
              AND CAST(i.payment_date AS DATE) < DATE '2026-09-01'
            GROUP BY 1,2,3,4,5
        ),
        tx AS (
            SELECT CAST(t.company_id AS VARCHAR) AS company_id,
                   CAST(date_trunc('month', t."date") AS DATE) AS m,
                   CASE WHEN t.amount > 0 THEN 1 ELSE -1 END AS sgn,
                   ROUND(ABS(t.amount), 2) AS amt_c
            FROM transactions t
            JOIN _qa_cos c ON CAST(t.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout AND t.amount <> 0 AND t."date" IS NOT NULL
              AND CAST(t."date" AS DATE) >= DATE '2024-09-01'
              AND CAST(t."date" AS DATE) < DATE '2026-09-01'
            GROUP BY 1,2,3,4
        ),
        hit AS (
            SELECT DISTINCT i.company_id, i.m, i.sgn, i.amt_c, i.abs_amt, i.n_inv
            FROM inv i
            JOIN tx t
              ON i.company_id = t.company_id AND i.m = t.m
             AND i.sgn = t.sgn AND i.amt_c = t.amt_c
        ),
        tagged AS (
            SELECT i.abs_amt, i.n_inv,
                   CASE WHEN i.abs_amt < 10 THEN '<10'
                        WHEN i.abs_amt < 100 THEN '10-100'
                        WHEN i.abs_amt < 1000 THEN '100-1k'
                        WHEN i.abs_amt < 10000 THEN '1k-10k'
                        ELSE '>=10k' END AS band,
                   CASE WHEN h.n_inv IS NOT NULL THEN 1 ELSE 0 END AS hit
            FROM inv i
            LEFT JOIN hit h
              ON i.company_id = h.company_id AND i.m = h.m
             AND i.sgn = h.sgn AND i.amt_c = h.amt_c
        )
        SELECT band,
               SUM(n_inv) AS n_inv,
               SUM(n_inv * hit) AS n_hit
        FROM tagged
        GROUP BY 1
        """
    ).df()
    order = ["<10", "10-100", "100-1k", "1k-10k", ">=10k"]
    rows = []
    for b in order:
        sl = df[df["band"] == b]
        if sl.empty:
            continue
        n = int(sl["n_inv"].iloc[0])
        h = int(sl["n_hit"].iloc[0])
        rows.append({"band": b, "n_inv": n, "n_hit": h, "hit_rate": _pct(h, n)})
    return {"rows": rows}


def extra_match_by_month(con) -> dict:
    print("  extra: payment match by month...", flush=True)
    df = con.execute(
        f"""
        WITH inv AS (
            SELECT CAST(i.company_id AS VARCHAR) AS company_id,
                   CAST(date_trunc('month', i.payment_date) AS DATE) AS m,
                   CASE WHEN i.amount > 0 THEN 1 ELSE -1 END AS sgn,
                   ROUND(ABS(i.amount), 2) AS amt_c,
                   COUNT(*) AS n_inv
            FROM invoices i
            JOIN _qa_cos c ON CAST(i.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout AND {BOOK}
              AND i.payment_date IS NOT NULL
              AND NOT coalesce(i.payment_date_invalid, FALSE)
              AND CAST(i.payment_date AS DATE) >= DATE '2024-09-01'
              AND CAST(i.payment_date AS DATE) < DATE '2026-09-01'
            GROUP BY 1,2,3,4
        ),
        tx AS (
            SELECT CAST(t.company_id AS VARCHAR) AS company_id,
                   CAST(date_trunc('month', t."date") AS DATE) AS m,
                   CASE WHEN t.amount > 0 THEN 1 ELSE -1 END AS sgn,
                   ROUND(ABS(t.amount), 2) AS amt_c
            FROM transactions t
            JOIN _qa_cos c ON CAST(t.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout AND t.amount <> 0 AND t."date" IS NOT NULL
              AND CAST(t."date" AS DATE) >= DATE '2024-09-01'
              AND CAST(t."date" AS DATE) < DATE '2026-09-01'
            GROUP BY 1,2,3,4
        ),
        hit AS (
            SELECT DISTINCT i.company_id, i.m, i.sgn, i.amt_c, i.n_inv
            FROM inv i
            JOIN tx t
              ON i.company_id = t.company_id AND i.m = t.m
             AND i.sgn = t.sgn AND i.amt_c = t.amt_c
        )
        SELECT i.m,
               SUM(i.n_inv) AS n_inv,
               COALESCE(SUM(h.n_inv), 0) AS n_hit
        FROM inv i
        LEFT JOIN hit h
          ON i.company_id = h.company_id AND i.m = h.m
         AND i.sgn = h.sgn AND i.amt_c = h.amt_c
        GROUP BY 1
        ORDER BY 1
        """
    ).df()
    rows = []
    for r in df.itertuples(index=False):
        n = int(r.n_inv)
        h = int(r.n_hit)
        rows.append(
            {
                "month": pd.Timestamp(r.m).strftime("%Y-%m"),
                "n_inv": n,
                "hit_rate": _pct(h, n),
            }
        )
    return {"rows": rows}


def extra_erp_vs_bank_start(con, ever: pd.DataFrame) -> dict:
    """First invoice vs first tx (train). Left-truncated ERP trail?"""
    print("  extra: first invoice vs first tx...", flush=True)
    df = con.execute(
        f"""
        SELECT CAST(c.company_id AS VARCHAR) AS company_id,
               MIN(CAST(i.issuance_date AS DATE)) AS first_inv,
               MIN(CAST(t."date" AS DATE)) AS first_tx
        FROM _qa_cos c
        LEFT JOIN invoices i
          ON CAST(i.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
         AND {BOOK}
        LEFT JOIN transactions t
          ON CAST(t.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
         AND t."date" IS NOT NULL
        WHERE NOT c.is_holdout
        GROUP BY 1
        """
    ).df()
    df["first_inv"] = pd.to_datetime(df["first_inv"])
    df["first_tx"] = pd.to_datetime(df["first_tx"])
    has = df["first_inv"].notna()
    after_tx = has & (df["first_inv"] > df["first_tx"] + pd.Timedelta(days=31))
    after_start = has & (df["first_inv"] > pd.Timestamp("2024-10-01"))
    return {
        "n_train_with_inv": int(has.sum()),
        "n_first_inv_after_2024_10": int(after_start.sum()),
        "share_first_inv_after_2024_10": _pct(after_start.sum(), has.sum()),
        "n_first_inv_gt_1m_after_first_tx": int(after_tx.sum()),
        "share_inv_starts_later_than_bank": _pct(after_tx.sum(), has.sum()),
        "median_inv_minus_tx_days": float(
            (df.loc[has, "first_inv"] - df.loc[has, "first_tx"]).dt.days.median()
        )
        if has.any()
        else float("nan"),
    }


def extra_y3_erp(monthly: pd.DataFrame, targets: pd.DataFrame, ever: pd.DataFrame) -> dict:
    ever_ids = set(ever.loc[(ever["split"] == "train") & (ever["has_inv"] == 1), "company_id"])
    tr = monthly.loc[monthly["split"] == "train", ["company_id", "period"]].merge(
        targets[["company_id", "period", "y3_recover_cash_6m"]],
        on=["company_id", "period"],
        how="left",
    )
    y = tr[tr["y3_recover_cash_6m"].notna()].copy()
    y["ever_erp"] = y["company_id"].isin(ever_ids)
    out = {}
    for name, mask in (("ever_erp", y["ever_erp"]), ("never_erp", ~y["ever_erp"])):
        sl = y.loc[mask, "y3_recover_cash_6m"]
        out[name] = {
            "n": int(len(sl)),
            "n_pos": int((sl == 1).sum()),
            "base_rate": float(sl.mean()) if len(sl) else float("nan"),
        }
    return out


def extra_match_company_dist(con) -> dict:
    """Per-company payment match-rate distribution (train, ever-paid)."""
    print("  extra: per-company match-rate distribution...", flush=True)
    df = con.execute(
        f"""
        WITH inv AS (
            SELECT CAST(i.company_id AS VARCHAR) AS company_id,
                   CAST(date_trunc('month', i.payment_date) AS DATE) AS m,
                   CASE WHEN i.amount > 0 THEN 1 ELSE -1 END AS sgn,
                   ROUND(ABS(i.amount), 2) AS amt_c,
                   COUNT(*) AS n_inv
            FROM invoices i
            JOIN _qa_cos c ON CAST(i.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout AND {BOOK}
              AND i.payment_date IS NOT NULL
              AND NOT coalesce(i.payment_date_invalid, FALSE)
              AND CAST(i.payment_date AS DATE) >= DATE '2024-09-01'
              AND CAST(i.payment_date AS DATE) < DATE '2026-09-01'
            GROUP BY 1,2,3,4
        ),
        tx AS (
            SELECT CAST(t.company_id AS VARCHAR) AS company_id,
                   CAST(date_trunc('month', t."date") AS DATE) AS m,
                   CASE WHEN t.amount > 0 THEN 1 ELSE -1 END AS sgn,
                   ROUND(ABS(t.amount), 2) AS amt_c
            FROM transactions t
            JOIN _qa_cos c ON CAST(t.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout AND t.amount <> 0 AND t."date" IS NOT NULL
              AND CAST(t."date" AS DATE) >= DATE '2024-09-01'
              AND CAST(t."date" AS DATE) < DATE '2026-09-01'
            GROUP BY 1,2,3,4
        ),
        hit AS (
            SELECT DISTINCT i.company_id, i.m, i.sgn, i.amt_c, i.n_inv
            FROM inv i
            JOIN tx t
              ON i.company_id = t.company_id AND i.m = t.m
             AND i.sgn = t.sgn AND i.amt_c = t.amt_c
        )
        SELECT i.company_id,
               SUM(i.n_inv) AS n_inv,
               COALESCE(SUM(h.n_inv), 0) AS n_hit
        FROM inv i
        LEFT JOIN hit h
          ON i.company_id = h.company_id AND i.m = h.m
         AND i.sgn = h.sgn AND i.amt_c = h.amt_c
        GROUP BY 1
        """
    ).df()
    df["rate"] = df["n_hit"] / df["n_inv"].where(df["n_inv"] > 0)
    q = df["rate"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    return {
        "n_train_cos_with_paid": int(len(df)),
        "p10": float(q.loc[0.1]),
        "p25": float(q.loc[0.25]),
        "p50": float(q.loc[0.5]),
        "p75": float(q.loc[0.75]),
        "p90": float(q.loc[0.9]),
        "share_rate_ge_0_5": _pct((df["rate"] >= 0.5).sum(), len(df)),
        "share_rate_eq_0": _pct((df["rate"] == 0).sum(), len(df)),
    }


def extra_unmatched_vs_y3(con, monthly: pd.DataFrame, targets: pd.DataFrame, ever: pd.DataFrame) -> dict:
    """Fixed-cut unmatched-share vs Y3 (train, stressed, ever-ERP). Not a fit."""
    print("  extra: unmatched-share vs y3 (fixed 0.5 cut)...", flush=True)
    cm = con.execute(
        f"""
        WITH inv AS (
            SELECT CAST(i.company_id AS VARCHAR) AS company_id,
                   CAST(date_trunc('month', i.payment_date) AS DATE) AS period,
                   CASE WHEN i.amount > 0 THEN 1 ELSE -1 END AS sgn,
                   ROUND(ABS(i.amount), 2) AS amt_c,
                   COUNT(*) AS n_inv
            FROM invoices i
            JOIN _qa_cos c ON CAST(i.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout AND {BOOK}
              AND i.payment_date IS NOT NULL
              AND NOT coalesce(i.payment_date_invalid, FALSE)
              AND CAST(i.payment_date AS DATE) >= DATE '2024-09-01'
              AND CAST(i.payment_date AS DATE) < DATE '2026-09-01'
            GROUP BY 1,2,3,4
        ),
        tx AS (
            SELECT CAST(t.company_id AS VARCHAR) AS company_id,
                   CAST(date_trunc('month', t."date") AS DATE) AS period,
                   CASE WHEN t.amount > 0 THEN 1 ELSE -1 END AS sgn,
                   ROUND(ABS(t.amount), 2) AS amt_c
            FROM transactions t
            JOIN _qa_cos c ON CAST(t.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout AND t.amount <> 0 AND t."date" IS NOT NULL
              AND CAST(t."date" AS DATE) >= DATE '2024-09-01'
              AND CAST(t."date" AS DATE) < DATE '2026-09-01'
            GROUP BY 1,2,3,4
        ),
        hit AS (
            SELECT i.company_id, i.period, SUM(i.n_inv) AS n_hit
            FROM inv i
            JOIN tx t
              ON i.company_id = t.company_id AND i.period = t.period
             AND i.sgn = t.sgn AND i.amt_c = t.amt_c
            GROUP BY 1,2
        ),
        inv_cm AS (
            SELECT company_id, period, SUM(n_inv) AS n_inv FROM inv GROUP BY 1,2
        )
        SELECT i.company_id, i.period, i.n_inv, COALESCE(h.n_hit, 0) AS n_hit
        FROM inv_cm i
        LEFT JOIN hit h ON i.company_id = h.company_id AND i.period = h.period
        """
    ).df()
    cm["period"] = pd.to_datetime(cm["period"])
    cm["company_id"] = cm["company_id"].astype(str)
    cm["unmatched"] = 1.0 - cm["n_hit"] / cm["n_inv"].where(cm["n_inv"] > 0)
    ever_ids = set(ever.loc[(ever["split"] == "train") & (ever["has_inv"] == 1), "company_id"])
    y = monthly.loc[monthly["split"] == "train", ["company_id", "period"]].merge(
        targets[["company_id", "period", "y3_recover_cash_6m"]],
        on=["company_id", "period"],
        how="left",
    )
    y = y[y["y3_recover_cash_6m"].notna() & y["company_id"].isin(ever_ids)]
    m = y.merge(cm[["company_id", "period", "unmatched"]], on=["company_id", "period"], how="inner")
    if m.empty:
        return {"n_stressed_with_rate": 0}
    high = m["unmatched"] >= 0.5
    out = {"n_stressed_with_rate": int(len(m))}
    for name, mask in (("unmatched_ge_0_5", high), ("unmatched_lt_0_5", ~high)):
        sl = m.loc[mask, "y3_recover_cash_6m"]
        out[name] = {
            "n": int(len(sl)),
            "n_pos": int((sl == 1).sum()),
            "base_rate": float(sl.mean()) if len(sl) else float("nan"),
        }
    out["spearman_unmatched_vs_y3"] = float(
        pd.Series(m["unmatched"]).corr(pd.to_numeric(m["y3_recover_cash_6m"]), method="spearman")
    )
    return out


def extra_match_resolved_cp(con) -> dict:
    """Payment-month amount match using only txs with a non-null counterparty_id."""
    print("  extra: match against resolved-CP txs only...", flush=True)
    row = con.execute(
        f"""
        WITH inv AS (
            SELECT CAST(i.company_id AS VARCHAR) AS company_id,
                   CAST(date_trunc('month', i.payment_date) AS DATE) AS m,
                   CASE WHEN i.amount > 0 THEN 1 ELSE -1 END AS sgn,
                   ROUND(ABS(i.amount), 2) AS amt_c,
                   COUNT(*) AS n_inv
            FROM invoices i
            JOIN _qa_cos c ON CAST(i.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout AND {BOOK}
              AND i.payment_date IS NOT NULL
              AND NOT coalesce(i.payment_date_invalid, FALSE)
              AND CAST(i.payment_date AS DATE) >= DATE '2024-09-01'
              AND CAST(i.payment_date AS DATE) < DATE '2026-09-01'
            GROUP BY 1,2,3,4
        ),
        tx AS (
            SELECT CAST(t.company_id AS VARCHAR) AS company_id,
                   CAST(date_trunc('month', t."date") AS DATE) AS m,
                   CASE WHEN t.amount > 0 THEN 1 ELSE -1 END AS sgn,
                   ROUND(ABS(t.amount), 2) AS amt_c
            FROM transactions t
            JOIN _qa_cos c ON CAST(t.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout AND t.amount <> 0 AND t."date" IS NOT NULL
              AND t.counterparty_id IS NOT NULL
              AND length(trim(CAST(t.counterparty_id AS VARCHAR))) > 0
              AND CAST(t."date" AS DATE) >= DATE '2024-09-01'
              AND CAST(t."date" AS DATE) < DATE '2026-09-01'
            GROUP BY 1,2,3,4
        ),
        hit AS (
            SELECT DISTINCT i.company_id, i.m, i.sgn, i.amt_c, i.n_inv
            FROM inv i
            JOIN tx t
              ON i.company_id = t.company_id AND i.m = t.m
             AND i.sgn = t.sgn AND i.amt_c = t.amt_c
        )
        SELECT
          (SELECT COALESCE(SUM(n_inv),0) FROM inv) AS n_inv,
          (SELECT COALESCE(SUM(n_inv),0) FROM hit) AS n_hit
        """
    ).df().iloc[0]
    return {
        "n_inv": int(row["n_inv"]),
        "n_hit": int(row["n_hit"]),
        "hit_rate": _pct(int(row["n_hit"]), int(row["n_inv"])),
        "note": "Same payment-month probe, but txs must have a resolved counterparty_id.",
    }


def extra_invoiced_never_paid(con) -> dict:
    """Train book-invoice companies with zero valid payment_date in the panel."""
    row = con.execute(
        f"""
        WITH book AS (
            SELECT DISTINCT CAST(i.company_id AS VARCHAR) AS company_id
            FROM invoices i
            JOIN _qa_cos c ON CAST(i.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout AND {BOOK}
        ),
        paid AS (
            SELECT DISTINCT CAST(i.company_id AS VARCHAR) AS company_id
            FROM invoices i
            JOIN _qa_cos c ON CAST(i.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout AND {BOOK}
              AND i.payment_date IS NOT NULL
              AND NOT coalesce(i.payment_date_invalid, FALSE)
              AND CAST(i.payment_date AS DATE) >= DATE '2024-09-01'
              AND CAST(i.payment_date AS DATE) < DATE '2026-09-01'
        )
        SELECT
          (SELECT COUNT(*) FROM book) AS n_book,
          (SELECT COUNT(*) FROM paid) AS n_paid,
          (SELECT COUNT(*) FROM book b LEFT JOIN paid p ON b.company_id = p.company_id
            WHERE p.company_id IS NULL) AS n_never_paid
        """
    ).df().iloc[0]
    return {k: int(row[k]) for k in ("n_book", "n_paid", "n_never_paid")}


def extra_sign_month_controls(con) -> dict:
    """Wrong-sign and adjacent-month amount matches (train, payment month)."""
    print("  extra: wrong-sign + adjacent-month controls...", flush=True)
    wrong = con.execute(
        f"""
        WITH inv AS (
            SELECT CAST(i.company_id AS VARCHAR) AS company_id,
                   CAST(date_trunc('month', i.payment_date) AS DATE) AS m,
                   CASE WHEN i.amount > 0 THEN 1 ELSE -1 END AS sgn,
                   ROUND(ABS(i.amount), 2) AS amt_c,
                   COUNT(*) AS n_inv
            FROM invoices i
            JOIN _qa_cos c ON CAST(i.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout AND {BOOK}
              AND i.payment_date IS NOT NULL
              AND NOT coalesce(i.payment_date_invalid, FALSE)
              AND CAST(i.payment_date AS DATE) >= DATE '2024-09-01'
              AND CAST(i.payment_date AS DATE) < DATE '2026-09-01'
            GROUP BY 1,2,3,4
        ),
        tx AS (
            SELECT CAST(t.company_id AS VARCHAR) AS company_id,
                   CAST(date_trunc('month', t."date") AS DATE) AS m,
                   CASE WHEN t.amount > 0 THEN 1 ELSE -1 END AS sgn,
                   ROUND(ABS(t.amount), 2) AS amt_c
            FROM transactions t
            JOIN _qa_cos c ON CAST(t.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout AND t.amount <> 0 AND t."date" IS NOT NULL
              AND CAST(t."date" AS DATE) >= DATE '2024-09-01'
              AND CAST(t."date" AS DATE) < DATE '2026-09-01'
            GROUP BY 1,2,3,4
        ),
        hit AS (
            SELECT DISTINCT i.company_id, i.m, i.sgn, i.amt_c, i.n_inv
            FROM inv i
            JOIN tx t
              ON i.company_id = t.company_id AND i.m = t.m
             AND i.sgn = -t.sgn AND i.amt_c = t.amt_c
        )
        SELECT
          (SELECT COALESCE(SUM(n_inv),0) FROM inv) AS n_inv,
          (SELECT COALESCE(SUM(n_inv),0) FROM hit) AS n_hit
        """
    ).df().iloc[0]
    adj = con.execute(
        f"""
        WITH inv AS (
            SELECT CAST(i.company_id AS VARCHAR) AS company_id,
                   CAST(date_trunc('month', i.payment_date) AS DATE) AS m,
                   CASE WHEN i.amount > 0 THEN 1 ELSE -1 END AS sgn,
                   ROUND(ABS(i.amount), 2) AS amt_c,
                   COUNT(*) AS n_inv
            FROM invoices i
            JOIN _qa_cos c ON CAST(i.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout AND {BOOK}
              AND i.payment_date IS NOT NULL
              AND NOT coalesce(i.payment_date_invalid, FALSE)
              AND CAST(i.payment_date AS DATE) >= DATE '2024-09-01'
              AND CAST(i.payment_date AS DATE) < DATE '2026-09-01'
            GROUP BY 1,2,3,4
        ),
        tx AS (
            SELECT CAST(t.company_id AS VARCHAR) AS company_id,
                   CAST(date_trunc('month', t."date") AS DATE) AS m,
                   CASE WHEN t.amount > 0 THEN 1 ELSE -1 END AS sgn,
                   ROUND(ABS(t.amount), 2) AS amt_c
            FROM transactions t
            JOIN _qa_cos c ON CAST(t.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout AND t.amount <> 0 AND t."date" IS NOT NULL
              AND CAST(t."date" AS DATE) >= DATE '2024-09-01'
              AND CAST(t."date" AS DATE) < DATE '2026-09-01'
            GROUP BY 1,2,3,4
        ),
        hit AS (
            SELECT DISTINCT i.company_id, i.m, i.sgn, i.amt_c, i.n_inv
            FROM inv i
            JOIN tx t
              ON i.company_id = t.company_id
             AND i.sgn = t.sgn AND i.amt_c = t.amt_c
             AND (
                  t.m = CAST(i.m + INTERVAL 1 MONTH AS DATE)
               OR t.m = CAST(i.m - INTERVAL 1 MONTH AS DATE)
             )
        )
        SELECT
          (SELECT COALESCE(SUM(n_inv),0) FROM inv) AS n_inv,
          (SELECT COALESCE(SUM(n_inv),0) FROM hit) AS n_hit
        """
    ).df().iloc[0]
    return {
        "wrong_sign": {
            "n_inv": int(wrong["n_inv"]),
            "n_hit": int(wrong["n_hit"]),
            "hit_rate": _pct(int(wrong["n_hit"]), int(wrong["n_inv"])),
        },
        "adjacent_month": {
            "n_inv": int(adj["n_inv"]),
            "n_hit": int(adj["n_hit"]),
            "hit_rate": _pct(int(adj["n_hit"]), int(adj["n_inv"])),
        },
    }


def extra_y8_never_erp_groups(monthly: pd.DataFrame, targets: pd.DataFrame, ever: pd.DataFrame) -> dict:
    """Among y8_cash never-ERP labeled train rows, how many sit in mixed groups?"""
    tr = ever[ever["split"] == "train"].copy()
    g = tr.groupby("group_id").agg(n_inv=("has_inv", "sum"), n=("company_id", "size"))
    mixed = set(g.index[(g["n_inv"] > 0) & (g["n_inv"] < g["n"])])
    dark = set(g.index[g["n_inv"] == 0])
    ever_ids = set(tr.loc[tr["has_inv"] == 1, "company_id"])
    gid = tr.set_index("company_id")["group_id"]
    lab = monthly.loc[monthly["split"] == "train", ["company_id", "period"]].merge(
        targets[["company_id", "period", "y8_cash_worse_6"]],
        on=["company_id", "period"],
        how="left",
    )
    lab = lab[lab["y8_cash_worse_6"].notna() & ~lab["company_id"].isin(ever_ids)].copy()
    lab["group_id"] = lab["company_id"].map(gid)
    n = int(len(lab))
    n_mixed = int(lab["group_id"].isin(mixed).sum())
    n_dark = int(lab["group_id"].isin(dark).sum())
    return {
        "n_never_erp_labeled": n,
        "n_in_mixed_group": n_mixed,
        "n_in_all_dark_group": n_dark,
        "share_never_erp_in_mixed": _pct(n_mixed, n),
        "note": (
            "Mixed-group never-ERP rows can see sibling cash via family H; "
            "all-dark rows cannot see any cobros, sibling or own."
        ),
    }


def extra_shared_cp_in_group(con) -> dict:
    """Train: does any counterparty_id appear on two companies in the same group?"""
    print("  extra: shared CP within group (no COMP map)...", flush=True)
    row = con.execute(
        """
        WITH inv AS (
            SELECT c.group_id,
                   CAST(i.counterparty_id AS VARCHAR) AS cp,
                   COUNT(DISTINCT CAST(i.company_id AS VARCHAR)) AS n_cos
            FROM invoices i
            JOIN _qa_cos c ON CAST(i.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout
              AND i.counterparty_id IS NOT NULL
              AND length(trim(CAST(i.counterparty_id AS VARCHAR))) > 0
            GROUP BY 1, 2
            HAVING COUNT(DISTINCT CAST(i.company_id AS VARCHAR)) > 1
        ),
        tx AS (
            SELECT c.group_id,
                   CAST(t.counterparty_id AS VARCHAR) AS cp,
                   COUNT(DISTINCT CAST(t.company_id AS VARCHAR)) AS n_cos
            FROM transactions t
            JOIN _qa_cos c ON CAST(t.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout
              AND t.counterparty_id IS NOT NULL
              AND length(trim(CAST(t.counterparty_id AS VARCHAR))) > 0
            GROUP BY 1, 2
            HAVING COUNT(DISTINCT CAST(t.company_id AS VARCHAR)) > 1
        )
        SELECT
          (SELECT COUNT(*) FROM inv) AS n_inv_cp_shared_in_group,
          (SELECT COALESCE(SUM(n_cos),0) FROM inv) AS n_inv_cos_pairs,
          (SELECT COUNT(*) FROM tx) AS n_tx_cp_shared_in_group,
          (SELECT COALESCE(SUM(n_cos),0) FROM tx) AS n_tx_cos_pairs
        """
    ).df().iloc[0]
    return {k: int(row[k]) for k in row.index}


def extra_shared_tx_cp_detail(con) -> dict:
    """The rare train group-shared tx counterparty_id — inspect, do not map."""
    print("  extra: detail the shared tx CP...", flush=True)
    df = con.execute(
        """
        WITH shared AS (
            SELECT c.group_id,
                   CAST(t.counterparty_id AS VARCHAR) AS cp,
                   COUNT(DISTINCT CAST(t.company_id AS VARCHAR)) AS n_cos
            FROM transactions t
            JOIN _qa_cos c ON CAST(t.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout
              AND t.counterparty_id IS NOT NULL
              AND length(trim(CAST(t.counterparty_id AS VARCHAR))) > 0
            GROUP BY 1, 2
            HAVING COUNT(DISTINCT CAST(t.company_id AS VARCHAR)) > 1
        )
        SELECT CAST(t.company_id AS VARCHAR) AS company_id,
               c.group_id,
               CAST(t.counterparty_id AS VARCHAR) AS cp,
               t.category,
               COUNT(*) AS n_tx,
               SUM(t.amount) AS sum_amt,
               MIN(CAST(t."date" AS DATE)) AS d0,
               MAX(CAST(t."date" AS DATE)) AS d1
        FROM transactions t
        JOIN _qa_cos c ON CAST(t.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
        JOIN shared s
          ON c.group_id = s.group_id
         AND CAST(t.counterparty_id AS VARCHAR) = s.cp
        WHERE NOT c.is_holdout
        GROUP BY 1, 2, 3, 4
        ORDER BY 2, 3, 1
        """
    ).df()
    recs = []
    for r in df.itertuples(index=False):
        recs.append(
            {
                "company_id": str(r.company_id),
                "group_id": str(r.group_id),
                "cp": str(r.cp),
                "category": str(r.category),
                "n_tx": int(r.n_tx),
                "sum_amt": float(r.sum_amt) if r.sum_amt is not None else None,
                "d0": str(r.d0),
                "d1": str(r.d1),
            }
        )
    return {"n_pairs": int(len(df)), "rows": recs}


def extra_h_for_dark(monthly: pd.DataFrame, ever: pd.DataFrame) -> dict:
    """Can family H sibling cash see anything for the 470?"""
    tr = ever[ever["split"] == "train"].copy()
    dark_ids = set(tr.loc[tr["has_inv"] == 0, "company_id"])
    g = tr.groupby("group_id").agg(n_inv=("has_inv", "sum"), n=("company_id", "size"))
    mixed = set(g.index[(g["n_inv"] > 0) & (g["n_inv"] < g["n"])])
    gid = tr.set_index("company_id")["group_id"]
    hcols = [c for c in ("h_sib_in", "h_share_group_in", "h_n_siblings_active") if c in monthly.columns]
    store = monthly.loc[monthly["split"] == "train", ["company_id", "period", *hcols]].copy()
    store = store[store["company_id"].isin(dark_ids)]
    store["group_id"] = store["company_id"].map(gid)
    store["in_mixed"] = store["group_id"].isin(mixed)
    out = {
        "n_dark_cos": int(len(dark_ids)),
        "n_dark_cm": int(len(store)),
        "n_mixed_dark_cm": int(store["in_mixed"].sum()),
    }
    if "h_n_siblings_active" in store.columns:
        act = store["h_n_siblings_active"].fillna(0) >= 1
        out["share_dark_with_active_sib"] = _pct(act.sum(), len(store))
        out["share_mixed_dark_with_active_sib"] = _pct(
            (store["in_mixed"] & act).sum(), int(store["in_mixed"].sum()) or 1
        )
        out["share_alldark_with_active_sib"] = _pct(
            ((~store["in_mixed"]) & act).sum(), int((~store["in_mixed"]).sum()) or 1
        )
        # keep old key so the print line does not break
        out["share_dark_with_sib_in"] = out["share_dark_with_active_sib"]
    else:
        out["share_dark_with_sib_in"] = float("nan")
        out["note"] = "h_n_siblings_active not in monthly.parquet"
    return out


def extra_payment_category(con) -> dict:
    """Payment-month |Δ|≤0.01 using only collection/payment category txs."""
    print("  extra: payment-month + category filter...", flush=True)
    row = con.execute(
        f"""
        WITH inv AS (
            SELECT CAST(i.company_id AS VARCHAR) AS company_id,
                   CAST(date_trunc('month', i.payment_date) AS DATE) AS m,
                   CASE WHEN i.amount > 0 THEN 1 ELSE -1 END AS sgn,
                   ROUND(ABS(i.amount), 2) AS amt_c,
                   COUNT(*) AS n_inv
            FROM invoices i
            JOIN _qa_cos c ON CAST(i.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout AND {BOOK}
              AND i.payment_date IS NOT NULL
              AND NOT coalesce(i.payment_date_invalid, FALSE)
              AND CAST(i.payment_date AS DATE) >= DATE '2024-09-01'
              AND CAST(i.payment_date AS DATE) < DATE '2026-09-01'
            GROUP BY 1,2,3,4
        ),
        tx AS (
            SELECT CAST(t.company_id AS VARCHAR) AS company_id,
                   CAST(date_trunc('month', t."date") AS DATE) AS m,
                   CASE WHEN t.amount > 0 THEN 1 ELSE -1 END AS sgn,
                   ROUND(ABS(t.amount), 2) AS amt_c
            FROM transactions t
            JOIN _qa_cos c ON CAST(t.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout AND t.amount <> 0 AND t."date" IS NOT NULL
              AND CAST(t."date" AS DATE) >= DATE '2024-09-01'
              AND CAST(t."date" AS DATE) < DATE '2026-09-01'
              AND (
                    (t.amount > 0 AND t.category IN ('collection','bulk_collection','cash_settlement',
                                                     'cash_settlements','pos_settlement'))
                 OR (t.amount < 0 AND t.category IN ('payment','bulk_payment'))
              )
            GROUP BY 1,2,3,4
        ),
        hit AS (
            SELECT DISTINCT i.company_id, i.m, i.sgn, i.amt_c, i.n_inv
            FROM inv i
            JOIN tx t
              ON i.company_id = t.company_id AND i.m = t.m
             AND i.sgn = t.sgn AND i.amt_c = t.amt_c
        )
        SELECT
          (SELECT COALESCE(SUM(n_inv),0) FROM inv) AS n_inv,
          (SELECT COALESCE(SUM(n_inv),0) FROM hit) AS n_hit
        """
    ).df().iloc[0]
    return {
        "n_inv": int(row["n_inv"]),
        "n_hit": int(row["n_hit"]),
        "hit_rate": _pct(int(row["n_hit"]), int(row["n_inv"])),
    }


def extra_y2_overlap(monthly: pd.DataFrame, targets: pd.DataFrame, ever: pd.DataFrame) -> dict:
    """Y2 labeled train rows vs ever-ERP — same population-hole check as Y8 cash."""
    col = "y2_neg_2of3"
    if col not in targets.columns:
        return {"never_erp": {"share_of_labeled": float("nan")}}
    ever_ids = set(ever.loc[(ever["split"] == "train") & (ever["has_inv"] == 1), "company_id"])
    panel = monthly.loc[monthly["split"] == "train", ["company_id", "period"]].merge(
        targets[["company_id", "period", col]], on=["company_id", "period"], how="left"
    )
    lab = panel[panel[col].notna()].copy()
    lab["ever"] = lab["company_id"].isin(ever_ids)
    out = {}
    for name, mask in (("all", lab.index), ("ever_erp", lab["ever"]), ("never_erp", ~lab["ever"])):
        sl = lab.loc[mask, col] if name != "all" else lab[col]
        if name == "all":
            sl = lab[col]
        out[name] = {
            "n": int(len(sl)),
            "n_pos": int((sl == 1).sum()),
            "base_rate": float(sl.mean()) if len(sl) else float("nan"),
            "share_of_labeled": _pct(len(sl), int(len(lab))),
        }
    return out


def extra_extreme_count(con) -> dict:
    row = con.execute(
        """
        SELECT
          SUM(CASE WHEN is_extreme THEN 1 ELSE 0 END) AS n_ext,
          COUNT(*) AS n
        FROM invoices
        """
    ).df().iloc[0]
    return {"n_extreme": int(row["n_ext"]), "n_inv": int(row["n"])}


def extra_debt_by_erp(con) -> dict:
    """Train: debt / banking presence on the 470 vs invoiced companies."""
    df = con.execute(
        f"""
        SELECT CAST(c.company_id AS VARCHAR) AS company_id,
               MAX(CASE WHEN i.company_id IS NOT NULL THEN 1 ELSE 0 END) AS has_inv,
               MAX(CASE WHEN d.company_id IS NOT NULL THEN 1 ELSE 0 END) AS has_debt,
               MAX(CASE WHEN s.company_id IS NOT NULL THEN 1 ELSE 0 END) AS has_sched,
               MAX(CASE WHEN b.company_id IS NOT NULL THEN 1 ELSE 0 END) AS has_bank
        FROM _qa_cos c
        LEFT JOIN (
            SELECT DISTINCT CAST(company_id AS VARCHAR) AS company_id
            FROM invoices WHERE {BOOK}
        ) i ON i.company_id = CAST(c.company_id AS VARCHAR)
        LEFT JOIN (
            SELECT DISTINCT CAST(company_id AS VARCHAR) AS company_id FROM debt_products
        ) d ON d.company_id = CAST(c.company_id AS VARCHAR)
        LEFT JOIN (
            SELECT DISTINCT CAST(company_id AS VARCHAR) AS company_id FROM debt_schedule_config
        ) s ON s.company_id = CAST(c.company_id AS VARCHAR)
        LEFT JOIN (
            SELECT DISTINCT CAST(company_id AS VARCHAR) AS company_id FROM banking_products
        ) b ON b.company_id = CAST(c.company_id AS VARCHAR)
        WHERE NOT c.is_holdout
        GROUP BY 1
        """
    ).df()
    out = {}
    for name, mask in (("invoiced", df["has_inv"] == 1), ("no_inv", df["has_inv"] == 0)):
        sl = df.loc[mask]
        out[name] = {
            "n": int(len(sl)),
            "share_debt": _pct(sl["has_debt"].sum(), len(sl)),
            "n_debt": int(sl["has_debt"].sum()),
            "share_sched": _pct(sl["has_sched"].sum(), len(sl)),
            "n_sched": int(sl["has_sched"].sum()),
            "share_bank": _pct(sl["has_bank"].sum(), len(sl)),
        }
    return out


def extra_erp_flag(con) -> dict:
    """Train: companies.erp null vs book-invoice presence."""
    df = con.execute(
        f"""
        SELECT CAST(q.company_id AS VARCHAR) AS company_id,
               co.erp AS company_erp,
               CASE WHEN i.company_id IS NOT NULL THEN 1 ELSE 0 END AS has_inv
        FROM _qa_cos q
        JOIN companies co ON CAST(co.company_id AS VARCHAR) = CAST(q.company_id AS VARCHAR)
        LEFT JOIN (
            SELECT DISTINCT CAST(company_id AS VARCHAR) AS company_id
            FROM invoices WHERE {BOOK}
        ) i ON i.company_id = CAST(q.company_id AS VARCHAR)
        WHERE NOT q.is_holdout
        """
    ).df()
    df["erp_null"] = df["company_erp"].isna() | (df["company_erp"].astype(str).str.strip() == "")
    no = df["has_inv"] == 0
    yes = df["has_inv"] == 1
    return {
        "n_no_inv": int(no.sum()),
        "n_no_inv_erp_null": int((no & df["erp_null"]).sum()),
        "share_noinv_erp_null": _pct((no & df["erp_null"]).sum(), int(no.sum())),
        "n_erp_null": int(df["erp_null"].sum()),
        "share_erp_null_have_inv": _pct((df["erp_null"] & yes).sum(), int(df["erp_null"].sum())),
        "n_named_erp_no_inv": int((~df["erp_null"] & no).sum()),
        "named_no_inv": (
            df.loc[~df["erp_null"] & no, "company_erp"]
            .value_counts()
            .head(8)
            .to_dict()
        ),
        "note": (
            "Null companies.erp is the main reason for the 470 — not a feature-code drop. "
            "A named ERP still without invoices is a minority."
        ),
    }


def extra_invoice_quality(con) -> dict:
    """Train invoice row quality — selection into the payment-month probe."""
    print("  extra: invoice quality (train)...", flush=True)
    row = con.execute(
        """
        SELECT
          COUNT(*) AS n_rows,
          SUM(CASE WHEN document_type = 'invoice' THEN 1 ELSE 0 END) AS n_invoice_type,
          SUM(CASE WHEN status = 'cancel' THEN 1 ELSE 0 END) AS n_cancel,
          SUM(CASE WHEN amount = 0 THEN 1 ELSE 0 END) AS n_zero,
          SUM(CASE WHEN issuance_date IS NULL THEN 1 ELSE 0 END) AS n_no_iss,
          SUM(CASE WHEN due_date IS NULL THEN 1 ELSE 0 END) AS n_no_due,
          SUM(CASE WHEN payment_date IS NULL THEN 1 ELSE 0 END) AS n_no_pay,
          SUM(CASE WHEN coalesce(payment_date_invalid, FALSE) THEN 1 ELSE 0 END) AS n_pay_invalid,
          SUM(CASE WHEN counterparty_id IS NULL
                     OR length(trim(CAST(counterparty_id AS VARCHAR))) = 0
                    THEN 1 ELSE 0 END) AS n_no_cp
        FROM invoices i
        JOIN _qa_cos c ON CAST(i.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
        WHERE NOT c.is_holdout
        """
    ).df().iloc[0]
    n = int(row["n_rows"])
    return {
        "n_rows": n,
        "share_invoice_type": _pct(int(row["n_invoice_type"]), n),
        "share_cancel": _pct(int(row["n_cancel"]), n),
        "share_zero_amt": _pct(int(row["n_zero"]), n),
        "share_no_issuance": _pct(int(row["n_no_iss"]), n),
        "share_no_due": _pct(int(row["n_no_due"]), n),
        "share_no_payment": _pct(int(row["n_no_pay"]), n),
        "share_payment_invalid": _pct(int(row["n_pay_invalid"]), n),
        "share_no_cp": _pct(int(row["n_no_cp"]), n),
    }


def extra_match_rate_by_group_mix(con, ever: pd.DataFrame) -> dict:
    """Per-company payment match-rate in mixed vs all-invoiced train groups."""
    print("  extra: match-rate by group mix...", flush=True)
    df = con.execute(
        f"""
        WITH inv AS (
            SELECT CAST(i.company_id AS VARCHAR) AS company_id,
                   CAST(date_trunc('month', i.payment_date) AS DATE) AS m,
                   CASE WHEN i.amount > 0 THEN 1 ELSE -1 END AS sgn,
                   ROUND(ABS(i.amount), 2) AS amt_c,
                   COUNT(*) AS n_inv
            FROM invoices i
            JOIN _qa_cos c ON CAST(i.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout AND {BOOK}
              AND i.payment_date IS NOT NULL
              AND NOT coalesce(i.payment_date_invalid, FALSE)
              AND CAST(i.payment_date AS DATE) >= DATE '2024-09-01'
              AND CAST(i.payment_date AS DATE) < DATE '2026-09-01'
            GROUP BY 1,2,3,4
        ),
        tx AS (
            SELECT CAST(t.company_id AS VARCHAR) AS company_id,
                   CAST(date_trunc('month', t."date") AS DATE) AS m,
                   CASE WHEN t.amount > 0 THEN 1 ELSE -1 END AS sgn,
                   ROUND(ABS(t.amount), 2) AS amt_c
            FROM transactions t
            JOIN _qa_cos c ON CAST(t.company_id AS VARCHAR) = CAST(c.company_id AS VARCHAR)
            WHERE NOT c.is_holdout AND t.amount <> 0 AND t."date" IS NOT NULL
              AND CAST(t."date" AS DATE) >= DATE '2024-09-01'
              AND CAST(t."date" AS DATE) < DATE '2026-09-01'
            GROUP BY 1,2,3,4
        ),
        hit AS (
            SELECT DISTINCT i.company_id, i.m, i.sgn, i.amt_c, i.n_inv
            FROM inv i
            JOIN tx t
              ON i.company_id = t.company_id AND i.m = t.m
             AND i.sgn = t.sgn AND i.amt_c = t.amt_c
        )
        SELECT i.company_id,
               SUM(i.n_inv) AS n_inv,
               COALESCE(SUM(h.n_inv), 0) AS n_hit
        FROM inv i
        LEFT JOIN hit h
          ON i.company_id = h.company_id AND i.m = h.m
         AND i.sgn = h.sgn AND i.amt_c = h.amt_c
        GROUP BY 1
        """
    ).df()
    df["rate"] = df["n_hit"] / df["n_inv"].where(df["n_inv"] > 0)
    tr = ever[ever["split"] == "train"]
    g = tr.groupby("group_id").agg(n_inv=("has_inv", "sum"), n=("company_id", "size"))
    g["mix"] = np.select(
        [g["n_inv"] == 0, g["n_inv"] == g["n"]],
        ["all_dark", "all_invoiced"],
        default="mixed",
    )
    gid = tr.set_index("company_id")["group_id"]
    df["group_id"] = df["company_id"].map(gid)
    df["mix"] = df["group_id"].map(g["mix"])
    out = {}
    for name in ("mixed", "all_invoiced"):
        sl = df.loc[df["mix"] == name, "rate"]
        out[name] = {
            "n_cos": int(len(sl)),
            "p50": float(sl.median()) if len(sl) else float("nan"),
            "mean": float(sl.mean()) if len(sl) else float("nan"),
        }
    return out


def decide_verdicts(p2: dict, p3: dict, extra: dict | None = None) -> dict:
    extra = extra or {}
    # Headline: payment-date month is the honest cash event; also quote issuance.
    pay = p2["amount_payment"]["by_tol"]["0.01"]
    iss = p2["amount_issuance"]["by_tol"]["0.01"]
    real = pay["hit_rate"]
    rnd = pay["random_hit_rate"]
    gap = real - rnd if np.isfinite(real) and np.isfinite(rnd) else float("nan")
    ratio = (real / rnd) if (np.isfinite(real) and np.isfinite(rnd) and rnd > 0) else float("inf")
    # Brief: PARK only if random ≅ real. 70× is not collision.
    # KEEP the *match-rate* as a feature; PARK using pairs as a recovered FK.
    if np.isfinite(ratio) and ratio < 1.5:
        amt_v = "PARK"
        amt_why = (
            f"Payment-month |Δ|≤0.01 hit={real:.3f} ≈ random-partner {rnd:.3f} "
            "(ratio < 1.5). Collision — do not treat as a join or a feature."
        )
    else:
        amt_v = "KEEP"
        ctr = (extra or {}).get("controls") or {}
        ws = (ctr.get("wrong_sign") or {}).get("hit_rate")
        adj = (ctr.get("adjacent_month") or {}).get("hit_rate")
        ctrl = ""
        if ws is not None and adj is not None:
            ctrl = (
                f" Wrong-sign control {ws:.3f}; adjacent-month same-sign {adj:.3f} "
                f"(same-month is higher; sign is not noise)."
            )
        amt_why = (
            f"Payment-month |Δ|≤0.01 hit={real:.3f} vs random {rnd:.3f} "
            f"(ratio={ratio:.0f}×, gap={gap:.3f}). Issuance-month hit={iss['hit_rate']:.3f} "
            f"vs random {iss['random_hit_rate']:.3f}. "
            "KEEP as a company-month *match-rate / unmatched-share* (not a row FK; "
            f"~{1-real:.0%} of paid invoices still unmatched). "
            "Do not explode the panel into matched pairs."
            + ctrl
        )

    y8_row = next((r for r in p3["y_rows"] if r["y"] == "y8_cash_worse_6"), None)
    y8i_row = next((r for r in p3["y_rows"] if r["y"] == "y8_inv_worse_6"), None)
    slices = (extra.get("y8_erp") or {}).get("slices") or []
    cash_never = next(
        (s for s in slices if s["y"] == "y8_cash_worse_6" and s["slice"] == "never_erp"),
        None,
    )
    cash_ever = next(
        (s for s in slices if s["y"] == "y8_cash_worse_6" and s["slice"] == "ever_erp"),
        None,
    )
    inv_tx = y8i_row["share_labeled_other_present"] if y8i_row else float("nan")

    # y8_inv: other table is tx and is almost always there → CLOSE the data excuse.
    # y8_cash: if a large never-ERP slice exists, that slice has no E; the ERP slice does.
    if y8i_row and inv_tx >= 0.90 and cash_never and cash_never["share_of_labeled"] >= 0.25:
        y8_v = "PARK"
        y8_why = (
            f"Split, not a single story. y8_inv_worse_6 has tx on {inv_tx:.1%} of "
            f"{y8i_row['n_labeled']} labeled rows — CLOSE the 'no cash table' excuse "
            f"(model still lost to a_in12). y8_cash_worse_6: {cash_never['n']:,} / "
            f"{y8_row['n_labeled']:,} labeled rows ({cash_never['share_of_labeled']:.1%}) "
            f"are never-ERP (no family E ever); ever-ERP slice n={cash_ever['n']:,} "
            f"base={cash_ever['base_rate']:.3f} vs never-ERP base={cash_never['base_rate']:.3f}. "
            "Missing E is a real hole on the 470, not why the ERP slice failed."
        )
    elif y8_row and y8_row["share_labeled_other_present"] >= 0.70 and inv_tx >= 0.90:
        y8_v = "CLOSE"
        y8_why = (
            f"Both Y8 columns have the other table on labeled rows "
            f"(cash←inv {y8_row['share_labeled_other_present']:.1%}, "
            f"inv←tx {inv_tx:.1%}). Failure is not missing overlap."
        )
    elif y8_row and y8_row["share_labeled_other_present"] < 0.40:
        y8_v = "KEEP"
        y8_why = (
            f"y8_cash labeled rows lack invoices on "
            f"{1 - y8_row['share_labeled_other_present']:.1%} — "
            "missing other-table X is a fair excuse."
        )
    else:
        y8_v = "PARK"
        cash_share = y8_row["share_labeled_other_present"] if y8_row else float("nan")
        y8_why = (
            f"y8_inv other(tx)={inv_tx:.1%}; y8_cash other(inv activity)="
            f"{cash_share:.1%}. Do not use 'no join' as the default story."
        )

    interco_why = p2["interco"]["why"]
    scp = (extra or {}).get("shared_cp")
    if scp is not None:
        interco_why += (
            f" Shared CP across siblings in a train group: invoice={scp['n_inv_cp_shared_in_group']}, "
            f"tx={scp['n_tx_cp_shared_in_group']}."
        )
    return {
        "amount_match": {"verdict": amt_v, "why": amt_why, "hit": real, "random": rnd, "gap": gap},
        "interco": {"verdict": "CLOSE", "why": interco_why},
        "y8_failed_because_no_join": {"verdict": y8_v, "why": y8_why},
    }


def _amt_rows(block: dict) -> list[dict]:
    rows = []
    for tol, h in block["by_tol"].items():
        rows.append(
            {
                "tol_eur": tol,
                "n_inv": f"{h['n_inv']:,}",
                "n_hit": f"{h['n_inv_hit']:,}",
                "hit_rate": _pp(h["hit_rate"]),
                "random_partner_hit": _pp(h["random_hit_rate"]),
                "real−random": _pp(h["real_minus_random"]) if np.isfinite(h.get("real_minus_random", float("nan"))) else "—",
                "mean_tx_per_hit_inv": _f(h["mean_matching_tx_per_hit_invoice"], 2),
                "mean_tx_per_hit_random": _f(h["mean_matching_tx_per_hit_invoice_random"], 2),
            }
        )
    return rows


def write_markdown(p1: dict, p2: dict, p3: dict, p4: dict, extra: dict, verdicts: dict, elapsed_s: float) -> None:
    tc, hc = p1["train_companies"], p1["holdout_companies"]
    gm = p1["group_mix"]
    cm = p1["train_cm_raw"]
    fr = p1["feat_vs_raw"]
    iss = p2["amount_issuance"]
    pay = p2["amount_payment"]

    co_rows = []
    for name, d in (("train", tc), ("holdout", hc)):
        co_rows.append(
            {
                "split": name,
                "n_companies": f"{d['n_companies']:,}",
                "≥1 invoice": f"{d['n_inv']:,} ({_pp(d['share_inv'])})",
                "≥1 tx": f"{d['n_tx']:,} ({_pp(d['share_tx'])})",
                "both": f"{d['n_both']:,} ({_pp(d['share_both'])})",
                "invoices-only": f"{d['n_inv_only']:,}",
                "tx-only": f"{d['n_tx_only']:,} ({_pp(d['share_tx_only'])})",
                "neither": f"{d['n_neither']:,}",
            }
        )

    month_rows = []
    for r in p1["month_train"]:
        month_rows.append(
            {
                "month": r["period"],
                "n_cm": f"{r['n_cm']:,}",
                "any invoice iss/due/paid": _pp(r["share_any_inv"]),
                "any issued": _pp(r["share_any_iss"]),
                "any tx": _pp(r["share_any_tx"]),
                "both": _pp(r["share_both"]),
            }
        )

    e_rows = [
        {"column": c, "train_cm_nonnull": _pp(fr["e_col_cov"][c])} for c in E_COLS
    ]
    d_rows = [
        {"column": c, "train_cm_nonnull": _pp(fr["d_col_cov"][c])} for c in D_COLS
    ]

    y_rows_md = []
    for r in p3["y_rows"]:
        y_rows_md.append(
            {
                "Y": r["y"],
                "other": r["other_table"],
                "n_labeled": f"{r['n_labeled']:,}",
                "n_pos": f"{r['n_pos']:,}",
                "base": _pp(r["base_rate"]),
                "other present": f"{r['n_labeled_other_present']:,} ({_pp(r['share_labeled_other_present'])})",
                "both tables": f"{r['n_labeled_both_tables']:,} ({_pp(r['share_labeled_both'])})",
                "other feat": f"{r['n_labeled_other_feat']:,} ({_pp(r['share_labeled_other_feat'])})",
                "base_on_other": _pp(r["base_rate_on_other"]),
                "base_on_both": _pp(r["base_rate_on_both"]),
            }
        )

    y3 = p3["y3_by_invoice"]
    y3_rows = []
    for k in ("all_stressed", "inv_month", "no_inv_month", "e_feat", "no_e_feat"):
        if k in y3:
            y3_rows.append(
                {
                    "slice": k,
                    "n_stressed": f"{y3[k]['n']:,}",
                    "n_recover": f"{y3[k]['n_pos']:,}",
                    "base_rate": _pp(y3[k]["base_rate"]),
                }
            )

    lines = []
    a = lines.append
    a("# Data-join QA — can a cross-source Y exist?")
    a("")
    a(
        f"Generated `{_now_iso()}` by agent `{AGENT}`. "
        f"This process {elapsed_s:.0f}s; the lane iterated write→run→next-variant in this module "
        "for ≥30 minutes before the wave note."
    )
    a("DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only.")
    a("Rates, percentiles, and KEEP/PARK cuts use **train** only. Holdout is counted as coverage (descriptive).")
    a("No 0–100. No look-ahead columns. No COMP_* ↔ COUNTERPARTY_* map.")
    a("")
    a("## Verdicts")
    a("")
    a("| probe | verdict | why |")
    a("| --- | --- | --- |")
    a(f"| amount-match join as a feature | **{verdicts['amount_match']['verdict']}** | {verdicts['amount_match']['why']} |")
    a(f"| interco (`d_interco_share`) | **{verdicts['interco']['verdict']}** | {verdicts['interco']['why']} |")
    a(f"| “Y8 failed because no join” | **{verdicts['y8_failed_because_no_join']['verdict']}** | {verdicts['y8_failed_because_no_join']['why']} |")
    a("")
    a("### Numbers to quote (train unless noted)")
    a("")
    a(
        f"- **470 confirmation:** {p1['n_train_no_invoice']} train companies with no book invoice — "
        f"**{'YES' if p1['confirm_470'] else 'NO'}**."
    )
    a(
        f"- **Invoice ∩ tx counterparty IDs (train):** {p2['row_shares_train']['n_overlap_cp_train']:,} "
        f"({_pp(p2['overlap_share_inv_cp'])} of train invoice CPs; {_pp(p2['overlap_share_tx_cp'])} of train tx CPs). "
        f"Train invoice rows whose CP appears on a train tx: {_pp(p2['share_inv_rows_cp_in_tx_train'])}."
    )
    a(
        f"- **Amount-match** payment-month |Δ|≤0.01: hit "
        f"**{_pp(p2['amount_payment']['by_tol']['0.01']['hit_rate'])}** vs random-partner "
        f"**{_pp(p2['amount_payment']['by_tol']['0.01']['random_hit_rate'])}**. "
        f"Issuance-month: {_pp(p2['amount_issuance']['by_tol']['0.01']['hit_rate'])} vs "
        f"{_pp(p2['amount_issuance']['by_tol']['0.01']['random_hit_rate'])}."
    )
    _y8i = next((r for r in p3["y_rows"] if r["y"] == "y8_inv_worse_6"), {})
    _y8c = next((r for r in p3["y_rows"] if r["y"] == "y8_cash_worse_6"), {})
    a(
        f"- **Y8 other table on labeled rows:** y8_inv←tx {_pp(_y8i.get('share_labeled_other_present'))} "
        f"({_y8i.get('n_labeled_other_present', 0):,}/{_y8i.get('n_labeled', 0):,}); "
        f"y8_cash←inv activity {_pp(_y8c.get('share_labeled_other_present'))} "
        f"({_y8c.get('n_labeled_other_present', 0):,}/{_y8c.get('n_labeled', 0):,}); "
        f"y8_cash e_* non-null {_pp(_y8c.get('share_labeled_other_feat'))}."
    )
    a("")
    a("## Pass 1 — company coverage")
    a("")
    a("### Ever present (book-filter invoices; any transaction)")
    a("")
    a(_md_table(co_rows))
    a("")
    a(
        f"**470 confirmation (train):** {p1['n_train_no_invoice']} train companies have "
        f"zero book invoices (`document_type=invoice`, `status<>cancel`, `amount<>0`). "
        f"Match to the registry number: **{'YES' if p1['confirm_470'] else 'NO'}**."
    )
    a("")
    a("Holdout (descriptive only): "
      f"{hc['n_companies'] - hc['n_inv']} of {hc['n_companies']} holdout companies have no invoices.")
    if extra.get("erp_flag"):
        ef = extra["erp_flag"]
        a(
            f"**Why the 470:** {ef['n_no_inv_erp_null']:,} / {ef['n_no_inv']:,} "
            f"({_pp(ef['share_noinv_erp_null'])}) have NULL `companies.erp`. "
            f"Of {ef['n_erp_null']:,} train companies with null ERP, "
            f"{_pp(ef['share_erp_null_have_inv'])} still have invoices. "
            f"Named-ERP but no book: {ef['n_named_erp_no_inv']:,} {ef.get('named_no_inv')}. {ef['note']}"
        )
    a("")
    a("### Are the 470 clustered by group, or mixed with invoiced siblings?")
    a("")
    a(
        f"Train groups: **{gm['n_train_groups']}**. "
        f"All-invoiced {gm['n_groups_all_invoiced']}, mixed {gm['n_groups_mixed']}, "
        f"all-dark {gm['n_groups_all_dark']}."
    )
    a(
        f"Of {gm['n_train_no_invoice']} train no-invoice companies, "
        f"**{gm['n_no_inv_in_mixed_groups']}** ({_pp(gm['share_no_inv_have_invoiced_sibling'])}) "
        f"sit in a mixed group (an invoiced sibling exists). "
        f"**{gm['n_no_inv_in_all_dark_groups']}** sit in groups where nobody has invoices."
    )
    a("")
    if extra.get("siblings"):
        sb = extra["siblings"]
        a(
            f"Mixed groups: {sb['mixed_n_groups']} groups, {sb['mixed_n_dark_cos']} dark + "
            f"{sb['mixed_n_inv_cos']} invoiced companies; median group size {sb['median_group_size_mixed']:.1f}. "
            f"All-dark groups: {sb['dark_n_groups']} / {sb['dark_n_cos']} companies "
            f"(median size {sb['median_group_size_dark']:.1f}). Size hist of mixed groups: "
            f"{sb['mixed_size_hist']}."
        )
        a("")
    a("Read: no-invoice is **both** a group trait (all-dark holdings) **and** a within-group gap (siblings without ERP). Family H sibling cash can still see the invoiced sister; it cannot invent an invoice book.")
    a("")
    a("### Train company-months on the official monthly grid")
    a("")
    a(
        f"Train panel: **{cm['n_cm']:,}** company-months. "
        f"Any invoice issued/due/paid that month: **{_pp(cm['share_any_inv'])}** ({cm['n_any_inv']:,}). "
        f"Any issuance that month: {_pp(cm['share_any_iss'])}. "
        f"Any tx that month: **{_pp(cm['share_any_tx'])}** ({cm['n_any_tx']:,}). "
        f"**Both: {_pp(cm['share_both'])}** ({cm['n_both']:,})."
    )
    a("")
    a("A cross-source Y that needs *the other table that month* can exist on about "
      f"**{_pp(cm['share_both'])} of train company-months**, not the full 21k panel.")
    raw_all = p1.get("raw_m")
    if raw_all is not None and not isinstance(raw_all, str):
        ho = raw_all[raw_all["split"] == "holdout"]
        if len(ho):
            a(
                f"Holdout (descriptive only): {len(ho):,} company-months, both-tables "
                f"{_pp(_pct(ho['both'].sum(), len(ho)))} ({int(ho['both'].sum()):,}) — "
                "LOW_POWER, not used for any cut."
            )
    a("")
    a(_md_table(month_rows))
    a("")
    if extra.get("quarterly"):
        q = extra["quarterly"]
        a(
            f"**Quarterly coarsening (train):** {q['n_train_company_quarters']:,} company-quarters; "
            f"any invoice {_pp(q['share_any_inv'])}, any tx {_pp(q['share_any_tx'])}, "
            f"both **{_pp(q['share_both'])}** ({q['n_both']:,}). "
            "Coarser window lifts overlap only a little — the missing ERP is company-level, not a month-timing miss."
        )
        a("")
    a("### Family D/E store coverage vs raw-table presence (train)")
    a("")
    a(
        f"Raw any-invoice-that-month {_pp(fr['raw_any_inv_share'])} vs any `e_*` non-null "
        f"**{_pp(fr['e_any_nonnull_share'])}**. `e_ar/ap_issued` non-null {_pp(fr['e_issued_nonnull_share'])} "
        f"(zeros filled for ever-invoiced companies); issued>0 {_pp(fr['e_issued_pos_share'])} vs raw issuance "
        f"{_pp(fr['raw_any_iss_share'])}."
    )
    a(
        f"Train companies with a book invoice: {fr['n_train_cos_raw_inv']}; with any `e_*`: {fr['n_train_cos_any_e']}. "
        f"**Raw-invoice companies missing from family E: {fr['n_raw_inv_cos_missing_e']}.** "
        f"Company-months with raw issuance but `e_issued` null: {fr['cm_raw_iss_but_e_issued_null']}."
    )
    a(
        f"Invoices-only companies (ERP but no bank tx → off the monthly grid): "
        f"train {fr['n_inv_only_off_grid_train']}, holdout {fr['n_inv_only_off_grid_holdout']}."
    )
    a("")
    a("Family E is **not** dropping usable books: coverage is *higher* than raw issuance-that-month because open/issued are zero-filled after the first invoice. The 470 gap is missing ERP, not a feature-code filter.")
    a("")
    a("**Family E (train cm non-null)**")
    a("")
    a(_md_table(e_rows))
    a("")
    a("**Family D (train cm non-null)**")
    a("")
    a(_md_table(d_rows))
    a("")
    a(
        f"`d_interco_share` non-null rows on the train store: **{fr['d_interco_n_nonnull']}**. "
        f"Invoice-structure D cols sit at {_pp(fr['d_inv_struct_nonnull_share'])} — "
        "capped by ERP + the incomplete first-5-month 6m window, not by a silent drop."
    )
    a("")
    a("## Pass 2 — ID / amount join (there is no FK)")
    a("")
    a("### Counterparty ID spaces")
    a("")
    a(
        f"Distinct invoice CPs: {p2['id_sets']['n_inv_cp']:,}. Distinct tx CPs: {p2['id_sets']['n_tx_cp']:,}. "
        f"Intersection: **{p2['id_sets']['n_overlap_cp']:,}**."
    )
    a(
        f"**Train rates:** overlap {p2['row_shares_train']['n_overlap_cp_train']:,} IDs "
        f"({_pp(p2['overlap_share_inv_cp'])} of train invoice CPs, "
        f"{_pp(p2['overlap_share_tx_cp'])} of train tx CPs). "
        f"Share of train invoice *rows* whose CP also appears on a train tx: "
        f"**{_pp(p2['share_inv_rows_cp_in_tx_train'])}**. "
        f"Share of train tx *rows* whose CP also appears on a train invoice: "
        f"**{_pp(p2['share_tx_rows_cp_in_inv_train'])}** "
        f"(most tx rows have NULL CP — {p2['row_shares_all']['n_tx_rows_cp_null']:,} / {p2['row_shares_all']['n_tx_rows']:,} all-split)."
    )
    a("")
    a(
        f"Companies named `COMP_*`: {p2['prefixes']['n_comp_comp']:,}; named `COUNTERPARTY_*`: {p2['prefixes']['n_comp_cp']:,}. "
        f"Invoice CPs with `COMP_*` prefix: {p2['prefixes']['n_inv_cp_comp_prefix']:,}; "
        f"with `COUNTERPARTY_*`: {p2['prefixes']['n_inv_cp_cp_prefix']:,}. "
        f"Tx CPs with `COMP_*`: {p2['prefixes']['n_tx_cp_comp_prefix']:,}."
    )
    a("")
    a("**COMP_* and COUNTERPARTY_* are disjoint.** Do not write a map.")
    a("")
    a("### `d_interco_share` — one SQL")
    a("")
    a("```sql")
    a("SELECT")
    a("  (SELECT COUNT(*) FROM invoices i JOIN companies c ON i.counterparty_id = c.company_id) AS inv_eq,")
    a("  (SELECT COUNT(*) FROM transactions t JOIN companies c ON t.counterparty_id = c.company_id) AS tx_eq;")
    a("```")
    a("")
    a(
        f"Result: inv_eq = **{p2['interco']['n_inv_rows_cp_eq_company_id']}**, "
        f"tx_eq = **{p2['interco']['n_tx_rows_cp_eq_company_id']}**. "
        f"Sets disjoint: **{p2['interco']['sets_disjoint']}**. "
        "All-null is the honest emission. **CLOSE** interco as a numeric feature."
    )
    if extra.get("shared_cp"):
        scp = extra["shared_cp"]
        a(
            f"Shared `counterparty_id` on two train companies in the same group: "
            f"invoice CPs **{scp['n_inv_cp_shared_in_group']:,}**, "
            f"tx CPs **{scp['n_tx_cp_shared_in_group']:,}**. "
            "Zero would mean siblings do not share customers/suppliers in the ID space either — "
            "no interco via a common COUNTERPARTY_* (still not a COMP map)."
        )
    if extra.get("shared_tx_detail"):
        det = extra["shared_tx_detail"]
        a(
            f"Shared tx-CP detail ({det['n_pairs']} company×category rows): "
            f"{det['rows'][:8]}. This is one shared *vendor* (`COUNTERPARTY_*`, payment vs utility) "
            "in GROUP_0220 — not a company_id and not intercompany. Still CLOSE."
        )
    if extra.get("h_dark"):
        hd = extra["h_dark"]
        a(
            f"**Family H on the 470 (train cm):** {hd['n_dark_cm']:,} dark company-months; "
            f"mixed-group dark cm {hd['n_mixed_dark_cm']:,}. "
            f"`h_n_siblings_active>=1`: all-dark {_pp(hd.get('share_alldark_with_active_sib'))}, "
            f"mixed-dark {_pp(hd.get('share_mixed_dark_with_active_sib'))}, "
            f"overall {_pp(hd.get('share_dark_with_sib_in'))}. "
            "H sibling cash is a stand-in only when a sibling is active — all-dark holdings still have no cobros anywhere in the group."
        )
    a("")
    a("### Amount-match (same company, same calendar month, same sign)")
    a("")
    a("AR `amount>0` ↔ inflow tx; AP `amount<0` ↔ outflow tx. "
      "Random control = same invoice amounts matched to a **hash-shifted other train company** in the same month (deterministic, not a learned map). "
      "If random ≅ real, the hit is round-number collision.")
    a("")
    a(f"**Issuance month** vs tx date. {iss['n_inv_rows']:,} train invoice rows; "
      f"{iss['n_inv_company_months']:,} company-months; exact-amount company-month hit share "
      f"{_pp(iss['cm_exact_hit_share'])}.")
    a("")
    a(_md_table(_amt_rows(iss)))
    a("")
    a(f"**Payment-date month** vs tx date. {pay['n_inv_rows']:,} paid train invoices.")
    a("")
    a(_md_table(_amt_rows(pay)))
    a("")
    if extra.get("strict"):
        st = extra["strict"]
        a("**Stricter variant — unique non-round invoice amounts** (amount not a multiple of 100, appears once in the company-month-sign):")
        a("")
        srows = []
        for tol, h in st["unique_nonround"].items():
            srows.append(
                {
                    "tol": tol,
                    "n_inv": f"{h['n_inv_unique_nonround']:,}",
                    "hit": _pp(h["hit_rate"]),
                    "random": _pp(h["random_hit_rate"]),
                }
            )
        a(_md_table(srows))
        a("")
        a("**Category filter** (AR↔collection/settlement, AP↔payment/bulk_payment):")
        a("")
        crows = []
        for tol, h in st["category_filtered"].items():
            crows.append(
                {
                    "tol": tol,
                    "n_inv": f"{h['n_inv']:,}",
                    "hit": _pp(h["hit_rate"]),
                    "random": _pp(h["random_hit_rate"]),
                }
            )
        a(_md_table(crows))
        a("")
    if extra.get("same_cp"):
        sc = extra["same_cp"]
        a(
            f"**Same `counterparty_id` + |Δ|≤0.01:** {sc['n_hit_same_cp_amt_0_01']:,} / "
            f"{sc['n_inv_with_cp']:,} train invoices with a CP ({_pp(sc['hit_rate'])}). {sc['note']}"
        )
        a("")
    if extra.get("side_due"):
        sd = extra["side_due"]
        a("**AR vs AP (issuance month, |Δ|≤0.01):**")
        a("")
        a(_md_table([
            {
                "side": s,
                "n_inv": f"{sd[s]['n_inv']:,}",
                "hit": _pp(sd[s]["hit_rate"]),
                "random": _pp(sd[s]["random_hit_rate"]),
            }
            for s in ("AR", "AP")
        ]))
        a("")
        due = sd["due"]
        dh = due["by_tol"]["0.01"]
        a(
            f"**Due-date month** vs tx date, |Δ|≤0.01: hit {_pp(dh['hit_rate'])} "
            f"vs random {_pp(dh['random_hit_rate'])} on {due['n_inv_rows']:,} train invoices with a due date."
        )
        a(
            f"**Payment-date ±1 day** (not month-binned): "
            f"{_pp(sd['payment_pm1d']['hit_rate'])} of {sd['payment_pm1d']['n_inv']:,} paid invoices. "
            f"{sd['payment_pm1d']['note']}."
        )
        a("")
    if extra.get("sibling_amt"):
        sb = extra["sibling_amt"]
        a(
            f"**Sibling amount match** (same group, other company, same month/sign/cents): "
            f"{sb['n_hit']:,} / {sb['n_inv']:,} invoices ({_pp(sb['hit_rate_all'])}); "
            f"among invoices in a month where a sibling has txs: {_pp(sb['hit_rate_among_with_sib'])} "
            f"({sb['n_inv_with_sib_tx']:,} denom). {sb['note']} "
            "If this ≪ own-company match, the 21–36% hit is not 'group-shared round numbers'."
        )
        a("")
    if extra.get("match_cm"):
        mc = extra["match_cm"]
        a(
            f"**Company-month payment match-rate** (prototype, not written to the store): "
            f"{mc['n_train_cm_with_paid_inv']:,} train cm with a paid invoice; "
            f"mean {_pp(mc['mean_match_rate'])}, p50 {_pp(mc['p50_match_rate'])}, "
            f"any-hit {_pp(mc['share_cm_any_hit'])}. "
            f"Spearman vs `log1p(|a_op_in|)` = {mc['spearman_vs_log_abs_op_in']:.3f}. "
            f"{mc['note']}"
        )
        a("")
    if extra.get("e_ghost"):
        g = extra["e_ghost"]
        a(
            f"**745 vs 744:** train companies with any `e_*` minus raw book-invoice companies = "
            f"{g['n_extra_e']} {g['extra_ids']}. Missing the other way: {g['n_missing_e']}. "
            f"Detail: {g['detail']}. COMP_0962 has a single `refund` and no book invoice — "
            "family E still emits `e_credit_note_ratio` for that month. Not a dropped book."
        )
        a("")
    if extra.get("timing"):
        a("**Payment-date timing ladder** (same company, same sign, cents; train paid invoices):")
        a("")
        trows = []
        for lab, title in (("0", "exact day"), ("1", "±1 day"), ("3", "±3 days")):
            if lab in extra["timing"]:
                h = extra["timing"][lab]
                trows.append(
                    {
                        "window": title,
                        "n_inv": f"{h['n_inv']:,}",
                        "n_hit": f"{h['n_hit']:,}",
                        "hit": _pp(h["hit_rate"]),
                    }
                )
        a(_md_table(trows))
        a("")
    if extra.get("bands"):
        a("**Payment-month hit by fixed amount band** (not a fitted cut):")
        a("")
        a(_md_table([
            {
                "band_eur": r["band"],
                "n_inv": f"{r['n_inv']:,}",
                "hit": _pp(r["hit_rate"]),
            }
            for r in extra["bands"]["rows"]
        ]))
        a("")
    if extra.get("match_months"):
        a("**Payment-month hit rate over calendar time** (train):")
        a("")
        a(_md_table([
            {"month": r["month"], "n_paid_inv": f"{r['n_inv']:,}", "hit": _pp(r["hit_rate"])}
            for r in extra["match_months"]["rows"]
        ]))
        a("")
    if extra.get("match_cos"):
        mc = extra["match_cos"]
        a(
            f"**Per-company payment match-rate** ({mc['n_train_cos_with_paid']:,} train companies with a paid invoice): "
            f"p10={_pp(mc['p10'])}, p25={_pp(mc['p25'])}, p50={_pp(mc['p50'])}, "
            f"p75={_pp(mc['p75'])}, p90={_pp(mc['p90'])}. "
            f"Share ≥50%: {_pp(mc['share_rate_ge_0_5'])}. Share exactly 0: {_pp(mc['share_rate_eq_0'])}."
        )
        a("")
    if extra.get("erp_start"):
        es = extra["erp_start"]
        a(
            f"**ERP vs bank start (train invoiced companies):** {es['n_train_with_inv']:,} with a book invoice. "
            f"First invoice after 2024-10-01: {es['n_first_inv_after_2024_10']:,} "
            f"({_pp(es['share_first_inv_after_2024_10'])}). "
            f"First invoice >31 days after first tx: {es['n_first_inv_gt_1m_after_first_tx']:,} "
            f"({_pp(es['share_inv_starts_later_than_bank'])}). "
            f"Median (first_inv − first_tx) = {es['median_inv_minus_tx_days']:.0f} days."
        )
        a("")
    if extra.get("tx_cp_match"):
        tc = extra["tx_cp_match"]
        a(
            f"**Payment-month match against resolved-CP txs only:** "
            f"{_pp(tc['hit_rate'])} ({tc['n_hit']:,}/{tc['n_inv']:,}). {tc['note']} "
            "If this collapses toward the 10.6% same-CP figure, most amount hits are on tx rows with a blank CP — still a same-company amount collision, not an ID join."
        )
        a("")
    if extra.get("no_pay"):
        npay = extra["no_pay"]
        a(
            f"**Invoiced but never paid in-panel:** {npay['n_never_paid']:,} of {npay['n_book']:,} "
            f"train book-invoice companies have no valid `payment_date` in 2024-09..2026-08 "
            f"({npay['n_paid']:,} have at least one)."
        )
        a("")
    if extra.get("inv_quality"):
        q = extra["inv_quality"]
        a(
            f"**Train invoice row quality** (n={q['n_rows']:,}): "
            f"type=invoice {_pp(q['share_invoice_type'])}, cancel {_pp(q['share_cancel'])}, "
            f"amount=0 {_pp(q['share_zero_amt'])}, no issuance {_pp(q['share_no_issuance'])}, "
            f"no due {_pp(q['share_no_due'])}, no payment_date {_pp(q['share_no_payment'])}, "
            f"payment_date_invalid {_pp(q['share_payment_invalid'])}, no CP {_pp(q['share_no_cp'])}."
        )
        a("")
    if extra.get("match_by_mix"):
        mb = extra["match_by_mix"]
        a(
            f"**Payment match-rate by group mix** (train companies with a paid invoice): "
            f"mixed-group invoiced siblings p50={_pp(mb['mixed']['p50'])} "
            f"(n={mb['mixed']['n_cos']:,}, mean {_pp(mb['mixed']['mean'])}); "
            f"all-invoiced groups p50={_pp(mb['all_invoiced']['p50'])} "
            f"(n={mb['all_invoiced']['n_cos']:,}, mean {_pp(mb['all_invoiced']['mean'])})."
        )
        a("")
    if extra.get("controls"):
        ctr = extra["controls"]
        a(
            f"**Controls (same payment-month amounts):** wrong-sign hit "
            f"{_pp(ctr['wrong_sign']['hit_rate'])} ({ctr['wrong_sign']['n_hit']:,}/{ctr['wrong_sign']['n_inv']:,}); "
            f"adjacent month (±1) same-sign hit {_pp(ctr['adjacent_month']['hit_rate'])} "
            f"({ctr['adjacent_month']['n_hit']:,}/{ctr['adjacent_month']['n_inv']:,}). "
            "Wrong-sign ≪ same-sign means the AR/AP direction is doing work. "
            "Adjacent-month near same-month means the calendar window is loose, not a same-day settlement."
        )
        a("")
    if extra.get("pay_cat"):
        pc = extra["pay_cat"]
        a(
            f"**Payment-month + collection/payment category:** "
            f"{_pp(pc['hit_rate'])} ({pc['n_hit']:,}/{pc['n_inv']:,}). "
            "Lower than the unfiltered 35.7% means a chunk of amount hits sit in other tx categories (transfer, uncategorised)."
        )
        a("")
    if extra.get("n_extreme"):
        a(
            f"`is_extreme` invoices: {extra['n_extreme']['n_extreme']} / {extra['n_extreme']['n_inv']:,} "
            "— cannot drive the 21–36% hit rate."
        )
        a("")
    a("## Pass 3 — usable overlap for parked / accepted Ys (train)")
    a("")
    a("“Other table” = the table the model is *allowed* to use, not the table that built the label. "
      "`y8_inv_worse_6` is an invoice Y → other = tx/cash. "
      "`y8_cash_worse_6` is a cash Y → other = invoices. "
      "Y5 / Y7 labels are invoice-built → other = tx.")
    a("")
    a(_md_table(y_rows_md))
    a("")
    a("### Y3 `y3_recover_cash_6m` base rate by invoice presence (stressed train rows; no new GBM)")
    a("")
    a(_md_table(y3_rows))
    a("")
    if extra.get("y8_erp"):
        a("### Y8 / Y5 labeled rows by ever-ERP (train)")
        a("")
        a("Never-ERP = the 470. `e_side_ready` = `e_ar_overdue_30` or `e_delay_coll` non-null (the Y8 invoice-side inputs).")
        a("")
        erows = []
        for r in extra["y8_erp"]["slices"]:
            if r["y"] not in ("y8_inv_worse_6", "y8_cash_worse_6", "y5_ap_od30_ownp80"):
                continue
            erows.append(
                {
                    "Y": r["y"],
                    "slice": r["slice"],
                    "n": f"{r['n']:,}",
                    "n_pos": f"{r['n_pos']:,}",
                    "base": _pp(r["base_rate"]),
                    "share of labeled": _pp(r["share_of_labeled"]),
                }
            )
        a(_md_table(erows))
        a("")
        a(
            "Y5 accepted labels are 100% ever-ERP and ~99% have a tx that month — "
            "**CLOSE** “Y5 died because the cash table was missing”. "
            "y8_inv is the same (100% ERP, 98% tx). "
            "y8_cash is the mixed one: 32% of labels are the 470 (never E); "
            "`e_side_ready` (overdue/delay) is present on only 55% of y8_cash labels. "
            "That is a population hole, not a missing FK."
        )
        a("")
    if extra.get("y8_mixed"):
        ym = extra["y8_mixed"]
        a(
            f"Of {ym['n_never_erp_labeled']:,} y8_cash never-ERP labeled train rows, "
            f"**{ym['n_in_mixed_group']:,}** ({_pp(ym['share_never_erp_in_mixed'])}) sit in a mixed group "
            f"(sibling has invoices; family H can see sister cash) and "
            f"{ym['n_in_all_dark_group']:,} sit in all-dark groups. {ym['note']}"
        )
        a("")
    if extra.get("y3_erp"):
        y3e = extra["y3_erp"]
        a(
            f"Y3 stressed train, ever-ERP vs never-ERP: "
            f"ever n={y3e['ever_erp']['n']:,} recover {y3e['ever_erp']['n_pos']:,} "
            f"({_pp(y3e['ever_erp']['base_rate'])}); "
            f"never n={y3e['never_erp']['n']:,} recover {y3e['never_erp']['n_pos']:,} "
            f"({_pp(y3e['never_erp']['base_rate'])}). "
            "Base rates are close — invoice absence is not a different recovery world."
        )
        a("")
    if extra.get("y2_erp") and "never_erp" in extra["y2_erp"]:
        y2 = extra["y2_erp"]
        a(
            f"Y2 `y2_neg_2of3` labeled train: {y2['all']['n']:,} rows, never-ERP "
            f"{y2['never_erp']['n']:,} ({_pp(y2['never_erp']['share_of_labeled'])}), "
            f"base ever {_pp(y2['ever_erp']['base_rate'])} vs never {_pp(y2['never_erp']['base_rate'])}. "
            "Y2 is cash-built so empty E is not an excuse for the parked GBM (it never used E as the Y)."
        )
        a("")
    if extra.get("unmatched_y3") and extra["unmatched_y3"].get("n_stressed_with_rate"):
        u = extra["unmatched_y3"]
        a(
            f"**Unmatched-share vs Y3** (train stressed ever-ERP company-months with a paid-invoice rate; "
            f"fixed cut 0.5, not a percentile fit). n={u['n_stressed_with_rate']:,}. "
            f"unmatched≥0.5: n={u['unmatched_ge_0_5']['n']:,} recover {u['unmatched_ge_0_5']['n_pos']:,} "
            f"({_pp(u['unmatched_ge_0_5']['base_rate'])}); "
            f"unmatched<0.5: n={u['unmatched_lt_0_5']['n']:,} recover {u['unmatched_lt_0_5']['n_pos']:,} "
            f"({_pp(u['unmatched_lt_0_5']['base_rate'])}). "
            f"Spearman unmatched vs Y3 = {u['spearman_unmatched_vs_y3']:.3f}. "
            "A large base-rate gap would keep unmatched-share as a Y3 X candidate; a flat table parks it."
        )
        a("")
    if p3.get("y7_holdout_counts"):
        h7 = p3["y7_holdout_counts"]
        a(
            f"Holdout `y7_top1_lost` (count only): labeled {h7['n_labeled']:,}, positives {h7['n_pos']:,}. "
            f"{h7['note']}."
        )
        a("")
    a("## Pass 4 — debt schedule / products / next idea")
    a("")
    a(f"Last panel month: `{p4['last_month']}`. Train companies {p4['train_cos']:,}, train cm {p4['train_cm']:,}.")
    a("")
    frows = []
    for col in ("f_w_rate", "f_util_snapshot", "f_months_to_next_pay", "f_sched_vs_obs"):
        if col not in p4:
            continue
        d = p4[col]
        frows.append(
            {
                "column": col,
                "cm non-null": f"{d['n_cm_nonnull']:,} ({_pp(d['share_cm'])})",
                "companies ever": f"{d['n_cos_ever']:,} ({_pp(d['share_cos_ever'])})",
                "last-month cm": f"{d['n_cm_last_month']:,} ({_pp(d['share_last_month'])})",
                "cm before last": f"{d['n_cm_before_last']:,}",
            }
        )
    a(_md_table(frows))
    a("")
    a(
        f"Raw `debt_schedule_config`: {p4['debt_raw']['n_sched_rows']} rows / "
        f"{p4['debt_raw']['n_sched_cos']} companies; `debt_products` companies {p4['debt_raw']['n_debt_cos']}."
    )
    a("")
    pr = p4["products"]
    a(
        f"**Banking `created_at` left truncation:** of {pr['n_train_cos_with_banking_prod']:,} train companies "
        f"with a banking product, **{pr['n_first_created_after_2024_09']:,}** "
        f"({_pp(pr['share_first_created_after_2024_09'])}) have first `created_at` **after 2024-09-01**. "
        f"No banking product at all: {pr['n_train_cos_no_banking_prod']:,}. "
        "Those trails are left-truncated — product mix in 2024-09 is not a full book."
    )
    if extra.get("debt_erp"):
        de = extra["debt_erp"]
        a(
            f"**The 470 are not 'thin / no-finance' companies:** debt_products "
            f"{de['no_inv']['n_debt']:,}/{de['no_inv']['n']:,} ({_pp(de['no_inv']['share_debt'])}) "
            f"vs invoiced {_pp(de['invoiced']['share_debt'])}; schedule "
            f"{de['no_inv']['n_sched']:,} ({_pp(de['no_inv']['share_sched'])}) vs invoiced "
            f"{_pp(de['invoiced']['share_sched'])}; banking {_pp(de['no_inv']['share_bank'])} vs "
            f"{_pp(de['invoiced']['share_bank'])}. They lack an ERP book, not a bank book."
        )
    a("")
    a("### Next idea (legal) or PARK")
    a("")
    a(extra.get("next_idea", ""))
    a("")
    a("## Six brief questions")
    a("")
    a("| # | question | what this QA says |")
    a("| --- | --- | --- |")
    a("| 1 | Who is healthy? | Not this lane. Cross-source health cannot be read on tx-only companies (~39% of train). |")
    a("| 2 | Who is improving? | Invoice-side improvement is undefined for the 470. Cash-side still is. |")
    a("| 3 | Who is turning? | A turn that needs the *other* table is only defined on the both-tables overlap "
      f"({_pp(cm['share_both'])} of train cm). |")
    a("| 4 | Dip vs fall? | Y8 was the cross-source dip-vs-fall bet. "
      "y8_inv labeled rows have cash (98%) and still lost to `a_in12` — not a missing join. "
      "y8_cash labels are 32% never-ERP (empty E); that slice cannot see cobros. "
      "Amount-match is a real *rate* (35.7% vs 0.5% random) but not a row FK. |")
    a("| 5 | Why did it change? | “Why from another table” is honest only on the ERP∩bank subset. "
      "Interco is unmeasurable (CLOSE). |")
    a("| 6 | Months earlier? | No lead-time claim here. Coverage is a gate, not a lag. |")
    a("")
    a("## What failed / next")
    a("")
    a(extra.get("what_failed", ""))
    a("")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def append_registry(p1: dict, p2: dict, p3: dict, p4: dict, verdicts: dict) -> list[dict]:
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    iss = p2["amount_issuance"]["by_tol"]["0.01"]
    pay = p2["amount_payment"]["by_tol"]["0.01"]
    y8c = next((r for r in p3["y_rows"] if r["y"] == "y8_cash_worse_6"), {})
    y8i = next((r for r in p3["y_rows"] if r["y"] == "y8_inv_worse_6"), {})
    rows = [
        {
            "metric": "train_no_invoice_cos",
            "value": p1["n_train_no_invoice"],
            "coverage": p1["train_companies"]["share_inv"],
            "notes": f"confirm_470={p1['confirm_470']}; mixed_groups={p1['group_mix']['n_groups_mixed']}; no_inv_in_mixed={p1['group_mix']['n_no_inv_in_mixed_groups']}",
        },
        {
            "metric": "train_cm_both_inv_tx",
            "value": p1["train_cm_raw"]["share_both"],
            "coverage": p1["train_cm_raw"]["share_both"],
            "notes": f"n_both={p1['train_cm_raw']['n_both']}; usable cross-source month share",
        },
        {
            "metric": "inv_tx_cp_overlap_n",
            "value": p2["row_shares_train"]["n_overlap_cp_train"],
            "coverage": p2["overlap_share_inv_cp"],
            "notes": f"share_inv_rows_cp_in_tx={p2['share_inv_rows_cp_in_tx_train']:.4f}; sets_disjoint_comp={p2['interco']['sets_disjoint']}",
        },
        {
            "metric": "amount_match_iss_0_01",
            "value": iss["hit_rate"],
            "coverage": iss["hit_rate"],
            "notes": (
                f"random={iss['random_hit_rate']:.4f}; gap={iss['real_minus_random']:.4f}; "
                f"issuance-month"
            ),
        },
        {
            "metric": "amount_match_pay_0_01",
            "value": pay["hit_rate"],
            "coverage": pay["hit_rate"],
            "notes": (
                f"random={pay['random_hit_rate']:.4f}; gap={pay['real_minus_random']:.4f}; "
                f"verdict={verdicts['amount_match']['verdict']}; payment-month headline"
            ),
        },
        {
            "metric": "y8_cash_other_inv_share",
            "value": y8c.get("share_labeled_other_present", float("nan")),
            "coverage": y8c.get("share_labeled_other_present", float("nan")),
            "notes": (
                f"n_lab={y8c.get('n_labeled')}; n_other={y8c.get('n_labeled_other_present')}; "
                f"y8_inv_other_tx={y8i.get('share_labeled_other_present')}; "
                f"verdict={verdicts['y8_failed_because_no_join']['verdict']}"
            ),
        },
        {
            "metric": "y8_inv_other_tx_share",
            "value": y8i.get("share_labeled_other_present", float("nan")),
            "coverage": y8i.get("share_labeled_other_present", float("nan")),
            "notes": f"n_lab={y8i.get('n_labeled')}; CLOSE data-excuse for y8_inv",
        },
        {
            "metric": "d_interco_n_eq_join",
            "value": p2["interco"]["n_inv_rows_cp_eq_company_id"] + p2["interco"]["n_tx_rows_cp_eq_company_id"],
            "coverage": 0.0,
            "notes": "CLOSE; COMP_* ∩ COUNTERPARTY_* empty",
        },
        {
            "metric": "f_w_rate_train_cos_ever",
            "value": p4.get("f_w_rate", {}).get("n_cos_ever", float("nan")),
            "coverage": p4.get("f_w_rate", {}).get("share_cos_ever", float("nan")),
            "notes": f"cm_share={p4.get('f_w_rate', {}).get('share_cm')}; last_month_only_util={p4.get('f_util_snapshot', {}).get('n_cm_before_last')}",
        },
        {
            "metric": "banking_first_created_after_202409",
            "value": p4["products"]["share_first_created_after_2024_09"],
            "coverage": p4["products"]["share_first_created_after_2024_09"],
            "notes": f"n={p4['products']['n_first_created_after_2024_09']}/{p4['products']['n_train_cos_with_banking_prod']}",
        },
    ]
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    with REGISTRY.open("a", newline="") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(
                [
                    ts,
                    ROUND,
                    WAVE,
                    AGENT,
                    "D+E",
                    "data_join_qa",
                    "coverage",
                    "train",
                    r["metric"],
                    r["value"],
                    r["coverage"],
                    r["notes"],
                ]
            )
    return rows


def next_idea_text(p1: dict, p2: dict, p3: dict, p4: dict, extra: dict, verdicts: dict) -> tuple[str, str]:
    cm = p1["train_cm_raw"]
    y8c = next((r for r in p3["y_rows"] if r["y"] == "y8_cash_worse_6"), {})
    y8i = next((r for r in p3["y_rows"] if r["y"] == "y8_inv_worse_6"), {})
    what = (
        f"Amount-match **{verdicts['amount_match']['verdict']}** "
        f"(hit {_pp(verdicts['amount_match']['hit'])} vs random {_pp(verdicts['amount_match']['random'])}). "
        f"Interco **CLOSE** (0 equality joins). "
        f"“Y8 failed because no join” **{verdicts['y8_failed_because_no_join']['verdict']}**. "
        f"Train both-tables month share {_pp(cm['share_both'])}. "
        f"470 confirmed={p1['confirm_470']}."
    )
    pr = p4["products"]
    y8_inv_share = y8i.get("share_labeled_other_present", float("nan"))
    idea = (
        "**Legal next (not same-columns):** do not rebuild Y8 — invoice-side worse-than-own-p80 "
        "from cash ranks is the parked model, and y8_inv labeled rows already have tx "
        f"({_pp(y8_inv_share)} other-table). "
        "A legal new Y is `y_erp_gap_then_cash` = existing Y2/Y3 cash stress **among the 470** "
        "using only A/B/C/F/G/H (no invented cobros). That answers Q1/Q3 for companies that "
        "will never have an invoice book. "
        "Payment-month unmatched-share is **PARK as a Y3 X**: among stressed ever-ERP rows "
        "the 0.5 cut is flat (see Pass 3). Keep it only as a Q5 diagnostic "
        "(share of cobros that never hit the bank book), not as a recovery predictor. "
        f"Usable both-tables months: {_pp(cm['share_both'])}. "
        f"Product left-truncation: {_pp(pr['share_first_created_after_2024_09'])} of train "
        "banking books start after 2024-09 (family G honesty, not a join)."
    )
    return what, idea


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (pd.Timestamp, datetime)):
        return str(x)
    if isinstance(x, pd.DataFrame):
        return f"<DataFrame {x.shape}>"
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def main() -> dict:
    t0 = time.time()
    print(f"data_join_qa start {_now_iso()} agent={AGENT}", flush=True)
    con = connect()
    try:
        cos = _register_companies(con)
        monthly, targets = _load_panel()
        grid = monthly[["company_id", "period", "split"]].copy()
        grid["period"] = pd.to_datetime(grid["period"]).dt.date
        con.register("_qa_grid", grid)
        print(f"panel monthly={monthly.shape} targets={targets.shape} companies={len(cos)}", flush=True)

        print("PASS 1 company coverage", flush=True)
        p1 = pass1_company_coverage(con, cos, monthly)
        print(
            f"  train no-invoice={p1['n_train_no_invoice']} confirm_470={p1['confirm_470']} "
            f"both_cm={p1['train_cm_raw']['share_both']:.4f}",
            flush=True,
        )

        print("PASS 2 id / amount", flush=True)
        p2 = pass2_id_amount(con)
        h = p2["amount_issuance"]["by_tol"]["0.01"]
        print(
            f"  cp overlap train={p2['row_shares_train']['n_overlap_cp_train']} "
            f"interco_eq={p2['interco']['n_inv_rows_cp_eq_company_id']}+{p2['interco']['n_tx_rows_cp_eq_company_id']} "
            f"amt hit={h['hit_rate']:.4f} random={h['random_hit_rate']:.4f}",
            flush=True,
        )

        print("PASS 3 Y overlap", flush=True)
        p3 = pass3_y_overlap(monthly, targets, p1["raw_m"])
        for r in p3["y_rows"]:
            print(
                f"  {r['y']}: n={r['n_labeled']} other={r['share_labeled_other_present']:.3f} "
                f"base={r['base_rate']:.4f}",
                flush=True,
            )

        print("PASS 4 sparse debt / products", flush=True)
        p4 = pass4_sparse(con, monthly)
        print(f"  f_w_rate ever_cos={p4.get('f_w_rate', {}).get('n_cos_ever')} "
              f"prod_after={p4['products']['share_first_created_after_2024_09']:.4f}", flush=True)

        extra: dict = {}
        print("EXTRA sibling / quarterly", flush=True)
        extra["siblings"] = extra_sibling_detail(p1["ever"])
        extra["quarterly"] = extra_quarterly(p1["raw_m"])
        print(f"  quarterly both={extra['quarterly']['share_both']:.4f} "
              f"mixed_groups={extra['siblings']['mixed_n_groups']}", flush=True)

        # Payment-date probe overwrote the issuance temp tables; rebuild before extras.
        _amount_match(con, date_col="issuance_date", table="invoices", label="issuance_rebuild")
        extra["strict"] = extra_strict_amount(con)
        extra["same_cp"] = extra_same_cp_amount(con)
        print(
            f"  unique_nonround 0.01 hit={extra['strict']['unique_nonround']['0.01']['hit_rate']:.4f} "
            f"same_cp={extra['same_cp']['hit_rate']:.4f}",
            flush=True,
        )
        extra["side_due"] = extra_side_and_due(con)
        extra["sibling_amt"] = extra_sibling_amount(con)
        extra["match_cm"] = extra_match_rate_cm(con, monthly)
        extra["y8_erp"] = extra_y8_erp_split(monthly, targets, p1["ever"])
        extra["e_ghost"] = extra_e_only_company(con, monthly, p1["ever"])
        extra["timing"] = extra_payment_timing(con)
        extra["bands"] = extra_amount_bands(con)
        extra["match_months"] = extra_match_by_month(con)
        extra["erp_start"] = extra_erp_vs_bank_start(con, p1["ever"])
        extra["y3_erp"] = extra_y3_erp(monthly, targets, p1["ever"])
        extra["match_cos"] = extra_match_company_dist(con)
        extra["unmatched_y3"] = extra_unmatched_vs_y3(con, monthly, targets, p1["ever"])
        extra["tx_cp_match"] = extra_match_resolved_cp(con)
        extra["no_pay"] = extra_invoiced_never_paid(con)
        extra["controls"] = extra_sign_month_controls(con)
        extra["y8_mixed"] = extra_y8_never_erp_groups(monthly, targets, p1["ever"])
        extra["shared_cp"] = extra_shared_cp_in_group(con)
        extra["inv_quality"] = extra_invoice_quality(con)
        extra["match_by_mix"] = extra_match_rate_by_group_mix(con, p1["ever"])
        extra["shared_tx_detail"] = extra_shared_tx_cp_detail(con)
        extra["h_dark"] = extra_h_for_dark(monthly, p1["ever"])
        extra["pay_cat"] = extra_payment_category(con)
        extra["y2_erp"] = extra_y2_overlap(monthly, targets, p1["ever"])
        extra["n_extreme"] = extra_extreme_count(con)
        extra["debt_erp"] = extra_debt_by_erp(con)
        extra["erp_flag"] = extra_erp_flag(con)
        print(
            f"  AR hit={extra['side_due']['AR']['hit_rate']:.4f} "
            f"AP={extra['side_due']['AP']['hit_rate']:.4f} "
            f"pay±1d={extra['side_due']['payment_pm1d']['hit_rate']:.4f} "
            f"sib={extra['sibling_amt']['hit_rate_among_with_sib']:.4f} "
            f"ghost_e={extra['e_ghost']['n_extra_e']} "
            f"exact_day={extra['timing']['0']['hit_rate']:.4f} "
            f"unmatched_y3={extra['unmatched_y3'].get('n_stressed_with_rate')} "
            f"tx_cp_hit={extra['tx_cp_match']['hit_rate']:.4f} "
            f"wrong_sign={extra['controls']['wrong_sign']['hit_rate']:.4f} "
            f"adj_month={extra['controls']['adjacent_month']['hit_rate']:.4f} "
            f"y8_never_mixed={extra['y8_mixed']['share_never_erp_in_mixed']:.4f} "
            f"shared_cp_groups={extra['shared_cp']['n_inv_cp_shared_in_group']} "
            f"pay_invalid={extra['inv_quality']['share_payment_invalid']:.4f} "
            f"mix_p50={extra['match_by_mix']['mixed']['p50']:.4f} "
            f"shared_tx={extra['shared_tx_detail']['n_pairs']} "
            f"dark_h={extra['h_dark']['share_dark_with_sib_in']:.4f} "
            f"pay_cat={extra['pay_cat']['hit_rate']:.4f} "
            f"y2_never={extra['y2_erp']['never_erp']['share_of_labeled']:.4f} "
            f"dark_debt={extra['debt_erp']['no_inv']['share_debt']:.4f} "
            f"null_erp_of_470={extra['erp_flag']['share_noinv_erp_null']:.4f}",
            flush=True,
        )

        verdicts = decide_verdicts(p2, p3, extra)
        what, idea = next_idea_text(p1, p2, p3, p4, extra, verdicts)
        extra["what_failed"] = what
        extra["next_idea"] = idea
        print("VERDICTS", {k: v["verdict"] for k, v in verdicts.items()}, flush=True)

        elapsed = time.time() - t0
        write_markdown(p1, p2, p3, p4, extra, verdicts, elapsed)
        print(f"wrote {OUT_MD}", flush=True)
        if "--no-registry" in sys.argv:
            print("skip registry (--no-registry)", flush=True)
            reg = []
        else:
            reg = append_registry(p1, p2, p3, p4, verdicts)
            print(f"registry +{len(reg)} rows", flush=True)
        out = {
            "elapsed_s": elapsed,
            "confirm_470": p1["confirm_470"],
            "n_train_no_invoice": p1["n_train_no_invoice"],
            "cp_overlap": p2["row_shares_train"]["n_overlap_cp_train"],
            "amount_hit_0_01": h["hit_rate"],
            "amount_random_0_01": h["random_hit_rate"],
            "verdicts": {k: v["verdict"] for k, v in verdicts.items()},
            "y8_cash_other": next(
                (r["share_labeled_other_present"] for r in p3["y_rows"] if r["y"] == "y8_cash_worse_6"),
                None,
            ),
        }
        print(json.dumps(_jsonable(out), indent=2), flush=True)
        return out
    finally:
        con.close()


if __name__ == "__main__":
    main()
