"""Clean-flag DQ QA — prevalence, amount mass, and whether flags are health.

Doubtful rows are kept and flagged in `clean` (analysis/README.md). This
module measures those flags. It does not drop rows, rewrite parquet, write
a 0–100, invent a Y assembler, or edit family modules.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.clean_flags_qa

Owned: analysis/evaluate/clean_flags_qa.py, analysis/outputs/clean_flags_qa.md,
optional PNG, registry appends, overnight/waves/wave4_clean_flags.md (end).
Read-only: duckdb `clean` via connect, monthly.parquet, targets.parquet.

Holdout 72 is coverage only. Rates / AUROC / KEEP-PARK-CLOSE on **train**.
Seed 20260918 group folds. No build_targets. No new GBM.

Iteration log (same module, not one-shot):
1. Prevalence + amount mass + calendar + extreme row cards
2. Y overlap on company-months + size control
3. is_dup clones vs same-day-same-amount + SIZE
4. product_known vs Family G first-month hole
5. balance_sentinel vs Family B cash walk
6. Debt exclude-extreme vs cashflow keep-extreme (f_ds_r / a_op_in)
7. Quoted singles if extremes dropped (days 0.711 / issued_lag1 0.630 / HHI 0.605)
8. Canceling extreme pairs + PDI July pile + B snapshot gap
9. HHI window overlap; 4997-clone whale; soft not-clone anatomy
10. Why f_ds_r is unchanged; PDI-as-Y gate (design note only)
11–16. Residual PDI as X; drop-dup / drop-PDI quotes; leftover 2.8%
17–22. Unk-product ≠ G; post-snapshot txs; all 24 extremes on checking; B-walk shift
23–33. Y2 0/67 on 4 train extreme-checking cos; 11/18 flips on COMP_0306;
       same-mag inv↔tx only that company (226d); GROUP_0199 holdout same two days;
       GROUP_0094 train siblings COMP_0306 + COMP_1192
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
from analysis.features.common import ANALYSIS, CAT_MAP, DATA, LAST_M, MONTHS, connect

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

STORE = DATA / "feature_store" / "monthly.parquet"
TARGETS = DATA / "feature_store" / "targets.parquet"
OUT_MD = ANALYSIS / "outputs" / "clean_flags_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "clean_flags_prevalence.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
AGENT = "d7235c84"
WAVE = "4"
ROUND = "R4"
N_FOLDS = 5
MIN_POS = 50

PANEL_START = pd.Timestamp("2024-09-01")
PANEL_END = pd.Timestamp("2026-08-01")
EXTRACT = pd.Timestamp("2026-09-01")

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
Y4 = "y4_ds_r_double"
Y7 = "y7_top1_lost"
Y9 = "y9_fee_r_ownp80"
Y_ACCEPTED = [Y2, Y3, Y4, Y7, Y9]

QUOTE_DAYS = 0.711
QUOTE_ISSUED = 0.630
QUOTE_HHI = 0.605
MOVE = 0.02

CASH_TYPES = ("checking", "saving", "tpv")
BOOK = (
    "document_type = 'invoice' AND status <> 'cancel' "
    "AND amount <> 0 AND issuance_date IS NOT NULL"
)


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


def _d(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    try:
        ts = pd.Timestamp(x)
        if pd.isna(ts):
            return ""
        return str(ts.date())
    except (TypeError, ValueError):
        return str(x)


def _sci(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    ax = abs(float(x))
    if ax >= 1e6 or (ax > 0 and ax < 1e-2):
        return f"{float(x):.3e}"
    return f"{float(x):,.2f}"


def md_table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(h for h, _ in cols) + " |"
    sep = "| " + " | ".join("---" if a != "right" else "---:" for _, a in cols) + " |"
    body = []
    for r in rows:
        cells = []
        for h, align in cols:
            v = r.get(h, "")
            cells.append(str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *body])


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


def signed_oof_auroc(y: pd.Series, x: pd.Series, folds: pd.Series, mask: pd.Series) -> dict:
    y = pd.to_numeric(y, errors="coerce")
    x = pd.to_numeric(x, errors="coerce")
    defined = mask & y.notna() & x.notna()
    n = int(defined.sum())
    n_pos = int((defined & (y == 1)).sum())
    n_neg = int((defined & (y == 0)).sum())
    if n_pos < MIN_POS or n_neg == 0:
        return {
            "cv": float("nan"),
            "sd": float("nan"),
            "n": n,
            "n_pos": n_pos,
            "low_power": True,
        }
    aucs = []
    for k in range(N_FOLDS):
        tr = defined & (folds != k)
        va = defined & (folds == k)
        sign = choose_sign(y[tr], x[tr])
        aucs.append(auroc(y[va], sign * x[va]))
    finite = [a for a in aucs if np.isfinite(a)]
    return {
        "cv": float(np.mean(finite)) if finite else float("nan"),
        "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan"),
        "n": n,
        "n_pos": n_pos,
        "low_power": False,
    }


def load_panel() -> pd.DataFrame:
    store = pd.read_parquet(STORE)
    tgt = pd.read_parquet(TARGETS)
    store["company_id"] = store["company_id"].astype(str)
    tgt["company_id"] = tgt["company_id"].astype(str)
    store["period"] = pd.to_datetime(store["period"]).dt.normalize()
    tgt["period"] = pd.to_datetime(tgt["period"]).dt.normalize()
    keep_s = [
        "company_id",
        "period",
        "a_op_in",
        "a_in3",
        "a_n_tx",
        "c_n_days_with_tx",
        "e_ar_issued",
        "d_cust_hhi",
        "f_ds_r",
        "g_n_accounts",
        "b_liq",
    ]
    keep_s = [c for c in keep_s if c in store.columns]
    ykeep = ["company_id", "period"] + [c for c in tgt.columns if c.startswith("y")]
    panel = store[keep_s].merge(tgt[ykeep], on=["company_id", "period"], how="left")
    hold = load_holdout()
    panel["split"] = np.where(panel["company_id"].isin(hold), "holdout", "train")
    return panel


def attach_groups(con, panel: pd.DataFrame) -> pd.DataFrame:
    cos = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               CAST(group_id AS VARCHAR) AS group_id
        FROM companies
        """
    ).df()
    out = panel.merge(cos, on="company_id", how="left")
    train = out.loc[out["split"] == "train", ["company_id", "group_id"]].drop_duplicates()
    assert_no_holdout(train["company_id"])
    folds = group_folds(train, n=N_FOLDS, seed=FOLD_SEED)
    out = out.merge(folds[["company_id", "fold"]], on="company_id", how="left")
    return out


def pass1_prevalence(con, hold: set[str]) -> dict:
    """n rows, n companies, amount mass — train vs holdout — every listed flag."""
    hold_df = pd.DataFrame({"company_id": sorted(hold)})
    try:
        con.unregister("_hold")
    except Exception:
        pass
    con.register("_hold", hold_df)

    specs = [
        (
            "transactions",
            "is_extreme",
            "is_extreme",
            "abs(amount)",
            "TRUE",
        ),
        (
            "transactions",
            "is_dup",
            "is_dup",
            "abs(amount)",
            "TRUE",
        ),
        (
            "transactions",
            "product_known=false",
            "NOT product_known",
            "abs(amount)",
            "TRUE",
        ),
        (
            "invoices",
            "is_extreme",
            "is_extreme",
            "abs(amount)",
            "TRUE",
        ),
        (
            "invoices",
            "payment_date_invalid",
            "payment_date_invalid",
            "abs(amount)",
            "TRUE",
        ),
        (
            "balances",
            "balance_sentinel",
            "balance_sentinel",
            "abs(coalesce(balance, 0))",
            "TRUE",
        ),
        (
            "balances",
            "product_known=false",
            "NOT product_known",
            "abs(coalesce(balance, 0))",
            "TRUE",
        ),
        (
            "banking_products",
            "created_after_snapshot",
            "created_after_snapshot",
            "1",
            "TRUE",
        ),
        (
            "debt_products",
            "created_after_snapshot",
            "created_after_snapshot",
            "1",
            "TRUE",
        ),
        (
            "debt_products",
            "outstanding_gt_granted",
            "outstanding_gt_granted",
            "abs(coalesce(outstanding, 0))",
            "TRUE",
        ),
        (
            "debt_schedule_config",
            "outstanding_gt_granted",
            "outstanding_gt_granted",
            "abs(coalesce(outstanding_balance, 0))",
            "TRUE",
        ),
    ]
    rows = []
    for table, flag, pred, mass_expr, _ in specs:
        q = f"""
        WITH base AS (
          SELECT CAST(company_id AS VARCHAR) AS company_id,
                 ({pred}) AS flagged,
                 {mass_expr} AS mass
          FROM {table}
        )
        SELECT
          CASE WHEN h.company_id IS NOT NULL THEN 'holdout' ELSE 'train' END AS split,
          count(*) AS n_rows,
          count(*) FILTER (WHERE flagged) AS n_flag,
          count(DISTINCT base.company_id) AS n_cos,
          count(DISTINCT base.company_id) FILTER (WHERE flagged) AS n_cos_flag,
          coalesce(sum(mass), 0) AS mass_all,
          coalesce(sum(mass) FILTER (WHERE flagged), 0) AS mass_flag
        FROM base
        LEFT JOIN _hold h ON base.company_id = h.company_id
        GROUP BY 1
        """
        df = con.execute(q).df()
        for rec in df.to_dict("records"):
            rows.append(
                {
                    "table": table,
                    "flag": flag,
                    "split": rec["split"],
                    "n_rows": int(rec["n_rows"]),
                    "n_flag": int(rec["n_flag"]),
                    "n_cos": int(rec["n_cos"]),
                    "n_cos_flag": int(rec["n_cos_flag"]),
                    "row_share": _pct(rec["n_flag"], rec["n_rows"]),
                    "mass_all": float(rec["mass_all"]),
                    "mass_flag": float(rec["mass_flag"]),
                    "mass_share": _pct(rec["mass_flag"], rec["mass_all"]),
                }
            )
        tot = con.execute(
            f"""
            SELECT count(*) AS n_rows,
                   count(*) FILTER (WHERE {pred}) AS n_flag,
                   count(DISTINCT company_id) AS n_cos,
                   count(DISTINCT company_id) FILTER (WHERE {pred}) AS n_cos_flag,
                   coalesce(sum({mass_expr}), 0) AS mass_all,
                   coalesce(sum({mass_expr}) FILTER (WHERE {pred}), 0) AS mass_flag
            FROM {table}
            """
        ).fetchdf()
        r = tot.iloc[0]
        rows.append(
            {
                "table": table,
                "flag": flag,
                "split": "all",
                "n_rows": int(r["n_rows"]),
                "n_flag": int(r["n_flag"]),
                "n_cos": int(r["n_cos"]),
                "n_cos_flag": int(r["n_cos_flag"]),
                "row_share": _pct(r["n_flag"], r["n_rows"]),
                "mass_all": float(r["mass_all"]),
                "mass_flag": float(r["mass_flag"]),
                "mass_share": _pct(r["mass_flag"], r["mass_all"]),
            }
        )

    # Sentinel original balances live in main (clean nulls them).
    sent_raw = con.execute(
        """
        SELECT CAST(b.company_id AS VARCHAR) AS company_id,
               b.product_id, b.balance,
               CASE WHEN h.company_id IS NOT NULL THEN 'holdout' ELSE 'train' END AS split
        FROM main.balances b
        LEFT JOIN _hold h ON CAST(b.company_id AS VARCHAR) = h.company_id
        WHERE b.balance IN (-999999999, -1000000000) OR abs(b.balance) >= 1e10
        """
    ).df()

    print("pass1 prevalence")
    for r in rows:
        if r["split"] != "all":
            continue
        print(
            f"  {r['table']:22s} {r['flag']:26s} "
            f"n={r['n_flag']:,}/{r['n_rows']:,} ({_pp(r['row_share'])}) "
            f"cos={r['n_cos_flag']}/{r['n_cos']} mass={_pp(r['mass_share'])}"
        )
    return {"rows": rows, "sent_raw": sent_raw}


def pass2_calendar(con, hold: set[str]) -> dict:
    """One-month pile vs spread, for row-level flags with a date."""
    specs = [
        ("transactions", "is_extreme", "is_extreme", "date", "abs(amount)"),
        ("transactions", "is_dup", "is_dup", "date", "abs(amount)"),
        ("transactions", "product_known=false", "NOT product_known", "date", "abs(amount)"),
        ("invoices", "is_extreme", "is_extreme", "issuance_date", "abs(amount)"),
        ("invoices", "payment_date_invalid", "payment_date_invalid", "issuance_date", "abs(amount)"),
    ]
    out = []
    monthly = {}
    hold_df = pd.DataFrame({"company_id": sorted(hold)})
    try:
        con.unregister("_hold")
    except Exception:
        pass
    con.register("_hold", hold_df)
    for table, flag, pred, date_col, mass_expr in specs:
        df = con.execute(
            f"""
            SELECT CAST(date_trunc('month', {date_col}) AS DATE) AS month,
                   CASE WHEN h.company_id IS NOT NULL THEN 'holdout' ELSE 'train' END AS split,
                   count(*) AS n_flag,
                   coalesce(sum({mass_expr}), 0) AS mass
            FROM {table} t
            LEFT JOIN _hold h ON CAST(t.company_id AS VARCHAR) = h.company_id
            WHERE {pred} AND {date_col} IS NOT NULL
            GROUP BY 1, 2
            ORDER BY 1, 2
            """
        ).df()
        monthly[(table, flag)] = df
        for split, g in [("train", df[df["split"] == "train"]), ("holdout", df[df["split"] == "holdout"]), ("all", df)]:
            if g.empty:
                out.append(
                    {
                        "table": table,
                        "flag": flag,
                        "split": split,
                        "n_months": 0,
                        "n_flag": 0,
                        "max_month": "",
                        "max_row_share": float("nan"),
                        "max_mass_share": float("nan"),
                    }
                )
                continue
            agg = g.groupby("month", as_index=False).agg(n_flag=("n_flag", "sum"), mass=("mass", "sum"))
            tot_n = float(agg["n_flag"].sum())
            tot_m = float(agg["mass"].sum())
            i = int(agg["n_flag"].idxmax())
            j = int(agg["mass"].idxmax())
            out.append(
                {
                    "table": table,
                    "flag": flag,
                    "split": split,
                    "n_months": int(agg.shape[0]),
                    "n_flag": int(tot_n),
                    "max_month": str(pd.to_datetime(agg.loc[i, "month"]).date()),
                    "max_row_share": _pct(agg.loc[i, "n_flag"], tot_n),
                    "max_mass_share": _pct(agg.loc[j, "mass"], tot_m),
                    "max_mass_month": str(pd.to_datetime(agg.loc[j, "month"]).date()),
                }
            )
    print("pass2 calendar")
    for r in out:
        if r["split"] != "train":
            continue
        print(
            f"  {r['table']:12s} {r['flag']:22s} months={r['n_months']} "
            f"max {r['max_month']} rows={_pp(r['max_row_share'])} "
            f"mass {r.get('max_mass_month','')}={_pp(r['max_mass_share'])}"
        )
    return {"rows": out, "monthly": monthly}


def pass3_extreme_cards(con, hold: set[str]) -> dict:
    """The 10 invoices and 24 txs — who, when, train vs holdout, category."""
    inv = con.execute(
        f"""
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               operation_id, document_type, status,
               CAST(issuance_date AS DATE) AS issuance_date,
               amount, counterparty_id, payment_date_invalid
        FROM invoices
        WHERE is_extreme
        ORDER BY abs(amount) DESC
        """
    ).df()
    inv["split"] = np.where(inv["company_id"].isin(hold), "holdout", "train")
    tx = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               transaction_id, product_id,
               CAST("date" AS DATE) AS date,
               amount, category, status, is_dup, product_known
        FROM transactions
        WHERE is_extreme
        ORDER BY abs(amount) DESC
        """
    ).df()
    tx["split"] = np.where(tx["company_id"].isin(hold), "holdout", "train")
    tx["grp"] = tx["category"].map(CAT_MAP).fillna("other")

    print("pass3 extreme cards")
    print(
        f"  invoices {len(inv)} train={int((inv.split=='train').sum())} "
        f"holdout={int((inv.split=='holdout').sum())} "
        f"cos_train={inv.loc[inv.split=='train','company_id'].nunique()} "
        f"cos_hold={inv.loc[inv.split=='holdout','company_id'].nunique()}"
    )
    print(
        f"  txs {len(tx)} train={int((tx.split=='train').sum())} "
        f"holdout={int((tx.split=='holdout').sum())} "
        f"grp={tx.grp.value_counts().to_dict()}"
    )
    return {"inv": inv, "tx": tx}


def pass4_y_overlap(con, panel: pd.DataFrame) -> dict:
    """Accepted-Y base rates on flagged vs not company-months. Size = log1p(a_in3)."""
    hold = set(panel.loc[panel["split"] == "holdout", "company_id"])
    hold_df = pd.DataFrame({"company_id": sorted(hold)})
    try:
        con.unregister("_hold")
    except Exception:
        pass
    con.register("_hold", hold_df)

    tx_cm = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               CAST(date_trunc('month', "date") AS DATE) AS period,
               MAX(CASE WHEN is_extreme THEN 1 ELSE 0 END) AS has_tx_extreme,
               MAX(CASE WHEN is_dup THEN 1 ELSE 0 END) AS has_tx_dup,
               MAX(CASE WHEN NOT product_known THEN 1 ELSE 0 END) AS has_tx_unkprod,
               SUM(CASE WHEN is_extreme THEN 1 ELSE 0 END) AS n_tx_extreme,
               SUM(CASE WHEN is_dup THEN 1 ELSE 0 END) AS n_tx_dup
        FROM transactions
        WHERE "date" IS NOT NULL
          AND CAST("date" AS DATE) >= DATE '2024-09-01'
          AND CAST("date" AS DATE) < DATE '2026-09-01'
        GROUP BY 1, 2
        """
    ).df()
    inv_cm = con.execute(
        f"""
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               CAST(date_trunc('month', issuance_date) AS DATE) AS period,
               MAX(CASE WHEN is_extreme THEN 1 ELSE 0 END) AS has_inv_extreme,
               MAX(CASE WHEN payment_date_invalid THEN 1 ELSE 0 END) AS has_pdi,
               SUM(CASE WHEN is_extreme THEN 1 ELSE 0 END) AS n_inv_extreme,
               SUM(CASE WHEN payment_date_invalid THEN 1 ELSE 0 END) AS n_pdi
        FROM invoices
        WHERE {BOOK}
        GROUP BY 1, 2
        """
    ).df()
    tx_cm["period"] = pd.to_datetime(tx_cm["period"])
    inv_cm["period"] = pd.to_datetime(inv_cm["period"])
    p = panel.copy()
    p = p.merge(tx_cm, on=["company_id", "period"], how="left")
    p = p.merge(inv_cm, on=["company_id", "period"], how="left")
    flag_cols = [
        "has_tx_extreme",
        "has_tx_dup",
        "has_tx_unkprod",
        "has_inv_extreme",
        "has_pdi",
    ]
    for c in flag_cols:
        p[c] = p[c].fillna(0).astype(int)
    p["log_in3"] = np.log1p(pd.to_numeric(p["a_in3"], errors="coerce").clip(lower=0))

    # Company-level snapshot flags → every panel month of that company.
    snap = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               MAX(CASE WHEN balance_sentinel THEN 1 ELSE 0 END) AS ever_sentinel,
               MAX(CASE WHEN NOT product_known THEN 1 ELSE 0 END) AS ever_bal_unk
        FROM balances
        GROUP BY 1
        """
    ).df()
    debtf = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               MAX(CASE WHEN outstanding_gt_granted THEN 1 ELSE 0 END) AS ever_ogtg,
               MAX(CASE WHEN created_after_snapshot THEN 1 ELSE 0 END) AS ever_debt_after
        FROM debt_products
        GROUP BY 1
        """
    ).df()
    bankf = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               MAX(CASE WHEN created_after_snapshot THEN 1 ELSE 0 END) AS ever_bank_after
        FROM banking_products
        GROUP BY 1
        """
    ).df()
    p = p.merge(snap, on="company_id", how="left")
    p = p.merge(debtf, on="company_id", how="left")
    p = p.merge(bankf, on="company_id", how="left")
    for c in ["ever_sentinel", "ever_bal_unk", "ever_ogtg", "ever_debt_after", "ever_bank_after"]:
        p[c] = p[c].fillna(0).astype(int)

    cm_flags = flag_cols + [
        "ever_sentinel",
        "ever_ogtg",
        "ever_bank_after",
        "ever_debt_after",
    ]
    tr = p[p["split"] == "train"].copy()
    rows = []
    for flag in cm_flags:
        on = tr[flag] == 1
        off = tr[flag] == 0
        rec = {
            "flag": flag,
            "n_on": int(on.sum()),
            "n_off": int(off.sum()),
            "share_cm": _pct(on.sum(), len(tr)),
            "n_cos_on": int(tr.loc[on, "company_id"].nunique()),
        }
        for y in Y_ACCEPTED:
            yy = pd.to_numeric(tr[y], errors="coerce")
            lab_on = on & yy.notna()
            lab_off = off & yy.notna()
            rec[f"{y}_n_on"] = int(lab_on.sum())
            rec[f"{y}_rate_on"] = float(yy[lab_on].mean()) if lab_on.any() else float("nan")
            rec[f"{y}_rate_off"] = float(yy[lab_off].mean()) if lab_off.any() else float("nan")
        size_on = tr.loc[on, "log_in3"]
        size_off = tr.loc[off, "log_in3"]
        rec["log_in3_on"] = float(size_on.mean()) if size_on.notna().any() else float("nan")
        rec["log_in3_off"] = float(size_off.mean()) if size_off.notna().any() else float("nan")
        size_auc = auroc((tr[flag] == 1).astype(float), tr["log_in3"])
        rec["size_auc"] = float(size_auc)
        rows.append(rec)
        print(
            f"  {flag:22s} cm={rec['n_on']:,} ({_pp(rec['share_cm'])}) "
            f"size_auc={_f(rec['size_auc'])} "
            f"y3_on={_pp(rec[f'{Y3}_rate_on'])} y3_off={_pp(rec[f'{Y3}_rate_off'])}"
        )
    return {"rows": rows, "panel": p}


def pass5_dup(con, panel: pd.DataFrame, hold: set[str]) -> dict:
    """Extract clones (full-content) vs same-day same-amount soft pairs."""
    clones = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               product_id, CAST("date" AS DATE) AS date,
               amount, category, status,
               count(*) AS n_copies,
               sum(CASE WHEN is_dup THEN 1 ELSE 0 END) AS n_flagged
        FROM transactions
        GROUP BY company_id, product_id, "date", value_date, amount, category,
                 description, counterparty_id, status
        HAVING count(*) > 1
        """
    ).df()
    clones["split"] = np.where(clones["company_id"].isin(hold), "holdout", "train")
    soft = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               CAST("date" AS DATE) AS date,
               amount,
               count(*) AS n,
               count(DISTINCT transaction_id) AS n_id,
               count(DISTINCT status) AS n_status,
               count(DISTINCT product_id) AS n_prod,
               count(DISTINCT coalesce(description, '')) AS n_desc,
               sum(CASE WHEN is_dup THEN 1 ELSE 0 END) AS n_is_dup
        FROM transactions
        GROUP BY 1, 2, 3
        HAVING count(*) > 1
        """
    ).df()
    soft["split"] = np.where(soft["company_id"].isin(hold), "holdout", "train")
    # Soft pairs that are NOT full-content clones (n_is_dup = 0).
    soft_not_clone = soft[soft["n_is_dup"] == 0]
    cat = (
        clones[clones["split"] == "train"]
        .groupby("category", as_index=False)
        .agg(n_groups=("n_copies", "size"), n_extra=("n_flagged", "sum"), mass=("amount", lambda s: s.abs().sum()))
        .sort_values("n_extra", ascending=False)
    )

    tr = panel[panel["split"] == "train"].copy()
    if "has_tx_dup" not in tr.columns:
        # filled in pass4 panel; if called standalone, skip size
        size_auc = float("nan")
        base = float("nan")
    else:
        size_auc = float(auroc((tr["has_tx_dup"] == 1).astype(float), tr["log_in3"]))
        base = float((tr["has_tx_dup"] == 1).mean())

    print("pass5 dup")
    print(
        f"  clone groups={len(clones):,} train={int((clones.split=='train').sum()):,} "
        f"extra_flagged={int(clones.n_flagged.sum()):,} "
        f"mult p50={clones.n_copies.median():.1f} max={int(clones.n_copies.max())}"
    )
    print(
        f"  soft same-day-amt groups={len(soft):,} "
        f"not_clone={len(soft_not_clone):,} "
        f"multi_status={int((soft.n_status>1).sum()):,}"
    )
    print(f"  has_tx_dup CM base={_pp(base)} size_auc={_f(size_auc)}")
    return {
        "clones": clones,
        "soft": soft,
        "soft_not_clone": soft_not_clone,
        "cat": cat,
        "size_auc": size_auc,
        "cm_base": base,
    }


def pass6_product_vs_g(con, panel: pd.DataFrame) -> dict:
    """Is product_known=false the 63.7% first-month G hole, or a different hole?"""
    unk = con.execute(
        """
        SELECT CAST(t.company_id AS VARCHAR) AS company_id,
               t.product_id,
               CAST(date_trunc('month', t."date") AS DATE) AS month,
               count(*) AS n,
               sum(abs(t.amount)) AS mass
        FROM transactions t
        WHERE NOT t.product_known
        GROUP BY 1, 2, 3
        """
    ).df()
    unk["month"] = pd.to_datetime(unk["month"])
    hold = set(panel.loc[panel["split"] == "holdout", "company_id"])
    unk["split"] = np.where(unk["company_id"].isin(hold), "holdout", "train")

    # First panel month per company + g_n_accounts.
    first = (
        panel.sort_values(["company_id", "period"])
        .groupby("company_id", as_index=False)
        .first()[["company_id", "period", "g_n_accounts", "split"]]
    )
    first["g0"] = first["g_n_accounts"].fillna(0).eq(0)
    tr_first = first[first["split"] == "train"]
    g0_share = float(tr_first["g0"].mean())

    # Tx in first month: product_known share among g0 vs g>0.
    first_tx = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               CAST(date_trunc('month', "date") AS DATE) AS month,
               count(*) AS n_tx,
               sum(CASE WHEN NOT product_known THEN 1 ELSE 0 END) AS n_unk,
               sum(abs(amount)) AS mass,
               sum(CASE WHEN NOT product_known THEN abs(amount) ELSE 0 END) AS mass_unk
        FROM transactions
        WHERE "date" IS NOT NULL
        GROUP BY 1, 2
        """
    ).df()
    first_tx["month"] = pd.to_datetime(first_tx["month"])
    ft = first.merge(first_tx, left_on=["company_id", "period"], right_on=["company_id", "month"], how="left")
    ft_tr = ft[ft["split"] == "train"]
    g0 = ft_tr[ft_tr["g0"]]
    g1 = ft_tr[~ft_tr["g0"]]

    # Unknown product_ids — exist in banking/debt? (should be none)
    in_books = con.execute(
        """
        SELECT count(DISTINCT t.product_id) AS n_unk_prod,
               sum(CASE WHEN b.product_id IS NOT NULL OR d.product_id IS NOT NULL THEN 1 ELSE 0 END) AS n_in_books
        FROM (SELECT DISTINCT product_id FROM transactions WHERE NOT product_known) t
        LEFT JOIN banking_products b ON t.product_id = b.product_id
        LEFT JOIN debt_products d ON t.product_id = d.product_id
        """
    ).fetchone()

    # Unknown-product months vs g_n_accounts=0 months on the panel.
    p = panel.copy()
    unk_cm = (
        unk.groupby(["company_id", "month"], as_index=False)["n"]
        .sum()
        .rename(columns={"month": "period", "n": "n_unk_tx"})
    )
    p = p.merge(unk_cm, on=["company_id", "period"], how="left")
    p["n_unk_tx"] = p["n_unk_tx"].fillna(0)
    tr = p[p["split"] == "train"]
    g0_cm = tr["g_n_accounts"].fillna(0).eq(0)
    unk_cm_m = tr["n_unk_tx"] > 0
    both = int((g0_cm & unk_cm_m).sum())

    print("pass6 product vs G")
    print(f"  first-month g_n_accounts=0 train={_pp(g0_share)} (G quote 63.7%)")
    print(
        f"  first-month unk row share g0={_pp(_pct(g0['n_unk'].sum(), g0['n_tx'].sum()))} "
        f"g>0={_pp(_pct(g1['n_unk'].sum(), g1['n_tx'].sum()))}"
    )
    print(f"  unk products={in_books[0]} in_books={in_books[1]}")
    print(
        f"  train CM g0={int(g0_cm.sum()):,} unk={int(unk_cm_m.sum()):,} both={both} "
        f"unk_among_g0={_pp(_pct(both, int(g0_cm.sum())))}"
    )
    return {
        "unk": unk,
        "g0_share": g0_share,
        "first_unk_g0": _pct(g0["n_unk"].sum(), g0["n_tx"].sum()),
        "first_unk_g1": _pct(g1["n_unk"].sum(), g1["n_tx"].sum()),
        "n_unk_prod": int(in_books[0]),
        "n_in_books": int(in_books[1]),
        "g0_cm": int(g0_cm.sum()),
        "unk_cm": int(unk_cm_m.sum()),
        "both_cm": both,
        "ft": ft_tr,
    }


def pass7_sentinel(con, panel: pd.DataFrame, hold: set[str]) -> dict:
    """Do sentinels sit on checking/saving/tpv (Family B walk)?"""
    rows = con.execute(
        f"""
        SELECT CAST(b.company_id AS VARCHAR) AS company_id,
               b.product_id,
               b.balance AS clean_balance,
               b.balance_sentinel,
               b.product_known,
               p.type AS bank_type,
               d.type AS debt_type,
               m.balance AS raw_balance
        FROM balances b
        LEFT JOIN banking_products p ON b.product_id = p.product_id
        LEFT JOIN debt_products d ON b.product_id = d.product_id
        LEFT JOIN main.balances m ON b.product_id = m.product_id AND b.company_id = m.company_id
        WHERE b.balance_sentinel
        """
    ).df()
    rows["split"] = np.where(rows["company_id"].isin(hold), "holdout", "train")
    rows["cash_type"] = rows["bank_type"].isin(CASH_TYPES)

    # Family B snapshot: cash types with NOT NULL balance.
    cash = con.execute(
        f"""
        SELECT CAST(p.company_id AS VARCHAR) AS company_id,
               p.product_id, p.type,
               b.balance, b.balance_sentinel
        FROM balances b
        JOIN banking_products p ON b.product_id = p.product_id
        WHERE p.type IN ('checking','saving','tpv')
        """
    ).df()
    cash["used"] = cash["balance"].notna()
    n_cash = len(cash)
    n_cash_used = int(cash["used"].sum())
    n_cash_sent = int(cash["balance_sentinel"].fillna(False).sum())
    # Companies whose *entire* cash book is sentinel / unused.
    by_co = cash.groupby("company_id").agg(
        n=("product_id", "nunique"),
        n_used=("used", "sum"),
        n_sent=("balance_sentinel", "sum"),
    )
    n_co_no_snap = int((by_co["n_used"] == 0).sum())
    n_co_sent_only = int(((by_co["n_sent"] > 0) & (by_co["n_used"] == 0)).sum())

    print("pass7 sentinel")
    print(f"  sentinel rows={len(rows)} cash_type={int(rows.cash_type.sum())}")
    print(f"  types bank={rows.bank_type.tolist()} debt={rows.debt_type.tolist()}")
    print(
        f"  B cash products={n_cash} used(not-null)={n_cash_used} "
        f"sentinel_on_cash={n_cash_sent} cos_no_snap={n_co_no_snap}"
    )
    return {
        "rows": rows,
        "n_cash": n_cash,
        "n_cash_used": n_cash_used,
        "n_cash_sent": n_cash_sent,
        "n_co_no_snap": n_co_no_snap,
        "n_co_sent_only": n_co_sent_only,
        "by_co": by_co,
    }


def pass8_debt_vs_cashflow(con, panel: pd.DataFrame) -> dict:
    """How much would f_ds_r / a_op_in move if the other extreme rule were used?"""
    cm = pd.DataFrame({"category": list(CAT_MAP), "grp": list(CAT_MAP.values())})
    try:
        con.unregister("_qa_cat")
    except Exception:
        pass
    con.register("_qa_cat", cm)
    flows = con.execute(
        """
        SELECT CAST(t.company_id AS VARCHAR) AS company_id,
               CAST(date_trunc('month', t."date") AS DATE) AS period,
               SUM(CASE WHEN c.grp = 'op_in' THEN t.amount ELSE 0 END) AS op_in_all,
               SUM(CASE WHEN c.grp = 'op_in' AND NOT coalesce(t.is_extreme, false)
                        THEN t.amount ELSE 0 END) AS op_in_ex,
               -SUM(CASE WHEN c.grp = 'debt_service' THEN t.amount ELSE 0 END) AS ds_all,
               -SUM(CASE WHEN c.grp = 'debt_service' AND NOT coalesce(t.is_extreme, false)
                         THEN t.amount ELSE 0 END) AS ds_ex,
               -SUM(CASE WHEN c.grp = 'fin_cost' THEN t.amount ELSE 0 END) AS fc_all,
               -SUM(CASE WHEN c.grp = 'fin_cost' AND NOT coalesce(t.is_extreme, false)
                         THEN t.amount ELSE 0 END) AS fc_ex
        FROM transactions t
        LEFT JOIN _qa_cat c ON t.category = c.category
        WHERE t."date" IS NOT NULL
        GROUP BY 1, 2
        """
    ).df()
    flows["period"] = pd.to_datetime(flows["period"])
    tr = panel[panel["split"] == "train"][["company_id", "period", "a_op_in", "f_ds_r"]].copy()
    tr = tr.merge(flows, on=["company_id", "period"], how="left")
    for c in ["op_in_all", "op_in_ex", "ds_all", "ds_ex", "fc_all", "fc_ex"]:
        tr[c] = tr[c].fillna(0.0)
    tr = tr.sort_values(["company_id", "period"]).reset_index(drop=True)
    g = tr.groupby("company_id", sort=False)
    for src, dst in [
        ("op_in_all", "in3_all"),
        ("op_in_ex", "in3_ex"),
        ("ds_all", "ds3_all"),
        ("ds_ex", "ds3_ex"),
        ("fc_all", "fc3_all"),
        ("fc_ex", "fc3_ex"),
    ]:
        tr[dst] = g[src].transform(lambda s: s.rolling(3, min_periods=3).sum())
    tr["fds_keep"] = (tr["ds3_all"] / np.maximum(tr["in3_all"], 1.0)).clip(0.0, 2.0)
    tr["fds_drop"] = (tr["ds3_ex"] / np.maximum(tr["in3_ex"], 1.0)).clip(0.0, 2.0)
    # Store f_ds_r is the drop-extreme (debt) rule.
    store = pd.to_numeric(tr["f_ds_r"], errors="coerce")
    rec_err = float((store - tr["fds_drop"]).abs().max()) if store.notna().any() else float("nan")

    d_in = tr["op_in_all"] - tr["op_in_ex"]
    d_fds = (tr["fds_keep"] - tr["fds_drop"]).abs()
    out = {
        "train_sum_op_in_keep": float(tr["op_in_all"].sum()),
        "train_sum_op_in_drop": float(tr["op_in_ex"].sum()),
        "rel_op_in": _pct(float(d_in.sum()), float(tr["op_in_all"].sum())),
        "n_cm_op_in_changed": int((d_in.abs() > 0).sum()),
        "train_sum_ds_keep": float(tr["ds_all"].sum()),
        "train_sum_ds_drop": float(tr["ds_ex"].sum()),
        "rel_ds": _pct(float((tr["ds_all"] - tr["ds_ex"]).sum()), float(tr["ds_all"].sum()) if tr["ds_all"].sum() else np.nan),
        "n_cm_fds_defined": int(tr["fds_drop"].notna().sum()),
        "n_cm_fds_changed": int((d_fds > 1e-12).sum()),
        "n_cm_fds_ge_001": int((d_fds >= 0.01).sum()),
        "mean_abs_fds": float(d_fds.mean()),
        "max_abs_fds": float(d_fds.max()) if len(d_fds) else float("nan"),
        "p99_abs_fds": float(d_fds.quantile(0.99)) if len(d_fds) else float("nan"),
        "store_vs_drop_maxabs": rec_err,
        "extreme_op_in_mass": float(d_in.clip(lower=0).sum() + (-d_in.clip(upper=0)).sum()),
    }
    print("pass8 debt vs cashflow")
    print(
        f"  a_op_in train keep={out['train_sum_op_in_keep']:.3e} "
        f"drop={out['train_sum_op_in_drop']:.3e} rel={_pp(out['rel_op_in'],3)} "
        f"cm_changed={out['n_cm_op_in_changed']}"
    )
    print(
        f"  f_ds_r |Δ| mean={_f(out['mean_abs_fds'])} max={_f(out['max_abs_fds'])} "
        f"cm_changed={out['n_cm_fds_changed']} ≥0.01={out['n_cm_fds_ge_001']} "
        f"store_vs_drop_maxabs={_f(out['store_vs_drop_maxabs'])}"
    )
    return out


def pass9_quoted(con, panel: pd.DataFrame) -> dict:
    """Would dropping extremes move days 0.711 / issued_lag1 0.630 / HHI 0.605 by ≥0.02?"""
    days = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               CAST(date_trunc('month', "date") AS DATE) AS period,
               count(DISTINCT CAST("date" AS DATE)) AS days_keep,
               count(DISTINCT CASE WHEN NOT coalesce(is_extreme, false)
                                   THEN CAST("date" AS DATE) END) AS days_drop
        FROM transactions
        WHERE "date" IS NOT NULL
        GROUP BY 1, 2
        """
    ).df()
    days["period"] = pd.to_datetime(days["period"])

    issued = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               CAST(date_trunc('month', issuance_date) AS DATE) AS period,
               SUM(CASE WHEN document_type = 'invoice' AND amount > 0
                        THEN abs(amount) ELSE 0 END) AS ar_keep,
               SUM(CASE WHEN document_type = 'invoice' AND amount > 0
                         AND NOT coalesce(is_extreme, false)
                        THEN abs(amount) ELSE 0 END) AS ar_drop
        FROM invoices
        WHERE amount <> 0 AND issuance_date IS NOT NULL
        GROUP BY 1, 2
        """
    ).df()
    issued["period"] = pd.to_datetime(issued["period"])

    # HHI: only recompute if an extreme AR invoice can sit in a 6m window.
    ext_ar = con.execute(
        f"""
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               CAST(issuance_date AS DATE) AS iss,
               counterparty_id, abs(amount) AS amt
        FROM invoices
        WHERE {BOOK} AND amount > 0 AND is_extreme AND counterparty_id IS NOT NULL
        """
    ).df()
    ext_ar["iss"] = pd.to_datetime(ext_ar["iss"])

    p = panel.copy()
    p = p.merge(days, on=["company_id", "period"], how="left")
    p = p.merge(issued, on=["company_id", "period"], how="left")
    p["days_keep"] = p["days_keep"].fillna(0)
    p["days_drop"] = p["days_drop"].fillna(0)
    p = p.sort_values(["company_id", "period"]).reset_index(drop=True)
    g = p.groupby("company_id", sort=False)
    p["issued_lag1_keep"] = g["ar_keep"].shift(1)
    p["issued_lag1_drop"] = g["ar_drop"].shift(1)
    p["hhi_lag3"] = g["d_cust_hhi"].shift(3)

    # Recompute HHI excluding extremes for companies that have one.
    hhi_changed = False
    p["hhi_drop"] = p["d_cust_hhi"]
    if not ext_ar.empty:
        hhi_changed = True
        inv = con.execute(
            f"""
            SELECT CAST(company_id AS VARCHAR) AS company_id,
                   CAST(issuance_date AS DATE) AS iss,
                   counterparty_id, abs(amount) AS amt, is_extreme
            FROM invoices
            WHERE {BOOK} AND amount > 0 AND counterparty_id IS NOT NULL
              AND company_id IN (SELECT company_id FROM invoices WHERE is_extreme AND amount > 0)
            """
        ).df()
        inv["iss"] = pd.to_datetime(inv["iss"])
        # 6-month windows on the official grid for affected companies.
        aff = set(inv["company_id"].astype(str))
        sub = p[p["company_id"].isin(aff)][["company_id", "period"]].drop_duplicates()
        new_rows = []
        for rec in sub.itertuples(index=False):
            end = pd.Timestamp(rec.period) + pd.offsets.MonthEnd(0)
            start = (pd.Timestamp(rec.period) - pd.DateOffset(months=5)).normalize()
            win = inv[(inv["company_id"] == rec.company_id) & (inv["iss"] >= start) & (inv["iss"] <= end)]
            for drop, col in ((False, "hhi_keep"), (True, "hhi_drop")):
                w = win if not drop else win[~win["is_extreme"].astype(bool)]
                w = w[w["amt"] > 0]
                if w.empty:
                    new_rows.append({"company_id": rec.company_id, "period": rec.period, col: np.nan})
                    continue
                gg = w.groupby("counterparty_id")["amt"].sum()
                tot = float(gg.sum())
                if tot <= 0:
                    new_rows.append({"company_id": rec.company_id, "period": rec.period, col: np.nan})
                    continue
                shares = gg / tot
                new_rows.append(
                    {"company_id": rec.company_id, "period": rec.period, col: float((shares ** 2).sum())}
                )
        alt = pd.DataFrame(new_rows)
        if not alt.empty:
            keep = alt.dropna(axis=1, how="all")
            # reshape if we appended keep/drop as separate rows
            kh = alt[alt.get("hhi_keep").notna()] if "hhi_keep" in alt.columns else pd.DataFrame()
            # rebuild cleanly
            rebuilt = {}
            for rec in sub.itertuples(index=False):
                end = pd.Timestamp(rec.period) + pd.offsets.MonthEnd(0)
                start = (pd.Timestamp(rec.period) - pd.DateOffset(months=5)).normalize()
                win = inv[(inv["company_id"] == rec.company_id) & (inv["iss"] >= start) & (inv["iss"] <= end)]
                vals = {}
                for drop, col in ((False, "hhi_keep_r"), (True, "hhi_drop_r")):
                    w = win if not drop else win[~win["is_extreme"].astype(bool)]
                    w = w[w["amt"] > 0]
                    if w.empty or float(w["amt"].sum()) <= 0:
                        vals[col] = np.nan
                    else:
                        gg = w.groupby("counterparty_id")["amt"].sum()
                        shares = gg / float(gg.sum())
                        vals[col] = float((shares ** 2).sum())
                rebuilt[(rec.company_id, pd.Timestamp(rec.period))] = vals
            p["hhi_keep_r"] = p.apply(
                lambda r: rebuilt.get((r["company_id"], r["period"]), {}).get("hhi_keep_r", np.nan),
                axis=1,
            )
            p["hhi_drop_r"] = p.apply(
                lambda r: rebuilt.get((r["company_id"], r["period"]), {}).get("hhi_drop_r", np.nan),
                axis=1,
            )
            p["hhi_drop"] = p["hhi_drop_r"].combine_first(p["d_cust_hhi"])
    p["hhi_lag3_drop"] = p.groupby("company_id", sort=False)["hhi_drop"].shift(3)

    tr = p[p["split"] == "train"].copy()
    results = []

    def _eval(name, ycol, x_keep, x_drop, quote):
        mask = tr[ycol].notna()
        keep = signed_oof_auroc(tr[ycol], x_keep, tr["fold"], mask)
        drop = signed_oof_auroc(tr[ycol], x_drop, tr["fold"], mask)
        n_changed = int((x_keep.notna() & x_drop.notna() & (x_keep != x_drop) & mask).sum())
        delta = (
            (drop["cv"] - keep["cv"])
            if np.isfinite(drop["cv"]) and np.isfinite(keep["cv"])
            else float("nan")
        )
        rec = {
            "name": name,
            "y": ycol,
            "quote": quote,
            "keep_cv": keep["cv"],
            "drop_cv": drop["cv"],
            "delta": delta,
            "keep_sd": keep["sd"],
            "n": keep["n"],
            "n_pos": keep["n_pos"],
            "n_changed": n_changed,
            "moves": bool(np.isfinite(delta) and abs(delta) >= MOVE),
        }
        results.append(rec)
        print(
            f"  {name}: keep={_f(keep['cv'])} drop={_f(drop['cv'])} "
            f"Δ={_f(delta)} quote={quote} changed_cm={n_changed} "
            f"{'MOVES' if rec['moves'] else 'no-move'}"
        )
        return rec

    _eval("days", Y3, tr["days_keep"], tr["days_drop"], QUOTE_DAYS)
    # Store days vs recomputed keep (sanity)
    store_days = signed_oof_auroc(
        tr[Y3], pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce"), tr["fold"], tr[Y3].notna()
    )
    _eval("issued_lag1", Y7, tr["issued_lag1_keep"], tr["issued_lag1_drop"], QUOTE_ISSUED)
    store_iss = signed_oof_auroc(
        tr[Y7],
        tr.groupby("company_id", sort=False)["e_ar_issued"].shift(1),
        tr["fold"],
        tr[Y7].notna(),
    )
    _eval("hhi_lag3", Y4, tr["hhi_lag3"], tr["hhi_lag3_drop"], QUOTE_HHI)

    days_vanish = int(((tr["days_keep"] - tr["days_drop"]) > 0).sum())
    print(
        f"  store days cv={_f(store_days['cv'])} store issued_lag1 cv={_f(store_iss['cv'])} "
        f"days_cm_vanish={days_vanish} ext_ar_inv={len(ext_ar)} hhi_recomputed={hhi_changed}"
    )
    return {
        "results": results,
        "store_days": store_days,
        "store_iss": store_iss,
        "days_vanish": days_vanish,
        "n_ext_ar": int(len(ext_ar)),
        "hhi_changed": hhi_changed,
        "ext_ar": ext_ar,
    }


def pass10_ever(panel: pd.DataFrame) -> dict:
    """Company-level ever-flagged vs never."""
    tr = panel[panel["split"] == "train"]
    ho = panel[panel["split"] == "holdout"]
    flags = [
        "has_tx_extreme",
        "has_tx_dup",
        "has_tx_unkprod",
        "has_inv_extreme",
        "has_pdi",
        "ever_sentinel",
        "ever_ogtg",
        "ever_bank_after",
        "ever_debt_after",
    ]
    rows = []
    for split, df in (("train", tr), ("holdout", ho)):
        n_co = df["company_id"].nunique()
        for f in flags:
            if f not in df.columns:
                continue
            ever = df.groupby("company_id")[f].max()
            rows.append(
                {
                    "split": split,
                    "flag": f,
                    "n_cos": n_co,
                    "n_ever": int((ever > 0).sum()),
                    "share_ever": _pct(int((ever > 0).sum()), n_co),
                }
            )
    print("pass10 ever-flagged")
    for r in rows:
        if r["split"] != "train":
            continue
        print(f"  {r['flag']:22s} ever {r['n_ever']}/{r['n_cos']} ({_pp(r['share_ever'])})")
    return {"rows": rows}


def _pass34(con, p3: dict) -> dict:
    """How large are the two shared extreme-tx groups? Coverage, not sibling H."""
    groups = ("GROUP_0094", "GROUP_0199")
    q = ",".join(f"'{g}'" for g in groups)
    df = con.execute(
        f"""
        SELECT CAST(group_id AS VARCHAR) AS group_id,
               CAST(company_id AS VARCHAR) AS company_id
        FROM companies
        WHERE group_id IN ({q})
        ORDER BY 1, 2
        """
    ).df()
    hold = load_holdout()
    ext = set(p3["tx"]["company_id"].unique())
    df["split"] = np.where(df["company_id"].isin(hold), "holdout", "train")
    df["has_ext_tx"] = df["company_id"].isin(ext)
    sizes = df.groupby("group_id").agg(n_cos=("company_id", "nunique")).reset_index()
    print("pass34 shared-group size")
    print(df.to_string(index=False))
    print(sizes.to_string(index=False))
    return {
        "n": int(len(df)),
        "rows": df,
        "sizes": sizes,
        "n_ext": int(df["has_ext_tx"].sum()),
    }


def _pass33(con, p3: dict) -> dict:
    """Are extreme-tx companies in the same group? Coverage fact, not sibling H."""
    ids = tuple(p3["tx"]["company_id"].unique().tolist())
    if not ids:
        return {"n": 0, "rows": pd.DataFrame(), "n_shared_groups": 0}
    q = ",".join(f"'{c}'" for c in ids)
    df = con.execute(
        f"""
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               CAST(group_id AS VARCHAR) AS group_id
        FROM companies
        WHERE company_id IN ({q})
        """
    ).df()
    hold = load_holdout()
    df["split"] = np.where(df["company_id"].isin(hold), "holdout", "train")
    vc = df["group_id"].value_counts()
    shared = int((vc > 1).sum())
    print("pass33 extreme-tx companies by group")
    print(df.to_string(index=False))
    print(f"  groups with >1 extreme-tx company={shared}")
    return {"n": int(len(df)), "rows": df, "n_shared_groups": shared}


def _pass32(p3: dict) -> dict:
    """Dates that host extreme txs on more than one company (sibling clock?)."""
    tx = p3["tx"].copy()
    tx["date"] = pd.to_datetime(tx["date"])
    g = (
        tx.groupby(tx["date"].dt.date)
        .agg(n=("company_id", "size"), n_cos=("company_id", "nunique"))
        .reset_index()
    )
    multi = g[g["n_cos"] > 1].sort_values("date")
    print(f"pass32 extreme-tx dates with >1 company: {len(multi)}")
    if len(multi):
        print(multi.to_string(index=False))
        detail = tx[tx["date"].dt.date.isin(set(multi["date"]))][
            ["company_id", "date", "amount", "split"]
        ]
        print(detail.to_string(index=False))
    else:
        detail = pd.DataFrame()
    return {
        "n_dates": int(len(multi)),
        "dates": multi,
        "detail": detail,
    }


def _pass31(p3: dict) -> dict:
    """Holdout extreme txs — coverage only (COMP_0900 / COMP_0276)."""
    tx = p3["tx"]
    ho = tx[tx["split"] == "holdout"][
        ["company_id", "date", "amount", "category", "grp"]
    ].copy()
    print(
        f"pass31 holdout extreme txs n={len(ho)} "
        f"cos={ho.company_id.nunique() if len(ho) else 0}"
    )
    if len(ho):
        print(ho.to_string(index=False))
    return {"n": int(len(ho)), "rows": ho}


def _pass30(p3: dict) -> dict:
    """Card the remaining train extreme-checking companies (0487 / 1192)."""
    tx = p3["tx"]
    want = ["COMP_0487", "COMP_1192"]
    sub = tx[tx["company_id"].isin(want)][
        ["company_id", "date", "amount", "category", "grp", "split"]
    ].copy()
    print("pass30 remaining train extreme-checking txs")
    print(sub.to_string(index=False) if len(sub) else "(none)")
    return {"n": int(len(sub)), "rows": sub}


def _pass29(p3: dict) -> dict:
    """Same-day opposite-sign extreme txs (B-walk net-zero washes)."""
    tx = p3["tx"].copy()
    tx["date"] = pd.to_datetime(tx["date"])
    tx["abs_amt"] = tx["amount"].abs()
    pairs = []
    for (cid, d, a), g in tx.groupby(["company_id", "date", "abs_amt"]):
        if len(g) < 2:
            continue
        if (g["amount"] > 0).any() and (g["amount"] < 0).any():
            pairs.append(
                {
                    "company": cid,
                    "date": str(pd.Timestamp(d).date()),
                    "n": int(len(g)),
                    "pos": float(g.loc[g["amount"] > 0, "amount"].sum()),
                    "neg": float(g.loc[g["amount"] < 0, "amount"].sum()),
                    "net": float(g["amount"].sum()),
                    "cats": ",".join(sorted(g["category"].astype(str).unique())),
                }
            )
    print(f"pass29 same-day opposite-sign extreme tx washes={len(pairs)}")
    if pairs:
        print(pd.DataFrame(pairs).to_string(index=False))
    return {"n_pairs": len(pairs), "pairs": pd.DataFrame(pairs)}


def _pass28(p3: dict) -> dict:
    """COMP_0629 has both extreme invoices and txs but no same-mag pair — show amounts."""
    inv = p3["inv"]
    tx = p3["tx"]
    cid = "COMP_0629"
    inv_c = inv[inv["company_id"] == cid][
        ["issuance_date", "amount", "document_type", "status"]
    ].copy()
    tx_c = tx[tx["company_id"] == cid][["date", "amount", "category", "grp"]].copy()
    print("pass28 COMP_0629 extreme amounts (no same-mag inv↔tx pair)")
    print(inv_c.to_string(index=False) if len(inv_c) else "(no inv)")
    print(tx_c.to_string(index=False) if len(tx_c) else "(no tx)")
    return {
        "n_inv": int(len(inv_c)),
        "n_tx": int(len(tx_c)),
        "inv": inv_c,
        "tx": tx_c,
    }


def _pass27(con) -> dict:
    """Same-magnitude extreme inv↔tx pairs on companies that have both (not join QA)."""
    inv = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               CAST(issuance_date AS DATE) AS issuance_date,
               amount, document_type, status
        FROM invoices
        WHERE is_extreme
        """
    ).df()
    tx = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               CAST("date" AS DATE) AS date, amount, category
        FROM transactions
        WHERE is_extreme
        """
    ).df()
    hold = load_holdout()
    both = sorted(set(inv.company_id) & set(tx.company_id))
    pairs = []
    for cid in both:
        for _, i in inv[inv.company_id == cid].iterrows():
            for _, t in tx[tx.company_id == cid].iterrows():
                rel = abs(abs(float(i.amount)) - abs(float(t.amount))) / max(
                    abs(float(i.amount)), 1.0
                )
                days = abs((pd.Timestamp(i.issuance_date) - pd.Timestamp(t.date)).days)
                if rel < 0.02:
                    pairs.append(
                        {
                            "company": cid,
                            "split": "holdout" if cid in hold else "train",
                            "inv_iss": str(pd.Timestamp(i.issuance_date).date()),
                            "inv_amt": float(i.amount),
                            "inv_status": i.status,
                            "tx_date": str(pd.Timestamp(t.date).date()),
                            "tx_amt": float(t.amount),
                            "tx_cat": t.category,
                            "amt_rel": rel,
                            "days": int(days),
                        }
                    )
    print(
        f"pass27 extreme inv∩tx companies={both} same-mag pairs={len(pairs)}"
    )
    if pairs:
        print(pd.DataFrame(pairs).to_string(index=False))
    return {
        "both": both,
        "n_both": len(both),
        "pairs": pd.DataFrame(pairs),
        "n_pairs": len(pairs),
    }


def _pass26(con) -> dict:
    """Is COMP_0306's extreme invoice the same economic event as the checking outflow?"""
    inv = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               CAST(issuance_date AS DATE) AS issuance_date,
               CAST(payment_date AS DATE) AS payment_date,
               amount, document_type, status,
               payment_date_invalid, is_extreme
        FROM invoices
        WHERE company_id = 'COMP_0306' AND is_extreme
        """
    ).df()
    tx = con.execute(
        """
        SELECT CAST("date" AS DATE) AS date, amount, category, product_id
        FROM transactions
        WHERE company_id = 'COMP_0306' AND is_extreme
        """
    ).df()
    near = []
    if len(inv) and len(tx):
        for _, i in inv.iterrows():
            for _, t in tx.iterrows():
                amt_rel = abs(abs(float(i.amount)) - abs(float(t.amount))) / max(
                    abs(float(i.amount)), 1.0
                )
                d_iss = abs((pd.Timestamp(i.issuance_date) - pd.Timestamp(t.date)).days)
                d_pay = (
                    abs((pd.Timestamp(i.payment_date) - pd.Timestamp(t.date)).days)
                    if pd.notna(i.payment_date)
                    else None
                )
                near.append(
                    {
                        "inv_iss": str(i.issuance_date),
                        "inv_pay": str(i.payment_date),
                        "inv_amt": float(i.amount),
                        "inv_type": i.document_type,
                        "inv_status": i.status,
                        "tx_date": str(t.date),
                        "tx_amt": float(t.amount),
                        "amt_rel": amt_rel,
                        "days_iss": int(d_iss),
                        "days_pay": d_pay,
                    }
                )
    print("pass26 COMP_0306 inv vs tx extremes")
    print(inv.to_string(index=False) if len(inv) else "(no extreme inv)")
    if near:
        print(pd.DataFrame(near).to_string(index=False))
    same = bool(near and min(r["amt_rel"] for r in near) < 0.02)
    print(f"  same-magnitude (|Δamt|/|inv|<2%)={same} n_pairs={len(near)}")
    return {"inv": inv, "near": pd.DataFrame(near), "same_mag": same, "n_pairs": len(near)}


def _pass25(con, panel: pd.DataFrame) -> dict:
    """COMP_0306 is the only train company whose Y2 labels would flip.

    Card the extreme checking row and list months that change liq sign.
    Also count months that would become Y3-eligible via liq<0 (Y3 already
    uses the B cash path; no assembler).
    """
    cid = "COMP_0306"
    txs = con.execute(
        """
        SELECT CAST(t.company_id AS VARCHAR) AS company_id,
               CAST(t."date" AS DATE) AS date,
               t.amount, t.category, t.status, t.product_id,
               p.type AS bank_type
        FROM transactions t
        JOIN banking_products p ON t.product_id = p.product_id
        WHERE t.is_extreme AND t.company_id = 'COMP_0306'
        """
    ).df()
    ext = con.execute(
        """
        SELECT CAST(date_trunc('month', t."date") AS DATE) AS month,
               SUM(t.amount) AS flow
        FROM transactions t
        JOIN banking_products p ON t.product_id = p.product_id
        WHERE t.is_extreme AND t.company_id = 'COMP_0306'
          AND p.type IN ('checking','saving','tpv')
        GROUP BY 1
        """
    ).df()
    ext["month"] = pd.to_datetime(ext["month"])
    sub = panel[panel["company_id"] == cid][
        ["period", "b_liq", Y2, Y3, "split"]
    ].copy().sort_values("period")
    afters = [
        float(ext.loc[ext["month"] > per, "flow"].sum()) for per in sub["period"]
    ]
    liq = pd.to_numeric(sub["b_liq"], errors="coerce")
    alt = liq + pd.Series(afters, index=sub.index)
    sign = ((liq < 0) != (alt < 0)) & liq.notna() & alt.notna()
    flip_m = [
        {
            "period": str(pd.Timestamp(p).date()),
            "b_liq": float(a) if pd.notna(a) else float("nan"),
            "alt_liq": float(b) if pd.notna(b) else float("nan"),
            "y2": float(y) if pd.notna(y) else float("nan"),
            "y3": float(z) if pd.notna(z) else float("nan"),
        }
        for p, a, b, y, z in zip(
            sub.loc[sign, "period"],
            liq[sign],
            alt[sign],
            pd.to_numeric(sub.loc[sign, Y2], errors="coerce"),
            pd.to_numeric(sub.loc[sign, Y3], errors="coerce"),
        )
    ]
    y3 = pd.to_numeric(sub[Y3], errors="coerce")
    rec = {
        "n_tx": int(len(txs)),
        "txs": txs,
        "n_sign": int(sign.sum()),
        "n_new_neg": int((alt < 0).sum()) - int((liq < 0).sum()),
        "y3_lab": int(y3.notna().sum()),
        "y3_pos": int((y3 == 1).sum()),
        "n_y3_eligible_alt": int((alt < 0).sum()),
        "flip_months": flip_m,
    }
    print("pass25 COMP_0306 extreme card")
    print(txs.to_string(index=False) if len(txs) else "(none)")
    print(
        f"  liq sign flips={rec['n_sign']} new_neg_months={rec['n_new_neg']} "
        f"store Y3 lab={rec['y3_lab']} pos={rec['y3_pos']} "
        f"alt_liq<0 months={rec['n_y3_eligible_alt']}"
    )
    return rec


def _pass24(con, panel: pd.DataFrame) -> dict:
    """Y2 flips if the B walk dropped extreme checking flows.

    Identity: end_bal_t = snapshot − sum(flows after t).
    Drop extremes → alt_liq_t = b_liq_t + sum(extreme cash-type flow after t).
    Y2 = 1 if alt_liq < 0 in at least 2 of the next 3 months (same horizon as y2_stress).
    Report only — do not rewrite liquidity.py or rebuild targets.
    """
    ext = con.execute(
        """
        SELECT CAST(t.company_id AS VARCHAR) AS company_id,
               CAST(date_trunc('month', t."date") AS DATE) AS month,
               SUM(t.amount) AS flow
        FROM transactions t
        JOIN banking_products p ON t.product_id = p.product_id
        WHERE t.is_extreme AND p.type IN ('checking','saving','tpv')
        GROUP BY 1, 2
        """
    ).df()
    ext["month"] = pd.to_datetime(ext["month"])
    hold = load_holdout()
    cutoff = LAST_M - pd.DateOffset(months=3)
    rows = []
    n_flip_train = 0
    n_lab_train = 0
    n_pos_store = 0
    n_pos_alt = 0
    n_sign_train = 0
    for cid, g in ext.groupby("company_id"):
        sub = panel[panel["company_id"] == cid][
            ["company_id", "period", "b_liq", Y2, "split"]
        ].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("period")
        g2 = g.groupby("month", as_index=False)["flow"].sum()
        afters = [
            float(g2.loc[g2["month"] > per, "flow"].sum()) for per in sub["period"]
        ]
        sub = sub.assign(ext_after=afters)
        liq = pd.to_numeric(sub["b_liq"], errors="coerce")
        alt = liq + sub["ext_after"]
        neg = pd.Series(np.where(alt.isna(), np.nan, (alt < 0).astype(float)), index=sub.index)
        n1 = neg.shift(-1)
        n2 = neg.shift(-2)
        n3 = neg.shift(-3)
        neg_count = n1 + n2 + n3
        has_h3 = sub["period"] <= cutoff
        y_alt = np.where(has_h3 & neg_count.notna(), (neg_count >= 2).astype(float), np.nan)
        y_store = pd.to_numeric(sub[Y2], errors="coerce")
        both = y_store.notna() & pd.notna(y_alt)
        flip = both & (y_store.to_numpy() != y_alt)
        split = "holdout" if cid in hold else "train"
        sign_flip = int((((liq < 0) != (alt < 0)) & alt.notna() & liq.notna()).sum())
        rec = {
            "company": cid,
            "split": split,
            "n_lab": int(both.sum()),
            "n_flip": int(flip.sum()),
            "store_pos": int((y_store[both] == 1).sum()),
            "alt_pos": int((pd.Series(y_alt, index=sub.index)[both] == 1).sum()),
            "n_neg_store": int((liq < 0).sum()),
            "n_neg_alt": int((alt < 0).sum()),
            "n_sign_flip": sign_flip,
        }
        rows.append(rec)
        if split == "train":
            n_flip_train += rec["n_flip"]
            n_lab_train += rec["n_lab"]
            n_pos_store += rec["store_pos"]
            n_pos_alt += rec["alt_pos"]
            n_sign_train += sign_flip
    print("pass24 Y2 flips if B walk dropped extreme checking flows")
    print(pd.DataFrame(rows).to_string(index=False))
    print(
        f"  train labeled={n_lab_train} flips={n_flip_train} "
        f"store_pos={n_pos_store} alt_pos={n_pos_alt} liq_sign_flips={n_sign_train}"
    )
    return {
        "rows": pd.DataFrame(rows),
        "n_flip_train": n_flip_train,
        "n_lab_train": n_lab_train,
        "n_pos_store": n_pos_store,
        "n_pos_alt": n_pos_alt,
        "n_sign_train": n_sign_train,
    }


def _pass23(panel: pd.DataFrame) -> dict:
    """Y2 on the four train companies whose checking book carries extremes."""
    cos = ["COMP_0629", "COMP_0306", "COMP_0487", "COMP_1192"]
    tr = panel[panel["split"] == "train"]
    y = pd.to_numeric(tr[Y2], errors="coerce")
    on = tr["company_id"].isin(cos)
    rec = {
        "n_lab_on": int((on & y.notna()).sum()),
        "n_pos_on": int((on & (y == 1)).sum()),
        "rate_on": float(y[on & y.notna()].mean()) if (on & y.notna()).any() else float("nan"),
        "n_lab_off": int((~on & y.notna()).sum()),
        "rate_off": float(y[~on & y.notna()].mean()) if (~on & y.notna()).any() else float("nan"),
        "n_cos": 4,
    }
    print(
        f"pass23 Y2 on 4 train extreme-checking cos: "
        f"lab={rec['n_lab_on']} pos={rec['n_pos_on']} rate={_pp(rec['rate_on'])} "
        f"vs rest {_pp(rec['rate_off'])}"
    )
    return rec


def _pass22(con, panel: pd.DataFrame) -> dict:
    """Estimate B-walk liq shift = −sum(extreme checking flows after the period)."""
    ext = con.execute(
        """
        SELECT CAST(t.company_id AS VARCHAR) AS company_id,
               t.product_id,
               CAST(date_trunc('month', t."date") AS DATE) AS month,
               SUM(t.amount) AS flow
        FROM transactions t
        JOIN banking_products p ON t.product_id = p.product_id
        WHERE t.is_extreme AND p.type IN ('checking','saving','tpv')
        GROUP BY 1, 2, 3
        """
    ).df()
    ext["month"] = pd.to_datetime(ext["month"])
    hold = load_holdout()
    rows = []
    for cid, g in ext.groupby("company_id"):
        tot_after_start = float(g["flow"].sum())
        # last official month: flows after 2026-08 are only 2026-09 extract-day, ignored by B last=LAST_M
        after_aug = float(g.loc[g["month"] > PANEL_END, "flow"].sum())
        rows.append(
            {
                "company": cid,
                "split": "holdout" if cid in hold else "train",
                "n_months": int(g["month"].nunique()),
                "sum_extreme_flow": tot_after_start,
                "after_aug": after_aug,
            }
        )
        # merge onto that company's panel b_liq if present
        sub = panel[(panel["company_id"] == cid)][["period", "b_liq", "split"]].copy()
        if sub.empty or "b_liq" not in sub.columns:
            continue
        # shift at period t = −sum(extreme flow with month > t)
        g2 = g.groupby("month", as_index=False)["flow"].sum().sort_values("month")
        cum = g2["flow"].sum()
        # for each period, after = total − cum through period
        shifts = []
        for per in sub["period"]:
            after = float(g2.loc[g2["month"] > per, "flow"].sum())
            shifts.append(-after)
        sub = sub.assign(shift=shifts)
        liq = pd.to_numeric(sub["b_liq"], errors="coerce")
        rows[-1]["max_abs_shift"] = float(np.nanmax(np.abs(shifts))) if shifts else float("nan")
        rows[-1]["n_cm"] = int(len(sub))
        rows[-1]["liq_p50"] = float(liq.median()) if liq.notna().any() else float("nan")
    print("pass22 B-walk extreme-flow shift")
    print(pd.DataFrame(rows).to_string(index=False))
    return {"rows": pd.DataFrame(rows)}


def _pass21(con) -> dict:
    """Do extreme txs sit on checking/saving/tpv (Family B flow walk)?"""
    df = con.execute(
        """
        SELECT CAST(t.company_id AS VARCHAR) AS company_id,
               t.product_id, p.type AS bank_type, d.type AS debt_type,
               t.amount, t.category,
               CASE WHEN p.type IN ('checking','saving','tpv') THEN 1 ELSE 0 END AS cash_type
        FROM transactions t
        LEFT JOIN banking_products p ON t.product_id = p.product_id
        LEFT JOIN debt_products d ON t.product_id = d.product_id
        WHERE t.is_extreme
        """
    ).df()
    n_cash = int(df["cash_type"].sum())
    print(
        f"pass21 extreme txs on cash types={n_cash}/{len(df)} "
        f"bank_types={df['bank_type'].value_counts(dropna=False).to_dict()}"
    )
    return {
        "n": int(len(df)),
        "n_cash": n_cash,
        "bank_types": df["bank_type"].value_counts(dropna=False).to_dict(),
        "rows": df,
    }


def _pass20(con) -> dict:
    """Holdout extreme companies: share of *their own* tx |amount| that is extreme."""
    df = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               count(*) AS n_tx,
               sum(CASE WHEN is_extreme THEN 1 ELSE 0 END) AS n_ext,
               sum(abs(amount)) AS mass,
               sum(CASE WHEN is_extreme THEN abs(amount) ELSE 0 END) AS mass_ext
        FROM transactions
        WHERE company_id IN (
            SELECT DISTINCT company_id FROM transactions WHERE is_extreme
        )
        GROUP BY 1
        ORDER BY mass_ext DESC
        """
    ).df()
    hold = load_holdout()
    df["split"] = np.where(df["company_id"].isin(hold), "holdout", "train")
    df["mass_share"] = df["mass_ext"] / df["mass"]
    print("pass20 extreme companies own-mass share")
    print(df[["company_id", "split", "n_ext", "n_tx", "mass_share"]].to_string(index=False))
    return {"rows": df}


def _pass19(con, panel: pd.DataFrame) -> dict:
    """Flows on post-snapshot banking products (G excludes them; cashflow keeps)."""
    cm = pd.DataFrame({"category": list(CAT_MAP), "grp": list(CAT_MAP.values())})
    try:
        con.unregister("_qa_cat2")
    except Exception:
        pass
    con.register("_qa_cat2", cm)
    flows = con.execute(
        """
        SELECT CAST(t.company_id AS VARCHAR) AS company_id,
               CAST(date_trunc('month', t."date") AS DATE) AS period,
               SUM(CASE WHEN c.grp = 'op_in' THEN t.amount ELSE 0 END) AS op_in_after,
               SUM(t.amount) AS net_after,
               count(*) AS n
        FROM transactions t
        JOIN banking_products p ON t.product_id = p.product_id
        LEFT JOIN _qa_cat2 c ON t.category = c.category
        WHERE p.created_after_snapshot
          AND CAST(t."date" AS DATE) < DATE '2026-09-01'
        GROUP BY 1, 2
        """
    ).df()
    flows["period"] = pd.to_datetime(flows["period"])
    tr = panel[panel["split"] == "train"][["company_id", "period", "a_op_in"]].copy()
    m = tr.merge(flows, on=["company_id", "period"], how="left")
    m["op_in_after"] = m["op_in_after"].fillna(0.0)
    tot = float(m["a_op_in"].sum())
    after = float(m["op_in_after"].sum())
    n_cm = int((m["op_in_after"].abs() > 0).sum())
    print(
        f"pass19 post-snapshot banking op_in on train panel={after:.3e} / {tot:.3e} "
        f"rel={_pp(_pct(after, tot), 4)} cm={n_cm}"
    )
    return {
        "op_in_after": after,
        "op_in_all": tot,
        "rel": _pct(after, tot),
        "n_cm": n_cm,
        "n_tx_rows": int(flows["n"].sum()) if len(flows) else 0,
    }


def _pass18(con, panel: pd.DataFrame) -> dict:
    """Post-snapshot products: any pre-extract txs? Last-month ogtg vs store."""
    pre = con.execute(
        """
        SELECT 'banking' AS book,
               count(DISTINCT p.product_id) AS n_prod,
               count(DISTINCT t.transaction_id) AS n_tx,
               count(DISTINCT t.company_id) AS n_cos
        FROM banking_products p
        LEFT JOIN transactions t
          ON t.product_id = p.product_id
         AND CAST(t."date" AS DATE) < DATE '2026-09-01'
        WHERE p.created_after_snapshot
        UNION ALL
        SELECT 'debt',
               count(DISTINCT p.product_id),
               count(DISTINCT t.transaction_id),
               count(DISTINCT t.company_id)
        FROM debt_products p
        LEFT JOIN transactions t
          ON t.product_id = p.product_id
         AND CAST(t."date" AS DATE) < DATE '2026-09-01'
        WHERE p.created_after_snapshot
        """
    ).df()
    last = panel[(panel["split"] == "train") & (panel["period"] == PANEL_END)].copy()
    # store may have f_outstanding_gt_granted
    store = pd.read_parquet(STORE, columns=["company_id", "period", "f_outstanding_gt_granted", "g_created_after_snapshot"])
    store["company_id"] = store["company_id"].astype(str)
    store["period"] = pd.to_datetime(store["period"]).dt.normalize()
    sl = store[(store["company_id"].isin(set(last["company_id"]))) & (store["period"] == PANEL_END)]
    ogtg = pd.to_numeric(sl["f_outstanding_gt_granted"], errors="coerce")
    g_after = pd.to_numeric(sl["g_created_after_snapshot"], errors="coerce")
    print("pass18 post-snapshot pre-extract txs")
    print(pre.to_string(index=False))
    print(
        f"      last-month train f_ogtg nn={int(ogtg.notna().sum())} mean={_f(ogtg.mean())} "
        f"g_created_after_snapshot nn={int(g_after.notna().sum())} max={_f(g_after.max())}"
    )
    return {
        "pre": pre,
        "ogtg_nn": int(ogtg.notna().sum()),
        "ogtg_mean": float(ogtg.mean()) if ogtg.notna().any() else float("nan"),
        "g_after_max": float(g_after.max()) if g_after.notna().any() else float("nan"),
    }


def _pass17(con, panel: pd.DataFrame) -> dict:
    """Unknown-product txs: category mix, overlap with unknown balances, later g_n_accounts."""
    cat = con.execute(
        """
        SELECT category, count(*) AS n, sum(abs(amount)) AS mass,
               count(DISTINCT company_id) AS n_cos
        FROM transactions
        WHERE NOT product_known
        GROUP BY 1
        ORDER BY n DESC
        """
    ).df()
    overlap = con.execute(
        """
        SELECT count(DISTINCT t.product_id) AS n_tx_unk,
               count(DISTINCT b.product_id) AS n_bal_unk,
               count(DISTINCT CASE WHEN b.product_id IS NOT NULL THEN t.product_id END) AS both
        FROM (SELECT DISTINCT product_id FROM transactions WHERE NOT product_known) t
        FULL OUTER JOIN (SELECT DISTINCT product_id FROM balances WHERE NOT product_known) b
          ON t.product_id = b.product_id
        """
    ).fetchone()
    # companies with unk txs: do they later get g_n_accounts>0?
    tr = panel[panel["split"] == "train"]
    unk_cos = set(
        con.execute(
            """
            SELECT DISTINCT CAST(company_id AS VARCHAR)
            FROM transactions WHERE NOT product_known
            """
        ).fetchdf().iloc[:, 0].astype(str)
    )
    later = tr[tr["company_id"].isin(unk_cos)]
    last = later.sort_values("period").groupby("company_id").tail(1)
    n_later_g = int((last["g_n_accounts"].fillna(0) > 0).sum()) if len(last) else 0
    print(
        f"pass17 unk cat top={cat.head(5).to_dict('records')} "
        f"tx_unk_prod={overlap[0]} bal_unk={overlap[1]} both={overlap[2]} "
        f"unk_cos_train_last_g>0={n_later_g}/{len(last)}"
    )
    return {
        "cat": cat,
        "n_tx_unk": int(overlap[0] or 0),
        "n_bal_unk": int(overlap[1] or 0),
        "n_both": int(overlap[2] or 0),
        "n_later_g": n_later_g,
        "n_unk_cos_last": int(len(last)),
    }


def _pass16(con, panel: pd.DataFrame) -> dict:
    """Family E drops PDI from open/delay but not from issued. Would issued_lag1 move?"""
    iss = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               CAST(date_trunc('month', issuance_date) AS DATE) AS period,
               SUM(CASE WHEN document_type = 'invoice' AND amount > 0
                        THEN abs(amount) ELSE 0 END) AS ar_keep,
               SUM(CASE WHEN document_type = 'invoice' AND amount > 0
                         AND NOT coalesce(payment_date_invalid, false)
                        THEN abs(amount) ELSE 0 END) AS ar_nopdi
        FROM invoices
        WHERE amount <> 0 AND issuance_date IS NOT NULL
        GROUP BY 1, 2
        """
    ).df()
    iss["period"] = pd.to_datetime(iss["period"])
    tr = panel[panel["split"] == "train"].copy()
    tr = tr.merge(iss, on=["company_id", "period"], how="left")
    tr = tr.sort_values(["company_id", "period"])
    g = tr.groupby("company_id", sort=False)
    tr["lag_keep"] = g["ar_keep"].shift(1)
    tr["lag_nopdi"] = g["ar_nopdi"].shift(1)
    mask = tr[Y7].notna()
    keep = signed_oof_auroc(tr[Y7], tr["lag_keep"], tr["fold"], mask)
    drop = signed_oof_auroc(tr[Y7], tr["lag_nopdi"], tr["fold"], mask)
    delta = (
        drop["cv"] - keep["cv"]
        if np.isfinite(drop["cv"]) and np.isfinite(keep["cv"])
        else float("nan")
    )
    n_changed = int((tr["lag_keep"].notna() & tr["lag_nopdi"].notna() & (tr["lag_keep"] != tr["lag_nopdi"]) & mask).sum())
    rec = {
        "keep": keep["cv"],
        "drop": drop["cv"],
        "delta": delta,
        "n_changed": n_changed,
        "moves": bool(np.isfinite(delta) and abs(delta) >= MOVE),
    }
    print(
        f"pass16 issued_lag1 drop-PDI keep={_f(keep['cv'])} drop={_f(drop['cv'])} "
        f"Δ={_f(delta)} changed={n_changed} {'MOVES' if rec['moves'] else 'no-move'}"
    )
    return rec


def _pass15(con, panel: pd.DataFrame) -> dict:
    """Would dropping is_dup extras move Y3 a_n_tx / days? First copy stays."""
    cm = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               CAST(date_trunc('month', "date") AS DATE) AS period,
               count(*) AS n_keep,
               sum(CASE WHEN NOT coalesce(is_dup, false) THEN 1 ELSE 0 END) AS n_drop,
               count(DISTINCT CAST("date" AS DATE)) AS days_keep,
               count(DISTINCT CASE WHEN NOT coalesce(is_dup, false)
                                   THEN CAST("date" AS DATE) END) AS days_drop
        FROM transactions
        WHERE "date" IS NOT NULL
        GROUP BY 1, 2
        """
    ).df()
    cm["period"] = pd.to_datetime(cm["period"])
    tr = panel[panel["split"] == "train"].copy()
    tr = tr.merge(cm, on=["company_id", "period"], how="left")
    for c in ["n_keep", "n_drop", "days_keep", "days_drop"]:
        tr[c] = tr[c].fillna(0)
    mask = tr[Y3].notna()
    keep = signed_oof_auroc(tr[Y3], tr["n_keep"], tr["fold"], mask)
    drop = signed_oof_auroc(tr[Y3], tr["n_drop"], tr["fold"], mask)
    store = signed_oof_auroc(tr[Y3], pd.to_numeric(tr["a_n_tx"], errors="coerce"), tr["fold"], mask)
    days_k = signed_oof_auroc(tr[Y3], tr["days_keep"], tr["fold"], mask)
    days_d = signed_oof_auroc(tr[Y3], tr["days_drop"], tr["fold"], mask)
    delta = (
        drop["cv"] - keep["cv"]
        if np.isfinite(drop["cv"]) and np.isfinite(keep["cv"])
        else float("nan")
    )
    n_changed = int(((tr["n_keep"] != tr["n_drop"]) & mask).sum())
    days_vanish = int(((tr["days_keep"] - tr["days_drop"]) > 0).sum())
    rec = {
        "ntx_keep": keep["cv"],
        "ntx_drop": drop["cv"],
        "ntx_store": store["cv"],
        "ntx_delta": delta,
        "days_keep": days_k["cv"],
        "days_drop": days_d["cv"],
        "n_changed": n_changed,
        "days_vanish": days_vanish,
        "moves": bool(np.isfinite(delta) and abs(delta) >= MOVE),
    }
    print(
        f"pass15 drop-dup a_n_tx keep={_f(keep['cv'])} drop={_f(drop['cv'])} "
        f"Δ={_f(delta)} store={_f(store['cv'])} changed={n_changed} "
        f"days { _f(days_k['cv']) }→{_f(days_d['cv'])} vanish={days_vanish} "
        f"{'MOVES' if rec['moves'] else 'no-move'}"
    )
    return rec


def _pass13(con, panel: pd.DataFrame, p3: dict, tr: pd.DataFrame) -> dict:
    """Holdout extreme txs + whether leftover extreme companies are ever Y4-labeled."""
    tx = p3["tx"]
    ho = tx[tx["split"] == "holdout"].copy()
    ho_pairs = 0
    recs = list(ho.itertuples(index=True))
    used = set()
    for i, a in enumerate(recs):
        if i in used:
            continue
        for j, b in enumerate(recs):
            if j <= i or j in used:
                continue
            if a.company_id == b.company_id and abs(abs(a.amount) - abs(b.amount)) < 1 and a.amount * b.amount < 0:
                ho_pairs += 1
                used.add(i)
                used.add(j)
                break
    y4_ever = {}
    for cid in ["COMP_0629", "COMP_0163", "COMP_0306", "COMP_0042", "COMP_0487", "COMP_1192"]:
        sub = tr[tr["company_id"] == cid]
        y4 = pd.to_numeric(sub[Y4], errors="coerce") if Y4 in sub.columns else pd.Series(dtype=float)
        y3 = pd.to_numeric(sub[Y3], errors="coerce") if Y3 in sub.columns else pd.Series(dtype=float)
        y4_ever[cid] = {
            "n_cm": int(len(sub)),
            "y4_lab": int(y4.notna().sum()) if len(y4) else 0,
            "y4_pos": int((y4 == 1).sum()) if len(y4) else 0,
            "y3_lab": int(y3.notna().sum()) if len(y3) else 0,
        }
    unk_months = con.execute(
        """
        SELECT count(DISTINCT product_id) AS n_prod,
               count(DISTINCT CAST(date_trunc('month', "date") AS DATE)) AS n_months,
               min(CAST("date" AS DATE)) AS first_d,
               max(CAST("date" AS DATE)) AS last_d
        FROM transactions
        WHERE NOT product_known
        """
    ).fetchone()
    print(
        f"pass13 holdout ext n={len(ho)} wash_pairs={ho_pairs} leftover={len(ho)-2*ho_pairs} "
        f"unk_prod months span={unk_months[2]}→{unk_months[3]} n_prod={unk_months[0]}"
    )
    print(f"      y4 ever on extreme cos: {y4_ever}")
    return {
        "ho_n": int(len(ho)),
        "ho_washes": ho_pairs,
        "ho_leftover": int(len(ho) - 2 * ho_pairs),
        "y4_ever": y4_ever,
        "unk_n_prod": int(unk_months[0]),
        "unk_n_months": int(unk_months[1]),
        "unk_first": str(unk_months[2]),
        "unk_last": str(unk_months[3]),
    }


def _pass14(con, panel: pd.DataFrame) -> dict:
    """Residual PDI (no extremes) calendar + has_pdi as X vs accepted Ys (CLOSE check)."""
    cal = con.execute(
        """
        SELECT CAST(date_trunc('month', issuance_date) AS DATE) AS month,
               count(*) AS n, sum(abs(amount)) AS mass
        FROM invoices
        WHERE payment_date_invalid AND NOT is_extreme
        GROUP BY 1
        ORDER BY 1
        """
    ).df()
    cal["month"] = pd.to_datetime(cal["month"])
    tot_n = float(cal["n"].sum())
    tot_m = float(cal["mass"].sum())
    max_n = float(cal["n"].max()) if len(cal) else 0
    max_m = float(cal["mass"].max()) if len(cal) else 0
    max_n_m = str(cal.loc[cal["n"].idxmax(), "month"].date()) if len(cal) else ""
    max_m_m = str(cal.loc[cal["mass"].idxmax(), "month"].date()) if len(cal) else ""
    tr = panel[panel["split"] == "train"]
    x_rows = []
    if "has_pdi" in tr.columns and "fold" in tr.columns:
        for y in Y_ACCEPTED:
            res = signed_oof_auroc(tr[y], tr["has_pdi"].astype(float), tr["fold"], tr[y].notna())
            x_rows.append(
                {
                    "y": y,
                    "cv": res["cv"],
                    "n": res["n"],
                    "n_pos": res["n_pos"],
                    "low_power": res["low_power"],
                }
            )
    print(
        f"pass14 residual PDI months={len(cal)} max_row {max_n_m}={_pp(_pct(max_n, tot_n))} "
        f"max_mass {max_m_m}={_pp(_pct(max_m, tot_m))}"
    )
    for r in x_rows:
        print(f"      has_pdi vs {r['y']}: cv={_f(r['cv'])} n_pos={r['n_pos']} lp={r['low_power']}")
    return {
        "n_months": int(len(cal)),
        "max_row_month": max_n_m,
        "max_row_share": _pct(max_n, tot_n),
        "max_mass_month": max_m_m,
        "max_mass_share": _pct(max_m, tot_m),
        "x_rows": x_rows,
    }


def pass11_deeper(con, panel: pd.DataFrame, p3: dict, p5: dict, p7: dict, p8: dict, p9: dict) -> dict:
    """Next cuts after prevalence: washes, PDI pile, B gap, HHI windows, clones, soft dups."""
    hold = set(panel.loc[panel["split"] == "holdout", "company_id"])

    # --- 11a canceling extreme invoice pairs ---
    inv = p3["inv"].copy()
    inv["abs_amt"] = inv["amount"].abs()
    inv["sign"] = np.sign(inv["amount"])
    pair_rows = []
    for cid, g in inv.groupby("company_id"):
        net = float(g["amount"].sum())
        pair_rows.append(
            {
                "company": cid,
                "n": int(len(g)),
                "gross": float(g["abs_amt"].sum()),
                "net": net,
                "types": ",".join(sorted(g["document_type"].astype(str).unique())),
                "statuses": ",".join(sorted(g["status"].astype(str).unique())),
            }
        )
    # exact opposite pairs (same abs, opposite sign)
    washes = 0
    used = set()
    recs = list(inv.itertuples(index=True))
    for i, a in enumerate(recs):
        if i in used:
            continue
        for j, b in enumerate(recs):
            if j <= i or j in used:
                continue
            if a.company_id == b.company_id and abs(abs(a.amount) - abs(b.amount)) < 1.0 and a.amount * b.amount < 0:
                washes += 1
                used.add(i)
                used.add(j)
                break
    leftover = [recs[i] for i in range(len(recs)) if i not in used]

    # book-filter AR extremes (what HHI / issued see)
    ar_ext = con.execute(
        f"""
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               CAST(issuance_date AS DATE) AS iss,
               document_type, status, amount, counterparty_id,
               payment_date_invalid
        FROM invoices
        WHERE is_extreme
        ORDER BY abs(amount) DESC
        """
    ).df()
    ar_ext["iss"] = pd.to_datetime(ar_ext["iss"])
    in_hhi = ar_ext[
        (ar_ext["document_type"] == "invoice")
        & (ar_ext["status"] != "cancel")
        & (ar_ext["amount"] > 0)
        & ar_ext["counterparty_id"].notna()
    ]

    print("pass11a extreme washes")
    print(f"  companies={len(pair_rows)} wash_pairs={washes} leftover={len(leftover)}")
    print(f"  book AR+CP extremes (HHI-eligible)={len(in_hhi)}")

    # --- 11b PDI July pile ---
    pdi = con.execute(
        """
        SELECT CAST(date_trunc('month', issuance_date) AS DATE) AS month,
               count(*) AS n,
               sum(abs(amount)) AS mass,
               sum(CASE WHEN is_extreme THEN 1 ELSE 0 END) AS n_ext,
               sum(CASE WHEN is_extreme THEN abs(amount) ELSE 0 END) AS mass_ext
        FROM invoices
        WHERE payment_date_invalid
        GROUP BY 1
        ORDER BY mass DESC
        """
    ).df()
    pdi["month"] = pd.to_datetime(pdi["month"])
    tot_pdi_mass = float(pdi["mass"].sum())
    july = pdi[pdi["month"] == pd.Timestamp("2026-07-01")]
    july_mass = float(july["mass"].sum()) if len(july) else 0.0
    july_ext_mass = float(july["mass_ext"].sum()) if len(july) else 0.0
    # COMP_0629 share of PDI mass
    pdi_co = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               count(*) AS n, sum(abs(amount)) AS mass,
               sum(CASE WHEN is_extreme THEN 1 ELSE 0 END) AS n_ext
        FROM invoices
        WHERE payment_date_invalid
        GROUP BY 1
        ORDER BY mass DESC
        LIMIT 8
        """
    ).df()
    print(
        f"pass11b PDI July mass={july_mass:.3e} / {tot_pdi_mass:.3e} "
        f"({_pp(_pct(july_mass, tot_pdi_mass))}); of which extreme={july_ext_mass:.3e}"
    )

    # --- 11c B snapshot gap ---
    sent_ids = ["COMP_0420", "COMP_1068"]
    cash_gap = con.execute(
        f"""
        SELECT CAST(p.company_id AS VARCHAR) AS company_id,
               p.product_id, p.type,
               b.balance AS clean_bal,
               b.balance_sentinel,
               m.balance AS raw_bal
        FROM banking_products p
        JOIN balances b ON b.product_id = p.product_id
        LEFT JOIN main.balances m ON m.product_id = b.product_id AND m.company_id = b.company_id
        WHERE p.type IN ('checking','saving','tpv')
          AND p.company_id IN ('COMP_0420','COMP_1068')
        ORDER BY p.company_id, p.product_id
        """
    ).df()
    gap_rows = []
    for cid, g in cash_gap.groupby("company_id"):
        used = g[g["clean_bal"].notna()]
        sent = g[g["balance_sentinel"].fillna(False)]
        gap_rows.append(
            {
                "company": cid,
                "n_cash": int(len(g)),
                "n_used": int(len(used)),
                "n_sent": int(len(sent)),
                "snap_used": float(used["clean_bal"].sum()) if len(used) else 0.0,
                "raw_sent": float(sent["raw_bal"].sum()) if len(sent) else 0.0,
                "types": ",".join(sorted(g["type"].astype(str).unique())),
            }
        )
        print(
            f"pass11c {cid} cash={len(g)} used={len(used)} sent={len(sent)} "
            f"snap_used={used['clean_bal'].sum() if len(used) else 0:.3e} "
            f"raw_sent={sent['raw_bal'].sum() if len(sent) else 0:.3e}"
        )

    # --- 11d HHI windows ---
    # For each HHI-eligible extreme AR, how many train CMs have defined d_cust_hhi
    # whose 6m window contains that issuance?
    hhi_hits = []
    tr = panel[panel["split"] == "train"].copy()
    for rec in in_hhi.itertuples(index=False):
        cid = rec.company_id
        iss = pd.Timestamp(rec.iss)
        sub = tr[(tr["company_id"] == cid) & tr["d_cust_hhi"].notna()]
        hit = 0
        y4_hit = 0
        for r in sub.itertuples(index=False):
            end = pd.Timestamp(r.period) + pd.offsets.MonthEnd(0)
            start = (pd.Timestamp(r.period) - pd.DateOffset(months=5)).normalize()
            if start <= iss <= end:
                hit += 1
                yy = getattr(r, Y4, np.nan)
                if pd.notna(yy):
                    y4_hit += 1
        hhi_hits.append(
            {
                "company": cid,
                "iss": str(iss.date()),
                "amount": float(rec.amount),
                "defined_hhi_cm": hit,
                "y4_labeled_cm": y4_hit,
            }
        )
    print(f"pass11d HHI-eligible extremes in defined windows: {hhi_hits}")

    # --- 11e clone whale ---
    whale = (
        p5["clones"]
        .sort_values("n_copies", ascending=False)
        .head(5)[
            ["company_id", "date", "amount", "category", "status", "n_copies", "n_flagged", "split"]
        ]
    )
    print("pass11e clone whales")
    print(whale.to_string(index=False))

    # --- 11f soft not-clone anatomy ---
    soft = p5["soft_not_clone"].copy()
    # join a sample of keys back is expensive; aggregate from the grouped table we have
    n_soft = int(len(soft))
    n_2 = int((soft["n"] == 2).sum())
    n_status = int((soft["n_status"] > 1).sum())
    n_prod = int((soft["n_prod"] > 1).sum())
    n_desc = int((soft["n_desc"] > 1).sum())
    # pending vs booked: pull status mix for same-day-same-amt with n_is_dup=0
    soft_status = con.execute(
        """
        WITH g AS (
          SELECT company_id, CAST("date" AS DATE) AS d, amount
          FROM transactions
          GROUP BY 1, 2, 3
          HAVING count(*) > 1
             AND sum(CASE WHEN is_dup THEN 1 ELSE 0 END) = 0
        )
        SELECT t.status, count(*) AS n
        FROM transactions t
        JOIN g ON t.company_id = g.company_id
              AND CAST(t."date" AS DATE) = g.d
              AND t.amount = g.amount
        GROUP BY 1
        ORDER BY n DESC
        """
    ).df()
    # pending+booked pairs
    pb = con.execute(
        """
        WITH g AS (
          SELECT company_id, CAST("date" AS DATE) AS d, amount,
                 count(DISTINCT status) AS n_status,
                 count(*) AS n
          FROM transactions
          GROUP BY 1, 2, 3
          HAVING count(*) > 1
             AND sum(CASE WHEN is_dup THEN 1 ELSE 0 END) = 0
        )
        SELECT
          count(*) AS n_groups,
          sum(CASE WHEN n_status > 1 THEN 1 ELSE 0 END) AS n_multi_status
        FROM g
        """
    ).fetchone()
    print(
        f"pass11f soft_not_clone n={n_soft:,} n=2={n_2:,} multi_status={n_status:,} "
        f"multi_prod={n_prod:,} multi_desc={n_desc:,} groups={pb[0]} multi_st={pb[1]}"
    )
    print(f"  status mix: {soft_status.to_dict('records')[:8]}")

    # --- 11g why f_ds_r unchanged ---
    # The 2 train op_in extremes are COMP_0629. ds3 on those CMs?
    ext_in = p3["tx"]
    ext_in = ext_in[(ext_in["split"] == "train") & (ext_in["grp"] == "op_in")]
    fds_why = []
    for rec in ext_in.itertuples(index=False):
        per = pd.Timestamp(rec.date).replace(day=1)
        row = panel[(panel["company_id"] == rec.company_id) & (panel["period"] == per)]
        fds = float(row["f_ds_r"].iloc[0]) if len(row) and pd.notna(row["f_ds_r"].iloc[0]) else float("nan")
        ain = float(row["a_op_in"].iloc[0]) if len(row) else float("nan")
        fds_why.append(
            {
                "company": rec.company_id,
                "date": str(pd.Timestamp(rec.date).date()),
                "amount": float(rec.amount),
                "a_op_in": ain,
                "f_ds_r": fds,
            }
        )
    print(f"pass11g op_in extremes vs f_ds_r: {fds_why}")

    # --- 11h PDI as Y gate (design note only) ---
    trp = panel[panel["split"] == "train"]
    pdi_base = float((trp["has_pdi"] == 1).mean()) if "has_pdi" in trp.columns else float("nan")
    pdi_size = float(auroc((trp["has_pdi"] == 1).astype(float), trp["log_in3"])) if "has_pdi" in trp.columns else float("nan")
    pdi_y_note = (
        pdi_base >= 0.05
        and pdi_base <= 0.30
        and np.isfinite(pdi_size)
        and pdi_size < 0.60
    )
    print(f"pass11h PDI CM base={_pp(pdi_base)} size_auc={_f(pdi_size)} gate={pdi_y_note} (design note only)")

    # --- 11i cross-flag overlap ---
    xflag = con.execute(
        """
        SELECT
          sum(CASE WHEN is_extreme AND is_dup THEN 1 ELSE 0 END) AS ext_dup,
          sum(CASE WHEN is_extreme AND NOT product_known THEN 1 ELSE 0 END) AS ext_unk,
          sum(CASE WHEN is_dup AND NOT product_known THEN 1 ELSE 0 END) AS dup_unk
        FROM transactions
        """
    ).fetchone()
    xinv = con.execute(
        """
        SELECT
          sum(CASE WHEN is_extreme AND payment_date_invalid THEN 1 ELSE 0 END) AS ext_pdi,
          sum(CASE WHEN is_extreme AND payment_date_invalid THEN abs(amount) ELSE 0 END) AS ext_pdi_mass
        FROM invoices
        """
    ).fetchone()
    print(f"pass11i tx overlap ext∩dup={xflag[0]} ext∩unk={xflag[1]} dup∩unk={xflag[2]}")
    print(f"         inv ext∩pdi={xinv[0]} mass={xinv[1]:.3e}")

    # --- 12a HHI level change on defined (often unlabeled) windows ---
    hhi_delta = []
    for cid in sorted(set(in_hhi["company_id"])):
        sub = tr[(tr["company_id"] == cid) & tr["d_cust_hhi"].notna()].copy()
        if sub.empty:
            continue
        inv_c = con.execute(
            f"""
            SELECT CAST(issuance_date AS DATE) AS iss, counterparty_id,
                   abs(amount) AS amt, is_extreme
            FROM invoices
            WHERE {BOOK} AND amount > 0 AND counterparty_id IS NOT NULL
              AND company_id = ?
            """,
            [cid],
        ).df()
        inv_c["iss"] = pd.to_datetime(inv_c["iss"])
        dvals = []
        for r in sub.itertuples(index=False):
            end = pd.Timestamp(r.period) + pd.offsets.MonthEnd(0)
            start = (pd.Timestamp(r.period) - pd.DateOffset(months=5)).normalize()
            win = inv_c[(inv_c["iss"] >= start) & (inv_c["iss"] <= end)]
            if win.empty or float(win["amt"].sum()) <= 0:
                continue
            sh = win.groupby("counterparty_id")["amt"].sum()
            h_keep = float(((sh / sh.sum()) ** 2).sum())
            w2 = win[~win["is_extreme"].astype(bool)]
            if w2.empty or float(w2["amt"].sum()) <= 0:
                h_drop = float("nan")
            else:
                sh2 = w2.groupby("counterparty_id")["amt"].sum()
                h_drop = float(((sh2 / sh2.sum()) ** 2).sum())
            dvals.append(abs(h_keep - h_drop) if np.isfinite(h_drop) else float("nan"))
        hhi_delta.append(
            {
                "company": cid,
                "n_defined": int(len(sub)),
                "n_changed": int(sum(d > 1e-12 for d in dvals if np.isfinite(d))),
                "max_abs_d": float(np.nanmax(dvals)) if dvals else float("nan"),
                "mean_abs_d": float(np.nanmean(dvals)) if dvals else float("nan"),
            }
        )
    print(f"pass12a HHI level Δ (defined windows, not Y4): {hhi_delta}")

    # --- 12b COMP_0611 whale ---
    whale_detail = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               CAST("date" AS DATE) AS date, amount, category,
               product_id, counterparty_id, status,
               count(*) AS n, count(DISTINCT transaction_id) AS n_id,
               count(DISTINCT coalesce(description,'')) AS n_desc
        FROM transactions
        WHERE company_id = 'COMP_0611' AND CAST("date" AS DATE) = DATE '2026-03-31'
          AND amount IN (20.00, 100.00) AND category = 'collection'
        GROUP BY 1,2,3,4,5,6,7
        ORDER BY n DESC
        """
    ).df()
    print("pass12b COMP_0611 whale groups")
    print(whale_detail.head(8).to_string(index=False))

    # --- 12c leftover unpaired extreme invoices + gross vs net mass ---
    inv_abs = float(inv["abs_amt"].sum())
    tot_inv_mass = float(
        con.execute("SELECT sum(abs(amount)) FROM invoices").fetchone()[0]
    )
    leftover_abs = float(sum(abs(r.amount) for r in leftover))
    print(
        f"pass12c extreme gross={inv_abs:.3e} leftover={leftover_abs:.3e} "
        f"all_inv={tot_inv_mass:.3e} leftover_share={_pp(_pct(leftover_abs, tot_inv_mass))}"
    )

    # --- 12d PDI without the one extreme row ---
    pdi_wo = con.execute(
        """
        SELECT count(*) AS n, sum(abs(amount)) AS mass
        FROM invoices
        WHERE payment_date_invalid AND NOT is_extreme
        """
    ).fetchone()
    print(f"pass12d PDI without extremes n={pdi_wo[0]:,} mass={pdi_wo[1]:.3e}")

    # --- 12e is_dup vs a_n_tx (row-SIZE not amount-SIZE) ---
    dup_vs_ntx = float(auroc((trp["has_tx_dup"] == 1).astype(float), trp["a_n_tx"])) if "has_tx_dup" in trp.columns else float("nan")
    print(f"pass12e has_tx_dup vs a_n_tx AUROC={_f(dup_vs_ntx)} (vs log_in3 {p5['size_auc']})")

    return {
        "pairs": pair_rows,
        "washes": washes,
        "leftover_n": len(leftover),
        "n_hhi_eligible": int(len(in_hhi)),
        "pdi": pdi,
        "july_mass": july_mass,
        "july_ext_mass": july_ext_mass,
        "tot_pdi_mass": tot_pdi_mass,
        "pdi_co": pdi_co,
        "gap_rows": gap_rows,
        "cash_gap": cash_gap,
        "hhi_hits": hhi_hits,
        "whale": whale,
        "soft_n": n_soft,
        "soft_n2": n_2,
        "soft_multi_status": n_status,
        "soft_multi_prod": n_prod,
        "soft_multi_desc": n_desc,
        "soft_status": soft_status,
        "soft_pb": {"n_groups": int(pb[0]), "n_multi_status": int(pb[1])},
        "fds_why": fds_why,
        "pdi_base": pdi_base,
        "pdi_size": pdi_size,
        "pdi_y_note": pdi_y_note,
        "xflag": {"ext_dup": int(xflag[0]), "ext_unk": int(xflag[1]), "dup_unk": int(xflag[2])},
        "xinv": {"ext_pdi": int(xinv[0]), "ext_pdi_mass": float(xinv[1])},
        "hhi_delta": hhi_delta,
        "whale_detail": whale_detail,
        "ext_gross": inv_abs,
        "ext_leftover": leftover_abs,
        "all_inv_mass": tot_inv_mass,
        "pdi_wo_n": int(pdi_wo[0]),
        "pdi_wo_mass": float(pdi_wo[1]),
        "dup_vs_ntx": dup_vs_ntx,
        "dup_size_in3": p5["size_auc"],
        "p13": _pass13(con, panel, p3, tr),
        "p14": _pass14(con, panel),
        "p15": _pass15(con, panel),
        "p16": _pass16(con, panel),
        "p17": _pass17(con, panel),
        "p18": _pass18(con, panel),
        "p19": _pass19(con, panel),
        "p20": _pass20(con),
        "p21": _pass21(con),
        "p22": _pass22(con, panel),
        "p23": _pass23(panel),
        "p24": _pass24(con, panel),
        "p25": _pass25(con, panel),
        "p26": _pass26(con),
        "p27": _pass27(con),
        "p28": _pass28(p3),
        "p29": _pass29(p3),
        "p30": _pass30(p3),
        "p31": _pass31(p3),
        "p32": _pass32(p3),
        "p33": _pass33(con, p3),
        "p34": _pass34(con, p3),
    }


def write_png(p1: dict) -> None:
    if not HAS_MPL:
        print("no matplotlib — skip PNG")
        return
    # Train amount-mass + row share for amount-bearing flags.
    want = [
        ("transactions", "is_extreme"),
        ("transactions", "is_dup"),
        ("transactions", "product_known=false"),
        ("invoices", "is_extreme"),
        ("invoices", "payment_date_invalid"),
    ]
    labels, row_s, mass_s = [], [], []
    for table, flag in want:
        recs = [r for r in p1["rows"] if r["table"] == table and r["flag"] == flag and r["split"] == "train"]
        if not recs:
            continue
        r = recs[0]
        labels.append(f"{table[:3]} {flag}")
        row_s.append(100.0 * r["row_share"] if np.isfinite(r["row_share"]) else 0.0)
        mass_s.append(100.0 * r["mass_share"] if np.isfinite(r["mass_share"]) else 0.0)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    y = np.arange(len(labels))
    h = 0.38
    ax.barh(y + h / 2, row_s, h, label="row share %", color="#4C78A8")
    ax.barh(y - h / 2, mass_s, h, label="|amount| mass %", color="#F58518")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("train share of table (%)")
    ax.set_title("Clean flags — row share vs |amount| mass (train)\ninv is_extreme 52% is GROSS of 3 wash pairs")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


def _p11_md(p11: dict) -> str:
    pair_tbl = md_table(
        [
            {
                "company": r["company"],
                "n": r["n"],
                "gross": _sci(r["gross"]),
                "net": _sci(r["net"]),
                "types": r["types"],
                "statuses": r["statuses"],
            }
            for r in p11["pairs"]
        ],
        [
            ("company", ""),
            ("n", "right"),
            ("gross", "right"),
            ("net", "right"),
            ("types", ""),
            ("statuses", ""),
        ],
    )
    pdi_top = md_table(
        [
            {
                "company": rec["company_id"],
                "n": f"{int(rec['n']):,}",
                "|amt|": _sci(rec["mass"]),
                "n_ext": int(rec["n_ext"]),
            }
            for rec in p11["pdi_co"].to_dict("records")
        ],
        [("company", ""), ("n", "right"), ("|amt|", "right"), ("n_ext", "right")],
    )
    gap_tbl = md_table(
        [
            {
                "company": r["company"],
                "cash products": r["n_cash"],
                "used": r["n_used"],
                "sentinel": r["n_sent"],
                "snapshot used": _sci(r["snap_used"]),
                "raw sentinel": _sci(r["raw_sent"]),
            }
            for r in p11["gap_rows"]
        ],
        [
            ("company", ""),
            ("cash products", "right"),
            ("used", "right"),
            ("sentinel", "right"),
            ("snapshot used", "right"),
            ("raw sentinel", "right"),
        ],
    )
    hhi_tbl = md_table(
        [
            {
                "company": r["company"],
                "iss": r["iss"],
                "amount": _sci(r["amount"]),
                "defined HHI CM": r["defined_hhi_cm"],
                "Y4 labeled CM": r["y4_labeled_cm"],
            }
            for r in p11["hhi_hits"]
        ],
        [
            ("company", ""),
            ("iss", ""),
            ("amount", "right"),
            ("defined HHI CM", "right"),
            ("Y4 labeled CM", "right"),
        ],
    ) if p11["hhi_hits"] else "(none HHI-eligible)"
    whale_tbl = md_table(
        [
            {
                "company": rec["company_id"],
                "split": rec["split"],
                "date": str(rec["date"]),
                "cat": rec["category"],
                "n copies": int(rec["n_copies"]),
                "amount": _sci(rec["amount"]),
            }
            for rec in p11["whale"].to_dict("records")
        ],
        [
            ("company", ""),
            ("split", ""),
            ("date", ""),
            ("cat", ""),
            ("n copies", "right"),
            ("amount", "right"),
        ],
    )
    fds_tbl = md_table(
        [
            {
                "company": r["company"],
                "date": r["date"],
                "extreme": _sci(r["amount"]),
                "a_op_in": _sci(r["a_op_in"]),
                "f_ds_r": _f(r["f_ds_r"]),
            }
            for r in p11["fds_why"]
        ],
        [
            ("company", ""),
            ("date", ""),
            ("extreme", "right"),
            ("a_op_in", "right"),
            ("f_ds_r", "right"),
        ],
    ) if p11["fds_why"] else "(no train op_in extremes)"
    st = p11["soft_status"]
    st_tbl = md_table(
        [{"status": rec["status"], "n txs": f"{int(rec['n']):,}"} for rec in st.to_dict("records")[:8]],
        [("status", ""), ("n txs", "right")],
    ) if len(st) else "(none)"
    return "\n".join(
        [
            "### 11a — extreme invoices are mostly washes",
            "",
            f"Opposite-sign same-|amount| pairs: **{p11['washes']}**. Leftover unpaired extremes: {p11['leftover_n']}. "
            f"HHI-eligible (book invoice, not cancel, amount>0, has CP): **{p11['n_hhi_eligible']}**. "
            "The 52% invoice amount-mass is a **gross** figure — COMP_0629's ±6.24e10 invoice/note pair nets toward 0. "
            "That is why issued_lag1 / HHI AUROC did not move.",
            "",
            pair_tbl,
            "",
            "### 11b — `payment_date_invalid` July pile",
            "",
            f"July 2026 is {_pp(_pct(p11['july_mass'], p11['tot_pdi_mass']))} of PDI |amount| mass "
            f"({_sci(p11['july_mass'])} / {_sci(p11['tot_pdi_mass'])}). "
            f"Of that July mass, extremes are {_sci(p11['july_ext_mass'])} "
            f"({_pp(_pct(p11['july_ext_mass'], p11['july_mass']))}). "
            f"Invoice ext∩PDI rows={p11['xinv']['ext_pdi']}, mass={_sci(p11['xinv']['ext_pdi_mass'])}.",
            "",
            "Top PDI companies by |amount|:",
            "",
            pdi_top,
            "",
            "### 11c — Family B snapshot gap on the two sentinel companies",
            "",
            gap_tbl,
            "",
            "Walk **exists** (other used cash products). The omit is silent: COMP_1068's +1e11 raw checking "
            "never enters `_cash_balances`. Do not rewrite `liquidity.py` here — report only.",
            "",
            "### 11d — do extreme AR invoices sit in a defined HHI window?",
            "",
            hhi_tbl,
            "",
            "Defined HHI windows **exist** (2–6 CM). Y4-labeled CM on those windows = **0**, "
            "and pass 13 shows these companies are never Y4-labeled at all. "
            "Dropping extremes cannot move `d_cust_hhi_lag3` 0.605. That matches pass 9 n_changed=0.",
            "",
            "### 11e — clone whale (max copies)",
            "",
            whale_tbl,
            "",
            "### 11f — soft same-day-same-amount that are **not** `is_dup`",
            "",
            f"Groups={p11['soft_n']:,} (n=2: {p11['soft_n2']:,}). "
            f"Multi-status={p11['soft_multi_status']:,}; multi-product={p11['soft_multi_prod']:,}; "
            f"multi-description={p11['soft_multi_desc']:,}. "
            "These are a different booking (status/product/description differs) — not extract clones. "
            "The clean key is right to leave them unflagged as `is_dup`.",
            "",
            st_tbl,
            "",
            "### 11g — why `f_ds_r` does not move",
            "",
            "Train op_in extremes (cashflow keep, debt drop). Debt-service totals were identical "
            "(no extreme in `debt_repayment`). If `f_ds_r` is 0 on those months, the denominator change does not move the ratio.",
            "",
            fds_tbl,
            "",
            "### 11h — PDI as a health Y? Design note only",
            "",
            f"Train CM `has_pdi` base={_pp(p11['pdi_base'])} (in 5–30%? "
            f"{'yes' if 0.05 <= p11['pdi_base'] <= 0.30 else 'no'}). "
            f"Size AUROC=`log1p(a_in3)` {_f(p11['pdi_size'])} "
            f"({'<0.60' if np.isfinite(p11['pdi_size']) and p11['pdi_size'] < 0.60 else 'SIZE or NA'}). "
            f"Numeric gate {'PASS' if p11['pdi_y_note'] else 'FAIL'}. "
            "**PARK anyway** — invalid payment dates are DQ, July-mass piled, Family E already drops them from timing. "
            "No assembler. No new Y file.",
            "",
            "### 11i — cross-flag overlap",
            "",
            f"tx ext∩dup={p11['xflag']['ext_dup']}; ext∩unk={p11['xflag']['ext_unk']}; "
            f"dup∩unk={p11['xflag']['dup_unk']}. "
            f"inv ext∩PDI={p11['xinv']['ext_pdi']} mass={_sci(p11['xinv']['ext_pdi_mass'])}.",
            "",
            "### 12a — HHI *levels* on defined windows (not the Y4 quote)",
            "",
            md_table(
                [
                    {
                        "company": r["company"],
                        "defined CM": r["n_defined"],
                        "HHI changed": r["n_changed"],
                        "max |Δ|": _f(r["max_abs_d"]),
                        "mean |Δ|": _f(r["mean_abs_d"]),
                    }
                    for r in p11.get("hhi_delta", [])
                ],
                [
                    ("company", ""),
                    ("defined CM", "right"),
                    ("HHI changed", "right"),
                    ("max |Δ|", "right"),
                    ("mean |Δ|", "right"),
                ],
            )
            if p11.get("hhi_delta")
            else "(none)",
            "",
            "HHI **levels** can move on unlabeled months. The quoted single is AUROC vs `y4_ds_r_double`; "
            "those labeled rows do not overlap the extreme-AR windows (11d). Quote stays.",
            "",
            "### 12b — COMP_0611 clone whale",
            "",
            md_table(
                [
                    {
                        "company": rec["company_id"],
                        "date": str(rec["date"]),
                        "amount": _sci(rec["amount"]),
                        "product": rec["product_id"],
                        "cp": rec["counterparty_id"],
                        "n": int(rec["n"]),
                        "n_id": int(rec["n_id"]),
                        "n_desc": int(rec["n_desc"]),
                    }
                    for rec in p11.get("whale_detail", pd.DataFrame()).to_dict("records")[:6]
                ],
                [
                    ("company", ""),
                    ("date", ""),
                    ("amount", "right"),
                    ("product", ""),
                    ("cp", ""),
                    ("n", "right"),
                    ("n_id", "right"),
                    ("n_desc", "right"),
                ],
            )
            if p11.get("whale_detail") is not None and len(p11.get("whale_detail"))
            else "(none)",
            "",
            "Same product, same CP, same description, distinct `transaction_id` — extract clones, not a second booking.",
            "",
            "### 12c — leftover unpaired extremes (the real amount-mass after washes)",
            "",
            f"Extreme |amount| gross={_sci(p11.get('ext_gross'))} "
            f"({_pp(_pct(p11.get('ext_gross', 0), p11.get('all_inv_mass', 1)))} of all invoices). "
            f"After 3 wash pairs, leftover unpaired |amount|={_sci(p11.get('ext_leftover'))} "
            f"({_pp(_pct(p11.get('ext_leftover', 0), p11.get('all_inv_mass', 1)))} of all invoices). "
            "Quote the 52% as **gross**; net leftover is the 4 unpaired rows.",
            "",
            "### 12d — PDI without the one extreme row",
            "",
            f"PDI rows excluding `is_extreme`: {p11.get('pdi_wo_n', 0):,} ; "
            f"|amount|={_sci(p11.get('pdi_wo_mass'))}. "
            "The July 96% pile is almost entirely COMP_0629's −6.24e10 paid invoice "
            "(also `is_extreme`). Residual PDI is ordinary invalid dates, not a second giant.",
            "",
            "### 12e — `is_dup` is row-SIZE",
            "",
            f"Train CM `has_tx_dup` vs `a_n_tx` AUROC={_f(p11.get('dup_vs_ntx'))} "
            f"(vs `log1p(a_in3)` {_f(p11.get('dup_size_in3'))}). "
            "More txs → more clones. **SIZE**. CLOSE as X. Base 36.1% is outside 5–30% anyway. PARK as Y.",
            "",
            "### 13 — holdout extremes + Y4 ever-label on leftover companies",
            "",
            (
                f"Holdout extreme txs={p11.get('p13', {}).get('ho_n', 0)} "
                f"(COMP_0900 + COMP_0276, uncategorized). "
                f"Opposite-sign pairs={p11.get('p13', {}).get('ho_washes', 0)}; "
                f"unpaired={p11.get('p13', {}).get('ho_leftover', 0)}. "
                "33% of holdout tx |amount| is this DQ pile — coverage only; do not fit on it."
            ),
            "",
            md_table(
                [
                    {
                        "company": cid,
                        "CM": v["n_cm"],
                        "Y4 labeled": v["y4_lab"],
                        "Y4 pos": v["y4_pos"],
                        "Y3 labeled": v["y3_lab"],
                    }
                    for cid, v in p11.get("p13", {}).get("y4_ever", {}).items()
                ],
                [
                    ("company", ""),
                    ("CM", "right"),
                    ("Y4 labeled", "right"),
                    ("Y4 pos", "right"),
                    ("Y3 labeled", "right"),
                ],
            )
            if p11.get("p13", {}).get("y4_ever")
            else "(none)",
            "",
            f"Unknown-product IDs span {p11.get('p13', {}).get('unk_first')} → {p11.get('p13', {}).get('unk_last')} "
            f"({p11.get('p13', {}).get('unk_n_months')} months, {p11.get('p13', {}).get('unk_n_prod')} products) "
            "— a persistent missing-ID hole, not a first-month-only G artifact.",
            "",
            "### 14 — residual PDI is spread; `has_pdi` as X is CLOSE",
            "",
            f"PDI excluding the one extreme: {p11.get('p14', {}).get('n_months', 0)} months. "
            f"Max-month rows {p11.get('p14', {}).get('max_row_month')} = {_pp(p11.get('p14', {}).get('max_row_share'))}. "
            f"Max-month mass {p11.get('p14', {}).get('max_mass_month')} = {_pp(p11.get('p14', {}).get('max_mass_share'))}. "
            "Without COMP_0629's giant, invalid payment dates are a **spread** DQ, not a one-month event.",
            "",
            md_table(
                [
                    {
                        "Y": r["y"],
                        "CV AUROC": "LOW_POWER" if r["low_power"] else _f(r["cv"]),
                        "n": f"{r['n']:,}",
                        "n_pos": r["n_pos"],
                    }
                    for r in p11.get("p14", {}).get("x_rows", [])
                ],
                [("Y", ""), ("CV AUROC", "right"), ("n", "right"), ("n_pos", "right")],
            )
            if p11.get("p14", {}).get("x_rows")
            else "(none)",
            "",
            "`has_pdi` as X vs accepted Ys: do not KEEP. Even if a CV is above 0.55 it is a DQ bit "
            "(Family E already drops invalid dates from timing). **CLOSE as X. PARK as Y.**",
            "",
            "### 15 — dropping `is_dup` vs the Y3 15-col (`a_n_tx` / days)",
            "",
            (
                f"First copy stays (`is_dup` is rn>1), so `c_n_days_with_tx` is unchanged "
                f"(days vanish={p11.get('p15', {}).get('days_vanish', 0)}). "
                f"`a_n_tx` keep CV={_f(p11.get('p15', {}).get('ntx_keep'))} "
                f"drop-dup CV={_f(p11.get('p15', {}).get('ntx_drop'))} "
                f"Δ={_f(p11.get('p15', {}).get('ntx_delta'))} "
                f"(changed CM={p11.get('p15', {}).get('n_changed', 0)}). "
                f"{'MOVES ≥0.02' if p11.get('p15', {}).get('moves') else 'Does not move ≥0.02'} — "
                "never-drop still **KEEP** for clones as well as extremes."
            ),
            "",
            "### 16 — Family E PDI on issued (open/delay already drop it)",
            "",
            (
                f"`e_ar_issued_lag1` if PDI rows are also dropped from issued: "
                f"keep={_f(p11.get('p16', {}).get('keep'))} "
                f"drop-PDI={_f(p11.get('p16', {}).get('drop'))} "
                f"Δ={_f(p11.get('p16', {}).get('delta'))} "
                f"changed CM={p11.get('p16', {}).get('n_changed', 0)}. "
                f"{'MOVES ≥0.02 — would justify unifying the E filter' if p11.get('p16', {}).get('moves') else 'Does not move ≥0.02 — do not patch invoices.py'}."
            ),
            "",
            "### 17 — unknown-product IDs vs unknown balances",
            "",
            (
                f"Unknown tx `product_id`={p11.get('p17', {}).get('n_tx_unk')} ; "
                f"unknown balance `product_id`={p11.get('p17', {}).get('n_bal_unk')} ; "
                f"intersection={p11.get('p17', {}).get('n_both')}. "
                f"Train companies with unk txs whose last-month `g_n_accounts`>0: "
                f"{p11.get('p17', {}).get('n_later_g')}/{p11.get('p17', {}).get('n_unk_cos_last')} "
                "— they get a real book later; the unknown IDs never join it."
            ),
            "",
            md_table(
                [
                    {
                        "category": rec["category"],
                        "n": f"{int(rec['n']):,}",
                        "|amt|": _sci(rec["mass"]),
                        "cos": int(rec["n_cos"]),
                    }
                    for rec in p11.get("p17", {}).get("cat", pd.DataFrame()).to_dict("records")[:8]
                ],
                [("category", ""), ("n", "right"), ("|amt|", "right"), ("cos", "right")],
            )
            if p11.get("p17", {}).get("cat") is not None and len(p11.get("p17", {}).get("cat", []))
            else "(none)",
            "",
            "### 18 — `created_after_snapshot` products still have a pre-extract book?",
            "",
            md_table(
                [
                    {
                        "book": rec["book"],
                        "products": int(rec["n_prod"]),
                        "pre-extract txs": int(rec["n_tx"]) if pd.notna(rec["n_tx"]) else 0,
                        "cos": int(rec["n_cos"]) if pd.notna(rec["n_cos"]) else 0,
                    }
                    for rec in p11.get("p18", {}).get("pre", pd.DataFrame()).to_dict("records")
                ],
                [
                    ("book", ""),
                    ("products", "right"),
                    ("pre-extract txs", "right"),
                    ("cos", "right"),
                ],
            )
            if p11.get("p18", {}).get("pre") is not None and len(p11.get("p18", {}).get("pre", []))
            else "(none)",
            "",
            f"Last-month train `f_outstanding_gt_granted` nn={p11.get('p18', {}).get('ogtg_nn')} "
            f"mean={_f(p11.get('p18', {}).get('ogtg_mean'))} "
            f"(Family F already NaNs this before last month). "
            f"`g_created_after_snapshot` max on last-month train={_f(p11.get('p18', {}).get('g_after_max'))} "
            "(G said 0 on every train CM — confirm).",
            "",
            "### 19 — those post-snapshot products still flow on the train panel",
            "",
            (
                f"Train `a_op_in` sitting on `created_after_snapshot` banking products: "
                f"{_sci(p11.get('p19', {}).get('op_in_after'))} / {_sci(p11.get('p19', {}).get('op_in_all'))} "
                f"= {_pp(p11.get('p19', {}).get('rel'), 4)} "
                f"({p11.get('p19', {}).get('n_cm', 0)} CM). "
                "G excludes the product from inventory; cashflow keeps the txs (`product_known` is true). "
                "Report only — do not patch G or A."
            ),
            "",
            "### 20 — extreme companies: share of *their own* tx |amount|",
            "",
            md_table(
                [
                    {
                        "company": rec["company_id"],
                        "split": rec["split"],
                        "n ext / n tx": f"{int(rec['n_ext'])}/{int(rec['n_tx'])}",
                        "own |amt| share": _pp(rec["mass_share"], 1),
                    }
                    for rec in p11.get("p20", {}).get("rows", pd.DataFrame()).to_dict("records")
                ],
                [
                    ("company", ""),
                    ("split", ""),
                    ("n ext / n tx", "right"),
                    ("own |amt| share", "right"),
                ],
            )
            if p11.get("p20", {}).get("rows") is not None and len(p11.get("p20", {}).get("rows", []))
            else "(none)",
            "",
            "### 21 — extreme txs on Family B cash products?",
            "",
            (
                f"{p11.get('p21', {}).get('n_cash', 0)} / {p11.get('p21', {}).get('n', 0)} "
                f"extreme txs sit on checking/saving/tpv. "
                f"Bank types: {p11.get('p21', {}).get('bank_types', {})}. "
                "`liquidity.py` `_product_flows` has **no** `is_extreme` filter. "
                "If cash-type count > 0, those amounts walk the snapshot backward. "
                "Holdout COMP_0900 is coverage-only; train cash-type extremes would bias `b_liq`."
            ),
            "",
            "### 22 — estimated `b_liq` shift from those checking extremes",
            "",
            "Identity: `end_bal_t = snapshot − sum(flows after t)`. "
            "Drop extremes → `alt_liq_t = b_liq_t + sum(extreme checking flow after t)`. "
            "Table `max |shift|` is `|sum(extreme after t)|`. Report only.",
            "",
            md_table(
                [
                    {
                        "company": rec["company"],
                        "split": rec["split"],
                        "ext months": rec.get("n_months", ""),
                        "sum ext flow": _sci(rec.get("sum_extreme_flow")),
                        "max |shift|": _sci(rec.get("max_abs_shift")),
                        "b_liq p50": _sci(rec.get("liq_p50")),
                    }
                    for rec in p11.get("p22", {}).get("rows", pd.DataFrame()).to_dict("records")
                ],
                [
                    ("company", ""),
                    ("split", ""),
                    ("ext months", "right"),
                    ("sum ext flow", "right"),
                    ("max |shift|", "right"),
                    ("b_liq p50", "right"),
                ],
            )
            if p11.get("p22", {}).get("rows") is not None and len(p11.get("p22", {}).get("rows", []))
            else "(none)",
            "",
            "### 23 — Y2 on those four train companies",
            "",
            (
                f"Y2 labeled CM on COMP_0629 / 0306 / 0487 / 1192: "
                f"{p11.get('p23', {}).get('n_lab_on', 0)} "
                f"(pos={p11.get('p23', {}).get('n_pos_on', 0)}, "
                f"rate={_pp(p11.get('p23', {}).get('rate_on'))}) "
                f"vs rest of train {_pp(p11.get('p23', {}).get('rate_off'))}. "
                "Y2 already PARK as a model. This is a B-walk footnote, not a new Y."
            ),
            "",
            "### 24 — Y2 flips if the B walk dropped those checking extremes",
            "",
            "Identity (corrected): drop extremes → `alt_liq_t = b_liq_t + sum(extreme cash-type flow after t)`. "
            "Y2 rebuilt from `alt_liq` with the same 3-month horizon. Report only.",
            "",
            md_table(
                [
                    {
                        "company": rec["company"],
                        "split": rec["split"],
                        "labeled": rec.get("n_lab", ""),
                        "Y2 flips": rec.get("n_flip", ""),
                        "store pos": rec.get("store_pos", ""),
                        "alt pos": rec.get("alt_pos", ""),
                        "liq sign flips": rec.get("n_sign_flip", ""),
                    }
                    for rec in p11.get("p24", {}).get("rows", pd.DataFrame()).to_dict("records")
                ],
                [
                    ("company", ""),
                    ("split", ""),
                    ("labeled", "right"),
                    ("Y2 flips", "right"),
                    ("store pos", "right"),
                    ("alt pos", "right"),
                    ("liq sign flips", "right"),
                ],
            )
            if p11.get("p24", {}).get("rows") is not None and len(p11.get("p24", {}).get("rows", []))
            else "(none)",
            "",
            (
                f"Train: {p11.get('p24', {}).get('n_flip_train', 0)} Y2 flips / "
                f"{p11.get('p24', {}).get('n_lab_train', 0)} labeled "
                f"(store pos={p11.get('p24', {}).get('n_pos_store', 0)}, "
                f"alt pos={p11.get('p24', {}).get('n_pos_alt', 0)}; "
                f"{p11.get('p24', {}).get('n_sign_train', 0)} months change `b_liq` sign). "
                "Y2 is already PARK. Do not rewrite `liquidity.py` in this lane."
            ),
            "",
            "### 25 — COMP_0306 is the train flip (Y2 0→11)",
            "",
            md_table(
                [
                    {
                        "date": _d(r.get("date", "")),
                        "amount": _sci(r.get("amount")),
                        "category": r.get("category", ""),
                        "status": r.get("status", ""),
                        "bank type": r.get("bank_type", ""),
                    }
                    for r in p11.get("p25", {}).get("txs", pd.DataFrame()).to_dict("records")
                ],
                [
                    ("date", ""),
                    ("amount", "right"),
                    ("category", ""),
                    ("status", ""),
                    ("bank type", ""),
                ],
            )
            if p11.get("p25", {}).get("n_tx", 0)
            else "(none)",
            "",
            (
                f"{p11.get('p25', {}).get('n_sign', 0)} months change `b_liq` sign "
                f"({p11.get('p25', {}).get('n_new_neg', 0)} newly negative). "
                f"Store Y3 labeled={p11.get('p25', {}).get('y3_lab', 0)} "
                f"(pos={p11.get('p25', {}).get('y3_pos', 0)}). "
                f"Months with alt_liq<0 (would enter Y3 stressed via liq): "
                f"{p11.get('p25', {}).get('n_y3_eligible_alt', 0)}. "
                "Y3 already forbids family B as X. Footnote for Family B QA — no patch here."
            ),
            "",
            "### 26 — COMP_0306 extreme invoice vs the checking outflow",
            "",
            md_table(
                [
                    {
                        "inv iss": _d(r.get("inv_iss", "")),
                        "inv pay": _d(r.get("inv_pay", "")),
                        "inv amt": _sci(r.get("inv_amt")),
                        "type": r.get("inv_type", ""),
                        "status": r.get("inv_status", ""),
                        "tx date": _d(r.get("tx_date", "")),
                        "tx amt": _sci(r.get("tx_amt")),
                        "|Δamt|/|inv|": _f(r.get("amt_rel"), 4),
                        "days vs iss": r.get("days_iss", ""),
                    }
                    for r in p11.get("p26", {}).get("near", pd.DataFrame()).to_dict("records")
                ],
                [
                    ("inv iss", ""),
                    ("inv pay", ""),
                    ("inv amt", "right"),
                    ("type", ""),
                    ("status", ""),
                    ("tx date", ""),
                    ("tx amt", "right"),
                    ("|Δamt|/|inv|", "right"),
                    ("days vs iss", "right"),
                ],
            )
            if p11.get("p26", {}).get("n_pairs", 0)
            else "(no pair)",
            "",
            (
                f"Same-magnitude (|Δamt|/|inv|<2%): "
                f"{'yes' if p11.get('p26', {}).get('same_mag') else 'no'}. "
                "If yes, one economic event is double-counted across invoice HHI and the B walk; "
                "still KEEP never-drop (quoted singles unmoved). Do not redo join QA."
            ),
            "",
            "### 27 — same-magnitude extreme inv↔tx on companies that have both",
            "",
            (
                f"Companies with at least one extreme invoice **and** one extreme tx: "
                f"{p11.get('p27', {}).get('both', [])} (n={p11.get('p27', {}).get('n_both', 0)}). "
                f"Same-magnitude pairs (|Δamt|/|inv|<2%): {p11.get('p27', {}).get('n_pairs', 0)}."
            ),
            "",
            md_table(
                [
                    {
                        "company": r.get("company", ""),
                        "split": r.get("split", ""),
                        "inv iss": _d(r.get("inv_iss", "")),
                        "inv amt": _sci(r.get("inv_amt")),
                        "status": r.get("inv_status", ""),
                        "tx date": _d(r.get("tx_date", "")),
                        "tx amt": _sci(r.get("tx_amt")),
                        "cat": r.get("tx_cat", ""),
                        "|Δamt|/|inv|": _f(r.get("amt_rel"), 4),
                        "days": r.get("days", ""),
                    }
                    for r in p11.get("p27", {}).get("pairs", pd.DataFrame()).to_dict("records")
                ],
                [
                    ("company", ""),
                    ("split", ""),
                    ("inv iss", ""),
                    ("inv amt", "right"),
                    ("status", ""),
                    ("tx date", ""),
                    ("tx amt", "right"),
                    ("cat", ""),
                    ("|Δamt|/|inv|", "right"),
                    ("days", "right"),
                ],
            )
            if p11.get("p27", {}).get("n_pairs", 0)
            else "(none)",
            "",
            "### 28 — COMP_0629 extremes are two different books",
            "",
            (
                f"Extreme invoices={p11.get('p28', {}).get('n_inv', 0)}, "
                f"extreme txs={p11.get('p28', {}).get('n_tx', 0)}. "
                "No same-magnitude pair (pass 27). Washes live on invoices; "
                "checking `op_in` giants are a separate book. Y2 does not flip here "
                "(inflows raise `alt_liq` further above zero)."
            ),
            "",
            "### 29 — same-day opposite-sign extreme txs (net-zero on the B walk)",
            "",
            (
                f"Pairs: {p11.get('p29', {}).get('n_pairs', 0)}. "
                "COMP_0629 2025-02-18 ±1.694e9 is one; it nets out of `sum_extreme_flow` "
                "and does not move `alt_liq`. Unpaired leftovers are what pass 22/24 see."
            ),
            "",
            md_table(
                [
                    {
                        "company": r.get("company", ""),
                        "date": r.get("date", ""),
                        "n": r.get("n", ""),
                        "pos": _sci(r.get("pos")),
                        "neg": _sci(r.get("neg")),
                        "net": _sci(r.get("net")),
                        "cats": r.get("cats", ""),
                    }
                    for r in p11.get("p29", {}).get("pairs", pd.DataFrame()).to_dict("records")
                ],
                [
                    ("company", ""),
                    ("date", ""),
                    ("n", "right"),
                    ("pos", "right"),
                    ("neg", "right"),
                    ("net", "right"),
                    ("cats", ""),
                ],
            )
            if p11.get("p29", {}).get("n_pairs", 0)
            else "(none)",
            "",
            "### 30 — remaining train extreme-checking txs (0487 / 1192)",
            "",
            md_table(
                [
                    {
                        "company": r.get("company_id", ""),
                        "date": _d(r.get("date", "")),
                        "amount": _sci(r.get("amount")),
                        "category": r.get("category", ""),
                        "grp": r.get("grp", ""),
                    }
                    for r in p11.get("p30", {}).get("rows", pd.DataFrame()).to_dict("records")
                ],
                [
                    ("company", ""),
                    ("date", ""),
                    ("amount", "right"),
                    ("category", ""),
                    ("grp", ""),
                ],
            )
            if p11.get("p30", {}).get("n", 0)
            else "(none)",
            "",
            "Neither flips Y2 (pass 24). Inflows / leftover `other` do not push `alt_liq` below 0.",
            "",
            "### 31 — holdout extreme txs (coverage only)",
            "",
            md_table(
                [
                    {
                        "company": r.get("company_id", ""),
                        "date": _d(r.get("date", "")),
                        "amount": _sci(r.get("amount")),
                        "category": r.get("category", ""),
                        "grp": r.get("grp", ""),
                    }
                    for r in p11.get("p31", {}).get("rows", pd.DataFrame()).to_dict("records")
                ],
                [
                    ("company", ""),
                    ("date", ""),
                    ("amount", "right"),
                    ("category", ""),
                    ("grp", ""),
                ],
            )
            if p11.get("p31", {}).get("n", 0)
            else "(none)",
            "",
            "COMP_0900 / COMP_0276 are 33–37% of their own tx |amount|. Do not tune on them.",
            "",
            "### 32 — extreme-tx dates shared by more than one company",
            "",
            (
                f"Dates with extremes on >1 company: {p11.get('p32', {}).get('n_dates', 0)}. "
                "Holdout COMP_0900 and COMP_0276 share 2026-02-10 / 2026-02-13 "
                "(uncategorized giants). Coverage only — do not redo sibling H."
            ),
            "",
            "### 33 — extreme-tx companies and groups",
            "",
            md_table(
                [
                    {
                        "company": r.get("company_id", ""),
                        "group": r.get("group_id", ""),
                        "split": r.get("split", ""),
                    }
                    for r in p11.get("p33", {}).get("rows", pd.DataFrame()).to_dict("records")
                ],
                [("company", ""), ("group", ""), ("split", "")],
            )
            if p11.get("p33", {}).get("n", 0)
            else "(none)",
            "",
            (
                f"Groups that contain more than one extreme-tx company: "
                f"{p11.get('p33', {}).get('n_shared_groups', 0)}. "
                "Coverage fact. Do not redo sibling H."
            ),
            "",
            "### 34 — size of those two groups",
            "",
            md_table(
                [
                    {
                        "group": r.get("group_id", ""),
                        "n companies": r.get("n_cos", ""),
                    }
                    for r in p11.get("p34", {}).get("sizes", pd.DataFrame()).to_dict("records")
                ],
                [("group", ""), ("n companies", "right")],
            )
            if p11.get("p34", {}).get("n", 0)
            else "(none)",
            "",
            (
                f"Members listed: {p11.get('p34', {}).get('n', 0)}; "
                f"of which extreme-tx: {p11.get('p34', {}).get('n_ext', 0)}. "
                "GROUP_0094 has 13 train companies including extreme-invoice COMP_0163 "
                "plus extreme-tx COMP_0306 / COMP_1192. GROUP_0199 has 11 holdout companies; "
                "only 0900 / 0276 carry extreme txs. Coverage — do not redo sibling H."
            ),
        ]
    )


def write_md(ctx: dict) -> None:
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = (
        ctx["p1"],
        ctx["p2"],
        ctx["p3"],
        ctx["p4"],
        ctx["p5"],
        ctx["p6"],
        ctx["p7"],
        ctx["p8"],
        ctx["p9"],
        ctx["p10"],
    )
    if "p11" in ctx:
        ctx["p11_md"] = _p11_md(ctx["p11"])
    inv = p3["inv"]
    tx = p3["tx"]
    q = {r["name"]: r for r in p9["results"]}
    moves_any = any(r["moves"] for r in p9["results"])

    def prev_rows(split):
        out = []
        for r in p1["rows"]:
            if r["split"] != split:
                continue
            out.append(
                {
                    "table": r["table"],
                    "flag": r["flag"],
                    "n_flag": f"{r['n_flag']:,}",
                    "n_rows": f"{r['n_rows']:,}",
                    "row %": _pp(r["row_share"], 3),
                    "cos": f"{r['n_cos_flag']}/{r['n_cos']}",
                    "|amt| %": _pp(r["mass_share"], 4) if r["table"] not in ("banking_products",) else "—",
                }
            )
        return out

    y_rows = []
    for r in p4["rows"]:
        y_rows.append(
            {
                "flag": r["flag"],
                "train CM": f"{r['n_on']:,}",
                "CM %": _pp(r["share_cm"]),
                "y2 on/off": f"{_pp(r[f'{Y2}_rate_on'])} / {_pp(r[f'{Y2}_rate_off'])}",
                "y3 on/off": f"{_pp(r[f'{Y3}_rate_on'])} / {_pp(r[f'{Y3}_rate_off'])}",
                "y4 on/off": f"{_pp(r[f'{Y4}_rate_on'])} / {_pp(r[f'{Y4}_rate_off'])}",
                "y7 on/off": f"{_pp(r[f'{Y7}_rate_on'])} / {_pp(r[f'{Y7}_rate_off'])}",
                "y9 on/off": f"{_pp(r[f'{Y9}_rate_on'])} / {_pp(r[f'{Y9}_rate_off'])}",
                "size AUROC": _f(r["size_auc"]),
            }
        )

    cal_rows = []
    for r in p2["rows"]:
        if r["split"] != "train":
            continue
        cal_rows.append(
            {
                "table": r["table"],
                "flag": r["flag"],
                "n months": r["n_months"],
                "max month (rows)": r["max_month"],
                "max row %": _pp(r["max_row_share"]),
                "max month (mass)": r.get("max_mass_month", ""),
                "max mass %": _pp(r["max_mass_share"]),
            }
        )

    ever_rows = []
    for r in p10["rows"]:
        ever_rows.append(
            {
                "split": r["split"],
                "flag": r["flag"],
                "ever": f"{r['n_ever']}/{r['n_cos']}",
                "share": _pp(r["share_ever"]),
            }
        )

    inv_rows = []
    for rec in inv.to_dict("records"):
        inv_rows.append(
            {
                "company": rec["company_id"],
                "split": rec["split"],
                "iss": str(rec["issuance_date"]),
                "type": rec["document_type"],
                "status": rec["status"],
                "amount": _sci(rec["amount"]),
            }
        )
    tx_rows = []
    for rec in tx.to_dict("records"):
        tx_rows.append(
            {
                "company": rec["company_id"],
                "split": rec["split"],
                "date": str(rec["date"]),
                "cat": rec["category"],
                "grp": rec["grp"],
                "amount": _sci(rec["amount"]),
            }
        )

    q_rows = []
    for r in p9["results"]:
        q_rows.append(
            {
                "single": r["name"],
                "quote": _f(r["quote"]),
                "keep CV": _f(r["keep_cv"]),
                "drop-ext CV": _f(r["drop_cv"]),
                "Δ": _f(r["delta"]),
                "changed CM": r["n_changed"],
                "≥0.02?": "YES" if r["moves"] else "no",
            }
        )

    sent = p7["rows"]
    sent_rows = []
    for rec in sent.to_dict("records"):
        sent_rows.append(
            {
                "company": rec["company_id"],
                "split": rec["split"],
                "bank type": rec["bank_type"] if pd.notna(rec["bank_type"]) else "",
                "debt type": rec["debt_type"] if pd.notna(rec["debt_type"]) else "",
                "cash?": "yes" if rec["cash_type"] else "no",
                "raw balance": _sci(rec["raw_balance"]),
            }
        )

    # Verdicts
    tx_ext_all = next(r for r in p1["rows"] if r["table"] == "transactions" and r["flag"] == "is_extreme" and r["split"] == "all")
    inv_ext_all = next(r for r in p1["rows"] if r["table"] == "invoices" and r["flag"] == "is_extreme" and r["split"] == "all")
    tx_ext_tr = next(r for r in p1["rows"] if r["table"] == "transactions" and r["flag"] == "is_extreme" and r["split"] == "train")
    inv_ext_tr = next(r for r in p1["rows"] if r["table"] == "invoices" and r["flag"] == "is_extreme" and r["split"] == "train")
    dup_tr = next(r for r in p1["rows"] if r["table"] == "transactions" and r["flag"] == "is_dup" and r["split"] == "train")
    unk_tr = next(r for r in p1["rows"] if r["table"] == "transactions" and r["flag"] == "product_known=false" and r["split"] == "train")

    drop_rule = "KEEP never-drop"
    if moves_any:
        drop_rule = "REVIEW never-drop — a quoted single moved ≥0.02"

    lines = [
        "# Clean flags QA — DQ, not health",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. No `build_targets`. Family modules not edited.",
        "",
        "Flags are the README keep-and-flag set: `is_dup`, `is_extreme` (|amount|≥1e9), "
        "`product_known`, `payment_date_invalid`, `created_after_snapshot`, "
        "`outstanding_gt_granted`, `balance_sentinel`.",
        "",
        "## Verdicts",
        "",
        "| probe | verdict | why |",
        "| --- | --- | --- |",
        f"| flags as health Y | **PARK** | They are DQ. Row bases are tiny (tx extreme {tx_ext_all['n_flag']}, "
        f"inv extreme {inv_ext_all['n_flag']}) or not a trajectory (snapshot sentinels / post-extract created_at). |",
        "| flags as X | **CLOSE** | Do not put DQ bits on the score path. Size/overlap below; none is a six-question signal. |",
        f"| never-drop extremes | **{drop_rule.split()[0]}** | {drop_rule}. "
        f"Quoted singles days/issued/HHI move ≥0.02 if extremes dropped: "
        f"{'YES' if moves_any else 'no'}. |",
        f"| product_known vs G 63.7% hole | **different hole** | First-month `g_n_accounts=0` train={_pp(p6['g0_share'])}. "
        f"Unknown-product share of first-month txs on those companies={_pp(p6['first_unk_g0'])} "
        f"(g>0={_pp(p6['first_unk_g1'])}). Unknown products in banking/debt books={p6['n_in_books']}/{p6['n_unk_prod']}. |",
        f"| Family B sentinel risk | **snapshot OK; flow walk sees extremes** | "
        f"Sentinels (3 checking) are NULLed — garbage does not seed the snapshot. "
        f"All 24 extreme txs are checking. COMP_0306: 11/18 Y2 flips if the walk dropped the −1.61e9 Aug-2026 outflow. |",
        "| debt exclude-extreme vs cashflow keep | **report only** | "
        f"Train `a_op_in` rel move if debt rule used={_pp(p8['rel_op_in'], 4)}. "
        f"`f_ds_r` |Δ| mean={_f(p8['mean_abs_fds'])} max={_f(p8['max_abs_fds'])}; "
        f"CM changed={p8['n_cm_fds_changed']}. Do not edit family modules. |",
        "",
        "### Numbers to quote (train unless noted)",
        "",
        f"- **tx is_extreme:** {tx_ext_all['n_flag']:,} / {tx_ext_all['n_rows']:,} rows "
        f"(train {tx_ext_tr['n_flag']:,} / {tx_ext_tr['n_rows']:,}); "
        f"amount mass train {_pp(tx_ext_tr['mass_share'], 4)}; "
        f"companies train {tx_ext_tr['n_cos_flag']} / holdout "
        f"{next(r['n_cos_flag'] for r in p1['rows'] if r['table']=='transactions' and r['flag']=='is_extreme' and r['split']=='holdout')}.",
        f"- **inv is_extreme:** {inv_ext_all['n_flag']:,} / {inv_ext_all['n_rows']:,} "
        f"(join QA said 10 / 896,711 — **CONFIRM**); "
        f"**gross** amount mass train {_pp(inv_ext_tr['mass_share'], 4)} — 3 wash pairs; "
        f"unpaired leftover is the real mass (pass 12c). All 10 are train (holdout 0 companies). "
        f"Train companies {inv_ext_tr['n_cos_flag']}.",
        f"- **tx is_dup:** train {dup_tr['n_flag']:,} / {dup_tr['n_rows']:,} ({_pp(dup_tr['row_share'])}), "
        f"amount mass {_pp(dup_tr['mass_share'], 3)}. Full-content extract clones, not a second booking with a new description.",
        f"- **product_known=false txs:** train {unk_tr['n_flag']:,} ({_pp(unk_tr['row_share'], 3)}), "
        f"mass {_pp(unk_tr['mass_share'], 4)} — not the G connection hole.",
        f"- **Quoted singles if drop extremes:** days keep { _f(q.get('days',{}).get('keep_cv')) } → drop { _f(q.get('days',{}).get('drop_cv')) }; "
        f"issued_lag1 { _f(q.get('issued_lag1',{}).get('keep_cv')) } → { _f(q.get('issued_lag1',{}).get('drop_cv')) }; "
        f"HHI_lag3 { _f(q.get('hhi_lag3',{}).get('keep_cv')) } → { _f(q.get('hhi_lag3',{}).get('drop_cv')) }. "
        f"Threshold ≥0.02: {'hit' if moves_any else 'not hit'}.",
        f"- **B-walk / Y2 footnote:** train COMP_0306 checking −1.613e9 (2026-08-14) same-magnitude as overdue invoice −1.616e9 (2025-12-31, 226d). "
        f"Drop-from-walk would flip 11/18 Y2 labels (0→11). Sibling COMP_1192 is GROUP_0094. Y2 already PARK. Do not rewrite `liquidity.py`. "
        f"Holdout extremes are GROUP_0199 (COMP_0900 + COMP_0276) on 2026-02-10 / 02-13.",
        "",
        "## Brief questions",
        "",
        "1. **Who is healthy?** — a DQ flag is not a health reading.",
        "2. **Who is improving?** — clones / sentinels / unknown products do not say 45→65.",
        "3. **Who is turning?** — `created_after_snapshot` is post-extract (Family G already 0 on the panel).",
        "4. **Dip vs fall?** — not these bits.",
        "5. **Why did it change?** — do not explain a score move with `is_dup`.",
        "6. **Months earlier?** — flags are not lead time.",
        "",
        "## Pass 1 — prevalence (n, companies, amount mass)",
        "",
        "Clean-table denominators (amount=0 already dropped). `dq_log` counts on raw; "
        "tx `is_extreme` 24 vs raw 24; inv 10 vs raw 10; `product_known=false` txs 1,313 vs raw 1,314 "
        "(the extra raw row was amount=0).",
        "",
        "### Train",
        "",
        md_table(
            prev_rows("train"),
            [
                ("table", ""),
                ("flag", ""),
                ("n_flag", "right"),
                ("n_rows", "right"),
                ("row %", "right"),
                ("cos", "right"),
                ("|amt| %", "right"),
            ],
        ),
        "",
        "### Holdout (coverage only)",
        "",
        md_table(
            prev_rows("holdout"),
            [
                ("table", ""),
                ("flag", ""),
                ("n_flag", "right"),
                ("n_rows", "right"),
                ("row %", "right"),
                ("cos", "right"),
                ("|amt| %", "right"),
            ],
        ),
        "",
        "Balances sentinels have `balance` NULLed in `clean`. Raw values:",
        "",
    ]
    if p1["sent_raw"].empty:
        lines.append("(none)")
    else:
        sr = []
        for rec in p1["sent_raw"].to_dict("records"):
            sr.append(
                {
                    "company": rec["company_id"],
                    "split": rec["split"],
                    "product": rec["product_id"],
                    "raw balance": _sci(rec["balance"]),
                }
            )
        lines += [
            md_table(
                sr,
                [
                    ("company", ""),
                    ("split", ""),
                    ("product", ""),
                    ("raw balance", "right"),
                ],
            )
        ]
    cat_rows = []
    if p5["cat"] is not None and len(p5["cat"]):
        for rec in p5["cat"].head(12).to_dict("records"):
            cat_rows.append(
                {
                    "category": rec["category"],
                    "groups": f"{int(rec['n_groups']):,}",
                    "extra copies": f"{int(rec['n_extra']):,}",
                    "|amt|": _sci(rec["mass"]),
                }
            )

    lines += [
        "",
        f"Plot: `{OUT_PNG.name}`.",
        "",
        "## Pass 2 — calendar concentration",
        "",
        "Train. One-month pile if max-month share is high; spread if many months and max share is low.",
        "",
        md_table(
            cal_rows,
            [
                ("table", ""),
                ("flag", ""),
                ("n months", "right"),
                ("max month (rows)", ""),
                ("max row %", "right"),
                ("max month (mass)", ""),
                ("max mass %", "right"),
            ],
        ),
        "",
        "## Pass 3 — extreme row cards",
        "",
        f"Invoices: **{len(inv)}** (train {int((inv['split']=='train').sum())} / "
        f"holdout {int((inv['split']=='holdout').sum())}). "
        f"Join QA 10 / 896,711 — confirm {inv_ext_all['n_flag']} / {inv_ext_all['n_rows']:,}.",
        "",
        md_table(
            inv_rows,
            [
                ("company", ""),
                ("split", ""),
                ("iss", ""),
                ("type", ""),
                ("status", ""),
                ("amount", "right"),
            ],
        ),
        "",
        f"Transactions: **{len(tx)}** (train {int((tx['split']=='train').sum())} / "
        f"holdout {int((tx['split']=='holdout').sum())}). "
        f"Category groups: {tx['grp'].value_counts().to_dict()}.",
        "",
        md_table(
            tx_rows,
            [
                ("company", ""),
                ("split", ""),
                ("date", ""),
                ("cat", ""),
                ("grp", ""),
                ("amount", "right"),
            ],
        ),
        "",
        "## Pass 4 — accepted-Y rates on flagged vs not (train company-months)",
        "",
        "Company-month flags from booking / issuance month. Snapshot flags "
        "(`ever_sentinel`, `ever_ogtg`, `ever_*_after`) mark every panel month of that company. "
        "Size AUROC is `log1p(a_in3)` vs the flag (Mann–Whitney). ≥0.60 = SIZE.",
        "",
        md_table(
            y_rows,
            [
                ("flag", ""),
                ("train CM", "right"),
                ("CM %", "right"),
                ("y2 on/off", ""),
                ("y3 on/off", ""),
                ("y4 on/off", ""),
                ("y7 on/off", ""),
                ("y9 on/off", ""),
                ("size AUROC", "right"),
            ],
        ),
        "",
        "## Pass 5 — `is_dup`: extract clones vs same-day same amount",
        "",
        f"Full-content clone groups (the clean key: company, product, date, value_date, "
        f"amount, category, description, counterparty, status): **{len(p5['clones']):,}**. "
        f"Train groups {int((p5['clones']['split']=='train').sum()):,}. "
        f"Extra copies (the `is_dup` flag) {int(p5['clones']['n_flagged'].sum()):,}. "
        f"Copies per group p50={p5['clones']['n_copies'].median():.1f} "
        f"max={int(p5['clones']['n_copies'].max())}.",
        "",
        f"Same-day same-amount (looser): **{len(p5['soft']):,}** groups. "
        f"Of those, **{len(p5['soft_not_clone']):,}** have zero `is_dup` "
        f"(same payment booked twice with a different description/status/product — not flagged). "
        f"Multi-status among soft groups: {int((p5['soft']['n_status']>1).sum()):,}.",
        "",
        f"Train CM `has_tx_dup` base={_pp(p5['cm_base'])}. "
        f"Size AUROC vs `log1p(a_in3)` = {_f(p5['size_auc'])} "
        f"({'SIZE' if np.isfinite(p5['size_auc']) and p5['size_auc'] >= 0.60 else 'not SIZE'}).",
        "",
        "Train clone extras by category:",
        "",
        md_table(
            cat_rows,
            [
                ("category", ""),
                ("groups", "right"),
                ("extra copies", "right"),
                ("|amt|", "right"),
            ],
        )
        if cat_rows
        else "(none)",
        "",
        "## Pass 6 — `product_known` vs Family G first-month hole",
        "",
        f"G quote: first-month `g_n_accounts` p50=0, **63.7% still 0**. "
        f"This module: train first-month g=0 share **{_pp(p6['g0_share'])}**.",
        "",
        f"Unknown-product txs are a **different hole**. "
        f"First-month unk-product row share on g=0 companies={_pp(p6['first_unk_g0'])}; "
        f"on g>0 companies={_pp(p6['first_unk_g1'])}. "
        f"Distinct unknown `product_id`={p6['n_unk_prod']}; "
        f"of those in banking∪debt books={p6['n_in_books']} (must be 0 by construction).",
        "",
        f"Train panel CM: `g_n_accounts=0` {p6['g0_cm']:,}; "
        f"any unknown-product tx {p6['unk_cm']:,}; both {p6['both_cm']:,} "
        f"(unk among g0 months {_pp(_pct(p6['both_cm'], p6['g0_cm']))}). "
        f"Trail already said g=0 months still have txs — those txs are almost all `product_known=true` "
        f"(product exists in the extract, `created_at` is later). Unknown product is a missing-ID hole, "
        f"not the connection clock.",
        "",
        "## Pass 7 — `balance_sentinel` vs Family B walk",
        "",
        f"`liquidity.py` `_cash_balances` keeps checking/saving/tpv with `balance IS NOT NULL`. "
        f"Clean nulls sentinel balances, so a cash-type sentinel is **silent-omitted** from the snapshot.",
        "",
        f"Sentinel rows={len(sent)}. On cash types={p7['n_cash_sent']}. "
        f"B cash products={p7['n_cash']}; used (not-null)={p7['n_cash_used']}; "
        f"companies with no usable cash snapshot={p7['n_co_no_snap']}; "
        f"of which sentinel-only={p7['n_co_sent_only']}.",
        "",
        md_table(
            sent_rows,
            [
                ("company", ""),
                ("split", ""),
                ("bank type", ""),
                ("debt type", ""),
                ("cash?", ""),
                ("raw balance", "right"),
            ],
        )
        if sent_rows
        else "(none)",
        "",
        "## Pass 8 — debt exclude-extreme vs cashflow keep-extreme",
        "",
        "Cashflow / ops / groupctx: flags are **not** drop filters. "
        "Debt `_flows`: `AND NOT coalesce(t.is_extreme, false)` — so `f_ds_r` / `f_fc_r` "
        "omit extremes from op_in, debt_service, and fin_cost. Family E also drops "
        "`payment_date_invalid` from open/delay (timing unknown). Family D HHI does **not** "
        "drop `is_extreme` invoices. Inconsistency stays in the report; modules not edited.",
        "",
        f"- Train sum `a_op_in` keep-extreme (cashflow) = {_sci(p8['train_sum_op_in_keep'])}.",
        f"- Train sum `a_op_in` drop-extreme (debt rule) = {_sci(p8['train_sum_op_in_drop'])}.",
        f"- Relative move = {_pp(p8['rel_op_in'], 4)}. Company-months with any op_in change: {p8['n_cm_op_in_changed']:,}.",
        f"- Train sum debt_service keep={_sci(p8['train_sum_ds_keep'])} drop={_sci(p8['train_sum_ds_drop'])} rel={_pp(p8['rel_ds'], 4)}.",
        f"- `f_ds_r` if cashflow rule vs debt rule: mean |Δ|={_f(p8['mean_abs_fds'])}, "
        f"p99={_f(p8['p99_abs_fds'])}, max={_f(p8['max_abs_fds'])}. "
        f"CM changed={p8['n_cm_fds_changed']:,}; |Δ|≥0.01 → {p8['n_cm_fds_ge_001']:,}.",
        f"- Store `f_ds_r` vs recomputed drop-extreme max|Δ|={_f(p8['store_vs_drop_maxabs'])} (sanity).",
        "",
        "## Pass 9 — quoted singles if extremes dropped",
        "",
        "Night quotes: `c_n_days_with_tx` 0.711 (Y3), `e_ar_issued_lag1` 0.630 (Y7), "
        "`d_cust_hhi_lag3` 0.605 (Y4). Recompute the single with extremes removed; "
        "train group-fold signed AUROC. Move = |Δ|≥0.02.",
        "",
        md_table(
            q_rows,
            [
                ("single", ""),
                ("quote", "right"),
                ("keep CV", "right"),
                ("drop-ext CV", "right"),
                ("Δ", "right"),
                ("changed CM", "right"),
                ("≥0.02?", ""),
            ],
        ),
        "",
        f"Store `c_n_days_with_tx` CV={_f(p9['store_days']['cv'])} (night 0.711). "
        f"Store issued_lag1 CV={_f(p9['store_iss']['cv'])} (night 0.630). "
        f"In-module issued_lag1 keep=0.6157 is a 0-fill / cancel-filter gap vs the store, "
        f"**not** a drop-extreme effect (keep and drop match). "
        f"Days that vanish if extremes dropped: {p9['days_vanish']:,} train CM. "
        f"Extreme AR invoices with a CP (can touch HHI): {p9['n_ext_ar']}.",
        "",
        f"**Never-drop:** {drop_rule}.",
        "",
        "## Pass 10 — company-level ever-flagged vs never",
        "",
        md_table(
            ever_rows,
            [
                ("split", ""),
                ("flag", ""),
                ("ever", "right"),
                ("share", "right"),
            ],
        ),
        "",
        "## Family-module inconsistency (not patched)",
        "",
        "- `cashflow.py` / `ops.py` / `groupctx.py` / `catmix.py`: *Clean flags are not drop filters.*",
        "- `debt.py` `_flows`: excludes `is_extreme`. `created_after_snapshot` is excluded from as-of inventory.",
        "- `invoices.py` / `match.py`: `payment_date_invalid` dropped from timing / paid-match.",
        "- `liquidity.py`: sentinels already NULL, then `balance IS NOT NULL` — cash-type sentinels never enter the walk.",
        "- `counterparties.py` HHI: no `is_extreme` filter on invoices.",
        "",
        "No family file was edited. If a later wave unifies the extreme rule, pick **keep** "
        "(never-drop) unless pass 9 is revisited and a quote moves.",
        "",
        "## PARK / CLOSE / KEEP",
        "",
        "| item | call |",
        "| --- | --- |",
        "| any flag as a health Y | **PARK** (DQ; do not invent a Y from a flag) |",
        "| any flag as X | **CLOSE** |",
        "| never-drop doubtful rows | **KEEP** unless pass 9 says a quote moved ≥0.02 |",
        "| debt vs cashflow extreme rule | **report** — do not patch tonight |",
        "| product_known as the G hole | **CLOSE** that story — different hole |",
        "| B snapshot sentinels | **protects** — 3 checking NULLed; other cash remains |",
        "| B flow walk vs extreme txs | **footnote** — all 24 extremes are checking. COMP_0306: 11/18 Y2 flips (0→11) if the walk dropped the −1.61e9 checking flow. Do not rewrite `liquidity.py` here |",
        "",
        "## Pass 11 — canceling pairs, PDI pile, B gap, HHI windows",
        "",
        ctx.get("p11_md", "(pass 11 not run)"),
        "",
        "## Still unknown / next cut in this module",
        "",
        "- Family B QA owns whether to footnote `b_liq` / Y2 on COMP_0306 (11/18 flips). This lane does not rewrite `liquidity.py`.",
        "- Hidden-test GROUP_0199 (COMP_0900 + COMP_0276) books uncategorized giants on the same two days (2026-02-10 / 02-13). Coverage only; do not redo sibling H.",
        "- Train GROUP_0094 pairs COMP_0306 (Y2-flip outflow) with sibling COMP_1192 (+1.13e9). Coverage fact.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")


def append_registry(ctx: dict) -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    with REGISTRY.open("r", encoding="utf-8") as f:
        header = next(csv.reader(f))
    existing = REGISTRY.read_text(encoding="utf-8")
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    p1, p8, p9, p6 = ctx["p1"], ctx["p8"], ctx["p9"], ctx["p6"]

    def grab(table, flag, split, field):
        for r in p1["rows"]:
            if r["table"] == table and r["flag"] == flag and r["split"] == split:
                return r[field]
        return ""

    q = {r["name"]: r for r in p9["results"]}
    rows = [
        {
            "x_families": "DQ",
            "y": "-",
            "model": "coverage",
            "split": "train",
            "metric": "tx_is_extreme_n",
            "value": grab("transactions", "is_extreme", "train", "n_flag"),
            "coverage": grab("transactions", "is_extreme", "train", "row_share"),
            "notes": "clean txs is_extreme train n",
        },
        {
            "x_families": "DQ",
            "y": "-",
            "model": "coverage",
            "split": "all",
            "metric": "tx_is_extreme_n",
            "value": grab("transactions", "is_extreme", "all", "n_flag"),
            "coverage": grab("transactions", "is_extreme", "all", "row_share"),
            "notes": "confirm vs join-QA invoices-only 10",
        },
        {
            "x_families": "DQ",
            "y": "-",
            "model": "coverage",
            "split": "train",
            "metric": "tx_is_extreme_amt_share",
            "value": grab("transactions", "is_extreme", "train", "mass_share"),
            "coverage": grab("transactions", "is_extreme", "train", "row_share"),
            "notes": "amount mass not row count",
        },
        {
            "x_families": "DQ",
            "y": "-",
            "model": "coverage",
            "split": "all",
            "metric": "inv_is_extreme_n",
            "value": grab("invoices", "is_extreme", "all", "n_flag"),
            "coverage": grab("invoices", "is_extreme", "all", "row_share"),
            "notes": "confirm 10/896711",
        },
        {
            "x_families": "DQ",
            "y": "-",
            "model": "coverage",
            "split": "train",
            "metric": "inv_is_extreme_amt_share",
            "value": grab("invoices", "is_extreme", "train", "mass_share"),
            "coverage": grab("invoices", "is_extreme", "train", "row_share"),
            "notes": "amount mass train",
        },
        {
            "x_families": "DQ",
            "y": "-",
            "model": "coverage",
            "split": "train",
            "metric": "tx_is_dup_n",
            "value": grab("transactions", "is_dup", "train", "n_flag"),
            "coverage": grab("transactions", "is_dup", "train", "row_share"),
            "notes": "extract clones",
        },
        {
            "x_families": "DQ",
            "y": "-",
            "model": "coverage",
            "split": "train",
            "metric": "tx_unk_product_share",
            "value": grab("transactions", "product_known=false", "train", "row_share"),
            "coverage": p6["g0_share"],
            "notes": "unk share vs G first-month g0",
        },
        {
            "x_families": "DQ",
            "y": "-",
            "model": "coverage",
            "split": "train",
            "metric": "a_op_in_rel_if_drop_extreme",
            "value": p8["rel_op_in"],
            "coverage": p8["n_cm_op_in_changed"],
            "notes": "cashflow keep vs debt drop",
        },
        {
            "x_families": "DQ",
            "y": Y3,
            "model": "single_drop_ext",
            "split": "train",
            "metric": "days_cv_delta",
            "value": q.get("days", {}).get("delta", ""),
            "coverage": q.get("days", {}).get("n", ""),
            "notes": f"keep={q.get('days',{}).get('keep_cv')} drop={q.get('days',{}).get('drop_cv')}",
        },
        {
            "x_families": "DQ",
            "y": Y7,
            "model": "single_drop_ext",
            "split": "train",
            "metric": "issued_lag1_cv_delta",
            "value": q.get("issued_lag1", {}).get("delta", ""),
            "coverage": q.get("issued_lag1", {}).get("n", ""),
            "notes": f"keep={q.get('issued_lag1',{}).get('keep_cv')} drop={q.get('issued_lag1',{}).get('drop_cv')}",
        },
        {
            "x_families": "DQ",
            "y": Y4,
            "model": "single_drop_ext",
            "split": "train",
            "metric": "hhi_lag3_cv_delta",
            "value": q.get("hhi_lag3", {}).get("delta", ""),
            "coverage": q.get("hhi_lag3", {}).get("n", ""),
            "notes": f"keep={q.get('hhi_lag3',{}).get('keep_cv')} drop={q.get('hhi_lag3',{}).get('drop_cv')}",
        },
        {
            "x_families": "DQ",
            "y": "-",
            "model": "coverage",
            "split": "all",
            "metric": "inv_extreme_wash_pairs",
            "value": ctx.get("p11", {}).get("washes", ""),
            "coverage": ctx.get("p11", {}).get("n_hhi_eligible", ""),
            "notes": "opposite-sign pairs; coverage=HHI-eligible n",
        },
        {
            "x_families": "DQ",
            "y": "-",
            "model": "coverage",
            "split": "train",
            "metric": "pdi_july_mass_share",
            "value": _pct(ctx.get("p11", {}).get("july_mass", 0), ctx.get("p11", {}).get("tot_pdi_mass", 1)),
            "coverage": ctx.get("p11", {}).get("pdi_base", ""),
            "notes": "PDI amount mass in 2026-07; coverage=CM has_pdi",
        },
        {
            "x_families": "B",
            "y": "-",
            "model": "coverage",
            "split": "train",
            "metric": "cash_sentinel_n",
            "value": ctx.get("p7", {}).get("n_cash_sent", ""),
            "coverage": ctx.get("p7", {}).get("n_co_no_snap", ""),
            "notes": "checking sentinels omitted from B walk; coverage=cos with no snap",
        },
        {
            "x_families": "DQ",
            "y": "-",
            "model": "coverage",
            "split": "train",
            "metric": "inv_extreme_leftover_amt_share",
            "value": _pct(ctx.get("p11", {}).get("ext_leftover", 0), ctx.get("p11", {}).get("all_inv_mass", 1)),
            "coverage": ctx.get("p11", {}).get("leftover_n", ""),
            "notes": "unpaired extreme |amt| / all invoices after 3 washes",
        },
        {
            "x_families": "DQ",
            "y": "-",
            "model": "coverage",
            "split": "train",
            "metric": "has_tx_dup_vs_n_tx_auc",
            "value": ctx.get("p11", {}).get("dup_vs_ntx", ""),
            "coverage": ctx.get("p11", {}).get("pdi_base", ""),
            "notes": "is_dup is row-SIZE; coverage unused (pdi base parked nearby)",
        },
        {
            "x_families": "DQ",
            "y": Y3,
            "model": "single_flag",
            "split": "train",
            "metric": "has_pdi_cv",
            "value": next(
                (r["cv"] for r in ctx.get("p11", {}).get("p14", {}).get("x_rows", []) if r["y"] == Y3),
                "",
            ),
            "coverage": ctx.get("p11", {}).get("pdi_base", ""),
            "notes": "CLOSE as X; residual PDI is spread",
        },
        {
            "x_families": "DQ",
            "y": Y3,
            "model": "single_drop_dup",
            "split": "train",
            "metric": "a_n_tx_cv_delta",
            "value": ctx.get("p11", {}).get("p15", {}).get("ntx_delta", ""),
            "coverage": ctx.get("p11", {}).get("p15", {}).get("n_changed", ""),
            "notes": "drop is_dup extras; first copy stays",
        },
        {
            "x_families": "G",
            "y": "-",
            "model": "coverage",
            "split": "train",
            "metric": "post_snapshot_bank_opin_share",
            "value": ctx.get("p11", {}).get("p19", {}).get("rel", ""),
            "coverage": ctx.get("p11", {}).get("p19", {}).get("n_cm", ""),
            "notes": "G excludes product; cashflow keeps txs",
        },
        {
            "x_families": "B",
            "y": "-",
            "model": "coverage",
            "split": "all",
            "metric": "extreme_tx_on_checking",
            "value": ctx.get("p11", {}).get("p21", {}).get("n_cash", ""),
            "coverage": ctx.get("p11", {}).get("p21", {}).get("n", ""),
            "notes": "all extreme txs sit on checking; B flow walk sees them",
        },
        {
            "x_families": "B",
            "y": Y2,
            "model": "coverage",
            "split": "train",
            "metric": "y2_flips_if_drop_extreme_checking",
            "value": ctx.get("p11", {}).get("p24", {}).get("n_flip_train", ""),
            "coverage": ctx.get("p11", {}).get("p24", {}).get("n_lab_train", ""),
            "notes": "alt_liq = b_liq + sum(extreme cash flow after t); Y2 already PARK",
        },
        {
            "x_families": "H",
            "y": "-",
            "model": "coverage",
            "split": "all",
            "metric": "extreme_tx_shared_groups",
            "value": ctx.get("p11", {}).get("p33", {}).get("n_shared_groups", ""),
            "coverage": ctx.get("p11", {}).get("p33", {}).get("n", ""),
            "notes": "GROUP_0199 holdout 0900+0276; GROUP_0094 train 0306+1192; do not redo H",
        },
    ]
    written = 0
    with REGISTRY.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        for r in rows:
            rec = {
                "ts": ts,
                "round": ROUND,
                "wave": WAVE,
                "agent": AGENT,
                "x_families": r.get("x_families", "DQ"),
                "y": r.get("y", ""),
                "model": r.get("model", ""),
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


def main() -> int:
    t0 = time.time()
    print(f"clean_flags_qa start {_now_iso()} agent={AGENT}")
    hold = load_holdout()
    print(f"holdout companies={len(hold)}")
    con = connect()
    p1 = pass1_prevalence(con, hold)
    p2 = pass2_calendar(con, hold)
    p3 = pass3_extreme_cards(con, hold)
    panel = attach_groups(con, load_panel())
    p4 = pass4_y_overlap(con, panel)
    panel = p4["panel"]
    p5 = pass5_dup(con, panel, hold)
    p6 = pass6_product_vs_g(con, panel)
    p7 = pass7_sentinel(con, panel, hold)
    p8 = pass8_debt_vs_cashflow(con, panel)
    p9 = pass9_quoted(con, panel)
    p10 = pass10_ever(panel)
    p11 = pass11_deeper(con, panel, p3, p5, p7, p8, p9)
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
    }
    write_png(p1)
    write_md(ctx)
    append_registry(ctx)
    con.close()
    print(f"done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
