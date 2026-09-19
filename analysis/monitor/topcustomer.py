"""'Top customer went quiet' alert (plan step 3, decided). Trigger, exposure and ranking. Not a score input.

Trigger (transparent rule): last quarter's top customer got no invoice this month.
  - top customer at month t = largest counterparty by invoiced AR amount in months t-3..t-1 (as-of, no look-ahead)
  - quiet at t when that counterparty has no AR invoice issued in month t
Measured on train (analysis/monitor/y7_alert_eval.py, .agents/persistent-memory/2026-09-19-1300-y7-alert-grade-eval.md):
58% of flagged months lose the customer vs 29% base (83% recall, 40% of rows flagged); with a sustained 25% inflow drop the
precision is 17% vs 6%. Wording is "top customer stopped billing, review exposure and collections", never "revenue at risk".
The night's TURNOVER card (shallow LightGBM) only ranks alerts: its score is not a probability.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.models.gbm_y7_core import CORE_TURNOVER, Y_COL, _fit_shallow, _resolve_cols, prepare_panel
from analysis.targets.y7_concentration import _ar_month_cp, _roll_cp

MODEL_PATH = Path(__file__).resolve().parent / "y7_rank_model.txt"
MODEL_COLS_PATH = Path(__file__).resolve().parent / "y7_rank_cols.json"


def quiet_events(con) -> pd.DataFrame:
    """One row per company-month where last quarter's top customer billed nothing.

    Columns: company_id, period (month start), customer_id, share (of last quarter's billing), last_quarter_amount,
    months_billed_of_3 (how many of the 3 months the customer was billed), open_receivable_eur (unpaid AR to that
    customer at month-end, as-of), and `flagged_rows` is the count of company-months with a defined top customer
    (the denominator for the flag rate) in .attrs.
    """
    mcp = _ar_month_cp(con)
    if mcp.empty:
        return pd.DataFrame(columns=["company_id", "period", "customer_id", "share", "last_quarter_amount", "months_billed_of_3", "open_receivable_eur"])
    book = _roll_cp(mcp, (0, 1, 2))                     # book at p covers issuance in p-2..p
    tot = book.groupby(["company_id", "period"])["amt"].transform("sum")
    book = book[tot > 0].copy()
    book["share"] = book["amt"] / tot[tot > 0]
    idx = book.groupby(["company_id", "period"])["amt"].idxmax()
    top = book.loc[idx, ["company_id", "period", "counterparty_id", "share", "amt"]].rename(
        columns={"counterparty_id": "customer_id", "amt": "last_quarter_amount"})
    # the book at p is the quarter before month p+1
    top["period"] = top["period"] + pd.DateOffset(months=1)
    billed = mcp.rename(columns={"month": "period", "counterparty_id": "customer_id", "amt": "amt_now"})
    top = top.merge(billed, on=["company_id", "period", "customer_id"], how="left")
    top["amt_now"] = top["amt_now"].fillna(0.0)
    months = []
    for off in (1, 2, 3):
        m = mcp.copy()
        m["period"] = m["month"] + pd.DateOffset(months=off)
        months.append(m[["company_id", "period", "counterparty_id"]].rename(columns={"counterparty_id": "customer_id"}))
    n_billed = pd.concat(months).groupby(["company_id", "period", "customer_id"]).size().rename("months_billed_of_3").reset_index()
    top = top.merge(n_billed, on=["company_id", "period", "customer_id"], how="left")
    top["months_billed_of_3"] = top["months_billed_of_3"].fillna(0).astype(int)
    n_rows = len(top)
    ev = top[top["amt_now"] <= 0].drop(columns="amt_now").reset_index(drop=True)
    ev = ev.merge(_open_receivable(con, ev), on=["company_id", "period", "customer_id"], how="left")
    ev.attrs["defined_rows"] = n_rows
    return ev


def onset_events(con) -> pd.DataFrame:
    """quiet_events kept to the ONSET only: the same customer was not already quiet last month (a quiet customer alerts once)."""
    ev = quiet_events(con)
    defined = ev.attrs.get("defined_rows")
    ev["company_id"] = ev["company_id"].astype(str)
    ev["period"] = pd.to_datetime(ev["period"])
    ev["customer_id"] = ev["customer_id"].astype(str)
    prev = ev[["company_id", "customer_id", "period"]].copy()
    prev["period"] = prev["period"] + pd.DateOffset(months=1)
    prev["repeat"] = True
    ev = ev.merge(prev, on=["company_id", "customer_id", "period"], how="left")
    out = ev[ev["repeat"].isna()].drop(columns="repeat").reset_index(drop=True)
    out.attrs["defined_rows"] = defined
    return out


def _open_receivable(con, ev: pd.DataFrame) -> pd.DataFrame:
    """Unpaid AR to the customer at the end of the month (issued by then, not paid by then, payment dates that are impossible dropped)."""
    if ev.empty:
        return ev[["company_id", "period", "customer_id"]].assign(open_receivable_eur=[])
    e = ev[["company_id", "period", "customer_id"]].copy()
    e["month_end"] = e["period"] + pd.offsets.MonthEnd(0)
    con.register("_tc_ev", e)
    try:
        out = con.execute(
            """
            SELECT e.company_id, e.period, e.customer_id, SUM(i.amount) AS open_receivable_eur
            FROM _tc_ev e
            JOIN invoices i ON i.company_id = e.company_id AND i.counterparty_id = e.customer_id
            WHERE i.document_type = 'invoice' AND i.status <> 'cancel' AND i.amount > 0
              AND i.issuance_date <= e.month_end + INTERVAL 1 DAY
              AND NOT i.payment_date_invalid
              AND (i.payment_date IS NULL OR i.payment_date > e.month_end + INTERVAL 1 DAY)
            GROUP BY 1, 2, 3
            """
        ).df()
    finally:
        con.unregister("_tc_ev")
    out["company_id"] = out["company_id"].astype(str)
    out["customer_id"] = out["customer_id"].astype(str)
    return out


def _panel(store: pd.DataFrame, y: pd.DataFrame | None) -> pd.DataFrame:
    s = store.copy()
    s["company_id"] = s["company_id"].astype(str)
    s["period"] = pd.to_datetime(s["period"])
    ypan = (y if y is not None else s[["company_id", "period"]].assign(**{Y_COL: np.nan}))[["company_id", "period", Y_COL]]
    folds = s[["company_id"]].drop_duplicates().assign(fold=0)
    return prepare_panel(s, ypan, folds)


def fit_rank_model(store: pd.DataFrame, y7: pd.DataFrame, train_ids: set[str]) -> dict:
    """Fit the night's TURNOVER card (shallow LightGBM) on labelled train rows and save it. Holdout companies never enter."""
    panel = _panel(store, y7)
    cols = _resolve_cols(panel, CORE_TURNOVER)
    tr = panel[panel["company_id"].isin(train_ids) & panel[Y_COL].notna()]
    clf = _fit_shallow(tr[cols], tr[Y_COL].astype(int))
    clf.booster_.save_model(str(MODEL_PATH))
    MODEL_COLS_PATH.write_text('{"cols": ' + str(list(cols)).replace("'", '"') + "}\n")
    return {"rows": int(len(tr)), "companies": int(tr["company_id"].nunique()), "cols": list(cols)}


def oof_rank_scores(store: pd.DataFrame, y7: pd.DataFrame, train_df: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame:
    """Out-of-fold model score on train (group folds), so evaluation of the ranking is not in-sample."""
    from analysis.evaluate.protocol import FOLD_SEED, group_folds

    folds = group_folds(train_df, n=n_folds, seed=FOLD_SEED)
    panel = _panel(store, y7)
    panel = panel.drop(columns=[c for c in ("fold",) if c in panel.columns]).merge(folds[["company_id", "fold"]], on="company_id", how="inner")
    cols = _resolve_cols(panel, CORE_TURNOVER)
    out = panel[["company_id", "period"]].copy()
    out["rank_score"] = np.nan
    lab = panel[Y_COL].notna()
    for k in range(n_folds):
        clf = _fit_shallow(panel.loc[lab & (panel["fold"] != k), cols], panel.loc[lab & (panel["fold"] != k), Y_COL].astype(int))
        te = panel["fold"] == k
        out.loc[te, "rank_score"] = clf.predict_proba(panel.loc[te, cols])[:, 1]
    return out


def rank_scores(store: pd.DataFrame) -> pd.DataFrame:
    """Model score (ranking only) per company-month from the saved model. NaN where the card's inputs are missing."""
    import json

    import lightgbm as lgb

    cols = json.loads(MODEL_COLS_PATH.read_text())["cols"]
    panel = _panel(store, None)
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    out = panel[["company_id", "period"]].copy()
    out["rank_score"] = booster.predict(panel[cols])
    return out
