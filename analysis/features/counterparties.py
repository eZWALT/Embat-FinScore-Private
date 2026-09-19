"""Family D — counterparty structure (customers, suppliers, tx fill).

All features use only events with date <= period end. Invoice IDs are the
customer/supplier source (near-complete fill). Transaction counterparties are
used for ``d_tx_cp_share`` and for a same-ID intercompany probe.

Formulas (6-month window = first day of (period_end month − 5 months) through
period end; same window as ``score_pipeline`` concentration):

- d_cust_hhi / d_supp_hhi: sum_i (amt_i / tot)^2 over counterparties (Herfindahl).
- d_cust_top1 / d_supp_top1: max_i amt_i / tot (pipeline-style concentration).
- d_n_cust / d_n_supp: distinct non-null counterparty_id on AR / AP invoices.
- d_cust_new / d_cust_lost: |cur \\ prev| and |prev \\ cur| of AR counterparties
  in the calendar quarter of ``period`` (through period end) vs the previous
  calendar quarter.
- d_tx_cp_share: share of transactions in the 6-month window with non-null
  counterparty_id.
- d_interco_share: |flows| whose counterparty_id equals another company_id in
  the same group_id, over |tx amounts| in the window. Emitted only if that
  equality join matches any row in the database; otherwise all-NaN. Do not
  invent a COMP_* mapping — invoice/tx IDs are COUNTERPARTY_*, companies are
  COMP_*, and the two sets do not overlap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import MONTHS

SOURCE_TABLES = ["transactions", "invoices", "companies"]
FAMILY = "d"

DATA_START = pd.Timestamp(MONTHS[0])  # 2024-09-01
_COLS = [
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


def _infer_freq(period: pd.Series) -> str:
    u = pd.to_datetime(period).drop_duplicates().sort_values()
    if len(u) >= 2:
        med = u.diff().dropna().median()
        if pd.notna(med) and med <= pd.Timedelta(days=8):
            return "W"
        return "M"
    ts = pd.Timestamp(u.iloc[0])
    return "M" if ts.is_month_start else "W"


def _period_end(period: pd.Series, freq: str) -> pd.Series:
    p = pd.to_datetime(period)
    if freq == "M":
        return p + pd.offsets.MonthEnd(0)
    return p + pd.Timedelta(days=6)


def _windows(period: pd.Series, freq: str) -> pd.DataFrame:
    p = pd.to_datetime(period)
    end = pd.to_datetime(_period_end(p, freq)).dt.normalize()
    win6 = (end - pd.DateOffset(months=5)).dt.to_period("M").dt.to_timestamp()
    q = p.dt.to_period("Q")
    return pd.DataFrame(
        {
            "period_end": end,
            "win6_start": win6,
            "q_start": q.dt.start_time,
            "prev_q_start": (q - 1).dt.start_time,
            "prev_q_end": (q - 1).dt.end_time.dt.normalize(),
            "full6": win6 >= DATA_START,
            "prev_q_complete": (q - 1).dt.start_time >= DATA_START,
        }
    )


def _id_overlap_count(con) -> int:
    """Rows where counterparty_id equals a company_id (only non-invented link)."""
    return int(
        con.execute(
            """
            SELECT
              (SELECT COUNT(*) FROM invoices i
                 INNER JOIN companies c ON i.counterparty_id = c.company_id)
              +
              (SELECT COUNT(*) FROM transactions t
                 INNER JOIN companies c ON t.counterparty_id = c.company_id)
            """
        ).fetchone()[0]
    )


def _concentration(win: pd.DataFrame, side: str, period) -> pd.DataFrame:
    """Return company-level HHI, top-1, n for one side in one period."""
    prefix = "d_cust" if side == "AR" else "d_supp"
    s = win.loc[win["side"] == side, ["company_id", "counterparty_id", "amt"]]
    s = s[s["counterparty_id"].notna() & (s["amt"] > 0)]
    empty = pd.DataFrame(
        columns=["company_id", "period", f"{prefix}_hhi", f"{prefix}_top1", f"{prefix}_n"]
    )
    if s.empty:
        return empty
    g = s.groupby(["company_id", "counterparty_id"], as_index=False)["amt"].sum()
    tot = g.groupby("company_id")["amt"].transform("sum")
    share = g["amt"] / tot
    g = g.assign(share2=share.to_numpy() ** 2, share=share)
    out = g.groupby("company_id", as_index=False).agg(
        **{
            f"{prefix}_hhi": ("share2", "sum"),
            f"{prefix}_top1": ("share", "max"),
            f"{prefix}_n": ("counterparty_id", "nunique"),
        }
    )
    out["period"] = period
    return out


def build(con, grid: pd.DataFrame) -> pd.DataFrame:
    keys = grid[["company_id", "period"]].copy().reset_index(drop=True)
    keys["period"] = pd.to_datetime(keys["period"])
    if keys.empty:
        return keys.assign(**{c: np.float64(np.nan) for c in _COLS})

    freq = _infer_freq(keys["period"])
    meta = pd.concat([keys, _windows(keys["period"], freq).reset_index(drop=True)], axis=1)

    first_inv = con.execute(
        """
        SELECT company_id, MIN(CAST(issuance_date AS DATE)) AS first_iss
        FROM invoices
        WHERE document_type = 'invoice' AND status <> 'cancel' AND amount <> 0
          AND issuance_date IS NOT NULL
        GROUP BY 1
        """
    ).df()
    first_inv["first_iss"] = pd.to_datetime(first_inv["first_iss"])
    meta = meta.merge(first_inv, on="company_id", how="left")
    meta["has_erp"] = meta["first_iss"].notna() & (meta["first_iss"] <= meta["period_end"])
    meta["churn_ok"] = (
        meta["has_erp"]
        & meta["prev_q_complete"]
        & meta["first_iss"].notna()
        & (meta["first_iss"] <= meta["prev_q_end"])
    )

    inv = con.execute(
        """
        SELECT company_id,
               CAST(issuance_date AS DATE) AS iss,
               counterparty_id,
               CASE WHEN amount > 0 THEN 'AR' ELSE 'AP' END AS side,
               abs(amount) AS amt
        FROM invoices
        WHERE document_type = 'invoice'
          AND status <> 'cancel'
          AND amount <> 0
          AND issuance_date IS NOT NULL
        """
    ).df()
    inv["iss"] = pd.to_datetime(inv["iss"])

    tx_daily = con.execute(
        """
        SELECT company_id,
               CAST("date" AS DATE) AS d,
               COUNT(*) AS n_tx,
               SUM(CASE WHEN counterparty_id IS NOT NULL
                         AND length(trim(CAST(counterparty_id AS VARCHAR))) > 0
                        THEN 1 ELSE 0 END) AS n_tx_cp
        FROM transactions
        WHERE "date" IS NOT NULL
        GROUP BY 1, 2
        """
    ).df()
    tx_daily["d"] = pd.to_datetime(tx_daily["d"])

    overlap = _id_overlap_count(con)
    interco_daily = pd.DataFrame(columns=["company_id", "d", "interco_amt", "tx_amt"])
    if overlap > 0:
        interco_daily = con.execute(
            """
            SELECT t.company_id,
                   CAST(t."date" AS DATE) AS d,
                   SUM(abs(t.amount)) AS tx_amt,
                   SUM(CASE WHEN other.company_id IS NOT NULL THEN abs(t.amount) ELSE 0 END)
                     AS interco_amt
            FROM transactions t
            INNER JOIN companies self ON t.company_id = self.company_id
            LEFT JOIN companies other
              ON t.counterparty_id = other.company_id
             AND other.group_id = self.group_id
             AND other.company_id <> self.company_id
            WHERE t."date" IS NOT NULL
            GROUP BY 1, 2
            """
        ).df()
        if not interco_daily.empty:
            interco_daily["d"] = pd.to_datetime(interco_daily["d"])

    conc_parts: list[pd.DataFrame] = []
    churn_parts: list[pd.DataFrame] = []
    tx_parts: list[pd.DataFrame] = []
    interco_parts: list[pd.DataFrame] = []

    for period, sl in meta.groupby("period", sort=False):
        e = sl["period_end"].iloc[0]
        w0 = sl["win6_start"].iloc[0]
        q0 = sl["q_start"].iloc[0]
        pq0 = sl["prev_q_start"].iloc[0]
        pq1 = sl["prev_q_end"].iloc[0]
        cos = sl["company_id"]

        if sl["full6"].iloc[0]:
            inv_w = inv[(inv["iss"] >= w0) & (inv["iss"] <= e) & inv["company_id"].isin(cos)]
            if not inv_w.empty:
                for side in ("AR", "AP"):
                    piece = _concentration(inv_w, side, period)
                    if not piece.empty:
                        conc_parts.append(piece)

        if sl["prev_q_complete"].iloc[0]:
            ar = inv.loc[
                (inv["side"] == "AR")
                & inv["counterparty_id"].notna()
                & inv["company_id"].isin(cos),
                ["company_id", "iss", "counterparty_id"],
            ]
            cur = ar.loc[(ar["iss"] >= q0) & (ar["iss"] <= e), ["company_id", "counterparty_id"]].drop_duplicates()
            prev = ar.loc[(ar["iss"] >= pq0) & (ar["iss"] <= pq1), ["company_id", "counterparty_id"]].drop_duplicates()
            both = cur.merge(prev, on=["company_id", "counterparty_id"], how="outer", indicator=True)
            if not both.empty:
                new = (
                    both.loc[both["_merge"].eq("left_only")]
                    .groupby("company_id", as_index=False)
                    .size()
                    .rename(columns={"size": "d_cust_new"})
                )
                lost = (
                    both.loc[both["_merge"].eq("right_only")]
                    .groupby("company_id", as_index=False)
                    .size()
                    .rename(columns={"size": "d_cust_lost"})
                )
                ch = new.merge(lost, on="company_id", how="outer")
                ch["d_cust_new"] = ch["d_cust_new"].fillna(0.0)
                ch["d_cust_lost"] = ch["d_cust_lost"].fillna(0.0)
                ch["period"] = period
                churn_parts.append(ch)

        tx_w = tx_daily[(tx_daily["d"] >= w0) & (tx_daily["d"] <= e) & tx_daily["company_id"].isin(cos)]
        if not tx_w.empty:
            agg = tx_w.groupby("company_id", as_index=False).agg(n_tx=("n_tx", "sum"), n_tx_cp=("n_tx_cp", "sum"))
            agg["d_tx_cp_share"] = np.where(agg["n_tx"] > 0, agg["n_tx_cp"] / agg["n_tx"], np.nan)
            agg["period"] = period
            tx_parts.append(agg[["company_id", "period", "d_tx_cp_share"]])

        if overlap > 0 and not interco_daily.empty:
            ic = interco_daily[
                (interco_daily["d"] >= w0)
                & (interco_daily["d"] <= e)
                & interco_daily["company_id"].isin(cos)
            ]
            if not ic.empty:
                ic = ic.groupby("company_id", as_index=False).agg(
                    interco_amt=("interco_amt", "sum"), tx_amt=("tx_amt", "sum")
                )
                ic["d_interco_share"] = np.where(ic["tx_amt"] > 0, ic["interco_amt"] / ic["tx_amt"], np.nan)
                ic["period"] = period
                interco_parts.append(ic[["company_id", "period", "d_interco_share"]])

    out = meta[["company_id", "period", "has_erp", "churn_ok", "full6"]].copy()

    if conc_parts:
        conc = pd.concat(conc_parts, ignore_index=True)
        if "d_cust_hhi" in conc.columns:
            cust = conc.dropna(subset=["d_cust_hhi"])[
                ["company_id", "period", "d_cust_hhi", "d_cust_top1", "d_cust_n"]
            ].rename(columns={"d_cust_n": "d_n_cust"})
            out = out.merge(cust, on=["company_id", "period"], how="left")
        if "d_supp_hhi" in conc.columns:
            supp = conc.dropna(subset=["d_supp_hhi"])[
                ["company_id", "period", "d_supp_hhi", "d_supp_top1", "d_supp_n"]
            ].rename(columns={"d_supp_n": "d_n_supp"})
            out = out.merge(supp, on=["company_id", "period"], how="left")
    for c in ("d_cust_hhi", "d_cust_top1", "d_n_cust", "d_supp_hhi", "d_supp_top1", "d_n_supp"):
        if c not in out.columns:
            out[c] = np.nan

    # Full 6m window + ERP history but no identified CPs → n=0, HHI/top1 stay NaN.
    full6 = out["full6"].fillna(False).astype(bool)
    has_erp = out["has_erp"].fillna(False).astype(bool)
    out.loc[full6 & has_erp & out["d_n_cust"].isna(), "d_n_cust"] = 0.0
    out.loc[full6 & has_erp & out["d_n_supp"].isna(), "d_n_supp"] = 0.0

    if churn_parts:
        ch = pd.concat(churn_parts, ignore_index=True)
        out = out.merge(ch, on=["company_id", "period"], how="left")
    else:
        out["d_cust_new"] = np.nan
        out["d_cust_lost"] = np.nan
    if "d_cust_new" not in out.columns:
        out["d_cust_new"] = np.nan
        out["d_cust_lost"] = np.nan
    churn_ok = out["churn_ok"].fillna(False).astype(bool)
    out.loc[churn_ok & out["d_cust_new"].isna(), "d_cust_new"] = 0.0
    out.loc[churn_ok & out["d_cust_lost"].isna(), "d_cust_lost"] = 0.0
    out.loc[~churn_ok, ["d_cust_new", "d_cust_lost"]] = np.nan

    if tx_parts:
        tx = pd.concat(tx_parts, ignore_index=True)
        out = out.merge(tx, on=["company_id", "period"], how="left")
    else:
        out["d_tx_cp_share"] = np.nan

    if interco_parts:
        ic = pd.concat(interco_parts, ignore_index=True)
        out = out.merge(ic, on=["company_id", "period"], how="left")
    else:
        out["d_interco_share"] = np.nan

    inv_cols = [
        "d_cust_hhi",
        "d_cust_top1",
        "d_n_cust",
        "d_supp_hhi",
        "d_supp_top1",
        "d_n_supp",
        "d_cust_new",
        "d_cust_lost",
    ]
    no_erp = ~out["has_erp"].fillna(False).astype(bool)
    incomplete6 = ~out["full6"].fillna(False).astype(bool)
    out.loc[no_erp, inv_cols] = np.nan
    out.loc[incomplete6, ["d_cust_hhi", "d_cust_top1", "d_n_cust", "d_supp_hhi", "d_supp_top1", "d_n_supp"]] = np.nan

    result = keys.merge(out[["company_id", "period"] + _COLS], on=["company_id", "period"], how="left")
    if result.duplicated(["company_id", "period"]).any():
        raise ValueError("Family D produced duplicate company_id, period rows")
    for c in _COLS:
        result[c] = pd.to_numeric(result[c], errors="coerce").astype("float64")
    return result
