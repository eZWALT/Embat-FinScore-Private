"""Family B — liquidity / balance path.

Reconstruct company cash on checking + saving + tpv accounts by walking
backward from the 2026-09-01 `balances` snapshot, matching
`analysis.score_pipeline._liquidity` (and the R `liquidity_by_month`).

The snapshot walk uses later booked flows to recover the historical
end-of-period balance. That is not feature look-ahead: it is an identity
(end_bal_t = snapshot - sum of flows after t). Outflow features use only
transactions with date < AS_OF and only periods <= the row's period.

Formulas (monthly; weekly uses 13 / 26 periods ≈ 3 / 6 months, and the
same runway scale `out3 / 3` = average monthly operational outflow):

- b_liq              reconstructed period-end cash
- b_runway           clip(liq / max(out3 / 3, 1), -6, 24)
- b_d_runway         runway - runway_{t-3} (weekly: t-13)
- b_neg_liq_3        share of last 3 months (13 weeks) with liq < 0
- b_min_liq_3        min liq in that window
- b_mean_liq_3       mean liq in that window
- b_below_0          1 if liq < 0
- b_below_half_runway 1 if liq < 0.5 * max(out3 / 3, 1)
- b_bal_vol          rolling 6 sd(liq) / max(mean |month out|, 1)
                     (weekly: mean |week out| scaled by 52/12)
- b_neg_episodes     count of non-negative → negative onsets in last 6 months
                     (26 weeks)

`out3` is the rolling 3-month (13-week) sum of operational outflow, computed
here from `transactions` + CAT_MAP. This module does not import cashflow.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import AS_OF, CAT_MAP, LAST_M, MONTHS, WEEKS

SOURCE_TABLES = ["balances", "transactions", "banking_products"]
FAMILY = "b"

CASH_TYPES = ("checking", "saving", "tpv")
OP_OUT_CATS = tuple(k for k, v in CAT_MAP.items() if v == "op_out")

B_COLS = [
    "b_liq",
    "b_runway",
    "b_d_runway",
    "b_neg_liq_3",
    "b_min_liq_3",
    "b_mean_liq_3",
    "b_below_0",
    "b_below_half_runway",
    "b_bal_vol",
    "b_neg_episodes",
]


def _freq(periods: pd.Series) -> str:
    p = pd.to_datetime(periods).drop_duplicates()
    ps = set(p)
    if ps and ps <= set(MONTHS):
        return "M"
    if ps and ps <= set(WEEKS):
        return "W"
    if len(p) >= 2:
        return "W" if p.sort_values().diff().median() <= pd.Timedelta(days=8) else "M"
    return "M"


def _cash_types_sql() -> str:
    return ", ".join(f"'{t}'" for t in CASH_TYPES)


def _cash_balances(con) -> pd.DataFrame:
    return con.execute(
        f"""
        SELECT b.product_id, p.company_id, b.balance
        FROM balances b
        JOIN banking_products p ON b.product_id = p.product_id
        WHERE p.type IN ({_cash_types_sql()})
          AND b.balance IS NOT NULL
        """
    ).df()


def _product_flows(con, trunc: str) -> pd.DataFrame:
    # No date filter: same as _liquidity. Snapshot-day txs sit in 2026-09.
    return con.execute(
        f"""
        SELECT t.product_id,
               CAST(date_trunc('{trunc}', t."date") AS DATE) AS stamp,
               SUM(t.amount) AS flow
        FROM transactions t
        WHERE t.product_id IN (
            SELECT b.product_id
            FROM balances b
            JOIN banking_products p ON b.product_id = p.product_id
            WHERE p.type IN ({_cash_types_sql()})
              AND b.balance IS NOT NULL
        )
        GROUP BY 1, 2
        """
    ).df()


def _reconstruct(con, trunc: str, stamps: list[pd.Timestamp], last: pd.Timestamp) -> pd.DataFrame:
    """Period-end cash: snapshot minus the sum of flows after the period."""
    bal = _cash_balances(con)
    if bal.empty:
        return pd.DataFrame(columns=["company_id", "period", "liq"])
    flow = _product_flows(con, trunc)
    flow["stamp"] = pd.to_datetime(flow["stamp"])
    extra = last + (pd.DateOffset(months=1) if trunc == "month" else pd.DateOffset(weeks=1))
    axis = list(stamps) + [pd.Timestamp(extra)]
    grid = pd.MultiIndex.from_product(
        [bal["product_id"].unique(), axis], names=["product_id", "stamp"]
    ).to_frame(index=False)
    grid = grid.merge(flow, on=["product_id", "stamp"], how="left").fillna({"flow": 0.0})
    grid = grid.merge(bal, on="product_id").sort_values(["product_id", "stamp"]).reset_index(drop=True)
    g = grid.groupby("product_id")["flow"]
    after = g.transform("sum") - g.cumsum()
    grid["end_bal"] = grid["balance"] - after
    grid = grid[grid["stamp"] <= last]
    return (
        grid.groupby(["company_id", "stamp"], as_index=False)["end_bal"]
        .sum()
        .rename(columns={"end_bal": "liq", "stamp": "period"})
    )


def _op_out(con, trunc: str) -> pd.DataFrame:
    """Company operational outflow. Sign: amount is negative for outflows, so -sum >= 0."""
    cats = ", ".join(f"'{c}'" for c in OP_OUT_CATS)
    as_of = AS_OF.strftime("%Y-%m-%d")
    df = con.execute(
        f"""
        SELECT t.company_id,
               CAST(date_trunc('{trunc}', t."date") AS DATE) AS period,
               -SUM(CASE WHEN t.category IN ({cats}) THEN t.amount ELSE 0 END) AS op_out
        FROM transactions t
        WHERE t."date" < TIMESTAMP '{as_of}'
        GROUP BY 1, 2
        """
    ).df()
    df["period"] = pd.to_datetime(df["period"])
    return df


def _windows(freq: str) -> tuple[int, int, int]:
    if freq == "W":
        return 13, 26, 13
    return 3, 6, 3


def build(con, grid: pd.DataFrame) -> pd.DataFrame:
    out = grid[["company_id", "period"]].copy()
    out["period"] = pd.to_datetime(out["period"])
    freq = _freq(out["period"])
    if freq == "W":
        liq = _reconstruct(con, "week", list(WEEKS), WEEKS[-1])
        flows = _op_out(con, "week")
        out_scale = 52.0 / 12.0
    else:
        liq = _reconstruct(con, "month", list(MONTHS), LAST_M)
        flows = _op_out(con, "month")
        out_scale = 1.0

    win3, win6, lag3 = _windows(freq)
    P = out.merge(liq, on=["company_id", "period"], how="left")
    P = P.merge(flows, on=["company_id", "period"], how="left")
    P["op_out"] = P["op_out"].fillna(0.0)
    P = P.sort_values(["company_id", "period"]).reset_index(drop=True)

    g = P.groupby("company_id", sort=False)
    P["out3"] = g["op_out"].transform(lambda s: s.rolling(win3).sum())
    month_out = np.maximum(P["out3"] / 3.0, 1.0)

    P["b_liq"] = P["liq"]
    P["b_runway"] = (P["liq"] / month_out).clip(-6, 24)
    P["b_d_runway"] = P["b_runway"] - g["b_runway"].shift(lag3)

    neg = pd.Series(np.where(P["liq"].isna(), np.nan, (P["liq"] < 0).astype(float)), index=P.index)
    P["b_neg_liq_3"] = neg.groupby(P["company_id"], sort=False).transform(lambda s: s.rolling(win3).mean())
    P["b_min_liq_3"] = g["liq"].transform(lambda s: s.rolling(win3).min())
    P["b_mean_liq_3"] = g["liq"].transform(lambda s: s.rolling(win3).mean())
    P["b_below_0"] = np.where(P["liq"].isna(), np.nan, (P["liq"] < 0).astype(float))
    P["b_below_half_runway"] = np.where(
        P["liq"].isna() | P["out3"].isna(),
        np.nan,
        (P["liq"] < 0.5 * month_out).astype(float),
    )

    sd6 = g["liq"].transform(lambda s: s.rolling(win6).std())
    mean_abs_out = g["op_out"].transform(lambda s: s.abs().rolling(win6).mean()) * out_scale
    P["b_bal_vol"] = sd6 / np.maximum(mean_abs_out, 1.0)

    prev = g["liq"].shift(1)
    onset = pd.Series(
        np.where(P["liq"].isna(), np.nan, ((prev >= 0) & (P["liq"] < 0)).astype(float)),
        index=P.index,
    )
    P["b_neg_episodes"] = onset.groupby(P["company_id"], sort=False).transform(
        lambda s: s.rolling(win6).sum()
    )

    return P[["company_id", "period", *B_COLS]]
