"""Y1 — reconstruction / forecast targets (continuous).

Labels are values at t+h, so they intentionally use future cash and the
reconstructed balance path. Features must still stop at period end.

Horizons are calendar months (h = 1, 3). Weekly grid rows are aligned to
the containing month so the same columns can be joined.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.features.common import CAT_MAP, LAST_M, MONTHS

Y1_COLS = [
    "y1_net_h1",
    "y1_net_h3",
    "y1_in_h1",
    "y1_in_h3",
    "y1_liq_h1",
    "y1_liq_h3",
]

META = {
    "name": "y1_forecast",
    "horizon": 3,
    "source_tables": ["transactions", "balances", "banking_products"],
    "forbidden_x_families": [],
    "literature": (
        "Yao, Levy-Chapira, Margaryan 2017 (arXiv 1707.00757): checking-account "
        "activity beats ratios for corporate default. Y1 is the cash-path "
        "forecast target (cross-horizon, not a same-column label)."
    ),
    "kind": "continuous",
    "columns": list(Y1_COLS),
    "definition": (
        "y1_net_h* = operational net (op_in - op_out) at t+h; "
        "y1_in_h* = operational inflow at t+h; "
        "y1_liq_h* = reconstructed month-end liquidity at t+h "
        "(same unwind as score_pipeline._liquidity). "
        "None of these are known at t."
    ),
}

SOURCE_TABLES = META["source_tables"]

_OP_IN = tuple(k for k, v in CAT_MAP.items() if v == "op_in")
_OP_OUT = tuple(k for k, v in CAT_MAP.items() if v == "op_out")


def _sql_in(values: tuple[str, ...]) -> str:
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


def monthly_ops(con) -> pd.DataFrame:
    """Monthly operational inflow/outflow. Outflow stored as a positive magnitude."""
    df = con.execute(
        f"""
        SELECT
          t.company_id,
          CAST(date_trunc('month', t."date") AS DATE) AS month,
          SUM(CASE WHEN t.category IN ({_sql_in(_OP_IN)}) THEN t.amount ELSE 0 END) AS op_in,
          -SUM(CASE WHEN t.category IN ({_sql_in(_OP_OUT)}) THEN t.amount ELSE 0 END) AS op_out
        FROM transactions t
        WHERE t."date" < TIMESTAMP '2026-09-01'
        GROUP BY 1, 2
        """
    ).df()
    df["month"] = pd.to_datetime(df["month"])
    return df


def reconstruct_liq(con) -> pd.DataFrame:
    """Month-end liquidity: snapshot balance minus subsequent product flows.

    Same reconstruction as ``score_pipeline._liquidity`` (checking/saving/tpv).
    """
    bal = con.execute(
        """
        SELECT b.product_id, p.company_id, b.balance
        FROM balances b
        JOIN banking_products p ON b.product_id = p.product_id
        WHERE p.type IN ('checking', 'saving', 'tpv') AND b.balance IS NOT NULL
        """
    ).df()
    flow = con.execute(
        """
        SELECT t.product_id,
               CAST(date_trunc('month', t."date") AS DATE) AS month,
               SUM(t.amount) AS flow
        FROM transactions t
        WHERE t.product_id IN (
            SELECT b.product_id
            FROM balances b
            JOIN banking_products p ON b.product_id = p.product_id
            WHERE p.type IN ('checking', 'saving', 'tpv') AND b.balance IS NOT NULL
        )
        GROUP BY 1, 2
        """
    ).df()
    flow["month"] = pd.to_datetime(flow["month"])
    months = list(MONTHS) + [LAST_M + pd.DateOffset(months=1)]
    grid = pd.MultiIndex.from_product(
        [bal["product_id"].unique(), months], names=["product_id", "month"]
    ).to_frame(index=False)
    grid = grid.merge(flow, on=["product_id", "month"], how="left").fillna({"flow": 0.0})
    grid = grid.merge(bal, on="product_id").sort_values(["product_id", "month"]).reset_index(drop=True)
    g = grid.groupby("product_id")["flow"]
    after = g.transform("sum") - g.cumsum()
    grid["end_bal"] = grid["balance"] - after
    grid = grid[grid["month"] <= LAST_M]
    return (
        grid.groupby(["company_id", "month"], as_index=False)["end_bal"]
        .sum()
        .rename(columns={"end_bal": "liq"})
    )


def cash_month_panel(con) -> pd.DataFrame:
    """Dense company × month path used by Y1 and Y2.

    Columns: company_id, month, op_in, op_out, net, liq, out3, runway.
    Months without transactions after first activity have flows filled with 0.
    """
    ops = monthly_ops(con)
    liq = reconstruct_liq(con)
    first = ops.groupby("company_id")["month"].min().rename("first_m")
    companies = pd.Index(ops["company_id"].unique()).union(pd.Index(liq["company_id"].unique()))
    panel = pd.MultiIndex.from_product(
        [companies, MONTHS], names=["company_id", "month"]
    ).to_frame(index=False)
    panel = panel.merge(first, left_on="company_id", right_index=True, how="left")
    panel = panel[panel["first_m"].isna() | (panel["month"] >= panel["first_m"])]
    panel = panel.merge(ops, on=["company_id", "month"], how="left")
    panel["op_in"] = panel["op_in"].fillna(0.0)
    panel["op_out"] = panel["op_out"].fillna(0.0)
    panel["net"] = panel["op_in"] - panel["op_out"]
    panel = panel.merge(liq, on=["company_id", "month"], how="left")
    panel = panel.sort_values(["company_id", "month"]).reset_index(drop=True)
    g = panel.groupby("company_id", sort=False)
    panel["out3"] = g["op_out"].transform(lambda s: s.rolling(3).sum())
    panel["runway"] = (panel["liq"] / np.maximum(panel["out3"] / 3.0, 1.0)).clip(-6, 24)
    return panel.drop(columns=["first_m"])


def _month_key(period: pd.Series) -> pd.Series:
    return pd.to_datetime(period).dt.to_period("M").dt.to_timestamp()


def build(con, grid: pd.DataFrame) -> pd.DataFrame:
    """Return company_id, period and the six Y1 columns (values at t+h)."""
    panel = cash_month_panel(con).sort_values(["company_id", "month"]).reset_index(drop=True)
    g = panel.groupby("company_id", sort=False)
    panel["y1_net_h1"] = g["net"].shift(-1)
    panel["y1_net_h3"] = g["net"].shift(-3)
    panel["y1_in_h1"] = g["op_in"].shift(-1)
    panel["y1_in_h3"] = g["op_in"].shift(-3)
    panel["y1_liq_h1"] = g["liq"].shift(-1)
    panel["y1_liq_h3"] = g["liq"].shift(-3)
    # Dense panel through LAST_M: shift(-h) is NaN when t+h is past the sample.

    out = grid[["company_id", "period"]].copy()
    out["period"] = pd.to_datetime(out["period"])
    out["_m"] = _month_key(out["period"])
    lab = panel[["company_id", "month", *Y1_COLS]].rename(columns={"month": "_m"})
    out = out.merge(lab, on=["company_id", "_m"], how="left").drop(columns=["_m"])
    return out.reset_index(drop=True)
