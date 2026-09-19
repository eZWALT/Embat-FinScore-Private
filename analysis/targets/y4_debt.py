"""Y4 — debt-pressure targets (binary).

Forbidden X for models that predict these labels: family F (debt / financing
columns). Thresholds are fixed from the catalogue; they are not fit on train
or holdout.

Flows are computed here from `transactions` via `CAT_MAP`. This module does
not import `analysis.features.debt`.

`outstanding_gt_granted` is a 2026-09-01 snapshot flag on `debt_products`
(and `debt_schedule_config`). It is not a time series — do not treat it as
something that “appears” in 2024.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.features.common import AS_OF, CAT_MAP, LAST_M, MONTHS

Y4_COLS = [
    "y4_ds_r_gt05_sust",
    "y4_ds_r_double",
    "y4_new_facility_after_dip",
    "y4_ogtg_appear",
]

META = {
    "name": "y4_debt_pressure",
    "horizon": 6,
    "source_tables": ["transactions", "debt_products"],
    "forbidden_x_families": ["f"],
    "literature": (
        "Yao, Levy-Chapira, Margaryan 2017 (arXiv 1707.00757): credit-line "
        "violations and cash inflows are central for corporate default. "
        "Catalogue Y4: debt-service / inflow rise (sustained or doubling) "
        "and new facility after a cash dip (distress borrowing)."
    ),
    "kind": "binary",
    "columns": list(Y4_COLS),
    "definitions": {
        "y4_ds_r_gt05_sust": (
            "1 if ds_r = debt_service_3m / max(in3, 1) > 0.5 in each of "
            "the next 3 months (t+1, t+2, t+3). debt_service is CAT_MAP "
            "debt_repayment only (fee / interest_charge are fin_cost)."
        ),
        "y4_ds_r_double": (
            "1 if ds_r at t+3 >= 2 * ds_r at t, both defined, and "
            "ds_r_t > 0.05 (floor so a near-zero base cannot explode). "
            "NaN when the floor or either side is missing."
        ),
        "y4_new_facility_after_dip": (
            "1 if a debt_products.created_at falls in (t, t+6m] AND net < 0 "
            "in at least 2 of the last 3 months at t. Dip uses operational "
            "net, not reconstructed liquidity (avoids copying family B)."
        ),
        "y4_ogtg_appear": (
            "Snapshot-only: 1 if the company has any debt_products row with "
            "outstanding_gt_granted at extract; populated only for 2026-08, "
            "NaN on every other month. Not an onset in 2024–2025."
        ),
    },
}

SOURCE_TABLES = META["source_tables"]

_OP_IN = tuple(k for k, v in CAT_MAP.items() if v == "op_in")
_OP_OUT = tuple(k for k, v in CAT_MAP.items() if v == "op_out")
_DEBT_SVC = tuple(k for k, v in CAT_MAP.items() if v == "debt_service")
_FIN_COST = tuple(k for k, v in CAT_MAP.items() if v == "fin_cost")


def _sql_in(values: tuple[str, ...]) -> str:
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


def _month_key(period: pd.Series) -> pd.Series:
    return pd.to_datetime(period).dt.to_period("M").dt.to_timestamp()


def _has_horizon(month: pd.Series, h: int) -> pd.Series:
    """t+h is inside the sample when month <= LAST_M - h months."""
    cutoff = LAST_M - pd.DateOffset(months=h)
    return month <= cutoff


def _auroc(score, y) -> float:
    d = pd.DataFrame({"s": score, "y": y}).dropna()
    n1 = int((d["y"] == 1).sum())
    n0 = int((d["y"] == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = d["s"].rank()
    return float((r[d["y"] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def monthly_debt_flows(con) -> pd.DataFrame:
    """Monthly op_in, op_out, debt_service, fin_cost from transactions.

    Outflow groups are negated (same bank-sign convention as the pipeline).
    ``debt_service`` = debt_repayment; ``fin_cost`` = fee + interest_charge.
    """
    df = con.execute(
        f"""
        SELECT
          t.company_id,
          CAST(date_trunc('month', t."date") AS DATE) AS month,
          SUM(CASE WHEN t.category IN ({_sql_in(_OP_IN)}) THEN t.amount ELSE 0 END) AS op_in,
          -SUM(CASE WHEN t.category IN ({_sql_in(_OP_OUT)}) THEN t.amount ELSE 0 END) AS op_out,
          -SUM(CASE WHEN t.category IN ({_sql_in(_DEBT_SVC)}) THEN t.amount ELSE 0 END) AS debt_service,
          -SUM(CASE WHEN t.category IN ({_sql_in(_FIN_COST)}) THEN t.amount ELSE 0 END) AS fin_cost
        FROM transactions t
        WHERE t."date" < TIMESTAMP '2026-09-01'
        GROUP BY 1, 2
        """
    ).df()
    df["month"] = pd.to_datetime(df["month"])
    return df


def debt_month_panel(con) -> pd.DataFrame:
    """Dense company × month path for Y4.

    Columns: company_id, month, op_in, op_out, net, debt_service, fin_cost,
    in3, ds3, ds_r. Months after first activity with no txs are 0-filled.
    ``ds_r`` is unclipped so a true doubling is not capped at 2.
    """
    flows = monthly_debt_flows(con)
    first = flows.groupby("company_id")["month"].min().rename("first_m")
    panel = pd.MultiIndex.from_product(
        [first.index, MONTHS], names=["company_id", "month"]
    ).to_frame(index=False)
    panel = panel.merge(first, left_on="company_id", right_index=True)
    panel = panel[panel["month"] >= panel["first_m"]]
    panel = panel.merge(flows, on=["company_id", "month"], how="left")
    for c in ("op_in", "op_out", "debt_service", "fin_cost"):
        panel[c] = panel[c].fillna(0.0)
    panel["net"] = panel["op_in"] - panel["op_out"]
    panel = panel.sort_values(["company_id", "month"]).reset_index(drop=True)
    g = panel.groupby("company_id", sort=False)
    panel["in3"] = g["op_in"].transform(lambda s: s.rolling(3).sum())
    panel["ds3"] = g["debt_service"].transform(lambda s: s.rolling(3).sum())
    panel["ds_r"] = panel["ds3"] / np.maximum(panel["in3"], 1.0)
    return panel.drop(columns=["first_m"])


def _new_facility_flag(panel: pd.DataFrame, fac: pd.DataFrame) -> pd.Series:
    """1 if any created_at is in (month, month+6m], else 0."""
    keys = panel[["company_id", "month"]]
    if fac.empty:
        return pd.Series(0.0, index=panel.index)
    fac = fac.copy()
    fac["created_at"] = pd.to_datetime(fac["created_at"])
    m = keys.merge(fac, on="company_id", how="left")
    t6 = m["month"] + pd.DateOffset(months=6)
    hit = (
        m["created_at"].notna()
        & (m["created_at"] > m["month"])
        & (m["created_at"] <= t6)
    )
    flagged = (
        m.loc[hit, ["company_id", "month"]]
        .drop_duplicates()
        .assign(_hit=1.0)
    )
    out = keys.merge(flagged, on=["company_id", "month"], how="left")["_hit"]
    return out.fillna(0.0).astype(float)


def build(con, grid: pd.DataFrame) -> pd.DataFrame:
    """Return company_id, period and the four Y4 binaries (0/1/NaN)."""
    panel = debt_month_panel(con)
    cid = panel["company_id"]
    g = panel.groupby(cid, sort=False)

    has_h3 = _has_horizon(panel["month"], 3)
    ds = panel["ds_r"]
    d1 = g["ds_r"].shift(-1)
    d2 = g["ds_r"].shift(-2)
    d3 = g["ds_r"].shift(-3)
    hi1 = d1 > 0.5
    hi2 = d2 > 0.5
    hi3 = d3 > 0.5
    fut_ok = has_h3 & d1.notna() & d2.notna() & d3.notna()
    panel["y4_ds_r_gt05_sust"] = np.where(
        fut_ok, (hi1 & hi2 & hi3).astype(float), np.nan
    )

    both = has_h3 & ds.notna() & d3.notna() & (ds > 0.05)
    panel["y4_ds_r_double"] = np.where(both, (d3 >= 2.0 * ds).astype(float), np.nan)

    neg = (panel["net"] < 0).astype(float)
    n0 = neg
    n1 = neg.groupby(cid).shift(1)
    n2 = neg.groupby(cid).shift(2)
    dip = pd.Series(
        np.where(n1.notna() & n2.notna(), ((n0 + n1 + n2) >= 2).astype(float), np.nan),
        index=panel.index,
    )
    fac = con.execute(
        """
        SELECT company_id, created_at
        FROM debt_products
        WHERE created_at IS NOT NULL
        """
    ).df()
    panel["new_fac"] = _new_facility_flag(panel, fac).to_numpy()
    has_h6 = (panel["month"] + pd.DateOffset(months=6)) <= AS_OF
    panel["y4_new_facility_after_dip"] = np.where(
        has_h6 & dip.notna(),
        ((dip == 1.0) & (panel["new_fac"] == 1.0)).astype(float),
        np.nan,
    )

    og = con.execute(
        """
        SELECT company_id,
               MAX(CASE WHEN outstanding_gt_granted THEN 1 ELSE 0 END) AS ogtg
        FROM debt_products
        GROUP BY 1
        """
    ).df()
    panel = panel.merge(og, on="company_id", how="left")
    panel["y4_ogtg_appear"] = np.where(
        panel["month"].eq(LAST_M), panel["ogtg"].fillna(0.0), np.nan
    )

    out = grid[["company_id", "period"]].copy()
    out["period"] = pd.to_datetime(out["period"])
    out["_m"] = _month_key(out["period"])
    lab = panel[["company_id", "month", *Y4_COLS]].rename(columns={"month": "_m"})
    out = out.merge(lab, on=["company_id", "_m"], how="left").drop(columns=["_m"])
    return out.reset_index(drop=True)


def size_auroc(y: pd.Series, op_in: pd.Series) -> float:
    """AUROC of log1p(|monthly op_in|) vs a binary label."""
    return _auroc(np.log1p(pd.to_numeric(op_in, errors="coerce").abs()), y)


def train_acceptance(y4: pd.DataFrame, op_in: pd.Series, is_train: pd.Series) -> pd.DataFrame:
    """Base rates and size AUROC on train company-months only (no threshold search).

    ACCEPTED when train base rate is in [5%, 30%] and size AUROC < 0.60.
    """
    rows = []
    for col in Y4_COLS:
        y = y4.loc[is_train, col]
        ok = y.notna()
        n = int(ok.sum())
        rate = float(y[ok].mean()) if n else float("nan")
        auc = size_auroc(y, op_in.loc[is_train])
        rate_ok = bool(n and 0.05 <= rate <= 0.30)
        size_ok = bool(np.isfinite(auc) and auc < 0.60)
        rows.append(
            {
                "column": col,
                "n_train": n,
                "n_pos": int((y == 1).sum()),
                "base_rate": rate,
                "size_auroc": auc,
                "size_proxy": bool(auc >= 0.60) if np.isfinite(auc) else False,
                "accepted": bool(rate_ok and size_ok),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from analysis.features.common import connect, train_mask
    from analysis.features.grid import monthly_grid

    con = connect()
    grid = monthly_grid(con)
    y4 = build(con, grid)
    keys = grid[["company_id", "period"]].copy()
    keys["period"] = pd.to_datetime(keys["period"])
    tr = train_mask(keys["company_id"])
    panel = debt_month_panel(con).rename(columns={"month": "period"})
    op = keys.merge(
        panel[["company_id", "period", "op_in"]],
        on=["company_id", "period"],
        how="left",
    )["op_in"]
    print("train company-months", int(tr.sum()), "companies", keys.loc[tr, "company_id"].nunique())
    acc = train_acceptance(y4, op, tr)
    print("Y4 train acceptance")
    print(acc.to_string(index=False))
    print("META.forbidden_x_families", META["forbidden_x_families"])
    con.close()
