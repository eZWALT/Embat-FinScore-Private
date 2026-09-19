"""Y7 — concentration shock (binary).

Loss of the current top AR customer next quarter, optionally followed by a
sustained operational-inflow drop. Invoice HHI / top-1 are recomputed here
from `invoices` (3-month trailing AR book). Family D is not imported.

Definitions (horizon = next 3 calendar months = “next quarter”):

- y7_top1_lost: the AR counterparty with the largest share of |amount| in
  months t-2..t has 0 AR issuance in t+1..t+3 (share drops to 0).
- y7_top1_lost_inflow: that event AND op_in in each of t+1, t+2, t+3 is
  more than 25% below the trailing-3m mean op_in at t (sustained; not a
  one-month crash).

Forbidden X for models that predict these labels: family D.
Thresholds are fixed from the catalogue; they are not fit on train or holdout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.evaluate.protocol import auroc
from analysis.features.common import LAST_M, MONTHS, train_mask
from analysis.targets.y1_forecast import _month_key, cash_month_panel

Y7_COLS = ["y7_top1_lost", "y7_top1_lost_inflow"]

HORIZON = 3
ID_WIN = 3  # trailing months used to name the top customer / HHI
INFLOW_DROP = 0.25  # each future month < (1 - this) * trailing-3m mean

META = {
    "name": "y7_concentration_shock",
    "horizon": HORIZON,
    "source_tables": ["invoices", "transactions"],
    "forbidden_x_families": ["d"],
    "literature": (
        "Perez-Salazar, Marquez, Vidal-Silva 2026 (Computers 15:135): "
        "supplier HHI and operational volatility beat revenue level for "
        "micro-enterprise solvency. Y7 is the customer-side analogue: "
        "loss of the current top-1 AR counterparty, with a sustained "
        "inflow drop so the event is not a silent ERP gap."
    ),
    "kind": "binary",
    "columns": list(Y7_COLS),
    "definitions": {
        "y7_top1_lost": (
            "1 if the top-1 AR counterparty of the trailing 3-month invoice "
            "book (HHI / top-1 recomputed here) has 0 AR amount in t+1..t+3"
        ),
        "y7_top1_lost_inflow": (
            "1 if y7_top1_lost and op_in in each of t+1..t+3 is < 75% of "
            "the trailing-3m mean op_in at t (baseline > 0)"
        ),
    },
}

SOURCE_TABLES = META["source_tables"]


def _ar_month_cp(con) -> pd.DataFrame:
    """Monthly AR |amount| by company × counterparty (issuance month)."""
    df = con.execute(
        """
        SELECT company_id,
               CAST(date_trunc('month', issuance_date) AS DATE) AS month,
               counterparty_id,
               SUM(abs(amount)) AS amt
        FROM invoices
        WHERE document_type = 'invoice'
          AND status <> 'cancel'
          AND amount > 0
          AND issuance_date IS NOT NULL
          AND issuance_date < TIMESTAMP '2026-09-01'
          AND counterparty_id IS NOT NULL
          AND length(trim(CAST(counterparty_id AS VARCHAR))) > 0
        GROUP BY 1, 2, 3
        """
    ).df()
    df["company_id"] = df["company_id"].astype(str)
    df["month"] = pd.to_datetime(df["month"])
    df["counterparty_id"] = df["counterparty_id"].astype(str)
    return df


def _roll_cp(mcp: pd.DataFrame, month_offsets: tuple[int, ...]) -> pd.DataFrame:
    """Map each issuance month into the company-periods whose window contains it."""
    parts = []
    for off in month_offsets:
        tmp = mcp.copy()
        tmp["period"] = tmp["month"] + pd.DateOffset(months=int(off))
        parts.append(tmp)
    w = pd.concat(parts, ignore_index=True)
    return w.groupby(["company_id", "period", "counterparty_id"], as_index=False)["amt"].sum()


def _invoice_top1_panel(con) -> pd.DataFrame:
    """Company-month top-1 AR id, share, HHI, and next-quarter amount for that id.

    Identification window = months t-2..t (complete from MONTHS[2]).
    Next quarter = months t+1..t+3 (complete when t+3 <= LAST_M).
    HHI = sum_i share_i^2 on the identification window (invoice AR only).
    """
    mcp = _ar_month_cp(con)
    if mcp.empty:
        return pd.DataFrame(
            columns=[
                "company_id",
                "period",
                "top1_id",
                "top1_share",
                "hhi",
                "n_cp",
                "nxt_amt",
                "y7_top1_lost",
            ]
        )

    # Issuance in month m sits in the trailing-3m book of t = m, m+1, m+2.
    id_win = _roll_cp(mcp, (0, 1, 2))
    # Issuance in month m sits in the next-quarter book of t = m-1, m-2, m-3.
    nxt_win = _roll_cp(mcp, (-1, -2, -3))

    id_lo = MONTHS[ID_WIN - 1]
    id_hi = LAST_M - pd.DateOffset(months=HORIZON)
    id_win = id_win[(id_win["period"] >= id_lo) & (id_win["period"] <= id_hi)]
    nxt_win = nxt_win[(nxt_win["period"] >= id_lo) & (nxt_win["period"] <= id_hi)]

    tot = id_win.groupby(["company_id", "period"])["amt"].transform("sum")
    id_win = id_win.loc[tot > 0].copy()
    id_win["share"] = id_win["amt"] / tot.loc[id_win.index]
    id_win["share2"] = id_win["share"] ** 2

    book = id_win.groupby(["company_id", "period"], as_index=False).agg(
        hhi=("share2", "sum"),
        n_cp=("counterparty_id", "nunique"),
        tot=("amt", "sum"),
    )
    idx = id_win.groupby(["company_id", "period"])["amt"].idxmax()
    top = id_win.loc[idx, ["company_id", "period", "counterparty_id", "share"]].rename(
        columns={"counterparty_id": "top1_id", "share": "top1_share"}
    )
    top = top.merge(book, on=["company_id", "period"], how="left")

    nxt = nxt_win.rename(columns={"amt": "nxt_amt", "counterparty_id": "top1_id"})
    top = top.merge(nxt, on=["company_id", "period", "top1_id"], how="left")
    top["nxt_amt"] = top["nxt_amt"].fillna(0.0)
    top["y7_top1_lost"] = (top["nxt_amt"] <= 0.0).astype(float)
    return top


def _inflow_drop25(panel: pd.DataFrame) -> pd.Series:
    """1 if each of the next 3 months is < 75% of trailing-3m mean op_in (> 0)."""
    g = panel.groupby("company_id", sort=False)
    f1 = g["op_in"].shift(-1)
    f2 = g["op_in"].shift(-2)
    f3 = g["op_in"].shift(-3)
    base = g["op_in"].transform(lambda s: s.rolling(ID_WIN, min_periods=ID_WIN).mean())
    cut = (1.0 - INFLOW_DROP) * base
    ok = f1.notna() & f2.notna() & f3.notna() & base.notna() & (base > 0)
    return pd.Series(
        np.where(ok, ((f1 < cut) & (f2 < cut) & (f3 < cut)).astype(float), np.nan),
        index=panel.index,
    )


def build(con, grid: pd.DataFrame) -> pd.DataFrame:
    """Return company_id, period and the two Y7 binaries (0/1/NaN)."""
    out = grid[["company_id", "period"]].copy()
    out["company_id"] = out["company_id"].astype(str)
    out["period"] = pd.to_datetime(out["period"])
    out["_m"] = _month_key(out["period"])

    top = _invoice_top1_panel(con)
    cash = cash_month_panel(con).sort_values(["company_id", "month"]).reset_index(drop=True)
    cash["company_id"] = cash["company_id"].astype(str)
    cash["drop25"] = _inflow_drop25(cash)

    lab = top[["company_id", "period", "y7_top1_lost"]].rename(columns={"period": "_m"})
    drop = cash[["company_id", "month", "drop25"]].rename(columns={"month": "_m"})
    lab = lab.merge(drop, on=["company_id", "_m"], how="left")
    lab["y7_top1_lost_inflow"] = np.where(
        lab["y7_top1_lost"].notna() & lab["drop25"].notna(),
        ((lab["y7_top1_lost"] == 1.0) & (lab["drop25"] == 1.0)).astype(float),
        np.nan,
    )

    out = out.merge(
        lab[["company_id", "_m", *Y7_COLS]],
        on=["company_id", "_m"],
        how="left",
    ).drop(columns=["_m"])
    return out.reset_index(drop=True)


def train_acceptance(y7: pd.DataFrame, op_in: pd.Series, is_train: pd.Series) -> pd.DataFrame:
    """Base rates and size AUROC on train company-months (no threshold search)."""
    rows = []
    for col in Y7_COLS:
        y = y7.loc[is_train, col]
        ok = y.notna()
        n = int(ok.sum())
        rate = float(y[ok].mean()) if n else float("nan")
        auc = auroc(y, np.log1p(pd.to_numeric(op_in.loc[is_train], errors="coerce").abs()))
        auc_abs = float(max(auc, 1.0 - auc)) if np.isfinite(auc) else float("nan")
        in_rate = bool(n and 0.05 <= rate <= 0.30)
        size_ok = bool(np.isfinite(auc_abs) and auc_abs < 0.60)
        rows.append(
            {
                "column": col,
                "n_train": n,
                "n_pos": int((y == 1).sum()),
                "base_rate": rate,
                "size_auroc": auc,
                "size_auroc_two_sided": auc_abs,
                "accepted": in_rate and size_ok,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from analysis.features.common import connect
    from analysis.features.grid import monthly_grid

    con = connect()
    grid = monthly_grid(con)[["company_id", "period"]]
    y7 = build(con, grid)
    keys = grid[["company_id", "period"]].copy()
    keys["company_id"] = keys["company_id"].astype(str)
    keys["period"] = pd.to_datetime(keys["period"])
    tr = train_mask(keys["company_id"])
    panel = cash_month_panel(con).rename(columns={"month": "period"})
    panel["company_id"] = panel["company_id"].astype(str)
    panel["period"] = pd.to_datetime(panel["period"])
    op = keys.merge(panel[["company_id", "period", "op_in"]], on=["company_id", "period"], how="left")["op_in"]
    print("train company-months", int(tr.sum()), "companies", keys.loc[tr, "company_id"].nunique())
    acc = train_acceptance(y7, op, tr)
    print("Y7 train acceptance")
    print(acc.to_string(index=False))
    con.close()
