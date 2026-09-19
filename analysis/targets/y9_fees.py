"""Y9 — sustained fee/interest pressure (binary).

FinRegLab NSF/fee style: bank-statement fee and interest burden as a
distress signal. Not a bankruptcy label.

Definitions (horizon = 3 future calendar months; no look-ahead):

- y9_fee_r_ownp80: fee_r = (fee+interest)_3m / max(in3, 1) exceeds that
  company's own-history p80 for 3 consecutive future months. p80 uses
  only that company's months <= t (never future, never a pooled
  percentile, so holdout never enters a train cut).
- y9_fee_spike: fee_r at least doubles vs the trailing-6m mean of fee_r
  at t, in 2 of the next 3 months. Doubling is undefined when the
  trailing mean is 0 or missing (NaN, not 0) — a zero base cannot
  double.

Flows are computed here from `transactions` via `CAT_MAP`
(fee / interest_charge → fin_cost; op_in categories → in3). This
module does not import `analysis.features.debt` or cashflow.

Forbidden later X: family F and prefixes f_, a_fin_cost, a_fc
(same-column leak: f_fc_r / a_fin_cost are this Y's raw material).

Literature: FinRegLab 2025, Sharpening the Focus — NSF counts and fee
pressure on bank statements predict SMB default.

Maps to brief questions 3 (who is turning) and 5 (why did it change).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.evaluate.protocol import assert_no_holdout, auroc, leakage_check
from analysis.features.common import CAT_MAP, LAST_M, MONTHS, train_mask

Y9_COLS = ["y9_fee_r_ownp80", "y9_fee_spike"]

HORIZON = 3
MIN_OWN_HIST = 6
TRAIL = 6
OWN_P = 0.80
SPIKE_K = 2.0
SPIKE_OF = 2  # 2 of next 3 months

_OP_IN = tuple(k for k, v in CAT_MAP.items() if v == "op_in")
_FIN_COST = tuple(k for k, v in CAT_MAP.items() if v == "fin_cost")

META = {
    "name": "y9_fee_pressure",
    "horizon": HORIZON,
    "source_tables": ["transactions"],
    "forbidden_x_families": ["f"],
    "forbidden_x_prefixes": ["f_", "a_fin_cost", "a_fc"],
    "literature": (
        "FinRegLab 2025, Sharpening the Focus: NSF counts and fee/interest "
        "pressure on bank statements predict SMB default (young firms "
        "especially). Y9 is sustained fee+interest / inflow, not a "
        "bankruptcy label."
    ),
    "kind": "binary",
    "columns": list(Y9_COLS),
    "accepted": {
        "y9_fee_r_ownp80": True,
        "y9_fee_spike": True,
    },
    "brief_questions": ["turning", "why"],
    "definitions": {
        "y9_fee_r_ownp80": (
            "1 if fee_r = (fee+interest)_3m / max(in3, 1) exceeds that "
            "company's own expanding p80 (months <= t, min 6 finite) in "
            "each of t+1, t+2, t+3"
        ),
        "y9_fee_spike": (
            "1 if fee_r >= 2 * trailing-6m mean(fee_r) at t in at least "
            "2 of t+1..t+3. NaN when the 6m mean is 0 or missing "
            "(doubling undefined) or the 3-month horizon is incomplete"
        ),
    },
}

SOURCE_TABLES = META["source_tables"]


def _sql_in(values: tuple[str, ...]) -> str:
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


def _month_key(period: pd.Series) -> pd.Series:
    return pd.to_datetime(period).dt.to_period("M").dt.to_timestamp()


def _has_horizon(month: pd.Series, h: int) -> pd.Series:
    cutoff = LAST_M - pd.DateOffset(months=h)
    return month <= cutoff


def _expanding_quantile_skipna(s: pd.Series, q: float, min_periods: int) -> pd.Series:
    """Expanding percentile that counts only finite observations."""
    out = np.full(len(s), np.nan, dtype=float)
    vals: list[float] = []
    for i, v in enumerate(s.to_numpy(dtype=float)):
        if np.isfinite(v):
            vals.append(float(v))
        if len(vals) >= min_periods:
            out[i] = float(np.quantile(vals, q))
    return pd.Series(out, index=s.index)


def monthly_fee_flows(con) -> pd.DataFrame:
    """Monthly op_in and fin_cost (fee + interest_charge) from transactions.

    Outflow-side fin_cost is negated (same bank-sign convention as Y4).
    """
    df = con.execute(
        f"""
        SELECT
          t.company_id,
          CAST(date_trunc('month', t."date") AS DATE) AS month,
          SUM(CASE WHEN t.category IN ({_sql_in(_OP_IN)}) THEN t.amount ELSE 0 END) AS op_in,
          -SUM(CASE WHEN t.category IN ({_sql_in(_FIN_COST)}) THEN t.amount ELSE 0 END) AS fin_cost
        FROM transactions t
        WHERE t."date" < TIMESTAMP '2026-09-01'
        GROUP BY 1, 2
        """
    ).df()
    df["company_id"] = df["company_id"].astype(str)
    df["month"] = pd.to_datetime(df["month"])
    return df


def fee_month_panel(con) -> pd.DataFrame:
    """Dense company × month path for Y9.

    Columns: company_id, month, op_in, fin_cost, in3, fc3, fee_r.
    Months after first activity with no txs are 0-filled.
    ``fee_r`` is unclipped so a true doubling is not capped at 1.
    """
    flows = monthly_fee_flows(con)
    first = flows.groupby("company_id")["month"].min().rename("first_m")
    panel = pd.MultiIndex.from_product(
        [first.index, MONTHS], names=["company_id", "month"]
    ).to_frame(index=False)
    panel["company_id"] = panel["company_id"].astype(str)
    panel = panel.merge(first, left_on="company_id", right_index=True)
    panel = panel[panel["month"] >= panel["first_m"]]
    panel = panel.merge(flows, on=["company_id", "month"], how="left")
    panel["op_in"] = panel["op_in"].fillna(0.0)
    panel["fin_cost"] = panel["fin_cost"].fillna(0.0)
    panel = panel.sort_values(["company_id", "month"]).reset_index(drop=True)
    g = panel.groupby("company_id", sort=False)
    panel["in3"] = g["op_in"].transform(lambda s: s.rolling(3).sum())
    panel["fc3"] = g["fin_cost"].transform(lambda s: s.rolling(3).sum())
    panel["fee_r"] = panel["fc3"] / np.maximum(panel["in3"], 1.0)
    return panel.drop(columns=["first_m"])


def build(con, grid: pd.DataFrame) -> pd.DataFrame:
    """Return company_id, period and the two Y9 binaries (0/1/NaN)."""
    need = {"company_id", "period"}
    if not need <= set(grid.columns):
        raise ValueError("grid must have company_id, period")

    panel = fee_month_panel(con)
    cid = panel["company_id"]
    g = panel.groupby(cid, sort=False)

    panel["own_p80"] = g["fee_r"].transform(
        lambda s: _expanding_quantile_skipna(s, OWN_P, MIN_OWN_HIST)
    )
    panel["trail6"] = g["fee_r"].transform(
        lambda s: s.rolling(TRAIL, min_periods=TRAIL).mean()
    )

    f1 = g["fee_r"].shift(-1)
    f2 = g["fee_r"].shift(-2)
    f3 = g["fee_r"].shift(-3)
    has_h3 = _has_horizon(panel["month"], HORIZON)
    fut_ok = has_h3 & f1.notna() & f2.notna() & f3.notna()

    hi = (f1 > panel["own_p80"]) & (f2 > panel["own_p80"]) & (f3 > panel["own_p80"])
    panel["y9_fee_r_ownp80"] = np.where(
        fut_ok & panel["own_p80"].notna(), hi.astype(float), np.nan
    )

    # Doubling requires a strictly positive trailing mean (0 * 2 is not a double).
    base_ok = panel["trail6"].notna() & (panel["trail6"] > 0)
    n_dbl = (
        (f1 >= SPIKE_K * panel["trail6"]).astype(float)
        + (f2 >= SPIKE_K * panel["trail6"]).astype(float)
        + (f3 >= SPIKE_K * panel["trail6"]).astype(float)
    )
    panel["y9_fee_spike"] = np.where(
        fut_ok & base_ok, (n_dbl >= SPIKE_OF).astype(float), np.nan
    )

    out = grid[["company_id", "period"]].copy()
    out["company_id"] = out["company_id"].astype(str)
    out["period"] = pd.to_datetime(out["period"])
    out["_m"] = _month_key(out["period"])
    lab = panel[["company_id", "month", *Y9_COLS]].rename(columns={"month": "_m"})
    out = out.merge(lab, on=["company_id", "_m"], how="left").drop(columns=["_m"])
    return out.reset_index(drop=True)


def train_acceptance(y9: pd.DataFrame, op_in: pd.Series, is_train: pd.Series) -> pd.DataFrame:
    """Base rates and two-sided size AUROC on train company-months only.

    ACCEPTED when train base rate is in [5%, 30%] and
    max(auc, 1-auc) < 0.60. Holdout is never used.
    """
    assert_no_holdout(y9.loc[is_train, "company_id"])
    rows = []
    n_train = int(is_train.sum())
    for col in Y9_COLS:
        y = y9.loc[is_train, col]
        ok = y.notna()
        n = int(ok.sum())
        rate = float(y[ok].mean()) if n else float("nan")
        auc = auroc(y, np.log1p(pd.to_numeric(op_in.loc[is_train], errors="coerce").abs()))
        auc_abs = float(max(auc, 1.0 - auc)) if np.isfinite(auc) else float("nan")
        in_rate = bool(n and 0.05 <= rate <= 0.30)
        size_ok = bool(np.isfinite(auc_abs) and auc_abs < 0.60)
        accepted = bool(in_rate and size_ok)
        rows.append(
            {
                "column": col,
                "n_train": n_train,
                "n_labeled": n,
                "n_pos": int((y == 1).sum()),
                "coverage": n / n_train if n_train else float("nan"),
                "base_rate": rate,
                "size_auroc": auc,
                "size_auroc_two_sided": auc_abs,
                "accepted": accepted,
                "verdict": "ACCEPTED" if accepted else "REJECTED",
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from analysis.features.common import connect
    from analysis.features.grid import monthly_grid

    con = connect()
    grid = monthly_grid(con)[["company_id", "period"]]
    y9 = build(con, grid)
    keys = grid[["company_id", "period"]].copy()
    keys["company_id"] = keys["company_id"].astype(str)
    keys["period"] = pd.to_datetime(keys["period"])
    tr = train_mask(keys["company_id"])
    panel = fee_month_panel(con).rename(columns={"month": "period"})
    panel["company_id"] = panel["company_id"].astype(str)
    op = keys.merge(
        panel[["company_id", "period", "op_in"]],
        on=["company_id", "period"],
        how="left",
    )["op_in"]
    print("rows", len(y9), "dups", int(y9.duplicated(["company_id", "period"]).sum()))
    print("train company-months", int(tr.sum()), "companies", keys.loc[tr, "company_id"].nunique())
    acc = train_acceptance(y9, op, tr)
    print("Y9 train acceptance")
    print(acc.to_string(index=False))
    print("META.forbidden_x_families", META["forbidden_x_families"])
    print("META.forbidden_x_prefixes", META["forbidden_x_prefixes"])
    leak = leakage_check(
        ["a_op_in", "a_fin_cost", "a_fc_r", "f_fc_r", "f_fin_cost", "b_runway"],
        "y9_fee_r_ownp80",
        forbidden_prefixes=META["forbidden_x_prefixes"],
    )
    print("leakage_check demo", leak)
    con.close()
