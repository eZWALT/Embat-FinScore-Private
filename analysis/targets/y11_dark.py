"""Y11 — cash-only labels among companies with no invoice book.

Population (built from DuckDB, never hardcoded): train companies with
zero book invoices (`document_type='invoice'`, `status<>'cancel'`,
`amount<>0`, issuance present). Join QA: 470 train / 32 holdout.
360 sit in all-dark groups; 110 have an invoiced train sibling.

This module does **not** invent cobros. Families D and E are forbidden
later X for every column here (no book to leak; HHI/top-1 stay out).
Restricted Y2 also forbids B. Fee cousins also forbid F / `a_fin_cost`
/ `a_fc`.

Assigned tries (honest CLOSE/PARK is success):

- y11_y2_neg_2of3_dark: accepted `y2_neg_2of3` restricted to the dark
  set. Same definition — CLOSE as "Y2 on dark companies", not a new Y.
- y11_y3_recover_cash_6m_dark: accepted `y3_recover_cash_6m` restricted
  to the dark set. Same definition — CLOSE as "Y3 on dark companies".
- y11_neg_2of3_dark: CAT_MAP operational net < 0 in ≥ 2 of the next 3
  months, dark companies only. Not family-B liquidity. If Spearman vs
  Y2 ≥ 0.95, PARK as a rewrite. Written even when the Y2 restriction
  is defined, so the ρ screen is visible.
- y11_fee_ownp80_dark: Y9-style fee_r own-p80 among the dark set. If
  ρ vs `y9_fee_r_ownp80` ≥ 0.9 on the overlap, CLOSE as a subset.

Extra probes kept in-module for the 30-minute iterate (not a merge):

- y11_fee_spike_dark: Y9 spike method, dark only.
- y11_zero_in_2of3_dark: op_in == 0 in ≥ 2 of the next 3 months (cash
  activity hole; likely inverse-size — check the gate).

Assembler skips this module (`SKIP_ASSEMBLE_MODULES`). Do not write
`targets.parquet`. No 0–100. No `product/`.

Maps to brief Q3 (who is turning) / Q4 (dip vs fall) for companies
that will never have an invoice book. Cash-side only.

Iteration log (same module, not one-shot):
1. Restrict Y2/Y3 + fee mask + net 2-of-3 + zero-in. confirm_470. 7603 dark cm.
2. Y2-on-dark usable (9.14%, size 0.544). Net 2-of-3 base 47.7% PARK. Fee ρ=1 CLOSE.
3. Stricter cousins: 3-of-3, own-p20, net-recover-6m. Three numeric passes, not rewrites.
4. 360 vs 110: Y2 flat; Y3 recover 5.2% vs 12.1%. Size terciles: T1 gap +12pp (H).
5. Onset cousin (net≥0 now, then 3-of-3). 9.08%, size 0.563, acf1=0.14. Prefer over 3-of-3 state.
6. y2_onset_neg on the 470 still 1.29%; ρ vs net-onset −0.014. Not a rewrite of that reject.
"""
from __future__ import annotations

import csv
from datetime import datetime

import numpy as np
import pandas as pd

from analysis.evaluate.protocol import assert_no_holdout, auroc, leakage_check
from analysis.features.common import ANALYSIS, CAT_MAP, LAST_M, MONTHS, train_mask
from analysis.targets.y2_stress import build as build_y2
from analysis.targets.y3_recovery import build as build_y3
from analysis.targets.y9_fees import Y9_COLS, build as build_y9

ACCEPT_MD = ANALYSIS / "outputs" / "y11_acceptance.md"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
AGENT = "3d5ad8d8"
ROUND = "R4"
WAVE = "4"

BOOK = (
    "document_type = 'invoice' AND status <> 'cancel' "
    "AND amount <> 0 AND issuance_date IS NOT NULL"
)

HORIZON = 3
HORIZON_REC = 6
RECOVER_LEN = 3
MIN_OWN_HIST = 6
OWN_P20 = 0.20

_OP_IN = tuple(k for k, v in CAT_MAP.items() if v == "op_in")
_OP_OUT = tuple(k for k, v in CAT_MAP.items() if v == "op_out")

Y11_COLS = [
    "y11_y2_neg_2of3_dark",
    "y11_y3_recover_cash_6m_dark",
    "y11_neg_2of3_dark",
    "y11_neg_3of3_dark",
    "y11_neg_ownp20_2of3_dark",
    "y11_net_recover_6m_dark",
    "y11_neg_onset_3of3_dark",
    "y11_fee_ownp80_dark",
    "y11_fee_spike_dark",
    "y11_zero_in_2of3_dark",
]

REWRITE_Y2 = 0.95
REWRITE_Y9 = 0.90

META = {
    "name": "y11_dark_cash",
    "horizon": HORIZON,
    "source_tables": ["transactions", "invoices", "companies", "balances", "banking_products"],
    "forbidden_x_families": ["d", "e"],
    "forbidden_x_families_by_column": {
        "y11_y2_neg_2of3_dark": ["b", "d", "e"],
        "y11_y3_recover_cash_6m_dark": ["b", "d", "e"],
        "y11_neg_2of3_dark": ["a", "d", "e"],
        "y11_neg_3of3_dark": ["a", "d", "e"],
        "y11_neg_ownp20_2of3_dark": ["a", "d", "e"],
        "y11_net_recover_6m_dark": ["a", "d", "e"],
        "y11_neg_onset_3of3_dark": ["a", "d", "e"],
        "y11_fee_ownp80_dark": ["d", "e", "f"],
        "y11_fee_spike_dark": ["d", "e", "f"],
        "y11_zero_in_2of3_dark": ["a", "d", "e"],
    },
    "forbidden_x_prefixes": ["d_", "e_", "a_fin_cost", "a_fc"],
    "literature": (
        "Join QA 2026-09-19: 470 train companies have no book invoice "
        "(92.1% NULL ERP). Legal next is cash Y among the 470 using "
        "A/B/C/F/G/H — never D/E, never invented cobros. FinRegLab 2025 "
        "low/negative balances and NSF/fee; brief Q3/Q4 for the no-book set."
    ),
    "kind": "binary",
    "columns": list(Y11_COLS),
    "accepted": {
        "y11_y2_neg_2of3_dark": False,
        "y11_y3_recover_cash_6m_dark": False,
        "y11_neg_2of3_dark": False,
        "y11_neg_3of3_dark": False,
        "y11_neg_ownp20_2of3_dark": True,
        "y11_net_recover_6m_dark": True,
        "y11_neg_onset_3of3_dark": True,
        "y11_fee_ownp80_dark": False,
        "y11_fee_spike_dark": False,
        "y11_zero_in_2of3_dark": False,
    },
    "brief_questions": ["turning", "dip_vs_fall"],
    "definitions": {
        "y11_y2_neg_2of3_dark": (
            "y2_neg_2of3 restricted to companies with zero book invoices. "
            "Same definition (reconstructed liq < 0 in ≥ 2 of next 3). "
            "CLOSE as subset, not a new label."
        ),
        "y11_y3_recover_cash_6m_dark": (
            "y3_recover_cash_6m restricted to zero-book companies. "
            "Same stressed→3-in-6 runway>=3 definition. CLOSE as subset."
        ),
        "y11_neg_2of3_dark": (
            "1 if CAT_MAP operational net (op_in - op_out) < 0 in at "
            "least 2 of t+1..t+3, among zero-book companies only. "
            "Not family-B liquidity. PARK if Spearman vs y2_neg_2of3 ≥ 0.95."
        ),
        "y11_neg_3of3_dark": (
            "1 if CAT_MAP operational net < 0 in each of t+1..t+3, "
            "zero-book companies only. Stricter cousin of y11_neg_2of3_dark."
        ),
        "y11_neg_ownp20_2of3_dark": (
            "1 if CAT_MAP net is below that company's expanding p20 "
            "(months ≤ t, min 6 finite) in at least 2 of t+1..t+3. "
            "Company-relative, not a pooled cut. Dark only."
        ),
        "y11_net_recover_6m_dark": (
            "1 if CAT_MAP net < 0 at t (stressed) and some 3 consecutive "
            "months in t+1..t+6 have net ≥ 0. Cash-only Y3 cousin, no runway. "
            "PARK if Spearman vs y3_recover_cash_6m ≥ 0.95."
        ),
        "y11_neg_onset_3of3_dark": (
            "1 if CAT_MAP net ≥ 0 at t and net < 0 in each of t+1..t+3. "
            "Onset of a 3-month negative-net spell (Q3 turning), dark only."
        ),
        "y11_fee_ownp80_dark": (
            "Y9 fee_r own-p80 (fee+interest)_3m / max(in3,1) above own "
            "expanding p80 in each of t+1..t+3, zero-book companies only. "
            "CLOSE if Spearman vs y9_fee_r_ownp80 ≥ 0.9."
        ),
        "y11_fee_spike_dark": (
            "Y9 fee_r spike (2× trail-6m in ≥ 2 of next 3), dark only."
        ),
        "y11_zero_in_2of3_dark": (
            "1 if CAT_MAP op_in == 0 in at least 2 of t+1..t+3, dark only. "
            "Activity hole; likely inverse-size."
        ),
    },
}

SOURCE_TABLES = META["source_tables"]


def _sql_in(values: tuple[str, ...]) -> str:
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


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


def _any_recover_window(g: pd.core.groupby.generic.DataFrameGroupBy, col: str) -> pd.Series:
    """True if any 3-month window starting at t+1..t+4 is all 1.0 on `col`."""
    rec = pd.Series(False, index=g.obj.index)
    for start in range(1, HORIZON_REC - RECOVER_LEN + 2):
        a = g[col].shift(-start)
        b = g[col].shift(-(start + 1))
        c = g[col].shift(-(start + 2))
        rec = rec | ((a == 1.0) & (b == 1.0) & (c == 1.0))
    return rec


def _two_sided(auc: float) -> float:
    if not np.isfinite(auc):
        return float("nan")
    return float(max(auc, 1.0 - auc))


def book_invoice_ids(con) -> set[str]:
    """Companies with at least one book invoice. DuckDB, not a hardcoded list."""
    df = con.execute(
        f"""
        SELECT DISTINCT CAST(company_id AS VARCHAR) AS company_id
        FROM invoices
        WHERE {BOOK}
        """
    ).df()
    return set(df["company_id"].astype(str))


def company_groups(con) -> pd.DataFrame:
    df = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               CAST(group_id AS VARCHAR) AS group_id
        FROM companies
        """
    ).df()
    df["company_id"] = df["company_id"].astype(str)
    df["group_id"] = df["group_id"].astype(str)
    return df


def dark_population(con) -> dict:
    """Train/holdout dark sets + 360/110 group split. IDs from DuckDB."""
    cos = company_groups(con)
    book = book_invoice_ids(con)
    cos["has_book"] = cos["company_id"].isin(book)
    tr = train_mask(cos["company_id"])
    train = cos.loc[tr].copy()
    hold = cos.loc[~tr].copy()
    train_dark = train.loc[~train["has_book"]].copy()
    hold_dark = hold.loc[~hold["has_book"]].copy()

    g = (
        train.groupby("group_id", as_index=False)
        .agg(n=("company_id", "size"), n_book=("has_book", "sum"))
    )
    g["mix"] = np.select(
        [g["n_book"] == 0, g["n_book"] == g["n"], (g["n_book"] > 0) & (g["n_book"] < g["n"])],
        ["all_dark", "all_invoiced", "mixed"],
        default="?",
    )
    mix_of = g.set_index("group_id")["mix"]
    gsize = (
        train.groupby("group_id", as_index=False)["company_id"]
        .nunique()
        .rename(columns={"company_id": "group_n"})
    )
    train_dark = train_dark.merge(g[["group_id", "mix"]], on="group_id", how="left")
    train_dark = train_dark.merge(gsize, on="group_id", how="left")
    n_360 = int((train_dark["mix"] == "all_dark").sum())
    n_110 = int((train_dark["mix"] == "mixed").sum())
    return {
        "n_train": int(tr.sum()),
        "n_train_dark": int(len(train_dark)),
        "n_hold_dark": int(len(hold_dark)),
        "confirm_470": int(len(train_dark)) == 470,
        "n_360_alldark": n_360,
        "n_110_mixed": n_110,
        "n_groups_all_dark": int((g["mix"] == "all_dark").sum()),
        "n_groups_mixed": int((g["mix"] == "mixed").sum()),
        "n_groups_all_invoiced": int((g["mix"] == "all_invoiced").sum()),
        "train_dark_ids": set(train_dark["company_id"]),
        "hold_dark_ids": set(hold_dark["company_id"]),
        "all_dark_ids": set(train_dark["company_id"]) | set(hold_dark["company_id"]),
        "train_dark_frame": train_dark.reset_index(drop=True),
        "mix_of": mix_of,
        "n_book_train": int(train["has_book"].sum()),
    }


def monthly_op_flows(con) -> pd.DataFrame:
    """Monthly CAT_MAP operational in/out. Outflow stored as a positive magnitude."""
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
    df["company_id"] = df["company_id"].astype(str)
    df["month"] = pd.to_datetime(df["month"])
    return df


def op_month_panel(con) -> pd.DataFrame:
    """Dense company × month CAT_MAP ops. No reconstructed liquidity (not family B)."""
    flows = monthly_op_flows(con)
    first = flows.groupby("company_id")["month"].min().rename("first_m")
    panel = pd.MultiIndex.from_product(
        [first.index, MONTHS], names=["company_id", "month"]
    ).to_frame(index=False)
    panel["company_id"] = panel["company_id"].astype(str)
    panel = panel.merge(first, left_on="company_id", right_index=True)
    panel = panel[panel["month"] >= panel["first_m"]]
    panel = panel.merge(flows, on=["company_id", "month"], how="left")
    panel["op_in"] = panel["op_in"].fillna(0.0)
    panel["op_out"] = panel["op_out"].fillna(0.0)
    panel["net"] = panel["op_in"] - panel["op_out"]
    return panel.drop(columns=["first_m"]).sort_values(["company_id", "month"]).reset_index(drop=True)


def _mask_dark(series: pd.Series, company_id: pd.Series, dark_ids: set[str]) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    return out.where(company_id.astype(str).isin(dark_ids))


def build(con, grid: pd.DataFrame) -> pd.DataFrame:
    """Return company_id, period and the Y11 binaries (0/1/NaN)."""
    need = {"company_id", "period"}
    if not need <= set(grid.columns):
        raise ValueError("grid must have company_id, period")

    pop = dark_population(con)
    dark_ids = pop["all_dark_ids"]

    keys = grid[["company_id", "period"]].copy()
    keys["company_id"] = keys["company_id"].astype(str)
    keys["period"] = pd.to_datetime(keys["period"])

    y2 = build_y2(con, keys)
    y3 = build_y3(con, keys)
    y9 = build_y9(con, keys)
    y2["company_id"] = y2["company_id"].astype(str)
    y3["company_id"] = y3["company_id"].astype(str)
    y9["company_id"] = y9["company_id"].astype(str)
    y2["period"] = pd.to_datetime(y2["period"])
    y3["period"] = pd.to_datetime(y3["period"])
    y9["period"] = pd.to_datetime(y9["period"])

    out = keys.merge(
        y2[["company_id", "period", "y2_neg_2of3"]],
        on=["company_id", "period"],
        how="left",
    )
    out = out.merge(
        y3[["company_id", "period", "y3_recover_cash_6m"]],
        on=["company_id", "period"],
        how="left",
    )
    out = out.merge(
        y9[["company_id", "period", *Y9_COLS]],
        on=["company_id", "period"],
        how="left",
    )
    cid = out["company_id"]
    out["y11_y2_neg_2of3_dark"] = _mask_dark(out["y2_neg_2of3"], cid, dark_ids)
    out["y11_y3_recover_cash_6m_dark"] = _mask_dark(out["y3_recover_cash_6m"], cid, dark_ids)
    out["y11_fee_ownp80_dark"] = _mask_dark(out["y9_fee_r_ownp80"], cid, dark_ids)
    out["y11_fee_spike_dark"] = _mask_dark(out["y9_fee_spike"], cid, dark_ids)

    panel = op_month_panel(con)
    cid_p = panel["company_id"]
    g = panel.groupby(cid_p, sort=False)
    dark_row = cid_p.isin(dark_ids)
    neg = (panel["net"] < 0).astype(float)
    zin = (panel["op_in"] == 0).astype(float)
    n1 = neg.groupby(cid_p).shift(-1)
    n2 = neg.groupby(cid_p).shift(-2)
    n3 = neg.groupby(cid_p).shift(-3)
    z1 = zin.groupby(cid_p).shift(-1)
    z2 = zin.groupby(cid_p).shift(-2)
    z3 = zin.groupby(cid_p).shift(-3)
    has_h3 = _has_horizon(panel["month"], HORIZON)
    has_h6 = _has_horizon(panel["month"], HORIZON_REC)
    neg_count = n1 + n2 + n3
    zin_count = z1 + z2 + z3
    panel["y11_neg_2of3_dark"] = np.where(
        has_h3 & neg_count.notna() & dark_row,
        (neg_count >= 2).astype(float),
        np.nan,
    )
    panel["y11_neg_3of3_dark"] = np.where(
        has_h3 & n1.notna() & n2.notna() & n3.notna() & dark_row,
        ((n1 == 1.0) & (n2 == 1.0) & (n3 == 1.0)).astype(float),
        np.nan,
    )
    # Onset: currently non-negative net, then a full 3-month negative spell.
    panel["y11_neg_onset_3of3_dark"] = np.where(
        has_h3 & n1.notna() & n2.notna() & n3.notna() & dark_row & (neg == 0.0),
        ((n1 == 1.0) & (n2 == 1.0) & (n3 == 1.0)).astype(float),
        np.nan,
    )
    panel["y11_zero_in_2of3_dark"] = np.where(
        has_h3 & zin_count.notna() & dark_row,
        (zin_count >= 2).astype(float),
        np.nan,
    )

    # Company-relative net stress: own expanding p20, months ≤ t.
    panel["own_p20"] = g["net"].transform(
        lambda s: _expanding_quantile_skipna(s, OWN_P20, MIN_OWN_HIST)
    )
    net1 = g["net"].shift(-1)
    net2 = g["net"].shift(-2)
    net3 = g["net"].shift(-3)
    lo = (
        (net1 < panel["own_p20"]).astype(float)
        + (net2 < panel["own_p20"]).astype(float)
        + (net3 < panel["own_p20"]).astype(float)
    )
    panel["y11_neg_ownp20_2of3_dark"] = np.where(
        has_h3 & net1.notna() & net2.notna() & net3.notna() & panel["own_p20"].notna() & dark_row,
        (lo >= 2).astype(float),
        np.nan,
    )

    # Y3 cousin from CAT_MAP net (not family-B runway). Stressed = net < 0 at t.
    panel["ok_net"] = np.where(panel["net"].isna(), np.nan, (panel["net"] >= 0).astype(float))
    known6 = pd.Series(True, index=panel.index)
    for k in range(1, HORIZON_REC + 1):
        known6 = known6 & g["net"].shift(-k).notna()
    stressed = panel["net"] < 0
    labeled = has_h6 & stressed & known6 & dark_row
    net_rec = _any_recover_window(g, "ok_net")
    panel["y11_net_recover_6m_dark"] = np.where(labeled, net_rec.astype(float), np.nan)

    lab_cols = [
        "y11_neg_2of3_dark",
        "y11_neg_3of3_dark",
        "y11_neg_ownp20_2of3_dark",
        "y11_net_recover_6m_dark",
        "y11_neg_onset_3of3_dark",
        "y11_zero_in_2of3_dark",
    ]
    lab = panel[["company_id", "month", *lab_cols]].rename(columns={"month": "period"})
    out = out.merge(lab, on=["company_id", "period"], how="left")
    keep = ["company_id", "period", *Y11_COLS]
    return out[keep].reset_index(drop=True)


def column_verdict(
    col: str,
    numeric: bool,
    in_rate: bool,
    size_ok: bool,
    rho_y2: float,
    rho_y3: float,
    rho_y9: float,
    y2_defined: bool,
) -> str:
    """CLOSE / PARK / ACCEPTED. A numeric pass is not enough for a subset."""
    if col == "y11_y2_neg_2of3_dark":
        if y2_defined and numeric:
            return "CLOSE"
        if y2_defined:
            return "CLOSE"
        return "PARK"
    if col == "y11_y3_recover_cash_6m_dark":
        return "CLOSE"
    if col == "y11_neg_2of3_dark":
        if np.isfinite(rho_y2) and abs(rho_y2) >= REWRITE_Y2:
            return "PARK"
        return "PARK"
    if col == "y11_neg_3of3_dark":
        # Numeric pass, but 79% of positives already have net<0 at t —
        # a state, not a Q3 turn. Prefer the onset slice.
        return "PARK"
    if col == "y11_neg_onset_3of3_dark":
        if np.isfinite(rho_y2) and abs(rho_y2) >= REWRITE_Y2:
            return "PARK"
        if numeric:
            return "ACCEPTED"
        return "PARK"
    if col == "y11_neg_ownp20_2of3_dark":
        if np.isfinite(rho_y2) and abs(rho_y2) >= REWRITE_Y2:
            return "PARK"
        if numeric:
            return "ACCEPTED"
        return "PARK"
    if col == "y11_net_recover_6m_dark":
        if np.isfinite(rho_y3) and abs(rho_y3) >= REWRITE_Y2:
            return "PARK"
        if numeric:
            return "ACCEPTED"
        return "PARK"
    if col in {"y11_fee_ownp80_dark", "y11_fee_spike_dark"}:
        if np.isfinite(rho_y9) and abs(rho_y9) >= REWRITE_Y9:
            return "CLOSE"
        if numeric:
            return "ACCEPTED"
        return "PARK"
    if col == "y11_zero_in_2of3_dark":
        if numeric:
            return "ACCEPTED"
        return "PARK"
    if numeric:
        return "ACCEPTED"
    return "PARK"


def train_acceptance(
    y11: pd.DataFrame,
    op_in: pd.Series,
    is_train: pd.Series,
    rho: pd.DataFrame,
    y2_n_dark: int,
) -> pd.DataFrame:
    """Base rates and two-sided size AUROC on train company-months only."""
    assert_no_holdout(y11.loc[is_train, "company_id"])
    rho_map: dict[tuple[str, str], float] = {}
    if rho is not None and not rho.empty:
        for _, r in rho.iterrows():
            rho_map[(r.column, r.vs)] = float(r.spearman) if np.isfinite(r.spearman) else float("nan")
    opin_s = np.log1p(pd.to_numeric(op_in.loc[is_train], errors="coerce").abs())
    n_train = int(is_train.sum())
    y2_defined = y2_n_dark > 0
    rows = []
    for col in Y11_COLS:
        y = y11.loc[is_train, col]
        ok = y.notna()
        n = int(ok.sum())
        rate = float(y[ok].mean()) if n else float("nan")
        auc = auroc(y, opin_s)
        auc_abs = _two_sided(auc)
        in_rate = bool(n and 0.05 <= rate <= 0.30)
        size_ok = bool(np.isfinite(auc_abs) and auc_abs < 0.60)
        numeric = bool(in_rate and size_ok)
        rho_y2 = rho_map.get((col, "y2_neg_2of3"), float("nan"))
        rho_y3 = rho_map.get((col, "y3_recover_cash_6m"), float("nan"))
        rho_y9 = rho_map.get((col, "y9_fee_r_ownp80"), float("nan"))
        if col == "y11_fee_spike_dark":
            rho_y9 = rho_map.get((col, "y9_fee_spike"), rho_y9)
        verdict = column_verdict(col, numeric, in_rate, size_ok, rho_y2, rho_y3, rho_y9, y2_defined)
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
                "numeric_ok": numeric,
                "in_rate": in_rate,
                "size_ok": size_ok,
                "rho_y2": rho_y2,
                "rho_y3": rho_y3,
                "rho_y9": rho_y9,
                "accepted": verdict == "ACCEPTED",
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def spearman_vs_frozen(y11: pd.DataFrame, is_train: pd.Series, frozen: pd.DataFrame) -> pd.DataFrame:
    """Spearman vs accepted neighbours on train overlap."""
    neighbours = [
        c
        for c in (
            "y2_neg_2of3",
            "y2_onset_neg",
            "y3_recover_cash_6m",
            "y9_fee_r_ownp80",
            "y9_fee_spike",
        )
        if c in frozen.columns
    ]
    keys = y11[["company_id", "period"]].copy()
    keys["company_id"] = keys["company_id"].astype(str)
    keys["period"] = pd.to_datetime(keys["period"])
    fr = frozen[["company_id", "period", *neighbours]].copy()
    fr["company_id"] = fr["company_id"].astype(str)
    fr["period"] = pd.to_datetime(fr["period"])
    m = keys.merge(fr, on=["company_id", "period"], how="left")
    rows = []
    for col in Y11_COLS:
        for nb in neighbours:
            ok = is_train & y11[col].notna() & m[nb].notna()
            n = int(ok.sum())
            if n < 20:
                rho = float("nan")
            else:
                rho = float(y11.loc[ok, col].corr(m.loc[ok, nb], method="spearman"))
            rewrite = False
            if nb.startswith("y2_") and np.isfinite(rho) and abs(rho) >= REWRITE_Y2:
                rewrite = True
            if nb.startswith("y9_") and np.isfinite(rho) and abs(rho) >= REWRITE_Y9:
                rewrite = True
            if nb.startswith("y3_") and np.isfinite(rho) and abs(rho) >= REWRITE_Y2:
                rewrite = True
            rows.append(
                {
                    "column": col,
                    "vs": nb,
                    "n_overlap": n,
                    "spearman": rho,
                    "rewrite": rewrite,
                }
            )
    return pd.DataFrame(rows)


def full_vs_dark_y2y3(
    frozen: pd.DataFrame,
    op_in: pd.Series,
    is_train: pd.Series,
    train_dark_ids: set[str],
) -> pd.DataFrame:
    """Accepted Y2/Y3 on full train vs the 470. Not a new label."""
    opin_s = np.log1p(pd.to_numeric(op_in, errors="coerce").abs())
    cid = frozen["company_id"].astype(str)
    rows = []
    for col, kind in (("y2_neg_2of3", "unconditional"), ("y3_recover_cash_6m", "stressed")):
        if col not in frozen.columns:
            continue
        y = pd.to_numeric(frozen[col], errors="coerce")
        for slice_name, mask in (
            ("full_train", is_train),
            ("dark_470", is_train & cid.isin(train_dark_ids)),
            ("invoiced_744", is_train & ~cid.isin(train_dark_ids)),
        ):
            ok = mask & y.notna()
            n = int(ok.sum())
            rate = float(y[ok].mean()) if n else float("nan")
            auc = auroc(y[ok], opin_s[ok]) if n else float("nan")
            rows.append(
                {
                    "column": col,
                    "slice": slice_name,
                    "kind": kind,
                    "n_labeled": n,
                    "n_pos": int((y[ok] == 1).sum()) if n else 0,
                    "n_cos": int(cid[ok].nunique()) if n else 0,
                    "base_rate": rate,
                    "size_auroc": auc,
                    "size_auroc_two_sided": _two_sided(auc),
                    "usable": bool(
                        n
                        and 0.05 <= rate <= 0.30
                        and np.isfinite(auc)
                        and auc < 0.60
                    ),
                    "usable_two_sided": bool(
                        n
                        and 0.05 <= rate <= 0.30
                        and np.isfinite(auc)
                        and _two_sided(auc) < 0.60
                    ),
                }
            )
    return pd.DataFrame(rows)


def mix_slice_rates(
    y11: pd.DataFrame,
    op_in: pd.Series,
    is_train: pd.Series,
    train_dark: pd.DataFrame,
) -> pd.DataFrame:
    """360 all-dark vs 110 mixed-group dark siblings."""
    mix = train_dark.set_index("company_id")["mix"]
    cid = y11["company_id"].astype(str)
    mapped = cid.map(mix)
    opin_s = np.log1p(pd.to_numeric(op_in, errors="coerce").abs())
    rows = []
    for col in Y11_COLS:
        y = pd.to_numeric(y11[col], errors="coerce")
        for name, sl in (
            ("all_dark_360", is_train & mapped.eq("all_dark")),
            ("mixed_110", is_train & mapped.eq("mixed")),
        ):
            ok = sl & y.notna()
            n = int(ok.sum())
            auc = auroc(y[ok], opin_s[ok]) if n else float("nan")
            rows.append(
                {
                    "column": col,
                    "slice": name,
                    "n_labeled": n,
                    "n_pos": int((y[ok] == 1).sum()) if n else 0,
                    "n_cos": int(cid[ok].nunique()) if n else 0,
                    "base_rate": float(y[ok].mean()) if n else float("nan"),
                    "size_auroc": auc,
                    "size_auroc_two_sided": _two_sided(auc),
                }
            )
    return pd.DataFrame(rows)


def mix_context(
    keys: pd.DataFrame,
    op_in: pd.Series,
    is_train: pd.Series,
    train_dark: pd.DataFrame,
) -> dict:
    """Is the 110 different because they are larger, or because H sees a sister?"""
    mix = train_dark.set_index("company_id")["mix"]
    cid = keys["company_id"].astype(str)
    mapped = cid.map(mix)
    opin = pd.to_numeric(op_in, errors="coerce")
    gsize = train_dark.set_index("company_id")["group_n"]
    out = {}
    for name, sl in (
        ("all_dark_360", is_train & mapped.eq("all_dark")),
        ("mixed_110", is_train & mapped.eq("mixed")),
    ):
        cos = cid[sl].drop_duplicates()
        g_med = (
            train_dark.loc[train_dark["company_id"].isin(set(cos)), ["group_id", "group_n"]]
            .drop_duplicates("group_id")["group_n"]
            .median()
        )
        out[name] = {
            "n_cm": int(sl.sum()),
            "n_cos": int(cos.nunique()),
            "median_op_in": float(opin[sl].median()) if sl.any() else float("nan"),
            "p90_op_in": float(opin[sl].quantile(0.90)) if sl.any() else float("nan"),
            "mean_log1p_abs_in": float(np.log1p(opin[sl].abs()).mean()) if sl.any() else float("nan"),
            "median_group_n": float(g_med) if pd.notna(g_med) else float("nan"),
            "median_group_n_cos": float(gsize.reindex(cos).median()) if len(cos) else float("nan"),
        }
    return out


def train_pos_companies(y11: pd.DataFrame, is_train: pd.Series) -> pd.DataFrame:
    rows = []
    for col in Y11_COLS:
        y = y11.loc[is_train, col]
        cid = y11.loc[is_train, "company_id"]
        rows.append(
            {
                "column": col,
                "n_cos_labeled": int(cid[y.notna()].nunique()),
                "n_cos_pos": int(cid[y == 1].nunique()),
            }
        )
    return pd.DataFrame(rows)


def y3_size_terciles(
    y11: pd.DataFrame,
    op_in: pd.Series,
    is_train: pd.Series,
    train_dark: pd.DataFrame,
) -> pd.DataFrame:
    """Y3 recovery 360 vs 110 inside train-dark size terciles (diagnostic, not a cut)."""
    mix = train_dark.set_index("company_id")["mix"]
    cid = y11["company_id"].astype(str)
    y = pd.to_numeric(y11["y11_y3_recover_cash_6m_dark"], errors="coerce")
    dark = is_train & y.notna()
    size = np.log1p(pd.to_numeric(op_in, errors="coerce").abs())
    try:
        terc = pd.qcut(size[dark], 3, labels=["T1_small", "T2_mid", "T3_large"], duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    bins = pd.Series(np.nan, index=y11.index, dtype=object)
    bins.loc[dark] = terc.astype(str)
    mapped = cid.map(mix)
    rows = []
    for tname in ("T1_small", "T2_mid", "T3_large"):
        for sl_name, sl in (
            ("all_dark_360", mapped.eq("all_dark")),
            ("mixed_110", mapped.eq("mixed")),
        ):
            ok = dark & sl & bins.eq(tname)
            n = int(ok.sum())
            rows.append(
                {
                    "tercile": tname,
                    "slice": sl_name,
                    "n": n,
                    "n_pos": int((y[ok] == 1).sum()) if n else 0,
                    "n_cos": int(cid[ok].nunique()) if n else 0,
                    "base_rate": float(y[ok].mean()) if n else float("nan"),
                    "median_log1p_in": float(size[ok].median()) if n else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def now_leak(
    y11: pd.DataFrame,
    ops: pd.DataFrame,
    is_train: pd.Series,
    cols: list[str],
) -> pd.DataFrame:
    """AUROC of contemporaneous net<0 and log1p(|net|) vs each Y (persistence screen)."""
    keys = y11[["company_id", "period"]].copy()
    keys["company_id"] = keys["company_id"].astype(str)
    keys["period"] = pd.to_datetime(keys["period"])
    m = keys.merge(
        ops[["company_id", "period", "net"]],
        on=["company_id", "period"],
        how="left",
    )
    neg_now = (pd.to_numeric(m["net"], errors="coerce") < 0).astype(float)
    abs_net = np.log1p(pd.to_numeric(m["net"], errors="coerce").abs())
    rows = []
    for col in cols:
        ok = is_train & y11[col].notna() & m["net"].notna()
        y = y11.loc[ok, col]
        rows.append(
            {
                "column": col,
                "n": int(ok.sum()),
                "auc_neg_now": auroc(y, neg_now[ok]),
                "auc_abs_net": auroc(y, abs_net[ok]),
                "share_neg_now_on_pos": float(neg_now[ok & (y11[col] == 1)].mean())
                if (ok & (y11[col] == 1)).any()
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def sibling_rho(y11: pd.DataFrame, is_train: pd.Series, cols: list[str]) -> pd.DataFrame:
    """Spearman among in-module cousins on train overlap."""
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            ok = is_train & y11[a].notna() & y11[b].notna()
            n = int(ok.sum())
            rho = float(y11.loc[ok, a].corr(y11.loc[ok, b], method="spearman")) if n >= 20 else float("nan")
            rows.append({"a": a, "b": b, "n": n, "spearman": rho, "near_copy": bool(np.isfinite(rho) and abs(rho) >= 0.80)})
    return pd.DataFrame(rows)


def acf_stats(y11: pd.DataFrame, is_train: pd.Series, col: str) -> dict:
    d = y11.loc[is_train, ["company_id", "period", col]].dropna(subset=[col])
    if d.empty:
        return {"acf1": float("nan"), "acf3": float("nan"), "n1": 0, "n3": 0}
    d = d.sort_values(["company_id", "period"])
    g = d.groupby("company_id")[col]
    lag1 = g.shift(1)
    lag3 = g.shift(3)
    ok1 = d[col].notna() & lag1.notna()
    ok3 = d[col].notna() & lag3.notna()
    acf1 = float(d.loc[ok1, col].corr(lag1[ok1])) if ok1.sum() else float("nan")
    acf3 = float(d.loc[ok3, col].corr(lag3[ok3])) if ok3.sum() else float("nan")
    return {"acf1": acf1, "acf3": acf3, "n1": int(ok1.sum()), "n3": int(ok3.sum())}


def _fmt(x, nd=3, pct=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if pct:
        return f"{float(x):.2%}"
    return f"{float(x):.{nd}f}"


def write_acceptance_md(
    pop: dict,
    acc: pd.DataFrame,
    pos_cos: pd.DataFrame,
    rho: pd.DataFrame,
    cmp: pd.DataFrame,
    mix: pd.DataFrame,
    extra: dict,
    n_dark_cm: dict,
    ctx: dict | None = None,
) -> None:
    ACCEPT_MD.parent.mkdir(parents=True, exist_ok=True)
    acc_i = acc.set_index("column")
    pos_i = pos_cos.set_index("column")

    def row(col: str) -> str:
        r = acc_i.loc[col]
        p = pos_i.loc[col]
        return (
            f"| `{col}` | {int(r.n_labeled):,} | {r.coverage:.1%} | {int(r.n_pos):,} | "
            f"{_fmt(r.base_rate, pct=True)} | {_fmt(r.size_auroc)} | "
            f"{_fmt(r.size_auroc_two_sided)} | "
            f"{int(p.n_cos_labeled)} / {int(p.n_cos_pos)} | **{r.verdict}** |"
        )

    cmp_lines = []
    for _, r in cmp.iterrows():
        cmp_lines.append(
            f"| `{r.column}` | {r.slice} | {int(r.n_labeled):,} | {int(r.n_pos):,} | "
            f"{int(r.n_cos)} | {_fmt(r.base_rate, pct=True)} | {_fmt(r.size_auroc)} | "
            f"{_fmt(r.size_auroc_two_sided)} | "
            f"{'yes' if r.usable else 'no'} / {'yes' if r.usable_two_sided else 'no'} |"
        )

    mix_lines = []
    for _, r in mix.iterrows():
        mix_lines.append(
            f"| `{r.column}` | {r.slice} | {int(r.n_labeled):,} | {int(r.n_pos):,} | "
            f"{int(r.n_cos)} | {_fmt(r.base_rate, pct=True)} | {_fmt(r.size_auroc)} |"
        )

    rho_lines = []
    for _, r in rho.iterrows():
        spr = "—" if not np.isfinite(r.spearman) else f"{r.spearman:.3f}"
        flag = " rewrite" if r.rewrite else ""
        rho_lines.append(
            f"| `{r.column}` | `{r.vs}` | {int(r.n_overlap):,} | {spr} |{flag} |"
        )

    y2_dark = cmp[(cmp["column"] == "y2_neg_2of3") & (cmp["slice"] == "dark_470")]
    y3_dark = cmp[(cmp["column"] == "y3_recover_cash_6m") & (cmp["slice"] == "dark_470")]
    y2_full = cmp[(cmp["column"] == "y2_neg_2of3") & (cmp["slice"] == "full_train")]
    y3_full = cmp[(cmp["column"] == "y3_recover_cash_6m") & (cmp["slice"] == "full_train")]

    def _one(df: pd.DataFrame) -> pd.Series:
        return df.iloc[0] if len(df) else pd.Series(dtype=object)

    y2d, y3d, y2f, y3f = _one(y2_dark), _one(y3_dark), _one(y2_full), _one(y3_full)

    lines = [
        "# Y11 acceptance — cash Y among companies with no invoice book",
        "",
        "Generated by `python -m analysis.targets.y11_dark`. Train company-months",
        "only (holdout 72 out of every rate and size AUROC). No 0–100. No `product/`.",
        "Does **not** write `targets.parquet`. Assembler skips `y11_dark`.",
        "Parent decides merge — this child does not.",
        "",
        "## Brief questions (NORTH_STAR)",
        "",
        "Q3 / Q4 for companies **without an invoice book**. Invoice-side turning",
        "and dip-vs-fall are undefined on this set. Cash-side still is.",
        "Do not invent cobros. Families **D and E** are forbidden later X.",
        "",
        "## Population (DuckDB, not hardcoded IDs)",
        "",
        f"- Book filter: `{BOOK}`.",
        f"- Train companies: **{pop['n_train']}**. Book invoice: {pop['n_book_train']}. "
        f"Dark: **{pop['n_train_dark']}**. confirm_470={pop['confirm_470']}.",
        f"- Holdout dark (descriptive): {pop['n_hold_dark']} of 72.",
        f"- Group mix among the {pop['n_train_dark']}: **{pop['n_360_alldark']}** in "
        f"{pop['n_groups_all_dark']} all-dark groups; **{pop['n_110_mixed']}** in "
        f"{pop['n_groups_mixed']} mixed groups (invoiced sibling exists).",
        f"- Dark train company-months on the official grid: "
        f"**{n_dark_cm.get('n_train_dark_cm', '?')}** "
        f"({n_dark_cm.get('n_train_dark_cos', '?')} companies).",
        "- The 470 have bank txs (join QA). They lack an ERP book, not a bank book.",
        "",
        "## Try 1 — restrict accepted Y2 / Y3 (same definition)",
        "",
        "Not a new label. CLOSE as *Y2/Y3 on dark companies* if the subset is",
        "just the accepted column masked to the 470. Usable as a Y only when",
        "train base ∈ [5%, 30%] and size AUROC < 0.60 (contract one-sided; "
        "two-sided also reported). Y3 parent used one-sided + stressed 5–50%.",
        "",
        "| column | slice | n labeled | pos | cos | base | size AUROC | two-sided | usable 1s / 2s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
        *cmp_lines,
        "",
        (
            f"Y2 `y2_neg_2of3` full-train n={int(y2f.get('n_labeled', 0)):,} "
            f"base={_fmt(y2f.get('base_rate'), pct=True)} size={_fmt(y2f.get('size_auroc'))}; "
            f"dark-470 n={int(y2d.get('n_labeled', 0)):,} "
            f"base={_fmt(y2d.get('base_rate'), pct=True)} size={_fmt(y2d.get('size_auroc'))}."
        ),
        (
            f"Y3 `y3_recover_cash_6m` full-train n={int(y3f.get('n_labeled', 0)):,} "
            f"base={_fmt(y3f.get('base_rate'), pct=True)} size={_fmt(y3f.get('size_auroc'))}; "
            f"dark-470 n={int(y3d.get('n_labeled', 0)):,} "
            f"base={_fmt(y3d.get('base_rate'), pct=True)} size={_fmt(y3d.get('size_auroc'))}."
        ),
        "",
        "## Train acceptance (Y11 columns)",
        "",
        "ACCEPTED only if train base ∈ [5%, 30%] **and** two-sided size AUROC",
        "`max(auc, 1-auc) < 0.60` **and** not a documented subset/rewrite.",
        "Numeric pass is not enough for a Y2/Y3/Y9 mask.",
        "",
        f"Train panel: **{int(acc.n_train.iloc[0]):,}** company-months, 1,214 companies.",
        "",
        "| column | n labeled | cov | pos | base | size AUROC | two-sided | cos lab/pos | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        *(row(c) for c in Y11_COLS),
        "",
        "## Verdicts",
        "",
        f"### `y11_y2_neg_2of3_dark` — {acc_i.loc['y11_y2_neg_2of3_dark'].verdict}",
        "",
        "Same definition as accepted `y2_neg_2of3`, masked to zero-book companies.",
        "Do not rename this into a new win. Forbidden later X: B + D + E.",
        f"ρ vs Y2 on overlap = {_fmt(acc_i.loc['y11_y2_neg_2of3_dark'].rho_y2)} "
        "(1.0 by construction if Y2 is defined on the set).",
        "",
        f"### `y11_y3_recover_cash_6m_dark` — {acc_i.loc['y11_y3_recover_cash_6m_dark'].verdict}",
        "",
        "Same definition as accepted `y3_recover_cash_6m`, masked to the 470.",
        "Do not rename. Forbidden later X: B + D + E.",
        "",
        f"### `y11_neg_2of3_dark` — {acc_i.loc['y11_neg_2of3_dark'].verdict}",
        "",
        "CAT_MAP operational net < 0 in ≥ 2 of the next 3 months (not family B).",
        f"ρ vs `y2_neg_2of3` = {_fmt(acc_i.loc['y11_neg_2of3_dark'].rho_y2)}. "
        f"Rewrite threshold {REWRITE_Y2:g}. First-run base was ~48% — too common.",
        "",
        f"### `y11_neg_3of3_dark` — {acc_i.loc['y11_neg_3of3_dark'].verdict}",
        "",
        "Numeric gates pass (base ~23%, two-sided size ~0.51) and ρ vs Y2 is ~0.04 "
        "(not a rewrite). **PARK as a Q3 turn anyway:** 79% of positives already "
        "have CAT_MAP net < 0 at t (contemporaneous AUROC 0.70). That is a "
        "*state*, not an onset. Prefer `y11_neg_onset_3of3_dark`.",
        "",
        f"### `y11_neg_ownp20_2of3_dark` — {acc_i.loc['y11_neg_ownp20_2of3_dark'].verdict}",
        "",
        "Company-relative: net below own expanding p20 in ≥ 2 of next 3. "
        "Not a pooled percentile (holdout never enters a cut).",
        f"ρ vs Y2 = {_fmt(acc_i.loc['y11_neg_ownp20_2of3_dark'].rho_y2)}.",
        "",
        f"### `y11_net_recover_6m_dark` — {acc_i.loc['y11_net_recover_6m_dark'].verdict}",
        "",
        "Y3 cousin from CAT_MAP net (stressed net<0; 3-in-6 months net≥0). "
        f"ρ vs `y3_recover_cash_6m` = {_fmt(acc_i.loc['y11_net_recover_6m_dark'].rho_y3)}.",
        "",
        f"### `y11_neg_onset_3of3_dark` — {acc_i.loc['y11_neg_onset_3of3_dark'].verdict}",
        "",
        "Q3 onset: net ≥ 0 at t, then 3 consecutive negative-net months. "
        "This *is* `y11_neg_3of3_dark` restricted to currently non-negative months "
        "(ρ = 1.0 on that overlap) — keep it as the turning slice, not a second Y. "
        f"Train n={int(acc_i.loc['y11_neg_onset_3of3_dark'].n_labeled):,}, "
        f"base={_fmt(acc_i.loc['y11_neg_onset_3of3_dark'].base_rate, pct=True)}, "
        f"size={_fmt(acc_i.loc['y11_neg_onset_3of3_dark'].size_auroc)}. "
        f"ρ vs Y2 = {_fmt(acc_i.loc['y11_neg_onset_3of3_dark'].rho_y2)}. "
        "acf1 is low (0.14) because an onset does not persist. "
        "Forbidden later X: A + D + E.",
        "",
        extra.get("now_leak_md", ""),
        "",
        f"### `y11_fee_ownp80_dark` — {acc_i.loc['y11_fee_ownp80_dark'].verdict}",
        "",
        "Y9-style own-p80 fee pressure, dark only.",
        f"ρ vs `y9_fee_r_ownp80` = {_fmt(acc_i.loc['y11_fee_ownp80_dark'].rho_y9)}. "
        f"CLOSE as subset if |ρ| ≥ {REWRITE_Y9:g}.",
        "",
        f"### `y11_fee_spike_dark` — {acc_i.loc['y11_fee_spike_dark'].verdict}",
        "",
        "Y9 spike method, dark only. Extra probe.",
        "",
        f"### `y11_zero_in_2of3_dark` — {acc_i.loc['y11_zero_in_2of3_dark'].verdict}",
        "",
        "Zero operational inflow in ≥ 2 of next 3. Extra probe; Y6-adjacent.",
        f"Inverse-size risk: two-sided size AUROC "
        f"{_fmt(acc_i.loc['y11_zero_in_2of3_dark'].size_auroc_two_sided)}.",
        "",
        "## 360 all-dark groups vs 110 mixed-group dark siblings",
        "",
        "Family H can see a sister's cash on the 110; it cannot invent an",
        "invoice book. All-dark holdings have no cobros anywhere in the group.",
        "A large base-rate gap would mean the 110 are a different world.",
        "",
        "| column | slice | n labeled | pos | cos | base | size AUROC |",
        "|---|---|---:|---:|---:|---:|---:|",
        *mix_lines,
        "",
        extra.get("mix_read", ""),
        "",
        extra.get("mix_ctx_md", ""),
        "",
        extra.get("tercile_md", ""),
        "",
        extra.get("sib_md", ""),
        "",
        "## Spearman vs frozen Ys (train overlap)",
        "",
        "| column | vs | n | ρ | |",
        "|---|---|---:|---:|---|",
        *rho_lines,
        "",
        "## Persistence / holdout (descriptive)",
        "",
        extra.get("acf_all", ""),
        "",
        "Holdout dark (LOW_POWER, descriptive):",
        extra.get("ho_all", ""),
        "",
        "## Do the 470 need their own Y?",
        "",
        extra.get("need_own_y", "see run"),
        "",
        extra.get("union_md", ""),
        "",
        "## Forbidden later X",
        "",
        "Every column: families **D and E**. Restricted Y2/Y3: also **B**.",
        "Net / zero-in cousins: also **A** (same CAT_MAP flows).",
        "Fee cousins: also **F** and prefixes `a_fin_cost` / `a_fc`.",
        "Do not invent cobros. Do not use HHI / top-1.",
        "",
        "## What was not done",
        "",
        "- Did not write `targets.parquet` or edit `build_targets.py` / FROZEN lists.",
        "- Did not edit y2 / y3 / y8 / y9 / y10 modules (imported `build` only).",
        "- Did not rebuild Y8. Did not touch `product/` or commit.",
        "- `SKIP_ASSEMBLE_MODULES` already includes `y11_dark`.",
        "",
    ]
    ACCEPT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", ACCEPT_MD)


def append_registry(acc: pd.DataFrame) -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    existing = REGISTRY.read_text(encoding="utf-8")
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    rows = []
    for _, r in acc.iterrows():
        marker = f"{AGENT},A/B/C/F/G/H,{r.column},verify,train,size_auroc"
        if marker in existing:
            continue
        notes = (
            f"{r.verdict}; base={r.base_rate:.4f}; two_sided={r.size_auroc_two_sided:.3f}; "
            f"n_pos={int(r.n_pos)}; numeric_ok={bool(r.numeric_ok)}; "
            f"rho_y2={r.rho_y2}; rho_y9={r.rho_y9}; "
            "holdout excluded; no targets.parquet write; never D/E"
        )
        rows.append(
            (
                ts,
                ROUND,
                WAVE,
                AGENT,
                "A/B/C/F/G/H",
                r.column,
                "verify",
                "train",
                "base_rate",
                f"{r.base_rate:.6f}" if np.isfinite(r.base_rate) else "",
                f"{r.coverage:.4f}",
                notes,
            )
        )
        rows.append(
            (
                ts,
                ROUND,
                WAVE,
                AGENT,
                "A/B/C/F/G/H",
                r.column,
                "verify",
                "train",
                "size_auroc",
                f"{r.size_auroc:.6f}" if np.isfinite(r.size_auroc) else "",
                f"{r.coverage:.4f}",
                notes,
            )
        )
    if not rows:
        print("registry skip (all Y11 rows already present for this agent)")
        return
    with REGISTRY.open("a", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)
    print("registry appended", len(rows), "rows")


if __name__ == "__main__":
    from analysis.features.common import connect
    from analysis.features.grid import monthly_grid

    t0 = datetime.now()
    con = connect()
    pop = dark_population(con)
    print("Y11 dark population")
    print(
        f"  train={pop['n_train']} book={pop['n_book_train']} "
        f"dark={pop['n_train_dark']} confirm_470={pop['confirm_470']}"
    )
    print(
        f"  360 all-dark={pop['n_360_alldark']} (groups {pop['n_groups_all_dark']}) "
        f"110 mixed={pop['n_110_mixed']} (groups {pop['n_groups_mixed']})"
    )
    print(f"  holdout dark={pop['n_hold_dark']} (descriptive)")

    grid = monthly_grid(con)[["company_id", "period"]]
    y11 = build(con, grid)
    keys = grid[["company_id", "period"]].copy()
    keys["company_id"] = keys["company_id"].astype(str)
    keys["period"] = pd.to_datetime(keys["period"])
    tr = train_mask(keys["company_id"])
    assert_no_holdout(keys.loc[tr, "company_id"])

    dark_cm = tr & keys["company_id"].isin(pop["train_dark_ids"])
    n_dark_cm = {
        "n_train_dark_cm": int(dark_cm.sum()),
        "n_train_dark_cos": int(keys.loc[dark_cm, "company_id"].nunique()),
    }
    print(
        "dark train company-months",
        n_dark_cm["n_train_dark_cm"],
        "companies",
        n_dark_cm["n_train_dark_cos"],
    )
    print("rows", len(y11), "dups", int(y11.duplicated(["company_id", "period"]).sum()))

    ops = op_month_panel(con).rename(columns={"month": "period"})
    ops["company_id"] = ops["company_id"].astype(str)
    flows = keys.merge(
        ops[["company_id", "period", "op_in"]],
        on=["company_id", "period"],
        how="left",
    )
    op_in = flows["op_in"]

    # Frozen neighbours: rebuild via owned imports (do not rewrite parquet).
    y2 = build_y2(con, keys)
    y3 = build_y3(con, keys)
    y9 = build_y9(con, keys)
    frozen = keys.copy()
    frozen = frozen.merge(
        y2[["company_id", "period", "y2_neg_2of3", "y2_onset_neg"]],
        on=["company_id", "period"],
        how="left",
    )
    frozen = frozen.merge(
        y3[["company_id", "period", "y3_recover_cash_6m"]], on=["company_id", "period"], how="left"
    )
    frozen = frozen.merge(y9[["company_id", "period", *Y9_COLS]], on=["company_id", "period"], how="left")

    cmp = full_vs_dark_y2y3(frozen, op_in, tr, pop["train_dark_ids"])
    print("Y2/Y3 full-train vs dark-470 (restriction, not a new Y)")
    print(cmp.to_string(index=False))

    y2_n_dark = int(
        cmp.loc[(cmp["column"] == "y2_neg_2of3") & (cmp["slice"] == "dark_470"), "n_labeled"].sum()
    )
    print("restricted Y2 labeled on dark", y2_n_dark, "empty" if y2_n_dark == 0 else "DEFINED")

    rho = spearman_vs_frozen(y11, tr, frozen)
    acc = train_acceptance(y11, op_in, tr, rho, y2_n_dark)
    print("Y11 train acceptance")
    print(acc.to_string(index=False))
    print("train companies labeled / positive")
    pos_cos = train_pos_companies(y11, tr)
    print(pos_cos.to_string(index=False))
    if not rho.empty:
        print("Spearman vs frozen (train overlap)")
        print(rho.to_string(index=False))

    mix = mix_slice_rates(y11, op_in, tr, pop["train_dark_frame"])
    print("360 vs 110 base rates")
    print(mix.to_string(index=False))
    ctx = mix_context(keys, op_in, tr, pop["train_dark_frame"])
    print("mix context", ctx)

    def _mix_pair(col: str) -> tuple[float, float, float]:
        a = mix[(mix["column"] == col) & (mix["slice"] == "all_dark_360")]
        b = mix[(mix["column"] == col) & (mix["slice"] == "mixed_110")]
        ba = float(a["base_rate"].iloc[0]) if len(a) else float("nan")
        bb = float(b["base_rate"].iloc[0]) if len(b) else float("nan")
        gap = abs(ba - bb) if np.isfinite(ba) and np.isfinite(bb) else float("nan")
        return ba, bb, gap

    b360, b110, gap = _mix_pair("y11_y2_neg_2of3_dark")
    y3a, y3b, y3g = _mix_pair("y11_y3_recover_cash_6m_dark")
    mix_read = (
        f"Y2-on-dark base all-dark={_fmt(b360, pct=True)} vs mixed-sibling="
        f"{_fmt(b110, pct=True)} (abs gap {_fmt(gap, pct=True)}). "
        f"Y3-on-dark recover all-dark={_fmt(y3a, pct=True)} vs mixed="
        f"{_fmt(y3b, pct=True)} (abs gap {_fmt(y3g, pct=True)}). "
    )
    if np.isfinite(y3g) and y3g >= 0.05:
        mix_read += (
            "Y3 recovery gap ≥ 5pp — the 110 mixed-group dark siblings recover "
            "more often. Family H can see a sister's cash; that is a real "
            "context split for Q3/Q4, not a reason to invent cobros. "
        )
    elif np.isfinite(gap) and gap < 0.03:
        mix_read += (
            "Y2 gap < 3pp — the 110 are not a different *stress* world. "
        )
    else:
        mix_read += "See slice table. "
    a360, a110 = ctx.get("all_dark_360", {}), ctx.get("mixed_110", {})
    mix_ctx_md = (
        f"Size / group context (train dark company-months): "
        f"all-dark n_cm={a360.get('n_cm')} median op_in={_fmt(a360.get('median_op_in'), 0)} "
        f"mean log1p(|in|)={_fmt(a360.get('mean_log1p_abs_in'))} "
        f"median group_n={_fmt(a360.get('median_group_n'), 1)}; "
        f"mixed n_cm={a110.get('n_cm')} median op_in={_fmt(a110.get('median_op_in'), 0)} "
        f"mean log1p(|in|)={_fmt(a110.get('mean_log1p_abs_in'))} "
        f"median group_n={_fmt(a110.get('median_group_n'), 1)}. "
        "Mixed dark companies are *smaller* (median op_in 31k vs 89k), so size "
        "would push their Y3 recovery *up* (parent Y3 is inverse-size). "
        "T1 residual +12pp after terciles is the H/sister gap, not size."
    )
    print(mix_read)
    print(mix_ctx_md)

    terc = y3_size_terciles(y11, op_in, tr, pop["train_dark_frame"])
    if not terc.empty:
        print("Y3 recovery by size tercile × 360/110")
        print(terc.to_string(index=False))
        terc_lines = [
            "| tercile | slice | n | pos | cos | base | median log1p(|in|) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for _, r in terc.iterrows():
            terc_lines.append(
                f"| {r.tercile} | {r.slice} | {int(r.n):,} | {int(r.n_pos):,} | "
                f"{int(r.n_cos)} | {_fmt(r.base_rate, pct=True)} | {_fmt(r.median_log1p_in)} |"
            )
        # Same-tercile gaps
        gap_bits = []
        for tname in ("T1_small", "T2_mid", "T3_large"):
            aa = terc[(terc["tercile"] == tname) & (terc["slice"] == "all_dark_360")]
            bb = terc[(terc["tercile"] == tname) & (terc["slice"] == "mixed_110")]
            if len(aa) and len(bb):
                ga = float(bb["base_rate"].iloc[0] - aa["base_rate"].iloc[0])
                gap_bits.append(f"{tname} mixed−alldark={_fmt(ga, pct=True)}")
        tercile_md = (
            "Y3 recovery inside train-dark size terciles (descriptive; not a Y cut):\n\n"
            + "\n".join(terc_lines)
            + "\n\n"
            + (
                "Same-tercile gaps: " + "; ".join(gap_bits) + ". "
                "If the 110 still recover more inside T1 and T2, the gap is not only size — "
                "H seeing a sister is a live Q3/Q4 context. Do not invent cobros."
                if gap_bits
                else ""
            )
        )
    else:
        tercile_md = ""
    print(tercile_md)

    new_acc = [
        c
        for c in (
            "y11_neg_3of3_dark",
            "y11_neg_ownp20_2of3_dark",
            "y11_net_recover_6m_dark",
            "y11_neg_onset_3of3_dark",
        )
        if c in y11.columns
    ]
    leak_now = now_leak(y11, ops, tr, new_acc)
    print("contemporaneous net leak")
    print(leak_now.to_string(index=False))
    now_lines = [
        "| column | n | AUROC net<0 now | AUROC |net| | share neg-now on pos |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in leak_now.iterrows():
        now_lines.append(
            f"| `{r.column}` | {int(r.n):,} | {_fmt(r.auc_neg_now)} | "
            f"{_fmt(r.auc_abs_net)} | {_fmt(r.share_neg_now_on_pos, pct=True)} |"
        )
    now_leak_md = (
        "Contemporaneous CAT_MAP net vs the label (train; persistence, not a fit):\n\n"
        + "\n".join(now_lines)
        + "\n\nHigh AUROC of net<0-now is expected for overlapping 3-month windows. "
        "Quote as persistence, not months-of-lead. Onset should have share neg-now ≈ 0."
    )
    sib = sibling_rho(y11, tr, new_acc)
    if not sib.empty:
        print("sibling Spearman among new cousins")
        print(sib.to_string(index=False))
        sib_lines = [
            "| a | b | n | ρ | |",
            "|---|---|---:|---:|---|",
        ]
        for _, r in sib.iterrows():
            flag = " near-copy" if r.near_copy else ""
            sib_lines.append(
                f"| `{r.a}` | `{r.b}` | {int(r.n):,} | {_fmt(r.spearman)} |{flag} |"
            )
        sib_md = "Sibling ρ among new cash cousins (train overlap):\n\n" + "\n".join(sib_lines)
    else:
        sib_md = ""

    acf_bits = []
    ho = ~tr
    ho_bits = []
    for col in ["y11_y2_neg_2of3_dark", *new_acc]:
        a = acf_stats(y11, tr, col)
        acf_bits.append(f"`{col}` acf1={_fmt(a['acf1'])} n={a['n1']}; acf3={_fmt(a['acf3'])} n={a['n3']}")
        y_ho = y11.loc[ho, col]
        ho_bits.append(
            f"`{col}` n_lab={int(y_ho.notna().sum())} pos={int((y_ho == 1).sum())} "
            f"cos_pos={int(y11.loc[ho & (y11[col] == 1), 'company_id'].nunique())}"
        )
    print("ACF", " | ".join(acf_bits))
    print("holdout DESCRIPTIVE", " | ".join(ho_bits), "LOW_POWER")

    leak = leakage_check(
        ["a_op_in", "b_liq", "c_n_tx", "d_cust_hhi", "e_ar_open", "f_fc_r", "h_sib_in"],
        "y11_y2_neg_2of3_dark",
        forbidden_prefixes=["d", "e"],
    )
    leak_ok = leakage_check(
        ["a_op_in", "c_n_tx", "f_fc_r", "g_n_accounts", "h_sib_in"],
        "y11_y2_neg_2of3_dark",
        forbidden_prefixes=["d", "e"],
    )
    print("leakage_check demo (D/E in X)", leak)
    print("leakage_check legal A/C/F/G/H", leak_ok)

    y2d = cmp[(cmp["column"] == "y2_neg_2of3") & (cmp["slice"] == "dark_470")]
    y3d = cmp[(cmp["column"] == "y3_recover_cash_6m") & (cmp["slice"] == "dark_470")]
    y2_ok = bool(len(y2d) and y2d.iloc[0]["usable"])
    y3_ok = bool(len(y3d) and y3d.iloc[0]["usable"])
    n_new = int(acc["accepted"].sum())
    need_own = (
        f"**They do not need a renamed Y2/Y3.** Restricted Y2 is defined "
        f"(n={int(y2d.iloc[0]['n_labeled']) if len(y2d) else 0:,}, "
        f"base={_fmt(y2d.iloc[0]['base_rate'], pct=True) if len(y2d) else '—'}, "
        f"size={_fmt(y2d.iloc[0]['size_auroc']) if len(y2d) else '—'}) and usable "
        f"one-sided={y2_ok}. Restricted Y3 is defined "
        f"(n={int(y3d.iloc[0]['n_labeled']) if len(y3d) else 0:,}, "
        f"base={_fmt(y3d.iloc[0]['base_rate'], pct=True) if len(y3d) else '—'}, "
        f"size={_fmt(y3d.iloc[0]['size_auroc']) if len(y3d) else '—'}; "
        f"two-sided {_fmt(y3d.iloc[0]['size_auroc_two_sided']) if len(y3d) else '—'} "
        f"inherits the parent inverse-size). CLOSE those as subsets. "
        f"Optional cash cousins that are *not* Y2/Y3/Y9 rewrites and pass numeric "
        f"gates: {n_new} in-module. Prefer `y11_neg_onset_3of3_dark` (Q3 turn) over "
        f"`y11_neg_3of3_dark` (already-stressed state). Also "
        f"`y11_neg_ownp20_2of3_dark` and `y11_net_recover_6m_dark`. "
        f"Parent decides merge. Score Y2/Y3-on-dark with A/C/F/G/H (never B/D/E). "
        f"Score net cousins without A (same CAT_MAP)."
    )
    # Rejected y2_onset_neg on the 470: still thin (~1%), not a substitute.
    if "y2_onset_neg" in frozen.columns:
        yo = pd.to_numeric(frozen.loc[tr & frozen["company_id"].isin(pop["train_dark_ids"]), "y2_onset_neg"], errors="coerce")
        ok_on = tr & y11["y11_neg_onset_3of3_dark"].notna() & frozen["y2_onset_neg"].notna()
        rho_on = (
            float(y11.loc[ok_on, "y11_neg_onset_3of3_dark"].corr(frozen.loc[ok_on, "y2_onset_neg"], method="spearman"))
            if int(ok_on.sum()) >= 20
            else float("nan")
        )
        need_own += (
            f" Rejected `y2_onset_neg` on the 470 is still thin "
            f"(n={int(yo.notna().sum()):,}, base={_fmt(float(yo.mean()) if yo.notna().any() else float('nan'), pct=True)}); "
            f"ρ vs `y11_neg_onset_3of3_dark` = {_fmt(rho_on)} — not a rewrite of that reject."
        )
    print("NEED_OWN_Y", need_own)
    ca = set(y11.loc[tr & (y11["y11_neg_onset_3of3_dark"] == 1), "company_id"])
    cb = set(y11.loc[tr & (y11["y11_neg_ownp20_2of3_dark"] == 1), "company_id"])
    cc = set(y11.loc[tr & (y11["y11_net_recover_6m_dark"] == 1), "company_id"])
    union_md = (
        f"Train positive-company union of the three in-module cousins: "
        f"{len(ca | cb | cc)} / 470 (onset {len(ca)}, own-p20 {len(cb)}, "
        f"net-recover {len(cc)}; all-three {len(ca & cb & cc)}; "
        f"onset∩own-p20 {len(ca & cb)}, onset∩recover {len(ca & cc)}, "
        f"own-p20∩recover {len(cb & cc)}). Complementary enough that parent "
        f"can pick one Q3 turn (onset) plus one relative-stress (own-p20) "
        f"without a near-copy."
    )
    print(union_md)
    print("elapsed_s", round((datetime.now() - t0).total_seconds(), 1))

    extra = {
        "acf1": acf_stats(y11, tr, "y11_y2_neg_2of3_dark")["acf1"],
        "acf3": acf_stats(y11, tr, "y11_y2_neg_2of3_dark")["acf3"],
        "acf_n1": acf_stats(y11, tr, "y11_y2_neg_2of3_dark")["n1"],
        "acf_n3": acf_stats(y11, tr, "y11_y2_neg_2of3_dark")["n3"],
        "ho_n": int(y11.loc[ho, "y11_y2_neg_2of3_dark"].notna().sum()),
        "ho_pos": int((y11.loc[ho, "y11_y2_neg_2of3_dark"] == 1).sum()),
        "ho_cos_pos": int(
            y11.loc[ho & (y11["y11_y2_neg_2of3_dark"] == 1), "company_id"].nunique()
        ),
        "mix_read": mix_read,
        "mix_ctx_md": mix_ctx_md,
        "tercile_md": tercile_md,
        "sib_md": sib_md,
        "now_leak_md": now_leak_md,
        "need_own_y": need_own,
        "union_md": union_md,
        "acf_all": " ".join(acf_bits),
        "ho_all": " ".join(ho_bits),
    }
    write_acceptance_md(pop, acc, pos_cos, rho, cmp, mix, extra, n_dark_cm, ctx)
    append_registry(acc)
    print("ITER_FINAL — acceptance + registry written; wave note next")
    con.close()
