"""Family B / balances still QA (reconstruction identity + Q1 honesty).

NORTH_STAR: balances is a 2026-09-01 still (a few rows sit on the closest
prior day). Liquidity is reconstructed backwards on checking+saving+tpv:
end_bal_t = snapshot − sum(flows after t). Night quote: Q1 health is
last-value liquidity (Y1 liq path PARK; Spearman liq_t vs liq_t+3 = 0.85).

Holdout 72 (seed 20260918) is coverage only. Rates / Spearman / PARK-CLOSE-
KEEP are train. No parquet rewrite. No new GBM. No 0–100. Does not run
build_targets. Does not invent a Y from b_liq / b_runway. Does not use
Family B as X for Y2 or Y3. Does not redo Family G.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.balances_b_qa

Owned: analysis/outputs/balances_b_qa.md, optional PNG,
overnight/waves/wave4_balances_b.md (one note at the END),
append-only registry coverage / identity-error rows.

Do not edit liquidity.py unless a real bug. Do not edit y5/y9/G modules.
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.evaluate.protocol import FOLD_SEED, assert_no_holdout, load_holdout
from analysis.features.common import ANALYSIS, AS_OF, DATA, LAST_M, MONTHS, connect
from analysis.features.liquidity import CASH_TYPES, _reconstruct

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:  # pragma: no cover
    HAS_MPL = False

REGISTRY = ANALYSIS / "experiments" / "registry.csv"
STORE = DATA / "feature_store" / "monthly.parquet"
OUT_MD = ANALYSIS / "outputs" / "balances_b_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "balances_b_residual.png"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_balances_b.md"
AGENT = "086b8ed0"
WAVE = "4"
ROUND = "R4"
PANEL_START = MONTHS[0]
PANEL_END = LAST_M
EXTRACT = AS_OF
EXTRA_M = LAST_M + pd.DateOffset(months=1)
SIZE_RHO = 0.50
CASH_SQL = ", ".join(f"'{t}'" for t in CASH_TYPES)
B_COLS = (
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
)


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")


def _fmt(v) -> str:
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return ""
        return f"{float(v):.6g}"
    return "" if v is None else str(v)


def _pp(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{100.0 * float(x):.1f}%"


def spearman(a, b) -> float:
    d = pd.DataFrame(
        {"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}
    ).dropna()
    if len(d) < 8:
        return float("nan")
    if d["a"].nunique() < 2 or d["b"].nunique() < 2:
        return float("nan")
    return float(d["a"].corr(d["b"], method="spearman"))


def _md_table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    right = {
        "n", "n_co", "n_cm", "n_prod", "n_rows", "n_hold", "n_train",
        "rows", "companies", "products", "n_null", "n_zombie", "n_pos",
        "n_neg", "n_labeled", "n_pairs", "n_tx",
    }
    head = "| " + " | ".join(title for _, title in cols) + " |"
    sep = "| " + " | ".join("---:" if k in right else "---" for k, _ in cols) + " |"
    lines = [head, sep]
    shareish = (
        "_rate", "_share", "cov_cm", "cov_co", "share", "prev",
        "share_gt1", "share_gt1pct", "share_neg", "share_lt1",
    )
    rhoish = (
        "rho", "spearman", "acf1", "acf3", "median_abs", "p50", "p10",
        "p25", "p75", "p90", "mean", "mae", "max_abs",
    )
    for r in rows:
        cells = []
        for k, _ in cols:
            v = r.get(k, "")
            if isinstance(v, (float, np.floating, np.integer)) and np.isfinite(float(v)):
                v = float(v)
                if k.endswith("_rate") or k.endswith("_share") or k in {
                    "cov_cm", "cov_co", "share", "prev", "share_gt1", "share_gt1pct",
                    "share_neg", "share_lt1", "share_null", "share_zero",
                    "share_up", "share_closer", "share_farther", "first_neg",
                    "last_neg", "head_neg", "tail_neg", "share_eq",
                    "share_0720", "share_jul", "share_aug",
                } or any(k.endswith(s) for s in shareish):
                    cells.append(f"{100.0 * v:.1f}%")
                elif k.endswith("rho") or k.endswith("_rho") or k in rhoish or k.startswith("acf"):
                    cells.append(f"{v:.3f}")
                else:
                    cells.append(f"{v:.4g}")
            else:
                cells.append("" if v is None or (isinstance(v, float) and not np.isfinite(v)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_store() -> pd.DataFrame:
    need = [
        "company_id",
        "period",
        "a_in3",
        "a_op_in",
        "g_n_accounts",
        *B_COLS,
    ]
    raw = pd.read_parquet(STORE, columns=need)
    raw["company_id"] = raw["company_id"].astype(str)
    raw["period"] = pd.to_datetime(raw["period"])
    return raw


def train_store(store: pd.DataFrame) -> pd.DataFrame:
    hold = load_holdout()
    train = store.loc[~store["company_id"].isin(hold)].copy()
    assert_no_holdout(train["company_id"])
    return train


def _cash_snap(con) -> pd.DataFrame:
    df = con.execute(
        f"""
        SELECT CAST(p.company_id AS VARCHAR) AS company_id,
               SUM(b.balance) AS snap,
               COUNT(*) AS n_cash_prod,
               SUM(ABS(b.balance)) AS snap_abs
        FROM balances b
        JOIN banking_products p ON b.product_id = p.product_id
        WHERE p."type" IN ({CASH_SQL})
          AND b.balance IS NOT NULL
        GROUP BY 1
        """
    ).df()
    df["company_id"] = df["company_id"].astype(str)
    return df


def _cash_flows(con) -> pd.DataFrame:
    df = con.execute(
        f"""
        SELECT CAST(p.company_id AS VARCHAR) AS company_id,
               CAST(date_trunc('month', t."date") AS DATE) AS period,
               SUM(t.amount) AS flow
        FROM transactions t
        JOIN banking_products p ON t.product_id = p.product_id
        WHERE t.product_id IN (
            SELECT b.product_id
            FROM balances b
            JOIN banking_products p2 ON b.product_id = p2.product_id
            WHERE p2."type" IN ({CASH_SQL})
              AND b.balance IS NOT NULL
        )
        GROUP BY 1, 2
        """
    ).df()
    df["company_id"] = df["company_id"].astype(str)
    df["period"] = pd.to_datetime(df["period"])
    return df


def _after_flows(flows: pd.DataFrame, snap: pd.DataFrame) -> pd.DataFrame:
    """Company-month sum of cash-product flows strictly after the period.

    Axis includes 2026-09 so snapshot-day txs sit in the after-sum for August
    (same as liquidity._reconstruct).
    """
    axis = list(MONTHS) + [pd.Timestamp(EXTRA_M)]
    grid = pd.MultiIndex.from_product(
        [snap["company_id"].unique(), axis], names=["company_id", "period"]
    ).to_frame(index=False)
    grid = grid.merge(flows, on=["company_id", "period"], how="left").fillna({"flow": 0.0})
    grid = grid.sort_values(["company_id", "period"]).reset_index(drop=True)
    g = grid.groupby("company_id")["flow"]
    grid["after"] = g.transform("sum") - g.cumsum()
    return grid.loc[grid["period"] <= PANEL_END, ["company_id", "period", "flow", "after"]]


def _trail(con, store: pd.DataFrame) -> pd.DataFrame:
    """First-tx month → panel end. Same span idea as trail_length."""
    tx = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               MIN("date") AS first_tx,
               MAX(CASE WHEN "date" < TIMESTAMP '2026-09-01' THEN "date" END) AS last_grid_tx
        FROM transactions
        WHERE "date" IS NOT NULL
        GROUP BY 1
        """
    ).df()
    tx["company_id"] = tx["company_id"].astype(str)
    tx["first_tx"] = pd.to_datetime(tx["first_tx"])
    tx["first_tx_month"] = tx["first_tx"].dt.to_period("M").dt.to_timestamp()
    grid_n = store.groupby("company_id").size().rename("n_grid")
    first_grid = store.groupby("company_id")["period"].min().rename("first_grid")
    out = tx.merge(grid_n, on="company_id", how="left")
    out = out.merge(first_grid, on="company_id", how="left")
    out["n_grid"] = out["n_grid"].fillna(0).astype(int)
    out["months_on_book"] = (
        (PANEL_END.year - out["first_tx_month"].dt.year) * 12
        + (PANEL_END.month - out["first_tx_month"].dt.month)
        + 1
    )
    out["late"] = out["first_tx_month"] > PANEL_START
    out["short"] = out["months_on_book"] < 12
    out["full24"] = out["n_grid"] == 24
    return out


# ---------------------------------------------------------------------------
# Pass 1 — raw balances still
# ---------------------------------------------------------------------------


def pass1_raw(con) -> dict:
    hold = load_holdout()
    raw = con.execute(
        """
        SELECT
          COUNT(*) AS n_rows,
          COUNT(DISTINCT company_id) AS n_co,
          COUNT(DISTINCT product_id) AS n_prod,
          COUNT(DISTINCT date) AS n_dates,
          MIN(date) AS mind,
          MAX(date) AS maxd,
          SUM(CASE WHEN date = DATE '2026-09-01' THEN 1 ELSE 0 END) AS n_asof,
          SUM(CASE WHEN balance IS NULL THEN 1 ELSE 0 END) AS n_null_bal,
          SUM(CASE WHEN product_known THEN 0 ELSE 1 END) AS n_unknown,
          SUM(CASE WHEN balance_sentinel THEN 1 ELSE 0 END) AS n_sentinel
        FROM balances
        """
    ).df().iloc[0]
    dates = con.execute(
        "SELECT CAST(date AS DATE) AS date, COUNT(*) AS n FROM balances GROUP BY 1 ORDER BY 1"
    ).df()
    dates["date"] = pd.to_datetime(dates["date"])
    types = con.execute(
        """
        SELECT
          CASE
            WHEN p.product_id IS NULL AND d.product_id IS NOT NULL THEN 'debt_orphan'
            WHEN p.product_id IS NULL THEN 'unknown_orphan'
            ELSE coalesce(p."type", '(null)')
          END AS bucket,
          coalesce(p."type", d."type", '(no product)') AS ptype,
          COUNT(*) AS n_rows,
          COUNT(DISTINCT b.company_id) AS n_co,
          SUM(b.balance) AS sum_bal,
          SUM(ABS(b.balance)) AS sum_abs,
          SUM(CASE WHEN b.balance IS NULL THEN 1 ELSE 0 END) AS n_null
        FROM balances b
        LEFT JOIN banking_products p ON b.product_id = p.product_id
        LEFT JOIN debt_products d ON b.product_id = d.product_id
        GROUP BY 1, 2
        ORDER BY n_rows DESC
        """
    ).df()
    cos = con.execute("SELECT DISTINCT CAST(company_id AS VARCHAR) AS company_id FROM balances").df()
    cos["company_id"] = cos["company_id"].astype(str)
    n_train = int((~cos.company_id.isin(hold)).sum())
    n_hold = int(cos.company_id.isin(hold).sum())
    n_all_co = int(con.execute("SELECT COUNT(*) FROM companies").fetchone()[0])
    return {
        "n_rows": int(raw["n_rows"]),
        "n_co": int(raw["n_co"]),
        "n_prod": int(raw["n_prod"]),
        "n_dates": int(raw["n_dates"]),
        "mind": pd.Timestamp(raw["mind"]),
        "maxd": pd.Timestamp(raw["maxd"]),
        "n_asof": int(raw["n_asof"] or 0),
        "share_asof": float(raw["n_asof"] or 0) / float(raw["n_rows"]) if raw["n_rows"] else float("nan"),
        "n_not_asof": int(raw["n_rows"] - (raw["n_asof"] or 0)),
        "n_null_bal": int(raw["n_null_bal"] or 0),
        "n_unknown": int(raw["n_unknown"] or 0),
        "n_sentinel": int(raw["n_sentinel"] or 0),
        "n_train_co": n_train,
        "n_hold_co": n_hold,
        "n_all_co": n_all_co,
        "dates": dates,
        "types": types,
    }


# ---------------------------------------------------------------------------
# Pass 2 — identity: b_liq + after-flows = snapshot
# ---------------------------------------------------------------------------


def pass2_identity(con, store: pd.DataFrame) -> dict:
    hold = load_holdout()
    snap = _cash_snap(con)
    flows = _cash_flows(con)
    after = _after_flows(flows, snap)
    train = train_store(store)
    m = train.merge(snap, on="company_id", how="left")
    m = m.merge(after, on=["company_id", "period"], how="left")
    m["after"] = m["after"].fillna(0.0)
    has = m["b_liq"].notna() & m["snap"].notna()
    rec = m["b_liq"] + m["after"]
    resid = rec - m["snap"]
    absr = resid.abs()
    snap_abs = m["snap"].abs()
    gt1 = absr > 1.0
    gt1pct = (snap_abs > 0) & (absr > 0.01 * snap_abs)
    # rebuild vs store
    rebuilt = _reconstruct(con, "month", list(MONTHS), LAST_M)
    rebuilt["company_id"] = rebuilt["company_id"].astype(str)
    rebuilt["period"] = pd.to_datetime(rebuilt["period"])
    cmp = train.merge(rebuilt, on=["company_id", "period"], how="inner")
    both = cmp["b_liq"].notna() & cmp["liq"].notna()
    store_gap = (cmp.loc[both, "b_liq"] - cmp.loc[both, "liq"]).abs()
    # last-month reconstructed vs snap (diff = September flows)
    last = m.loc[m["period"] == PANEL_END].copy()
    last_has = last["b_liq"].notna() & last["snap"].notna()
    last_resid = (last.loc[last_has, "b_liq"] + last.loc[last_has, "after"]) - last.loc[last_has, "snap"]
    # product-level identity (tighter)
    prod = con.execute(
        f"""
        SELECT CAST(p.company_id AS VARCHAR) AS company_id,
               b.product_id,
               b.balance AS snap
        FROM balances b
        JOIN banking_products p ON b.product_id = p.product_id
        WHERE p."type" IN ({CASH_SQL}) AND b.balance IS NOT NULL
        """
    ).df()
    prod["company_id"] = prod["company_id"].astype(str)
    pflow = con.execute(
        f"""
        SELECT t.product_id,
               CAST(date_trunc('month', t."date") AS DATE) AS period,
               SUM(t.amount) AS flow
        FROM transactions t
        WHERE t.product_id IN (
            SELECT b.product_id FROM balances b
            JOIN banking_products p ON b.product_id = p.product_id
            WHERE p."type" IN ({CASH_SQL}) AND b.balance IS NOT NULL
        )
        GROUP BY 1, 2
        """
    ).df()
    pflow["period"] = pd.to_datetime(pflow["period"])
    paxis = pd.MultiIndex.from_product(
        [prod["product_id"].unique(), list(MONTHS) + [pd.Timestamp(EXTRA_M)]],
        names=["product_id", "period"],
    ).to_frame(index=False)
    paxis = paxis.merge(pflow, on=["product_id", "period"], how="left").fillna({"flow": 0.0})
    paxis = paxis.merge(prod, on="product_id")
    paxis = paxis.sort_values(["product_id", "period"]).reset_index(drop=True)
    pg = paxis.groupby("product_id")["flow"]
    paxis["end_bal"] = paxis["snap"] - (pg.transform("sum") - pg.cumsum())
    paxis = paxis.loc[paxis["period"] <= PANEL_END]
    # compare product-sum to store b_liq
    psum = paxis.groupby(["company_id", "period"], as_index=False)["end_bal"].sum()
    psum["company_id"] = psum["company_id"].astype(str)
    psum["period"] = pd.to_datetime(psum["period"])
    vs = train.merge(psum, on=["company_id", "period"], how="inner")
    vs_both = vs["b_liq"].notna()
    vs_gap = (vs.loc[vs_both, "b_liq"] - vs.loc[vs_both, "end_bal"]).abs()

    def _summ(s: pd.Series) -> dict:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return {"n": 0, "median_abs": float("nan"), "p90": float("nan"), "max_abs": float("nan")}
        return {
            "n": int(len(s)),
            "median_abs": float(s.abs().median()),
            "p90": float(s.abs().quantile(0.90)),
            "max_abs": float(s.abs().max()),
        }

    out = {
        "n_cm": int(has.sum()),
        "n_co": int(m.loc[has, "company_id"].nunique()),
        "median_abs": float(absr[has].median()) if has.any() else float("nan"),
        "p90_abs": float(absr[has].quantile(0.90)) if has.any() else float("nan"),
        "max_abs": float(absr[has].max()) if has.any() else float("nan"),
        "share_gt1": float(gt1[has].mean()) if has.any() else float("nan"),
        "share_gt1pct": float(gt1pct[has].mean()) if has.any() else float("nan"),
        "n_gt1": int(gt1[has].sum()),
        "n_gt1pct": int(gt1pct[has].sum()),
        "store_vs_rebuild": _summ(store_gap),
        "store_vs_product": _summ(vs_gap),
        "last_resid": _summ(last_resid),
        "n_last": int(last_has.sum()),
        "resid": resid[has].to_numpy(dtype=float) if has.any() else np.array([]),
        "rec": rec[has].to_numpy(dtype=float) if has.any() else np.array([]),
        "snap_v": m.loc[has, "snap"].to_numpy(dtype=float) if has.any() else np.array([]),
        "tiny": bool(has.any() and float(absr[has].median()) < 1e-6 and float(gt1[has].mean()) < 0.01),
        "n_train_cm": int(len(train)),
        "n_hold_excluded": int(store["company_id"].isin(hold).sum()),
    }
    return out


# ---------------------------------------------------------------------------
# Pass 3 — walk types vs excluded cash
# ---------------------------------------------------------------------------


def pass3_types(con) -> dict:
    hold = load_holdout()
    raw = con.execute(
        f"""
        SELECT
          CASE
            WHEN p.product_id IS NOT NULL AND p."type" IN ({CASH_SQL}) THEN 'walk'
            WHEN p.product_id IS NOT NULL THEN 'bank_excluded'
            WHEN d.product_id IS NOT NULL THEN 'debt_orphan'
            ELSE 'unknown_orphan'
          END AS pile,
          CASE
            WHEN p.product_id IS NOT NULL THEN p."type"
            WHEN d.product_id IS NOT NULL THEN d."type"
            ELSE '(unknown)'
          END AS ptype,
          CASE
            WHEN p.product_id IS NOT NULL AND (
              lower(coalesce(p.service, '')) = 'custom'
              OR p.bank_name ILIKE '%customer-defined%'
              OR p.bank_name ILIKE 'Other%'
            ) THEN 1 ELSE 0
          END AS is_custom,
          COUNT(*) AS n_rows,
          COUNT(DISTINCT b.company_id) AS n_co,
          SUM(b.balance) AS sum_bal,
          SUM(ABS(COALESCE(b.balance, 0))) AS sum_abs,
          SUM(CASE WHEN b.balance > 0 THEN b.balance ELSE 0 END) AS pos_bal
        FROM balances b
        LEFT JOIN banking_products p ON b.product_id = p.product_id
        LEFT JOIN debt_products d ON b.product_id = d.product_id
        GROUP BY 1, 2, 3
        ORDER BY pile, n_rows DESC
        """
    ).df()
    # pile totals
    pile = (
        raw.groupby("pile", as_index=False)
        .agg(n_rows=("n_rows", "sum"), n_co=("n_co", "sum"), sum_bal=("sum_bal", "sum"),
             sum_abs=("sum_abs", "sum"), pos_bal=("pos_bal", "sum"))
        .sort_values("n_rows", ascending=False)
    )
    tot_abs = float(raw["sum_abs"].sum())
    tot_pos = float(raw["pos_bal"].sum())
    walk_abs = float(raw.loc[raw["pile"] == "walk", "sum_abs"].sum())
    excl_abs = tot_abs - walk_abs
    walk_pos = float(raw.loc[raw["pile"] == "walk", "pos_bal"].sum())
    # custom that IS in the walk (type filter only — custom checking is walked)
    custom_walk = raw.loc[(raw["pile"] == "walk") & (raw["is_custom"] == 1)]
    custom_excl = raw.loc[(raw["pile"] != "walk") & (raw["is_custom"] == 1)]
    named = (
        raw.groupby("ptype", as_index=False)
        .agg(n_rows=("n_rows", "sum"), n_co=("n_co", "sum"), sum_bal=("sum_bal", "sum"),
             sum_abs=("sum_abs", "sum"), pos_bal=("pos_bal", "sum"))
        .sort_values("n_rows", ascending=False)
    )
    named["in_walk"] = named["ptype"].isin(CASH_TYPES)
    # train companies whose entire snapshot is excluded (no cash walk)
    cash_cos = set(
        con.execute(
            f"""
            SELECT DISTINCT CAST(p.company_id AS VARCHAR) AS company_id
            FROM balances b
            JOIN banking_products p ON b.product_id = p.product_id
            WHERE p."type" IN ({CASH_SQL}) AND b.balance IS NOT NULL
            """
        ).df()["company_id"].astype(str)
    )
    bal_cos = set(
        con.execute("SELECT DISTINCT CAST(company_id AS VARCHAR) AS company_id FROM balances").df()["company_id"].astype(str)
    )
    train_bal = {c for c in bal_cos if c not in hold}
    n_train_no_walk = len(train_bal - cash_cos)
    return {
        "raw": raw,
        "pile": pile,
        "named": named,
        "tot_abs": tot_abs,
        "tot_pos": tot_pos,
        "walk_abs": walk_abs,
        "excl_abs": excl_abs,
        "excl_share_abs": excl_abs / tot_abs if tot_abs else float("nan"),
        "walk_pos": walk_pos,
        "excl_pos_share": (tot_pos - walk_pos) / tot_pos if tot_pos else float("nan"),
        "custom_walk_n": int(custom_walk["n_rows"].sum()) if len(custom_walk) else 0,
        "custom_walk_abs": float(custom_walk["sum_abs"].sum()) if len(custom_walk) else 0.0,
        "custom_excl_n": int(custom_excl["n_rows"].sum()) if len(custom_excl) else 0,
        "n_train_no_walk": n_train_no_walk,
        "n_train_bal": len(train_bal),
        "n_cash_co": len(cash_cos),
    }


# ---------------------------------------------------------------------------
# Pass 4 — coverage vs txs vs G connection hole
# ---------------------------------------------------------------------------


def pass4_coverage(con, store: pd.DataFrame) -> dict:
    hold = load_holdout()
    train = train_store(store)
    n_train_co = int(train["company_id"].nunique())
    bal = con.execute("SELECT DISTINCT CAST(company_id AS VARCHAR) AS company_id FROM balances").df()
    bal["company_id"] = bal["company_id"].astype(str)
    tx = con.execute("SELECT DISTINCT CAST(company_id AS VARCHAR) AS company_id FROM transactions").df()
    tx["company_id"] = tx["company_id"].astype(str)
    cash = _cash_snap(con)
    train_ids = set(train["company_id"])
    bal_tr = set(bal.company_id) & train_ids
    tx_tr = set(tx.company_id) & train_ids
    cash_tr = set(cash.company_id) & train_ids
    ever_g = set(train.loc[pd.to_numeric(train["g_n_accounts"], errors="coerce") > 0, "company_id"])
    first = train.sort_values("period").groupby("company_id", sort=False).first()
    first_g0 = pd.to_numeric(first["g_n_accounts"], errors="coerce") == 0
    first_liq = pd.to_numeric(first["b_liq"], errors="coerce")
    un = first.loc[first_g0]
    un_liq = pd.to_numeric(un["b_liq"], errors="coerce")
    # last-month G vs B
    last = train.sort_values("period").groupby("company_id", sort=False).last()
    last_g0 = pd.to_numeric(last["g_n_accounts"], errors="coerce") == 0
    last_liq = pd.to_numeric(last["b_liq"], errors="coerce")
    # CM: g=0 and b_liq
    g0 = pd.to_numeric(train["g_n_accounts"], errors="coerce") == 0
    g0_liq = pd.to_numeric(train.loc[g0, "b_liq"], errors="coerce")
    return {
        "n_train_co": n_train_co,
        "n_train_cm": int(len(train)),
        "n_bal": len(bal_tr),
        "n_tx": len(tx_tr),
        "n_cash": len(cash_tr),
        "n_ever_g": len(ever_g),
        "n_bal_not_tx": len(bal_tr - tx_tr),
        "n_tx_not_bal": len(tx_tr - bal_tr),
        "n_bal_not_cash": len(bal_tr - cash_tr),
        "n_ever_g_no_cash": len(ever_g - cash_tr),
        "n_cash_no_g": len(cash_tr - ever_g),
        "first_g0_n": int(first_g0.sum()),
        "first_g0_share": float(first_g0.mean()),
        "first_g0_liq_null": float(un_liq.isna().mean()) if len(un) else float("nan"),
        "first_g0_liq_zero": float((un_liq == 0).mean()) if len(un) else float("nan"),
        "first_g0_liq_finite": float(un_liq.notna().mean()) if len(un) else float("nan"),
        "first_g0_liq_neg": float((un_liq < 0).mean()) if len(un) else float("nan"),
        "first_g0_liq_p50": float(un_liq.median()) if un_liq.notna().any() else float("nan"),
        "first_all_liq_null": float(first_liq.isna().mean()),
        "last_g0_n": int(last_g0.sum()),
        "last_g0_liq_null": float(last_liq[last_g0].isna().mean()) if last_g0.any() else float("nan"),
        "g0_cm": int(g0.sum()),
        "g0_liq_null": float(g0_liq.isna().mean()) if g0.any() else float("nan"),
        "g0_liq_zero": float((g0_liq == 0).mean()) if g0.any() else float("nan"),
        "g0_liq_finite": float(g0_liq.notna().mean()) if g0.any() else float("nan"),
        "g0_liq_p50": float(g0_liq.median()) if g0_liq.notna().any() else float("nan"),
        "b_liq_cm_null": float(train["b_liq"].isna().mean()),
        "b_liq_cm_zero": float((pd.to_numeric(train["b_liq"], errors="coerce") == 0).mean()),
        "b_liq_co_any": float((train.groupby("company_id")["b_liq"].apply(lambda s: s.notna().any())).mean()),
    }


# ---------------------------------------------------------------------------
# Pass 5 — persistence short vs long (is the still leaking?)
# ---------------------------------------------------------------------------


def _lag_pairs(df: pd.DataFrame, col: str, lag: int) -> pd.DataFrame:
    rows = []
    for cid, g in df.sort_values(["company_id", "period"]).groupby("company_id", sort=False):
        x = pd.to_numeric(g[col], errors="coerce").to_numpy()
        if len(x) <= lag:
            continue
        a = x[:-lag]
        b = x[lag:]
        per = g["period"].to_numpy()[:-lag]
        rows.append(pd.DataFrame({"company_id": cid, "period": per, "a": a, "b": b}))
    if not rows:
        return pd.DataFrame(columns=["company_id", "period", "a", "b"])
    return pd.concat(rows, ignore_index=True)


def pass5_persist(con, store: pd.DataFrame) -> dict:
    train = train_store(store)
    trail = _trail(con, store)
    hold = load_holdout()
    trail_tr = trail.loc[~trail["company_id"].isin(hold)].copy()
    assert_no_holdout(trail_tr["company_id"])
    train = train.merge(
        trail_tr[["company_id", "months_on_book", "short", "full24", "late", "n_grid", "first_tx_month"]],
        on="company_id",
        how="left",
    )
    snap = _cash_snap(con)
    train = train.merge(snap[["company_id", "snap"]], on="company_id", how="left")

    def _rho_lag(part: pd.DataFrame, lag: int) -> dict:
        pairs = _lag_pairs(part, "b_liq", lag).dropna()
        rho = spearman(pairs["a"], pairs["b"]) if len(pairs) else float("nan")
        return {"n_pairs": int(len(pairs)), "n_co": int(pairs["company_id"].nunique()) if len(pairs) else 0, "rho": rho}

    slices = {
        "all": train,
        "short_<12": train.loc[train["short"].fillna(False)],
        "long_24": train.loc[train["full24"].fillna(False)],
        "late_first_tx": train.loc[train["late"].fillna(False)],
        "from_2024_09": train.loc[~train["late"].fillna(False)],
    }
    lag_rows = []
    for name, part in slices.items():
        for lag in (1, 3):
            r = _rho_lag(part, lag)
            r["slice"] = name
            r["lag"] = lag
            lag_rows.append(r)

    # company first vs last vs snap
    first = train.sort_values("period").groupby("company_id", sort=False).first().reset_index()
    last = train.sort_values("period").groupby("company_id", sort=False).last().reset_index()
    co = first[["company_id", "b_liq", "months_on_book", "short", "full24", "late", "snap"]].rename(
        columns={"b_liq": "liq_first"}
    )
    co = co.merge(last[["company_id", "b_liq"]].rename(columns={"b_liq": "liq_last"}), on="company_id")
    rng = train.groupby("company_id")["b_liq"].agg(["min", "max", "std", "count"])
    rng["range"] = rng["max"] - rng["min"]
    co = co.merge(rng.reset_index(), on="company_id")
    co["rel_range"] = co["range"] / np.maximum(co["snap"].abs(), 1.0)
    co["abs_first_snap"] = (co["liq_first"] - co["snap"]).abs()
    co["abs_last_snap"] = (co["liq_last"] - co["snap"]).abs()
    co["rel_first"] = co["abs_first_snap"] / np.maximum(co["snap"].abs(), 1.0)
    co["rel_last"] = co["abs_last_snap"] / np.maximum(co["snap"].abs(), 1.0)

    def _co_slice(part: pd.DataFrame, name: str) -> dict:
        lab = part["liq_first"].notna() & part["snap"].notna()
        return {
            "slice": name,
            "n_co": int(len(part)),
            "rho_first_snap": spearman(part.loc[lab, "liq_first"], part.loc[lab, "snap"]),
            "rho_last_snap": spearman(part.loc[lab, "liq_last"], part.loc[lab, "snap"]),
            "rho_first_last": spearman(part["liq_first"], part["liq_last"]),
            "rel_range_p50": float(part["rel_range"].median()) if part["rel_range"].notna().any() else float("nan"),
            "rel_first_p50": float(part["rel_first"].median()) if part["rel_first"].notna().any() else float("nan"),
            "rel_last_p50": float(part["rel_last"].median()) if part["rel_last"].notna().any() else float("nan"),
        }

    co_rows = [
        _co_slice(co, "all"),
        _co_slice(co.loc[co["short"].fillna(False)], "short_<12"),
        _co_slice(co.loc[co["full24"].fillna(False)], "long_24"),
        _co_slice(co.loc[co["late"].fillna(False)], "late_first_tx"),
        _co_slice(co.loc[~co["late"].fillna(False)], "from_2024_09"),
    ]
    all_lag3 = next(r for r in lag_rows if r["slice"] == "all" and r["lag"] == 3)
    short_lag3 = next(r for r in lag_rows if r["slice"] == "short_<12" and r["lag"] == 3)
    long_lag3 = next(r for r in lag_rows if r["slice"] == "long_24" and r["lag"] == 3)
    # still-leak test: short persist >> long, and first≈snap on short
    leak_short = (
        np.isfinite(short_lag3["rho"])
        and np.isfinite(long_lag3["rho"])
        and short_lag3["rho"] > long_lag3["rho"] + 0.03
    )
    first_like_snap_short = False
    short_co = next(r for r in co_rows if r["slice"] == "short_<12")
    long_co = next(r for r in co_rows if r["slice"] == "long_24")
    if np.isfinite(short_co["rho_first_snap"]) and np.isfinite(long_co["rho_first_snap"]):
        first_like_snap_short = short_co["rho_first_snap"] > long_co["rho_first_snap"] + 0.03
    still_shining = bool(leak_short and first_like_snap_short)
    # real trail on long books: first-vs-snap much weaker than last-vs-snap, or rel_range not tiny
    long_real = bool(
        np.isfinite(long_co["rho_first_snap"])
        and np.isfinite(long_co["rho_last_snap"])
        and (long_co["rho_last_snap"] - long_co["rho_first_snap"]) >= 0.05
        or (np.isfinite(long_co["rel_range_p50"]) and long_co["rel_range_p50"] >= 0.20)
    )
    return {
        "lag_rows": lag_rows,
        "co_rows": co_rows,
        "all_lag3": all_lag3,
        "short_lag3": short_lag3,
        "long_lag3": long_lag3,
        "confirm_085": bool(np.isfinite(all_lag3["rho"]) and abs(all_lag3["rho"] - 0.85) < 0.02),
        "leak_short": leak_short,
        "first_like_snap_short": first_like_snap_short,
        "still_shining": still_shining,
        "long_real": long_real,
        "n_short_co": int(trail_tr["short"].fillna(False).sum()),
        "n_long_co": int(trail_tr["full24"].fillna(False).sum()),
        "n_late_co": int(trail_tr["late"].fillna(False).sum()),
    }


# ---------------------------------------------------------------------------
# Pass 6 — Q1 last-value description (NOT a Y2/Y3 model)
# ---------------------------------------------------------------------------


def pass6_q1(store: pd.DataFrame) -> dict:
    train = train_store(store)
    last = train.sort_values("period").groupby("company_id", sort=False).last().reset_index()
    liq = pd.to_numeric(last["b_liq"], errors="coerce")
    rw = pd.to_numeric(last["b_runway"], errors="coerce")
    cm_liq = pd.to_numeric(train["b_liq"], errors="coerce")
    cm_rw = pd.to_numeric(train["b_runway"], errors="coerce")

    def _dist(s: pd.Series) -> dict:
        nn = s.dropna()
        if nn.empty:
            return {k: float("nan") for k in ("n", "mean", "p10", "p25", "p50", "p75", "p90", "share_neg", "share_null")}
        return {
            "n": int(len(s)),
            "n_nn": int(len(nn)),
            "mean": float(nn.mean()),
            "p10": float(nn.quantile(0.10)),
            "p25": float(nn.quantile(0.25)),
            "p50": float(nn.quantile(0.50)),
            "p75": float(nn.quantile(0.75)),
            "p90": float(nn.quantile(0.90)),
            "share_neg": float((nn < 0).mean()),
            "share_null": float(s.isna().mean()),
        }

    last_liq = _dist(liq)
    last_rw = _dist(rw)
    last_rw["share_lt1"] = float((rw.dropna() < 1).mean()) if rw.notna().any() else float("nan")
    last_stressed = float(((liq < 0) | (rw < 1)).mean()) if (liq.notna() | rw.notna()).any() else float("nan")
    # among non-null last liq
    both = liq.notna()
    last_stressed_nn = float(((liq[both] < 0) | (rw[both] < 1)).mean()) if both.any() else float("nan")
    cm = {
        "liq": _dist(cm_liq),
        "rw": _dist(cm_rw),
        "share_rw_lt1": float((cm_rw.dropna() < 1).mean()) if cm_rw.notna().any() else float("nan"),
        "share_stressed": float(((cm_liq < 0) | (cm_rw < 1)).mean()),
    }
    # clipped runway pile (the [-6, 24] walls)
    rw_nn = rw.dropna()
    return {
        "n_co": int(len(last)),
        "last_liq": last_liq,
        "last_rw": last_rw,
        "last_stressed": last_stressed,
        "last_stressed_nn": last_stressed_nn,
        "cm": cm,
        "rw_floor_share": float((rw_nn <= -6 + 1e-9).mean()) if len(rw_nn) else float("nan"),
        "rw_ceil_share": float((rw_nn >= 24 - 1e-9).mean()) if len(rw_nn) else float("nan"),
        "n_last_liq_nn": int(liq.notna().sum()),
        "n_last_rw_nn": int(rw.notna().sum()),
    }


# ---------------------------------------------------------------------------
# Pass 7 — holdout coverage (descriptive only)
# ---------------------------------------------------------------------------


def pass7_holdout(con, store: pd.DataFrame) -> dict:
    hold = load_holdout()
    bal = con.execute("SELECT DISTINCT CAST(company_id AS VARCHAR) AS company_id FROM balances").df()
    bal["company_id"] = bal["company_id"].astype(str)
    cash = _cash_snap(con)
    ho = store.loc[store["company_id"].isin(hold)].copy()
    n_hold = len(hold)
    n_bal = int(sum(1 for c in hold if c in set(bal.company_id)))
    n_cash = int(sum(1 for c in hold if c in set(cash.company_id)))
    ho_liq = pd.to_numeric(ho["b_liq"], errors="coerce")
    last = ho.sort_values("period").groupby("company_id", sort=False).last()
    return {
        "n_hold": n_hold,
        "n_bal": n_bal,
        "n_cash": n_cash,
        "share_bal": n_bal / n_hold if n_hold else float("nan"),
        "share_cash": n_cash / n_hold if n_hold else float("nan"),
        "n_cm": int(len(ho)),
        "liq_cm_cov": float(ho_liq.notna().mean()) if len(ho) else float("nan"),
        "n_co_any_liq": int(last["b_liq"].notna().sum()),
    }


# ---------------------------------------------------------------------------
# Pass 8 — zombie snapshot cash (0 txs in 24 months)
# ---------------------------------------------------------------------------


def pass8_zombies(con) -> dict:
    hold = load_holdout()
    raw = con.execute(
        f"""
        SELECT CAST(p.company_id AS VARCHAR) AS company_id,
               b.product_id,
               p."type" AS ptype,
               b.balance,
               coalesce(t.n_tx, 0) AS n_tx,
               t.min_d,
               t.max_d
        FROM balances b
        JOIN banking_products p ON b.product_id = p.product_id
        LEFT JOIN (
            SELECT product_id, COUNT(*) AS n_tx, MIN("date") AS min_d, MAX("date") AS max_d
            FROM transactions
            GROUP BY 1
        ) t ON b.product_id = t.product_id
        WHERE p."type" IN ({CASH_SQL}) AND b.balance IS NOT NULL
        """
    ).df()
    raw["company_id"] = raw["company_id"].astype(str)
    raw["is_hold"] = raw["company_id"].isin(hold)
    tr = raw.loc[~raw["is_hold"]].copy()
    assert_no_holdout(tr["company_id"])
    z = tr["n_tx"] == 0
    by_type = []
    for t, part in tr.groupby("ptype"):
        zz = part["n_tx"] == 0
        by_type.append(
            {
                "ptype": t,
                "n_prod": int(len(part)),
                "n_zombie": int(zz.sum()),
                "share": float(zz.mean()),
                "zombie_bal": float(part.loc[zz, "balance"].sum()),
                "all_bal": float(part["balance"].sum()),
                "zombie_abs": float(part.loc[zz, "balance"].abs().sum()),
            }
        )
    zbal = tr.loc[z].groupby("company_id")["balance"].sum().rename("zombie_bal")
    csnap = tr.groupby("company_id")["balance"].sum().rename("snap")
    cz = csnap.to_frame().join(zbal, how="left").fillna({"zombie_bal": 0.0})
    cz["n_prod"] = tr.groupby("company_id").size()
    cz["n_zombie"] = tr.loc[z].groupby("company_id").size().reindex(cz.index).fillna(0)
    all_z = cz["n_zombie"] == cz["n_prod"]
    return {
        "n_prod": int(len(tr)),
        "n_zombie": int(z.sum()),
        "share_prod": float(z.mean()) if len(tr) else float("nan"),
        "zombie_bal": float(tr.loc[z, "balance"].sum()),
        "zombie_abs": float(tr.loc[z, "balance"].abs().sum()),
        "cash_bal": float(tr["balance"].sum()),
        "cash_abs": float(tr["balance"].abs().sum()),
        "share_bal": float(tr.loc[z, "balance"].abs().sum() / tr["balance"].abs().sum()) if tr["balance"].abs().sum() else float("nan"),
        "n_co_any": int((cz["n_zombie"] > 0).sum()),
        "n_co_all": int(all_z.sum()),
        "all_z_bal": float(cz.loc[all_z, "snap"].sum()) if all_z.any() else 0.0,
        "by_type": by_type,
        "n_train_cash_co": int(len(cz)),
    }


# ---------------------------------------------------------------------------
# Pass 9 — is b_bal_vol SIZE?
# ---------------------------------------------------------------------------


def pass9_vol(store: pd.DataFrame) -> dict:
    train = train_store(store)
    vol = pd.to_numeric(train["b_bal_vol"], errors="coerce")
    liq = pd.to_numeric(train["b_liq"], errors="coerce")
    ain = pd.to_numeric(train["a_in3"], errors="coerce")
    login = np.log1p(ain.clip(lower=0))
    last = train.sort_values("period").groupby("company_id", sort=False).last()
    last_vol = pd.to_numeric(last["b_bal_vol"], errors="coerce")
    last_log = np.log1p(pd.to_numeric(last["a_in3"], errors="coerce").clip(lower=0))
    last_liq = pd.to_numeric(last["b_liq"], errors="coerce")
    return {
        "n_cm": int(vol.notna().sum()),
        "cov_cm": float(vol.notna().mean()),
        "p50": float(vol.median()) if vol.notna().any() else float("nan"),
        "p90": float(vol.quantile(0.90)) if vol.notna().any() else float("nan"),
        "share_gt1": float((vol.dropna() > 1).mean()) if vol.notna().any() else float("nan"),
        "rho_login3": spearman(vol, login),
        "rho_ain3": spearman(vol, ain),
        "rho_liq": spearman(vol, liq),
        "rho_abs_liq": spearman(vol, liq.abs()),
        "last_rho_login3": spearman(last_vol, last_log),
        "last_rho_liq": spearman(last_vol, last_liq.abs()),
        "is_size": bool(
            (abs(spearman(vol, login)) >= SIZE_RHO) or (abs(spearman(vol, ain)) >= SIZE_RHO)
        ),
        "n_co_last": int(last_vol.notna().sum()),
    }


# ---------------------------------------------------------------------------
# Pass 10 — negatives at the start of late books (left-trunc + still)
# ---------------------------------------------------------------------------


def pass10_neg(con, store: pd.DataFrame) -> dict:
    train = train_store(store)
    trail = _trail(con, store)
    hold = load_holdout()
    trail_tr = trail.loc[~trail["company_id"].isin(hold)]
    train = train.merge(
        trail_tr[["company_id", "first_tx_month", "late", "months_on_book", "full24"]],
        on="company_id",
        how="left",
    )
    train["pre_tx"] = train["period"] < train["first_tx_month"]
    liq = pd.to_numeric(train["b_liq"], errors="coerce")
    train["neg"] = liq < 0
    # first month of each company
    first = train.sort_values("period").groupby("company_id", sort=False).first().reset_index()
    # among late books, share of pre-tx months that are negative
    late = train.loc[train["late"].fillna(False)]
    pre = late.loc[late["pre_tx"].fillna(False)]
    on = late.loc[~late["pre_tx"].fillna(False)]
    early = train.loc[~train["late"].fillna(False)]
    # constant pre-tx path? (same reconstructed liq on every pre-tx month)
    pre_std = (
        pre.groupby("company_id")["b_liq"].std()
        if len(pre)
        else pd.Series(dtype=float)
    )
    n_pre_co = int(pre["company_id"].nunique()) if len(pre) else 0
    n_const = int((pre_std.fillna(0) < 1e-6).sum()) if n_pre_co else 0
    # first-month negatives: late vs on-time
    first_neg_late = float((pd.to_numeric(first.loc[first["late"].fillna(False), "b_liq"], errors="coerce") < 0).mean())
    first_neg_ontime = float((pd.to_numeric(first.loc[~first["late"].fillna(False), "b_liq"], errors="coerce") < 0).mean())
    # calendar of negative share
    cal = (
        train.assign(neg=train["neg"].astype(float))
        .groupby("period")
        .agg(n_cm=("company_id", "size"), n_neg=("neg", "sum"), n_late=("late", "sum"))
        .reset_index()
    )
    cal["share_neg"] = cal["n_neg"] / cal["n_cm"]
    # late-book first-month vs last-month neg
    last = train.sort_values("period").groupby("company_id", sort=False).last().reset_index()
    late_ids = set(first.loc[first["late"].fillna(False), "company_id"])
    last_neg_late = float((pd.to_numeric(last.loc[last["company_id"].isin(late_ids), "b_liq"], errors="coerce") < 0).mean())
    return {
        "cm_neg_share": float(train["neg"].mean()),
        "cm_neg_n": int(train["neg"].sum()),
        "late_pre_n": int(len(pre)),
        "late_pre_neg": float(pre["neg"].mean()) if len(pre) else float("nan"),
        "late_on_neg": float(on["neg"].mean()) if len(on) else float("nan"),
        "ontime_neg": float((pd.to_numeric(early["b_liq"], errors="coerce") < 0).mean()) if len(early) else float("nan"),
        "n_pre_co": n_pre_co,
        "n_const_pre": n_const,
        "share_const_pre": n_const / n_pre_co if n_pre_co else float("nan"),
        "first_neg_late": first_neg_late,
        "first_neg_ontime": first_neg_ontime,
        "last_neg_late": last_neg_late,
        "cal": cal,
        "clusters": bool(
            np.isfinite(first_neg_late)
            and np.isfinite(first_neg_ontime)
            and first_neg_late > first_neg_ontime + 0.03
        ),
        "grid_starts_at_first_tx": n_pre_co == 0,
    }


# ---------------------------------------------------------------------------
# Pass 11 — why short persist is LOWER (opposite of still-leak)
# ---------------------------------------------------------------------------


def pass11_short_why(con, store: pd.DataFrame) -> dict:
    """Drop birth months; persist by trail bucket; first-vs-snap by first-tx month."""
    train = train_store(store)
    trail = _trail(con, store)
    hold = load_holdout()
    trail_tr = trail.loc[~trail["company_id"].isin(hold)].copy()
    assert_no_holdout(trail_tr["company_id"])
    train = train.merge(
        trail_tr[["company_id", "first_tx_month", "months_on_book", "short", "full24", "late"]],
        on="company_id",
        how="left",
    )
    snap = _cash_snap(con)
    train = train.merge(snap[["company_id", "snap"]], on="company_id", how="left")
    train = train.sort_values(["company_id", "period"]).reset_index(drop=True)
    train["age"] = (
        (train["period"].dt.year - train["first_tx_month"].dt.year) * 12
        + (train["period"].dt.month - train["first_tx_month"].dt.month)
    )
    # persist after dropping first 3 on-book months
    mature = train.loc[train["age"].fillna(-1) >= 3]

    def _rho(part: pd.DataFrame, lag: int = 3) -> dict:
        pairs = _lag_pairs(part, "b_liq", lag).dropna()
        return {
            "n_pairs": int(len(pairs)),
            "n_co": int(pairs["company_id"].nunique()) if len(pairs) else 0,
            "rho": spearman(pairs["a"], pairs["b"]) if len(pairs) else float("nan"),
        }

    buckets = [
        ("<6", train["months_on_book"] < 6),
        ("6-11", (train["months_on_book"] >= 6) & (train["months_on_book"] < 12)),
        ("12-17", (train["months_on_book"] >= 12) & (train["months_on_book"] < 18)),
        ("18-23", (train["months_on_book"] >= 18) & (train["months_on_book"] < 24)),
        ("24", train["months_on_book"] >= 24),
    ]
    buck_rows = []
    for name, mask in buckets:
        r = _rho(train.loc[mask.fillna(False)])
        r["slice"] = name
        r["kind"] = "all_months"
        buck_rows.append(r)
        r2 = _rho(mature.loc[mask.fillna(False)])
        r2["slice"] = name
        r2["kind"] = "age>=3"
        buck_rows.append(r2)

    short_all = _rho(train.loc[train["short"].fillna(False)])
    short_mat = _rho(train.loc[train["short"].fillna(False) & (train["age"].fillna(-1) >= 3)])
    long_mat = _rho(train.loc[train["full24"].fillna(False) & (train["age"].fillna(-1) >= 3)])

    # first-month vs snap by first-tx calendar
    first = train.sort_values("period").groupby("company_id", sort=False).first().reset_index()
    first["abs_rel"] = (first["b_liq"] - first["snap"]).abs() / np.maximum(first["snap"].abs(), 1.0)
    cal = []
    for ym, part in first.groupby(first["first_tx_month"].dt.strftime("%Y-%m")):
        lab = part["b_liq"].notna() & part["snap"].notna()
        cal.append(
            {
                "first_tx": ym,
                "n_co": int(len(part)),
                "rho_first_snap": spearman(part.loc[lab, "b_liq"], part.loc[lab, "snap"]),
                "rel_p50": float(part.loc[lab, "abs_rel"].median()) if lab.any() else float("nan"),
                "neg_share": float((pd.to_numeric(part["b_liq"], errors="coerce") < 0).mean()),
            }
        )
    # still-leak ranking test: later first_tx → higher first-vs-snap
    late_first = first.loc[first["late"].fillna(False)]
    ontime_first = first.loc[~first["late"].fillna(False)]
    rho_late = spearman(late_first["b_liq"], late_first["snap"])
    rho_ontime = spearman(ontime_first["b_liq"], ontime_first["snap"])
    leak_rank = bool(np.isfinite(rho_late) and np.isfinite(rho_ontime) and rho_late > rho_ontime + 0.03)
    # birth-month jump: |Δ liq| / max(|liq|,1) on first→second
    jumps = []
    for cid, g in train.groupby("company_id", sort=False):
        g = g.sort_values("period")
        if len(g) < 2:
            continue
        a, b = g["b_liq"].iloc[0], g["b_liq"].iloc[1]
        if pd.isna(a) or pd.isna(b):
            continue
        jumps.append(
            {
                "company_id": cid,
                "short": bool(g["short"].iloc[0]),
                "rel_jump": abs(b - a) / max(abs(a), 1.0),
            }
        )
    jdf = pd.DataFrame(jumps)
    return {
        "buck_rows": buck_rows,
        "short_all": short_all,
        "short_mat": short_mat,
        "long_mat": long_mat,
        "cal": cal,
        "rho_late_first": rho_late,
        "rho_ontime_first": rho_ontime,
        "leak_rank": leak_rank,
        "jump_p50_short": float(jdf.loc[jdf["short"], "rel_jump"].median()) if len(jdf) and jdf["short"].any() else float("nan"),
        "jump_p50_long": float(jdf.loc[~jdf["short"], "rel_jump"].median()) if len(jdf) and (~jdf["short"]).any() else float("nan"),
        "birth_explains": bool(
            np.isfinite(short_mat["rho"])
            and np.isfinite(short_all["rho"])
            and short_mat["rho"] - short_all["rho"] >= 0.05
        ),
    }


# ---------------------------------------------------------------------------
# Pass 12 — early-on-book negatives (grid starts at first tx)
# ---------------------------------------------------------------------------


def pass12_age_neg(con, store: pd.DataFrame) -> dict:
    train = train_store(store)
    trail = _trail(con, store)
    hold = load_holdout()
    trail_tr = trail.loc[~trail["company_id"].isin(hold)]
    train = train.merge(
        trail_tr[["company_id", "first_tx_month", "late", "short", "full24", "months_on_book"]],
        on="company_id",
        how="left",
    )
    train["age"] = (
        (train["period"].dt.year - train["first_tx_month"].dt.year) * 12
        + (train["period"].dt.month - train["first_tx_month"].dt.month)
    )
    liq = pd.to_numeric(train["b_liq"], errors="coerce")
    train["neg"] = liq < 0
    age_rows = []
    for late_flag, lname in ((True, "late"), (False, "ontime")):
        part = train.loc[train["late"].fillna(False) == late_flag]
        for lo, hi, aname in ((0, 0, "age0"), (1, 2, "age1-2"), (3, 5, "age3-5"), (6, 99, "age6+")):
            sl = part.loc[part["age"].between(lo, hi)]
            age_rows.append(
                {
                    "book": lname,
                    "age": aname,
                    "n_cm": int(len(sl)),
                    "n_co": int(sl["company_id"].nunique()) if len(sl) else 0,
                    "share_neg": float(sl["neg"].mean()) if len(sl) else float("nan"),
                }
            )
    # first 3 vs last 3 within late / short
    def _head_tail(mask, name):
        part = train.loc[mask].sort_values(["company_id", "period"])
        head = part.groupby("company_id", sort=False).head(3)
        tail = part.groupby("company_id", sort=False).tail(3)
        return {
            "slice": name,
            "head_neg": float(head["neg"].mean()) if len(head) else float("nan"),
            "tail_neg": float(tail["neg"].mean()) if len(tail) else float("nan"),
            "n_head": int(len(head)),
            "n_tail": int(len(tail)),
            "n_co": int(part["company_id"].nunique()) if len(part) else 0,
        }

    ht = [
        _head_tail(train["late"].fillna(False), "late"),
        _head_tail(train["short"].fillna(False), "short_<12"),
        _head_tail(train["full24"].fillna(False), "long_24"),
        _head_tail(~train["late"].fillna(False), "ontime"),
    ]
    # g=0 months that are negative (connection hole + reconstructed red)
    g0 = pd.to_numeric(train["g_n_accounts"], errors="coerce") == 0
    g0_neg = float(train.loc[g0, "neg"].mean()) if g0.any() else float("nan")
    gpos_neg = float(train.loc[~g0, "neg"].mean()) if (~g0).any() else float("nan")
    late_head = next(r for r in ht if r["slice"] == "late")
    clusters_early = bool(
        np.isfinite(late_head["head_neg"])
        and np.isfinite(late_head["tail_neg"])
        and late_head["head_neg"] > late_head["tail_neg"] + 0.03
    )
    return {
        "age_rows": age_rows,
        "ht": ht,
        "g0_neg": g0_neg,
        "gpos_neg": gpos_neg,
        "clusters_early": clusters_early,
        "note": "monthly grid starts at first_tx — there are no pre-tx company-months",
    }


# ---------------------------------------------------------------------------
# Pass 13 — runway<1 anatomy (not a Y2/Y3 model)
# ---------------------------------------------------------------------------


def pass13_runway(store: pd.DataFrame) -> dict:
    train = train_store(store)
    last = train.sort_values("period").groupby("company_id", sort=False).last().reset_index()
    liq = pd.to_numeric(last["b_liq"], errors="coerce")
    rw = pd.to_numeric(last["b_runway"], errors="coerce")
    login = np.log1p(pd.to_numeric(last["a_in3"], errors="coerce").clip(lower=0))
    nn = rw.notna() & liq.notna()
    thin = nn & (rw < 1)
    neg = nn & (liq < 0)
    pos_thin = thin & (liq >= 0)
    ceil = nn & (rw >= 24 - 1e-9)
    floor = nn & (rw <= -6 + 1e-9)
    # implied monthly burn from runway = liq / clip(liq/out, -6, 24)
    # cannot invert at the walls. among interior: out = liq / rw
    interior = nn & (rw > -6 + 1e-6) & (rw < 24 - 1e-6) & (rw != 0)
    implied_out = (liq[interior] / rw[interior]).abs()
    rows = [
        {"group": "all_nn", "n_co": int(nn.sum()), "share": 1.0, "liq_p50": float(liq[nn].median()), "rw_p50": float(rw[nn].median())},
        {"group": "rw<1", "n_co": int(thin.sum()), "share": float(thin.sum() / nn.sum()) if nn.any() else float("nan"),
         "liq_p50": float(liq[thin].median()) if thin.any() else float("nan"),
         "rw_p50": float(rw[thin].median()) if thin.any() else float("nan")},
        {"group": "liq<0", "n_co": int(neg.sum()), "share": float(neg.sum() / nn.sum()) if nn.any() else float("nan"),
         "liq_p50": float(liq[neg].median()) if neg.any() else float("nan"),
         "rw_p50": float(rw[neg].median()) if neg.any() else float("nan")},
        {"group": "liq>=0 & rw<1", "n_co": int(pos_thin.sum()), "share": float(pos_thin.sum() / nn.sum()) if nn.any() else float("nan"),
         "liq_p50": float(liq[pos_thin].median()) if pos_thin.any() else float("nan"),
         "rw_p50": float(rw[pos_thin].median()) if pos_thin.any() else float("nan")},
        {"group": "rw=24 wall", "n_co": int(ceil.sum()), "share": float(ceil.sum() / nn.sum()) if nn.any() else float("nan"),
         "liq_p50": float(liq[ceil].median()) if ceil.any() else float("nan"),
         "rw_p50": 24.0},
        {"group": "rw=-6 wall", "n_co": int(floor.sum()), "share": float(floor.sum() / nn.sum()) if nn.any() else float("nan"),
         "liq_p50": float(liq[floor].median()) if floor.any() else float("nan"),
         "rw_p50": -6.0},
    ]
    return {
        "rows": rows,
        "n_nn": int(nn.sum()),
        "share_thin": float(thin.mean()) if nn.any() else float("nan"),
        "share_pos_thin": float(pos_thin.sum() / nn.sum()) if nn.any() else float("nan"),
        "share_neg": float(neg.sum() / nn.sum()) if nn.any() else float("nan"),
        "rho_rw_login": spearman(rw, login),
        "rho_thin_login": spearman(thin.astype(float), login),
        "implied_out_p50": float(implied_out.median()) if len(implied_out) else float("nan"),
        "implied_out_p10": float(implied_out.quantile(0.10)) if len(implied_out) else float("nan"),
        "is_size": abs(spearman(rw, login)) >= SIZE_RHO if np.isfinite(spearman(rw, login)) else False,
    }


# ---------------------------------------------------------------------------
# Pass 14 — 13 no-balance + 6 no-cash-walk
# ---------------------------------------------------------------------------


def pass14_holes(con, store: pd.DataFrame) -> dict:
    hold = load_holdout()
    train = train_store(store)
    train_ids = set(train["company_id"])
    bal = set(con.execute("SELECT DISTINCT CAST(company_id AS VARCHAR) AS company_id FROM balances").df()["company_id"].astype(str))
    cash = set(_cash_snap(con)["company_id"])
    no_bal = sorted(train_ids - bal)
    bal_no_cash = sorted((train_ids & bal) - cash)
    last = train.sort_values("period").groupby("company_id", sort=False).last()
    rows = []
    for cid, kind in [(c, "tx_no_balance") for c in no_bal] + [(c, "balance_no_cash_walk") for c in bal_no_cash]:
        if cid not in last.index:
            continue
        r = last.loc[cid]
        rows.append(
            {
                "company_id": cid,
                "kind": kind,
                "g_n_accounts": float(r["g_n_accounts"]) if pd.notna(r["g_n_accounts"]) else float("nan"),
                "b_liq": float(r["b_liq"]) if pd.notna(r["b_liq"]) else float("nan"),
                "a_in3": float(r["a_in3"]) if pd.notna(r["a_in3"]) else float("nan"),
            }
        )
    types = []
    if bal_no_cash:
        q = ",".join(f"'{c}'" for c in bal_no_cash)
        types = con.execute(
            f"""
            SELECT CAST(b.company_id AS VARCHAR) AS company_id,
                   coalesce(p."type", d."type", '(unknown)') AS ptype,
                   COUNT(*) AS n,
                   SUM(b.balance) AS sum_bal
            FROM balances b
            LEFT JOIN banking_products p ON b.product_id = p.product_id
            LEFT JOIN debt_products d ON b.product_id = d.product_id
            WHERE CAST(b.company_id AS VARCHAR) IN ({q})
            GROUP BY 1, 2
            """
        ).df().to_dict("records")
    return {
        "n_no_bal": len(no_bal),
        "n_no_cash": len(bal_no_cash),
        "ids_no_bal": no_bal,
        "ids_no_cash": bal_no_cash,
        "rows": rows,
        "types": types,
        "no_bal_liq_all_null": bool(all(pd.isna(last.loc[c, "b_liq"]) for c in no_bal if c in last.index)) if no_bal else True,
    }


# ---------------------------------------------------------------------------
# Pass 15 — last-month vs snapshot is September flows
# ---------------------------------------------------------------------------


def pass15_sep(con, store: pd.DataFrame) -> dict:
    train = train_store(store)
    last = train.loc[train["period"] == PANEL_END].copy()
    snap = _cash_snap(con)
    last = last.merge(snap, on="company_id", how="left")
    both = last["b_liq"].notna() & last["snap"].notna()
    delta = last.loc[both, "snap"] - last.loc[both, "b_liq"]
    # independent Sep cash-product flow
    sep = con.execute(
        f"""
        SELECT CAST(p.company_id AS VARCHAR) AS company_id, SUM(t.amount) AS sep_flow
        FROM transactions t
        JOIN banking_products p ON t.product_id = p.product_id
        WHERE t.product_id IN (
            SELECT b.product_id FROM balances b
            JOIN banking_products p2 ON b.product_id = p2.product_id
            WHERE p2."type" IN ({CASH_SQL}) AND b.balance IS NOT NULL
        )
          AND date_trunc('month', t."date") = DATE '2026-09-01'
        GROUP BY 1
        """
    ).df()
    sep["company_id"] = sep["company_id"].astype(str)
    last = last.merge(sep, on="company_id", how="left")
    last["sep_flow"] = last["sep_flow"].fillna(0.0)
    gap = (delta - last.loc[both, "sep_flow"]).abs()
    return {
        "n": int(both.sum()),
        "delta_p50": float(delta.abs().median()) if both.any() else float("nan"),
        "delta_p90": float(delta.abs().quantile(0.90)) if both.any() else float("nan"),
        "share_eq": float((delta.abs() < 1.0).mean()) if both.any() else float("nan"),
        "rho_last_snap": spearman(last.loc[both, "b_liq"], last.loc[both, "snap"]),
        "sep_vs_delta_max": float(gap.max()) if both.any() else float("nan"),
        "sep_vs_delta_med": float(gap.median()) if both.any() else float("nan"),
        "n_no_sep": int((last.loc[both, "sep_flow"].abs() < 1e-9).sum()) if both.any() else 0,
    }


# ---------------------------------------------------------------------------
# Pass 16 — last-month b_liq vs raw snapshot by type mix
# ---------------------------------------------------------------------------


def pass16_last_still(con, store: pd.DataFrame) -> dict:
    """Is last-value Q1 literally the still? Yes up to Sep flows (pass 15)."""
    train = train_store(store)
    last = train.sort_values("period").groupby("company_id", sort=False).last().reset_index()
    snap = _cash_snap(con)
    last = last.merge(snap, on="company_id", how="left")
    both = last["b_liq"].notna() & last["snap"].notna()
    last["rel"] = (last["b_liq"] - last["snap"]).abs() / np.maximum(last["snap"].abs(), 1.0)
    # companies whose last grid month is not 2026-08 (shouldn't happen on official grid)
    not_aug = last["period"] != PANEL_END
    return {
        "n": int(both.sum()),
        "rho": spearman(last.loc[both, "b_liq"], last.loc[both, "snap"]),
        "rel_p50": float(last.loc[both, "rel"].median()) if both.any() else float("nan"),
        "rel_p90": float(last.loc[both, "rel"].quantile(0.90)) if both.any() else float("nan"),
        "n_not_aug": int(not_aug.sum()),
        "q1_is_still": bool(
            both.any()
            and spearman(last.loc[both, "b_liq"], last.loc[both, "snap"]) >= 0.95
            and float(last.loc[both, "rel"].median()) < 0.05
        ),
    }


# ---------------------------------------------------------------------------
# Pass 17 — calendar red decline: composition vs within-company accumulation
# ---------------------------------------------------------------------------


def pass17_accum(con, store: pd.DataFrame) -> dict:
    """Head>tail red on *every* cohort. Is that late-arrival or cash piled at extract?"""
    train = train_store(store)
    trail = _trail(con, store)
    hold = load_holdout()
    trail_tr = trail.loc[~trail["company_id"].isin(hold)]
    train = train.merge(
        trail_tr[["company_id", "full24", "late", "short"]],
        on="company_id",
        how="left",
    )
    liq = pd.to_numeric(train["b_liq"], errors="coerce")
    train["neg"] = liq < 0
    first = train.sort_values("period").groupby("company_id", sort=False).first().reset_index()
    last = train.sort_values("period").groupby("company_id", sort=False).last().reset_index()
    co = first[["company_id", "b_liq", "full24", "late", "short"]].rename(columns={"b_liq": "liq_first"})
    co = co.merge(last[["company_id", "b_liq"]].rename(columns={"b_liq": "liq_last"}), on="company_id")
    co["up"] = co["liq_last"] > co["liq_first"]
    co["both"] = co["liq_first"].notna() & co["liq_last"].notna()

    def _up(part: pd.DataFrame, name: str) -> dict:
        lab = part["both"]
        return {
            "slice": name,
            "n_co": int(lab.sum()),
            "share_up": float(part.loc[lab, "up"].mean()) if lab.any() else float("nan"),
            "first_neg": float((part.loc[lab, "liq_first"] < 0).mean()) if lab.any() else float("nan"),
            "last_neg": float((part.loc[lab, "liq_last"] < 0).mean()) if lab.any() else float("nan"),
        }

    up_rows = [
        _up(co, "all"),
        _up(co.loc[co["full24"].fillna(False)], "long_24"),
        _up(co.loc[co["late"].fillna(False)], "late"),
        _up(co.loc[co["short"].fillna(False)], "short_<12"),
    ]
    # calendar of the 435 on-time / 24-grid books only
    long = train.loc[train["full24"].fillna(False)]
    cal = (
        long.groupby("period")
        .agg(n_cm=("company_id", "size"), n_neg=("neg", "sum"))
        .reset_index()
    )
    cal["share_neg"] = cal["n_neg"] / cal["n_cm"]
    cal_rows = [
        {
            "period": pd.Timestamp(r["period"]).strftime("%Y-%m"),
            "n_cm": int(r["n_cm"]),
            "n_neg": int(r["n_neg"]),
            "share_neg": float(r["share_neg"]),
        }
        for _, r in cal.iterrows()
    ]
    share0 = float(cal["share_neg"].iloc[0]) if len(cal) else float("nan")
    share1 = float(cal["share_neg"].iloc[-1]) if len(cal) else float("nan")
    declined = bool(np.isfinite(share0) and np.isfinite(share1) and share0 > share1 + 0.02)
    all_up = next(r for r in up_rows if r["slice"] == "all")
    long_co = co.loc[co["full24"].fillna(False) & co["both"]]
    n2p = int(((long_co["liq_first"] < 0) & (long_co["liq_last"] >= 0)).sum())
    p2n = int(((long_co["liq_first"] >= 0) & (long_co["liq_last"] < 0)).sum())
    n2n = int(((long_co["liq_first"] < 0) & (long_co["liq_last"] < 0)).sum())
    p2p = int(((long_co["liq_first"] >= 0) & (long_co["liq_last"] >= 0)).sum())
    return {
        "up_rows": up_rows,
        "cal_24": cal_rows,
        "long_first_neg": share0,
        "long_last_neg": share1,
        "long_declined": declined,
        "share_up_all": all_up["share_up"],
        "accumulation": bool(np.isfinite(all_up["share_up"]) and all_up["share_up"] >= 0.55),
        "n2p": n2p,
        "p2n": p2n,
        "n2n": n2n,
        "p2p": p2p,
        "n_long": int(len(long_co)),
        "net_cross": n2p - p2n,
    }


# ---------------------------------------------------------------------------
# Pass 18 — runway persist (is Q1 last-runway as sticky as last-liq?)
# ---------------------------------------------------------------------------


def pass18_runway_persist(con, store: pd.DataFrame) -> dict:
    train = train_store(store)
    trail = _trail(con, store)
    hold = load_holdout()
    trail_tr = trail.loc[~trail["company_id"].isin(hold)]
    train = train.merge(
        trail_tr[["company_id", "short", "full24"]],
        on="company_id",
        how="left",
    )

    def _rho(part, col, lag):
        pairs = _lag_pairs(part, col, lag).dropna()
        return {
            "n_pairs": int(len(pairs)),
            "rho": spearman(pairs["a"], pairs["b"]) if len(pairs) else float("nan"),
        }

    rows = []
    for name, part in (
        ("all", train),
        ("short_<12", train.loc[train["short"].fillna(False)]),
        ("long_24", train.loc[train["full24"].fillna(False)]),
    ):
        for col in ("b_liq", "b_runway"):
            r = _rho(part, col, 3)
            r["slice"] = name
            r["col"] = col
            rows.append(r)
    liq = next(r for r in rows if r["slice"] == "all" and r["col"] == "b_liq")
    rw = next(r for r in rows if r["slice"] == "all" and r["col"] == "b_runway")
    return {
        "rows": rows,
        "rho_liq": liq["rho"],
        "rho_rw": rw["rho"],
        "rw_stickier": bool(np.isfinite(rw["rho"]) and np.isfinite(liq["rho"]) and rw["rho"] > liq["rho"] + 0.02),
    }


# ---------------------------------------------------------------------------
# Pass 19 — the 13 no-balance books: what txs do they have?
# ---------------------------------------------------------------------------


def pass19_hole_txs(con, store: pd.DataFrame) -> dict:
    hold = load_holdout()
    train_ids = set(train_store(store)["company_id"])
    bal = set(con.execute("SELECT DISTINCT CAST(company_id AS VARCHAR) AS company_id FROM balances").df()["company_id"].astype(str))
    no_bal = sorted(train_ids - bal)
    if not no_bal:
        return {"n": 0, "rows": [], "by_type": []}
    q = ",".join(f"'{c}'" for c in no_bal)
    txs = con.execute(
        f"""
        SELECT CAST(t.company_id AS VARCHAR) AS company_id,
               coalesce(p."type", d."type", '(no product)') AS ptype,
               COUNT(*) AS n_tx,
               SUM(t.amount) AS sum_amt,
               MIN(t."date") AS first_d,
               MAX(t."date") AS last_d
        FROM transactions t
        LEFT JOIN banking_products p ON t.product_id = p.product_id
        LEFT JOIN debt_products d ON t.product_id = d.product_id
        WHERE CAST(t.company_id AS VARCHAR) IN ({q})
        GROUP BY 1, 2
        ORDER BY 1, n_tx DESC
        """
    ).df()
    by_type = (
        txs.groupby("ptype", as_index=False)
        .agg(n_tx=("n_tx", "sum"), n_co=("company_id", "nunique"), sum_amt=("sum_amt", "sum"))
        .sort_values("n_tx", ascending=False)
        .to_dict("records")
    )
    # any cash-type txs without a balance? those would be a Family B miss
    cash_tx = txs.loc[txs["ptype"].isin(CASH_TYPES)]
    return {
        "n": len(no_bal),
        "rows": txs.to_dict("records"),
        "by_type": by_type,
        "n_cash_tx_rows": int(cash_tx["n_tx"].sum()) if len(cash_tx) else 0,
        "n_co_cash_tx": int(cash_tx["company_id"].nunique()) if len(cash_tx) else 0,
        "b_miss": int(cash_tx["n_tx"].sum()) > 0 if len(cash_tx) else False,
    }


# ---------------------------------------------------------------------------
# Pass 20 — are the 13's checking books dead before extract?
# ---------------------------------------------------------------------------


def pass20_hole_timing(con, store: pd.DataFrame) -> dict:
    hold = load_holdout()
    train_ids = set(train_store(store)["company_id"])
    bal = set(con.execute("SELECT DISTINCT CAST(company_id AS VARCHAR) AS company_id FROM balances").df()["company_id"].astype(str))
    no_bal = sorted(train_ids - bal)
    q = ",".join(f"'{c}'" for c in no_bal)
    raw = con.execute(
        f"""
        SELECT CAST(t.company_id AS VARCHAR) AS company_id,
               t.product_id,
               coalesce(p."type", '(no product)') AS ptype,
               COUNT(*) AS n_tx,
               MIN(t."date") AS first_d,
               MAX(t."date") AS last_d,
               SUM(t.amount) AS sum_amt,
               p.created_at
        FROM transactions t
        LEFT JOIN banking_products p ON t.product_id = p.product_id
        WHERE CAST(t.company_id AS VARCHAR) IN ({q})
          AND p."type" IN ({CASH_SQL})
        GROUP BY 1, 2, 3, 8
        """
    ).df()
    raw["first_d"] = pd.to_datetime(raw["first_d"])
    raw["last_d"] = pd.to_datetime(raw["last_d"])
    raw["created_at"] = pd.to_datetime(raw["created_at"])
    raw["last_before_extract"] = raw["last_d"] < EXTRACT
    raw["quiet_30"] = raw["last_d"] < EXTRACT - pd.Timedelta(days=30)
    raw["quiet_90"] = raw["last_d"] < EXTRACT - pd.Timedelta(days=90)
    co = raw.groupby("company_id").agg(
        n_prod=("product_id", "nunique"),
        n_tx=("n_tx", "sum"),
        last_d=("last_d", "max"),
        first_d=("first_d", "min"),
        sum_amt=("sum_amt", "sum"),
    ).reset_index()
    co["quiet_30"] = co["last_d"] < EXTRACT - pd.Timedelta(days=30)
    co["quiet_90"] = co["last_d"] < EXTRACT - pd.Timedelta(days=90)
    # products in banking_products for these cos that have a cash type but no balance
    prods = con.execute(
        f"""
        SELECT CAST(p.company_id AS VARCHAR) AS company_id,
               p.product_id,
               p."type" AS ptype,
               p.created_at,
               CASE WHEN b.product_id IS NULL THEN 0 ELSE 1 END AS has_bal
        FROM banking_products p
        LEFT JOIN balances b ON p.product_id = b.product_id
        WHERE CAST(p.company_id AS VARCHAR) IN ({q})
          AND p."type" IN ({CASH_SQL})
        """
    ).df()
    n_cash_prod = int(len(prods))
    n_no_bal_prod = int((prods["has_bal"] == 0).sum()) if len(prods) else 0
    co_rows = []
    for _, r in co.iterrows():
        co_rows.append(
            {
                "company_id": r["company_id"],
                "n_prod": int(r["n_prod"]),
                "n_tx": int(r["n_tx"]),
                "first_d": pd.Timestamp(r["first_d"]).strftime("%Y-%m-%d"),
                "last_d": pd.Timestamp(r["last_d"]).strftime("%Y-%m-%d"),
                "quiet_30": bool(r["quiet_30"]),
                "sum_amt": float(r["sum_amt"]),
            }
        )
    return {
        "n_co": len(no_bal),
        "n_cash_prod": n_cash_prod,
        "n_no_bal_prod": n_no_bal_prod,
        "n_quiet_30": int(co["quiet_30"].sum()) if len(co) else 0,
        "n_quiet_90": int(co["quiet_90"].sum()) if len(co) else 0,
        "n_live": int((~co["quiet_30"]).sum()) if len(co) else 0,
        "last_p50": pd.Timestamp(co["last_d"].median()).strftime("%Y-%m-%d") if len(co) else "",
        "rows": co_rows,
        "all_products_unphotographed": n_cash_prod == n_no_bal_prod and n_cash_prod > 0,
        "n_last_0720": int((co["last_d"].dt.normalize() == pd.Timestamp("2026-07-20")).sum()) if len(co) else 0,
        "live_id": str(co.loc[~co["quiet_30"], "company_id"].iloc[0]) if len(co) and (~co["quiet_30"]).any() else "",
    }


# ---------------------------------------------------------------------------
# Pass 21 — does the path approach the still? ( |liq-snap| shrinks )
# ---------------------------------------------------------------------------


def pass21_approach(con, store: pd.DataFrame) -> dict:
    """|b_liq - snap| = |flows after t|. Share of steps that move closer to the still."""
    train = train_store(store)
    snap = _cash_snap(con)
    train = train.merge(snap[["company_id", "snap"]], on="company_id", how="left")
    trail = _trail(con, store)
    hold = load_holdout()
    trail_tr = trail.loc[~trail["company_id"].isin(hold)]
    train = train.merge(trail_tr[["company_id", "short", "full24"]], on="company_id", how="left")
    train["dist"] = (pd.to_numeric(train["b_liq"], errors="coerce") - train["snap"]).abs()
    train = train.sort_values(["company_id", "period"])
    train["dist_next"] = train.groupby("company_id")["dist"].shift(-1)
    step = train.loc[train["dist"].notna() & train["dist_next"].notna()].copy()
    step["closer"] = step["dist_next"] < step["dist"] - 1e-6
    step["farther"] = step["dist_next"] > step["dist"] + 1e-6
    rows = []
    for name, part in (
        ("all", step),
        ("short_<12", step.loc[step["short"].fillna(False)]),
        ("long_24", step.loc[step["full24"].fillna(False)]),
    ):
        rows.append(
            {
                "slice": name,
                "n_steps": int(len(part)),
                "share_closer": float(part["closer"].mean()) if len(part) else float("nan"),
                "share_farther": float(part["farther"].mean()) if len(part) else float("nan"),
                "n_co": int(part["company_id"].nunique()) if len(part) else 0,
            }
        )
    # company: share of its own steps that approach
    co_share = step.groupby("company_id")["closer"].mean()
    return {
        "rows": rows,
        "share_closer": rows[0]["share_closer"],
        "share_farther": rows[0]["share_farther"],
        "co_p50": float(co_share.median()) if len(co_share) else float("nan"),
        "co_share_majority_closer": float((co_share > 0.5).mean()) if len(co_share) else float("nan"),
        "mostly_approach": bool(rows[0]["share_closer"] >= 0.65),
    }


# ---------------------------------------------------------------------------
# Pass 22 — custom checking in the walk (1.6B)
# ---------------------------------------------------------------------------


def pass22_custom(con) -> dict:
    hold = load_holdout()
    raw = con.execute(
        f"""
        SELECT CAST(p.company_id AS VARCHAR) AS company_id,
               COUNT(*) AS n_prod,
               SUM(b.balance) AS snap,
               SUM(ABS(b.balance)) AS snap_abs,
               SUM(CASE WHEN lower(coalesce(p.service,''))='custom'
                          OR p.bank_name ILIKE '%customer-defined%'
                          OR p.bank_name ILIKE 'Other%' THEN b.balance ELSE 0 END) AS custom_bal,
               SUM(CASE WHEN lower(coalesce(p.service,''))='custom'
                          OR p.bank_name ILIKE '%customer-defined%'
                          OR p.bank_name ILIKE 'Other%' THEN ABS(b.balance) ELSE 0 END) AS custom_abs
        FROM balances b
        JOIN banking_products p ON b.product_id = p.product_id
        WHERE p."type" IN ({CASH_SQL}) AND b.balance IS NOT NULL
        GROUP BY 1
        """
    ).df()
    raw["company_id"] = raw["company_id"].astype(str)
    tr = raw.loc[~raw["company_id"].isin(hold)].copy()
    assert_no_holdout(tr["company_id"])
    tot = float(tr["snap_abs"].sum())
    cust = float(tr["custom_abs"].sum())
    any_c = tr["custom_abs"] > 0
    all_c = (tr["custom_abs"] >= tr["snap_abs"] - 1e-6) & (tr["snap_abs"] > 0)
    return {
        "n_co": int(len(tr)),
        "n_any": int(any_c.sum()),
        "n_all": int(all_c.sum()),
        "share_abs": cust / tot if tot else float("nan"),
        "custom_abs": cust,
        "walk_abs": tot,
        "top": tr.loc[any_c].nlargest(8, "custom_abs")[["company_id", "custom_abs", "snap_abs"]].to_dict("records"),
    }


# ---------------------------------------------------------------------------
# Pass 23 — company-median vs last (is the typical month the still?)
# ---------------------------------------------------------------------------


def pass23_median(con, store: pd.DataFrame) -> dict:
    train = train_store(store)
    snap = _cash_snap(con)
    g = train.groupby("company_id")["b_liq"]
    co = pd.DataFrame(
        {
            "liq_med": g.median(),
            "liq_first": g.first(),
            "liq_last": g.last(),
        }
    ).reset_index()
    co = co.merge(snap[["company_id", "snap"]], on="company_id", how="left")
    lab = co["liq_med"].notna() & co["liq_last"].notna()
    return {
        "n": int(lab.sum()),
        "rho_med_last": spearman(co.loc[lab, "liq_med"], co.loc[lab, "liq_last"]),
        "rho_med_snap": spearman(co.loc[lab, "liq_med"], co.loc[lab, "snap"]),
        "rho_med_first": spearman(co.loc[lab, "liq_med"], co.loc[lab, "liq_first"]),
        "rho_first_last": spearman(co.loc[lab, "liq_first"], co.loc[lab, "liq_last"]),
        "rel_med_last_p50": float(
            ((co.loc[lab, "liq_med"] - co.loc[lab, "liq_last"]).abs() / np.maximum(co.loc[lab, "liq_last"].abs(), 1.0)).median()
        ) if lab.any() else float("nan"),
        "typical_is_still": bool(
            lab.any()
            and spearman(co.loc[lab, "liq_med"], co.loc[lab, "liq_last"]) >= 0.90
        ),
    }


# ---------------------------------------------------------------------------
# Pass 24 — negative-month stickiness (description; not a Y2 model)
# ---------------------------------------------------------------------------


def pass24_neg_stick(store: pd.DataFrame) -> dict:
    train = train_store(store)
    train = train.sort_values(["company_id", "period"])
    neg = pd.to_numeric(train["b_liq"], errors="coerce") < 0
    train = train.assign(neg=neg)
    # run lengths of consecutive negatives
    runs = []
    for cid, g in train.groupby("company_id", sort=False):
        n = g["neg"].to_numpy()
        if not n.any():
            continue
        length = 0
        for v in n:
            if v:
                length += 1
            elif length:
                runs.append(length)
                length = 0
        if length:
            runs.append(length)
    rs = np.array(runs, dtype=float) if runs else np.array([])
    ever = train.groupby("company_id")["neg"].any()
    pairs = _lag_pairs(train.assign(negf=train["neg"].astype(float)), "negf", 1).dropna()
    # P(neg_{t+1}|neg_t)
    p11 = float(pairs.loc[pairs["a"] == 1, "b"].mean()) if (pairs["a"] == 1).any() else float("nan")
    p01 = float(pairs.loc[pairs["a"] == 0, "b"].mean()) if (pairs["a"] == 0).any() else float("nan")
    return {
        "n_runs": int(len(rs)),
        "run_p50": float(np.median(rs)) if len(rs) else float("nan"),
        "run_p90": float(np.quantile(rs, 0.90)) if len(rs) else float("nan"),
        "share_run1": float((rs == 1).mean()) if len(rs) else float("nan"),
        "share_run_ge3": float((rs >= 3).mean()) if len(rs) else float("nan"),
        "ever_neg_share": float(ever.mean()),
        "n_ever": int(ever.sum()),
        "p_stay_neg": p11,
        "p_enter_neg": p01,
        "isolated_blips": bool(len(rs) and float((rs == 1).mean()) >= 0.50),
    }


# ---------------------------------------------------------------------------
# Pass 25 — September flow size vs snapshot
# ---------------------------------------------------------------------------


def pass25_sep_size(con, store: pd.DataFrame) -> dict:
    train = train_store(store)
    last = train.loc[train["period"] == PANEL_END].copy()
    snap = _cash_snap(con)
    last = last.merge(snap, on="company_id", how="left")
    sep = con.execute(
        f"""
        SELECT CAST(p.company_id AS VARCHAR) AS company_id, SUM(t.amount) AS sep_flow
        FROM transactions t
        JOIN banking_products p ON t.product_id = p.product_id
        WHERE t.product_id IN (
            SELECT b.product_id FROM balances b
            JOIN banking_products p2 ON b.product_id = p2.product_id
            WHERE p2."type" IN ({CASH_SQL}) AND b.balance IS NOT NULL
        )
          AND date_trunc('month', t."date") = DATE '2026-09-01'
        GROUP BY 1
        """
    ).df()
    sep["company_id"] = sep["company_id"].astype(str)
    last = last.merge(sep, on="company_id", how="left")
    last["sep_flow"] = last["sep_flow"].fillna(0.0)
    both = last["snap"].notna() & last["b_liq"].notna()
    last["sep_rel"] = last["sep_flow"].abs() / np.maximum(last["snap"].abs(), 1.0)
    rel = last.loc[both, "sep_rel"]
    fat = both & (last["sep_rel"] > 0.10)
    thin = both & (last["sep_rel"] <= 0.10)
    return {
        "n": int(both.sum()),
        "sep_abs_p50": float(last.loc[both, "sep_flow"].abs().median()) if both.any() else float("nan"),
        "rel_p50": float(rel.median()) if both.any() else float("nan"),
        "rel_p90": float(rel.quantile(0.90)) if both.any() else float("nan"),
        "share_gt10pct": float((rel > 0.10).mean()) if both.any() else float("nan"),
        "n_zero": int((last.loc[both, "sep_flow"].abs() < 1e-9).sum()) if both.any() else 0,
        "rho_all": spearman(last.loc[both, "b_liq"], last.loc[both, "snap"]),
        "rho_fat": spearman(last.loc[fat, "b_liq"], last.loc[fat, "snap"]),
        "rho_thin": spearman(last.loc[thin, "b_liq"], last.loc[thin, "snap"]),
        "n_fat": int(fat.sum()),
        "rho_sep_login": spearman(
            last.loc[both, "sep_rel"],
            np.log1p(pd.to_numeric(last.loc[both, "a_in3"], errors="coerce").clip(lower=0)),
        ),
    }


# ---------------------------------------------------------------------------
# Pass 26 — unique pile companies + last-tx cluster on the 13
# ---------------------------------------------------------------------------


def pass26_pile_unique_and_hole_cluster(con, store: pd.DataFrame) -> dict:
    """Pass-3 n_co sums across types. Unique pile membership + is 2026-07-20 a hole stamp?"""
    hold = load_holdout()
    train_ids = set(train_store(store)["company_id"])
    pile = con.execute(
        f"""
        SELECT
          CASE
            WHEN p.product_id IS NOT NULL AND p."type" IN ({CASH_SQL}) THEN 'walk'
            WHEN p.product_id IS NOT NULL THEN 'bank_excluded'
            WHEN d.product_id IS NOT NULL THEN 'debt_orphan'
            ELSE 'unknown_orphan'
          END AS pile,
          CAST(b.company_id AS VARCHAR) AS company_id
        FROM balances b
        LEFT JOIN banking_products p ON b.product_id = p.product_id
        LEFT JOIN debt_products d ON b.product_id = d.product_id
        """
    ).df()
    pile["company_id"] = pile["company_id"].astype(str)
    pile = pile.drop_duplicates()
    sets = {name: set(pile.loc[pile["pile"] == name, "company_id"]) for name in pile["pile"].unique()}
    walk = sets.get("walk", set())
    bank_ex = sets.get("bank_excluded", set())
    debt = sets.get("debt_orphan", set())
    unk = sets.get("unknown_orphan", set())
    any_excl = bank_ex | debt | unk
    pile_rows = [
        {"pile": "walk", "n_co": len(walk), "n_train": len(walk & train_ids)},
        {"pile": "bank_excluded", "n_co": len(bank_ex), "n_train": len(bank_ex & train_ids)},
        {"pile": "debt_orphan", "n_co": len(debt), "n_train": len(debt & train_ids)},
        {"pile": "unknown_orphan", "n_co": len(unk), "n_train": len(unk & train_ids)},
        {"pile": "any_excluded", "n_co": len(any_excl), "n_train": len(any_excl & train_ids)},
        {"pile": "walk ∩ debt_orphan", "n_co": len(walk & debt), "n_train": len(walk & debt & train_ids)},
        {"pile": "walk-only (no excluded row)", "n_co": len(walk - any_excl), "n_train": len((walk - any_excl) & train_ids)},
        {"pile": "excluded-only (no walk)", "n_co": len(any_excl - walk), "n_train": len((any_excl - walk) & train_ids)},
    ]

    bal = set(
        con.execute("SELECT DISTINCT CAST(company_id AS VARCHAR) AS company_id FROM balances")
        .df()["company_id"]
        .astype(str)
    )
    no_bal = sorted(train_ids - bal)
    last_all = con.execute(
        f"""
        SELECT CAST(p.company_id AS VARCHAR) AS company_id, MAX(t."date") AS last_d
        FROM transactions t
        JOIN banking_products p ON t.product_id = p.product_id
        WHERE p."type" IN ({CASH_SQL})
        GROUP BY 1
        """
    ).df()
    last_all["company_id"] = last_all["company_id"].astype(str)
    last_all["last_d"] = pd.to_datetime(last_all["last_d"])
    last_all["train"] = last_all["company_id"].isin(train_ids)
    last_all["hole"] = last_all["company_id"].isin(no_bal)
    last_all["photo"] = last_all["company_id"].isin(walk & train_ids)
    cut = pd.Timestamp("2026-07-20")

    def _last_stats(mask: pd.Series, name: str) -> dict:
        part = last_all.loc[mask]
        if part.empty:
            return {
                "slice": name, "n_co": 0, "last_p50": "",
                "n_0720": 0, "share_0720": float("nan"),
                "n_jul": 0, "share_jul": float("nan"),
                "n_aug": 0, "share_aug": float("nan"),
            }
        last_d = part["last_d"]
        jul = (last_d.dt.year == 2026) & (last_d.dt.month == 7)
        aug = (last_d.dt.year == 2026) & (last_d.dt.month == 8)
        d0720 = last_d.dt.normalize() == cut
        return {
            "slice": name,
            "n_co": int(len(part)),
            "last_p50": pd.Timestamp(last_d.median()).strftime("%Y-%m-%d"),
            "n_0720": int(d0720.sum()),
            "share_0720": float(d0720.mean()),
            "n_jul": int(jul.sum()),
            "share_jul": float(jul.mean()),
            "n_aug": int(aug.sum()),
            "share_aug": float(aug.mean()),
        }

    last_rows = [
        _last_stats(last_all["hole"], "13 holes"),
        _last_stats(last_all["photo"], "train photographed walk"),
        _last_stats(last_all["train"], "train any cash-tx"),
    ]
    hole_last = last_all.loc[last_all["hole"]].copy()
    hole_hist = (
        hole_last.assign(last_day=hole_last["last_d"].dt.strftime("%Y-%m-%d"))
        .groupby("last_day", as_index=False)
        .agg(n_co=("company_id", "nunique"))
        .sort_values("n_co", ascending=False)
        .to_dict("records")
        if len(hole_last)
        else []
    )
    banks = pd.DataFrame()
    if no_bal:
        q = ",".join(f"'{c}'" for c in no_bal)
        banks = con.execute(
            f"""
            SELECT coalesce(p.bank_name, '(none)') AS bank_name,
                   p."type" AS ptype,
                   COUNT(DISTINCT CAST(p.company_id AS VARCHAR)) AS n_co,
                   COUNT(*) AS n_prod
            FROM banking_products p
            WHERE CAST(p.company_id AS VARCHAR) IN ({q})
              AND p."type" IN ({CASH_SQL})
            GROUP BY 1, 2
            ORDER BY n_co DESC, n_prod DESC
            """
        ).df()
    hole_0720 = float((hole_last["last_d"].dt.normalize() == cut).mean()) if len(hole_last) else float("nan")
    photo_0720 = float(
        (last_all.loc[last_all["photo"], "last_d"].dt.normalize() == cut).mean()
    ) if last_all["photo"].any() else float("nan")
    cluster_is_hole = bool(
        np.isfinite(hole_0720) and np.isfinite(photo_0720) and hole_0720 >= 0.30 and hole_0720 > photo_0720 + 0.15
    )
    santander = "Santander (Corporate UK)"
    bank_last = pd.DataFrame()
    n_0720_sant = 0
    n_sant = 0
    n_0720 = int((hole_last["last_d"].dt.normalize() == cut).sum()) if len(hole_last) else 0
    if no_bal:
        q = ",".join(f"'{c}'" for c in no_bal)
        bank_last = con.execute(
            f"""
            SELECT CAST(p.company_id AS VARCHAR) AS company_id,
                   coalesce(p.bank_name, '(none)') AS bank_name,
                   MAX(t."date") AS last_d
            FROM transactions t
            JOIN banking_products p ON t.product_id = p.product_id
            WHERE CAST(p.company_id AS VARCHAR) IN ({q})
              AND p."type" IN ({CASH_SQL})
            GROUP BY 1, 2
            """
        ).df()
        if len(bank_last):
            bank_last["company_id"] = bank_last["company_id"].astype(str)
            bank_last["last_d"] = pd.to_datetime(bank_last["last_d"])
            d0720 = bank_last["last_d"].dt.normalize() == cut
            is_sant = bank_last["bank_name"] == santander
            n_sant = int(bank_last.loc[is_sant, "company_id"].nunique())
            n_0720_sant = int(bank_last.loc[d0720 & is_sant, "company_id"].nunique())
    santander_is_cluster = bool(n_0720 > 0 and n_0720_sant == n_0720 and n_sant == n_0720)
    return {
        "pile_rows": pile_rows,
        "n_walk_unique": len(walk),
        "n_walk_train": len(walk & train_ids),
        "n_walk_debt": len(walk & debt),
        "n_excl_only_train": len((any_excl - walk) & train_ids),
        "last_rows": last_rows,
        "hole_hist": hole_hist,
        "banks": banks.to_dict("records") if len(banks) else [],
        "n_banks": int(banks["bank_name"].nunique()) if len(banks) else 0,
        "hole_0720": hole_0720,
        "photo_0720": photo_0720,
        "cluster_is_hole": cluster_is_hole,
        "n_hole": len(no_bal),
        "n_0720": n_0720,
        "n_sant": n_sant,
        "n_0720_sant": n_0720_sant,
        "santander_is_cluster": santander_is_cluster,
    }


def make_png(p2: dict) -> bool:
    if not HAS_MPL or p2["resid"] is None or len(p2["resid"]) == 0:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    r = p2["resid"]
    clip = np.clip(r, np.nanpercentile(r, 0.5), np.nanpercentile(r, 99.5)) if len(r) > 20 else r
    axes[0].hist(clip, bins=40, color="#1f4e79", edgecolor="white")
    axes[0].axvline(0, color="#c45911", lw=1.2)
    axes[0].set_title("Identity residual (recovered − snapshot)")
    axes[0].set_xlabel("euro")
    axes[0].set_ylabel("train company-months")
    # last-month-ish scatter is all months; subsample
    rec, snap = p2["rec"], p2["snap_v"]
    if len(rec) > 4000:
        rng = np.random.default_rng(FOLD_SEED)
        idx = rng.choice(len(rec), size=4000, replace=False)
        rec, snap = rec[idx], snap[idx]
    axes[1].scatter(snap, rec, s=6, alpha=0.25, c="#1f4e79", linewidths=0)
    lo = np.nanmin([np.nanmin(snap), np.nanmin(rec)])
    hi = np.nanmax([np.nanmax(snap), np.nanmax(rec)])
    axes[1].plot([lo, hi], [lo, hi], color="#c45911", lw=1)
    axes[1].set_xlabel("cash snapshot (checking+saving+tpv)")
    axes[1].set_ylabel("b_liq + flows after t")
    axes[1].set_title("Recovered vs snapshot (train sample)")
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    return True


def write_md(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, png_ok: bool) -> None:
    lines = []
    a = lines.append
    a("# Family B — balances still / reconstruction QA")
    a("")
    a(f"Generated `{_utc_ts()}` UTC by `python -m analysis.evaluate.balances_b_qa`.")
    a("Holdout 72 (seed 20260918) is **coverage only**. Rates, Spearman,")
    a("and PARK/CLOSE/KEEP are train. No parquet rewrite. No new GBM. No 0–100.")
    a("Does not invent a Y from `b_liq` / `b_runway`. Does not use Family B as X for Y2/Y3.")
    a("Family G is a connection clock — this module does not redo it.")
    a("")
    a("## Headline")
    a("")
    a(
        f"- Raw `balances`: **{p1['n_rows']:,} rows / {p1['n_co']:,} companies**. "
        f"As-of 2026-09-01: {p1['n_asof']:,} ({_pp(p1['share_asof'])}). "
        f"Not the still: {p1['n_not_asof']} rows on {p1['n_dates'] - 1} late-August dates "
        f"(dictionary: closest prior day). Null balance sentinels: {p1['n_null_bal']}."
    )
    a(
        f"- Identity `b_liq + after-flows = snapshot`: train n={p2['n_cm']:,} CM / {p2['n_co']:,} companies. "
        f"median |residual| = **{p2['median_abs']:.4g}** euro; "
        f"|resid|>1€ {_pp(p2['share_gt1'])}; |resid|>1% of |snap| {_pp(p2['share_gt1pct'])}. "
        + (
            "**CLOSE the reconstruction** — the walk is an identity."
            if p2["tiny"]
            else "Residuals are not tiny — inspect before CLOSEing the walk."
        )
    )
    a(
        f"- Walk types = checking + saving + tpv. Excluded |snapshot| share **{_pp(p3['excl_share_abs'])}** "
        f"(debt orphans in `balances`, plus card / investment / leftover bank types). "
        f"Custom *checking* is **in** the walk ({p3['custom_walk_n']:,} products, "
        f"|bal| {p3['custom_walk_abs']:.4g}) — Family B filters type, not service."
    )
    a(
        f"- First-month `g_n_accounts=0` is {_pp(p4['first_g0_share'])} of train companies "
        f"(G 63.7% replica). On those first months `b_liq` is finite {_pp(p4['first_g0_liq_finite'])}, "
        f"null {_pp(p4['first_g0_liq_null'])}, exactly 0 {_pp(p4['first_g0_liq_zero'])}. "
        "The walk ignores `created_at` — unconnected ≠ missing cash."
    )
    a(
        f"- Spearman(`b_liq_t`, `b_liq_t+3`) train = **{p5['all_lag3']['rho']:.3f}** "
        f"(n={p5['all_lag3']['n_pairs']:,}"
        f"{'; CONFIRM ~0.85' if p5['confirm_085'] else '; does not match the 0.85 quote'}). "
        f"Short <12: {p5['short_lag3']['rho']:.3f} (n={p5['short_lag3']['n_pairs']:,}). "
        f"24-month books: {p5['long_lag3']['rho']:.3f} (n={p5['long_lag3']['n_pairs']:,}). "
        + (
            "Short books stick harder to the still — leak is real on left-trunc."
            if p5["leak_short"]
            else "Short books are not *more* persistent than long books."
        )
        + (
            " Long books still have a path (first-vs-snap weaker than last-vs-snap / range)."
            if p5["long_real"]
            else " Long books also look like the still shining through."
        )
    )
    a(
        f"- Last-value Q1 (train last month): `b_liq` p50={p6['last_liq']['p50']:.4g}, "
        f"share<0 {_pp(p6['last_liq']['share_neg'])}; `b_runway` p50={p6['last_rw']['p50']:.3f}, "
        f"share<1 {_pp(p6['last_rw']['share_lt1'])}. Not a Y2/Y3 model."
    )
    a(
        f"- Holdout coverage (descriptive): {p7['n_bal']}/{p7['n_hold']} have a balance row "
        f"({_pp(p7['share_bal'])}); cash-walk snapshot {p7['n_cash']}/{p7['n_hold']}."
    )
    a(
        f"- Zombie cash products (0 txs, train): {p8['n_zombie']:,}/{p8['n_prod']:,} "
        f"({_pp(p8['share_prod'])}) holding {_pp(p8['share_bal'])} of |cash snapshot|."
    )
    a(
        f"- `b_bal_vol` vs log1p(a_in3) ρ={p9['rho_login3']:.3f} "
        f"({'SIZE' if p9['is_size'] else 'not SIZE'}; |ρ|≥{SIZE_RHO:.2f})."
    )
    a(
        f"- Late-book pre-first-tx months: **none** (grid starts at first tx). "
        f"Late on-book neg {_pp(p10['late_on_neg'])} vs on-time {_pp(p10['ontime_neg'])}. "
        f"First-month neg late {_pp(p10['first_neg_late'])} vs on-time {_pp(p10['first_neg_ontime'])} — not a left-trunc red pile."
    )
    a(
        f"- Short persist after dropping age<3: {p11['short_mat']['rho']:.3f} vs long {p11['long_mat']['rho']:.3f}. "
        f"Later arrivals first-vs-snap {p11['rho_late_first']:.3f} vs on-time {p11['rho_ontime_first']:.3f} "
        f"(still-leak ranking: {'yes' if p11['leak_rank'] else 'no'})."
    )
    a(
        f"- Last-month vs cash snapshot ρ={p16['rho']:.3f}; |snap−b_liq| is Sep flows "
        f"(median {p15['delta_p50']:.4g}, identity max {p15['sep_vs_delta_max']:.4g}). "
        f"Last>first on {_pp(p17['share_up_all'])} of train; 24-month neg share "
        f"{_pp(p17['long_first_neg'])} → {_pp(p17['long_last_neg'])} inside the same 435 "
        f"(neg→pos {p17['n2p']} / pos→neg {p17['p2n']}). "
        f"13 train companies have checking txs ({p19['n_cash_tx_rows']:,}) and no snapshot "
        f"({p20['n_live']} still live in the last 30d, {p20['live_id'] or '—'}) — null `b_liq`, do not invent a 0-still. "
        f"Steps closer to the still {_pp(p21['share_closer'])}. "
        f"ρ(company-median, last)={p23['rho_med_last']:.3f} "
        f"({'typical month ≈ still' if p23['typical_is_still'] else 'typical month ≠ still'}). "
        f"Unique walk companies {p26['n_walk_unique']:,} (train {p26['n_walk_train']:,}); "
        f"2026-07-20 last cash-tx on {_pp(p26['hole_0720'])} of the 13 vs {_pp(p26['photo_0720'])} of photographed walk"
        f"{' — hole stamp' if p26['cluster_is_hole'] else ' — not a hole-only stamp'}."
    )
    a("")
    a("## Brief questions")
    a("")
    a("1. **Who is healthy?** — last-value `b_liq` / `b_runway` is the night Q1 *description*. It is the 2026-09 still walked backward, not a forecast.")
    a("2. **Who is improving?** — last>first `b_liq` is a coin flip (~49%). The *negative tail* shrinks toward extract (11.5%→5.1% on the same 435). Description only; not a new Y. Y1 liq path stays PARK.")
    a("3. **Who is turning?** — do not invent a Y from B (circular 16h death). Y2 already *is* the B path.")
    a("4. **Dip vs fall?** — reconstruction identity, not a dip detector.")
    a("5. **Why did it change?** — excluded snapshot cash is mostly debt orphans + investment, not a health why.")
    a("6. **Months earlier?** — short books persist *less* (0.73 vs 0.86), not more. First-vs-snap is ~0.61 on long and short. The still is the *last* month; it does not leak a constant ranking backward. Lead time on liquidity is the walk through later flows, not a photograph of 2024.")
    a("")
    a("## 1. Raw `balances`")
    a("")
    a("| item | n |")
    a("| --- | ---: |")
    a(f"| rows | {p1['n_rows']:,} |")
    a(f"| companies | {p1['n_co']:,} (train {p1['n_train_co']} / holdout {p1['n_hold_co']}; universe {p1['n_all_co']}) |")
    a(f"| product_id | {p1['n_prod']:,} |")
    a(f"| distinct dates | {p1['n_dates']} ({p1['mind'].date()} → {p1['maxd'].date()}) |")
    a(f"| date = 2026-09-01 | {p1['n_asof']:,} ({_pp(p1['share_asof'])}) |")
    a(f"| not 2026-09-01 | {p1['n_not_asof']} |")
    a(f"| null balance (sentinel) | {p1['n_null_bal']} |")
    a(f"| product_known=false | {p1['n_unknown']} |")
    a("")
    a("Not everything is 2026-09-01. The dictionary already allowed the closest prior day. Those 16 late-August rows are real extract jitter, not a second still.")
    a("")
    a("Dates:")
    a("")
    date_rows = [{"date": pd.Timestamp(r.date).strftime("%Y-%m-%d"), "n": int(r.n)} for r in p1["dates"].itertuples()]
    a(_md_table(date_rows, [("date", "date"), ("n", "n")]))
    a("")
    a("Product mix (banking type, or debt type when the balance row has no `banking_products` join):")
    a("")
    a(_md_table(
        p1["types"].to_dict("records"),
        [
            ("bucket", "bucket"),
            ("ptype", "type"),
            ("n_rows", "rows"),
            ("n_co", "companies"),
            ("sum_bal", "sum balance"),
            ("sum_abs", "sum |balance|"),
            ("n_null", "null bal"),
        ],
    ))
    a("")
    a("## 2. Identity")
    a("")
    a("Formula: `end_bal_t = snapshot − sum(checking/saving/tpv flows with month > t)`. Snapshot-day txs sit in 2026-09. Train company-months with both `b_liq` and a cash snapshot:")
    a("")
    a("| check | n | median \\|err\\| | p90 | max | share >1€ | share >1% \\|snap\\| |")
    a("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    a(
        f"| store `b_liq` + after vs snap | {p2['n_cm']:,} | {p2['median_abs']:.4g} | {p2['p90_abs']:.4g} | {p2['max_abs']:.4g} | "
        f"{_pp(p2['share_gt1'])} | {_pp(p2['share_gt1pct'])} |"
    )
    a(
        f"| store vs `liquidity._reconstruct` | {p2['store_vs_rebuild']['n']:,} | {p2['store_vs_rebuild']['median_abs']:.4g} | "
        f"{p2['store_vs_rebuild']['p90']:.4g} | {p2['store_vs_rebuild']['max_abs']:.4g} | — | — |"
    )
    a(
        f"| store vs product-level walk | {p2['store_vs_product']['n']:,} | {p2['store_vs_product']['median_abs']:.4g} | "
        f"{p2['store_vs_product']['p90']:.4g} | {p2['store_vs_product']['max_abs']:.4g} | — | — |"
    )
    a(
        f"| last-month recovered vs snap | {p2['n_last']:,} | {p2['last_resid']['median_abs']:.4g} | "
        f"{p2['last_resid']['p90']:.4g} | {p2['last_resid']['max_abs']:.4g} | — | — |"
    )
    a("")
    if p2["tiny"]:
        a("**CLOSE.** Residuals are numerical noise. Do not rewrite `liquidity.py`. Do not treat the walk as a feature that *learns* history — it is the still minus later booked flows.")
    else:
        a("Residuals are large enough to look at. If store vs `_reconstruct` is tiny but store+after vs snap is not, the after-flow join is the bug in this QA, not Family B.")
    a("")
    a("PARK any idea of using the snapshot as a Y. That would score the extract photograph.")
    a("")
    a("## 3. Walk vs excluded")
    a("")
    a(
        f"Walk |cash| = {p3['walk_abs']:.4g} ({_pp(1 - p3['excl_share_abs'])} of all |balances|). "
        f"Excluded |cash| share {_pp(p3['excl_share_abs'])}. Positive-cash share excluded {_pp(p3['excl_pos_share'])}."
    )
    a(f"Train companies with a balance row but no cash-walk snapshot: {p3['n_train_no_walk']:,} / {p3['n_train_bal']:,}.")
    a("")
    a("Piles (n_co here **sums across types** — a company with checking+saving is counted twice. Unique membership is §26):")
    a("")
    a(_md_table(
        p3["pile"].to_dict("records"),
        [
            ("pile", "pile"),
            ("n_rows", "rows"),
            ("n_co", "companies (type-sum)"),
            ("sum_bal", "sum bal"),
            ("sum_abs", "sum |bal|"),
            ("pos_bal", "positive bal"),
        ],
    ))
    a("")
    a("By type:")
    a("")
    a(_md_table(
        p3["named"].to_dict("records"),
        [
            ("ptype", "type"),
            ("in_walk", "in walk?"),
            ("n_rows", "rows"),
            ("n_co", "companies"),
            ("sum_bal", "sum bal"),
            ("sum_abs", "sum |bal|"),
        ],
    ))
    a("")
    a("Debt product outstanding sitting in `balances` is the bulk of the excluded pile (loan / LOC / confirming…). Card and investment are the named *bank* exclusions. TPV is in the walk but the extract stock is all-zero. Custom checking is walked.")
    a("")
    a("## 4. Coverage (train)")
    a("")
    a("| set | n companies |")
    a("| --- | ---: |")
    a(f"| train on the monthly grid | {p4['n_train_co']:,} |")
    a(f"| with a `balances` row | {p4['n_bal']:,} |")
    a(f"| with any tx | {p4['n_tx']:,} |")
    a(f"| with a cash-walk snapshot | {p4['n_cash']:,} |")
    a(f"| ever `g_n_accounts>0` | {p4['n_ever_g']:,} |")
    a(f"| tx but no balance | {p4['n_tx_not_bal']:,} |")
    a(f"| balance but no cash-walk | {p4['n_bal_not_cash']:,} |")
    a(f"| cash-walk but never G-connected | {p4['n_cash_no_g']:,} |")
    a("")
    a(
        f"First-month unconnected (`g_n_accounts=0`): {p4['first_g0_n']:,} / {p4['n_train_co']:,} = {_pp(p4['first_g0_share'])}. "
        f"`b_liq` on those rows: finite {_pp(p4['first_g0_liq_finite'])}, null {_pp(p4['first_g0_liq_null'])}, "
        f"zero {_pp(p4['first_g0_liq_zero'])}, negative {_pp(p4['first_g0_liq_neg'])}, p50={p4['first_g0_liq_p50']:.4g}."
    )
    a(
        f"All first-month `b_liq` null share {_pp(p4['first_all_liq_null'])}. "
        f"CM with `g_n_accounts=0`: {p4['g0_cm']:,}; `b_liq` finite {_pp(p4['g0_liq_finite'])}, "
        f"null {_pp(p4['g0_liq_null'])}, zero {_pp(p4['g0_liq_zero'])}, p50={p4['g0_liq_p50']:.4g}."
    )
    a("")
    a("The 63.7% hole is G's connection clock, not a missing `b_liq`. Unconnected first months usually already have a reconstructed cash path.")
    a("")
    a("## 5. Persistence — short vs long")
    a("")
    a(
        f"Train companies: short <12 first-tx trail {p5['n_short_co']:,}; 24-month grid {p5['n_long_co']:,}; "
        f"late first tx {p5['n_late_co']:,}."
    )
    a("")
    a(_md_table(
        p5["lag_rows"],
        [("slice", "slice"), ("lag", "lag"), ("n_pairs", "n pairs"), ("n_co", "n companies"), ("rho", "Spearman")],
    ))
    a("")
    a("Company first / last reconstructed cash vs the extract snapshot:")
    a("")
    a(_md_table(
        p5["co_rows"],
        [
            ("slice", "slice"),
            ("n_co", "n"),
            ("rho_first_snap", "ρ first vs snap"),
            ("rho_last_snap", "ρ last vs snap"),
            ("rho_first_last", "ρ first vs last"),
            ("rel_range_p50", "p50 range/|snap|"),
            ("rel_first_p50", "p50 |first−snap|/|snap|"),
            ("rel_last_p50", "p50 |last−snap|/|snap|"),
        ],
    ))
    a("")
    if p5["still_shining"] and not p5["long_real"]:
        a("**PARK last-value as a trail.** Persistence is the still shining through, including on 24-month books.")
    elif p5["long_real"] and p5["confirm_085"]:
        a("**KEEP last-value Q1 as a description of the extract still** (last-vs-snap ρ≈0.97). It is **not** 'everyone is the still all the way back': first-vs-snap is ~0.61 on both short and long books. Short 3-month persist is *lower* (0.73), not higher. Do not promote a 3-month liq forecast (already PARK).")
    else:
        a("Mixed: quote the table. Last-value remains the honest Q1 *level*; do not claim a 24-month trajectory from a 3-month Spearman alone.")
    a("")
    a("## 6. Q1 last-value honesty (description only)")
    a("")
    a("Last grid month per train company — the month-24 still the brief warned about. Not a model. Not a Y.")
    a("")
    a("| series | n | null | p10 | p25 | p50 | p75 | p90 | share <0 |")
    a("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    liq, rw = p6["last_liq"], p6["last_rw"]
    a(
        f"| last `b_liq` | {liq.get('n_nn', liq['n'])} | {_pp(liq['share_null'])} | "
        f"{liq['p10']:.4g} | {liq['p25']:.4g} | {liq['p50']:.4g} | {liq['p75']:.4g} | "
        f"{liq['p90']:.4g} | {_pp(liq['share_neg'])} |"
    )
    a(
        f"| last `b_runway` | {rw.get('n_nn', rw['n'])} | {_pp(rw['share_null'])} | "
        f"{rw['p10']:.3f} | {rw['p25']:.3f} | {rw['p50']:.3f} | {rw['p75']:.3f} | "
        f"{rw['p90']:.3f} | {_pp(rw['share_neg'])} |"
    )
    a("")
    a(
        f"Last-month runway < 1: {_pp(rw['share_lt1'])}. "
        f"Last-month stressed (`b_liq<0` or `b_runway<1`) among non-null liq: {_pp(p6['last_stressed_nn'])}. "
        f"Runway sitting on the clip walls: floor −6 {_pp(p6['rw_floor_share'])}, ceiling 24 {_pp(p6['rw_ceil_share'])}."
    )
    a(
        f"Company-month (all train): `b_liq` < 0 {_pp(p6['cm']['liq']['share_neg'])}; "
        f"runway < 1 {_pp(p6['cm']['share_rw_lt1'])}; stressed {_pp(p6['cm']['share_stressed'])}."
    )
    a("")
    a("Do **not** predict Y2/Y3 with Family B. Y2 is already 2-of-3 negative reconstructed months; Y3 is stressed-only recovery. Using B as X is the contract leak.")
    a("")
    a("## 7. Holdout coverage (count only)")
    a("")
    a("| item | n |")
    a("| --- | ---: |")
    a(f"| holdout companies | {p7['n_hold']} |")
    a(f"| with a balance row | {p7['n_bal']} ({_pp(p7['share_bal'])}) |")
    a(f"| with a cash-walk snapshot | {p7['n_cash']} ({_pp(p7['share_cash'])}) |")
    a(f"| holdout CM | {p7['n_cm']:,} |")
    a(f"| `b_liq` non-null CM | {_pp(p7['liq_cm_cov'])} |")
    a(f"| companies with any `b_liq` | {p7['n_co_any_liq']} |")
    a("")
    a("## 8. Zombie snapshot cash")
    a("")
    a(
        f"Train cash-walk products with **zero** txs in the extract: "
        f"{p8['n_zombie']:,} / {p8['n_prod']:,} = {_pp(p8['share_prod'])}. "
        f"|zombie| / |cash snap| = {_pp(p8['share_bal'])} "
        f"(zombie signed sum {p8['zombie_bal']:.4g} vs cash {p8['cash_bal']:.4g}). "
        f"Companies with any zombie cash product: {p8['n_co_any']:,}. "
        f"Companies whose *entire* cash snapshot is zombie: {p8['n_co_all']:,} "
        f"(signed sum {p8['all_z_bal']:.4g})."
    )
    a("")
    a(_md_table(
        p8["by_type"],
        [
            ("ptype", "type"),
            ("n_prod", "products"),
            ("n_zombie", "zombies"),
            ("share", "zombie share"),
            ("zombie_bal", "zombie bal"),
            ("zombie_abs", "zombie |bal|"),
            ("all_bal", "all bal"),
        ],
    ))
    a("")
    a("Zombie cash is a still of unused accounts, not a trail. Small |share| means it does not move company-sum `b_liq` much — but a dead checking account can hide inside a live sum.")
    a("")
    a("## 9. Is `b_bal_vol` SIZE?")
    a("")
    a(
        f"Train CM coverage {_pp(p9['cov_cm'])} (6-month roll). "
        f"p50={p9['p50']:.3f}, p90={p9['p90']:.3f}, share>1 {_pp(p9['share_gt1'])}. "
        f"Spearman vs log1p(a_in3)={p9['rho_login3']:.3f}; vs a_in3={p9['rho_ain3']:.3f}; "
        f"vs `b_liq`={p9['rho_liq']:.3f}; vs |b_liq|={p9['rho_abs_liq']:.3f}. "
        f"Last-month vs log size {p9['last_rho_login3']:.3f}."
    )
    a(
        "**SIZE** — do not read vol as a health residual."
        if p9["is_size"]
        else "**not SIZE** — vol is not company scale. Still not a Y (it is a roll of the same still-walked path)."
    )
    a("")
    a("## 10. Negatives on late books (left-trunc + still)")
    a("")
    a(
        f"Train CM `b_liq<0`: {p10['cm_neg_n']:,} ({_pp(p10['cm_neg_share'])}). "
        f"Late-book months *before* first tx: {p10['late_pre_n']:,} CM / {p10['n_pre_co']:,} companies; "
        f"neg {_pp(p10['late_pre_neg'])}; constant reconstructed path {_pp(p10['share_const_pre'])}. "
        f"Late on-book neg {_pp(p10['late_on_neg'])}; on-time-book neg {_pp(p10['ontime_neg'])}."
    )
    a(
        f"First-month neg: late {_pp(p10['first_neg_late'])} vs on-time {_pp(p10['first_neg_ontime'])}. "
        f"Late last-month neg {_pp(p10['last_neg_late'])}."
    )
    a("")
    if len(p10["cal"]):
        cal_rows = []
        for _, r in p10["cal"].iterrows():
            cal_rows.append(
                {
                    "period": pd.Timestamp(r["period"]).strftime("%Y-%m"),
                    "n_cm": int(r["n_cm"]),
                    "n_neg": int(r["n_neg"]),
                    "share_neg": float(r["share_neg"]),
                    "n_late": int(r["n_late"]),
                }
            )
        a("Negative share by calendar month (train):")
        a("")
        a(_md_table(
            cal_rows,
            [
                ("period", "month"),
                ("n_cm", "n CM"),
                ("n_neg", "n neg"),
                ("share_neg", "share neg"),
                ("n_late", "n late-book CM"),
            ],
        ))
        a("")
    if p10.get("grid_starts_at_first_tx"):
        a("The official monthly grid **starts at first tx** — there are no pre-connection company-months to hold a constant opening photograph. Left-truncation was already applied by the grid, not by Family B. See §12 for early-on-book (age 0–2) negatives.")
    elif p10["clusters"]:
        a("First-month negatives are richer on late arrivals. That is left-truncation of a still, not a 24-month stress trail. Do not treat early-grid `b_below_0` on a 6-month book as Q6 lead time.")
    else:
        a("Late first months are not clearly more negative than on-time first months.")
    a("")
    a("## 11. Why short persist is lower")
    a("")
    a(
        f"Still-leak predicted short books *more* like the snapshot. Observed Spearman t vs t+3 is **lower** on short "
        f"({p11['short_all']['rho']:.3f}) than on 24-month books. After dropping the first 3 on-book months: "
        f"short {p11['short_mat']['rho']:.3f} (n={p11['short_mat']['n_pairs']:,}) vs long {p11['long_mat']['rho']:.3f}. "
        + (
            "Birth months explain the short-book dip."
            if p11["birth_explains"]
            else "Dropping birth months does not close the short-vs-long gap — short books are noisier, not flatter."
        )
    )
    a(
        f"First-month vs snap ranking: late arrivals {p11['rho_late_first']:.3f} vs on-time {p11['rho_ontime_first']:.3f} "
        f"({'later arrivals rank *more* like the still' if p11['leak_rank'] else 'later arrivals do **not** rank more like the still'}). "
        f"First→second month |Δliq|/|liq| p50: short {p11['jump_p50_short']:.3f} vs not-short {p11['jump_p50_long']:.3f}."
    )
    a("")
    a(_md_table(
        p11["buck_rows"],
        [("slice", "trail"), ("kind", "months used"), ("n_pairs", "n pairs"), ("n_co", "n companies"), ("rho", "Spearman t,t+3")],
    ))
    a("")
    a("First-month vs snapshot by first-tx calendar (train):")
    a("")
    a(_md_table(
        p11["cal"],
        [
            ("first_tx", "first tx"),
            ("n_co", "n"),
            ("rho_first_snap", "ρ first vs snap"),
            ("rel_p50", "p50 |first−snap|/|snap|"),
            ("neg_share", "first-month neg"),
        ],
    ))
    a("")
    a("## 12. Early-on-book negatives (age, not pre-tx)")
    a("")
    a(p12["note"] + ".")
    a(
        f" `g_n_accounts=0` months that are `b_liq<0`: {_pp(p12['g0_neg'])} vs connected months {_pp(p12['gpos_neg'])}."
    )
    a("")
    a(_md_table(
        p12["age_rows"],
        [("book", "book"), ("age", "age"), ("n_cm", "n CM"), ("n_co", "n companies"), ("share_neg", "share neg")],
    ))
    a("")
    a("First 3 vs last 3 months within company:")
    a("")
    a(_md_table(
        p12["ht"],
        [
            ("slice", "slice"),
            ("n_co", "n"),
            ("n_head", "n head CM"),
            ("head_neg", "head neg"),
            ("n_tail", "n tail CM"),
            ("tail_neg", "tail neg"),
        ],
    ))
    a("")
    if p12["clusters_early"]:
        a("Late books are more negative in their first three months than their last three — **but so are 24-month on-time books**. That is cash accumulating toward a healthier extract still, not late-arrival left-trunc (late age0 8.9% is *below* on-time age0 11.5%). **CLOSE** early-grid `b_below_0` as Q6. See §17.")
    else:
        a("Late-book head months are not clearly redder than their own tail. Combined with §10 (late first-month 8.9% vs on-time 11.5%), negatives do **not** pile up at the start of late books.")
    a("")
    a("## 13. Runway < 1 anatomy")
    a("")
    a(
        f"Last-month train, non-null: runway<1 {_pp(p13['share_thin'])} of which `liq>=0` {_pp(p13['share_pos_thin'])} "
        f"and `liq<0` {_pp(p13['share_neg'])}. Spearman(runway, log1p a_in3)={p13['rho_rw_login']:.3f} "
        f"({'SIZE' if p13['is_size'] else 'not SIZE'}). "
        f"Implied monthly burn among interior (unclipped) runways: p10={p13['implied_out_p10']:.4g}, p50={p13['implied_out_p50']:.4g}."
    )
    a("")
    a(_md_table(
        p13["rows"],
        [("group", "group"), ("n_co", "n"), ("share", "share of nn"), ("liq_p50", "liq p50"), ("rw_p50", "runway p50")],
    ))
    a("")
    a("Half the last-month book is `runway<1` mostly because **positive cash over a fat 3-month burn**, not because 49% are below zero (only ~5% are). The 24-month clip wall is the other tail (idle / tiny out3). This is a description of the still, not a license to invent a runway Y.")
    a("")
    a("## 14. Coverage holes")
    a("")
    a(
        f"Tx but no `balances` row: {p14['n_no_bal']} train companies "
        f"(`b_liq` all-null on last month: {p14['no_bal_liq_all_null']}). "
        f"Balance row but no cash-walk snapshot: {p14['n_no_cash']}."
    )
    a("")
    if p14["rows"]:
        a(_md_table(
            p14["rows"],
            [
                ("company_id", "company"),
                ("kind", "kind"),
                ("g_n_accounts", "last g_n_accounts"),
                ("b_liq", "last b_liq"),
                ("a_in3", "last a_in3"),
            ],
        ))
        a("")
    if p14["types"]:
        a("No-cash-walk companies — what sits in `balances`:")
        a("")
        a(_md_table(
            p14["types"],
            [("company_id", "company"), ("ptype", "type"), ("n", "n"), ("sum_bal", "sum bal")],
        ))
        a("")
    a("## 15. Last month vs snapshot = September flows")
    a("")
    a(
        f"Train last-grid-month (2026-08) with a cash snap: {p15['n']:,}. "
        f"median |snap − b_liq| = {p15['delta_p50']:.4g}; share |diff|<1€ {_pp(p15['share_eq'])}. "
        f"ρ(last, snap)={p15['rho_last_snap']:.3f}. "
        f"| (snap−b_liq) − Sep cash flow | median {p15['sep_vs_delta_med']:.4g}, max {p15['sep_vs_delta_max']:.4g}. "
        f"Companies with no Sep cash-product flow: {p15['n_no_sep']:,}."
    )
    a("Last-value Q1 **is** the still, minus extract-month txs. That is why last-value wins Y1 liq and why a snapshot Y is PARK.")
    a("")
    a("## 16. Last-value = the still")
    a("")
    a(
        f"Company last month vs cash snapshot: n={p16['n']:,}, ρ={p16['rho']:.3f}, "
        f"p50 |liq−snap|/|snap|={p16['rel_p50']:.4g}, p90={p16['rel_p90']:.4g}. "
        f"Last month ≠ 2026-08: {p16['n_not_aug']}. "
        + (
            "**Yes — last-value Q1 is the extract still.**"
            if p16["q1_is_still"]
            else "Last-value is close to the still but not a photocopy on every company (Sep flows / non-August last month)."
        )
    )
    a("")
    a("## 17. Accumulation toward the still (not late-arrival red)")
    a("")
    a(
        f"Share of train companies with last `b_liq` > first `b_liq`: {_pp(p17['share_up_all'])}. "
        f"On the 24-month cohort, calendar neg share { _pp(p17['long_first_neg'])} → {_pp(p17['long_last_neg'])} "
        f"({'declines inside the same 435 companies' if p17['long_declined'] else 'does not decline within the 435'}). "
        + (
            "The panel accumulated cash toward a richer extract photograph. Head>tail red is Q2 *description* of that walk, not a late-book artifact."
            if p17["accumulation"]
            else "Last>first is a coin flip — the red decline is a *tail* recovery, not a richer book."
        )
        + (
            f" 24-month crossings: neg→pos {p17['n2p']}, pos→neg {p17['p2n']}, stayed neg {p17['n2n']}, stayed pos {p17['p2p']} (n={p17['n_long']})."
        )
    )
    a("")
    a(_md_table(
        p17["up_rows"],
        [
            ("slice", "slice"),
            ("n_co", "n"),
            ("share_up", "share last>first"),
            ("first_neg", "first-month neg"),
            ("last_neg", "last-month neg"),
        ],
    ))
    a("")
    a("Neg share on the **same** 24-month train companies (composition held fixed):")
    a("")
    a(_md_table(
        p17["cal_24"],
        [("period", "month"), ("n_cm", "n CM"), ("n_neg", "n neg"), ("share_neg", "share neg")],
    ))
    a("")
    a("## 18. Runway persist vs liq persist")
    a("")
    a(
        f"Spearman t vs t+3: `b_liq` {p18['rho_liq']:.3f} vs `b_runway` {p18['rho_rw']:.3f}. "
        + (
            "Runway is stickier than the euro stock."
            if p18["rw_stickier"]
            else "Runway is **not** stickier than `b_liq` — last-value Q1 should quote the stock (or both), not treat runway as a smoother path."
        )
    )
    a("")
    a(_md_table(
        p18["rows"],
        [("slice", "slice"), ("col", "series"), ("n_pairs", "n pairs"), ("rho", "Spearman t,t+3")],
    ))
    a("")
    a("## 19. The 13 with txs and no `balances` row")
    a("")
    a(
        f"Cash-type txs on those 13: {p19['n_cash_tx_rows']:,} txs / {p19['n_co_cash_tx']} companies. "
        + (
            "Family B **misses** cash-product history that has no extract snapshot — those `b_liq` stay null. Not a liquidity.py rewrite (no snapshot to walk from)."
            if p19["b_miss"]
            else "No checking/saving/tpv txs — their activity sits on non-walk types. Null `b_liq` is correct."
        )
    )
    a("")
    if p19["by_type"]:
        a(_md_table(
            p19["by_type"],
            [("ptype", "type"), ("n_tx", "n tx"), ("n_co", "n companies"), ("sum_amt", "sum amount")],
        ))
        a("")
    a("## 20. Timing of the 13 unphotographed checking books")
    a("")
    a(
        f"Cash products on the 13: {p20['n_cash_prod']} (unphotographed {p20['n_no_bal_prod']}). "
        f"Last cash-tx ≥30d before extract: {p20['n_quiet_30']}/{p20['n_co']}; "
        f"≥90d quiet: {p20['n_quiet_90']}; still live in the last 30d: {p20['n_live']}. "
        f"Median last cash-tx {p20['last_p50']}."
    )
    a("")
    if p20["rows"]:
        a(_md_table(
            p20["rows"],
            [
                ("company_id", "company"),
                ("n_prod", "cash products"),
                ("n_tx", "n tx"),
                ("first_d", "first tx"),
                ("last_d", "last tx"),
                ("quiet_30", "quiet ≥30d"),
                ("sum_amt", "sum amount"),
            ],
        ))
        a("")
    if p20["n_live"] > 0:
        a("Some of the 13 still transact in the last 30 days and have no photograph. That is an extract hole, not a closed account. Still **do not** invent a 0-snapshot in `liquidity.py`.")
    else:
        a("All 13 went quiet ≥30d before extract. Closed / unphotographed is the same for a backward walk: no still, no `b_liq`.")
    a("")
    a("## 21. Does the path approach the still?")
    a("")
    a(
        f"|b_liq − snap| is |remaining after-flows|. Share of month-to-month steps that get *closer* to the snapshot: "
        f"{_pp(p21['share_closer'])} (farther {_pp(p21['share_farther'])}). "
        f"Company-level p50 of that share {p21['co_p50']:.3f}; majority-approach companies {_pp(p21['co_share_majority_closer'])}."
    )
    a("")
    a(_md_table(
        p21["rows"],
        [
            ("slice", "slice"),
            ("n_co", "n"),
            ("n_steps", "n steps"),
            ("share_closer", "closer"),
            ("share_farther", "farther"),
        ],
    ))
    a("")
    if p21["mostly_approach"]:
        a("Most steps zoom toward the photograph. The *shape* of the path is the still leaking backward even though first-vs-snap ranking is only 0.61. Last-value Q1 is still the honest *level*.")
    else:
        a("The walk oscillates — remaining flow sums do not shrink every month. That is a real cash path, not a monotone zoom into the still.")
    a("")
    a("## 22. Custom checking inside the walk")
    a("")
    a(
        f"Train cash-walk |snapshot|: custom service {p22['n_any']:,} companies / {p22['n_all']:,} all-custom, "
        f"|share| {_pp(p22['share_abs'])} ({p22['custom_abs']:.4g} of {p22['walk_abs']:.4g}). "
        "Family B already walks these (type filter only). Not a hole. Not a G redo."
    )
    a("")
    if p22["top"]:
        a("Largest custom cash snapshots (train):")
        a("")
        a(_md_table(
            p22["top"],
            [("company_id", "company"), ("custom_abs", "custom |bal|"), ("snap_abs", "cash |bal|")],
        ))
        a("")
    a("## 23. Company-median vs last (is the typical month the still?)")
    a("")
    a(
        f"Train companies with a cash path n={p23['n']:,}. "
        f"ρ(median, last)={p23['rho_med_last']:.3f}; ρ(median, snap)={p23['rho_med_snap']:.3f}; "
        f"ρ(median, first)={p23['rho_med_first']:.3f}; ρ(first, last)={p23['rho_first_last']:.3f}. "
        f"p50 |median−last|/|last|={p23['rel_med_last_p50']:.3f}."
    )
    if p23["typical_is_still"]:
        a("The typical month already ranks like last-value. Q1 last-value is not only the final frame — the book's center of mass is the still.")
    else:
        a("The typical month is **not** the still (ρ(median,last) < 0.90). Last-value Q1 is the extract photograph; the rest of the book is a different ranking. That is a real trail, not everyone being the 2026-09 still.")
    a("")
    a("## 24. Negative-month stickiness (not a Y2 model)")
    a("")
    a(
        f"Train companies ever `b_liq<0`: {p24['n_ever']:,} ({_pp(p24['ever_neg_share'])}). "
        f"{p24['n_runs']:,} negative runs; length p50={p24['run_p50']:.0f}, p90={p24['run_p90']:.0f}; "
        f"share of runs that are a single month {_pp(p24['share_run1'])}; share ≥3 months {_pp(p24['share_run_ge3'])}. "
        f"P(neg next | neg now)={p24['p_stay_neg']:.3f}; P(enter | ok)={p24['p_enter_neg']:.3f}."
    )
    a(
        "Most red months are isolated blips — Y2's 2-of-3 is the clustered tail, not the typical dip. Do not build another Y from this."
        if p24["isolated_blips"]
        else "Negative months tend to come in streaks. That is already Y2's 2-of-3; do not invent a cousin."
    )
    a("")
    a("## 25. September flow vs snapshot")
    a("")
    a(
        f"Train last-month companies with a cash snap: {p25['n']:,}. "
        f"median |Sep cash flow|={p25['sep_abs_p50']:.4g}; "
        f"p50 |Sep|/|snap|={p25['rel_p50']:.3f}; p90={p25['rel_p90']:.3f}; "
        f"share >10% of |snap| {_pp(p25['share_gt10pct'])} (n={p25['n_fat']:,}); zero Sep flow {p25['n_zero']:,}. "
        f"ρ(last, snap) all={p25['rho_all']:.3f}; fat-Sep {p25['rho_fat']:.3f}; thin-Sep {p25['rho_thin']:.3f}. "
        f"|Sep|/|snap| vs log size ρ={p25['rho_sep_login']:.3f} "
        f"({'SIZE' if abs(p25['rho_sep_login']) >= SIZE_RHO else 'not SIZE'})."
    )
    a("Last-value differs from the still by a typically small extract-month flow. On the fat-Sep fifth, ranking peels (0.80 vs 1.00). Q1 last-value is the still *plus* extract-month noise, not a SIZE artifact.")
    a("")
    a("## 26. Unique pile companies + last-tx cluster on the 13")
    a("")
    a(
        f"Unique companies with a walk row: {p26['n_walk_unique']:,} (train {p26['n_walk_train']:,}). "
        f"Walk ∩ debt-orphan: {p26['n_walk_debt']:,} (same company can sit in both piles). "
        f"Train excluded-only (balance but no cash walk): {p26['n_excl_only_train']:,}."
    )
    a("")
    a(_md_table(
        p26["pile_rows"],
        [("pile", "pile"), ("n_co", "unique companies"), ("n_train", "of which train")],
    ))
    a("")
    a(
        f"Last cash-tx on **2026-07-20**: {_pp(p26['hole_0720'])} of the 13 holes vs "
        f"{_pp(p26['photo_0720'])} of train photographed walk. "
        + (
            "That date is a hole stamp — not the typical photographed last-tx. Still do not invent a 0-snapshot."
            if p26["cluster_is_hole"]
            else "That date is not hole-only (photographed books land there too). Not a second still; not a rewrite."
        )
        + f" Distinct banks on the 13 cash products: {p26['n_banks']}. "
        f"Santander (Corporate UK) on {p26['n_sant']} of the 13; "
        f"{p26['n_0720_sant']}/{p26['n_0720']} of the 2026-07-20 last-tx books are that bank"
        f"{' — the cluster is one bank extract' if p26['santander_is_cluster'] else ''}."
    )
    a("")
    if p26["last_rows"]:
        a(_md_table(
            p26["last_rows"],
            [
                ("slice", "slice"),
                ("n_co", "n"),
                ("last_p50", "median last cash-tx"),
                ("n_0720", "n on 2026-07-20"),
                ("share_0720", "share 07-20"),
                ("n_jul", "n in 2026-07"),
                ("share_jul", "share Jul"),
                ("n_aug", "n in 2026-08"),
                ("share_aug", "share Aug"),
            ],
        ))
        a("")
    if p26["hole_hist"]:
        a("Last cash-tx dates on the 13:")
        a("")
        a(_md_table(p26["hole_hist"], [("last_day", "last cash-tx"), ("n_co", "n")]))
        a("")
    if p26["banks"]:
        a("Cash-product banks on the 13 (not a G redo):")
        a("")
        a(_md_table(
            p26["banks"],
            [("bank_name", "bank"), ("ptype", "type"), ("n_co", "n companies"), ("n_prod", "n products")],
        ))
        a("")
    a("Do not rewrite `liquidity.py` around a 2026-07-20 cutoff. Extract hole, not a walk identity bug.")
    a("")
    a("## PARK / CLOSE / KEEP")
    a("")
    a("| object | decision | why |")
    a("| --- | --- | --- |")
    if p2["tiny"]:
        a("| reconstruction walk | **CLOSE** | identity; median |resid| is noise; do not rewrite `liquidity.py` |")
    else:
        a("| reconstruction walk | **inspect** | residuals not tiny — see §2 |")
    a("| snapshot as a Y | **PARK** | would score the 2026-09 photograph |")
    if p5["long_real"] and p5["confirm_085"]:
        a("| last-value `b_liq` / `b_runway` as Q1 description | **KEEP** | last-vs-snap ≈ 0.97 (thin-Sep = 1.00; fat-Sep fifth 0.80); first-vs-snap ≈ 0.61; persist 0.85 is a real path |")
    elif p5["still_shining"] and not p5["long_real"]:
        a("| last-value as Q1 *trail* | **PARK** | persist is the still leaking backward, including on long books |")
    else:
        a("| last-value as Q1 description | **KEEP thin** | level is the honest still; do not claim a trajectory |")
    a("| last-value as a 3-month forecast (Y1 liq) | **PARK** (CONFIRM) | already lost to last-value on CV OOF |")
    a("| Family B as X for Y2 / Y3 | **forbidden** | contract; Y2/Y3 *are* the B path |")
    a("| invent a Y from `b_liq` / `b_runway` / neg-run length | **PARK** | circular 16h death; Y2 already is the streak |")
    a("| zombie cash as a health flag | **PARK** | unused-account still; small |share| |")
    a(
        "| `b_bal_vol` as X | "
        + ("**CAUTION SIZE**" if p9["is_size"] else "**not SIZE**; still a roll of the walk")
        + " | same still-walked path |"
    )
    a("| early-grid negatives on late books as Q6 | **CLOSE** | late age0 is *less* red than on-time; head>tail red is accumulation on every cohort |")
    a("| snapshot / last-value as a *forecast* Y | **PARK** | last month *is* the still minus Sep flows |")
    a("| rewrite `liquidity.py` for the 13 nulls | **no** | they have checking txs but no extract still; walking from 0 would invent a snapshot |")
    a("| 2026-07-20 last-tx as a second still / cutoff Y | **PARK** | hole timing, not a health signal |")
    a("| redo Family G | **no** | connection clock already closed |")
    a("")
    if png_ok:
        a("## Plot")
        a("")
        a("- `analysis/outputs/balances_b_residual.png` — identity residual hist + recovered vs snapshot (train sample).")
        a("")
    a("## Closed in this module")
    a("")
    a("- Raw still: almost all 2026-09-01; 16 closest-prior-day rows.")
    a("- Identity of the walk: measured on train company-months.")
    a("- Excluded snapshot cash: debt orphans + named bank types.")
    a("- 63.7% first-month G hole is not a null `b_liq`.")
    a("- Spearman t vs t+3 on short vs 24-month books.")
    a("- Q1 last-value distribution; no B→Y2/Y3 model.")
    a("- Zombies / vol-SIZE / left-trunc negatives.")
    a("- Short persist is lower, not higher; first-vs-snap is not the still.")
    a("- Runway<1 is fat burn over positive cash, not 49% below zero.")
    a("- Last-value minus snapshot = September cash-product flows.")
    a("- Head>tail red is a shrinking *negative tail* on the 24-month cohort (last>first is a coin flip).")
    a("- Runway persist vs liq persist; 13 no-balance checking books (no still → no walk).")
    a("- Approach-to-still rate; custom checking already inside the walk.")
    a("- Typical month ≠ still (median vs last); neg-run lengths; Sep flow size.")
    a("- Unique pile n_co (pass-3 type-sum overcount); 2026-07-20 last-tx vs photographed books.")
    a("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_registry(p1, p2, p3, p4, p5, p7, p8, p9, p11, p15, p16, p17, p18, p19, p20, p21, p23, p24, p25, p26) -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    ts = _utc_ts()
    rows = [
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "B", "y": "-", "model": "balances_b_qa",
            "split": "train", "metric": "n_balance_rows",
            "value": _fmt(p1["n_rows"]), "coverage": _fmt(p1["share_asof"]),
            "notes": f"cos={p1['n_co']} asof={p1['n_asof']} not_asof={p1['n_not_asof']} dates={p1['n_dates']}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "B", "y": "-", "model": "balances_b_qa",
            "split": "train", "metric": "identity_median_abs_resid",
            "value": _fmt(p2["median_abs"]), "coverage": _fmt(p2["n_cm"] / p2["n_train_cm"] if p2["n_train_cm"] else float("nan")),
            "notes": f"share_gt1e={p2['share_gt1']:.4f} share_gt1pct={p2['share_gt1pct']:.4f} tiny={p2['tiny']} CLOSE_WALK={p2['tiny']}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "B", "y": "-", "model": "balances_b_qa",
            "split": "train", "metric": "identity_share_gt1euro",
            "value": _fmt(p2["share_gt1"]), "coverage": _fmt(p2["n_cm"] / p2["n_train_cm"] if p2["n_train_cm"] else float("nan")),
            "notes": f"n_gt1={p2['n_gt1']} n_gt1pct={p2['n_gt1pct']} store_vs_rebuild_max={p2['store_vs_rebuild']['max_abs']}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "B", "y": "-", "model": "balances_b_qa",
            "split": "train", "metric": "excl_snapshot_abs_share",
            "value": _fmt(p3["excl_share_abs"]), "coverage": "1.0000",
            "notes": f"walk_abs={p3['walk_abs']:.6g} debt_orphans_in_balances; custom_walk_n={p3['custom_walk_n']}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "B", "y": "-", "model": "balances_b_qa",
            "split": "train", "metric": "first_g0_bliq_finite_share",
            "value": _fmt(p4["first_g0_liq_finite"]), "coverage": _fmt(p4["first_g0_share"]),
            "notes": f"first_g0={p4['first_g0_n']} null={p4['first_g0_liq_null']:.3f} zero={p4['first_g0_liq_zero']:.3f} unconnected_ne_missing",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "B", "y": "-", "model": "balances_b_qa",
            "split": "train", "metric": "spearman_bliq_t_t3",
            "value": _fmt(p5["all_lag3"]["rho"]),
            "coverage": _fmt(p5["all_lag3"]["n_pairs"] / 21157 if p5["all_lag3"]["n_pairs"] else float("nan")),
            "notes": f"confirm_085={p5['confirm_085']} short={p5['short_lag3']['rho']:.3f} long24={p5['long_lag3']['rho']:.3f} still_shining={p5['still_shining']} long_real={p5['long_real']}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "B", "y": "-", "model": "balances_b_qa",
            "split": "holdout", "metric": "share_with_balance_row",
            "value": _fmt(p7["share_bal"]), "coverage": _fmt(p7["share_bal"]),
            "notes": f"n_bal={p7['n_bal']}/{p7['n_hold']} cash_walk={p7['n_cash']} descriptive_only",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "B", "y": "-", "model": "balances_b_qa",
            "split": "train", "metric": "zombie_cash_prod_share",
            "value": _fmt(p8["share_prod"]), "coverage": _fmt(p8["share_bal"]),
            "notes": f"n_zombie={p8['n_zombie']}/{p8['n_prod']} all_zombie_cos={p8['n_co_all']}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "B", "y": "-", "model": "balances_b_qa",
            "split": "train", "metric": "b_bal_vol_rho_login3",
            "value": _fmt(p9["rho_login3"]), "coverage": _fmt(p9["cov_cm"]),
            "notes": f"is_size={p9['is_size']} rho_abs_liq={p9['rho_abs_liq']:.3f}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "B", "y": "-", "model": "balances_b_qa",
            "split": "train", "metric": "spearman_bliq_t_t3_short_mature",
            "value": _fmt(p11["short_mat"]["rho"]),
            "coverage": _fmt(p11["short_mat"]["n_pairs"] / 21157 if p11["short_mat"]["n_pairs"] else float("nan")),
            "notes": f"short_all={p11['short_all']['rho']:.3f} long_mat={p11['long_mat']['rho']:.3f} leak_rank={p11['leak_rank']} birth_explains={p11['birth_explains']}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "B", "y": "-", "model": "balances_b_qa",
            "split": "train", "metric": "last_vs_snap_spearman",
            "value": _fmt(p16["rho"]), "coverage": _fmt(p16["n"] / 1214 if p16["n"] else float("nan")),
            "notes": f"q1_is_still={p16['q1_is_still']} rel_p50={p16['rel_p50']:.4g} sep_delta_p50={p15['delta_p50']:.4g} sep_id_max={p15['sep_vs_delta_max']:.4g}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "B", "y": "-", "model": "balances_b_qa",
            "split": "train", "metric": "share_last_gt_first_bliq",
            "value": _fmt(p17["share_up_all"]),
            "coverage": "1.0000",
            "notes": f"long24_neg {p17['long_first_neg']:.3f}->{p17['long_last_neg']:.3f} n2p={p17['n2p']} p2n={p17['p2n']} accumulation={p17['accumulation']}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "B", "y": "-", "model": "balances_b_qa",
            "split": "train", "metric": "spearman_brunway_t_t3",
            "value": _fmt(p18["rho_rw"]),
            "coverage": _fmt(p18["rows"][1]["n_pairs"] / 21157 if p18["rows"] else float("nan")),
            "notes": f"liq_t3={p18['rho_liq']:.3f} rw_stickier={p18['rw_stickier']}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "B", "y": "-", "model": "balances_b_qa",
            "split": "train", "metric": "n_tx_no_balance_checking",
            "value": _fmt(p19["n_cash_tx_rows"]),
            "coverage": _fmt(p19["n"] / 1214 if p19.get("n") else float("nan")),
            "notes": f"n_co=13 live30d={p20['n_live']} quiet30={p20['n_quiet_30']} no_bal_prod={p20['n_no_bal_prod']} NO_REWRITE",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "B", "y": "-", "model": "balances_b_qa",
            "split": "train", "metric": "share_steps_closer_to_snap",
            "value": _fmt(p21["share_closer"]),
            "coverage": "1.0000",
            "notes": f"farther={p21['share_farther']:.3f} co_p50={p21['co_p50']:.3f} majority={p21['co_share_majority_closer']:.3f} mostly_approach={p21['mostly_approach']}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "B", "y": "-", "model": "balances_b_qa",
            "split": "train", "metric": "spearman_bliq_median_vs_last",
            "value": _fmt(p23["rho_med_last"]),
            "coverage": _fmt(p23["n"] / 1214 if p23["n"] else float("nan")),
            "notes": f"rho_med_snap={p23['rho_med_snap']:.3f} rho_first_last={p23['rho_first_last']:.3f} typical_is_still={p23['typical_is_still']}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "B", "y": "-", "model": "balances_b_qa",
            "split": "train", "metric": "neg_run_p50",
            "value": _fmt(p24["run_p50"]),
            "coverage": _fmt(p24["ever_neg_share"]),
            "notes": f"n_runs={p24['n_runs']} p_stay={p24['p_stay_neg']:.3f} share_ge3={p24['share_run_ge3']:.3f} isolated={p24['isolated_blips']}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "B", "y": "-", "model": "balances_b_qa",
            "split": "train", "metric": "sep_flow_rel_p50",
            "value": _fmt(p25["rel_p50"]),
            "coverage": _fmt(p25["n"] / 1214 if p25["n"] else float("nan")),
            "notes": f"gt10={p25['share_gt10pct']:.3f} n_fat={p25['n_fat']} rho_fat={p25['rho_fat']:.3f} rho_thin={p25['rho_thin']:.3f}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "B", "y": "-", "model": "balances_b_qa",
            "split": "train", "metric": "unique_walk_n_co",
            "value": _fmt(p26["n_walk_train"]), "coverage": _fmt(p26["n_walk_train"] / 1214 if p26["n_walk_train"] else float("nan")),
            "notes": f"universe_walk={p26['n_walk_unique']} walk_debt={p26['n_walk_debt']} excl_only_train={p26['n_excl_only_train']}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "B", "y": "-", "model": "balances_b_qa",
            "split": "train", "metric": "hole_last_0720_share",
            "value": _fmt(p26["hole_0720"]), "coverage": _fmt(p26["n_hole"] / 1214 if p26["n_hole"] else float("nan")),
            "notes": f"photo_0720={p26['photo_0720']:.3f} cluster_is_hole={p26['cluster_is_hole']} n_banks={p26['n_banks']} sant={p26['n_sant']} 0720_sant={p26['n_0720_sant']} sant_cluster={p26['santander_is_cluster']}",
        },
    ]
    with REGISTRY.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        seen = {
            (r.get("agent"), r.get("y"), r.get("model"), r.get("split"), r.get("metric"))
            for r in reader
        }
    fresh = []
    for r in rows:
        key = (str(r.get("agent", "")), str(r.get("y", "")), str(r.get("model", "")),
               str(r.get("split", "")), str(r.get("metric", "")))
        if key in seen:
            continue
        fresh.append(r)
        seen.add(key)
    if not fresh:
        print("registry: nothing new to append")
        return
    with REGISTRY.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        for r in fresh:
            w.writerow({k: r.get(k, "") for k in header})
    print(f"appended {len(fresh)} registry rows")


def main() -> int:
    print(f"balances_b_qa start seed={FOLD_SEED} holdout=72 agent={AGENT}")
    if not STORE.exists():
        raise FileNotFoundError(STORE)
    con = connect()
    store = load_store()

    print("pass 1 raw balances")
    p1 = pass1_raw(con)
    print(
        f"  rows={p1['n_rows']} cos={p1['n_co']} asof={p1['n_asof']} "
        f"not_asof={p1['n_not_asof']} dates={p1['n_dates']}"
    )

    print("pass 2 identity")
    p2 = pass2_identity(con, store)
    print(
        f"  n={p2['n_cm']} median_abs={p2['median_abs']:.6g} "
        f"gt1={p2['share_gt1']:.4f} gt1pct={p2['share_gt1pct']:.4f} tiny={p2['tiny']}"
    )

    print("pass 3 walk vs excluded")
    p3 = pass3_types(con)
    print(f"  excl_abs_share={p3['excl_share_abs']:.3f} custom_walk={p3['custom_walk_n']}")

    print("pass 4 coverage")
    p4 = pass4_coverage(con, store)
    print(
        f"  first_g0={p4['first_g0_share']:.3f} liq_finite={p4['first_g0_liq_finite']:.3f} "
        f"liq_null={p4['first_g0_liq_null']:.3f} liq_zero={p4['first_g0_liq_zero']:.3f}"
    )

    print("pass 5 persist short vs long")
    p5 = pass5_persist(con, store)
    print(
        f"  rho_t3={p5['all_lag3']['rho']:.3f} short={p5['short_lag3']['rho']:.3f} "
        f"long={p5['long_lag3']['rho']:.3f} still={p5['still_shining']} long_real={p5['long_real']}"
    )

    print("pass 6 Q1 last-value")
    p6 = pass6_q1(store)
    print(
        f"  liq_p50={p6['last_liq']['p50']:.4g} liq<0={p6['last_liq']['share_neg']:.3f} "
        f"rw<1={p6['last_rw']['share_lt1']:.3f}"
    )

    print("pass 7 holdout coverage")
    p7 = pass7_holdout(con, store)
    print(f"  bal={p7['n_bal']}/{p7['n_hold']} cash={p7['n_cash']}")

    print("pass 8 zombies")
    p8 = pass8_zombies(con)
    print(f"  zombie_prod={p8['share_prod']:.3f} zombie_|bal|={p8['share_bal']:.4f}")

    print("pass 9 b_bal_vol SIZE")
    p9 = pass9_vol(store)
    print(f"  rho_login3={p9['rho_login3']:.3f} is_size={p9['is_size']}")

    print("pass 10 left-trunc negatives")
    p10 = pass10_neg(con, store)
    print(
        f"  late_pre_n={p10['late_pre_n']} grid_at_first_tx={p10['grid_starts_at_first_tx']} "
        f"first_neg_late={p10['first_neg_late']:.3f} ontime={p10['first_neg_ontime']:.3f}"
    )

    print("pass 11 short persist why")
    p11 = pass11_short_why(con, store)
    print(
        f"  short_all={p11['short_all']['rho']:.3f} short_mat={p11['short_mat']['rho']:.3f} "
        f"long_mat={p11['long_mat']['rho']:.3f} leak_rank={p11['leak_rank']}"
    )

    print("pass 12 age negatives")
    p12 = pass12_age_neg(con, store)
    print(f"  clusters_early={p12['clusters_early']} g0_neg={p12['g0_neg']}")

    print("pass 13 runway anatomy")
    p13 = pass13_runway(store)
    print(f"  thin={p13['share_thin']:.3f} pos_thin={p13['share_pos_thin']:.3f} rho_size={p13['rho_rw_login']:.3f}")

    print("pass 14 holes")
    p14 = pass14_holes(con, store)
    print(f"  no_bal={p14['n_no_bal']} no_cash={p14['n_no_cash']}")

    print("pass 15 last vs snap = Sep")
    p15 = pass15_sep(con, store)
    print(f"  rho={p15['rho_last_snap']:.3f} delta_p50={p15['delta_p50']:.4g} sep_id_max={p15['sep_vs_delta_max']:.4g}")

    print("pass 16 last-value is the still")
    p16 = pass16_last_still(con, store)
    print(f"  q1_is_still={p16['q1_is_still']} rho={p16['rho']:.3f} rel_p50={p16['rel_p50']:.4g}")

    print("pass 17 accumulation vs composition")
    p17 = pass17_accum(con, store)
    print(
        f"  share_up={p17['share_up_all']:.3f} long_neg {p17['long_first_neg']:.3f}->{p17['long_last_neg']:.3f} "
        f"n2p={p17['n2p']} p2n={p17['p2n']}"
    )

    print("pass 18 runway persist")
    p18 = pass18_runway_persist(con, store)
    print(f"  rho_liq={p18['rho_liq']:.3f} rho_rw={p18['rho_rw']:.3f} rw_stickier={p18['rw_stickier']}")

    print("pass 19 hole txs")
    p19 = pass19_hole_txs(con, store)
    print(f"  cash_txs={p19['n_cash_tx_rows']} cos={p19['n_co_cash_tx']} b_miss={p19['b_miss']}")

    print("pass 20 hole timing")
    p20 = pass20_hole_timing(con, store)
    print(f"  cash_prod={p20['n_cash_prod']} quiet30={p20['n_quiet_30']} live={p20['n_live']} live_id={p20['live_id']}")

    print("pass 21 approach still")
    p21 = pass21_approach(con, store)
    print(f"  closer={p21['share_closer']:.3f} farther={p21['share_farther']:.3f} mostly={p21['mostly_approach']}")

    print("pass 22 custom in walk")
    p22 = pass22_custom(con)
    print(f"  custom_|share|={p22['share_abs']:.3f} n_any={p22['n_any']} n_all={p22['n_all']}")

    print("pass 23 median vs last")
    p23 = pass23_median(con, store)
    print(f"  rho_med_last={p23['rho_med_last']:.3f} typical_is_still={p23['typical_is_still']}")

    print("pass 24 neg stickiness")
    p24 = pass24_neg_stick(store)
    print(f"  ever={p24['ever_neg_share']:.3f} run_p50={p24['run_p50']} stay={p24['p_stay_neg']:.3f} blips={p24['isolated_blips']}")

    print("pass 25 sep size")
    p25 = pass25_sep_size(con, store)
    print(f"  sep_rel_p50={p25['rel_p50']:.4g} gt10={p25['share_gt10pct']:.3f} rho_fat={p25['rho_fat']:.3f} rho_thin={p25['rho_thin']:.3f}")

    print("pass 26 unique piles + hole last-tx")
    p26 = pass26_pile_unique_and_hole_cluster(con, store)
    print(
        f"  walk_unique={p26['n_walk_unique']} train={p26['n_walk_train']} "
        f"hole_0720={p26['hole_0720']:.3f} photo_0720={p26['photo_0720']:.3f} "
        f"cluster={p26['cluster_is_hole']}"
    )

    png_ok = make_png(p2)
    print(f"png={OUT_PNG if png_ok else 'skipped'}")

    write_md(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, png_ok)
    print(f"wrote {OUT_MD}")
    append_registry(p1, p2, p3, p4, p5, p7, p8, p9, p11, p15, p16, p17, p18, p19, p20, p21, p23, p24, p25, p26)
    con.close()
    print("balances_b_qa done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
