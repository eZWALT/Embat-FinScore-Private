"""Assemble monthly (and later weekly) feature store from family modules.

Families that are not imported yet are skipped. Run:

    python -m analysis.features.build_feature_store
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pandas as pd

# allow `python analysis/features/build_feature_store.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.features.common import DATA, TABLES, connect
from analysis.features.grid import company_meta, monthly_grid

FAMILIES = [
    ("a", "analysis.features.cashflow"),
    ("b", "analysis.features.liquidity"),
    ("c", "analysis.features.ops"),
    ("d", "analysis.features.counterparties"),
    ("e", "analysis.features.invoices"),
    ("f", "analysis.features.debt"),
    ("g", "analysis.features.products"),
    ("h", "analysis.features.groupctx"),
]


def coverage_report(panel: pd.DataFrame, used_tables: list[str]) -> pd.DataFrame:
    feat_cols = [c for c in panel.columns if c not in {"company_id", "period", "freq", "first_month", "first_week"}]
    rows = []
    n_cm = len(panel)
    n_co = panel["company_id"].nunique()
    for c in feat_cols:
        s = panel[c]
        rows.append(
            {
                "kind": "feature",
                "name": c,
                "pct_company_months": float(s.notna().mean()),
                "pct_companies": float(panel.loc[s.notna(), "company_id"].nunique() / n_co if n_co else 0),
                "n_company_months": n_cm,
            }
        )
    for t in TABLES:
        rows.append(
            {
                "kind": "table",
                "name": t,
                "pct_company_months": 1.0 if t in used_tables else 0.0,
                "pct_companies": 1.0 if t in used_tables else 0.0,
                "n_company_months": n_cm,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    out = DATA / "feature_store"
    out.mkdir(parents=True, exist_ok=True)
    con = connect()
    grid = monthly_grid(con)
    meta = company_meta(con)
    panel = grid.merge(meta, on="company_id", how="left")
    used = ["transactions", "companies", "groups", "banking_products", "debt_products"]
    loaded = []
    for letter, modname in FAMILIES:
        try:
            mod = importlib.import_module(modname)
        except ModuleNotFoundError:
            print(f"skip {modname} (not written yet)")
            continue
        part = mod.build(con, grid[["company_id", "period"]].copy())
        extra = [c for c in part.columns if c not in {"company_id", "period"}]
        clash = set(extra) & set(panel.columns)
        if clash:
            raise ValueError(f"{modname} column clash: {clash}")
        panel = panel.merge(part, on=["company_id", "period"], how="left")
        loaded.append(letter)
        used.extend(getattr(mod, "SOURCE_TABLES", []))
    con.close()
    used = sorted(set(used))
    path = out / "monthly.parquet"
    panel.to_parquet(path, index=False)
    cov = coverage_report(panel, used)
    cov.to_csv(out / "coverage.csv", index=False)
    (out / "build.json").write_text(
        json.dumps(
            {
                "rows": int(len(panel)),
                "companies": int(panel.company_id.nunique()),
                "families": loaded,
                "tables": used,
                "n_cols": int(panel.shape[1]),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {path} shape={panel.shape} families={loaded}")


if __name__ == "__main__":
    main()
