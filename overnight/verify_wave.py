#!/usr/bin/env python3
"""Smoke-check family/target modules after a wave. Exit 1 on contract break."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.features.common import connect, load_holdout
from analysis.features.grid import monthly_grid

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
TARGETS = [
    "analysis.targets.y1_forecast",
    "analysis.targets.y2_stress",
    "analysis.targets.y3_recovery",
    "analysis.targets.y4_debt",
    "analysis.targets.y5_payment",
    "analysis.targets.y6_activity",
    "analysis.targets.y7_concentration",
    "analysis.targets.y8_cross",
]


def check_module(modname: str, prefix: str | None, grid, con, holdout) -> str:
    try:
        mod = importlib.import_module(modname)
    except ModuleNotFoundError:
        return f"ABSENT {modname}"
    if not hasattr(mod, "build"):
        return f"FAIL {modname}: no build()"
    part = mod.build(con, grid.copy())
    need = {"company_id", "period"}
    if not need <= set(part.columns):
        return f"FAIL {modname}: missing keys"
    extra = [c for c in part.columns if c not in need]
    if prefix and extra and not all(c.startswith(prefix) for c in extra):
        return f"FAIL {modname}: columns not prefixed {prefix}_ ({extra[:8]})"
    if part.duplicated(["company_id", "period"]).any():
        return f"FAIL {modname}: duplicate company_id,period"
    # holdout rows may exist in the panel; fitting on them is the crime — flag if module stored a ref
    if hasattr(mod, "REF") and getattr(mod, "REF", None) is not None:
        return f"FAIL {modname}: module-level REF (possible holdout leak)"
    n = part["company_id"].nunique()
    return f"OK {modname} rows={len(part)} companies={n} cols={len(extra)}"


def main() -> int:
    holdout = load_holdout()
    con = connect()
    grid = monthly_grid(con)[["company_id", "period"]]
    print(f"grid={grid.shape} holdout={len(holdout)}")
    bad = 0
    for letter, name in FAMILIES:
        msg = check_module(name, f"{letter}_", grid, con, holdout)
        print(msg)
        if msg.startswith("FAIL"):
            bad += 1
    for name in TARGETS:
        msg = check_module(name, None, grid, con, holdout)
        print(msg)
        if msg.startswith("FAIL"):
            bad += 1
    con.close()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
