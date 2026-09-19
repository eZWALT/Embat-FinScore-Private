"""Assemble the Y panel from target modules.

Families / Y modules that are not imported yet are skipped. Does not write
``monthly.parquet``. Run:

    python -m analysis.targets.build_targets
"""
from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

# allow `python analysis/targets/build_targets.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.features.common import DATA, ROOT, connect, train_mask
from analysis.features.grid import monthly_grid

TARGETS_DIR = Path(__file__).resolve().parent
WAVES_DIR = ROOT / "overnight" / "waves"

# Frozen verdicts. Wave-note scanners are greedy (ACCEPTED on the same line
# as another Y name can flip a parked label). These sets win.
FROZEN_ACCEPTED = frozenset(
    {
        "y2_neg_2of3",
        "y3_recover_cash_6m",
        "y4_ds_r_double",
        "y5_ap_od30_ownp80",
        "y5_ar_od30_sust",
        "y7_top1_lost",
        "y7_top1_lost_inflow",
        "y8_inv_worse_6",
        "y8_cash_worse_6",
        "y9_fee_r_ownp80",
        "y9_fee_spike",
    }
)
FROZEN_REJECTED = frozenset(
    {
        "y2_runway_lt1_sust",
        "y2_onset_neg",
        "y3_recover_6m",
        "y4_ds_r_gt05_sust",
        "y4_new_facility_after_dip",
        "y4_ogtg_appear",
        "y5_ap_delay_up15",
        "y6_zero_in_3",
        "y6_missed_payroll",
        "y6_silent_60",
        "y10_new_loc_after_stress",
        "y10_ogtg_last_month",
        "y10_add_loc_6m",
        "y10_loc_book_then_fees",
    }
)

# y10_util is in-module only until a parent merge. discover() must skip it
# or a casual `python -m analysis.targets.build_targets` writes six new
# columns and the wave scanner can flip parked Ys. The two interest-on-LOC
# labels (y10_loc_then_int_ownp80 / spike) stay out of FROZEN_ACCEPTED.
SKIP_ASSEMBLE_MODULES = frozenset({"y10_util", "y11_dark"})
ACCEPT_NOTES = {
    "y2_neg_2of3": "LOW_POWER holdout; GBM failed",
    "y2_runway_lt1_sust": "SIZE_PROXY",
    "y3_recover_cash_6m": "GBM CV 0.71 / shallow 0.762; SHAP quiet-stressed",
    "y6_zero_in_3": "inverse size proxy",
    "y6_missed_payroll": "GBM loses to e_dpo_proxy",
    "y8_inv_worse_6": "label accepted; model PARK",
    "y8_cash_worse_6": "label accepted; model PARK",
    "y9_fee_r_ownp80": "FinRegLab NSF/fee; also forbid a_fin_cost,a_fc",
    "y9_fee_spike": "FinRegLab NSF/fee; also forbid a_fin_cost,a_fc",
    "y10_new_loc_after_stress": "PARK subset of rejected y4_new_facility_after_dip",
    "y10_loc_book_then_fees": "CLOSE: Spearman 1.0 vs y9_fee_r_ownp80",
    "y10_ogtg_last_month": "extract still, not an onset",
    "y10_add_loc_6m": "SIZE after birth-month leak removed",
}

FEATURE_FAMILIES = [
    ("a", "analysis.features.cashflow", "Cash flow levels and shape"),
    ("b", "analysis.features.liquidity", "Liquidity / balance path"),
    ("c", "analysis.features.ops", "Operational regularity"),
    ("d", "analysis.features.counterparties", "Counterparty structure"),
    ("e", "analysis.features.invoices", "Receivables / payables"),
    ("f", "analysis.features.debt", "Debt and financing"),
    ("g", "analysis.features.products", "Product mix and access"),
    ("h", "analysis.features.groupctx", "Group context"),
]

META_ROWS = [
    ("freq", "meta", ["transactions"], "M on the monthly panel; W on the weekly panel"),
    ("first_month", "meta", ["transactions"], "date_trunc(month, min(tx.date)); company enters the monthly grid here"),
    ("group_id", "meta", ["companies"], "companies.group_id"),
    ("country", "meta", ["companies"], "ISO-2 country from clean.companies"),
    ("currency", "meta", ["companies"], "accounting currency"),
    ("erp", "meta", ["companies"], "company ERP flag"),
    ("created_at", "meta", ["companies"], "company created_at (extract snapshot; not a 2024 event)"),
    ("group_size", "meta", ["groups"], "groups.n_companies_in_sample"),
    ("group_erp", "meta", ["groups"], "groups.erp"),
    ("n_banking", "meta", ["banking_products"], "count of banking_products (extract stock)"),
    ("n_banks", "meta", ["banking_products"], "count distinct bank_name (extract stock)"),
    ("n_bank_types", "meta", ["banking_products"], "count distinct type (extract stock)"),
    ("n_debt", "meta", ["debt_products"], "count of debt_products (extract stock)"),
    ("n_debt_types", "meta", ["debt_products"], "count distinct type (extract stock)"),
]

_COL_ATTRS = (
    "FEATURE_COLS",
    "B_COLS",
    "Y1_COLS",
    "Y2_COLS",
    "Y3_COLS",
    "Y4_COLS",
    "Y5_COLS",
    "Y6_COLS",
    "Y7_COLS",
    "Y8_COLS",
    "Y_COLS",
    "_COLS",
)

_COL_IN_TEXT = re.compile(r"\b(y\d+_[a-z0-9_]+)\b")
_VERDICT_IN_TEXT = re.compile(r"\b(ACCEPTED|REJECTED)\b", re.I)


def discover_target_modules() -> list[str]:
    names = []
    for path in sorted(TARGETS_DIR.glob("y*.py")):
        if path.stem in SKIP_ASSEMBLE_MODULES:
            continue
        names.append(f"analysis.targets.{path.stem}")
    return names


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["company_id", "period"]].copy()
    out["company_id"] = out["company_id"].astype(str)
    out["period"] = pd.to_datetime(out["period"])
    return out


def _module_columns(mod) -> list[str]:
    meta = getattr(mod, "META", {}) or {}
    cols = meta.get("columns")
    if cols:
        return list(cols)
    for attr in _COL_ATTRS:
        if hasattr(mod, attr):
            val = getattr(mod, attr)
            if val:
                return list(val)
    return []


def _iter_bullets(doc: str):
    cur = None
    for raw in (doc or "").splitlines():
        s = raw.strip()
        if s.startswith("-") or s.startswith("*"):
            if cur:
                yield cur
            cur = s[1:].strip()
        elif cur and s and (raw.startswith(" ") or raw.startswith("\t")):
            cur = cur + " " + s
        elif cur and not s:
            yield cur
            cur = None
    if cur:
        yield cur


def formulas_from_docstring(doc: str) -> dict[str, str]:
    """Parse '- name = formula' / '- a / b: formula' lines from a module docstring."""
    out: dict[str, str] = {}
    for bullet in _iter_bullets(doc):
        line = bullet.replace("``", "").replace("`", "")
        m = re.match(
            r"^([A-Za-z][A-Za-z0-9_/,\s]+?)\s*(?:=|:|—|--)\s+(.+)$",
            line,
        )
        if not m:
            m = re.match(
                r"^([a-z][a-z0-9_]+(?:\s*/\s*[a-z][a-z0-9_]+)*)\s{2,}(.+)$",
                line,
            )
        if not m:
            # Family B: "- b_below_half_runway 1 if liq < ..." (single space)
            m = re.match(r"^([a-z]_[a-z0-9_]+)\s+(.+)$", line)
        if not m:
            continue
        names, form = m.group(1), re.sub(r"\s+", " ", m.group(2)).strip()
        for name in re.split(r"\s*/\s*|,\s*", names):
            name = name.strip()
            if re.fullmatch(r"[a-z]\w+", name):
                out[name] = form
    return out


def y1_formulas(definition: str) -> dict[str, str]:
    """Split Y1's packed definition into per-column one-liners."""
    out: dict[str, str] = {}
    text = re.sub(r"\s+", " ", definition or "")
    parts = [p.strip() for p in text.split(";") if p.strip()]
    mapping = {
        "y1_net_h": "y1_net_h* = operational net (op_in - op_out) at t+h",
        "y1_in_h": "y1_in_h* = operational inflow at t+h",
        "y1_liq_h": (
            "y1_liq_h* = reconstructed month-end liquidity at t+h "
            "(same unwind as score_pipeline._liquidity)"
        ),
    }
    for part in parts:
        low = part.lower()
        if "y1_net" in low:
            mapping["y1_net_h"] = part
        elif "y1_in" in low:
            mapping["y1_in_h"] = part
        elif "y1_liq" in low:
            mapping["y1_liq_h"] = part
    for col, h in (
        ("y1_net_h1", 1),
        ("y1_net_h3", 3),
        ("y1_in_h1", 1),
        ("y1_in_h3", 3),
        ("y1_liq_h1", 1),
        ("y1_liq_h3", 3),
    ):
        prefix = "y1_net_h" if "net" in col else ("y1_in_h" if "_in_" in col else "y1_liq_h")
        base = mapping[prefix]
        out[col] = base.replace("t+h", f"t+{h}").replace("h*", f"h{h}")
    return out


def scan_wave_verdicts() -> dict[str, str]:
    """Later wave notes may raise a shipped Y to accepted=1. Frozen list still wins.

    Same-line only, so a table row's REJECTED is not stolen from the previous ACCEPTED.
    Last mention in filename order wins.
    """
    verdicts: dict[str, str] = {}
    if not WAVES_DIR.exists():
        return verdicts
    for path in sorted(WAVES_DIR.glob("*.md")):
        for line in path.read_text(encoding="utf-8").splitlines():
            cols = list(_COL_IN_TEXT.finditer(line))
            verds = [(m.start(), m.group(1).upper()) for m in _VERDICT_IN_TEXT.finditer(line)]
            if not cols or not verds:
                continue
            for cm in cols:
                after = [v for pos, v in verds if pos >= cm.start()]
                before = [v for pos, v in verds if pos < cm.start()]
                verdicts[cm.group(1)] = after[0] if after else before[-1]
    return verdicts


def meta_accepted_flag(meta: dict, col: str) -> bool | None:
    if not meta or "accepted" not in meta:
        return None
    val = meta["accepted"]
    if isinstance(val, bool):
        return val
    if isinstance(val, dict):
        if col in val:
            return bool(val[col])
        return None
    if isinstance(val, (list, tuple, set, frozenset)):
        return col in val
    return None


def resolve_accepted(col: str, meta: dict, wave: dict[str, str]) -> tuple[int, str]:
    if col.startswith("y1_"):
        return 1, "y1_continuous"
    if col in FROZEN_REJECTED:
        return 0, "frozen_reject"
    if col in FROZEN_ACCEPTED:
        return 1, "frozen_wave"
    flag = meta_accepted_flag(meta, col)
    if flag is True:
        return 1, "meta"
    if flag is False:
        return 0, "meta"
    if wave.get(col) == "REJECTED":
        return 0, "wave_note"
    if wave.get(col) == "ACCEPTED":
        return 1, "wave_note"
    return 0, "shipped"


def is_binary_series(s: pd.Series, kind: str | None) -> bool:
    if kind == "continuous":
        return False
    if kind == "binary":
        return True
    vals = pd.to_numeric(s.dropna(), errors="coerce")
    if vals.empty:
        return kind != "continuous"
    uniq = set(vals.unique().tolist())
    return uniq <= {0.0, 1.0}


def acceptance_table(panel: pd.DataFrame, y_meta: dict[str, dict], wave: dict[str, str]) -> pd.DataFrame:
    tr = train_mask(panel["company_id"])
    n_train = int(tr.sum())
    rows = []
    y_cols = [c for c in panel.columns if c.startswith("y")]
    for col in y_cols:
        meta = y_meta.get(col, {})
        kind = meta.get("kind")
        s = panel.loc[tr, col]
        labeled = s.notna()
        n_lab = int(labeled.sum())
        binary = is_binary_series(s, kind)
        if binary and n_lab:
            base = float(pd.to_numeric(s[labeled], errors="coerce").mean())
            n_pos = int((pd.to_numeric(s[labeled], errors="coerce") == 1).sum())
        else:
            base = float("nan")
            n_pos = 0
        accepted, source = resolve_accepted(col, meta, wave)
        forb = meta.get("forbidden_x_families") or []
        if isinstance(forb, dict):
            forb = forb.get(col, meta.get("forbidden_x_families_default") or [])
        rows.append(
            {
                "column": col,
                "kind": "binary" if binary else "continuous",
                "n_train": n_train,
                "n_labeled": n_lab,
                "n_pos": n_pos if binary else "",
                "base_rate": None if not binary else (round(base, 6) if n_lab else None),
                "accepted": int(accepted),
                "accepted_source": source,
                "forbidden_x": ",".join(str(x) for x in forb) if forb else "",
                "horizon": meta.get("horizon", ""),
                "notes": ACCEPT_NOTES.get(col, ""),
            }
        )
    return pd.DataFrame(rows)


def write_feature_dictionary(
    path: Path,
    family_mods: list[tuple[str, object, str]],
    target_mods: list[tuple[str, object]],
    y_cols: list[str],
) -> None:
    lines = [
        "# Feature dictionary",
        "",
        "Human one-liners from family module docstrings and target `META`.",
        "Generated by `analysis.targets.build_targets`. Features use only events",
        "with date ≤ period end. Snapshot extract fields are flagged as such.",
        "`d_interco_share` is all-NaN: invoice/tx IDs are `COUNTERPARTY_*`,",
        "companies are `COMP_*`, and the two sets do not overlap.",
        "",
        "## Grid / company metadata",
        "",
    ]
    for name, family, tables, formula in META_ROWS:
        lines.extend(_section(name, family, tables, formula, forbidden=None))

    for letter, mod, title in family_mods:
        src = list(getattr(mod, "SOURCE_TABLES", []) or [])
        forms = formulas_from_docstring(getattr(mod, "__doc__", "") or "")
        cols = _module_columns(mod)
        if not cols:
            cols = [c for c in forms if c.startswith(f"{letter}_")]
        if letter == "c" and "c_missed_tax" not in cols:
            cols.append("c_missed_tax")
            forms.setdefault(
                "c_missed_tax",
                "1 if sum(c_tax_month) over last ≤6 months ≥ 3 and c_tax_month = 0 this month",
            )
        lines.extend(["", f"## Family {letter.upper()} — {title}", ""])
        if src:
            lines.append(f"Source tables: {', '.join(src)}.")
            lines.append("")
        for col in cols:
            form = forms.get(col, "see module docstring")
            if col == "d_interco_share":
                form = (
                    "|flows| whose counterparty_id equals another company_id in "
                    "the same group_id, over |tx amounts| in the 6-month window. "
                    "All-NaN here: COMP_* and COUNTERPARTY_* do not overlap "
                    "(do not invent a mapping)"
                )
            lines.extend(_section(col, letter, src, form, forbidden=None))

    lines.extend(["", "## Targets (Y)", ""])
    seen = set()
    for modname, mod in target_mods:
        meta = getattr(mod, "META", {}) or {}
        src = list(meta.get("source_tables") or getattr(mod, "SOURCE_TABLES", []) or [])
        defs = dict(meta.get("definitions") or {})
        if meta.get("definition") and str(meta.get("name", "")).startswith("y1"):
            defs.update(y1_formulas(str(meta["definition"])))
        defs.update(formulas_from_docstring(getattr(mod, "__doc__", "") or ""))
        forb_all = meta.get("forbidden_x_families") or []
        forb_by = meta.get("forbidden_x_families_by_column") or meta.get("forbidden_x_by_column") or {}
        cols = [c for c in _module_columns(mod) if c in y_cols or not y_cols]
        if not cols:
            cols = [c for c in y_cols if c.startswith(modname.rsplit(".", 1)[-1][:2] + "_")]
        family = str(meta.get("name") or modname.rsplit(".", 1)[-1])
        lines.extend(["", f"### Module `{modname}`", ""])
        if meta.get("literature"):
            lines.append(f"Literature: {meta['literature']}")
            lines.append("")
        for col in cols:
            if col in seen:
                continue
            seen.add(col)
            forb = forb_by.get(col, forb_all)
            form = defs.get(col, meta.get("definition") or "see META")
            form = re.sub(r"\s+", " ", str(form)).strip()
            lines.extend(_section(col, family, src, form, forbidden=forb))

    extra = [c for c in y_cols if c not in seen]
    if extra:
        lines.extend(["", "### Other Y columns", ""])
        for col in extra:
            lines.extend(_section(col, col.split("_")[0], [], "see target module META", forbidden=None))

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _section(name: str, family: str, tables: list[str], formula: str, forbidden) -> list[str]:
    rows = [
        f"#### `{name}`",
        "",
        f"- **family:** {family}",
        f"- **source tables:** {', '.join(tables) if tables else '—'}",
        f"- **formula:** {formula}",
    ]
    if forbidden is not None:
        fx = ", ".join(str(x) for x in forbidden) if forbidden else "none (cross-horizon / cross-source)"
        rows.append(f"- **forbidden X:** {fx}")
    rows.append("")
    return rows


def main() -> None:
    out = DATA / "feature_store"
    out.mkdir(parents=True, exist_ok=True)

    con = connect()
    grid = monthly_grid(con)
    panel = _keys(grid)
    loaded: list[str] = []
    y_meta: dict[str, dict] = {}
    target_mods: list[tuple[str, object]] = []

    for modname in discover_target_modules():
        try:
            mod = importlib.import_module(modname)
        except ModuleNotFoundError:
            print(f"skip {modname} (not written yet)")
            continue
        if not hasattr(mod, "build"):
            print(f"skip {modname} (no build)")
            continue
        raw = mod.build(con, grid[["company_id", "period"]].copy())
        extra = [c for c in raw.columns if c not in {"company_id", "period"} and str(c).startswith("y")]
        if not extra:
            print(f"skip {modname} (no y* columns)")
            continue
        part = raw[["company_id", "period", *extra]].copy()
        part["company_id"] = part["company_id"].astype(str)
        part["period"] = pd.to_datetime(part["period"])
        clash = set(extra) & set(panel.columns)
        if clash:
            raise ValueError(f"{modname} column clash: {clash}")
        if part.duplicated(["company_id", "period"]).any():
            raise ValueError(f"{modname}: duplicate company_id,period")
        panel = panel.merge(part[["company_id", "period", *extra]], on=["company_id", "period"], how="left")
        loaded.append(modname.rsplit(".", 1)[-1])
        target_mods.append((modname, mod))
        meta = dict(getattr(mod, "META", {}) or {})
        forb_by = meta.get("forbidden_x_families_by_column") or meta.get("forbidden_x_by_column") or {}
        for col in extra:
            y_meta[col] = {
                **meta,
                "forbidden_x_families": forb_by.get(col, meta.get("forbidden_x_families") or []),
            }

    con.close()

    path = out / "targets.parquet"
    panel.to_parquet(path, index=False)

    wave = scan_wave_verdicts()
    acc = acceptance_table(panel, y_meta, wave)
    acc_path = out / "y_acceptance.csv"
    acc.to_csv(acc_path, index=False)

    family_mods: list[tuple[str, object, str]] = []
    for letter, modname, title in FEATURE_FAMILIES:
        try:
            family_mods.append((letter, importlib.import_module(modname), title))
        except ModuleNotFoundError:
            print(f"dictionary skip {modname} (not written yet)")

    y_cols = [c for c in panel.columns if c.startswith("y")]
    dict_path = out / "feature_dictionary.md"
    write_feature_dictionary(dict_path, family_mods, target_mods, y_cols)

    accepted = acc.loc[acc["accepted"] == 1, "column"].tolist() if not acc.empty else []
    summary = {
        "rows": int(len(panel)),
        "companies": int(panel["company_id"].nunique()),
        "n_cols": int(panel.shape[1]),
        "modules": loaded,
        "y_columns": y_cols,
        "accepted": accepted,
    }
    print(f"wrote {path} shape={panel.shape} modules={loaded}")
    print(f"wrote {acc_path}")
    print(f"wrote {dict_path}")
    print("y_columns", y_cols)
    print("accepted", accepted)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
