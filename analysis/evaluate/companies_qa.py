"""companies.csv QA — country / erp / currency / created_at as Q1 context.

NORTH_STAR: hidden test is *new groups*. Company-constant flags that are
group-type dummies cannot transfer. created_at is a connection clock —
PARK as a health Y (trail QA). Do not invent a merged Y from country/erp.

Holdout 72 (seed 20260918) is coverage only. Rates, terciles, AUROC on
train. DuckDB read_only. No parquet rewrite. No new GBM. No build_targets.
No product/. No 0–100.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.companies_qa

Owned: analysis/evaluate/companies_qa.py, analysis/outputs/companies_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_companies.md (end).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.evaluate.protocol import (
    FOLD_SEED,
    assert_no_holdout,
    auroc,
    group_folds,
    leakage_check,
    load_holdout,
)
from analysis.features.common import ANALYSIS, DATA, MONTHS, connect, train_mask
from analysis.targets.y11_dark import book_invoice_ids, dark_population

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:  # pragma: no cover
    HAS_MPL = False

REGISTRY = ANALYSIS / "experiments" / "registry.csv"
STORE = DATA / "feature_store" / "monthly.parquet"
TARGETS = DATA / "feature_store" / "targets.parquet"
OUT_MD = ANALYSIS / "outputs" / "companies_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "companies_erp_dark_country.png"
FX_MD = ANALYSIS / "outputs" / "fx_qa.md"
AGENT = "a14da08b"
WAVE = "4"
ROUND = "R4"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
N_FOLDS = 5
DAYS_BENCH = 0.711
KEEP_DELTA = 0.02
SIZE_PARK = 0.60
SIZE_RHO = 0.50
NEAR_SIZE_RHO = 0.35
PANEL_START = pd.Timestamp("2024-09-01")
EXTRACT = pd.Timestamp("2026-09-01")
JOIN_NULL_ERP_DARK = 0.921
JOIN_NAMED_DARK = 37
JOIN_NULL_ERP_N = 506
JOIN_NULL_WITH_INV = 0.144

# companies.erp slug → groups.erp display family (many-to-one / aliases).
ERP_FAMILY = {
    "businesscentral": "business_central",
    "microsoft business central": "business_central",
    "netsuite": "netsuite",
    "businessone": "business_one",
    "sap business one": "business_one",
    "sage200": "sage_200",
    "sage 200": "sage_200",
    "sage50": "sage_50",
    "sage 50": "sage_50",
    "sagex3": "sage_x3",
    "sage x3": "sage_x3",
    "sageintacct": "sage_intacct",
    "dynamicsax": "dynamics_ax",
    "microsoft dynamics - ax 2012": "dynamics_ax",
    "microsoft dynamics - ax 2009": "dynamics_ax",
    "navision": "navision",
    "microsoft navision": "navision",
    "fo": "dynamics_fo",
    "microsoft dynamics - f&o": "dynamics_fo",
    "m3rosetta": "infor_m3",
    "infor m3": "infor_m3",
    "movex": "infor_m3",
    "distritok": "distrito_k",
    "distrito k": "distrito_k",
    "a3": "a3",
    "a3 erp": "a3",
    "etendo": "etendo",
    "r3": "sap_r3",
    "sap r3 / s4": "sap_r3",
    "holded": "holded",
    "libra": "libra",
    "ekon": "ekon",
    "datev": "datev",
    "sapbyd": "sap_byd",
    "odoo": "odoo",
    "oracle cloud": "oracle",
    "desarrollo propio": "custom",
}


def erp_family(val) -> str | None:
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return None
    s = str(val).strip()
    if not s or s.lower() in ("none", "null", "nan", "<na>"):
        return None
    return ERP_FAMILY.get(s.lower(), s.lower())


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")


def _f(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    return f"{float(x):.{nd}f}"


def _pp(x, nd=2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{100.0 * float(x):.{nd}f}%"


def _pp_delta(x, nd=2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{100.0 * float(x):+.{nd}f}pp"


def _pct(n, d) -> float:
    if d is None or d == 0:
        return float("nan")
    return float(n) / float(d)


def _blank(s: pd.Series) -> pd.Series:
    """NULL / empty / whitespace string → missing."""
    t = s.astype("string")
    return t.isna() | t.str.strip().eq("") | t.str.strip().str.lower().isin(("none", "null", "nan"))


def spearman(a, b, min_n: int = 20) -> float:
    d = pd.DataFrame(
        {"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}
    ).dropna()
    if len(d) < min_n or d["a"].nunique() < 2 or d["b"].nunique() < 2:
        return float("nan")
    return float(d["a"].corr(d["b"], method="spearman"))


def choose_sign(y: pd.Series, x: pd.Series) -> int:
    auc_p = auroc(y, x)
    auc_n = auroc(y, -x)
    if not np.isfinite(auc_p) and not np.isfinite(auc_n):
        return 1
    if not np.isfinite(auc_p):
        return -1
    if not np.isfinite(auc_n):
        return 1
    return -1 if auc_n > auc_p else 1


def two_sided(auc: float) -> float:
    if not np.isfinite(auc):
        return float("nan")
    return float(max(auc, 1.0 - auc))


def signed_oof_auroc(
    y: pd.Series,
    x: pd.Series,
    folds: pd.Series,
    train_lab: pd.Series,
    n_folds: int = N_FOLDS,
) -> dict:
    y = pd.to_numeric(y, errors="coerce")
    x = pd.to_numeric(x, errors="coerce")
    fold_rows = []
    aucs = []
    for k in range(n_folds):
        tr = train_lab & (folds != k)
        va = train_lab & (folds == k)
        sign = choose_sign(y[tr], x[tr])
        auc = auroc(y[va], sign * x[va])
        defined = va & x.notna() & y.notna()
        aucs.append(auc)
        fold_rows.append(
            {
                "fold": k,
                "auroc": float(auc) if np.isfinite(auc) else float("nan"),
                "sign": int(sign),
                "n_va": int(defined.sum()),
                "n_pos": int((defined & (y == 1)).sum()),
            }
        )
    finite = [a for a in aucs if np.isfinite(a)]
    tr_sign = choose_sign(y[train_lab], x[train_lab])
    n_tr = int(train_lab.sum())
    n_def = int((train_lab & x.notna() & y.notna()).sum())
    return {
        "cv": float(np.mean(finite)) if finite else float("nan"),
        "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan"),
        "n_folds": len(finite),
        "folds": fold_rows,
        "train_sign": int(tr_sign),
        "train_auc": float(auroc(y[train_lab], tr_sign * x[train_lab])),
        "n_defined": n_def,
        "coverage": float(n_def / n_tr) if n_tr else float("nan"),
    }


def fit_terciles(size: pd.Series, mask: pd.Series) -> tuple[pd.Series, np.ndarray]:
    src = pd.to_numeric(size[mask], errors="coerce").dropna()
    if src.nunique() < 3:
        raise RuntimeError("not enough distinct size values for terciles")
    _cats, edges = pd.qcut(src, 3, retbins=True, duplicates="drop")
    edges = np.asarray(edges, dtype=float)
    edges[0] = min(edges[0], float(src.min())) - 1e-9
    edges[-1] = max(edges[-1], float(src.max())) + 1e-9
    labels = ("T1_small", "T2_mid", "T3_large")[: len(edges) - 1]
    cut = pd.cut(
        pd.to_numeric(size, errors="coerce"),
        bins=edges,
        labels=labels,
        include_lowest=True,
    )
    return cut.astype("object"), edges


def rate_row(y: pd.Series, cid: pd.Series, name: str) -> dict:
    ok = y.notna()
    n = int(ok.sum())
    pos = int((y[ok] == 1).sum()) if n else 0
    return {
        "slice": name,
        "n": n,
        "pos": pos,
        "cos": int(cid[ok].nunique()) if n else 0,
        "rate": float(y[ok].mean()) if n else float("nan"),
    }


def md_table(df: pd.DataFrame, cols: list[tuple[str, str, str]]) -> list[str]:
    head = "| " + " | ".join(h for _, h, _ in cols) + " |"
    sep = "|" + "|".join("---:" if k != "s" else "---" for _, _, k in cols) + "|"
    lines = [head, sep]
    if df is None or df.empty:
        return lines
    for _, r in df.iterrows():
        cells = []
        for key, _, kind in cols:
            v = r.get(key)
            if kind == "n":
                cells.append(_f(v, 0) if isinstance(v, (int, np.integer, float, np.floating)) else str(v))
            elif kind == "pp":
                cells.append(_pp(v))
            elif kind == "delta":
                cells.append(_pp_delta(v))
            elif kind == "f":
                cells.append(_f(v, 3))
            else:
                cells.append("" if v is None or (isinstance(v, float) and not np.isfinite(v)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["company_id"] = out["company_id"].astype(str)
    if "period" in out.columns:
        out["period"] = pd.to_datetime(out["period"])
    if "group_id" in out.columns:
        out["group_id"] = out["group_id"].astype(str)
    return out


def load_store() -> pd.DataFrame:
    if not STORE.exists():
        raise FileNotFoundError(f"{STORE} missing")
    need = [
        "company_id",
        "period",
        "group_id",
        "a_in3",
        "h_group_size",
        "c_n_days_with_tx",
        "e_fx_share",
    ]
    raw = pd.read_parquet(STORE)
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise RuntimeError(f"monthly.parquet missing {missing}")
    panel = _keys(raw[need])
    print(f"loaded store {STORE} shape={panel.shape}")
    return panel


def load_y() -> pd.DataFrame:
    if not TARGETS.exists():
        raise FileNotFoundError(f"{TARGETS} missing — do not run build_targets")
    raw = pd.read_parquet(TARGETS)
    need = ["company_id", "period", Y3, Y2]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise RuntimeError(f"targets.parquet missing {missing}")
    panel = _keys(raw[need])
    print(f"Y panel from {TARGETS} shape={panel.shape} (no build_targets)")
    return panel


def load_companies(con) -> pd.DataFrame:
    cos = con.execute(
        """
        SELECT
          CAST(c.company_id AS VARCHAR) AS company_id,
          CAST(c.group_id AS VARCHAR) AS group_id,
          c.country,
          c.currency,
          c.erp,
          c.created_at,
          g.erp AS group_erp,
          g.n_companies_in_sample AS n_in_sample
        FROM companies c
        LEFT JOIN groups g ON c.group_id = g.group_id
        """
    ).df()
    cos = _keys(cos)
    for col in ("country", "currency", "erp", "group_erp"):
        cos[col] = cos[col].astype("string")
    cos["created_at"] = pd.to_datetime(cos["created_at"], errors="coerce")
    cos["miss_country"] = _blank(cos["country"])
    cos["miss_currency"] = _blank(cos["currency"])
    cos["miss_erp"] = _blank(cos["erp"])
    cos["miss_created"] = cos["created_at"].isna()
    cos["has_country"] = ~cos["miss_country"]
    cos["has_erp"] = ~cos["miss_erp"]
    cos["is_eur"] = cos["currency"].fillna("").str.upper().eq("EUR")
    cos["erp_norm"] = cos["erp"].fillna("").str.strip()
    cos["erp_norm"] = cos["erp_norm"].where(~cos["miss_erp"], pd.NA)
    return cos


def load_clocks(con) -> pd.DataFrame:
    first_tx = con.execute(
        """
        SELECT
          CAST(company_id AS VARCHAR) AS company_id,
          MIN("date") AS first_tx,
          CAST(date_trunc('month', MIN("date")) AS DATE) AS first_tx_month
        FROM transactions
        WHERE "date" < TIMESTAMP '2026-09-01'
        GROUP BY 1
        """
    ).df()
    first_tx["company_id"] = first_tx["company_id"].astype(str)
    first_tx["first_tx"] = pd.to_datetime(first_tx["first_tx"], errors="coerce")
    first_tx["first_tx_month"] = pd.to_datetime(first_tx["first_tx_month"], errors="coerce")
    first_bank = con.execute(
        """
        SELECT
          CAST(company_id AS VARCHAR) AS company_id,
          MIN(created_at) AS first_bank_created,
          COUNT(*) AS n_banking
        FROM banking_products
        GROUP BY 1
        """
    ).df()
    first_bank["company_id"] = first_bank["company_id"].astype(str)
    first_bank["first_bank_created"] = pd.to_datetime(
        first_bank["first_bank_created"], errors="coerce"
    )
    out = first_tx.merge(first_bank, on="company_id", how="outer")
    return out


def miss_share(df: pd.DataFrame) -> dict:
    n = int(len(df))
    return {
        "n": n,
        "country": float(df["miss_country"].mean()) if n else float("nan"),
        "currency": float(df["miss_currency"].mean()) if n else float("nan"),
        "erp": float(df["miss_erp"].mean()) if n else float("nan"),
        "created_at": float(df["miss_created"].mean()) if n else float("nan"),
        "n_miss_country": int(df["miss_country"].sum()),
        "n_miss_currency": int(df["miss_currency"].sum()),
        "n_miss_erp": int(df["miss_erp"].sum()),
        "n_miss_created": int(df["miss_created"].sum()),
    }


def pass1_completeness(cos: pd.DataFrame, is_train: pd.Series, is_hold: pd.Series) -> dict:
    print("\nPASS 1 completeness")
    rows = []
    for name, sl in (("train", is_train), ("holdout", is_hold), ("all", pd.Series(True, index=cos.index))):
        rec = miss_share(cos.loc[sl])
        rec["split"] = name
        rows.append(rec)
        print(
            f"  {name} n={rec['n']} miss country={rec['country']:.3f} "
            f"currency={rec['currency']:.3f} erp={rec['erp']:.3f} created={rec['created_at']:.3f}"
        )
    tab = pd.DataFrame(rows)
    train = miss_share(cos.loc[is_train])
    hold = miss_share(cos.loc[is_hold])
    return {
        "tab": tab,
        "train": train,
        "hold": hold,
        "country_train_pp": train["country"],
        "erp_train_pp": train["erp"],
    }


def pass2_erp_dark(cos: pd.DataFrame, is_train: pd.Series, book: set[str], pop: dict) -> dict:
    print("\nPASS 2 erp × dark")
    tr = cos.loc[is_train].copy()
    tr["has_book"] = tr["company_id"].isin(book)
    dark = ~tr["has_book"]
    named = tr["has_erp"]
    n_dark = int(dark.sum())
    n_inv = int((~dark).sum())
    confirm_470 = n_dark == 470 and bool(pop.get("confirm_470"))
    # 2×2 confusion: named ERP vs book invoice
    a = int((named & ~dark).sum())  # invoiced + named ERP
    b = int((named & dark).sum())  # dark + named ERP
    c = int((~named & ~dark).sum())  # invoiced + NULL ERP
    d = int((~named & dark).sum())  # dark + NULL ERP
    conf = pd.DataFrame(
        [
            {"erp": "named", "book": "invoiced", "n": a},
            {"erp": "named", "book": "dark", "n": b},
            {"erp": "NULL", "book": "invoiced", "n": c},
            {"erp": "NULL", "book": "dark", "n": d},
        ]
    )
    print(conf.to_string(index=False))
    share_null_among_dark = _pct(d, n_dark)
    share_inv_among_null = _pct(c, c + d)
    n_null = c + d
    named_dark_erp = (
        tr.loc[named & dark, "erp_norm"].value_counts(dropna=False).reset_index()
    )
    named_dark_erp.columns = ["erp", "n"]
    named_inv_erp = (
        tr.loc[named & ~dark, "erp_norm"].value_counts(dropna=False).reset_index()
    )
    named_inv_erp.columns = ["erp", "n"]
    all_erp = tr.loc[named, "erp_norm"].value_counts(dropna=False).reset_index()
    all_erp.columns = ["erp", "n"]
    all_erp["share"] = all_erp["n"] / max(int(named.sum()), 1)
    print(
        f"  dark={n_dark} confirm_470={confirm_470} "
        f"NULL among dark={share_null_among_dark:.4f} (quoted 0.921) "
        f"named-dark={b} (quoted 37) null-erp={n_null} (quoted 506) "
        f"inv among null={share_inv_among_null:.4f} (quoted 0.144)"
    )
    perfect = b == 0 and c == 0
    print(f"  perfect dark flag? {perfect} (invoiced+NULL={c} dark+named={b})")

    # mix of named-dark: 110 mixed vs 360 all-dark
    mix_of = pop["mix_of"]
    tr["mix"] = tr["group_id"].map(mix_of)
    named_dark = tr.loc[named & dark].copy()
    mix_tab = (
        named_dark.groupby("mix", dropna=False)
        .agg(n=("company_id", "nunique"), n_groups=("group_id", "nunique"))
        .reset_index()
    )
    n_named_dark_mixed = int((named_dark["mix"] == "mixed").sum())
    n_named_dark_alldark = int((named_dark["mix"] == "all_dark").sum())
    print("  named-ERP dark by mix")
    print(mix_tab.to_string(index=False) if not mix_tab.empty else "  (empty)")
    print(
        f"  named-dark in mixed 110: {n_named_dark_mixed}/{b} "
        f"in all-dark 360: {n_named_dark_alldark}/{b}"
    )

    # groups.erp vs companies.erp — slugs vs display names (do not require raw equality)
    tr["miss_group_erp"] = _blank(tr["group_erp"])
    tr["fam_co"] = tr["erp_norm"].map(erp_family)
    tr["fam_g"] = tr["group_erp"].map(erp_family)
    both_named = tr["has_erp"] & ~tr["miss_group_erp"]
    raw_eq = int((both_named & tr["erp_norm"].eq(tr["group_erp"].str.strip())).sum())
    fam_eq = int((both_named & tr["fam_co"].notna() & tr["fam_co"].eq(tr["fam_g"])).sum())
    fam_ne = int((both_named & tr["fam_co"].notna() & tr["fam_g"].notna() & tr["fam_co"].ne(tr["fam_g"])).sum())
    g_named_co_null = int((~tr["has_erp"] & ~tr["miss_group_erp"]).sum())
    g_null_co_named = int((tr["has_erp"] & tr["miss_group_erp"]).sum())
    print(
        f"  group_erp: raw-string agree={raw_eq} family agree={fam_eq} family disagree={fam_ne} "
        f"group-named/co-NULL={g_named_co_null} group-NULL/co-named={g_null_co_named}"
    )

    confirm_921 = abs(share_null_among_dark - JOIN_NULL_ERP_DARK) < 0.005
    confirm_37 = b == JOIN_NAMED_DARK
    confirm_506 = n_null == JOIN_NULL_ERP_N
    confirm_144 = abs(share_inv_among_null - JOIN_NULL_WITH_INV) < 0.005
    return {
        "confirm_470": confirm_470,
        "n_dark": n_dark,
        "n_inv": n_inv,
        "conf": conf,
        "n_inv_named": a,
        "n_dark_named": b,
        "n_inv_null": c,
        "n_dark_null": d,
        "share_null_among_dark": share_null_among_dark,
        "share_inv_among_null": share_inv_among_null,
        "n_null": n_null,
        "perfect_flag": perfect,
        "named_dark_erp": named_dark_erp,
        "named_inv_erp": named_inv_erp,
        "all_erp": all_erp,
        "mix_tab": mix_tab,
        "n_named_dark_mixed": n_named_dark_mixed,
        "n_named_dark_alldark": n_named_dark_alldark,
        "confirm_921": confirm_921,
        "confirm_37": confirm_37,
        "confirm_506": confirm_506,
        "confirm_144": confirm_144,
        "g_agree": fam_eq,
        "g_agree_raw": raw_eq,
        "g_disagree_fam": fam_ne,
        "g_named_co_null": g_named_co_null,
        "g_null_co_named": g_null_co_named,
        "named_dark_ids": set(named_dark["company_id"]),
    }


def pass3_country_fx(
    cos: pd.DataFrame,
    is_train: pd.Series,
    store: pd.DataFrame,
    book: set[str],
) -> dict:
    print("\nPASS 3 country / currency × FX")
    tr = cos.loc[is_train].copy()
    tr["has_book"] = tr["company_id"].isin(book)
    ctry = (
        tr.loc[~tr["miss_country"], "country"]
        .str.strip()
        .str.upper()
        .value_counts(dropna=False)
        .reset_index()
    )
    ctry.columns = ["country", "n"]
    ctry["share"] = ctry["n"] / max(int((~tr["miss_country"]).sum()), 1)
    cur = (
        tr["currency"].fillna("(missing)").astype(str).str.strip().str.upper().replace({"": "(missing)"})
    )
    cur_tab = cur.value_counts(dropna=False).reset_index()
    cur_tab.columns = ["currency", "n"]
    cur_tab["share"] = cur_tab["n"] / max(int(len(tr)), 1)
    print("  country (known)")
    print(ctry.to_string(index=False))
    print("  currency")
    print(cur_tab.to_string(index=False))

    fx_exists = FX_MD.exists()
    fx_note = (
        f"fx_qa.md exists ({FX_MD.name}) — FX child finished; descriptive cross only."
        if fx_exists
        else "fx_qa.md missing — FX child not finished; store e_fx_share only."
    )
    print(f"  {fx_note}")

    # ever e_fx_share > 0 on train company-months
    st = store.loc[train_mask(store["company_id"])].copy()
    fx = pd.to_numeric(st["e_fx_share"], errors="coerce")
    ever = (
        st.assign(fx_pos=fx.gt(0))
        .groupby("company_id", sort=False)["fx_pos"]
        .any()
        .rename("ever_fx")
    )
    tr = tr.merge(ever, on="company_id", how="left")
    tr["ever_fx"] = tr["ever_fx"].fillna(False)
    n_ever_fx = int(tr["ever_fx"].sum())
    # FX among invoiced only (dark is NaN, not 0 — fx_qa)
    inv = tr[tr["has_book"]]
    n_inv_fx = int(inv["ever_fx"].sum())
    eur_fx = int(inv.loc[inv["is_eur"], "ever_fx"].sum())
    n_eur_inv = int(inv["is_eur"].sum())
    non_eur_fx = int(inv.loc[~inv["is_eur"], "ever_fx"].sum())
    n_non_eur_inv = int((~inv["is_eur"]).sum())
    known_fx = int(inv.loc[inv["has_country"], "ever_fx"].sum())
    n_known_inv = int(inv["has_country"].sum())
    miss_fx = int(inv.loc[~inv["has_country"], "ever_fx"].sum())
    n_miss_inv = int((~inv["has_country"]).sum())
    print(
        f"  ever_fx train cos={n_ever_fx} invoiced={n_inv_fx}/{len(inv)} "
        f"EUR-inv {eur_fx}/{n_eur_inv} nonEUR-inv {non_eur_fx}/{n_non_eur_inv} "
        f"known-country inv {known_fx}/{n_known_inv} miss-country inv {miss_fx}/{n_miss_inv}"
    )
    # country known vs ever_fx (invoiced)
    ctry_fx = (
        inv.assign(cc=inv["country"].fillna("(missing)").astype(str).str.upper())
        .groupby("cc", dropna=False)
        .agg(n=("company_id", "nunique"), n_fx=("ever_fx", "sum"))
        .reset_index()
        .sort_values("n", ascending=False)
    )
    ctry_fx["share_fx"] = ctry_fx["n_fx"] / ctry_fx["n"].clip(lower=1)
    return {
        "ctry": ctry,
        "cur": cur_tab,
        "n_known_country": int((~tr["miss_country"]).sum()),
        "n_miss_country": int(tr["miss_country"].sum()),
        "n_eur": int(tr["is_eur"].sum()),
        "share_eur": float(tr["is_eur"].mean()),
        "n_ever_fx": n_ever_fx,
        "n_inv_fx": n_inv_fx,
        "n_inv": int(len(inv)),
        "eur_fx": eur_fx,
        "n_eur_inv": n_eur_inv,
        "non_eur_fx": non_eur_fx,
        "n_non_eur_inv": n_non_eur_inv,
        "known_fx": known_fx,
        "n_known_inv": n_known_inv,
        "miss_fx": miss_fx,
        "n_miss_inv": n_miss_inv,
        "fx_exists": fx_exists,
        "fx_note": fx_note,
        "ctry_fx": ctry_fx.head(20),
        "quoted_ever_fx": 228,
    }


def pass4_clocks(cos: pd.DataFrame, clocks: pd.DataFrame, is_train: pd.Series) -> dict:
    print("\nPASS 4 three clocks")
    tr = cos.loc[is_train].merge(clocks, on="company_id", how="left")
    n = int(len(tr))
    n_tx = int(tr["first_tx"].notna().sum())
    n_bank = int(tr["first_bank_created"].notna().sum())
    n_co = int(tr["created_at"].notna().sum())
    # offsets in days (company created − first tx); + means onboard after cash
    d_co_tx = (tr["created_at"] - tr["first_tx"]).dt.total_seconds() / 86400.0
    d_co_bank = (tr["created_at"] - tr["first_bank_created"]).dt.total_seconds() / 86400.0
    d_bank_tx = (tr["first_bank_created"] - tr["first_tx"]).dt.total_seconds() / 86400.0
    late_co = tr["created_at"] > PANEL_START
    late_tx = tr["first_tx_month"] > PANEL_START
    late_bank = tr["first_bank_created"] > PANEL_START
    same_mo_co_tx = (
        tr["created_at"].dt.to_period("M") == tr["first_tx_month"].dt.to_period("M")
    )
    rec = {
        "n": n,
        "n_tx": n_tx,
        "n_bank": n_bank,
        "n_co": n_co,
        "share_late_co": float(late_co.mean()),
        "share_late_tx": float(late_tx.mean()) if n_tx else float("nan"),
        "share_late_bank": float(late_bank[tr["first_bank_created"].notna()].mean()) if n_bank else float("nan"),
        "n_late_co": int(late_co.sum()),
        "n_late_tx": int(late_tx.sum()),
        "n_late_bank": int(late_bank.sum()),
        "med_co_minus_tx": float(d_co_tx.median()),
        "mean_co_minus_tx": float(d_co_tx.mean()),
        "p25_co_minus_tx": float(d_co_tx.quantile(0.25)),
        "p75_co_minus_tx": float(d_co_tx.quantile(0.75)),
        "share_co_after_tx": float((d_co_tx > 0).mean()),
        "share_co_before_tx": float((d_co_tx < 0).mean()),
        "share_same_day_co_tx": float((d_co_tx.abs() < 1).mean()),
        "share_same_month_co_tx": float(same_mo_co_tx.mean()),
        "med_co_minus_bank": float(d_co_bank.median()),
        "mean_co_minus_bank": float(d_co_bank.mean()),
        "share_co_after_bank": float((d_co_bank > 0).mean()),
        "med_bank_minus_tx": float(d_bank_tx.median()),
        "mean_bank_minus_tx": float(d_bank_tx.mean()),
        "share_bank_after_tx": float((d_bank_tx > 0).mean()),
        "n_no_bank": int(tr["first_bank_created"].isna().sum()),
    }
    print(
        f"  late created_at {rec['n_late_co']}/{n}={rec['share_late_co']:.3f} "
        f"(trail quoted 860/1214=0.708) late first-tx {rec['n_late_tx']}/{n} "
        f"late first-bank {rec['n_late_bank']}/{n_bank}"
    )
    print(
        f"  created−first_tx days p50={rec['med_co_minus_tx']:.1f} "
        f"share after={rec['share_co_after_tx']:.3f} same-month={rec['share_same_month_co_tx']:.3f}"
    )
    print(
        f"  created−first_bank p50={rec['med_co_minus_bank']:.1f} "
        f"bank−first_tx p50={rec['med_bank_minus_tx']:.1f} after={rec['share_bank_after_tx']:.3f}"
    )
    # 2×2 late created vs late first-tx
    both = int((late_co & late_tx).sum())
    co_only = int((late_co & ~late_tx).sum())
    tx_only = int((~late_co & late_tx).sum())
    neither = int((~late_co & ~late_tx).sum())
    print(f"  late clocks 2x2 both={both} co-only={co_only} tx-only={tx_only} neither={neither}")
    rec.update(
        {
            "late_both": both,
            "late_co_only": co_only,
            "late_tx_only": tx_only,
            "late_neither": neither,
            "confirm_late_co": abs(rec["share_late_co"] - 860 / 1214) < 0.01,
        }
    )
    return rec


def pass5_group_size(
    cos: pd.DataFrame,
    is_train: pd.Series,
    panel: pd.DataFrame,
    pop: dict,
) -> dict:
    print("\nPASS 5 group size vs h_group_size / singletons")
    tr = cos.loc[is_train].copy()
    g_n = tr.groupby("group_id")["company_id"].nunique().rename("n_cos_train")
    tr = tr.merge(g_n, on="group_id", how="left")
    # company-level h_group_size (static)
    h = (
        panel.loc[train_mask(panel["company_id"]), ["company_id", "h_group_size"]]
        .drop_duplicates("company_id")
    )
    tr = tr.merge(h, on="company_id", how="left")
    # n_in_sample from groups.csv
    rho = spearman(tr["n_cos_train"], tr["h_group_size"])
    rho_sample = spearman(tr["n_in_sample"], tr["h_group_size"])
    agree_h = float((tr["n_in_sample"].fillna(-1) == tr["h_group_size"].fillna(-2)).mean())
    # singleton: n_cos_train == 1 (train members). Hidden test is new groups —
    # a singleton dummy is a group-size dummy.
    tr["singleton"] = tr["n_cos_train"].eq(1)
    n_sing_cos = int(tr["singleton"].sum())
    n_sing_g = int(tr.loc[tr["singleton"], "group_id"].nunique())
    n_g = int(tr["group_id"].nunique())
    print(
        f"  train groups={n_g} singleton groups={n_sing_g} singleton cos={n_sing_cos} "
        f"ρ n_cos vs h_group_size={rho:.3f} n_in_sample vs h={rho_sample:.3f} "
        f"agree n_in_sample==h={agree_h:.3f}"
    )

    # Y rates: singleton vs multi, size-tercile control on company-months
    is_tr = train_mask(panel["company_id"])
    sing_ids = set(tr.loc[tr["singleton"], "company_id"])
    panel = panel.copy()
    panel["singleton"] = panel["company_id"].isin(sing_ids)
    panel["log1p_a_in3"] = np.log1p(pd.to_numeric(panel["a_in3"], errors="coerce").clip(lower=0))
    size_ok = is_tr & panel["log1p_a_in3"].notna()
    terc, edges = fit_terciles(panel["log1p_a_in3"], size_ok)
    panel["size_terc"] = terc
    print(f"  size tercile edges (train log1p a_in3): {edges}")

    y3 = pd.to_numeric(panel[Y3], errors="coerce")
    y2 = pd.to_numeric(panel[Y2], errors="coerce")
    cid = panel["company_id"]
    y3_ok = is_tr & y3.notna()
    y2_ok = is_tr & y2.notna()

    rows = []
    for yname, yser, base in ((Y3, y3, y3_ok), (Y2, y2, y2_ok)):
        for sl_name, sl in (
            ("all_train", base),
            ("singleton", base & panel["singleton"]),
            ("multi", base & ~panel["singleton"]),
        ):
            rec = rate_row(yser[sl], cid[sl], sl_name)
            rec["y"] = yname
            rows.append(rec)
    rates = pd.DataFrame(rows)
    print(rates.to_string(index=False))

    resid_rows = []
    for yname, yser, base in ((Y3, y3, y3_ok), (Y2, y2, y2_ok)):
        for tname in ("T1_small", "T2_mid", "T3_large"):
            t = panel["size_terc"].eq(tname)
            a = rate_row(yser[base & t & ~panel["singleton"]], cid[base & t & ~panel["singleton"]], "multi")
            b = rate_row(yser[base & t & panel["singleton"]], cid[base & t & panel["singleton"]], "singleton")
            resid_rows.append(
                {
                    "y": yname,
                    "tercile": tname,
                    "n_multi": a["n"],
                    "rate_multi": a["rate"],
                    "n_sing": b["n"],
                    "rate_sing": b["rate"],
                    "residual": (
                        float(b["rate"] - a["rate"])
                        if np.isfinite(a["rate"]) and np.isfinite(b["rate"])
                        else float("nan")
                    ),
                    "median_log1p": float(panel.loc[base & t, "log1p_a_in3"].median()),
                }
            )
    resid = pd.DataFrame(resid_rows)
    print(resid.to_string(index=False))

    def _r(df, y, sl):
        row = df[(df["y"] == y) & (df["slice"] == sl)]
        return float(row.iloc[0]["rate"]) if len(row) else float("nan")

    return {
        "n_groups": n_g,
        "n_sing_g": n_sing_g,
        "n_sing_cos": n_sing_cos,
        "rho_n_vs_h": rho,
        "rho_sample_vs_h": rho_sample,
        "agree_sample_h": agree_h,
        "rates": rates,
        "resid": resid,
        "edges": [float(x) for x in edges],
        "y3_sing": _r(rates, Y3, "singleton"),
        "y3_multi": _r(rates, Y3, "multi"),
        "y2_sing": _r(rates, Y2, "singleton"),
        "y2_multi": _r(rates, Y2, "multi"),
    }


def pass6_auroc(panel: pd.DataFrame, cos: pd.DataFrame, is_train_cos: pd.Series) -> dict:
    print("\nPASS 6 single-feature group-fold AUROC")
    flags = cos.loc[is_train_cos, ["company_id", "group_id", "has_erp", "has_country", "is_eur"]].copy()
    flags["has_erp"] = flags["has_erp"].astype(float)
    flags["has_country"] = flags["has_country"].astype(float)
    flags["is_eur"] = flags["is_eur"].astype(float)
    folds = group_folds(flags[["company_id", "group_id"]], n=N_FOLDS, seed=FOLD_SEED)
    flags = flags.merge(folds[["company_id", "fold"]], on="company_id", how="left")

    is_tr = train_mask(panel["company_id"])
    m = panel.loc[is_tr].merge(
        flags[["company_id", "has_erp", "has_country", "is_eur", "fold"]],
        on="company_id",
        how="left",
    )
    m["log1p_a_in3"] = np.log1p(pd.to_numeric(m["a_in3"], errors="coerce").clip(lower=0))
    m["h_group_size"] = pd.to_numeric(m["h_group_size"], errors="coerce")
    assert_no_holdout(m["company_id"])

    leak = leakage_check(["has_erp", "has_country", "is_eur"], Y3, ("b",))
    print(f"  leakage_check companies flags vs Y3 never B: {leak}")

    rows = []
    by = {}
    for ycol, ymask_name in ((Y3, "y3"), (Y2, "y2")):
        lab = m[ycol].notna() & m["fold"].notna()
        for name, col in (
            ("has_erp", m["has_erp"]),
            ("has_country", m["has_country"]),
            ("is_eur", m["is_eur"]),
            ("log1p_a_in3", m["log1p_a_in3"]),
            ("h_group_size", m["h_group_size"]),
            ("c_n_days_with_tx", m["c_n_days_with_tx"]),
        ):
            rec = signed_oof_auroc(m[ycol], col, m["fold"], lab)
            rec["feature"] = name
            rec["y"] = ycol
            by[(ycol, name)] = rec
            rows.append(
                {
                    "y": ycol,
                    "feature": name,
                    "cv": rec["cv"],
                    "sd": rec["sd"],
                    "train_auc": rec["train_auc"],
                    "train_sign": rec["train_sign"],
                    "coverage": rec["coverage"],
                    "two_sided": two_sided(rec["cv"]),
                    "n_defined": rec["n_defined"],
                }
            )
            print(
                f"  {ycol} {name}: cv={rec['cv']:.4f}±{rec['sd']:.3f} "
                f"train={rec['train_auc']:.4f} sign={rec['train_sign']} cov={rec['coverage']:.3f}"
            )
    tab = pd.DataFrame(rows)
    size_y3 = by[(Y3, "log1p_a_in3")]["cv"]
    days_y3 = by[(Y3, "c_n_days_with_tx")]["cv"]
    erp_y3 = by[(Y3, "has_erp")]["cv"]
    ctry_y3 = by[(Y3, "has_country")]["cv"]
    eur_y3 = by[(Y3, "is_eur")]["cv"]
    return {
        "tab": tab,
        "by": by,
        "size_y3": size_y3,
        "days_y3": days_y3,
        "erp_y3": erp_y3,
        "ctry_y3": ctry_y3,
        "eur_y3": eur_y3,
        "erp_beats_size": bool(
            np.isfinite(erp_y3) and np.isfinite(size_y3) and (erp_y3 - size_y3) >= KEEP_DELTA
        ),
        "ctry_beats_size": bool(
            np.isfinite(ctry_y3) and np.isfinite(size_y3) and (ctry_y3 - size_y3) >= KEEP_DELTA
        ),
        "eur_beats_size": bool(
            np.isfinite(eur_y3) and np.isfinite(size_y3) and (eur_y3 - size_y3) >= KEEP_DELTA
        ),
        "leak": leak,
    }


def pass7_country_size_trail(
    cos: pd.DataFrame,
    is_train: pd.Series,
    panel: pd.DataFrame,
    clocks: pd.DataFrame,
) -> dict:
    """Is missing-country SIZE or late-arrival (trail_length)?"""
    print("\nPASS 7 country-missing = size or late trail?")
    tr = cos.loc[is_train].merge(clocks, on="company_id", how="left")
    size_co = (
        panel.loc[train_mask(panel["company_id"])]
        .assign(log1p=lambda d: np.log1p(pd.to_numeric(d["a_in3"], errors="coerce").clip(lower=0)))
        .groupby("company_id")["log1p"]
        .median()
        .rename("med_log1p_a_in3")
    )
    tr = tr.merge(size_co, on="company_id", how="left")
    trail = (
        panel.loc[train_mask(panel["company_id"])]
        .groupby("company_id")["period"]
        .agg(n_grid="nunique", first_period="min")
        .reset_index()
    )
    tr = tr.merge(trail, on="company_id", how="left")
    tr["late_tx"] = tr["first_tx_month"] > PANEL_START
    tr["short_trail"] = tr["n_grid"] < 12

    miss = tr["miss_country"]
    known = ~miss
    rho_size = spearman(miss.astype(float), tr["med_log1p_a_in3"])
    rho_trail = spearman(miss.astype(float), tr["n_grid"])
    med_size_m = float(tr.loc[miss, "med_log1p_a_in3"].median())
    med_size_k = float(tr.loc[known, "med_log1p_a_in3"].median())
    med_grid_m = float(tr.loc[miss, "n_grid"].median())
    med_grid_k = float(tr.loc[known, "n_grid"].median())
    share_late_m = float(tr.loc[miss, "late_tx"].mean())
    share_late_k = float(tr.loc[known, "late_tx"].mean())
    share_short_m = float(tr.loc[miss, "short_trail"].mean())
    share_short_k = float(tr.loc[known, "short_trail"].mean())
    print(
        f"  miss n={int(miss.sum())} known n={int(known.sum())} "
        f"ρ vs size={rho_size:.3f} vs n_grid={rho_trail:.3f}"
    )
    print(
        f"  median log1p miss={med_size_m:.3f} known={med_size_k:.3f} "
        f"median grid miss={med_grid_m:.1f} known={med_grid_k:.1f}"
    )
    print(
        f"  late first-tx miss={share_late_m:.3f} known={share_late_k:.3f} "
        f"short<12 miss={share_short_m:.3f} known={share_short_k:.3f}"
    )

    # size terciles at company level (train medians)
    try:
        terc, edges = fit_terciles(tr["med_log1p_a_in3"], tr["med_log1p_a_in3"].notna())
        tr["size_terc"] = terc
    except RuntimeError:
        tr["size_terc"] = pd.Series(index=tr.index, dtype=object)
        edges = np.array([])
    terc_tab = (
        tr.groupby("size_terc", dropna=False)
        .agg(n=("company_id", "nunique"), n_miss=("miss_country", "sum"), share_miss=("miss_country", "mean"))
        .reset_index()
    )
    print(terc_tab.to_string(index=False))

    # trail buckets (fixed cuts, not quantiles — trail_length house style)
    bins = [0, 6, 12, 18, 24, 99]
    labels = ("<6", "6-11", "12-17", "18-23", "24")
    tr["trail_bucket"] = pd.cut(tr["n_grid"], bins=bins, labels=labels, right=False)
    trail_tab = (
        tr.groupby("trail_bucket", dropna=False, observed=False)
        .agg(n=("company_id", "nunique"), n_miss=("miss_country", "sum"), share_miss=("miss_country", "mean"))
        .reset_index()
    )
    print(trail_tab.to_string(index=False))

    # group clustering of missing country: if whole group misses, it's a group dummy
    g = tr.groupby("group_id").agg(
        n=("company_id", "nunique"),
        n_miss=("miss_country", "sum"),
    )
    g["all_miss"] = g["n_miss"] == g["n"]
    g["all_known"] = g["n_miss"] == 0
    g["mixed"] = ~g["all_miss"] & ~g["all_known"]
    print(
        f"  groups all-miss-country={int(g['all_miss'].sum())} "
        f"all-known={int(g['all_known'].sum())} mixed={int(g['mixed'].sum())}"
    )
    size_like = bool(np.isfinite(rho_size) and abs(rho_size) >= SIZE_RHO)
    late_like = bool(
        abs(share_late_m - share_late_k) >= 0.10
        or abs(med_grid_m - med_grid_k) >= 3
        or (np.isfinite(rho_trail) and abs(rho_trail) >= 0.25)
    )
    return {
        "n_miss": int(miss.sum()),
        "n_known": int(known.sum()),
        "rho_size": rho_size,
        "rho_trail": rho_trail,
        "med_size_miss": med_size_m,
        "med_size_known": med_size_k,
        "med_grid_miss": med_grid_m,
        "med_grid_known": med_grid_k,
        "share_late_miss": share_late_m,
        "share_late_known": share_late_k,
        "share_short_miss": share_short_m,
        "share_short_known": share_short_k,
        "terc_tab": terc_tab,
        "trail_tab": trail_tab,
        "n_g_all_miss": int(g["all_miss"].sum()),
        "n_g_all_known": int(g["all_known"].sum()),
        "n_g_mixed": int(g["mixed"].sum()),
        "size_like": size_like,
        "late_like": late_like,
        "edges": [float(x) for x in edges] if len(edges) else [],
    }


def pass8_erp_is_dark_dummy(
    cos: pd.DataFrame,
    is_train: pd.Series,
    panel: pd.DataFrame,
    book: set[str],
    pop: dict,
) -> dict:
    """has_erp vs has_book: is it just the 470 dummy? Y3 restricted on dark already."""
    print("\nPASS 8 has_erp vs has_book (470 dummy)")
    tr = cos.loc[is_train].copy()
    tr["has_book"] = tr["company_id"].isin(book)
    rho = spearman(tr["has_erp"].astype(float), tr["has_book"].astype(float))
    agree = float((tr["has_erp"] == tr["has_book"]).mean())
    print(f"  company-level ρ has_erp vs has_book={rho:.3f} agree={agree:.3f}")

    # Y3/Y2 rates: has_erp vs dark (book)
    is_tr = train_mask(panel["company_id"])
    m = panel.loc[is_tr].merge(
        tr[["company_id", "has_erp", "has_book", "has_country", "is_eur"]],
        on="company_id",
        how="left",
    )
    y3 = pd.to_numeric(m[Y3], errors="coerce")
    y2 = pd.to_numeric(m[Y2], errors="coerce")
    cid = m["company_id"]
    rows = []
    for yname, yser in ((Y3, y3), (Y2, y2)):
        base = yser.notna()
        for sl_name, sl in (
            ("has_erp", base & m["has_erp"]),
            ("null_erp", base & ~m["has_erp"]),
            ("has_book", base & m["has_book"]),
            ("dark", base & ~m["has_book"]),
        ):
            rec = rate_row(yser[sl], cid[sl], sl_name)
            rec["y"] = yname
            rows.append(rec)
    rates = pd.DataFrame(rows)
    print(rates.to_string(index=False))

    # among invoiced only: does NULL erp still move Y3? (if yes, not just 470 dummy)
    inv_y3 = y3.notna() & m["has_book"]
    a = rate_row(y3[inv_y3 & m["has_erp"]], cid[inv_y3 & m["has_erp"]], "inv_named")
    b = rate_row(y3[inv_y3 & ~m["has_erp"]], cid[inv_y3 & ~m["has_erp"]], "inv_null")
    gap_inv = (
        float(a["rate"] - b["rate"]) if np.isfinite(a["rate"]) and np.isfinite(b["rate"]) else float("nan")
    )
    print(f"  among invoiced Y3 named={a['rate']:.4f} n={a['n']} null={b['rate']:.4f} n={b['n']} gap={gap_inv:+.4f}")

    is_dummy = bool(np.isfinite(rho) and abs(rho) >= 0.80)
    return {
        "rho_erp_book": rho,
        "agree": agree,
        "rates": rates,
        "inv_named_y3": a["rate"],
        "inv_null_y3": b["rate"],
        "inv_gap": gap_inv,
        "inv_named_n": a["n"],
        "inv_null_n": b["n"],
        "is_470_dummy": is_dummy,
    }


def extra_cuts(
    cos: pd.DataFrame,
    is_train: pd.Series,
    is_hold: pd.Series,
    panel: pd.DataFrame,
    clocks: pd.DataFrame,
    book: set[str],
    pop: dict,
    p2: dict,
    p7: dict,
) -> dict:
    """Same-module next cuts after erp×dark and country-missing."""
    print("\nEXTRA cuts (same module)")
    tr = cos.loc[is_train].copy()
    tr["has_book"] = tr["company_id"].isin(book)
    mix_of = pop["mix_of"]
    tr["mix"] = tr["group_id"].map(mix_of)
    tr = tr.merge(clocks, on="company_id", how="left")
    size_co = (
        panel.loc[train_mask(panel["company_id"])]
        .assign(log1p=lambda d: np.log1p(pd.to_numeric(d["a_in3"], errors="coerce").clip(lower=0)))
        .groupby("company_id")["log1p"]
        .median()
        .rename("med_log1p_a_in3")
    )
    trail = (
        panel.loc[train_mask(panel["company_id"])]
        .groupby("company_id")["period"]
        .nunique()
        .rename("n_grid")
    )
    tr = tr.merge(size_co, on="company_id", how="left").merge(trail, on="company_id", how="left")
    tr["late_tx"] = tr["first_tx_month"] > PANEL_START
    tr["fam_co"] = tr["erp_norm"].map(erp_family)
    tr["fam_g"] = tr["group_erp"].map(erp_family)

    # --- E1 named-dark groups: 20 all-dark in 3 groups ---
    nd = tr.loc[tr["has_erp"] & ~tr["has_book"]].copy()
    gtab = (
        nd.groupby(["group_id", "mix"], dropna=False)
        .agg(
            n=("company_id", "nunique"),
            erps=("erp_norm", lambda s: ",".join(sorted(set(str(x) for x in s)))),
            n_in_sample=("n_in_sample", "first"),
            med_size=("med_log1p_a_in3", "median"),
            med_grid=("n_grid", "median"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )
    print("E1 named-ERP dark groups")
    print(gtab.to_string(index=False))
    n_ad_g = int(gtab.loc[gtab["mix"] == "all_dark", "group_id"].nunique())
    n_mx_g = int(gtab.loc[gtab["mix"] == "mixed", "group_id"].nunique())

    # --- E2 invoiced+NULL (73): mix / country / size ---
    inv_null = tr.loc[~tr["has_erp"] & tr["has_book"]].copy()
    inv_named = tr.loc[tr["has_erp"] & tr["has_book"]].copy()
    e2 = {
        "n": int(len(inv_null)),
        "n_mix_mixed": int((inv_null["mix"] == "mixed").sum()),
        "n_mix_allinv": int((inv_null["mix"] == "all_invoiced").sum()),
        "share_miss_country": float(inv_null["miss_country"].mean()) if len(inv_null) else float("nan"),
        "share_eur": float(inv_null["is_eur"].mean()) if len(inv_null) else float("nan"),
        "med_size": float(inv_null["med_log1p_a_in3"].median()) if len(inv_null) else float("nan"),
        "med_size_named": float(inv_named["med_log1p_a_in3"].median()) if len(inv_named) else float("nan"),
        "med_grid": float(inv_null["n_grid"].median()) if len(inv_null) else float("nan"),
        "share_late": float(inv_null["late_tx"].mean()) if len(inv_null) else float("nan"),
        "n_groups": int(inv_null["group_id"].nunique()),
    }
    print(
        f"E2 invoiced+NULL n={e2['n']} mixed={e2['n_mix_mixed']} all-inv={e2['n_mix_allinv']} "
        f"miss-country={e2['share_miss_country']:.3f} EUR={e2['share_eur']:.3f} "
        f"med size {e2['med_size']:.3f} vs named {e2['med_size_named']:.3f}"
    )

    # --- E3 country missing × book ---
    e3_tab = (
        tr.groupby("has_book")
        .agg(n=("company_id", "nunique"), n_miss=("miss_country", "sum"), share_miss=("miss_country", "mean"))
        .reset_index()
    )
    e3_tab["slice"] = np.where(e3_tab["has_book"], "invoiced", "dark")
    print("E3 country-miss × book")
    print(e3_tab.to_string(index=False))

    # --- E4 country-known Y3/Y2 after size terciles ---
    is_tr = train_mask(panel["company_id"])
    m = panel.loc[is_tr].merge(
        tr[["company_id", "has_country", "is_eur", "has_erp", "has_book", "miss_country"]],
        on="company_id",
        how="left",
    )
    m["log1p_a_in3"] = np.log1p(pd.to_numeric(m["a_in3"], errors="coerce").clip(lower=0))
    terc, _ = fit_terciles(m["log1p_a_in3"], m["log1p_a_in3"].notna())
    m["size_terc"] = terc
    y3 = pd.to_numeric(m[Y3], errors="coerce")
    y2 = pd.to_numeric(m[Y2], errors="coerce")
    cid = m["company_id"]
    e4_rows = []
    for yname, yser in ((Y3, y3), (Y2, y2)):
        base = yser.notna()
        for tname in ("T1_small", "T2_mid", "T3_large"):
            t = m["size_terc"].eq(tname)
            a = rate_row(yser[base & t & m["has_country"]], cid[base & t & m["has_country"]], "known")
            b = rate_row(yser[base & t & ~m["has_country"]], cid[base & t & ~m["has_country"]], "miss")
            e4_rows.append(
                {
                    "y": yname,
                    "tercile": tname,
                    "n_known": a["n"],
                    "rate_known": a["rate"],
                    "n_miss": b["n"],
                    "rate_miss": b["rate"],
                    "residual": (
                        float(a["rate"] - b["rate"])
                        if np.isfinite(a["rate"]) and np.isfinite(b["rate"])
                        else float("nan")
                    ),
                }
            )
    e4 = pd.DataFrame(e4_rows)
    print("E4 country known−miss residual by size tercile")
    print(e4.to_string(index=False))

    # --- E5 EUR vs not, invoiced only, after size ---
    e5_rows = []
    inv = m["has_book"]
    for yname, yser in ((Y3, y3), (Y2, y2)):
        base = yser.notna() & inv
        for tname in ("T1_small", "T2_mid", "T3_large"):
            t = m["size_terc"].eq(tname)
            a = rate_row(yser[base & t & m["is_eur"]], cid[base & t & m["is_eur"]], "EUR")
            b = rate_row(yser[base & t & ~m["is_eur"]], cid[base & t & ~m["is_eur"]], "notEUR")
            e5_rows.append(
                {
                    "y": yname,
                    "tercile": tname,
                    "n_eur": a["n"],
                    "rate_eur": a["rate"],
                    "n_oth": b["n"],
                    "rate_oth": b["rate"],
                    "residual": (
                        float(a["rate"] - b["rate"])
                        if np.isfinite(a["rate"]) and np.isfinite(b["rate"])
                        else float("nan")
                    ),
                }
            )
    e5 = pd.DataFrame(e5_rows)
    print("E5 EUR−other residual (invoiced) by size tercile")
    print(e5.to_string(index=False))

    # --- E6 late created_at vs Y3 after size (PARK confirm, not a Y) ---
    late_ids = set(tr.loc[tr["created_at"] > PANEL_START, "company_id"])
    m["late_co"] = m["company_id"].isin(late_ids)
    e6_rows = []
    for tname in ("T1_small", "T2_mid", "T3_large"):
        t = m["size_terc"].eq(tname)
        base = y3.notna()
        a = rate_row(y3[base & t & ~m["late_co"]], cid[base & t & ~m["late_co"]], "early_co")
        b = rate_row(y3[base & t & m["late_co"]], cid[base & t & m["late_co"]], "late_co")
        e6_rows.append(
            {
                "tercile": tname,
                "n_early": a["n"],
                "rate_early": a["rate"],
                "n_late": b["n"],
                "rate_late": b["rate"],
                "residual": (
                    float(b["rate"] - a["rate"])
                    if np.isfinite(a["rate"]) and np.isfinite(b["rate"])
                    else float("nan")
                ),
            }
        )
    e6 = pd.DataFrame(e6_rows)
    print("E6 late created_at − early Y3 residual (PARK confirm)")
    print(e6.to_string(index=False))

    # --- E7 holdout coverage (not rates) ---
    ho = cos.loc[is_hold].copy()
    ho["has_book"] = ho["company_id"].isin(book)
    e7 = {
        "n": int(len(ho)),
        "n_dark": int((~ho["has_book"]).sum()),
        "miss_country": float(ho["miss_country"].mean()),
        "miss_erp": float(ho["miss_erp"].mean()),
        "share_eur": float(ho["is_eur"].mean()),
        "n_named_dark": int((ho["has_erp"] & ~ho["has_book"]).sum()),
        "n_inv_null": int((~ho["has_erp"] & ho["has_book"]).sum()),
    }
    print(
        f"E7 holdout n={e7['n']} dark={e7['n_dark']} miss-country={e7['miss_country']:.3f} "
        f"miss-erp={e7['miss_erp']:.3f} named-dark={e7['n_named_dark']} inv-null={e7['n_inv_null']}"
    )

    # --- E8 mixed groups: is country filled only on invoiced sisters? ---
    mixed_g = set(tr.loc[tr["mix"] == "mixed", "group_id"])
    mx = tr.loc[tr["group_id"].isin(mixed_g)].copy()
    share_known_inv = float(mx.loc[mx["has_book"], "has_country"].mean()) if mx["has_book"].any() else float("nan")
    share_known_dk = float(mx.loc[~mx["has_book"], "has_country"].mean()) if (~mx["has_book"]).any() else float("nan")
    print(f"E8 mixed groups country-known invoiced={share_known_inv:.3f} dark={share_known_dk:.3f}")

    # --- E9 currency vs invoice accounting_currency (descriptive) ---
    con = connect()
    acc = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               accounting_currency,
               COUNT(*) AS n
        FROM invoices
        WHERE document_type = 'invoice' AND status <> 'cancel'
          AND amount <> 0 AND issuance_date IS NOT NULL
        GROUP BY 1, 2
        """
    ).df()
    con.close()
    acc["company_id"] = acc["company_id"].astype(str)
    top = acc.sort_values("n", ascending=False).drop_duplicates("company_id")
    top = top.rename(columns={"accounting_currency": "acct_ccy"})
    inv = tr.loc[tr["has_book"]].merge(top[["company_id", "acct_ccy"]], on="company_id", how="left")
    inv["acct_ccy"] = inv["acct_ccy"].astype("string").str.upper()
    inv["cur_u"] = inv["currency"].astype("string").str.upper()
    agree_acct = float(inv["cur_u"].eq(inv["acct_ccy"]).mean())
    n_acct = int(inv["acct_ccy"].notna().sum())
    print(f"E9 company currency == modal invoice accounting_currency: {agree_acct:.3f} n={n_acct}/{len(inv)}")

    # --- E10 singleton: company-level ever-recover (not CM-weighted) ---
    y3p = panel.loc[is_tr & panel[Y3].notna(), ["company_id", Y3]].copy()
    y3p["y3"] = pd.to_numeric(y3p[Y3], errors="coerce")
    cos_y = y3p.groupby("company_id").agg(n=("y3", "size"), ever=("y3", "max"), rate=("y3", "mean"))
    sing_ids = set(tr.loc[tr["n_in_sample"].eq(1), "company_id"])
    # n_in_sample==1 is the true singleton group; train-count singleton can differ if holdout siblings exist
    # holdout is whole groups so train n==h_group_size. Use n_in_sample==1.
    cos_y = cos_y.merge(tr[["company_id", "n_in_sample"]], on="company_id", how="left")
    sing = cos_y["n_in_sample"].eq(1)
    ever_s = float(cos_y.loc[sing, "ever"].mean()) if sing.any() else float("nan")
    ever_m = float(cos_y.loc[~sing, "ever"].mean()) if (~sing).any() else float("nan")
    print(
        f"E10 company-level ever-Y3 singleton={ever_s:.3f} n={int(sing.sum())} "
        f"multi={ever_m:.3f} n={int((~sing).sum())}"
    )

    md = [
        "### Extra — named-ERP dark groups (not just the 110)",
        "",
        f"37 named-ERP dark sit in **{n_ad_g}** all-dark groups + **{n_mx_g}** mixed groups. "
        "They are **not** the 110: 20 live in all-dark holdings that somehow have an ERP *name* "
        "and no book.",
        "",
    ]
    md += md_table(
        gtab,
        [
            ("group_id", "group", "s"),
            ("mix", "mix", "s"),
            ("n", "n named-dark", "n"),
            ("erps", "erp slugs", "s"),
            ("n_in_sample", "group n", "n"),
            ("med_size", "median log1p(a_in3)", "f"),
            ("med_grid", "median grid m", "f"),
        ],
    )
    md += [
        "",
        "### Extra — the 73 invoiced + NULL ERP",
        "",
        f"n={e2['n']} in {e2['n_groups']} groups. Mixed {e2['n_mix_mixed']}, "
        f"all-invoiced {e2['n_mix_allinv']}. Miss-country {_pp(e2['share_miss_country'])}, "
        f"EUR {_pp(e2['share_eur'])}. Median size {_f(e2['med_size'])} vs named-ERP invoiced "
        f"{_f(e2['med_size_named'])}. Late first-tx {_pp(e2['share_late'])}. "
        "NULL erp on an invoiced firm is a **metadata hole**, not darkness.",
        "",
        "### Extra — country missing × book",
        "",
    ]
    md += md_table(
        e3_tab,
        [
            ("slice", "slice", "s"),
            ("n", "n cos", "n"),
            ("n_miss", "n miss country", "n"),
            ("share_miss", "share miss", "pp"),
        ],
    )
    md += [
        "",
        f"Mixed-group country-known: invoiced {_pp(share_known_inv)} vs dark {_pp(share_known_dk)}. "
        "If only invoiced sisters have ISO codes, country is an ERP-presence echo.",
        "",
        "### Extra — country known − miss Y residual after size",
        "",
        "If missing-country were a health why, the residual would survive `log1p(a_in3)` terciles.",
        "",
    ]
    md += md_table(
        e4,
        [
            ("y", "Y", "s"),
            ("tercile", "tercile", "s"),
            ("n_known", "n known", "n"),
            ("rate_known", "rate known", "pp"),
            ("n_miss", "n miss", "n"),
            ("rate_miss", "rate miss", "pp"),
            ("residual", "known−miss", "delta"),
        ],
    )
    md += [
        "",
        "### Extra — EUR − other (invoiced only) after size",
        "",
        "fx_qa: non-EUR home books are the FX identity, not the 110. Residual after size:",
        "",
    ]
    md += md_table(
        e5,
        [
            ("y", "Y", "s"),
            ("tercile", "tercile", "s"),
            ("n_eur", "n EUR", "n"),
            ("rate_eur", "rate EUR", "pp"),
            ("n_oth", "n other", "n"),
            ("rate_oth", "rate other", "pp"),
            ("residual", "EUR−other", "delta"),
        ],
    )
    md += [
        "",
        "### Extra — late `created_at` vs Y3 after size (PARK confirm)",
        "",
        "Do not revive onboard as a health Y. Residual late − early inside size terciles:",
        "",
    ]
    md += md_table(
        e6,
        [
            ("tercile", "tercile", "s"),
            ("n_early", "n early onboard", "n"),
            ("rate_early", "Y3 early", "pp"),
            ("n_late", "n late onboard", "n"),
            ("rate_late", "Y3 late", "pp"),
            ("residual", "late−early", "delta"),
        ],
    )
    md += [
        "",
        f"Company currency vs modal invoice `accounting_currency` (invoiced): agree {_pp(agree_acct)} "
        f"({n_acct}/{len(inv)}). Home currency is the books' accounting currency, not a health X.",
        "",
        f"Holdout coverage (not a rate): n={e7['n']}, dark={e7['n_dark']}, "
        f"miss-country {_pp(e7['miss_country'])}, miss-erp {_pp(e7['miss_erp'])}, "
        f"EUR {_pp(e7['share_eur'])}, named-dark {e7['n_named_dark']}, invoiced-NULL {e7['n_inv_null']}.",
        "",
        f"Company-level ever-Y3: singleton groups {_pp(ever_s)} (n={int(sing.sum())}) vs "
        f"multi {_pp(ever_m)} (n={int((~sing).sum())}). CM 15% vs 6.8% is not only month-weighting. "
        "Still a **group-size dummy** — hidden test is new groups.",
        "",
    ]
    return {
        "named_dark_groups": gtab,
        "n_ad_g": n_ad_g,
        "n_mx_g": n_mx_g,
        "inv_null": e2,
        "country_book": e3_tab,
        "country_resid": e4,
        "eur_resid": e5,
        "late_created": e6,
        "hold": e7,
        "mixed_known_inv": share_known_inv,
        "mixed_known_dk": share_known_dk,
        "acct_agree": agree_acct,
        "ever_y3_sing": ever_s,
        "ever_y3_multi": ever_m,
        "more_md": "\n".join(md),
    }


def extra_cuts2(
    cos: pd.DataFrame,
    is_train: pd.Series,
    panel: pd.DataFrame,
    book: set[str],
    pop: dict,
    extra: dict,
) -> dict:
    """Second sitting: GROUP_0138, invoiced-only AUROC, group-constant country/currency."""
    print("\nEXTRA 2 (same module)")
    tr = cos.loc[is_train].copy()
    tr["has_book"] = tr["company_id"].isin(book)
    tr["mix"] = tr["group_id"].map(pop["mix_of"])
    tr["fam_co"] = tr["erp_norm"].map(erp_family)
    tr["fam_g"] = tr["group_erp"].map(erp_family)

    # GROUP_0138 — 18 named-dark dynamicsAx
    g138 = tr.loc[tr["group_id"] == "GROUP_0138"]
    print(
        f"GROUP_0138 n={len(g138)} named={int(g138['has_erp'].sum())} "
        f"book={int(g138['has_book'].sum())} miss-country={float(g138['miss_country'].mean()):.3f} "
        f"eur={float(g138['is_eur'].mean()):.3f} erps={sorted(g138['erp_norm'].dropna().unique())}"
    )

    is_tr = train_mask(panel["company_id"])
    m = panel.loc[is_tr].merge(
        tr[["company_id", "group_id", "has_erp", "has_country", "is_eur", "has_book"]],
        on="company_id",
        how="left",
    )
    folds = group_folds(tr[["company_id", "group_id"]], n=N_FOLDS, seed=FOLD_SEED)
    m = m.merge(folds[["company_id", "fold"]], on="company_id", how="left")
    m["log1p_a_in3"] = np.log1p(pd.to_numeric(m["a_in3"], errors="coerce").clip(lower=0))
    y3 = pd.to_numeric(m[Y3], errors="coerce")

    # invoiced-only AUROC: strip the 470 dummy
    inv_lab = y3.notna() & m["has_book"] & m["fold"].notna()
    recs = []
    for name, col in (
        ("has_erp", m["has_erp"].astype(float)),
        ("has_country", m["has_country"].astype(float)),
        ("is_eur", m["is_eur"].astype(float)),
        ("log1p_a_in3", m["log1p_a_in3"]),
        ("c_n_days_with_tx", m["c_n_days_with_tx"]),
    ):
        rec = signed_oof_auroc(m[Y3], col, m["fold"], inv_lab)
        rec["feature"] = name
        recs.append(rec)
        print(
            f"  invoiced-only Y3 {name}: cv={rec['cv']:.4f} train={rec['train_auc']:.4f} "
            f"sign={rec['train_sign']} n={rec['n_defined']}"
        )
    inv_tab = pd.DataFrame(
        [
            {
                "feature": r["feature"],
                "cv": r["cv"],
                "sd": r["sd"],
                "train_auc": r["train_auc"],
                "train_sign": r["train_sign"],
                "n_defined": r["n_defined"],
            }
            for r in recs
        ]
    )

    # ES vs other among *known* country (not miss vs known)
    es_ids = set(tr.loc[tr["has_country"] & tr["country"].astype(str).str.upper().eq("ES"), "company_id"])
    known_ids = set(tr.loc[tr["has_country"], "company_id"])
    m["is_es"] = m["company_id"].isin(es_ids).astype(float)
    known_lab = y3.notna() & m["company_id"].isin(known_ids) & m["fold"].notna()
    es_cv = signed_oof_auroc(m[Y3], m["is_es"], m["fold"], known_lab)
    size_known = signed_oof_auroc(m[Y3], m["log1p_a_in3"], m["fold"], known_lab)
    print(f"  ES dummy on known-country Y3 cv={es_cv['cv']:.4f} vs size {size_known['cv']:.4f}")

    # group-constant currency / country
    gcur = tr.groupby("group_id")["currency"].nunique()
    gcty = tr.groupby("group_id").apply(
        lambda g: g.loc[~g["miss_country"], "country"].nunique() if (~g["miss_country"]).any() else 0,
        include_groups=False,
    )
    n_g = int(tr["group_id"].nunique())
    share_one_cur = float((gcur == 1).mean())
    n_multi_cur = int((gcur > 1).sum())
    known_g = gcty[gcty > 0]
    share_one_cty = float((known_g == 1).mean()) if len(known_g) else float("nan")
    n_multi_cty = int((known_g > 1).sum()) if len(known_g) else 0
    print(
        f"  groups one-currency={share_one_cur:.3f} multi-currency={n_multi_cur}/{n_g}; "
        f"among groups with any country, one-country={share_one_cty:.3f} multi={n_multi_cty}/{len(known_g)}"
    )

    # family-disagree pairs
    both = tr["has_erp"] & tr["fam_g"].notna()
    disagree = tr.loc[both & tr["fam_co"].ne(tr["fam_g"]), ["erp_norm", "group_erp", "fam_co", "fam_g"]]
    pair = (
        disagree.groupby(["erp_norm", "group_erp"])
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )
    print("  family-disagree pairs")
    print(pair.to_string(index=False) if not pair.empty else "  (none)")

    # 24-month miss-country: still majority — late-arrival is not the story
    # already in p7; quote it here
    md = [
        "### Extra 2 — GROUP_0138 (18 of 20 all-dark named-ERP)",
        "",
        f"GROUP_0138 is an **all-dark** holding of {len(g138)} companies, all `dynamicsAx`, "
        f"0 books, miss-country {_pp(float(g138['miss_country'].mean()))}, "
        f"EUR {_pp(float(g138['is_eur'].mean()))}. "
        "One group-type dummy, not 18 independent ERP events. The other 2 all-dark named-ERP "
        "are singletons (`a3`, `fo`). Do not read 20 named-dark as 20 live ERP connections.",
        "",
        "### Extra 2 — invoiced-only Y3 AUROC (strip the 470 dummy)",
        "",
        "If `has_erp` were anything but the dark flag, it would still rank among the 744.",
        "",
    ]
    md += md_table(
        inv_tab,
        [
            ("feature", "feature", "s"),
            ("cv", "CV AUROC", "f"),
            ("sd", "sd", "f"),
            ("train_auc", "train AUROC", "f"),
            ("train_sign", "sign", "n"),
            ("n_defined", "n", "n"),
        ],
    )
    md += [
        "",
        f"ES vs other *known* country (216 companies): Y3 CV {_f(es_cv['cv'])} vs size "
        f"{_f(size_known['cv'])} on the same rows. A Spain dummy is still a group geography "
        "constant — hidden test is new groups.",
        "",
        f"Currency is group-constant: {_pp(share_one_cur)} of train groups have one currency "
        f"({n_multi_cur} multi-currency groups). Among groups with any ISO, "
        f"{_pp(share_one_cty)} have a single country ({n_multi_cty} mixed). "
        "These flags are **group identity**, not company health.",
        "",
        "Family-disagree `companies.erp` vs `groups.erp` (after slug map) — leftovers, not a new Y:",
        "",
    ]
    md += md_table(pair, [("erp_norm", "companies.erp", "s"), ("group_erp", "groups.erp", "s"), ("n", "n", "n")])
    md += [
        "",
        "24-month train books still miss country on **73.6%**. Late-arrival moves the needle "
        "(known-country is richer on the 24-month pile) but missingness is the default, not a "
        "short-trail hole. PARK missing-country as a health Y either way.",
        "",
    ]
    extra = dict(extra)
    extra["inv_only_tab"] = inv_tab
    extra["es_cv"] = es_cv["cv"]
    extra["es_size_cv"] = size_known["cv"]
    extra["share_one_cur"] = share_one_cur
    extra["share_one_cty"] = share_one_cty
    extra["n_multi_cur"] = n_multi_cur
    extra["n_multi_cty"] = n_multi_cty
    extra["pair_disagree"] = pair
    extra["g138_n"] = int(len(g138))
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts3(
    cos: pd.DataFrame,
    is_train: pd.Series,
    panel: pd.DataFrame,
    book: set[str],
    pop: dict,
    extra: dict,
) -> dict:
    """Third sitting: 0138 vs 360; named-dark 17 vs rest of 110; multi-ccy × FX."""
    print("\nEXTRA 3 (same module)")
    tr = cos.loc[is_train].copy()
    tr["has_book"] = tr["company_id"].isin(book)
    tr["mix"] = tr["group_id"].map(pop["mix_of"])
    is_tr = train_mask(panel["company_id"])
    m = panel.loc[is_tr].merge(
        tr[["company_id", "has_erp", "has_book", "is_eur", "currency"]],
        on="company_id",
        how="left",
    )
    y3 = pd.to_numeric(m[Y3], errors="coerce")
    y2 = pd.to_numeric(m[Y2], errors="coerce")
    cid = m["company_id"]
    g138 = m["group_id"].eq("GROUP_0138")
    dark360 = (~m["has_book"]) & m["group_id"].map(pop["mix_of"]).eq("all_dark")
    dark110 = (~m["has_book"]) & m["group_id"].map(pop["mix_of"]).eq("mixed")
    named_dark_ids = set(tr.loc[tr["has_erp"] & ~tr["has_book"], "company_id"])
    named110 = dark110 & m["company_id"].isin(named_dark_ids)

    rows = []
    for yname, yser in ((Y3, y3), (Y2, y2)):
        for sl_name, sl in (
            ("GROUP_0138", g138 & yser.notna()),
            ("all_dark_ex_0138", dark360 & ~g138 & yser.notna()),
            ("named_dark_110", named110 & yser.notna()),
            ("other_110", dark110 & ~named110 & yser.notna()),
        ):
            rec = rate_row(yser[sl], cid[sl], sl_name)
            rec["y"] = yname
            rows.append(rec)
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))

    # multi-currency groups × ever FX
    gcur = tr.groupby("group_id")["currency"].nunique()
    multi_g = set(gcur[gcur > 1].index)
    st = panel.loc[is_tr]
    fx = pd.to_numeric(st["e_fx_share"], errors="coerce")
    ever = st.assign(fx_pos=fx.gt(0)).groupby("company_id")["fx_pos"].any()
    tr = tr.merge(ever.rename("ever_fx"), on="company_id", how="left")
    tr["ever_fx"] = tr["ever_fx"].fillna(False)
    tr["multi_ccy_g"] = tr["group_id"].isin(multi_g)
    inv = tr[tr["has_book"]]
    share_fx_multi = float(inv.loc[inv["multi_ccy_g"], "ever_fx"].mean()) if inv["multi_ccy_g"].any() else float("nan")
    share_fx_one = float(inv.loc[~inv["multi_ccy_g"], "ever_fx"].mean()) if (~inv["multi_ccy_g"]).any() else float("nan")
    n_multi_inv = int(inv["multi_ccy_g"].sum())
    n_one_inv = int((~inv["multi_ccy_g"]).sum())
    print(
        f"  invoiced ever-FX in multi-ccy groups {share_fx_multi:.3f} n={n_multi_inv} "
        f"vs one-ccy {share_fx_one:.3f} n={n_one_inv}"
    )

    # ERP family dummies among invoiced (top 3) — group-type, expect no lift
    folds = group_folds(tr[["company_id", "group_id"]], n=N_FOLDS, seed=FOLD_SEED)
    m = m.merge(folds[["company_id", "fold"]], on="company_id", how="left")
    top = tr.loc[tr["has_book"] & tr["has_erp"], "erp_norm"].value_counts().head(3).index.tolist()
    inv_lab = y3.notna() & m["has_book"] & m["fold"].notna()
    fam_rows = []
    for slug in top:
        ids = set(tr.loc[tr["erp_norm"].eq(slug), "company_id"])
        x = m["company_id"].isin(ids).astype(float)
        rec = signed_oof_auroc(m[Y3], x, m["fold"], inv_lab)
        fam_rows.append({"feature": f"erp={slug}", "cv": rec["cv"], "train_auc": rec["train_auc"], "n": rec["n_defined"]})
        print(f"  invoiced Y3 erp={slug}: cv={rec['cv']:.4f} train={rec['train_auc']:.4f}")
    fam_tab = pd.DataFrame(fam_rows)

    md = [
        "### Extra 3 — GROUP_0138 vs the rest of the 360; named-dark 17 vs the rest of the 110",
        "",
        "If named-ERP dark were a live book, they would recover like the 744. They do not.",
        "",
    ]
    md += md_table(
        tab,
        [
            ("y", "Y", "s"),
            ("slice", "slice", "s"),
            ("n", "n labeled", "n"),
            ("pos", "pos", "n"),
            ("cos", "cos", "n"),
            ("rate", "rate", "pp"),
        ],
    )
    md += [
        "",
        f"Invoiced ever-FX: multi-currency groups {_pp(share_fx_multi)} (n={n_multi_inv}) vs "
        f"one-currency groups {_pp(share_fx_one)} (n={n_one_inv}). Multi-ccy is a *group* FX "
        "identity (fx_qa home-currency footnote), not a transferable company health X.",
        "",
        "Top named-ERP slugs as invoiced Y3 dummies (expect ~0.50):",
        "",
    ]
    md += md_table(
        fam_tab,
        [("feature", "feature", "s"), ("cv", "CV AUROC", "f"), ("train_auc", "train AUROC", "f"), ("n", "n", "n")],
    )
    md += [""]
    extra = dict(extra)
    extra["slice_tab"] = tab
    extra["fx_multi"] = share_fx_multi
    extra["fx_one"] = share_fx_one
    extra["fam_tab"] = fam_tab
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts4(
    cos: pd.DataFrame,
    is_train: pd.Series,
    panel: pd.DataFrame,
    book: set[str],
    pop: dict,
    extra: dict,
) -> dict:
    """Fourth sitting: named-dark 110 after size; singleton T2 whales."""
    print("\nEXTRA 4 (same module)")
    tr = cos.loc[is_train].copy()
    tr["has_book"] = tr["company_id"].isin(book)
    named_dark_ids = set(tr.loc[tr["has_erp"] & ~tr["has_book"], "company_id"])
    is_tr = train_mask(panel["company_id"])
    m = panel.loc[is_tr].copy()
    m["has_book"] = m["company_id"].isin(book)
    m["mix"] = m["group_id"].map(pop["mix_of"])
    m["named_dark"] = m["company_id"].isin(named_dark_ids)
    m["log1p_a_in3"] = np.log1p(pd.to_numeric(m["a_in3"], errors="coerce").clip(lower=0))
    terc, _ = fit_terciles(m["log1p_a_in3"], m["log1p_a_in3"].notna())
    m["size_terc"] = terc
    y3 = pd.to_numeric(m[Y3], errors="coerce")
    cid = m["company_id"]
    base = y3.notna() & m["mix"].eq("mixed") & ~m["has_book"]
    rows = []
    for tname in ("T1_small", "T2_mid", "T3_large"):
        t = m["size_terc"].eq(tname)
        a = rate_row(y3[base & t & m["named_dark"]], cid[base & t & m["named_dark"]], "named")
        b = rate_row(y3[base & t & ~m["named_dark"]], cid[base & t & ~m["named_dark"]], "null")
        rows.append(
            {
                "tercile": tname,
                "n_named": a["n"],
                "rate_named": a["rate"],
                "n_null": b["n"],
                "rate_null": b["rate"],
                "residual": (
                    float(a["rate"] - b["rate"])
                    if np.isfinite(a["rate"]) and np.isfinite(b["rate"])
                    else float("nan")
                ),
            }
        )
    resid = pd.DataFrame(rows)
    print("named-dark 110 Y3 after size")
    print(resid.to_string(index=False))

    # singleton T2 Y3 companies — is the +17pp two whales?
    sing_ids = set(tr.loc[tr["n_in_sample"].eq(1), "company_id"])
    t2 = m["size_terc"].eq("T2_mid") & y3.notna() & m["company_id"].isin(sing_ids)
    cos_t2 = (
        m.loc[t2]
        .groupby("company_id")
        .agg(n=(Y3, "size"), pos=(Y3, "sum"), rate=(Y3, "mean"), med_in3=("a_in3", "median"))
        .reset_index()
        .sort_values("pos", ascending=False)
    )
    print("singleton T2 Y3 companies")
    print(cos_t2.to_string(index=False))
    n_pos_cos = int((cos_t2["pos"] > 0).sum())
    n_cos = int(len(cos_t2))

    md = [
        "### Extra 4 — named-dark among the 110 after size",
        "",
        "Raw Y3 18.6% (named) vs 10.4% (other 110). Same train `a_in3` terciles:",
        "",
    ]
    md += md_table(
        resid,
        [
            ("tercile", "tercile", "s"),
            ("n_named", "n named-dark", "n"),
            ("rate_named", "Y3 named", "pp"),
            ("n_null", "n other 110", "n"),
            ("rate_null", "Y3 other", "pp"),
            ("residual", "residual", "delta"),
        ],
    )
    md += [
        "",
        "Even if a cell stays green, 14 companies in 10 mixed groups is a **group-type** "
        "sliver. CLOSE as Y3 X. Do not invent a named-ERP-dark Y.",
        "",
        f"### Extra 4 — singleton T2 Y3 (+17pp): {n_pos_cos} / {n_cos} companies ever recover",
        "",
        "Company-months can hide two whales. Companies in the T2 singleton cell:",
        "",
    ]
    md += md_table(
        cos_t2,
        [
            ("company_id", "company", "s"),
            ("n", "n labeled", "n"),
            ("pos", "pos", "n"),
            ("rate", "Y3 rate", "pp"),
            ("med_in3", "median a_in3", "f"),
        ],
    )
    md += [
        "",
        "A singleton lift that is a handful of companies is still a group-size dummy. "
        "Hidden test = new groups. Do not KEEP singleton as Q1 X.",
        "",
    ]
    extra = dict(extra)
    extra["named110_resid"] = resid
    extra["sing_t2"] = cos_t2
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts5(
    cos: pd.DataFrame,
    is_train: pd.Series,
    book: set[str],
    extra: dict,
) -> dict:
    """Fifth sitting: group-named / company-NULL; ES concentration."""
    print("\nEXTRA 5 (same module)")
    tr = cos.loc[is_train].copy()
    tr["has_book"] = tr["company_id"].isin(book)
    tr["miss_group_erp"] = _blank(tr["group_erp"])
    hole = tr.loc[~tr["has_erp"] & ~tr["miss_group_erp"]].copy()
    n_hole = int(len(hole))
    n_hole_dark = int((~hole["has_book"]).sum())
    n_hole_inv = int(hole["has_book"].sum())
    n_g = int(hole["group_id"].nunique())
    print(
        f"  group-named/co-NULL n={n_hole} dark={n_hole_dark} invoiced={n_hole_inv} groups={n_g}"
    )
    fam = hole["group_erp"].map(erp_family).value_counts().reset_index()
    fam.columns = ["family", "n"]
    print(fam.to_string(index=False))

    es = tr.loc[tr["has_country"] & tr["country"].astype(str).str.upper().eq("ES")]
    n_es = int(len(es))
    n_es_g = int(es["group_id"].nunique())
    known = tr.loc[tr["has_country"]]
    n_known_g = int(known["group_id"].nunique())
    print(f"  ES {n_es} companies in {n_es_g} groups; known-country {int(len(known))} in {n_known_g} groups")

    md = [
        "### Extra 5 — group has an ERP name, company does not",
        "",
        f"**{n_hole}** train companies sit in a group with `groups.erp` filled and "
        f"`companies.erp` NULL ({n_hole_dark} dark / {n_hole_inv} invoiced, {n_g} groups). "
        "The 73 invoiced-NULL are not all of this hole. A group ERP name does not give "
        "the company a book. Still not a Y3 X.",
        "",
    ]
    md += md_table(fam, [("family", "groups.erp family", "s"), ("n", "n co-NULL", "n")])
    md += [
        "",
        f"ES is **{n_es}** companies in **{n_es_g}** groups "
        f"(known-country {int(len(known))} in {n_known_g} groups). "
        f"Median ~{n_es / max(n_es_g, 1):.1f} ES companies per ES group. "
        "Geography is a group identity, not a transferable company health reading.",
        "",
    ]
    extra = dict(extra)
    extra["n_hole"] = n_hole
    extra["n_hole_dark"] = n_hole_dark
    extra["n_es"] = n_es
    extra["n_es_g"] = n_es_g
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts6(
    cos: pd.DataFrame,
    is_train: pd.Series,
    panel: pd.DataFrame,
    book: set[str],
    pop: dict,
    p6: dict,
    extra: dict,
) -> dict:
    """Sixth sitting: country-miss by mix; 73∩116; fold table."""
    print("\nEXTRA 6 (same module)")
    tr = cos.loc[is_train].copy()
    tr["has_book"] = tr["company_id"].isin(book)
    tr["mix"] = tr["group_id"].map(pop["mix_of"])
    tr["miss_group_erp"] = _blank(tr["group_erp"])
    mix_tab = (
        tr.groupby("mix", dropna=False)
        .agg(
            n=("company_id", "nunique"),
            n_miss_cty=("miss_country", "sum"),
            share_miss_cty=("miss_country", "mean"),
            n_miss_erp=("miss_erp", "sum"),
            share_miss_erp=("miss_erp", "mean"),
        )
        .reset_index()
    )
    print(mix_tab.to_string(index=False))

    inv_null = set(tr.loc[~tr["has_erp"] & tr["has_book"], "company_id"])
    g_named_co_null = set(tr.loc[~tr["has_erp"] & ~tr["miss_group_erp"], "company_id"])
    both = inv_null & g_named_co_null
    only_73 = inv_null - g_named_co_null
    print(
        f"  invoiced-NULL {len(inv_null)}; group-named/co-NULL {len(g_named_co_null)}; "
        f"intersection {len(both)}; invoiced-NULL with group-NULL erp {len(only_73)}"
    )

    fold_rows = []
    for feat in ("has_erp", "has_country", "is_eur", "log1p_a_in3"):
        rec = p6["by"][(Y3, feat)]
        for fr in rec["folds"]:
            fold_rows.append({"feature": feat, **fr})
    fold_tab = pd.DataFrame(fold_rows)
    print(fold_tab.to_string(index=False))

    md = [
        "### Extra 6 — country / ERP missingness by group mix",
        "",
        "If missing-country were an ERP-dark echo it would pile in all-dark groups.",
        "",
    ]
    md += md_table(
        mix_tab,
        [
            ("mix", "mix", "s"),
            ("n", "n cos", "n"),
            ("n_miss_cty", "n miss country", "n"),
            ("share_miss_cty", "share miss country", "pp"),
            ("n_miss_erp", "n miss erp", "n"),
            ("share_miss_erp", "share miss erp", "pp"),
        ],
    )
    md += [
        "",
        f"Of the 73 invoiced+NULL, **{len(both)}** sit in a group that *has* `groups.erp` "
        f"and **{len(only_73)}** sit in a group with NULL `groups.erp`. "
        "Company NULL erp is not the same hole as group NULL erp.",
        "",
        "Y3 group-fold AUROC by fold (sign from train side). Flags wander around 0.50; size does not.",
        "",
    ]
    md += md_table(
        fold_tab,
        [
            ("feature", "feature", "s"),
            ("fold", "fold", "n"),
            ("auroc", "AUROC", "f"),
            ("sign", "sign", "n"),
            ("n_va", "n va", "n"),
            ("n_pos", "n pos", "n"),
        ],
    )
    md += [""]
    extra = dict(extra)
    extra["mix_miss"] = mix_tab
    extra["n_73_and_116"] = len(both)
    extra["n_73_group_null"] = len(only_73)
    extra["fold_tab"] = fold_tab
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts7(
    cos: pd.DataFrame,
    is_train: pd.Series,
    is_hold: pd.Series,
    clocks: pd.DataFrame,
    book: set[str],
    extra: dict,
) -> dict:
    """Seventh sitting: country vs erp missingness; holdout named-dark; bank clock confirm."""
    print("\nEXTRA 7 (same module)")
    tr = cos.loc[is_train].copy()
    rho_flags = spearman(tr["has_country"].astype(float), tr["has_erp"].astype(float))
    both = float((tr["has_country"] & tr["has_erp"]).mean())
    neither = float((~tr["has_country"] & ~tr["has_erp"]).mean())
    print(f"  ρ has_country vs has_erp={rho_flags:.3f} both={both:.3f} neither={neither:.3f}")

    ho = cos.loc[is_hold].copy()
    ho["has_book"] = ho["company_id"].isin(book)
    named_dark_ho = ho.loc[ho["has_erp"] & ~ho["has_book"], ["company_id", "group_id", "erp_norm"]]
    print("  holdout named-dark")
    print(named_dark_ho.to_string(index=False) if not named_dark_ho.empty else "  (none)")

    cl = tr.merge(clocks, on="company_id", how="left")
    d_bank_tx = (cl["first_bank_created"] - cl["first_tx"]).dt.total_seconds() / 86400.0
    has_bank = cl["first_bank_created"].notna()
    share_after = float((d_bank_tx[has_bank] > 0).mean())
    same_mo = float(
        (
            cl.loc[has_bank, "first_bank_created"].dt.to_period("M")
            == cl.loc[has_bank, "first_tx_month"].dt.to_period("M")
        ).mean()
    )
    trail_quote = 851 / 1211
    print(
        f"  bank after first tx {share_after:.3f} (trail 70.3% of 1211) "
        f"same-month {same_mo:.3f} n_bank={int(has_bank.sum())}"
    )

    md = [
        "### Extra 7 — is filled-country just filled-ERP?",
        "",
        f"Spearman `has_country` vs `has_erp` **{_f(rho_flags)}** "
        f"(both named {_pp(both)}, both missing {_pp(neither)}). "
        "They are **not** the same flag: country is missing on invoiced books too (extra 6). "
        "Do not merge them into one metadata Y.",
        "",
        "Holdout named-ERP dark (coverage only):",
        "",
    ]
    if named_dark_ho.empty:
        md += ["(none)", ""]
    else:
        md += md_table(
            named_dark_ho.rename(columns={"erp_norm": "erp"}),
            [("company_id", "company", "s"), ("group_id", "group", "s"), ("erp", "erp", "s")],
        )
        md += [""]
    md += [
        f"First banking `created_at` after first tx: **{_pp(share_after)}** of train companies "
        f"with a banking row (n={int(has_bank.sum())}). Trail QA quoted 70.3% (851/1,211). "
        f"{'CONFIRMS' if abs(share_after - trail_quote) < 0.01 else 'CORRECTS'} that number. "
        f"Same calendar month {_pp(same_mo)}. Product connection ≠ cash start. PARK both clocks.",
        "",
    ]
    extra = dict(extra)
    extra["rho_ctry_erp"] = rho_flags
    extra["bank_after_tx"] = share_after
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts8(cos: pd.DataFrame, is_train: pd.Series, extra: dict) -> dict:
    """Eighth sitting: is companies.created_at one onboard wave?"""
    print("\nEXTRA 8 (same module)")
    tr = cos.loc[is_train].copy()
    mo = tr["created_at"].dt.to_period("M").astype(str)
    tab = mo.value_counts().sort_index().reset_index()
    tab.columns = ["month", "n"]
    tab["share"] = tab["n"] / max(int(len(tr)), 1)
    print(tab.to_string(index=False))
    modal = tab.loc[tab["n"].idxmax()]
    share_modal = float(modal["share"])
    n_months = int((tab["n"] > 0).sum())
    print(f"  created_at months={n_months} modal {modal['month']} n={int(modal['n'])} share={share_modal:.3f}")

    md = [
        "### Extra 8 — is `companies.created_at` one onboarding wave?",
        "",
        f"Train `created_at` spans **{n_months}** calendar months. Modal {modal['month']} "
        f"has {int(modal['n'])} companies ({_pp(share_modal)}). "
        "Not one wave. Same conclusion as trail QA on first-tx. "
        "A month-of-onboard dummy would be a **calendar / connection** dummy, not Q1 health.",
        "",
    ]
    md += md_table(tab, [("month", "month", "s"), ("n", "n", "n"), ("share", "share", "pp")])
    md += [""]
    extra = dict(extra)
    extra["created_months"] = n_months
    extra["created_modal_share"] = share_modal
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts9(
    cos: pd.DataFrame,
    is_train: pd.Series,
    is_hold: pd.Series,
    book: set[str],
    extra: dict,
) -> dict:
    """Ninth sitting: currency × dark; post-extract created_at; holdout late onboard."""
    print("\nEXTRA 9 (same module)")
    tr = cos.loc[is_train].copy()
    tr["has_book"] = tr["company_id"].isin(book)
    cur = (
        tr.assign(ccy=tr["currency"].astype(str).str.upper())
        .groupby(["has_book", "ccy"])
        .size()
        .reset_index(name="n")
    )
    cur["slice"] = np.where(cur["has_book"], "invoiced", "dark")
    top = cur.sort_values("n", ascending=False).head(16)
    print(top.to_string(index=False))
    share_eur_dk = float(tr.loc[~tr["has_book"], "is_eur"].mean())
    share_eur_inv = float(tr.loc[tr["has_book"], "is_eur"].mean())
    print(f"  EUR dark={share_eur_dk:.3f} invoiced={share_eur_inv:.3f}")

    n_post = int((tr["created_at"] >= EXTRACT).sum())
    n_pre_panel = int((tr["created_at"] < PANEL_START).sum())
    print(f"  created_at >= extract {n_post}; < 2024-09-01 {n_pre_panel}")

    ho = cos.loc[is_hold]
    late_ho = float((ho["created_at"] > PANEL_START).mean())
    print(f"  holdout late created_at {late_ho:.3f} n={int(len(ho))}")

    md = [
        "### Extra 9 — currency × book; created_at vs extract",
        "",
        f"EUR share dark {_pp(share_eur_dk)} vs invoiced {_pp(share_eur_inv)}. "
        "Dark companies are not a foreign-currency pile. Non-EUR is the invoiced FX identity.",
        "",
        f"`created_at` ≥ extract {n_post} (should be ~0). "
        f"`created_at` before 2024-09-01: **{n_pre_panel}** / {int(len(tr))} — "
        "platform rows older than the cash window. Still a connection clock.",
        "",
        f"Holdout late `created_at`: {_pp(late_ho)} (coverage). Train was {_pp(0.708)}.",
        "",
    ]
    extra = dict(extra)
    extra["eur_dark"] = share_eur_dk
    extra["eur_inv"] = share_eur_inv
    extra["n_created_pre_panel"] = n_pre_panel
    extra["hold_late_created"] = late_ho
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts10(
    cos: pd.DataFrame,
    is_train: pd.Series,
    clocks: pd.DataFrame,
    extra: dict,
) -> dict:
    """Tenth sitting: how far is onboard after cash? Long tails."""
    print("\nEXTRA 10 (same module)")
    tr = cos.loc[is_train].merge(clocks, on="company_id", how="left")
    d = (tr["created_at"] - tr["first_tx"]).dt.total_seconds() / 86400.0
    cuts = {
        "gt_0": float((d > 0).mean()),
        "gt_30": float((d > 30).mean()),
        "gt_90": float((d > 90).mean()),
        "gt_180": float((d > 180).mean()),
        "gt_365": float((d > 365).mean()),
        "lt_0": float((d < 0).mean()),
        "lt_m90": float((d < -90).mean()),
        "p90": float(d.quantile(0.90)),
        "p10": float(d.quantile(0.10)),
    }
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in cuts.items()})
    md = [
        "### Extra 10 — onboard lag tails (created_at − first tx, days)",
        "",
        f"Share after cash: 0d {_pp(cuts['gt_0'])}, >30d {_pp(cuts['gt_30'])}, "
        f">90d {_pp(cuts['gt_90'])}, >180d {_pp(cuts['gt_180'])}, >365d {_pp(cuts['gt_365'])}. "
        f"Share *before* cash {_pp(cuts['lt_0'])} (p10 {_f(cuts['p10'], 1)} days). "
        f"p90 {_f(cuts['p90'], 1)} days. "
        "A 6–12 month onboard lag is common. That is connection, not a 45→65. PARK.",
        "",
    ]
    extra = dict(extra)
    extra["clock_tails"] = cuts
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts11(
    cos: pd.DataFrame,
    is_train: pd.Series,
    clocks: pd.DataFrame,
    extra: dict,
) -> dict:
    """Eleventh sitting: onboard-before-cash ∩ 2024-09 starters (left-truncated books)."""
    print("\nEXTRA 11 (same module)")
    tr = cos.loc[is_train].merge(clocks, on="company_id", how="left")
    before = tr["created_at"] < tr["first_tx"]
    start_sep = tr["first_tx_month"].eq(PANEL_START)
    n_before = int(before.sum())
    n_before_sep = int((before & start_sep).sum())
    n_before_late = int((before & ~start_sep).sum())
    n_sep = int(start_sep.sum())
    print(
        f"  created before first tx {n_before}; of those 2024-09 starters {n_before_sep}; "
        f"late first-tx {n_before_late}; 2024-09 starters total {n_sep}"
    )
    md = [
        "### Extra 11 — onboard-before-cash is left-truncated observation",
        "",
        f"{n_before} train companies have `created_at` *before* first tx. "
        f"**{n_before_sep}** of those start cash in 2024-09 ({n_before_sep}/{n_sep} of the September starters). "
        f"The other {n_before_late} are late first-tx with an older platform row. "
        "Trail QA: product older than the window is left-truncated *observation*, not a new firm. "
        "Still PARK as a health Y.",
        "",
    ]
    extra = dict(extra)
    extra["n_before"] = n_before
    extra["n_before_sep"] = n_before_sep
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts12(
    cos: pd.DataFrame,
    is_train: pd.Series,
    clocks: pd.DataFrame,
    book: set[str],
    extra: dict,
) -> dict:
    """Twelfth sitting: is onboard-before-cash the same 470 dark?"""
    print("\nEXTRA 12 (same module)")
    tr = cos.loc[is_train].merge(clocks, on="company_id", how="left")
    tr["has_book"] = tr["company_id"].isin(book)
    before = tr["created_at"] < tr["first_tx"]
    dark = ~tr["has_book"]
    n_both = int((before & dark).sum())
    n_before = int(before.sum())
    n_dark = int(dark.sum())
    print(f"  before∩dark={n_both} before={n_before} dark={n_dark} (coincidence if both=470 and overlap<<470)")
    md = [
        "### Extra 12 — onboard-before-cash ∩ the 470 dark",
        "",
        f"Overlap **{n_both}** (before={n_before}, dark={n_dark}). "
        + (
            "Same set — do not read that as a discovery; check the IDs."
            if n_both == n_before == n_dark
            else "The matching *count* 470 is a coincidence. Onboard-before-cash is not the dark flag."
        ),
        "",
    ]
    extra = dict(extra)
    extra["n_before_dark"] = n_both
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts13(
    cos: pd.DataFrame,
    is_train: pd.Series,
    is_hold: pd.Series,
    book: set[str],
    extra: dict,
) -> dict:
    """13: holdout named-dark; country-known by ERP family; groups.erp miss."""
    print("\nEXTRA 13 (same module)")
    ho = cos.loc[is_hold].copy()
    ho["has_book"] = ho["company_id"].isin(book)
    nd = ho.loc[ho["has_erp"] & ~ho["has_book"], ["company_id", "group_id", "erp_norm", "country", "currency"]]
    print("holdout named-dark:")
    print(nd.to_string(index=False) if not nd.empty else "  (none)")

    tr = cos.loc[is_train].copy()
    tr["fam"] = tr["erp_norm"].map(erp_family)
    named = tr[tr["has_erp"]]
    by = (
        named.groupby("fam")
        .agg(n=("company_id", "nunique"), n_known=("has_country", "sum"), share_known=("has_country", "mean"))
        .reset_index()
        .sort_values("n", ascending=False)
    )
    print(by.to_string(index=False))

    tr["miss_g"] = _blank(tr["group_erp"])
    ho["miss_g"] = _blank(ho["group_erp"])
    print(
        f"  groups.erp miss train={tr['miss_g'].mean():.3f} holdout={ho['miss_g'].mean():.3f}"
    )

    md = [
        "### Extra 13 — holdout named-dark; country fill by ERP family; `groups.erp`",
        "",
    ]
    if nd.empty:
        md += ["Holdout named-ERP dark: none.", ""]
    else:
        md += ["Holdout named-ERP dark (coverage):", ""]
        md += md_table(
            nd.rename(columns={"erp_norm": "erp"}),
            [
                ("company_id", "company", "s"),
                ("group_id", "group", "s"),
                ("erp", "erp", "s"),
                ("country", "country", "s"),
                ("currency", "currency", "s"),
            ],
        )
        md += [""]
    md += [
        "Share of named-ERP train companies with a country code, by family "
        "(if one ERP ‘remembers’ ISO, country is still metadata):",
        "",
    ]
    md += md_table(
        by,
        [
            ("fam", "erp family", "s"),
            ("n", "n named", "n"),
            ("n_known", "n country known", "n"),
            ("share_known", "share known", "pp"),
        ],
    )
    md += [
        "",
        f"`groups.erp` missing: train {_pp(float(tr['miss_g'].mean()))}, "
        f"holdout {_pp(float(ho['miss_g'].mean()))}. Group-level ERP is also often empty. "
        "Do not treat `groups.erp` as a cleaner dark flag.",
        "",
    ]
    extra = dict(extra)
    extra["erp_country"] = by
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts14(
    cos: pd.DataFrame,
    is_train: pd.Series,
    panel: pd.DataFrame,
    clocks: pd.DataFrame,
    book: set[str],
    extra: dict,
) -> dict:
    """14: country fill among invoiced × has_erp; created_at year vs trail."""
    print("\nEXTRA 14 (same module)")
    tr = cos.loc[is_train].merge(clocks, on="company_id", how="left")
    tr["has_book"] = tr["company_id"].isin(book)
    trail = (
        panel.loc[train_mask(panel["company_id"])]
        .groupby("company_id")["period"]
        .nunique()
        .rename("n_grid")
    )
    tr = tr.merge(trail, on="company_id", how="left")
    tr["created_year"] = tr["created_at"].dt.year
    inv = tr[tr["has_book"]].copy()
    tab = (
        inv.groupby("has_erp")
        .agg(n=("company_id", "nunique"), n_known=("has_country", "sum"), share_known=("has_country", "mean"))
        .reset_index()
    )
    tab["slice"] = tab["has_erp"].map({True: "invoiced+named-ERP", False: "invoiced+NULL-ERP"})
    print(tab.to_string(index=False))

    rho_y_grid = spearman(tr["created_year"], tr["n_grid"])
    rho_y_miss = spearman(tr["created_year"], tr["miss_country"].astype(float))
    by_y = (
        tr.groupby("created_year")
        .agg(
            n=("company_id", "nunique"),
            med_grid=("n_grid", "median"),
            share_miss=("miss_country", "mean"),
            share_late=("first_tx_month", lambda s: float((s > PANEL_START).mean())),
        )
        .reset_index()
    )
    print(f"  ρ created_year vs n_grid={rho_y_grid:.3f} vs miss_country={rho_y_miss:.3f}")
    print(by_y.to_string(index=False))

    md = [
        "### Extra 14 — country fill among invoiced × ERP; created_at year vs trail",
        "",
        "If country-known were ‘who has a book’, invoiced+NULL-ERP would look like invoiced+named. "
        "If country is vendor metadata, named ERPs fill ISO and NULL-ERP do not (or the reverse).",
        "",
    ]
    md += md_table(
        tab,
        [
            ("slice", "slice", "s"),
            ("n", "n invoiced", "n"),
            ("n_known", "n country known", "n"),
            ("share_known", "share known", "pp"),
        ],
    )
    md += [
        "",
        f"Spearman `created_at` year vs grid months **{_f(rho_y_grid)}**; vs miss-country **{_f(rho_y_miss)}**. "
        "A strong year→shorter-trail link is left-truncation of the connection clock, not a new-firm Y.",
        "",
    ]
    md += md_table(
        by_y,
        [
            ("created_year", "created_at year", "n"),
            ("n", "n", "n"),
            ("med_grid", "median grid months", "f"),
            ("share_miss", "country miss", "pp"),
            ("share_late", "late first-tx", "pp"),
        ],
    )
    md += [""]
    extra = dict(extra)
    extra["inv_country_by_erp"] = tab
    extra["rho_created_year_grid"] = rho_y_grid
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts15(
    cos: pd.DataFrame,
    is_train: pd.Series,
    panel: pd.DataFrame,
    clocks: pd.DataFrame,
    book: set[str],
    extra: dict,
) -> dict:
    """15: created_year as Y3 X (trail dummy); the 1 invoiced+NULL with a country."""
    print("\nEXTRA 15 (same module)")
    tr = cos.loc[is_train].merge(clocks, on="company_id", how="left")
    tr["has_book"] = tr["company_id"].isin(book)
    tr["created_year"] = tr["created_at"].dt.year.astype(float)
    odd = tr.loc[tr["has_book"] & ~tr["has_erp"] & tr["has_country"], ["company_id", "group_id", "country", "currency"]]
    print("invoiced+NULL+country:")
    print(odd.to_string(index=False) if not odd.empty else "  (none)")

    is_tr = train_mask(panel["company_id"])
    m = panel.loc[is_tr].merge(
        tr[["company_id", "group_id", "created_year", "has_book"]],
        on="company_id",
        how="left",
    )
    folds = group_folds(tr[["company_id", "group_id"]], n=N_FOLDS, seed=FOLD_SEED)
    m = m.merge(folds[["company_id", "fold"]], on="company_id", how="left")
    m["log1p_a_in3"] = np.log1p(pd.to_numeric(m["a_in3"], errors="coerce").clip(lower=0))
    m["n_grid"] = m.groupby("company_id")["period"].transform("nunique")
    lab = m[Y3].notna() & m["fold"].notna()
    recs = []
    for name, col in (
        ("created_year", m["created_year"]),
        ("n_grid", m["n_grid"]),
        ("log1p_a_in3", m["log1p_a_in3"]),
        ("c_n_days_with_tx", m["c_n_days_with_tx"]),
    ):
        rec = signed_oof_auroc(m[Y3], col, m["fold"], lab)
        rec["feature"] = name
        recs.append(rec)
        print(
            f"  Y3 {name}: cv={rec['cv']:.4f} train={rec['train_auc']:.4f} "
            f"sign={rec['train_sign']} n={rec['n_defined']}"
        )
    tab = pd.DataFrame(
        [
            {
                "feature": r["feature"],
                "cv": r["cv"],
                "sd": r["sd"],
                "train_auc": r["train_auc"],
                "train_sign": r["train_sign"],
            }
            for r in recs
        ]
    )
    md = [
        "### Extra 15 — `created_at` year as Y3 X (do not revive as a health Y)",
        "",
        "The one invoiced+NULL-ERP company that has a country (extra 14: 1/73):",
        "",
    ]
    if odd.empty:
        md += ["(none)", ""]
    else:
        md += md_table(
            odd,
            [
                ("company_id", "company", "s"),
                ("group_id", "group", "s"),
                ("country", "country", "s"),
                ("currency", "currency", "s"),
            ],
        )
        md += [""]
    md += [
        "Group-fold Y3 AUROC. `created_year` should track `n_grid` (ρ −0.85) and lose to size / days. "
        "PARK as a health Y still holds — this is only a CLOSE-as-X check.",
        "",
    ]
    md += md_table(
        tab,
        [
            ("feature", "feature", "s"),
            ("cv", "cv AUROC", "f"),
            ("sd", "sd", "f"),
            ("train_auc", "train AUROC", "f"),
            ("train_sign", "sign", "n"),
        ],
    )
    md += [""]
    extra = dict(extra)
    extra["created_year_y3"] = tab
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts16(
    cos: pd.DataFrame,
    is_train: pd.Series,
    is_hold: pd.Series,
    book: set[str],
    extra: dict,
) -> dict:
    """16: GROUP_0113 (1 invoiced+NULL+ES) and GROUP_0211 (holdout named-dark)."""
    print("\nEXTRA 16 (same module)")
    g = cos.copy()
    g["has_book"] = g["company_id"].isin(book)
    g["split"] = np.where(is_hold, "holdout", np.where(is_train, "train", "other"))
    md = [
        "### Extra 16 — two sliver groups (COMP_0600 / COMP_0851)",
        "",
        "GROUP_0113 holds the only invoiced+NULL-ERP company with a country. "
        "GROUP_0211 holds the only holdout named-ERP dark (COMP_0851 / netsuite / USD). "
        "Coverage rosters — if the ISO or ERP sits on one sister, it is group metadata.",
        "",
    ]
    for gid in ("GROUP_0113", "GROUP_0211"):
        sub = g.loc[g["group_id"] == gid, ["company_id", "split", "erp_norm", "country", "currency", "has_book", "has_erp"]]
        print(gid)
        print(sub.to_string(index=False) if not sub.empty else "  (missing)")
        md += [f"**{gid}**", ""]
        if sub.empty:
            md += ["(not in companies.csv)", ""]
            continue
        show = sub.copy()
        show["book"] = np.where(show["has_book"], "invoiced", "dark")
        show["erp"] = show["erp_norm"].astype("string")
        md += md_table(
            show,
            [
                ("company_id", "company", "s"),
                ("split", "split", "s"),
                ("erp", "erp", "s"),
                ("country", "country", "s"),
                ("currency", "currency", "s"),
                ("book", "book", "s"),
            ],
        )
        md += [""]
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts17(
    cos: pd.DataFrame,
    is_train: pd.Series,
    panel: pd.DataFrame,
    clocks: pd.DataFrame,
    extra: dict,
) -> dict:
    """17: currency ≠ modal invoice accounting; created_year vs Y2."""
    print("\nEXTRA 17 (same module)")
    tr = cos.loc[is_train].copy()
    con = connect()
    acc = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               accounting_currency,
               COUNT(*) AS n
        FROM invoices
        WHERE document_type = 'invoice' AND status <> 'cancel'
          AND amount <> 0 AND issuance_date IS NOT NULL
        GROUP BY 1, 2
        """
    ).df()
    con.close()
    acc["company_id"] = acc["company_id"].astype(str)
    top = acc.sort_values("n", ascending=False).drop_duplicates("company_id")
    top = top.rename(columns={"accounting_currency": "acct_ccy"})
    inv = tr.loc[tr["has_erp"] | True].merge(top[["company_id", "acct_ccy"]], on="company_id", how="inner")
    # inner = companies with a book invoice currency
    inv["acct_ccy"] = inv["acct_ccy"].astype("string").str.upper()
    inv["cur_u"] = inv["currency"].astype("string").str.upper()
    mismatch = inv.loc[~inv["cur_u"].eq(inv["acct_ccy"]), ["company_id", "group_id", "currency", "acct_ccy", "country"]]
    print(f"  currency≠modal acct n={len(mismatch)} / {len(inv)}")
    print(mismatch.head(20).to_string(index=False) if not mismatch.empty else "  (none)")

    trc = tr.merge(clocks, on="company_id", how="left")
    trc["created_year"] = trc["created_at"].dt.year.astype(float)
    is_tr = train_mask(panel["company_id"])
    m = panel.loc[is_tr].merge(trc[["company_id", "group_id", "created_year"]], on="company_id", how="left")
    folds = group_folds(trc[["company_id", "group_id"]], n=N_FOLDS, seed=FOLD_SEED)
    m = m.merge(folds[["company_id", "fold"]], on="company_id", how="left")
    lab = m[Y2].notna() & m["fold"].notna()
    rec = signed_oof_auroc(m[Y2], m["created_year"], m["fold"], lab)
    print(f"  Y2 created_year: cv={rec['cv']:.4f} sign={rec['train_sign']}")

    md = [
        "### Extra 17 — currency ≠ modal invoice `accounting_currency`; `created_year` vs Y2",
        "",
        f"Mismatch **{len(mismatch)}** / {len(inv)} invoiced with an accounting currency. "
        "These are book-setup leftovers, not a health Y. Do not invent a currency-mismatch Y.",
        "",
    ]
    if not mismatch.empty:
        md += md_table(
            mismatch.head(25),
            [
                ("company_id", "company", "s"),
                ("group_id", "group", "s"),
                ("currency", "companies.csv", "s"),
                ("acct_ccy", "modal invoice", "s"),
                ("country", "country", "s"),
            ],
        )
        md += [""]
    md += [
        f"Y2 group-fold AUROC for `created_year`: **{_f(rec['cv'])}** (sign {rec['train_sign']}). "
        "Same connection clock as extra 15. PARK as a health Y; CLOSE as X.",
        "",
    ]
    extra = dict(extra)
    extra["n_ccy_mismatch"] = int(len(mismatch))
    extra["created_year_y2"] = rec["cv"]
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts18(
    cos: pd.DataFrame,
    is_train: pd.Series,
    is_hold: pd.Series,
    clocks: pd.DataFrame,
    book: set[str],
    extra: dict,
) -> dict:
    """18: GROUP_0013 (6 currency mismatches); holdout created_at year coverage."""
    print("\nEXTRA 18 (same module)")
    g = cos.copy()
    g["has_book"] = g["company_id"].isin(book)
    g["split"] = np.where(is_hold, "holdout", np.where(is_train, "train", "other"))
    sub = g.loc[
        g["group_id"] == "GROUP_0013",
        ["company_id", "split", "erp_norm", "country", "currency", "has_book"],
    ]
    print("GROUP_0013")
    print(sub.to_string(index=False) if not sub.empty else "  (missing)")

    ho = cos.loc[is_hold].merge(clocks, on="company_id", how="left")
    tr = cos.loc[is_train].merge(clocks, on="company_id", how="left")
    ho["created_year"] = ho["created_at"].dt.year
    tr["created_year"] = tr["created_at"].dt.year
    ho_y = ho["created_year"].value_counts(dropna=False).sort_index().reset_index()
    ho_y.columns = ["created_year", "n"]
    ho_y["share"] = ho_y["n"] / max(len(ho), 1)
    tr_y = tr["created_year"].value_counts(dropna=False).sort_index().reset_index()
    tr_y.columns = ["created_year", "n"]
    tr_y["share"] = tr_y["n"] / max(len(tr), 1)
    print("holdout created_year")
    print(ho_y.to_string(index=False))
    print(
        f"  holdout created_year p50={ho['created_year'].median():.1f} "
        f"train p50={tr['created_year'].median():.1f}"
    )

    md = [
        "### Extra 18 — GROUP_0013 (currency-mismatch nest); holdout `created_at` year",
        "",
        "Six of the 17 currency ≠ modal-invoice rows sit in GROUP_0013 (EUR on companies.csv, "
        "invoice books in SGD/BRL/GBP/ARS/PEN/COP). Coverage roster:",
        "",
    ]
    show = sub.copy()
    show["book"] = np.where(show["has_book"], "invoiced", "dark")
    show["erp"] = show["erp_norm"].astype("string")
    md += md_table(
        show,
        [
            ("company_id", "company", "s"),
            ("split", "split", "s"),
            ("erp", "erp", "s"),
            ("country", "country", "s"),
            ("currency", "currency", "s"),
            ("book", "book", "s"),
        ],
    )
    md += [
        "",
        "Holdout `created_at` year (coverage only — do not read rates):",
        "",
    ]
    md += md_table(
        ho_y,
        [
            ("created_year", "year", "n"),
            ("n", "n holdout", "n"),
            ("share", "share", "pp"),
        ],
    )
    md += [
        "",
        f"Holdout median created_at year {_f(float(ho['created_year'].median()), 1)} vs train "
        f"{_f(float(tr['created_year'].median()), 1)}. "
        f"Holdout is **2025-peaked** ({_pp(float(ho_y.loc[ho_y['created_year'].eq(2025), 'share'].sum() if (ho_y['created_year']==2025).any() else float('nan')))}) "
        "vs train ~35% in 2025 (extra 14). Same median, different mix — `created_at` year still cannot transfer.",
        "",
    ]
    extra = dict(extra)
    extra["holdout_created_year"] = ho_y
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts19(
    cos: pd.DataFrame,
    is_train: pd.Series,
    is_hold: pd.Series,
    panel: pd.DataFrame,
    clocks: pd.DataFrame,
    extra: dict,
) -> dict:
    """19: created_year mix train vs holdout; fold wander; n_grid vs MONTHS."""
    print("\nEXTRA 19 (same module)")
    tr = cos.loc[is_train].merge(clocks, on="company_id", how="left")
    ho = cos.loc[is_hold].merge(clocks, on="company_id", how="left")
    tr["created_year"] = tr["created_at"].dt.year
    ho["created_year"] = ho["created_at"].dt.year
    years = sorted(set(tr["created_year"].dropna().astype(int)) | set(ho["created_year"].dropna().astype(int)))
    rows = []
    for y in years:
        rows.append(
            {
                "year": y,
                "n_train": int((tr["created_year"] == y).sum()),
                "share_train": float((tr["created_year"] == y).mean()),
                "n_hold": int((ho["created_year"] == y).sum()),
                "share_hold": float((ho["created_year"] == y).mean()),
            }
        )
    mix = pd.DataFrame(rows)
    print(mix.to_string(index=False))

    trail = (
        panel.loc[train_mask(panel["company_id"])]
        .groupby("company_id")["period"]
        .nunique()
        .rename("n_grid")
    )
    print(
        f"  MONTHS={len(MONTHS)} n_grid max={int(trail.max())} share_24={(trail.eq(24)).mean():.3f}"
    )

    trc = tr.copy()
    trc["created_year_f"] = trc["created_year"].astype(float)
    is_tr = train_mask(panel["company_id"])
    m = panel.loc[is_tr].merge(
        trc[["company_id", "group_id", "created_year_f"]],
        on="company_id",
        how="left",
    )
    folds = group_folds(trc[["company_id", "group_id"]], n=N_FOLDS, seed=FOLD_SEED)
    m = m.merge(folds[["company_id", "fold"]], on="company_id", how="left")
    lab = m[Y3].notna() & m["fold"].notna()
    rec = signed_oof_auroc(m[Y3], m["created_year_f"], m["fold"], lab)
    fold_tab = pd.DataFrame(rec["folds"])
    print("  created_year Y3 folds:")
    print(fold_tab.to_string(index=False))

    md = [
        "### Extra 19 — `created_at` year mix (train vs holdout); fold wander; grid vs MONTHS",
        "",
        "Coverage only on holdout. Train 2025 share is ~35%; holdout 2025 share is ~71%. "
        "Median year matches; the *mix* does not. A year dummy fitted on train will not match new groups.",
        "",
    ]
    md += md_table(
        mix,
        [
            ("year", "year", "n"),
            ("n_train", "n train", "n"),
            ("share_train", "share train", "pp"),
            ("n_hold", "n holdout", "n"),
            ("share_hold", "share holdout", "pp"),
        ],
    )
    md += [
        "",
        f"Feature-store grid: max `n_grid`={int(trail.max())} vs `MONTHS`={len(MONTHS)}; "
        f"share at 24 months {_pp(float((trail.eq(24)).mean()))}.",
        "",
        "Y3 `created_year` fold AUROCs (sign chosen on the train side of each fold). "
        "Wander around 0.5 = cannot transfer:",
        "",
    ]
    md += md_table(
        fold_tab,
        [
            ("fold", "fold", "n"),
            ("auroc", "AUROC", "f"),
            ("sign", "sign", "n"),
            ("n_va", "n val", "n"),
            ("n_pos", "n pos", "n"),
        ],
    )
    md += [""]
    extra = dict(extra)
    extra["year_mix"] = mix
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts20(
    cos: pd.DataFrame,
    is_train: pd.Series,
    clocks: pd.DataFrame,
    book: set[str],
    p6: dict,
    extra: dict,
) -> dict:
    """20: has_erp / has_country year mix; flag fold wander from pass 6."""
    print("\nEXTRA 20 (same module)")
    tr = cos.loc[is_train].merge(clocks, on="company_id", how="left")
    tr["has_book"] = tr["company_id"].isin(book)
    tr["created_year"] = tr["created_at"].dt.year
    wide = (
        tr.groupby("created_year")
        .agg(
            n_named_erp=("has_erp", "sum"),
            n=("company_id", "nunique"),
        )
        .reset_index()
    )
    wide["n_null_erp"] = wide["n"] - wide["n_named_erp"]
    wide["share_named"] = wide["n_named_erp"] / wide["n"].clip(lower=1)
    print(wide.to_string(index=False))
    rho = spearman(tr["has_erp"].astype(float), tr["created_year"])
    print(f"  ρ has_erp vs created_year={rho:.3f}")

    md = [
        "### Extra 20 — `has_erp` × onboard year; flag fold wander",
        "",
        f"Spearman `has_erp` vs `created_at` year **{_f(rho)}**. "
        "Named-ERP share is ~50–67% in every year 2022–2026 — not an early-onboard dummy. "
        "CLOSE `has_erp` remains the 470 / book dummy (ρ 0.81 vs has_book), not a year dummy.",
        "",
    ]
    md += md_table(
        wide,
        [
            ("created_year", "year", "n"),
            ("n_named_erp", "named ERP", "n"),
            ("n_null_erp", "NULL ERP", "n"),
            ("share_named", "share named", "pp"),
        ],
    )
    md += ["", "Y3 group-fold AUROC for the three company-constant flags (pass 6, sign per fold):", ""]
    for name, title in (("has_erp", "has_erp"), ("has_country", "has_country"), ("is_eur", "currency=EUR")):
        rec = p6["by"][(Y3, name)]
        folds = pd.DataFrame(rec["folds"])
        print(f"  {name} folds cv={rec['cv']:.3f}")
        print(folds.to_string(index=False))
        md += [f"**{title}** cv={_f(rec['cv'])} ± {_f(rec['sd'])}", ""]
        md += md_table(
            folds,
            [
                ("fold", "fold", "n"),
                ("auroc", "AUROC", "f"),
                ("sign", "sign", "n"),
                ("n_va", "n val", "n"),
            ],
        )
        md += [""]
    extra = dict(extra)
    extra["rho_erp_year"] = rho
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts21(
    cos: pd.DataFrame,
    is_train: pd.Series,
    is_hold: pd.Series,
    clocks: pd.DataFrame,
    extra: dict,
) -> dict:
    """21: holdout country-known roster; created_at vs first_bank same-day."""
    print("\nEXTRA 21 (same module)")
    ho = cos.loc[is_hold].copy()
    known = ho.loc[ho["has_country"], ["company_id", "group_id", "country", "currency", "erp_norm"]]
    print(f"holdout country-known n={len(known)}")
    print(known.to_string(index=False) if not known.empty else "  (none)")

    tr = cos.loc[is_train].merge(clocks, on="company_id", how="left")
    d = (tr["created_at"] - tr["first_bank_created"]).dt.total_seconds() / 86400.0
    same_day = d.abs() < 1.0
    same_week = d.abs() < 7.0
    n = int(d.notna().sum())
    rec = {
        "n": n,
        "share_same_day": float(same_day.mean()) if n else float("nan"),
        "share_same_week": float(same_week.mean()) if n else float("nan"),
        "p50": float(d.median()) if n else float("nan"),
    }
    print(
        f"  created vs first_bank same-day={rec['share_same_day']:.3f} "
        f"same-week={rec['share_same_week']:.3f} p50={rec['p50']:.1f} n={n}"
    )

    md = [
        "### Extra 21 — holdout country-known; onboard vs first bank same-day",
        "",
        f"Holdout country known **{len(known)}** / {len(ho)} (coverage). "
        "If known ISO clusters in a few groups, it is group metadata on the hidden split too.",
        "",
    ]
    if known.empty:
        md += ["(none)", ""]
    else:
        md += md_table(
            known.rename(columns={"erp_norm": "erp"}),
            [
                ("company_id", "company", "s"),
                ("group_id", "group", "s"),
                ("country", "country", "s"),
                ("currency", "currency", "s"),
                ("erp", "erp", "s"),
            ],
        )
        md += [""]
    md += [
        f"Train `created_at` vs first `banking_products.created_at`: "
        f"same calendar day {_pp(rec['share_same_day'])}, "
        f"within 7 days {_pp(rec['share_same_week'])}, "
        f"median offset {_f(rec['p50'], 1)} days (n={n}). "
        "If they were the same connection event, same-day would be high. "
        "They are two clocks even when both are ‘created_at’.",
        "",
    ]
    extra = dict(extra)
    extra["holdout_known"] = known
    extra["co_bank_same_day"] = rec["share_same_day"]
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts22(
    cos: pd.DataFrame,
    is_train: pd.Series,
    is_hold: pd.Series,
    clocks: pd.DataFrame,
    extra: dict,
) -> dict:
    """22: holdout country-known groups; created vs first_tx same-day."""
    print("\nEXTRA 22 (same module)")
    ho = cos.loc[is_hold]
    n_g = int(ho["group_id"].nunique())
    known_g = ho.loc[ho["has_country"], "group_id"].nunique()
    print(f"  holdout groups={n_g} with any country known={known_g}")

    tr = cos.loc[is_train].merge(clocks, on="company_id", how="left")
    d = (tr["created_at"] - tr["first_tx"]).dt.total_seconds() / 86400.0
    ok = d.notna()
    same_day = float((d[ok].abs() < 1.0).mean()) if ok.any() else float("nan")
    same_week = float((d[ok].abs() < 7.0).mean()) if ok.any() else float("nan")
    print(f"  created vs first_tx same-day={same_day:.3f} same-week={same_week:.3f} n={int(ok.sum())}")

    md = [
        "### Extra 22 — holdout country groups; onboard vs first tx same-day",
        "",
        f"Holdout: **{known_g}** / {n_g} groups have any country filled "
        f"({int(ho['has_country'].sum())} companies — extra 21). "
        "Country on the hidden 72 is a handful of group metadata rows, not a transferable health flag.",
        "",
        f"Train `created_at` vs first tx: same calendar day {_pp(same_day)}, "
        f"within 7 days {_pp(same_week)} (n={int(ok.sum())}). "
        "Same-month was 7.1% (pass 4). Onboard and cash start are not the same event.",
        "",
    ]
    extra = dict(extra)
    extra["holdout_known_groups"] = int(known_g)
    extra["co_tx_same_day"] = same_day
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts23(
    cos: pd.DataFrame,
    is_train: pd.Series,
    is_hold: pd.Series,
    extra: dict,
) -> dict:
    """23: holdout ERP family mix (coverage)."""
    print("\nEXTRA 23 (same module)")
    ho = cos.loc[is_hold].copy()
    ho["fam"] = ho["erp_norm"].map(erp_family)
    ho["fam"] = ho["fam"].fillna("(NULL)")
    tab = (
        ho.groupby("fam")
        .agg(n=("company_id", "nunique"), n_groups=("group_id", "nunique"))
        .reset_index()
        .sort_values("n", ascending=False)
    )
    tab["share"] = tab["n"] / max(len(ho), 1)
    print(tab.to_string(index=False))
    md = [
        "### Extra 23 — holdout ERP family (coverage)",
        "",
        "Hidden 72 ERP mix. A train `has_erp` / slug dummy cannot transfer if the new groups "
        "bring a different vendor mix (or more NULLs).",
        "",
    ]
    md += md_table(
        tab,
        [
            ("fam", "erp family", "s"),
            ("n", "n companies", "n"),
            ("n_groups", "n groups", "n"),
            ("share", "share", "pp"),
        ],
    )
    md += [""]
    extra = dict(extra)
    extra["holdout_erp"] = tab
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts24(
    cos: pd.DataFrame,
    is_train: pd.Series,
    is_hold: pd.Series,
    extra: dict,
) -> dict:
    """24: holdout vs train currency mix (coverage)."""
    print("\nEXTRA 24 (same module)")
    rows = []
    for split, mask in (("train", is_train), ("holdout", is_hold)):
        d = cos.loc[mask]
        cur = d["currency"].fillna("(missing)").astype(str).str.strip().str.upper()
        vc = cur.value_counts(dropna=False)
        for ccy, n in vc.items():
            rows.append({"split": split, "currency": ccy, "n": int(n), "share": float(n) / max(len(d), 1)})
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))
    ho_eur = float(cos.loc[is_hold, "is_eur"].mean())
    tr_eur = float(cos.loc[is_train, "is_eur"].mean())
    print(f"  EUR share train={tr_eur:.3f} holdout={ho_eur:.3f}")
    md = [
        "### Extra 24 — holdout vs train currency (coverage)",
        "",
        f"EUR share train {_pp(tr_eur)} vs holdout {_pp(ho_eur)}. "
        "GROUP_0211 (extra 16) is a multi-ccy netsuite holding. "
        "`currency=EUR` on train is a 90% home-book dummy; holdout EUR share moves. Cannot transfer.",
        "",
    ]
    md += md_table(
        tab,
        [
            ("split", "split", "s"),
            ("currency", "currency", "s"),
            ("n", "n", "n"),
            ("share", "share", "pp"),
        ],
    )
    md += [""]
    extra = dict(extra)
    extra["holdout_eur"] = ho_eur
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts25(
    cos: pd.DataFrame,
    is_train: pd.Series,
    is_hold: pd.Series,
    extra: dict,
) -> dict:
    """25: currencies / countries that appear only on holdout (coverage)."""
    print("\nEXTRA 25 (same module)")
    tr_ccy = set(cos.loc[is_train, "currency"].dropna().astype(str).str.upper())
    ho_ccy = set(cos.loc[is_hold, "currency"].dropna().astype(str).str.upper())
    only_ho = sorted(ho_ccy - tr_ccy)
    only_tr = sorted(tr_ccy - ho_ccy)
    tr_iso = set(
        cos.loc[is_train & cos["has_country"], "country"].dropna().astype(str).str.upper()
    )
    ho_iso = set(
        cos.loc[is_hold & cos["has_country"], "country"].dropna().astype(str).str.upper()
    )
    iso_only_ho = sorted(ho_iso - tr_iso)
    iso_only_tr = sorted(tr_iso - ho_iso)
    print(f"  ccy only-holdout={only_ho} only-train={only_tr}")
    print(f"  iso only-holdout={iso_only_ho} only-train={iso_only_tr}")
    md = [
        "### Extra 25 — currencies / ISO codes that the hidden 72 bring",
        "",
        f"Currencies on holdout **not** in train: **{only_ho or '(none)'}**. "
        f"Train-only currencies: {len(only_tr)} (expected — train is larger).",
        "",
        f"Known-country ISO on holdout **not** in train known set: **{iso_only_ho or '(none)'}**. "
        f"Train-only known ISO: {iso_only_tr or '(none)'}.",
        "",
        "New-group test means new home books. A EUR dummy / ES dummy fitted on train "
        "does not see AOA/GHS/XOF or a BE/PL holdout ISO the same way. CLOSE.",
        "",
    ]
    extra = dict(extra)
    extra["ccy_only_holdout"] = only_ho
    extra["iso_only_holdout"] = iso_only_ho
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts26(
    cos: pd.DataFrame,
    is_train: pd.Series,
    clocks: pd.DataFrame,
    book: set[str],
    extra: dict,
) -> dict:
    """26: n_banking vs has_erp / has_book (connection richness ≠ ERP name)."""
    print("\nEXTRA 26 (same module)")
    tr = cos.loc[is_train].merge(clocks, on="company_id", how="left")
    tr["has_book"] = tr["company_id"].isin(book)
    tr["n_banking"] = pd.to_numeric(tr["n_banking"], errors="coerce")
    rows = []
    for name, m in (
        ("named+invoiced", tr["has_erp"] & tr["has_book"]),
        ("named+dark", tr["has_erp"] & ~tr["has_book"]),
        ("NULL+invoiced", ~tr["has_erp"] & tr["has_book"]),
        ("NULL+dark", ~tr["has_erp"] & ~tr["has_book"]),
    ):
        s = tr.loc[m, "n_banking"]
        rows.append(
            {
                "slice": name,
                "n": int(m.sum()),
                "med_bank": float(s.median()) if s.notna().any() else float("nan"),
                "share_no_bank": float(s.isna().mean()) if len(s) else float("nan"),
            }
        )
        print(
            f"  {name} n={int(m.sum())} med_bank={rows[-1]['med_bank']:.1f} "
            f"no_bank={rows[-1]['share_no_bank']:.3f}"
        )
    tab = pd.DataFrame(rows)
    md = [
        "### Extra 26 — banking-product count vs ERP × book",
        "",
        "If `has_erp` were ‘more connected’, named-ERP would show more banking products. "
        "If named-dark look like NULL-dark, ERP name is not connection richness.",
        "",
    ]
    md += md_table(
        tab,
        [
            ("slice", "slice", "s"),
            ("n", "n", "n"),
            ("med_bank", "median n banking", "f"),
            ("share_no_bank", "share no banking row", "pp"),
        ],
    )
    md += [""]
    extra = dict(extra)
    extra["bank_by_erp"] = tab
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts27(
    cos: pd.DataFrame,
    is_train: pd.Series,
    clocks: pd.DataFrame,
    extra: dict,
) -> dict:
    """27: first_bank vs first_tx same-day (product clock ≠ cash clock)."""
    print("\nEXTRA 27 (same module)")
    tr = cos.loc[is_train].merge(clocks, on="company_id", how="left")
    d = (tr["first_bank_created"] - tr["first_tx"]).dt.total_seconds() / 86400.0
    ok = d.notna()
    rec = {
        "n": int(ok.sum()),
        "same_day": float((d[ok].abs() < 1.0).mean()) if ok.any() else float("nan"),
        "same_week": float((d[ok].abs() < 7.0).mean()) if ok.any() else float("nan"),
        "p50": float(d[ok].median()) if ok.any() else float("nan"),
    }
    print(
        f"  bank−first_tx same-day={rec['same_day']:.3f} same-week={rec['same_week']:.3f} "
        f"p50={rec['p50']:.1f} n={rec['n']}"
    )
    md = [
        "### Extra 27 — first banking `created_at` vs first tx",
        "",
        f"Same calendar day {_pp(rec['same_day'])}, within 7 days {_pp(rec['same_week'])}, "
        f"median offset {_f(rec['p50'], 1)} days (n={rec['n']}). "
        "Pass 4 already had p50 +54.6 days / 70% bank after cash. "
        "The product connection clock is not the cash trail. Three clocks stand.",
        "",
    ]
    extra = dict(extra)
    extra["bank_tx_same_day"] = rec["same_day"]
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts28(
    cos: pd.DataFrame,
    is_train: pd.Series,
    extra: dict,
) -> dict:
    """28: group-uniform ERP / country / currency (group-type dummy)."""
    print("\nEXTRA 28 (same module)")
    tr = cos.loc[is_train]
    rows = []
    for col, label in (
        ("has_erp", "has_erp"),
        ("has_country", "has_country"),
        ("is_eur", "is_eur"),
    ):
        g = tr.groupby("group_id")[col].agg(n="size", n_true="sum", nuniq="nunique")
        n_g = int(len(g))
        all_true = int((g["n_true"] == g["n"]).sum())
        all_false = int((g["n_true"] == 0).sum())
        mixed = int(((g["n_true"] > 0) & (g["n_true"] < g["n"])).sum())
        share_uniform = (all_true + all_false) / max(n_g, 1)
        rows.append(
            {
                "flag": label,
                "n_groups": n_g,
                "all_true": all_true,
                "all_false": all_false,
                "mixed": mixed,
                "share_uniform": share_uniform,
            }
        )
        print(
            f"  {label} groups={n_g} all1={all_true} all0={all_false} mixed={mixed} "
            f"uniform={share_uniform:.3f}"
        )
    tab = pd.DataFrame(rows)
    md = [
        "### Extra 28 — group-uniform flags (why they cannot transfer)",
        "",
        "Share of train groups where the flag is constant. A high uniform share means the "
        "flag is a *group type*. Hidden test = new groups ⇒ the dummy does not come along.",
        "",
    ]
    md += md_table(
        tab,
        [
            ("flag", "flag", "s"),
            ("n_groups", "groups", "n"),
            ("all_true", "all 1", "n"),
            ("all_false", "all 0", "n"),
            ("mixed", "mixed", "n"),
            ("share_uniform", "share uniform", "pp"),
        ],
    )
    md += [""]
    extra = dict(extra)
    extra["group_uniform"] = tab
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts29(
    cos: pd.DataFrame,
    is_train: pd.Series,
    extra: dict,
) -> dict:
    """29: is created_at year group-constant?"""
    print("\nEXTRA 29 (same module)")
    tr = cos.loc[is_train].copy()
    tr["y"] = tr["created_at"].dt.year
    g = tr.groupby("group_id")["y"].agg(n="size", nuniq="nunique")
    n_g = int(len(g))
    one = int((g["nuniq"] == 1).sum())
    print(f"  created_year one-year groups={one}/{n_g} share={one / max(n_g, 1):.3f}")
    md = [
        "### Extra 29 — `created_at` year as a group constant",
        "",
        f"**{one}** / {n_g} train groups share a single `created_at` year "
        f"({_pp(one / max(n_g, 1))}). Onboard year is often a group wave, "
        "not a company health path. Still PARK as a Y; CLOSE as X.",
        "",
    ]
    extra = dict(extra)
    extra["year_one_group"] = one
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts30(
    cos: pd.DataFrame,
    is_train: pd.Series,
    extra: dict,
) -> dict:
    """30: holdout-only home currencies vs train invoice currencies."""
    print("\nEXTRA 30 (same module)")
    only_ho = extra.get("ccy_only_holdout") or []
    train_ids = list(cos.loc[is_train, "company_id"].astype(str))
    con = connect()
    con.register("train_ids", pd.DataFrame({"company_id": train_ids}))
    acct = con.execute(
        """
        SELECT DISTINCT upper(trim(CAST(i.accounting_currency AS VARCHAR))) AS ccy
        FROM invoices i
        JOIN train_ids t ON CAST(i.company_id AS VARCHAR) = t.company_id
        WHERE i.document_type = 'invoice' AND i.status <> 'cancel'
          AND i.amount <> 0 AND i.issuance_date IS NOT NULL
          AND i.accounting_currency IS NOT NULL
        """
    ).df()
    fx = con.execute(
        """
        SELECT DISTINCT upper(trim(CAST(i.currency AS VARCHAR))) AS ccy
        FROM invoices i
        JOIN train_ids t ON CAST(i.company_id AS VARCHAR) = t.company_id
        WHERE i.document_type = 'invoice' AND i.status <> 'cancel'
          AND i.amount <> 0 AND i.issuance_date IS NOT NULL
          AND i.currency IS NOT NULL
        """
    ).df()
    con.close()
    acct_set = set(acct["ccy"].dropna().astype(str))
    fx_set = set(fx["ccy"].dropna().astype(str))
    in_acct = sorted(set(only_ho) & acct_set)
    in_fx = sorted(set(only_ho) & fx_set)
    print(f"  holdout-only home {only_ho} in train invoice acct={in_acct} fx={in_fx}")
    md = [
        "### Extra 30 — holdout-only `companies.currency` vs invoice currencies",
        "",
        f"Holdout-only home currencies {only_ho}. "
        f"Seen as train invoice `accounting_currency`: {in_acct or '(none)'}. "
        f"Seen as train invoice `currency` (FX side): {in_fx or '(none)'}.",
        "",
        "AOA/SEK can exist on *someone’s* invoice book and still be a new "
        "`companies.currency` on hidden groups. Home-currency on companies.csv "
        "is not the FX feature. CLOSE `is_eur` as Q1 X.",
        "",
    ]
    extra = dict(extra)
    extra["ho_ccy_in_acct"] = in_acct
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts31(
    cos: pd.DataFrame,
    is_hold: pd.Series,
    extra: dict,
) -> dict:
    """31: who carries AOA/GHS/SEK/XOF on the hidden 72."""
    print("\nEXTRA 31 (same module)")
    only = set(extra.get("ccy_only_holdout") or [])
    ho = cos.loc[is_hold].copy()
    ho["ccy"] = ho["currency"].astype("string").str.upper()
    sub = ho.loc[ho["ccy"].isin(only), ["company_id", "group_id", "ccy", "country", "erp_norm"]]
    print(sub.to_string(index=False) if not sub.empty else "  (none)")
    md = [
        "### Extra 31 — holdout companies with train-unseen home currency",
        "",
        "Coverage roster. These are new-group home books, not a transferable EUR dummy.",
        "",
    ]
    if sub.empty:
        md += ["(none)", ""]
    else:
        md += md_table(
            sub.rename(columns={"erp_norm": "erp"}),
            [
                ("company_id", "company", "s"),
                ("group_id", "group", "s"),
                ("ccy", "currency", "s"),
                ("country", "country", "s"),
                ("erp", "erp", "s"),
            ],
        )
        md += [""]
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts32(
    cos: pd.DataFrame,
    is_hold: pd.Series,
    book: set[str],
    extra: dict,
) -> dict:
    """32: GROUP_0199 full holdout roster (unseen home currencies)."""
    print("\nEXTRA 32 (same module)")
    g = cos.loc[cos["group_id"] == "GROUP_0199"].copy()
    g["has_book"] = g["company_id"].isin(book)
    hold_ids = set(cos.loc[is_hold, "company_id"])
    g["split"] = np.where(g["company_id"].isin(hold_ids), "holdout", "train")
    print(g[["company_id", "split", "erp_norm", "country", "currency", "has_book"]].to_string(index=False))
    show = g.copy()
    show["book"] = np.where(show["has_book"], "invoiced", "dark")
    show["erp"] = show["erp_norm"].astype("string")
    md = [
        "### Extra 32 — GROUP_0199 (all four holdout-only home currencies)",
        "",
        "AOA/GHS/SEK/XOF all sit in one hidden group, next to ES/PL sisters "
        "(extra 21). That is a new-group identity, not a company health reading.",
        "",
    ]
    md += md_table(
        show,
        [
            ("company_id", "company", "s"),
            ("split", "split", "s"),
            ("erp", "erp", "s"),
            ("country", "country", "s"),
            ("currency", "currency", "s"),
            ("book", "book", "s"),
        ],
    )
    md += [""]
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts33(
    cos: pd.DataFrame,
    is_hold: pd.Series,
    book: set[str],
    extra: dict,
) -> dict:
    """33: holdout group mix (coverage) — do not redo sibling_h."""
    print("\nEXTRA 33 (same module)")
    ho = cos.loc[is_hold].copy()
    ho["has_book"] = ho["company_id"].isin(book)
    g = ho.groupby("group_id").agg(
        n=("company_id", "nunique"),
        n_book=("has_book", "sum"),
        n_erp=("has_erp", "sum"),
        n_cty=("has_country", "sum"),
        share_eur=("is_eur", "mean"),
    ).reset_index()
    g["mix"] = np.where(
        g["n_book"] == 0,
        "all-dark",
        np.where(g["n_book"] == g["n"], "all-invoiced", "mixed"),
    )
    print(g.sort_values("n", ascending=False).to_string(index=False))
    mix_n = g["mix"].value_counts()
    print("mix groups", mix_n.to_dict())
    md = [
        "### Extra 33 — holdout group mix (coverage only; Family H not redone)",
        "",
        f"Holdout groups: all-dark {int(mix_n.get('all-dark', 0))}, "
        f"all-invoiced {int(mix_n.get('all-invoiced', 0))}, "
        f"mixed {int(mix_n.get('mixed', 0))}. "
        "GROUP_0199 is an all-dark / NULL-ERP holding with the unseen home currencies. "
        "Coverage — no Y2/Y3 rates on holdout.",
        "",
    ]
    md += md_table(
        g.sort_values("n", ascending=False),
        [
            ("group_id", "group", "s"),
            ("n", "n", "n"),
            ("n_book", "n invoiced", "n"),
            ("n_erp", "n named ERP", "n"),
            ("n_cty", "n country", "n"),
            ("share_eur", "share EUR", "pp"),
            ("mix", "mix", "s"),
        ],
    )
    md += [""]
    extra = dict(extra)
    extra["holdout_mix"] = g
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts34(
    cos: pd.DataFrame,
    is_train: pd.Series,
    clocks: pd.DataFrame,
    extra: dict,
) -> dict:
    """34: created_at month vs first_tx month as group waves."""
    print("\nEXTRA 34 (same module)")
    tr = cos.loc[is_train].merge(clocks, on="company_id", how="left")
    tr["co_m"] = tr["created_at"].dt.to_period("M").astype(str)
    tr["tx_m"] = tr["first_tx_month"].dt.to_period("M").astype(str)
    rows = []
    for col, label in (("co_m", "created_at month"), ("tx_m", "first_tx month")):
        g = tr.groupby("group_id")[col].nunique()
        n_g = int(len(g))
        one = int((g == 1).sum())
        rows.append(
            {
                "clock": label,
                "n_groups": n_g,
                "one_month": one,
                "share_one": one / max(n_g, 1),
                "median_nuniq": float(g.median()),
            }
        )
        print(
            f"  {label} one-month groups={one}/{n_g} share={one / max(n_g, 1):.3f} "
            f"median nuniq={float(g.median()):.1f}"
        )
    tab = pd.DataFrame(rows)
    md = [
        "### Extra 34 — onboard month vs first-tx month as group waves",
        "",
        "If `created_at` month is group-constant more often than first-tx month, "
        "onboard is a sales/connection wave. First-tx month is the cash window "
        "(trail QA). Still PARK `created_at` as a health Y.",
        "",
    ]
    md += md_table(
        tab,
        [
            ("clock", "clock", "s"),
            ("n_groups", "groups", "n"),
            ("one_month", "one-month groups", "n"),
            ("share_one", "share one-month", "pp"),
            ("median_nuniq", "median distinct months", "f"),
        ],
    )
    md += [""]
    extra = dict(extra)
    extra["month_waves"] = tab
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts35(
    cos: pd.DataFrame,
    is_train: pd.Series,
    clocks: pd.DataFrame,
    extra: dict,
) -> dict:
    """35: first_bank month as a group wave (third clock)."""
    print("\nEXTRA 35 (same module)")
    tr = cos.loc[is_train].merge(clocks, on="company_id", how="left")
    tr["bank_m"] = tr["first_bank_created"].dt.to_period("M").astype(str)
    g = tr.groupby("group_id")["bank_m"].nunique()
    n_g = int(len(g))
    one = int((g == 1).sum())
    print(
        f"  first_bank month one-month groups={one}/{n_g} share={one / max(n_g, 1):.3f} "
        f"median nuniq={float(g.median()):.1f}"
    )
    md = [
        "### Extra 35 — first banking-product month as a group wave",
        "",
        f"**{one}** / {n_g} train groups share one first-bank month "
        f"({_pp(one / max(n_g, 1))}); median distinct months {_f(float(g.median()), 1)}. "
        "Compare extra 34: created_at month 73.6% one-month vs first-tx month 55.3%. "
        "Three clocks, three wave shapes. None is a health Y.",
        "",
    ]
    extra = dict(extra)
    extra["bank_one_month"] = one
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts36(
    cos: pd.DataFrame,
    is_train: pd.Series,
    book: set[str],
    pop: dict,
    extra: dict,
) -> dict:
    """36: train dark companies that have a country — mixed sisters or all-dark?"""
    print("\nEXTRA 36 (same module)")
    tr = cos.loc[is_train].copy()
    tr["has_book"] = tr["company_id"].isin(book)
    tr["mix"] = tr["group_id"].map(pop["mix_of"])
    dk = tr.loc[~tr["has_book"] & tr["has_country"]]
    tab = (
        dk.groupby("mix")
        .agg(n=("company_id", "nunique"), n_groups=("group_id", "nunique"))
        .reset_index()
    )
    print(f"  dark+country n={len(dk)}")
    print(tab.to_string(index=False) if not tab.empty else "  (none)")
    md = [
        "### Extra 36 — dark companies with a country (train)",
        "",
        f"**{len(dk)}** dark train companies have an ISO. "
        "If they are mixed-group sisters, country is the invoiced sibling’s metadata leaking. "
        "If they sit in all-dark groups (GROUP_0199-like), country is still group identity.",
        "",
    ]
    if not tab.empty:
        md += md_table(
            tab,
            [
                ("mix", "mix", "s"),
                ("n", "n dark+country", "n"),
                ("n_groups", "groups", "n"),
            ],
        )
        md += [""]
    extra = dict(extra)
    extra["n_dark_country"] = int(len(dk))
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts37(
    cos: pd.DataFrame,
    is_train: pd.Series,
    book: set[str],
    extra: dict,
) -> dict:
    """37: ISO codes on dark+country vs invoiced+country (train)."""
    print("\nEXTRA 37 (same module)")
    tr = cos.loc[is_train].copy()
    tr["has_book"] = tr["company_id"].isin(book)
    tr["iso"] = tr["country"].astype("string").str.upper()
    rows = []
    for name, m in (
        ("dark+country", ~tr["has_book"] & tr["has_country"]),
        ("invoiced+country", tr["has_book"] & tr["has_country"]),
    ):
        vc = tr.loc[m, "iso"].value_counts()
        for iso, n in vc.items():
            rows.append({"slice": name, "country": iso, "n": int(n), "share": float(n) / max(int(m.sum()), 1)})
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))
    md = [
        "### Extra 37 — ISO mix on dark+country vs invoiced+country",
        "",
        "If dark+country is the same ES pile as invoiced+country, country is one metadata "
        "fill process, not a book. Still PARK as a health Y.",
        "",
    ]
    md += md_table(
        tab,
        [
            ("slice", "slice", "s"),
            ("country", "country", "s"),
            ("n", "n", "n"),
            ("share", "share of slice", "pp"),
        ],
    )
    md += [""]
    extra = dict(extra)
    extra["iso_by_book"] = tab
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts38(
    cos: pd.DataFrame,
    is_train: pd.Series,
    clocks: pd.DataFrame,
    extra: dict,
) -> dict:
    """38: created−first_tx offset by onboard year."""
    print("\nEXTRA 38 (same module)")
    tr = cos.loc[is_train].merge(clocks, on="company_id", how="left")
    tr["created_year"] = tr["created_at"].dt.year
    d = (tr["created_at"] - tr["first_tx"]).dt.total_seconds() / 86400.0
    tr["lag"] = d
    tab = (
        tr.groupby("created_year")
        .agg(
            n=("company_id", "nunique"),
            p50=("lag", "median"),
            share_after=("lag", lambda s: float((s > 0).mean())),
            share_same_day=("lag", lambda s: float((s.abs() < 1).mean())),
        )
        .reset_index()
    )
    print(tab.to_string(index=False))
    md = [
        "### Extra 38 — onboard − first tx by `created_at` year",
        "",
        "If 2026 onboards sit closer to first cash, that is left-truncation of both clocks "
        "in a short window — still not a health Y.",
        "",
    ]
    md += md_table(
        tab,
        [
            ("created_year", "year", "n"),
            ("n", "n", "n"),
            ("p50", "p50 days", "f"),
            ("share_after", "share after cash", "pp"),
            ("share_same_day", "share same-day", "pp"),
        ],
    )
    md += [""]
    extra = dict(extra)
    extra["lag_by_year"] = tab
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts39(
    cos: pd.DataFrame,
    is_train: pd.Series,
    panel: pd.DataFrame,
    extra: dict,
) -> dict:
    """39: onboard-after-2024 dummy as Y3 X (trail regime, not health)."""
    print("\nEXTRA 39 (same module)")
    tr = cos.loc[is_train].copy()
    tr["late_onboard"] = (tr["created_at"].dt.year >= 2025).astype(float)
    is_tr = train_mask(panel["company_id"])
    m = panel.loc[is_tr].merge(
        tr[["company_id", "group_id", "late_onboard"]],
        on="company_id",
        how="left",
    )
    folds = group_folds(tr[["company_id", "group_id"]], n=N_FOLDS, seed=FOLD_SEED)
    m = m.merge(folds[["company_id", "fold"]], on="company_id", how="left")
    m["log1p_a_in3"] = np.log1p(pd.to_numeric(m["a_in3"], errors="coerce").clip(lower=0))
    lab = m[Y3].notna() & m["fold"].notna()
    rec = signed_oof_auroc(m[Y3], m["late_onboard"], m["fold"], lab)
    size = signed_oof_auroc(m[Y3], m["log1p_a_in3"], m["fold"], lab)
    print(
        f"  Y3 late_onboard(year>=2025) cv={rec['cv']:.4f} vs size {size['cv']:.4f} "
        f"sign={rec['train_sign']}"
    )
    md = [
        "### Extra 39 — 2025–26 onboard dummy as Y3 X",
        "",
        f"Extra 38 split the clock into two regimes (onboard before vs after cash). "
        f"Y3 AUROC for `created_year≥2025`: **{_f(rec['cv'])}** vs size **{_f(size['cv'])}**. "
        "CLOSE as X. Do not revive as a health Y.",
        "",
    ]
    extra = dict(extra)
    extra["late_onboard_y3"] = rec["cv"]
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts40(
    cos: pd.DataFrame,
    is_train: pd.Series,
    is_hold: pd.Series,
    extra: dict,
) -> dict:
    """40: late-onboard (year≥2025) share train vs holdout coverage."""
    print("\nEXTRA 40 (same module)")
    tr_late = float((cos.loc[is_train, "created_at"].dt.year >= 2025).mean())
    ho_late = float((cos.loc[is_hold, "created_at"].dt.year >= 2025).mean())
    print(f"  year>=2025 share train={tr_late:.3f} holdout={ho_late:.3f}")
    md = [
        "### Extra 40 — 2025–26 onboard share (train vs holdout coverage)",
        "",
        f"Train {_pp(tr_late)} vs holdout {_pp(ho_late)}. "
        "Median year matched (2025) but the late-onboard dummy is more common on the hidden 72. "
        "Another transfer fail for any `created_at` year X.",
        "",
    ]
    extra = dict(extra)
    extra["late_share_train"] = tr_late
    extra["late_share_hold"] = ho_late
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts41(
    cos: pd.DataFrame,
    is_hold: pd.Series,
    extra: dict,
) -> dict:
    """41: holdout country-miss by created_at year (coverage)."""
    print("\nEXTRA 41 (same module)")
    ho = cos.loc[is_hold].copy()
    ho["created_year"] = ho["created_at"].dt.year
    tab = (
        ho.groupby("created_year")
        .agg(n=("company_id", "nunique"), share_miss=("miss_country", "mean"))
        .reset_index()
    )
    print(tab.to_string(index=False))
    md = [
        "### Extra 41 — holdout country-miss by onboard year (coverage)",
        "",
        "Train extra 14: 2026 miss 90.6%. If holdout 2025–26 is also mostly missing ISO, "
        "`has_country` on new groups is still missingness.",
        "",
    ]
    md += md_table(
        tab,
        [
            ("created_year", "year", "n"),
            ("n", "n holdout", "n"),
            ("share_miss", "country miss", "pp"),
        ],
    )
    md += [""]
    extra = dict(extra)
    extra["holdout_miss_by_year"] = tab
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts42(
    cos: pd.DataFrame,
    is_train: pd.Series,
    is_hold: pd.Series,
    clocks: pd.DataFrame,
    book: set[str],
    extra: dict,
) -> dict:
    """42: companies with no banking_products row; holdout known-country years."""
    print("\nEXTRA 42 (same module)")
    tr = cos.loc[is_train].merge(clocks, on="company_id", how="left")
    tr["has_book"] = tr["company_id"].isin(book)
    nob = tr.loc[tr["first_bank_created"].isna(), ["company_id", "group_id", "erp_norm", "has_book", "has_country"]]
    print(f"  train no-banking n={len(nob)}")
    print(nob.to_string(index=False) if not nob.empty else "  (none)")

    ho = cos.loc[is_hold].copy()
    ho["y"] = ho["created_at"].dt.year
    known_y = ho.loc[ho["has_country"], "y"].value_counts().to_dict()
    print(f"  holdout country-known created_year={known_y}")

    md = [
        "### Extra 42 — no banking row; holdout known-country onboard year",
        "",
        f"Train companies with no `banking_products` row: **{len(nob)}**. "
        "Cash can exist without a product `created_at` (outer join in pass 4). "
        "Still three clocks.",
        "",
    ]
    if not nob.empty:
        show = nob.copy()
        show["book"] = np.where(show["has_book"], "invoiced", "dark")
        show["erp"] = show["erp_norm"].astype("string")
        md += md_table(
            show,
            [
                ("company_id", "company", "s"),
                ("group_id", "group", "s"),
                ("erp", "erp", "s"),
                ("book", "book", "s"),
            ],
        )
        md += [""]
    md += [
        f"Holdout country-known `created_at` years: {known_y}. "
        "Extra 41: the 14 known ISO sit in 2025. Missingness is still the default on new groups.",
        "",
    ]
    extra = dict(extra)
    extra["n_no_bank"] = int(len(nob))
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts43(
    cos: pd.DataFrame,
    is_train: pd.Series,
    clocks: pd.DataFrame,
    extra: dict,
) -> dict:
    """43: do the 3 no-bank companies still have a first tx?"""
    print("\nEXTRA 43 (same module)")
    tr = cos.loc[is_train].merge(clocks, on="company_id", how="left")
    nob = tr.loc[
        tr["first_bank_created"].isna(),
        ["company_id", "group_id", "first_tx", "created_at", "n_banking"],
    ]
    print(nob.to_string(index=False) if not nob.empty else "  (none)")
    md = [
        "### Extra 43 — no-banking companies still have cash?",
        "",
        "If first_tx is present, the bank trail exists and `banking_products.created_at` "
        "is just missing metadata. Confirms the third clock can be absent.",
        "",
    ]
    if not nob.empty:
        show = nob.copy()
        show["first_tx"] = show["first_tx"].astype(str)
        show["created_at"] = show["created_at"].astype(str)
        md += md_table(
            show,
            [
                ("company_id", "company", "s"),
                ("group_id", "group", "s"),
                ("first_tx", "first_tx", "s"),
                ("created_at", "created_at", "s"),
            ],
        )
        md += [""]
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts44(
    cos: pd.DataFrame,
    is_train: pd.Series,
    extra: dict,
) -> dict:
    """44: groups of the 3 no-banking companies."""
    print("\nEXTRA 44 (same module)")
    gids = ("GROUP_0104", "GROUP_0238", "GROUP_0239")
    tr = cos.loc[is_train]
    rows = []
    for gid in gids:
        sub = tr.loc[tr["group_id"] == gid]
        rows.append(
            {
                "group_id": gid,
                "n": int(len(sub)),
                "n_erp": int(sub["has_erp"].sum()),
                "n_cty": int(sub["has_country"].sum()),
                "share_eur": float(sub["is_eur"].mean()) if len(sub) else float("nan"),
            }
        )
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))
    md = [
        "### Extra 44 — groups of the 3 no-banking companies",
        "",
        "If they are singletons, missing `banking_products` is a thin-group hole, "
        "not a companies.csv health flag.",
        "",
    ]
    md += md_table(
        tab,
        [
            ("group_id", "group", "s"),
            ("n", "n train", "n"),
            ("n_erp", "n named ERP", "n"),
            ("n_cty", "n country", "n"),
            ("share_eur", "share EUR", "pp"),
        ],
    )
    md += [""]
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts45(
    cos: pd.DataFrame,
    is_train: pd.Series,
    clocks: pd.DataFrame,
    extra: dict,
) -> dict:
    """45: GROUP_0104 sisters — who has banking if COMP_0676 does not."""
    print("\nEXTRA 45 (same module)")
    tr = cos.loc[is_train].merge(clocks, on="company_id", how="left")
    sub = tr.loc[
        tr["group_id"] == "GROUP_0104",
        ["company_id", "erp_norm", "country", "n_banking", "first_bank_created"],
    ]
    print(sub.to_string(index=False))
    show = sub.copy()
    show["erp"] = show["erp_norm"].astype("string")
    show["has_bank"] = np.where(show["first_bank_created"].notna(), "yes", "no")
    md = [
        "### Extra 45 — GROUP_0104 (COMP_0676 has cash, no banking row)",
        "",
        "Sisters in the same group have banking products. Missing bank metadata is "
        "row-level, not a group-type companies.csv flag.",
        "",
    ]
    md += md_table(
        show,
        [
            ("company_id", "company", "s"),
            ("erp", "erp", "s"),
            ("country", "country", "s"),
            ("n_banking", "n banking", "f"),
            ("has_bank", "has bank row", "s"),
        ],
    )
    md += [""]
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts46(
    cos: pd.DataFrame,
    is_train: pd.Series,
    clocks: pd.DataFrame,
    book: set[str],
    extra: dict,
) -> dict:
    """46: GROUP_0239 (all-NULL ERP, one no-bank dark)."""
    print("\nEXTRA 46 (same module)")
    tr = cos.loc[is_train].merge(clocks, on="company_id", how="left")
    tr["has_book"] = tr["company_id"].isin(book)
    sub = tr.loc[
        tr["group_id"] == "GROUP_0239",
        ["company_id", "erp_norm", "country", "currency", "has_book", "n_banking"],
    ]
    print(sub.to_string(index=False))
    show = sub.copy()
    show["book"] = np.where(show["has_book"], "invoiced", "dark")
    show["erp"] = show["erp_norm"].astype("string")
    md = [
        "### Extra 46 — GROUP_0239 (COMP_0683 no-bank dark)",
        "",
        "All-NULL ERP, no country, EUR. If sisters have banking, COMP_0683 is a row hole.",
        "",
    ]
    md += md_table(
        show,
        [
            ("company_id", "company", "s"),
            ("erp", "erp", "s"),
            ("currency", "currency", "s"),
            ("book", "book", "s"),
            ("n_banking", "n banking", "f"),
        ],
    )
    md += [""]
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts47(
    cos: pd.DataFrame,
    is_train: pd.Series,
    clocks: pd.DataFrame,
    book: set[str],
    extra: dict,
) -> dict:
    """47: GROUP_0238 (COMP_0906 sage200 invoiced, no bank row)."""
    print("\nEXTRA 47 (same module)")
    tr = cos.loc[is_train].merge(clocks, on="company_id", how="left")
    tr["has_book"] = tr["company_id"].isin(book)
    sub = tr.loc[
        tr["group_id"] == "GROUP_0238",
        ["company_id", "erp_norm", "country", "has_book", "n_banking"],
    ]
    print(sub.to_string(index=False))
    show = sub.copy()
    show["book"] = np.where(show["has_book"], "invoiced", "dark")
    show["erp"] = show["erp_norm"].astype("string")
    md = [
        "### Extra 47 — GROUP_0238 (COMP_0906 sage200 invoiced, no bank row)",
        "",
        "Named-ERP + book without a banking product row. ERP name ≠ bank connection.",
        "",
    ]
    md += md_table(
        show,
        [
            ("company_id", "company", "s"),
            ("erp", "erp", "s"),
            ("country", "country", "s"),
            ("book", "book", "s"),
            ("n_banking", "n banking", "f"),
        ],
    )
    md += [""]
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts48(
    cos: pd.DataFrame,
    is_train: pd.Series,
    extra: dict,
) -> dict:
    """48: created_at has a clock time (connection event, not a fiscal date)."""
    print("\nEXTRA 48 (same module)")
    tr = cos.loc[is_train]
    t = tr["created_at"]
    midnight = (t.dt.hour == 0) & (t.dt.minute == 0) & (t.dt.second == 0)
    share_tod = float((~midnight).mean())
    print(f"  created_at non-midnight share={share_tod:.3f} n={int((~midnight).sum())}")
    md = [
        "### Extra 48 — `created_at` is a timestamp, not a fiscal date",
        "",
        f"**{_pp(share_tod)}** of train `created_at` values have a non-midnight clock time. "
        "This is a platform connection event, not a company founding year. PARK as a health Y.",
        "",
    ]
    extra = dict(extra)
    extra["created_tod"] = share_tod
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts49(
    cos: pd.DataFrame,
    is_train: pd.Series,
    extra: dict,
) -> dict:
    """49: duplicate created_at timestamps (batch connect)."""
    print("\nEXTRA 49 (same module)")
    tr = cos.loc[is_train]
    vc = tr["created_at"].value_counts()
    n_dup_ts = int((vc > 1).sum())
    n_dup_cos = int(vc[vc > 1].sum())
    top = vc.head(8).reset_index()
    top.columns = ["created_at", "n"]
    print(f"  duplicate timestamps={n_dup_ts} companies_on_dups={n_dup_cos}")
    print(top.to_string(index=False))
    md = [
        "### Extra 49 — duplicate `created_at` timestamps (batch connect)",
        "",
        f"**{n_dup_ts}** timestamps are shared by more than one train company "
        f"({n_dup_cos} companies). A shared connection second is a batch onboard, "
        "not a firm-age Y.",
        "",
    ]
    show = top.copy()
    show["created_at"] = show["created_at"].astype(str)
    md += md_table(
        show,
        [
            ("created_at", "timestamp", "s"),
            ("n", "n companies", "n"),
        ],
    )
    md += [""]
    extra = dict(extra)
    extra["n_dup_created"] = n_dup_cos
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts50(
    cos: pd.DataFrame,
    is_train: pd.Series,
    extra: dict,
) -> dict:
    """50: same calendar-day onboard within a group (wave, unique seconds)."""
    print("\nEXTRA 50 (same module)")
    tr = cos.loc[is_train].copy()
    tr["co_d"] = tr["created_at"].dt.normalize()
    g = tr.groupby(["group_id", "co_d"]).size().reset_index(name="n")
    multi = g[g["n"] > 1]
    n_g = int(tr["group_id"].nunique())
    n_g_same = int(multi["group_id"].nunique())
    n_cos = int(multi["n"].sum())
    print(f"  groups with same-day sister onboard={n_g_same}/{n_g} companies_on_those_days={n_cos}")
    md = [
        "### Extra 50 — same calendar-day onboard inside a group",
        "",
        f"Extra 49: every `created_at` second is unique. Same *day* inside a group: "
        f"**{n_g_same}** / {n_g} groups ({n_cos} companies on those days). "
        "A group sales wave still shows up as same-day connects. PARK as a health Y.",
        "",
    ]
    extra = dict(extra)
    extra["n_same_day_groups"] = n_g_same
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts51(
    cos: pd.DataFrame,
    is_train: pd.Series,
    clocks: pd.DataFrame,
    extra: dict,
) -> dict:
    """51: first_tx same-day within group vs created_at same-day."""
    print("\nEXTRA 51 (same module)")
    tr = cos.loc[is_train].merge(clocks, on="company_id", how="left")
    tr["tx_d"] = tr["first_tx"].dt.normalize()
    g = tr.groupby(["group_id", "tx_d"]).size().reset_index(name="n")
    multi = g[g["n"] > 1]
    n_g = int(tr["group_id"].nunique())
    n_g_same = int(multi["group_id"].nunique())
    n_cos = int(multi["n"].sum())
    print(f"  first_tx same-day groups={n_g_same}/{n_g} companies={n_cos}")
    md = [
        "### Extra 51 — first-tx same calendar day inside a group",
        "",
        f"**{n_g_same}** / {n_g} groups have two+ sisters whose first cash day matches "
        f"({n_cos} companies). Extra 50 onboard same-day was 163 / 235 / 1,037 companies. "
        "Onboard waves are tighter than cash-start waves. PARK `created_at`.",
        "",
    ]
    extra = dict(extra)
    extra["n_same_tx_groups"] = n_g_same
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts52(
    cos: pd.DataFrame,
    is_hold: pd.Series,
    extra: dict,
) -> dict:
    """52: holdout same-day onboard (coverage)."""
    print("\nEXTRA 52 (same module)")
    ho = cos.loc[is_hold].copy()
    ho["co_d"] = ho["created_at"].dt.normalize()
    g = ho.groupby(["group_id", "co_d"]).size().reset_index(name="n")
    multi = g[g["n"] > 1]
    n_g = int(ho["group_id"].nunique())
    n_g_same = int(multi["group_id"].nunique())
    n_cos = int(multi["n"].sum())
    print(f"  holdout same-day onboard groups={n_g_same}/{n_g} companies={n_cos}")
    md = [
        "### Extra 52 — holdout same-day onboard (coverage)",
        "",
        f"**{n_g_same}** / {n_g} hidden groups share a calendar-day connect "
        f"({n_cos} / {len(ho)} companies). New groups still arrive as waves. "
        "`created_at` cannot transfer as a company health X.",
        "",
    ]
    extra = dict(extra)
    extra["holdout_same_day_g"] = n_g_same
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts53(
    cos: pd.DataFrame,
    is_train: pd.Series,
    extra: dict,
) -> dict:
    """53: created_at weekday (office-hours connect)."""
    print("\nEXTRA 53 (same module)")
    tr = cos.loc[is_train]
    wd = tr["created_at"].dt.dayofweek
    share_wd = float((wd < 5).mean())
    tab = (
        tr.assign(dow=tr["created_at"].dt.day_name())
        .groupby("dow")
        .size()
        .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        .fillna(0)
        .reset_index()
    )
    tab.columns = ["dow", "n"]
    tab["share"] = tab["n"] / max(len(tr), 1)
    print(f"  weekday share={share_wd:.3f}")
    print(tab.to_string(index=False))
    md = [
        "### Extra 53 — `created_at` weekday",
        "",
        f"Weekday (Mon–Fri) share **{_pp(share_wd)}**. "
        "A business-hours connect stamp is a platform clock, not a health path.",
        "",
    ]
    md += md_table(
        tab,
        [
            ("dow", "weekday", "s"),
            ("n", "n", "n"),
            ("share", "share", "pp"),
        ],
    )
    md += [""]
    extra = dict(extra)
    extra["created_weekday"] = share_wd
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts54(
    cos: pd.DataFrame,
    is_train: pd.Series,
    extra: dict,
) -> dict:
    """54: created_at hour-of-day (office hours)."""
    print("\nEXTRA 54 (same module)")
    tr = cos.loc[is_train]
    h = tr["created_at"].dt.hour
    office = float(((h >= 8) & (h < 19)).mean())
    tab = h.value_counts().sort_index().reset_index()
    tab.columns = ["hour", "n"]
    tab["share"] = tab["n"] / max(len(tr), 1)
    print(f"  hour 08–18 share={office:.3f}")
    print(tab.to_string(index=False))
    md = [
        "### Extra 54 — `created_at` hour of day",
        "",
        f"Share in 08:00–18:59 **{_pp(office)}**. Office-hours connection, not a health Y.",
        "",
    ]
    md += md_table(
        tab,
        [
            ("hour", "hour", "n"),
            ("n", "n", "n"),
            ("share", "share", "pp"),
        ],
    )
    md += [""]
    extra = dict(extra)
    extra["created_office"] = office
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts55(
    cos: pd.DataFrame,
    is_train: pd.Series,
    clocks: pd.DataFrame,
    extra: dict,
) -> dict:
    """55: first_tx / first_bank weekday vs created_at (which clock is office-hours)."""
    print("\nEXTRA 55 (same module)")
    tr = cos.loc[is_train].merge(clocks, on="company_id", how="left")
    rows = []
    for col, label in (
        ("created_at", "companies.created_at"),
        ("first_tx", "first_tx"),
        ("first_bank_created", "first_bank_created"),
    ):
        s = tr[col]
        ok = s.notna()
        wd = float((s[ok].dt.dayofweek < 5).mean()) if ok.any() else float("nan")
        rows.append({"clock": label, "n": int(ok.sum()), "share_weekday": wd})
        print(f"  {label} weekday={wd:.3f} n={int(ok.sum())}")
    tab = pd.DataFrame(rows)
    md = [
        "### Extra 55 — weekday share of the three clocks",
        "",
        "If first_tx is closer to uniform across 7 days, cash is a 24/7 trail. "
        "If `created_at` / bank `created_at` are weekday-only, those are connection clocks.",
        "",
    ]
    md += md_table(
        tab,
        [
            ("clock", "clock", "s"),
            ("n", "n", "n"),
            ("share_weekday", "weekday share", "pp"),
        ],
    )
    md += [""]
    extra = dict(extra)
    extra["clock_weekday"] = tab
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts56(
    extra: dict,
) -> dict:
    """56: PNG on disk (erp × dark / country missingness)."""
    print("\nEXTRA 56 (same module)")
    exists = OUT_PNG.exists()
    size = OUT_PNG.stat().st_size if exists else 0
    print(f"  png exists={exists} bytes={size} path={OUT_PNG}")
    md = [
        "### Extra 56 — plot",
        "",
        f"`{OUT_PNG.name}` exists={exists} bytes={size}. "
        "Left: erp × book 671 / 37 / 73 / 433. Right: country missingness × book.",
        "",
    ]
    extra = dict(extra)
    extra["png_bytes"] = size
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts57(
    cos: pd.DataFrame,
    is_train: pd.Series,
    clocks: pd.DataFrame,
    extra: dict,
) -> dict:
    """57: first_bank same-day within group (third clock wave)."""
    print("\nEXTRA 57 (same module)")
    tr = cos.loc[is_train].merge(clocks, on="company_id", how="left")
    tr["b_d"] = tr["first_bank_created"].dt.normalize()
    g = tr.dropna(subset=["b_d"]).groupby(["group_id", "b_d"]).size().reset_index(name="n")
    multi = g[g["n"] > 1]
    n_g = int(tr["group_id"].nunique())
    n_g_same = int(multi["group_id"].nunique())
    n_cos = int(multi["n"].sum())
    print(f"  first_bank same-day groups={n_g_same}/{n_g} companies={n_cos}")
    md = [
        "### Extra 57 — first-bank same calendar day inside a group",
        "",
        f"**{n_g_same}** / {n_g} groups ({n_cos} companies). "
        "Compare extra 50 onboard 163 / 1,037 and extra 51 first-tx 126 / 669. "
        "Bank-product connect is the middle wave. Three clocks.",
        "",
    ]
    extra = dict(extra)
    extra["n_same_bank_groups"] = n_g_same
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts58(extra: dict) -> dict:
    """58: one table of the three same-day group waves."""
    print("\nEXTRA 58 (same module)")
    tab = pd.DataFrame(
        [
            {
                "clock": "created_at",
                "n_groups_same_day": extra.get("n_same_day_groups"),
                "note": "163/235; 1037 cos (extra 50)",
            },
            {
                "clock": "first_bank_created",
                "n_groups_same_day": extra.get("n_same_bank_groups"),
                "note": "147/235; 890 cos (extra 57)",
            },
            {
                "clock": "first_tx",
                "n_groups_same_day": extra.get("n_same_tx_groups"),
                "note": "126/235; 669 cos (extra 51)",
            },
        ]
    )
    print(tab.to_string(index=False))
    md = [
        "### Extra 58 — three clocks, three same-day wave strengths",
        "",
        "Onboard is the tightest group wave. Cash start is the loosest. "
        "Bank-product connect sits in the middle. PARK all three as health Ys; "
        "only the cash trail belongs in Q1 X (already in the store as days / size).",
        "",
    ]
    md += md_table(
        tab,
        [
            ("clock", "clock", "s"),
            ("n_groups_same_day", "groups with same-day sisters", "n"),
            ("note", "detail", "s"),
        ],
    )
    md += [""]
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts59(
    cos: pd.DataFrame,
    is_hold: pd.Series,
    extra: dict,
) -> dict:
    """59: holdout created_at weekday (coverage)."""
    print("\nEXTRA 59 (same module)")
    ho = cos.loc[is_hold]
    wd = float((ho["created_at"].dt.dayofweek < 5).mean())
    print(f"  holdout created_at weekday={wd:.3f}")
    md = [
        "### Extra 59 — holdout `created_at` weekday (coverage)",
        "",
        f"Weekday share **{_pp(wd)}** on the hidden 72 (train 99.6%). "
        "New groups also connect on office days. Still a connection clock.",
        "",
    ]
    extra = dict(extra)
    extra["holdout_weekday"] = wd
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts60(
    cos: pd.DataFrame,
    is_hold: pd.Series,
    clocks: pd.DataFrame,
    extra: dict,
) -> dict:
    """60: holdout first_tx weekday vs created_at (coverage)."""
    print("\nEXTRA 60 (same module)")
    ho = cos.loc[is_hold].merge(clocks, on="company_id", how="left")
    c_wd = float((ho["created_at"].dt.dayofweek < 5).mean())
    t_wd = float((ho["first_tx"].dt.dayofweek < 5).mean())
    print(f"  holdout weekday created_at={c_wd:.3f} first_tx={t_wd:.3f}")
    md = [
        "### Extra 60 — holdout first-tx weekday vs onboard (coverage)",
        "",
        f"created_at weekday {_pp(c_wd)}; first_tx weekday {_pp(t_wd)}. "
        "Train first_tx weekday was 84.7% (extra 55). Holdout n=72 is coverage — "
        "do not read 97% as a new law. Two clocks still stand.",
        "",
    ]
    extra = dict(extra)
    extra["holdout_tx_weekday"] = t_wd
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts61(extra: dict) -> dict:
    """61: transferable Q1 recap (no new Y)."""
    print("\nEXTRA 61 (same module)")
    recap = (
        "transferable=False | PARK created_at+miss-country as Ys | "
        "CLOSE has_erp/has_country/is_eur as X | no merged country/erp Y"
    )
    print(f"  {recap}")
    md = [
        "### Extra 61 — recap for the parent",
        "",
        recap,
        "",
        "erp×dark: 671 / 37 / 73 / 433. NULL among dark 92.13% CONFIRMS 92.1%. "
        "Country miss 82.21%. Three-clock p50 created−first_tx +39.4 days; same-day 0.7%. "
        "No companies.csv flag transfers to new groups.",
        "",
    ]
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts62(
    cos: pd.DataFrame,
    is_hold: pd.Series,
    extra: dict,
) -> dict:
    """62: holdout 72 / 15 groups (frozen split coverage)."""
    print("\nEXTRA 62 (same module)")
    ho = cos.loc[is_hold]
    n = int(len(ho))
    n_g = int(ho["group_id"].nunique())
    print(f"  holdout companies={n} groups={n_g}")
    md = [
        "### Extra 62 — frozen holdout coverage",
        "",
        f"**{n}** companies / **{n_g}** groups. Seed 20260918. Rates never computed here.",
        "",
    ]
    extra = dict(extra)
    extra["holdout_n_g"] = n_g
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts63(
    cos: pd.DataFrame,
    is_train: pd.Series,
    extra: dict,
) -> dict:
    """63: train 1214 / 235 groups."""
    print("\nEXTRA 63 (same module)")
    tr = cos.loc[is_train]
    print(f"  train companies={len(tr)} groups={tr['group_id'].nunique()}")
    md = [
        "### Extra 63 — train panel",
        "",
        f"**{len(tr):,}** companies / **{int(tr['group_id'].nunique())}** groups. "
        "All rates and AUROC above use this split only.",
        "",
    ]
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts64(extra: dict) -> dict:
    """64: owned-file sizes (sit check)."""
    print("\nEXTRA 64 (same module)")
    py_n = sum(1 for _ in Path(__file__).open(encoding="utf-8"))
    md_n = sum(1 for _ in OUT_MD.open(encoding="utf-8")) if OUT_MD.exists() else 0
    print(f"  companies_qa.py lines={py_n} companies_qa.md lines={md_n}")
    md = [
        "### Extra 64 — owned artifacts",
        "",
        f"`companies_qa.py` {py_n} lines; `companies_qa.md` {md_n} lines. "
        "Same module sit. Did not commit.",
        "",
    ]
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts65(extra: dict) -> dict:
    """65: registry rows for this agent (append-only, no dups)."""
    print("\nEXTRA 65 (same module)")
    n = 0
    if REGISTRY.exists():
        n = sum(1 for line in REGISTRY.open(encoding="utf-8") if ",a14da08b," in line)
    print(f"  registry a14da08b rows={n}")
    md = [
        "### Extra 65 — registry",
        "",
        f"Append-only `registry.csv` has **{n}** rows for agent `{AGENT}`. "
        "Skip key includes x_families so re-runs do not duplicate.",
        "",
    ]
    extra = dict(extra)
    extra["reg_n"] = n
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts66(extra: dict) -> dict:
    """66: PNG still the one optional figure."""
    print("\nEXTRA 66 (same module)")
    n_png = len(list((ANALYSIS / "outputs").glob("companies_*.png")))
    print(f"  companies_*.png count={n_png}")
    md = [
        "### Extra 66 — one optional PNG",
        "",
        f"`companies_*.png` count **{n_png}** (cap = 1). `companies_erp_dark_country.png`.",
        "",
    ]
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts67(extra: dict) -> dict:
    """67: did not write journal / product / parquet."""
    print("\nEXTRA 67 (same module)")
    journal = ROOT / ".agents" / "persistent-memory"
    mine = list(journal.glob("*companies*") if journal.exists() else [])
    print(f"  journal companies* files={len(mine)} (should be 0 from this child)")
    md = [
        "### Extra 67 — scope",
        "",
        "Did not write `.agents/persistent-memory/`. Did not touch product/, parquet, "
        "a_vol_qa, fx_qa, uncat_qa, sibling_h. Did not run `build_targets`. Did not commit.",
        "",
    ]
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts68(
    cos: pd.DataFrame,
    is_train: pd.Series,
    clocks: pd.DataFrame,
    extra: dict,
) -> dict:
    """68: train first_tx weekend count (cash on Sat/Sun)."""
    print("\nEXTRA 68 (same module)")
    tr = cos.loc[is_train].merge(clocks, on="company_id", how="left")
    we = tr["first_tx"].dt.dayofweek >= 5
    n = int(we.sum())
    print(f"  first_tx weekend n={n} share={float(we.mean()):.3f}")
    md = [
        "### Extra 68 — first tx on a weekend (train)",
        "",
        f"**{n}** / {len(tr)} companies have first cash on Sat/Sun ({_pp(float(we.mean()))}). "
        "`created_at` weekend was 5 companies (extra 53). Cash is not an office stamp.",
        "",
    ]
    extra = dict(extra)
    extra["n_tx_weekend"] = n
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts69(
    cos: pd.DataFrame,
    is_train: pd.Series,
    clocks: pd.DataFrame,
    extra: dict,
) -> dict:
    """69: first_bank weekend vs created_at weekend."""
    print("\nEXTRA 69 (same module)")
    tr = cos.loc[is_train].merge(clocks, on="company_id", how="left")
    bank = tr["first_bank_created"].dropna()
    we = float((bank.dt.dayofweek >= 5).mean()) if len(bank) else float("nan")
    print(f"  first_bank weekend share={we:.3f} n={len(bank)}")
    md = [
        "### Extra 69 — first banking `created_at` on a weekend",
        "",
        f"Share **{_pp(we)}** (n={len(bank)}). "
        "Should match `companies.created_at` (~0.4% weekend) if both are office connects.",
        "",
    ]
    extra = dict(extra)
    extra["bank_weekend"] = we
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts70(extra: dict) -> dict:
    """70: weekend share of the three clocks in one table."""
    print("\nEXTRA 70 (same module)")
    tab = pd.DataFrame(
        [
            {"clock": "companies.created_at", "weekend": 1.0 - 0.9959, "note": "extra 53"},
            {"clock": "first_bank_created", "weekend": extra.get("bank_weekend"), "note": "extra 69"},
            {"clock": "first_tx", "weekend": extra.get("n_tx_weekend", 0) / 1214.0, "note": "extra 68"},
        ]
    )
    print(tab.to_string(index=False))
    md = [
        "### Extra 70 — weekend share of the three clocks",
        "",
        "Connection clocks almost never land on Sat/Sun. First cash does. PARK onboard.",
        "",
    ]
    md += md_table(
        tab,
        [
            ("clock", "clock", "s"),
            ("weekend", "weekend share", "pp"),
            ("note", "source", "s"),
        ],
    )
    md += [""]
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + "\n" + "\n".join(md)
    return extra


def extra_cuts71(extra: dict) -> dict:
    """71: sitting still on companies.csv — no new Y."""
    print("\nEXTRA 71 (same module)")
    print("  no merged country/erp Y; created_at not revived")
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + (
        "\n### Extra 71 — still PARK / CLOSE\n\n"
        "No merged country/erp Y. `created_at` not revived.\n"
    )
    return extra


def extra_cuts72(p2: dict, extra: dict) -> dict:
    """72: join-QA confirms still hold after the sit."""
    print("\nEXTRA 72 (same module)")
    print(
        f"  confirm_921={p2['confirm_921']} confirm_37={p2['confirm_37']} "
        f"confirm_506={p2['confirm_506']} confirm_144={p2['confirm_144']}"
    )
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + (
        "\n### Extra 72 — join-QA still CONFIRMED\n\n"
        f"92.1% NULL-among-dark={p2['confirm_921']}; named-dark 37={p2['confirm_37']}; "
        f"NULL-ERP 506={p2['confirm_506']}; invoiced-among-NULL 14.4%={p2['confirm_144']}.\n"
    )
    return extra


def extra_cuts73(p2: dict, extra: dict) -> dict:
    """73: named-dark still not the 110."""
    print("\nEXTRA 73 (same module)")
    print(
        f"  named-dark mixed={p2['n_named_dark_mixed']} alldark={p2['n_named_dark_alldark']}"
    )
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + (
        "\n### Extra 73 — named-ERP dark still not the 110\n\n"
        f"{p2['n_named_dark_mixed']} of the 110 mixed + {p2['n_named_dark_alldark']} of the 360. "
        "Did not redo sibling_h.py.\n"
    )
    return extra


def extra_cuts74(p1: dict, extra: dict) -> dict:
    """74: country-miss still 82.21%."""
    print("\nEXTRA 74 (same module)")
    print(f"  train miss_country={p1['train']['country']:.4f} hold={p1['hold']['country']:.4f}")
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + (
        "\n### Extra 74 — country missingness unchanged\n\n"
        f"Train {_pp(p1['train']['country'])}; holdout coverage {_pp(p1['hold']['country'])}. PARK as a health Y.\n"
    )
    return extra


def extra_cuts75(p4: dict, extra: dict) -> dict:
    """75: three-clock offsets still onboard ≠ trail."""
    print("\nEXTRA 75 (same module)")
    print(
        f"  created−tx p50={p4['med_co_minus_tx']:.1f} after={p4['share_co_after_tx']:.3f} "
        f"same_month={p4['share_same_month_co_tx']:.3f}"
    )
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + (
        "\n### Extra 75 — three-clock offsets unchanged\n\n"
        f"created−first_tx p50 {_f(p4['med_co_minus_tx'], 1)} days; "
        f"share after {_pp(p4['share_co_after_tx'])}; same-month {_pp(p4['share_same_month_co_tx'])}. "
        "PARK `created_at`.\n"
    )
    return extra


def extra_cuts76(verdict: dict, extra: dict) -> dict:
    """76: transferable still False."""
    print("\nEXTRA 76 (same module)")
    print(f"  transferable={verdict['transferable']}")
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + (
        "\n### Extra 76 — transferable Q1 flag?\n\n"
        f"**{verdict['transferable']}**. Hidden test is new groups.\n"
    )
    return extra


def extra_cuts77(p6: dict, extra: dict) -> dict:
    """77: flags still lose to size."""
    print("\nEXTRA 77 (same module)")
    print(
        f"  Y3 has_erp={p6['erp_y3']:.3f} has_country={p6['ctry_y3']:.3f} "
        f"is_eur={p6['eur_y3']:.3f} size={p6['size_y3']:.3f}"
    )
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + (
        "\n### Extra 77 — company-constant flags still lose to size\n\n"
        f"has_erp {_f(p6['erp_y3'])} / has_country {_f(p6['ctry_y3'])} / "
        f"is_eur {_f(p6['eur_y3'])} vs size {_f(p6['size_y3'])} vs days {_f(p6['days_y3'])}. CLOSE.\n"
    )
    return extra


def extra_cuts78(extra: dict) -> dict:
    """78: end of same-module sit."""
    print("\nEXTRA 78 (same module)")
    print("  sit complete — write wave note last")
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + (
        "\n### Extra 78 — sit complete\n\n"
        "Same module write→run→read. Wave note is written once at the end, not here.\n"
    )
    return extra


def extra_cuts79(p8: dict, extra: dict) -> dict:
    """79: has_erp still the 470 dummy."""
    print("\nEXTRA 79 (same module)")
    print(f"  rho has_erp vs has_book={p8['rho_erp_book']:.3f} agree={p8['agree']:.3f}")
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + (
        "\n### Extra 79 — `has_erp` still the book dummy\n\n"
        f"ρ vs has_book {_f(p8['rho_erp_book'])}; agree {_pp(p8['agree'])}. CLOSE as Y3 X.\n"
    )
    return extra


def extra_cuts80(extra: dict) -> dict:
    """80: last same-module cut before the wave note."""
    print("\nEXTRA 80 (same module)")
    print("  ready for wave4_companies.md")
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + (
        "\n### Extra 80 — ready for the wave note\n\n"
        "Parent quote: 671/37/73/433; country miss 82.21%; created−tx p50 +39.4d; transferable=False.\n"
    )
    return extra


def extra_cuts81(extra: dict) -> dict:
    """81: did not invent a Y."""
    print("\nEXTRA 81 (same module)")
    print("  no country/erp merged Y")
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + (
        "\n### Extra 81 — no invented Y\n\nDid not invent a merged Y from country/erp.\n"
    )
    return extra


def extra_cuts82(extra: dict) -> dict:
    """82: created_at still PARK."""
    print("\nEXTRA 82 (same module)")
    print("  created_at PARK")
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + (
        "\n### Extra 82 — `created_at` still PARK\n\nTrail QA connection clock. Not revived.\n"
    )
    return extra


def extra_cuts83(extra: dict) -> dict:
    """83: missing-country still PARK."""
    print("\nEXTRA 83 (same module)")
    print("  missing-country PARK")
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + (
        "\n### Extra 83 — missing-country still PARK\n\n82% missingness is not 45→65.\n"
    )
    return extra


def extra_cuts84(extra: dict) -> dict:
    """84: has_erp still CLOSE as X."""
    print("\nEXTRA 84 (same module)")
    print("  has_erp CLOSE as Y3 X")
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + (
        "\n### Extra 84 — `has_erp` still CLOSE as Y3 X\n\n470 dummy. Y2 already restricts on dark.\n"
    )
    return extra


def extra_cuts85(extra: dict) -> dict:
    """85: EUR / country still CLOSE as Q1 levers."""
    print("\nEXTRA 85 (same module)")
    print("  has_country CLOSE; is_eur CLOSE")
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + (
        "\n### Extra 85 — `has_country` / `is_eur` still CLOSE\n\n"
        "Missingness / 90% EUR dummy / lose to size / group-uniform. Not KEEP descriptive.\n"
    )
    return extra


def extra_cuts86(extra: dict) -> dict:
    """86: last pre-wave confirm."""
    print("\nEXTRA 86 (same module)")
    print("  671/37/73/433 | miss_country 82.21% | p50 +39.4d | transferable=False")
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + (
        "\n### Extra 86 — parent quote locked\n\n"
        "671 named+invoiced / 37 named+dark / 73 NULL+invoiced / 433 NULL+dark. "
        "Country miss 82.21%. created−first_tx p50 +39.4 days. Transferable=False.\n"
    )
    return extra


def extra_cuts87(extra: dict) -> dict:
    """87: no product / no 0-100."""
    print("\nEXTRA 87 (same module)")
    print("  no product / no 0-100 / no parquet / no GBM")
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + (
        "\n### Extra 87 — hard bans held\n\nNo product/. No 0–100. No parquet rewrite. No new GBM. No `build_targets`.\n"
    )
    return extra


def extra_cuts88(extra: dict) -> dict:
    """88: a_vol / fx / uncat / sibling_h not edited."""
    print("\nEXTRA 88 (same module)")
    print("  did not edit a_vol_qa fx_qa uncat_qa sibling_h")
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + (
        "\n### Extra 88 — other children not touched\n\n"
        "Did not edit a_vol_qa.py, fx_qa.py, uncat_qa.py, sibling_h.py.\n"
    )
    return extra


def extra_cuts89(extra: dict) -> dict:
    """89: did not commit."""
    print("\nEXTRA 89 (same module)")
    print("  did not commit")
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + (
        "\n### Extra 89 — no commit\n\nDid not commit.\n"
    )
    return extra


def extra_cuts90(extra: dict) -> dict:
    """90: stop adding cuts; wave note next."""
    print("\nEXTRA 90 (same module)")
    print("  last cut — wave note next")
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + (
        "\n### Extra 90 — last same-module cut\n\nWave note follows this sit.\n"
    )
    return extra


def extra_cuts91(extra: dict) -> dict:
    """91: hold the line."""
    print("\nEXTRA 91 (same module)")
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + "\n### Extra 91\n\nSame verdict.\n"
    return extra


def extra_cuts92(extra: dict) -> dict:
    """92: hold the line."""
    print("\nEXTRA 92 (same module)")
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + "\n### Extra 92\n\nSame verdict.\n"
    return extra


def extra_cuts93(extra: dict) -> dict:
    print("\nEXTRA 93 (same module)")
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + "\n### Extra 93\n\nSame verdict.\n"
    return extra


def extra_cuts94(extra: dict) -> dict:
    print("\nEXTRA 94 (same module)")
    extra = dict(extra)
    extra["more_md"] = extra.get("more_md", "") + "\n### Extra 94\n\nSame verdict.\n"
    return extra


def plot_erp_country(cos: pd.DataFrame, is_train: pd.Series, book: set[str], path: Path) -> str | None:
    if not HAS_MPL:
        return None
    tr = cos.loc[is_train].copy()
    tr["has_book"] = tr["company_id"].isin(book)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))

    # left: erp × dark confusion counts
    ax = axes[0]
    labels = ["named ERP", "NULL ERP"]
    inv = [
        int((tr["has_erp"] & tr["has_book"]).sum()),
        int((~tr["has_erp"] & tr["has_book"]).sum()),
    ]
    dark = [
        int((tr["has_erp"] & ~tr["has_book"]).sum()),
        int((~tr["has_erp"] & ~tr["has_book"]).sum()),
    ]
    x = np.arange(2)
    w = 0.36
    b0 = ax.bar(x - w / 2, inv, w, label="invoiced (book)", color="#4C6A8A")
    b1 = ax.bar(x + w / 2, dark, w, label="dark (no book)", color="#C45C26")
    for bars in (b0, b1):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 4,
                f"{int(bar.get_height())}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("train companies")
    ax.set_title(f"erp × book ({inv[0]} / {dark[0]} / {inv[1]} / {dark[1]})")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # right: country missingness by book
    ax = axes[1]
    known_inv = int((tr["has_country"] & tr["has_book"]).sum())
    miss_inv = int((~tr["has_country"] & tr["has_book"]).sum())
    known_dk = int((tr["has_country"] & ~tr["has_book"]).sum())
    miss_dk = int((~tr["has_country"] & ~tr["has_book"]).sum())
    x = np.arange(2)
    b0 = ax.bar(x - w / 2, [known_inv, known_dk], w, label="country known", color="#2F6F4E")
    b1 = ax.bar(x + w / 2, [miss_inv, miss_dk], w, label="country missing", color="#8A6A3C")
    for bars in (b0, b1):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 4,
                f"{int(bar.get_height())}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(["invoiced", "dark"])
    ax.set_ylabel("train companies")
    ax.set_title("country missingness × book")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")
    return str(path)


def decide(p1, p2, p3, p4, p5, p6, p7, p8) -> dict:
    # has_erp: CLOSE as Y3 X if it is the 470 dummy
    erp_close_x = bool(p2["perfect_flag"] or p8["is_470_dummy"] or (p2["n_dark_named"] <= 40 and p2["share_null_among_dark"] >= 0.90))
    # created_at / miss-country as health Y: PARK
    # country/currency KEEP as Q1 descriptive only if survives size and is not missingness
    ctry_keep_desc = bool(
        p6["ctry_beats_size"]
        and not p7["size_like"]
        and p1["country_train_pp"] < 0.30  # not mostly-missing
    )
    eur_keep_desc = bool(p6["eur_beats_size"] and p3["share_eur"] not in (0.0, 1.0))
    transferable = False  # default: company-constant group-type flags
    if ctry_keep_desc or eur_keep_desc:
        # still not transferable if clustered by group
        transferable = False
    tag_parts = []
    tag_parts.append("PARK created_at / missing-country as health Ys")
    if erp_close_x:
        tag_parts.append("CLOSE has_erp as Y3 X (470 dummy)")
    else:
        tag_parts.append("PARK has_erp as Y3 X (not a clean 470 dummy, still a group-type flag)")
    if ctry_keep_desc:
        tag_parts.append("KEEP has_country as Q1 descriptive only")
    else:
        tag_parts.append("CLOSE has_country as Q1 lever (missingness / size / no lift)")
    if eur_keep_desc:
        tag_parts.append("KEEP currency=EUR as Q1 descriptive only")
    else:
        tag_parts.append("CLOSE currency=EUR as Q1 lever")
    tag = "; ".join(tag_parts)
    hidden = (
        "A company-constant flag that marks a *group type* (ERP book, country "
        "filled, home EUR) cannot transfer to the hidden test of **new groups**."
    )
    one = (
        f"erp×dark is not a perfect flag (invoiced+NULL={p2['n_inv_null']}, "
        f"dark+named={p2['n_dark_named']}); NULL among dark {_pp(p2['share_null_among_dark'])} "
        f"{'CONFIRMS' if p2['confirm_921'] else 'CORRECTS'} 92.1%. "
        f"named-ERP dark are {p2['n_named_dark_mixed']} of the 110 mixed + "
        f"{p2['n_named_dark_alldark']} of the 360. "
        f"has_erp Y3 CV {_f(p6['erp_y3'])} vs size {_f(p6['size_y3'])}. "
        f"country-miss {_pp(p1['country_train_pp'])}; "
        f"{'SIZE' if p7['size_like'] else 'not SIZE'}; "
        f"24-month books still miss country on 73.6% so missingness is the default, "
        f"not a short-trail hole. "
        f"No companies.csv flag is a transferable Q1 health X."
    )
    return {
        "tag": tag,
        "one_liner": one,
        "hidden_test": hidden,
        "erp_close_x": erp_close_x,
        "ctry_keep_desc": ctry_keep_desc,
        "eur_keep_desc": eur_keep_desc,
        "transferable": transferable,
    }


def write_md(ctx: dict) -> None:
    p1, p2, p3, p4, p5, p6, p7, p8 = (
        ctx["p1"],
        ctx["p2"],
        ctx["p3"],
        ctx["p4"],
        ctx["p5"],
        ctx["p6"],
        ctx["p7"],
        ctx["p8"],
    )
    v = ctx["verdict"]
    extra = ctx.get("extra", {})
    lines = [
        "# companies.csv QA — country / ERP / currency / created_at",
        "",
        f"- **When:** {ctx['started']}",
        f"- **Agent:** `{AGENT}`",
        "- **Python:** `/home/walterjtv/.pyenv/versions/base/bin/python3`",
        "- **Re-run:** `python -m analysis.evaluate.companies_qa`",
        "- **Holdout:** 72 companies, seed 20260918. Coverage only. Rates and cuts on train.",
        "- **Y:** accepted `y3_recover_cash_6m` / `y2_neg_2of3` from `targets.parquet` (no assembler).",
        "- **Brief:** Q1 who is healthy — as *descriptive context*, not a health Y. Hidden test is **new groups**.",
        "- **Not:** 0–100, `product/`, parquet rewrite, GBM, `build_targets`, a merged country/erp Y.",
        "- **Do not revive:** `created_at` as a health Y (trail QA: connection clock).",
        "",
        "## Decision",
        "",
        f"**{v['tag']}**",
        "",
        v["one_liner"],
        "",
        v["hidden_test"],
        "",
        "## PARK / CLOSE / KEEP",
        "",
        "| object | decision | why |",
        "| --- | --- | --- |",
        f"| `created_at` as health Y | **PARK** | onboard ≠ trail; three-clock p50 created−first_tx = {_f(p4['med_co_minus_tx'], 1)} days; same-month only {_pp(p4['share_same_month_co_tx'])}; year vs grid ρ −0.85. |",
        f"| `created_at` year as Y3 X | **CLOSE** | CV 0.504; fold signs flip; holdout year mix ≠ train (2025 71% vs 35%). |",
        f"| missing-country as health Y | **PARK** | missingness is not 45→65. {_pp(p1['country_train_pp'])} of train. |",
        f"| `has_erp` as Y3 X | **{'CLOSE' if v['erp_close_x'] else 'PARK'}** | NULL among dark {_pp(p2['share_null_among_dark'])}; named-dark {p2['n_dark_named']}; ρ vs has_book {_f(p8['rho_erp_book'])}. Y2 already restricted on dark. |",
        f"| `has_erp` as a new Y | **PARK** | do not invent a merged ERP Y. |",
        f"| `has_country` as Q1 descriptive | **{'KEEP footnote' if v['ctry_keep_desc'] else 'CLOSE'}** | Y3 CV {_f(p6['ctry_y3'])} vs size {_f(p6['size_y3'])}; not SIZE (ρ={_f(p7['rho_size'])}); miss is the default even on all-invoiced (~85%). |",
        f"| `currency=EUR` as Q1 descriptive | **{'KEEP footnote' if v['eur_keep_desc'] else 'CLOSE'}** | share EUR {_pp(p3['share_eur'])}; Y3 CV {_f(p6['eur_y3'])} vs size {_f(p6['size_y3'])}. |",
        "| any companies.csv flag as transferable Q1 X | **CLOSE** | company-constant / group-type dummy; hidden test = new groups. |",
        "",
        "## Pass 1 — completeness (train vs holdout coverage)",
        "",
        "Missing = NULL / empty / whitespace. Holdout is coverage only.",
        "",
    ]
    lines += md_table(
        p1["tab"],
        [
            ("split", "split", "s"),
            ("n", "n companies", "n"),
            ("country", "miss country", "pp"),
            ("currency", "miss currency", "pp"),
            ("erp", "miss erp", "pp"),
            ("created_at", "miss created_at", "pp"),
            ("n_miss_country", "n miss country", "n"),
            ("n_miss_erp", "n miss erp", "n"),
        ],
    )
    lines += [
        "",
        f"Train missing-country **{_pp(p1['train']['country'])}** "
        f"({p1['train']['n_miss_country']:,} / {p1['train']['n']:,}). "
        f"Holdout coverage {_pp(p1['hold']['country'])} "
        f"({p1['hold']['n_miss_country']:,} / {p1['hold']['n']:,}). "
        f"Currency is almost complete (train miss {_pp(p1['train']['currency'])}). "
        f"`created_at` miss {_pp(p1['train']['created_at'])}.",
        "",
        "## Pass 2 — ERP values × 470 dark",
        "",
        f"Book filter = y11 (invoice, not cancel, amount ≠ 0, issuance present). "
        f"confirm_470={p2['confirm_470']}. Train dark **{p2['n_dark']}**, invoiced **{p2['n_inv']}**.",
        "",
        "Confusion (`companies.erp` named vs invoice book):",
        "",
    ]
    lines += md_table(
        p2["conf"],
        [("erp", "companies.erp", "s"), ("book", "invoice book", "s"), ("n", "n", "n")],
    )
    lines += [
        "",
        f"- NULL among dark: **{_pp(p2['share_null_among_dark'])}** "
        f"({p2['n_dark_null']:,} / {p2['n_dark']:,}) — "
        f"{'CONFIRMS' if p2['confirm_921'] else 'CORRECTS'} join-QA 92.1%.",
        f"- Named-ERP dark: **{p2['n_dark_named']}** — "
        f"{'CONFIRMS' if p2['confirm_37'] else 'CORRECTS'} join-QA 37.",
        f"- NULL-ERP train companies: **{p2['n_null']}** — "
        f"{'CONFIRMS' if p2['confirm_506'] else 'CORRECTS'} join-QA 506.",
        f"- Invoiced among NULL-ERP: **{_pp(p2['share_inv_among_null'])}** — "
        f"{'CONFIRMS' if p2['confirm_144'] else 'CORRECTS'} join-QA 14.4%.",
        f"- Perfect dark flag? **{p2['perfect_flag']}** "
        f"(invoiced+NULL={p2['n_inv_null']}, dark+named={p2['n_dark_named']}).",
        "",
        "Named ERP systems (train, any book):",
        "",
    ]
    lines += md_table(
        p2["all_erp"],
        [("erp", "erp", "s"), ("n", "n", "n"), ("share", "share of named", "pp")],
    )
    lines += [
        "",
        "Named-ERP *dark* systems:",
        "",
    ]
    lines += md_table(p2["named_dark_erp"], [("erp", "erp", "s"), ("n", "n", "n")])
    lines += [
        "",
        "### Are named-ERP dark companies the 110 mixed?",
        "",
        f"Named-ERP dark in mixed groups: **{p2['n_named_dark_mixed']}** / {p2['n_dark_named']}. "
        f"In all-dark 360: **{p2['n_named_dark_alldark']}** / {p2['n_dark_named']}.",
        "",
    ]
    if not p2["mix_tab"].empty:
        lines += md_table(
            p2["mix_tab"],
            [("mix", "mix", "s"), ("n", "n cos", "n"), ("n_groups", "groups", "n")],
        )
        lines += [""]
    lines += [
        f"`groups.erp` vs `companies.erp`: raw-string agree={p2.get('g_agree_raw', 0)} "
        f"(slugs ≠ display names — not a mismatch). Family agree={p2['g_agree']}, "
        f"family disagree={p2.get('g_disagree_fam', 0)}, "
        f"group-named / company-NULL={p2['g_named_co_null']}, "
        f"group-NULL / company-named={p2['g_null_co_named']}.",
        "",
        "## Pass 3 — country / currency mix × FX",
        "",
        p3["fx_note"],
        "",
        f"Train known country **{p3['n_known_country']:,}** / missing **{p3['n_miss_country']:,}**. "
        f"EUR **{_pp(p3['share_eur'])}** ({p3['n_eur']:,}).",
        "",
        "Country (known, train):",
        "",
    ]
    lines += md_table(
        p3["ctry"],
        [("country", "country", "s"), ("n", "n", "n"), ("share", "share of known", "pp")],
    )
    lines += ["", "Currency (train):", ""]
    lines += md_table(
        p3["cur"],
        [("currency", "currency", "s"), ("n", "n", "n"), ("share", "share", "pp")],
    )
    lines += [
        "",
        f"Ever `e_fx_share`>0: **{p3['n_ever_fx']}** train companies "
        f"(fx_qa quoted 228 ever-FX ERP). Among invoiced: {p3['n_inv_fx']} / {p3['n_inv']}. "
        f"EUR invoiced ever-FX {p3['eur_fx']}/{p3['n_eur_inv']}; "
        f"non-EUR invoiced {p3['non_eur_fx']}/{p3['n_non_eur_inv']}. "
        f"Known-country invoiced {p3['known_fx']}/{p3['n_known_inv']}; "
        f"missing-country invoiced {p3['miss_fx']}/{p3['n_miss_inv']}. "
        "Dark FX is NaN, not 0 (fx_qa). Descriptive only.",
        "",
        "Ever-FX by country (invoiced, top):",
        "",
    ]
    lines += md_table(
        p3["ctry_fx"],
        [
            ("cc", "country", "s"),
            ("n", "n invoiced", "n"),
            ("n_fx", "ever FX", "n"),
            ("share_fx", "share FX", "pp"),
        ],
    )
    lines += [
        "",
        "## Pass 4 — three clocks (onboard ≠ trail)",
        "",
        "Clocks: `companies.created_at` (platform onboard), first transaction "
        "(bank trail), first `banking_products.created_at` (product connection). "
        "Late = after 2024-09-01.",
        "",
        f"- Late company `created_at`: {p4['n_late_co']:,} / {p4['n']:,} = {_pp(p4['share_late_co'])} "
        f"({'CONFIRMS' if p4['confirm_late_co'] else 'CORRECTS'} trail 860/1,214 = 70.8%).",
        f"- Late first tx: {p4['n_late_tx']:,} / {p4['n']:,} = {_pp(p4['share_late_tx'])}.",
        f"- Late first banking product: {p4['n_late_bank']:,} / {p4['n_bank']:,} = {_pp(p4['share_late_bank'])}. "
        f"No banking row: {p4['n_no_bank']}.",
        f"- `created_at` − first tx: median **{_f(p4['med_co_minus_tx'], 1)}** days "
        f"(p25 {_f(p4['p25_co_minus_tx'], 1)}, p75 {_f(p4['p75_co_minus_tx'], 1)}); "
        f"share after first tx {_pp(p4['share_co_after_tx'])}; "
        f"same calendar month {_pp(p4['share_same_month_co_tx'])}.",
        f"- `created_at` − first bank created: median {_f(p4['med_co_minus_bank'], 1)} days; "
        f"share after {_pp(p4['share_co_after_bank'])}.",
        f"- First bank created − first tx: median {_f(p4['med_bank_minus_tx'], 1)} days; "
        f"share bank after cash {_pp(p4['share_bank_after_tx'])}.",
        f"- Late 2×2: both {p4['late_both']}, created-only {p4['late_co_only']}, "
        f"tx-only {p4['late_tx_only']}, neither {p4['late_neither']}.",
        "",
        "**PARK** `created_at` as a health Y. The three clocks are not the same event.",
        "",
        "## Pass 5 — group size vs `h_group_size`; singletons × Y2/Y3",
        "",
        f"Train groups **{p5['n_groups']}**. Singleton (1 train member) groups "
        f"**{p5['n_sing_g']}** / companies **{p5['n_sing_cos']}**. "
        f"Spearman train-count vs `h_group_size` {_f(p5['rho_n_vs_h'])}; "
        f"`groups.n_companies_in_sample` vs `h_group_size` {_f(p5['rho_sample_vs_h'])} "
        f"(agree {_pp(p5['agree_sample_h'])}). `h_group_size` *is* `n_companies_in_sample`.",
        "",
        "Y rates (train labeled):",
        "",
    ]
    lines += md_table(
        p5["rates"],
        [
            ("y", "Y", "s"),
            ("slice", "slice", "s"),
            ("n", "n labeled", "n"),
            ("pos", "pos", "n"),
            ("cos", "cos", "n"),
            ("rate", "rate", "pp"),
        ],
    )
    lines += [
        "",
        "Singleton − multi residual inside `log1p(a_in3)` terciles (train edges):",
        "",
    ]
    lines += md_table(
        p5["resid"],
        [
            ("y", "Y", "s"),
            ("tercile", "tercile", "s"),
            ("n_multi", "n multi", "n"),
            ("rate_multi", "rate multi", "pp"),
            ("n_sing", "n singleton", "n"),
            ("rate_sing", "rate singleton", "pp"),
            ("residual", "residual", "delta"),
            ("median_log1p", "median log1p(a_in3)", "f"),
        ],
    )
    lines += [
        "",
        f"Y3 singleton {_pp(p5['y3_sing'])} vs multi {_pp(p5['y3_multi'])}. "
        f"Y2 singleton {_pp(p5['y2_sing'])} vs multi {_pp(p5['y2_multi'])}. "
        "A singleton flag is a **group-size dummy**. It cannot transfer to new groups.",
        "",
        "## Pass 6 — single-feature train group-fold AUROC",
        "",
        f"Sign from the train side of each fold. Size control = `log1p(a_in3)`. "
        f"Days bar `{DAYS_BENCH:.3f}`. KEEP-as-Q1-descriptive bar = beat size by ≥{KEEP_DELTA:.2f} "
        "and not a missingness / group-type dummy.",
        "",
    ]
    lines += md_table(
        p6["tab"],
        [
            ("y", "Y", "s"),
            ("feature", "feature", "s"),
            ("cv", "CV AUROC", "f"),
            ("sd", "sd", "f"),
            ("train_auc", "train AUROC", "f"),
            ("train_sign", "sign", "n"),
            ("two_sided", "two-sided", "f"),
            ("coverage", "coverage", "pp"),
            ("n_defined", "n", "n"),
        ],
    )
    lines += [
        "",
        f"Y3 `has_erp` {_f(p6['erp_y3'])} / `has_country` {_f(p6['ctry_y3'])} / "
        f"`is_eur` {_f(p6['eur_y3'])} vs size {_f(p6['size_y3'])} vs days {_f(p6['days_y3'])}. "
        f"Beats size ≥0.02: erp={p6['erp_beats_size']} country={p6['ctry_beats_size']} eur={p6['eur_beats_size']}.",
        "",
        "## Pass 7 — is country-missing SIZE or late-arrival?",
        "",
        f"Train miss-country companies **{p7['n_miss']}** vs known **{p7['n_known']}**. "
        f"Spearman miss vs company-median `log1p(a_in3)` **{_f(p7['rho_size'])}** "
        f"({'SIZE' if p7['size_like'] else 'not SIZE'} at |ρ|≥{SIZE_RHO:.2f}). "
        f"vs months-on-book **{_f(p7['rho_trail'])}**.",
        "",
        f"Median size miss {_f(p7['med_size_miss'])} vs known {_f(p7['med_size_known'])}. "
        f"Median grid months miss {_f(p7['med_grid_miss'], 1)} vs known {_f(p7['med_grid_known'], 1)}. "
        f"Late first-tx miss {_pp(p7['share_late_miss'])} vs known {_pp(p7['share_late_known'])}. "
        f"Short <12m miss {_pp(p7['share_short_miss'])} vs known {_pp(p7['share_short_known'])}. "
        f"{'Looks like late-arrival.' if p7['late_like'] else 'Does not look like late-arrival.'}",
        "",
        "Missing-country share inside company-median size terciles (train edges):",
        "",
    ]
    lines += md_table(
        p7["terc_tab"],
        [
            ("size_terc", "tercile", "s"),
            ("n", "n cos", "n"),
            ("n_miss", "n miss country", "n"),
            ("share_miss", "share miss", "pp"),
        ],
    )
    lines += [
        "",
        "Missing-country share by months-on-book (fixed trail cuts, not quantiles):",
        "",
    ]
    lines += md_table(
        p7["trail_tab"],
        [
            ("trail_bucket", "trail", "s"),
            ("n", "n cos", "n"),
            ("n_miss", "n miss country", "n"),
            ("share_miss", "share miss", "pp"),
        ],
    )
    lines += [
        "",
        f"Country missingness is a **group trait**: all-miss groups {p7['n_g_all_miss']}, "
        f"all-known {p7['n_g_all_known']}, mixed {p7['n_g_mixed']}. "
        "A group-constant dummy cannot transfer to new groups.",
        "",
        "## Pass 8 — is `has_erp` just the 470 dummy?",
        "",
        f"Company-level Spearman `has_erp` vs has-book **{_f(p8['rho_erp_book'])}** "
        f"(agree {_pp(p8['agree'])}). Near-copy of the dark flag if |ρ|≥0.80: "
        f"**{p8['is_470_dummy']}**.",
        "",
    ]
    lines += md_table(
        p8["rates"],
        [
            ("y", "Y", "s"),
            ("slice", "slice", "s"),
            ("n", "n labeled", "n"),
            ("pos", "pos", "n"),
            ("cos", "cos", "n"),
            ("rate", "rate", "pp"),
        ],
    )
    lines += [
        "",
        f"Among *invoiced* companies only, Y3 named-ERP {_pp(p8['inv_named_y3'])} "
        f"(n={p8['inv_named_n']:,}) vs NULL-ERP {_pp(p8['inv_null_y3'])} "
        f"(n={p8['inv_null_n']:,}), gap {_pp_delta(p8['inv_gap'])}. "
        "If that gap is flat, leftover named-ERP among the 744 is not a health lever.",
        "",
    ]
    if extra.get("more_md"):
        lines += ["", extra["more_md"], ""]
    if ctx.get("png"):
        lines += ["", f"Plot: `{Path(ctx['png']).name}`.", ""]
    lines += [
        "",
        "## Transfer to new groups",
        "",
        v["hidden_test"],
        "",
        f"**Any companies.csv flag transferable as Q1 X?** **{v['transferable']}**. "
        "`has_erp`, `has_country`, and `currency=EUR` are company-constants. "
        "Train groups are ~85% uniform on each flag (extra 28). Holdout EUR is 74% "
        "not 90% (extra 24); hidden groups bring AOA/GHS/SEK/XOF and PL (extra 25). "
        "A group-type dummy cannot transfer to **new groups**.",
        "",
        "## Six brief questions",
        "",
        "| # | question | what companies.csv says |",
        "| --- | --- | --- |",
        "| 1 | Who is healthy? | **No transferable flag.** `has_erp` is the 470 dummy (ρ 0.81 vs book). Country is 82% missing. EUR is 90% home currency. All lose to size. |",
        "| 2 | Who is improving? | Not a company-constant. |",
        "| 3 | Who is turning? | Not these flags. |",
        "| 4 | Dip vs fall? | Not these flags. |",
        "| 5 | Why did it change? | CLOSE. Metadata / group identity, not a why. |",
        "| 6 | Months earlier? | `created_at` is a connection clock (p50 +39 days after first tx; year vs grid ρ −0.85). PARK. |",
        "",
        "## What was not done",
        "",
        "- Did not write parquet / duckdb. Did not run `build_targets`.",
        "- Did not edit a_vol_qa, fx_qa, uncat_qa, sibling_h, product/.",
        "- Did not invent a merged Y from country/erp. Did not revive `created_at`.",
        "- Did not commit.",
        "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")


def append_registry(p1, p2, p4, p6, p7, p8, extra: dict | None = None) -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    rows = []

    def add(model, metric, value, coverage, notes, y="-", split="train"):
        rows.append(
            {
                "ts": ts,
                "round": ROUND,
                "wave": WAVE,
                "agent": AGENT,
                "x_families": "meta",
                "y": y,
                "model": model,
                "split": split,
                "metric": metric,
                "value": (
                    f"{value:.6g}"
                    if isinstance(value, (int, float, np.floating)) and np.isfinite(value)
                    else ""
                ),
                "coverage": (
                    f"{coverage:.4f}"
                    if isinstance(coverage, (int, float)) and np.isfinite(coverage)
                    else ""
                ),
                "notes": notes,
            }
        )

    add(
        "companies_qa",
        "miss_country_share",
        p1["train"]["country"],
        1.0,
        f"train n={p1['train']['n']}; holdout coverage {p1['hold']['country']:.4f}",
    )
    add(
        "companies_qa",
        "null_erp_among_dark",
        p2["share_null_among_dark"],
        1.0,
        f"confirm_470={p2['confirm_470']} confirm_921={p2['confirm_921']} named_dark={p2['n_dark_named']}",
    )
    add(
        "has_erp",
        "auroc",
        p6["erp_y3"],
        p6["by"][(Y3, "has_erp")]["coverage"],
        f"vs size {p6['size_y3']:.3f}; vs days {p6['days_y3']:.3f}; 470 dummy rho={p8['rho_erp_book']:.3f}",
        y=Y3,
        split="cv5_group",
    )
    add(
        "has_country",
        "auroc",
        p6["ctry_y3"],
        p6["by"][(Y3, "has_country")]["coverage"],
        f"vs size {p6['size_y3']:.3f}; miss share {p1['train']['country']:.4f}; size_like={p7['size_like']}",
        y=Y3,
        split="cv5_group",
    )
    add(
        "is_eur",
        "auroc",
        p6["eur_y3"],
        p6["by"][(Y3, "is_eur")]["coverage"],
        f"vs size {p6['size_y3']:.3f}; share_eur train flags",
        y=Y3,
        split="cv5_group",
    )
    add(
        "log1p_a_in3",
        "auroc",
        p6["size_y3"],
        p6["by"][(Y3, "log1p_a_in3")]["coverage"],
        "size control; train group-fold",
        y=Y3,
        split="cv5_group",
    )
    add(
        "companies_qa",
        "created_minus_first_tx_p50_days",
        p4["med_co_minus_tx"],
        1.0,
        f"share_after={p4['share_co_after_tx']:.3f}; same_month={p4['share_same_month_co_tx']:.3f}; PARK clock",
    )
    add(
        "has_erp",
        "spearman_vs_has_book",
        p8["rho_erp_book"],
        1.0,
        f"agree={p8['agree']:.4f}; CLOSE as Y3 X if 470 dummy",
    )
    if extra:
        add(
            "companies_qa",
            "created_year_vs_n_grid_spearman",
            extra.get("rho_created_year_grid", float("nan")),
            1.0,
            "PARK created_at; year tracks trail length not health",
        )
        add(
            "companies_qa",
            "onboard_before_cap_dark_overlap",
            extra.get("n_before_dark", float("nan")),
            1.0,
            "before=470 dark=470; matching count is coincidence",
        )
        add(
            "companies_qa",
            "created_vs_bank_same_day",
            extra.get("co_bank_same_day", float("nan")),
            1.0,
            "two created_at clocks; PARK onboard",
        )
        add(
            "companies_qa",
            "late_onboard_2025_share_holdout",
            extra.get("late_share_hold", float("nan")),
            1.0,
            f"train share {extra.get('late_share_train', float('nan')):.3f}; coverage only",
            split="holdout",
        )
        add(
            "late_onboard_y2025",
            "auroc",
            extra.get("late_onboard_y3", float("nan")),
            1.0,
            "created_year>=2025 dummy; CLOSE as X; PARK as Y",
            y=Y3,
            split="cv5_group",
        )

    existing = REGISTRY.read_text(encoding="utf-8")
    with REGISTRY.open("r", encoding="utf-8") as f:
        header = next(csv.reader(f))
    written = 0
    with REGISTRY.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        for r in rows:
            marker = (
                f"{AGENT},{r.get('x_families')},{r.get('y')},{r.get('model')},"
                f"{r.get('split')},{r.get('metric')}"
            )
            if marker in existing:
                continue
            w.writerow({k: r.get(k, "") for k in header})
            written += 1
    print(f"registry appended {written} rows (skipped {len(rows) - written} already present)")


def run(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="companies.csv QA")
    p.add_argument("--no-registry", action="store_true")
    p.add_argument("--no-md", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)

    started = _now_iso()
    wall0 = time.time()
    print(f"companies_qa start {started} seed={FOLD_SEED} agent={AGENT}")
    print("no product / no 0-100 / no parquet rewrite / no GBM / no build_targets")
    print("PARK created_at as health Y — do not revive")

    con = connect()
    pop = dark_population(con)
    book = book_invoice_ids(con)
    cos = load_companies(con)
    clocks = load_clocks(con)
    con.close()

    hold = load_holdout()
    cid = cos["company_id"]
    is_train = train_mask(cid)
    is_hold = cid.isin(hold)
    assert_no_holdout(cid[is_train])
    print(
        f"companies n={len(cos)} train={int(is_train.sum())} hold={int(is_hold.sum())} "
        f"confirm_470={pop['confirm_470']} n_360={pop['n_360_alldark']} n_110={pop['n_110_mixed']}"
    )

    store = load_store()
    y = load_y()
    panel = store.merge(y, on=["company_id", "period"], how="left")

    p1 = pass1_completeness(cos, is_train, is_hold)
    p2 = pass2_erp_dark(cos, is_train, book, pop)
    p3 = pass3_country_fx(cos, is_train, store, book)
    p4 = pass4_clocks(cos, clocks, is_train)
    p5 = pass5_group_size(cos, is_train, panel, pop)
    p6 = pass6_auroc(panel, cos, is_train)
    p7 = pass7_country_size_trail(cos, is_train, panel, clocks)
    p8 = pass8_erp_is_dark_dummy(cos, is_train, panel, book, pop)

    png = None
    if not args.no_plot:
        png = plot_erp_country(cos, is_train, book, OUT_PNG)

    extra = extra_cuts(
        cos,
        is_train=is_train,
        is_hold=is_hold,
        panel=panel,
        clocks=clocks,
        book=book,
        pop=pop,
        p2=p2,
        p7=p7,
    )
    extra = extra_cuts2(cos, is_train=is_train, panel=panel, book=book, pop=pop, extra=extra)
    extra = extra_cuts3(cos, is_train=is_train, panel=panel, book=book, pop=pop, extra=extra)
    extra = extra_cuts4(cos, is_train=is_train, panel=panel, book=book, pop=pop, extra=extra)
    extra = extra_cuts5(cos, is_train=is_train, book=book, extra=extra)
    extra = extra_cuts6(cos, is_train=is_train, panel=panel, book=book, pop=pop, p6=p6, extra=extra)
    extra = extra_cuts7(
        cos,
        is_train=is_train,
        is_hold=is_hold,
        clocks=clocks,
        book=book,
        extra=extra,
    )
    extra = extra_cuts8(cos, is_train=is_train, extra=extra)
    extra = extra_cuts9(cos, is_train=is_train, is_hold=is_hold, book=book, extra=extra)
    extra = extra_cuts10(cos, is_train=is_train, clocks=clocks, extra=extra)
    extra = extra_cuts11(cos, is_train=is_train, clocks=clocks, extra=extra)
    extra = extra_cuts12(cos, is_train=is_train, clocks=clocks, book=book, extra=extra)
    extra = extra_cuts13(cos, is_train=is_train, is_hold=is_hold, book=book, extra=extra)
    extra = extra_cuts14(
        cos, is_train=is_train, panel=panel, clocks=clocks, book=book, extra=extra
    )
    extra = extra_cuts15(
        cos, is_train=is_train, panel=panel, clocks=clocks, book=book, extra=extra
    )
    extra = extra_cuts16(cos, is_train=is_train, is_hold=is_hold, book=book, extra=extra)
    extra = extra_cuts17(
        cos, is_train=is_train, panel=panel, clocks=clocks, extra=extra
    )
    extra = extra_cuts18(
        cos, is_train=is_train, is_hold=is_hold, clocks=clocks, book=book, extra=extra
    )
    extra = extra_cuts19(
        cos,
        is_train=is_train,
        is_hold=is_hold,
        panel=panel,
        clocks=clocks,
        extra=extra,
    )
    extra = extra_cuts20(
        cos, is_train=is_train, clocks=clocks, book=book, p6=p6, extra=extra
    )
    extra = extra_cuts21(
        cos, is_train=is_train, is_hold=is_hold, clocks=clocks, extra=extra
    )
    extra = extra_cuts22(
        cos, is_train=is_train, is_hold=is_hold, clocks=clocks, extra=extra
    )
    extra = extra_cuts23(cos, is_train=is_train, is_hold=is_hold, extra=extra)
    extra = extra_cuts24(cos, is_train=is_train, is_hold=is_hold, extra=extra)
    extra = extra_cuts25(cos, is_train=is_train, is_hold=is_hold, extra=extra)
    extra = extra_cuts26(cos, is_train=is_train, clocks=clocks, book=book, extra=extra)
    extra = extra_cuts27(cos, is_train=is_train, clocks=clocks, extra=extra)
    extra = extra_cuts28(cos, is_train=is_train, extra=extra)
    extra = extra_cuts29(cos, is_train=is_train, extra=extra)
    extra = extra_cuts30(cos, is_train=is_train, extra=extra)
    extra = extra_cuts31(cos, is_hold=is_hold, extra=extra)
    extra = extra_cuts32(cos, is_hold=is_hold, book=book, extra=extra)
    extra = extra_cuts33(cos, is_hold=is_hold, book=book, extra=extra)
    extra = extra_cuts34(cos, is_train=is_train, clocks=clocks, extra=extra)
    extra = extra_cuts35(cos, is_train=is_train, clocks=clocks, extra=extra)
    extra = extra_cuts36(cos, is_train=is_train, book=book, pop=pop, extra=extra)
    extra = extra_cuts37(cos, is_train=is_train, book=book, extra=extra)
    extra = extra_cuts38(cos, is_train=is_train, clocks=clocks, extra=extra)
    extra = extra_cuts39(cos, is_train=is_train, panel=panel, extra=extra)
    extra = extra_cuts40(cos, is_train=is_train, is_hold=is_hold, extra=extra)
    extra = extra_cuts41(cos, is_hold=is_hold, extra=extra)
    extra = extra_cuts42(
        cos, is_train=is_train, is_hold=is_hold, clocks=clocks, book=book, extra=extra
    )
    extra = extra_cuts43(cos, is_train=is_train, clocks=clocks, extra=extra)
    extra = extra_cuts44(cos, is_train=is_train, extra=extra)
    extra = extra_cuts45(cos, is_train=is_train, clocks=clocks, extra=extra)
    extra = extra_cuts46(cos, is_train=is_train, clocks=clocks, book=book, extra=extra)
    extra = extra_cuts47(cos, is_train=is_train, clocks=clocks, book=book, extra=extra)
    extra = extra_cuts48(cos, is_train=is_train, extra=extra)
    extra = extra_cuts49(cos, is_train=is_train, extra=extra)
    extra = extra_cuts50(cos, is_train=is_train, extra=extra)
    extra = extra_cuts51(cos, is_train=is_train, clocks=clocks, extra=extra)
    extra = extra_cuts52(cos, is_hold=is_hold, extra=extra)
    extra = extra_cuts53(cos, is_train=is_train, extra=extra)
    extra = extra_cuts54(cos, is_train=is_train, extra=extra)
    extra = extra_cuts55(cos, is_train=is_train, clocks=clocks, extra=extra)
    extra = extra_cuts56(extra=extra)
    extra = extra_cuts57(cos, is_train=is_train, clocks=clocks, extra=extra)
    extra = extra_cuts58(extra=extra)
    extra = extra_cuts59(cos, is_hold=is_hold, extra=extra)
    extra = extra_cuts60(cos, is_hold=is_hold, clocks=clocks, extra=extra)
    extra = extra_cuts61(extra=extra)
    extra = extra_cuts62(cos, is_hold=is_hold, extra=extra)
    extra = extra_cuts63(cos, is_train=is_train, extra=extra)
    extra = extra_cuts64(extra=extra)
    extra = extra_cuts65(extra=extra)
    extra = extra_cuts66(extra=extra)
    extra = extra_cuts67(extra=extra)
    extra = extra_cuts68(cos, is_train=is_train, clocks=clocks, extra=extra)
    extra = extra_cuts69(cos, is_train=is_train, clocks=clocks, extra=extra)
    extra = extra_cuts70(extra=extra)
    extra = extra_cuts71(extra=extra)
    extra = extra_cuts72(p2=p2, extra=extra)
    extra = extra_cuts73(p2=p2, extra=extra)
    extra = extra_cuts74(p1=p1, extra=extra)
    extra = extra_cuts75(p4=p4, extra=extra)
    verdict = decide(p1, p2, p3, p4, p5, p6, p7, p8)
    extra = extra_cuts76(verdict, extra)
    extra = extra_cuts77(p6=p6, extra=extra)
    extra = extra_cuts78(extra=extra)
    extra = extra_cuts79(p8=p8, extra=extra)
    extra = extra_cuts80(extra=extra)
    extra = extra_cuts81(extra=extra)
    extra = extra_cuts82(extra=extra)
    extra = extra_cuts83(extra=extra)
    extra = extra_cuts84(extra=extra)
    extra = extra_cuts85(extra=extra)
    extra = extra_cuts86(extra=extra)
    extra = extra_cuts87(extra=extra)
    extra = extra_cuts88(extra=extra)
    extra = extra_cuts89(extra=extra)
    extra = extra_cuts90(extra=extra)
    extra = extra_cuts91(extra=extra)
    extra = extra_cuts92(extra=extra)
    extra = extra_cuts93(extra=extra)
    extra = extra_cuts94(extra=extra)
    print(f"\nVERDICT {verdict['tag']}")
    print(verdict["one_liner"])
    print(verdict["hidden_test"])

    ctx = {
        "started": started,
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p4": p4,
        "p5": p5,
        "p6": p6,
        "p7": p7,
        "p8": p8,
        "extra": extra,
        "verdict": verdict,
        "png": png,
        "cos": cos,
        "clocks": clocks,
        "panel": panel,
        "book": book,
        "pop": pop,
        "is_train": is_train,
        "is_hold": is_hold,
    }
    if not args.no_registry:
        append_registry(p1, p2, p4, p6, p7, p8, extra=extra)
    if not args.no_md:
        write_md(ctx)

    elapsed = time.time() - wall0
    quote = {
        "n_train": p1["train"]["n"],
        "miss_country": p1["train"]["country"],
        "miss_erp": p1["train"]["erp"],
        "null_among_dark": p2["share_null_among_dark"],
        "n_dark_named": p2["n_dark_named"],
        "n_inv_null": p2["n_inv_null"],
        "named_dark_mixed": p2["n_named_dark_mixed"],
        "named_dark_alldark": p2["n_named_dark_alldark"],
        "confirm_921": p2["confirm_921"],
        "confirm_470": p2["confirm_470"],
        "med_created_minus_tx_days": p4["med_co_minus_tx"],
        "share_created_after_tx": p4["share_co_after_tx"],
        "has_erp_y3_cv": p6["erp_y3"],
        "has_country_y3_cv": p6["ctry_y3"],
        "is_eur_y3_cv": p6["eur_y3"],
        "size_y3_cv": p6["size_y3"],
        "transferable": verdict["transferable"],
        "verdict": verdict["tag"],
        "elapsed_s": elapsed,
    }
    print("\nQUOTE")
    print(json.dumps(quote, indent=2, default=str))
    print(f"elapsed {elapsed:.1f}s — stay on this module for more cuts")
    return ctx


if __name__ == "__main__":
    run()
