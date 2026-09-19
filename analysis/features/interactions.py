"""Family I — cross-family shapes (t-only products of assembled A–H).

Reads already-built columns from ``data/feature_store/monthly.parquet``
(or uses A–H columns already present on ``grid``). Does **not** rewrite
the parquet. Does not import other family modules. No look-ahead: every
``i_*`` is a same-period transform of features that are already causal.

Holdout companies are scored with the same formulas; nothing here fits
percentiles, bins, or other train-only refs. Train-only diagnostics live
in ``analysis/outputs/interactions_report.md`` (``python -m
analysis.features.interactions``).

Why these products (brief six questions)
----------------------------------------
Y3 SHAP + the keep list say the useful story is *pairs*, not a 0–100
score: transfer with payroll/SS regularity, DSO with debt service,
runway with concentration / late AR / dying inflow.

- i_runway_x_hhi     = b_runway * (1 - d_cust_hhi)
                       Q1/Q4: liquidity cushion × diversified customers
- i_io_x_zeroin      = a_io_ratio * c_zero_in_share_6
                       Q4: coverage while activity is fading
- i_transfer_x_ss    = a_transfer * c_ss_month
                       Q5: Y3 SHAP pair (transfer × social-security month)
- i_dso_x_dsr        = clip(e_dso_proxy, 0, 24) * f_ds_r
                       Q5: collections cycle × debt-service burden
- i_runway_x_ar30    = b_runway * e_ar_overdue_30
                       Q4: Hirshleifer — late AR with / without cushion
- i_runway_x_zeroin  = b_runway * c_zero_in_share_6
                       Q4: dying inflow with cash still on the book
- i_gap_x_supphhi    = c_gap_sd * d_supp_hhi
                       Q5: Pérez-Salazar ops vol × supplier concentration
- i_io_x_dsr         = a_io_ratio * f_ds_r
                       Q5: cash coverage × debt service (keep + SHAP)
- i_below0_x_payroll = b_below_0 * c_missed_salary
                       Q4: NSF-like month × missed payroll = fall, not dip
- i_transfer_x_salary = a_transfer * c_salary_month
                       Q5: Y3 SHAP pair (transfer × salary month)
- i_miss_e           = 1 if no invoice book this company-month
- i_miss_d           = 1 if both customer and supplier HHI are null

``e_dso_proxy`` is clipped at 24 months inside the two DSO products only
(domain bound from ``feature_report.md``; not a train percentile). Raw
A–H columns are not modified.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# allow `python analysis/features/interactions.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.features.common import ANALYSIS, DATA, connect, train_mask
from analysis.features.grid import monthly_grid

SOURCE_TABLES: list[str] = []  # assembled monthly.parquet only; no raw-table scan
FAMILY = "i"

STORE = DATA / "feature_store" / "monthly.parquet"
REPORT_PATH = ANALYSIS / "outputs" / "interactions_report.md"

DSO_CLIP = 24.0
SIZE_RHO = 0.85
CONSTANT_THRESH = 0.999
NZV_THRESH = 0.95
MIN_ACF_PAIRS = 4

NEEDED = (
    "a_io_ratio",
    "a_transfer",
    "a_in3",
    "b_runway",
    "b_below_0",
    "c_zero_in_share_6",
    "c_ss_month",
    "c_salary_month",
    "c_gap_sd",
    "c_missed_salary",
    "d_cust_hhi",
    "d_supp_hhi",
    "e_dso_proxy",
    "e_ar_overdue_30",
    "e_ar_open",
    "e_ap_open",
    "f_ds_r",
)

SPECS: tuple[dict, ...] = (
    {
        "name": "i_runway_x_hhi",
        "formula": "b_runway * (1 - d_cust_hhi)",
        "brief": "Q1/Q4 healthy vs concentrated+thin — cushion × diversified customers",
    },
    {
        "name": "i_io_x_zeroin",
        "formula": "a_io_ratio * c_zero_in_share_6",
        "brief": "Q4 dip vs fall — coverage while trailing months show zero inflow",
    },
    {
        "name": "i_transfer_x_ss",
        "formula": "a_transfer * c_ss_month",
        "brief": "Q5 why / turning — Y3 SHAP pair (a_transfer × c_ss_month)",
    },
    {
        "name": "i_dso_x_dsr",
        "formula": "clip(e_dso_proxy, 0, 24) * f_ds_r",
        "brief": "Q5 why — collections months outstanding × debt-service / inflow",
    },
    {
        "name": "i_runway_x_ar30",
        "formula": "b_runway * e_ar_overdue_30",
        "brief": "Q4 dip vs fall — Hirshleifer: late AR with / without liquidity cushion",
    },
    {
        "name": "i_runway_x_zeroin",
        "formula": "b_runway * c_zero_in_share_6",
        "brief": "Q4 dip vs fall — dying activity with cash still on the book",
    },
    {
        "name": "i_gap_x_supphhi",
        "formula": "c_gap_sd * d_supp_hhi",
        "brief": "Q5 why — Pérez-Salazar: inter-tx gap sd × supplier HHI",
    },
    {
        "name": "i_io_x_dsr",
        "formula": "a_io_ratio * f_ds_r",
        "brief": "Q5 why — cash coverage × debt-service ratio (keep + Y3 SHAP)",
    },
    {
        "name": "i_below0_x_payroll",
        "formula": "b_below_0 * c_missed_salary",
        "brief": "Q4 fall not dip — negative reconstructed cash × missed payroll",
    },
    {
        "name": "i_transfer_x_salary",
        "formula": "a_transfer * c_salary_month",
        "brief": "Q5 why / turning — Y3 SHAP pair (a_transfer × c_salary_month)",
    },
    {
        "name": "i_miss_e",
        "formula": "1 if e_ar_open and e_ap_open are both null",
        "brief": "coverage — no invoice book this company-month (~36% train)",
    },
    {
        "name": "i_miss_d",
        "formula": "1 if d_cust_hhi and d_supp_hhi are both null",
        "brief": "coverage — no customer/supplier HHI (no ERP or incomplete 6m window)",
    },
)

I_COLS = tuple(s["name"] for s in SPECS)


def _keys(grid: pd.DataFrame) -> pd.DataFrame:
    keys = grid[["company_id", "period"]].copy()
    keys["company_id"] = keys["company_id"].astype(str)
    keys["period"] = pd.to_datetime(keys["period"])
    return keys.reset_index(drop=True)


def _store_columns() -> set[str]:
    import pyarrow.parquet as pq

    return set(pq.read_schema(STORE).names)


def _load_store(columns: list[str]) -> pd.DataFrame:
    if not STORE.exists():
        raise FileNotFoundError(
            f"{STORE} missing — parent must assemble A–H before Family I can load it"
        )
    present = [c for c in columns if c in _store_columns()]
    store = pd.read_parquet(STORE, columns=["company_id", "period", *present])
    store["company_id"] = store["company_id"].astype(str)
    store["period"] = pd.to_datetime(store["period"])
    return store


def _source_panel(grid: pd.DataFrame) -> pd.DataFrame:
    """Prefer A–H already on ``grid``; otherwise left-join monthly.parquet."""
    keys = _keys(grid)
    on_grid = [c for c in NEEDED if c in grid.columns]
    if set(NEEDED) <= set(grid.columns):
        src = grid[["company_id", "period", *NEEDED]].copy()
        src["company_id"] = src["company_id"].astype(str)
        src["period"] = pd.to_datetime(src["period"])
        return keys.merge(src, on=["company_id", "period"], how="left")

    store = _load_store(list(NEEDED))
    extra = [c for c in NEEDED if c not in store.columns]
    for c in extra:
        store[c] = np.nan
    # overlay any subset already on the grid (future assembler may pass a panel)
    if on_grid:
        overlay = grid[["company_id", "period", *on_grid]].copy()
        overlay["company_id"] = overlay["company_id"].astype(str)
        overlay["period"] = pd.to_datetime(overlay["period"])
        store = store.drop(columns=on_grid, errors="ignore")
        store = store.merge(overlay, on=["company_id", "period"], how="left")
    return keys.merge(store, on=["company_id", "period"], how="left")


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _clip_dso(s: pd.Series) -> pd.Series:
    return _num(s).clip(lower=0.0, upper=DSO_CLIP)


def _compute(src: pd.DataFrame) -> pd.DataFrame:
    out = src[["company_id", "period"]].copy()
    a_io = _num(src["a_io_ratio"])
    a_xfer = _num(src["a_transfer"])
    b_run = _num(src["b_runway"])
    b_neg = _num(src["b_below_0"])
    c_z6 = _num(src["c_zero_in_share_6"])
    c_ss = _num(src["c_ss_month"])
    c_sal = _num(src["c_salary_month"])
    c_gap = _num(src["c_gap_sd"])
    c_miss = _num(src["c_missed_salary"])
    d_chhi = _num(src["d_cust_hhi"])
    d_shhi = _num(src["d_supp_hhi"])
    e_dso = _clip_dso(src["e_dso_proxy"])
    e_ar30 = _num(src["e_ar_overdue_30"])
    f_dsr = _num(src["f_ds_r"])

    out["i_runway_x_hhi"] = b_run * (1.0 - d_chhi)
    out["i_io_x_zeroin"] = a_io * c_z6
    out["i_transfer_x_ss"] = a_xfer * c_ss
    out["i_dso_x_dsr"] = e_dso * f_dsr
    out["i_runway_x_ar30"] = b_run * e_ar30
    out["i_runway_x_zeroin"] = b_run * c_z6
    out["i_gap_x_supphhi"] = c_gap * d_shhi
    out["i_io_x_dsr"] = a_io * f_dsr
    out["i_below0_x_payroll"] = b_neg * c_miss
    out["i_transfer_x_salary"] = a_xfer * c_sal
    out["i_miss_e"] = (
        src["e_ar_open"].isna() & src["e_ap_open"].isna()
    ).astype(np.float64)
    out["i_miss_d"] = (d_chhi.isna() & d_shhi.isna()).astype(np.float64)
    return out


def build(con, grid: pd.DataFrame) -> pd.DataFrame:
    """Return ``company_id, period`` plus ``i_*``. ``con`` is unused (parquet)."""
    del con  # read-only duckdb is available; Family I does not query raw tables
    if grid.empty:
        empty = grid[["company_id", "period"]].copy() if len(grid.columns) else pd.DataFrame(
            columns=["company_id", "period"]
        )
        for c in I_COLS:
            empty[c] = np.float64(np.nan)
        return empty
    src = _source_panel(grid)
    out = _compute(src)
    extra = [c for c in out.columns if c not in {"company_id", "period"}]
    bad = [c for c in extra if not c.startswith("i_")]
    if bad:
        raise ValueError(f"Family I must only emit i_* columns, got {bad}")
    if out.duplicated(["company_id", "period"]).any():
        raise ValueError("Family I produced duplicate company_id, period rows")
    return out[["company_id", "period", *I_COLS]].reset_index(drop=True)


def _median_acf1(panel: pd.DataFrame, col: str) -> float:
    acc: list[float] = []
    work = panel[["company_id", "period", col]].sort_values(["company_id", "period"])
    for _, g in work.groupby("company_id", sort=False):
        s = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
        if s.size <= 1:
            continue
        x, y = s[:-1], s[1:]
        m = np.isfinite(x) & np.isfinite(y)
        if int(m.sum()) < MIN_ACF_PAIRS:
            continue
        xx, yy = x[m], y[m]
        if xx.std() <= 1e-15 or yy.std() <= 1e-15:
            continue
        r = float(np.corrcoef(xx, yy)[0, 1])
        if np.isfinite(r):
            acc.append(r)
    return float(np.nanmedian(acc)) if acc else float("nan")


def _train_stats(part: pd.DataFrame, src: pd.DataFrame) -> pd.DataFrame:
    tr = part.loc[train_mask(part["company_id"])].copy()
    size_df = src.loc[train_mask(src["company_id"]), ["company_id", "period", "a_in3"]]
    tr = tr.merge(size_df, on=["company_id", "period"], how="left")
    size = np.log1p(np.maximum(_num(tr["a_in3"]), 0.0))
    n_cm = len(tr)
    n_co = int(tr["company_id"].nunique())
    rows = []
    for spec in SPECS:
        c = spec["name"]
        s = _num(tr[c])
        nn = s.dropna()
        cov_cm = float(s.notna().mean()) if n_cm else float("nan")
        cov_co = (
            float(tr.loc[s.notna(), "company_id"].nunique() / n_co) if n_co else float("nan")
        )
        if nn.empty:
            modal_share = float("nan")
            n_unique = 0
        else:
            vc = nn.value_counts()
            modal_share = float(vc.iloc[0] / len(nn))
            n_unique = int(nn.nunique())
        if nn.size >= 50 and n_unique > 1 and size.notna().sum() >= 50:
            size_rho = float(s.corr(size, method="spearman"))
        else:
            size_rho = float("nan")
        acf1 = _median_acf1(tr, c)
        flags = []
        if nn.empty or n_unique <= 1 or (
            np.isfinite(modal_share) and modal_share >= CONSTANT_THRESH
        ):
            flags.append("CONSTANT")
        elif np.isfinite(modal_share) and modal_share >= NZV_THRESH:
            flags.append("NZV")
        if np.isfinite(size_rho) and abs(size_rho) > SIZE_RHO:
            flags.append("SIZE")
        rows.append(
            {
                "feature": c,
                "formula": spec["formula"],
                "brief": spec["brief"],
                "cov_cm": cov_cm,
                "cov_co": cov_co,
                "n_unique": n_unique,
                "modal_share": modal_share,
                "size_rho": size_rho,
                "acf1": acf1,
                "flags": ",".join(flags) if flags else "—",
            }
        )
    return pd.DataFrame(rows)


def _fmt_pct(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{100.0 * x:.1f}%"


def _fmt_num(x: float, digits: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    if abs(x) >= 1e5 or (abs(x) > 0 and abs(x) < 1e-3):
        return f"{x:.2e}"
    return f"{x:.{digits}f}"


def write_train_report(part: pd.DataFrame, src: pd.DataFrame, path: Path = REPORT_PATH) -> pd.DataFrame:
    stats = _train_stats(part, src)
    n_train = int(train_mask(part["company_id"]).sum())
    n_co = int(part.loc[train_mask(part["company_id"]), "company_id"].nunique())
    size_hits = stats.loc[stats["flags"].str.contains("SIZE", na=False), "feature"].tolist()
    const_hits = stats.loc[stats["flags"].str.contains("CONSTANT", na=False), "feature"].tolist()

    lines = [
        "# Interactions report (train only)",
        "",
        "Holdout `analysis/splits/holdout_companies.csv` (72 companies) is **excluded** "
        "from every number below. No percentiles or bins were fit.",
        "",
        f"- Panel: **{n_train}** company-months, **{n_co}** train companies "
        "(monthly grid 2024-09 … 2026-08).",
        "- Family: **I** (`i_*`). Source: `data/feature_store/monthly.parquet` A–H columns. "
        "Parquet was **not** rewritten.",
        "- Size proxy: Spearman vs `log1p(max(a_in3, 0))`. Flag `|ρ| > 0.85` as SIZE.",
        "- Constant: modal-value share `≥ 0.999` or a single value. NZV: modal share `≥ 0.95`.",
        "- Persistence: median company-wise Pearson acf at lag 1 "
        "(≥ 4 finite pairs, non-zero s.d.).",
        "- DSO products clip `e_dso_proxy` at 24 months (domain bound; not a train fit). "
        "Raw A–H columns are untouched.",
        "",
        "## Brief map",
        "",
        "These are *shapes* that answer dip vs fall / why / lead time. They are not a 0–100 score.",
        "",
        "| feature | formula | six-question map |",
        "| --- | --- | --- |",
    ]
    for spec in SPECS:
        lines.append(f"| `{spec['name']}` | `{spec['formula']}` | {spec['brief']} |")

    lines += [
        "",
        "## Battery",
        "",
        "| feature | cov_cm | cov_co | size_ρ vs log1p(a_in3) | acf1 | modal% | n_unique | flags |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in stats.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.feature}`",
                    _fmt_pct(row.cov_cm),
                    _fmt_pct(row.cov_co),
                    _fmt_num(row.size_rho),
                    _fmt_num(row.acf1),
                    _fmt_pct(row.modal_share),
                    str(row.n_unique),
                    row.flags,
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## SIZE / CONSTANT",
        "",
        f"- SIZE (`|ρ| > 0.85` vs `log1p(a_in3)`): "
        + (", ".join(f"`{c}`" for c in size_hits) if size_hits else "none"),
        f"- CONSTANT: "
        + (", ".join(f"`{c}`" for c in const_hits) if const_hits else "none"),
        "",
        "`i_transfer_x_ss` and `i_transfer_x_salary` keep the raw euro `a_transfer` "
        "(parked as a level in the feature report, size ρ ≈ 0.02 vs log inflow). "
        "They inherit that scale; check the SIZE flag above before putting them in a GBM "
        "next to `log1p(a_in3)`.",
        "",
        "`i_below0_x_payroll` is a rare joint event (keep even if NZV): NSF-like cash "
        "and a missed payroll month together is the fall-not-dip flag.",
        "",
        "`i_miss_e` / `i_miss_d` are coverage diagnostics so a model can see "
        "'no invoice book' / 'no counterparty HHI' instead of silently imputing. "
        "`i_miss_e` never flips inside a train company (acf1 undefined): it is "
        "has-ERP vs never-ERP. `i_miss_d` does flip (early incomplete 6m window "
        "then HHI appears) and is persistent (acf1 0.87).",
        "",
        "`i_transfer_x_ss` / `i_transfer_x_salary` have acf1 ≈ 0 (same as raw "
        "`a_transfer`). Y3 lead time sits in `a_transfer_lag1`, not in these "
        "t-only products.",
        "",
        "Do not treat holdout as confirmation. Do not drop A–H columns.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return stats


def main() -> None:
    con = connect()
    grid = monthly_grid(con)[["company_id", "period"]]
    src = _source_panel(grid)
    part = build(con, grid)
    con.close()
    stats = write_train_report(part, src)
    print(f"wrote {REPORT_PATH}")
    print(stats[["feature", "cov_cm", "size_rho", "acf1", "flags"]].to_string(index=False))
    print("did not write monthly.parquet")


if __name__ == "__main__":
    main()
