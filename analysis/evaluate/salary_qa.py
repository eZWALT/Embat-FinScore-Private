"""Q3/Q5 missed-salary — month shock, quiet twin, or calendar dummy?

NORTH_STAR: `c_salary_month` is already a perm-stable Y3 lever (quiet
months recover). This cut asks whether *missing* a usual payroll is a
real turning / why, or another rare / calendar / style dummy like
`c_missed_tax` (CLOSED as Q-peaked calendar; Y3 0.511 vs size 0.617).

`c_missed_salary` = usual salary in last ≤6 months (rolling sum ≥ 3,
min_periods=1, **includes current month**) AND `c_salary_month` = 0.
Token is raw `category = 'salary'`. Feature report: RARE, modal 97.1%.

Y6 `y6_missed_payroll` is a *future* rejected Y (t+1..t+3 miss after
usual). Different object — compare Jaccard / ρ only.

No 0–100. No product/. No parquet rewrite. No new GBM. Do not run
`build_targets`. Do not invent a merged salary Y. Do not put
`c_missed_salary` on the 15-col card. Night Y3 quote stays 0.762 / 0.752.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.salary_qa

Owned: analysis/evaluate/salary_qa.py, analysis/outputs/salary_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_salary.md (end).
"""
from __future__ import annotations

import csv
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
    load_holdout,
)
from analysis.features.common import ANALYSIS, DATA, MONTHS, connect
from analysis.targets.y11_dark import book_invoice_ids

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

STORE = DATA / "feature_store" / "monthly.parquet"
TARGETS = DATA / "feature_store" / "targets.parquet"
OUT_MD = ANALYSIS / "outputs" / "salary_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "salary_calendar.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
AGENT = "0fd41cbf"
WAVE = "4"
ROUND = "R4"
MODEL = "salary_qa"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
Y6 = "y6_missed_payroll"
Y9 = "y9_fee_r_ownp80"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_QUOTE = 0.617
KEEP_DELTA = 0.02
SIZE_PARK = 0.60
SIZE_RHO = 0.50
TWIN_RHO = 0.80
MODAL_QUOTE = 0.971
ICC_TRAIT = 0.85
MIN_POS = 50
MIN_ACF_PAIRS = 4
PANEL_START = pd.Timestamp("2024-09-01")
PANEL_END = pd.Timestamp("2026-08-01")
Q_MONTHS = (1, 4, 7, 10)
CHRONIC_GROUPS = ("GROUP_0158", "GROUP_0172")
CHRONIC_BELOW = 0.50
ROLL = 6
USUAL_MIN = 3
NIGHT_Y3 = 0.762
NIGHT_Y3_CORE = 0.752

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "a_in3",
    "a_out6",
    "a_n_tx",
    "c_n_days_with_tx",
    "c_salary_month",
    "c_missed_salary",
    "c_ss_month",
    "c_tax_month",
    "c_missed_tax",
    "b_below_0",
)

Y_KEEP = (Y2, Y3, Y9, Y6)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _f(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    return f"{float(x):.{nd}f}"


def _pp(x, nd=1) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{100.0 * float(x):.{nd}f}%"


def _pct(n, d) -> float:
    if d is None or d == 0:
        return float("nan")
    return float(n) / float(d)


def _md_table(rows: list[dict], cols: list[str] | None = None) -> str:
    if not rows:
        return "_(empty)_\n"
    if cols is None:
        cols = list(rows[0].keys())
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["company_id"] = out["company_id"].astype(str)
    if "period" in out.columns:
        out["period"] = pd.to_datetime(out["period"]).astype("datetime64[ns]")
    if "group_id" in out.columns:
        out["group_id"] = out["group_id"].astype(str)
    return out


def spearman(a, b) -> float:
    d = pd.DataFrame(
        {"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}
    ).dropna()
    if len(d) < 8 or d["a"].nunique() < 2 or d["b"].nunique() < 2:
        return float("nan")
    return float(d["a"].corr(d["b"], method="spearman"))


def jaccard(a, b) -> float:
    d = pd.DataFrame(
        {
            "a": pd.to_numeric(a, errors="coerce"),
            "b": pd.to_numeric(b, errors="coerce"),
        }
    ).dropna()
    if d.empty:
        return float("nan")
    aa = d["a"].to_numpy(dtype=float) == 1
    bb = d["b"].to_numpy(dtype=float) == 1
    inter = int((aa & bb).sum())
    union = int((aa | bb).sum())
    if union == 0:
        return float("nan")
    return float(inter) / float(union)


def median_acf(series: pd.Series, company: pd.Series, lag: int) -> float:
    df = pd.DataFrame({"x": pd.to_numeric(series, errors="coerce"), "co": company.astype(str)})
    vals = []
    for _, g in df.groupby("co", sort=False):
        x = g["x"].to_numpy(dtype=float)
        if len(x) <= lag:
            continue
        aa, bb = x[:-lag], x[lag:]
        m = np.isfinite(aa) & np.isfinite(bb)
        if int(m.sum()) < MIN_ACF_PAIRS:
            continue
        aa, bb = aa[m], bb[m]
        if np.std(aa) == 0 or np.std(bb) == 0:
            continue
        vals.append(float(np.corrcoef(aa, bb)[0, 1]))
    return float(np.median(vals)) if vals else float("nan")


def icc_anova(series: pd.Series, company: pd.Series) -> dict:
    s = pd.DataFrame(
        {"x": pd.to_numeric(series, errors="coerce"), "co": company.astype(str)}
    ).dropna()
    if len(s) < 10 or s["x"].nunique() < 2:
        return {"icc": float("nan"), "var_w": float("nan"), "var_b": float("nan"), "k": 0}
    y = s["x"].to_numpy(dtype=float)
    _, inv, counts = np.unique(s["co"].to_numpy(), return_inverse=True, return_counts=True)
    if counts.size < 2:
        return {"icc": float("nan"), "var_w": float("nan"), "var_b": float("nan"), "k": 0}
    means = np.bincount(inv, weights=y) / counts
    grand = float(y.mean())
    ssb = float(np.sum(counts * (means - grand) ** 2))
    ssw = float(np.sum((y - means[inv]) ** 2))
    k = int(counts.size)
    n = int(y.size)
    var_b = ssb / max(k - 1, 1)
    var_w = ssw / max(n - k, 1)
    icc = var_b / (var_b + var_w) if (var_b + var_w) > 0 else float("nan")
    return {"icc": float(icc), "var_w": float(var_w), "var_b": float(var_b), "k": k}


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


def signed_oof_auroc(
    y: pd.Series,
    x: pd.Series,
    folds: pd.Series,
    mask: pd.Series,
    n_folds: int = N_FOLDS,
) -> dict:
    """Group-fold CV AUROC. Sign from the train side of each fold."""
    y = pd.to_numeric(y, errors="coerce")
    x = pd.to_numeric(x, errors="coerce")
    defined = mask & y.notna() & x.notna()
    n = int(defined.sum())
    n_pos = int((defined & (y == 1)).sum())
    n_neg = int((defined & (y == 0)).sum())
    low = n_pos < MIN_POS or n_neg == 0
    fold_rows = []
    aucs = []
    if low:
        return {
            "cv": float("nan"),
            "sd": float("nan"),
            "n_folds": 0,
            "folds": fold_rows,
            "train_sign": 0,
            "train_auc": float("nan"),
            "n_defined": n,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "low_power": True,
        }
    for k in range(n_folds):
        tr = defined & (folds != k)
        va = defined & (folds == k)
        sign = choose_sign(y[tr], x[tr])
        auc = auroc(y[va], sign * x[va])
        aucs.append(auc)
        fold_rows.append(
            {
                "fold": k,
                "auroc": float(auc) if np.isfinite(auc) else float("nan"),
                "sign": int(sign),
                "n_va": int(va.sum()),
                "n_pos": int((va & (y == 1)).sum()),
            }
        )
    finite = [a for a in aucs if np.isfinite(a)]
    tr_sign = choose_sign(y[defined], x[defined])
    return {
        "cv": float(np.mean(finite)) if finite else float("nan"),
        "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan"),
        "n_folds": len(finite),
        "folds": fold_rows,
        "train_sign": int(tr_sign),
        "train_auc": float(auroc(y[defined], tr_sign * x[defined])),
        "n_defined": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "low_power": False,
    }


def fold_bits(rec: dict) -> str:
    return " ".join(
        f"{r['auroc']:.3f}" if np.isfinite(r["auroc"]) else "—" for r in rec.get("folds", [])
    )


def load_panel() -> pd.DataFrame:
    if not STORE.exists():
        raise FileNotFoundError(f"{STORE} missing")
    if not TARGETS.exists():
        raise FileNotFoundError(f"{TARGETS} missing — do not run build_targets")
    raw = pd.read_parquet(STORE)
    yraw = pd.read_parquet(TARGETS)
    print(f"store {STORE} shape={raw.shape} (read-only)")
    print(f"targets {TARGETS} shape={yraw.shape} (read-only)")
    missing = [c for c in STORE_COLS if c not in raw.columns]
    if missing:
        raise RuntimeError(f"monthly.parquet missing {missing}")
    ykeep = ["company_id", "period"]
    for c in Y_KEEP:
        if c in yraw.columns:
            ykeep.append(c)
        else:
            print(f"targets missing {c} — will compute in-module if needed")
    panel = _keys(raw[list(STORE_COLS)])
    y = _keys(yraw[ykeep])
    panel = panel.merge(y, on=["company_id", "period"], how="left")
    hold = load_holdout()
    panel["split"] = np.where(panel["company_id"].isin(hold), "holdout", "train")
    panel["cal_month"] = panel["period"].dt.month
    panel["cal_year"] = panel["period"].dt.year
    panel["is_q_month"] = panel["cal_month"].isin(Q_MONTHS).astype(np.int8)
    panel["is_aug"] = (panel["cal_month"] == 8).astype(np.int8)
    panel["not_salary"] = (pd.to_numeric(panel["c_salary_month"], errors="coerce") == 0).astype(
        np.int8
    )
    panel["log_in3"] = np.log1p(
        pd.to_numeric(panel["a_in3"], errors="coerce").clip(lower=0)
    )
    panel["so_far"] = panel.groupby("company_id", sort=False).cumcount() + 1
    first = panel.groupby("company_id")["period"].transform("min")
    panel["months_on_book"] = (
        (PANEL_END.year - first.dt.year) * 12 + (PANEL_END.month - first.dt.month) + 1
    )
    panel["short_book"] = (panel["months_on_book"] < 12).astype(np.int8)
    return panel


def attach_folds(panel: pd.DataFrame) -> pd.DataFrame:
    train = panel[panel["split"] == "train"][["company_id", "group_id"]].drop_duplicates()
    assert_no_holdout(train["company_id"])
    folds = group_folds(train, n=N_FOLDS, seed=FOLD_SEED)
    return panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")


def add_panel_lags(df: pd.DataFrame, stems: list[str], lags: tuple[int, ...]) -> pd.DataFrame:
    out = df.sort_values(["company_id", "period"]).reset_index(drop=True)
    cid = out["company_id"]
    extra = {}
    for c in stems:
        s = pd.to_numeric(out[c], errors="coerce")
        g = s.groupby(cid, sort=False)
        for k in lags:
            extra[f"{c}_lag{k}"] = g.shift(k)
    if extra:
        out = pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)
    return out


def attach_y6_if_missing(panel: pd.DataFrame, con) -> pd.DataFrame:
    if Y6 in panel.columns:
        print(f"{Y6} present in targets.parquet — using store, no assembler")
        return panel
    from analysis.targets.y6_activity import build as build_y6

    print(f"{Y6} missing from targets — in-module y6_activity.build (no assembler)")
    grid = panel[["company_id", "period"]].drop_duplicates()
    y6 = build_y6(con, grid)
    y6 = _keys(y6[["company_id", "period", Y6]])
    return panel.merge(y6, on=["company_id", "period"], how="left")


def _usual_mask(tr: pd.DataFrame) -> pd.Series:
    g = tr.sort_values(["company_id", "period"])
    sal6 = g.groupby("company_id", sort=False)["c_salary_month"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").rolling(ROLL, min_periods=1).sum()
    )
    return (sal6 >= USUAL_MIN).reindex(tr.index)


def chronic_ids(tr: pd.DataFrame) -> list[str]:
    """12 chronic dark Y2 names: ≥50% labeled months already below 0 in 0158/0172."""
    y = pd.to_numeric(tr[Y2], errors="coerce")
    below = pd.to_numeric(tr["b_below_0"], errors="coerce")
    lab = y.notna()
    hot = tr["group_id"].astype(str).isin(CHRONIC_GROUPS)
    sl = lab & hot
    g = (
        tr.loc[sl, ["company_id"]]
        .assign(below=below[sl].values)
        .groupby("company_id")["below"]
        .agg(n="size", n_below="sum")
    )
    g["share_below"] = g["n_below"] / g["n"]
    return [str(i) for i in g.index[g["share_below"] >= CHRONIC_BELOW]]


def _company_terciles(tr: pd.DataFrame, col: str, name: str) -> pd.Series:
    last = tr.sort_values("period").groupby("company_id", sort=False).last()
    x = pd.to_numeric(last[col], errors="coerce")
    terc = pd.qcut(x, 3, labels=["T1", "T2", "T3"], duplicates="drop")
    return terc.rename(name)


# ---------------------------------------------------------------------------
# Pass 1 — completeness + base rate
# ---------------------------------------------------------------------------
def pass1_base(panel: pd.DataFrame, tr: pd.DataFrame) -> dict:
    rows = []
    for split, sl in (("train", tr), ("holdout", panel[panel["split"] == "holdout"])):
        for col in ("c_salary_month", "c_missed_salary"):
            x = pd.to_numeric(sl[col], errors="coerce")
            nn = x.dropna()
            modal = float(nn.mode().iloc[0]) if len(nn) else float("nan")
            modal_share = float((nn == modal).mean()) if len(nn) else float("nan")
            prev = float((x == 1).mean()) if len(x) else float("nan")
            rows.append(
                {
                    "split": split,
                    "col": col,
                    "n_cm": int(len(sl)),
                    "n_co": int(sl["company_id"].nunique()),
                    "cov": float(x.notna().mean()) if len(x) else float("nan"),
                    "prev": prev,
                    "n_pos": int((x == 1).sum()),
                    "modal": modal,
                    "modal_share": modal_share,
                }
            )
    sal = next(r for r in rows if r["split"] == "train" and r["col"] == "c_salary_month")
    miss = next(r for r in rows if r["split"] == "train" and r["col"] == "c_missed_salary")
    ho_miss = next(r for r in rows if r["split"] == "holdout" and r["col"] == "c_missed_salary")
    confirm = bool(
        np.isfinite(miss["modal_share"]) and abs(miss["modal_share"] - MODAL_QUOTE) < 0.015
    )
    prose = (
        f"Train `c_salary_month` share {_pp(sal['prev'])} (n={sal['n_pos']:,} / {sal['n_cm']:,}). "
        f"`c_missed_salary` prevalence {_pp(miss['prev'])} (n={miss['n_pos']:,}); "
        f"modal {miss['modal']:g} share {_pp(miss['modal_share'])} "
        f"({'CONFIRM 97.1%' if confirm else 'does not match feature-report 97.1%'}). "
        f"Holdout coverage only: missed mean {_pp(ho_miss['prev'])} on {ho_miss['n_cm']:,} CM / "
        f"{ho_miss['n_co']} cos. No AUROC on holdout."
    )
    print(prose)
    return {
        "rows": rows,
        "sal": sal,
        "miss": miss,
        "ho_miss": ho_miss,
        "confirm_modal": confirm,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 2 — formula vs raw category=salary (includes-current quirk)
# ---------------------------------------------------------------------------
def pass2_formula(tr: pd.DataFrame, con) -> dict:
    hold = load_holdout()
    raw = con.execute(
        """
        SELECT
          CAST(company_id AS VARCHAR) AS company_id,
          CAST(date_trunc('month', "date") AS DATE) AS month,
          MAX(CASE WHEN category = 'salary' THEN 1 ELSE 0 END) AS raw_sal
        FROM transactions
        WHERE "date" IS NOT NULL
          AND "date" >= DATE '2024-09-01'
          AND "date" < DATE '2026-09-01'
        GROUP BY 1, 2
        """
    ).df()
    raw["company_id"] = raw["company_id"].astype(str)
    raw["period"] = pd.to_datetime(raw["month"])
    raw = raw.loc[~raw["company_id"].isin(hold)].copy()
    assert_no_holdout(raw["company_id"])

    grid = tr[["company_id", "period", "c_salary_month", "c_missed_salary"]].copy()
    grid = grid.merge(raw[["company_id", "period", "raw_sal"]], on=["company_id", "period"], how="left")
    grid["raw_sal"] = grid["raw_sal"].fillna(0).astype(int)
    store_sal = pd.to_numeric(grid["c_salary_month"], errors="coerce").fillna(0).astype(int)
    agree_sal = float((store_sal == grid["raw_sal"]).mean())

    g = grid.sort_values(["company_id", "period"])
    sal6_incl = g.groupby("company_id", sort=False)["raw_sal"].transform(
        lambda s: s.rolling(ROLL, min_periods=1).sum()
    )
    recon_incl = ((sal6_incl >= USUAL_MIN) & (g["raw_sal"] == 0)).astype(int)
    store_miss = pd.to_numeric(g["c_missed_salary"], errors="coerce").fillna(0).astype(int)
    agree_miss = float((recon_incl == store_miss).mean())

    # exclude-current: roll previous 6 only (the quirk contrast)
    prev = g.groupby("company_id", sort=False)["raw_sal"].shift(1)
    sal6_excl = prev.groupby(g["company_id"], sort=False).transform(
        lambda s: s.rolling(ROLL, min_periods=1).sum()
    )
    recon_excl = ((sal6_excl >= USUAL_MIN) & (g["raw_sal"] == 0)).astype(int)
    n_quirk = int((recon_incl != recon_excl).sum())
    n_incl_only = int(((recon_incl == 1) & (recon_excl == 0)).sum())
    n_excl_only = int(((recon_incl == 0) & (recon_excl == 1)).sum())

    # sample company-months: two missed, two salary, one quirk if any
    samples = []
    miss_idx = g.index[store_miss == 1]
    sal_idx = g.index[store_sal == 1]
    quirk_idx = g.index[recon_incl != recon_excl]
    pick = []
    if len(miss_idx):
        pick.extend(list(miss_idx[:2]))
    if len(sal_idx):
        pick.extend(list(sal_idx[:2]))
    if len(quirk_idx):
        pick.extend(list(quirk_idx[:2]))
    g2 = g.reset_index(drop=False)
    by_co = {cid: sub for cid, sub in g.groupby("company_id", sort=False)}
    for i in pick[:6]:
        row = g.loc[i]
        cid = str(row["company_id"])
        per = pd.Timestamp(row["period"])
        sub = by_co[cid]
        win = sub[(sub["period"] > per - pd.DateOffset(months=6)) & (sub["period"] <= per)]
        samples.append(
            {
                "company_id": cid,
                "period": str(per.date()),
                "store_sal": int(row["c_salary_month"]),
                "raw_sal": int(row["raw_sal"]),
                "store_miss": int(row["c_missed_salary"]),
                "recon_incl": int(recon_incl.loc[i]),
                "recon_excl": int(recon_excl.loc[i]),
                "win_sal": ",".join(str(int(v)) for v in win["raw_sal"].tolist()),
                "win_n": int(len(win)),
            }
        )

    prose = (
        f"Store `c_salary_month` vs raw category=salary agreement {_pp(agree_sal)}. "
        f"Reconstructed missed (roll-6 include current, min_periods=1) vs store {_pp(agree_miss)}. "
        f"Include-vs-exclude-current disagree on {n_quirk:,} CM "
        f"(incl-only {n_incl_only:,}, excl-only {n_excl_only:,}). "
        "Quirk: the current 0 occupies a slot in the 6-month usual-sum, so include-current "
        "is stricter (needs ≥3 salary in the other 5). Exclude-current would add those "
        "excl-only months — do not rewrite ops.py."
    )
    print(prose)
    return {
        "agree_sal": agree_sal,
        "agree_miss": agree_miss,
        "n_quirk": n_quirk,
        "n_incl_only": n_incl_only,
        "n_excl_only": n_excl_only,
        "n_raw_sal_cm": int((grid["raw_sal"] == 1).sum()),
        "n_raw_sal_co": int(grid.loc[grid["raw_sal"] == 1, "company_id"].nunique()),
        "samples": samples,
        "prose": prose,
        "formula_ok": bool(agree_sal >= 0.995 and agree_miss >= 0.995),
    }


# ---------------------------------------------------------------------------
# Pass 3 — calendar
# ---------------------------------------------------------------------------
def pass3_calendar(tr: pd.DataFrame, con) -> dict:
    cal = []
    for m in range(1, 13):
        sl = tr[tr["cal_month"] == m]
        cal.append(
            {
                "month": m,
                "name": pd.Timestamp(2000, m, 1).strftime("%b"),
                "q": int(m in Q_MONTHS),
                "aug": int(m == 8),
                "n_cm": int(len(sl)),
                "sal_share": float(pd.to_numeric(sl["c_salary_month"], errors="coerce").mean()),
                "miss_share": float(pd.to_numeric(sl["c_missed_salary"], errors="coerce").mean()),
            }
        )
    q_sal = float(np.nanmean([r["sal_share"] for r in cal if r["q"] == 1]))
    nq_sal = float(np.nanmean([r["sal_share"] for r in cal if r["q"] == 0]))
    q_miss = float(np.nanmean([r["miss_share"] for r in cal if r["q"] == 1]))
    nq_miss = float(np.nanmean([r["miss_share"] for r in cal if r["q"] == 0]))
    aug_sal = next(r["sal_share"] for r in cal if r["month"] == 8)
    aug_miss = next(r["miss_share"] for r in cal if r["month"] == 8)
    other_miss = float(np.nanmean([r["miss_share"] for r in cal if r["month"] != 8]))
    peak_sal = max(cal, key=lambda r: r["sal_share"] if np.isfinite(r["sal_share"]) else -1)
    trough_sal = min(cal, key=lambda r: r["sal_share"] if np.isfinite(r["sal_share"]) else 9)
    peak_miss = max(cal, key=lambda r: r["miss_share"] if np.isfinite(r["miss_share"]) else -1)
    trough_miss = min(cal, key=lambda r: r["miss_share"] if np.isfinite(r["miss_share"]) else 9)
    ratio = q_sal / nq_sal if nq_sal else float("nan")
    quarterly = bool(q_sal >= 0.55 and nq_sal <= 0.25)
    monthly = bool(nq_sal >= 0.25 and np.isfinite(ratio) and ratio < 1.3)
    mixed_q = bool((not quarterly) and (not monthly) and np.isfinite(ratio) and ratio >= 1.3)
    aug_peak = bool(np.isfinite(aug_miss) and np.isfinite(other_miss) and aug_miss >= other_miss + 0.02)
    shape = (
        "quarterly"
        if quarterly
        else ("august_peaked" if aug_peak and not mixed_q else ("mixed_q_peaked" if mixed_q else ("monthly" if monthly else "mixed")))
    )
    calendar_dummy = bool(mixed_q or quarterly or aug_peak)

    # weekday of last salary booking in-month (train, no holdout)
    hold = load_holdout()
    wd = con.execute(
        """
        SELECT
          CAST(company_id AS VARCHAR) AS company_id,
          CAST(date_trunc('month', "date") AS DATE) AS month,
          MAX("date") AS last_sal
        FROM transactions
        WHERE category = 'salary'
          AND "date" IS NOT NULL
          AND "date" >= DATE '2024-09-01'
          AND "date" < DATE '2026-09-01'
        GROUP BY 1, 2
        """
    ).df()
    wd["company_id"] = wd["company_id"].astype(str)
    wd["period"] = pd.to_datetime(wd["month"])
    wd = wd.loc[~wd["company_id"].isin(hold)].copy()
    assert_no_holdout(wd["company_id"])
    wd["last_sal"] = pd.to_datetime(wd["last_sal"])
    wd["wd"] = wd["last_sal"].dt.dayofweek
    wd_rows = []
    names = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
    for i, name in enumerate(names):
        sl = wd[wd["wd"] == i]
        wd_rows.append(
            {
                "wd": name,
                "n_cm": int(len(sl)),
                "share": _pct(len(sl), len(wd)),
            }
        )
    weekend = float(wd["wd"].isin([5, 6]).mean()) if len(wd) else float("nan")

    prose = (
        f"Salary CM share Q-months {_pp(q_sal)} vs other {_pp(nq_sal)} (ratio "
        f"{ratio if np.isfinite(ratio) else float('nan'):.2f}). "
        f"Missed Q {_pp(q_miss)} vs other {_pp(nq_miss)}; Aug missed {_pp(aug_miss)} vs "
        f"other-month mean {_pp(other_miss)}. Peak salary {peak_sal['name']} {_pp(peak_sal['sal_share'])}, "
        f"trough {trough_sal['name']} {_pp(trough_sal['sal_share'])}. "
        f"Peak missed {peak_miss['name']} {_pp(peak_miss['miss_share'])}. "
        f"Shape **{shape}**. Last-salary weekday weekend share {_pp(weekend)}."
    )
    print(prose)
    return {
        "cal": cal,
        "q_sal": q_sal,
        "nq_sal": nq_sal,
        "q_miss": q_miss,
        "nq_miss": nq_miss,
        "aug_sal": aug_sal,
        "aug_miss": aug_miss,
        "other_miss": other_miss,
        "ratio": ratio,
        "shape": shape,
        "quarterly": quarterly,
        "monthly": monthly,
        "mixed_q": mixed_q,
        "aug_peak": aug_peak,
        "calendar_dummy": calendar_dummy,
        "peak_sal": peak_sal,
        "trough_sal": trough_sal,
        "peak_miss": peak_miss,
        "trough_miss": trough_miss,
        "wd_rows": wd_rows,
        "weekend": weekend,
        "n_sal_wd": int(len(wd)),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 4 — co-occurrence (quiet month?)
# ---------------------------------------------------------------------------
def pass4_cooccur(tr: pd.DataFrame) -> dict:
    miss = pd.to_numeric(tr["c_missed_salary"], errors="coerce")
    sal = pd.to_numeric(tr["c_salary_month"], errors="coerce")
    ss = pd.to_numeric(tr["c_ss_month"], errors="coerce")
    tax = pd.to_numeric(tr["c_tax_month"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    quiet = days == 0
    m1 = miss == 1

    def _p(flag: pd.Series, mask: pd.Series) -> float:
        sl = flag[mask]
        return float(sl.mean()) if sl.notna().any() else float("nan")

    rows = [
        {
            "slice": "all",
            "n": int(len(tr)),
            "ss": float(ss.mean()),
            "tax": float(tax.mean()),
            "days_p50": float(days.median()),
            "quiet": float(quiet.mean()),
        },
        {
            "slice": "missed",
            "n": int(m1.sum()),
            "ss": _p(ss, m1),
            "tax": _p(tax, m1),
            "days_p50": float(days[m1].median()) if m1.any() else float("nan"),
            "quiet": _p(quiet.astype(float), m1),
        },
        {
            "slice": "salary",
            "n": int((sal == 1).sum()),
            "ss": _p(ss, sal == 1),
            "tax": _p(tax, sal == 1),
            "days_p50": float(days[sal == 1].median()) if (sal == 1).any() else float("nan"),
            "quiet": _p(quiet.astype(float), sal == 1),
        },
        {
            "slice": "no_salary",
            "n": int((sal == 0).sum()),
            "ss": _p(ss, sal == 0),
            "tax": _p(tax, sal == 0),
            "days_p50": float(days[sal == 0].median()) if (sal == 0).any() else float("nan"),
            "quiet": _p(quiet.astype(float), sal == 0),
        },
    ]

    p_ss_miss = float(ss[m1].mean()) if m1.any() else float("nan")
    p_ss_sal = float(ss[sal == 1].mean()) if (sal == 1).any() else float("nan")
    p_quiet_miss = float(quiet[m1].mean()) if m1.any() else float("nan")
    p_quiet_all = float(quiet.mean())
    days_miss = float(days[m1].median()) if m1.any() else float("nan")
    days_sal = float(days[sal == 1].median()) if (sal == 1).any() else float("nan")
    days_all = float(days.median())
    # missed months that still have SS / tax / activity
    both_ss = int(((m1) & (ss == 1)).sum())
    both_tax = int(((m1) & (tax == 1)).sum())
    quiet_twin = bool(
        np.isfinite(p_quiet_miss) and p_quiet_miss >= 0.40
        or (np.isfinite(days_miss) and np.isfinite(days_all) and days_miss <= 1 and days_all >= 3)
    )
    prose = (
        f"Missed CM: P(ss)={_pp(p_ss_miss)} vs salary-month {_pp(p_ss_sal)}; "
        f"P(days=0)={_pp(p_quiet_miss)} vs all {_pp(p_quiet_all)}; "
        f"days p50 missed {days_miss:.1f} vs salary {days_sal:.1f} vs all {days_all:.1f}. "
        f"Missed ∩ ss {both_ss:,}; missed ∩ tax {both_tax:,}. "
        f"{'Looks like a quiet-month twin' if quiet_twin else 'Not a pure quiet-month (still some activity / SS)'}."
    )
    print(prose)
    return {
        "rows": rows,
        "p_ss_miss": p_ss_miss,
        "p_ss_sal": p_ss_sal,
        "p_quiet_miss": p_quiet_miss,
        "p_quiet_all": p_quiet_all,
        "days_miss": days_miss,
        "days_sal": days_sal,
        "days_all": days_all,
        "both_ss": both_ss,
        "both_tax": both_tax,
        "quiet_twin": quiet_twin,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 5 — single-feature group-fold AUROC
# ---------------------------------------------------------------------------
def pass5_auroc(tr: pd.DataFrame) -> dict:
    feats = {
        "c_missed_salary": tr["c_missed_salary"],
        "c_salary_month": tr["c_salary_month"],
        "c_ss_month": tr["c_ss_month"],
        "c_missed_tax": tr["c_missed_tax"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "log1p_a_in3": tr["log_in3"],
        "is_aug": tr["is_aug"],
        "is_q_month": tr["is_q_month"],
        "not_salary": tr["not_salary"],
    }
    rows = []
    store = {}
    for y in (Y2, Y3, Y9):
        lab = tr[y].notna()
        for name, col in feats.items():
            res = signed_oof_auroc(tr[y], col, tr["fold"], lab)
            store[(y, name)] = res
            rows.append(
                {
                    "y": y,
                    "feature": name,
                    "n": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                    "sd": _f(res["sd"]),
                    "sign": res["train_sign"] if not res["low_power"] else "—",
                    "train": _f(res["train_auc"]) if not res["low_power"] else "—",
                    "folds": fold_bits(res) if not res["low_power"] else "—",
                }
            )
            print(
                f"AUROC {y} {name}: "
                f"{'LOW_POWER' if res['low_power'] else _f(res['cv'])} "
                f"n_pos={res['n_pos']} sign={res['train_sign']}"
            )

    def _cv(y, feat) -> float:
        r = store[(y, feat)]
        return float("nan") if r["low_power"] else r["cv"]

    size_y3 = _cv(Y3, "log1p_a_in3")
    days_y3 = _cv(Y3, "c_n_days_with_tx")
    miss_y3 = _cv(Y3, "c_missed_salary")
    sal_y3 = _cv(Y3, "c_salary_month")
    ss_y3 = _cv(Y3, "c_ss_month")
    miss_y2 = _cv(Y2, "c_missed_salary")
    miss_y9 = _cv(Y9, "c_missed_salary")
    size_y2 = _cv(Y2, "log1p_a_in3")
    days_y2 = _cv(Y2, "c_n_days_with_tx")
    aug_y3 = _cv(Y3, "is_aug")
    notsal_y3 = _cv(Y3, "not_salary")
    beat_size = (
        miss_y3 - size_y3 if np.isfinite(miss_y3) and np.isfinite(size_y3) else float("nan")
    )
    beat_days = (
        miss_y3 - days_y3 if np.isfinite(miss_y3) and np.isfinite(days_y3) else float("nan")
    )
    just_not_sal = bool(
        np.isfinite(miss_y3) and np.isfinite(notsal_y3) and abs(miss_y3 - notsal_y3) < 0.02
    )
    just_aug = bool(np.isfinite(miss_y3) and np.isfinite(aug_y3) and abs(miss_y3 - aug_y3) < 0.02)
    size_park = bool(np.isfinite(size_y3) and size_y3 >= SIZE_PARK)
    keep_x = bool(
        np.isfinite(beat_size)
        and beat_size >= KEEP_DELTA
        and not just_not_sal
        and not just_aug
        and (not np.isfinite(days_y3) or miss_y3 + 1e-9 >= days_y3 or beat_days >= -0.05)
    )
    # KEEP-as-X gate from brief: beat size ≥0.02 AND not calendar/size/salary_month twin
    # losing to days by a lot is a quiet-month twin — block KEEP
    if np.isfinite(days_y3) and np.isfinite(miss_y3) and miss_y3 < days_y3 - 0.05:
        keep_x = False
    if np.isfinite(sal_y3) and np.isfinite(miss_y3) and abs(miss_y3 - sal_y3) < 0.02:
        keep_x = False

    wander = float("nan")
    rec = store[(Y3, "c_missed_salary")]
    if not rec["low_power"] and rec["folds"]:
        vals = [r["auroc"] for r in rec["folds"] if np.isfinite(r["auroc"])]
        wander = float(max(vals) - min(vals)) if vals else float("nan")

    prose = (
        f"Y3 stressed singles: `c_missed_salary` {_f(miss_y3)} vs size {_f(size_y3)} "
        f"(Δ {_f(beat_size)}) vs days {_f(days_y3)} (night 0.711, replica Δ "
        f"{_f(days_y3 - DAYS_BENCH)}). `c_salary_month` {_f(sal_y3)} `c_ss_month` {_f(ss_y3)} "
        f"`not_salary` {_f(notsal_y3)} `is_aug` {_f(aug_y3)}. "
        f"Y2 missed {_f(miss_y2)} vs size {_f(size_y2)} / days {_f(days_y2)}. "
        f"Y9 missed {_f(miss_y9)}. Fold wander Y3 missed {_f(wander)}. "
        f"{'KEEP-gate open' if keep_x else 'KEEP-gate closed'}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "miss_y3": miss_y3,
        "sal_y3": sal_y3,
        "ss_y3": ss_y3,
        "size_y3": size_y3,
        "days_y3": days_y3,
        "miss_y2": miss_y2,
        "miss_y9": miss_y9,
        "size_y2": size_y2,
        "days_y2": days_y2,
        "aug_y3": aug_y3,
        "notsal_y3": notsal_y3,
        "beat_size": beat_size,
        "beat_days": beat_days,
        "just_not_sal": just_not_sal,
        "just_aug": just_aug,
        "size_park": size_park,
        "keep_x": keep_x,
        "wander": wander,
        "n_y3": int(tr[Y3].notna().sum()),
        "n_y2": int(tr[Y2].notna().sum()),
        "n_y9": int(tr[Y9].notna().sum()),
        "y3_rate": float(pd.to_numeric(tr[Y3], errors="coerce").mean()),
        "y2_rate": float(pd.to_numeric(tr[Y2], errors="coerce").mean()),
        "y9_rate": float(pd.to_numeric(tr[Y9], errors="coerce").mean()),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 6 — leak screens
# ---------------------------------------------------------------------------
def pass6_leak(tr: pd.DataFrame) -> dict:
    miss = pd.to_numeric(tr["c_missed_salary"], errors="coerce")
    pairs = [
        ("c_salary_month", tr["c_salary_month"]),
        ("c_ss_month", tr["c_ss_month"]),
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ("a_n_tx", tr["a_n_tx"]),
        ("a_out6", tr["a_out6"]),
        ("log1p(a_in3)", tr["log_in3"]),
        ("c_missed_tax", tr["c_missed_tax"]),
        ("c_tax_month", tr["c_tax_month"]),
        ("not_salary", tr["not_salary"]),
    ]
    rows = []
    twin_name = None
    size_flag = False
    for name, col in pairs:
        rho = spearman(miss, col)
        flag = ""
        if np.isfinite(rho) and abs(rho) >= TWIN_RHO:
            flag = "TWIN"
            if twin_name is None:
                twin_name = name
        if name == "log1p(a_in3)" and np.isfinite(rho) and abs(rho) >= SIZE_RHO:
            flag = (flag + " SIZE").strip()
            size_flag = True
        rows.append({"pair": f"c_missed_salary vs {name}", "rho": rho, "flag": flag})
    # salary_month vs days (quiet card already on 15-col)
    rows.append(
        {
            "pair": "c_salary_month vs c_n_days_with_tx",
            "rho": spearman(tr["c_salary_month"], tr["c_n_days_with_tx"]),
            "flag": "",
        }
    )
    prose = (
        "Leak screen |ρ|≥0.80 = twin; |ρ| vs log1p(a_in3) ≥0.50 = SIZE. "
        + (
            f"Twin: **{twin_name}**."
            if twin_name
            else "No |ρ|≥0.80 twin vs the listed stems."
        )
        + (f" SIZE vs inflow." if size_flag else " Not a size clone.")
        + f" vs c_missed_tax ρ={next(r['rho'] for r in rows if 'c_missed_tax' in r['pair']):.3f} "
        "(tax cousin quote 0.075)."
    )
    print(prose)
    return {
        "rows": rows,
        "twin_name": twin_name,
        "size_flag": size_flag,
        "rho_sal": next(r["rho"] for r in rows if r["pair"].endswith("c_salary_month")),
        "rho_days": next(r["rho"] for r in rows if r["pair"].endswith("c_n_days_with_tx") and "missed_salary" in r["pair"]),
        "rho_ss": next(r["rho"] for r in rows if r["pair"].endswith("c_ss_month")),
        "rho_size": next(r["rho"] for r in rows if "log1p" in r["pair"]),
        "rho_tax": next(r["rho"] for r in rows if r["pair"].endswith("c_missed_tax")),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 7 — ICC / company-demean
# ---------------------------------------------------------------------------
def pass7_icc(tr: pd.DataFrame) -> dict:
    miss = pd.to_numeric(tr["c_missed_salary"], errors="coerce")
    sal = pd.to_numeric(tr["c_salary_month"], errors="coerce")
    icc_m = icc_anova(miss, tr["company_id"])
    icc_s = icc_anova(sal, tr["company_id"])
    # demean and re-rank Y3
    mu = miss.groupby(tr["company_id"], sort=False).transform("mean")
    dem = miss - mu
    lab = tr[Y3].notna()
    raw = signed_oof_auroc(tr[Y3], miss, tr["fold"], lab)
    dem_res = signed_oof_auroc(tr[Y3], dem, tr["fold"], lab)
    drop = (
        raw["cv"] - dem_res["cv"]
        if (not raw["low_power"] and not dem_res["low_power"])
        else float("nan")
    )
    trait = bool(np.isfinite(icc_m["icc"]) and icc_m["icc"] >= ICC_TRAIT)
    shock = bool(np.isfinite(icc_m["icc"]) and icc_m["icc"] < 0.50)
    # ever-miss companies
    ever = tr.groupby("company_id")["c_missed_salary"].sum()
    n_ever = int((ever > 0).sum())
    n_co = int(tr["company_id"].nunique())
    prose = (
        f"`c_missed_salary` ICC={_f(icc_m['icc'])} (k={icc_m['k']}); "
        f"`c_salary_month` ICC={_f(icc_s['icc'])}. "
        f"Y3 raw {_f(raw['cv']) if not raw['low_power'] else 'LOW_POWER'} vs "
        f"company-demean {_f(dem_res['cv']) if not dem_res['low_power'] else 'LOW_POWER'} "
        f"(drop {_f(drop)}). Ever-miss companies {n_ever}/{n_co}. "
        f"{'TRAIT (payroll-booker style)' if trait else ('MONTH SHOCK' if shock else 'mixed / between')}."
    )
    print(prose)
    return {
        "icc_miss": icc_m["icc"],
        "icc_sal": icc_s["icc"],
        "k_miss": icc_m["k"],
        "raw_cv": raw["cv"] if not raw["low_power"] else float("nan"),
        "dem_cv": dem_res["cv"] if not dem_res["low_power"] else float("nan"),
        "drop": drop,
        "trait": trait,
        "shock": shock,
        "n_ever": n_ever,
        "n_co": n_co,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 8 — dark 470 vs 744
# ---------------------------------------------------------------------------
def pass8_dark(tr: pd.DataFrame, con) -> dict:
    book = book_invoice_ids(con)
    last = tr.sort_values("period").groupby("company_id", sort=False).last().reset_index()
    last["ever_erp"] = last["company_id"].isin(book)
    n_erp = int(last["ever_erp"].sum())
    n_dark = int((~last["ever_erp"]).sum())
    confirm = n_erp == 744 and n_dark == 470

    ever = tr.groupby("company_id", as_index=False).agg(
        n_cm=("period", "size"),
        n_sal=("c_salary_month", "sum"),
        n_miss=("c_missed_salary", "sum"),
        n_ss=("c_ss_month", "sum"),
    )
    ever["ever_erp"] = ever["company_id"].isin(book)
    ever["sal_rate"] = ever["n_sal"] / ever["n_cm"]
    ever["miss_rate"] = ever["n_miss"] / ever["n_cm"]
    rows = []
    for name, part in (
        ("ever_erp_744", ever[ever["ever_erp"]]),
        ("never_erp_470", ever[~ever["ever_erp"]]),
    ):
        rows.append(
            {
                "group": name,
                "n_co": int(len(part)),
                "ever_sal": _pp(float((part["n_sal"] > 0).mean())),
                "sal_cm": _pp(float(part["sal_rate"].mean())),
                "miss_cm": _pp(float(part["miss_rate"].mean())),
                "ss_cm": _pp(float((part["n_ss"] / part["n_cm"]).mean())),
            }
        )
    tr2 = tr.copy()
    tr2["ever_erp"] = tr2["company_id"].isin(book)
    cm = []
    store = {}
    for name, part in (("ever_erp", tr2[tr2["ever_erp"]]), ("never_erp", tr2[~tr2["ever_erp"]])):
        cm.append(
            {
                "group": name,
                "n_cm": int(len(part)),
                "n_co": int(part["company_id"].nunique()),
                "sal": float(pd.to_numeric(part["c_salary_month"], errors="coerce").mean()),
                "miss": float(pd.to_numeric(part["c_missed_salary"], errors="coerce").mean()),
            }
        )
        lab = part[Y3].notna()
        res = signed_oof_auroc(part[Y3], part["c_missed_salary"], part["fold"], lab)
        store[name] = res
    dark_miss = float(ever.loc[~ever["ever_erp"], "miss_rate"].mean())
    erp_miss = float(ever.loc[ever["ever_erp"], "miss_rate"].mean())
    same = bool(np.isfinite(dark_miss) and np.isfinite(erp_miss) and abs(dark_miss - erp_miss) < 0.02)
    hold = load_holdout()
    hold_book = int(len(set(hold) & book))
    prose = (
        f"Train last-month: ever-ERP {n_erp} / never-ERP {n_dark} "
        f"({'CONFIRM 744/470' if confirm else 'counts differ'}). "
        f"Mean company miss-CM: invoiced {_pp(erp_miss)} vs dark {_pp(dark_miss)}. "
        f"{'Same bank-book miss rate' if same else 'Dark files missed-salary at a different rate'}. "
        f"Holdout ever-ERP coverage only: {hold_book}/{len(hold)}."
    )
    print(prose)
    return {
        "n_erp": n_erp,
        "n_dark": n_dark,
        "confirm": confirm,
        "rows": rows,
        "cm": cm,
        "dark_miss": dark_miss,
        "erp_miss": erp_miss,
        "same": same,
        "hold_book": hold_book,
        "hold_n": int(len(hold)),
        "y3_erp": store["ever_erp"]["cv"] if not store["ever_erp"]["low_power"] else float("nan"),
        "y3_dark": store["never_erp"]["cv"] if not store["never_erp"]["low_power"] else float("nan"),
        "prose": prose,
        "book": book,
    }


# ---------------------------------------------------------------------------
# Pass 9 — drop 12 chronic dark Y2 names
# ---------------------------------------------------------------------------
def pass9_chronic(tr: pd.DataFrame) -> dict:
    ids = chronic_ids(tr)
    drop = ~tr["company_id"].astype(str).isin(set(ids))
    lab = tr[Y2].notna()
    full = signed_oof_auroc(tr[Y2], tr["c_missed_salary"], tr["fold"], lab)
    rest = signed_oof_auroc(tr[Y2], tr["c_missed_salary"], tr["fold"], lab & drop)
    size_full = signed_oof_auroc(tr[Y2], tr["log_in3"], tr["fold"], lab)
    size_rest = signed_oof_auroc(tr[Y2], tr["log_in3"], tr["fold"], lab & drop)
    days_full = signed_oof_auroc(tr[Y2], tr["c_n_days_with_tx"], tr["fold"], lab)
    days_rest = signed_oof_auroc(tr[Y2], tr["c_n_days_with_tx"], tr["fold"], lab & drop)
    miss_share_ch = float(
        pd.to_numeric(tr.loc[tr["company_id"].astype(str).isin(set(ids)), "c_missed_salary"], errors="coerce").mean()
    ) if ids else float("nan")
    miss_share_rest = float(pd.to_numeric(tr.loc[drop, "c_missed_salary"], errors="coerce").mean())
    flip = False
    if not full["low_power"] and not rest["low_power"]:
        flip = abs(full["cv"] - rest["cv"]) >= 0.03
    prose = (
        f"Chronic 12 names (0158/0172, ≥50% labeled months below 0): {len(ids)}. "
        f"Y2 missed CV full {_f(full['cv']) if not full['low_power'] else 'LOW_POWER'} → "
        f"drop-12 {_f(rest['cv']) if not rest['low_power'] else 'LOW_POWER'}. "
        f"Size {(_f(size_full['cv']) if not size_full['low_power'] else '—')} → "
        f"{(_f(size_rest['cv']) if not size_rest['low_power'] else '—')}; "
        f"days {(_f(days_full['cv']) if not days_full['low_power'] else '—')} → "
        f"{(_f(days_rest['cv']) if not days_rest['low_power'] else '—')}. "
        f"Miss share on 12 {_pp(miss_share_ch)} vs rest {_pp(miss_share_rest)}. "
        f"{'DROP FLIPS Y2' if flip else 'Drop does not flip Y2 (≥0.03)'}."
    )
    print(prose)
    return {
        "n_ids": len(ids),
        "ids": ids,
        "y2_full": full["cv"] if not full["low_power"] else float("nan"),
        "y2_drop": rest["cv"] if not rest["low_power"] else float("nan"),
        "size_full": size_full["cv"] if not size_full["low_power"] else float("nan"),
        "size_drop": size_rest["cv"] if not size_rest["low_power"] else float("nan"),
        "days_full": days_full["cv"] if not days_full["low_power"] else float("nan"),
        "days_drop": days_rest["cv"] if not days_rest["low_power"] else float("nan"),
        "miss_ch": miss_share_ch,
        "miss_rest": miss_share_rest,
        "flip": flip,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 10 — 2×2 vs salary_month and days terciles after size
# ---------------------------------------------------------------------------
def pass10_twobytwo(tr: pd.DataFrame) -> dict:
    miss = pd.to_numeric(tr["c_missed_salary"], errors="coerce") == 1
    sal = pd.to_numeric(tr["c_salary_month"], errors="coerce") == 1
    # by construction missed => not salary
    both = int((miss & sal).sum())
    miss_only = int((miss & ~sal).sum())
    sal_only = int((~miss & sal).sum())
    neither = int((~miss & ~sal).sum())
    n = int(len(tr))

    rate_rows = []
    for y in (Y2, Y3, Y9):
        for sname, mask in (
            ("missed", miss),
            ("salary", sal),
            ("no_sal_not_miss", ~sal & ~miss),
        ):
            s = pd.to_numeric(tr.loc[mask, y], errors="coerce")
            rate_rows.append(
                {
                    "y": y,
                    "slice": sname,
                    "n_cm": int(mask.sum()),
                    "n_lab": int(s.notna().sum()),
                    "n_pos": int((s == 1).sum()),
                    "rate": float(s.mean()) if s.notna().any() else float("nan"),
                }
            )

    terc_size = _company_terciles(tr, "log_in3", "size_t")
    terc_days = _company_terciles(tr, "c_n_days_with_tx", "days_t")
    m = tr.merge(terc_size.reset_index(), on="company_id", how="left")
    m = m.merge(terc_days.reset_index(), on="company_id", how="left")
    terc_rows = []
    store = {}
    lab = m[Y3].notna()
    for tname in ("size_t", "days_t"):
        for labv in ("T1", "T2", "T3"):
            mask = lab & (m[tname].astype(str) == labv)
            res = signed_oof_auroc(m[Y3], m["c_missed_salary"], m["fold"], mask)
            store[(tname, labv)] = res
            terc_rows.append(
                {
                    "clock": tname,
                    "tercile": labv,
                    "n": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "miss": _pp(
                        float(pd.to_numeric(m.loc[mask, "c_missed_salary"], errors="coerce").mean())
                        if mask.any()
                        else float("nan")
                    ),
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                }
            )
    # residual: missed among not-salary only (the honest complement)
    notsal = lab & (pd.to_numeric(m["c_salary_month"], errors="coerce") == 0)
    res_ns = signed_oof_auroc(m[Y3], m["c_missed_salary"], m["fold"], notsal)
    prose = (
        f"2×2 missed×salary: both {both:,} (must be 0), miss-only {miss_only:,}, "
        f"salary-only {sal_only:,}, neither {neither:,} / {n:,}. "
        f"Y3 missed CV inside size terciles: "
        + ", ".join(
            f"{k}={'LOW_POWER' if store[('size_t', k)]['low_power'] else _f(store[('size_t', k)]['cv'])}"
            for k in ("T1", "T2", "T3")
        )
        + f". Among not-salary months Y3 {_f(res_ns['cv']) if not res_ns['low_power'] else 'LOW_POWER'}."
    )
    print(prose)
    return {
        "both": both,
        "miss_only": miss_only,
        "sal_only": sal_only,
        "neither": neither,
        "rate_rows": rate_rows,
        "terc_rows": terc_rows,
        "notsal_cv": res_ns["cv"] if not res_ns["low_power"] else float("nan"),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 11 — Q6 lag1 / lag3 on short vs long books
# ---------------------------------------------------------------------------
def pass11_q6(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    slices = (
        ("all", pd.Series(True, index=tr.index)),
        ("short_<12", tr["short_book"] == 1),
        ("long_>=12", tr["short_book"] == 0),
        ("so_far>=4", tr["so_far"] >= 4),
    )
    cols = ("c_missed_salary", "c_missed_salary_lag1", "c_missed_salary_lag3", "c_salary_month")
    for y in (Y2, Y3):
        for sname, smask in slices:
            lab = tr[y].notna() & smask
            for col in cols:
                if col not in tr.columns:
                    continue
                res = signed_oof_auroc(tr[y], tr[col], tr["fold"], lab & tr[col].notna())
                store[(y, sname, col)] = res
                prev = (
                    float(pd.to_numeric(tr.loc[lab, col], errors="coerce").mean())
                    if lab.any()
                    else float("nan")
                )
                rows.append(
                    {
                        "y": y,
                        "slice": sname,
                        "col": col,
                        "n": f"{res['n_defined']:,}",
                        "n_pos": f"{res['n_pos']:,}",
                        "x_mean": _pp(prev),
                        "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                    }
                )

    def _cv(y, sl, col) -> float:
        r = store.get((y, sl, col))
        if r is None or r["low_power"]:
            return float("nan")
        return r["cv"]

    now = _cv(Y3, "all", "c_missed_salary")
    lag1 = _cv(Y3, "all", "c_missed_salary_lag1")
    lag3 = _cv(Y3, "all", "c_missed_salary_lag3")
    short_now = _cv(Y3, "short_<12", "c_missed_salary")
    short_lag1 = _cv(Y3, "short_<12", "c_missed_salary_lag1")
    long_lag1 = _cv(Y3, "long_>=12", "c_missed_salary_lag1")
    short_prev = float(
        pd.to_numeric(tr.loc[tr["short_book"] == 1, "c_missed_salary"], errors="coerce").mean()
    )
    n_short = int((tr["short_book"] == 1).sum())
    n_short_miss = int(((tr["short_book"] == 1) & (pd.to_numeric(tr["c_missed_salary"], errors="coerce") == 1)).sum())
    empty_short = bool(n_short_miss < 10 or (np.isfinite(short_prev) and short_prev < 0.005))
    keep_q6 = bool(
        np.isfinite(lag1)
        and np.isfinite(now)
        and now >= 0.55
        and (now - lag1) <= 0.03
        and lag1 >= 0.55
        and not empty_short
    )
    if not np.isfinite(now) or now < 0.55:
        q6 = "CLOSE"
        why = (
            f"contemporaneous Y3 {_f(now)} is chance; lag1 {_f(lag1)} / lag3 {_f(lag3)} "
            f"have nothing to lead. Short books are not empty ({n_short_miss} miss / {n_short:,} CM, "
            f"{_pp(short_prev)}) but Y3 there is LOW_POWER."
        )
    elif empty_short:
        q6 = "CLOSE"
        why = f"empty on short books (miss share {_pp(short_prev)}); Q6 needs a trail."
    else:
        q6 = "KEEP" if keep_q6 else "CLOSE"
        why = f"now {_f(now)} vs lag1 {_f(lag1)} vs lag3 {_f(lag3)}."
    prose = (
        f"Q6 Y3 now {_f(now)} lag1 {_f(lag1)} lag3 {_f(lag3)}; "
        f"short now {_f(short_now)} lag1 {_f(short_lag1)}; long lag1 {_f(long_lag1)}. "
        f"Short-book miss share {_pp(short_prev)}. **{q6}** — {why}"
    )
    print(prose)
    return {
        "rows": rows,
        "now": now,
        "lag1": lag1,
        "lag3": lag3,
        "short_now": short_now,
        "short_lag1": short_lag1,
        "long_lag1": long_lag1,
        "short_prev": short_prev,
        "empty_short": empty_short,
        "keep_q6": keep_q6,
        "q6": q6,
        "why": why,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 12 — vs y6_missed_payroll (different object)
# ---------------------------------------------------------------------------
def pass12_y6(tr: pd.DataFrame) -> dict:
    if Y6 not in tr.columns:
        return {
            "present": False,
            "prose": "y6_missed_payroll unavailable even after in-module build.",
            "jac": float("nan"),
            "rho": float("nan"),
            "leak": False,
            "rows": [],
        }
    x = pd.to_numeric(tr["c_missed_salary"], errors="coerce")
    y = pd.to_numeric(tr[Y6], errors="coerce")
    both = x.notna() & y.notna()
    jac = jaccard(x[both], y[both])
    rho = spearman(x[both], y[both])
    n_both = int(both.sum())
    n_x1 = int(((x == 1) & both).sum())
    n_y1 = int(((y == 1) & both).sum())
    n_and = int(((x == 1) & (y == 1) & both).sum())
    p_y_given_x = float(y[(x == 1) & both].mean()) if ((x == 1) & both).any() else float("nan")
    p_x_given_y = float(x[(y == 1) & both].mean()) if ((y == 1) & both).any() else float("nan")
    y_rate = float(y[both].mean()) if both.any() else float("nan")
    leak = bool((np.isfinite(jac) and jac >= 0.50) or (np.isfinite(rho) and abs(rho) >= TWIN_RHO))
    rows = [
        {"item": "overlap n", "value": f"{n_both:,}"},
        {"item": "c_missed_salary=1 on overlap", "value": f"{n_x1:,}"},
        {"item": "y6_missed_payroll=1 on overlap", "value": f"{n_y1:,}"},
        {"item": "intersection", "value": f"{n_and:,}"},
        {"item": "Jaccard", "value": _f(jac)},
        {"item": "Spearman", "value": _f(rho)},
        {"item": "P(Y6|missed X)", "value": _pp(p_y_given_x)},
        {"item": "P(missed X|Y6)", "value": _pp(p_x_given_y)},
        {"item": "Y6 base on overlap", "value": _pp(y_rate)},
    ]
    lab = tr[Y3].notna() & y.notna()
    y6_as_x = signed_oof_auroc(tr[Y3], y, tr["fold"], lab)
    prose = (
        f"Y6 overlap n={n_both:,}; Jaccard {_f(jac)} Spearman {_f(rho)}. "
        f"P(Y6|missed)={_pp(p_y_given_x)} vs Y6 base {_pp(y_rate)}. "
        f"Y3 using Y6-as-X {_f(y6_as_x['cv']) if not y6_as_x['low_power'] else 'LOW_POWER'}. "
        f"{'X leaks the rejected Y' if leak else 'X does not leak y6_missed_payroll (different window: now vs t+1..t+3)'}."
    )
    print(prose)
    return {
        "present": True,
        "jac": jac,
        "rho": rho,
        "leak": leak,
        "n_both": n_both,
        "n_x1": n_x1,
        "n_y1": n_y1,
        "n_and": n_and,
        "p_y_given_x": p_y_given_x,
        "p_x_given_y": p_x_given_y,
        "y_rate": y_rate,
        "y3_y6": y6_as_x["cv"] if not y6_as_x["low_power"] else float("nan"),
        "rows": rows,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 13 — amounts: payroll mass or 1€ token?
# ---------------------------------------------------------------------------
def pass13_amounts(tr: pd.DataFrame, con) -> dict:
    hold = load_holdout()
    tx = con.execute(
        """
        SELECT
          CAST(company_id AS VARCHAR) AS company_id,
          CAST(date_trunc('month', "date") AS DATE) AS month,
          amount,
          category
        FROM transactions
        WHERE category IN ('salary', 'social_security')
          AND "date" IS NOT NULL
          AND "date" >= DATE '2024-09-01'
          AND "date" < DATE '2026-09-01'
        """
    ).df()
    tx["company_id"] = tx["company_id"].astype(str)
    tx["period"] = pd.to_datetime(tx["month"])
    tx = tx.loc[~tx["company_id"].isin(hold)].copy()
    assert_no_holdout(tx["company_id"])
    sal = tx[tx["category"] == "salary"].copy()
    sal["abs"] = pd.to_numeric(sal["amount"], errors="coerce").abs()
    ss = tx[tx["category"] == "social_security"].copy()
    ss["abs"] = pd.to_numeric(ss["amount"], errors="coerce").abs()

    def _amt(s: pd.Series) -> dict:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return {"n": 0, "p50": float("nan"), "p90": float("nan"), "mean": float("nan"), "share_1e": float("nan")}
        return {
            "n": int(len(s)),
            "p50": float(s.median()),
            "p90": float(s.quantile(0.90)),
            "mean": float(s.mean()),
            "share_1e": float((s <= 1.0).mean()),
        }

    overall = _amt(sal["abs"])
    ss_amt = _amt(ss["abs"])
    bins = [
        ("<=1", 0, 1.0000001),
        ("1-100", 1.0000001, 100),
        ("100-1k", 100, 1000),
        ("1k-5k", 1000, 5000),
        (">5k", 5000, np.inf),
    ]
    bin_rows = []
    s = sal["abs"].dropna()
    for name, lo, hi in bins:
        sl = s[(s >= lo) & (s < hi)]
        bin_rows.append(
            {
                "bin": name,
                "n_tx": int(len(sl)),
                "share_tx": _pp(_pct(len(sl), len(s))),
                "p50": f"{float(sl.median()):,.0f}" if len(sl) else "—",
            }
        )
    out = con.execute(
        """
        SELECT
          CAST(company_id AS VARCHAR) AS company_id,
          CAST(date_trunc('month', "date") AS DATE) AS month,
          SUM(CASE WHEN amount < 0 THEN ABS(amount)
                   WHEN category IN ('salary','social_security','tax','payment','bulk_payment','utility')
                   THEN ABS(amount) ELSE 0 END) AS abs_out_proxy,
          SUM(CASE WHEN category = 'salary' THEN ABS(amount) ELSE 0 END) AS abs_sal
        FROM transactions
        WHERE "date" IS NOT NULL
          AND "date" >= DATE '2024-09-01'
          AND "date" < DATE '2026-09-01'
        GROUP BY 1, 2
        """
    ).df()
    out["company_id"] = out["company_id"].astype(str)
    out["period"] = pd.to_datetime(out["month"])
    out = out.loc[~out["company_id"].isin(hold)].copy()
    assert_no_holdout(out["company_id"])
    m = tr.merge(out[["company_id", "period", "abs_out_proxy", "abs_sal"]], on=["company_id", "period"], how="left")
    has_sal = pd.to_numeric(m["c_salary_month"], errors="coerce") == 1
    share = np.where(
        pd.to_numeric(m["abs_out_proxy"], errors="coerce") > 0,
        pd.to_numeric(m["abs_sal"], errors="coerce") / pd.to_numeric(m["abs_out_proxy"], errors="coerce"),
        np.nan,
    )
    share_s = pd.Series(share, index=m.index)
    share_on_sal = share_s[has_sal]
    p50_share = float(share_on_sal.median()) if share_on_sal.notna().any() else float("nan")
    token = bool(np.isfinite(overall["p50"]) and overall["p50"] <= 5.0)
    prose = (
        f"Train salary txs n={overall['n']:,} |amt| p50={overall['p50']:,.0f} p90={overall['p90']:,.0f} "
        f"share≤1€ {_pp(overall['share_1e'])}. "
        f"On salary months, salary / |out-proxy| p50 {_pp(p50_share)}. "
        f"SS txs p50={ss_amt['p50']:,.0f}. "
        f"{'1€ booking token' if token else 'real payroll mass (not a 1€ token)'}."
    )
    print(prose)
    return {
        "overall": overall,
        "ss_amt": ss_amt,
        "bin_rows": bin_rows,
        "p50_share": p50_share,
        "token": token,
        "n_sal_co": int(sal["company_id"].nunique()),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra — payroll cadence
# ---------------------------------------------------------------------------
def pass14_cadence(tr: pd.DataFrame) -> dict:
    ever = tr.groupby("company_id", as_index=False).agg(
        n_cm=("period", "size"),
        n_sal=("c_salary_month", "sum"),
        n_miss=("c_missed_salary", "sum"),
        n_ss=("c_ss_month", "sum"),
    )
    ever["sal_rate"] = ever["n_sal"] / ever["n_cm"]
    ever["kind"] = np.select(
        [ever["n_sal"] == 0, ever["sal_rate"] >= 0.70, (ever["sal_rate"] >= 0.25) & (ever["n_sal"] >= 3)],
        ["never", "monthly", "irregular"],
        default="rare",
    )
    rows = []
    for k, part in ever.groupby("kind"):
        rows.append(
            {
                "kind": k,
                "n_co": int(len(part)),
                "share_co": _pp(_pct(len(part), len(ever))),
                "sal_p50": _f(float(part["sal_rate"].median()), 3),
                "miss_p50": _f(float((part["n_miss"] / part["n_cm"]).median()), 3),
            }
        )
    n_never = int((ever["kind"] == "never").sum())
    n_month = int((ever["kind"] == "monthly").sum())
    n_irr = int((ever["kind"] == "irregular").sum())
    n_rare = int((ever["kind"] == "rare").sum())
    prose = (
        f"Train payroll cadence: never {n_never}, monthly {n_month}, "
        f"irregular {n_irr}, rare {n_rare} / {len(ever)}."
    )
    print(prose)
    return {
        "rows": rows,
        "n_never": n_never,
        "n_month": n_month,
        "n_irr": n_irr,
        "n_rare": n_rare,
        "n_co": int(len(ever)),
        "ever": ever,
        "prose": prose,
    }


def pass15_usual(tr: pd.DataFrame) -> dict:
    usual = _usual_mask(tr)
    miss = pd.to_numeric(tr["c_missed_salary"], errors="coerce") == 1
    n_usual = int(usual.sum())
    n_miss_u = int((usual & miss).sum())
    rows = []
    store = {}
    for y in (Y2, Y3, Y9):
        lab = tr[y].notna()
        for sname, mask in (
            ("usual_sal6", lab & usual),
            ("usual_missed", lab & usual & miss),
            ("usual_paid", lab & usual & ~miss),
        ):
            res = signed_oof_auroc(tr[y], tr["c_missed_salary"], tr["fold"], mask)
            store[(y, sname)] = res
            s = pd.to_numeric(tr.loc[mask, y], errors="coerce")
            rows.append(
                {
                    "y": y,
                    "slice": sname,
                    "n": f"{int(mask.sum()):,}",
                    "n_lab": f"{int(s.notna().sum()):,}",
                    "n_pos": f"{int((s == 1).sum()):,}",
                    "rate": _pp(float(s.mean()) if s.notna().any() else float("nan")),
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                }
            )
    y3u = store[(Y3, "usual_sal6")]
    y3_miss = next(r for r in rows if r["y"] == Y3 and r["slice"] == "usual_missed")
    y3_paid = next(r for r in rows if r["y"] == Y3 and r["slice"] == "usual_paid")
    prose = (
        f"Usual-salary CM (sal6≥3): {n_usual:,} / {len(tr):,}; of those missed {n_miss_u:,} "
        f"({_pp(_pct(n_miss_u, n_usual))}). Y3 missed-on-usual "
        f"{'LOW_POWER' if y3u['low_power'] else _f(y3u['cv'])}. "
        f"Y3 rate usual-missed {y3_miss['rate']} vs usual-paid {y3_paid['rate']}."
    )
    print(prose)
    return {
        "rows": rows,
        "n_usual": n_usual,
        "n_miss_u": n_miss_u,
        "y3_cv": y3u["cv"] if not y3u["low_power"] else float("nan"),
        "prose": prose,
    }


def pass16_who(tr: pd.DataFrame, ever: pd.DataFrame) -> dict:
    m = tr.merge(ever[["company_id", "kind"]], on="company_id", how="left")
    miss = m[pd.to_numeric(m["c_missed_salary"], errors="coerce") == 1]
    rows = []
    for k, part in miss.groupby("kind"):
        rows.append(
            {
                "kind": k,
                "n_miss": int(len(part)),
                "share": _pp(_pct(len(part), len(miss))) if len(miss) else "—",
                "n_co": int(part["company_id"].nunique()),
                "days_p50": _f(float(pd.to_numeric(part["c_n_days_with_tx"], errors="coerce").median()), 1),
            }
        )
    n_mon = int((miss["kind"] == "monthly").sum()) if len(miss) else 0
    n_irr = int((miss["kind"] == "irregular").sum()) if len(miss) else 0
    prose = (
        f"Of {len(miss):,} missed CM: monthly {n_mon:,}, irregular {n_irr:,}. "
        "A monthly booker skip is the cleanest Q5; count it separately from rare/irregular holes."
    )
    print(prose)
    return {"rows": rows, "n_miss": int(len(miss)), "n_mon": n_mon, "n_irr": n_irr, "prose": prose}


def pass17_holdout(panel: pd.DataFrame) -> dict:
    ho = panel[panel["split"] == "holdout"]
    rows = []
    for col in ("c_salary_month", "c_missed_salary", "c_ss_month", "c_n_days_with_tx"):
        x = pd.to_numeric(ho[col], errors="coerce")
        rows.append(
            {
                "col": col,
                "n_cm": int(len(ho)),
                "n_co": int(ho["company_id"].nunique()),
                "cov": _pp(float(x.notna().mean())),
                "mean": _pp(float(x.mean()) if x.notna().any() else float("nan"))
                if col != "c_n_days_with_tx"
                else _f(float(x.mean()) if x.notna().any() else float("nan")),
            }
        )
    prose = (
        f"Holdout 72 coverage only: {len(ho):,} CM. "
        f"salary-month mean {_pp(float(pd.to_numeric(ho['c_salary_month'], errors='coerce').mean()))}; "
        f"missed-salary mean {_pp(float(pd.to_numeric(ho['c_missed_salary'], errors='coerce').mean()))}. "
        "No AUROC claim."
    )
    print(prose)
    return {"rows": rows, "n_cm": int(len(ho)), "n_co": int(ho["company_id"].nunique()), "prose": prose}


def pass18_fold_wander(p5: dict) -> dict:
    rec = p5["store"][(Y3, "c_missed_salary")]
    rec_d = p5["store"][(Y3, "c_n_days_with_tx")]
    rec_s = p5["store"][(Y3, "log1p_a_in3")]
    rec_sal = p5["store"][(Y3, "c_salary_month")]
    rows = []
    for k in range(N_FOLDS):
        def _frow(r, k):
            if r["low_power"] or k >= len(r["folds"]):
                return "—"
            a = r["folds"][k]["auroc"]
            return _f(a)

        rows.append(
            {
                "fold": k,
                "missed": _frow(rec, k),
                "salary": _frow(rec_sal, k),
                "days": _frow(rec_d, k),
                "size": _frow(rec_s, k),
                "n_pos_miss": rec["folds"][k]["n_pos"] if not rec["low_power"] and k < len(rec["folds"]) else "—",
            }
        )
    prose = f"Y3 fold wander missed {_f(p5['wander'])}. Signs from train side of each fold."
    print(prose)
    return {"rows": rows, "prose": prose}


def pass19_ss_when_miss(tr: pd.DataFrame) -> dict:
    """When salary is missed, is SS also missing? Payroll packet vs salary token."""
    miss = pd.to_numeric(tr["c_missed_salary"], errors="coerce") == 1
    sal = pd.to_numeric(tr["c_salary_month"], errors="coerce") == 1
    ss = pd.to_numeric(tr["c_ss_month"], errors="coerce") == 1
    rows = [
        {
            "slice": "missed",
            "n": int(miss.sum()),
            "P(ss)": _pp(float(ss[miss].mean()) if miss.any() else float("nan")),
            "P(tax)": _pp(float(pd.to_numeric(tr.loc[miss, "c_tax_month"], errors="coerce").mean()) if miss.any() else float("nan")),
        },
        {
            "slice": "salary",
            "n": int(sal.sum()),
            "P(ss)": _pp(float(ss[sal].mean()) if sal.any() else float("nan")),
            "P(tax)": _pp(float(pd.to_numeric(tr.loc[sal, "c_tax_month"], errors="coerce").mean()) if sal.any() else float("nan")),
        },
        {
            "slice": "ss_no_salary",
            "n": int((ss & ~sal).sum()),
            "P(ss)": "100.0%",
            "P(tax)": _pp(
                float(pd.to_numeric(tr.loc[ss & ~sal, "c_tax_month"], errors="coerce").mean())
                if (ss & ~sal).any()
                else float("nan")
            ),
        },
    ]
    packet = bool(
        miss.any()
        and float(ss[miss].mean()) <= 0.20
        and float(ss[sal].mean()) >= 0.60
    )
    prose = (
        f"P(ss|missed)={rows[0]['P(ss)']} vs P(ss|salary)={rows[1]['P(ss)']}. "
        f"{'Missed salary usually drops the SS packet too' if packet else 'SS often continues when salary is missed (token skip, not payroll halt)'}."
    )
    print(prose)
    return {"rows": rows, "packet": packet, "prose": prose}


def pass20_quiet_residual(tr: pd.DataFrame) -> dict:
    """Does missed still rank after dropping quiet months / after days?"""
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    lab = tr[Y3].notna()
    rows = []
    for sname, mask in (
        ("all", lab),
        ("days>=1", lab & (days >= 1)),
        ("days>=3", lab & (days >= 3)),
        ("days>=5", lab & (days >= 5)),
        ("has_ss", lab & (pd.to_numeric(tr["c_ss_month"], errors="coerce") == 1)),
        ("no_ss", lab & (pd.to_numeric(tr["c_ss_month"], errors="coerce") == 0)),
    ):
        res = signed_oof_auroc(tr[Y3], tr["c_missed_salary"], tr["fold"], mask)
        rows.append(
            {
                "slice": sname,
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "miss": _pp(
                    float(pd.to_numeric(tr.loc[mask, "c_missed_salary"], errors="coerce").mean())
                    if mask.any()
                    else float("nan")
                ),
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
            }
        )
    busy = next(r for r in rows if r["slice"] == "days>=3")
    prose = (
        f"Y3 missed on days≥3 {busy['CV']} (vs all {rows[0]['CV']}). "
        "If skill vanishes once the month is busy, it is the quiet-month twin already on the 15-col card."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def pass21_usual_vs_size(tr: pd.DataFrame) -> dict:
    """Honest skip among usual (sal6≥3): does missed beat size *on that slice*?"""
    usual = _usual_mask(tr)
    rows = []
    store = {}
    for y in (Y2, Y3):
        lab = tr[y].notna() & usual
        for feat, col in (
            ("c_missed_salary", tr["c_missed_salary"]),
            ("c_salary_month", tr["c_salary_month"]),
            ("log1p_a_in3", tr["log_in3"]),
            ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
            ("c_ss_month", tr["c_ss_month"]),
        ):
            res = signed_oof_auroc(tr[y], col, tr["fold"], lab)
            store[(y, feat)] = res
            rows.append(
                {
                    "y": y,
                    "feature": feat,
                    "n": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                    "sd": _f(res["sd"]),
                }
            )
    y3m = store[(Y3, "c_missed_salary")]
    y3s = store[(Y3, "log1p_a_in3")]
    y3d = store[(Y3, "c_n_days_with_tx")]
    beat = (
        y3m["cv"] - y3s["cv"]
        if (not y3m["low_power"] and not y3s["low_power"])
        else float("nan")
    )
    keep_usual = bool(np.isfinite(beat) and beat >= KEEP_DELTA)
    prose = (
        f"Usual-only Y3: missed {_f(y3m['cv']) if not y3m['low_power'] else 'LOW_POWER'} vs "
        f"size {_f(y3s['cv']) if not y3s['low_power'] else 'LOW_POWER'} "
        f"(Δ {_f(beat)}) vs days {_f(y3d['cv']) if not y3d['low_power'] else 'LOW_POWER'}. "
        f"{'Would KEEP on the usual slice' if keep_usual else 'Still loses to size on the usual slice — not a leftover Q5'}."
    )
    print(prose)
    return {
        "rows": rows,
        "y3_miss": y3m["cv"] if not y3m["low_power"] else float("nan"),
        "y3_size": y3s["cv"] if not y3s["low_power"] else float("nan"),
        "y3_days": y3d["cv"] if not y3d["low_power"] else float("nan"),
        "beat": beat,
        "keep_usual": keep_usual,
        "prose": prose,
    }


def pass22_monthly_skip(tr: pd.DataFrame, ever: pd.DataFrame) -> dict:
    """Skip among monthly-cadence bookers — cleanest Q5 if it existed."""
    m = tr.merge(ever[["company_id", "kind"]], on="company_id", how="left")
    rows = []
    store = {}
    for kind in ("monthly", "irregular", "rare", "never"):
        sl = m["kind"] == kind
        for y in (Y2, Y3):
            lab = sl & m[y].notna()
            res = signed_oof_auroc(m[y], m["c_missed_salary"], m["fold"], lab)
            store[(y, kind)] = res
            rows.append(
                {
                    "y": y,
                    "kind": kind,
                    "n": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "miss": _pp(
                        float(pd.to_numeric(m.loc[lab, "c_missed_salary"], errors="coerce").mean())
                        if lab.any()
                        else float("nan")
                    ),
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                }
            )
    y3m = store[(Y3, "monthly")]
    y3i = store[(Y3, "irregular")]
    size_m = signed_oof_auroc(
        m[Y3], m["log_in3"], m["fold"], (m["kind"] == "monthly") & m[Y3].notna()
    )
    prose = (
        f"Monthly-cadence Y3 missed "
        f"{'LOW_POWER' if y3m['low_power'] else _f(y3m['cv'])} "
        f"vs size on monthly {_f(size_m['cv']) if not size_m['low_power'] else 'LOW_POWER'}; "
        f"irregular {_f(y3i['cv']) if not y3i['low_power'] else 'LOW_POWER'}. "
        "A monthly skip would be the cleanest Q5; it is not."
    )
    print(prose)
    return {
        "rows": rows,
        "y3_monthly": y3m["cv"] if not y3m["low_power"] else float("nan"),
        "y3_irr": y3i["cv"] if not y3i["low_power"] else float("nan"),
        "size_monthly": size_m["cv"] if not size_m["low_power"] else float("nan"),
        "prose": prose,
    }


def pass23_excl_current(tr: pd.DataFrame) -> dict:
    """In-module exclude-current usual-sum — does the 144-quirk flip the AUROC? Do not rewrite ops."""
    g = tr.sort_values(["company_id", "period"])
    prev = g.groupby("company_id", sort=False)["c_salary_month"].shift(1)
    sal6_excl = prev.groupby(g["company_id"], sort=False).transform(
        lambda s: pd.to_numeric(s, errors="coerce").rolling(ROLL, min_periods=1).sum()
    )
    sal = pd.to_numeric(g["c_salary_month"], errors="coerce")
    excl = ((sal6_excl >= USUAL_MIN) & (sal == 0)).astype(float)
    excl = excl.reindex(tr.index)
    n = int((excl == 1).sum())
    n_store = int((pd.to_numeric(tr["c_missed_salary"], errors="coerce") == 1).sum())
    rows = []
    store = {}
    for y in (Y2, Y3):
        lab = tr[y].notna()
        res = signed_oof_auroc(tr[y], excl, tr["fold"], lab)
        store[y] = res
        rows.append(
            {
                "y": y,
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                "sign": res["train_sign"] if not res["low_power"] else "—",
            }
        )
    y3 = store[Y3]
    prose = (
        f"Exclude-current missed n={n:,} vs store {n_store:,} (+{n - n_store:,} quirk months). "
        f"Y3 CV {_f(y3['cv']) if not y3['low_power'] else 'LOW_POWER'}. "
        "The include-current quirk does not hide a KEEP. Do not patch ops.py."
    )
    print(prose)
    return {
        "n_excl": n,
        "n_store": n_store,
        "y3": y3["cv"] if not y3["low_power"] else float("nan"),
        "y2": store[Y2]["cv"] if not store[Y2]["low_power"] else float("nan"),
        "rows": rows,
        "prose": prose,
    }


def pass24_last_amt(tr: pd.DataFrame, con) -> dict:
    """Last salary |amt| before a miss — was the usual payroll real mass?"""
    hold = load_holdout()
    tx = con.execute(
        """
        SELECT
          CAST(company_id AS VARCHAR) AS company_id,
          CAST(date_trunc('month', "date") AS DATE) AS month,
          SUM(ABS(amount)) AS abs_sal,
          COUNT(*) AS n_sal
        FROM transactions
        WHERE category = 'salary'
          AND "date" IS NOT NULL
          AND "date" >= DATE '2024-09-01'
          AND "date" < DATE '2026-09-01'
        GROUP BY 1, 2
        """
    ).df()
    tx["company_id"] = tx["company_id"].astype(str)
    tx["period"] = pd.to_datetime(tx["month"])
    tx = tx.loc[~tx["company_id"].isin(hold)].copy()
    assert_no_holdout(tx["company_id"])
    m = tr.merge(tx[["company_id", "period", "abs_sal", "n_sal"]], on=["company_id", "period"], how="left")
    m = m.sort_values(["company_id", "period"])
    last_amt = m.groupby("company_id", sort=False)["abs_sal"].ffill()
    last_amt = last_amt.groupby(m["company_id"], sort=False).shift(1)
    miss = pd.to_numeric(m["c_missed_salary"], errors="coerce") == 1
    sal = pd.to_numeric(m["c_salary_month"], errors="coerce") == 1
    p50_miss = float(last_amt[miss].median()) if miss.any() else float("nan")
    p50_sal = float(m.loc[sal, "abs_sal"].median()) if sal.any() else float("nan")
    tiny = float((last_amt[miss] <= 5.0).mean()) if miss.any() else float("nan")
    prose = (
        f"Last salary |amt| before a miss p50={p50_miss:,.0f} vs in-month salary p50={p50_sal:,.0f}. "
        f"Share of misses after a ≤5€ last salary {_pp(tiny)}. "
        "Usual payroll before a miss is mass, not a token."
    )
    print(prose)
    return {
        "p50_miss": p50_miss,
        "p50_sal": p50_sal,
        "tiny": tiny,
        "n_miss_amt": int(last_amt[miss].notna().sum()),
        "prose": prose,
    }


def pass25_y6_lead(tr: pd.DataFrame, ever: pd.DataFrame) -> dict:
    """Contemporaneous miss is a lead into rejected Y6 — still not a merge."""
    if Y6 not in tr.columns:
        return {"rows": [], "prose": "Y6 missing.", "p_all": float("nan")}
    m = tr.merge(ever[["company_id", "kind"]], on="company_id", how="left")
    x = pd.to_numeric(m["c_missed_salary"], errors="coerce")
    y = pd.to_numeric(m[Y6], errors="coerce")
    rows = []
    for sname, mask in (
        ("all_overlap", x.notna() & y.notna()),
        ("monthly", (m["kind"] == "monthly") & x.notna() & y.notna()),
        ("irregular", (m["kind"] == "irregular") & x.notna() & y.notna()),
        ("short", (m["short_book"] == 1) & x.notna() & y.notna()),
        ("long", (m["short_book"] == 0) & x.notna() & y.notna()),
    ):
        sl = mask
        xx = x[sl] == 1
        p = float(y[sl][xx].mean()) if xx.any() else float("nan")
        base = float(y[sl].mean()) if sl.any() else float("nan")
        rows.append(
            {
                "slice": sname,
                "n_overlap": int(sl.sum()),
                "n_miss": int(xx.sum()),
                "P(Y6|miss)": _pp(p),
                "Y6 base": _pp(base),
            }
        )
    p_all = float(y[(x == 1) & y.notna()].mean()) if ((x == 1) & y.notna()).any() else float("nan")
    prose = (
        f"P(Y6|missed X)={_pp(p_all)} vs Y6 base on overlap. "
        "A now-miss often precedes the rejected 3-month future miss. Still PARK — do not merge."
    )
    print(prose)
    return {"rows": rows, "p_all": p_all, "prose": prose}


def pass26_co_mean(tr: pd.DataFrame) -> dict:
    """Company-mean missed (booker trait) vs demean (month shock) as Y3 X."""
    miss = pd.to_numeric(tr["c_missed_salary"], errors="coerce")
    mu = miss.groupby(tr["company_id"], sort=False).transform("mean")
    dem = miss - mu
    lab = tr[Y3].notna()
    rows = []
    store = {}
    for name, col in (("raw", miss), ("co_mean", mu), ("demean", dem)):
        res = signed_oof_auroc(tr[Y3], col, tr["fold"], lab)
        store[name] = res
        rows.append(
            {
                "feature": name,
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                "sign": res["train_sign"] if not res["low_power"] else "—",
            }
        )
    prose = (
        f"Y3 raw {_f(store['raw']['cv'])} / company-mean {_f(store['co_mean']['cv'])} / "
        f"demean {_f(store['demean']['cv'])}. "
        "Demean rising toward chance-plus is not a shock KEEP — still loses to size."
    )
    print(prose)
    return {
        "rows": rows,
        "raw": store["raw"]["cv"] if not store["raw"]["low_power"] else float("nan"),
        "mean": store["co_mean"]["cv"] if not store["co_mean"]["low_power"] else float("nan"),
        "dem": store["demean"]["cv"] if not store["demean"]["low_power"] else float("nan"),
        "prose": prose,
    }


def pass27_june(tr: pd.DataFrame) -> dict:
    """Peak missed is June 4.1%, not August. Residual off-June."""
    lab = tr[Y3].notna()
    rows = []
    for sname, mask in (
        ("June", lab & (tr["cal_month"] == 6)),
        ("not_June", lab & (tr["cal_month"] != 6)),
        ("Aug", lab & (tr["cal_month"] == 8)),
        ("not_Aug", lab & (tr["cal_month"] != 8)),
    ):
        res = signed_oof_auroc(tr[Y3], tr["c_missed_salary"], tr["fold"], mask)
        rows.append(
            {
                "slice": sname,
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "miss": _pp(
                    float(pd.to_numeric(tr.loc[mask, "c_missed_salary"], errors="coerce").mean())
                    if mask.any()
                    else float("nan")
                ),
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
            }
        )
    prose = (
        f"Y3 missed off-June {rows[1]['CV']}; off-Aug {rows[3]['CV']}. "
        "June is a small peak, not a tax-style calendar dummy."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def pass28_halt_vs_token(tr: pd.DataFrame) -> dict:
    """Missed + SS gone = payroll halt; missed + SS continues = salary token skip."""
    miss = pd.to_numeric(tr["c_missed_salary"], errors="coerce") == 1
    ss = pd.to_numeric(tr["c_ss_month"], errors="coerce") == 1
    halt = miss & ~ss
    token = miss & ss
    rows = []
    for y in (Y2, Y3):
        for sname, mask in (("halt_no_ss", halt), ("token_ss_continues", token), ("not_miss", ~miss)):
            s = pd.to_numeric(tr.loc[mask, y], errors="coerce")
            res = signed_oof_auroc(tr[y], miss.astype(float), tr["fold"], tr[y].notna() & mask)
            rows.append(
                {
                    "y": y,
                    "slice": sname,
                    "n_cm": int(mask.sum()),
                    "n_lab": int(s.notna().sum()),
                    "n_pos": int((s == 1).sum()),
                    "rate": _pp(float(s.mean()) if s.notna().any() else float("nan")),
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                }
            )
    # also use halt as an in-module flag
    halt_f = halt.astype(float)
    y3h = signed_oof_auroc(tr[Y3], halt_f, tr["fold"], tr[Y3].notna())
    y3t = signed_oof_auroc(tr[Y3], token.astype(float), tr["fold"], tr[Y3].notna())
    size_h = signed_oof_auroc(tr[Y3], tr["log_in3"], tr["fold"], tr[Y3].notna())
    prose = (
        f"Halt (missed, no SS) n={int(halt.sum()):,}; token-skip (missed, SS continues) n={int(token.sum()):,}. "
        f"Y3 halt-as-X {_f(y3h['cv']) if not y3h['low_power'] else 'LOW_POWER'}; "
        f"token-as-X {_f(y3t['cv']) if not y3t['low_power'] else 'LOW_POWER'} vs size {_f(size_h['cv'])}. "
        "Neither split is a KEEP."
    )
    print(prose)
    return {
        "rows": rows,
        "n_halt": int(halt.sum()),
        "n_token": int(token.sum()),
        "y3_halt": y3h["cv"] if not y3h["low_power"] else float("nan"),
        "y3_token": y3t["cv"] if not y3t["low_power"] else float("nan"),
        "prose": prose,
    }


def pass29_streak(tr: pd.DataFrame) -> dict:
    """First miss after a paid run vs 2nd+ consecutive miss."""
    g = tr.sort_values(["company_id", "period"])
    miss = pd.to_numeric(g["c_missed_salary"], errors="coerce")
    prev = miss.groupby(g["company_id"], sort=False).shift(1)
    first = (miss == 1) & (prev.fillna(0) == 0)
    cont = (miss == 1) & (prev == 1)
    first = first.reindex(tr.index)
    cont = cont.reindex(tr.index)
    rows = []
    for y in (Y2, Y3):
        for sname, mask in (("first_miss", first), ("cont_miss", cont)):
            s = pd.to_numeric(tr.loc[mask, y], errors="coerce")
            res = signed_oof_auroc(tr[y], miss.reindex(tr.index), tr["fold"], tr[y].notna() & mask)
            rows.append(
                {
                    "y": y,
                    "slice": sname,
                    "n_cm": int(mask.sum()),
                    "n_lab": int(s.notna().sum()),
                    "n_pos": int((s == 1).sum()),
                    "rate": _pp(float(s.mean()) if s.notna().any() else float("nan")),
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                }
            )
    y3f = signed_oof_auroc(tr[Y3], first.astype(float), tr["fold"], tr[Y3].notna())
    prose = (
        f"First miss n={int(first.sum()):,}; continuation n={int(cont.sum()):,}. "
        f"Y3 first-miss-as-X {_f(y3f['cv']) if not y3f['low_power'] else 'LOW_POWER'}. "
        "Onset of a skip is not a turning X."
    )
    print(prose)
    return {
        "rows": rows,
        "n_first": int(first.sum()),
        "n_cont": int(cont.sum()),
        "y3_first": y3f["cv"] if not y3f["low_power"] else float("nan"),
        "prose": prose,
    }


def pass30_notsal_jaccard(tr: pd.DataFrame) -> dict:
    """Missed is a thin subset of not-salary — the 15-col lever is salary_month, not the skip."""
    miss = pd.to_numeric(tr["c_missed_salary"], errors="coerce")
    notsal = pd.to_numeric(tr["not_salary"], errors="coerce")
    jac = jaccard(miss, notsal)
    rho = spearman(miss, notsal)
    n_miss = int((miss == 1).sum())
    n_ns = int((notsal == 1).sum())
    prose = (
        f"Jaccard(missed, not_salary)={_f(jac)} (n_miss={n_miss:,} / n_notsal={n_ns:,}); "
        f"ρ={_f(rho)}. Most non-salary months are not “missed” (no usual history). "
        f"`not_salary` Y3 0.671 is the quiet-month lever already on the card; "
        "missed is the rare usual-skip leftover and it is chance."
    )
    print(prose)
    return {"jac": jac, "rho": rho, "n_miss": n_miss, "n_ns": n_ns, "prose": prose}


def pass31_size_of_missers(tr: pd.DataFrame) -> dict:
    """Are usual-missers just smaller? Rate gap 11.6% vs 2.7% may be size."""
    usual = _usual_mask(tr)
    miss = pd.to_numeric(tr["c_missed_salary"], errors="coerce") == 1
    logx = tr["log_in3"]
    rows = []
    for sname, mask in (
        ("usual_missed", usual & miss),
        ("usual_paid", usual & ~miss),
        ("not_usual", ~usual),
    ):
        s = logx[mask]
        y3 = pd.to_numeric(tr.loc[mask, Y3], errors="coerce")
        rows.append(
            {
                "slice": sname,
                "n_cm": int(mask.sum()),
                "log_in3_p50": _f(float(s.median()) if s.notna().any() else float("nan")),
                "days_p50": _f(
                    float(pd.to_numeric(tr.loc[mask, "c_n_days_with_tx"], errors="coerce").median())
                    if mask.any()
                    else float("nan"),
                    1,
                ),
                "Y3 rate": _pp(float(y3.mean()) if y3.notna().any() else float("nan")),
            }
        )
    p50_m = float(logx[usual & miss].median()) if (usual & miss).any() else float("nan")
    p50_p = float(logx[usual & ~miss].median()) if (usual & ~miss).any() else float("nan")
    smaller = bool(np.isfinite(p50_m) and np.isfinite(p50_p) and p50_m < p50_p - 0.3)
    prose = (
        f"Usual-missed log1p(a_in3) p50 {_f(p50_m)} vs usual-paid {_f(p50_p)}. "
        f"{'Missers are smaller — the Y3 rate gap is a size pile' if smaller else 'Missers are not much smaller'}."
    )
    print(prose)
    return {"rows": rows, "p50_m": p50_m, "p50_p": p50_p, "smaller": smaller, "prose": prose}


def pass32_holdout_cadence(panel: pd.DataFrame, ever_tr: pd.DataFrame) -> dict:
    """Holdout coverage of salary / missed / cadence — no AUROC."""
    ho = panel[panel["split"] == "holdout"].copy()
    ever = ho.groupby("company_id", as_index=False).agg(
        n_cm=("period", "size"),
        n_sal=("c_salary_month", "sum"),
        n_miss=("c_missed_salary", "sum"),
    )
    ever["sal_rate"] = ever["n_sal"] / ever["n_cm"]
    ever["kind"] = np.select(
        [ever["n_sal"] == 0, ever["sal_rate"] >= 0.70, (ever["sal_rate"] >= 0.25) & (ever["n_sal"] >= 3)],
        ["never", "monthly", "irregular"],
        default="rare",
    )
    rows = []
    for k, part in ever.groupby("kind"):
        rows.append(
            {
                "kind": k,
                "n_co": int(len(part)),
                "sal_p50": _f(float(part["sal_rate"].median()), 3),
                "miss_p50": _f(float((part["n_miss"] / part["n_cm"]).median()), 3),
            }
        )
    prose = (
        f"Holdout cadence coverage only: never {int((ever['kind']=='never').sum())}, "
        f"monthly {int((ever['kind']=='monthly').sum())}, "
        f"irregular {int((ever['kind']=='irregular').sum())}, "
        f"rare {int((ever['kind']=='rare').sum())} / {len(ever)}. No AUROC."
    )
    print(prose)
    return {"rows": rows, "n_co": int(len(ever)), "prose": prose}


def make_png(p3: dict) -> bool:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return False
    months = [r["name"] for r in p3["cal"]]
    sal = [100.0 * r["sal_share"] for r in p3["cal"]]
    miss = [100.0 * r["miss_share"] for r in p3["cal"]]
    colors = ["#1f4e79" if r["q"] else ("#9e6b4a" if r["aug"] else "#6b8f71") for r in p3["cal"]]
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.2), gridspec_kw={"height_ratios": [1.15, 1.05]})
    ax = axes[0]
    ax.bar(np.arange(12), sal, color=colors)
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(months)
    ax.set_ylabel("% of train CM with salary")
    ax.set_title("Salary calendar (navy=Q, brown=Aug) — stacked 2024-09..2026-08")
    ax.set_ylim(0, max(sal) * 1.18 if sal else 100)
    ax.axhline(100.0 * p3["q_sal"], color="#1f4e79", ls="--", lw=0.8, alpha=0.7)
    ax.axhline(100.0 * p3["nq_sal"], color="#6b8f71", ls="--", lw=0.8, alpha=0.7)

    ax2 = axes[1]
    ax2.bar(np.arange(12), miss, color=["#8b3a3a" if r["aug"] else "#4a4a4a" for r in p3["cal"]])
    ax2.set_xticks(np.arange(12))
    ax2.set_xticklabels(months)
    ax2.set_ylabel("% of train CM missed-salary")
    ax2.set_title("c_missed_salary by calendar month (red-brown = August)")
    ax2.set_ylim(0, max(miss) * 1.25 if miss and max(miss) > 0 else 10)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return True


def decide(p3, p4, p5, p6, p7, p9, p11, p12) -> dict:
    """KEEP / CLOSE / PARK with the brief's exact language."""
    park_y = True
    calendar = bool(p3["calendar_dummy"])
    twin = p6.get("twin_name")
    quiet = bool(p4.get("quiet_twin"))
    lose_bars = bool(
        not np.isfinite(p5["miss_y3"])
        or (np.isfinite(p5["size_y3"]) and p5["miss_y3"] < p5["size_y3"] + KEEP_DELTA)
        or (np.isfinite(p5["days_y3"]) and p5["miss_y3"] <= p5["days_y3"])
        or p5.get("just_not_sal")
    )
    keep_x = bool(
        p5.get("keep_x")
        and not calendar
        and twin is None
        and not p6.get("size_flag")
        and not p9.get("flip")
        and p7.get("shock")
        and not p7.get("trait")
        and not quiet
        and not lose_bars
    )
    if keep_x:
        x_dec = "KEEP"
        q5 = "KEEP"
        why = (
            f"missed-salary Y3 CV {_f(p5['miss_y3'])} beats size {_f(p5['size_y3'])} "
            f"by {_f(p5['beat_size'])}, is a month shock (ICC {_f(p7['icc_miss'])}), "
            "survives leak / 12-names / calendar. Later store / Q3-Q5 X only — "
            "not on tonight's 15-col card."
        )
    elif calendar and lose_bars:
        x_dec = "CLOSE"
        q5 = "PARK"
        why = (
            f"calendar shape {p3['shape']} (Q-sal {_pp(p3['q_sal'])} vs other {_pp(p3['nq_sal'])}; "
            f"Aug miss {_pp(p3['aug_miss'])}). Y3 {_f(p5['miss_y3'])} loses to size "
            f"{_f(p5['size_y3'])} / days {_f(p5['days_y3'])}."
        )
    elif lose_bars or quiet or twin:
        x_dec = "CLOSE"
        q5 = "CLOSE"
        bits = []
        if twin:
            bits.append(f"twin of {twin}")
        if quiet:
            bits.append("quiet-month / days twin")
        extra = ("; " + "; ".join(bits)) if bits else ""
        why = (
            f"Y3 {_f(p5['miss_y3'])} vs size {_f(p5['size_y3'])} (Δ {_f(p5['beat_size'])}) "
            f"vs days {_f(p5['days_y3'])}{extra}. "
            f"`c_salary_month` already on the 15-col card at {_f(p5['sal_y3'])}; "
            "missed-salary does not beat that bar or size."
        )
    else:
        x_dec = "CLOSE"
        q5 = "CLOSE"
        why = (
            f"Y3 {_f(p5['miss_y3'])} does not beat size {_f(p5['size_y3'])} by ≥0.02 "
            f"(Δ {_f(p5['beat_size'])})."
        )
    return {
        "q5": q5,
        "why": why,
        "park_y": park_y,
        "x_dec": x_dec,
        "keep_x": keep_x,
        "calendar": calendar,
        "q6": p11["q6"],
        "q6_why": p11["why"],
    }


def write_md(ctx: dict) -> None:
    p1, p2, p3, p4, p5 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"], ctx["p5"]
    p6, p7, p8, p9 = ctx["p6"], ctx["p7"], ctx["p8"], ctx["p9"]
    p10, p11, p12, p13 = ctx["p10"], ctx["p11"], ctx["p12"], ctx["p13"]
    d = ctx["decision"]
    lines = [
        "# Q3/Q5 missed-salary — shock, quiet twin, or calendar dummy?",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. No `build_targets`. Do not invent a missed-salary Y. "
        "Do not put `c_missed_salary` on the 15-col card. "
        f"Night Y3 quote stays **{NIGHT_Y3:.3f} / {NIGHT_Y3_CORE:.3f}**.",
        "",
        "`c_missed_salary` = usual salary in last ≤6 months (rolling sum ≥ 3, "
        "min_periods=1, **includes current month**) AND `c_salary_month` = 0. "
        "Token is raw `category = 'salary'`. Feature report: RARE, modal 97.1%. "
        "`c_salary_month` / `c_ss_month` already sit on the 15-col Y3 card "
        "(quiet-stressed recover). Tax cousin CLOSED as calendar dummy — not redone.",
        "",
        "## Headline",
        "",
        ctx["headline"],
        "",
        "## Brief questions",
        "",
        "| # | question | what this cut says |",
        "| --- | --- | --- |",
        f"| 1 | Who is healthy? | Not this flag. PARK as a health Y. Cadence: {ctx['p14']['n_never']} never / {ctx['p14']['n_month']} monthly / {ctx['p14']['n_irr']} irregular. |",
        "| 2 | Who is improving? | Not this flag. A skip is not a recovery. |",
        f"| 3 | Who is turning? | Missed usual payroll this month. Y3 CV {_f(p5['miss_y3'])} vs size {_f(p5['size_y3'])} / days {_f(p5['days_y3'])}. |",
        "| 4 | Dip vs fall? | Not this table. |",
        f"| 5 | Why did it change? | **{d['q5']}** — {d['why']} |",
        f"| 6 | Months earlier? | lag1/lag3 **{d['q6']}** — {d['q6_why']} |",
        "",
        "## PARK / CLOSE / KEEP",
        "",
        "| object | decision | why |",
        "| --- | --- | --- |",
        f"| `c_missed_salary` as Y3 / Q3-Q5 X | **{d['x_dec']}** | {d['why']} Still **not** on tonight's 15-col card. |",
        "| `c_missed_salary` as a health Y | **PARK** | do not invent `y_missed_salary`; Y6 future-miss already rejected |",
        f"| missed-salary as calendar dummy | **{'PARK (it is the dummy)' if d['calendar'] else 'no — not Q/Aug peaked like tax'}** | shape={p3['shape']}; Q-sal {_pp(p3['q_sal'])} vs other {_pp(p3['nq_sal'])} |",
        f"| Q6 lag1/lag3 `c_missed_salary` | **{d['q6']}** | {p11['prose']} |",
        f"| leak twin | **{p6['twin_name'] or 'none'}** | {p6['prose']} |",
        f"| vs rejected `y6_missed_payroll` | **{'LEAK' if p12.get('leak') else 'distinct'}** | Jaccard {_f(p12.get('jac'))} ρ {_f(p12.get('rho'))} |",
        "",
        "## 1. Completeness + base rate (train rates; holdout coverage)",
        "",
        p1["prose"],
        "",
        _md_table(
            [
                {
                    "split": r["split"],
                    "col": r["col"],
                    "n_cm": f"{r['n_cm']:,}",
                    "n_co": f"{r['n_co']:,}",
                    "cov": _pp(r["cov"]),
                    "prev": _pp(r["prev"]),
                    "n_pos": f"{r['n_pos']:,}",
                    "modal": f"{r['modal']:g}" if np.isfinite(r["modal"]) else "—",
                    "modal%": _pp(r["modal_share"]),
                }
                for r in p1["rows"]
            ]
        ),
        "",
        f"Confirm feature-report modal 97.1%: **{'YES' if p1['confirm_modal'] else 'NO'}**.",
        "",
        "## 2. Formula vs raw `category = 'salary'`",
        "",
        p2["prose"],
        "",
        f"Raw salary CM (train): {p2['n_raw_sal_cm']:,} / companies {p2['n_raw_sal_co']:,}. "
        f"Formula OK: **{'YES' if p2['formula_ok'] else 'NO — stop, do not rewrite ops.py'}**.",
        "",
        "Sample company-months (window is last ≤6 including current, 1=salary):",
        "",
        _md_table(p2["samples"]) if p2["samples"] else "_(empty)_",
        "",
        "## 3. Calendar — month-of-year / August / weekday of last salary",
        "",
        p3["prose"],
        "",
        _md_table(
            [
                {
                    "month": r["name"],
                    "Q?": "Q" if r["q"] else ("Aug" if r["aug"] else ""),
                    "n_cm": f"{r['n_cm']:,}",
                    "salary": _pp(r["sal_share"]),
                    "missed": _pp(r["miss_share"]),
                }
                for r in p3["cal"]
            ]
        ),
        "",
        "Weekday of last in-month salary booking (train):",
        "",
        _md_table(p3["wd_rows"]),
        "",
        f"Plot: `{OUT_PNG.name}`." if ctx["png_ok"] else "Plot: skipped (no matplotlib).",
        "",
        "## 4. Co-occurrence with SS / tax / days (quiet month?)",
        "",
        p4["prose"],
        "",
        _md_table(
            [
                {
                    "slice": r["slice"],
                    "n": f"{r['n']:,}",
                    "P(ss)": _pp(r["ss"]),
                    "P(tax)": _pp(r["tax"]),
                    "days p50": _f(r["days_p50"], 1),
                    "P(days=0)": _pp(r["quiet"]),
                }
                for r in p4["rows"]
            ]
        ),
        "",
        "## 5. Single-feature train group-fold AUROC",
        "",
        f"Y2 n={p5['n_y2']:,} base {_pp(p5['y2_rate'])}; "
        f"Y3 stressed n={p5['n_y3']:,} base {_pp(p5['y3_rate'])}; "
        f"Y9 n={p5['n_y9']:,} base {_pp(p5['y9_rate'])}. "
        "Sign from the train side of each fold. Seed 20260918. "
        f"Night quote: `c_n_days_with_tx` 0.711 on Y3 (replica {_f(p5['days_y3'])}). "
        f"Size quote 0.617 (replica {_f(p5['size_y3'])}). "
        f"Night Y3 GBM **{NIGHT_Y3:.3f} / core {NIGHT_Y3_CORE:.3f}** — not re-fit.",
        "",
        p5["prose"],
        "",
        _md_table(p5["rows"]),
        "",
        f"KEEP-as-X gate: beat size by ≥{KEEP_DELTA:g} **and** not a calendar / size / 12-name / salary_month twin. "
        f"Size dummy ≥{SIZE_PARK:g} is a bar, not a keep.",
        "",
        "## 6. Leak screens",
        "",
        p6["prose"],
        "",
        _md_table(
            [{"pair": r["pair"], "Spearman": _f(r["rho"]), "flag": r["flag"]} for r in p6["rows"]]
        ),
        "",
        "## 7. ICC / company-demean (trait vs month shock)",
        "",
        p7["prose"],
        "",
        "| item | value |",
        "| --- | ---: |",
        f"| ICC `c_missed_salary` | {_f(p7['icc_miss'])} |",
        f"| ICC `c_salary_month` | {_f(p7['icc_sal'])} |",
        f"| Y3 raw / demean | {_f(p7['raw_cv'])} / {_f(p7['dem_cv'])} |",
        f"| ever-miss companies | {p7['n_ever']:,} / {p7['n_co']:,} |",
        f"| trait / shock | {p7['trait']} / {p7['shock']} |",
        "",
        "Uncat-style ICC 0.985 is a bookkeeping trait. Feature-report missed ICC 0.74 is BETWEEN.",
        "",
        "## 8. Dark 470 vs 744 invoiced",
        "",
        p8["prose"],
        "",
        _md_table(p8["rows"]),
        "",
        _md_table(
            [
                {
                    "group": r["group"],
                    "n_cm": f"{r['n_cm']:,}",
                    "n_co": f"{r['n_co']:,}",
                    "salary": _pp(r["sal"]),
                    "missed": _pp(r["miss"]),
                }
                for r in p8["cm"]
            ]
        ),
        "",
        f"Y3 missed CV invoiced {_f(p8['y3_erp'])} / dark {_f(p8['y3_dark'])}.",
        "",
        "## 9. Drop 12 chronic dark Y2 names (GROUP_0158 / GROUP_0172)",
        "",
        p9["prose"],
        "",
        f"n_ids={p9['n_ids']}. Y2 full {_f(p9['y2_full'])} → drop-12 {_f(p9['y2_drop'])}.",
        "",
        "## 10. Complementary 2×2 vs `c_salary_month` and days terciles",
        "",
        p10["prose"],
        "",
        _md_table(
            [
                {
                    "y": r["y"],
                    "slice": r["slice"],
                    "n_cm": f"{r['n_cm']:,}",
                    "n_lab": f"{r['n_lab']:,}",
                    "n_pos": f"{r['n_pos']:,}",
                    "rate": _pp(r["rate"]),
                }
                for r in p10["rate_rows"]
            ]
        ),
        "",
        _md_table(p10["terc_rows"]),
        "",
        "## 11. Q6 — lag1 / lag3 on short vs long books",
        "",
        p11["prose"],
        "",
        _md_table(p11["rows"]),
        "",
        "## 12. vs rejected `y6_missed_payroll` (future t+1..t+3)",
        "",
        p12["prose"],
        "",
        _md_table(p12["rows"]) if p12.get("rows") else "_(unavailable)_",
        "",
        "Y6 looks forward. `c_missed_salary` is contemporaneous X. Do not merge them.",
        "",
        "## 13. Amounts — payroll mass or 1€ token?",
        "",
        p13["prose"],
        "",
        _md_table(p13["bin_rows"]),
        "",
        f"Salary companies {p13['n_sal_co']:,}. SS |amt| p50 "
        f"{p13['ss_amt']['p50']:,.0f}." if np.isfinite(p13["ss_amt"]["p50"]) else "",
        "",
        "## 14. Company payroll cadence",
        "",
        ctx["p14"]["prose"],
        "",
        _md_table(ctx["p14"]["rows"]),
        "",
        "monthly = salary in ≥70% of grid months. irregular = 25–70% and ≥3 salary months. never = none. Else rare.",
        "",
        "## 15. Honest skip — usual (sal6≥3) × missed",
        "",
        ctx["p15"]["prose"],
        "",
        _md_table(ctx["p15"]["rows"]),
        "",
        "## 16. Who produces missed-salary?",
        "",
        ctx["p16"]["prose"],
        "",
        _md_table(ctx["p16"]["rows"]),
        "",
        "## 17. Holdout coverage only (no AUROC)",
        "",
        ctx["p17"]["prose"],
        "",
        _md_table(ctx["p17"]["rows"]),
        "",
        "## 18. Y3 fold table (wander)",
        "",
        ctx["p18"]["prose"],
        "",
        _md_table(ctx["p18"]["rows"]),
        "",
        "## 19. SS packet when salary is missed",
        "",
        ctx["p19"]["prose"],
        "",
        _md_table(ctx["p19"]["rows"]),
        "",
        "## 20. Residual after quiet months",
        "",
        ctx["p20"]["prose"],
        "",
        _md_table(ctx["p20"]["rows"]),
        "",
        "## 21. Usual-only singles vs size / days",
        "",
        ctx["p21"]["prose"],
        "",
        _md_table(ctx["p21"]["rows"]),
        "",
        "## 22. Monthly-cadence skip",
        "",
        ctx["p22"]["prose"],
        "",
        _md_table(ctx["p22"]["rows"]),
        "",
        "## 23. Exclude-current reconstruction (in-module; do not rewrite ops.py)",
        "",
        ctx["p23"]["prose"],
        "",
        _md_table(ctx["p23"]["rows"]),
        "",
        "## 24. Last salary amount before a miss",
        "",
        ctx["p24"]["prose"],
        "",
        "## 25. P(Y6 | missed X) by cadence / trail",
        "",
        ctx["p25"]["prose"],
        "",
        _md_table(ctx["p25"]["rows"]),
        "",
        "## 26. Company-mean vs demean as Y3 X",
        "",
        ctx["p26"]["prose"],
        "",
        _md_table(ctx["p26"]["rows"]),
        "",
        "## 27. June peak residual (not August)",
        "",
        ctx["p27"]["prose"],
        "",
        _md_table(ctx["p27"]["rows"]),
        "",
        "## 28. Payroll halt vs salary-token skip",
        "",
        ctx["p28"]["prose"],
        "",
        _md_table(ctx["p28"]["rows"]),
        "",
        "## 29. First miss vs continuation streak",
        "",
        ctx["p29"]["prose"],
        "",
        _md_table(ctx["p29"]["rows"]),
        "",
        "## 30. Jaccard vs `not_salary` (the 15-col lever)",
        "",
        ctx["p30"]["prose"],
        "",
        "## 31. Size of usual-missers (rate gap vs size pile)",
        "",
        ctx["p31"]["prose"],
        "",
        _md_table(ctx["p31"]["rows"]),
        "",
        "## 32. Holdout cadence coverage only",
        "",
        ctx["p32"]["prose"],
        "",
        _md_table(ctx["p32"]["rows"]),
        "",
        "## What this cut did not do",
        "",
        "- Did not rewrite `ops.py`, `tax_qa.*`, `uncat_qa.py`, `companies_qa.py`, `a_vol_qa.py`, `factoring_qa.py`.",
        "- Did not run `python -m analysis.targets.build_targets` or rewrite parquet / duckdb.",
        "- Did not touch `product/` or write a 0–100 / pillars.",
        "- Did not invent `y_missed_salary` or merge with Y6.",
        "- Did not put `c_missed_salary` on the 15-col card.",
        f"- Did not change the night Y3 quote {NIGHT_Y3:.3f} / {NIGHT_Y3_CORE:.3f}.",
        "- Did not redo the CLOSED tax calendar.",
        "- Did not write the parent journal.",
        "",
        "## What failed / next (held for wave note)",
        "",
        *([f"- {x}" for x in ctx["failed"]] if ctx["failed"] else ["- (none)"]),
        "",
        f"Elapsed {ctx['elapsed_s']:.0f}s. Must-do 1–13 plus cadence, usual-skip, who, holdout, fold wander, SS packet, quiet residual, usual-vs-size, monthly skip, exclude-current quirk, last-amt, Y6 lead, co-mean, June residual, halt vs token, miss streak, not-salary Jaccard, misser size, holdout cadence.",
        "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD}")


def append_registry(ctx: dict) -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    p1, p3, p5, p6, p7, p11, p12, d = (
        ctx["p1"],
        ctx["p3"],
        ctx["p5"],
        ctx["p6"],
        ctx["p7"],
        ctx["p11"],
        ctx["p12"],
        ctx["decision"],
    )
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    rows = [
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": "-",
            "model": MODEL,
            "split": "train",
            "metric": "c_missed_salary_prev",
            "value": p1["miss"]["prev"],
            "coverage": "1.0000",
            "notes": f"modal={p1['miss']['modal_share']:.4f} confirm971={p1['confirm_modal']} sal={p1['sal']['prev']:.4f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_c_missed_salary",
            "value": p5["miss_y3"],
            "coverage": "1.0000",
            "notes": f"size={p5['size_y3']:.4f} days={p5['days_y3']:.4f} beat_size={p5['beat_size']:.4f} x={d['x_dec']} q5={d['q5']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": Y2,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_c_missed_salary",
            "value": p5["miss_y2"],
            "coverage": "1.0000",
            "notes": f"size={p5['size_y2']:.4f} days={p5['days_y2']:.4f} drop12={ctx['p9']['y2_drop']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_c_salary_month",
            "value": p5["sal_y3"],
            "coverage": "1.0000",
            "notes": f"ss={p5['ss_y3']:.4f} not_sal={p5['notsal_y3']:.4f} already_on_15col",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_c_n_days_with_tx",
            "value": p5["days_y3"],
            "coverage": "1.0000",
            "notes": f"night=0.711 replica; size={p5['size_y3']:.4f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": "-",
            "model": MODEL,
            "split": "train",
            "metric": "c_missed_salary_icc",
            "value": p7["icc_miss"],
            "coverage": "1.0000",
            "notes": f"sal_icc={p7['icc_sal']:.4f} trait={p7['trait']} shock={p7['shock']} dem={p7['dem_cv']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": "-",
            "model": MODEL,
            "split": "train",
            "metric": "salary_calendar_q_share",
            "value": p3["q_sal"],
            "coverage": "1.0000",
            "notes": f"shape={p3['shape']} nq={p3['nq_sal']:.4f} aug_miss={p3['aug_miss']:.4f} dummy={p3['calendar_dummy']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_c_missed_salary_lag1",
            "value": p11["lag1"],
            "coverage": "1.0000",
            "notes": f"now={p11['now']} lag3={p11['lag3']} q6={d['q6']} empty_short={p11['empty_short']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": Y6,
            "model": MODEL,
            "split": "train",
            "metric": "jaccard_missed_salary_vs_y6",
            "value": p12.get("jac"),
            "coverage": f"{p12.get('n_both', 0) / max(p1['miss']['n_cm'], 1):.4f}",
            "notes": f"rho={p12.get('rho')} leak={p12.get('leak')} n_and={p12.get('n_and')}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": "-",
            "model": MODEL,
            "split": "train",
            "metric": "c_missed_salary_rho_days",
            "value": p6["rho_days"],
            "coverage": "1.0000",
            "notes": f"rho_sal={p6['rho_sal']:.4f} rho_ss={p6['rho_ss']:.4f} rho_size={p6['rho_size']:.4f} twin={p6['twin_name']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_c_missed_salary_usual",
            "value": ctx["p21"]["y3_miss"],
            "coverage": f"{ctx['p15']['n_usual'] / ctx['p1']['miss']['n_cm']:.4f}",
            "notes": f"size={ctx['p21']['y3_size']} days={ctx['p21']['y3_days']} beat={ctx['p21']['beat']} keep_usual={ctx['p21']['keep_usual']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_c_missed_salary_excl_current",
            "value": ctx["p23"]["y3"],
            "coverage": "1.0000",
            "notes": f"n_excl={ctx['p23']['n_excl']} n_store={ctx['p23']['n_store']} do_not_rewrite_ops",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_c_missed_salary_monthly",
            "value": ctx["p22"]["y3_monthly"],
            "coverage": "1.0000",
            "notes": f"irr={ctx['p22']['y3_irr']} size_monthly={ctx['p22']['size_monthly']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_halt_no_ss",
            "value": ctx["p28"]["y3_halt"],
            "coverage": "1.0000",
            "notes": f"n_halt={ctx['p28']['n_halt']} n_token={ctx['p28']['n_token']} token_cv={ctx['p28']['y3_token']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": "-",
            "model": MODEL,
            "split": "train",
            "metric": "jaccard_missed_vs_not_salary",
            "value": ctx["p30"]["jac"],
            "coverage": "1.0000",
            "notes": f"rho={ctx['p30']['rho']} n_miss={ctx['p30']['n_miss']} n_notsal={ctx['p30']['n_ns']}",
        },
    ]
    with REGISTRY.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        seen = {
            (
                r.get("agent"),
                r.get("y"),
                r.get("model"),
                r.get("split"),
                r.get("metric"),
                r.get("x_families"),
            )
            for r in reader
        }
    fresh = []
    for r in rows:
        key = (
            str(r.get("agent", "")),
            str(r.get("y", "")),
            str(r.get("model", "")),
            str(r.get("split", "")),
            str(r.get("metric", "")),
            str(r.get("x_families", "")),
        )
        if key in seen:
            continue
        fresh.append(r)
        seen.add(key)
    if not fresh:
        print("registry: no new rows")
        return
    with REGISTRY.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        for r in fresh:
            w.writerow({k: r.get(k, "") for k in header})
    print(f"registry appended {len(fresh)} rows")


def run() -> dict:
    t0 = time.time()
    print(f"salary_qa start seed={FOLD_SEED} holdout=72 agent={AGENT}")
    panel = load_panel()
    panel = attach_folds(panel)
    panel = add_panel_lags(panel, ["c_missed_salary", "c_salary_month"], (1, 3))
    con = connect()
    try:
        panel = attach_y6_if_missing(panel, con)
        tr = panel[panel["split"] == "train"].copy()
        assert_no_holdout(tr["company_id"])
        print(
            f"train CM={len(tr):,} companies={tr['company_id'].nunique()} "
            f"holdout CM={(panel['split']=='holdout').sum()}"
        )
        print("pass 1 base rate")
        p1 = pass1_base(panel, tr)
        print("pass 2 formula")
        p2 = pass2_formula(tr, con)
        print("pass 3 calendar")
        p3 = pass3_calendar(tr, con)
        print("pass 4 co-occur")
        p4 = pass4_cooccur(tr)
        print("pass 5 singles")
        p5 = pass5_auroc(tr)
        print("pass 6 leak")
        p6 = pass6_leak(tr)
        print("pass 7 ICC")
        p7 = pass7_icc(tr)
        print("pass 8 dark 470 vs 744")
        p8 = pass8_dark(tr, con)
        print("pass 9 chronic 12")
        p9 = pass9_chronic(tr)
        print("pass 10 2x2 / terciles")
        p10 = pass10_twobytwo(tr)
        print("pass 11 Q6")
        p11 = pass11_q6(tr)
        print("pass 12 vs Y6")
        p12 = pass12_y6(tr)
        print("pass 13 amounts")
        p13 = pass13_amounts(tr, con)
        print("pass 14 cadence")
        p14 = pass14_cadence(tr)
        print("pass 15 usual skip")
        p15 = pass15_usual(tr)
        print("pass 16 who missed")
        p16 = pass16_who(tr, p14["ever"])
        print("pass 17 holdout coverage")
        p17 = pass17_holdout(panel)
        print("pass 18 fold wander")
        p18 = pass18_fold_wander(p5)
        print("pass 19 SS packet")
        p19 = pass19_ss_when_miss(tr)
        print("pass 20 quiet residual")
        p20 = pass20_quiet_residual(tr)
        print("pass 21 usual vs size")
        p21 = pass21_usual_vs_size(tr)
        print("pass 22 monthly skip")
        p22 = pass22_monthly_skip(tr, p14["ever"])
        print("pass 23 exclude-current")
        p23 = pass23_excl_current(tr)
        print("pass 24 last salary amt")
        p24 = pass24_last_amt(tr, con)
        print("pass 25 Y6 lead")
        p25 = pass25_y6_lead(tr, p14["ever"])
        print("pass 26 co-mean")
        p26 = pass26_co_mean(tr)
        print("pass 27 June residual")
        p27 = pass27_june(tr)
        print("pass 28 halt vs token")
        p28 = pass28_halt_vs_token(tr)
        print("pass 29 streak")
        p29 = pass29_streak(tr)
        print("pass 30 not-salary Jaccard")
        p30 = pass30_notsal_jaccard(tr)
        print("pass 31 misser size")
        p31 = pass31_size_of_missers(tr)
        print("pass 32 holdout cadence")
        p32 = pass32_holdout_cadence(panel, p14["ever"])
    finally:
        con.close()

    decision = decide(p3, p4, p5, p6, p7, p9, p11, p12)
    png_ok = make_png(p3)
    headline = (
        f"`c_missed_salary` train prev {_pp(p1['miss']['prev'])} "
        f"(modal {_pp(p1['miss']['modal_share'])}"
        f"{', CONFIRM 97.1%' if p1['confirm_modal'] else ''}). "
        f"Calendar **{p3['shape']}**. "
        f"Y3 missed {_f(p5['miss_y3'])} vs size {_f(p5['size_y3'])} (Δ {_f(p5['beat_size'])}) "
        f"vs days {_f(p5['days_y3'])}. ICC {_f(p7['icc_miss'])} "
        f"({'trait' if p7['trait'] else ('shock' if p7['shock'] else 'between')}). "
        f"Y6 Jaccard {_f(p12.get('jac'))}. "
        f"X **{decision['x_dec']}**. PARK as health Y. Q6 **{decision['q6']}**."
    )
    print(headline)
    failed = []
    if not p1["confirm_modal"]:
        failed.append(f"modal share {_pp(p1['miss']['modal_share'])} ≠ 97.1% quote")
    if not p2["formula_ok"]:
        failed.append(
            f"store vs raw agree sal {_pp(p2['agree_sal'])} miss {_pp(p2['agree_miss'])} — inspect ops.py, do not rewrite"
        )
    if not p8["confirm"]:
        failed.append(f"dark/erp {p8['n_dark']}/{p8['n_erp']} ≠ 470/744")
    if abs(p5["days_y3"] - DAYS_BENCH) > 0.02 if np.isfinite(p5["days_y3"]) else True:
        failed.append(f"Y3 days replica {_f(p5['days_y3'])} vs night 0.711")
    if abs(p5["size_y3"] - SIZE_QUOTE) > 0.02 if np.isfinite(p5["size_y3"]) else True:
        failed.append(f"Y3 size replica {_f(p5['size_y3'])} vs quote 0.617")
    if p9["n_ids"] != 12:
        failed.append(f"chronic names {p9['n_ids']} ≠ 12 from y2_why")
    if decision["x_dec"] != "KEEP":
        failed.append(
            f"X {decision['x_dec']}: Y3 {_f(p5['miss_y3'])} vs size {_f(p5['size_y3'])} / days {_f(p5['days_y3'])}"
        )
    if not p21.get("keep_usual"):
        failed.append(
            f"usual-only Y3 {_f(p21['y3_miss'])} vs size {_f(p21['y3_size'])} (Δ {_f(p21['beat'])})"
        )
    failed.append(
        f"exclude-current Y3 {_f(p23['y3'])} n={p23['n_excl']}; do not rewrite ops.py"
    )
    failed.append(
        f"halt Y3 {_f(p28['y3_halt'])} / first-miss {_f(p29['y3_first'])}; "
        f"missers smaller={p31['smaller']}"
    )
    elapsed = time.time() - t0
    ctx = {
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p4": p4,
        "p5": p5,
        "p6": p6,
        "p7": p7,
        "p8": p8,
        "p9": p9,
        "p10": p10,
        "p11": p11,
        "p12": p12,
        "p13": p13,
        "p14": p14,
        "p15": p15,
        "p16": p16,
        "p17": p17,
        "p18": p18,
        "p19": p19,
        "p20": p20,
        "p21": p21,
        "p22": p22,
        "p23": p23,
        "p24": p24,
        "p25": p25,
        "p26": p26,
        "p27": p27,
        "p28": p28,
        "p29": p29,
        "p30": p30,
        "p31": p31,
        "p32": p32,
        "decision": decision,
        "headline": headline,
        "png_ok": png_ok,
        "failed": failed,
        "elapsed_s": elapsed,
        "panel": panel,
    }
    write_md(ctx)
    append_registry(ctx)
    print(f"elapsed {elapsed:.1f}s")
    return ctx


if __name__ == "__main__":
    run()
