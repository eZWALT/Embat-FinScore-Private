"""Train-only feature evaluation battery (plan section 4).

Holdout companies are excluded from every statistic: near-zero-variance
filters, Spearman / hierarchical clustering, VIF, k-means centroids, and
the StandardScaler used for company clustering. Features are computed for
the full monthly grid (same as the assembler) and then restricted to train.

    python -m analysis.evaluate.feature_report
"""
from __future__ import annotations

import importlib
import math
import sys
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.features.common import ANALYSIS, connect, load_holdout
from analysis.features.grid import company_meta, monthly_grid

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    HAS_VIF = True
except ImportError:
    HAS_VIF = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


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

OUTPUT_DIR = ANALYSIS / "outputs"
REPORT_PATH = OUTPUT_DIR / "feature_report.md"

NZV_THRESH = 0.95
CONSTANT_THRESH = 0.999
SIZE_RHO = 0.85
RHO_CUT = 0.80
MIN_COV_CORR = 0.20
MIN_N_CORR = 400
MIN_ACF_PAIRS = 4
RANDOM_STATE = 20260918
K_RANGE = range(4, 9)
MAX_HEATMAP = 36
BETWEEN_ICC = 0.85

ID_COLS = {
    "company_id",
    "period",
    "freq",
    "first_month",
    "first_week",
    "group_id",
    "country",
    "currency",
    "erp",
    "created_at",
    "group_erp",
}

SHAPE_TOKENS = (
    "share",
    "ratio",
    "margin",
    "growth",
    "hhi",
    "overdue",
    "delay",
    "runway",
    "recency",
    "gap",
    "vol",
    "util",
    "dso",
    "dpo",
    "ds_r",
    "fc_r",
)

# Raw euro / count levels — singleton Spearman clusters still collinear at
# company-median scale (VIF explodes: a_net = a_op_in - a_op_out).
AMOUNT_TOKENS = (
    "op_in",
    "op_out",
    "a_net",
    "a_in3",
    "a_in6",
    "a_in12",
    "a_out3",
    "a_out6",
    "a_out12",
    "b_liq",
    "min_liq",
    "mean_liq",
    "_open",
    "_issued",
    "sib_in",
    "sib_out",
    "sib_net",
    "fin_cost",
    "debt_service",
    "a_transfer",
    "a_invest",
)

CLUSTER_CANDIDATES = [
    "a_io_ratio",
    "a_growth_3",
    "a_uncat_share",
    "a_net_margin",
    "b_runway",
    "b_bal_vol",
    "b_neg_liq_3",
    "b_d_runway",
    "c_zero_in_share_6",
    "c_gap_sd",
    "c_recency_days",
    "d_cust_hhi",
    "d_tx_cp_share",
    "d_interco_share",
    "e_ap_overdue_30",
    "e_ar_overdue_30",
    "e_pending_amt_share",
    "f_ds_r",
    "f_fc_r",
    "g_n_types",
    "g_custom_share",
    "h_share_group_in",
    "h_sib_neg_share",
]

RARE_HINTS = (
    "missed",
    "last_tx",
    "below_0",
    "has_",
    "new_facility",
    "outstanding_gt",
    "zero_in_month",
    "created_after",
    "neg_episodes",
)


@dataclass
class Battery:
    panel: pd.DataFrame
    features: list[str]
    loaded: list[dict]
    skipped: list[dict]
    stats: pd.DataFrame
    rho: pd.DataFrame
    cluster_rows: pd.DataFrame
    vif: pd.Series
    km: dict
    plots: list[str] = field(default_factory=list)
    mpl_ok: bool = HAS_MPL


def _is_feature(name: str) -> bool:
    return len(name) > 2 and name[1] == "_" and name[0] in "abcdefgh"


def feature_columns(panel: pd.DataFrame) -> list[str]:
    cols = []
    for c in panel.columns:
        if c in ID_COLS or not _is_feature(c):
            continue
        if pd.api.types.is_numeric_dtype(panel[c]):
            cols.append(c)
    return cols


def family_of(name: str) -> str:
    return name[0] if _is_feature(name) else "?"


def is_shape(name: str) -> bool:
    low = name.lower()
    return any(tok in low for tok in SHAPE_TOKENS)


def is_amount(name: str) -> bool:
    low = name.lower()
    return any(tok in low for tok in AMOUNT_TOKENS)


def is_rare_event(name: str, n_unique: int, modal_share: float) -> bool:
    if not np.isfinite(modal_share) or modal_share >= CONSTANT_THRESH:
        return False
    if n_unique <= 3 and 0.90 <= modal_share < CONSTANT_THRESH:
        return True
    if "new_facility" in name and modal_share >= 0.90:
        return True
    return any(h in name for h in RARE_HINTS) and n_unique <= 12 and modal_share >= 0.85


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in df.itertuples(index=False):
        cells = []
        for v in row:
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                cells.append("—")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fmt_pct(x: float, digits: int = 1) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{100.0 * x:.{digits}f}%"


def fmt_num(x: float, digits: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    if abs(x) >= 1e5 or (abs(x) > 0 and abs(x) < 1e-3):
        return f"{x:.2e}"
    return f"{x:.{digits}f}"


def _cluster_size_note(km: dict) -> str:
    bits = []
    ct = km.get("crosstab_size")
    if isinstance(ct, pd.DataFrame) and not ct.empty:
        for idx, row in ct.iterrows():
            tot = float(row.sum())
            if tot <= 0:
                continue
            share = row.astype(float) / tot
            tname = str(share.idxmax())
            if float(share.max()) >= 0.80 and tot >= 20:
                bits.append(
                    f"C{int(idx)} is {fmt_pct(float(share.max()))} {tname} (n={int(tot)}) — a size pocket, not a type"
                )
    tiny = [f"C{k}={v}" for k, v in (km.get("sizes") or {}).items() if v < 15]
    if tiny:
        bits.append(f"{', '.join(tiny)} is too small to model")
    if not bits:
        return "If a single cluster eats one tertile, the 'types' are size."
    return " ".join(b if b.endswith(".") else b + "." for b in bits)


def cramers_v(ct: pd.DataFrame) -> float:
    if ct.size == 0 or ct.values.sum() == 0:
        return float("nan")
    chi2 = chi2_contingency(ct.values, correction=False)[0]
    n = float(ct.values.sum())
    r, c = ct.shape
    k = min(r, c) - 1
    if n <= 0 or k <= 0:
        return float("nan")
    return float(math.sqrt(chi2 / (n * k)))


def load_panel() -> tuple[pd.DataFrame, list[dict], list[dict]]:
    con = connect()
    grid = monthly_grid(con)
    meta = company_meta(con)
    hold = load_holdout()
    panel = grid.merge(meta, on="company_id", how="left")
    panel["company_id"] = panel["company_id"].astype(str)
    panel["period"] = pd.to_datetime(panel["period"])
    loaded: list[dict] = []
    skipped: list[dict] = []
    for letter, modname in FAMILIES:
        try:
            mod = importlib.import_module(modname)
        except ModuleNotFoundError:
            skipped.append({"letter": letter, "mod": modname, "err": "not importable"})
            print(f"skip {modname} (not written yet)", flush=True)
            continue
        try:
            part = mod.build(con, grid[["company_id", "period"]].copy())
            extra = [c for c in part.columns if c not in {"company_id", "period"}]
            clash = set(extra) & set(panel.columns)
            if clash:
                raise ValueError(f"column clash: {sorted(clash)}")
            part = part.copy()
            part["company_id"] = part["company_id"].astype(str)
            part["period"] = pd.to_datetime(part["period"])
            panel = panel.merge(part, on=["company_id", "period"], how="left")
            loaded.append(
                {
                    "letter": letter,
                    "mod": modname,
                    "cols": extra,
                    "n_cols": len(extra),
                }
            )
            print(f"loaded {letter} n_cols={len(extra)}", flush=True)
        except Exception as e:
            skipped.append({"letter": letter, "mod": modname, "err": f"{type(e).__name__}: {e}"})
            print(f"skip {modname} ({type(e).__name__}: {e})", flush=True)
            traceback.print_exc()
    con.close()
    train = panel.loc[~panel["company_id"].isin(hold)].copy()
    train["company_id"] = train["company_id"].astype(str)
    train["period"] = pd.to_datetime(train["period"])
    print(
        f"train panel {train.shape} companies={train.company_id.nunique()} "
        f"(holdout {len(hold)} excluded)",
        flush=True,
    )
    return train, loaded, skipped


def _anova(panel: pd.DataFrame, col: str) -> tuple[float, float, float, float]:
    s = panel[["company_id", col]].dropna()
    if len(s) < 10 or s[col].nunique(dropna=True) < 2:
        return np.nan, np.nan, np.nan, np.nan
    y = s[col].to_numpy(dtype=float)
    _, inv, counts = np.unique(s["company_id"].to_numpy(), return_inverse=True, return_counts=True)
    if counts.size < 2:
        return np.nan, np.nan, np.nan, np.nan
    means = np.bincount(inv, weights=y) / counts
    grand = float(y.mean())
    ssb = float(np.sum(counts * (means - grand) ** 2))
    ssw = float(np.sum((y - means[inv]) ** 2))
    k = int(counts.size)
    n = int(y.size)
    var_b = ssb / max(k - 1, 1)
    var_w = ssw / max(n - k, 1)
    ratio = var_w / var_b if var_b > 1e-18 else np.inf
    icc = var_b / (var_b + var_w) if (var_b + var_w) > 0 else np.nan
    return var_w, var_b, ratio, icc


def median_acfs(panel: pd.DataFrame, cols: list[str], lags: tuple[int, ...] = (1, 3, 6)) -> pd.DataFrame:
    work = panel[["company_id", "period", *cols]].sort_values(["company_id", "period"])
    acc = {c: {lag: [] for lag in lags} for c in cols}
    n_cos = {c: {lag: 0 for lag in lags} for c in cols}
    for _, g in work.groupby("company_id", sort=False):
        for c in cols:
            s = pd.to_numeric(g[c], errors="coerce").to_numpy(dtype=float)
            for lag in lags:
                if s.size <= lag:
                    continue
                x, y = s[:-lag], s[lag:]
                m = np.isfinite(x) & np.isfinite(y)
                if int(m.sum()) < MIN_ACF_PAIRS:
                    continue
                xx, yy = x[m], y[m]
                if xx.std() <= 1e-15 or yy.std() <= 1e-15:
                    continue
                r = float(np.corrcoef(xx, yy)[0, 1])
                if np.isfinite(r):
                    acc[c][lag].append(r)
                    n_cos[c][lag] += 1
    rows = []
    for c in cols:
        rec = {"feature": c}
        for lag in lags:
            vals = acc[c][lag]
            rec[f"acf_{lag}"] = float(np.nanmedian(vals)) if vals else np.nan
            rec[f"acf_{lag}_n"] = n_cos[c][lag]
        rows.append(rec)
    return pd.DataFrame(rows)


def per_feature_stats(panel: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    n_cm = len(panel)
    n_co = int(panel["company_id"].nunique())
    size = np.log1p(np.abs(pd.to_numeric(panel["a_op_in"], errors="coerce")))
    rows = []
    for c in features:
        s = pd.to_numeric(panel[c], errors="coerce")
        nn = s.dropna()
        cov_cm = float(s.notna().mean()) if n_cm else np.nan
        cov_co = float(panel.loc[s.notna(), "company_id"].nunique() / n_co) if n_co else np.nan
        if nn.empty:
            modal = np.nan
            modal_share = np.nan
            n_unique = 0
        else:
            vc = nn.value_counts()
            modal = float(vc.index[0]) if np.isfinite(vc.index[0]) else vc.index[0]
            modal_share = float(vc.iloc[0] / len(nn))
            n_unique = int(nn.nunique())
        if nn.size >= 50 and n_unique > 1 and size.notna().sum() >= 50:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                size_rho = float(s.corr(size, method="spearman"))
        else:
            size_rho = np.nan
        var_w, var_b, ratio, icc = _anova(panel, c)
        rare = is_rare_event(c, n_unique, modal_share if np.isfinite(modal_share) else 0.0)
        flags = []
        if nn.empty or n_unique <= 1 or (np.isfinite(modal_share) and modal_share >= CONSTANT_THRESH):
            flags.append("CONSTANT")
        elif np.isfinite(modal_share) and modal_share >= NZV_THRESH:
            flags.append("RARE" if rare else "NZV")
        if np.isfinite(size_rho) and abs(size_rho) > SIZE_RHO:
            flags.append("SIZE")
        if np.isfinite(icc) and icc >= BETWEEN_ICC:
            flags.append("BETWEEN")
        if cov_cm < 0.05:
            flags.append("LOWCOV")
        rows.append(
            {
                "feature": c,
                "family": family_of(c),
                "cov_cm": cov_cm,
                "cov_co": cov_co,
                "n_unique": n_unique,
                "modal": modal,
                "modal_share": modal_share,
                "size_rho": size_rho,
                "var_within": var_w,
                "var_between": var_b,
                "ratio_wb": ratio,
                "icc": icc,
                "rare_event": rare,
                "flags": ",".join(flags),
            }
        )
    stats = pd.DataFrame(rows)
    print("computing company-wise autocorr (lags 1,3,6)…", flush=True)
    acf = median_acfs(panel, features)
    stats = stats.merge(acf, on="feature", how="left")
    low_persist = (stats["acf_1"].abs() < 0.25) & stats["acf_1"].notna()
    change_like = stats["feature"].str.contains("growth|d_runway|new_|missed", regex=True)
    merged_flags = []
    for flags, lp, ch in zip(stats["flags"], low_persist, change_like):
        bits = [b for b in str(flags).split(",") if b]
        if bool(lp) and not bool(ch):
            bits.append("LOW_PERSIST")
        merged_flags.append(",".join(bits))
    stats["flags"] = merged_flags
    return stats


def _eligible_for_corr(stats: pd.DataFrame) -> list[str]:
    ok = stats[
        (stats["cov_cm"] >= MIN_COV_CORR)
        & (stats["n_unique"] > 1)
        & (stats["modal_share"].fillna(1.0) < CONSTANT_THRESH)
    ]
    return ok["feature"].tolist()


def spearman_matrix(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame()
    x = panel[cols].apply(pd.to_numeric, errors="coerce")
    keep = [c for c in cols if x[c].notna().sum() >= MIN_N_CORR and x[c].nunique(dropna=True) > 1]
    if len(keep) < 2:
        return pd.DataFrame(index=keep, columns=keep, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = x[keep].corr(method="spearman")
    return rho


def cluster_features(rho: pd.DataFrame, stats: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    if rho.empty or len(rho) < 2:
        empty = pd.DataFrame(columns=["feature", "cluster", "n_in_cluster", "representative", "members"])
        return empty, np.array([])
    abs_rho = rho.abs().clip(0.0, 1.0).fillna(0.0)
    abs_rho = (abs_rho + abs_rho.T) / 2.0
    np.fill_diagonal(abs_rho.values, 1.0)
    dist = (1.0 - abs_rho).clip(lower=0.0)
    np.fill_diagonal(dist.values, 0.0)
    condensed = squareform(dist.values, checks=False)
    z = linkage(condensed, method="average")
    labels = fcluster(z, t=1.0 - RHO_CUT, criterion="distance")
    st = stats.set_index("feature")
    score_map = {}
    for name in rho.columns:
        if name not in st.index:
            score_map[name] = 0.0
            continue
        r = st.loc[name]
        cov = float(r["cov_cm"]) if np.isfinite(r["cov_cm"]) else 0.0
        sr = abs(float(r["size_rho"])) if np.isfinite(r["size_rho"]) else 0.0
        ms = float(r["modal_share"]) if np.isfinite(r["modal_share"]) else 1.0
        shape = 2.5 if is_shape(str(name)) else 0.0
        amount_pen = 2.0 if is_amount(str(name)) else 0.0
        flags = str(r["flags"])
        nzv_pen = 2.0 if ("NZV" in flags or "CONSTANT" in flags) else 0.0
        size_pen = 2.0 if "SIZE" in flags else 0.0
        score_map[str(name)] = (
            3.0 * cov
            + 1.5 * (1.0 - min(sr, 1.0))
            + 1.0 * (1.0 - min(ms, 1.0))
            + shape
            - amount_pen
            - nzv_pen
            - size_pen
        )
    rows = []
    for cid in sorted(set(labels)):
        members = [rho.columns[i] for i, lab in enumerate(labels) if lab == cid]
        members_sorted = sorted(members, key=lambda n: score_map.get(n, 0.0), reverse=True)
        # Prefer trailing-3m inflow as the size-cluster name (persists; raw month does not).
        if "a_in3" in members_sorted:
            members_sorted = ["a_in3"] + [m for m in members_sorted if m != "a_in3"]
        elif "a_op_in" in members_sorted:
            members_sorted = ["a_op_in"] + [m for m in members_sorted if m != "a_op_in"]
        rep = members_sorted[0]
        pair_max = 0.0
        if len(members) > 1:
            sub = abs_rho.loc[members, members]
            iu = np.triu_indices(len(members), k=1)
            pair_max = float(sub.values[iu].max()) if iu[0].size else 0.0
        for m in members_sorted:
            rows.append(
                {
                    "feature": m,
                    "cluster": int(cid),
                    "n_in_cluster": len(members),
                    "representative": rep,
                    "is_rep": m == rep,
                    "members": ", ".join(members_sorted),
                    "cluster_max_abs_rho": pair_max,
                }
            )
    return pd.DataFrame(rows), z


def vif_on_reps(panel: pd.DataFrame, reps: list[str]) -> pd.Series:
    if not HAS_VIF or len(reps) < 2:
        return pd.Series(dtype=float)
    # Raw euro levels are known collinear at the company-median scale.
    use = [c for c in reps if c in panel.columns and not is_amount(c)]
    if len(use) < 2:
        use = [c for c in reps if c in panel.columns]
    med = panel.groupby("company_id")[use].median(numeric_only=True)
    keep = [c for c in use if med[c].notna().mean() >= 0.50 and med[c].nunique(dropna=True) > 1]
    X = med[keep].dropna()
    if len(X) < 40 or X.shape[1] < 2:
        return pd.Series(dtype=float)
    vals = StandardScaler().fit_transform(X.to_numpy(dtype=float))
    out = {}
    for i, col in enumerate(keep):
        try:
            out[col] = float(variance_inflation_factor(vals, i))
        except Exception:
            out[col] = np.nan
    return pd.Series(out).sort_values(ascending=False)


def _select_cluster_features(stats: pd.DataFrame, cluster_rows: pd.DataFrame) -> list[str]:
    present = []
    used_family: dict[str, int] = {}
    used_cluster: set[int] = set()
    st = stats.set_index("feature")
    clus = cluster_rows.set_index("feature") if not cluster_rows.empty else pd.DataFrame()
    for name in CLUSTER_CANDIDATES:
        if name not in st.index:
            continue
        r = st.loc[name]
        if r["cov_cm"] < 0.35 or r["n_unique"] <= 1:
            continue
        if "CONSTANT" in str(r["flags"]) or "NZV" in str(r["flags"]):
            continue
        if "SIZE" in str(r["flags"]):
            continue
        if name in clus.index:
            cid = int(clus.loc[name, "cluster"])
            if cid in used_cluster:
                continue
        else:
            cid = None
        fam = family_of(name)
        if used_family.get(fam, 0) >= 2:
            continue
        present.append(name)
        used_family[fam] = used_family.get(fam, 0) + 1
        if cid is not None:
            used_cluster.add(cid)
        if len(present) >= 12:
            break
    if len(present) >= 6:
        return present
    extras = st[
        (st["cov_cm"] >= 0.50)
        & (st["n_unique"] > 2)
        & ~st["flags"].str.contains("SIZE|CONSTANT|NZV", na=False)
    ].sort_values("cov_cm", ascending=False)
    for name in extras.index:
        if name in present:
            continue
        present.append(str(name))
        if len(present) >= 8:
            break
    return present


def company_clustering(panel: pd.DataFrame, stats: pd.DataFrame, cluster_rows: pd.DataFrame) -> dict:
    feats = _select_cluster_features(stats, cluster_rows)
    out = {
        "features": feats,
        "n_companies": 0,
        "silhouette_by_k": {},
        "best_k": None,
        "best_sil": np.nan,
        "sizes": {},
        "cramer_size": np.nan,
        "nmi_size": np.nan,
        "nmi_group": np.nan,
        "group_purity": np.nan,
        "n_multi_groups": 0,
        "pure_multi_groups": 0,
        "crosstab_size": pd.DataFrame(),
        "labels": pd.Series(dtype=int),
        "med": pd.DataFrame(),
        "tertile": pd.Series(dtype=str),
        "scaled": None,
    }
    if len(feats) < 3 or "a_op_in" not in panel.columns:
        return out
    med = panel.groupby("company_id")[feats].median(numeric_only=True)
    size_med = np.log1p(np.abs(panel.groupby("company_id")["a_op_in"].median()))
    obs = med.notna().mean(axis=1)
    med = med.loc[obs >= 0.5].copy()
    if len(med) < 40:
        return out
    med = med.fillna(med.median(numeric_only=True))
    scaler = StandardScaler()
    xs = scaler.fit_transform(med.to_numpy(dtype=float))
    sils = {}
    models = {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE, max_iter=400)
        lab = km.fit_predict(xs)
        if len(set(lab)) < 2:
            sils[k] = np.nan
            continue
        sils[k] = float(silhouette_score(xs, lab))
        models[k] = lab
    finite = {k: v for k, v in sils.items() if np.isfinite(v)}
    if not finite:
        return out
    best_k = max(finite, key=finite.get)
    labels = pd.Series(models[best_k], index=med.index, name="cluster")
    tert = pd.qcut(size_med.reindex(med.index), 3, labels=["T1_small", "T2_mid", "T3_large"])
    ct = pd.crosstab(labels, tert)
    groups = panel.drop_duplicates("company_id").set_index("company_id")["group_id"]
    g = groups.reindex(med.index)
    nmi_g = float(normalized_mutual_info_score(g.astype(str), labels.astype(str))) if g.notna().all() else np.nan
    multi = g.value_counts()
    multi_ids = set(multi[multi >= 3].index)
    n_multi = 0
    n_pure = 0
    for gid in multi_ids:
        labs = labels[g == gid]
        if labs.empty:
            continue
        n_multi += 1
        if labs.nunique() == 1:
            n_pure += 1
    out.update(
        {
            "n_companies": int(len(med)),
            "silhouette_by_k": sils,
            "best_k": int(best_k),
            "best_sil": float(finite[best_k]),
            "sizes": {int(k): int(v) for k, v in labels.value_counts().sort_index().items()},
            "cramer_size": cramers_v(ct),
            "nmi_size": float(normalized_mutual_info_score(tert.astype(str), labels.astype(str))),
            "nmi_group": nmi_g,
            "group_purity": (n_pure / n_multi) if n_multi else np.nan,
            "n_multi_groups": n_multi,
            "pure_multi_groups": n_pure,
            "crosstab_size": ct,
            "labels": labels,
            "med": med,
            "tertile": tert,
            "scaled": xs,
        }
    )
    return out


def recommend(stats: pd.DataFrame, cluster_rows: pd.DataFrame, vif: pd.Series | None = None) -> dict:
    st = stats.set_index("feature")
    drop: dict[str, str] = {}
    park: dict[str, str] = {}
    keep: dict[str, str] = {}

    reps = set()
    clus_idx = None
    if not cluster_rows.empty:
        reps = set(cluster_rows.loc[cluster_rows["is_rep"], "feature"])
        clus_idx = cluster_rows.set_index("feature")

    for feat, r in st.iterrows():
        flags = str(r["flags"])
        if feat == "a_in3":
            keep[feat] = "canonical size control — trailing 3m inflow (use log1p); more persistent than a_op_in"
            continue
        if "CONSTANT" in flags or r["n_unique"] <= 1 or (np.isfinite(r["cov_cm"]) and r["cov_cm"] == 0):
            drop[feat] = "constant or all-null (no variance)"
            continue
        if np.isfinite(r["cov_cm"]) and r["cov_cm"] < 0.05:
            park[feat] = f"coverage {fmt_pct(r['cov_cm'])} — last-month / sparse only"
            continue
        if bool(r["rare_event"]):
            keep[feat] = f"rare-event flag (modal {fmt_pct(r['modal_share'])})"
            continue
        if "NZV" in flags:
            drop[feat] = f"near-zero variance (modal share {fmt_pct(r['modal_share'])})"
            continue
        if feat == "a_op_in":
            drop[feat] = "raw monthly inflow: SIZE vs its own log and median acf1≈0 (extremes); use log1p(a_in3)"
            continue
        if feat == "a_net":
            drop[feat] = "identity a_op_in − a_op_out (VIF explodes with the two levels)"
            continue
        if "SIZE" in flags:
            drop[feat] = f"size proxy |ρ|={fmt_num(abs(r['size_rho']))} vs log1p(|a_op_in|)"
            continue
        if np.isfinite(r["size_rho"]) and abs(r["size_rho"]) > 0.75 and feat != "a_in3":
            park[feat] = f"near-size |ρ|={fmt_num(abs(r['size_rho']))} vs log inflow — do not add besides a_in3"
            continue
        if clus_idx is not None and feat in clus_idx.index and not bool(clus_idx.loc[feat, "is_rep"]):
            drop[feat] = f"redundant with {clus_idx.loc[feat, 'representative']} (|ρ| cluster cut {RHO_CUT})"
            continue
        if is_amount(feat):
            park[feat] = "raw level amount — singleton cluster on the panel, but company-median VIF collinear with size; log if needed"
            continue
        if feat in reps:
            keep[feat] = "cluster representative (share / ratio / flag)"
        elif np.isfinite(r["cov_cm"]) and r["cov_cm"] < 0.30:
            park[feat] = f"coverage {fmt_pct(r['cov_cm'])} — usable with missingness handling"
        else:
            keep[feat] = "not redundant; keep for now"

    missing = set(st.index) - set(drop) - set(park) - set(keep)
    for feat in sorted(missing):
        keep[feat] = "unclassified fallback — keep pending review"

    vif = vif if vif is not None else pd.Series(dtype=float)
    if "d_cust_new" in keep and "d_n_cust" in keep and float(vif.get("d_cust_new", 0) or 0) > 10:
        park["d_cust_new"] = f"company-median VIF={fmt_num(float(vif['d_cust_new']), 1)} with d_n_cust"
        keep.pop("d_cust_new")
    if "g_n_types" in keep and float(vif.get("g_n_types", 0) or 0) > 15:
        park["g_n_types"] = f"company-median VIF={fmt_num(float(vif['g_n_types']), 1)} with product-mix flags"
        keep.pop("g_n_types")

    return {"drop": drop, "park": park, "keep": keep}


def make_plots(bat: Battery) -> list[str]:
    if not HAS_MPL:
        print("matplotlib not available — skipping PNGs", flush=True)
        return []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    reps = []
    if not bat.cluster_rows.empty:
        reps = bat.cluster_rows.loc[bat.cluster_rows["is_rep"], "feature"].tolist()
    rho = bat.rho
    if not rho.empty and reps:
        use = [c for c in reps if c in rho.columns]
        if len(use) > MAX_HEATMAP:
            cov = bat.stats.set_index("feature")["cov_cm"]
            use = sorted(use, key=lambda n: float(cov.get(n, 0.0)), reverse=True)[:MAX_HEATMAP]
        if len(use) >= 2:
            sub = rho.loc[use, use]
            fig, ax = plt.subplots(figsize=(max(8, 0.32 * len(use) + 2), max(7, 0.32 * len(use) + 1)))
            im = ax.imshow(sub.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(use)))
            ax.set_yticks(range(len(use)))
            ax.set_xticklabels(use, rotation=90, fontsize=7)
            ax.set_yticklabels(use, fontsize=7)
            ax.set_title("Spearman — cluster representatives (train)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            p = OUTPUT_DIR / "feature_corr_heatmap.png"
            fig.savefig(p, dpi=140)
            plt.close(fig)
            paths.append(str(p))

    if not rho.empty and len(rho) >= 3:
        abs_rho = rho.abs().clip(0.0, 1.0).fillna(0.0)
        abs_rho = (abs_rho + abs_rho.T) / 2.0
        np.fill_diagonal(abs_rho.values, 1.0)
        dist = (1.0 - abs_rho).clip(lower=0.0)
        np.fill_diagonal(dist.values, 0.0)
        z = linkage(squareform(dist.values, checks=False), method="average")
        fig, ax = plt.subplots(figsize=(11, max(8, 0.18 * len(rho))))
        dendrogram(z, labels=list(rho.columns), orientation="right", ax=ax, leaf_font_size=6, color_threshold=1.0 - RHO_CUT)
        ax.axvline(1.0 - RHO_CUT, color="crimson", ls="--", lw=0.8, label=f"|ρ|={RHO_CUT} cut")
        ax.set_xlabel("distance = 1 − |Spearman ρ|")
        ax.set_title("Feature hierarchical clustering (average, train)")
        ax.legend(loc="lower right")
        fig.tight_layout()
        p = OUTPUT_DIR / "feature_dendrogram.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(str(p))

    km = bat.km
    if km.get("scaled") is not None and km.get("best_k") is not None:
        xs = km["scaled"]
        lab = km["labels"]
        tert = km["tertile"]
        xy = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(xs)
        fig, ax = plt.subplots(figsize=(8.2, 6.4))
        markers = {"T1_small": "o", "T2_mid": "s", "T3_large": "^"}
        cmap = plt.get_cmap("tab10")
        for tname, marker in markers.items():
            m = tert.astype(str) == tname
            if not m.any():
                continue
            ax.scatter(
                xy[m.values, 0],
                xy[m.values, 1],
                c=[cmap(int(v) % 10) for v in lab[m]],
                marker=marker,
                s=22,
                alpha=0.75,
                edgecolors="none",
                label=tname,
            )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"Company k-means k={km['best_k']} (colour) vs inflow tertile (shape)")
        ax.legend(title="log-inflow tertile", fontsize=8)
        fig.tight_layout()
        p = OUTPUT_DIR / "company_clusters_pca.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(str(p))
    return paths


def _stats_display(stats: pd.DataFrame, family: str | None = None) -> pd.DataFrame:
    s = stats if family is None else stats[stats["family"] == family]
    if s.empty:
        return pd.DataFrame()
    s = s.sort_values(["family", "feature"])
    return pd.DataFrame(
        {
            "feature": s["feature"],
            "cov_cm": s["cov_cm"].map(fmt_pct),
            "cov_co": s["cov_co"].map(fmt_pct),
            "modal%": s["modal_share"].map(fmt_pct),
            "size_ρ": s["size_rho"].map(lambda x: fmt_num(x, 3)),
            "acf1": s["acf_1"].map(lambda x: fmt_num(x, 2)),
            "acf3": s["acf_3"].map(lambda x: fmt_num(x, 2)),
            "acf6": s["acf_6"].map(lambda x: fmt_num(x, 2)),
            "w/b": s["ratio_wb"].map(lambda x: fmt_num(x, 2)),
            "ICC": s["icc"].map(lambda x: fmt_num(x, 2)),
            "flags": s["flags"].replace("", "—"),
        }
    )


def write_report(bat: Battery, rec: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    p = bat.panel
    st = bat.stats
    n_cm = len(p)
    n_co = int(p["company_id"].nunique())
    families = ",".join(x["letter"] for x in bat.loaded) or "—"
    skipped = ", ".join(f"{x['letter']} ({x['err']})" for x in bat.skipped) or "none"

    size = st.dropna(subset=["size_rho"]).assign(abs_rho=lambda d: d["size_rho"].abs())
    size = size.sort_values("abs_rho", ascending=False)
    size_flagged = size[size["abs_rho"] > SIZE_RHO]
    nzv = st[st["flags"].str.contains("NZV|CONSTANT", na=False)].sort_values("modal_share", ascending=False)

    drop_df = pd.DataFrame(
        [{"feature": k, "reason": v} for k, v in sorted(rec["drop"].items(), key=lambda kv: (family_of(kv[0]), kv[0]))]
    )
    park_df = pd.DataFrame(
        [{"feature": k, "reason": v} for k, v in sorted(rec["park"].items(), key=lambda kv: (family_of(kv[0]), kv[0]))]
    )
    keep_df = pd.DataFrame(
        [{"feature": k, "reason": v} for k, v in sorted(rec["keep"].items(), key=lambda kv: (family_of(kv[0]), kv[0]))]
    )

    clus = bat.cluster_rows
    if not clus.empty:
        clus_view = (
            clus.sort_values(["cluster", "is_rep"], ascending=[True, False])
            .groupby("cluster", as_index=False)
            .agg(
                n=("feature", "size"),
                representative=("representative", "first"),
                members=("members", "first"),
                max_abs_rho=("cluster_max_abs_rho", "first"),
            )
        )
        clus_view["max_abs_rho"] = clus_view["max_abs_rho"].map(lambda x: fmt_num(x, 2))
        n_multi = int((clus_view["n"] > 1).sum())
    else:
        clus_view = pd.DataFrame()
        n_multi = 0

    vif_df = pd.DataFrame()
    if len(bat.vif):
        vif_df = pd.DataFrame({"feature": bat.vif.index, "VIF": [fmt_num(v, 2) for v in bat.vif.values]})

    km = bat.km
    sil_rows = []
    for k, v in km.get("silhouette_by_k", {}).items():
        sil_rows.append({"k": k, "silhouette": fmt_num(v, 3), "chosen": "yes" if k == km.get("best_k") else ""})
    sil_df = pd.DataFrame(sil_rows)
    size_ct = km.get("crosstab_size", pd.DataFrame())
    size_ct_md = pd.DataFrame()
    if isinstance(size_ct, pd.DataFrame) and not size_ct.empty:
        size_ct_md = size_ct.reset_index().rename(columns={"cluster": "cluster"})
        size_ct_md["cluster"] = size_ct_md["cluster"].map(lambda x: f"C{int(x)}")
        for c in size_ct_md.columns:
            if c != "cluster":
                size_ct_md[c] = size_ct_md[c].map(lambda x: str(int(x)) if pd.notna(x) else "—")

    plots_md = (
        "\n".join(f"- `{Path(x).name}`" for x in bat.plots)
        if bat.plots
        else ("matplotlib was not available — plots skipped." if not bat.mpl_ok else "No plots produced.")
    )

    km_feats = ", ".join(f"`{c}`" for c in km.get("features", [])) or "—"
    sil = km.get("best_sil")
    sil_note = (
        "strong structure"
        if np.isfinite(sil or np.nan) and sil >= 0.50
        else (
            "weak / overlapping structure — per-cluster models are not justified on this profile set"
            if np.isfinite(sil or np.nan) and sil < 0.25
            else "modest structure — treat per-cluster models as an experiment, not a default"
        )
    )

    lines = [
        "# Feature evaluation report (train only)",
        "",
        "Holdout `analysis/splits/holdout_companies.csv` (72 companies) is **excluded** from every number below. "
        "No percentiles, bins, cluster centroids, or scalers were fit on holdout.",
        "",
        f"- Panel: **{n_cm:,}** company-months, **{n_co:,}** train companies (monthly grid 2024-09 … 2026-08).",
        f"- Families loaded: **{families.upper()}** ({len(st)} numeric features).",
        f"- Families skipped: {skipped}.",
        f"- Size proxy: Spearman of each feature vs `log1p(|a_op_in|)`. Flag `|ρ| > {SIZE_RHO}`. "
        "The **model** size control is `log1p(a_in3)` (trailing 3m; median acf1 ≈ 0.66). "
        "Raw `a_op_in` has median Pearson acf1 ≈ 0 — monthly extremes, not a stable scale.",
        f"- Near-zero variance: modal-value share `≥ {NZV_THRESH}` (constant if `≥ {CONSTANT_THRESH}`).",
        f"- Persistence: Pearson autocorr at lags 1 / 3 / 6, company-wise, then the median across companies "
        f"(need ≥ {MIN_ACF_PAIRS} finite pairs and non-zero s.d.).",
        "- Variance split: ANOVA `var_within` / `var_between`; ICC = between / (between + within). "
        f"ICC `≥ {BETWEEN_ICC}` is flagged BETWEEN (mostly a company identity).",
        f"- Correlation clusters: average-linkage on `1 − |ρ|`, cut at `|ρ| = {RHO_CUT}` "
        f"(features with coverage `< {fmt_pct(MIN_COV_CORR)}` excluded from the matrix).",
        f"- Company clustering: k-means `k = 4…8`, seed `{RANDOM_STATE}`, on standardised company medians. "
        "Silhouette picks k. Clusters are checked against log-inflow tertiles and `group_id`.",
        "",
        "## Headline",
        "",
        "### Top size-proxies",
        "",
        "Ranked by `|Spearman ρ|` vs `log1p(|a_op_in|)`. Only rows with `|ρ| > 0.85` are *flagged* SIZE; "
        "the table shows the top 12 so weaker level-amount leaks are visible too.",
        "",
        md_table(
            pd.DataFrame(
                {
                    "feature": size.head(12)["feature"],
                    "ρ": size.head(12)["size_rho"].map(lambda x: fmt_num(x, 3)),
                    "|ρ|": size.head(12)["abs_rho"].map(lambda x: fmt_num(x, 3)),
                    "flagged": ["SIZE" if x > SIZE_RHO else "" for x in size.head(12)["abs_rho"]],
                }
            )
        )
        if len(size)
        else "_No size correlations._",
        "",
        f"Flagged SIZE (`|ρ| > {SIZE_RHO}`): "
        + (", ".join(f"`{x}`" for x in size_flagged["feature"]) if len(size_flagged) else "none")
        + ".",
        "",
        "### Near-zero variance / constants",
        "",
        md_table(
            pd.DataFrame(
                {
                    "feature": nzv["feature"],
                    "modal%": nzv["modal_share"].map(fmt_pct),
                    "n_unique": nzv["n_unique"].map(lambda x: str(int(x))),
                    "flags": nzv["flags"],
                    "rare_event": nzv["rare_event"].map(lambda x: "yes" if x else ""),
                }
            )
        )
        if len(nzv)
        else "_None._",
        "",
        "Rare-event flags (payroll miss, NSF-like `b_below_0`, product presence, `outstanding_gt_granted`) "
        "are **kept** even when the modal share is high: the minority class is the signal.",
        "",
        "### Company-cluster silhouette",
        "",
        (
            f"Best **k = {km.get('best_k')}**, silhouette **{fmt_num(km.get('best_sil'), 3)}** "
            f"on **{km.get('n_companies')}** companies using {km_feats}."
            if km.get("best_k") is not None
            else "Company clustering did not run (too few usable profile features)."
        ),
        "",
        md_table(sil_df) if len(sil_df) else "",
        "",
        (
            f"Cramér's V vs log-inflow tertile: **{fmt_num(km.get('cramer_size'), 3)}** "
            f"(NMI {fmt_num(km.get('nmi_size'), 3)}). "
            f"NMI vs `group_id`: **{fmt_num(km.get('nmi_group'), 3)}**. "
            f"Among {km.get('n_multi_groups')} groups with ≥ 3 train companies, "
            f"{km.get('pure_multi_groups')} are mono-cluster "
            f"(purity {fmt_pct(km.get('group_purity') or float('nan'))})."
            if km.get("best_k") is not None
            else ""
        ),
        "",
        f"Read: {sil_note}. "
        + (
            "Cramér's V ≥ 0.35 plus a one-tertile pocket means part of the partition is a size cut — "
            "do not treat these as operating types."
            if np.isfinite(km.get("cramer_size") or np.nan) and km.get("cramer_size") >= 0.35
            else "Cramér's V vs size is low enough that the partition is not *only* a size tertile."
            if km.get("best_k") is not None
            else ""
        ),
        "",
        "### Recommended drop list",
        "",
        md_table(drop_df) if len(drop_df) else "_Nothing dropped._",
        "",
        "### Park (sparse, near-size, or raw level — not in the core GBM set)",
        "",
        md_table(park_df) if len(park_df) else "_None._",
        "",
        "### Recommended keep",
        "",
        md_table(keep_df) if len(keep_df) else "_None._",
        "",
        "Core GBM starter set (size + shares/ratios/flags from the keep list): "
        + (
            ", ".join(
                f"`{f}`"
                for f in keep_df["feature"]
                if f == "a_in3" or is_shape(f) or any(h in f for h in RARE_HINTS) or f.startswith("c_missed")
            )
            if len(keep_df)
            else "—"
        )
        + ".",
        "",
        "## 1. Coverage by family",
        "",
    ]

    cov_rows = []
    for rec_f in bat.loaded:
        cols = [c for c in rec_f["cols"] if c in st["feature"].values]
        sub = st[st["feature"].isin(cols)]
        cov_rows.append(
            {
                "family": rec_f["letter"],
                "n_cols": len(cols),
                "median cov_cm": fmt_pct(sub["cov_cm"].median()) if len(sub) else "—",
                "min cov_cm": fmt_pct(sub["cov_cm"].min()) if len(sub) else "—",
                "median ICC": fmt_num(sub["icc"].median(), 2) if len(sub) else "—",
                "n SIZE": int(sub["flags"].str.contains("SIZE").sum()) if len(sub) else 0,
                "n NZV/CONST": int(sub["flags"].str.contains("NZV|CONSTANT").sum()) if len(sub) else 0,
            }
        )
    lines += [md_table(pd.DataFrame(cov_rows)), ""]
    lines += [
        "`d_interco_share` is all-null on this extract (no usable intercompany IDs). "
        "Family F snapshot fields (`f_util_snapshot`, `f_w_rate`, `f_months_to_next_pay`, "
        "`f_sched_vs_obs`) are populated only near the 2026-09-01 book — they are not a panel series.",
        "",
        "## 2. Per-feature battery",
        "",
        "Flags: `SIZE` `|ρ|>0.85` vs log inflow; `NZV` modal share ≥ 0.95; `CONSTANT` modal share ≥ 0.999 "
        "or a single value; `BETWEEN` ICC ≥ 0.85; `LOWCOV` company-month coverage < 5%; `LOW_PERSIST` "
        "median company acf1 `|r|<0.25` on a level (not a change feature); `RARE` = rare-event keep.",
        "",
    ]
    for fam in sorted(st["family"].unique()):
        title = {
            "a": "A — cash flow",
            "b": "B — liquidity",
            "c": "C — operations",
            "d": "D — counterparties",
            "e": "E — invoices",
            "f": "F — debt",
            "g": "G — products",
            "h": "H — group context",
        }.get(fam, fam)
        lines += [f"### Family {title}", "", md_table(_stats_display(st, fam)), ""]

    lines += [
        "## 3. Correlation clusters (`|ρ|` cut 0.80)",
        "",
        f"{len(clus) if not clus.empty else 0} features entered the Spearman matrix; "
        f"{int(clus['cluster'].nunique()) if not clus.empty else 0} clusters; "
        f"{n_multi} clusters have more than one member. One representative is kept per cluster "
        "(highest score: coverage, not-size, not-NZV, share/ratio names; `a_op_in` wins a size cluster).",
        "",
        md_table(clus_view) if len(clus_view) else "_Correlation matrix too small._",
        "",
        "### VIF on company-median representatives",
        "",
        (
            "VIF is computed on **train company medians** of *non-amount* cluster representatives "
            "with ≥ 50% company coverage, after standardising. Raw euro levels are left out on purpose: "
            "their median-scale VIF is 10³–10⁵ because size is one factor. VIF `> 10` among the "
            "remaining reps still wants a drop."
            if HAS_VIF
            else "statsmodels was not available — VIF skipped."
        ),
        "",
        md_table(vif_df) if len(vif_df) else "_VIF not computed (too few complete representatives)._ ",
        "",
        "## 4. Company clustering",
        "",
        "Input is a **small standardised set of company medians**, not the full 100-d panel. "
        "Raw level amounts and SIZE-flagged columns are excluded so the partition is a chance to "
        "find operating types rather than 'big vs small'.",
        "",
        f"Features used: {km_feats}.",
        "",
        (
            "Cluster sizes: " + ", ".join(f"C{k}={v}" for k, v in km.get("sizes", {}).items())
            if km.get("sizes")
            else ""
        ),
        "",
        "Crosstab vs log-inflow tertile (T1 = smallest third of company-median `a_op_in`):",
        "",
        md_table(size_ct_md) if len(size_ct_md) else "_No crosstab._",
        "",
        _cluster_size_note(km)
        + " If groups with ≥ 3 members are mostly mono-cluster, the types are group membership "
        "(the holdout is already group-aware, so a second group-cluster model would double-count that cut).",
        "",
        "## 5. How to use this in models",
        "",
        "- Always put **one** size control in the GBM: `log1p(a_in3)`. Do not also keep `a_op_in` / "
        "`a_in6` / `a_in12` / `a_out*` / `a_net`. Company-median VIF of raw euro levels is huge even "
        "when panel `|ρ|` is below 0.8 (`a_net = a_op_in − a_op_out`).",
        "- Prefer shares, ratios, overdue-30, runway, gap-sd, HHI over open-amount / issued-amount levels.",
        "- Change features (`a_growth_*`, `b_d_runway`) are allowed to have low lag-1 autocorr — that is "
        "the point; do not drop them for persistence alone. Javier already saw level persistence and "
        "change mean-reversion.",
        "- Family E DSO/DPO means are unusable (extreme invoices / tiny issued). If `e_dso_proxy` / "
        "`e_dpo_proxy` survive as representatives, winsorise at 24 months in the model layer; this "
        "report does not fit that clip on data.",
        "- Do not fit anything that produced this report on the 72 holdout companies.",
        "",
        "## 6. Plots",
        "",
        plots_md,
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines).replace("\n\n\n\n", "\n\n").strip() + "\n")
    print(f"wrote {REPORT_PATH}", flush=True)


def run() -> Battery:
    panel, loaded, skipped = load_panel()
    if "a_op_in" not in panel.columns:
        raise RuntimeError("Family A did not load — cannot form the log1p(|a_op_in|) size proxy")
    features = feature_columns(panel)
    print(f"numeric features: {len(features)}", flush=True)
    stats = per_feature_stats(panel, features)
    eligible = _eligible_for_corr(stats)
    print(f"spearman matrix on {len(eligible)} features", flush=True)
    rho = spearman_matrix(panel, eligible)
    cluster_rows, _z = cluster_features(rho, stats)
    reps = (
        cluster_rows.loc[cluster_rows["is_rep"], "feature"].tolist()
        if not cluster_rows.empty
        else []
    )
    vif = vif_on_reps(panel, reps)
    print("k-means on company medians…", flush=True)
    km = company_clustering(panel, stats, cluster_rows)
    bat = Battery(
        panel=panel,
        features=features,
        loaded=loaded,
        skipped=skipped,
        stats=stats,
        rho=rho,
        cluster_rows=cluster_rows,
        vif=vif,
        km=km,
        mpl_ok=HAS_MPL,
    )
    rec = recommend(stats, cluster_rows, vif)
    bat.plots = make_plots(bat)
    write_report(bat, rec)

    size = stats.dropna(subset=["size_rho"]).assign(a=lambda d: d["size_rho"].abs()).sort_values("a", ascending=False)
    flagged = size.loc[size["a"] > SIZE_RHO, "feature"].tolist()
    nzv = stats.loc[stats["flags"].str.contains("NZV|CONSTANT", na=False), "feature"].tolist()
    print("\n=== RETURN ===", flush=True)
    print("top_size_proxies:", ", ".join(flagged) if flagged else "(none > 0.85)")
    print("top_size_ranked:", ", ".join(size.head(8)["feature"].tolist()))
    print("near_zero_var:", ", ".join(nzv))
    print(
        f"cluster_silhouette: k={km.get('best_k')} sil={km.get('best_sil')} "
        f"cramer_size={km.get('cramer_size')} nmi_group={km.get('nmi_group')}"
    )
    print("drop:", ", ".join(sorted(rec["drop"])))
    return bat


def main() -> None:
    run()


if __name__ == "__main__":
    main()
