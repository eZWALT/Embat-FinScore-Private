"""Behaviour clusters (plan step 3): group companies by how their money moves, not by how big they are. Fit on train.

Company-level behaviour vector from the feature store (ratios and shares, then each feature is regressed on log
inflow and only the residual is kept, so the clusters carry no linear size signal): volatility of inflows and
outflows, months without incoming money, activity density and regularity, payroll/tax/social-security presence,
debt service and fee shares, coverage of outflows by inflows and, for companies with invoices, collection delay,
overdue share, customer concentration and credit notes. Invoice features of companies without invoices are set to the
train median (neutral) and `has_invoices` is a feature, so bank-only companies land together.

The vector uses the company's whole available trail, so membership is descriptive (a trait), not an as-of signal:
it is used for the "vs cluster" comparison and never triggers an alert. Companies with fewer than MIN_MONTHS months
of trail get no cluster.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_MONTHS = 6
SEED = 20260919
K_RANGE = range(4, 9)

# feature -> (source description, phrase when high, phrase when low)
FEATURES = {
    "in_cv": ("variation of monthly inflows", "lumpy inflows", "steady inflows"),
    "out_cv": ("variation of monthly outflows", "lumpy outflows", "steady outflows"),
    "zero_in": ("share of months without incoming money", "months without incoming money", "money coming in every month"),
    "days": ("median days with movements per month", "movements almost daily", "few days with movements"),
    "gap_sd": ("spread of gaps between movements", "irregular timing", "regular timing"),
    "salary": ("share of months with payroll", "runs payroll", "no payroll"),
    "tax": ("share of months with tax payments", "regular tax payments", "few tax payments"),
    "ss": ("share of months with social security", "pays social security", "no social security"),
    "ds": ("mean debt service / inflows", "heavy debt service", "little debt service"),
    "fc": ("mean bank fees and interest / inflows", "high bank costs", "low bank costs"),
    "margin": ("median net margin", "positive operating margin", "negative operating margin"),
    "io": ("median inflow / outflow coverage", "inflows well above outflows", "outflows above inflows"),
    "has_invoices": ("has invoice data", "invoice-based", "bank-only"),
    "delay_coll": ("median collection delay", "slow-paying customers", "fast-paying customers"),
    "ar_od30": ("median share of receivables > 30 days overdue", "many overdue receivables", "few overdue receivables"),
    "hhi": ("median customer concentration", "dependent on few customers", "diversified customers"),
    "credit_note": ("median credit-note share", "many credit notes", "few credit notes"),
}
INVOICE_FEATURES = ["delay_coll", "ar_od30", "hhi", "credit_note"]
STORE_COLS = ["company_id", "period", "a_op_in", "a_op_out", "a_io_ratio", "a_net_margin", "c_zero_in_month", "c_n_days_with_tx",
              "c_gap_sd", "c_salary_month", "c_tax_month", "c_ss_month", "f_ds_r", "f_fc_r", "e_delay_coll", "e_ar_overdue_30",
              "d_cust_hhi", "e_credit_note_ratio", "e_ar_issued"]


def _cv(s: pd.Series) -> float:
    s = s.dropna()
    return float(min(3.0, s.std() / max(s.mean(), 1.0))) if len(s) >= 3 else np.nan


def behaviour_table(store: pd.DataFrame) -> pd.DataFrame:
    """One row per company (index company_id): FEATURES columns plus n_months and log_inflow (diagnostic only)."""
    s = store[STORE_COLS].sort_values(["company_id", "period"])
    g = s.groupby("company_id", sort=True)
    t = pd.DataFrame({
        "n_months": g.size(),
        "in_cv": g["a_op_in"].apply(_cv), "out_cv": g["a_op_out"].apply(_cv),
        "zero_in": g["c_zero_in_month"].mean(), "days": g["c_n_days_with_tx"].median(), "gap_sd": g["c_gap_sd"].median(),
        "salary": g["c_salary_month"].mean(), "tax": g["c_tax_month"].mean(), "ss": g["c_ss_month"].mean(),
        "ds": g["f_ds_r"].mean().clip(upper=1.0), "fc": g["f_fc_r"].mean(),
        "margin": g["a_net_margin"].median(), "io": g["a_io_ratio"].median(),
        "has_invoices": g["e_ar_issued"].apply(lambda x: float((x > 0).any())),
        "delay_coll": g["e_delay_coll"].median(), "ar_od30": g["e_ar_overdue_30"].median(),
        "hhi": g["d_cust_hhi"].median(), "credit_note": g["e_credit_note_ratio"].median(),
        "log_inflow": g["a_op_in"].apply(lambda x: float(np.log1p(max(x.mean(), 0.0)))),
    })
    return t


class Clusterer:
    """Winsorise, standardise, k-means. All parameters come from the train fit."""

    def __init__(self, params: dict):
        self.p = params

    @staticmethod
    def fit(table: pd.DataFrame, k: int | None = None) -> dict:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        t = table[table["n_months"] >= MIN_MONTHS].copy()
        feats = list(FEATURES)
        t.loc[t["has_invoices"] == 0, INVOICE_FEATURES] = np.nan
        lo, hi = t[feats].quantile(0.01), t[feats].quantile(0.99)
        w = t[feats].clip(lo, hi, axis=1)
        med = w.median()
        w = w.fillna(med)
        size = t["log_inflow"]
        coef = {}
        for f in feats:  # remove the linear size signal from every feature
            b = float(np.cov(size, w[f])[0, 1] / size.var())
            a = float(w[f].mean() - b * size.mean())
            coef[f] = (a, b)
            w[f] = w[f] - (a + b * size)
        mu, sd = w.mean(), w.std().replace(0, 1.0)
        X = ((w - mu) / sd).to_numpy()
        sil = {}
        rng = np.random.default_rng(SEED)
        sample = rng.choice(len(X), min(len(X), 1500), replace=False)
        for kk in K_RANGE:
            lab = KMeans(kk, n_init=10, random_state=SEED).fit_predict(X)
            sil[kk] = float(silhouette_score(X[sample], lab[sample]))
        k = k or max(sil, key=sil.get)
        km = KMeans(k, n_init=30, random_state=SEED).fit(X)
        order = np.argsort(-np.bincount(km.labels_))          # cluster ids by size, largest first
        remap = {int(old): new for new, old in enumerate(order)}
        cent = km.cluster_centers_[order]
        labels = pd.Series([remap[int(l)] for l in km.labels_], index=t.index)
        names, descs = [], []
        for c in range(k):
            z = pd.Series(cent[c], index=feats)
            top = z.abs().sort_values(ascending=False).index[:4]
            phr = [FEATURES[f][1] if z[f] > 0 else FEATURES[f][2] for f in top]
            names.append(" · ".join(phr[:2]))
            descs.append("; ".join(phr))
        return {
            "k": int(k), "silhouette_by_k": {str(a): round(b, 4) for a, b in sil.items()}, "features": feats,
            "winsor_lo": lo.tolist(), "winsor_hi": hi.tolist(), "impute": med.tolist(), "size_a": [coef[f][0] for f in feats],
            "size_b": [coef[f][1] for f in feats], "size_mean": float(size.mean()), "mean": mu.tolist(), "std": sd.tolist(),
            "centroids": cent.tolist(), "labels": names, "descriptions": descs,
            "sizes": np.bincount(labels, minlength=k).tolist(), "train_assignment": labels,
        }

    def assign(self, table: pd.DataFrame) -> pd.Series:
        """Cluster id per company (NaN for trails shorter than MIN_MONTHS)."""
        p = self.p
        feats = p["features"]
        t = table.copy()
        t.loc[t["has_invoices"] == 0, INVOICE_FEATURES] = np.nan
        w = t[feats].clip(pd.Series(p["winsor_lo"], index=feats), pd.Series(p["winsor_hi"], index=feats), axis=1)
        w = w.fillna(pd.Series(p["impute"], index=feats))
        a, b = pd.Series(p["size_a"], index=feats), pd.Series(p["size_b"], index=feats)
        w = w - (a.values[None, :] + np.outer(t["log_inflow"].to_numpy(), b.values))
        X = ((w - pd.Series(p["mean"], index=feats)) / pd.Series(p["std"], index=feats)).to_numpy()
        d = ((X[:, None, :] - np.asarray(p["centroids"])[None, :, :]) ** 2).sum(axis=2)
        out = pd.Series(d.argmin(axis=1).astype(float), index=t.index)
        out[t["n_months"] < MIN_MONTHS] = np.nan
        return out
