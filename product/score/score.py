"""Train-only percentile dummy score on the monthly feature store."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.evaluate.protocol import load_holdout
from analysis.features.common import DATA, ROOT
from product.score.card import (
    ACTIVE_WINDOW,
    CATEGORY_WEIGHTS,
    DARK_DAYS_CAP,
    DARK_LONG_CAP,
    DARK_LONG_DAYS,
    DIP_POINTS,
    ITEMS,
    MIN_ACTIVE_WINDOW,
    SCORE_STRETCH,
    SMOOTH_WINDOW,
    STORE_COLUMNS,
    THIN_TRAIL_MONTHS,
    VERSION,
    Item,
    format_reason,
)

OUT_DIR = ROOT / "product" / "score" / "outputs"
STORE_PATH = DATA / "feature_store" / "monthly.parquet"


def derive_columns(panel: pd.DataFrame) -> pd.DataFrame:
    """Add trail_months and active_share_6. No train fit."""
    out = panel.copy()
    out["company_id"] = out["company_id"].astype(str)
    out["period"] = pd.to_datetime(out["period"])
    first = pd.to_datetime(out["first_month"])
    out["trail_months"] = (
        (out["period"].dt.year - first.dt.year) * 12
        + (out["period"].dt.month - first.dt.month)
        + 1
    )
    days = pd.to_numeric(out["c_n_days_with_tx"], errors="coerce").fillna(0.0)
    active = (days > 0).astype(float)
    out["_active_month"] = active
    out["active_share_6"] = (
        out.groupby("company_id", sort=False)["_active_month"].transform(
            lambda s: s.rolling(ACTIVE_WINDOW, min_periods=MIN_ACTIVE_WINDOW).mean()
        )
    )
    out["going_dark"] = days.eq(0.0)
    recency = pd.to_numeric(out.get("c_recency_days"), errors="coerce")
    out["going_dark_long"] = out["going_dark"] & recency.gt(DARK_LONG_DAYS).fillna(False)
    out["thin_file"] = out["trail_months"] < THIN_TRAIL_MONTHS
    return out


def _finite(s: pd.Series) -> np.ndarray:
    v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    return v[np.isfinite(v)]


def fit_ref(panel: pd.DataFrame, holdout: set[str] | None = None) -> dict:
    """Empirical sorted values on train company-months only."""
    if holdout is None:
        holdout = load_holdout()
    train = panel.loc[~panel["company_id"].astype(str).isin(holdout)]
    leaked = set(train["company_id"]).intersection(holdout)
    if leaked:
        raise ValueError(f"holdout companies in percentile fit: {sorted(leaked)[:5]}")
    ref: dict = {
        "version": VERSION,
        "n_train_rows": int(len(train)),
        "n_train_companies": int(train["company_id"].nunique()),
        "signals": {},
    }
    for item in ITEMS:
        if item.kind != "percentile":
            continue
        if item.name not in train.columns:
            raise KeyError(f"missing column for fit: {item.name}")
        values = np.sort(_finite(train[item.name]))
        ref["signals"][item.name] = {
            "direction": item.direction,
            "n": int(values.size),
            "values": values.tolist(),
        }
    return ref


def _pct_score(values: np.ndarray, ref_values: np.ndarray, direction: int) -> np.ndarray:
    r = ref_values
    if r.size == 0:
        return np.full(values.shape, np.nan)
    left = np.searchsorted(r, values, "left")
    right = np.searchsorted(r, values, "right")
    pct = (left + right) / 2.0 / r.size
    return 100.0 * np.where(direction == 1, pct, 1.0 - pct)


def _threshold_score(values: np.ndarray, item: Item) -> np.ndarray:
    lo, hi = float(item.lo), float(item.hi)
    span = hi - lo
    raw = np.clip((hi - values) / span, 0.0, 1.0)
    if item.direction == -1:
        return 100.0 * raw
    return 100.0 * (1.0 - raw)


def item_scores(panel: pd.DataFrame, ref: dict) -> pd.DataFrame:
    rows = {"company_id": panel["company_id"].to_numpy(), "period": panel["period"].to_numpy()}
    for item in ITEMS:
        raw = pd.to_numeric(panel[item.name], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(raw)
        scored = np.full(raw.shape, np.nan)
        if item.kind == "percentile":
            stored = np.asarray(ref["signals"][item.name]["values"], dtype=float)
            scored[ok] = _pct_score(raw[ok], stored, item.direction)
        else:
            scored[ok] = _threshold_score(raw[ok], item)
        rows[item.name] = raw
        rows[f"{item.name}__pts"] = scored
    return pd.DataFrame(rows)


def _category_block(scores: pd.DataFrame) -> pd.DataFrame:
    out = scores[["company_id", "period"]].copy()
    for cat, weight in CATEGORY_WEIGHTS.items():
        cols = [f"{it.name}__pts" for it in ITEMS if it.category == cat]
        block = scores[cols]
        out[f"cat_{cat}"] = block.mean(axis=1, skipna=True)
        out[f"n_{cat}"] = block.notna().sum(axis=1)
        out[f"w_{cat}"] = np.where(out[f"n_{cat}"] > 0, weight, 0.0)
    wcols = [f"w_{c}" for c in CATEGORY_WEIGHTS]
    out["cov_w"] = out[wcols].sum(axis=1)
    out["w_sum"] = out["cov_w"].replace(0.0, np.nan)
    raw = np.zeros(len(out), dtype=float)
    for cat in CATEGORY_WEIGHTS:
        part = out[f"w_{cat}"] / out["w_sum"] * out[f"cat_{cat}"]
        raw = raw + np.where(out[f"w_{cat}"] > 0, part.to_numpy(dtype=float), 0.0)
    out["score_raw"] = np.where(out["cov_w"] > 0, raw, np.nan)
    out["score_pre_cap"] = np.clip(50.0 + SCORE_STRETCH * (out["score_raw"] - 50.0), 0.0, 100.0)
    return out


def _apply_dark_cap(panel: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    dark = panel["going_dark"].to_numpy()
    very = panel["going_dark_long"].to_numpy()
    pre = out["score_pre_cap"].to_numpy(dtype=float)
    capped = pre.copy()
    capped[dark] = np.minimum(capped[dark], DARK_DAYS_CAP)
    capped[very] = np.minimum(capped[very], DARK_LONG_CAP)
    out["score"] = capped
    out["dark_cap"] = dark | very
    return out


def _confidence(panel: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    pay = out["n_payment_history"]
    mix = out["n_mix"]
    out["no_invoices"] = (pay == 0) & (mix == 0)
    out["thin_file"] = panel["thin_file"].to_numpy()
    out["going_dark"] = panel["going_dark"].to_numpy()
    out["going_dark_long"] = panel["going_dark_long"].to_numpy()
    conf = out["cov_w"] / 100.0
    conf = np.where(out["thin_file"], conf * 0.8, conf)
    conf = np.where(out["going_dark"], np.minimum(conf, 0.55), conf)
    out["confidence"] = conf
    out["confidence_band"] = np.select(
        [conf >= 0.75, conf >= 0.40],
        ["high", "medium"],
        default="low",
    )
    return out


def _trajectory(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.sort_values(["company_id", "period"]).reset_index(drop=True)
    g = out.groupby("company_id", sort=False)["score"]
    out["score_3m"] = g.transform(
        lambda s: s.rolling(SMOOTH_WINDOW, min_periods=1).median()
    )
    d3 = out["score_3m"] - out.groupby("company_id", sort=False)["score_3m"].shift(3)
    out["delta3"] = d3
    out["state"] = np.select(
        [
            out["going_dark_long"],
            d3 >= DIP_POINTS,
            d3 <= -DIP_POINTS,
        ],
        ["deteriorating", "improving", "deteriorating"],
        default="stable",
    )
    out.loc[out["score"].isna(), "state"] = None
    return out


def _reasons(scored: pd.DataFrame) -> pd.DataFrame:
    """Weakest items by how far they sit below 50, weighted by category."""
    recs = []
    has_recency = "c_recency_days" in scored.columns
    for _, row in scored.iterrows():
        cands = []
        if bool(row.get("going_dark_long")):
            recency = row["c_recency_days"] if has_recency else np.nan
            recency_f = float(recency) if pd.notna(recency) else float(DARK_LONG_DAYS)
            cands.append(
                (
                    100.0,
                    "going_dark_long",
                    f"no bank movements for {recency_f:.0f} days",
                    f"sin movimientos bancarios desde hace {recency_f:.0f} días",
                )
            )
        elif bool(row.get("going_dark")):
            cands.append(
                (
                    80.0,
                    "going_dark",
                    "no bank movements this month",
                    "sin movimientos bancarios este mes",
                )
            )
        for item in ITEMS:
            pts = row.get(f"{item.name}__pts")
            if not np.isfinite(pts):
                continue
            drag = max(0.0, 50.0 - float(pts)) * (CATEGORY_WEIGHTS[item.category] / 100.0)
            if drag <= 0:
                continue
            raw = row.get(item.name)
            raw_f = float(raw) if pd.notna(raw) else float("nan")
            cands.append(
                (
                    drag,
                    item.name,
                    format_reason(item, raw_f, "en"),
                    format_reason(item, raw_f, "es"),
                )
            )
        if not cands:
            best = []
            for item in ITEMS:
                pts = row.get(f"{item.name}__pts")
                if not np.isfinite(pts):
                    continue
                raw = row.get(item.name)
                raw_f = float(raw) if pd.notna(raw) else float("nan")
                best.append((float(pts), item, raw_f))
            best.sort(key=lambda x: -x[0])
            for pts, item, raw_f in best[:3]:
                cands.append(
                    (
                        pts,
                        item.name,
                        format_reason(item, raw_f, "en"),
                        format_reason(item, raw_f, "es"),
                    )
                )
        cands.sort(key=lambda x: -x[0])
        rec = {}
        for k in range(3):
            if k < len(cands):
                rec[f"reason_{k+1}_code"] = cands[k][1]
                rec[f"reason_{k+1}_en"] = cands[k][2]
                rec[f"reason_{k+1}_es"] = cands[k][3]
            else:
                rec[f"reason_{k+1}_code"] = None
                rec[f"reason_{k+1}_en"] = None
                rec[f"reason_{k+1}_es"] = None
        recs.append(rec)
    reasons = pd.DataFrame(recs, index=scored.index)
    return pd.concat([scored, reasons], axis=1)


def score_panel(panel: pd.DataFrame, ref: dict) -> pd.DataFrame:
    derived = derive_columns(panel).reset_index(drop=True)
    items = item_scores(derived, ref)
    for extra in ("going_dark", "going_dark_long", "thin_file", "trail_months", "c_recency_days"):
        if extra in derived.columns:
            items[extra] = derived[extra].to_numpy()
    if "group_id" in derived.columns:
        items["group_id"] = derived["group_id"].to_numpy()
    block = _category_block(items)
    merged = items.drop(columns=["company_id", "period"]).copy()
    merged = pd.concat([block, merged], axis=1)
    merged = _apply_dark_cap(derived, merged)
    merged = _confidence(derived, merged)
    merged = _trajectory(merged)
    merged = _reasons(merged)
    merged["score_version"] = VERSION
    front = [
        "company_id",
        "period",
        "score",
        "score_3m",
        "score_raw",
        "score_pre_cap",
        "state",
        "delta3",
        "confidence",
        "confidence_band",
        "cov_w",
        "going_dark",
        "going_dark_long",
        "thin_file",
        "no_invoices",
        "dark_cap",
        "trail_months",
        "score_version",
    ]
    if "group_id" in merged.columns:
        front.insert(2, "group_id")
    rest = [c for c in merged.columns if c not in front]
    return merged[front + rest]


def load_store(path: Path | None = None) -> pd.DataFrame:
    path = path or STORE_PATH
    if not path.exists():
        from analysis.features.build_feature_store import main as build_store

        build_store()
    panel = pd.read_parquet(path)
    missing = [c for c in STORE_COLUMNS if c not in panel.columns]
    if missing:
        raise KeyError(f"feature store missing {missing}")
    return panel


def save_ref(ref: dict, path: Path | None = None) -> Path:
    path = path or (OUT_DIR / "ref_v0.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ref) + "\n", encoding="utf-8")
    return path


def load_ref(path: Path | None = None) -> dict:
    path = path or (OUT_DIR / "ref_v0.json")
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(scored: pd.DataFrame, holdout: set[str]) -> dict:
    train = scored.loc[~scored["company_id"].isin(holdout)]
    latest = train.sort_values("period").groupby("company_id", as_index=False).tail(1)
    s = latest["score_3m"].dropna()
    return {
        "version": VERSION,
        "train_rows": int(len(train)),
        "train_companies": int(train["company_id"].nunique()),
        "latest_n": int(len(s)),
        "latest_p10": float(s.quantile(0.10)) if len(s) else None,
        "latest_p50": float(s.quantile(0.50)) if len(s) else None,
        "latest_p90": float(s.quantile(0.90)) if len(s) else None,
        "share_no_invoices": float(latest["no_invoices"].mean()) if len(latest) else None,
        "share_going_dark": float(latest["going_dark"].mean()) if len(latest) else None,
        "share_thin": float(latest["thin_file"].mean()) if len(latest) else None,
        "state_counts": latest["state"].value_counts(dropna=False).to_dict(),
        "confidence_counts": latest["confidence_band"].value_counts(dropna=False).to_dict(),
    }
