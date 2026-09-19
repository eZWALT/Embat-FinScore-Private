"""Per-company weekly net-flow bake-off (Y1 path).

Sample of 40 train companies with >=40 activity weeks. Forecast the next
4 and 8 complete ISO weeks of operational net (op_in - op_out).

Baselines: last value, historical mean, seasonal naive (m=52 if the train
series is at least 52 weeks, else m=4).

Models: ETS (Holt / damped trend), SARIMAX(1,0,0), SARIMAX(1,0,0) +
known-in-advance exog (invoices due that week, calendar month-end dummy).
Debt schedule is skipped (too sparse). Prophet is skipped if missing.

Metric: MAE / company train-period mean |net|. Report median and win-rate
versus historical mean. Holdout companies never enter the sample or a fit.

Monthly SARIMAX already lost to the historical mean; this file is the
weekly retry, not a claim that monthly ARIMA works.
"""
from __future__ import annotations

import csv
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.evaluate.protocol import assert_no_holdout
from analysis.features.common import (
    ANALYSIS,
    AS_OF,
    CAT_MAP,
    WEEKS,
    connect,
    load_holdout,
)

try:
    from prophet import Prophet as _Prophet
except Exception:  # Prophet is optional; not installed in this env
    _Prophet = None

from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

REGISTRY = ANALYSIS / "experiments" / "registry.csv"
AGENT = "2b391abc"
SAMPLE_N = 40
SAMPLE_SEED = 20260918
MIN_ACTIVITY = 40
H_SHORT = 4
H_LONG = 8
HORIZONS = (H_SHORT, H_LONG)

_OP_IN = tuple(k for k, v in CAT_MAP.items() if v == "op_in")
_OP_OUT = tuple(k for k, v in CAT_MAP.items() if v == "op_out")

# Last W-MON in WEEKS is 2026-08-31: only Monday exists before AS_OF 2026-09-01.
COMPLETE_WEEKS = WEEKS[(WEEKS + pd.Timedelta(days=6)) < AS_OF]
ORIGIN = COMPLETE_WEEKS[-1 - H_LONG]
TEST_WEEKS = COMPLETE_WEEKS[-H_LONG:]

BASELINES = ("last_value", "hist_mean", "seas_naive")
FITTED_MODELS = ("ets", "sarimax100", "sarimax100_exog")
ALL_MODELS = BASELINES + FITTED_MODELS
if _Prophet is not None:
    ALL_MODELS = ALL_MODELS + ("prophet",)


def _sql_in(values: tuple[str, ...]) -> str:
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


def _week_ends(weeks: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(weeks, index=weeks) + pd.Timedelta(days=6)


def month_end_dummy(weeks: pd.DatetimeIndex) -> pd.Series:
    """1 if the Mon–Sun week contains a calendar month-end."""
    flags = []
    for w in weeks:
        flags.append(int(any((w + pd.Timedelta(days=i)).is_month_end for i in range(7))))
    return pd.Series(flags, index=weeks, dtype=float, name="month_end")


def seasonal_naive(y: np.ndarray, h: int) -> tuple[np.ndarray, int]:
    """Hyndman seasonal naive: ŷ_{T+h} = y_{T+h-m(k+1)}, k=floor((h-1)/m)."""
    y = np.asarray(y, dtype=float)
    t = len(y)
    m = 52 if t >= 52 else 4
    out = np.empty(h, dtype=float)
    for step in range(1, h + 1):
        k = (step - 1) // m
        idx = t + step - m * (k + 1) - 1
        out[step - 1] = y[idx] if 0 <= idx < t else y[-1]
    return out, m


def last_value(y: np.ndarray, h: int) -> np.ndarray:
    return np.full(h, float(y[-1]), dtype=float)


def hist_mean(y: np.ndarray, h: int) -> np.ndarray:
    return np.full(h, float(np.mean(y)), dtype=float)


def norm_mae(y_true: np.ndarray, yhat: np.ndarray, scale: float) -> float:
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(yhat, dtype=float)
    if yt.size == 0 or yt.size != yp.size:
        return float("nan")
    if not np.isfinite(yt).all() or not np.isfinite(yp).all():
        return float("nan")
    return float(np.mean(np.abs(yt - yp)) / scale)


def load_weekly_ops(con) -> pd.DataFrame:
    """Observed company × ISO-week operational flows (no zero fill)."""
    df = con.execute(
        f"""
        SELECT
          t.company_id,
          CAST(date_trunc('week', t."date") AS DATE) AS period,
          SUM(CASE WHEN t.category IN ({_sql_in(_OP_IN)}) THEN t.amount ELSE 0 END) AS op_in,
          -SUM(CASE WHEN t.category IN ({_sql_in(_OP_OUT)}) THEN t.amount ELSE 0 END) AS op_out,
          COUNT(*) AS n_tx
        FROM transactions t
        WHERE t."date" IS NOT NULL
          AND t."date" < TIMESTAMP '{AS_OF.date()}'
        GROUP BY 1, 2
        """
    ).df()
    df["company_id"] = df["company_id"].astype(str)
    df["period"] = pd.to_datetime(df["period"])
    df["net"] = df["op_in"] - df["op_out"]
    return df


def eligible_train_ids(ops: pd.DataFrame, hold: set[str]) -> list[str]:
    """Train companies with >= MIN_ACTIVITY weeks that have a tx at or before origin."""
    cal = set(COMPLETE_WEEKS)
    d = ops[(ops["period"].isin(cal)) & (ops["period"] <= ORIGIN)]
    n_act = d.groupby("company_id").size()
    ids = [i for i in n_act.index.astype(str) if i not in hold and int(n_act[i]) >= MIN_ACTIVITY]
    return sorted(ids)


def pick_sample(eligible: list[str], n: int, seed: int) -> list[str]:
    if len(eligible) < n:
        raise RuntimeError(f"only {len(eligible)} eligible train companies, need {n}")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.array(eligible, dtype=object), size=n, replace=False)
    return sorted(str(x) for x in chosen)


def dense_company_net(ops: pd.DataFrame, company_id: str) -> pd.Series:
    """Weekly net from first activity week on the complete calendar; silent weeks = 0."""
    raw = ops.loc[ops["company_id"] == company_id, ["period", "net"]].copy()
    raw = raw[raw["period"].isin(COMPLETE_WEEKS)].sort_values("period")
    if raw.empty:
        return pd.Series(dtype=float)
    first = raw["period"].min()
    idx = COMPLETE_WEEKS[COMPLETE_WEEKS >= first]
    s = raw.drop_duplicates("period").set_index("period")["net"].reindex(idx).fillna(0.0)
    s.index = pd.DatetimeIndex(s.index, freq="W-MON")
    return s.astype(float)


def load_invoice_dues(con, company_ids: list[str]) -> pd.DataFrame:
    """Invoice rows needed to build known-in-advance weekly due amounts."""
    if not company_ids:
        return pd.DataFrame(columns=["company_id", "due_week", "issuance_date", "amount"])
    con.register("_ts_sample_cos", pd.DataFrame({"company_id": company_ids}))
    df = con.execute(
        """
        SELECT i.company_id,
               CAST(date_trunc('week', i.due_date) AS DATE) AS due_week,
               CAST(i.issuance_date AS DATE) AS issuance_date,
               i.amount
        FROM invoices i
        JOIN _ts_sample_cos s ON i.company_id = s.company_id
        WHERE i.document_type = 'invoice'
          AND i.status <> 'cancel'
          AND i.amount <> 0
          AND i.due_date IS NOT NULL
          AND i.issuance_date IS NOT NULL
        """
    ).df()
    df["company_id"] = df["company_id"].astype(str)
    df["due_week"] = pd.to_datetime(df["due_week"])
    df["issuance_date"] = pd.to_datetime(df["issuance_date"])
    return df


def weekly_invoice_exog(
    inv: pd.DataFrame,
    company_id: str,
    weeks: pd.DatetimeIndex,
    known_asof: pd.Timestamp,
) -> pd.DataFrame:
    """AR/AP amounts due in each week, using only invoices issued by known_asof.

    known_asof is the week-end of the last train week (origin). Training weeks
    further restrict issuance to that week's own end so a feature at t never
    uses documents issued after t.
    """
    ends = _week_ends(weeks)
    raw = inv.loc[inv["company_id"] == company_id]
    rows = []
    for w, end in zip(weeks, ends):
        asof = min(pd.Timestamp(end), pd.Timestamp(known_asof))
        part = raw[(raw["due_week"] == w) & (raw["issuance_date"] <= asof)]
        ar = float(part.loc[part["amount"] > 0, "amount"].sum()) if not part.empty else 0.0
        ap = float((-part.loc[part["amount"] < 0, "amount"]).sum()) if not part.empty else 0.0
        rows.append((w, ar, ap))
    out = pd.DataFrame(rows, columns=["period", "inv_due_ar", "inv_due_ap"]).set_index("period")
    out["month_end"] = month_end_dummy(weeks).to_numpy()
    return out


def _drop_zero_var(train: pd.DataFrame, fut: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep = [c for c in train.columns if float(train[c].std(ddof=0)) > 1e-12]
    return train[keep].copy(), fut[keep].copy()


def _zscore(train: pd.DataFrame, fut: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mu = train.mean(axis=0)
    sd = train.std(axis=0, ddof=0).replace(0.0, 1.0)
    return ((train - mu) / sd).to_numpy(dtype=float), ((fut - mu) / sd).to_numpy(dtype=float)


def fit_ets(y: np.ndarray, h: int) -> np.ndarray | None:
    specs = (
        dict(trend="add", damped_trend=True, seasonal=None, initialization_method="estimated"),
        dict(trend="add", damped_trend=False, seasonal=None, initialization_method="estimated"),
        dict(trend=None, damped_trend=False, seasonal=None, initialization_method="legacy-heuristic"),
    )
    for kw in specs:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ExponentialSmoothing(y, **kw).fit(optimized=True, remove_bias=False)
                fc = np.asarray(fit.forecast(h), dtype=float)
            if fc.shape == (h,) and np.isfinite(fc).all():
                return fc
        except Exception:
            continue
    return None


def fit_sarimax(
    y: np.ndarray,
    h: int,
    exog_train: np.ndarray | None = None,
    exog_fut: np.ndarray | None = None,
) -> np.ndarray | None:
    if exog_train is not None and (exog_train.size == 0 or exog_train.shape[1] == 0):
        exog_train, exog_fut = None, None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore")
            mod = SARIMAX(
                y,
                order=(1, 0, 0),
                trend="c",
                exog=exog_train,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = mod.fit(disp=False, maxiter=80, method="lbfgs")
            fc = np.asarray(fit.forecast(h, exog=exog_fut), dtype=float)
        if fc.shape == (h,) and np.isfinite(fc).all():
            return fc
    except Exception:
        return None
    return None


def fit_prophet(index: pd.DatetimeIndex, y: np.ndarray, h: int, future_idx: pd.DatetimeIndex) -> np.ndarray | None:
    if _Prophet is None:
        return None
    try:
        df = pd.DataFrame({"ds": index, "y": y})
        m = _Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(df)
        fut = pd.DataFrame({"ds": future_idx[:h]})
        fc = m.predict(fut)["yhat"].to_numpy(dtype=float)
        if fc.shape == (h,) and np.isfinite(fc).all():
            return fc
    except Exception:
        return None
    return None


def _forecasts_for_company(
    series: pd.Series,
    exog: pd.DataFrame,
) -> tuple[dict[str, np.ndarray | None], int, float, dict]:
    train = series.loc[series.index <= ORIGIN]
    test = series.reindex(TEST_WEEKS)
    y = train.to_numpy(dtype=float)
    y_test = test.to_numpy(dtype=float)
    scale = float(np.mean(np.abs(y))) if len(y) else float("nan")
    info = {
        "n_train": int(len(y)),
        "n_test": int(y_test.size),
        "scale": scale,
        "seas_m": 52 if len(y) >= 52 else 4,
        "n_exog": 0,
        "exog_cols": "",
    }
    out: dict[str, np.ndarray | None] = {}
    out["last_value"] = last_value(y, H_LONG)
    out["hist_mean"] = hist_mean(y, H_LONG)
    seas, m = seasonal_naive(y, H_LONG)
    out["seas_naive"] = seas
    info["seas_m"] = m

    out["ets"] = fit_ets(y, H_LONG)
    out["sarimax100"] = fit_sarimax(y, H_LONG)

    ex_tr = exog.loc[train.index]
    ex_te = exog.reindex(TEST_WEEKS)
    ex_tr, ex_te = _drop_zero_var(ex_tr, ex_te)
    info["n_exog"] = int(ex_tr.shape[1])
    info["exog_cols"] = ",".join(ex_tr.columns.astype(str))
    if ex_tr.shape[1] == 0:
        out["sarimax100_exog"] = None
    else:
        z_tr, z_te = _zscore(ex_tr, ex_te)
        out["sarimax100_exog"] = fit_sarimax(y, H_LONG, z_tr, z_te)

    if _Prophet is not None:
        out["prophet"] = fit_prophet(train.index, y, H_LONG, TEST_WEEKS)
    return out, m, scale, info | {"y_test": y_test}


def append_registry(rows: list[dict]) -> None:
    if not rows:
        return
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    with REGISTRY.open("r", encoding="utf-8") as f:
        header = next(csv.reader(f))
    with REGISTRY.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def _summarise(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    recs = []
    for h in HORIZONS:
        sub_h = df[df["horizon"] == h]
        mean_mae = sub_h.loc[sub_h["model"] == "hist_mean"].set_index("company_id")["norm_mae"]
        for model in ALL_MODELS:
            part = sub_h[sub_h["model"] == model]
            ok = part["norm_mae"].notna()
            vals = part.loc[ok, "norm_mae"]
            n_fit = int(ok.sum())
            n_fail = int((~ok).sum())
            med = float(vals.median()) if n_fit else float("nan")
            aligned = part.loc[ok].set_index("company_id")["norm_mae"]
            both = aligned.index.intersection(mean_mae.dropna().index)
            if len(both):
                win = float((aligned.loc[both] < mean_mae.loc[both]).mean())
            else:
                win = float("nan")
            mean_med = float(mean_mae.median()) if len(mean_mae) else float("nan")
            # Strict improvement only; ULP ties (ETS h=8 == hist mean) are not wins.
            beats = bool(
                n_fit
                and np.isfinite(med)
                and np.isfinite(mean_med)
                and med < mean_med - 1e-9
            )
            recs.append(
                {
                    "horizon": h,
                    "model": model,
                    "n_fitted": n_fit,
                    "n_fail": n_fail,
                    "median_norm_mae": med,
                    "winrate_vs_mean": win,
                    "beats_mean_median": beats,
                }
            )
    return pd.DataFrame(recs)


def run(n_companies: int = SAMPLE_N, seed: int = SAMPLE_SEED, write_registry: bool = True) -> dict:
    hold = load_holdout()
    con = connect()
    print(
        f"complete_weeks={len(COMPLETE_WEEKS)} origin={ORIGIN.date()} "
        f"test={TEST_WEEKS[0].date()}..{TEST_WEEKS[-1].date()} "
        f"prophet={'yes' if _Prophet else 'skip'}"
    )
    ops = load_weekly_ops(con)
    eligible = eligible_train_ids(ops, hold)
    sample = pick_sample(eligible, n_companies, seed)
    assert_no_holdout(sample)
    print(f"eligible_train={len(eligible)} sample={len(sample)} holdout={len(hold)}")

    inv = load_invoice_dues(con, sample)
    origin_end = ORIGIN + pd.Timedelta(days=6)
    con.close()

    detail_rows: list[dict] = []
    company_meta: list[dict] = []
    n_inv_cos = 0
    for i, cid in enumerate(sample, 1):
        series = dense_company_net(ops, cid)
        train = series.loc[series.index <= ORIGIN]
        if len(train) < MIN_ACTIVITY:
            print(f"skip {cid}: train weeks {len(train)} < {MIN_ACTIVITY}")
            continue
        scale = float(np.mean(np.abs(train.to_numpy(dtype=float))))
        if not np.isfinite(scale) or scale <= 0:
            print(f"skip {cid}: train mean |net| is 0")
            continue
        weeks_all = series.index
        exog = weekly_invoice_exog(inv, cid, weeks_all, origin_end)
        if float(exog.loc[train.index, ["inv_due_ar", "inv_due_ap"]].abs().sum().sum()) > 0:
            n_inv_cos += 1
        fcs, seas_m, scale, info = _forecasts_for_company(series, exog)
        y_test = info["y_test"]
        company_meta.append(
            {
                "company_id": cid,
                "n_train": info["n_train"],
                "scale": scale,
                "seas_m": seas_m,
                "n_exog": info["n_exog"],
                "exog_cols": info["exog_cols"],
            }
        )
        for model in ALL_MODELS:
            yhat = fcs.get(model)
            for h in HORIZONS:
                mae = (
                    norm_mae(y_test[:h], yhat[:h], scale)
                    if yhat is not None
                    else float("nan")
                )
                detail_rows.append(
                    {
                        "company_id": cid,
                        "model": model,
                        "horizon": h,
                        "norm_mae": mae,
                    }
                )
        if i == 1 or i % 10 == 0:
            print(f"fitted {i}/{len(sample)} last={cid} n_train={info['n_train']} seas_m={seas_m}")

    detail = pd.DataFrame(detail_rows)
    summary = _summarise(detail_rows)
    meta = pd.DataFrame(company_meta)
    n_used = int(meta["company_id"].nunique()) if not meta.empty else 0
    print("n_companies_used", n_used, "with_invoice_due_exog", n_inv_cos)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    any_model_beats = bool(
        summary.loc[summary["model"].isin(FITTED_MODELS), "beats_mean_median"].any()
    )
    if any_model_beats:
        print("NOTE: at least one fitted model beats hist mean on median MAE.")
    else:
        print("NO WIN: no ETS/SARIMAX variant beats historical mean on median norm MAE.")

    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    reg: list[dict] = []
    for _, r in summary.iterrows():
        notes = (
            f"n_eligible={len(eligible)}; n_used={n_used}; "
            f"inv_due_cos={n_inv_cos}; origin={ORIGIN.date()}; "
            f"test={TEST_WEEKS[0].date()}..{TEST_WEEKS[-1].date()}; "
            f"drop_stub_2026-08-31; debt_schedule=skipped; "
            f"prophet={'fit' if _Prophet else 'skip'}; "
            f"beats_mean_median={int(r['beats_mean_median'])}; "
            f"n_fail={int(r['n_fail'])}"
        )
        xfam = "calendar+inv_due" if r["model"] == "sarimax100_exog" else "-"
        for metric, key in (("median_norm_mae", "median_norm_mae"), ("winrate_vs_mean", "winrate_vs_mean")):
            val = r[key]
            reg.append(
                {
                    "ts": ts,
                    "round": "R4",
                    "wave": 3,
                    "agent": AGENT,
                    "x_families": xfam,
                    "y": f"y1_weekly_net_h{int(r['horizon'])}",
                    "model": r["model"],
                    "split": f"train_sample{n_used}",
                    "metric": metric,
                    "value": f"{val:.6g}" if np.isfinite(val) else "",
                    "coverage": f"{(r['n_fitted'] / n_used):.4f}" if n_used else "",
                    "notes": notes,
                }
            )
    if write_registry:
        append_registry(reg)
        print(f"appended {len(reg)} registry rows")

    return {
        "eligible": len(eligible),
        "n_used": n_used,
        "n_inv_cos": n_inv_cos,
        "origin": str(ORIGIN.date()),
        "test_start": str(TEST_WEEKS[0].date()),
        "test_end": str(TEST_WEEKS[-1].date()),
        "summary": summary,
        "detail": detail,
        "meta": meta,
        "any_model_beats": any_model_beats,
        "prophet": bool(_Prophet),
        "registry_rows": len(reg) if write_registry else 0,
    }


if __name__ == "__main__":
    run()
