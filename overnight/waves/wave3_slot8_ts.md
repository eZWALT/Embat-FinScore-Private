# Wave 3 slot 8 — weekly per-company TS + exog (Y1 net)

- **When:** 2026-09-19 ~00:07 CEST
- **Python:** `/home/walterjtv/.pyenv/versions/base/bin/python3` (statsmodels 0.14; Prophet not installed, skipped)
- **Holdout:** `analysis/splits/holdout_companies.csv` (72 companies). Never sampled, never fitted.

## Files written

- `analysis/models/ts_per_company.py`
- `overnight/waves/wave3_slot8_ts.md` (this file)
- 24 rows appended to `analysis/experiments/registry.csv` (`y1_weekly_net_h4` / `h8`)

Did not edit Y5 or `protocol.py`. Did not fit on holdout.

## Setup

Weekly **operational net** (`op_in − op_out`, same groups as Family A / Y1). Dense `W-MON` grid after first activity; silent weeks = 0.

`weekly_grid` includes 2026-08-31, but that week is a **1-day stub** (AS_OF 2026-09-01). Dropped it. Complete calendar: 104 weeks, 2024-09-02 .. 2026-08-24.

- Origin (last train week): **2026-06-29**
- Test: **2026-07-06 .. 2026-08-24** (8 weeks; h=4 is the first four)
- Eligible: train companies with ≥40 weeks that have a transaction at or before origin → **750 / 1214**
- Sample: **40** companies, `numpy` seed **20260918**
- Scale: company **train** mean `|net|`. Metric = MAE / that scale
- Seasonal naive: m=52 if train length ≥52 else m=4 (37 used 52, 3 used 4)
- Exog (known in advance): invoices `document_type=invoice`, not cancel, due that week, **issued by min(week-end, origin-end)**; plus calendar month-end dummy. Per-company z-score on train; zero-variance columns dropped
- **Debt schedule skipped:** 87 rows / 40 companies in `debt_schedule_config`; only a snapshot `next_payment_date`, not a weekly installment path. Probe: 22 of the (pre-stub) eligible set
- Median train length in the sample: 91.5 weeks (min 44, max 96). 25/40 companies had any invoice-due mass

## Results — n fitted = 40 / 40 (all models, both horizons)

Median of per-company (MAE / mean|net|). Win-rate = share of companies with model MAE **strictly** below hist-mean MAE.

| horizon | model | median norm MAE | win-rate vs mean | vs mean median |
|---------|-------|----------------:|-----------------:|----------------|
| 4 | last_value | 1.327 | 37.5% | lose |
| 4 | **hist_mean** | **1.075** | — | — |
| 4 | seas_naive | 1.409 | 32.5% | lose |
| 4 | ets | 1.060 | 65.0% | −1.3% (near-copy of the mean) |
| 4 | sarimax100 | 1.000 | 40.0% | −7.0% median, but loses 24/40 pairwise |
| 4 | sarimax100_exog | 1.072 | 35.0% | −0.2% (noise) |
| 8 | last_value | 1.344 | 32.5% | lose |
| 8 | **hist_mean** | **0.857** | — | — |
| 8 | seas_naive | 1.103 | 20.0% | lose |
| 8 | ets | 0.857 | 55.0% | **tie** (median identical) |
| 8 | sarimax100 | 0.868 | 35.0% | lose |
| 8 | sarimax100_exog | 0.920 | 35.0% | lose |

## Verdict: do not claim victory

Weekly TS does **not** beat the historical mean in a way that should change the stack.

- **h=8:** hist mean is best. ETS median is a numerical tie (12.5% of companies have identical ETS and mean MAE; median pairwise gap is 0). SARIMAX and SARIMAX+exog both lose.
- **h=4:** SARIMAX(1,0,0) has the lowest median (1.000 vs 1.075) but **wins only 16/40** companies. Median pairwise gap is *+0.006* (typical company is slightly worse); the median-of-MAE “win” is a right-tail effect (one company −0.42). ETS median is 1.3% better with 65% win-rate, but corr(ETS, mean) = 0.98 and 42% of companies are within 1% of the mean — damped Holt is the mean with a small trend.
- **Exog invoices-due + month-end made SARIMAX worse** than the univariate AR(1) on both horizons (h=4: 1.072 vs 1.000; h=8: 0.920 vs 0.868). Same pattern as the monthly check (invoices-due hurt). 15/40 companies had only the month-end dummy (no invoice-due variance).
- Last value and seasonal naive both lose. Weekly net is noisy; a year-ago week is not a useful twin.

This is **not** a repeat of “monthly SARIMAX wins.” Monthly already lost (median norm MAE 0.58–0.94 vs mean 0.49). Weekly is closer, still not a reason to keep per-company ETS/SARIMAX as a Y1 workhorse.

## What failed

- Prophet missing — skipped, no install attempted
- Debt schedule too sparse / not a weekly series — skipped
- Seasonal naive-52 (and fallback-4) worse than the mean
- Known-in-advance invoice due amounts did not help (and made AR(1) worse)
- No material win at h=8, which is the more relevant cash-path horizon

## Next idea

Park per-company weekly SARIMAX/ETS for **net**. Two failures in a row on this family (monthly loss, weekly not a real win). If Y1 continues: try **inflow** (less cancellation noise than net), or a **pooled** model, or stop at the hist-mean baseline for the point forecast and spend the slot on GBM / Y2–Y5. Do not add Prophet until the mean is actually beaten.
