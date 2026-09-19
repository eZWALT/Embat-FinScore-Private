# 2026-09-19-1503 — forecast rework: from a flat naive fan to a pooled quantile fan (supersedes the forecast part of `1830-monitor-and-forecast.md`)

- **Author:** Claude Code (Sonnet 5) for Javier Boix
- **Trigger:** Javier found the forecast module thin next to the rest of the tool (flat median only, no trend, no seasonality, no ARIMA-type models).
- **Scope:** `analysis/monitor/forecast.py` rewritten; bundle schema 1.1.0 -> **1.2.0** (additive); docs and two POC agent statements that would have become false. Step 5 (web) untouched: the contract types are the hand-off. Nothing committed.

## What was measured (train companies, group-fold CV, holdout untouched; `analysis/monitor/forecast_evaluation.md`)

- **The score is not a trending series.** Lag-1 autocorrelation of monthly changes -0.14; the score is pulled toward the company's own average and the portfolio level. Damped-trend Holt is 0.3-5% worse than the last value at 1-3 months and ties at 4-6. Per-company statsmodels ARIMA (220 sampled companies): 1-27% worse at 1-3 months; at 4-6 only the mean-reverting AR(1)+const gains (1-3.5%). 24 points per company are too few to fit per company.
- **No seasonality:** lag-12 autocorrelation -0.014 (CI -0.05 to +0.03); calendar-month means do not repeat in year 2 (r -0.21). Power is limited with 24 months: rules out a strong yearly pattern, not a weak one. Not shipped.
- **What works:** pooled linear quantile regression per horizon and quantile on 7 features (own deviation, level, 1- and 3-month moves, deviation from 6-month mean, own volatility, level^2) -> `reversion_quantile`. Pinball skill over the naive fan +3.3% (1 month) ... +15.9% (6), all CIs above zero, 50/80% coverage on target out of fold. The **median alone gains little** (MAE +1.0% at 3 months, +4.7% at 6): the value is the skewed, company-conditional fan.
- **Stricter check (also hold out later months; fit up to 2025-09, test held-out companies after):** +1.5% (1 month) to +11.6% (6), CIs above zero, but 80% interval covers 74-78%. **Quote this as the cautious reading**; it is in the manifest (`monitor.forecast.time_split`).
- **Moves do not simply reverse.** After a fall of 8+ points in 3 months (median -15): median next 3 months +0.3, only 28% recover at least half, mean +3.9 (minority of big reversals). My first read from the regression slope ("3-month moves mostly reverse") was mean-driven and wrong for the typical case; the report says so.

## Bundle (schema 1.2.0, additive; a 1.1.0 consumer still reads it)

- `Forecast`: per-point `method`, `own_average`, `drivers` (parts add up to the 3-month expected change, each with a plain-language `text`), `skill_by_horizon`; `skill_vs_naive` now means pinball skill at 3 months (was the smoothed level's MAE skill ~0). `manifest.monitor.forecast` carries method per horizon, seasonality, time-split check and move persistence.
- Full export validates (1,282 of 1,286 companies have a fan, alerts unchanged at 2,365); sample bundle regenerated (only schema version, manifest and forecast blocks changed).
- `poc/agent/tools.py` docstring and `poc/agent/prompts/product_context.md` said "method is naive_last / ties": updated so the agent does not repeat a false claim. That is the colleagues' area; the edit is limited to those two statements.

## Decisions

- Pure numpy at export time; statsmodels only for the optional `--arima` benchmark (already in `analysis/requirements.txt`, installed into the system Python here).
- Ship rule per horizon: reversion fan if its pinball-skill CI over the naive fan is above zero, else naive. Everything shipped at every horizon.
- No trend and no seasonal term: measured, not assumed.
- `analysis/monitor/forecast_test.py` (runnable module): no look-ahead in features, fans ordered inside 0-100, drivers add up, short trail -> None, gap -> fan, pull has the right sign.

## Still unknown / limits

- 24 months, one lap: the calendar and the reversion strength cannot be checked across cycles. Out-of-time coverage is under-nominal, so fans are slightly too narrow on later months.
- `level_centre` and the typical volatility are means/medians over all train companies, including the held-out fold in each CV split (two scalars; negligible, not removed).
- Companies with a gap in the trail fall back to the naive fan (none in train; would matter on messier data).
- Forecasting the score's drivers (items/EUR) or cash/runway was not attempted; the night's cash forecasts tied last value.
- `AGENTS.md` had an unrelated uncommitted line (`product/agent/` pointer) and `product/agent/` is untracked: not mine.
