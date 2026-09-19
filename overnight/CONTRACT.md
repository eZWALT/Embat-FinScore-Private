# Overnight contract (do not break)

Code lives in `analysis/`. `overnight/` is status, logs, and the frozen holdout copy.

**Re-read `overnight/NORTH_STAR.md` before every task.** That is the X Ray brief
working copy (six questions, trajectory not snapshot, hidden 72, no product tonight).

Tonight's scope: analysis, feature engineering, explainability. Do not touch
`product/`. Do not write a 0–100 formula.

## Hard rules

- Python only. Read `data/embat.duckdb` **read_only**. Schema `clean` first (`analysis.features.common.connect`).
- No look-ahead: a feature for `period` uses only events with date <= period end (month-end or week-end).
- Holdout `analysis/splits/holdout_companies.csv` (same file in `overnight/splits/`) is never used to fit percentiles, bins, models, or cluster centroids.
- **Y is never built from the same columns as the X allowed for the model that predicts it.**
- Do not write a 0–100 score formula. Do not touch `product/`. Do not commit parquet / duckdb.
- One file owner per wave (see `overnight/ORCHESTRATION.md`). Do not edit another agent's files.
- After your module works, write `overnight/waves/<wave>_<slot>.md` with: files written, columns, coverage on train, what failed, next idea.

## Feature module API

```python
SOURCE_TABLES = ["transactions"]  # tables you actually query
FAMILY = "a"

def build(con, grid: pd.DataFrame) -> pd.DataFrame:
    # grid columns: company_id, period (Timestamp, month-start or week-start)
    # return company_id, period + columns prefixed a_ / b_ / ...
```

## Target module API

```python
META = {
    "name": "y2_neg_balance_2of3",
    "horizon": 3,
    "source_tables": ["balances", "transactions"],
    "forbidden_x_families": ["b"],
    "literature": "FinRegLab 2025 low/negative ending balances",
}

def build(con, grid: pd.DataFrame) -> pd.DataFrame:
    # return company_id, period, y2_...
```

## Acceptance for a binary Y (before any model)

- Base rate on **train** company-months in 5–30%.
- AUROC of `log1p(|op_in|)` (or log size) vs Y < 0.60.
- Not identical to any allowed X column.
- Sustained window (>= 3 months or 6-month aggregate). No single-month 40% inflow crash.

## Verification the orchestrator will run

```bash
python -c "from analysis.features import monthly_grid, connect; print(monthly_grid(connect()).shape)"
python -m analysis.features.build_feature_store
```
