# analysis/ — signals, features, explainability

Find signals in the X Ray treasury trail. Brief working copy: `overnight/NORTH_STAR.md`
(six questions, trajectory not last-month, hidden 72). Re-read it before new work.
Morning briefing: `overnight/dashboards/MORNING_REPORT.md`.

Tonight: **analysis + features + SHAP/explainability only.** Do not put a 0–100
formula here. `product/` is frozen until the team opens goals 2 and 4 again.

The brief still applies, per company, per month: who is healthy / improving / turning; dip vs fall; what moved; how many months earlier it showed.

Hidden test: 60–80 companies must not be used to fit anything. Split when one exists.

## Local database (DuckDB)

The notebooks read `data/embat.duckdb`, built locally from the CSVs (not committed). CSVs can sit in `data/` or `data/raw/output/`.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r analysis/requirements.txt
python analysis/build_db.py        # ~10 s: loads the CSVs and builds the `clean` schema; --force to rebuild
```

Then open the `.Rmd` files in RStudio and Knit (R packages install on the first chunk). Nobody else may hold the database open for writing.

### Raw vs clean

The database has two schemas. `main` holds the CSVs untouched; `clean` holds the cleaned tables (same names) plus `clean.dq_log`, one row per cleaning rule with the rows affected. Rules live in `analysis/clean_db.py`.

- Rows are only dropped when they carry no information (amount = 0: 369 transactions, 1,183 invoices).
- Impossible values become `NULL` (transaction `value_date` far from the booking date, invoice `payment_date` before issue / in the future / year 2000-6913 / on unpaid invoices, `due_date` year 7025, sentinel balances, `exchange_rate <= 0`).
- Doubtful rows are kept and flagged, never dropped: `is_dup` (repeated content), `is_extreme` (|amount| >= 1e9), `product_known`, `payment_date_invalid`, `created_after_snapshot`, `outstanding_gt_granted`, `balance_sentinel`.
- Normalised: `category` ('-' and empty -> `uncategorized`), `status` (empty -> `unknown`), `companies.country` to ISO-2, the always-empty `balances.available` column dropped.

Both the Python pipeline and the R notebooks read `clean` first (`SET search_path = 'clean,main'`). For raw data use `main.<table>`.

From Python: `duckdb.connect("data/embat.duckdb", read_only=True)`; inspect the rules with `SELECT * FROM clean.dq_log`.

## Python pipeline (runs today)

`score_pipeline.py` is a Python port of the score in `02_score_salud_financiera.Rmd` (monthly signals -> percentile reference -> 5 pillars -> score -> trajectory/states -> alerts). It runs end to end in ~5 s and is what we use to test ideas quickly.

```bash
cd analysis
python validate_pipeline.py        # features, sign check, score stats, group-fold stability, proxy-event AUC, alert lead time
```

First findings (details in `.agents/persistent-memory/2026-09-18-2210-python-port-first-validation.md`): invoice sign convention confirmed; the naive "cash falls 40%" proxy event is mean reversion, not deterioration, so the current AUCs are not evidence of predictive value.

## Layout

```text
analysis/
├── README.md
├── build_db.py                    # CSV -> data/embat.duckdb (raw in main, cleaned in clean)
├── clean_db.py                    # cleaning rules -> clean schema + clean.dq_log
├── score_pipeline.py              # Python port of the score pipeline (reads clean)
├── validate_pipeline.py           # runs the pipeline and prints diagnostics
├── requirements.txt
├── eda.Rmd                        # EDA of the 8 tables
└── 02_score_salud_financiera.Rmd  # signals, score, trajectory, explanation, monitor, product; writes analysis/outputs/*.csv
```

## Next: feature store + Y catalogue

Plan: `.agents/persistent-memory/2026-09-18-2350-feature-store-and-y-plan.md`.
Night contract and waves: `overnight/CONTRACT.md`. Holdout: `analysis/splits/holdout_companies.csv`.

## Status

The two `.Rmd` notebooks are written but **not yet knitted** (R was not installed where they were authored); expect small fixes on first knit. The Python pipeline has been run end to end.
