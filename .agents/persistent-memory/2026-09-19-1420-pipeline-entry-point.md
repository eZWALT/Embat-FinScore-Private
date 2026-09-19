# 2026-09-19-1420 — plan step 1: one command from CSV folder to clean DB + feature store

- **Author:** Claude Code (Sonnet 5) for Javier Boix
- **When:** 2026-09-19 ~14:20 CEST

## What changed

- `analysis/pipeline.py` (new): `PYTHONUTF8=1 python -m analysis.pipeline --input <csv_folder> --work-dir <out>` -> `<out>/embat.duckdb` (`main` raw, `clean`, `clean.dq_log`) -> `<out>/feature_store/monthly.parquet` + `dq_log.csv` + `pipeline_run.json` (input sha256, clean row counts, dq counts, parquet sha256). ~17-35 s on the full data. Default work dir is `data/` (it would overwrite `data/embat.duckdb`); I have not run it there, all tests wrote to a scratch folder.
- `analysis/build_db.py`: refactored to `build(csv_dir, db_path)`; column types are now fixed (they equal what autodetect gave on train, so `main` is unchanged) instead of inferred; rows that do not parse are rejected and counted; missing/extra columns and missing optional files are logged instead of crashing or being ignored; key columns or required files missing -> clear error. Sort keys got the id as a tie-break.
- `analysis/clean_db.py`: kept all 26 rules; added `GUARDS` (detectors for dirt not seen on train) and a `source` column in `dq_log` (`clean` / `guard` / `load`). Guard actions: NaN/inf/NULL amount, NULL or orphan `company_id` on transactions/invoices, NULL dates, repeated `transaction_id` / `(company_id, operation_id)` are dropped and counted (orphans would create phantom companies in the grid because the grid comes from transactions); out-of-window dates and future-issued invoices are flagged; unknown categories, statuses, invoice types, orphan products/balances, stale balance snapshot, NULL group are kept and counted. Every detector is logged even with 0 rows, so a run shows the check ran. On the current CSVs all guards count 0, so `clean` equals the old one.
- `analysis/pipeline_check.py` (new): acceptance checks. All pass: (1) two runs give byte-identical parquet, identical dq_log and row counts; (2) a 40-company subset (20 with invoices, 20 without, 2 with a 3-month trail) completes and the no-invoice companies have null invoice features; (3) a folder with injected dirt (repeated ids, orphans, NaN amounts, unknown category/status, extra/missing column, missing `invoices` column and `debt_schedule_config.csv`, a malformed row) completes and each kind appears in `dq_log` with the injected count.
- `analysis/README.md`: two lines on the command and on `dq_log.source`.

## Decisions / findings

- **DuckDB float sums are thread-order dependent.** Without `SET threads = 1` the parquet differed between runs in `h_sib_*` (1e-7). The pipeline forces one thread for the feature build (17 s total, no cost worth mentioning).
- **The night's store has float noise at zero cash.** Versus the committed-era `monthly.parquet` (built multi-threaded) the rebuild differs only in `b_*` columns: `b_liq` by up to 1e-6, and 168 rows of `b_below_0`, 216 of `b_neg_liq_3`, 106 of `b_neg_episodes` flip because a company with exactly zero cash gets `liq = -1e-10` or `+1e-11` depending on summation order (`liquidity.py` compares `liq < 0` unrounded; I did not edit it). The score recomputes the negative-cash indicators from `b_liq` rounded to cents.
- The pipeline sorts the panel by `company_id, period` (the old build left DuckDB's group order); nothing downstream joins by position.
- Feature families do not fit anything across companies (grepped), so "same rules on hidden data, no re-fitting" holds at the feature level. `h_*` (group context) depend on which siblings are in the folder.

## Still unknown

- Hidden folder's calendar window: `MONTHS`, `AS_OF`, `SNAPSHOT` are constants (2024-09..2026-09). If the hidden data covers another window the features need parametrising (not touched). Out-of-window transactions are flagged in `dq_log`.
- Real hidden dirt classes; the guards are generic, not measured on hidden data.
- Whitespace/case variants of `company_id` are not normalised (they show up as orphans and are dropped, counted).
