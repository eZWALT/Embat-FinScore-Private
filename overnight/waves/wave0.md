# Wave 0 — environment

- Deleted old `overnight/` (1.4 GB candidates, search_16h, compare, watchdog).
- Recreated `overnight/{logs,waves,splits}` and `analysis/{features,targets,evaluate,models,experiments,splits}`.
- Holdout 72 companies copied to both split paths (seed 20260918).
- Shared API: `analysis/features/common.py`, `grid.py`, `build_feature_store.py`.
- Contract: `overnight/CONTRACT.md`.
- DuckDB rebuild started if missing.
