# analysis/

Exploration between `data/` and `product/`.

The X Ray brief asks the system to answer these **per company, per month**. Use this list as the analysis checklist; there are no results here yet.

1. Who is healthy
2. Who is improving
3. Who is starting to turn
4. One-month dip vs structural fall
5. Why the score changed (which signal, when)
6. How many months earlier the change was visible

Hidden test: 60–80 companies the model must not train on. Split work should respect that once a split exists.

## Local database (DuckDB)

The notebooks read `data/embat.duckdb`, built locally from the CSVs (not committed). CSVs can sit in `data/` or `data/raw/output/`.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r analysis/requirements.txt
python analysis/build_db.py        # ~5 s; --force to rebuild
```

Then open the `.Rmd` files in RStudio and Knit (R packages install on the first chunk). Nobody else may hold the database open for writing.

From Python: `duckdb.connect("data/embat.duckdb", read_only=True)`.

## Layout

```text
analysis/
├── README.md
├── build_db.py                    # CSV -> data/embat.duckdb
├── requirements.txt
├── eda.Rmd                        # EDA of the 8 tables
└── 02_score_salud_financiera.Rmd  # signals, score, trajectory, explanation, monitor, product; writes analysis/outputs/*.csv
```

## Status

Both notebooks are written but **not yet run end to end** (R was not available where they were authored) — expect small fixes on first knit.
