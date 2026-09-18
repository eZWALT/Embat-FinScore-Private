# analysis/ — goal 1: signals

Find signals in the X Ray treasury trail. Do not put a 0–100 formula here; that is `product/score/`.

The brief still applies, per company, per month: who is healthy / improving / turning; dip vs fall; what moved; how many months earlier it showed.

Hidden test: 60–80 companies must not be used to fit anything. Split when one exists.

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
