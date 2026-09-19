"""Acceptance checks for analysis.pipeline. Writes only under a temp folder.

    python -m analysis.pipeline_check [--csv-dir DIR] [--keep]

1. determinism   two runs on the same CSVs give byte-identical parquet and identical dq_log
2. subset        a folder with ~40 companies (no invoices for some, a company with a 3-month trail) completes
3. new dirt      a folder with injected dirt (duplicate ids, orphan and NaN rows, unknown category/status,
                 extra and missing columns, a malformed CSV row, no invoices.csv) completes and each kind
                 shows up in dq_log
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import build_db, pipeline


def _write_subset(src: Path, dst: Path, companies: list[str], short_trail: list[str]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    ids = ", ".join(f"'{c}'" for c in companies)
    read = lambda csv: f"read_csv('{(src / csv).as_posix()}', header = true, all_varchar = true, sample_size = -1)"
    for table, (csv, *_rest) in build_db.TABLES.items():
        if table == "groups":
            where = f"group_id IN (SELECT group_id FROM {read('companies.csv')} WHERE company_id IN ({ids}))"
        else:
            where = f"company_id IN ({ids})"
        date_col = {"transactions": "date", "invoices": "issuance_date"}.get(table)
        if date_col:  # cutting activity before 2026-06 also cuts the older invoices and their payments
            late = ", ".join(f"'{c}'" for c in short_trail)
            where += f" AND (company_id NOT IN ({late}) OR CAST({date_col} AS TIMESTAMP) >= TIMESTAMP '2026-06-01')"
        con.execute(f"COPY (SELECT * FROM {read(csv)} WHERE {where}) TO '{(dst / csv).as_posix()}' (HEADER, DELIMITER ',')")
    con.close()


def _check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", type=Path, default=None)
    ap.add_argument("--keep", action="store_true", help="keep the temp folder")
    args = ap.parse_args()
    src = args.csv_dir or build_db.find_csv_dir()
    tmp = Path(tempfile.mkdtemp(prefix="pipeline_check_"))
    ok = True
    try:
        print("1. determinism")
        a = pipeline.run(src, tmp / "run1", verbose=False)
        b = pipeline.run(src, tmp / "run2", verbose=False)
        ok &= _check("parquet sha256 equal", a["feature_store"]["parquet_sha256"] == b["feature_store"]["parquet_sha256"])
        ok &= _check("dq_log equal", a["dq_log_rows_affected"] == b["dq_log_rows_affected"])
        ok &= _check("clean row counts equal", a["clean_rows"] == b["clean_rows"])

        print("2. subset of companies")
        full = duckdb.connect(str(tmp / "run1" / "embat.duckdb"), read_only=True)
        inv = {r[0] for r in full.execute("SELECT DISTINCT company_id FROM clean.invoices").fetchall()}
        allc = [r[0] for r in full.execute("SELECT company_id FROM clean.companies ORDER BY company_id").fetchall()]
        full.close()
        with_inv, without_inv = [c for c in allc if c in inv][:20], [c for c in allc if c not in inv][:20]
        subset = sorted(with_inv + without_inv)
        late = [with_inv[0], without_inv[0]]   # short trail: keep only their last 3 months (2026-06..08)
        sub_dir = tmp / "subset_csv"
        _write_subset(src, sub_dir, subset, short_trail=late)
        s = pipeline.run(sub_dir, tmp / "subset_out", verbose=False)
        ok &= _check(f"completes on {len(subset)} companies", s["feature_store"]["companies"] <= len(subset),
                     f"{s['feature_store']['companies']} in store, {len(late)} with a 3-month trail")
        panel = pd.read_parquet(tmp / "subset_out" / "feature_store" / "monthly.parquet")
        ok &= _check("short-trail companies have <= 3 grid months per company",
                     bool((panel[panel.company_id.isin(late)].groupby("company_id").size() <= 3).all()))
        no_inv = panel[~panel.company_id.isin(inv)]
        ok &= _check("companies without invoices have null invoice features", no_inv["e_delay_coll"].isna().all())

        print("3. injected dirt")
        dirt = tmp / "dirt_csv"
        shutil.copytree(sub_dir, dirt)
        tx = pd.read_csv(dirt / "transactions.csv", dtype=str)
        extra = tx.iloc[:5].copy()                     # duplicate transaction_id
        orphan = tx.iloc[5:8].copy(); orphan["company_id"] = "COMP_NOPE"; orphan["transaction_id"] = ["TX_ORPH_1", "TX_ORPH_2", "TX_ORPH_3"]
        nan_amt = tx.iloc[8:10].copy(); nan_amt["transaction_id"] = ["TX_NAN_1", "TX_NAN_2"]; nan_amt["amount"] = ["nan", ""]
        newcat = tx.iloc[10:14].copy(); newcat["transaction_id"] = [f"TX_CAT_{i}" for i in range(4)]; newcat["category"] = "crypto_swap"
        tx = pd.concat([tx, extra, orphan, nan_amt, newcat], ignore_index=True)
        tx["extra_col"] = "x"
        tx.to_csv(dirt / "transactions.csv", index=False)
        with open(dirt / "transactions.csv", "a", encoding="utf8") as f:   # malformed row: wrong types
            f.write("TX_BAD,COMP_X,PROD_X,not-a-date,,abc,,booked,,payment,desc,,x\n")
        inv_df = pd.read_csv(dirt / "invoices.csv", dtype=str)
        inv_df.loc[0, "status"] = "teleported"
        inv_df.drop(columns=["concept"]).to_csv(dirt / "invoices.csv", index=False)
        (dirt / "debt_schedule_config.csv").unlink()
        d = pipeline.run(dirt, tmp / "dirt_out", verbose=False)
        log = d["dq_log_rows_affected"]
        want = {
            "guard:transactions:transaction_id repetido": 5,
            "guard:transactions:company_id nulo o ausente de companies (huerfana; crearia empresas fantasma en el grid)": 3,
            "guard:transactions:importe nulo / NaN / infinito": 2,
            "guard:transactions:categoria fuera del mapa conocido": 4,
            "guard:invoices:status fuera de la lista conocida": 1,
            "load:transactions:columna 'extra_col' no esperada en el CSV": 0,
            "load:invoices:columna 'concept' ausente en el CSV": 0,
            "load:debt_schedule_config:archivo debt_schedule_config.csv ausente": 0,
        }
        for k, v in want.items():
            ok &= _check(k.split(":", 1)[1][:70], log.get(k) == v, f"logged {log.get(k)}, expected {v}")
        bad_key = "load:transactions:fila del CSV que no parsea con el esquema (tipo/comillas/columnas)"
        ok &= _check("malformed CSV row rejected and logged", log.get(bad_key, 0) >= 1, f"logged {log.get(bad_key)}")
        con = duckdb.connect(str(tmp / "dirt_out" / "embat.duckdb"), read_only=True)
        n_clean = con.execute("SELECT count(*) FROM clean.transactions").fetchone()[0]
        n_orig = duckdb.connect().execute(
            f"SELECT count(*) FROM read_csv('{(sub_dir / 'transactions.csv').as_posix()}', header=true, all_varchar=true) WHERE amount <> '0'").fetchone()[0]
        con.close()
        ok &= _check("dirty rows removed, 4 unknown-category rows kept", n_clean == n_orig + 4, f"{n_clean} vs {n_orig}+4")
        ok &= _check("store built without the phantom company", "COMP_NOPE" not in set(
            pd.read_parquet(tmp / "dirt_out" / "feature_store" / "monthly.parquet").company_id))
    finally:
        if args.keep:
            print("kept", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print("\nALL OK" if ok else "\nFAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
