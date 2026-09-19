"""Wave D — NSF / overdraft token hole (FinRegLab distress as X).

The data dictionary has no NSF token. This cut confirms that on the
dictionary, the feature dictionary, CAT_MAP, the monthly store, and raw
``transactions.category`` / ``description`` (train only). Leftover-after-days
is **undefined**. Do not reconstruct NSF / overdraft / neg-days from family B
(Y2 lock; walk is an identity). Do not invent ``y_nsf``. Do not score vs Y9
(Y9 is the fee label; ``m_fin`` leaks). PARK utilisation / Y10 stays.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.nsf_count_qa
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.evaluate.protocol import assert_no_holdout, load_holdout
from analysis.features.common import ANALYSIS, CAT_MAP, DATA, connect

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

DATA_DICT = ROOT / "data" / "data_dictionary.md"
FEAT_DICT = DATA / "feature_store" / "feature_dictionary.md"
STORE = DATA / "feature_store" / "monthly.parquet"
TARGETS = DATA / "feature_store" / "targets.parquet"
OUT_MD = ANALYSIS / "outputs" / "nsf_count_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "nsf_count_qa.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"

Y3_NIGHT = (0.762, 0.752)
DAYS_BAR = 0.711
SIZE_BAR = 0.617
Y7_TURNOVER = (0.720, 0.712)
SS_LEFT = 0.635
SALARY_LEFT = 0.603
P50_QUOTE = 1.079

# Dictionary / store / category names that would be a legal NSF token.
# Word-boundary: bare "nsf" is a substring of "transfer" / a_transfer.
TOKEN_NAME_RE = re.compile(
    r"(?<![a-z0-9_])("
    r"nsf|n\.s\.f|overdraft|overdrawn|descubierto|insufficient|"
    r"returned.?item|bounce|bounced|cumover|y_nsf"
    r")(?![a-z0-9_])",
    re.I,
)
# Description patterns. Not a leftover X — a hole scan.
# Word-bound "nsf" is the FinRegLab count token. "descubierto" is Spanish
# overdraft *interest/fee* narrative (Y9 / Family M), not CUMOVER.
COUNT_PATTERNS = (
    "nsf",
    "n.s.f",
    "overdraft",
    "overdrawn",
    "insufficient",
    "bounce",
    "bounced",
    "fondos insuficientes",
)
FEE_NARRATIVE = ("descubierto",)
OTHER_INSPECT = ("returned item",)
DESC_PATTERNS = COUNT_PATTERNS + FEE_NARRATIVE + OTHER_INSPECT
# Store / target columns that would be the reconstructed lock, not a token.
B_LOCK = ("b_below_0", "b_neg_episodes", "b_neg_liq_3", "b_min_liq_3")
Y9_COLS = ("y9_fee_r_ownp80", "y9_fee_spike")
FORBIDDEN_PROXIES = B_LOCK + ("f_util_snapshot", "a_fin_cost", "f_fc_r", "f_fin_cost")


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _f(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{float(x):.{nd}f}"


def _md_table(rows: list[dict], cols: list[str] | None = None) -> str:
    if not rows:
        return "_(empty)_\n"
    if cols is None:
        cols = list(rows[0].keys())
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _scan_text(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        if TOKEN_NAME_RE.search(line):
            hits.append(
                {
                    "file": path.name,
                    "line": i,
                    "snippet": line.strip()[:160],
                }
            )
    return hits


def _scan_names(names: list[str], where: str) -> list[dict]:
    rows = []
    for n in names:
        m = TOKEN_NAME_RE.search(n)
        rows.append(
            {
                "where": where,
                "name": n,
                "token_hit": bool(m),
                "match": m.group(0) if m else "",
            }
        )
    return [r for r in rows if r["token_hit"]]


def load_scan() -> dict:
    hold = {str(x) for x in load_holdout()}
    data_hits = _scan_text(DATA_DICT)
    feat_hits = _scan_text(FEAT_DICT)
    # Literature mentions of NSF in the feature dictionary are cites, not columns.
    feat_col_hits = [
        h
        for h in feat_hits
        if re.search(r"^#{2,4}\s+`", h["snippet"]) or re.search(r"`[a-z0-9_]*nsf", h["snippet"], re.I)
    ]

    store_cols = [str(c) for c in pq.ParquetFile(STORE).schema_arrow.names]
    tgt_cols = [str(c) for c in pq.ParquetFile(TARGETS).schema_arrow.names]
    store_tok = _scan_names(store_cols, "monthly.parquet")
    tgt_tok = _scan_names(tgt_cols, "targets.parquet")
    cat_tok = _scan_names(list(CAT_MAP), "CAT_MAP")
    grp_tok = _scan_names(sorted(set(CAT_MAP.values())), "CAT_MAP group")

    con = connect()
    hold_df = pd.DataFrame({"company_id": sorted(hold)})
    con.register("_hold", hold_df)
    train_ids = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id
        FROM companies
        WHERE CAST(company_id AS VARCHAR) NOT IN (SELECT company_id FROM _hold)
        """
    ).df()
    assert_no_holdout(train_ids["company_id"])
    cats = con.execute(
        """
        SELECT CAST(t.category AS VARCHAR) AS category, COUNT(*) AS n_tx
        FROM transactions t
        WHERE CAST(t.company_id AS VARCHAR) NOT IN (SELECT company_id FROM _hold)
        GROUP BY 1
        ORDER BY n_tx DESC
        """
    ).df()
    cats["category"] = cats["category"].astype(str)
    cat_name_hits = _scan_names(cats["category"].tolist(), "transactions.category")

    def _desc_pred(p: str) -> str:
        # Word-ish boundary so "nsf" does not hit "transfer" / "transferencia".
        esc = p.lower().replace(".", r"\.")
        return (
            f"regexp_matches(lower(CAST(t.description AS VARCHAR)), "
            f"'(^|[^a-z0-9]){esc}([^a-z0-9]|$)')"
        )

    desc_sql = ",\n            ".join(
        f"SUM(CASE WHEN {_desc_pred(p)} THEN 1 ELSE 0 END) AS p{i}"
        for i, p in enumerate(DESC_PATTERNS)
    )
    desc_hit = con.execute(
        f"""
        SELECT {desc_sql}
        FROM transactions t
        WHERE CAST(t.company_id AS VARCHAR) NOT IN (SELECT company_id FROM _hold)
        """
    ).fetchone()
    desc_rows = [
        {"pattern": p, "n_tx_train": int(desc_hit[i])} for i, p in enumerate(DESC_PATTERNS)
    ]

    n_hold = int(
        con.execute("SELECT COUNT(*) FROM companies WHERE CAST(company_id AS VARCHAR) IN (SELECT company_id FROM _hold)").fetchone()[0]
    )
    n_train_co = int(len(train_ids))
    n_tx_train = int(cats["n_tx"].sum()) if len(cats) else 0
    fee_n = int(cats.loc[cats["category"] == "fee", "n_tx"].sum()) if "fee" in set(cats["category"]) else 0
    int_n = (
        int(cats.loc[cats["category"] == "interest_charge", "n_tx"].sum())
        if "interest_charge" in set(cats["category"])
        else 0
    )
    uncat_n = (
        int(cats.loc[cats["category"] == "uncategorized", "n_tx"].sum())
        if "uncategorized" in set(cats["category"])
        else 0
    )
    desc_nco = {}
    for p in DESC_PATTERNS:
        desc_nco[p] = int(
            con.execute(
                f"""
                SELECT COUNT(DISTINCT CAST(t.company_id AS VARCHAR))
                FROM transactions t
                WHERE CAST(t.company_id AS VARCHAR) NOT IN (SELECT company_id FROM _hold)
                  AND {_desc_pred(p)}
                """
            ).fetchone()[0]
        )
    desc_mix = con.execute(
        f"""
        SELECT CAST(t.category AS VARCHAR) AS category, COUNT(*) AS n_tx,
               COUNT(DISTINCT CAST(t.company_id AS VARCHAR)) AS n_co
        FROM transactions t
        WHERE CAST(t.company_id AS VARCHAR) NOT IN (SELECT company_id FROM _hold)
          AND {_desc_pred('descubierto')}
        GROUP BY 1
        ORDER BY n_tx DESC
        """
    ).df()
    desc_samples = con.execute(
        f"""
        SELECT CAST(t.category AS VARCHAR) AS category,
               left(CAST(t.description AS VARCHAR), 80) AS description,
               t.amount
        FROM transactions t
        WHERE CAST(t.company_id AS VARCHAR) NOT IN (SELECT company_id FROM _hold)
          AND {_desc_pred('descubierto')}
        LIMIT 8
        """
    ).df()
    hold_desc = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM transactions t
            WHERE CAST(t.company_id AS VARCHAR) IN (SELECT company_id FROM _hold)
              AND {_desc_pred('descubierto')}
            """
        ).fetchone()[0]
    )
    hold_nsf = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM transactions t
            WHERE CAST(t.company_id AS VARCHAR) IN (SELECT company_id FROM _hold)
              AND {_desc_pred('nsf')}
            """
        ).fetchone()[0]
    )
    nsf_like_transfer = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM transactions t
            WHERE CAST(t.company_id AS VARCHAR) NOT IN (SELECT company_id FROM _hold)
              AND lower(CAST(t.description AS VARCHAR)) LIKE '%nsf%'
            """
        ).fetchone()[0]
    )
    con.close()

    legal_cats = [c for c in cats["category"] if TOKEN_NAME_RE.search(c)]
    count_hits = [r for r in desc_rows if r["pattern"] in COUNT_PATTERNS and r["n_tx_train"] > 0]
    hole = (
        not legal_cats
        and not cat_tok
        and not store_tok
        and not tgt_tok
        and not cat_name_hits
        and not count_hits
        and not feat_col_hits
    )
    # data_hits may mention nothing; feat_hits mention NSF as literature, not a column.
    return {
        "hold": hold,
        "n_hold": int(n_hold),
        "n_train_co": int(n_train_co),
        "n_tx_train": n_tx_train,
        "data_hits": data_hits,
        "feat_hits": feat_hits,
        "feat_col_hits": feat_col_hits,
        "store_cols": store_cols,
        "tgt_cols": tgt_cols,
        "store_tok": store_tok,
        "tgt_tok": tgt_tok,
        "cat_tok": cat_tok,
        "grp_tok": grp_tok,
        "cats": cats,
        "cat_name_hits": cat_name_hits,
        "desc_rows": desc_rows,
        "legal_cats": legal_cats,
        "legal_desc": count_hits,
        "desc_nco": desc_nco,
        "desc_mix": desc_mix,
        "desc_samples": desc_samples,
        "hold_desc": hold_desc,
        "hold_nsf": hold_nsf,
        "nsf_like_transfer": nsf_like_transfer,
        "fee_n": fee_n,
        "int_n": int_n,
        "uncat_n": uncat_n,
        "hole": hole,
        "b_lock_in_store": [c for c in B_LOCK if c in store_cols],
        "y9_in_tgt": [c for c in Y9_COLS if c in tgt_cols],
        "util_in_store": "f_util_snapshot" in store_cols,
    }


def _plot(ctx: dict) -> None:
    if not HAS_MPL:
        return
    cats = ctx["cats"].head(12)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.barh(cats["category"][::-1], cats["n_tx"][::-1], color="#1b4f72")
    ax.set_xlabel("train transaction count")
    ax.set_title("No NSF / overdraft category — top tokens (train)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)


def write_md(ctx: dict) -> None:
    hole = ctx["hole"]
    role = "HOLE — leftover undefined" if hole else "TOKEN FOUND — do not KEEP without leftover"
    headline = (
        "FinRegLab NSF-as-X is a dataset hole. Leftover-after-days is undefined. "
        "Last-value Q1 + quiet-stressed Y3 stay the cash-flow engine. "
        "No NSF / overdraft token in data_dictionary.md, CAT_MAP, monthly.parquet, "
        "or train descriptions. Do not reconstruct from B. Do not invent y_nsf. "
        "Do not score vs Y9. PARK utilisation / Y10 stays. "
        f"Night Y3 {Y3_NIGHT[0]}/{Y3_NIGHT[1]}, days {DAYS_BAR}, size {SIZE_BAR} unchanged."
    )
    if not hole:
        headline = (
            "A token-like string appeared — leftover-after-days is still not a KEEP. "
            "Do not reconstruct from B. Do not invent y_nsf. Do not score vs Y9."
        )

    cat_rows = [
        {
            "category": r["category"],
            "n_tx": f"{int(r['n_tx']):,}",
            "in CAT_MAP": "yes" if r["category"] in CAT_MAP else "no",
            "NSF token": "yes" if TOKEN_NAME_RE.search(str(r["category"])) else "no",
        }
        for _, r in ctx["cats"].iterrows()
    ]
    desc_rows = []
    for r in ctx["desc_rows"]:
        p = r["pattern"]
        if p in COUNT_PATTERNS:
            kind = "count token (FinRegLab NSF)"
            legal = "yes — leftover defined" if r["n_tx_train"] else "no"
        elif p in FEE_NARRATIVE:
            kind = "fee/interest narrative (Y9)"
            legal = "no — do not leftover / do not score vs Y9"
        else:
            kind = "other inspect"
            legal = "no — not NSF"
        desc_rows.append(
            {
                "pattern": p,
                "n_tx_train": f"{r['n_tx_train']:,}",
                "n_co": f"{ctx['desc_nco'].get(p, 0):,}",
                "kind": kind,
                "legal count": legal,
            }
        )
    mix_rows = [
        {
            "category": r["category"],
            "n_tx": f"{int(r['n_tx']):,}",
            "n_co": f"{int(r['n_co']):,}",
        }
        for _, r in ctx["desc_mix"].iterrows()
    ]
    samp_rows = [
        {
            "category": r["category"],
            "description": str(r["description"]),
            "amount": f"{float(r['amount']):.2f}",
        }
        for _, r in ctx["desc_samples"].iterrows()
    ]
    dict_rows = []
    if not ctx["data_hits"]:
        dict_rows.append(
            {
                "file": "data_dictionary.md",
                "NSF / overdraft token": "none",
                "note": "category examples are utility / tax / collection; placeholders are [X] / COUNTERPARTY",
            }
        )
    else:
        for h in ctx["data_hits"]:
            dict_rows.append(
                {
                    "file": h["file"],
                    "NSF / overdraft token": f"line {h['line']}",
                    "note": h["snippet"][:80],
                }
            )
    feat_note = (
        f"{len(ctx['feat_hits'])} literature mentions of NSF (Y2/Y9 cites); "
        f"{len(ctx['feat_col_hits'])} column headings. No `nsf_*` feature."
    )

    md = f"""# NSF / overdraft token hole (Wave D)

Generated `{ctx['now']}` by `analysis/evaluate/nsf_count_qa.py`.
DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only.
Train only. Holdout {ctx['n_hold']} coverage only. Seed 20260918.
No 0–100. No parquet rewrite. No `build_targets`. No new GBM.
Never B as Y2/Y3 X. Do not invent `y_nsf`. Do not score vs Y9.
Do not reconstruct neg-days from family B. Do not quote `a_out_vol` 0.722
as the engine. PARK utilisation / Y10 stays.

FinRegLab 2025 names NSF **counts**, low/neg ending balances, and daily-pay
MCAs as application distress X. Norden (no line): ΔCUMOVER is the only
activity predictor. Formisano/Modina: overdraft **days**. This dataset has
**no NSF token**. Leftover-after-days is therefore undefined — not CLOSE,
not DROP-as-weak, a hole.

## Headline

{headline}

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | Last-value `b_runway` KEEP (p50 {P50_QUOTE}). Not NSF. Never B as X. |
| 2 | Who is improving? | Not an NSF path. Y1 stays PARK. |
| 3 | Who is turning? | Quiet-stressed stays SS leftover {SS_LEFT} / salary {SALARY_LEFT} / days {DAYS_BAR}. NSF-as-X is a hole. |
| 4 | Dip vs fall? | Out. Sibling TURNOVER {Y7_TURNOVER[0]}/{Y7_TURNOVER[1]}. |
| 5 | Why did it change? | No NSF token to name. Fee/interest is Y9, not an X. |
| 6 | Months earlier? | Hidden {ctx['n_hold']} is coverage. No NSF lead. |

## KEEP / CLOSE / DROP / PARK / HOLE

| object | decision | why |
| --- | --- | --- |
| NSF / overdraft **token** as Y3 X | **{role}** | dictionary + CAT_MAP + store + train descriptions |
| leftover-after-days of NSF count | **undefined** | no legal count to residualise after days |
| reconstruct NSF from `b_below_0` / `b_neg_episodes` | **never** | Y2 lock; walk is an identity |
| invent `y_nsf` | **never** | new Y from the same hole |
| score the hole vs Y9 | **never** | Y9 is the fee label; `m_fin` leaks |
| `fee` / `interest_charge` as NSF proxy | **never** | euro fee, not a count; Family M / Y9 |
| `f_util_snapshot` / Y10 | **PARK** | last-month-only 1.6% |
| last-value Q1 + quiet-stressed Y3 | **KEEP engine** | night {Y3_NIGHT[0]}/{Y3_NIGHT[1]} · days {DAYS_BAR} · size {SIZE_BAR} |

## 1. Dictionaries

{ _md_table(dict_rows) }

Feature dictionary: {feat_note}

CAT_MAP keys with NSF-like names: {len(ctx['cat_tok'])}.
CAT_MAP groups with NSF-like names: {len(ctx['grp_tok'])}.
`fee` and `interest_charge` map to `fin_cost` — that is Y9 / Family M, not NSF.

## 2. Store / targets (schema only)

monthly.parquet columns: {len(ctx['store_cols'])}. NSF-like names: {len(ctx['store_tok'])}.
B-lock present (do not use as NSF): {', '.join(ctx['b_lock_in_store']) or '—'}.
`f_util_snapshot` in store: {ctx['util_in_store']}.

targets.parquet columns: {len(ctx['tgt_cols'])}. NSF-like names: {len(ctx['tgt_tok'])}.
Y9 present (do not score the hole vs them): {', '.join(ctx['y9_in_tgt']) or '—'}.
`y_nsf` is not in the store.

## 3. Train categories (holdout out)

n_train companies = {ctx['n_train_co']:,}. n_tx = {ctx['n_tx_train']:,}.
`fee` n_tx = {ctx['fee_n']:,}. `interest_charge` n_tx = {ctx['int_n']:,}.
`uncategorized` n_tx = {ctx['uncat_n']:,}. NSF-like category names: {len(ctx['legal_cats'])}.

{_md_table(cat_rows)}

## 4. Train descriptions (ILIKE hole scan, not an X)

{_md_table(desc_rows)}

Word-bound `nsf` / `overdraft` / `bounce` = **0**. A raw `LIKE '%nsf%'`
hits {ctx['nsf_like_transfer']:,} train txs because **`nsf` sits inside
`transfer` / `transferencia`** — not an NSF token.

`descubierto` is Spanish overdraft **interest / claim-fee** text
(`INTERES.DESCUBIERTO`, `GASTOS POR RECLAMACIÓN DE DESCUBIERTO`). That is
Y9 / Family M, not Norden ΔCUMOVER and not a FinRegLab NSF **count**.
Do not leftover it after days. Do not score it vs Y9.

`returned item` is a credit-note / furniture narrative, not a returned check.

Holdout coverage (no fit): `descubierto` n_tx = {ctx['hold_desc']:,};
word-bound `nsf` n_tx = {ctx['hold_nsf']:,}.

### Extra — `descubierto` category mix (train, not an X)

{_md_table(mix_rows)}

{_md_table(samp_rows)}

## Extra — what we refuse to compute

- Leftover of `b_below_0` / `b_neg_episodes` / reconstructed neg-days after days.
  That is the Y2 lock, not CUMOVER.
- Leftover of `a_fin_cost` / `f_fc_r` / `m_fee_share` vs Y3 or vs Y9.
- Any utilisation / Y10 rewrite.
- Hidden-72 fit. New GBM. 0–100.

## Explicitly out

- Editing `liquidity.py`. Putting B on Y2/Y3 X or the 15-col card.
- AMPLI (HIGH−LOW of balances). Invent `y_nsf`. Score vs Y9.
- Quote `a_out_vol` 0.722 as the engine.
- Overwrite `runway_window_qa.*` / `days_delta_qa.*` / `y3_reasons.*` /
  leftover QA owners (`debt_svc` / `fin_cost` / `cust_lost`) / `lit_invoice.*`.

Plot: `nsf_count_qa.png`.

Night Y3 {Y3_NIGHT[0]}/{Y3_NIGHT[1]}, days {DAYS_BAR}, size {SIZE_BAR},
TURNOVER {Y7_TURNOVER[0]}/{Y7_TURNOVER[1]}, SS leftover {SS_LEFT},
salary {SALARY_LEFT} unchanged.
"""
    OUT_MD.write_text(md, encoding="utf-8")


def _append_registry(ctx: dict) -> None:
    ts = datetime.now().astimezone().isoformat(timespec="seconds")
    n_desc = sum(r["n_tx_train"] for r in ctx["desc_rows"])
    row = (
        f"{ts},R4,4,nsf_count_qa,-,y3_recover_cash_6m,nsf_count_qa,train,"
        f"nsf_token_hits,{len(ctx['legal_desc'])},{n_desc},"
        f"{'HOLE leftover undefined' if ctx['hole'] else 'TOKEN inspect'} "
        f"n_cat={len(ctx['cats'])} fee_n={ctx['fee_n']}"
    )
    with REGISTRY.open("a") as f:
        f.write(row + "\n")


def main() -> None:
    t0 = datetime.now(timezone.utc)
    ctx = load_scan()
    ctx["now"] = _now_iso()
    if HAS_MPL:
        _plot(ctx)
    write_md(ctx)
    _append_registry(ctx)
    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    print(
        f"wrote {OUT_MD} hole={ctx['hole']} n_cat={len(ctx['cats'])} "
        f"desc_hits={sum(r['n_tx_train'] for r in ctx['desc_rows'])} {elapsed:.0f}s"
    )


if __name__ == "__main__":
    main()
