# Wave 4 — data-join QA (one note)

Agent `d56ee5fe`. Lane: `analysis/evaluate/data_join_qa.py`. No commit. No parquet/duckdb rewrite. No `product/`. No 0–100. Holdout never used to choose a cut.

## Files written

- `analysis/evaluate/data_join_qa.py` — create; 11 write→run variants in this module
- `analysis/outputs/data_join_qa.md` — tables + KEEP/PARK/CLOSE
- `overnight/waves/wave4_data_qa.md` — this note
- `analysis/experiments/registry.csv` — 10 append-only `split=train` coverage rows (`d56ee5fe`)

Did not edit family modules, models, `explain_y7.py`, `catmix.py`, `gbm_core.py`, holdout membership, `LIVE.json`, canvas.

## Coverage (train; quote these)

- **470 confirmed.** 1,214 train companies: 744 have a book invoice, 470 do not (38.7%). Holdout 40/72 invoiced (descriptive). All companies have txs (grid is tx-born). Invoices-only off-grid: 0.
- **Why the 470:** 433 / 470 (**92.1%**) have NULL `companies.erp`. Named-ERP but no book: 37 (dynamicsAx 18, netsuite 8, businessCentral 6, …). They have bank books (debt 30.6% vs 28.8% invoiced). Not a feature-code drop.
- No-invoice is **both** clustered and mixed: 79 all-dark groups (360 cos), 38 mixed groups (110 dark + 230 invoiced), 118 all-invoiced. 23.4% of the 470 have an invoiced sibling.
- Train company-months **21,157**. Any invoice iss/due/paid that month 54.4%; any tx 95.8%; **both 53.4% (11,297)**. Quarterly both only 55.9% — the hole is company-level ERP, not month timing.
- Family E does **not** drop usable books: 0 raw-invoice companies missing `e_*`. `e_*` non-null 64.1% (zero-fill after first invoice) vs raw issuance-that-month 52.8%. One extra `e_*` company: `COMP_0962` (single refund, no book invoice).
- Family D invoice-structure ~53%; `d_interco_share` **0** non-null.

## ID / amount

- Invoice ∩ tx CP IDs **train 40,670** (34.9% of train invoice CPs; 88.0% of train tx CPs). 50.0% of train invoice rows have a CP that also appears on a train tx; 8.8% of train tx rows (90% of txs have NULL CP).
- `COMP_*` ∩ `COUNTERPARTY_*` **disjoint**. Equality join `counterparty_id = company_id`: **0 + 0**. Shared CP across siblings in a train group: invoice **0**, tx **1** (GROUP_0220 `COUNTERPARTY_08536`, payment vs utility — a shared vendor, not interco).

## Amount-match (hit vs random)

| window | hit | random partner |
| --- | ---: | ---: |
| issuance month, \|Δ\|≤0.01 | **21.2%** | **0.5%** |
| payment month, \|Δ\|≤0.01 | **35.7%** | **0.5%** |
| payment ±1 day | 25.0% | — |
| unique non-round | 18.3% | 0.3% |
| same CP + amount | 10.6% | — |
| sibling company amounts | 5.9% | — |
| wrong sign | 2.2% | — |
| adjacent month | 20.0% | — |

Random ≪ real → not collision. Spearman vs `log1p(|a_op_in|)` = 0.03 (not size). Still **not a row FK** (~64% of paid invoices unmatched).

## Did Y8 have the other table?

- `y8_inv_worse_6`: 3,883 labeled; **tx present 98.1%**. 100% ever-ERP. **CLOSE** the “no cash table” excuse (lost to `a_in12` anyway).
- `y8_cash_worse_6`: 6,986 labeled; invoice *activity* that month **58.0%**; any `e_*` **68.0%**; `e_side_ready` **55.1%**. **32.0% (2,238)** never-ERP. Of those, 616 (27.5%) are mixed-group (H can see sister cash); 1,622 are all-dark.
- Y5 / Y7: **~99% have tx**. **CLOSE** “Y5 died because cash was missing”.

Y3 stressed: inv-month recover 6.1% vs no-inv 8.4%; ever-ERP 7.3% vs never 6.8%. Unmatched-share vs Y3 Spearman 0.05 — **PARK as a Y3 X**.

## Verdicts

| probe | verdict |
| --- | --- |
| amount-match join as a **feature** (match-rate, not pairs) | **KEEP** |
| interco / `d_interco_share` | **CLOSE** |
| “Y8 failed because no join” | **PARK** (split: y8_inv CLOSE excuse; y8_cash has a 32% empty-E hole, ERP slice still failed) |

## What failed / next idea

Amount-match cannot recover a 1–1 invoice↔tx FK. Interco is unmeasurable. Y8 inv had cash and still died — not a data-join failure. Y8 cash mixes 470 empty-E rows into the label.

**Legal next:** `y_erp_gap_then_cash` = existing Y2/Y3 **among the 470** using only A/B/C/F/G/H (no invented cobros). Do not rebuild Y8. Unmatched-share stays a Q5 diagnostic, not a Y3 X. Family G: 73.6% of train banking books have first `created_at` after 2024-09. `f_w_rate` ever on 38 train companies (1.7% cm); `f_util_snapshot` last-month only (334 cos).

---

## Family J — amount-match rates (same agent, same lane)

`data_join_qa.py` is now read-only. New owner files: `analysis/features/match.py`, `analysis/outputs/match_report.md`. No parquet merge. No FAMILIES edit. No Y8 rebuild. No interco map.

`build(con, grid)` → `company_id, period, j_pay_match, j_iss_match, j_has_book`.
`SOURCE_TABLES=["transactions","invoices"]`, `FAMILY="j"`. Payment-month, `|Δ|≤0.01` from QA (not holdout-fit). Greedy 1-1 inside `(company, period, sign, cents)`: `n_matched = min(n_inv, n_tx)`.

Smokes: 470 → `j_pay_match` all NaN (not 0); July `n_paid` matches July SQL only; weekly 2-co grid OK. 22,230 × 3 on the monthly grid.

### Emitted (train; holdout 72 out)

| col | cov_all_cm | cov_ever-ERP | size ρ vs log1p(a_in3) | acf1 | rewrite vs e_* | vs Y3 | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `j_pay_match` | 45.6% | 71.2% | 0.000 | 0.226 | max \|ρ\| 0.09 (`e_pending_amt_share`) | −0.060 | **KEEP-Q5** |
| `j_iss_match` | 52.8% | 82.5% | 0.018 | 0.262 | max \|ρ\| 0.34 (`e_pending_amt_share`) | −0.027 | **KEEP-Q5** |
| `j_has_book` | 100% | 100% | −0.015 | 0.835 | 0.28 / 0.32 vs issued/open | −0.009 | **KEEP-Q5** (miss flag) |

- `j_pay_match` vs `j_iss_match` ρ = **0.678** (< 0.9) → keep both. Iss residual after pay vs Y3 = −0.012.
- `j_has_book` acf1 is the causal step (156 train cos flip 0→1; 470 stay 0; 588 start 1), not a cycle.
- Cm-mean 1-1 = 39.2%; invoice-weighted 1-1 = 33.6%; invoice-weighted set-overlap = 36.7% (QA headline 35.7%). Wrong-sign cm-mean 1.5%.

### PARK / CLOSE (not emitted)

| col | why |
| --- | --- |
| `j_pay_unmatched` | ρ = −1 with pay |
| `j_pay_match_t3` | acf1 = 0.689 (overlap); **acf3 = −0.003** → CLOSE, same honesty as mix t3 |
| `j_n_paid` / `j_n_matched` | not SIZE (ρ 0.45 / 0.42) but Y3 raw −0.14 / −0.16 **dies after log1p(a_in3)** (residual −0.000) and after `e_ar_issued` (0.015). Prefer rates. |
| `j_pay_set` | near-twin of 1-1 |
| `j_pay_wrong` | control; stay in report |
| `j_interco_*` / row FK / COMP↔CP | never built |

### J is a Y3 X?

**No.** `j_pay_match` vs `y3_recover_cash_6m` = −0.060 (unmatched +0.060). Y7 / Y8 all \|ρ\| < 0.08 on train labeled (iss vs Y7 on *paid* months only is 0.082 — do not quote as a Y7 X; official labeled ρ = 0.047). **KEEP Family J as a Q5 diagnostic / miss flag**, not a recovery predictor.

### Appendix — `y_erp_gap_then_cash` label design (no module this pass)

Slice, not a new formula. Do **not** invent cobros.

- **Population:** the 470 never-ERP train companies (`j_has_book` stays 0). Holdout 72 stay out of any cut.
- **Label:** existing `y2_*` / `y3_recover_cash_6m` restricted to that population. Cash-only Y3 already forbids Family E as X — the 470 are the empty-E hole, so this is the honest place to ask the cash question.
- **Allowed X:** A / B / C / F / G / H only. No E, no D invoice-structure, no Family J rates (they are NaN here by construction), no invented invoice↔tx pairs.
- **Forbidden:** rebuilding Y8; filling E with tx lookalikes; using mixed-group sister invoices as own books.
- **Acceptance sketch:** same Y3 stressed-rate band (5–50%) *inside the 470*; must beat `log1p(a_in3)` / runway single-feature; group-aware CV. If the 470 stressed n is tiny, CLOSE the label (power), do not relax the band on holdout.
- **Why this and not J-as-Y3-X:** match-rate vs Y3 is 0.06; the 470 question is “cash-only health when ERP was never connected”, which J cannot see.
