# NSF / overdraft token hole (Wave D)

Generated `2026-09-19T09:07:06+02:00` by `analysis/evaluate/nsf_count_qa.py`.
DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only.
Train only. Holdout 72 coverage only. Seed 20260918.
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

FinRegLab NSF-as-X is a dataset hole. Leftover-after-days is undefined. Last-value Q1 + quiet-stressed Y3 stay the cash-flow engine. No NSF / overdraft token in data_dictionary.md, CAT_MAP, monthly.parquet, or train descriptions. Do not reconstruct from B. Do not invent y_nsf. Do not score vs Y9. PARK utilisation / Y10 stays. Night Y3 0.762/0.752, days 0.711, size 0.617 unchanged.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | Last-value `b_runway` KEEP (p50 1.079). Not NSF. Never B as X. |
| 2 | Who is improving? | Not an NSF path. Y1 stays PARK. |
| 3 | Who is turning? | Quiet-stressed stays SS leftover 0.635 / salary 0.603 / days 0.711. NSF-as-X is a hole. |
| 4 | Dip vs fall? | Out. Sibling TURNOVER 0.72/0.712. |
| 5 | Why did it change? | No NSF token to name. Fee/interest is Y9, not an X. |
| 6 | Months earlier? | Hidden 72 is coverage. No NSF lead. |

## KEEP / CLOSE / DROP / PARK / HOLE

| object | decision | why |
| --- | --- | --- |
| NSF / overdraft **token** as Y3 X | **HOLE — leftover undefined** | dictionary + CAT_MAP + store + train descriptions |
| leftover-after-days of NSF count | **undefined** | no legal count to residualise after days |
| reconstruct NSF from `b_below_0` / `b_neg_episodes` | **never** | Y2 lock; walk is an identity |
| invent `y_nsf` | **never** | new Y from the same hole |
| score the hole vs Y9 | **never** | Y9 is the fee label; `m_fin` leaks |
| `fee` / `interest_charge` as NSF proxy | **never** | euro fee, not a count; Family M / Y9 |
| `f_util_snapshot` / Y10 | **PARK** | last-month-only 1.6% |
| last-value Q1 + quiet-stressed Y3 | **KEEP engine** | night 0.762/0.752 · days 0.711 · size 0.617 |

## 1. Dictionaries

| file | NSF / overdraft token | note |
| --- | --- | --- |
| data_dictionary.md | none | category examples are utility / tax / collection; placeholders are [X] / COUNTERPARTY |


Feature dictionary: 2 literature mentions of NSF (Y2/Y9 cites); 0 column headings. No `nsf_*` feature.

CAT_MAP keys with NSF-like names: 0.
CAT_MAP groups with NSF-like names: 0.
`fee` and `interest_charge` map to `fin_cost` — that is Y9 / Family M, not NSF.

## 2. Store / targets (schema only)

monthly.parquet columns: 118. NSF-like names: 0.
B-lock present (do not use as NSF): b_below_0, b_neg_episodes, b_neg_liq_3, b_min_liq_3.
`f_util_snapshot` in store: True.

targets.parquet columns: 29. NSF-like names: 0.
Y9 present (do not score the hole vs them): y9_fee_r_ownp80, y9_fee_spike.
`y_nsf` is not in the store.

## 3. Train categories (holdout out)

n_train companies = 1,214. n_tx = 2,424,704.
`fee` n_tx = 167,758. `interest_charge` n_tx = 7,701.
`uncategorized` n_tx = 595,540. NSF-like category names: 0.

| category | n_tx | in CAT_MAP | NSF token |
| --- | --- | --- | --- |
| uncategorized | 595,540 | no | no |
| collection | 537,032 | yes | no |
| payment | 345,980 | yes | no |
| utility | 249,063 | yes | no |
| fee | 167,758 | yes | no |
| transfer | 146,486 | yes | no |
| bulk_collection | 64,930 | yes | no |
| tax | 52,410 | yes | no |
| cash_settlement | 45,590 | yes | no |
| pos_settlement | 44,545 | yes | no |
| salary | 41,101 | yes | no |
| bulk_payment | 40,228 | yes | no |
| social_security | 22,926 | yes | no |
| debt_repayment | 21,122 | yes | no |
| cash_withdrawal | 13,494 | yes | no |
| collection_refund | 11,512 | yes | no |
| interest_charge | 7,701 | yes | no |
| pos_withdrawal | 7,329 | yes | no |
| investment_deployment | 3,725 | yes | no |
| investment_return | 3,245 | yes | no |
| payment_refund | 2,985 | yes | no |
| tax_refund | 2 | yes | no |


## 4. Train descriptions (ILIKE hole scan, not an X)

| pattern | n_tx_train | n_co | kind | legal count |
| --- | --- | --- | --- | --- |
| nsf | 0 | 0 | count token (FinRegLab NSF) | no |
| n.s.f | 0 | 0 | count token (FinRegLab NSF) | no |
| overdraft | 0 | 0 | count token (FinRegLab NSF) | no |
| overdrawn | 0 | 0 | count token (FinRegLab NSF) | no |
| insufficient | 0 | 0 | count token (FinRegLab NSF) | no |
| bounce | 0 | 0 | count token (FinRegLab NSF) | no |
| bounced | 0 | 0 | count token (FinRegLab NSF) | no |
| fondos insuficientes | 0 | 0 | count token (FinRegLab NSF) | no |
| descubierto | 250 | 83 | fee/interest narrative (Y9) | no — do not leftover / do not score vs Y9 |
| returned item | 2 | 2 | other inspect | no — not NSF |


Word-bound `nsf` / `overdraft` / `bounce` = **0**. A raw `LIKE '%nsf%'`
hits 480,882 train txs because **`nsf` sits inside
`transfer` / `transferencia`** — not an NSF token.

`descubierto` is Spanish overdraft **interest / claim-fee** text
(`INTERES.DESCUBIERTO`, `GASTOS POR RECLAMACIÓN DE DESCUBIERTO`). That is
Y9 / Family M, not Norden ΔCUMOVER and not a FinRegLab NSF **count**.
Do not leftover it after days. Do not score it vs Y9.

`returned item` is a credit-note / furniture narrative, not a returned check.

Holdout coverage (no fit): `descubierto` n_tx = 12;
word-bound `nsf` n_tx = 0.

### Extra — `descubierto` category mix (train, not an X)

| category | n_tx | n_co |
| --- | --- | --- |
| utility | 104 | 44 |
| fee | 83 | 46 |
| uncategorized | 47 | 23 |
| collection | 9 | 7 |
| payment | 4 | 4 |
| transfer | 3 | 2 |


| category | description | amount |
| --- | --- | --- |
| payment | INTERES.DESCUBIERTO | -221.34 |
| fee | INTERES.DESCUBIERTO | -0.15 |
| utility | INTERES.DESCUBIERTO | -1.47 |
| fee | INTERES.DESCUBIERTO | -0.77 |
| utility | INTERES.DESCUBIERTO | -1.42 |
| utility | INTERES.DESCUBIERTO | -0.12 |
| utility | INTERES.DESCUBIERTO | -0.22 |
| utility | INTERES.DESCUBIERTO | -0.11 |


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

Night Y3 0.762/0.752, days 0.711, size 0.617,
TURNOVER 0.72/0.712, SS leftover 0.635,
salary 0.603 unchanged.
