# Wave 1 slot 6 — Family D counterparties

- **Owner:** `analysis/features/counterparties.py`
- **Note:** `overnight/waves/wave1_slot6.md`
- **Smoke:** `/tmp/slot6_smoke.py` → `/tmp/wave1_slot6_features.parquet`, `/tmp/wave1_slot6_train_coverage.csv`
- **API:** `SOURCE_TABLES=["transactions","invoices","companies"]`, `FAMILY="d"`, `build(con, grid) -> company_id, period, d_*`
- **Python:** `/home/walterjtv/.pyenv/versions/base/bin/python3`

## Files written

- `analysis/features/counterparties.py` — Family D
- this note

Did not edit other wave files, `product/`, or the feature-store assembler.

## Columns

| column | definition | window / rule |
|--------|------------|----------------|
| `d_cust_hhi` | Herfindahl \(\sum_i (amt_i / tot)^2\) on AR invoice `abs(amount)` | last 6 months, `document_type=invoice`, `status<>cancel` |
| `d_cust_top1` | max customer / total AR (same as pipeline `concentration`) | same |
| `d_n_cust` | distinct non-null AR `counterparty_id` | same; `0` if ERP history exists but no AR CPs; NaN if no ERP |
| `d_supp_hhi` | HHI on AP invoices (`amount < 0`) | same 6-month window |
| `d_supp_top1` | max supplier / total AP | same |
| `d_n_supp` | distinct non-null AP `counterparty_id` | same zero/NaN rule as customers |
| `d_cust_new` | AR counterparties in current calendar quarter (through period end) not in previous quarter | NaN until previous quarter is fully inside the dataset (2025-01+) and the company had an invoice by previous-quarter end |
| `d_cust_lost` | AR counterparties in previous quarter not in current | same eligibility |
| `d_tx_cp_share` | share of transactions with non-null `counterparty_id` | same 6-month date window; truncated at 2024-09-01 |
| `d_interco_share` | **all NaN** | see below — no reliable `COMP_*` link |

No look-ahead: invoice `issuance_date` and tx `date` must be `<=` month-end / week-end. 6-month start = first day of (period-end month − 5 months), matching `score_pipeline._invoice_features` concentration. HHI / top-1 / n_* are NaN for 2024-09..2025-01 (incomplete global 6-month window), same cutoff as the pipeline (`i >= 5`).

Invoice amounts only (not mixed with cash). Tx counterparties are not folded into HHI: 90% blank and amounts are a different event.

`d_cust_top1` vs pipeline `concentration` on 9,398 overlapping train/holdout rows: Pearson **0.995**, median |diff| **0**. Residual is the pipeline dropping `due_date` nulls and paid rows with invalid `payment_date`; we keep those invoices if `issuance_date` is present.

## Train coverage

Holdout excluded via `analysis.features.common.train_mask`. Grid: 22,230 company-months (1,286 companies × up to 24 months). **Train: 21,157 company-months, 1,214 companies.**

| column | % company-months | % companies (any non-null) | n CM | mean | p50 |
|--------|------------------|----------------------------|------|------|-----|
| `d_cust_hhi` | 42.2% | 55.5% | 8,928 | 0.521 | 0.478 |
| `d_cust_top1` | 42.2% | 55.5% | 8,928 | 0.613 | 0.624 |
| `d_n_cust` | 53.4% | 61.3% | 11,293 | 30.1 | 4 |
| `d_supp_hhi` | 50.1% | 61.0% | 10,595 | 0.396 | 0.326 |
| `d_supp_top1` | 50.1% | 61.0% | 10,595 | 0.524 | 0.494 |
| `d_n_supp` | 53.4% | 61.3% | 11,293 | 39.7 | 20 |
| `d_cust_new` | 53.6% | 61.2% | 11,338 | 8.20 | 1 |
| `d_cust_lost` | 53.6% | 61.2% | 11,338 | 11.58 | 1 |
| `d_tx_cp_share` | **99.2%** | **100%** | 20,983 | 0.126 | **0.000** |
| `d_interco_share` | **0%** | **0%** | 0 | — | — |

Conditional on 2025-02+ (full 6-month window): HHI/top-1 rise only to 48% / 57% of company-months (still capped by missing ERP). Among the 744 train companies that ever have invoice features, 2025-02+ coverage is 76% (AR HHI), 90% (AP HHI), 96% (`d_n_*`).

HHI and top-1 sit in (0, 1]. Median customer is concentrated (top-1 62%); suppliers are more diverse (top-1 52%, n_supp p50 = 20 vs n_cust p50 = 4).

## Honesty: missing counterparties

**Transactions are almost unused as an ID source.**

- `clean.transactions`: 2,556,068 rows; **2,305,294 (90.2%) `counterparty_id` NULL**. The rest are `COUNTERPARTY_*` (250,774 rows, 47,796 distinct IDs). No empties other than NULL.
- `d_tx_cp_share` mean 0.126, **median 0**. 57.8% of train company-months have *zero* resolved tx counterparties in the 6-month window. A handful of companies are well-filled (p90 0.46). This column is a coverage diagnostic, not a structure feature.

**Invoices are the usable customer/supplier file.**

- `clean.invoices`: 896,711 rows; **11,452 (1.3%) NULL** `counterparty_id`. All non-null IDs are `COUNTERPARTY_*` (123,929 distinct).
- Only **785 / 1,286 companies** appear in invoices at all (784 after `document_type=invoice` / non-cancel). The other **~501 companies have no ERP trail** — invoice columns stay NaN, not zero. That is why invoice-feature company coverage tops out near 61%.
- AR vs AP (invoice, not cancel): 717 companies have AR, 782 have AP. More AP fill than AR, which is why `d_supp_*` coverage > `d_cust_hhi`.

**Shared ID space is real between tx and invoices, not with companies.**

- 42,101 of 47,796 tx counterparty IDs also appear on invoices. The dictionary is right: one `COUNTERPARTY_*` space.
- **0 rows** where `counterparty_id = companies.company_id` (invoices or transactions).
- **0** `COMP_*` tokens in `transactions.description` or `invoices.concept`.
- **0** shared counterparties across two companies in the same `group_id`.
- Custom / `Other (customer-defined)` products exist (305 checking, 171 loans, …) but their transactions still carry `COUNTERPARTY_*` or NULL, never a sibling `COMP_*`.

## `d_interco_share` is NaN on purpose

The only non-invented join is equality: `counterparty_id = company_id` and `group_id` of that company equals the row’s group, `company_id` ≠ self.

That join hits **0 rows**. Emitting `0` would pretend we measured “no intercompany flows”. We did not; we cannot see them. Column is all-NaN. `companies` is still in `SOURCE_TABLES` because `build()` runs this probe.

Did **not** invent: strip digits (`COUNTERPARTY_0001` ↔ `COMP_0001` is also 0), parse `[COMPANY]` placeholders, or treat custom-product flows as interco without an ID.

## What failed / limits

- Intercompany feature: **failed as a numeric column** (no link). Honest NaN.
- Tx-based customer/supplier structure: **not viable** at panel scale (90% blank; median fill 0).
- Invoice features missing for ~39% of companies (no ERP) and for the first five calendar months (short window).
- `d_n_cust` max 2,371 / `d_n_supp` max 3,141: a few firms have huge counterparties lists; HHI still defined. Not cleaned further (`is_extreme` kept, per clean-schema policy).
- Weekly grid works (subset smoke 240 rows, no dups) but the period loop is slower than monthly; verify path is monthly.

## Next idea

- Do not spend another slot on a `COMP_*` interco join unless new identifiers appear.
- Optional later: recover extra tx CPs from `COUNTERPARTY_*` tokens in `description` (same ID space as invoices). That can lift `d_tx_cp_share`; it still cannot label intercompany.
- Add `d_cust_churn = d_cust_lost / n_prev` for Y7 (top-customer disappearance). Forbidden X for Y7 is this family.
- Supplier HHI is the Perez-Salazar-style column; check persistence and size-correlation in `feature_report` before using it as a distress X.
- Family H can do sibling cash aggregates from `group_id` without needing counterparties.
