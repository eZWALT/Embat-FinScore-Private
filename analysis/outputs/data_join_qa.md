# Data-join QA — can a cross-source Y exist?

Generated `2026-09-19T00:46:50+02:00` by agent `d56ee5fe`. This process 32s; the lane iterated write→run→next-variant in this module for ≥30 minutes before the wave note.
DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only.
Rates, percentiles, and KEEP/PARK cuts use **train** only. Holdout is counted as coverage (descriptive).
No 0–100. No look-ahead columns. No COMP_* ↔ COUNTERPARTY_* map.

## Verdicts

| probe | verdict | why |
| --- | --- | --- |
| amount-match join as a feature | **KEEP** | Payment-month |Δ|≤0.01 hit=0.357 vs random 0.005 (ratio=74×, gap=0.353). Issuance-month hit=0.212 vs random 0.005. KEEP as a company-month *match-rate / unmatched-share* (not a row FK; ~64% of paid invoices still unmatched). Do not explode the panel into matched pairs. Wrong-sign control 0.022; adjacent-month same-sign 0.200 (same-month is higher; sign is not noise). |
| interco (`d_interco_share`) | **CLOSE** | ID spaces are disjoint. Equality join hits 0 rows. Emitting 0 would pretend we measured no intercompany flow. Do not invent a COMP_* ↔ COUNTERPARTY_* map. Shared CP across siblings in a train group: invoice=0, tx=1. |
| “Y8 failed because no join” | **PARK** | Split, not a single story. y8_inv_worse_6 has tx on 98.1% of 3883 labeled rows — CLOSE the 'no cash table' excuse (model still lost to a_in12). y8_cash_worse_6: 2,238 / 6,986 labeled rows (32.0%) are never-ERP (no family E ever); ever-ERP slice n=4,748 base=0.224 vs never-ERP base=0.246. Missing E is a real hole on the 470, not why the ERP slice failed. |

### Numbers to quote (train unless noted)

- **470 confirmation:** 470 train companies with no book invoice — **YES**.
- **Invoice ∩ tx counterparty IDs (train):** 40,670 (34.9% of train invoice CPs; 88.0% of train tx CPs). Train invoice rows whose CP appears on a train tx: 50.0%.
- **Amount-match** payment-month |Δ|≤0.01: hit **35.7%** vs random-partner **0.5%**. Issuance-month: 21.2% vs 0.5%.
- **Y8 other table on labeled rows:** y8_inv←tx 98.1% (3,810/3,883); y8_cash←inv activity 58.0% (4,053/6,986); y8_cash e_* non-null 68.0%.

## Pass 1 — company coverage

### Ever present (book-filter invoices; any transaction)

| split | n_companies | ≥1 invoice | ≥1 tx | both | invoices-only | tx-only | neither |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1,214 | 744 (61.3%) | 1,214 (100.0%) | 744 (61.3%) | 0 | 470 (38.7%) | 0 |
| holdout | 72 | 40 (55.6%) | 72 (100.0%) | 40 (55.6%) | 0 | 32 (44.4%) | 0 |


**470 confirmation (train):** 470 train companies have zero book invoices (`document_type=invoice`, `status<>cancel`, `amount<>0`). Match to the registry number: **YES**.

Holdout (descriptive only): 32 of 72 holdout companies have no invoices.
**Why the 470:** 433 / 470 (92.1%) have NULL `companies.erp`. Of 506 train companies with null ERP, 14.4% still have invoices. Named-ERP but no book: 37 {'dynamicsAx': 18, 'netsuite': 8, 'businessCentral': 6, 'businessOne': 2, 'a3': 1, 'fo': 1, 'm3Rosetta': 1}. Null companies.erp is the main reason for the 470 — not a feature-code drop. A named ERP still without invoices is a minority.

### Are the 470 clustered by group, or mixed with invoiced siblings?

Train groups: **235**. All-invoiced 118, mixed 38, all-dark 79.
Of 470 train no-invoice companies, **110** (23.4%) sit in a mixed group (an invoiced sibling exists). **360** sit in groups where nobody has invoices.

Mixed groups: 38 groups, 110 dark + 230 invoiced companies; median group size 7.5. All-dark groups: 79 / 360 companies (median size 2.0). Size hist of mixed groups: {'1': 0, '2': 0, '3-4': 9, '5-8': 12, '9+': 17}.

Read: no-invoice is **both** a group trait (all-dark holdings) **and** a within-group gap (siblings without ERP). Family H sibling cash can still see the invoiced sister; it cannot invent an invoice book.

### Train company-months on the official monthly grid

Train panel: **21,157** company-months. Any invoice issued/due/paid that month: **54.4%** (11,503). Any issuance that month: 52.8%. Any tx that month: **95.8%** (20,268). **Both: 53.4%** (11,297).

A cross-source Y that needs *the other table that month* can exist on about **53.4% of train company-months**, not the full 21k panel.
Holdout (descriptive only): 1,073 company-months, both-tables 49.4% (530) — LOW_POWER, not used for any cut.

| month | n_cm | any invoice iss/due/paid | any issued | any tx | both |
| --- | --- | --- | --- | --- | --- |
| 2024-09 | 435 | 54.3% | 54.3% | 100.0% | 54.3% |
| 2024-10 | 473 | 54.8% | 53.7% | 98.7% | 54.5% |
| 2024-11 | 483 | 55.9% | 55.5% | 97.9% | 54.9% |
| 2024-12 | 525 | 58.1% | 57.0% | 98.3% | 57.1% |
| 2025-01 | 636 | 57.2% | 55.7% | 98.7% | 56.8% |
| 2025-02 | 677 | 56.9% | 55.7% | 97.9% | 56.1% |
| 2025-03 | 714 | 56.0% | 54.8% | 97.1% | 55.0% |
| 2025-04 | 733 | 56.1% | 53.3% | 97.5% | 55.1% |
| 2025-05 | 752 | 54.8% | 52.1% | 96.9% | 53.9% |
| 2025-06 | 762 | 55.2% | 53.9% | 96.3% | 53.9% |
| 2025-07 | 796 | 54.6% | 53.0% | 97.9% | 54.3% |
| 2025-08 | 833 | 52.2% | 50.3% | 95.4% | 51.0% |
| 2025-09 | 864 | 54.1% | 52.4% | 97.0% | 53.5% |
| 2025-10 | 908 | 54.8% | 53.0% | 96.7% | 54.1% |
| 2025-11 | 945 | 52.8% | 50.6% | 96.0% | 51.4% |
| 2025-12 | 1,006 | 54.1% | 52.3% | 95.6% | 52.8% |
| 2026-01 | 1,132 | 54.3% | 52.2% | 96.3% | 53.5% |
| 2026-02 | 1,204 | 53.0% | 51.7% | 95.3% | 51.9% |
| 2026-03 | 1,211 | 54.1% | 52.8% | 95.2% | 53.0% |
| 2026-04 | 1,212 | 54.1% | 53.2% | 95.6% | 53.5% |
| 2026-05 | 1,214 | 53.6% | 52.9% | 94.5% | 53.0% |
| 2026-06 | 1,214 | 54.1% | 52.9% | 93.6% | 52.9% |
| 2026-07 | 1,214 | 53.9% | 52.0% | 93.2% | 52.7% |
| 2026-08 | 1,214 | 52.4% | 50.3% | 90.0% | 50.3% |


**Quarterly coarsening (train):** 7,945 company-quarters; any invoice 56.4%, any tx 97.7%, both **55.9%** (4,445). Coarser window lifts overlap only a little — the missing ERP is company-level, not a month-timing miss.

### Family D/E store coverage vs raw-table presence (train)

Raw any-invoice-that-month 54.4% vs any `e_*` non-null **64.1%**. `e_ar/ap_issued` non-null 64.1% (zeros filled for ever-invoiced companies); issued>0 52.8% vs raw issuance 52.8%.
Train companies with a book invoice: 744; with any `e_*`: 745. **Raw-invoice companies missing from family E: 0.** Company-months with raw issuance but `e_issued` null: 0.
Invoices-only companies (ERP but no bank tx → off the monthly grid): train 0, holdout 0.

Family E is **not** dropping usable books: coverage is *higher* than raw issuance-that-month because open/issued are zero-filled after the first invoice. The 470 gap is missing ERP, not a feature-code filter.

**Family E (train cm non-null)**

| column | train_cm_nonnull |
| --- | --- |
| e_ar_open | 64.1% |
| e_ap_open | 64.1% |
| e_ar_overdue | 46.4% |
| e_ap_overdue | 57.5% |
| e_ar_overdue_30 | 46.4% |
| e_ap_overdue_30 | 57.5% |
| e_delay_coll | 31.9% |
| e_delay_paid | 40.7% |
| e_dso_proxy | 40.6% |
| e_dpo_proxy | 51.2% |
| e_credit_note_ratio | 53.0% |
| e_pending_amt_share | 60.3% |
| e_fx_share | 52.8% |
| e_ar_issued | 64.1% |
| e_ap_issued | 64.1% |


**Family D (train cm non-null)**

| column | train_cm_nonnull |
| --- | --- |
| d_cust_hhi | 42.2% |
| d_cust_top1 | 42.2% |
| d_n_cust | 53.4% |
| d_supp_hhi | 50.1% |
| d_supp_top1 | 50.1% |
| d_n_supp | 53.4% |
| d_cust_new | 53.6% |
| d_cust_lost | 53.6% |
| d_tx_cp_share | 99.2% |
| d_interco_share | 0.0% |


`d_interco_share` non-null rows on the train store: **0**. Invoice-structure D cols sit at 53.4% — capped by ERP + the incomplete first-5-month 6m window, not by a silent drop.

## Pass 2 — ID / amount join (there is no FK)

### Counterparty ID spaces

Distinct invoice CPs: 123,929. Distinct tx CPs: 47,796. Intersection: **42,101**.
**Train rates:** overlap 40,670 IDs (34.9% of train invoice CPs, 88.0% of train tx CPs). Share of train invoice *rows* whose CP also appears on a train tx: **50.0%**. Share of train tx *rows* whose CP also appears on a train invoice: **8.8%** (most tx rows have NULL CP — 2,305,294 / 2,556,068 all-split).

Companies named `COMP_*`: 1,286; named `COUNTERPARTY_*`: 0. Invoice CPs with `COMP_*` prefix: 0; with `COUNTERPARTY_*`: 885,259. Tx CPs with `COMP_*`: 0.

**COMP_* and COUNTERPARTY_* are disjoint.** Do not write a map.

### `d_interco_share` — one SQL

```sql
SELECT
  (SELECT COUNT(*) FROM invoices i JOIN companies c ON i.counterparty_id = c.company_id) AS inv_eq,
  (SELECT COUNT(*) FROM transactions t JOIN companies c ON t.counterparty_id = c.company_id) AS tx_eq;
```

Result: inv_eq = **0**, tx_eq = **0**. Sets disjoint: **True**. All-null is the honest emission. **CLOSE** interco as a numeric feature.
Shared `counterparty_id` on two train companies in the same group: invoice CPs **0**, tx CPs **1**. Zero would mean siblings do not share customers/suppliers in the ID space either — no interco via a common COUNTERPARTY_* (still not a COMP map).
Shared tx-CP detail (2 company×category rows): [{'company_id': 'COMP_0354', 'group_id': 'GROUP_0220', 'cp': 'COUNTERPARTY_08536', 'category': 'payment', 'n_tx': 17, 'sum_amt': -5475.4, 'd0': '2025-04-08 00:00:00', 'd1': '2026-08-05 00:00:00'}, {'company_id': 'COMP_0909', 'group_id': 'GROUP_0220', 'cp': 'COUNTERPARTY_08536', 'category': 'utility', 'n_tx': 11, 'sum_amt': -1716.7599999999998, 'd0': '2025-04-08 00:00:00', 'd1': '2026-03-05 00:00:00'}]. This is one shared *vendor* (`COUNTERPARTY_*`, payment vs utility) in GROUP_0220 — not a company_id and not intercompany. Still CLOSE.
**Family H on the 470 (train cm):** 7,603 dark company-months; mixed-group dark cm 1,943. `h_n_siblings_active>=1`: all-dark 89.4%, mixed-dark 100.0%, overall 92.1%. H sibling cash is a stand-in only when a sibling is active — all-dark holdings still have no cobros anywhere in the group.

### Amount-match (same company, same calendar month, same sign)

AR `amount>0` ↔ inflow tx; AP `amount<0` ↔ outflow tx. Random control = same invoice amounts matched to a **hash-shifted other train company** in the same month (deterministic, not a learned map). If random ≅ real, the hit is round-number collision.

**Issuance month** vs tx date. 709,267 train invoice rows; 12,893 company-months; exact-amount company-month hit share 69.7%.

| tol_eur | n_inv | n_hit | hit_rate | random_partner_hit | real−random | mean_tx_per_hit_inv | mean_tx_per_hit_random |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 709,267 | 148,900 | 21.0% | 0.4% | 20.6% | 2.26 | 1.64 |
| 0.01 | 709,267 | 150,607 | 21.2% | 0.5% | 20.7% | 2.26 | 1.62 |
| 1.0 | 709,267 | 196,634 | 27.7% | 5.5% | 22.2% | 2.91 | 2.49 |


**Payment-date month** vs tx date. 496,914 paid train invoices.

| tol_eur | n_inv | n_hit | hit_rate | random_partner_hit | real−random | mean_tx_per_hit_inv | mean_tx_per_hit_random |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 496,914 | 176,248 | 35.5% | 0.4% | 35.1% | 1.82 | 1.60 |
| 0.01 | 496,914 | 177,594 | 35.7% | 0.5% | 35.3% | 1.82 | 1.59 |
| 1.0 | 496,914 | 207,343 | 41.7% | 5.8% | 36.0% | 2.44 | 2.75 |


**Stricter variant — unique non-round invoice amounts** (amount not a multiple of 100, appears once in the company-month-sign):

| tol | n_inv | hit | random |
| --- | --- | --- | --- |
| 0.0 | 554,514 | 18.0% | 0.2% |
| 0.01 | 554,514 | 18.3% | 0.3% |
| 1.0 | 554,514 | 24.7% | 5.1% |


**Category filter** (AR↔collection/settlement, AP↔payment/bulk_payment):

| tol | n_inv | hit | random |
| --- | --- | --- | --- |
| 0.0 | 709,267 | 11.2% | 0.2% |
| 0.01 | 709,267 | 11.3% | 0.2% |
| 1.0 | 709,267 | 14.7% | 2.0% |


**Same `counterparty_id` + |Δ|≤0.01:** 75,429 / 708,963 train invoices with a CP (10.6%). Requires resolved tx counterparty (rare). Not a FK; just a tighter collision screen.

**AR vs AP (issuance month, |Δ|≤0.01):**

| side | n_inv | hit | random |
| --- | --- | --- | --- |
| AR | 284,921 | 19.4% | 0.4% |
| AP | 424,346 | 22.5% | 0.5% |


**Due-date month** vs tx date, |Δ|≤0.01: hit 26.7% vs random 0.5% on 669,970 train invoices with a due date.
**Payment-date ±1 day** (not month-binned): 25.0% of 496,914 paid invoices. same company, same sign, |Δamt|≤0.01, |Δdays|≤1 around payment_date.

**Sibling amount match** (same group, other company, same month/sign/cents): 37,760 / 709,267 invoices (5.3%); among invoices in a month where a sibling has txs: 5.9% (639,863 denom). Same group, other company, same month/sign/cents. Not a COMP↔CP map. If this ≪ own-company match, the 21–36% hit is not 'group-shared round numbers'.

**Company-month payment match-rate** (prototype, not written to the store): 9,648 train cm with a paid invoice; mean 41.1%, p50 36.4%, any-hit 85.0%. Spearman vs `log1p(|a_op_in|)` = 0.030. Prototype only (not written to parquet). Spearman vs size; |ρ|>0.85 would be a size proxy.

**745 vs 744:** train companies with any `e_*` minus raw book-invoice companies = 1 ['COMP_0962']. Missing the other way: 0. Detail: [{'company_id': 'COMP_0962', 'document_type': 'refund', 'status': 'overdue', 'n': 1, 'n_zero': 0.0, 'n_no_iss': 0.0}]. COMP_0962 has a single `refund` and no book invoice — family E still emits `e_credit_note_ratio` for that month. Not a dropped book.

**Payment-date timing ladder** (same company, same sign, cents; train paid invoices):

| window | n_inv | n_hit | hit |
| --- | --- | --- | --- |
| exact day | 496,914 | 99,688 | 20.1% |
| ±1 day | 496,914 | 124,122 | 25.0% |
| ±3 days | 496,914 | 141,906 | 28.6% |


**Payment-month hit by fixed amount band** (not a fitted cut):

| band_eur | n_inv | hit |
| --- | --- | --- |
| <10 | 17,962 | 27.8% |
| 10-100 | 92,267 | 36.4% |
| 100-1k | 168,469 | 37.8% |
| 1k-10k | 145,797 | 34.9% |
| >=10k | 72,419 | 31.9% |


**Payment-month hit rate over calendar time** (train):

| month | n_paid_inv | hit |
| --- | --- | --- |
| 2024-09 | 3,951 | 36.5% |
| 2024-10 | 7,333 | 36.2% |
| 2024-11 | 8,661 | 32.7% |
| 2024-12 | 9,627 | 36.4% |
| 2025-01 | 11,387 | 33.9% |
| 2025-02 | 13,097 | 34.5% |
| 2025-03 | 14,625 | 32.1% |
| 2025-04 | 14,608 | 35.1% |
| 2025-05 | 15,245 | 34.1% |
| 2025-06 | 17,448 | 32.5% |
| 2025-07 | 20,515 | 33.7% |
| 2025-08 | 16,808 | 30.9% |
| 2025-09 | 21,366 | 32.3% |
| 2025-10 | 22,706 | 34.8% |
| 2025-11 | 21,422 | 34.0% |
| 2025-12 | 21,940 | 35.4% |
| 2026-01 | 24,447 | 34.8% |
| 2026-02 | 23,558 | 37.0% |
| 2026-03 | 30,549 | 40.5% |
| 2026-04 | 30,864 | 39.2% |
| 2026-05 | 32,760 | 37.2% |
| 2026-06 | 34,108 | 34.7% |
| 2026-07 | 44,145 | 37.1% |
| 2026-08 | 35,744 | 35.4% |


**Per-company payment match-rate** (693 train companies with a paid invoice): p10=8.4%, p25=19.7%, p50=35.9%, p75=55.3%, p90=74.9%. Share ≥50%: 30.4%. Share exactly 0: 3.9%.

**ERP vs bank start (train invoiced companies):** 744 with a book invoice. First invoice after 2024-10-01: 356 (47.8%). First invoice >31 days after first tx: 144 (19.4%). Median (first_inv − first_tx) = -1 days.

**Payment-month match against resolved-CP txs only:** 21.7% (107,778/496,914). Same payment-month probe, but txs must have a resolved counterparty_id. If this collapses toward the 10.6% same-CP figure, most amount hits are on tx rows with a blank CP — still a same-company amount collision, not an ID join.

**Invoiced but never paid in-panel:** 51 of 744 train book-invoice companies have no valid `payment_date` in 2024-09..2026-08 (693 have at least one).

**Train invoice row quality** (n=840,634): type=invoice 86.3%, cancel 1.6%, amount=0 0.0%, no issuance 0.0%, no due 0.0%, no payment_date 31.0%, payment_date_invalid 4.4%, no CP 1.3%.

**Payment match-rate by group mix** (train companies with a paid invoice): mixed-group invoiced siblings p50=39.9% (n=227, mean 42.5%); all-invoiced groups p50=31.8% (n=466, mean 36.9%).

**Controls (same payment-month amounts):** wrong-sign hit 2.2% (10,767/496,914); adjacent month (±1) same-sign hit 20.0% (99,578/496,914). Wrong-sign ≪ same-sign means the AR/AP direction is doing work. Adjacent-month near same-month means the calendar window is loose, not a same-day settlement.

**Payment-month + collection/payment category:** 19.1% (94,952/496,914). Lower than the unfiltered 35.7% means a chunk of amount hits sit in other tx categories (transfer, uncategorised).

`is_extreme` invoices: 10 / 896,711 — cannot drive the 21–36% hit rate.

## Pass 3 — usable overlap for parked / accepted Ys (train)

“Other table” = the table the model is *allowed* to use, not the table that built the label. `y8_inv_worse_6` is an invoice Y → other = tx/cash. `y8_cash_worse_6` is a cash Y → other = invoices. Y5 / Y7 labels are invoice-built → other = tx.

| Y | other | n_labeled | n_pos | base | other present | both tables | other feat | base_on_other | base_on_both |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y8_inv_worse_6 | tx | 3,883 | 761 | 19.6% | 3,810 (98.1%) | 3,590 (92.5%) | 3,810 (98.1%) | 19.2% | 18.9% |
| y8_cash_worse_6 | inv | 6,986 | 1,615 | 23.1% | 4,053 (58.0%) | 3,979 (57.0%) | 4,748 (68.0%) | 22.1% | 21.5% |
| y5_ap_od30_ownp80 | tx | 4,905 | 418 | 8.5% | 4,854 (99.0%) | 4,825 (98.4%) | 4,854 (99.0%) | 8.5% | 8.6% |
| y5_ar_od30_sust | tx | 3,315 | 237 | 7.1% | 3,283 (99.0%) | 3,279 (98.9%) | 3,283 (99.0%) | 7.2% | 7.2% |
| y5_ap_delay_up15 | tx | 3,408 | 153 | 4.5% | 3,371 (98.9%) | 3,355 (98.4%) | 3,371 (98.9%) | 4.5% | 4.5% |
| y7_top1_lost | tx | 7,464 | 2,149 | 28.8% | 7,370 (98.7%) | 7,298 (97.8%) | 7,370 (98.7%) | 28.7% | 28.2% |
| y7_top1_lost_inflow | tx | 6,739 | 420 | 6.2% | 6,706 (99.5%) | 6,650 (98.7%) | 6,706 (99.5%) | 6.1% | 6.0% |


### Y3 `y3_recover_cash_6m` base rate by invoice presence (stressed train rows; no new GBM)

| slice | n_stressed | n_recover | base_rate |
| --- | --- | --- | --- |
| all_stressed | 5,648 | 402 | 7.1% |
| inv_month | 3,168 | 194 | 6.1% |
| no_inv_month | 2,480 | 208 | 8.4% |
| e_feat | 3,618 | 264 | 7.3% |
| no_e_feat | 2,030 | 138 | 6.8% |


### Y8 / Y5 labeled rows by ever-ERP (train)

Never-ERP = the 470. `e_side_ready` = `e_ar_overdue_30` or `e_delay_coll` non-null (the Y8 invoice-side inputs).

| Y | slice | n | n_pos | base | share of labeled |
| --- | --- | --- | --- | --- | --- |
| y8_inv_worse_6 | all | 3,883 | 761 | 19.6% | 100.0% |
| y8_inv_worse_6 | ever_erp | 3,883 | 761 | 19.6% | 100.0% |
| y8_inv_worse_6 | never_erp | 0 | 0 | — | 0.0% |
| y8_inv_worse_6 | e_any | 3,883 | 761 | 19.6% | 100.0% |
| y8_inv_worse_6 | e_side_ready | 3,875 | 761 | 19.6% | 99.8% |
| y8_cash_worse_6 | all | 6,986 | 1,615 | 23.1% | 100.0% |
| y8_cash_worse_6 | ever_erp | 4,748 | 1,065 | 22.4% | 68.0% |
| y8_cash_worse_6 | never_erp | 2,238 | 550 | 24.6% | 32.0% |
| y8_cash_worse_6 | e_any | 4,748 | 1,065 | 22.4% | 68.0% |
| y8_cash_worse_6 | e_side_ready | 3,849 | 801 | 20.8% | 55.1% |
| y5_ap_od30_ownp80 | all | 4,905 | 418 | 8.5% | 100.0% |
| y5_ap_od30_ownp80 | ever_erp | 4,905 | 418 | 8.5% | 100.0% |
| y5_ap_od30_ownp80 | never_erp | 0 | 0 | — | 0.0% |
| y5_ap_od30_ownp80 | e_any | 4,905 | 418 | 8.5% | 100.0% |
| y5_ap_od30_ownp80 | e_side_ready | 4,550 | 374 | 8.2% | 92.8% |


Y5 accepted labels are 100% ever-ERP and ~99% have a tx that month — **CLOSE** “Y5 died because the cash table was missing”. y8_inv is the same (100% ERP, 98% tx). y8_cash is the mixed one: 32% of labels are the 470 (never E); `e_side_ready` (overdue/delay) is present on only 55% of y8_cash labels. That is a population hole, not a missing FK.

Of 2,238 y8_cash never-ERP labeled train rows, **616** (27.5%) sit in a mixed group (sibling has invoices; family H can see sister cash) and 1,622 sit in all-dark groups. Mixed-group never-ERP rows can see sibling cash via family H; all-dark rows cannot see any cobros, sibling or own.

Y3 stressed train, ever-ERP vs never-ERP: ever n=3,618 recover 264 (7.3%); never n=2,030 recover 138 (6.8%). Base rates are close — invoice absence is not a different recovery world.

Y2 `y2_neg_2of3` labeled train: 17,356 rows, never-ERP 6,081 (35.0%), base ever 6.3% vs never 9.1%. Y2 is cash-built so empty E is not an excuse for the parked GBM (it never used E as the Y).

**Unmatched-share vs Y3** (train stressed ever-ERP company-months with a paid-invoice rate; fixed cut 0.5, not a percentile fit). n=2,646. unmatched≥0.5: n=1,672 recover 100 (6.0%); unmatched<0.5: n=974 recover 51 (5.2%). Spearman unmatched vs Y3 = 0.050. A large base-rate gap would keep unmatched-share as a Y3 X candidate; a flat table parks it.

Holdout `y7_top1_lost` (count only): labeled 391, positives 122. LOW_POWER except this count — no rate used to choose a cut.

## Pass 4 — debt schedule / products / next idea

Last panel month: `2026-08-01`. Train companies 1,214, train cm 21,157.

| column | cm non-null | companies ever | last-month cm | cm before last |
| --- | --- | --- | --- | --- |
| f_w_rate | 368 (1.7%) | 38 (3.1%) | 38 (3.1%) | 330 |
| f_util_snapshot | 334 (1.6%) | 334 (27.5%) | 334 (27.5%) | 0 |
| f_months_to_next_pay | 368 (1.7%) | 38 (3.1%) | 38 (3.1%) | 330 |
| f_sched_vs_obs | 368 (1.7%) | 38 (3.1%) | 38 (3.1%) | 330 |


Raw `debt_schedule_config`: 87 rows / 40 companies; `debt_products` companies 378.

**Banking `created_at` left truncation:** of 1,211 train companies with a banking product, **891** (73.6%) have first `created_at` **after 2024-09-01**. No banking product at all: 3. Those trails are left-truncated — product mix in 2024-09 is not a full book.
**The 470 are not 'thin / no-finance' companies:** debt_products 144/470 (30.6%) vs invoiced 28.8%; schedule 13 (2.8%) vs invoiced 3.5%; banking 99.8% vs 99.7%. They lack an ERP book, not a bank book.

### Next idea (legal) or PARK

**Legal next (not same-columns):** do not rebuild Y8 — invoice-side worse-than-own-p80 from cash ranks is the parked model, and y8_inv labeled rows already have tx (98.1% other-table). A legal new Y is `y_erp_gap_then_cash` = existing Y2/Y3 cash stress **among the 470** using only A/B/C/F/G/H (no invented cobros). That answers Q1/Q3 for companies that will never have an invoice book. Payment-month unmatched-share is **PARK as a Y3 X**: among stressed ever-ERP rows the 0.5 cut is flat (see Pass 3). Keep it only as a Q5 diagnostic (share of cobros that never hit the bank book), not as a recovery predictor. Usable both-tables months: 53.4%. Product left-truncation: 73.6% of train banking books start after 2024-09 (family G honesty, not a join).

## Six brief questions

| # | question | what this QA says |
| --- | --- | --- |
| 1 | Who is healthy? | Not this lane. Cross-source health cannot be read on tx-only companies (~39% of train). |
| 2 | Who is improving? | Invoice-side improvement is undefined for the 470. Cash-side still is. |
| 3 | Who is turning? | A turn that needs the *other* table is only defined on the both-tables overlap (53.4% of train cm). |
| 4 | Dip vs fall? | Y8 was the cross-source dip-vs-fall bet. y8_inv labeled rows have cash (98%) and still lost to `a_in12` — not a missing join. y8_cash labels are 32% never-ERP (empty E); that slice cannot see cobros. Amount-match is a real *rate* (35.7% vs 0.5% random) but not a row FK. |
| 5 | Why did it change? | “Why from another table” is honest only on the ERP∩bank subset. Interco is unmeasurable (CLOSE). |
| 6 | Months earlier? | No lead-time claim here. Coverage is a gate, not a lag. |

## What failed / next

Amount-match **KEEP** (hit 35.7% vs random 0.5%). Interco **CLOSE** (0 equality joins). “Y8 failed because no join” **PARK**. Train both-tables month share 53.4%. 470 confirmed=True.
