# Debt schedule + utilisation snapshot QA

Generated `2026-09-18T23:44` UTC by `python -m analysis.evaluate.debt_schedule_qa`.
Holdout 72 (seed 20260918) is **coverage only**. Rates, tertiles, AUROC,
and PARK/CLOSE are train. No parquet rewrite. No new GBM. No 0–100.
Y10 utilisation labels stay parked — this file does not revive them.

## Headline

CORRECT last-month-only for f_util_snapshot / f_outstanding_gt_granted; CORRECT ~1.7% coverage for f_w_rate / f_months_to_next_pay / f_sched_vs_obs; WRONG to call those three last-month-only — they are a thin growing panel (368 train CM / 38 train companies; 10.3% of non-nulls are 2026-08).

- Raw `debt_schedule_config`: **87 rows / 40 companies / 87 products**.
  debt.py citation (~87 rows / 40 companies) is **still true**.
  Train companies with a schedule **row**: **39**. Holdout (count only): **1**.
  Monthly store trains down to **38** — raw-not-store `['COMP_1027']` is the
  post-extract `created_after_snapshot` loan (Family F already drops it).
- Train store: `38` companies / `368` company-months have `f_w_rate`
  (1.7% of 21,157 train CM). Last-month share of those non-nulls: 10.3%.
- `f_util_snapshot` last-month-only: **True** (coverage 1.6%). `f_outstanding_gt_granted` last-month-only: **True**.
- Inventory (`f_n_facilities` / `f_new_facility` / `f_has_loc`) is a **panel**: yes. Facility count rises 555 times and drops 0 times (drops should be rare — as-of `created_at`).
- PARK snapshot columns as GBM X and as a health Y. Keep the *flow* `f_ds_r`.
- Q6: `next_payment_date` / `total_periods` are **not** lead time. next_payment_date is a static extract field (81/87 already before 2026-09-01). f_months_to_next_pay is a countdown to that frozen date and is usually already negative. total_periods is the original installment count, not remaining tenor. Neither is a living lead-time clock (Q6).

## Brief questions

5. **Why did it change?** — only if a schedule vs no-schedule split is a real trail,
   not a 40-company bookkeeping tag. CM AUROC of ever-schedule vs accepted Ys is ~0.50.
   Y4 is a bit higher on schedule companies (they have a debt service by construction).
   That is not a why-trail. Do not treat schedule presence as Q5.
6. **How many months earlier?** — `next_payment_date` is the last-book snapshot,
   usually already past. Countdown `f_months_to_next_pay` is not visibility of a turn.
   Q6 stays closed for these columns.

## 1. Raw `debt_schedule_config` + join to `debt_products`

| item | n |
| --- | ---: |
| schedule rows | 87 |
| distinct companies | 40 (train 39 / holdout 1 `COMP_1036`) |
| distinct product_id | 87 |
| debt_products rows / companies / products | 2239 / 378 / 2239 |
| schedule rows joined on product_id | 87 |
| schedule with no debt_products row | 0 |
| company_id mismatch | 0 |
| schedule product `created_after_snapshot` | 1 |

Join is clean. One schedule product is post-extract and Family F already drops
`created_after_snapshot` in `_schedule_asof`.

### Product-type mix (schedule ∩ debt_products)

No LOC and no factoring on the schedule table. Formal amortisation is almost all loans.

| type | rows | companies | products |
| --- | ---: | ---: | ---: |
| loan | 75 | 36 | 75 |
| leasing | 7 | 2 | 7 |
| guarantee | 2 | 2 | 2 |
| renting | 2 | 2 | 2 |
| mortgage | 1 | 1 | 1 |

## 2. Which months have non-null schedule fields

Train panel: **21,157** company-months / **1214** companies (2024-09 … 2026-08). Holdout coverage only: 1,073 / 72.

| split | column | n non-null | cov CM | n companies | cov companies | n in last month | share of non-nulls in last month | last-month only? |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: | --- |
| train | f_w_rate | 368 | 1.7% | 38 | 3.1% | 38 | 10.3% | False |
| train | f_util_snapshot | 334 | 1.6% | 334 | 27.5% | 334 | 100.0% | True |
| train | f_months_to_next_pay | 368 | 1.7% | 38 | 3.1% | 38 | 10.3% | False |
| train | f_sched_vs_obs | 368 | 1.7% | 38 | 3.1% | 38 | 10.3% | False |
| train | f_outstanding_gt_granted | 1214 | 5.7% | 1214 | 100.0% | 1214 | 100.0% | True |
| train | f_n_facilities | 21157 | 100.0% | 1214 | 100.0% | 1214 | 5.7% | False |
| train | f_new_facility | 21157 | 100.0% | 1214 | 100.0% | 1214 | 5.7% | False |
| train | f_has_loc | 21157 | 100.0% | 1214 | 100.0% | 1214 | 5.7% | False |
| holdout | f_w_rate | 2 | 0.2% | 1 | 1.4% | 1 | 50.0% | False |
| holdout | f_util_snapshot | 13 | 1.2% | 13 | 18.1% | 13 | 100.0% | True |
| holdout | f_months_to_next_pay | 2 | 0.2% | 1 | 1.4% | 1 | 50.0% | False |
| holdout | f_sched_vs_obs | 2 | 0.2% | 1 | 1.4% | 1 | 50.0% | False |
| holdout | f_outstanding_gt_granted | 72 | 6.7% | 72 | 100.0% | 72 | 100.0% | True |
| holdout | f_n_facilities | 1073 | 100.0% | 72 | 100.0% | 72 | 6.7% | False |
| holdout | f_new_facility | 1073 | 100.0% | 72 | 100.0% | 72 | 6.7% | False |
| holdout | f_has_loc | 1073 | 100.0% | 72 | 100.0% | 72 | 6.7% | False |

### Train coverage by calendar month

| period | n companies | f_w_rate | f_months_to_next_pay | f_sched_vs_obs | f_util_snapshot | fac>0 | new>0 | ogtg non-null |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024-09-01 | 435 | 0.9% | 0.9% | 0.9% | 0.0% | 18.2% | 2.8% | 0.0% |
| 2024-10-01 | 473 | 1.1% | 1.1% | 1.1% | 0.0% | 18.8% | 3.0% | 0.0% |
| 2024-11-01 | 483 | 1.0% | 1.0% | 1.0% | 0.0% | 19.3% | 1.4% | 0.0% |
| 2024-12-01 | 525 | 1.1% | 1.1% | 1.1% | 0.0% | 19.2% | 1.5% | 0.0% |
| 2025-01-01 | 636 | 0.9% | 0.9% | 0.9% | 0.0% | 16.4% | 0.9% | 0.0% |
| 2025-02-01 | 677 | 0.9% | 0.9% | 0.9% | 0.0% | 17.4% | 3.2% | 0.0% |
| 2025-03-01 | 714 | 0.8% | 0.8% | 0.8% | 0.0% | 18.6% | 2.8% | 0.0% |
| 2025-04-01 | 733 | 1.0% | 1.0% | 1.0% | 0.0% | 19.2% | 2.2% | 0.0% |
| 2025-05-01 | 752 | 1.1% | 1.1% | 1.1% | 0.0% | 20.7% | 2.8% | 0.0% |
| 2025-06-01 | 762 | 1.0% | 1.0% | 1.0% | 0.0% | 21.7% | 3.9% | 0.0% |
| 2025-07-01 | 796 | 1.1% | 1.1% | 1.1% | 0.0% | 21.6% | 1.9% | 0.0% |
| 2025-08-01 | 833 | 1.2% | 1.2% | 1.2% | 0.0% | 20.9% | 0.8% | 0.0% |
| 2025-09-01 | 864 | 1.5% | 1.5% | 1.5% | 0.0% | 22.3% | 3.5% | 0.0% |
| 2025-10-01 | 908 | 1.5% | 1.5% | 1.5% | 0.0% | 23.8% | 4.7% | 0.0% |
| 2025-11-01 | 945 | 1.6% | 1.6% | 1.6% | 0.0% | 23.9% | 3.1% | 0.0% |
| 2025-12-01 | 1006 | 1.5% | 1.5% | 1.5% | 0.0% | 23.2% | 2.1% | 0.0% |
| 2026-01-01 | 1132 | 1.5% | 1.5% | 1.5% | 0.0% | 21.4% | 2.3% | 0.0% |
| 2026-02-01 | 1204 | 1.7% | 1.7% | 1.7% | 0.0% | 21.3% | 2.4% | 0.0% |
| 2026-03-01 | 1211 | 2.0% | 2.0% | 2.0% | 0.0% | 23.5% | 3.9% | 0.0% |
| 2026-04-01 | 1212 | 2.3% | 2.3% | 2.3% | 0.0% | 25.6% | 2.9% | 0.0% |
| 2026-05-01 | 1214 | 2.6% | 2.6% | 2.6% | 0.0% | 27.0% | 3.7% | 0.0% |
| 2026-06-01 | 1214 | 2.9% | 2.9% | 2.9% | 0.0% | 27.5% | 2.2% | 0.0% |
| 2026-07-01 | 1214 | 3.1% | 3.1% | 3.1% | 0.0% | 29.0% | 3.6% | 0.0% |
| 2026-08-01 | 1214 | 3.1% | 3.1% | 3.1% | 27.5% | 29.2% | 1.6% | 100.0% |

Read: rate / next-pay / sched_vs_obs grow from ~0.9% to 3.1% as `created_at`
brings products onto the book. That is a **thin panel**, not a last-month still.
`f_util_snapshot` and `f_outstanding_gt_granted` are NaN until 2026-08 — Family F
blanks snapshot amounts before `AS_OF`. NORTH_STAR's '~1.7% last-month' sentence
mixes those two facts. The train panel itself is **unbalanced** (435 companies in 2024-09 → 1214 by 2026-05),
so the `fac>0` share can dip when new companies enter without a connected facility.
Within company, `f_n_facilities` never falls.

## 3. Inventory is a panel; amounts are a still

`created_at` decides which facilities exist as-of `period_end`. Counts and type
flags therefore move through 2024–2026. Granted / outstanding / rate / next-pay
on those rows are still the 2026-09-01 extract (Family F already names them snapshot).

| column | cov CM | share > 0 | n companies > 0 | acf1 | acf3 | acf6 |
| --- | --- | --- | --- | --- | --- | --- |
| f_n_facilities | 100.0% | 0.2294 | 354 | 0.852 | 0.730 | 0.534 |
| f_new_facility | 100.0% | 0.02708 | 309 | -0.062 | -0.074 | -0.083 |
| f_has_loc | 100.0% | 0.1173 | 186 | 0.691 | 0.696 | 0.395 |

- `f_n_facilities` month-to-month: 555 rises, 0 drops, 19388 flats. A real as-of inventory path.
- `f_w_rate` is almost a company constant: 33 / 38 train schedule companies have one unique rate (5 change when a second product is created).
- `f_months_to_next_pay` has median 7 distinct values per company
  — the frozen `next_payment_date` minus a moving `period_end`.
- Months of non-null `f_w_rate` per train schedule company: median 7, max 24 (of 24).

## 4. Schedule companies vs observed financing flows

Train companies: **1214**. Ever a schedule row: **39** (3.2%). Ever a repayment / interest / fee transaction: **1055**.
Schedule ∩ `debt_repayment`: 34. Schedule with **no** repayment: 5. Schedule ∩ (fee ∪ interest): 38.

| set | n train companies | share of train | ∩ schedule | share that have a schedule |
| --- | ---: | --- | --- | --- |
| schedule | 39 | 0.03213 | 39 | 1.000 |
| debt_repayment | 493 | 0.4061 | 34 | 0.069 |
| interest_charge | 451 | 0.3715 | 22 | 0.049 |
| fee | 997 | 0.8213 | 38 | 0.038 |
| any_fin_flow | 1055 | 0.869 | 39 | 0.037 |
| debt_product | 354 | 0.2916 | 38 | 0.107 |

A schedule row is rare. Observed debt service is common. The health-relevant
financing trail is already in Family F as `f_ds_r` / `f_fc_r`, not in the 40-row book.

## 5. Accepted Y base rates — schedule vs not

Company-month rates on train. `ever_sched` = company has any `debt_schedule_config` row.
`log1p(a_in3)` is the size control (AUROC vs Y; not a model).

| Y | group | n labeled CM | n pos | base rate | n companies |
| --- | --- | ---: | ---: | --- | ---: |
| y3_recover_cash_6m | schedule | 193 | 11 | 0.05699 | 22 |
| y3_recover_cash_6m | no_schedule | 5455 | 391 | 0.07168 | 703 |
| y4_ds_r_double | schedule | 233 | 44 | 0.1888 | 27 |
| y4_ds_r_double | no_schedule | 2137 | 285 | 0.1334 | 279 |
| y9_fee_r_ownp80 | schedule | 279 | 36 | 0.129 | 23 |
| y9_fee_r_ownp80 | no_schedule | 9312 | 1314 | 0.1411 | 885 |
| y2_neg_2of3 | schedule | 495 | 17 | 0.03434 | 37 |
| y2_neg_2of3 | no_schedule | 16861 | 1254 | 0.07437 | 1158 |

| Y | AUROC ever-schedule | AUROC has-schedule this month | AUROC log1p(a_in3) |
| --- | --- | --- | --- |
| y3_recover_cash_6m | 0.4963 | 0.497 | 0.3805 |
| y4_ds_r_double | 0.5206 | 0.516 | 0.4935 |
| y9_fee_r_ownp80 | 0.4986 | 0.4967 | 0.5336 |
| y2_neg_2of3 | 0.4918 | 0.4944 | 0.5402 |

Company-ever (max of the Y on that company):

| Y | group | n companies with a label | n ever-positive | ever rate |
| --- | --- | ---: | ---: | --- |
| y3_recover_cash_6m | schedule | 22 | 4 | 0.1818 |
| y3_recover_cash_6m | no_schedule | 703 | 170 | 0.2418 |
| y4_ds_r_double | schedule | 27 | 16 | 0.5926 |
| y4_ds_r_double | no_schedule | 279 | 101 | 0.362 |
| y9_fee_r_ownp80 | schedule | 23 | 16 | 0.6957 |
| y9_fee_r_ownp80 | no_schedule | 885 | 478 | 0.5401 |
| y2_neg_2of3 | schedule | 37 | 6 | 0.1622 |
| y2_neg_2of3 | no_schedule | 1158 | 164 | 0.1416 |

Size tertiles (train company-median `log1p(a_in3)`):

| tertile | Y | group | n labeled CM | n pos | base rate | n companies |
| --- | --- | --- | ---: | ---: | --- | ---: |
| T1_small | y3_recover_cash_6m | schedule | 13 | 0 | 0 | 4 |
| T1_small | y3_recover_cash_6m | no_schedule | 1318 | 222 | 0.1684 | 401 |
| T1_small | y4_ds_r_double | schedule | 34 | 7 | 0.2059 | 4 |
| T1_small | y4_ds_r_double | no_schedule | 461 | 87 | 0.1887 | 401 |
| T1_small | y9_fee_r_ownp80 | schedule | 26 | 5 | 0.1923 | 4 |
| T1_small | y9_fee_r_ownp80 | no_schedule | 3387 | 370 | 0.1092 | 401 |
| T1_small | y2_neg_2of3 | schedule | 43 | 0 | 0 | 4 |
| T1_small | y2_neg_2of3 | no_schedule | 5985 | 298 | 0.04979 | 401 |
| T2_mid | y3_recover_cash_6m | schedule | 36 | 3 | 0.08333 | 13 |
| T2_mid | y3_recover_cash_6m | no_schedule | 2048 | 113 | 0.05518 | 391 |
| T2_mid | y4_ds_r_double | schedule | 85 | 22 | 0.2588 | 13 |
| T2_mid | y4_ds_r_double | no_schedule | 744 | 114 | 0.1532 | 391 |
| T2_mid | y9_fee_r_ownp80 | schedule | 69 | 10 | 0.1449 | 13 |
| T2_mid | y9_fee_r_ownp80 | no_schedule | 3216 | 526 | 0.1636 | 391 |
| T2_mid | y2_neg_2of3 | schedule | 141 | 3 | 0.02128 | 13 |
| T2_mid | y2_neg_2of3 | no_schedule | 5746 | 435 | 0.0757 | 391 |
| T3_large | y3_recover_cash_6m | schedule | 144 | 8 | 0.05556 | 22 |
| T3_large | y3_recover_cash_6m | no_schedule | 2089 | 56 | 0.02681 | 383 |
| T3_large | y4_ds_r_double | schedule | 114 | 15 | 0.1316 | 22 |
| T3_large | y4_ds_r_double | no_schedule | 932 | 84 | 0.09013 | 383 |
| T3_large | y9_fee_r_ownp80 | schedule | 184 | 21 | 0.1141 | 22 |
| T3_large | y9_fee_r_ownp80 | no_schedule | 2709 | 418 | 0.1543 | 383 |
| T3_large | y2_neg_2of3 | schedule | 311 | 14 | 0.04502 | 22 |
| T3_large | y2_neg_2of3 | no_schedule | 5130 | 521 | 0.1016 | 383 |

Read: ever-schedule is not a classifier (AUROC ≈ 0.50). The naive Y4 gap
(18.9% schedule vs 13.3% no-schedule) is **not** schedule vs everyone — Y4 is
unlabeled on companies with no repayment (0 labeled no-rep CM).
Fair split: schedule 18.9% vs repayment-no-schedule 13.3%. Small-n bump, still CLOSE. Y2 is *lower* on schedule.
Company-ever Y4 59% vs 36% sits on n=27 labeled schedule companies — do not
promote a split that small. Size-controlled tertiles do not flip the PARK.

## 6. `outstanding > granted`

- `debt_products`: 50 true / 2020 false / 169 null (35 companies).
- `debt_schedule_config`: 6 true (5 companies).
- Train last month `2026-08-01`: `32` / `1214` companies have `f_outstanding_gt_granted` = 1 (2.6%).
- Last-month labeled Y counts (horizon → almost empty): `y3_recover_cash_6m`=0, `y4_ds_r_double`=0, `y9_fee_r_ownp80`=0, `y2_neg_2of3`=0.
  Overlap is therefore company-ever Y among the 32 last-month OGTG companies.

| Y | group | n companies with a label | n ever-positive | ever rate |
| --- | --- | ---: | ---: | --- |
| y3_recover_cash_6m | ogtg | 24 | 3 | 0.125 |
| y3_recover_cash_6m | not_ogtg | 701 | 171 | 0.2439 |
| y4_ds_r_double | ogtg | 17 | 6 | 0.3529 |
| y4_ds_r_double | not_ogtg | 289 | 111 | 0.3841 |
| y9_fee_r_ownp80 | ogtg | 24 | 15 | 0.625 |
| y9_fee_r_ownp80 | not_ogtg | 884 | 479 | 0.5419 |
| y2_neg_2of3 | ogtg | 32 | 5 | 0.1562 |
| y2_neg_2of3 | not_ogtg | 1163 | 165 | 0.1419 |

No material Y4 / Y9 enrichment. Same extract-still as Y10 `y10_ogtg_last_month` (PARK).

## 7. Q6 honesty — lead time

| item | value |
| --- | --- |
| next_payment_date range | 2024-03-19 00:00:00 → 2027-01-22 00:00:00 |
| last_payment_date range | 2024-02-27 15:09:49 → 2026-09-15 17:34:50 |
| next before / on-or-after / after 2026-09-01 | 81 / 6 / 5 |
| last_payment after extract | 1 |
| last_payment > next_payment | 4 (all 4 are the same calendar day — timestamp, not a broken book) |
| total_periods min / mean / max | 1 / 49.5 / 180 |
| granted_balance + / 0 / − | 83 / 4 / 0 |
| f_months_to_next_pay train mean / p50 | -7.65 / -5.55 |
| share of non-null months_to_next < 0 | 91.6% |
| median within-company month step | -1.02 |
| usable as lead time? | **no** |

Amortising frequency / interest type (all 87 rows):

| frequency | n |
| --- | ---: |
| monthly | 79 |
| quarterly | 5 |
| semiannually | 3 |

| interest | n |
| --- | ---: |
| fixed | 58 |
| variable | 29 |

next_payment_date is a static extract field (81/87 already before 2026-09-01). f_months_to_next_pay is a countdown to that frozen date and is usually already negative. total_periods is the original installment count, not remaining tenor. Neither is a living lead-time clock (Q6).

## 8. Do groups share a schedule?

30 groups have at least one schedule company (29 with a train member).
23 groups have exactly one schedule company; 7 have 2–3.
Those groups contain 179 companies of which 40 have a schedule.
Schedule is a company-level tag, not a group book. Siblings usually do **not** share it.

Group-size counts (all / train-member groups): `{1: 23, 2: 4, 3: 3}` / `{1: 22, 2: 4, 3: 3}`.

## 9. `f_new_facility` as a Q3 turning flag (no tree)

Train: 573 company-months (2.7%) across 309 companies have `f_new_facility` > 0.
Median company acf1 of the count: -0.062; of the >0 flag: -0.062 (acf3 -0.074, acf6 -0.083).
AUROC of `log1p(a_in3)` vs the new-facility flag: 0.725 — larger firms connect more products (size-tilted, not a health signal).

Single-feature AUROC of the new-facility flag vs accepted Ys (train, not a GBM):

| Y | AUROC f_new_facility>0 | AUROC log1p(a_in3) | n labeled |
| --- | --- | --- | ---: |
| y3_recover_cash_6m | 0.4843 | 0.3805 | 5648 |
| y4_ds_r_double | 0.506 | 0.4935 | 2370 |
| y9_fee_r_ownp80 | 0.5031 | 0.5336 | 9591 |
| y2_neg_2of3 | 0.4949 | 0.5402 | 17356 |

Same-month coincidence (descriptive):

| Y | n new ∧ Y | rate Y | new | rate Y | no new |
| --- | --- | --- | --- |
| y3_recover_cash_6m | 4 | 0.0181 | 0.07334 |
| y4_ds_r_double | 32 | 0.1553 | 0.1372 |
| y9_fee_r_ownp80 | 36 | 0.1765 | 0.14 |
| y2_neg_2of3 | 23 | 0.04802 | 0.07395 |

`f_new_facility` is a usable *inventory clock* (rare, low persistence, already on the
feature-report keep list as a rare-event flag). It is **not** a new Y and it does
not rescue the snapshot rate / util columns. Dictionary `created_at` is *when the
product was connected*: 29/34 train schedule
companies with a repayment paid **before** their first schedule product was connected
(median lag -86 days). Among any debt_product + repayment, 180/269 also paid first. 283/573 new-facility months are a first-ever connected facility.
Leave it as X inventory with a **CAUTION** on Q3 (connection ≠ origination).
Do not build a turning label from `created_at` here (Y10 already parked `y10_new_loc_after_stress`).

## 10. Settlement, granted holes, 39 vs 38

- Settlement: **77/87** hit `banking_products` (all 77 same-company checking).
  The other 10: 8 settle to another of the company's **debt** products
  (LOC), 2 orphan `settlement_product_id`s (not in bank or debt).

| non-bank settlement dest | n |
| --- | ---: |
| lineofcredit | 8 |
| (orphan) | 2 |

- `granted_balance` vs `abs(debt_products.granted)` match within €1 on **19/87** rows.
  Sign convention: 83 schedule rows have positive granted and negative product granted.
  **4** schedule rows have `granted_balance = 0` but a large outstanding
  (Family F then emits scheduled installment 0, so `f_sched_vs_obs` can be 0 rather than null).
  Not a debt.py bug — the schedule extract is just incomplete. Still PARK the column.
- Raw train schedule companies 39 vs store 38: missing `['COMP_1027']`.
  Post-extract schedule companies: `['COMP_1027']`.
- Schedule with no `debt_repayment` transaction (train): `['COMP_0365', 'COMP_0415', 'COMP_1068', 'COMP_1121', 'COMP_1261']` (n=5).
  All five have fees; the book is not the flow.
- `next_payment_date` before `created_at`: 2 rows.

## 11. `f_sched_vs_obs` clip and snapshot rate vs observed cost

Train non-null `f_sched_vs_obs`: n=368, mean=7.06, p50=1.16,
share exactly 0 = 7.1%, share clipped at 20 = 31.2%.
p10=0.03, p90=20.00. A third of the thin panel sits on the clip —
the snapshot installment is not a historical expected path.

Snapshot `f_w_rate` on the same months: p50=0.037 (min 0.000, max 0.110) — a 0–11% annual rate, not bps.
Spearman vs observed `f_fc_r` = -0.068; vs `f_ds_r` = -0.106.
`f_sched_vs_obs` vs `f_ds_r` = -0.428; vs `f_fc_r` = -0.061.
The extract rate does not track observed financing-cost pressure. Another PARK nail.

## 12. Group-fold single-feature AUROC (protocol.py, not a GBM)

5 group folds, seed 20260918, 1214 train companies. Score used as-is.

| Y | feature | CV AUROC | sd | n folds |
| --- | --- | --- | --- | --- |
| y3_recover_cash_6m | ever_sched | 0.4984 | 0.01394 | 5 |
| y3_recover_cash_6m | has_sched_cm | 0.4984 | 0.006121 | 5 |
| y3_recover_cash_6m | new_facility_gt0 | 0.4854 | 0.004822 | 5 |
| y3_recover_cash_6m | log1p_a_in3 | 0.3834 | 0.06064 | 5 |
| y4_ds_r_double | ever_sched | 0.5195 | 0.0245 | 5 |
| y4_ds_r_double | has_sched_cm | 0.5147 | 0.03092 | 5 |
| y4_ds_r_double | new_facility_gt0 | 0.4991 | 0.03067 | 5 |
| y4_ds_r_double | log1p_a_in3 | 0.4929 | 0.01819 | 5 |
| y9_fee_r_ownp80 | ever_sched | 0.4997 | 0.006151 | 5 |
| y9_fee_r_ownp80 | has_sched_cm | 0.4972 | 0.007909 | 5 |
| y9_fee_r_ownp80 | new_facility_gt0 | 0.504 | 0.004663 | 5 |
| y9_fee_r_ownp80 | log1p_a_in3 | 0.5339 | 0.01786 | 5 |
| y2_neg_2of3 | ever_sched | 0.4949 | 0.0193 | 5 |
| y2_neg_2of3 | has_sched_cm | 0.4974 | 0.01489 | 5 |
| y2_neg_2of3 | new_facility_gt0 | 0.4991 | 0.01689 | 5 |
| y2_neg_2of3 | log1p_a_in3 | 0.5519 | 0.04628 | 5 |

Ever-schedule and same-month schedule stay at chance. `f_new_facility>0` stays at chance.
`log1p(a_in3)` is the only column here that moves (and it is the size control, not a debt signal).
Confirm CLOSE on schedule-as-Y / Q5.

## 13. Flow vs book

Train companies with a LOC on `debt_products`: 186. Schedule ∩ LOC: 22 — the amortisation table is not the credit-line book.
Repayment without a schedule: 459 companies. The observed `f_ds_r` trail is there.

| group | n companies | n CM | f_ds_r p50 | f_ds_r mean | share f_ds_r>0 | f_fc_r p50 | share f_fc_r>0 |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| schedule | 39 | 627 | 0.07687 | 0.2477 | 0.7416 | 0.001276 | 0.7895 |
| repayment_no_schedule | 459 | 8270 | 0.01166 | 0.1748 | 0.5956 | 0.0007018 | 0.7499 |
| neither | 716 | 12260 | 0 | 0 | 0 | 0.0001215 | 0.5419 |

Schedule companies look like the repayment-no-schedule majority on `f_ds_r`, just smaller n.
The book does not add a second financing trail.

## 14. Fair Y4 split + odd schedule rows

| Y | group | n labeled CM | n pos | base rate | n companies |
| --- | --- | ---: | ---: | --- | ---: |
| y3_recover_cash_6m | no_rep | 2633 | 277 | 0.1052 | 716 |
| y3_recover_cash_6m | rep_no_sched | 2822 | 114 | 0.0404 | 459 |
| y3_recover_cash_6m | schedule | 193 | 11 | 0.05699 | 39 |
| y4_ds_r_double | no_rep | 0 | 0 |  | 716 |
| y4_ds_r_double | rep_no_sched | 2137 | 285 | 0.1334 | 459 |
| y4_ds_r_double | schedule | 233 | 44 | 0.1888 | 39 |
| y9_fee_r_ownp80 | no_rep | 5434 | 725 | 0.1334 | 716 |
| y9_fee_r_ownp80 | rep_no_sched | 3878 | 589 | 0.1519 | 459 |
| y9_fee_r_ownp80 | schedule | 279 | 36 | 0.129 | 39 |
| y2_neg_2of3 | no_rep | 10005 | 695 | 0.06947 | 716 |
| y2_neg_2of3 | rep_no_sched | 6856 | 559 | 0.08153 | 459 |
| y2_neg_2of3 | schedule | 495 | 17 | 0.03434 | 39 |

Type mix / holes on the 87-row book (not a model input):

| type | n | granted=0 | ogtg | granted>1e8 |
| --- | ---: | --- | --- | --- |
| loan | 75 | 4 | 5 | 1 |
| leasing | 7 | 0 | 0 | 0 |
| renting | 2 | 0 | 0 | 0 |
| guarantee | 2 | 0 | 1 | 0 |
| mortgage | 1 | 0 | 0 | 0 |

Guarantee / renting / one 300M loan sit on this table. COMP_0415 is a 300M loan with
no `debt_repayment` transaction. Do not treat the 87 rows as a clean amortising book.

## 15. First-birth vs add-on; variable vs fixed

Same-month Y rates when `f_new_facility>0` is a first connected facility vs an add-on.

| Y | kind | n CM | n labeled | n pos | base rate | n companies |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| y3_recover_cash_6m | add_on | 290 | 112 | 0 | 0 | 146 |
| y3_recover_cash_6m | first_birth | 283 | 109 | 4 | 0.0367 | 283 |
| y3_recover_cash_6m | none | 20584 | 5427 | 398 | 0.07334 | 1214 |
| y4_ds_r_double | add_on | 290 | 127 | 20 | 0.1575 | 146 |
| y4_ds_r_double | first_birth | 283 | 79 | 12 | 0.1519 | 283 |
| y4_ds_r_double | none | 20584 | 2164 | 297 | 0.1372 | 1214 |
| y9_fee_r_ownp80 | add_on | 290 | 145 | 24 | 0.1655 | 146 |
| y9_fee_r_ownp80 | first_birth | 283 | 59 | 12 | 0.2034 | 283 |
| y9_fee_r_ownp80 | none | 20584 | 9387 | 1314 | 0.14 | 1214 |
| y2_neg_2of3 | add_on | 290 | 226 | 10 | 0.04425 | 146 |
| y2_neg_2of3 | first_birth | 283 | 253 | 13 | 0.05138 | 283 |
| y2_neg_2of3 | none | 20584 | 16877 | 1248 | 0.07395 | 1214 |

Add-on months have **zero** labeled Y3 recoveries (n=112). First-birth Y9 is a bit
higher (small n). Neither is a turning label. Reinforces Q3 CAUTION: connection clock.

Snapshot rate type on schedule company-months (train):

| interest | n CM | n companies | f_w_rate p50 | f_fc_r p50 | f_ds_r p50 |
| --- | ---: | ---: | --- | --- | --- |
| fixed | 328 | 27 | 0.04 | 0.0008874 | 0.07661 |
| variable | 40 | 11 | 0.02 | 0.0008701 | 0.08559 |

Variable vs fixed does not split observed `f_fc_r`. PARK `f_w_rate` stands.

## PARK / CLOSE

| object | decision | why |
| --- | --- | --- |
| `f_w_rate` as GBM X | **PARK** | 1.7% CM; almost a company constant; snapshot rate |
| `f_months_to_next_pay` as GBM X | **PARK** | 1.7% CM; countdown to a frozen, usually past, date |
| `f_sched_vs_obs` as GBM X | **PARK** | 1.7% CM; numerator is snapshot granted/total_periods |
| `f_util_snapshot` as GBM X or Y | **PARK** | last-month only; Y10 already parked utilisation |
| `f_outstanding_gt_granted` as Y | **PARK** | last-month extract flag; no Y4/Y9 trail |
| schedule presence as Y / Q5 | **CLOSE** | AUROC ~0.50; n=39 train companies |
| `next_payment_date` / `total_periods` as Q6 lead | **CLOSE** | last-book snapshot, not remaining tenor |
| `f_ds_r` / `f_fc_r` flow | **KEEP** | already Family F; this is the observed financing trail |
| `f_n_facilities` / `f_has_*` / `f_new_facility` | **KEEP** (inventory, Q3 CAUTION) | real connection panel; not origination |

A *flow* of observed repayments vs expected installment is already sketched as
`f_sched_vs_obs`. The expected side is the snapshot book, so the ratio cannot be
a historical miss/hit path. Do not invent a new Y from it.

## Plot

- `analysis/outputs/debt_schedule_who_when.png` — train % with a schedule field
  vs % with any facility, and when schedule products were connected.

## Closed in this module (no leftover cut)

- Settlement join: 77 checking + 8 LOC + 2 orphan. Not a trail.
- 4 `granted_balance = 0` rows: leave `f_sched_vs_obs` as-is; column stays PARK.
- `f_n_facilities` drops: **0**. Inventory is monotone within company.
- 39 vs 38: COMP_1027 post-extract only. Not a bug.

