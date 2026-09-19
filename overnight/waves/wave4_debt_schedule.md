# Wave 4 — debt schedule snapshot QA

Long-lived child. 15 cuts in `analysis/evaluate/debt_schedule_qa.py`.
No parquet rewrite. No new GBM. No 0–100. Y10 utilisation stays parked.

## Files

- `analysis/evaluate/debt_schedule_qa.py` (owner)
- `analysis/outputs/debt_schedule_qa.md`
- `analysis/outputs/debt_schedule_who_when.png`
- registry appends (`agent=1882a607`, coverage + one group-fold diagnostic)

Did **not** edit `debt.py` (no bug). Did **not** run `build_targets`.

## NORTH_STAR sentence

"Debt schedule and utilisation snapshots are last-month thin (~1.7%)" is **half right**:

- `f_util_snapshot` / `f_outstanding_gt_granted`: **last-month only** (100% of non-nulls are 2026-08). Util 1.6% CM / 27.5% companies. OGTG last-month 32/1214 train = 2.6%.
- `f_w_rate` / `f_months_to_next_pay` / `f_sched_vs_obs`: **1.7% CM is correct, last-month-only is wrong**. Thin growing panel: 368 train CM / **38** train companies; only 10.3% of those non-nulls are last month. Coverage 0.9% → 3.1% as `created_at` connects products.

## Coverage (train; holdout count only)

| item | n |
| --- | ---: |
| schedule rows / companies / products | 87 / 40 / 87 (**still true**) |
| train companies with a schedule row | 39 (3.2%) |
| store companies with `f_w_rate` | 38 (COMP_1027 post-extract, dropped) |
| holdout schedule (count only) | 1 (`COMP_1036`, 2 CM) |
| train CM with rate/next-pay/sched_vs_obs | 368 / 21,157 (1.7%) |
| ever `debt_repayment` | 493 |
| repayment **without** a schedule | 459 |
| schedule with no repayment | 5 |
| inventory `f_n_facilities` rises / drops | 555 / **0** (panel) |
| `f_new_facility>0` | 573 CM / 309 companies (2.7%) |

No LOC / no factoring on the schedule table (75 loans). 30 groups; 23 have a single schedule company. Siblings do not share the book.

## Y base rates (train)

Ever-schedule AUROC ≈ 0.50 on y3 / y4 / y9 / y2. Group-fold same (y4 CV 0.519).
Y4 is unlabeled without repayment. Fair split: schedule **18.9%** vs repayment-no-schedule **13.3%** (n=233 vs 2137). Small-n bump, not a why-trail.
OGTG last-month 32 companies: no Y4/Y9 enrichment.
Q5 is **not** a real trail. Q6 is **not** usable.

## Q6 honesty

`next_payment_date` is a frozen extract (81/87 already before 2026-09-01). `f_months_to_next_pay` p50 = −5.6 months, 91.6% negative, steps −1.02 per month. `total_periods` is original tenor, not remaining. **CLOSE** as lead time.

`created_at` is **connection**, not origination: 29/34 schedule+repay companies paid first (median −86 days). `f_new_facility` stays KEEP inventory with **Q3 CAUTION**.

## PARK / CLOSE (snapshot X)

| object | decision |
| --- | --- |
| `f_w_rate` / `f_months_to_next_pay` / `f_sched_vs_obs` as GBM X | **PARK** |
| `f_util_snapshot` as X or Y | **PARK** (Y10 already) |
| `f_outstanding_gt_granted` as Y | **PARK** |
| schedule presence as Y / Q5 | **CLOSE** |
| next_pay / total_periods as Q6 | **CLOSE** |
| `f_ds_r` / `f_fc_r` | **KEEP** (the flow) |
| `f_n_facilities` / `f_has_*` / `f_new_facility` | **KEEP** inventory, Q3 CAUTION |

`f_sched_vs_obs` is already the observed-vs-expected sketch; 31% of the thin panel is clipped at 20. Do not invent a new Y from it.

## What failed / next

Nothing to fix in Family F. Next idea (not this owner): leave the 1.7% snapshot columns out of every GBM spec; if someone wants financing-stress Q3, it is the flow (`f_ds_r` / Y4 / Y9), not the 40-company book.
