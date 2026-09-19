# 2026-09-19-0206 — debt schedule snapshot QA

- **Author:** agent 1882a607 (night child)
- **When:** 2026-09-19 02:06 CEST

## What changed

Confirmed NORTH_STAR: utilisation is last-month-only (1.6%); schedule
rate/next-pay/sched_vs_obs are 1.7% but a **thin growing panel**, not
last-month-only (38 train companies, 368 CM). 87 rows / 40 companies
still true. Snapshot columns PARK as X/Y. Q6 CLOSE. Inventory is a
real `created_at` panel (0 drops). `created_at` is connection, not
origination.

## Decisions

- Do not put `f_w_rate` / `f_months_to_next_pay` / `f_sched_vs_obs` /
  `f_util_snapshot` in a GBM.
- Do not revive a utilisation Y (Y10 already parked).
- Keep `f_ds_r` / `f_fc_r` and inventory flags. Q3 CAUTION on
  `f_new_facility`.
- Did not edit `debt.py`.

## Still unknown

None that change the PARK. Settlement 2 orphans and 4 zero-granted
rows are data holes on a parked table.
