# Wave 4 — unused leftover of `d_supp_hhi`

- **When:** 2026-09-19T04:50:32+02:00
- **Agent:** `c9e14b20`
- **Python:** `/home/walterjtv/.pyenv/versions/base/bin/python3`
- **Re-run:** `python -m analysis.evaluate.supp_hhi_qa`
- **Holdout:** 72 companies, seed 20260918. Rates / AUROC on train. Holdout = coverage.
- **Owned:** `analysis/evaluate/supp_hhi_qa.py`, `analysis/outputs/supp_hhi_qa.md`, `analysis/outputs/supp_hhi_quintiles.png`, append-only registry, this note.

## Decision

**DROP** as X. Y3 **DROP** leftover-after-days 0.464. Y4 **DROP** leftover after cust_hhi_lag3 0.523 body 0.523. Y5 **PARK** leftover-after-size 0.525. not SIZE (ρ=-0.356); twin of d_supp_top1 ρ=0.987. Tail vs body: Y5 protective 0.027 vs 0.086 (CONFIRM True); Y4 body 0.523 dead like 0.445. Object: top1 rewrite; Y5 tail is protective (not a Y4 crash). 44 should lose `d_supp_hhi`. Y5 leftover 65% → 65.5% after dropping tail (Δ 0.004). Q6 CLOSE. Night quotes unchanged: Y3 0.762 / 0.752; days 0.711; size 0.617; TURNOVER 0.720 / 0.712.

| object | decision |
| --- | --- |
| Y3 X | **DROP** |
| Y4 X | **DROP** |
| Y5 X | **PARK** |
| the 44 | **DROP from the 44** |
| twin vs top1 | **DROP HHI** ρ=0.987 |
| Y5 tail | **CONFIRM protective** 0.027 vs 0.086 |
| Y4 body | **0.523** (customer body 0.445) |

Coverage train 50.1% ever-n 741. Dark 470 NaN not 0: True. Leftover Y3 after days 0.464; Y4 after cust_hhi_lag3 0.523; Y5 after size 0.525. Y5 65% leftover → 65.5% after drop-tail (Δ 0.004; increased=False).

## What failed

- twin of d_supp_top1 ρ=0.987 — DROP weaker HHI
- Y3 leftover after days 0.464 dies <0.55
- Y4 leftover after cust_hhi_lag3 0.523; body 0.523
- Y5 size-rank AUROC 0.663 ≥0.60 — PARK as Y5 X
- Y5 tail CONFIRM protective — not a Y4-style crash
- dropping the protective tail does not raise the 65% leftover (almost no tail pos)
- Y3 T2+T3 leftover after days 0.417 dies — 0.653 is activity
- Y3 body leftover after days 0.411 dies
- Y5 size-rank AUROC 0.663 CONFIRM vs 0.663
- Q6 short lag3 leftover after the honest bar dies — CLOSE
- Y3 leftover after n_supp+days 0.470 dies

## Next idea

- If the 44 drops HHI, keep `d_supp_top1` only as the Javier concentration object (already PARK as Y5 X / SIZE). Do not stack HHI+top1.
- Do not invent `y_supp_hhi`. Do not merge with Y4.

Elapsed 7s. Night quotes unchanged.
