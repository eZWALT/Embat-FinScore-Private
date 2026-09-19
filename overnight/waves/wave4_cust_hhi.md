# Wave 4 — unused leftover of `d_cust_hhi` as Y3 X

- **When:** 2026-09-19T05:10:50+02:00
- **Agent:** `b4e81c2a`
- **Python:** `/home/walterjtv/.pyenv/versions/base/bin/python3`
- **Re-run:** `python -m analysis.evaluate.cust_hhi_qa`
- **Holdout:** 72 companies, seed 20260918. Rates / AUROC on train. Holdout = coverage / mix.
- **Owned:** `analysis/evaluate/cust_hhi_qa.py`, `analysis/outputs/cust_hhi_qa.md`, `analysis/outputs/cust_hhi_qa.png`, append-only registry, this note.

## Decision

**DROP** as engine X. Y3 **DROP** leftover-after-days 0.419. Y4 engine X **DROP** leftover after top1_lag3 0.549 body 0.445 (CONFIRM 0.445). Y4 tail footnote **KEEP** (0.221 vs 0.115; lag3 0.605). not SIZE (ρ=-0.179); twin of d_cust_top1 ρ=0.994 / lag3 0.994. Object: top1 rewrite; Y4 0.605 is the >0.975 monopoly tail, not a gradient. 44 should lose `d_cust_hhi` as engine X; Y4 tail footnote may stay. Q6 CLOSE. Night quotes unchanged: Y3 0.762 / 0.752; days 0.711; size 0.617; TURNOVER 0.720 / 0.712.

| object | decision |
| --- | --- |
| Y3 X | **DROP** |
| Y4 engine X | **DROP** |
| Y4 tail footnote | **KEEP** |
| the 44 as X | **DROP from the 44 as engine X** |
| twin vs top1 | **DROP HHI as X** ρ=0.994 |
| Y4 body | **0.445** CONFIRM True |
| Q6 | **CLOSE** |

Coverage train 42.2% ever-n 674. Dark 470 NaN not 0: True. Leftover Y3 after days 0.419; Y4 after top1_lag3 0.549. Y4 lag3 0.605.

## What failed

- twin of d_cust_top1 ρ=0.994 / lag3 0.994 — DROP weaker HHI as X
- Y3 leftover after days 0.419 dies <0.55
- Y4 leftover after top1_lag3 0.549; body 0.445
- Y4 body CV 0.445 CONFIRM vs 0.445
- Y4 tail CONFIRM crash 0.221 vs 0.115
- Q6 short lag3 LOW_POWER present=0.217 pos=42
- supp HHI is a different object (protective) — do not merge
- same-n leftover after top1 is a near-identity (R²≥0.90)
- Y3 body leftover after days 0.344 dies
- Y3 T2+T3 leftover after days 0.383 dies
- Q6 short leftover after the honest bar dies — CLOSE
- holdout mix flip CONFIRM crash=0.375 spike=0.875
- Y3 leftover after n_cust+days+top1 0.445 dies
- Y4 body leftover after top1 0.495 dies
- days leftover after HHI 0.714 survives — days is the bar
- Y3 HHI_lag3 leftover after days 0.527 dies
- Y4 leftover after tail-flag 0.464 dies — 0.605 is the bin
- Y4 now-body leftover after days 0.569 ρ(resid,top1)=0.946; after days+top1 0.480

## Next idea

- If the 44 drops HHI as X, keep the Y4 >0.975 tail footnote only. Do not stack HHI+top1. Do not invent `y_cust_hhi`. Do not reopen trees.

Elapsed 5s. Night quotes unchanged.
