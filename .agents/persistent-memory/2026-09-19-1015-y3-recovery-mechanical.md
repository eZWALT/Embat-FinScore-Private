# 2026-09-19-1015 — Y3 recovery is mostly a shrinking outflow base

- **Author:** Javier Boix (with Claude Code)
- **When:** 2026-09-19 ~10:15 CEST, after pulling the overnight push (`fd33be1`)
- **Supersedes:** `2026-09-19-0930-overnight-harness-audit.md` (that audit was of `compare.py` / `search_16h.py`, both deleted in `e943582`; its conclusions match why the teammate replaced them).

## Check

Replicated `y3_recover_cash_6m` from `analysis/targets/y3_recovery.py` on train (n_pos 402, base rate 0.0712, same as `y_acceptance.csv`; 5,643 stressed months, 723 companies, **positives come from 174 companies**). Runway is `liq / max(out3/3, 1)` with `out3` the trailing 3-month operating outflow (`y1_forecast.cash_month_panel`), and the label needs 3 consecutive months of runway >= 3 in t+1..t+6.

Counterfactual: same rule but with the denominator fixed at its value at t, so recovery has to come from liquidity, not from a smaller outflow base.

| | as defined | denominator fixed at t |
|---|---|---|
| base rate among stressed months | 0.0712 | 0.0308 |
| positives that survive | 100% | **29.4%** |
| median out3(t+3) / out3(t), positives vs negatives | **0.099** vs 0.958 | - |
| positives with zero outflow in t+1 vs negatives | **24.9%** vs 4.5% | - |
| AUROC of -log1p(op_out_t) | 0.682 | 0.770 |
| AUROC of log1p(op_in_t) | 0.315 (0.685 inverted) | 0.290 |

## Reading

- About 70% of "recoveries" are a collapse of outflows (median -90% within 3 months, a quarter with zero outflow the next month), not liquidity rebuilding. Going dark counts as recovery.
- The Reading 1 story ("quiet stressed months recover", no payroll / no social security / fewer movement days) is then close to an identity: family C features measure this month's outflow, the label's denominator is built from trailing outflow. Family B is forbidden for Y3, family C is not.
- Size is inverse here (AUROC 0.315 for op_in); `Y3.train_acceptance` accepts on one-sided `auc < 0.60`, so an inverse size proxy passes even though `size_auroc_two_sided` is 0.685 in the same row.
- Not tested: the 0.762 model itself, other Ys. This is a property of the label. Some outflow decline can be genuine cost-cutting, so a corrected label may keep part of the signal.

## Suggested fix (not done)

- Require recovery to hold with the denominator at or above 0.5 x out3(t), and require inflow not to collapse (in3(t+k) >= 0.5 x in3(t)), or define recovery on net flow / liquidity change instead of runway.
- Make the acceptance size gate two-sided.
- Report CIs by group bootstrap; 174 positive companies is the effective n.

## Still unknown

- Whether the organizers' hidden truth looks anything like this Y.
- How much of the 0.762 survives a corrected Y3.
