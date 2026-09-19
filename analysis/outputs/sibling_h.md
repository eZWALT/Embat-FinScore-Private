# Family H — sister existence vs sister state (110 vs 360)

- **When:** 2026-09-19T01:56
- **Agent:** `5d5b1b81`
- **Python:** `/home/walterjtv/.pyenv/versions/base/bin/python3`
- **Re-run:** `python -m analysis.evaluate.sibling_h`
- **Holdout:** 72 companies, seed 20260918. Coverage only. Rates and cuts on train.
- **Y:** accepted `y3_recover_cash_6m` / `y2_neg_2of3` from `targets.parquet` (no assembler).
- **X question:** Family H as Y3 X (never B) vs a descriptive Q5 footnote.
- **Brief:** Q5 why / Q3 turning. Hidden test is **new groups**.
- **Not:** 0–100, `product/`, parquet rewrite, GBM tree bake-off, per-group models, y11 merge.

## Decision

**PARK H as Y3 X; KEEP as a descriptive Q5 footnote**

- **H as Y3 X:** PARK. `h_sib_neg_share` CV 0.434 vs `c_n_days_with_tx` 0.711; mixed dummy 0.494; `h_n_siblings_active` ρ 0.979 vs `h_group_size`; `h_share_group_in` is NEAR_SIZE (ρ 0.747 vs own A). Not a B-copy (worst |ρ| vs `b_runway` 0.111).
- **Q5 footnote:** KEEP sister *mean B-runway ≥ 1* on the 110 (after-size +6.97pp; company-level ever-recover 48.65% vs 13.51%). Family H does not carry that state (`h_sib_neg_share` gap +0.01pp). Sister B is forbidden as Y3 X. Invoiced firms in mixed groups recover *less* (4.97% vs 8.92%) — the 110 lift is dark-side, not a healthy-holding dummy.

If the useful signal is “this group has an invoiced sister”, it **cannot** transfer to the hidden test of new groups.

Sister *B-runway* (not Family H, not sister existence) moves Y3 +6.97pp after own-size terciles inside the 110 (company-level ever-recover gap +35.14pp). `h_sib_neg_share` on the same 110 is flat (+0.01pp). Do not add H or sister-B to Y3 X. H as stored is a group-type dummy; the Q5 keep is sister liquidity, which Y3 X forbids.

## Pass 1 — reproduce 360 / 110 / 744

Book filter = y11 (invoice, not cancel, amount ≠ 0, issuance present). confirm_470=True. Train invoiced **744**. Dark **470** = **360** all-dark (79 groups) + **110** mixed (38 groups).

Y3 is stressed-only (NaN if not stressed or no t+6). Y2 is the accepted 2-of-3 negative-liq label.

| slice | n labeled | pos | cos | Y3 recover |
|---|---:|---:|---:|---:|
| full_train | 5,648 | 402 | 725 | 7.12% |
| invoiced_744 | 3,618 | 264 | 463 | 7.30% |
| dark_470 | 2,030 | 138 | 262 | 6.80% |
| all_dark_360 | 1,557 | 81 | 188 | 5.20% |
| mixed_110 | 473 | 57 | 74 | 12.05% |

Y2 `y2_neg_2of3` on the same company slices:

| slice | n labeled | pos | cos | Y2 base |
|---|---:|---:|---:|---:|
| full_train | 17,356 | 1,271 | 1,195 | 7.32% |
| invoiced_744 | 11,275 | 715 | 736 | 6.34% |
| dark_470 | 6,081 | 556 | 459 | 9.14% |
| all_dark_360 | 4,489 | 426 | 352 | 9.49% |
| mixed_110 | 1,592 | 130 | 107 | 8.17% |

Y3 all-dark 5.20% vs mixed 12.05% (gap +6.85pp). Y2 all-dark 9.49% vs mixed 8.17% (gap -1.32pp).

## Pass 2 — size control (`log1p(a_in3)` terciles, train edges)

Tercile cuts from **all train company-months with finite `a_in3`**. Holdout never enters a cut. Residual = mixed − all-dark inside the tercile.

| tercile | slice | n | pos | cos | Y3 recover | median log1p(a_in3) |
|---|---|---:|---:|---:|---:|---:|
| T1_small | all_dark | 276 | 23 | 57 | 8.33% | 9.052 |
| T1_small | mixed | 116 | 29 | 32 | 25.00% | 9.477 |
| T2_mid | all_dark | 558 | 26 | 96 | 4.66% | 12.763 |
| T2_mid | mixed | 129 | 13 | 34 | 10.08% | 12.498 |
| T3_large | all_dark | 689 | 30 | 99 | 4.35% | 14.747 |
| T3_large | mixed | 217 | 13 | 38 | 5.99% | 14.801 |

Residuals (mixed − all-dark):

| tercile | n all-dark | n mixed | Y3 all-dark | Y3 mixed | residual |
|---|---:|---:|---:|---:|---:|
| T1_small | 276 | 116 | 8.33% | 25.00% | +16.67pp |
| T2_mid | 558 | 129 | 4.66% | 10.08% | +5.42pp |
| T3_large | 689 | 217 | 4.35% | 5.99% | +1.64pp |

T1 residual **+16.67pp**. Corrects the y11 +12.5pp (that cut used `log1p(|op_in|)` on dark-labeled rows, not `a_in3` on all train).

Same terciles on Y2:

| tercile | n all-dark | n mixed | Y2 all-dark | Y2 mixed | residual |
|---|---:|---:|---:|---:|---:|
| T1_small | 1,108 | 555 | 6.05% | 5.77% | -0.28pp |
| T2_mid | 1,300 | 324 | 12.62% | 10.49% | -2.12pp |
| T3_large | 1,379 | 499 | 9.14% | 7.62% | -1.52pp |

Plot: `y3_rate_mix_size_tercile.png`.


## Pass 3 — sister *state* among the 110 (not sister existence)

Invoiced sisters only, same `group_id` × `period`. Dark companies are never in the sister pool. Healthy = sister mean `b_runway` ≥ 3 / sister mean `a_io_ratio` ≥ 1 / no invoiced sister with `y2_neg_2of3`=1. Stressed = the complement among rows where the sister feature is defined. If H only marks “has a sister”, this gap is a **group-type dummy**, not why.

Among mixed-dark train company-months, invoiced sister present this month: 1,943 / 1,943 (100.00%). Median invoiced sisters/month = 4.0.

| sister cut | n healthy | pos | Y3 healthy | n stressed | pos | Y3 stressed | gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| runway>=3 | 192 | 28 | 14.58% | 273 | 28 | 10.26% | +4.33pp |
| runway>=1 | 257 | 40 | 15.56% | 208 | 16 | 7.69% | +7.87pp |
| io_ratio>=1 | 286 | 32 | 11.19% | 179 | 24 | 13.41% | -2.22pp |
| sister_not_y2 | 338 | 45 | 13.31% | 135 | 12 | 8.89% | +4.42pp |

Sister-state residual inside the same train `a_in3` terciles (110 only):

| cut | tercile | n healthy | n stressed | Y3 healthy | Y3 stressed | residual |
|---|---|---:|---:|---:|---:|---:|
| runway>=3 | T1_small | 45 | 71 | 28.89% | 22.54% | +6.35pp |
| runway>=3 | T2_mid | 38 | 91 | 13.16% | 8.79% | +4.37pp |
| runway>=3 | T3_large | 108 | 109 | 8.33% | 3.67% | +4.66pp |
| runway>=1 | T1_small | 64 | 52 | 28.12% | 21.15% | +6.97pp |
| runway>=1 | T2_mid | 61 | 68 | 16.39% | 4.41% | +11.98pp |
| runway>=1 | T3_large | 131 | 86 | 8.40% | 2.33% | +6.07pp |
| io_ratio>=1 | T1_small | 72 | 44 | 22.22% | 29.55% | -7.32pp |
| io_ratio>=1 | T2_mid | 66 | 63 | 6.06% | 14.29% | -8.23pp |
| io_ratio>=1 | T3_large | 146 | 71 | 7.53% | 2.82% | +4.72pp |
| sister_not_y2 | T1_small | 80 | 36 | 27.50% | 19.44% | +8.06pp |
| sister_not_y2 | T2_mid | 102 | 27 | 12.75% | 0.00% | +12.75pp |
| sister_not_y2 | T3_large | 149 | 68 | 5.37% | 7.35% | -1.98pp |

Best sister-state move after size (max |T1/T2/T3| residual, runway/io/not-y2): **+8.06pp**. KEEP-as-Q5 bar is ≥5pp after size. Clears the bar.

Same sister cuts on Y2 (110, train labeled):

| sister cut | n healthy | Y2 healthy | n stressed | Y2 stressed | gap |
|---|---:|---:|---:|---:|---:|
| runway>=1 | 874 | 7.67% | 593 | 8.09% | -0.43pp |
| io_ratio>=1 | 866 | 6.81% | 601 | 9.32% | -2.50pp |
| sister_not_y2 | 1,174 | 7.41% | 418 | 10.29% | -2.88pp |

## Pass 4 — leak screen (H vs own A / own B)

Y3 X never B. Flag B-copy if |ρ| vs `b_runway` ≥ 0.80 (NEAR at 0.50). SIZE if |ρ| vs `a_op_in` / `a_in3` ≥ 0.85 (NEAR at 0.70).

| slice | H | vs | n | Spearman ρ | flags |
|---|---|---|---:|---:|---|
| train_y3 | h_sib_in | a_op_in | 5,648 | 0.060 | — |
| train_y3 | h_sib_in | a_in3 | 5,528 | 0.086 | — |
| train_y3 | h_sib_in | b_runway | 5,528 | -0.111 | — |
| train_y3 | h_sib_neg_share | a_op_in | 5,408 | -0.006 | — |
| train_y3 | h_sib_neg_share | a_in3 | 5,296 | -0.028 | — |
| train_y3 | h_sib_neg_share | b_runway | 5,296 | -0.002 | — |
| train_y3 | h_share_group_in | a_op_in | 5,583 | 0.747 | NEAR_SIZE |
| train_y3 | h_share_group_in | a_in3 | 5,468 | 0.600 | — |
| train_y3 | h_share_group_in | b_runway | 5,468 | 0.106 | — |

Worst |ρ| vs own B (`b_runway`) on train Y3-labeled: **0.111** (`h_sib_in`). not a B-copy.
Worst |ρ| vs own size on train Y3-labeled: **0.747** (`h_share_group_in vs a_op_in`). On the 110 only, `h_share_group_in` vs `a_op_in` is SIZE (ρ 0.876).

## Pass 5 — single-feature train group-fold AUROC (Y3 stressed)

Sign from the train side of each fold. Compare to `c_n_days_with_tx` quoted **0.711**. KEEP as Q5 only if the H / mixed dummy beats a size or group-size dummy by ≥ 0.02 and is not a B-copy.

| feature | CV AUROC | sd | train AUROC | sign | coverage | vs 0.711 | vs size dummy | vs group-size dummy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h_sib_neg_share | 0.434 | 0.066 | 0.504 | -1 | 95.75% | -0.277 | -0.183 | -0.118 |
| mixed_dummy | 0.494 | 0.040 | 0.511 | -1 | 100.00% | -0.217 | -0.123 | -0.058 |
| log1p_a_in3 | 0.617 | 0.061 | 0.620 | -1 | 97.88% | -0.094 | 0.000 | 0.065 |
| h_group_size | 0.552 | 0.119 | 0.580 | -1 | 100.00% | -0.159 | -0.065 | 0.000 |
| c_n_days_with_tx | 0.711 | 0.031 | 0.723 | -1 | 100.00% | 0.000 | 0.095 | 0.159 |
| h_n_siblings_active | 0.548 | 0.118 | 0.575 | -1 | 100.00% | -0.163 | -0.069 | -0.004 |
| h_share_group_in | 0.638 | 0.073 | 0.639 | -1 | 98.85% | -0.073 | 0.022 | 0.086 |
| h_sib_in | 0.551 | 0.074 | 0.569 | -1 | 100.00% | -0.160 | -0.066 | -0.001 |

`h_sib_neg_share` CV **0.434**. Mixed-group dummy CV **0.494**. Size dummy `log1p(a_in3)` CV **0.617**. Group-size dummy CV **0.552**. `c_n_days_with_tx` CV **0.711**.
H/mixed beat size+group-size by ≥0.02: **False**.

## Extra cuts (same module)

### Are all-dark holdings larger groups?

| slice | cos | groups | median group_n | mean group_n | median h_group_size | median h_n_siblings_active | median a_in3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_dark_360 | 360 | 79 | 2.000 | 4.557 | 9.000 | 8.000 | 361385.490 |
| mixed_110 | 110 | 38 | 7.500 | 8.947 | 11.000 | 9.000 | 237921.660 |
| invoiced_744 | 744 | 156 | 3.000 | 5.474 | 11.000 | 9.000 | 241638.325 |

All-dark holdings are *smaller* groups (median 2.0 vs mixed 7.5). Mixed dark companies are the small firms inside large groups (median a_in3 237922 vs all-dark 361385).

### Is `h_n_siblings_active` just `h_group_size`?

Train Spearman `h_n_siblings_active` vs `h_group_size` = **0.979** (n=21,157). Feature report quoted 0.98. Yes — near-copy of group size, not a live sibling-activity lever.

Train Spearman `h_sib_in` vs `h_sib_out` = **0.925** (report 0.92). `h_share_group_in` vs `log1p(a_in3)` = **0.677** (report size ρ 0.792 vs log inflow).

### Holdout coverage (not a rate claim)

Holdout companies 72. Dark 32. H columns defined on holdout company-months: `h_sib_neg_share` 89.75%, `h_sib_in` 100.00%. Y3 labeled holdout n=235 pos=14 (LOW_POWER).

### y11-style size cut (confirm or correct +12.5pp)

Terciles of `log1p(|a_op_in|)` fit on **train-dark Y3-labeled** only — the y11 recipe.

| tercile | n all-dark | n mixed | Y3 all-dark | Y3 mixed | residual |
|---|---:|---:|---:|---:|---:|
| T1_small | 484 | 193 | 8.26% | 20.73% | +12.46pp |
| T2_mid | 558 | 118 | 4.12% | 6.78% | +2.66pp |
| T3_large | 515 | 162 | 3.50% | 5.56% | +2.06pp |

T1 residual **+12.46pp**. Confirms the quoted +12.5pp.

Same idea with `log1p(a_in3)` edges from train-dark Y3-labeled (not all train CM):

| tercile | n all-dark | n mixed | Y3 all-dark | Y3 mixed | residual |
|---|---:|---:|---:|---:|---:|
| T1_small | 483 | 179 | 7.25% | 20.67% | +13.42pp |
| T2_mid | 546 | 115 | 3.85% | 6.09% | +2.24pp |
| T3_large | 494 | 168 | 4.66% | 6.55% | +1.89pp |

T1 residual **+13.42pp**.

### Sister runway ≥ 1 after size (110)

This is the strongest raw sister-state cut (+7.87pp). Same train `a_in3` terciles.

| tercile | n sister ok | n sister stressed | Y3 sister ok | Y3 sister stressed | residual |
|---|---:|---:|---:|---:|---:|
| T1_small | 64 | 52 | 28.12% | 21.15% | +6.97pp |
| T2_mid | 61 | 68 | 16.39% | 4.41% | +11.98pp |
| T3_large | 131 | 86 | 8.40% | 2.33% | +6.07pp |

Weighted residual **+7.95pp**; T1 **+6.97pp** (cells ≥20: True). After-size quote **+6.97pp**. Sister runway vs own `log1p(a_in3)` ρ=0.102; vs own `b_runway` ρ=-0.188 (not an own-B copy). Sister `a_io_ratio` went the *wrong* way (−2.2pp) — do not KEEP on IO.

### H-legal sister-state: `h_sib_neg_share` median split on the 110

Train-110 Y3 median = 0.500. Low share (fewer red sisters) vs high:
 Y3 12.05% (n=224) vs 12.05% (n=249), gap +0.01pp. Weighted after size -0.64pp.

| tercile | n low-neg | n high-neg | Y3 low-neg | Y3 high-neg | residual |
|---|---:|---:|---:|---:|---:|
| T1_small | 63 | 53 | 22.22% | 28.30% | -6.08pp |
| T2_mid | 41 | 88 | 9.76% | 10.23% | -0.47pp |
| T3_large | 114 | 103 | 7.02% | 4.85% | +2.16pp |

If H only encodes “has a sister”, this split should be flat. It is flat enough that H is not the why.

### Is the 110 gap just bigger groups?

`h_group_size` terciles fit on all train. Residual mixed − all-dark:

| tercile | n all-dark | n mixed | Y3 all-dark | Y3 mixed | residual |
|---|---:|---:|---:|---:|---:|
| T1_small | 473 | 92 | 9.30% | 15.22% | +5.92pp |
| T2_mid | 538 | 316 | 6.69% | 8.86% | +2.17pp |
| T3_large | 546 | 65 | 0.18% | 23.08% | +22.89pp |

Mixed dummy on dark Y3 only (existence among the 470): CV 0.566 / train 0.597. A group-type dummy can look useful on *this* panel and still fail on new groups.

Company-month `h_group_size` medians (9 vs 11) are weighted by large groups. Company-level median group_n is 2.0 vs 7.5 — all-dark holdings are smaller groups.

### Sister size vs sister health (110 Y3)

Sister `b_runway` vs sister `log1p(a_in3)` ρ=-0.089 (not a sister-size clone).

| sister a_in3 tercile | n | pos | Y3 recover | share sister runway≥1 | median sister log1p(a_in3) |
|---|---:|---:|---:|---:|---:|
| T1_small | 155 | 14 | 9.03% | 56.77% | 12.085 |
| T2_mid | 155 | 23 | 14.84% | 52.90% | 13.589 |
| T3_large | 155 | 19 | 12.26% | 56.13% | 14.368 |

### Sister runway ≥ 1 inside *group-size* terciles (110)

If the state gap is just “bigger holding”, it dies here.

| group-size tercile | n sister ok | n sister stressed | Y3 sister ok | Y3 sister stressed | residual |
|---|---:|---:|---:|---:|---:|
| T1_small | 44 | 47 | 15.91% | 14.89% | +1.02pp |
| T2_mid | 157 | 152 | 14.01% | 3.29% | +10.72pp |
| T3_large | 56 | 9 | 19.64% | 44.44% | -24.80pp |

Weighted residual inside group-size bins **+3.86pp**.

### Single-feature on the 110 Y3 rows only (descriptive; not Y3 X)

Sister `b_runway` is family B of the *sister*. Y3 X never B — this rank is a Q5 footnote, not a column to add.

| feature | CV AUROC | sd | train AUROC | sign | n |
|---|---:|---:|---:|---:|---:|
| sister_runway | 0.662 | 0.147 | 0.581 | 1 | 465 |
| sister_io | 0.371 | 0.174 | 0.542 | -1 | 465 |
| sister_y2 | 0.599 | 0.125 | 0.564 | -1 | 473 |
| sister_in3 | 0.319 | 0.118 | 0.524 | 1 | 465 |
| h_sib_neg_share | 0.433 | 0.125 | 0.513 | -1 | 473 |
| n_inv_sisters_month | 0.603 | 0.131 | 0.599 | 1 | 473 |

### Y2: sister runway ≥ 1 after size (110)

Y2 is the stress label. A sister-state why should not just be “sister Y2”.

| tercile | n sister ok | n sister stressed | Y2 sister ok | Y2 sister stressed | residual |
|---|---:|---:|---:|---:|---:|
| T1_small | 305 | 250 | 8.20% | 2.80% | +5.40pp |
| T2_mid | 197 | 127 | 5.08% | 18.90% | -13.82pp |
| T3_large | 319 | 180 | 9.72% | 3.89% | +5.83pp |

### Company-level sister state (74 mixed Y3 companies)

Company-months can repeat the same holding. Split companies by mean sister `b_runway` (median 1.867).

- Companies with mean sister runway ≥ median: ever-recover **48.65%** (37 cos); mean CM rate 29.18%.
- Below median: ever-recover **13.51%** (37 cos); mean CM rate 10.04%.
- Gap ever-recover +35.14pp; CM-rate gap +19.15pp.
- Sister runway vs sister Y2 ρ=-0.332; vs n invoiced sisters ρ=0.407.

If the company-level ever-recover gap is small, the CM +7pp is a few months in the same holdings. Sister runway is **not** a Y2 clone of the sister (ρ well below 0.80).

### Sister aggregation (mean vs min vs max vs largest)

Mean can hide one healthy invoiced sister among four. Min = all sisters ok. Largest = invoiced sister with the highest `a_in3` this month.

| cut | n healthy | Y3 healthy | n stressed | Y3 stressed | gap |
|---|---:|---:|---:|---:|---:|
| mean_rw>=1 | 257 | 15.56% | 208 | 7.69% | +7.87pp |
| min_rw>=1 (all sisters ok) | 53 | 9.43% | 412 | 12.38% | -2.94pp |
| max_rw>=1 (any sister ok) | 306 | 14.38% | 159 | 7.55% | +6.83pp |
| largest_rw>=1 | 151 | 13.25% | 314 | 11.46% | +1.78pp |
| mean_rw>=3 | 192 | 14.58% | 273 | 10.26% | +4.33pp |

Sister-runway group-fold on the 110: CV 0.662 ± 0.147 (folds [0.875, 0.539, 0.51, 0.715, 0.671]). n_pos per fold is small — quote the CM rate gap and the company-level ever-recover split, not 0.662 as an engine number.

### Q6: does sister runway lead by a month?

Contemporaneous sister runway≥1: Y3 15.56% vs 7.69% (gap +7.87pp, n=257/208).
Lag-1 sister runway≥1: Y3 14.92% vs 7.81% (gap +7.11pp, n=248/192).
110-only CV now 0.662 vs lag1 0.679.

Lag-1 still clears +5pp — a short Q6 lead on the 110, still sister B, still not H, still no transfer to new groups.

### Do invoiced companies in mixed groups also recover more?

If yes, the 110 gap is a **mixed-group** type, not “H lets a dark firm see sister cash”.

- Invoiced in mixed groups: Y3 **4.97%** (n=1,488 / pos=74 / cos=166).
- Invoiced in all-invoiced groups: Y3 **8.92%** (n=2,130 / pos=190 / cos=297).
- Dark in mixed (the 110): Y3 **12.05%** (n=473).
- Invoiced mixed − all-invoiced gap **-3.95pp**.

Invoiced firms in mixed groups do not share the 110 lift — the gap is on the dark side.


## Verdict for H as Y3 X vs Q5 footnote

PARK H as a Y3 X: `h_n_siblings_active` ρ vs `h_group_size` = 0.979; `h_sib_neg_share` CV 0.434 loses to size 0.617 and to `c_n_days_with_tx` 0.711; mixed-group dummy CV 0.494; `h_share_group_in` CV 0.638 is NEAR_SIZE (ρ vs a_op_in 0.747) and SIZE on the 110. Sister *B-runway* (not Family H, not sister existence) moves Y3 +6.97pp after own-size terciles inside the 110 (company-level ever-recover gap +35.14pp). `h_sib_neg_share` on the same 110 is flat (+0.01pp). Do not add H or sister-B to Y3 X. H as stored is a group-type dummy; the Q5 keep is sister liquidity, which Y3 X forbids. If the useful signal is “this group has an invoiced sister”, it **cannot** transfer to the hidden test of new groups. Not a 0–100.

## What was not done

- Did not write parquet / duckdb. Did not run `build_targets`.
- Did not edit y11_dark, trail_length, debt_schedule_qa, family modules, explain_y3.
- Did not invent a new Y. Did not merge y11. Did not touch `product/`. Did not commit.
- Did not revive per-group Y3 (PARK 0.694 vs 0.710; hidden 72 = new groups).
