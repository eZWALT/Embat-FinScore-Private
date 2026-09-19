# Wave 4 — Family H sister existence vs sister state

Agent `5d5b1b81`. Lane: `analysis/evaluate/sibling_h.py`. No commit. No parquet/duckdb rewrite. No `product/`. No 0–100. No GBM. Holdout 72 coverage only. Did not run `build_targets`. Did not merge y11.

## Files written

- `analysis/evaluate/sibling_h.py` — create; 7 write→run cuts in this module
- `analysis/outputs/sibling_h.md` — tables + PARK/KEEP
- `analysis/outputs/y3_rate_mix_size_tercile.png` — Y3 rate by mix × size tercile
- `overnight/waves/wave4_sibling_h.md` — this note
- `analysis/experiments/registry.csv` — append-only (agent `5d5b1b81`; 10 rows)

Did not edit y11_dark, trail_length, debt_schedule_qa, family modules, parquet, product/, build_targets, explain_y3.

## Population (reproduced)

confirm_470=True. Train invoiced **744**. Dark **360** in 79 all-dark groups + **110** in 38 mixed groups.

| slice | Y3 n / pos / rate | Y2 n / pos / rate |
|---|---:|---:|
| all_dark_360 | 1,557 / 81 / **5.20%** | 4,489 / 426 / **9.49%** |
| mixed_110 | 473 / 57 / **12.05%** | 1,592 / 130 / **8.17%** |

Y3 gap +6.85pp. Y2 gap −1.32pp (flat / slightly lower stress on the 110).

## Size residual

- Specified control: `log1p(a_in3)` terciles, edges from **all train** CM. T1 mixed−all-dark **+16.67pp** (25.0% vs 8.3%).
- y11-style `log1p(|a_op_in|)` on train-dark Y3-labeled: T1 **+12.46pp**. **Confirms +12.5pp.**
- `log1p(a_in3)` edges from train-dark Y3-labeled: T1 **+13.42pp**.
- Quote the y11-style **+12.5pp** as the comparable number; the all-train `a_in3` T1 is a different (stricter) bin and is larger.

All-dark holdings are *smaller* groups (median group_n 2.0 vs mixed 7.5). Mixed dark firms are smaller (`a_in3` 238k vs 361k).

## Sister state vs existence (110)

| cut | Y3 healthy | Y3 stressed | gap | after size (T1 / weighted) |
|---|---:|---:|---:|---|
| mean sister `b_runway` ≥ 1 | 15.56% (257) | 7.69% (208) | **+7.87pp** | T1 **+6.97pp** / +7.95pp |
| mean sister `b_runway` ≥ 3 | 14.58% | 10.26% | +4.33pp | T1 +6.4pp / +5.0pp |
| sister `a_io_ratio` ≥ 1 | 11.19% | 13.41% | **−2.22pp** | wrong way |
| sister not Y2 | 13.31% | 8.89% | +4.42pp | T1 +8.1pp (T2 thin) |
| `h_sib_neg_share` median | 12.05% | 12.05% | **+0.01pp** | flat |

Company-level (74 mixed Y3 cos): mean sister runway ≥ median ever-recover **48.6% vs 13.5%** (+35pp). Not a sister-Y2 clone (ρ −0.33). Mean aggregation beats min / largest-sister. Lag-1 sister runway still +7.1pp (short Q6, still B).

Invoiced firms in mixed groups recover **less** (4.97% vs 8.92% all-invoiced). The 110 lift is dark-side, not a healthy-holding dummy.

## Leak + single-feature (Y3 stressed, train group-fold)

| feature | CV AUROC | vs 0.711 | vs size 0.617 |
|---|---:|---:|---:|
| `c_n_days_with_tx` | **0.711** | 0 | +0.095 |
| `log1p(a_in3)` | 0.617 | −0.094 | 0 |
| `h_share_group_in` | 0.638 | −0.073 | +0.022 (NEAR_SIZE ρ 0.747) |
| `h_group_size` | 0.552 | −0.159 | −0.065 |
| `h_n_siblings_active` | 0.548 | −0.163 | −0.069 |
| `h_sib_in` | 0.551 | −0.160 | −0.066 |
| mixed dummy | 0.494 | −0.217 | −0.123 |
| `h_sib_neg_share` | **0.434** | −0.277 | −0.183 |

`h_n_siblings_active` vs `h_group_size` ρ=**0.979**. Worst |ρ| vs `b_runway` = 0.111 — not a B-copy. H-neg / mixed dummy do **not** beat size/group-size by ≥0.02.

## Hidden-test honesty

If the useful signal is “this group has an invoiced sister”, it **cannot** transfer to the hidden test of new groups.

## Verdict

- **PARK H as a Y3 X.** Group-size copy, near-size share, `h_sib_neg_share` 0.434 vs 0.711. Do not brief H as a durable lever. Overturns high |SHAP| / perm ≈ 0: that rank is `h_group_size`.
- **KEEP as a descriptive Q5 footnote** (not X): sister *mean B-runway ≥ 1* moves Y3 ≥5pp inside the 110 after size. Family H does not carry that state. Sister B is forbidden as Y3 X.

## Next idea (parent)

Do not add H columns to Y3 X. Do not put sister `b_runway` in X. Optional footnote in the Y3 why card: among dark firms with an invoiced sister, recovery is higher when that sister is not currently stressed. Hidden 72 = new groups — do not score a mixed-group dummy.
