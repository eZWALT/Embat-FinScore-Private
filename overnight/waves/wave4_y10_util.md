# Wave 4 — Y10 financing stress / LOC inventory

- **Files:** `analysis/targets/y10_util.py`, `analysis/outputs/y10_acceptance.md`, this note, append-only `analysis/experiments/registry.csv`
- **API:** `META.forbidden_x_families = ["f"]`, prefixes `f_` / `a_fin_cost` / `a_fc`, `build(con, grid)` → `company_id`, `period`, `y10_*`
- **Re-run:** `python -m analysis.targets.y10_util`
- **Did not** write `targets.parquet`, edit `build_targets.py`, import family F, touch `product/`, or commit.
- **Assembler warning:** `discover_target_modules()` globs `y*.py`. Parent must not run the assembler until the merge decision.

## Brief (NORTH_STAR)

Q3 / Q5 from *financing stress*, not bankruptcy, not a 0–100. There is no historical outstanding/granted panel, so utilisation cannot be a Y.

## Inventory (DuckDB `debt_products`)

- Type string is `lineofcredit` (536 rows, 206 companies, 189 train).
- First LOC mid-panel: 152 companies (135 train).
- Extract `outstanding_gt_granted`: 35 companies (32 train). LAST-MONTH-ONLY.
- `f_util_snapshot` / `f_w_rate` remain last-month / 38-train-cos thin (join QA).

## Train-only verdicts (21,157 company-months, 1,214 companies)

Do not put a frozen-Y name on the same line as a scanner token. Numbers below.

| column | n | pos | base | size AUROC | two-sided | verdict |
|---|---:|---:|---:|---:|---:|---|
| `y10_new_loc_after_stress` | 12,674 | 209 | 1.65% | 0.681 | 0.681 | PARK |
| `y10_loc_book_then_fees` | 1,236 | 186 | 15.05% | 0.539 | 0.539 | CLOSE |
| `y10_ogtg_last_month` | 1,214 | 32 | 2.64% | 0.664 | 0.664 | PARK |
| `y10_add_loc_6m` | 1,497 | 148 | 9.89% | 0.611 | 0.611 | PARK |
| `y10_loc_then_int_ownp80` | 1,236 | 179 | 14.48% | 0.508 | 0.508 | ACCEPTED |
| `y10_loc_then_int_spike` | 900 | 158 | 17.56% | 0.480 | 0.520 | ACCEPTED |

Scanner tokens (Y10 columns only):

- `y10_new_loc_after_stress` PARK (not a keep)
- `y10_loc_book_then_fees` CLOSE (subset)
- `y10_ogtg_last_month` PARK (snapshot)
- `y10_add_loc_6m` PARK (size)
- `y10_loc_then_int_ownp80` ACCEPTED
- `y10_loc_then_int_spike` ACCEPTED

## Why those verdicts

1. **New LOC after 2-of-3 operational-net dip** is the same idea as the already-rejected facility-after-dip label, restricted to LOC. Strict subset (209 of 578). Base 1.65% + size 0.681. PARK, do not rename.
2. **LOC on book + fee_r own-p80** clears the numeric band but Spearman vs the fee own-p80 label is 1.000 on 1,236 rows. CLOSE as a subset, not a new Y.
3. **Extract ogtg on 2026-08 only** is the same leak as the frozen snapshot flag. PARK as a Y.
4. **Add-LOC in 6m** looked like a numeric pass when month-end inventory counted a first mid-month LOC as an add. After `created_at ≤ t` (month-start), size AUROC is 0.611. PARK.
5. **LOC on book + interest_charge/inflow own-p80 for t+1..t+3** clears base 14.48% and size 0.508. Spearman vs the fee own-p80 label is 0.348; vs the debt-service-double label 0.116; vs the 2-of-3 stress label 0.060. No allowed-X |ρ| ≥ 0.8. Left in the module. **Parent decides merge.** This is not utilisation.
6. **LOC on book + interest_r spike (2× trail-6m in 2 of next 3)** clears base 17.56% and two-sided size 0.520 on 900 labeled months / 101 companies. Sibling ρ vs the own-p80 column is 0.511; vs the fee-spike label 0.398. Not a rewrite. Positive-company union 73 / intersection 49. Same parent-merge rule.

## Can the debt table support a new Y?

**Inventory yes, utilisation no.** `created_at` can filter who already has a LOC (123 train companies have enough history for the interest label). Outstanding/granted cannot make a 2024–2025 util path. Catalogue “LOC util > 90%” stays impossible.

## Next idea (parent)

Merge the two interest-on-LOC columns only if a Y9-method sibling on the LOC book is wanted. Do not reopen snapshot ogtg or rename the LOC-after-dip column. Models must drop F and `a_fin_cost` / `a_fc`. Holdout is LOW_POWER on both keep columns (own-p80: 72 labeled / 17 pos / 4 cos; spike: 41 / 16 / 4).
