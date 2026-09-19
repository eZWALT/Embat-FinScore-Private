# Wave 3 · slot 6 — Y9 sustained fee/interest pressure

- **Files:** `analysis/targets/y9_fees.py` (this note)
- **API:** `META.forbidden_x_families = ["f"]`, `META.forbidden_x_prefixes = ["f_","a_fin_cost","a_fc"]`, `horizon = 3`, `build(con, grid)` → `company_id`, `period`, `y9_*`
- **Assembler:** `build_targets.discover_target_modules()` globs `y*.py`. No register line added.
- **Grid:** monthly `company_id × period` (22,230 rows, 1,286 companies). Weekly rows align to the containing month.
- **Train only** (holdout `analysis/splits/holdout_companies.csv` excluded from rates / AUROC): 21,157 company-months, 1,214 companies
- **Thresholds:** own-history p80 is per-company past (months ≤ t, min 6 finite). Not a global percentile. Not searched on train or holdout.
- **Flows:** computed here from `transactions` via `CAT_MAP` (`fee` / `interest_charge` → fin_cost). Does **not** import `analysis.features.debt` or cashflow.
- **Re-run:** `python -m analysis.targets.y9_fees`

## Brief questions (NORTH_STAR)

Y9 is **not** a bankruptcy label. It answers:

3. **Who is turning?** — fee+interest / inflow stays above that firm’s own p80 for the next quarter, or doubles vs the trailing-6m mean in 2 of the next 3 months (NSF/fee onset).
5. **Why did it change?** — the move is financing-cost / fee pressure (FinRegLab 2025 bank-statement NSF/fee), not cash level and not a default tag.

Y3 is the 45→65 recovery direction; Y2 is 82→68 liquidity stress. Y9 is the *why / turning* companion from fee+interest.

## Y9 (binary; size = AUROC of log1p(|monthly op_in|))

ACCEPTED if train base rate ∈ [5%, 30%] **and** two-sided size AUROC `max(auc, 1-auc) < 0.60`.

| column | train n | cov | pos | base rate | accepted | size AUROC | two-sided | verdict |
|--------|--------:|----:|----:|----------:|----------|-----------:|----------:|---------|
| y9_fee_r_ownp80 | 9,591 | 45.3% | 1,350 | **14.08%** | **yes** | 0.522 | 0.522 | ACCEPTED |
| y9_fee_spike | 7,879 | 37.2% | 1,502 | **19.06%** | **yes** | 0.440 | 0.560 | ACCEPTED |

908 / 1,214 train companies have any Y9 label (494 with a positive own-p80 event; 513 with a positive spike).

Definitions:

- `fee_r` = `(fee+interest)_3m / max(in3, 1)` unclipped (same 3-month windows as `score_pipeline.fin_cost_r`, without the [0, 1] cap so a true double is visible).
- `y9_fee_r_ownp80` = 1 if `fee_r` at t+1, t+2 and t+3 each exceed that company’s expanding p80 of `fee_r` using months ≤ t (min 6 finite observations).
- `y9_fee_spike` = 1 if `fee_r` ≥ 2 × trailing-6m mean(`fee_r`) at t in **at least 2 of** t+1..t+3. NaN when the 6m mean is 0 or missing (doubling is undefined on a zero base) or the horizon is incomplete.

## Forbidden later X

Models that predict these labels must not use family **F** or columns starting with `f_`, `a_fin_cost`, `a_fc` (`f_fc_r`, `f_fin_cost`, `a_fin_cost`, lagged copies). Intended X is A (except those prefixes) + B C D E G H. Same-column leak: `f_fc_r` *is* this Y’s raw material.

`protocol.leakage_check` appends `_` to prefixes that do not already end with `_`, so pass `a_fin_cost` as an exact drop in the model allow-list, not only through that helper.

## What failed / not done

- Spike **without** the positive-base rule sat at 33.5% (above 30%) and two-sided size AUROC 0.649 — companies with a 6-month zero fee path “double” at any later fee. That is not a train-fitted cut; 0 × 2 is not a double (same reason Y4 NaNs `ds_r_t ≤ 0.05`).
- A catalogue 0.05 floor on the 6m mean (Y4-style) would have passed the rate band but failed two-sided size AUROC (0.647). Not used.
- Did not write a 0–100 score. Did not touch `product/` or `build_targets.py`. Did not revert Y7/Y8 accepted or Y6 parked.

## Next idea (later wave; do not retune here)

Keep both columns. A later GBM must drop F + `a_fin_cost` / `a_fc*`. If a denser fee Y is needed, split `fee` vs `interest_charge` as two numerators — still own-history, still 3-month horizon.
