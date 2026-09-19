# Wave 4 — factoring / confirming / LOC inventory

Long-lived child `fd90198f`. Cuts in `analysis/evaluate/factoring_qa.py` (prevalence through holdout-group unseen). No parquet rewrite. No new GBM. No `build_targets`. No 0–100. Did not score F vs Y9. Did not edit `debt.py`, `a_vol_qa.py`, `companies_qa.py`, `fx_qa.py`, or `product/`.

## Files

- `analysis/evaluate/factoring_qa.py` (create / owner)
- `analysis/outputs/factoring_qa.md`
- `analysis/outputs/factoring_has_vs_size.png`
- append-only `analysis/experiments/registry.csv` (`agent=fd90198f`)
- `overnight/waves/wave4_factoring.md` — this note

## Columns

`f_has_factoring` / `f_has_confirming` / `f_has_loc` / `f_new_facility` — as-of `created_at` inventory (rise-only like G). Flow `f_ds_r` / `f_fc_r` untouched (KEEP).

## Coverage (train; holdout count only)

| item | n |
| --- | ---: |
| train companies / CM | 1,214 / 21,157 |
| ever factoring / confirming / LOC | **18** / **61** / **186** |
| CM share (modal 0) | 0.7% (99.3%) / 3.3% (96.7%) / 11.7% (88.3%) |
| ever WC fact∪conf | 65 |
| rises / drops | 17/0, 50/0, 135/0 — **rise-only** |
| `f_n_facilities` | 555 rises / 0 drops |
| holdout ever | 1 / 9 / 17 (last-month WC names 19; factoring = COMP_0269 stacked) |
| dark 470 vs ERP 744 | fact 10 vs 8; conf 28 vs 33; LOC 74 vs 112 |
| FX 228 ∩ factoring | 1 / 18 |
| Y5 leftover CM | 222 (y5_why match); fact ∩ leftover 5 co |

Size ρ vs log1p(a_in3): 0.083 / 0.156 / 0.255 — not SIZE.

## Y3 singles vs 0.711 / size 0.617

Oriented group-fold (seed 20260918). Days replica **0.711 MATCH**.

| flag | Y3 CV | Δ size | Y2 CV |
| --- | ---: | ---: | ---: |
| `f_has_factoring` | 0.505 | −0.112 | 0.496 |
| `f_has_confirming` | 0.485 | −0.132 | 0.504 |
| `f_has_loc` | 0.546 | −0.070 | 0.523 |
| `f_new_facility` | 0.515 | −0.102 | 0.501 |

KEEP-as-44 needs beat size ≥0.02 and not SIZE and not a group dummy. **None keep.**

## NORTH_STAR

Q3 turning: factoring/confirming are working-capital **tools**, not utilisation (Y10 already PARK). Debt schedule QA: inventory is a **connection panel**, not origination. New-WC month size-w Y3 −2.0% / Y2 −0.1% — Q3 footnote **CLOSE** (first-cut KEEP was new-factoring Y3 0% on 11 labeled months).

## PARK / CLOSE / KEEP

| object | decision |
| --- | --- |
| `f_has_factoring` / `f_has_confirming` / `f_has_loc` as 44-col Y3 X | **CLOSE** — drop |
| `f_new_facility` as 44-col X | **CLOSE** — drop (connection clock) |
| has_* as health Y | **PARK** |
| new-WC month as Q3 footnote | **CLOSE** |
| confirming as AP-side / Q5 tool footnote | **CLOSE** — only 8.8% vs rest 8.3%; +4.4pp is stacked WC |
| extract unused / outstanding / liquidity as X or Y | **PARK** (Y10) |
| `f_ds_r` / `f_fc_r` | **KEEP** (loan book on factoring-only names, not the flag) |
| score flags vs Y9 | **CLOSE** (Y9 forbids F) |

## What failed / next

- Confirming looked AP-side until confirming-only (14) matched rest. Stacked still +4.3pp after size — stacking, not the product.
- Drawn factoring Y2 +21pp after size is GROUP_0139 (3/8). Remaining 5: Y2 0%.
- 0/18 ever Y3+ is real vs T3 peers 7.7% and is still CLOSE as X (18-co dummy; holdout WC groups unseen, top GROUP_0103).
- Next (not this owner): drop the three flags + `f_new_facility` from `gbm_core` CORE/44 when someone re-cards. Not a parquet rewrite.
