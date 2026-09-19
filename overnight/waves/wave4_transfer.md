# Wave 4 — signed transfer / invest QA

Agent `1bc809f3`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/transfer_qa.py`
- `analysis/outputs/transfer_qa.md`
- `analysis/outputs/transfer_wash.png`
- append-only `analysis/experiments/registry.csv` (skip key agent+y+model+split+metric+x_families)
- this note

Did not touch `cashflow.py`, `catmix.py`, `interactions.py`, `salary_qa.*`, `a_vol_qa.py`, `missing_cp_qa.py`, `product/`, parquet / duckdb, `build_targets`, parent journal, Family M/I, or the 15-col card. Night Y3 quote stays **0.762 / 0.752**.

## Columns / coverage (train)

- Any-transfer 39.0% (8,242 / 21,157 CM, 882/1,214 cos). Store vs raw signed max|Δ|=1.9e-6 (float dust). Holdout any-xfer 31.9% — no AUROC.
- Any-invest 10.5% (2,231 CM, 402 cos). Holdout 14.3%.
- Tokens: `transfer` 152k (90k in / 61k out); `investment_deployment` 3,923 all out; `investment_return` 3,458 all in.

## Locked verdict

| object | decision |
| --- | --- |
| `a_transfer` as Y3 X | **CLOSE** |
| `a_transfer` as a health Y | **PARK** (do not invent `y_transfer`) |
| `a_invest` as Y3 X | **CLOSE** |
| `a_invest` as a health Y | **PARK** (do not invent `y_invest`) |
| two-way wash | **NO** (p50 \|net\|/gross = 1.0) |
| quiet twin | **YES** (honest leftover dies) |
| SIZE | **NO** (ρ vs log1p(a_in3) = 0.044) |
| Q6 lag1/lag3 | **CLOSE** |
| KEEP-Q5 footnote | **CLOSE** |

Y3 signed `a_transfer` **0.566** vs size **0.617** (Δ −0.051) vs days **0.711** vs salary **0.671**. ρ vs days 0.109 (not a \|ρ\|≥0.80 twin). Honest leftover: `has_xfer` after days **0.579** / after card **0.564**. Signed OLS leftover 0.666 is a zero-month −days leak (ρ(resid,days)=−0.881). ICC 0.956 TRAIT; company-demean drops 0.566→0.509. That is the SHAP-heavy / perm-light story: trees split on a company transferer trait that permutation kills.

`has_xfer` 0.655 / gross 0.664 beat size but lose to days/n_tx/salary. Still not on the card.

`a_invest` Y3 **0.505**. Deploy-only 920 / return-only 923 / both 388 — not G rise-only.

## What failed / extras that stayed CLOSE

- Not a wash: 55.5% of transfer CM are one-way; only 9.4% have \|net\|/gross ≤0.05. COMP_0003-style two-way months exist but are not typical.
- Transfer is real mass: \|amt\| p50 6,000 vs op_out 497 (≤1€ 0.7%).
- Descriptions 86.4% TRASP / 7.6% TRANSFERENCIA; missing CP 97.8%; owner/sweep/dividend 0%. No leftover Q5 sentence.
- Dark 470 vs 744: same transfer-CM rate (41.5% vs 38.5%). CONFIRM 744/470.
- Drop 12 chronic Y2 names: 0.524 → 0.516 (no flip). Those 12 transfer 92.9% of months vs rest 38.3%.
- Company-mean `has_xfer` Y3 **0.747** (company Y3-any 0.743 vs mean days 0.706). ρ vs company-mean days 0.518 — not a \|ρ\|≥0.80 twin, but ICC 0.96 is a **trait**, not a month shock. KEEP gate requires shock. Do not put on the card.
- Q6: now 0.566 / lag1 0.557 / lag3 0.542. SHAP lag1 perm-light CONFIRMED.

## Next (parent)

CLOSE signed A as Y3 X. PARK as Ys. Keep salary / days / n_tx on the 15-col card. Do not merge M or I. Do not invent `y_transfer` / `y_invest`. Company-mean transfer share is a style footnote only — parent decides; this child does not KEEP it.
