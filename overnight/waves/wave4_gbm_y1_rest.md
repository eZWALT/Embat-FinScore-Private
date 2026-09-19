# Wave 4 — Y1 rest (inflow h1/h3, liquidity h3)

- **When:** 2026-09-19 00:28–00:56 CEST (write → run → read → variant, same module)
- **Agent:** `a41f9c72` (did not touch `0137801b` net / liq_h1 rows)
- **Python:** `/home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.models.gbm_y1`
- **Gate:** default `ONLY = (y1_in_h1, y1_in_h3, y1_liq_h3)`; `--all` runs all six; `--residual-last` / `--residual-hist` / `--log-size` / `--diag` / `--no-registry`
- **Holdout:** 72 companies, seed 20260918. Never in a fit, percentile, or early-stop.
- **X:** families A–H, lags 1 and 3, meta `group_size` + `n_banking`. n_x=308 (302 after log-size). Y1 has no forbidden families (labels are t+h).
- **Win rule:** holdout median-norm MAE (median over companies of MAE / mean|y|) vs **hist_mean**. Also report last-value. Quote **train group-fold CV MAE** (and OOF median-norm) as the fit claim. Variant KEEP only if holdout gap vs hist ≥ `CLEAR_MARGIN` 0.02.

No product/. No 0–100. Did not edit parquet, duckdb, family modules, `gbm_core.py`, holdout membership, or LIVE.json.

## Files

- `analysis/models/gbm_y1.py` (appended TASKS + gate + residual / log-size / OOF median-norm; existing `run` / `run_task` / net / liq_h1 still there)
- `analysis/outputs/y1_rest.md` (quote table)
- this note
- append-only `analysis/experiments/registry.csv` (`split=cv5_group` and `holdout`; 72 rows, agent `a41f9c72`)

## KEEP / PARK

| series | CV MAE | CV OOF med-norm gbm / hist / last | holdout med-norm gbm / hist / last | holdout MAE | beat-share vs hist / last | verdict |
|--------|--------:|-----------------------------------|------------------------------------|-------------|---------------------------|---------|
| **y1_in_h1** | €3.78m (trees=54) | **0.726** / 0.781 / 0.822 | 0.799 / **0.795** / 0.981 | €3.02m vs hist €2.90m vs last €3.66m | 0.54 / 0.77 | **PARK** |
| **y1_in_h3** | €4.22m (trees=116) | 0.816 / 0.821 / 0.893 | **0.817** / 0.856 / 0.878 | €3.30m vs hist €3.18m vs last €3.28m | 0.51 / 0.76 | **KEEP thin** |
| **y1_liq_h3** | €4.61m (trees=117) | 0.632 / 0.645 / **0.598** | 0.568 / 0.625 / 0.586 | €7.87m vs hist €17.2m vs last €13.3m | 0.51 / 0.64 | **PARK path** |

Naive check matches `d0748ae4` (hist holdout MAE op_in h1 €2.90m / h3 €3.18m; last-value worse on inflow, stronger on liq).

### y1_in_h1 — PARK

Official holdout median-norm **loses** to hist (0.799 vs 0.795). Last-value is worse (0.981); trees beat last, not the company mean. CV OOF says the opposite (0.726 vs 0.781): fit claim looks like a win, holdout check fails. Do not trophy either side. Gain is raw `a_op_in` / `a_in3` (size). Company beat-share vs hist is 54% (coin flip). Path not reconstructed.

### y1_in_h3 — KEEP thin

Holdout median-norm beats hist (0.817 vs 0.856, gap 0.039) and last (0.878). Official WIN. CV OOF is only 0.816 vs 0.821 (gap 0.005) — direction agrees, magnitude does not. Raw holdout MAE still loses to hist (€3.30m vs €3.18m). Beat-share vs hist is 51%. Fold MAE is huge (fold 3 €12.5m vs fold 1 €0.75m) because groups differ in euro scale; that is why the quote is median-norm, not mean MAE. Trees are again `a_op_in`. Keep as a thin check vs hist, not as a 3-month inflow trajectory.

### y1_liq_h3 — PARK path

Holdout median-norm beats hist (0.568 vs 0.625) and **slightly** beats last (0.586). That holdout last-win is a trophy: **CV OOF last-value wins** (0.598 vs GBM 0.632). CV MAE €4.61m vs holdout €7.87m also disagree. Train persist (no model): liq_t vs liq_{t+3} Pearson 0.71 / Spearman **0.85** (n=17356); vs t+1 Spearman 0.91. Top gain is `b_liq`. Same story as posted `y1_liq_h1` (median-norm 0.409 vs last **0.385**). Last-value runway/liquidity is already the brief Q1 snapshot. Trees are not adding a trustworthy 3-month trajectory.

## Pass 2 variants (y1_in_h1 only)

No residual **KEEP**. Last-value residual and log-size miss hist or miss the 0.02 bar. Hist-mean residual meets the holdout bar but fails the fit claim.

| variant | CV MAE | CV OOF med-norm | holdout med-norm vs hist / last | holdout MAE | KEEP? |
|---------|--------:|-----------------|---------------------------------|-------------|-------|
| last-value residual (`--residual-last`) | €2.95m | 0.738 vs hist 0.781 | 0.798 vs **0.795** / 0.981 | €3.41m (worse than level GBM) | **PARK** — still loses hist |
| log-size (`--log-size`; drop `a_op_in`/`a_in3`/`a_in6`/`a_in12`, in-memory `log1p`) | €3.77m | 0.733 vs hist 0.781 | 0.786 vs 0.795 / 0.981 | €2.98m | **PARK** — gap 0.009 < 0.02 |
| hist-mean residual (`--residual-hist`) | €3.70m | 0.777 vs hist 0.781 | 0.763 vs 0.795 / 0.981 (gap 0.033) | €2.91m ≈ hist €2.90m | **PARK** — holdout gap ≥ 0.02 but CV OOF gap 0.005; beat-share vs hist **44%** (more firms worse). Do not trophy holdout. |

Seasonal residual was not run. Train month-of-year R² after company-demeaning is **0.0003** (`op_in`) / 0.0008 (`liq`). `d0748ae4` seasonal_naive only covers ~28% of holdout rows. No calendar season to residualise.

## What failed

- Inflow h1: GBM ≈ hist on holdout; CV and holdout disagree on the win metric.
- Inflow variants: last-residual overfits the delta (CV MAE down, holdout MAE up). log-size is the same model on a log stem; thin holdout edge is not a KEEP. Hist residual improves holdout median-norm (0.763 vs 0.795) but only 44% of companies beat hist and CV OOF is a rounding-level edge — PARK, not a path.
- Liquidity path: last-value wins the fit claim (CV OOF). Holdout GBM-vs-last is not trusted.
- Euro-level fold MAE is not a comparable fit claim across groups; OOF median-norm is.

## Judge note (brief Q1 / Q6)

Q1 *who is healthy* is already a last-value liquidity/runway snapshot: liq Spearman 0.91 at one month and 0.85 at three, and the h3 GBM loses to last-value on train OOF. Inflow next-month is a noisy company mean (hist beats last; GBM does not beat hist on the holdout check), so “healthy” is not a reconstructed inflow path. Q6 *how many months earlier* the path was visible: the lead is the persistence itself — about one month of usable level memory for liquidity, weaker and mean-reverting for operational inflow (Spearman 0.76 at h=1, 0.73 at h=3). Other families (C–H) do not show up in gain. Trees are not buying extra months of warning beyond “copy last liquidity” and “use the firm’s inflow average.”

## Next idea

Park further Y1 inflow GBMs. Hist-mean residual was the remaining inflow idea and it failed the fit claim (CV OOF) plus company beat-share. For liquidity, stop fitting trees for path reconstruction; publish last-value + persist (acf 1/3) as the Q1/Q6 evidence.

Re-run rest: `python -m analysis.models.gbm_y1`  
Last residual (parked): `python -m analysis.models.gbm_y1 --residual-last`  
Hist residual (parked): `python -m analysis.models.gbm_y1 --residual-hist`  
Log-size (parked): `python -m analysis.models.gbm_y1 --log-size`  
Persist / season only: `python -m analysis.models.gbm_y1 --diag`
