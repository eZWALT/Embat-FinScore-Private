# Wave 2 / slot 1 — Y7 concentration + Y8 cross-source

- **Files:** `analysis/targets/y7_concentration.py`, `analysis/targets/y8_cross.py` (owned). This note.
- **Not touched:** `analysis/features/products.py` (G done), `analysis/features/counterparties.py` (D may still be owned). Invoice HHI / top-1 recomputed in Y7 from `invoices`.
- **Grid:** monthly `company_id × period` (22,230 rows, 1,286 companies). Weekly rows, if passed, align via month-start like Y1.
- **Train only** (holdout `analysis/splits/holdout_companies.csv` excluded from rates / AUROC / rank refs): 21,157 company-months, 1,214 companies.
- **Thresholds:** fixed from the catalogue. Not searched on train or holdout.
- **Re-run:** `python -m analysis.targets.y7_concentration` and `python -m analysis.targets.y8_cross`

## Y7 (binary; next quarter = t+1..t+3)

Invoice AR book on the trailing 3 months (complete from 2024-11). HHI = Σ share², top-1 = max share; both computed here, not imported from D. Next-quarter amount for that same `counterparty_id` is 0 ⇒ share dropped to 0. Inflow clause: each of t+1..t+3 has `op_in` < 75% of trailing-3m mean `op_in` at t (baseline > 0). Forbidden X: `["d"]`.

| column | train n | cov | pos | base rate | accepted | size AUROC | two-sided |
|--------|--------:|----:|----:|----------:|----------|-----------:|----------:|
| y7_top1_lost | 7,464 | 35.3% | 2,149 | **28.79%** | **yes** | 0.462 | 0.538 |
| y7_top1_lost_inflow | 6,739 | 31.9% | 420 | **6.23%** | **yes** | 0.463 | 0.537 |

Size = AUROC of `log1p(|monthly op_in|)`. Inflow positives are a subset of `y7_top1_lost` on the 6,739 overlapping labeled rows. Identification book (train): 656 companies, mean HHI 0.56, median top-1 share 0.69, mean 23 counterparties. 655 / 1,214 train companies have any Y7 label (ERP AR is sparse; 714 companies ever issue AR).

`y7_top1_lost` is the looser column (no inflow clause). It already sits in 5–30%, so the inflow variant is not a rescue — it is the catalogue shock (~6%).

## Y8 (binary; horizon 6; cross-source)

Composites are 0–1 worse-is-higher ranks, **not** a 0–100 product score.

- **cash_side** (liquidez + caja): mean of available train-only within-month percentile ranks of `−runway`, `−net_margin`, `+neg_liq_3`. Cash path from `cash_month_panel` (same unwind as Y1/Y2).
- **invoice_side** (cobros): mean of available train-only ranks of `e_ar_overdue_30`, `e_delay_coll` (family E `build()`).

Label: the side at **t+4, t+5 and t+6** each exceeds that company’s own expanding p80 (min 6 finite months, history ≤ t). Point-only `side(t+6) > own p80` was 31.5% / 37.6% (rejected on the 30% cap); the last three months of the horizon are the contract’s sustained window.

| column | train n | cov | pos | base rate | accepted | size AUROC | two-sided | single-source AUROC |
|--------|--------:|----:|----:|----------:|----------|-----------:|----------:|--------------------:|
| y8_inv_worse_6 | 3,883 | 18.4% | 761 | **19.60%** | **yes** | 0.538 | 0.538 | 0.531 (best e_) |
| y8_cash_worse_6 | 6,986 | 33.0% | 1,615 | **23.12%** | **yes** | 0.434 | 0.566 | 0.571 (best cash) |

Single-feature leak screen: contemporaneous `e_ar_overdue_30` / `e_delay_coll` / `e_ar_overdue` vs the invoice Y, and `runway` / `net_margin` / `neg_liq` vs the cash Y. All two-sided AUROC < 0.70 (no e_ column explains y8). 815 / 1,214 train companies have any Y8 label. Invoice coverage is thinner because `e_delay_coll` is masked for the first 6 months and own p80 needs 6 observations before a 6-month horizon remains.

### Forbidden X (META)

Cross-source pairing (plan Y8 / Javier: cash at t → invoice at t+6 and reverse). Same-source X is the persistence leak this Y exists to avoid.

| Y column | built from | forbidden X | intended X |
|----------|------------|-------------|------------|
| y8_inv_worse_6 | invoice-side / cobros | **e** | a, b (cash) |
| y8_cash_worse_6 | cash-side / liquidez+caja | **a, b** | e (invoices) |

`META.forbidden_x_families = ["a","b","e"]` is the union. `META.forbidden_x_by_column` is what a model must use. Holdout never enters the rank reference; own p80 is per-company.

## What failed

- **Y8 point-in-time** (`side(t+6) > own p80` only): invoice 31.5%, cash 37.6% — above 30%. Fixed scales without ranks made invoice worse (44%). Onset (`not already > p80 at t`) also passed (~21%/22%) but the shipped rule is the sustained 3-month window, which matches the overnight contract.
- **Y7 coverage** is ERP-limited: only ~35% of train company-months have a 3-month AR book plus a visible next quarter. That is a coverage gap, not a rate fail.
- Did not import family D; did not write a 0–100 score; did not touch G or `product/`.

## Next idea

If a denser Y7 is needed, identify top-1 on the 6-month pipeline concentration window (still next-3m outcome) — expect a similar rate, more companies with a named customer. For Y8, a weekly analogue of the same ranks is only worth it after the monthly GBM baseline; invoice delay is already a 90-day window.
