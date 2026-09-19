# Wave 4 — companies.csv QA (a14da08b)

Long-lived data lane. Same module ≥30 min: write → run → next cut (passes 1–8 + extras 1–90).
No 0–100. No product/. No parquet rewrite. No new GBM. No `build_targets`.
Did not edit a_vol_qa / fx_qa / uncat_qa / sibling_h. Did not invent a merged country/erp Y.
Did not revive `created_at` as a health Y. Holdout 72 / 15 groups coverage only; rates / AUROC on train. Seed 20260918.

## Files

- `analysis/evaluate/companies_qa.py` (create)
- `analysis/outputs/companies_qa.md`
- `analysis/outputs/companies_erp_dark_country.png`
- append-only `analysis/experiments/registry.csv` (13 rows, agent a14da08b)

## Quote (train)

erp × 470 dark (CONFIRMS join-QA 92.1% / 37 / 506 / 14.4%):

| companies.erp | invoice book | n |
|---|---|---:|
| named | invoiced | 671 |
| named | dark | 37 |
| NULL | invoiced | 73 |
| NULL | dark | 433 |

- NULL among dark **92.13%**. Perfect dark flag? **False**.
- Named-ERP dark are **17 of the 110 mixed + 20 of the 360** (not the 110). 18/20 all-dark named = GROUP_0138 dynamicsAx.
- Country miss **82.21%** (998/1214). Holdout coverage 80.56% (58/72). Currency miss 0%. erp miss 41.68% (506). created_at miss 0%.
- Three clocks: created−first_tx p50 **+39.4 days**, share after 61.3%, same-month 7.1%, same-day **0.7%**. created−bank p50 −5.9d (same-day 24.4%). bank−first_tx p50 +54.6d (same-day 1.6%). Onboard ≠ trail.
- **Any companies.csv flag transferable as Q1 X?** **False**.

## PARK / CLOSE / KEEP

| object | decision |
| --- | --- |
| `created_at` as health Y | **PARK** — connection clock (year vs grid ρ −0.85; 99.6% weekday; same-day group wave 163/235) |
| missing-country as health Y | **PARK** — 82% missingness; not SIZE (ρ −0.028); 24m books still 73.6% miss |
| `has_erp` as Y3 X | **CLOSE** — 470 dummy (ρ 0.81 vs book); Y3 CV 0.480 vs size 0.617 |
| `has_country` as Q1 descriptive | **CLOSE** — Y3 0.481 vs size 0.617; ERP-vendor metadata; 85% group-uniform |
| `currency=EUR` as Q1 descriptive | **CLOSE** — 90.3% train vs 73.6% holdout; Y3 0.515 vs size 0.617 |
| any companies.csv flag as transferable Q1 X | **CLOSE** — group-type dummy; hidden test = new groups |

## What the later cuts added

- Onboard-before-cash n=470 ∩ dark n=470 overlap **182** — coincidence of counts.
- created_year vs n_grid ρ **−0.851**. Two regimes: 2021–23 all before cash; 2025–26 mostly after. late_onboard Y3 0.522 vs size 0.617.
- Holdout year mix 2025 **71%** vs train 35%; late-onboard share 83% vs 60%.
- Country fill by ERP family 0–100% (AX/X3/Infor 0%; Libra 100%). invoiced+NULL country-known **1/73**.
- Holdout named-dark: COMP_0851 / GROUP_0211 netsuite USD. Unseen home ccy AOA/GHS/SEK/XOF all in GROUP_0199 (all-dark).
- Train groups ~85% uniform on has_erp / has_country / is_eur. Same-day onboard 1037 cos vs first-tx 669.
- Weekend: created_at 0.4% / first_bank 0.6% / first_tx **15.3%**.

## What failed / next

- Skip key first missed `x_families` and duplicated 8 registry rows; duplicates removed; skip now includes family.
- Extra 18 first said holdout was “later-onboarded”; median year matches (2025) — mix differs (2025-peaked).
- **Next (not this lane):** do not put has_erp / has_country / is_eur / created_at on a 44-col Y3 list. Do not build a country or ERP Y. Trail length / days stay the transferable clocks.
