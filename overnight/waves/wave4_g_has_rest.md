# Wave 4 — leftover `g_has_*` / `g_custom_share` (4348a23d)

Long-lived data lane. Same module ≥30 min: write → run → next cut (must-do 1–12 plus extras 13–32).
No 0–100. No product/. No parquet rewrite. No new GBM. No `build_targets`.
`products.py` not edited. Did not reopen `g_new` / `created_at` as health Ys.
Holdout 72 coverage only; rates / AUROC on train. Seed 20260918.
Did not put `g_has_*` on the 15-col Y3 card. Night Y3 quote stays **0.762 / 0.752**.
Days bar **0.711**. Size **0.617**. Y7 TURNOVER **0.720 / 0.712** (do not grow).

## Files

- `analysis/evaluate/g_has_rest_qa.py` (create)
- `analysis/outputs/g_has_rest_qa.md`
- `analysis/outputs/g_has_rest_vs_size.png`
- append-only `analysis/experiments/registry.csv` (skip key includes `x_families`)
- this note

## What we measured (train)

- Prevalence CONFIRM: saving last-month **n=9** ever=9 modal0 **99.7%**; investment last **88** ever=88 modal0 **95.1%**; TPV last **n=10** ever=10. `g_custom_share>0` last=121 ever=121 cov 87.6%. Holdout coverage only (saving/TPV ever=0; invest 5; custom 19/72; card 8; checking 72).
- HAS flags **rise-only YES** (saving 9/0, invest 80/0, TPV 9/0, card 151/0, checking 773/0). `g_n_accounts` CONFIRM 1,561/0. `g_custom_share` 51↑/51↓ — **100% of drops are n_accounts↑ with n_custom flat** (mechanical dilution). n_custom itself 65↑/0↓.
- Spearman: no leftover flag is SIZE. No |ρ|≥0.80 twin vs accounts / checking / days / n_tx. checking vs `g_n_accounts` ρ=0.576 (not a twin); Jaccard(checking=0, accounts=0)=**0.991** (2,618/2,641). DROP checking from the 44 anyway (connection hole).
- Y3 oriented CV (G replica): card **0.551 CONFIRM**, days **0.711**, size **0.617**. saving 0.501 / invest 0.509 / TPV 0.502 / custom 0.530 / checking 0.506. None beat size ≥0.02.
- OLS leftover after days looks high (0.65–0.71) but ρ(resid,days) **0.84–0.99** — **fake days leak** (same as zero_in). Honest leftover **DIES**. Inside days terciles custom D2 0.566 vs size 0.555 (Δ 0.011) vs days 0.575 — CLOSE.
- `g_custom_share` leftover after accounts **0.548 <0.55**. ρ vs accounts 0.064 — **not a mix leftover, not an inventory twin**. acf1 0.844 CONFIRM / ICC 0.998 TRAIT.
- SIZE terciles: no leftover flag survives T1 after honest leftover. Investment last-month T3 **14.3%** vs T1 **3.7%** (size-tagged type). Card × TPV CONFIRM TPV n=10, does not survive.
- Dark 470 vs ERP 744 CONFIRM. Checking ~99.4% both. Dark invest **8.9% > ERP 6.2%**. Access ≠ ERP. TPV rare on both.
- Q6 **CLOSE**: flags are 0 until first type `created_at` (pre0=100%). First-on aligns with `g_new` (saving/TPV 100%; invest 92%). Misses are already-on-book at first grid month.
- Family F same rise-only + fake-days leftover (already dropped from 44). wallet/risk/lineofcomex/expensesPlatform sit in `other` (37 train cos) — do not invent `g_has_*`. expensesPlatform Y3 23.1% on 26 labeled / 7 cos — do not promote.

## PARK / CLOSE / KEEP

| object | decision |
| --- | --- |
| `g_has_saving` on the 44 | **DROP from the 44** / **CLOSE as X** / **PARK as Y** (n=9; Y3 0.501; leftover fake) |
| `g_has_investment` on the 44 | **DROP from the 44** / **CLOSE as X** / **PARK as Y** (Y3 0.509; T3-tagged) |
| `g_has_tpv` on the 44 | **DROP from the 44** / **CLOSE as X** / **PARK as Y** (n=10; do not invent `y_has_tpv`) |
| `g_custom_share` on the 44 | **DROP from the 44** / **CLOSE as X** / **PARK as Y** (leftover after accounts 0.548; dilution not mix) |
| `g_has_card` on the 44 | **DROP from the 44** / **CLOSE as X** (CONFIRM G 0.551) |
| `g_has_checking` on the 44 | **DROP from the 44** / **CLOSE as mix** (CONFIRM 99.1% hole) |
| any leftover `g_has_*` as Q6 | **CLOSE** (connection clock) |
| leftover types as new columns | **PARK** — do not invent parquet `g_has_*` |
| 15-col Y3 card | **no** |
| night quotes | **unchanged** |

## Brief map

1. Who is healthy? — not these flags (PARK as Y).
2. Who is improving? — flags only rise. custom_share ↓ is dilution.
3. Who is turning? — first-on = connection birth. PARK as Q3.
5. Why? — type tags / inventory clock, already on days / size.
6. Months earlier? — **CLOSE**.

## What we did not do

- Did not edit `products.py`, `banking_g_qa.py`, parquet / duckdb, `build_targets`, `product/`, the 15-col card, LIVE.json, CONTEXT.md, canvas, or the parent journal.
- Did not invent `y_has_tpv`. Did not reopen `g_new` / `created_at`.
- Did not fit on holdout 72. Did not commit.

## What failed / next

- First leftover table quoted OLS 0.65–0.71 as “lives”. Tightened: ρ(resid,days)≥0.30 → fake, honest leftover DIES.
- First custom verdict said “inventory twin” because leftover after accounts <0.55. Corrected: ρ=0.064, not a twin; dies as leftover, dilution explains the 51↓.
- checking self-ρ was listed as a twin; fixed (no |ρ|≥0.80 twin; hole is Jaccard 0.991).
- Next (not this owner): drop `g_custom_share`, `g_has_card`, `g_has_checking`, `g_has_investment`, `g_has_saving`, `g_has_tpv` from the 44-col starter when someone re-cards. Not a parquet rewrite. `g_n_accounts` stays KEEP-list inventory clock, off the 44.
