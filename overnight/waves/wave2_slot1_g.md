# Wave 2 / slot 1 — Family G (product mix and access)

## Files

- `analysis/features/products.py` (owned). `SOURCE_TABLES = ["banking_products"]`, `FAMILY = "g"`, `build(con, grid)`.
- This note. Smoke script only in `/tmp/wave2_slot1_g_smoke.py` (not in the repo).
- No other files edited. Family A not touched. No commit. No 0–100 score.

## Columns (all `g_`)

As-of inventory for period `P` counts products with `created_at < P+1 month` (weekly: `< P+1 week`). `created_at` null is present in every period. Holdout not used to fit anything (no refs). `company_meta.n_banking` stays the static extract count.

| column | formula / note |
|--------|----------------|
| `g_n_accounts` | count of as-of products |
| `g_n_banks` | distinct `bank_name` |
| `g_n_types` | distinct `type` (includes wallet / risk / expensesPlatform / lineofcomex) |
| `g_has_card` / `g_has_tpv` / `g_has_checking` / `g_has_saving` / `g_has_investment` | 1 if any as-of product has that type |
| `g_custom_share` | share with `service=custom` or `bank_name` like Other / customer-defined (347 rows; the two filters coincide) |
| `g_created_unknown_share` | share with `created_at` null |
| `g_created_after_snapshot` | count with `created_at > 2026-09-01` among as-of products |
| `g_created_after_snapshot_share` | that count / `g_n_accounts` |
| `g_new_this_month` | count with `created_at` in `[P, P+1 period)` |

Empty company-months on the grid: counts / has_* = 0, shares NaN. Checked: last-month `g_n_accounts` = `n_banking` − after-snapshot (5,943 vs 5,987); consecutive-month `Δ g_n_accounts = g_new_this_month`; `g_n_accounts` non-decreasing within company.

## Train coverage

Train = 21,157 company-months / 1,214 companies (holdout 1,073 / 72 excluded from these %). Panel shape 22,230 × 15.

| column | train non-null % |
|--------|------------------|
| g_n_accounts | 100.00 |
| g_n_banks | 100.00 |
| g_n_types | 100.00 |
| g_has_card | 100.00 |
| g_has_tpv | 100.00 |
| g_has_checking | 100.00 |
| g_has_saving | 100.00 |
| g_has_investment | 100.00 |
| g_created_after_snapshot | 100.00 |
| g_new_this_month | 100.00 |
| g_custom_share | 87.63 |
| g_created_unknown_share | 87.63 |
| g_created_after_snapshot_share | 87.63 |

Share columns are NaN on the 2,618 train months (12.37%) with `g_n_accounts = 0` (first txs before any product is connected, or never connected). 1,209 / 1,214 train companies have at least one as-of product by 2026-08.

## Issues

- `clean.dq_log` flags **44** banking products with `created_at > 2026-09-01` (journal “63” = 44 banking + 19 debt). Monthly last cut is `created_at < 2026-09-01`, so `g_created_after_snapshot` is 0 on every monthly row (no look-ahead). They are in `n_banking` but not in G. On the last ISO week (2026-08-31) the cut is `< 2026-09-07` and 15 of them appear.
- `created_at` is never null in this extract → `g_created_unknown_share` is 0 whenever defined (near-zero variance).
- `g_has_tpv` / `g_has_saving` are rare (10 / 9 train companies ever; 0.45% / 0.32% of train months). Card 16.2% of companies, investment 7.2%.
- Mean train `g_n_accounts` rises from 2.75 (2024-09, 21% zeros) to 4.60 (2026-08). Checking dominates (4,817 of 5,943 as-of products).
- Three companies have no banking products at all (`COMP_0676`, `COMP_0683`, `COMP_0906`).

## Next idea

Add `g_has_wallet` / `g_has_expenses_platform` (23–34 products) and months-since-first-connection vs first tx, so late-connected companies are not treated as “thin” only because `g_n_accounts` is still ramping.
