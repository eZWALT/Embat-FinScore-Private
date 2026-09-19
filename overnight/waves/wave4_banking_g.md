# Wave 4 — Family G / banking_products QA

Agent `7e5c43f8`. Same module, iterated write → run → read (inventory → oriented AUROC → size 2×2 → connection replica → checking=connected). No parquet rewrite. No new GBM. No 0–100. No `products.py` edit.

## Files

- `analysis/evaluate/banking_g_qa.py` (owned)
- `analysis/outputs/banking_g_qa.md`
- `analysis/outputs/banking_g_has_vs_size.png`
- append-only `analysis/experiments/registry.csv`

## What we measured (train)

- Raw `banking_products`: 5,987 rows / 1,283 companies. Null `created_at`: **0**. Post-snapshot: 44. Type mix includes wallet / risk / lineofcomex as **other** (no `g_has_*`).
- `g_n_accounts` is a **rise-only** as-of panel: 1,561 rises, **0 drops** (same `created_at` rule as debt facilities). First-month 63.7% still 0; last-month p50=3.
- `g_new_this_month>0`: 7.9% CM / 1,016 companies. acf1=−0.08. 52.8% of new months are a first-ever as-of account. Calendar is a spread, not one wave. **PARK as health Y / Q3.**
- Best `g_has_*` vs Y3 (group-fold, oriented): **card 0.551**. `c_n_days_with_tx` **0.711** (raw 0.289 flipped). Size dummy **0.617**. Connected-only card **0.556** vs size 0.622. **CLOSE as Y3 X.**
- Card × TPV after size terciles does **not** survive (TPV n=10). Do not revive clusters (sil 0.234).
- `g_created_*` all-zero on the monthly panel. Drop-list **CONFIRM**. 44-col starter **CONFIRM** has `g_has_*` and not `g_created_*`. `g_n_accounts` / `g_new` stay off the 44. **CAUTION:** `g_has_checking` in the 44 is 99.1% the connection hole.
- 744 / 470 **CONFIRM**. Dark do **not** have fewer accounts (p50=3 both). Access ≠ ERP. Custom/Other is not the 470 (18/42 all-custom are dark).
- `company_meta.n_banking` vs last-month `g_n_accounts`: 1,188 / 1,214 exact; **all** 26 mismatches = post-snapshot rows. Not a bug.
- 73.6% replica **CONFIRM** (891/1,211). Lag first-tx → first `g_n_accounts>0`: p50=2 months.

## PARK / CLOSE / KEEP

| object | decision |
| --- | --- |
| `g_new` / `created_*` as Y | **PARK** (onboarding clock) |
| `g_created_*` as X | **DROP** (CONFIRM) |
| `g_has_*` as Y3 X | **CLOSE** (lose to 0.711 and to size 0.617) |
| `g_has_*` as Q1 operating type | **CLOSE** (2×2 dies after size) |
| `g_n_accounts` inventory | **KEEP** as clock, not Y |
| G as Q1 health context | **no** — another connection clock, complementary to debt |

## What failed / next

- First AUROC table treated inverse-size 0.38 as the bar; chance flags looked like wins. Fixed with oriented max(auc,1−auc).
- T2 one-vs-many Y2 5.1% vs 10.8% is **not** residual size (log p50 matched) but T3 flips — still CLOSE.
- **Next (not this lane):** if someone trims the 44-col starter, drop `g_has_checking` (connection dummy) and the rare `g_has_saving` / `g_has_tpv` (n=9 / 10). Do not invent a dark-access Y.
