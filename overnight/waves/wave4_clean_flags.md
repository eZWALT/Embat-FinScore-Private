# Wave 4 — clean flags DQ QA (d7235c84)

Long-lived data lane. Same module ≥30 min: write → run → next cut (passes 1–34).
No 0–100. No product/. No parquet rewrite. No new GBM. No `build_targets`.
Family modules not edited. Holdout 72 coverage only; rates on train.

## Files

- `analysis/evaluate/clean_flags_qa.py` (create)
- `analysis/outputs/clean_flags_qa.md`
- `analysis/outputs/clean_flags_prevalence.png`
- `analysis/experiments/registry.csv` (append coverage / drop-extreme / drop-dup / B-walk)

## Verdicts

| item | call |
| --- | --- |
| flags as health Y | **PARK** (DQ) |
| flags as X | **CLOSE** (`has_pdi` vs accepted Ys CV 0.49–0.55) |
| never-drop | **KEEP** — days / issued_lag1 / HHI do not move ≥0.02; drop-dup `a_n_tx` Δ=+0.003 |
| product_known vs G 63.7% | **different hole** (unk first-month share 0.1%) |
| B snapshot sentinels | **protects** (3 checking NULLed; other cash ~33k / ~40k) |
| B flow walk | **footnote** — all 24 extreme txs are checking; COMP_0306 11/18 Y2 flips if dropped. Do not patch `liquidity.py` |
| debt exclude-extreme vs cashflow keep | **report** — `a_op_in` −2.78% on 2 CM; `f_ds_r` Δ=0 |

## Numbers to quote

- **tx is_extreme:** 24 / 2,556,068. Train 8 (2.4% amount mass, 4 cos) / holdout 16 (33.1% mass, 2 cos). All 24 on **checking**.
- **inv is_extreme:** 10 / 896,711 — **CONFIRM** join QA. All train, 0 holdout. Gross |amount| **52.0%**; after 3 wash pairs leftover **2.8%**. Extreme-invoice companies are never Y4-labeled.
- **Quoted singles if drop extremes:** days 0.7114→0.7114; issued_lag1 store 0.6295 (in-module keep=drop 0.6157); HHI_lag3 0.6047→0.6047. No ≥0.02 move.
- **product_known=false:** 1,313 txs / 29 products. Not the G hole. Span 2024-09-09→2026-07-10. Intersection with unknown balances = 1 of 29.
- **is_dup:** extract clones (p50=2, max=4997 COMP_0611 €20×4997, same product, no CP). CM base 36.1%. SIZE (`a_n_tx` 0.866). Soft not-clone 59k (mostly different description).
- **B-walk / Y2:** train 4 extreme-checking cos have Y2 rate 0% (67 labeled) vs rest 7.4%. Dropping extreme checking flows would flip **11/18** Y2 labels on COMP_0306 only (0→11). Cause: one uncategorized checking outflow −1.613e9 on 2026-08-14, same magnitude as overdue invoice −1.616e9 issued 2025-12-31 (226 days, no payment_date). COMP_0629 has both books, **no** same-mag pair; Feb ±1.694e9 is a same-day tx wash (net 0 on the walk).
- **Groups:** holdout extremes are **GROUP_0199** (11 cos; only COMP_0900 + COMP_0276 have extreme txs) on the same two days 2026-02-10 / 02-13. Train **GROUP_0094** has 13 cos including extreme-invoice COMP_0163 and extreme-tx COMP_0306 / COMP_1192. Coverage; do not redo sibling H.
- **`has_pdi`:** CM 17.1%, size 0.590 — numeric Y gate passes; **PARK anyway** (DQ). Residual PDI (no giant) is spread (25 months). Drop-PDI issued_lag1 Δ=−0.005.

## What failed / next

- Unifying debt vs cashflow extreme rule: do not patch; `f_ds_r` already 0 on the two train op_in extreme months.
- Family B QA owns whether to footnote `b_liq` / Y2 on COMP_0306. This lane does not rewrite `liquidity.py`.
- Hidden-test GROUP_0199 unpaired uncategorized extremes are a third of those companies' own tx mass.

## Iteration (this module)

1–7 prevalence, calendar, Y overlap, dup, G hole, sentinel, debt-vs-A, quoted singles
8–16 washes, PDI July, B gap, HHI windows, whale, leftover 2.8%, holdout Y4-never, residual PDI as X, drop-dup, drop-PDI issued
17–22 unk-product anatomy, post-snapshot 8,186 pre-extract txs (op_in 0.006%), own-mass share, all 24 extremes on checking, B-walk shift estimate
23–34 Y2 0/67 on four train extreme-checking cos; 11/18 flips on COMP_0306; inv↔tx same-mag only that company (226d); COMP_0629 same-day tx wash net 0; GROUP_0199 (11, holdout) same two days; GROUP_0094 (13, train) holds COMP_0163 + 0306 + 1192
