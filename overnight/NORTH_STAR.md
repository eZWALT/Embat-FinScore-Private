# North star — re-read before every analysis task

Live brief URL (Cloudflare-gated from agents; this file is the working copy):
https://claude.ai/artifact/8N8Q7QMjprCUWxGAiJaWoP?sk=5wYke4E8ukAw6afs6TrG1g

Captured 2026-09-18 in `.agents/persistent-memory/2026-09-18-initial-context.md`
and `2026-09-18-2020-four-goals.md`. If the artifact is reachable, re-fetch and
diff this file. Do not invent a different contest.

## What we are building (engine only, tonight)

Track: HackSpain 2026 · Embat **X Ray**.
Question: *¿Puede el dinero decir cómo está una empresa?*

Embat gives a 24-month treasury trail (synthetic SME, 2024-09-01 → 2026-09-01,
250 groups / 1,286 companies). The engine is a **FICO-like company health
reading of that trail**. It is **not** a bankruptcy classifier.

The brief's own examples: Northbrook Foods **45 → 65** vs Velasco Industrial
**82 → 68**. The same month-24 still can hide who is the better risk.
So the object is a **trajectory**, not a last-month snapshot.

## Six questions — every feature / Y / SHAP must map to one

Per company, per month:

1. Who is healthy?
2. Who is improving?
3. Who is turning?
4. Dip vs fall?
5. Why did it change?
6. How many months earlier was it visible?

Y3 cash-recovery is the 45→65 direction. Y2 stress is the 82→68 direction
(models failed). Y1 is "can we reconstruct the path". Y9 fee/interest
pressure is *turning / why* (FinRegLab NSF/fee). Y5/Y7/Y8 are other-table
why / dip-vs-fall. A model that cannot be explained in those six sentences
is not on-brief, even if AUROC is high.

## Four jobs in the brief (tonight we only do three of the engine)

| Brief job | Tonight | Where |
|-----------|---------|-------|
| Read the trail | yes | `analysis/features/`, `analysis/targets/` |
| Score trajectory (not only last month) | evidence only — no 0–100 formula | accepted Ys + models in `analysis/` |
| Explain the number | yes | SHAP / gain / single-feature, `analysis/outputs/` |
| Sellable product on top | **no** | `product/` is frozen |

Do not write a 0–100 index. Do not touch `product/`. Do not pick a web stack,
LLM, buyer story, or Docker runtime. Those stay empty until the team opens
goal 4 again.

## Constraints that are the contest, not our taste

- Hidden test: **60–80 companies never used to fit**. We froze 72 companies /
  15 groups (`analysis/splits/holdout_companies.csv`, seed 20260918).
  Never fit percentiles, bins, models, SHAP baselines, or cluster centroids
  on them. Quote **train group-fold CV** for claims; holdout is LOW_POWER.
- **Y is never built from the same columns as the X allowed to predict it.**
- No look-ahead: feature for `period` uses events with date ≤ period end.
- Judging (brief): being right, being on time, being worth something — equal.
  A simple, explainable signal that answers the six questions beats a clever
  number nobody can defend.
- Delivery later (not tonight): hidden-test scores, both directions,
  trajectory, explanation, product, buyer, navigable demo. Tonight produces
  the **evidence** those later pieces will stand on.
- Bonus the brief asked for: measured **lead time**; a monitor that alerts
  unprompted. Persistence (acf 1/3/6) and lag-SHAP are how we earn lead time.

## Dataset facts that must stay in the work

Eight tables, join `company_id`. No invoice↔bank FK. `balances` is a
2026-09-01 still (liquidity is reconstructed backwards). All 8 tables are
already in the monthly store. `d_interco_share` is all-null. Utilisation
(`f_util_snapshot`) is last-month-only (~1.6%). Schedule rate / next-pay
are a **1.7% growing panel** (38 train companies), not a last-month still.
Both stay PARK as GBM X. Keep flow `f_ds_r` / `f_fc_r`.
Leftover QAs **DROP `g_n_accounts` from the 44** (rise-only 1561/0;
leftover after days 0.428). **PARK `f_util_snapshot` as snapshot X**
(native Y3 n_pos=0; hole leftover 0.711 is a fake days leak).
**CLOSE unused leftover of `e_ap_overdue`** (rank 0.584 lives; beat-size
FAIL +0.008). Do not put any of them on the 15-col card.

## Honest referee (why the 16h 0.86 died)

The discarded overnight search built Y and score from the same `net` /
`overdue` columns. That measured persistence, not health. Do not recreate it.

## Scope check (paste into every child prompt)

```
READ overnight/NORTH_STAR.md first.
Work only under analysis/ (+ overnight/waves + registry + feature_store docs).
No product/. No 0–100 formula. Map the output to the six questions.
```
