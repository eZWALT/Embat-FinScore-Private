# product/score/ — v0 dummy FICO-like card

A first 0–100 number the product can show. Weights are fixed in advance.
The only fit is a train-only percentile table. Not a model, not a claim
about the hidden test.

```bash
PYTHONPATH=. python -m product.score
```

Writes `outputs/monthly_scores.parquet` and `outputs/ref_v0.json` (gitignored).

## Row contract (v0, for `poc/` and later `product/web/`)

One row per `company_id` × `period`. Bind these; treat the rest as extras.

| Column | Meaning |
|---|---|
| `score` | Monthly 0–100 after stretch and dark cap |
| `score_3m` | **Headline.** 3-month median of `score` |
| `state` | `improving` / `stable` / `deteriorating` from Δ `score_3m` over 3 months (long dark → deteriorating) |
| `confidence` | 0–1 from available category weight (thin file ×0.8; dark capped at 0.55) |
| `confidence_band` | `high` / `medium` / `low` |
| `no_invoices` | Payment + mix missing; do not show a 90 as “perfect payer” |
| `going_dark` / `going_dark_long` | Dark cap fired |
| `cat_payment_history` / `cat_amounts_owed` / `cat_length_stability` / `cat_mix` | Category 0–100 (NaN if dropped) |
| `reason_1_es` … `reason_3_es` (and `_en`, `_code`) | Weakest items; Spanish for the UI |
| `group_id` | From the store, when present |

Not in v0: fifth “new credit” subscore, € on each reason, owner/action alerts (those stay step 3 / Sentinel). Rebuild the parquet after card changes.

## What is on the card

New credit is empty here (utilisation 1.6%, no bounced-payment token,
facility counts are connection artefacts). Its 10 FICO points move to
amounts owed.

| Category | Weight | Variables | Direction |
|---|---:|---|---|
| Payment history | 35 | `e_ar_overdue_30`, `e_ap_overdue_30`, `e_delay_coll`, `e_delay_paid` | lower better |
| Amounts owed | 40 | `b_runway`, `f_ds_r`, `f_fc_r` | runway up, ratios down |
| Length / stability | 15 | `active_share_6` (share of last 6 months with a bank movement) | higher better |
| Mix | 10 | `d_cust_top1`, but only the tail: 0.70 → 100 pts, 0.975 → 0 pts | lower better |

Missing category → drop it and reweight the rest. No invoices (about 470
of 1,214 train companies) means payment and mix are empty, so the number
is runway + debt cost + “still operating”, with a medium/low confidence
flag.

Left out on purpose:

- Social security / payroll / movement-days as “quiet = healthier” (Y3 trap).
- `trail_months` as points (mostly when the company connected, not age).
- `d_supp_hhi` (night found concentration *protective*; unsigned).
- `e_credit_note_ratio` (experimental).
- `e_issued_top1` / customer-loss (alert layer, not this score).

## Guards

- **Going dark scores low, never high.** No movements this month → cap 50.
  Recency > 45 days → cap 35.
- **Thin file:** fewer than 6 months on the grid → confidence × 0.8.
  Not extra points for a long extract.
- **Holdout** (`analysis/splits/holdout_companies.csv`) never enters the
  percentile table.

Averaging several percentile items would leave almost everyone in the
40s–60s. After the weighted mean we stretch by 2 around 50 (a priori, not
fit): `clip(50 + 2*(raw − 50), 0, 100)`. Then the dark cap.

## Headline the product should show

`score_3m` = 3-month median of the monthly score, plus a coarse state
from the 3-month change (improving / stable / deteriorating). Long dark
is always deteriorating.

Reasons are the weakest items below 50 points, plus the dark cap when it
fires. English and Spanish sentences sit on the row.

## First run (train, latest month 2026-08)

1,214 companies. `score_3m` p10 / p50 / p90 = **17 / 47 / 80**.
39% have no invoices (cash + “still operating” only; flag the gap).
10% are going dark (median 35). Mix is 100 for almost everyone — the
body of concentration is treated as noise; only the >97.5% tail hurts.

## What this is not

It does not explain 82→68 without cash (runway *is* cash; that is the
photograph, not a cause). It is not validated yet against the eight
accepted outcomes. Do not quote an AUROC for this card until that check
runs.
