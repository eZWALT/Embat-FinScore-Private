# Wave 4 — leftover B cols on the 44 (c8b1440a)

Long-lived data lane. Same module ≥30 min: write → run → next cut (must-do 1–12 plus extras 13–29).
No 0–100. No product/. No parquet rewrite. No new GBM. No `build_targets`.
`liquidity.py` not edited. Did not reopen the walk / still-leak except persist confirm.
Holdout 72 coverage only; rates / AUROC on train. Seed 20260918.
Did not put B on the 15-col Y3 card. Night Y3 quote stays **0.762 / 0.752**.
Days bar **0.711**. Size **0.617**. Y7 TURNOVER **0.72 / 0.712**.

## Files

- `analysis/evaluate/b_on_44_qa.py` (create)
- `analysis/outputs/b_on_44_qa.md`
- `analysis/outputs/b_on_44_leftover.png`
- append-only `analysis/experiments/registry.csv`
- this note

## What we measured (train)

- Train 21,157 CM / 1214 companies. `b_liq` all-null=19 = 13 tx-no-balance + 6 no-cash-walk (CONFIRM 13+6=19). tx-no-balance train=13 (CONFIRM HOLE13). Santander UK on 6 of the 13; last-tx 2026-07-20 n=6. Zombie cash 538/4,408=12.2% (CONFIRM 12.2%) |zombie|/|cash|=0.4% (quote 0.36–0.4%). PARK as a flag.
- Y3 days=0.711 (CONFIRM 0.711) size=0.617 (CONFIRM 0.617). Y2 `b_below_0`=0.896 `b_runway`=0.924 — LOCK leak (not KEEP). Singles are diagnostics. Never B as Y2/Y3 X.
- Y3 leftover after days: b_bal_vol=0.734, b_below_0=0.649, b_d_runway=0.748, b_neg_episodes=0.687, b_runway=0.711. Fake-days any=True. Honest leftover still lives. Leftover is a DROP evidence, not a KEEP-as-X.
- Rank leftover kills `b_runway` (0.537). `b_d_runway` OLS leftover looks high; Q1 dummy≈continuous (0.649 vs 0.659); body leftover after days=0.527 DIES. Q1 owns 47.3% of Y3 recoveries.
- ρ(b_bal_vol, a_vol)=0.354 n=14,968 (CONFIRM 0.354 DRIFT). Y3 leftover of store vol after Javier=0.621 after days=0.734 after both=0.732.
- LOOK-AHEAD last-value still Y3=0.823 vs month-t 0.596. last p50=1.079 CONFIRM 1.079. Last-month Y3 labels=0.
- Demean *raises* B AUROC (runway 0.596→0.940) — month-t cash path is the Y. `a_out_vol` demean 0.722→0.549 (trait). `b_bal_vol` η²=0.129 ≠ a_out_vol dummy.
- Persist CONFIRM short 0.730 / 24m 0.862. b_liq persist short=0.730 (CONFIRM 0.730) long24=0.862 (CONFIRM 0.862). vol/d_runway need history — empty on so_far<6. Q6 KEEP=False (B forbidden).
- DROP all five from the 44 as Y2/Y3 X: **True**. KEEP as Q1 description: ['b_runway']. Night quotes unchanged 0.762/0.752.

## PARK / CLOSE / KEEP

| object | decision |
| --- | --- |
| `b_bal_vol` on the 44 | **DROP from the 44** / **CLOSE as X** / **PARK as Y** (DRIFT 0.354 vs Javier; leftover after days is fake ρ=0.978; not the a_out_vol trait) |
| `b_below_0` on the 44 | **DROP from the 44** / **CLOSE as X** / **PARK as flag** (Y2 leak 0.896) |
| `b_d_runway` on the 44 | **DROP from the 44** / **CLOSE as X** (Q1 crash leftover; body 0.527 DIES) |
| `b_neg_episodes` on the 44 | **DROP from the 44** / **CLOSE as X** / **PARK as Y2 cousin** (not a below_0 twin; leftover fake days) |
| `b_runway` on the 44 as Y3 X | **DROP from the 44 as X** (rank leftover 0.537 DIES) |
| last-value `b_runway` as Q1 description | **KEEP** |
| last-value as forecast Y | **PARK** |
| Family B as Y2/Y3 X | **forbidden** (lock) |
| invent `y_bal_vol` | **no** |
| 15-col Y3 card | **no** |
| night quotes | **unchanged** |

## Brief map

1. Who is healthy? — last-value runway KEEP as description.
2. Who is improving? — d_runway empty-on-short; leftover is the Y3 crash cell. DROP as X.
3. Who is turning? — do not invent y_bal_vol. Y2 is the B path.
5. Why? — store vol ≠ Javier vol (DRIFT 0.354) and ≠ a_out_vol trait.
6. Months earlier? — Q6 CLOSE. Short persist 0.73 < long 0.86.

## What we did not do

- Did not edit `liquidity.py`, `balances_b_qa.py`, parquet / duckdb, `build_targets`, `product/`, the 15-col card, LIVE.json, CONTEXT.md, canvas, or the parent journal.
- Did not invent `y_bal_vol`. Did not reopen the walk except persist confirm.
- Did not fit on holdout 72. Did not commit.

## What failed / next

- First leftover table quoted OLS 0.65–0.75 as “lives”. Tightened: ρ(resid,days)≥0.30 → fake; rank leftover kills `b_runway` (0.537); `b_d_runway` leftover is the Q1 crash cell (body 0.527 DIES).
- First hole-13 query joined balances→banking_products and counted 17. Fixed: `balances.company_id` → 13 tx-no-balance + 6 no-cash-walk = 19 all-null CONFIRM.
- Next (not this owner): drop the five B cols from the 44-col starter when someone re-cards. Not a parquet rewrite. Last-value `b_runway` stays Q1 description off the card.
