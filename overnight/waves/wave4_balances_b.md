# Wave 4 — Family B balances still QA

Long-lived child. 25 cuts in `analysis/evaluate/balances_b_qa.py`.
No parquet rewrite. No new GBM. No 0–100. No Family B as X for Y2/Y3.
Did **not** edit `liquidity.py` (no bug — the walk is an identity).
Did **not** run `build_targets`. Did **not** invent a Y from `b_liq` / `b_runway`.

## Files

- `analysis/evaluate/balances_b_qa.py` (owner)
- `analysis/outputs/balances_b_qa.md`
- `analysis/outputs/balances_b_residual.png`
- registry appends (`agent=086b8ed0`, coverage / identity / persist)
- this note

## NORTH_STAR sentence

Q1 health is last-value liquidity. That is **true of the last month**, not of the trail.

- Last-month `b_liq` vs cash snapshot ρ=0.966 (thin-Sep **1.000**; fat-Sep fifth **0.799**). Median |snap − last| = €200 = September cash-product flow (identity max 9e-9).
- First-month vs snapshot ρ≈0.61 on both short and 24-month books. Company-median vs last ρ=0.808. The typical month is **not** the still.
- Spearman(`b_liq_t`, `b_liq_{t+3}`) train = **0.848** (CONFIRM ~0.85, n=17,356). Short <12: **0.730**. 24-month: **0.862**. The still-leak prediction (short books *more* like the snapshot) is **false**.

## Identity (CLOSE)

`b_liq + sum(checking/saving/tpv flows after t) = snapshot`. Train 20,941 CM / 1,195 companies.

| check | median \|err\| | share >1€ | share >1% \|snap\| |
| --- | ---: | ---: | ---: |
| store + after vs snap | 9.1e-12 | 0.0% | 0.0% |
| store vs `liquidity._reconstruct` | 0 | — | — |

**CLOSE the reconstruction.** Do not rewrite `liquidity.py`. PARK the snapshot as a Y.

## Raw still

7,996 rows / 1,273 companies. 7,980 (99.8%) on 2026-09-01; 16 on late-August closest-prior days. 3 null sentinels. 2,207 debt-product orphans live in `balances` (loan/LOC/…). Walk |cash| is 79%; excluded |snapshot| 21% (mostly those debt orphans + investment). Custom checking is **in** the walk (type filter, not service). TPV extract stock is all-zero.

Holdout: **72/72** have a balance row and a cash-walk snapshot (coverage only).

## 63.7% first-month unconnected

`g_n_accounts=0` first month: 773 / 1,214. On those rows `b_liq` is **finite 97.8%**, null 2.2%, exactly 0 0.1%. Unconnected ≠ missing cash. The walk ignores `created_at`.

## Zombies / vol / left-trunc

- Zombie cash products (0 txs): 12.2% of train cash products, **0.4%** of |cash|. PARK as a health flag.
- `b_bal_vol` vs log1p(a_in3) ρ=−0.282 — **not SIZE**.
- Monthly grid **starts at first tx** — no pre-tx company-months. Late first-month neg 8.9% vs on-time 11.5%. Head>tail red is on *every* cohort (24-month 11.5%→5.1% inside the same 435). Last>first is a coin flip (48.6%). Tail recovery (38 neg→pos vs 10 pos→neg), not a richer book. **CLOSE** early-grid `b_below_0` as Q6.

## Q1 last-value (KEEP as description)

Last-month train: `b_liq` p50 €82k, share<0 **4.7%**; `b_runway` p50 1.08, share<1 **49.1%** (44.4% are `liq≥0` over a fat burn; only 4.7% are below zero). Runway clip wall 24 sits on 20.3%. Runway persist t vs t+3 = 0.757 < liq 0.848.

KEEP last-value as the honest Q1 *level*. PARK as a 3-month forecast (Y1 liq already). PARK inventing a Y from B.

## Holes (not a liquidity.py bug)

13 train companies have checking txs (8,645) and no `balances` row → `b_liq` null. 12 quiet ≥30d; 1 live (`COMP_0312`, last tx 2026-08-07). Walking from 0 would invent a still. **No rewrite.**

## PARK / CLOSE / KEEP

| object | decision |
| --- | --- |
| reconstruction walk | **CLOSE** |
| snapshot as a Y | **PARK** |
| last-value as Q1 description | **KEEP** |
| last-value / snapshot as a forecast Y | **PARK** (Y1 liq CONFIRM) |
| Family B as X for Y2/Y3 | **forbidden** |
| invent a Y from B | **PARK** |
| zombie / `b_bal_vol` as health | **PARK** / not SIZE |
| early-grid negatives as Q6 | **CLOSE** |
| rewrite `liquidity.py` | **no** |

## What failed / next

Nothing to fix in Family B. Next idea (not this owner): leave last-value `b_liq` / `b_runway` as the Q1 *photograph*; do not feed B to Y2/Y3; do not walk the 13 from a fake 0-snapshot.

## Pass 26 addendum (same module)

Pass-3 pile `n_co` **sums across types**. Unique walk companies = **1,267** (train **1,195**). Walk ∩ debt-orphan = 370. Train excluded-only = 6 (unchanged).

The 13 unphotographed checking books: last cash-tx on **2026-07-20** is **6/13 (46.2%)** vs **0.3%** of train photographed walk (median last-tx there is 2026-09-01). All 6 are **Santander (Corporate UK)**. Hole stamp = one bank extract, not a second still. **PARK** as a cutoff Y. Still **no** `liquidity.py` rewrite.
