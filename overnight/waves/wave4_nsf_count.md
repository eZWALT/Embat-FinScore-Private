# Wave 4 — NSF / overdraft token hole (FinRegLab distress as X)

Owner: cash-flow literature. Deliverable: `analysis/outputs/nsf_count_qa.md`.
Do not reconstruct from B. Do not invent `y_nsf`. Do not score vs Y9.
Hidden 72 is coverage only.

## Headline

**HOLE.** FinRegLab NSF-as-X is a dataset hole. Leftover-after-days is
**undefined**. Last-value Q1 + quiet-stressed Y3 stay the cash-flow engine.

`data_dictionary.md` has no NSF / overdraft token (category examples are
utility / tax / collection). Feature dictionary mentions NSF only as
Y2/Y9 *literature*, not a column. CAT_MAP and monthly.parquet: 0 NSF-like
names. 22 train categories; none is NSF.

Word-bound train descriptions: `nsf` / `overdraft` / `bounce` = **0**.
A naive `LIKE '%nsf%'` hits 480,882 txs because **`nsf` sits inside
`transfer`**. Not a token.

`descubierto` 250 train txs / 83 companies is Spanish overdraft
**interest / claim-fee** text (`INTERES.DESCUBIERTO`), often filed as
`utility`. That is Y9 / Family M, not Norden ΔCUMOVER. Do not leftover.
Do not score vs Y9. Holdout coverage: 12 `descubierto` txs; 0 word-bound
`nsf`.

## Six questions

| # | this cut |
| --- | --- |
| 1 | Last-value `b_runway` KEEP (p50 1.079). Not NSF. Never B as X. |
| 2 | Not an NSF path. Y1 stays PARK. |
| 3 | Quiet-stressed stays SS 0.635 / salary 0.603 / days 0.711. |
| 4 | Out. Sibling TURNOVER 0.720 / 0.712. |
| 5 | No NSF token to name. Fee/interest is Y9, not an X. |
| 6 | Hidden 72 is coverage. No NSF lead. |

## Do not do

- Reconstruct NSF / overdraft / neg-days from `b_below_0` / `b_neg_episodes`.
- Invent `y_nsf`. Score the hole vs Y9. Treat `fee` as an NSF count.
- Leftover `descubierto` after days. Edit `liquidity.py`.
- Quote `a_out_vol` 0.722 as the engine. Grow utilisation / Y10.
- Overwrite `runway_window_qa.*` / `days_delta_qa.*` / `y3_reasons.*` /
  leftover QA owners / `lit_invoice.*`.

## Night numbers this wave must not move

Y3 0.762 / 0.752. days 0.711. size 0.617. TURNOVER 0.720 / 0.712.
SS leftover 0.635. salary 0.603. last-value p50 1.079.
