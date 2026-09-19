# Wave 4 — Y11 cash Y among companies with no invoice book

Agent `3d5ad8d8`. Lane: `analysis/targets/y11_dark.py`. No commit. No parquet/duckdb rewrite. No `product/`. No 0–100. Holdout never used to choose a cut. Assembler already skips `y11_dark`.

## Files written

- `analysis/targets/y11_dark.py` — create; 6 write→run variants in this module
- `analysis/outputs/y11_acceptance.md` — tables + CLOSE/PARK/ACCEPTED
- `overnight/waves/wave4_y11_dark.md` — this note
- `analysis/experiments/registry.csv` — append-only train base_rate / size_auroc rows (`3d5ad8d8`)

Did not edit y2 / y3 / y8 / y9 / y10, `match.py`, `gbm_y1.py`, `xgb_y4.py`, `gbm_i_lift.py`, FROZEN lists, `targets.parquet`, or `product/`. Did not rebuild Y8.

## Population (DuckDB, not hardcoded IDs)

- Book filter: `document_type='invoice' AND status<>'cancel' AND amount<>0 AND issuance_date IS NOT NULL`.
- **confirm_470=True.** Train dark 470 / 1,214. Holdout dark 32 / 72 (descriptive).
- Dark train company-months on the official grid: **7,603** (470 companies).
- Group mix: **360** in 79 all-dark groups; **110** in 38 mixed groups.

## Brief (NORTH_STAR)

Q3 / Q4 for companies **without an invoice book**. Invoice-side turning is undefined. Cash-side still is. Never D/E. Do not invent cobros.

## Train-only verdicts

Do not put a frozen-Y name on the same line as a scanner token.

| column | n | pos | base | size AUROC | two-sided | ρ vs parent | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `y11_y2_neg_2of3_dark` | 6,081 | 556 | 9.14% | 0.544 | 0.544 | 1.000 vs Y2 | CLOSE |
| `y11_y3_recover_cash_6m_dark` | 2,030 | 138 | 6.80% | 0.331 | 0.669 | 1.000 vs Y3 | CLOSE |
| `y11_neg_2of3_dark` | 6,193 | 2,954 | 47.70% | 0.509 | 0.509 | 0.043 vs Y2 | PARK |
| `y11_neg_3of3_dark` | 6,193 | 1,425 | 23.01% | 0.488 | 0.512 | 0.040 vs Y2 | PARK |
| `y11_neg_ownp20_2of3_dark` | 3,905 | 825 | 21.13% | 0.541 | 0.541 | 0.027 vs Y2 | ACCEPTED |
| `y11_net_recover_6m_dark` | 2,289 | 506 | 22.11% | 0.491 | 0.509 | 0.146 vs Y3 | ACCEPTED |
| `y11_neg_onset_3of3_dark` | 3,227 | 293 | 9.08% | 0.563 | 0.563 | 0.013 vs Y2 | ACCEPTED |
| `y11_fee_ownp80_dark` | 3,183 | 448 | 14.07% | 0.536 | 0.536 | 1.000 vs Y9 | CLOSE |
| `y11_fee_spike_dark` | 2,488 | 471 | 18.93% | 0.409 | 0.591 | 1.000 vs Y9 spike | CLOSE |
| `y11_zero_in_2of3_dark` | 6,193 | 1,255 | 20.26% | 0.122 | 0.878 | −0.090 vs Y2 | PARK |

Scanner tokens (Y11 columns only):

- `y11_y2_neg_2of3_dark` CLOSE (subset)
- `y11_y3_recover_cash_6m_dark` CLOSE (subset)
- `y11_neg_2of3_dark` PARK (base)
- `y11_neg_3of3_dark` PARK (state, not a turn)
- `y11_neg_ownp20_2of3_dark` ACCEPTED
- `y11_net_recover_6m_dark` ACCEPTED
- `y11_neg_onset_3of3_dark` ACCEPTED
- `y11_fee_ownp80_dark` CLOSE (subset)
- `y11_fee_spike_dark` CLOSE (subset)
- `y11_zero_in_2of3_dark` PARK (size)

## Why those verdicts

1. **Y2 restricted to the 470** is defined (n=6,081, base 9.14% vs full-train 7.32%, size 0.544 vs 0.531). Same definition, ρ=1. CLOSE as *Y2 on dark companies*, not a new label. Usable one-sided and two-sided.
2. **Y3 restricted to the 470** is defined (n=2,030, base 6.80% vs 7.12%, size 0.331 vs 0.315). Same definition, ρ=1. CLOSE as *Y3 on dark companies*. Inherits the parent inverse-size (two-sided 0.669, same as full-train 0.685). One-sided gate still passes, matching how the parent was accepted.
3. **CAT_MAP net 2-of-3** is not a Y2 rewrite (ρ=0.043) but base **47.7%** is outside 5–30%. PARK. Restricted Y2 was not empty, so this cousin was only a screen.
4. **Net 3-of-3** clears numeric gates (23.0%, size 0.488) and is not a Y2 rewrite, but 79% of positives already have net<0 at t (AUROC-now 0.705). A *state*, not a Q3 turn. PARK. Prefer the onset slice.
5. **Onset 3-of-3** (net≥0 at t, then three negative-net months): n=3,227, pos=293, base **9.08%**, size 0.563, acf1=0.14. ρ vs Y2 = 0.013; vs rejected `y2_onset_neg` = −0.014 (`y2_onset_neg` on the 470 is still 1.29% thin). This *is* 3-of-3 on currently healthy months (ρ=1.0 there). Left as the turning slice. Forbidden later X: **A + D + E**.
6. **Own-p20 2-of-3** (company expanding p20, not a pooled cut): 21.13%, size 0.541, ρ vs Y2 = 0.027. Not a rewrite. Sibling ρ vs onset 0.25. In-module. Parent merge.
7. **Net-recover 6m** (stressed net<0; 3-in-6 months net≥0): 22.11%, size 0.491, ρ vs Y3 = 0.146 (not a rewrite). In-module. Parent merge.
8. **Fee own-p80 / spike on dark** are Y9 masks (ρ=1.0). CLOSE as subsets. Do not rename.
9. **Zero-in 2-of-3** is inverse-size (two-sided 0.878). PARK.

## 360 vs 110 — does H seeing a sister matter?

- **Y2 stress:** 9.49% all-dark vs 8.17% mixed (gap 1.32pp). Not a different stress world.
- **Y3 recover:** 5.20% all-dark vs **12.05%** mixed (gap 6.85pp). Mixed dark companies are *smaller* (median op_in 31k vs 89k; group median 7.5 vs 2.0), which would already lift Y3 (parent is inverse-size). After size terciles the T1 residual is still **+12.5pp** (20.7% vs 8.3%). Family H can see a sister on the 110; that is a real Q3/Q4 context split for *recovery*, not a reason to invent cobros. Onset / Y2 stay flat across the split.

## Do the 470 need their own Y?

**They do not need a renamed Y2/Y3.** Restricted accepted Y2 is usable on this set. Restricted Y3 is the same label (inverse-size inherited). Score those with A/C/F/G/H, never B/D/E.

Optional cash cousins that are *not* rewrites sit in-module for a parent merge: `y11_neg_onset_3of3_dark` (Q3 turn), `y11_neg_ownp20_2of3_dark`, `y11_net_recover_6m_dark`. Train pos-cos union **365 / 470** (onset 212, own-p20 236, recover 197; all-three only 56). Models of the net cousins must drop family **A** (same CAT_MAP flows) as well as D/E. Holdout is LOW_POWER (onset: 192 labeled / 22 pos / 16 companies).

## Next idea (parent)

Merge at most the onset + own-p20 pair if a no-book Q3 cousin is wanted. Do not merge the Y2/Y3/Y9 masks. Do not reopen net 2-of-3 (base 48%). H on the 110 is a recovery-context feature, not a cobros invention.
