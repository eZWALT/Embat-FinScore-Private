# Brief map — six questions (morning-quotable)

Generated `2026-09-19T05:01:37+0200` by `e4a91c7b` (`analysis.evaluate.brief_map`). Compile / consistency pass. DuckDB and parquet were **not** opened for a join. No GBM. No `build_targets`. No 0–100. No `product/`. Holdout seed `20260918`; holdout n=72 (coverage only).

Re-read `overnight/NORTH_STAR.md`: the object is a **trajectory** (45→65 recover vs 82→68 deteriorate), not a last-month snapshot. Every row below maps to one of the six questions.

## Night quotes (do not change)

- Y3 shallow 278 **0.762 ± 0.016**. 15-col A **0.752 ± 0.037**. Days bar **0.711**. Size **0.617**.
- Y7 TURNOVER **0.720** / B_shallow **0.712**. Do not quote 278-col 0.663 or holdout 0.680. Do not quote a_out_vol 0.722 as the engine.
- CN leftover after issued_lag1 **KEEP 0.597** — do not grow TURNOVER.
- delay_coll leftover after DSO **KEEP 0.581** — do not grow TURNOVER.

## 15-col Y3 card (do not add)

Stems + lags 1,3: `c_ss_month`, `c_salary_month`, `a_n_tx`, `f_ds_r`, `c_n_days_with_tx`. Never B. Never `a_op_in` (SIZE). `e_dso_proxy` left out. Family I CLOSE.

## Y7 even card (do not grow)

TURNOVER n_x=5: `e_ar_issued_lag1` + issued-lag CV + CN ±lag1 + `f_fc_r_lag3`. No DSO. Quote **0.720** / B_shallow **0.712**.

## Q1 — Who is healthy?

| object | decision | one number | source |
| --- | --- | --- | --- |
| last-value `b_liq` / `b_runway` (Q1 description) | **KEEP** | 0.966 | `balances_b_qa.md` |
| reconstruction walk (identity) | **CLOSE** | 9.095e-12 | `balances_b_qa.md` |
| `companies.csv` flags as transferable Q1 X | **CLOSE** | 0.480 | `companies_qa.md` |
| `created_at` / short-trail as health Y | **PARK** | 73.6% | `trail_length.md` |
| clean flags as health Y / X | **KEEP never-drop / PARK Y** | 24 | `clean_flags_qa.md` |

Notes:

- last-value `b_liq` / `b_runway` (Q1 description): last-vs-snap ρ; persist t↔t+3 = 0.848 (~0.85). Never B as Y2/Y3 X.
- reconstruction walk (identity): median |resid|; do not rewrite liquidity.py.
- `companies.csv` flags as transferable Q1 X: has_erp Y3 vs size 0.617; hidden test is new groups.
- `created_at` / short-trail as health Y: connection clock, not cash trail (first tx late 64.2%).
- clean flags as health Y / X: tx extremes 24; quoted singles do not move (days 0.711).

## Q2 — Who is improving?

| object | decision | one number | source |
| --- | --- | --- | --- |
| `y1_in_h3` thin path | **KEEP** | 0.817 | `y1_rest.md` |
| `y1_in_h1` | **PARK** | 0.799 | `y1_rest.md` |
| Y1 liquidity path (`y1_liq_h3`) | **PARK** | 0.598 | `y1_rest.md` |

Notes:

- `y1_in_h3` thin path: holdout med-norm vs hist 0.856; CV gap 0.005; beat-share 51%.
- `y1_in_h1`: holdout 0.799 vs hist 0.795; CV and holdout disagree.
- Y1 liquidity path (`y1_liq_h3`): last-value wins CV OOF on liq_h3 vs GBM 0.632; Spearman(liq_t, liq_t+3)=0.848.

## Q3 — Who is turning?

| object | decision | one number | source |
| --- | --- | --- | --- |
| Y3 278-col shallow engine | **KEEP** | 0.762 ± 0.016 | `overnight/dashboards/CONTEXT.md` |
| Y3 15-col shallow-A (stems + lags 1/3) | **KEEP** | 0.7520 ± 0.037 | `i_lift.md` |
| days bar `c_n_days_with_tx` | **KEEP** | 0.711 | `q6_quoted.md` |
| Family I (`i_*`) merge onto the 15-col card | **CLOSE** | 0.7615 | `i_lift.md` |
| Y2 trees (82→68 direction) | **PARK** | 0.571 | `y2_why.md` |
| Y9 trees / Family M as Y9 X | **CLOSE** | 0.525 | `y9_why.md` |
| Y6 `y6_zero_in_3` / missed-payroll | **PARK** | 0.854 | `zero_in_qa.md` |

Notes:

- Y3 278-col shallow engine: 50 trees, depth 3. Registry 67e8ef01 mean 0.7622 sd 0.0158. Do not average with 400+ES 0.710.
- Y3 15-col shallow-A (stems + lags 1/3): beats days 0.711. Never B. Never a_op_in. No new cols on the card.
- days bar `c_n_days_with_tx`: quiet-stressed recover bar. Size 0.617. Night quote.
- Family I (`i_*`) merge onto the 15-col card: best add C3 still under KEEP gate 0.772. Leave in-memory.
- Y2 trees (82→68 direction): best legal single days; drop 12 chronic names → 0.549. Direction real; non-B why is not.
- Y9 trees / Family M as Y9 X: legal leftover m_int_share; m_fin 0.618 is the Y (ρ 0.875). FinRegLab NSF/fee. Trees PARK.
- Y6 `y6_zero_in_3` / missed-payroll: Y6 vs size 0.854 — inverse-activity fail. Do not revive.

## Q4 — Dip vs fall?

| object | decision | one number | source |
| --- | --- | --- | --- |
| Y7 TURNOVER (no DSO; issued_lag1 + CV + CN ±lag1 + f_fc_r_lag3) | **KEEP** | 0.720 | `y7_core.md` |
| Y7 B_shallow SHAP card | **CLOSE** | 0.712 | `y7_core.md` |
| Y2 = Y4 crash? | **CLOSE** | 0.057 | `y2_why.md` |
| Y4 `d_cust_hhi_lag3` monopoly tail | **KEEP** | 0.605 | `y4_why.md` |
| Y4 2-col z-avg | **CLOSE** | 0.634 | `y4_why.md` |
| Y4 XGB / LGB trees | **PARK** | 0.585 | `y4_xgb.md` |
| Y8 both (inv / cash) | **PARK** | 53.4% | `data_join_qa.md` |
| Y10 utilisation | **PARK** | 1.65% | `y10_acceptance.md` |
| Y11 renamed Y2/Y3 on the 470 | **CLOSE** | 1.0 | `y11_acceptance.md` |

Notes:

- Y7 TURNOVER (no DSO; issued_lag1 + CV + CN ±lag1 + f_fc_r_lag3): n_x=5; fold 4 0.680. Do not grow. Do not quote 278-col 0.663 or holdout 0.680.
- Y7 B_shallow SHAP card: 6-col named card; DSO fails short quintile 0.410.
- Y2 = Y4 crash?: Jaccard. Y2 median in-ratio 1.03 vs Y4 0.36. Already-neg 82%. Y2 ≠ Y4.
- Y4 `d_cust_hhi_lag3` monopoly tail: HHI>0.975 tail; body ≤0.975 CV 0.445. ρ vs top1_lag3 0.991. CLOSE as Q6.
- Y4 2-col z-avg: zavg is n_supp on those same rows (0.631, gap +0.003).
- Y4 XGB / LGB trees: xgb_es 0.585 loses to the 0.605 single. Do not reopen trees.
- Y8 both (inv / cash): cross-source months; y8_inv had cash on 98.1% and still lost to a_in12.
- Y10 utilisation: no outstanding/granted history; snapshot last-month only. NORTH_STAR ~1.6%.
- Y11 renamed Y2/Y3 on the 470: restricted Y2 ρ=1 with accepted Y2. Do not run assembler.

## Q5 — Why did it change?

| object | decision | one number | source |
| --- | --- | --- | --- |
| CN leftover after `e_ar_issued_lag1` | **KEEP** | 0.597 | `credit_note_qa.md` |
| `e_delay_coll` leftover after DSO | **KEEP** | 0.581 | `delay_qa.md` |
| Y5 unexplained leftover (cash × HHI neither) | **KEEP** | 65% | `y5_why.md` |
| Y5 AR `d_tx_cp_share` (night quote) | **KEEP** | 0.611 | `y5_why.md` |
| missing-CP share as new Q5 | **CLOSE** | -0.947 | `missing_cp_qa.md` |
| Family J `j_pay_match` (amount-match rate) | **KEEP** | 35.7% | `data_join_qa.md` |
| Family M (`m_*`) merge | **CLOSE** | 0.525 | `y9_why.md` |
| Family H as Y3 X / sister mean `b_runway`≥1 | **PARK** | +6.97pp | `sibling_h.md` |
| Y3 SHAP story (all-negative quiet recover) | **KEEP** | 0.276 | `y3_importances.md` |
| Y7 Family E SHAP share | **KEEP** | 56% | `overnight/dashboards/CONTEXT.md` |
| `e_fx_share` as Y3 X / Q5 footnote | **CLOSE** | 0.528 | `fx_qa.md` |
| uncat amount/count as Q5 why | **CLOSE** | 0.542 | `uncat_qa.md` |
| `c_missed_salary` as Y3 / Q5 X | **CLOSE** | 0.513 | `salary_qa.md` |
| `c_missed_tax` as Q5 | **CLOSE** | 0.511 | `tax_qa.md` |
| signed `a_transfer` as Y3 X | **CLOSE** | 0.566 | `transfer_qa.md` |
| Javier `a_vol` / `a_out_vol` as engine | **CLOSE** | 0.626 | `a_vol_qa.md` |

Notes:

- CN leftover after `e_ar_issued_lag1`: ρ issued_lag1 0.237 — not a twin. CLOSE TURNOVER add-on. Do not grow 0.720.
- `e_delay_coll` leftover after DSO: also after issued_lag1 0.583 / both 0.584. ρ vs DSO 0.214. CLOSE TURNOVER add-on.
- Y5 unexplained leftover (cash × HHI neither): AP 65.1% / AR 65.3%. Trees PARK. Banque de France >30d / Hirshleifer PastDue%.
- Y5 AR `d_tx_cp_share` (night quote): CV 0.576. Fold 3 owns 85.9% of hole pos — not a leave-one-group law. PARK as X.
- missing-CP share as new Q5: twin of store d_tx_cp_share. Leftover after uncat+dtx dies 0.509.
- Family J `j_pay_match` (amount-match rate): KEEP-Q5 vs random 0.5%. 470 stay NaN. Not a Y3 X. Do not merge.
- Family M (`m_*`) merge: do not merge. Mix cannot answer Q6. m_fin 0.618 is the Y (ρ 0.875).
- Family H as Y3 X / sister mean `b_runway`≥1: after-size T1. KEEP-Q5 footnote only. H marks sister existence, not sister health.
- Y3 SHAP story (all-negative quiet recover): top SHAP `c_ss_month`; top-10 signs all −. Not dip-vs-fall.
- Y7 Family E SHAP share: lead-time/turning 52%. DSO SHAP #1 fails short-DSO fifth 0.410. shap_y7 e=1.4289.
- `e_fx_share` as Y3 X / Q5 footnote: DROP from the 44. KEEP footnote only (mixed import/export). vs size 0.628 / days 0.711.
- uncat amount/count as Q5 why: Y3 count vs size 0.617. ICC 0.985 bookkeeping style. PARK as X.
- `c_missed_salary` as Y3 / Q5 X: vs size 0.617 / days 0.711. Card already has c_salary_month 0.671.
- `c_missed_tax` as Q5: Q-peaked 69.2% vs 43.4%. PARK as Y and as Y3 X.
- signed `a_transfer` as Y3 X: vs days 0.711. Honest leftover after days 0.579. ICC 0.956 TRAIT.
- Javier `a_vol` / `a_out_vol` as engine: a_vol Δ+0.009 vs size is tercile mix. Do not quote a_out_vol 0.722 as the engine.

## Q6 — How many months earlier?

| object | decision | one number | source |
| --- | --- | --- | --- |
| `e_ar_issued_lag1` (Y7) | **KEEP** | 0.626 | `q6_quoted.md` |
| `c_n_days_with_tx_lag1` (Y3) | **KEEP** | 0.684 | `q6_quoted.md` |
| Y3 C/A lag3 on short companies | **CLOSE** | 36.3% | `q6_quoted.md` |
| F lag3 (`f_ds_r_lag3` / `f_fc_r_lag3`) | **CLOSE** | 16.1% | `q6_quoted.md` |
| Y4 `d_cust_hhi_lag3` as Q6 | **CLOSE** | 21.7% | `q6_quoted.md` |
| delay / overdue as Q6 | **CLOSE** | 0.0% | `delay_qa.md` |
| `e_pending_amt_share` as Q6 | **CLOSE** | 0.430 | `pending_qa.md` |
| `e_dpo_proxy` as Q6 | **CLOSE** | 0.450 | `dpo_qa.md` |
| `d_supp_hhi` as Q6 | **CLOSE** | 0.523 | `supp_hhi_qa.md` |
| `a_growth_12` as Q6 | **CLOSE** | 0.0% | `growth_qa.md` |
| CN lag1 as Q6 | **CLOSE** | 0.542 | `credit_note_qa.md` |
| `next_payment_date` / `f_months_to_next_pay` | **CLOSE** | -5.55 | `debt_schedule_qa.md` |

Notes:

- `e_ar_issued_lag1` (Y7): short so-far 95.2% present; night 0.630 Δ −0.004. Hidden 72: 1-month claims only.
- `c_n_days_with_tx_lag1` (Y3): short 100% present. Contemporaneous days is SIGNAL (0.696), not lead.
- Y3 C/A lag3 on short companies: company-short nn; so-far-short was 84.7% (full4 shift).
- F lag3 (`f_ds_r_lag3` / `f_fc_r_lag3`): Y7 company-short f_fc_r_lag3; empty until so-far≥6.
- Y4 `d_cust_hhi_lag3` as Q6: missing on short half; LOW_POWER 42 pos.
- delay / overdue as Q6: empty first 6 calendar months. Short lag1 0.576 is not issued_lag1.
- `e_pending_amt_share` as Q6: lag1 ≈ now 0.420. Stock populated early (58.3%) but not a lead.
- `e_dpo_proxy` as Q6: short lag1 dies. Exists early (93.7%) unlike delay — still not a lead.
- `d_supp_hhi` as Q6: leftover after cust HHI lag3 dies. Twin of d_supp_top1 ρ=0.987.
- `a_growth_12` as Q6: empty on short books (needs month 15+). n_def short=0.
- CN lag1 as Q6: short so-far vs issued_lag1 0.630.
- `next_payment_date` / `f_months_to_next_pay`: p50 months-to-next; 91.6% already past. KEEP flow f_ds_r / f_fc_r.

## Contradiction hunt (KEEP in one md, DROP in another)

Rule: the **later leftover QA** wins. Feature-report 44 is the starter, not the morning list. Role splits (KEEP leftover / CLOSE TURNOVER add-on; KEEP-Q5 / not Y3 X) are not fights.

| object | KEEP where | DROP where | resolve (later leftover) |
| --- | --- | --- | --- |
| e_credit_note_ratio as Y3 X / the 44 | feature_report.md (44-col starter) | credit_note_qa.md | later leftover: DROP from the 44 as Y3 X (0.579 vs days 0.711; leftover 0.560). KEEP leftover after issued_lag1 0.597 as Q5 footnote. CLOSE TURNOVER add-on. |
| e_delay_coll / overdue as Y3 X / the 44 | feature_report.md (44-col starter); shap_y7.md (Q5 why) | delay_qa.md | later leftover: DROP from the 44 as Y3 X (0.512; leftover-days 0.427). KEEP leftover after DSO 0.581 as Q5 footnote. CLOSE TURNOVER add-on. CLOSE e_delay_paid / e_ar_overdue leftovers (0.449 / 0.547). |
| e_pending_amt_share | feature_report.md (44-col starter) | pending_qa.md | later leftover: DROP from the 44. Leftover dies (Y7 after DSO 0.421 / issued_lag1 0.418). Q6 CLOSE. PARK as health Y. |
| e_dpo_proxy | feature_report.md (44-col starter); y7_core.md TURNDPO 0.723 | dpo_qa.md | later leftover: DROP from the 44. Y7 leftover CLOSE (0.471 / 0.443). TURNDPO 0.723 is not KEEP. Q6 CLOSE. |
| d_supp_hhi | feature_report.md (44-col starter) | supp_hhi_qa.md | later leftover: DROP from the 44. Twin of d_supp_top1 ρ=0.987. Y5 tail protective 2.7% vs 8.6% — not a Y4 crash. PARK as Y5 X (size-rank 0.663). |
| c_gap_sd | feature_report.md (44-col starter); shap_y3.md rank 8 | gap_sd_qa.md | later leftover: DROP from the 44. Twin of days ρ −0.905. Leftover after days 0.535 dies. Days leftover after gap 0.658 still lives. |
| c_recency_days | feature_report.md (44-col starter) | recency_qa.md | later leftover: DROP from the 44. Leftover after days 0.607 dies. June flag PARK as extract artifact (Javier 61 vs as-of 62; COMP_0981 is the +1). |
| a_io_ratio / a_growth_3 | feature_report.md (44-col starter) | growth_qa.md | later leftover: DROP both from the 44. Leftover after size+days dies (0.527 / 0.522). a_growth_12 CLOSE as Q6 (empty on short). |
| c_zero_in_month / c_zero_in_share_6 | feature_report.md (44-col starter) | zero_in_qa.md | later leftover: DROP from the 44. Leftover after days dies (0.553 / 0.537). Do not revive Y6. |
| e_fx_share | feature_report.md (44-col starter) | fx_qa.md | later leftover: DROP from the 44 as Y3 X. KEEP only as a Q5 footnote (mixed import/export). Q6 lag1 CLOSE. |
| f_has_factoring / f_has_confirming / f_has_loc / f_new_facility | feature_report.md (44-col starter) | factoring_qa.md | later leftover: DROP from the 44. Rise-only connection inventory. Y3 0.505 / 0.485 / 0.546 vs size 0.617. PARK as health Ys. KEEP flow f_ds_r / f_fc_r (debt_schedule_qa.md). |
| remaining g_has_* + g_custom_share | feature_report.md (44); banking_g_qa.md (g_has_* all in starter=True; CLOSE as Y3 X, card 0.551) | g_has_rest_qa.md | later leftover: DROP remaining g_has_saving / investment / tpv + g_custom_share. Honest leftover after days dies (OLS 0.65–0.71 is a fake days leak). Card already CLOSE 0.551; checking DROP (99.1% hole). banking_g confirmed they were on the starter, not that they stay as engines. |
| a_uncat_share / amount-uncat as Q5 | feature_report.md (keep-list) | uncat_qa.md | later leftover: CLOSE as Q5. PARK as Y and as X. ICC 0.985 style dummy. Do not merge amount-uncat. |
| c_missed_salary as Y3 X | feature_report.md (rare-event keep) | salary_qa.md | later leftover: CLOSE as Y3 / Q3–Q5 X. PARK as health Y. Lever is c_salary_month 0.671 already on the 15-col card. |
| Family J as Y3 X vs KEEP-Q5 | match_report.md / data_join_qa.md (KEEP-Q5 / KEEP amount-match rate) | not a DROP — role split | not a contradiction. KEEP as Q5 diagnostic (35.7% vs 0.5% random). Not a Y3 X (pay vs recover −0.060). Do not merge parquet. |
| Family M as Q5 mix vs Y9 merge | catmix_report.md (maps m_* to Q5 why) | y9_why.md (CLOSE merge) | later leftover: CLOSE merge. Fee/fin shares are the Y9 label (ρ 0.875 / 0.830). Mix cannot answer Q6. |
| a_out_vol 0.722 as engine | a_vol_qa.md (Y3 0.722 reproduced) | a_vol_qa.md (later-store KEEP False; X CLOSE) | same md, role split. Reproduced number is a company trait (demean 0.549, η² 0.741), not the Y3 engine. Do not quote 0.722 as the engine. Night quote stays 0.762 / 0.752. |
| Y7 278-col 0.663 vs TURNOVER 0.720 | shap_y7.md quotes 0.663 as this-run CV | y7_core.md (do not quote 278-col 0.663) | later core card: quote TURNOVER 0.720 / B_shallow 0.712. shap_y7.md is the 278-col SHAP story (Family E 56%), not the even card. |
| B columns on the 44 / DSO on the 44 | feature_report.md (b_runway, b_bal_vol, …, e_dso_proxy in the 44) | b_on_44_qa.md / dso_qa.md exist after CONTEXT 04:50 | IN-FLIGHT as of the 04:50 compile. Do not invent a settled DROP. balances_b_qa.md already KEEP last-value as Q1 description and forbids B as Y2/Y3 X. y7_core.md already drops DSO from TURNOVER. 44-drop list as of 04:50 does not wait for these two. |
| Y4 d_cust_hhi_lag3 KEEP as monopoly tail vs CLOSE as Q6 | y4_why.md (0.605 is the HHI>0.975 tail) | q6_quoted.md (21.7% present on short; LOW_POWER 42 pos) | role split, not a fight. KEEP as the Q4 single (monopoly tail). CLOSE as a 3-month lead on short / hidden-72 books. |
| Y4 2-col z-avg CLOSE vs PARK | y4_xgb.md (PARK as next idea, not XGB KEEP) | y4_why.md (CLOSE — is n_supp on the same rows) | later leftover: CLOSE the z-avg card. Trees stay PARK. Do not reopen. |
| created_at CLOSE as Y3 X vs PARK as health Y | not KEEP — companies_qa.md CLOSE as Y3 X (CV 0.504) | trail_length.md PARK as health Y (73.6% connection clock) | role split. PARK as a health Y. CLOSE as a transferable X. Same object, two jobs, same answer: not health. |
| e_fx_share CLOSE as Q5 X vs KEEP footnote | fx_qa.md (KEEP footnote — mixed import/export) | fx_qa.md (CLOSE as Y3 X / DROP from the 44) | same md, role split. DROP from the 44. KEEP only as a Q5 footnote. |
| d_tx_cp_share KEEP 0.611 vs DROP from the 44 | y5_why.md (KEEP quote 0.611 / PARK as X) | d_tx_qa.md (05:00 — after the 04:50 cut) | post-04:50 leftover. Do **not** add to the 04:50 44-drop list. If morning reads the 05:00 file: KEEP the 0.611 quote, PARK as X, DROP from the 44 as Y3 X (leftover-after-days 0.600). Same fold-3 hole 85.9%. Twin of miss_cp ρ −0.947 already CLOSE as new Q5. |

Parser also saw **0** table-level KEEP+DROP collisions (object-string match). Same-row leftover DROP/CLOSE on starter-44 names: **0** expected (in the 04:50 drop list), **0** unexpected. In-flight B / DSO are not resolved.

## 44-drop list as of 04:50

Do not wait for in-flight B-on-44 / DSO. Feature-report starter was 44; night leftover QAs have since dropped:

- `e_fx_share`
- `f_has_factoring / f_has_confirming / f_has_loc / f_new_facility`
- `c_gap_sd`
- `c_recency_days`
- `a_io_ratio`
- `a_growth_3`
- `c_zero_in_month / c_zero_in_share_6`
- `e_credit_note_ratio (as Y3 X)`
- `e_delay_coll / e_delay_paid / e_ar_overdue / e_*_overdue_30 (as Y3 X)`
- `e_pending_amt_share`
- `remaining g_has_* + g_custom_share (card CLOSE 0.551; checking 99.1% hole)`
- `e_dpo_proxy`
- `d_supp_hhi`

**In-flight (do not invent a verdict):**

- B-on-44 (b_* leftover on the 44)
- DSO (e_dso_proxy leftover on the 44)

  - `b_on_44_qa.md` exists after CONTEXT 04:50: **True**. Verdict still in-flight for this compile.
  - `dso_qa.md` exists after CONTEXT 04:50: **True**. Verdict still in-flight for this compile.

Still on the 44 as of 04:50 (not a KEEP-as-engine): `a_growth_12`, `a_in3`, `a_uncat_share`, `b_bal_vol`, `b_below_0`, `b_d_runway`, `b_neg_episodes`, `b_runway`, `c_last_tx_before_2026_06`, `c_missed_salary`, `c_missed_tax`, `d_cust_hhi`, `d_tx_cp_share`, `e_dso_proxy`, `f_ds_r`, `f_fc_r`, `f_outstanding_gt_granted`, `h_sib_neg_share`. Read: `a_in3` size control; `a_growth_12` CLOSE as Q6; `a_uncat_share` CLOSE Q5; `c_missed_salary` / `c_missed_tax` CLOSE/PARK as X; `d_cust_hhi` Y4 monopoly-tail single / CLOSE as Q6; `d_tx_cp_share` Y5 KEEP quote / PARK as X; `e_dso_proxy` **in-flight**; `f_ds_r` / `f_fc_r` KEEP flow; `f_outstanding_gt_granted` PARK snapshot; `h_sib_neg_share` PARK as Y3 X; five B columns **in-flight** as X (`b_runway` already KEEP as Q1 description).

`d_tx_qa.md` appeared **05:00** (after the 04:50 cut). Do not fold `d_tx_cp_share` into the 04:50 drop list. After-cut verdict if morning reads it: KEEP 0.611 quote, PARK as X, DROP from the 44 as Y3 X.

## What morning can say (six sentences)

1. Healthy, tonight, is last-value reconstructed cash / `b_runway` on the 2026-09 still (last-vs-snap ρ 0.966; persist 0.85) — not `companies.csv` (has_erp Y3 0.480) and not `created_at` (connection clock, 73.6% late).
2. Improving is only a thin inflow path: KEEP `y1_in_h3` (holdout med-norm 0.817 vs hist 0.856); the Y1 liquidity path stays PARK because last-value already wins (Spearman 0.85).
3. Turning toward 45→65 is Y3 quiet-stressed recover: 278-col shallow 0.762 ± 0.016, 15-col A 0.752 ± 0.037 beating days 0.711 / size 0.617; Family I CLOSE (best add 0.7615 < 0.772); no new column on the 15-col card.
4. Dip vs fall is Y7 TURNOVER 0.720 / B_shallow 0.712 on issued_lag1 (not DSO; not 278-col 0.663); Y2 is already-negative persistence (Jaccard vs Y4 0.057), Y4 is the HHI>0.975 monopoly tail (0.605), Y8 PARK.
5. Why: CN leftover after issued_lag1 KEEP 0.597 and delay leftover after DSO KEEP 0.581 as footnotes (do not grow TURNOVER); Y5 is a 65% leftover; missing-CP is the `d_tx_cp_share` twin (ρ −0.947); Family J KEEP-Q5 not Y3 X; Family M CLOSE.
6. Months earlier: KEEP issued_lag1 (short 0.626 / 95.2%) and days_lag1 (short 0.684 / 100%); CLOSE F lag3, Y4 HHI, delay, pending, DPO, supp HHI, and growth_12 — hidden 72 is 1-month claims only.

## What morning cannot say

- A 0–100 index, pillars, or product/web/Docker story.
- Holdout AUROC as a trophy. Hidden 72 is LOW_POWER (7–23 events) except `y7_top1_lost` (122 pos). Quote train group-fold CV.
- Hidden-72 lead time beyond one month (full-24 books 4.2% vs train 35.8%; Y4 HHI_lag3 almost undefined). `y7_top1_lost` holdout 122 events is the exception — still do not quote holdout AUROC.
- 278-col Y7 0.663, holdout Y7 0.680, or `a_out_vol` 0.722 as the engine.
- Y2 / Y4 / Y5 / Y9 trees as a why (all PARK). Family I/M/J parquet merges.
- New columns on the 15-col Y3 card, or a grown TURNOVER card.
- Settled B-on-44 or DSO leftover verdicts — those two were in-flight at the 04:50 compile.

## SHAP story (already written)

- **Y3:** among already-stressed months, all top-10 SHAP signs are **−**. Quiet firms (no SS/salary, fewer txs) are the ones whose runway later prints ≥ 3 for three months. That is de-escalation / mean reversion at the bottom — the 45→65 direction. It does **not** yet separate dip from fall. Canonical: `y3_importances.md` (50-tree). Do not mix parent 400-tree rank order.
- **Y7:** Family E holds **56%** of mean |SHAP|; lead-time / turning holds **52%**. DSO is SHAP #1 and **worse than chance** on the short-DSO fifth (0.410). TURNOVER drops DSO; fold 4 0.680 is issued, not DSO. Canonical: `shap_y7.md` + `y7_core.md`.

## Literature already on the desk (do not invent papers)

- FinRegLab 2025 NSF/fee / low-negative balances — NORTH_STAR Y9 turning/why; journal `2026-09-18-2350-feature-store-and-y-plan.md`.
- Banque de France Bulletin 227/8: late payments **>30d** move PD; ≤30d do not. Y5 is that clock; the 65% leftover is what cash/HHI do not explain.
- Hirshleifer et al. 2019 PastDue% — same Y5 plan note. Not a new paper.

## Compile checks

- Holdout companies: **72** (want 72). Seed `20260918`.
- Map rows: **52**. Source-file token+number misses: **0**.
- 44-drop leftover confirm misses: **0**.
- Registry 67e8ef01 / 0.762239: **ok**.
- Resolved KEEP-vs-DROP objects: **24**. Raw parser collisions: **0**.
- In-flight: B-on-44 (b_* leftover on the 44), DSO (e_dso_proxy leftover on the 44).
- Parquet / models: not opened / not fit.

Row compile is clean: every number and decision token is in its source md.

## What this module did not do

- Did not change night Y3 0.762 / 0.752 or Y7 TURNOVER 0.720 / B_shallow 0.712.
- Did not put anything on the 15-col card. Did not grow TURNOVER.
- Did not invent B-on-44 or DSO leftover verdicts.
- Did not rewrite parquet or duckdb. Did not run `build_targets`. Did not fit. Did not commit.
- Did not write the parent journal / LIVE / canvas / `product/`.

## Re-run

```bash
/home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.brief_map
```
