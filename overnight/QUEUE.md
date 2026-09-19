# Night queue (cap 3 — never kill a runner)

Policy: at most **3** concurrent child agents for *new* launches. Overflow
goes here, FIFO. Do not interrupt a running agent to make room. New children
iterate ≥30 minutes on one owner subspace (not one-shot). Parent owns merge
/ LIVE. After a models plateau (no +0.02 CV lift), skip the next GBM and
take an unused **data** or **FE** lane.

Every spawn: child reads `overnight/NORTH_STAR.md` first (X Ray brief).
Scope: analysis / features / explainability. No `product/`, no 0–100 formula.

Deadline: 2026-09-19 08:00 CEST.

## Running (do not stop)

Filled by `overnight/LIVE.json` `slots`.

## FIFO

| pri | id | lane | status | owner files | notes |
|-----|----|------|--------|-------------|-------|
| 1 | y9_fees | data | done | `analysis/targets/y9_fees.py` | ACCEPTED 14.1% / 19.1%. Models never F + a_fin_cost. |
| 1b | gbm_y9 | models | parked | `analysis/models/gbm_y9.py` | CV 0.554 / 0.576; loses or ties single. Label kept. |
| 2 | y3_per_group | models | parked | `analysis/models/gbm_y3_group.py` | CV 0.694 vs global 0.710. Hidden test = new groups. |
| 3 | shap_y3 | explainability | done | `analysis/outputs/y3_importances.md` | Signed TreeSHAP, 50-tree spec. Quiet-stressed recover. a_op_in SIZE. |
| 4 | shap_y7 | explainability | done | `analysis/outputs/shap_y7.md` | f02dbf98 — E 56% / lead 52%. Quote CV 0.663. |
| 5 | fe_interactions | feature engineering | done | `analysis/features/interactions.py` | 12 i_* none SIZE. Not merged into parquet yet. |
| 6 | gbm_y1_rest | models | done | `analysis/models/gbm_y1.py` | 3604ecd0 — KEEP thin in_h3; PARK in_h1 + liq_h3. Path not solved. |
| 7 | xgb_y4 | models | parked | `analysis/models/xgb_y4.py` | f248c993 — trees lose to d_cust_hhi_lag3 0.605. Label stays. |
| 8 | data_category_mix | data | done | `analysis/features/catmix.py` | bac5b3bb — 20 m_*, 95.8% cov, no SIZE. Not merged. |
| 8b | catmix_trail3 | feature engineering | close | `analysis/features/catmix.py` | bac5b3bb — t3 acf1 is overlap; acf3 −0.10. Mix ≠ Q6. |
| 9 | gbm_core_keep | models | close | `analysis/models/gbm_core.py` | 7f37f0fd — A 0.7099 / B 0.693 / shallow-A **0.752 n_x=15**. Do not grow B. |
| 10 | data_join_qa | data | done | `analysis/outputs/data_join_qa.md` | d56ee5fe — 470 yes; Y8 excuse PARK; match KEEP 35.7% vs 0.5%; interco CLOSE. |
| 10b | match_rates | feature engineering | done | `analysis/features/match.py` | d56ee5fe — 3 j_* KEEP-Q5, not Y3 X. 470 NaN. t3 PARK. |
| 11 | y10_util | data | parked | `analysis/targets/y10_util.py` | 14d1bb47 — util impossible. int-on-LOC in-module, not merged. Assembler skips. |
| 12 | dashboard_refresh | data | parent | `overnight/dashboards/` + canvas | Refresh CONTEXT + HTML + canvas whenever SHAP or a new registry block lands. |
| 13 | y7_core_card | models | close | `analysis/models/gbm_y7_core.py` | 410a183a — B_shallow **0.712**/6; TURNOVER **0.720**/5. Do not KEEP 400+ES. |
| 14 | fe_i_lift_y3 | feature engineering | close | `analysis/outputs/i_lift.md` | 88f17954 — CLOSE. C3 0.7615 < 0.772. Do not merge I. |
| 15 | y11_dark | data | close | `analysis/outputs/y11_acceptance.md` | 3d5ad8d8 — no renamed Y2/Y3. Cousins in-module. Do not assemble. |
| 16 | y4_why | explainability | done | `analysis/outputs/y4_why.md` | 79044b5e — 80% crash; HHI 0.605 is monopoly tail. z-avg CLOSE. |
| 17 | trail_len | data | done | `analysis/outputs/trail_length.md` | 6bf54618 — 73.6% is connection; trail 64.2%. PARK created_at Y. |
| 18 | debt_sched | data | done | `analysis/outputs/debt_schedule_qa.md` | 1882a607 — util last-month; rate is a 1.7% growing panel. PARK X. |
| 19 | sibling_h | explainability | done | `analysis/outputs/sibling_h.md` | 5d5b1b81 — PARK H as Y3 X. Sister runway≥1 is a Q5 footnote. |
| 20 | q6_quoted | explainability | done | `analysis/outputs/q6_quoted.md` | c0ddcae1 — KEEP issued_lag1 + days_lag1. CLOSE HHI / F lag3. |
| 21 | y9_why | explainability | done | `analysis/outputs/y9_why.md` | 0511f2af — not outflow, not mix. m_fin leaks. Do not merge M. |
| 25 | y2_why | explainability | done | `analysis/outputs/y2_why.md` | 1bb2643e — ≠Y4. Already-neg persistence. Non-B why fails. |
| 30 | fx_inv | data | done | `analysis/outputs/fx_qa.md` | 97d3db33 — drop e_fx from the 44. Q5 footnote only. |
| 33 | factoring | data | done | `analysis/outputs/factoring_qa.md` | fd90198f — drop f_has_* + f_new_facility from 44. KEEP f_ds_r/f_fc_r. |
| 22 | banking_g | data | done | `analysis/outputs/banking_g_qa.md` | 7e5c43f8 — rise-only connection. Card 0.551 < size. CLOSE as Y3 X. |
| 24 | balances_b | data | done | `analysis/outputs/balances_b_qa.md` | 086b8ed0 — walk identity. Still not leaking. KEEP last-value Q1. |
| 27 | javier_14 | data | done | `analysis/outputs/score_pipeline_qa.md` | 851eac72 — 11 SAME / 2 CLOSE / 1 DRIFT (vol). No 0–100. |
| 31 | a_vol | feature engineering | close | `analysis/outputs/a_vol_qa.md` | 6b456387 — Javier vol CLOSE. Do not merge. |
| 34 | a_out_vol | feature engineering | close | `analysis/outputs/a_vol_qa.md` | 6b456387 — 0.722 trait (demean 0.549). CLOSE / no merge. |
| 39 | recency | data | done | `analysis/outputs/recency_qa.md` | 4545d7a6 — DROP recency leftover 0.607. June flag extract 61/62. |
| 42 | credit_note | data | done | `analysis/outputs/credit_note_qa.md` | 0c3bf32d — KEEP Y7 leftover 0.597. DROP as Y3 X. Do not grow TURNOVER. |
| 45 | g_has_rest | data | done | `analysis/outputs/g_has_rest_qa.md` | 4348a23d — DROP remaining g_has_* + custom from 44. Leftover after days dies. |
| 48 | b_on_44 | data | done | `analysis/outputs/b_on_44_qa.md` | 5e278537 — DROP five B cols as X. b_runway KEEP Q1 description. Leftover dies. |
| 51 | d_tx_cp | data | done | `analysis/outputs/d_tx_qa.md` | 63209d6e — DROP from 44. Twin miss_cp ρ −0.947. KEEP Y5 0.611; PARK as X. |
| 53 | f_fc_r | data | done | `analysis/evaluate/fc_r_qa.py` | 572fb928 — DROP contemp from 44. Unused leftover ρ vs ds_r 0.205. KEEP lag3 on TURNOVER. |
| 23 | y5_why | explainability | done | `analysis/outputs/y5_why.md` | 234af73a — leftover 65%. Not cash, not supp tail. Trees PARK. |
| 26 | clean_flags | data | done | `analysis/outputs/clean_flags_qa.md` | d7235c84 — KEEP never-drop. Holdout extreme mass 33.1%. |
| 28 | tax_ops | data | done | `analysis/outputs/tax_qa.md` | 1094dc70 — calendar dummy. PARK as Y and Y3 X. |
| 29 | uncat | data | done | `analysis/outputs/uncat_qa.md` | ec17da1b — style dummy. PARK as Y and X. Do not merge. |
| 32 | companies | data | done | `analysis/outputs/companies_qa.md` | a14da08b — no transferable Q1 X. erp×dark 671/37/73/433. CLOSE has_erp. |
| 35 | salary | data | done | `analysis/outputs/salary_qa.md` | 0fd41cbf — CLOSE as Y3 X (0.513). PARK as Y. Not tax dummy. |
| 37 | transfer | data | done | `analysis/outputs/transfer_qa.md` | 1bc809f3 — CLOSE. 0.566 leftover after days 0.579. Not wash. |
| 40 | zero_in | data | done | `analysis/outputs/zero_in_qa.md` | 87c77f5b — DROP both from 44. All-out 64%. Leftover dies. Do not revive Y6. |
| 44 | pending | data | done | `analysis/outputs/pending_qa.md` | 86399952 — DROP from 44. Leftover dies (Y7 0.421 / Y3 0.443). Trait. Do not invent y_pending. |
| 47 | supp_hhi | data | done | `analysis/outputs/supp_hhi_qa.md` | ee13dffe — DROP from 44. Twin of top1 ρ 0.987. Y5 tail protective. |
| 50 | brief_map | explainability | done | `analysis/outputs/brief_map.md` | 60e18b0f — six-question morning map. 44-drop as-of 04:50. No 0–100. |
| 52 | cust_hhi | data | done | `analysis/evaluate/cust_hhi_qa.py` | 0940f0fc — DROP as engine X. Twin top1 ρ 0.994. KEEP Y4 tail 0.605. Body 0.445. |
| 36 | missing_cp | data | done | `analysis/outputs/missing_cp_qa.md` | ff4293ad — CLOSE. Twin of d_tx_cp_share (ρ −0.947). Not uncat. |
| 38 | gap_sd | data | done | `analysis/outputs/gap_sd_qa.md` | 87e59905 — DROP from 44. Twin of days (ρ −0.905). Leftover 0.535. |
| 41 | growth | data | done | `analysis/outputs/growth_qa.md` | f4320aab — DROP io/g3. CLOSE g12 as Q6. Leftover dies. |
| 43 | delay | data | done | `analysis/outputs/delay_qa.md` | 47815ca4 — KEEP delay_coll leftover 0.581. DROP as Y3 X. Do not grow TURNOVER. |
| 46 | dpo | data | done | `analysis/outputs/dpo_qa.md` | b4d143b5 — DROP from 44. Y7 leftover dies. Y3 leftover is |DPO|>24 tail. |
| 49 | dso | data | done | `analysis/evaluate/dso_qa.py` | 553c6ea4 — DROP from 44. Y3 leftover 0.474. Y7 leftover after issued 0.452. Do not put back on TURNOVER. |
| 54 | a_n_tx | data | done | `analysis/evaluate/n_tx_qa.py` | fd8977c1 — DROP from the 15-col card. Days twin ρ 0.938. Leftover rank 0.538. Identity of c_n_tx. |
| 55 | f_ds_r | data | done | `analysis/evaluate/ds_r_qa.py` | 572fb928 — DROP from the 15-col card and the 44. Leftover after days rank 0.528. Store flow stays. |
| 56 | issued | data | done | `analysis/evaluate/issued_qa.py` | 553c6ea4 — CLOSE unused leftover. KEEP off the card. Rank leftover 0.608 fake days clone. Y7 leftover after lag1 0.581 CLOSE add-on. |
| 63 | e_ap_issued | data | done | `analysis/evaluate/ap_issued_qa.py` | 553c6ea4 — CLOSE unused leftover. KEEP off the card. Rank leftover 0.591 fake days clone. SIZE ρ 0.514. Do not grow TURNOVER. |
| 70 | e_ap_open | data | done | `analysis/evaluate/ap_open_qa.py` | 553c6ea4 — CLOSE unused leftover. Leftover after days 0.418. Rewrite of issued 0.459. Do not grow TURNOVER. |
| 76 | e_ap_overdue | data | done | `analysis/evaluate/ap_overdue_qa.py` | 553c6ea4 — CLOSE unused leftover. Leftover after days 0.584 lives; beat-size FAIL +0.008. Twin overdue_30 ρ 0.849. Do not grow TURNOVER. |
| 57 | c_ss_month | data | done | `analysis/evaluate/ss_qa.py` | fd8977c1 — KEEP on the 15-col card. Leftover after days 0.635. Q6 KEEP lag1 leftover 0.631. |
| 59 | a_in3 | data | done | `analysis/evaluate/in3_qa.py` | 572fb928 — CLOSE unused leftover. DROP from the 44 as engine X. Leftover after days rank 0.521. Size bar 0.617 stays. |
| 61 | d_n_cust | data | done | `analysis/evaluate/n_cust_qa.py` | 572fb928 — CLOSE unused leftover. DROP from 44. Leftover after days rank 0.545. Twin of top1/HHI. |
| 64 | d_n_supp | data | done | `analysis/evaluate/n_supp_qa.py` | 572fb928 — DROP from 44 as Y3 X. Leftover 0.587 lives but SIZE ρ 0.546. Q6 leftover 0.598 not added to q6_keep. |
| 65 | d_cust_top1 | data | done | `analysis/evaluate/top1_qa.py` | 572fb928 — CLOSE unused leftover. DROP from 44 as Y3 X. Leftover 0.525. Y4 >0.975 footnote KEEP. |
| 66 | a_uncat_share | data | done | `analysis/evaluate/uncat_share_qa.py` | 572fb928 — CLOSE unused leftover. DROP from 44. Leftover 0.573 lives but single 0.542 fails beat-size. ICC 0.985 TRAIT. |
| 67 | h_sib_neg | data | done | `analysis/evaluate/sib_neg_qa.py` | 572fb928 — CLOSE unused leftover. DROP from 44. Leftover 0.453. Q5 sister-runway footnote KEEP. |
| 69 | d_supp_top1 | data | done | `analysis/evaluate/supp_top1_qa.py` | 572fb928 — CLOSE unused leftover. DROP from 44. Leftover 0.429. Twin of HHI ρ 0.987. Y5 tail footnote KEEP. |
| 71 | june_last_tx | data | done | `analysis/evaluate/june_tx_qa.py` | 572fb928 — PARK as extract. DROP from 44. Flag only 2026-06..08; 0 on labeled Y3. Single 0.500. Do not revive Y6. |
| 72 | j_pay_match | data | done | `analysis/evaluate/pay_match_qa.py` | 572fb928 — KEEP-Q5 only. DROP as Y3 X. Single 0.567 fails beat-size. Leftover 0.556 thin. Do not merge J. |
| 73 | e_ar_open | data | done | `analysis/evaluate/ar_open_qa.py` | 572fb928 — CLOSE unused leftover. DROP from 44. Leftover 0.527 dies (fake days OLS 0.719). Rewrite of issued 0.433. |
| 75 | g_n_accounts | data | done | `analysis/evaluate/n_accounts_qa.py` | 572fb928 — CLOSE unused leftover. Leftover after days 0.428 dies. Rise-only 1561/0. Twin g_n_banks ρ 0.880. DROP from 44. |
| 60 | c_tax_month | data | done | `analysis/evaluate/tax_month_qa.py` | fd8977c1 — CLOSE unused leftover. DROP from 44. Leftover after days 0.520. Q-peaked 25.7%. Do not put on card. |
| 62 | f_n_types | data | done | `analysis/evaluate/n_types_qa.py` | fd8977c1 — DROP from 44. Leftover after days CLOSE 0.534. Twin of f_n_facilities ρ 0.994. Rise-only 344/0. |
| 68 | f_ogtg | data | done | `analysis/evaluate/ogtg_qa.py` | fd8977c1 — PARK as snapshot X. DROP from 44. Native Y3 n_pos=0. Honest leftover after last-month days+n_types 0.540. |
| 74 | f_util_snapshot | data | done | `analysis/evaluate/util_snap_qa.py` | fd8977c1 — PARK as snapshot X. Last-month 1.6%. Native Y3 n_pos=0. Hole leftover 0.711 fake days. Y10 impossible. |
| 77 | a_op_out | data | running | `analysis/evaluate/op_out_qa.py` | 572fb928 — unused a_op_out leftover after days. Do not overwrite a_vol_qa / transfer_qa / growth_qa / in3_qa. Do not quote a_out_vol 0.722 as the engine. |
| 78 | a_fin_cost | data | running | `analysis/evaluate/fin_cost_qa.py` | fd8977c1 — unused a_fin_cost leftover after days. Y9 is the fee label. Do not overwrite y9_why / fc_r_qa / catmix. Do not merge M. |
| 79 | d_cust_lost | data | running | `analysis/evaluate/cust_lost_qa.py` | 553c6ea4 — unused d_cust_lost leftover after days. Y7 is top1 lost — different object. Do not overwrite top1_qa / cust_hhi_qa / n_cust_qa / y4_why. Do not grow TURNOVER. |
| 80 | lit_cashflow | explainability | running | `analysis/outputs/lit_cashflow.md` | f6fc63bd — user extra until 10:00. Cash-flow underwriting / FICO-like SME scorecards. Do not overwrite leftover QAs. |
| 81 | lit_invoice | explainability | running | `analysis/outputs/lit_invoice.md` | 689100e7 — user extra until 10:00. Invoice / concentration / trade-credit / lead-time. Do not overwrite leftover QAs. |
| 58 | c_salary_month | data | done | `analysis/evaluate/salary_month_qa.py` | 572fb928 — KEEP on the 15-col card. Leftover after days rank 0.603. Single 0.671 beats size. |

## Done enough (do not relaunch the original)

- XGB Y5 (`0c6820b4`): both accepted Y5s reported. PARK — does not beat single feature; holdout inverted.
- GBM Y1 net/liq_h1 (`0137801b`) + rest (`3604ecd0`): KEEP thin in_h3 only. PARK in_h1 + liq path.
- Feature report (`c343716b`): `analysis/outputs/feature_report.md` + three PNGs.
- Y7/Y8 GBM (`170793d7`): registry rows at 00:13.
- Holdout power (`22a1f47e`): all binaries LOW_POWER.

## Plateaus — pick the next unused lane

If models stall (no CV lift > 0.02 twice): do **data** or **feature engineering**, not another GBM on the same Y.
If data stalls: new Y from a table we under-used (products created_at, debt_schedule 1.7%, category mix).
If FE stalls: SHAP-guided drops + core-keep retrain.
