# Contextualized night summary — analysis / features / explainability

- **As of:** 2026-09-19 ~09:15 CEST — night closed; all seats idle
- **Morning briefing (start here):** `overnight/dashboards/MORNING_REPORT.md` and `overnight/dashboards/morning.html`
- **Brief:** `overnight/NORTH_STAR.md` (working copy of the X Ray artifact)
- **Product:** out of scope tonight

## What the contest actually asked

Can a 24-month treasury trail say how a company is — not whether it will
default. The brief’s pictures are a **recovery** (45→65) and a
**deterioration** (82→68). Same last month can hide the better risk, so the
object is a trajectory plus an explanation, plus lead time.

Tonight we only build the engine evidence: signals, proxy Ys that do not
leak into X, models that beat naive baselines on **train group-fold CV**,
and SHAP so “why” is not a black box. No 0–100. No web.

## What we built that is real

- Monthly panel **22,230 × 118** from all eight tables. Weekly grid exists;
  weekly TS models lost to the historical mean and are parked.
- Eight feature families A–H, dictionary, coverage, train-only feature
  report (drop 29 / park 16 / keep 44). Night has since dropped `e_fx`,
  `f_has_*`, **`c_gap_sd`**, **`c_recency_days`**, **`a_io_ratio`**,
  **`a_growth_3`**,   **`c_zero_in_*`**, **`e_credit_note_ratio`** (as Y3 X), and
  **delay / overdue** (as Y3 X), **`e_pending_amt_share`**, and remaining
  **`g_has_*` + `g_custom_share`**,   **`e_dpo_proxy`**, **`d_supp_hhi`**, and   leftover **B cols as Y2/Y3 X**
  (`b_runway` stays Q1 description only), **`d_tx_cp_share`**, and
  **`d_cust_hhi` as engine X** (Y4 >0.975 tail footnote stays), and
  contemporaneous **`f_fc_r`** (`f_fc_r_lag3` stays on TURNOVER), and
  **`e_dso_proxy`**, and **`f_ds_r`** from that 44, and **`a_in3` as
  engine X** (size *bar* 0.617 stays), and **`c_tax_month`**, and
  **`d_n_cust`**, and **`d_n_supp`**, and **`d_cust_top1` as Y3 X**
  (Y4 >0.975 footnote stays), and **`a_uncat_share`**, and
  **`f_n_types`**, and **`h_sib_neg_share`**, and **`d_supp_top1`**,
  and **`c_last_tx_before_2026_06`**, and **`f_outstanding_gt_granted`**,
  and **`e_ar_open`**, and **`e_ap_open`**, and
  **`g_n_accounts`**, and **`f_util_snapshot`**, and **`e_ap_overdue`**,
  and **`a_op_out`**, and **`a_fin_cost`**, and **`a_debt_service`**,
  and **`d_cust_lost`**.
  Size control is
  `log1p(a_in3)`, not raw this-month inflow (acf1 ≈ 0).
- **`c_gap_sd`** ([regularity QA](87e59905-b29f-452d-963a-854dcb3e8895)):
  **DROP from the 44. CLOSE as X. PARK as a Y.** Twin of days
  (ρ **−0.905**) and `a_n_tx` (−0.866). Y3 **0.686** vs size 0.617 vs
  days **0.711**. Leftover after days **0.535** dies. Days leftover
  after gap **0.658** still lives — not interchangeable. Feature
  report kept the weaker twin; the Y3 engine was right. Midnight
  collapse CONFIRMED (raw σ vs 1/n_tx 0.916). Do not rewrite
  `ops.py`. Not on the 15-col card.
- **Growth / coverage** ([shape QA](f4320aab-a766-43dd-9809-ba2267fe2918)):
  **DROP `a_io_ratio` and `a_growth_3` from the 44.** **CLOSE
  `a_growth_12` as Q6.** **PARK** as Ys. Y3 io **0.565** / g3 **0.515**
  / g12 **0.526** vs size 0.617 / days 0.711. Leftover after size+days
  **dies** (0.527 / 0.522 / 0.399). Mean-reversion is a **U-shape** plus
  mechanical acf3 **−0.433**, not monotone. io is a `a_net_margin` twin
  (ρ 0.989). growth_12 empty on short books; holdout **69/72**
  late-arrival. Do not invent `y_growth`. Do not merge with Y4.
- **Zero-in** ([activity QA](87c77f5b-2d51-4a01-a2fe-7fd4c634f4c3)):
  **DROP `c_zero_in_month` and `c_zero_in_share_6` from the 44.**
  **CLOSE as X. PARK as Ys.** The flag is **all-out** (64.2% of
  positives: txs with no amount>0), not empty-grid (35.8%). Y3
  **0.580 / 0.617** vs size 0.617 / days 0.711. Leftover after days
  **dies** (0.553 / 0.537). SIZE on `log1p(|a_op_in|)` (−0.508 /
  −0.519). Y6 `y6_zero_in_3` vs size **0.854** — same inverse-activity
  fail. Do not revive Y6. Off the 15-col card.
- **Credit notes** ([leftover QA](0c3bf32d-f4d0-4219-bd00-202d72a985db)):
  **KEEP** as a Y7 leftover after `e_ar_issued_lag1` (residual **0.597**;
  ρ **0.237**, not a twin). Demean leftover **0.509** dies — who-uses-notes,
  not this month’s correction. **CLOSE** as a TURNOVER add-on: do not
  grow **0.720 / 0.712**. **DROP from the 44** as Y3 X (**0.579** vs
  days **0.711**; leftover **0.560**). **PARK** as a health Y.
  **KEEP-Q5 footnote**. **CLOSE** as Q6 (short lag1 **0.542**). Fold 4
  **0.680** is issued, not CN. Fallback **note 98.6% / refund 1.4%**.
  Dark 470 stay NaN (COMP_0962 refund-only). Do not invent
  `y_credit_note`.
- **Delay / overdue** ([leftover QA](47815ca4-ee6d-4a51-b38e-8fdba454f445)):
  **KEEP** `e_delay_coll` as a Y7 leftover after DSO (**0.581**) and
  issued_lag1 (**0.583**; both **0.584**). Not a twin (ρ vs DSO **0.214**).
  Leftover ≈ same-n raw; ICC **0.922** / demean **0.521** — who-pays-late
  style, not a month shock. **CLOSE** as a TURNOVER add-on: do not
  change **0.720 / 0.712**. **CLOSE** `e_delay_paid` / `e_ar_overdue`
  leftovers (0.449 / 0.547). **DROP from the 44** as Y3 X (**0.512**;
  leftover after days **0.427** vs days **0.711**). **CLOSE** Q6 (empty
  until month 7). **CLOSE** 3m-window leftover (**0.509**). **PARK** as
  a health Y. Javier delay **ρ=1 SAME**. Fold 4 **0.680** is issued,
  not delay. Dark 470 stay NaN. Do not invent `y_delay`.
- **Pending amount-share** ([leftover QA](86399952-9dd9-44b6-b202-4df7d45d75de)):
  **DROP `e_pending_amt_share` from the 44** as Y7/Y3 X. Leftover dies
  (Y7 after DSO **0.421** / issued_lag1 **0.418**; Y3 after days
  **0.443**). Not SIZE (ρ **0.044**). Not a twin. Q6 **CLOSE** — stock
  is populated in the first 6 months (58.3% vs delay 0.0%) but lag ≈
  now. ICC **0.971** η² **0.672** trait. **PARK** as a health Y.
  TURNPEND 0.7184 already lost to TURNOVER 0.720. Fold 4 pending 0.363
  vs issued 0.647. Do not invent `y_pending`. Do not merge Family J.
- **Remaining `g_has_*`** ([leftover QA](4348a23d-c928-4bc6-affa-75f3c76cc61f)):
  **DROP from the 44 / CLOSE as Y3 X / PARK as Y** for `g_has_saving`
  (n=9, Y3 **0.501**), `g_has_investment` (**0.509**, T3-tagged),
  `g_has_tpv` (n=10), and `g_custom_share` (**0.530**; leftover after
  accounts **0.548**). Confirm card CLOSE **0.551**; checking DROP
  (99.1% hole, Jaccard **0.991**). Honest leftover after days **dies**
  (OLS 0.65–0.71 is a fake days leak). HAS flags rise-only; custom_share
  51↓ is inventory dilution. Dark invest 8.9% > ERP 6.2% (access ≠ ERP).
  Do not invent `y_has_tpv`.
- **DPO** ([leftover QA](b4d143b5-025b-479f-8569-c4affc344f93)):
  **DROP `e_dpo_proxy` from the 44.** Y7 leftover **CLOSE** (after DSO
  **0.471** / issued_lag1 **0.443**). Y3 leftover after days **0.653**
  is a **|DPO|>24 tail** (drop-tail **0.432**) and loses to days
  **0.711**. Not a twin (ρ DSO **0.456**). Not SIZE (ρ **−0.025**).
  Y5 as X **FORBIDDEN**. **PARK** as a health Y. Q6 **CLOSE**. TURNDPO
  0.723 is not KEEP. Clip-at-24 is model-layer only. Dark 470 NaN.
  Do not invent `y_dpo`.
- **DSO** ([leftover QA](553c6ea4-a745-42a3-962a-2299b600d939)):
  **DROP `e_dso_proxy` from the 44.** Y3 leftover after days **0.474**
  dies (not only a |DSO|>24 tail — drop-tail **0.424**). Y7 leftover
  after issued_lag1 **0.452** CLOSE. Leftover after delay **0.561** is
  a |DSO|>24 tail — SHAP #1 is not delay (ρ **0.214**). Short-DSO Q1
  univariate **0.455** confirms the 0.410 hole. Fold 4 issued **0.647**
  vs DSO **0.342**. **PARK** as a health Y. Q6 **CLOSE**. Do not put
  DSO back on TURNOVER **0.720**. Store SAME. Dark 470 NaN. Do not
  invent `y_dso`.
- **`a_n_tx`** ([leftover QA](fd8977c1-61b4-42c3-8c84-8f3faa2bb904)):
  **DROP from the 15-col Y3 card** (parent decision; do not rewrite
  `gbm_core.py` tonight). COUNT(*) identity of `c_n_tx`. Days twin
  ρ **0.938**. SIZE ρ **0.661**. Leftover after days rank **0.538**
  (OLS 0.623 rank-dies). Inverse days-after-n_tx **0.564** thin —
  days **0.711** stays. Perm 0.030 is the SIZE_STEM rewrite. Q6
  **CLOSE**. Do not invent `y_n_tx`.
- **`f_ds_r`** ([leftover QA](572fb928-40f3-4948-9ab9-0fde761a9d50)):
  **DROP from the 15-col card and the 44.** Unused leftover after days
  rank **0.528** (OLS 0.709 almost-fake days leak). Inverse days
  **0.682**. Single **0.620** fails beat-size (Δ 0.003). Leftover
  after fc **0.606**; ρ vs fc **0.205**. Twin of euro ds ρ **0.881**.
  Store flow stays. `f_fc_r_lag3` stays on TURNOVER. Javier
  `debt_serv_r` SAME. Do not invent `y_ds_r`.
- **`c_salary_month`** ([leftover QA](572fb928-40f3-4948-9ab9-0fde761a9d50)):
  **KEEP on the 15-col card.** Leftover after days rank **0.603**
  lives. Inverse days **0.647**. Single **0.671** beats size by
  **0.055**. Not SIZE (ρ **0.333**). Not a twin. Leftover after SS
  **0.626**. Calendar monthly. Q6 lag1 leftover **0.606**. Dark
  leftover dies (0.547); ERP 0.634 lives. `c_missed_salary` stays
  CLOSE 0.513. Do not invent `y_salary`.
- **`c_ss_month`** ([leftover QA](fd8977c1-61b4-42c3-8c84-8f3faa2bb904)):
  **KEEP on the 15-col card.** Leftover after days **0.635**. Leftover
  after salary **0.668** (not a payroll twin). Inverse days **0.641**.
  Single **0.693** beats size by **0.077**. BETWEEN payer identity
  (ICC **0.990**). **Q6 KEEP** lag1 leftover **0.631**. Calendar
  monthly. Do not invent `y_ss`.
- **`log1p(a_in3)`** ([leftover QA](572fb928-40f3-4948-9ab9-0fde761a9d50)):
  **CLOSE as unused leftover. DROP from the 44 as engine X.** Honest
  leftover after days rank **0.521** dies (OLS 0.531; not a fake
  leak). Inverse days **0.667**. Size **0.617** stays the KEEP-as-X
  *bar*, not a card stem. Twin of `a_op_in` / `a_in6` / `a_in12`.
  Q6 lag1 leftover **0.527** dies. Do not invent `y_in3`.
- **`c_tax_month`** ([leftover QA](fd8977c1-61b4-42c3-8c84-8f3faa2bb904)):
  **CLOSE unused leftover. DROP from the 44 as Y3 X.** Leftover after
  days **0.520** dies. Single **0.611** fails beat-size (Δ −0.006).
  Calendar **q_peaked** (gap **25.7%**), not a salary/SS cousin. Not
  the complement of missed-tax (Jaccard 0.000). Q6 **CLOSE**. Do not
  invent `y_tax`. Do not overwrite `tax_qa.py`.
- **`e_ar_issued` now** ([leftover QA](553c6ea4-a745-42a3-962a-2299b600d939)):
  **CLOSE unused leftover. KEEP off the 15-col card.** Rank leftover
  after days **0.608** is a fake days clone (OLS 0.694). Twin of
  `issued_lag1` (ρ **0.814**). Single **0.687** vs days **0.711**.
  Y7 leftover after lag1 **0.581** — **CLOSE** TURNOVER add-on.
  `issued_lag1` Q6 **KEEP locked**. Do not invent `y_issued`.
- **`d_n_cust`** ([leftover QA](572fb928-40f3-4948-9ab9-0fde761a9d50)):
  **CLOSE unused leftover. DROP from the 44 as Y3 X.** Leftover after
  days rank **0.545** dies. Twin of top1 (ρ **−0.806**) and HHI
  (ρ **−0.837**). Single **0.653** beats size but leftover dies. Y4
  leftover after top1_lag3 **0.530**. `d_n_supp` is a different object
  (ρ **0.632**). Do not invent `y_n_cust`.
- **`d_n_supp`** ([leftover QA](572fb928-40f3-4948-9ab9-0fde761a9d50)):
  **DROP from the 44 as Y3 X.** Leftover after days rank **0.587**
  lives and beat-size PASSES, but SIZE (ρ **0.546**). Not a twin of
  n_cust (ρ **0.632**). Leftover after days+top1 **0.543** dies. Q6
  leftover **0.598** lives — not added to q6_keep. Do not invent
  `y_n_supp`.
- **`d_cust_top1`** ([leftover QA](572fb928-40f3-4948-9ab9-0fde761a9d50)):
  **CLOSE unused leftover. DROP from the 44 as Y3 X.** Leftover after
  days rank **0.525** dies. Single **0.590** fails beat-size. Twin of
  HHI (ρ **0.994**). **Y4 >0.975 footnote KEEP.** `d_supp_top1` is a
  different object (ρ **0.117**). Do not invent `y_top1`.
- **`a_uncat_share`** ([leftover QA](572fb928-40f3-4948-9ab9-0fde761a9d50)):
  **CLOSE unused leftover. DROP from the 44 as Y3 X.** Leftover after
  days rank **0.573** lives but single **0.542** fails beat-size.
  ICC **0.985** TRAIT. Not a miss_cp twin (ρ **0.131**). **PARK as Y.**
  Do not invent `y_uncat`. Do not overwrite `uncat_qa.py`.
- **`f_n_types`** ([leftover QA](fd8977c1-61b4-42c3-8c84-8f3faa2bb904)):
  **DROP from the 44 as Y3 X.** Leftover after days **CLOSE 0.534**.
  Twin of `f_n_facilities` (ρ **0.994**). Single **0.578** fails
  beat-size. Rise-only **344 / 0**. Q6 **CLOSE**. PARK as a health Y.
  Do not invent `y_n_types`.
- **`h_sib_neg_share`** ([leftover QA](572fb928-40f3-4948-9ab9-0fde761a9d50)):
  **CLOSE unused leftover. DROP from the 44 as Y3 X.** Leftover after
  days rank **0.453** dies. Single **0.434** fails beat-size. **PARK
  as Y** — sister existence; hidden test is new groups. **Q5
  sister-runway ≥1 footnote KEEP.** Do not invent `y_sib_neg`.
- **`e_ap_issued` now** ([leftover QA](553c6ea4-a745-42a3-962a-2299b600d939)):
  **CLOSE unused leftover. KEEP off the 15-col card.** Rank leftover
  after days **0.591** is a fake days clone (same as AR issued 0.608).
  Single **0.675** vs days **0.711**. SIZE ρ **0.514**. Leftover after
  AR+days **0.524** dies. Do not grow TURNOVER. Do not invent
  `y_ap_issued`.
- **`d_supp_top1`** ([leftover QA](572fb928-40f3-4948-9ab9-0fde761a9d50)):
  **CLOSE unused leftover. DROP from the 44 as Y3 X.** Leftover after
  days rank **0.429** dies. Twin of HHI (ρ **0.987**). Single **0.641**
  barely beats size. vs cust top1 ρ **0.117**. **Y5 protective-tail
  footnote KEEP.** Do not invent `y_supp_top1`.
- **`c_last_tx_before_2026_06`** ([leftover QA](572fb928-40f3-4948-9ab9-0fde761a9d50)):
  **PARK as extract. DROP from the 44 as Y3 X.** Flag only 2026-06..08;
  identically 0 on labeled Y3. Single **0.500**. Honest leftover
  **0.500 const**. Javier **61 vs 62**; COMP_0981 is the +1. Do not
  invent `y_june`. Do not revive Y6.
- **`j_pay_match`** ([leftover QA](572fb928-40f3-4948-9ab9-0fde761a9d50)):
  **KEEP-Q5 only. DROP as Y3 X. Do not merge Family J.** Single
  **0.567** fails beat-size. Leftover after days **0.556** thin. Not a
  pending twin (ρ **−0.093**). Dark 470 stay NaN. Do not invent
  `y_pay_match`.
- **`f_outstanding_gt_granted`** ([leftover QA](fd8977c1-61b4-42c3-8c84-8f3faa2bb904)):
  **PARK as snapshot X. DROP from the 44 as Y3 X.** Native Y3 undefined
  on the 2026-08 extract (n_pos=0). fillna0 leftover 0.711 is a fake
  days leak. Honest leftover after last-month days+n_types **0.540**.
  Last-month-only cov **5.7%**. Q6 **CLOSE**. Do not invent a Y.
- **`e_ar_open`** ([leftover QA](572fb928-40f3-4948-9ab9-0fde761a9d50)):
  **CLOSE unused leftover. DROP from the 44 as Y3 X.** Leftover after
  days **0.527** dies (OLS 0.719 fake days clone). Single **0.587**
  fails beat-size. Rewrite of issued (leftover **0.433**). Not the
  DSO numerator.   Do not invent `y_ar_open`. Do not grow TURNOVER.
- **`e_ap_open`** ([leftover QA](553c6ea4-a745-42a3-962a-2299b600d939)):
  **CLOSE unused leftover. KEEP off the 15-col card.** Leftover after
  days rank **0.418** dies. Single **0.569** fails beat-size. Rewrite
  of issued (leftover **0.459**). Leftover after DPO+days **0.429**.
  Do not invent `y_ap_open`. Do not grow TURNOVER.
- **`g_n_accounts`** ([leftover QA](572fb928-40f3-4948-9ab9-0fde761a9d50)):
  **CLOSE unused leftover. DROP from the 44 as Y3 X.** Leftover after
  days rank **0.428** dies. Single **0.581** vs days **0.711**. Twin of
  `g_n_banks` (ρ **0.880**). Rise-only **1561/0**. Checking hole
  **99.1%**. Dark last p50=3 = ERP last p50=3 (access ≠ ERP). **PARK**
  as a Y. Do not invent `y_n_accounts`.
- **`f_util_snapshot`** ([leftover QA](fd8977c1-61b4-42c3-8c84-8f3faa2bb904)):
  **PARK as snapshot X. DROP from the 44 as Y3 X.** Last-month-only
  **1.6%** (334/21,157). Native Y3 n_pos=0. fillna0 leftover **0.711**
  is a fake days leak. Y10 utilisation **impossible**. Q6 **CLOSE**.
  Do not add y10 to `FROZEN_ACCEPTED`. Do not invent a utilisation Y.
- **`e_ap_overdue`** ([leftover QA](553c6ea4-a745-42a3-962a-2299b600d939)):
  **CLOSE unused leftover. KEEP off the 15-col card.** Leftover after
  days rank **0.584** lives but beat-size **FAIL +0.008** (Y3 **0.625**
  vs size **0.617**). Twin of `e_ap_overdue_30` (ρ **0.849**). Y7
  leftover **0.406**. Dark 470 stay NaN.   Do not invent `y_ap_overdue`.
  Do not grow TURNOVER.
- **`a_op_out`** ([leftover QA](572fb928-40f3-4948-9ab9-0fde761a9d50)):
  **CLOSE unused leftover. DROP from the 44 as Y3 X.** Rank leftover
  after days **0.586** lives but OLS **0.675** is a fake days clone.
  **SIZE** ρ **0.741**. Twin of `a_out3` **0.907** / `a_out6` **0.860**.
  Demean leftover **0.525** dies. Leftover after days+CV **0.518** dies.
  Q6 lag1 leftover **0.561** not added to `q6_keep`. **PARK** `y_op_out`.
  Outflow CV leftover **0.734** is a later owner. Do not quote
  a_out_vol 0.722 as the engine.
- **`a_fin_cost`** ([leftover QA](fd8977c1-61b4-42c3-8c84-8f3faa2bb904)):
  **CLOSE unused leftover. DROP from the 44 as Y3 X.** Honest leftover
  after days **0.483** dies. OLS **0.683** is a fake days leak
  (ρ **−0.845**). Exact twin of `f_fin_cost` (ρ **1.000**). Native
  0.634 vs days 0.711 vs size 0.617; beat-size **+0.018 FAIL**. KEEP
  `f_fc_r_lag3` on TURNOVER. Q6 lag1 leftover **0.469** CLOSE. Do not
  put amount or high-fee dummy on the 15-col card. Do not merge M.
- **`a_debt_service`** ([leftover QA](572fb928-40f3-4948-9ab9-0fde761a9d50)):
  **CLOSE unused leftover. DROP from the 44 as Y3 X.** Honest leftover
  after days **0.484** dies. OLS **0.631** is not a fake days clone.
  Beat-size **FAIL** (0.613 vs size 0.617). Exact twin of
  `f_debt_service` (ρ **1.000**) and twin of `f_ds_r` (ρ **0.881**).
  **PARK** `y_debt_service`. Off the 15-col card. Do not grow TURNOVER.
  `f_ds_r` leftover **0.528 DROP** stays. Slice leftovers on T3/Q4
  (0.610 / 0.642) are leftover of the already-DROP ratio.
- **Δdays** ([leftover QA](f6fc63bd-3cca-43b9-9560-d57ce46a4b8f)):
  **DROP as Y3 X.** On ≥18m books leftover after days-level is Δ3
  **0.484** / Δ6 **0.517**. Inverse days-after-Δ3 **0.712** lives —
  the level already ate the path. `q6_keep` stays `days_lag1` **0.684**.
  Off the 15-col card. Not AMPLI / not B. Hidden 72 stays 1-month.
- **top-1 PastDue%** ([leftover QA](689100e7-9d3a-41a4-b281-a0e6a8a87d65)):
  **CLOSE** Y7 leftover after issued_lag1+CN **0.576** (below 0.58) and
  a twin of firm `e_ar_overdue` (ρ **0.858**). Delay footnote **0.581**
  stays. Do not grow TURNOVER **0.720**. Y3 leftover after days
  **0.451**. Q6 short lag1 **0.581** KEEP; issued_lag1 **0.626** stays
  the invoice lead. Dark 470 stay NaN.
- **`d_cust_lost`** ([leftover QA](553c6ea4-a745-42a3-962a-2299b600d939)):
  **CLOSE unused leftover. DROP from the 44 as Y3 X.** Leftover after
  days rank **0.522** dies. Twin of `d_n_cust` (ρ **0.861**). Y3
  **0.581** vs days **0.711**. **PARK** `y_cust_lost`. Dark 470 stay
  NaN. Not Y7 top-1 lost. Y4 footnote KEEP locked. Q6 lag1 leftover
  **0.464** dies.
- **issued-to-top-1** ([leftover QA](689100e7-9d3a-41a4-b281-a0e6a8a87d65)):
  **KEEP** as a Y7 leftover after issued_lag1 (**0.789**). Not a twin.
  Beat-size PASS (raw 0.810 vs size 0.469). Engine is
  **thinning-to-zero** (Y7 57.7% vs 8.1% if they still billed).
  **CLOSE** as a TURNOVER add-on — issued_lag1 **0.626** stays the
  card lead. Q6 lag1 leftover **0.749** is a named-volume footnote,
  not a replace. Off the 15-col card. Dark 470 stay NaN.
- **Cash-flow literature** ([cluster](f6fc63bd-3cca-43b9-9560-d57ce46a4b8f)):
  11 papers, URLs fetched. Last-value `b_runway` **SAME** Farrell 2016
  (p50 1.079m ≈ 27 cash-buffer days). Quiet-stressed recover is **NEW**
  vs default PD. Leftover-after-days is the unpublished referee (Ng v4
  quotes raw 0.806 / 0.850). Lundmark: `a_out_vol` 0.722 stays a trait.
  Siddiqi reasons landed: say no SS **0.635**, no salary **0.603**,
  fewer days **0.711** + lag1. Do not say dropped SHAP names. Δdays
  leftover **DROP** 0.484 / 0.517. Hair 3-month mean is
  **APPLICATION-ONLY** (last-month still ρ **0.946** twin of last-value;
  `run3_prior` 0.873). Last-value KEEP Q1 p50 **1.079**. Y3 leftover of
  `run3` after last-value **0.608** is a B-after-B leak screen, never
  engine X. NSF/overdraft-as-X is a **dataset hole**: no token in the
  dictionary, CAT_MAP, store, or word-bound descriptions; `descubierto`
  (250 txs) is overdraft *fee* text — Y9, not a count. Leftover
  undefined. Do not reconstruct from B. Do not invent `y_nsf`.
- **Invoice literature** ([cluster](689100e7-9d3a-41a4-b281-a0e6a8a87d65)):
  12 papers, URLs fetched. **DSO is not PD; issued volume is** — SAME
  as TURNOVER **0.720** and the DSO DROP. Pérez-Salazar is the only
  CONTRADICT (synthetic supplier-HHI = fragility; our Y5 tail is
  protective 2.7% vs 8.6%). CN ratio is a literature gap. Wave A
  top-1 PastDue leftover **CLOSE 0.576**. Issued-to-top-1 leftover
  **KEEP 0.789** / CLOSE TURNOVER add-on. Y5 net TC × activity
  **CLOSE**: leftover after size+days **0.534** dies; neither cell still
  222/341 (65.1%); shock-month / AP / AR extras die. Y5 stays a 65%
  leftover. Never E as Y5 X.
  Do not grow TURNOVER.
- **Supplier HHI** ([leftover QA](ee13dffe-d5f0-4ddc-a0f3-b73a2992a169)):
  **DROP `d_supp_hhi` from the 44.** Twin of `d_supp_top1` (ρ **0.987**).
  Leftovers die after days (Y3 **0.464**), customer HHI lag3 (Y4
  **0.523**), and size (Y5 **0.525**). Y5 tail **CONFIRM protective**
  2.7% vs 8.6% — not a Y4 crash. Dropping the tail does not raise the
  65% leftover (65.5%). Q6 **CLOSE**. **PARK** as Y5 X (size-rank
  **0.663**). Do not invent `y_supp_hhi`. Do not merge with Y4.
- **B on the 44** ([leftover QA](5e278537-ac87-4fb1-a81c-6a735831dcc5)):
  **DROP all five leftover B columns from the 44 as Y2/Y3 X.**
  `b_runway` stays **KEEP as Q1 last-value description** (p50=1.079) and
  **PARK as a forecast Y**. Honest leftover after days dies: vol /
  below_0 / episodes are fake-days residuals; runway rank leftover
  **0.537**; `b_d_runway` leftover is the Q1 crash cell (body **0.527**).
  Y2 `b_below_0` **0.896** is the lock, not a KEEP. Store vol vs Javier
  **DRIFT 0.354**.   Do not invent `y_bal_vol`. Do not rewrite
  `liquidity.py`.
- **Brief map** ([six-question compile](60e18b0f-6071-45c8-83f8-87dd442fffa6)):
  morning-quotable `analysis/outputs/brief_map.md`. Later leftover QAs
  win KEEP-vs-DROP fights. Map’s 44-drop list is as-of **04:50**; B-on-44,
  `d_tx`, customer HHI, `f_fc_r`, and DSO landed after that cut.
  Six morning sentences: Q1 last-value `b_runway`; Q2 thin `y1_in_h3`;
  Q3 Y3 **0.762 / 0.752**; Q4 Y7 TURNOVER **0.720**; Q5 CN 0.597 + delay
  0.581 footnotes; Q6 issued_lag1 + days_lag1. No 0–100.
- **`d_tx_cp_share`** ([leftover QA](63209d6e-6533-4e44-8d3b-eb4d2a3476ad)):
  **DROP from the 44.** Twin of miss_cp **CONFIRM** (ρ **−0.947**);
  leftover after miss dies (Y3 **0.509** / Y5 **0.454**). Honest Y3
  leftover after days dies (rank **0.537**). **KEEP** the Y5 **0.611**
  quote; **PARK as X** — fold 3 owns **85.9%** of the hole; drop that
  cluster → **0.535**. Dark 470 is **0 not NaN**. Q6 **CLOSE**. Do not
  invent `y_d_tx`.
- **Customer HHI** ([leftover QA](0940f0fc-ba05-4dc8-b22c-94693b1c9f93)):
  **DROP as engine X on the 44.** Twin of `d_cust_top1` ρ **0.994** /
  lag3 **0.994** (quote 0.991 CONFIRM). Y3 leftover after days **0.419**
  dies. Y4 leftover after top1_lag3 **0.549** dies; after the tail-flag
  **0.464** dies (0.605 is the bin). Body CV **0.445** CONFIRM. Tail
  22.1% vs 11.5% CONFIRM. **KEEP** the Y4 >0.975 footnote. Q6 **CLOSE**
  (21.7%, 42 pos). vs supp HHI ρ **0.158** — different object. Off the
  15-col card. Do not invent `y_cust_hhi`. Do not reopen Y4 trees.
- **`f_fc_r`** ([leftover QA](572fb928-40f3-4948-9ab9-0fde761a9d50)):
  **DROP** contemporaneous `f_fc_r` from the 44 as unused leftover,
  not a twin of `f_ds_r` (ρ **0.205**). Y3 leftover after `f_ds_r`
  **0.464** / after days **0.449** (OLS 0.648 is a fake days leak).
  Single **0.559** vs size 0.617 / days 0.711. **KEEP** `f_fc_r_lag3`
  on TURNOVER (do not grow 0.720). Y7 leftover after issued **0.563**
  — CLOSE add-on. Not binary Y9 (ρ **0.132**). Q6 **CLOSE**. Store
  flow stays. Do not invent `y_fc`. Do not merge M.
- **Javier 14** ([reconcile](851eac72-10f7-41d8-ae5f-5c813f51b85a)):
  **11 SAME / 2 CLOSE / 1 DRIFT.** CAT_MAP identical. Concentration is
  **top1** (ρ 0.995), not HHI. Overdue CLOSE (3m vs all-open). Worst
  drift: `volatility` vs `b_bal_vol` ρ **0.354** — Javier is
  sd6(net)/in6, store is balance vol. No named `a_vol`. The 14 omit
  `c_n_days_with_tx` and `e_ar_issued_lag1` (beyond, not drift). B↔invoice
  median |ρ|=0.038 — do not average. No 0–100.
  [a_vol](6b456387-d6a5-4481-850e-1ef6d96bf101): in-memory Javier vol is
  **SAME** (ρ=1). Y3 **0.626** vs size 0.617 (Δ+0.009 is tercile mix;
  loses inside T1 and T2+T3). Y2 0.439. **CLOSE as X. PARK as Y. Do not
  merge.** `a_out_vol` leak screens: Y3 **0.722** reproduced (Δ+0.104
  vs size; no twin; 12-name Δ+0.002). Demean **0.549** (η² **0.741**)
  — **company trait**, not a month shock. Later-store KEEP **False**.
  **X CLOSE. Y PARK. Do not merge.** Not on the 15-col card. Do not
  quote 0.722 as the engine.
- Frozen holdout: 72 companies / 15 groups. Every accepted binary is
  LOW_POWER there (7–23 events) except **y7_top1_lost** (122 holdout pos).
  Claims use train CV. Holdout is also a **late-arrival** sample
  (=24 books **4.2%** vs train **35.8%**).
- **Clean flags** ([DQ QA](d7235c84-3eb3-4163-8560-87eab5308a96)):
  **KEEP never-drop. PARK as Y. CLOSE as X.** Tx extremes 24/2.56M
  (train 8, mass 2.4%; **holdout 16, mass 33.1%** on 2 cos). Invoice
  extremes 10/896,711 CONFIRM; leftover 2.8% after washes. Quoted
  singles do not move (days 0.711 / issued 0.630 / HHI 0.605).
  `product_known` (1,313 txs / 29 products) is **not** the G 63.7%
  hole. `is_dup` is SIZE. B snapshot is sentinel-protected; the
  **flow walk sees all 24 checking giants** — dropping COMP_0306
  would flip 11/18 Y2 labels. Do not rewrite `liquidity.py`.
- **Tax** ([missed-tax](1094dc70-fcab-4d18-9751-018f9f242b46)):
  **calendar dummy, not Q5.** Mixed Q-peaked (Jan/Apr/Jul/Oct 69.2% vs
  43.4% other). `c_missed_tax` Y3 **0.511** vs size 0.617 / days 0.711
  — PARK as Y and as Y3 X. Modal 91.7% CONFIRMED; P(missed|no tax)=17.3%.
  Dark 470 vs 744 file tax at **48.7% vs 51.8%**. Amounts VAT-like
  (p50 419). `tax_refund` = 2 txs. Q6 lag1 chance.
- **Missed salary** ([salary QA](0fd41cbf-2781-4904-976e-49f1963ca065)):
  **CLOSE as Y3 / Q3–Q5 X. PARK as a health Y.** Train **2.9%**
  (623 / 21,157; modal 97.1% CONFIRMED). Y3 **0.513** vs size 0.617
  (Δ −0.104) / days 0.711. `c_salary_month` already on the 15-col
  card at **0.671** — leftover after days **KEEP 0.603**. The lever
  is presence, not the skip. Calendar is **monthly** (Q-sal 40.5% vs
  other 39.9%), not tax-style Q-peak.
  No |ρ|≥0.80 twin (vs `c_missed_tax` 0.075). ICC 0.736 BETWEEN.
  vs rejected Y6 Jaccard **0.365** (now vs t+1..t+3). Q6 CLOSE
  (lag1 0.512). Dark miss 3.1% vs invoiced 2.7%. Do not invent
  `y_missed_salary`. Do not put it on the card.
- **Trail / Q6** ([trail length](6bf54618-71b5-43c3-b609-0b6c9fd485b5)):
  first-bank `created_at` after 2024-09 **73.6%** (891/1,211) is a
  **connection** clock. Bank trail = first tx after 2024-09 **64.2%**.
  Train months-on-book: <6 0.2%, <12 28.8%, ≥18 58.8%, =24 35.8%
  (median 20). Y bases short vs long are close — none is long-only.
  Y4 `HHI_lag3` missing on **78%** of short labeled rows. Y7
  `issued_lag1` present on 95% of short rows; honest≥6 only **47%**.
  Even among 435 full books, only 16.8% ever have a labeled Y4 row
  with the lag. PARK `created_at` / short-trail as health Ys.
- **Recency** ([last-tx QA](4545d7a6-0170-405e-9653-b3d2a330d5c4)):
  **DROP `c_recency_days` from the 44.** Y3 **0.659** vs size 0.617 vs
  days **0.711**. Leftover after days **0.607** dies. Not a |ρ|≥0.80
  twin (vs days −0.615) and not SIZE (−0.413) — still a quiet twin.
  **`c_last_tx_before_2026_06` PARK** as extract artifact: Javier 61 vs
  as-of 2026-08-31 **62**; **COMP_0981 is the +1**. Flag only on
  2026-06..08 (203 CM); Y3 labels end 2026-02 so the flag is 0.500.
  Recency tail piles at August. Onset ≥30d is chance (0.524). Do not
  revive Y6. Off the 15-col card.
- **companies.csv** ([metadata QA](a14da08b-dfa1-42b1-96d3-476beb79adcd)):
  **no transferable Q1 X.** Hidden test is new groups. erp×dark
  **671 / 37 / 73 / 433** — NULL among dark **92.13%** CONFIRMS
  92.1%; not a perfect 470 flag. Named-ERP dark are **17 of the 110
  mixed + 20 of the 360** (18 of those 20 = GROUP_0138 dynamicsAx).
  Country miss **82.21%** (holdout coverage 80.56%). Currency
  complete; EUR 90.3% train vs **73.6%** holdout (AOA/GHS/SEK/XOF
  all sit in GROUP_0199). Three clocks: created−first tx p50
  **+39.4 days**, same-day **0.7%**. **PARK** `created_at` and
  missing-country as health Ys. **CLOSE** `has_erp` (Y3 0.480; ρ
  0.81 vs book), `has_country` (0.481), `currency=EUR` (0.515) vs
  size 0.617 / days 0.711. ~85% of train groups are uniform on each
  flag. Do not invent a country/erp Y.
- **Quoted Q6** ([Q6 quoted leads](c0ddcae1-304f-4d8c-b0f7-df589ad3dff5)):
  Y3 C/A **lag1 100%** on short labels; **lag3 84.7%** (so-far&lt;4,
  company-relative full4 — not Y4 calendar full6). Y7 issued_lag1
  **95.2%**. F lag3 empty until so-far≥6. Short single CV: days
  **0.696** (vs 0.711), issued_lag1 **0.626** (vs 0.630), HHI
  LOW_POWER (42 pos). **KEEP** issued_lag1, `c_n_days_with_tx_lag1`,
  and `c_ss_month_lag1` leftover after days_lag1 **0.631**.
  **CLOSE** Y3 lag3 on short companies (36.3%), F lag3, Y4 HHI.
  Hidden 72: only 1-month claims.
- Accepted Ys mapped to the brief: Y3 = improving after stress; Y2 = turning
  worse (models failed); Y1 = reconstruct the path; Y9 = fee/interest turning
  (14.1% / 19.1%, size-clean); Y5/Y7/Y8 = why / dip vs fall from another table.

## What worked

LightGBM on **y3_recover_cash_6m** (stressed rows, never family B) is the
clean 45→65 win. Night quote stays the full-X shallow model
(**0.762 ± 0.016**, 50 trees, depth 3). The 400-tree / early-stop bake-off
posted **0.710** — same Y, noisier spec. Do not average them.

**Y3 core-keep ([7f37f0fd](7f37f0fd-92bf-4af6-bde7-a52ebd7fa701)): CLOSE,
not PARK.** Five stems + lags 1/3, never B, drop `a_op_in` SIZE,
`e_dso_proxy` left out:

| spec | n_x | CV AUROC | vs 0.710 / vs single |
|------|-----|----------|----------------------|
| A early-stop (published 0.710 spec) | 15 | 0.7099 ± 0.041 | −0.0001 / loses to `c_n_days_with_tx` 0.711 |
| B keep-list | 126 | 0.6929 ± 0.017 | −0.017 / beats `c_gap_sd` 0.671 |
| **Shallow-A** (50 / depth 3) | **15** | **0.7520 ± 0.037** | +0.042 / beats 0.711 |

Early stopping collapsed A to 1–7 trees, so 0.710 is almost the single
activity feature. The 278-col engine is unused capacity once trees may grow.
Prefer A if a core is named. Do not grow B. Per-group Y3 stays parked.

**y7_top1_lost** (dip vs fall, never D): do **not** quote holdout 0.680.
278-col claim was **0.663**. [Y7 SHAP card](410a183a-e392-4b74-aafe-d4ed1c973c4f)
is **CLOSE**, not PARK.

| spec | n_x | CV AUROC | note |
|------|-----|----------|------|
| B_shallow (SHAP card, drop sign-flip lag) | 6 | **0.712 ± 0.086** | quote if a SHAP card is named |
| **TURNOVER** (no DSO; issued lag1 + issued-lag CV + CN ±lag1 + `f_fc_r_lag3`) | **5** | **0.720 ± 0.034** | fold 4 **0.680** (278-col was 0.556) |
| 400+ES A/B | 6–7 | 0.70–0.71 | trees collapse 1–6 — do not KEEP |

DSO is SHAP #1 and **worse than chance** on the short-DSO fifth (0.410).
Q5-only 0.674 already beats 0.663; Q6-only 0.644 does not. Leak/size PASS.
Y8 both fail. PARK Y8.

Y1 path ([remainder](3604ecd0-87b1-4a80-9ec1-7e5e682dea99)): **KEEP thin
`y1_in_h3`** only (holdout med-norm 0.817 vs hist 0.856; CV gap 0.005;
beat-share 51%). **PARK `y1_in_h1`** (holdout 0.799 vs hist 0.795; CV
and holdout disagree). **PARK both liquidity horizons** — last-value
wins CV OOF on liq_h3 (0.598 vs GBM 0.632); train Spearman(liq_t,
liq_{t+3}) = **0.85**. Residuals none KEEP. Q1 health is last-value
liquidity; trees do not reconstruct the path.
  [Balances B](086b8ed0-62ba-4742-94a8-538dd178ea23): the walk is an
  **identity** (median |resid| 9.1e-12€). Spearman short **0.730** vs
  24m **0.862** — the still is **not** leaking backward. Last-vs-snap
  ρ=0.966 (last month *is* the photograph); first-vs-snap ≈0.61.
  Zombie cash 12.2% of products / **0.36%** of |cash| — PARK as a flag.
  13 nulls = Santander UK extract hole (6 last-tx 2026-07-20). KEEP
  last-value as Q1 description. PARK as forecast Y. Never B as Y2/Y3 X.

## What failed honestly

- Circular 0.86 overnight search (erased).
- Y2 GBM (82→68 direction) ≈ chance once B is forbidden.
  [Y2 why](1bb2643e-c8d5-476f-bcff-03fd6bfb57a6): **different turn from
  Y4** (Jaccard 0.057; Y2 median in-ratio **1.03** vs Y4 0.36). 82% of
  positives are already below 0 at *t*; clean-now leftover 1.5%. Best
  legal single `c_n_days_with_tx` **0.571** vs size/GBM 0.540 — drop 12
  chronic dark names in GROUP_0158/0172 and it falls to 0.549. Fold 0
  (0.344) was those 12 names (81.9% Y2) in two large all-dark groups
  the tree treated as safe. Direction is real; a non-B why is not.
  Trees stay PARK. Do not merge with Y4.
- Y6 is a size/activity proxy.
- Y5 XGB does not beat a single feature; holdout inverted.
  [Y5 why](234af73a-5824-4a35-90a4-6083910fa056): **unexplained leftover**
  (~65% of positives neither cash-stress nor counterpart HHI). Not
  cash-stress (days>0 on 99%+ leftover). Supplier HHI>0.975 is
  **protective** (2.7% vs 8.6%). AP `h_group_size` 0.581 vs 0.599 CLOSE
  (large groups protective). AR `d_tx_cp_share` 0.576 vs night **0.611**
  — KEEP the quote, PARK as X. Q5 “no bank counterparties on a thick
  book” is **one group** (fold 3 25.1% / 191 months), not a law. Q6
  CLOSE on short trails. Trees stay PARK.
- **Y4 trees PARK** ([XGB Y4](f248c993-06de-4dc9-9c49-62aa35d217bc)):
  XGB 0.585 / shallow 0.561 lose to `d_cust_hhi_lag3` **0.605**. 400+ES
  collapsed. Holdout n_pos=16, AUROC 0.467. Label stays: the “double”
  is mostly a future **inflow crash** (pos median in3 ratio 0.36; 80%
  drop inflow >20%), not a repayment spike (ds3 ratio 1.15).
  [Y4 why](79044b5e-4335-431b-9a9b-0ea520d88cc1): crash share **79.6%**,
  spike **34.3%**. HHI 0.605 is the **HHI>0.975 tail** (22% on 172
  months / 43 cos); body ≤0.975 CV **0.445**. Quintiles not monotone
  (8.2 / 15.4 / 10.6 / 11.8 / 22.4%). ρ vs `d_cust_top1_lag3` =
  **0.991**. 2-col z-avg CLOSE. Holdout 16 pos flip to 38% crash /
  88% spike — HHI should invert there.
- Y9 GBM PARK: own-p80 CV 0.554 loses to `a_out6` 0.565; spike +0.013 vs `a_op_in`. Label stays (turning/why), trees collapsed to 1–3.
  [Y9 why](0511f2af-bb1b-4658-b612-6a8c11c23f44): **not an outflow tail**
  (31% high out6; 42% neither) and **not a usable mix shift**. `m_fin`
  0.618 / `m_fee` 0.613 equal any-`a_fin_cost>0` (0.610) — that is the Y
  (ρ 0.875 / 0.830 FAIL). Legal leftover `m_int_share` **0.525** CLOSE.
  `a_out6` is a no-fee shield (6.7% vs 22.9%), dies on later-fin.
  Do **not** merge Family M. Not Q6 (lag1 0.571; acf1 −0.055).
- Y8 “failed because no join” is **PARK** (split). `y8_inv` had cash on
  **98.1%** of labeled rows and still lost to `a_in12`. `y8_cash` has a
  real **32% never-ERP** hole; the ERP slice still failed. Cross-source
  Y that needs both tables that month exists on **53.4%** of train
  company-months (11,297 / 21,157). 470 train companies have no invoices
  — confirmed; 92.1% have NULL `companies.erp`. Interco **CLOSE** (0
  equality joins). Amount-match as a *rate* is **KEEP** (payment-month
  35.7% vs 0.5% random) — not a recovered FK.
- Per-company SARIMAX/ETS/Prophet path: park.
- Company clusters (silhouette 0.234) are not operating types.
- Per-group Y3 mixture CV 0.694 vs global 0.710 — PARK. Hidden test is new groups.
- Y3 keep-list B: larger, worse. Early-stop on tiny X is the same collapse as Y2/Y9.
- **Debt schedule** ([snapshot QA](1882a607-6914-4907-a870-993fbd9579f9)):
  NORTH_STAR was half right. `f_util_snapshot` / OGTG are **last-month
  only** (1.6% CM; 100% of non-nulls are 2026-08). `f_w_rate` / next-pay
  / `f_sched_vs_obs` are a **1.7% growing panel** (368 CM / 38 store
  cos; only 10.3% last-month). Raw schedule still 87 rows / 40 cos.
  Inventory (`f_n_facilities`, `f_new_facility`, `f_has_loc`) is a real
  `created_at` panel (rises 555, drops 0) — connection, not origination.
  Ever-schedule AUROC ≈ 0.50. **PARK** snapshot cols as X and as a Y.
  **CLOSE** next_payment as Q6 (p50 months-to-next **−5.6**; 91.6%
  already past). **KEEP** flow `f_ds_r` in the store. **DROP**
  `f_ds_r` from the 15-col card and the 44 (unused leftover after
  days). Contemporaneous `f_fc_r` **DROP from the 44**;
  `f_fc_r_lag3` **KEEP** on TURNOVER.
- **Factoring / confirming** ([inventory QA](fd90198f-68bb-49bf-908c-7184a40a718f)):
  **drop `f_has_factoring` / `f_has_confirming` / `f_has_loc` /
  `f_new_facility` from the 44.** Ever-n train **18 / 61 / 186**
  (CM 0.7% / 3.3% / 11.7%). Rise-only (drops 0/0/0) — same
  connection inventory as G. Y3 singles **0.505 / 0.485 / 0.546**
  vs size 0.617 / days 0.711. Dark 470 vs invoiced 744 still have
  the products (fact 10 vs 8; conf 28 vs 33; LOC 74 vs 112).
  Factoring 18 are a never-recoverer sliver (0/18 ever Y3+ vs T3
  peers 7.7%) and still CLOSE as X. New-WC month size-w Y3
  **−2.0%** — Q3 footnote **CLOSE**. Confirming AP-side is stacked
  WC (confirming-only 8.8% vs rest 8.3%). **PARK** as health Ys.
  Do not score vs Y9 (F forbidden).
- **Family G** ([banking products](7e5c43f8-97d4-4e92-93df-dc28718aff29)):
  rise-only connection inventory (1,561 rises, **0 drops**). `g_new`
  **PARK** as a health Y (7.9% CM, acf1 −0.08). Best access flag
  `g_has_card` oriented CV **0.551** loses to size 0.617 and days 0.711
  — **CLOSE as Y3 X**. Drop `g_created_*` CONFIRM (all-zero on panel).
  `g_has_*` stay in the 44; **`g_has_checking` is 99.1% the connection
  hole**. Dark 470 do not have fewer accounts (p50=3 both). Access ≠ ERP.
- **Y10 utilisation is impossible** — no outstanding/granted history.
  new-LOC-after-stress PARK (1.65%, size 0.681, subset of rejected Y4).
  fee-on-LOC CLOSE (ρ = 1.0 vs Y9). Two interest-on-LOC labels pass
  gates in-module (14.5% / 0.508, 17.6% / 0.520) — **not merged**.
  Assembler skips `y10_util`.
- **Y11 / 470** ([cash Y among the 470](3d5ad8d8-bcd1-4c34-94fc-c3bb7345c734)):
  **no renamed Y2/Y3.** Restricted `y2_neg_2of3` on the 470 is
  CLOSE/subset (6,081 / 9.14% / size 0.544, ρ=1). Restricted Y3 is
  CLOSE/subset (inherits inverse-size). Three cash cousins stay
  in-module unmerged: onset 3-of-3, own-p20 2-of-3, net-recover 6m
  (pos-cos union 365/470). Fee masks = Y9 copies. Never D/E; net
  cousins also drop A. Mixed 110 recover more than all-dark 360
  (12.05% vs 5.20%; size-tercile T1 residual +12.5pp). Do not run the
  assembler.

## Explainability (signed SHAP — quote this)

Canonical write-up: `analysis/outputs/y3_importances.md` (50-tree bake-off
spec, CV 0.710, signs + permutation). Parent 400-tree plots in
`shap_y3_*.png` agree on the names, not the order — do not mix the two ranks.

All top-10 signs are **−** (higher value → lower P(recover)). Stable
permutation core: `c_ss_month` (ΔAUROC 0.034), `a_n_tx` (0.030 —
days twin, **DROP from the card**), `c_salary_month` / `a_op_in`
(0.020). Transfers, DSO, siblings are SHAP-heavy and perm-light —
not durable levers. `a_op_in` is a size clone (ρ 0.998). No A/C
copy of runway (worst |ρ| 0.20).
[Transfer](1bc809f3-d686-4017-b280-1b6b21811ad1): signed `a_transfer`
Y3 **0.566** vs size 0.617 / days 0.711 / salary 0.671. Not a wash
(p50 |net|/gross = 1). Honest leftover after days **0.579**. ICC
**0.956 TRAIT**. **CLOSE as X. PARK as Y.** `a_invest` 0.505 CLOSE.
Do not merge M/I. Not on the 15-col card.

**Family H** ([sibling Y3 gap](5d5b1b81-0a59-43d9-a1cf-aa65e07ee6ea)):
**PARK as Y3 X.** Mixed 110 vs all-dark 360 recover 12.05% vs 5.20%.
Size T1 residual **+12.5pp** (y11-style) / **+16.7pp** (`a_in3`).
`h_sib_neg_share` is flat (+0.01pp) and CV **0.434** vs 0.711. H columns
mark sister *existence*, not sister health. Sister mean `b_runway` ≥ 1
moves Y3 **+7.0pp** after size inside the 110 — **KEEP as a Q5 footnote
only**, not as X (Y3 never B; hidden test is new groups). Mixed dummy
0.494; `h_share_group_in` 0.638 is NEAR_SIZE.

Core-keep A stems (published 15-col spec still quotes **0.752**):
**`c_ss_month` KEEP** (leftover after days 0.635), **`c_salary_month`
KEEP** (0.603), `c_n_days_with_tx` + lags 1,3. **`a_n_tx` and
`f_ds_r` DROP from the card.** Do not rewrite `gbm_core.py` tonight.
Quiet-stressed recovery survives dropping SIZE and the unused
columns.

**Story:** among already-stressed months, quiet firms (no SS/salary, fewer
txs) are the ones whose runway later prints ≥ 3 for three months. Busy
payroll-like outflows and last-month debt service cut odds. That is
de-escalation / possible mean reversion at the bottom — the same honesty
Javier used when “inflow −40%” failed. It answers brief “why” only with
that caveat; it does **not** yet separate dip from fall.

**Y7 signed SHAP** (`analysis/outputs/shap_y7.md`, 50 trees, never D,
sample 4,000 of 7,464 train labeled / 2,149 pos). Family E holds **56%**
of mean |SHAP|; lead-time / turning holds **52%**. Top names: `e_dso_proxy`
+, `e_ar_issued` − (tied 0.216), `e_ar_issued_lag1` −, `f_fc_r_lag3` +,
credit-note ratio at t and t−1, `e_delay_coll` +. Static `n_banking` /
`group_size` entered the top 10 — not levers. Family B is allowed and
absent from the top 10. Size AUROC vs `log1p(a_in3)` = 0.465. Worst |ρ|
vs `d_cust_top1` is −0.30.

**Y7 story (Q4 label, Q5/Q6 SHAP + card):** stretched AR and credit notes
raise P(top-1 gone) in the *long-DSO majority*. DSO is SHAP #1 and worse
than chance on the short-DSO fifth. Thin last-month issuance and jumpy
issued volume are the t−1 levers. Named SHAP card: B_shallow **0.712 / 6**.
Even card: TURNOVER **0.720 / 5** (drop DSO). Do not resume this owner
for more add-ons (they hurt fold 4). Not the Y3 quiet-stressed story.

**FX** ([invoice share](97d3db33-489e-42e2-beb6-af228ec7027a)): ever-FX
228 train ERP cos (16.8% of ERP months). Size ρ 0.060 — not SIZE. Y3
**0.528** vs size 0.628 / days 0.711. **Drop `e_fx_share` from the 44.**
PARK as a health Y. CLOSE as Y7/Y5 X (forbidden E) and as Y3 X. KEEP
only as a Q5 footnote (mixed import/export 44/56, not the 110). Y7
descriptive 30.5% vs 26.8%. Q6 lag1 CLOSE.

Family I (`i_*`, [I lift on Y3](88f17954-a44f-48ad-84e8-6278a82c6935)):
**CLOSE — do not merge.** Shallow-A reproduced **0.7520 ± 0.037 / n_x=15**.
Planned A (SHAP pairs) 0.7522 / 17; B (all 8 legal `i_*`) 0.7524 / 23.
Best add C3 **0.7615 ± 0.024** — still under the **0.772** KEEP gate
(+0.02). Leak screens passed (max |ρ| vs `b_runway` 0.162 / 0.212).
Brier worse with I. Leave in-memory; do not add I to FAMILIES. Next Y3
idea is a core edit, not an I merge.

**Family M** (`m_*`, [category mix](bac5b3bb-19c9-4bbf-b8e4-de23b1793a72)):
20 monthly category-mix shares from `transactions` + `CAT_MAP`. Prefix `m_`
so it does not clash with I. Train coverage **95.8%** (85.8% for
`m_coll_vs_pay`). SIZE / NZV / CONSTANT: none. Highest size ρ vs
`log1p(a_in3)` is 0.333. Not merged into parquet / `FAMILIES`.

Parked as rewrites: uncat *count* share (≡ `a_uncat_share`),
`a_fin_cost/a_op_in`, count twins of debt/transfer/invest/interest/salary.
`m_fin_share` is in the simplex but ρ = 0.964 vs monthly fin/inflow — do
not stack with `a_fin_cost` / `f_fc_r` / Y9; use `m_fee_share` +
`m_int_share`. Amount-share **acf1 ≈ 0** (same noise as `a_op_in`).

Trailing-3m (`m_*_t3`, 15 cols, summed |amounts|, min_periods=3): acf1
0.57–0.71 is **window overlap** (two-thirds shared mass). Honest acf3 is
**−0.10** (t6 acf6 ≈ −0.3). `m_fin_share_t3` PARK (ρ 0.944 vs `f_fc_r`).
Y Spearman |ρ| ≤ 0.15 vs Y3/Y7. **Mix cannot answer Q6.** Y9 why:
do **not** merge monthly `m_*` either — fee/fin shares are the label.

**Uncat** ([share QA](ec17da1b-a50d-49b4-afdc-9fdb4bfb855c)): **Q5 CLOSE.**
Count mean 0.258 vs amount 0.226 (ρ 0.889). Only leftover token is
`uncategorized`. Y3 **0.542** vs size 0.617 / days 0.711. ICC 0.985 —
bookkeeping style, not a month shock. Dark 31.7% vs invoiced 25.0%;
uncat ≠ no ERP. Y2 amount 0.602 is the same style dummy. PARK as Y and
as X. Do not merge amount-uncat. Do not add `uncategorized` to CAT_MAP.

**Missing-CP** ([tx fill](ff4293ad-0821-4c36-a0a3-783cb1a25541)):
**CLOSE as Y3 X. PARK as a Y.** Not the uncat twin (ρ **0.131**).
Twin of store `d_tx_cp_share` (ρ **−0.947**). Train txs miss CP
**90.1% / 96.4% of |amt|**; mapped txs still miss **89.6%**. Invoice
fill **0.999** vs tx named **0.252**. Leftover after uncat + fill
**dies** (Y3 0.509 vs size 0.617). Dark 0.999 vs invoiced 0.775 —
level, not a 470-only X. ICC 0.967. Q5 is already the Y5
`d_tx_cp_share` sentence (fold 3 owns **85.9%** of hole pos). Do not
invent `y_missing_cp`. Do not merge a D column. Y7 never D. Night
Y5 0.611 unchanged.

**Family J** (`j_*`, [match-rate](d56ee5fe-34de-408d-bf21-ef6283a150b1)):
`j_pay_match` (cov 45.6% / 71.2% ever-ERP, size ρ 0.000, acf1 0.226),
`j_iss_match` (52.8%, ρ vs pay 0.678), `j_has_book` (causal miss flag).
**KEEP as Q5 diagnostic.** Not a Y3 X (pay vs recover **−0.060**). 470
companies stay **NaN, not 0**. t3 CLOSE (acf3 −0.003). Unmatched/counts
PARK. Do not merge as a Y3 column. No parquet rewrite.

## Still unknown (same as the brief)

What the hidden 60–80 test is scoring. Whether any cross-source Y will
predict deterioration the way Y3 predicts recovery (not on the full
panel: only 53.4% of months have both tables). Whether a later 0–100
should be a small monotone scorecard on the 15-col core, not a
278-column tree. Family J is a Q5 flag, not a Y3 X. Family H is a
group-type dummy — sister *runway* is a footnote, not a transferable
feature. Family M does not explain Y9. Y2 is already-negative
persistence, not Y4's crash. Y5 is leftover, not a supplier-tail why.
