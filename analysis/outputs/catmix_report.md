# Category mix report (train only)

Holdout `analysis/splits/holdout_companies.csv` (72 companies) is **excluded** from every number below. No percentiles or bins were fit.

- Panel: **21157** company-months, **1214** train companies (monthly grid 2024-09 … 2026-08).
- Family: **M** (`m_*`). Source: `clean.transactions` + `CAT_MAP`. `data/feature_store/monthly.parquet` was **not** rewritten and is read only for `a_in3` / rewrite checks.
- Mix is *why the money moved* (brief Q5 / Q3). `m_*_t3` is the Q6 candidate (summed |amounts| over t-2..t). Not a 0–100 score.
- No look-ahead: `date_trunc('month', date) = period`. September 2026-09-01 bookings are off the August row.
- Empty company-months (no txs → all shares NaN): 4.2%.
- Size proxy: Spearman vs `log1p(max(a_in3, 0))`. Flag `|ρ| > 0.85` as SIZE.
- Constant: modal-value share `≥ 0.999` or a single value. NZV: modal share `≥ 0.95`.
- Persistence: median company-wise Pearson acf at lag 1 (≥ 4 finite pairs, non-zero s.d.).
- Group amount shares sum to 1: max |sum−1| = 2.13e-14 on months with a defined mix.
- `m_fee_share + m_int_share = m_fin_share`: max abs residual 2.50e-16.

## Brief map

These are *shares* that say what the cash did. They are not a 0–100 index.

| feature | formula | six-question map |
| --- | --- | --- |
| `m_op_in_share` | `sum(|amt| | CAT_MAP=op_in) / sum(|amt|)` | Q1/Q5 why — operating-inflow mix (collection, settlements, refunds) |
| `m_op_out_share` | `sum(|amt| | CAT_MAP=op_out) / sum(|amt|)` | Q1/Q5 why — operating-outflow mix (payment, utility, payroll, tax, cash) |
| `m_fin_share` | `sum(|amt| | CAT_MAP=fin_cost) / sum(|amt|)` | Q3/Q5 turning — fee+interest as a share of activity (not fin/inflow) |
| `m_debt_share` | `sum(|amt| | CAT_MAP=debt_service) / sum(|amt|)` | Q5 why — debt_repayment volume in the month's cash |
| `m_xfer_share` | `sum(|amt| | category=transfer) / sum(|amt|)` | Q5 why — |transfer| mix; A only keeps signed net a_transfer |
| `m_invest_share` | `sum(|amt| | CAT_MAP=invest) / sum(|amt|)` | Q5 why — deploy+return volume share |
| `m_uncat_share` | `sum(|amt| | uncategorized or not in CAT_MAP) / sum(|amt|)` | Q5 why — amount-weighted opacity (A's a_uncat_share is count) |
| `m_fee_share` | `sum(|amt| | category=fee) / sum(|amt|)` | Q3/Q5 FinRegLab — fee slice of fin_cost (score uses the lump) |
| `m_int_share` | `sum(|amt| | category=interest_charge) / sum(|amt|)` | Q3/Q5 turning — interest_charge vs fee (Y9 lumps them) |
| `m_tax_share` | `sum(|amt| | category=tax) / sum(|amt|)` | Q5 why — tax amount intensity (C only has c_tax_month) |
| `m_salary_share` | `sum(|amt| | category=salary) / sum(|amt|)` | Q4/Q5 why — payroll amount intensity (C only has presence) |
| `m_coll_share` | `sum(|amt| | category=collection) / sum(|amt|)` | Q5 why — raw collection (not bulk / settlements) |
| `m_pay_share` | `sum(|amt| | category=payment) / sum(|amt|)` | Q5 why — raw payment (not bulk / utility / payroll) |
| `m_core_share` | `(|collection| + |payment|) / sum(|amt|)` | Q5 why — how much of the month is the named coll/pay pair |
| `m_coll_vs_pay` | `|collection| / (|collection| + |payment|)` | Q5 why — collection vs payment (1 = all collection, 0 = all payment) |
| `m_op_in_n_share` | `n(CAT_MAP=op_in) / n_tx` | Q5 why — op_in ticket mix (count ≠ amount, ρ≈0.65) |
| `m_op_out_n_share` | `n(CAT_MAP=op_out) / n_tx` | Q5 why — op_out ticket mix (count ≠ amount, ρ≈0.67) |
| `m_fee_n_share` | `n(category=fee) / n_tx` | Q3/Q5 — many small fees; amount share stays tiny |
| `m_coll_n_share` | `n(category=collection) / n_tx` | Q5 why — collection ticket mix vs euro mix (ρ≈0.69) |
| `m_pay_n_share` | `n(category=payment) / n_tx` | Q5 why — payment ticket mix vs euro mix (ρ≈0.77) |

## Battery

| feature | cov_cm | cov_co | size_ρ vs log1p(a_in3) | acf1 | modal% | n_unique | mean | flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `m_op_in_share` | 95.8% | 100.0% | 0.333 | 0.003 | 16.3% | 16687 | 0.293 | — |
| `m_op_out_share` | 95.8% | 100.0% | 0.048 | -0.011 | 8.9% | 17749 | 0.343 | — |
| `m_fin_share` | 95.8% | 100.0% | 0.135 | -0.059 | 41.0% | 11635 | 0.018 | — |
| `m_debt_share` | 95.8% | 100.0% | 0.257 | 0.037 | 75.4% | 4949 | 0.017 | — |
| `m_xfer_share` | 95.8% | 100.0% | 0.253 | 0.016 | 59.3% | 8106 | 0.096 | — |
| `m_invest_share` | 95.8% | 100.0% | 0.223 | -0.056 | 89.0% | 2231 | 0.007 | — |
| `m_uncat_share` | 95.8% | 100.0% | 0.032 | 0.050 | 21.1% | 15369 | 0.226 | — |
| `m_fee_share` | 95.8% | 100.0% | 0.200 | -0.056 | 45.9% | 10853 | 0.009 | — |
| `m_int_share` | 95.8% | 100.0% | 0.136 | -0.093 | 83.4% | 3175 | 0.009 | — |
| `m_tax_share` | 95.8% | 100.0% | 0.209 | -0.167 | 46.0% | 10752 | 0.041 | — |
| `m_salary_share` | 95.8% | 100.0% | 0.218 | -0.045 | 58.1% | 8483 | 0.035 | — |
| `m_coll_share` | 95.8% | 100.0% | 0.328 | -0.002 | 18.6% | 16230 | 0.264 | — |
| `m_pay_share` | 95.8% | 100.0% | 0.265 | -2.28e-04 | 23.1% | 15381 | 0.167 | — |
| `m_core_share` | 95.8% | 100.0% | 0.251 | 0.070 | 10.4% | 17637 | 0.431 | — |
| `m_coll_vs_pay` | 85.8% | 99.1% | 0.022 | 0.026 | 14.1% | 13775 | 0.616 | — |
| `m_op_in_n_share` | 95.8% | 100.0% | 0.330 | 0.128 | 16.3% | 5561 | 0.200 | — |
| `m_op_out_n_share` | 95.8% | 100.0% | 0.062 | 0.218 | 8.9% | 6174 | 0.406 | — |
| `m_fee_n_share` | 95.8% | 100.0% | 0.196 | 0.082 | 45.9% | 3608 | 0.059 | — |
| `m_coll_n_share` | 95.8% | 100.0% | 0.313 | 0.125 | 18.6% | 5260 | 0.170 | — |
| `m_pay_n_share` | 95.8% | 100.0% | 0.225 | 0.217 | 23.1% | 5096 | 0.177 | — |

## SIZE / NZV / CONSTANT

- SIZE (`|ρ| > 0.85` vs `log1p(a_in3)`): none
- NZV (modal ≥ 0.95): none
- CONSTANT: none

Monthly amount shares have **acf1 ≈ 0** (same noise as `a_op_in`). They answer *this month's mix* (Q5), not lead time. Trailing-3m columns below are the Q6 candidates.

## Parked (not emitted by `build`)

A share that is just a rewrite of `a_uncat_share` or `a_fin_cost/inflow` is parked. Count twins with Spearman `≥ 0.95` vs their amount share are parked too.

| parked | formula | why |
| --- | --- | --- |
| `m_uncat_n_share` | `n(uncategorized or not in CAT_MAP) / n_tx` | exact rewrite of a_uncat_share (only unmapped token is uncategorized) |
| `m_fc_over_in` | `a_fin_cost / a_op_in` | exact rewrite of monthly a_fin_cost / inflow; f_fc_r is the 3m version |
| `m_debt_n_share` | `n(debt_repayment) / n_tx` | count twin of m_debt_share (train Spearman ≥ 0.95) |
| `m_xfer_n_share` | `n(transfer) / n_tx` | count twin of m_xfer_share (train Spearman ≥ 0.95) |
| `m_invest_n_share` | `n(CAT_MAP=invest) / n_tx` | count twin of m_invest_share (train Spearman ≥ 0.95) |
| `m_int_n_share` | `n(interest_charge) / n_tx` | count twin of m_int_share (train Spearman ≥ 0.95) |
| `m_salary_n_share` | `n(salary) / n_tx` | count twin of m_salary_share (train Spearman ≥ 0.95) |
| `m_fin_n_share` | `n(CAT_MAP=fin_cost) / n_tx` | almost m_fee_n_share (interest is rare in count) |

## Rewrite / overlap checks (train Spearman)

| a | b | ρ | note |
| --- | --- | --- | --- |
| `m_uncat_n_share` | `a_uncat_share` | 1.000 | park — exact count uncat |
| `m_uncat_share` | `a_uncat_share` | 0.889 | keep — amount vs count |
| `m_fc_over_in` | `fc_inflow` | 1.000 | park — exact a_fin_cost/inflow |
| `m_fin_share` | `fc_inflow` | 0.964 | near monthly fin/inflow; not a rewrite of f_fc_r |
| `m_fin_share` | `f_fc_r` | 0.739 | vs trailing 3m f_fc_r |
| `m_fee_share` | `fc_inflow` | 0.876 | fee amount vs monthly fin/inflow |
| `m_int_share` | `fc_inflow` | 0.399 | interest amount vs monthly fin/inflow |
| `m_fee_share` | `m_fin_share` | 0.879 | fee is most of fin_cost euros |
| `m_int_share` | `m_fin_share` | 0.439 | interest slice of fin_cost |
| `m_op_in_n_share` | `m_op_in_share` | 0.654 | count vs amount |
| `m_op_out_n_share` | `m_op_out_share` | 0.669 | count vs amount |
| `m_fee_n_share` | `m_fee_share` | 0.900 | count vs amount |
| `m_coll_n_share` | `m_coll_share` | 0.691 | count vs amount |
| `m_pay_n_share` | `m_pay_share` | 0.771 | count vs amount |
| `m_debt_n_share` | `m_debt_share` | 0.990 | park if ≥ 0.95 |
| `m_xfer_n_share` | `m_xfer_share` | 0.969 | park if ≥ 0.95 |
| `m_invest_n_share` | `m_invest_share` | 0.998 | park if ≥ 0.95 |
| `m_int_n_share` | `m_int_share` | 0.995 | park if ≥ 0.95 |
| `m_salary_n_share` | `m_salary_share` | 0.952 | park if ≥ 0.95 |
| `m_salary_share` | `c_salary_month` | 0.953 | amount vs C presence flag |
| `m_tax_share` | `c_tax_month` | 0.908 | amount vs C presence flag |
| `m_op_in_share` | `m_coll_share` | 0.928 | group vs raw collection |
| `m_op_out_share` | `m_pay_share` | 0.605 | group vs raw payment |
| `m_coll_share` | `m_pay_share` | 0.138 | raw pair (should stay low) |

## How to read the keep set

- `m_uncat_share` is **not** `a_uncat_share`: amount-weighted opacity (train ρ vs the count share ≈ 0.89; means differ). Keep both stories if a model already has the count version — or swap to amount.
- `m_fin_share` ranks like monthly `a_fin_cost / a_op_in` (ρ high) but is bounded in `[0, 1]` and is only moderately tied to `f_fc_r`. **Do not stack** with `a_fin_cost`, `f_fc_r`, or Y9 raw material. Prefer `m_fee_share` + `m_int_share` when the question is fee vs interest.
- `m_xfer_share` is |transfer| / |all|. `a_transfer` is a signed net that can sit near 0 while two-way transfer volume is large.
- `m_salary_share` / `m_tax_share` add intensity on top of `c_salary_month` / `c_tax_month` (presence flags).
- `m_coll_vs_pay` is the named-pair mix the brief wants: what the operating cash is doing, collection vs payment.
- Count shares kept (`op_in`, `op_out`, `fee`, `collection`, `payment`) are the ones whose amount twin does **not** already tell the same ranking.

Do not treat holdout as confirmation. Do not drop Family A columns. Do not put these in a Y that is built from the same category sums (Y9 forbids fin-cost raw material).

## Trailing-3m mix (Q6)

Shares recomputed from **summed |amounts|** (or counts) over `t-2..t`, `min_periods=3`. Empty grid months contribute 0; they do not vote equally with a €10m month. First two company-months are NaN. Holdout excluded. No train bins.

- KEEP = acf1 gate **and** `acf3 ≥ 0.25` (real Q6). CLOSE = acf1 ≥ 0.25 (or +0.15 vs sibling) but acf3 dead (overlap smoother only). PARK = SIZE / CONSTANT / rewrite.
- PARK if SIZE (`|ρ| > 0.85`), CONSTANT, or rewrite (`|ρ| ≥ 0.95` vs `a_*` / `c_salary_month`; `|ρ| ≥ 0.90` vs `f_fc_r` / monthly fin/inflow). ρ vs the monthly sibling is expected and is not a park reason.
- `acf3` is the honest Q6 check: a 3m window at lag 3 shares **no** months. acf1 on overlapping windows is partly mechanical.
- `m_fin_share_t3` is optional; park if it clones `f_fc_r`.

| feature | sibling | cov_cm | size_ρ | acf1 | acf3 | sib acf1 | Δ acf1 | flags | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `m_coll_vs_pay_t3` | `m_coll_vs_pay` | 82.3% | 0.051 | 0.616 | -0.126 | 0.026 | 0.590 | — | CLOSE |
| `m_uncat_share_t3` | `m_uncat_share` | 86.7% | 0.021 | 0.648 | -0.044 | 0.050 | 0.597 | — | CLOSE |
| `m_fee_share_t3` | `m_fee_share` | 86.7% | 0.151 | 0.592 | -0.149 | -0.056 | 0.648 | — | CLOSE |
| `m_int_share_t3` | `m_int_share` | 86.7% | 0.163 | 0.646 | -0.062 | -0.093 | 0.739 | — | CLOSE |
| `m_fee_n_share_t3` | `m_fee_n_share` | 86.7% | 0.152 | 0.708 | -0.052 | 0.082 | 0.626 | — | CLOSE |
| `m_salary_share_t3` | `m_salary_share` | 86.7% | 0.236 | 0.636 | -0.119 | -0.045 | 0.682 | — | CLOSE |
| `m_xfer_share_t3` | `m_xfer_share` | 86.7% | 0.254 | 0.653 | -0.088 | 0.016 | 0.637 | — | CLOSE |
| `m_op_in_share_t3` | `m_op_in_share` | 86.7% | 0.347 | 0.585 | -0.100 | 0.003 | 0.582 | — | CLOSE |
| `m_debt_share_t3` | `m_debt_share` | 86.7% | 0.277 | 0.652 | -0.054 | 0.037 | 0.615 | — | CLOSE |
| `m_core_share_t3` | `m_core_share` | 86.7% | 0.250 | 0.645 | -0.041 | 0.070 | 0.575 | — | CLOSE |
| `m_tax_share_t3` | `m_tax_share` | 86.7% | 0.070 | 0.581 | -0.153 | -0.167 | 0.748 | — | CLOSE |
| `m_op_out_share_t3` | `m_op_out_share` | 86.7% | 0.006 | 0.569 | -0.163 | -0.011 | 0.581 | — | CLOSE |
| `m_coll_share_t3` | `m_coll_share` | 86.7% | 0.336 | 0.592 | -0.104 | -0.002 | 0.594 | — | CLOSE |
| `m_pay_share_t3` | `m_pay_share` | 86.7% | 0.214 | 0.610 | -0.105 | -2.28e-04 | 0.610 | — | CLOSE |
| `m_fin_share_t3` | `m_fin_share` | 86.7% | 0.074 | 0.600 | -0.143 | -0.059 | 0.658 | — | PARK |

- KEEP (0): none — none clear acf3
- CLOSE (14): `m_coll_vs_pay_t3`, `m_uncat_share_t3`, `m_fee_share_t3`, `m_int_share_t3`, `m_fee_n_share_t3`, `m_salary_share_t3`, `m_xfer_share_t3`, `m_op_in_share_t3`, `m_debt_share_t3`, `m_core_share_t3`, `m_tax_share_t3`, `m_op_out_share_t3`, `m_coll_share_t3`, `m_pay_share_t3` — acf1 gate only; overlap smoother, not Q6
- PARK (1): `m_fin_share_t3`

### t3 rewrite / overlap (train Spearman)

| a | b | ρ | note |
| --- | --- | --- | --- |
| `m_uncat_share_t3` | `a_uncat_share` | 0.794 | vs monthly count uncat |
| `m_salary_share_t3` | `c_salary_month` | 0.865 | vs C presence flag |
| `m_fin_share_t3` | `f_fc_r` | 0.944 | vs trailing f_fc_r |
| `m_fin_share_t3` | `fc_inflow` | 0.750 | vs monthly a_fin_cost/inflow |
| `m_fee_share_t3` | `f_fc_r` | 0.820 | vs f_fc_r (do not stack / not a Y9 win) |
| `m_int_share_t3` | `f_fc_r` | 0.424 | vs f_fc_r (do not stack / not a Y9 win) |
| `m_fee_n_share_t3` | `f_fc_r` | 0.682 | fee count 3m vs f_fc_r |
| `m_uncat_share_t3` | `m_uncat_share` | 0.890 | t3 vs monthly sibling |
| `m_coll_vs_pay_t3` | `m_coll_vs_pay` | 0.762 | t3 vs monthly sibling |
| `m_fee_share_t3` | `m_fee_share` | 0.790 | t3 vs monthly sibling |
| `m_op_in_share_t3` | `m_op_in_share` | 0.732 | t3 vs monthly sibling |

## Trailing-6m mix (Q6 fallback, not emitted)

No t3 column KEEPs Q6 (acf3 dead). Same summed-|amount| rule, `t-5..t`, `min_periods=6`, on the 4 highest-acf1 t3 stems. `acf6` is the no-overlap check for a 6m window. **Not in `build`.**

| feature | sibling | cov_cm | size_ρ | acf1 | acf6 | t3 acf1 | flags | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `m_uncat_share_t6` | `m_uncat_share` | 70.5% | -0.023 | 0.792 | -0.258 | 0.648 | — | CLOSE |
| `m_fee_n_share_t6` | `m_fee_n_share` | 70.5% | 0.011 | 0.829 | -0.370 | 0.708 | — | CLOSE |
| `m_xfer_share_t6` | `m_xfer_share` | 70.5% | -0.005 | 0.793 | -0.320 | 0.653 | — | CLOSE |
| `m_debt_share_t6` | `m_debt_share` | 70.5% | 0.037 | 0.829 | -0.392 | 0.652 | — | CLOSE |

### t3 (acf1-pass) vs accepted Ys (train labeled rows, no model)

Read `data/feature_store/targets.parquet` (not rewritten). Spearman on train rows where the Y is non-null. **Not a win:** do not read Y9 vs `m_fee_*` / `m_int_*` / `m_fin_*` as evidence — those share the same fee/interest raw material.

| feature | ρ y3_recover_cash_6m | ρ y7_top1_lost | ρ y9_fee_r_ownp80 | n labeled |
| --- | --- | --- | --- | --- |
| `m_coll_vs_pay_t3` | -0.043 | -0.029 | -0.014 | 8939 |
| `m_uncat_share_t3` | 0.018 | 0.116 | -0.013 | 9378 |
| `m_fee_share_t3` | -0.070 | 0.096 | n/a (same-source) | 9378 |
| `m_int_share_t3` | -0.058 | -0.048 | n/a (same-source) | 9378 |
| `m_fee_n_share_t3` | -0.032 | 0.116 | n/a (same-source) | 9378 |
| `m_salary_share_t3` | -0.148 | -0.109 | 0.038 | 9378 |
| `m_xfer_share_t3` | -0.147 | -0.076 | 0.065 | 9378 |
| `m_op_in_share_t3` | -0.063 | -0.060 | -0.030 | 9378 |
| `m_debt_share_t3` | -0.125 | -0.043 | 0.040 | 9378 |
| `m_core_share_t3` | 0.027 | 0.003 | -0.017 | 9378 |
| `m_tax_share_t3` | -0.108 | -0.057 | 0.012 | 9378 |
| `m_op_out_share_t3` | 0.057 | -0.020 | 0.003 | 9378 |
| `m_coll_share_t3` | -0.043 | -0.031 | -0.019 | 9378 |
| `m_pay_share_t3` | 0.025 | 0.029 | 0.017 | 9378 |


## Q6 verdict

t3 **CLOSE** for Q6: acf1 fires by overlap (3m windows at lag 1 share 2/3 of the mass). Median acf3 is -0.102 (lag 3 shares no months). t6 on the top-4 stems also CLOSEs (`acf6` ≈ −0.3). **Park the trail family as lead time.** Mix answers Q5 (why this month / this quarter); t3 is only a less-noisy mix if parent wants a smoother. Do not merge t3/t6 as Q6.

