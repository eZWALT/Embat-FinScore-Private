# Wave 4 — Family M (category mix)

## Files written

- `analysis/features/catmix.py` — `SOURCE_TABLES=["transactions"]`, `FAMILY="m"`, `build(con, grid) -> company_id, period, m_*`
- `analysis/outputs/catmix_report.md` — train-only coverage / size ρ / acf1 / rewrite checks
- This note

Did **not** rewrite `data/feature_store/monthly.parquet`. Did **not** add M to `FAMILIES`. Did not edit `common.py`, `cashflow.py`, Family I, `product/`, or other owners.

`build` queries `clean.transactions` and `CAT_MAP`. No look-ahead: `date_trunc('month', date) = period` (same cut as Family A). Holdout is scored with the same formulas; no percentiles or bins. Prefix is `m_` so it does not clash with Family I (`i_*`).

Smoke: `python -m analysis.features.catmix` → 22,230 rows, 1,286 companies, 20 `m_*`, no key dups. Holdout 1,073 company-months are on the panel (not in the report numbers). Empty-grid path returns NaN columns.

## Columns (emitted)

Amount shares of `sum(|amount|)` by CAT_MAP group (sum to 1) and under-used raw tokens. Count shares only where they are not a rewrite.

| column | formula | brief |
|--------|---------|-------|
| `m_op_in_share` | \|op_in\| / \|all\| | Q1/Q5 operating-in mix |
| `m_op_out_share` | \|op_out\| / \|all\| | Q1/Q5 operating-out mix |
| `m_fin_share` | \|fin_cost\| / \|all\| | Q3/Q5 fee+interest activity share |
| `m_debt_share` | \|debt_service\| / \|all\| | Q5 repayment mix |
| `m_xfer_share` | \|transfer\| / \|all\| | Q5 \|transfer\| (A keeps signed net) |
| `m_invest_share` | \|invest\| / \|all\| | Q5 deploy+return mix |
| `m_uncat_share` | \|uncat or unmapped\| / \|all\| | Q5 amount-weighted opacity |
| `m_fee_share` | \|fee\| / \|all\| | Q3/Q5 fee slice |
| `m_int_share` | \|interest_charge\| / \|all\| | Q3/Q5 interest vs fee |
| `m_tax_share` | \|tax\| / \|all\| | Q5 tax intensity |
| `m_salary_share` | \|salary\| / \|all\| | Q4/Q5 payroll intensity |
| `m_coll_share` | \|collection\| / \|all\| | Q5 raw collection |
| `m_pay_share` | \|payment\| / \|all\| | Q5 raw payment |
| `m_core_share` | (\|coll\|+\|pay\|) / \|all\| | Q5 named-pair weight |
| `m_coll_vs_pay` | \|coll\| / (\|coll\|+\|pay\|) | Q5 collection vs payment |
| `m_op_in_n_share` | n(op_in) / n_tx | Q5 ticket mix (ρ vs amount 0.65) |
| `m_op_out_n_share` | n(op_out) / n_tx | Q5 ticket mix (ρ vs amount 0.67) |
| `m_fee_n_share` | n(fee) / n_tx | Q3/Q5 many small fees |
| `m_coll_n_share` | n(collection) / n_tx | Q5 ticket vs euro (ρ 0.69) |
| `m_pay_n_share` | n(payment) / n_tx | Q5 ticket vs euro (ρ 0.77) |

## Train coverage / size ρ / acf1

Holdout 72 excluded. **21,157** company-months, **1,214** companies. Size = Spearman vs `log1p(max(a_in3, 0))`. SIZE if `|ρ| > 0.85`. Empty months (no txs): **4.2%** → shares NaN. `m_coll_vs_pay` also NaN when the named pair is absent.

Coverage is **95.8%** company-months / **100%** train companies for every share except `m_coll_vs_pay` (**85.8%** / 99.1%).

| column | cov_cm | size_ρ | acf1 | flags |
|--------|--------|--------|------|-------|
| `m_op_in_share` | 95.8% | 0.333 | 0.003 | — |
| `m_op_out_share` | 95.8% | 0.048 | -0.011 | — |
| `m_fin_share` | 95.8% | 0.135 | -0.059 | — |
| `m_debt_share` | 95.8% | 0.257 | 0.037 | — |
| `m_xfer_share` | 95.8% | 0.253 | 0.016 | — |
| `m_invest_share` | 95.8% | 0.223 | -0.056 | — |
| `m_uncat_share` | 95.8% | 0.032 | 0.050 | — |
| `m_fee_share` | 95.8% | 0.200 | -0.056 | — |
| `m_int_share` | 95.8% | 0.136 | -0.093 | — |
| `m_tax_share` | 95.8% | 0.209 | -0.167 | — |
| `m_salary_share` | 95.8% | 0.218 | -0.045 | — |
| `m_coll_share` | 95.8% | 0.328 | -0.002 | — |
| `m_pay_share` | 95.8% | 0.265 | -0.000 | — |
| `m_core_share` | 95.8% | 0.251 | 0.070 | — |
| `m_coll_vs_pay` | 85.8% | 0.022 | 0.026 | — |
| `m_op_in_n_share` | 95.8% | 0.330 | 0.128 | — |
| `m_op_out_n_share` | 95.8% | 0.062 | 0.218 | — |
| `m_fee_n_share` | 95.8% | 0.196 | 0.082 | — |
| `m_coll_n_share` | 95.8% | 0.313 | 0.125 | — |
| `m_pay_n_share` | 95.8% | 0.225 | 0.217 | — |

**SIZE: none. NZV: none. CONSTANT: none.** Highest size ρ is `m_op_in_share` 0.333 / `m_coll_share` 0.328.

## Parked (not emitted)

| parked | why |
|--------|-----|
| uncat *count* share | **exact** `a_uncat_share` (ρ = 1.000) |
| `a_fin_cost / a_op_in` | **exact** monthly fin-cost / inflow (ρ = 1.000) |
| count twins of debt / transfer / invest / interest / salary | ρ vs own amount share ≥ 0.95 |
| fin_cost *count* share | almost `m_fee_n_share` |

`m_uncat_share` (amount) is **kept**: ρ vs `a_uncat_share` = 0.889, means 0.226 vs 0.258.

`m_fin_share` is **emitted for the CAT_MAP simplex** but is a near-rewrite of monthly `a_fin_cost/inflow` (ρ = 0.964). It is **not** a rewrite of `f_fc_r` (ρ = 0.739). Do not stack with `a_fin_cost` / `f_fc_r` / Y9. Prefer `m_fee_share` + `m_int_share` (interest vs inflow ρ = 0.399).

## What failed

- Amount-share **acf1 ≈ 0**: monthly mix is as noisy as `a_op_in`. Answers Q5 this month, not Q6 lead time.
- `m_coll_vs_pay` coverage 85.8% (no named collection or payment that month).
- `m_salary_share` ranks like `c_salary_month` (ρ = 0.953) — intensity on a presence flag, not a new event.
- `m_invest_share` modal 89% zeros (still under the NZV cut).
- Weekly grid unsupported (`date_trunc` month).
- 24 `is_extreme` rows kept; they can own a company-month's mix.

## Next idea

Done below (trailing-3m). When parent adds M to `FAMILIES`, merge **monthly** `m_*` for Q5; do not merge t3/t6 as Q6.

## Trailing-3m (Q6 follow-up)

Added `m_*_t3` inside `build` via `_apply_trail`: within company, `t-2..t`, `min_periods=3`, shares from **summed |amounts|** (empty months contribute 0). Holdout scored with the same formulas. Monthly table in `catmix_report.md` kept. t6 computed on 4 stems as a diagnostic only — **not emitted**.

Gate: KEEP = `acf1 ≥ 0.25` (or +0.15 vs monthly) **and** `acf3 ≥ 0.25`. CLOSE = acf1 only (overlap). PARK = SIZE / rewrite. `acf3` is the no-shared-month check.

| column | cov_cm | size_ρ | acf1 | acf3 | vs sibling acf1 | decision |
|--------|--------|--------|------|------|-----------------|----------|
| `m_coll_vs_pay_t3` | 82.3% | 0.051 | 0.616 | -0.126 | +0.590 | CLOSE |
| `m_uncat_share_t3` | 86.7% | 0.021 | 0.648 | -0.044 | +0.597 | CLOSE |
| `m_fee_share_t3` | 86.7% | 0.151 | 0.592 | -0.149 | +0.648 | CLOSE |
| `m_int_share_t3` | 86.7% | 0.163 | 0.646 | -0.062 | +0.739 | CLOSE |
| `m_fee_n_share_t3` | 86.7% | 0.152 | 0.708 | -0.052 | +0.626 | CLOSE |
| `m_salary_share_t3` | 86.7% | 0.236 | 0.636 | -0.119 | +0.682 | CLOSE |
| `m_xfer_share_t3` | 86.7% | 0.254 | 0.653 | -0.088 | +0.637 | CLOSE |
| `m_op_in_share_t3` | 86.7% | 0.347 | 0.585 | -0.100 | +0.582 | CLOSE |
| `m_debt_share_t3` | 86.7% | 0.277 | 0.652 | -0.054 | +0.615 | CLOSE |
| `m_core_share_t3` | 86.7% | 0.250 | 0.645 | -0.041 | +0.575 | CLOSE |
| `m_tax_share_t3` | 86.7% | 0.070 | 0.581 | -0.153 | +0.748 | CLOSE |
| `m_op_out_share_t3` | 86.7% | 0.006 | 0.569 | -0.163 | +0.581 | CLOSE |
| `m_coll_share_t3` | 86.7% | 0.336 | 0.592 | -0.104 | +0.594 | CLOSE |
| `m_pay_share_t3` | 86.7% | 0.214 | 0.610 | -0.105 | +0.610 | CLOSE |
| `m_fin_share_t3` | 86.7% | 0.074 | 0.600 | -0.143 | +0.658 | PARK (ρ=0.944 vs `f_fc_r`) |

t6 (not emitted; cov 70.5%): `m_uncat_share_t6`, `m_fee_n_share_t6`, `m_xfer_share_t6`, `m_debt_share_t6` — acf1 0.79–0.83, **acf6 −0.26 to −0.39**, all CLOSE.

Train Y Spearman (labeled rows, no model): all `|ρ| ≤ 0.15` vs `y3_recover_cash_6m` / `y7_top1_lost`. Strongest: `m_salary_share_t3` −0.148 and `m_xfer_share_t3` −0.147 vs Y3. Y9 vs fee/int/fin not scored (same source).

**Q6: no.** Mix is a contemporaneous why (Q5). t3/t6 acf1 is mechanical overlap. Parent: do not merge the trail family as lead time. Monthly `m_*` still stand for Q5. Did not rewrite parquet / FAMILIES.
