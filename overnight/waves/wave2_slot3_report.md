# Wave 2 slot 3 — train-only feature report

## Files written

- `analysis/evaluate/feature_report.py` — battery (plan §4). Holdout excluded from every statistic (variance, Spearman, hierarchical cut, VIF, k-means / scaler).
- `analysis/outputs/feature_report.md` — readable tables + drop/keep.
- Plots (matplotlib available): `analysis/outputs/feature_corr_heatmap.png`, `feature_dendrogram.png`, `company_clusters_pca.png`.
- This note.

No family modules edited. No commit. No holdout fit. No 0–100 formula.

## Panel

Train: **21,157** company-months / **1,214** companies (72 holdout left out). Families imported: **A–H** (102 numeric columns). None skipped.

## Return (parent)

**Top size-proxies** vs `log1p(|a_op_in|)`:

| feature | \|ρ\| | flag |
|---------|------:|------|
| `a_op_in` | 0.996 | SIZE |
| `a_in3` | 0.875 | SIZE |
| `a_in6` | 0.809 | near-size |
| `h_share_group_in` | 0.792 | near-size |
| `a_op_out` | 0.738 | |
| `a_in12` | 0.736 | |
| `c_n_tx` / `a_n_tx` | 0.706 | same series |

Use **`log1p(a_in3)`** as the single size control. Raw `a_op_in` has median company Pearson acf1 ≈ 0 (monthly extremes); `a_in3` acf1 = 0.66.

**Near-zero-var / constants**

- Constants: `g_created_unknown_share`, `g_created_after_snapshot`, `g_created_after_snapshot_share` (all 100% one value on the monthly panel).
- All-null: `d_interco_share` (0% coverage).
- NZV continuous: `a_pending_share` (modal 97.3%).
- Rare flags with high modal share were **kept** (`c_missed_*`, `c_last_tx_before_2026_06`, `b_below_0`, `f_has_*`, `f_new_facility`, `f_outstanding_gt_granted`, `g_has_saving` / `tpv` / `investment`).

**Company-cluster silhouette**

- k-means on standardised company medians of 12 non-size features, seed 20260918, k = 4…8.
- Best **k = 4**, silhouette **0.234** (k=5…8 all ≤ 0.20).
- Cramér's V vs log-inflow tertile **0.396**; NMI vs `group_id` **0.112**.
- C0 = 156 companies, **97.4% T1_small** (size pocket). C3 = 7 (too small). Group purity among 138 groups with ≥ 3 train members: 25.4%.
- **Do not run per-cluster models** on this cut. Not just `group_id` either.

**Recommended drop list**

`a_op_in`, `a_in12`, `a_out3`, `a_out6`, `a_out12`, `a_net`, `a_net_margin`, `a_n_tx`, `a_debt_service`, `a_pending_share`, `b_min_liq_3`, `b_mean_liq_3`, `b_below_half_runway`, `b_neg_liq_3`, `c_n_tx`, `c_n_days_with_tx`, `d_cust_top1`, `d_cust_lost`, `d_supp_top1`, `d_interco_share`, `e_ap_overdue`, `f_debt_service`, `f_fin_cost`, `f_n_facilities`, `g_created_after_snapshot`, `g_created_after_snapshot_share`, `g_created_unknown_share`, `g_n_banks`, `h_group_size`, `h_sib_in`.

Park (not core GBM): raw euro levels (`a_op_out`, `b_liq`, `e_*_open` / `issued`, `h_sib_*` amounts), near-size `a_in6` / `h_share_group_in`, snapshot-only F (`f_util_snapshot`, `f_w_rate`, `f_months_to_next_pay`, `f_sched_vs_obs` ~1.7% coverage), plus `d_cust_new` (VIF 19 with `d_n_cust`) and `g_n_types` (VIF 23).

Keep starter: `log1p(a_in3)`, `a_io_ratio`, `a_growth_3`, `a_uncat_share`, `b_runway`, `b_d_runway`, `b_bal_vol`, `b_below_0`, `c_gap_sd`, `c_zero_in_share_6`, `c_missed_*`, `d_cust_hhi`, `d_supp_hhi`, `e_*_overdue_30`, `e_delay_*`, `f_ds_r`, `f_fc_r`, `h_sib_neg_share`. Full tables in `analysis/outputs/feature_report.md`.

## Other findings

- Almost every level has ICC ≥ 0.85 (BETWEEN): company identity dominates month-to-month. Trailing windows persist; raw month amounts and change features do not.
- 93 features in the Spearman matrix → 68 `|ρ|` clusters (cut 0.80); 17 multi-member. Ratios win over raw amounts as representatives (`f_ds_r` over `a_debt_service`, `a_io_ratio` over `a_net_margin`).
- Family F snapshot book is last-month only by construction. Family G `created_after_snapshot` is identically 0 on this monthly grid.

## What failed

- Nothing in the contract smoke: train-only, no holdout centroids / percentiles, families A–H imported without clash.
- HDBSCAN not installed — k-means only (as requested).
- Pearson ACF on raw `a_op_in` is not the same story as Javier's 6-month level r (use `a_in3`).
- Heatmap is a 36-rep subset (p = 93 is too wide); dendrogram has all 93.

## Next idea

- Winsorise `e_dso_proxy` / `e_dpo_proxy` at 24 months in the model layer (means are junk; p50 is ~1.7 months).
- Rebuild `d_interco_share` or drop the column from Family D.
- GBM bake-off on the keep starter vs a "kitchen sink" of all reps, group-fold CV, size-only baseline AUROC.
