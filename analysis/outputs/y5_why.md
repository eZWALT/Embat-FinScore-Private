# Y5 why — cash-stress, supplier-tail, or leftover

- **When:** 2026-09-19T02:33
- **Agent:** `234af73a`
- **Python:** `/home/walterjtv/.pyenv/versions/base/bin/python3`
- **Re-run:** `python -m analysis.evaluate.y5_why`
- **Holdout:** 72 companies, seed 20260918. Coverage only. Rates + singles on train.
- **Y:** `y5_ap_od30_ownp80` / `y5_ar_od30_sust` from `targets.parquet` (not rebuilt). `y5_ap_delay_up15` rejected — coverage only. Train AP 4905 / 418 / **8.52%**; AR 3315 / 237 / **7.15%**.
- **X candidates:** `d_supp_hhi` / `d_cust_hhi` / `a_out6` / `log1p(a_in3)` / `c_n_days_with_tx` / `a_io_ratio` / `c_gap_sd` + night singles `h_group_size` / `d_tx_cp_share`. **Never E.**
- **Quote:** train group-fold CV (5 folds). Holdout is LOW_POWER. Trees stay PARK.
- **Brief:** Q5 why / honest 1-month Q6. Not bankruptcy. Not a 0–100.

## Decision

**KEEP** a non-E Q5 sentence on AR only. Trees stay **PARK**. AP is **unexplained_leftover** / CLOSE. AR cash/HHI is leftover; the usable why is the night single `d_tx_cp_share` (KEEP).

Best non-E AP `h_group_size` CV **0.581** (night `h_group_size` train 0.599, this-run CV 0.581) — beats size but quintiles are not monotone (large groups 2.9% are protective).
Best non-E AR `d_tx_cp_share` CV **0.576** (night `d_tx_cp_share` train 0.611) — zero-named-cp head 12.8% plus intensity on cp>0 (CV 0.580). Cut 18: among above-median `d_n_cust`, unnamed bank months are **17.4%** vs **6.3%** named — a tagging hole, not invoice thinness. AP analogue is **false** (10.9% vs 10.1% named among many-supplier months). Cut 19: hole is cash-independent (16.2% vs named-ok 7.3%). Cut 21 hole_lag1 short CV 0.480 — Q6 stays CLOSE. Hi-cust Q1 unnamed **19.5%** vs Q5 3.4% (head-only). Hole is 27% of AR pos across 28 companies (fragile=False). Cut 31: fold 3 owns 86% of hole pos; without it hole 7.8% vs named 7.0% (survives=False). Cut 34: fold 3 zero+hi-cust is 25.1% (48/191); fold 2 zero+hi is 3.3% (2/60). KEEP the night single; CLOSE 17.4% as a leave-one-group law. Tenure (so-far) is CV 0.55/0.56 — not a book-age rewrite.

Thin-F first-run winners (`f_w_rate_lag1` 0.668 / `f_months_to_next_pay_lag3` 0.677) stay PARK — 1.7% debt-schedule panel, not a cash/D quote.

We **cannot** use family E to show “already overdue last month”. Non-E |ρ| vs `e_ap_overdue_30` / `e_ar_overdue_30` / delay stays <0.18, so cash and HHI are not overdue rewrites. AP has no monotone/tail non-E that clears size+0.02. AR does: unnamed / thinly-named bank counterparties. **Q6 CLOSE:** `d_tx_cp_share` dies on so-far<12 (CV 0.445 / lag1 0.431, 75 pos). Only the 1-month clock is even eligible, and it does not transfer to short books.

## Pass 1 — decompose positives

High flags are that company's own expanding p80 (months ≤ t, min 6). Low `a_io_ratio` is own p20. Y2 / Y4 / Y9 are accepted labels from the same parquet. Never E.

| Y | flag | n_hi / n_defined pos | share of pos | coverage of pos |
|---|---|---:|---:|---:|
| `y5_ap_od30_ownp80` | `y2_neg_2of3` | 32 / 418 | 0.077 | 1.000 |
| `y5_ap_od30_ownp80` | `y4_ds_r_double` | 12 / 80 | 0.150 | 0.191 |
| `y5_ap_od30_ownp80` | `y9_fee_r_ownp80` | 59 / 385 | 0.153 | 0.921 |
| `y5_ap_od30_ownp80` | `a_io_ratio_lo` | 92 / 385 | 0.239 | 0.921 |
| `y5_ap_od30_ownp80` | `d_supp_hhi_hi` | 53 / 366 | 0.145 | 0.876 |
| `y5_ap_od30_ownp80` | `c_gap_sd_hi` | 71 / 412 | 0.172 | 0.986 |
| `y5_ar_od30_sust` | `y2_neg_2of3` | 27 / 237 | 0.114 | 1.000 |
| `y5_ar_od30_sust` | `y4_ds_r_double` | 1 / 35 | 0.029 | 0.148 |
| `y5_ar_od30_sust` | `y9_fee_r_ownp80` | 22 / 224 | 0.098 | 0.945 |
| `y5_ar_od30_sust` | `a_io_ratio_lo` | 49 / 224 | 0.219 | 0.945 |
| `y5_ar_od30_sust` | `d_cust_hhi_hi` | 38 / 202 | 0.188 | 0.852 |
| `y5_ar_od30_sust` | `c_gap_sd_hi` | 50 / 233 | 0.215 | 0.983 |

### 2×2 low-`a_io_ratio` × high counterpart HHI among positives

- `y5_ap_od30_ownp80` n=341: cash_only=70 hhi_only=37 both=12 neither=222 (neither **65.1%**). Read: **unexplained_leftover**.
- `y5_ar_od30_sust` n=193: cash_only=30 hhi_only=25 both=12 neither=126 (neither **65.3%**). Read: **unexplained_leftover**.

## Pass 2 — quintiles (train labeled cuts)

### `d_supp_hhi` vs `y5_ap_od30_ownp80`

n=4902 bins=5 up=False down=False tail_only=False head_only=False

| Q | interval | n | n_pos | P(Y=1) | median X |
|---:|---|---:|---:|---:|---:|
| 1 | (0.00563, 0.117] | 981 | 95 | 0.097 | 0.07717 |
| 2 | (0.117, 0.218] | 980 | 91 | 0.093 | 0.1631 |
| 3 | (0.218, 0.342] | 982 | 73 | 0.074 | 0.2751 |
| 4 | (0.342, 0.532] | 978 | 82 | 0.084 | 0.4176 |
| 5 | (0.532, 1.0] | 981 | 77 | 0.078 | 0.7144 |

### `d_supp_hhi` vs `y5_ar_od30_sust`

n=3309 bins=5 up=False down=False tail_only=False head_only=False

| Q | interval | n | n_pos | P(Y=1) | median X |
|---:|---|---:|---:|---:|---:|
| 1 | (0.011, 0.106] | 662 | 37 | 0.056 | 0.0745 |
| 2 | (0.106, 0.191] | 662 | 46 | 0.069 | 0.1455 |
| 3 | (0.191, 0.308] | 661 | 52 | 0.079 | 0.2452 |
| 4 | (0.308, 0.504] | 662 | 50 | 0.076 | 0.3962 |
| 5 | (0.504, 1.0] | 662 | 51 | 0.077 | 0.6656 |

### `d_cust_hhi` vs `y5_ap_od30_ownp80`

n=4460 bins=5 up=False down=False tail_only=False head_only=False

| Q | interval | n | n_pos | P(Y=1) | median X |
|---:|---|---:|---:|---:|---:|
| 1 | (0.0017599999999999998, 0.129] | 892 | 57 | 0.064 | 0.06321 |
| 2 | (0.129, 0.318] | 892 | 90 | 0.101 | 0.2117 |
| 3 | (0.318, 0.564] | 892 | 87 | 0.098 | 0.4201 |
| 4 | (0.564, 0.945] | 892 | 78 | 0.087 | 0.7632 |
| 5 | (0.945, 1.0] | 892 | 69 | 0.077 | 1 |

### `d_cust_hhi` vs `y5_ar_od30_sust`

n=3311 bins=5 up=False down=False tail_only=False head_only=False

| Q | interval | n | n_pos | P(Y=1) | median X |
|---:|---|---:|---:|---:|---:|
| 1 | (0.0017599999999999998, 0.108] | 663 | 40 | 0.060 | 0.05491 |
| 2 | (0.108, 0.241] | 662 | 57 | 0.086 | 0.1676 |
| 3 | (0.241, 0.442] | 662 | 52 | 0.079 | 0.3357 |
| 4 | (0.442, 0.784] | 662 | 54 | 0.082 | 0.5813 |
| 5 | (0.784, 1.0] | 662 | 34 | 0.051 | 0.9825 |

### `a_out6` vs `y5_ap_od30_ownp80`

n=4905 bins=5 up=False down=False tail_only=False head_only=False

| Q | interval | n | n_pos | P(Y=1) | median X |
|---:|---|---:|---:|---:|---:|
| 1 | (-28157774.900999997, 97077.858] | 981 | 54 | 0.055 | 1.553e+04 |
| 2 | (97077.858, 452601.654] | 981 | 81 | 0.083 | 2.372e+05 |
| 3 | (452601.654, 1274415.418] | 981 | 81 | 0.083 | 7.97e+05 |
| 4 | (1274415.418, 4254456.872] | 981 | 107 | 0.109 | 2.359e+06 |
| 5 | (4254456.872, 17114407320.57] | 981 | 95 | 0.097 | 1.188e+07 |

### `a_out6` vs `y5_ar_od30_sust`

n=3315 bins=5 up=False down=False tail_only=False head_only=False

| Q | interval | n | n_pos | P(Y=1) | median X |
|---:|---|---:|---:|---:|---:|
| 1 | (-28157774.900999997, 151264.498] | 663 | 36 | 0.054 | 3.043e+04 |
| 2 | (151264.498, 514384.806] | 663 | 52 | 0.078 | 3.264e+05 |
| 3 | (514384.806, 1369367.58] | 663 | 55 | 0.083 | 8.664e+05 |
| 4 | (1369367.58, 4572547.922] | 663 | 53 | 0.080 | 2.62e+06 |
| 5 | (4572547.922, 17114407320.57] | 663 | 41 | 0.062 | 1.305e+07 |

### `log1p_a_in3` vs `y5_ap_od30_ownp80`

n=4905 bins=5 up=False down=False tail_only=False head_only=False

| Q | interval | n | n_pos | P(Y=1) | median X |
|---:|---|---:|---:|---:|---:|
| 1 | (-0.001, 10.696] | 981 | 65 | 0.066 | 9.341 |
| 2 | (10.696, 12.198] | 981 | 83 | 0.085 | 11.52 |
| 3 | (12.198, 13.3] | 981 | 68 | 0.069 | 12.72 |
| 4 | (13.3, 14.46] | 981 | 106 | 0.108 | 13.88 |
| 5 | (14.46, 22.713] | 981 | 96 | 0.098 | 15.31 |

### `log1p_a_in3` vs `y5_ar_od30_sust`

n=3315 bins=5 up=False down=False tail_only=False head_only=False

| Q | interval | n | n_pos | P(Y=1) | median X |
|---:|---|---:|---:|---:|---:|
| 1 | (-0.001, 11.262] | 663 | 48 | 0.072 | 10.04 |
| 2 | (11.262, 12.451] | 663 | 41 | 0.062 | 11.96 |
| 3 | (12.451, 13.474] | 663 | 56 | 0.084 | 12.97 |
| 4 | (13.474, 14.609] | 663 | 44 | 0.066 | 14.08 |
| 5 | (14.609, 22.713] | 663 | 48 | 0.072 | 15.43 |

### `c_n_days_with_tx` vs `y5_ap_od30_ownp80`

n=4905 bins=5 up=True down=False tail_only=False head_only=False

| Q | interval | n | n_pos | P(Y=1) | median X |
|---:|---|---:|---:|---:|---:|
| 1 | (-0.001, 8.0] | 1008 | 64 | 0.063 | 5 |
| 2 | (8.0, 14.0] | 1052 | 88 | 0.084 | 12 |
| 3 | (14.0, 18.0] | 893 | 80 | 0.090 | 17 |
| 4 | (18.0, 22.0] | 1038 | 97 | 0.093 | 21 |
| 5 | (22.0, 31.0] | 914 | 89 | 0.097 | 26 |

### `c_n_days_with_tx` vs `y5_ar_od30_sust`

n=3315 bins=5 up=False down=False tail_only=False head_only=False

| Q | interval | n | n_pos | P(Y=1) | median X |
|---:|---|---:|---:|---:|---:|
| 1 | (-0.001, 10.0] | 667 | 34 | 0.051 | 7 |
| 2 | (10.0, 16.0] | 797 | 60 | 0.075 | 14 |
| 3 | (16.0, 20.0] | 733 | 58 | 0.079 | 19 |
| 4 | (20.0, 23.0] | 599 | 38 | 0.063 | 22 |
| 5 | (23.0, 31.0] | 519 | 47 | 0.091 | 27 |

### `a_io_ratio` vs `y5_ap_od30_ownp80`

n=4905 bins=5 up=False down=False tail_only=False head_only=False

| Q | interval | n | n_pos | P(Y=1) | median X |
|---:|---|---:|---:|---:|---:|
| 1 | (-1696.131, 0.489] | 981 | 80 | 0.082 | 0.1188 |
| 2 | (0.489, 0.901] | 981 | 89 | 0.091 | 0.7351 |
| 3 | (0.901, 1.124] | 981 | 91 | 0.093 | 1.009 |
| 4 | (1.124, 1.871] | 981 | 96 | 0.098 | 1.335 |
| 5 | (1.871, 3.0] | 981 | 62 | 0.063 | 3 |

### `a_io_ratio` vs `y5_ar_od30_sust`

n=3315 bins=5 up=False down=False tail_only=False head_only=False

| Q | interval | n | n_pos | P(Y=1) | median X |
|---:|---|---:|---:|---:|---:|
| 1 | (-1696.131, 0.564] | 663 | 58 | 0.087 | 0.2409 |
| 2 | (0.564, 0.92] | 663 | 47 | 0.071 | 0.7751 |
| 3 | (0.92, 1.126] | 663 | 46 | 0.069 | 1.014 |
| 4 | (1.126, 1.818] | 663 | 41 | 0.062 | 1.334 |
| 5 | (1.818, 3.0] | 663 | 45 | 0.068 | 3 |

### `h_group_size` vs `y5_ap_od30_ownp80`

n=4905 bins=5 up=False down=False tail_only=False head_only=False

| Q | interval | n | n_pos | P(Y=1) | median X |
|---:|---|---:|---:|---:|---:|
| 1 | (0.999, 4.0] | 1072 | 108 | 0.101 | 3 |
| 2 | (4.0, 9.0] | 902 | 105 | 0.116 | 6 |
| 3 | (9.0, 13.0] | 999 | 95 | 0.095 | 12 |
| 4 | (13.0, 18.0] | 1162 | 88 | 0.076 | 16 |
| 5 | (18.0, 22.0] | 770 | 22 | 0.029 | 22 |

### `h_group_size` vs `y5_ar_od30_sust`

n=3315 bins=5 up=False down=False tail_only=False head_only=False

| Q | interval | n | n_pos | P(Y=1) | median X |
|---:|---|---:|---:|---:|---:|
| 1 | (0.999, 4.0] | 713 | 40 | 0.056 | 3 |
| 2 | (4.0, 8.0] | 618 | 37 | 0.060 | 6 |
| 3 | (8.0, 13.0] | 791 | 67 | 0.085 | 12 |
| 4 | (13.0, 17.0] | 577 | 40 | 0.069 | 15 |
| 5 | (17.0, 22.0] | 616 | 53 | 0.086 | 19 |

### `d_tx_cp_share` vs `y5_ap_od30_ownp80`

n=4894 bins=5 up=False down=False tail_only=False head_only=False

| Q | interval | n | n_pos | P(Y=1) | median X |
|---:|---|---:|---:|---:|---:|
| 1 | (-0.001, 0.00587] | 979 | 85 | 0.087 | 0 |
| 2 | (0.00587, 0.147] | 979 | 120 | 0.123 | 0.07463 |
| 3 | (0.147, 0.306] | 978 | 93 | 0.095 | 0.2205 |
| 4 | (0.306, 0.466] | 979 | 51 | 0.052 | 0.3897 |
| 5 | (0.466, 0.963] | 979 | 69 | 0.070 | 0.5973 |

### `d_tx_cp_share` vs `y5_ar_od30_sust`

n=3308 bins=5 up=False down=False tail_only=False head_only=True

| Q | interval | n | n_pos | P(Y=1) | median X |
|---:|---|---:|---:|---:|---:|
| 1 | (-0.001, 0.00652] | 662 | 85 | 0.128 | 0 |
| 2 | (0.00652, 0.164] | 661 | 40 | 0.061 | 0.0796 |
| 3 | (0.164, 0.338] | 663 | 53 | 0.080 | 0.2407 |
| 4 | (0.338, 0.482] | 660 | 30 | 0.045 | 0.4056 |
| 5 | (0.482, 0.955] | 662 | 28 | 0.042 | 0.6073 |

Plot: `y5_why_quintiles.png`.

## Pass 3 — single-feature CV

Night honest singles (quote these, not the tree): AP `h_group_size` train 0.599; AR `d_tx_cp_share` train 0.611. Chance = 0.50. Size = `log1p(a_in3)`. PARK a column as X if its size AUROC ≥ 0.6.

| Y | feature | CV AUROC ± sd | train | sign | coverage | size AUROC | size ρ |
|---|---|---:|---:|---:|---:|---:|---:|
| `y5_ap_od30_ownp80` | `d_supp_hhi` | 0.539 ± 0.044 | 0.526 | -1 | 0.999 | 0.663 | -0.331 |
| `y5_ap_od30_ownp80` | `d_cust_hhi` | 0.439 ± 0.095 | 0.505 | +1 | 0.909 | 0.547 | -0.138 |
| `y5_ap_od30_ownp80` | `a_out6` | 0.580 ± 0.064 | 0.563 | +1 | 1.000 | 0.903 | +0.789 |
| `y5_ap_od30_ownp80` | `log1p_a_in3` | 0.556 ± 0.086 | 0.545 | +1 | 1.000 | 1.000 | +1.000 |
| `y5_ap_od30_ownp80` | `c_n_days_with_tx` | 0.540 ± 0.083 | 0.540 | +1 | 1.000 | 0.763 | +0.511 |
| `y5_ap_od30_ownp80` | `a_io_ratio` | 0.525 ± 0.039 | 0.518 | -1 | 1.000 | 0.569 | +0.195 |
| `y5_ap_od30_ownp80` | `c_gap_sd` | 0.544 ± 0.081 | 0.545 | -1 | 0.991 | 0.765 | -0.507 |
| `y5_ap_od30_ownp80` | `d_tx_cp_share` | 0.532 ± 0.095 | 0.552 | -1 | 0.998 | 0.540 | -0.063 |
| `y5_ap_od30_ownp80` | `h_group_size` | 0.581 ± 0.086 | 0.599 | -1 | 1.000 | 0.517 | +0.011 |
| `y5_ar_od30_sust` | `d_supp_hhi` | 0.545 ± 0.063 | 0.522 | +1 | 0.998 | 0.655 | -0.320 |
| `y5_ar_od30_sust` | `d_cust_hhi` | 0.464 ± 0.065 | 0.521 | -1 | 0.999 | 0.548 | -0.113 |
| `y5_ar_od30_sust` | `a_out6` | 0.465 ± 0.036 | 0.503 | +1 | 1.000 | 0.905 | +0.808 |
| `y5_ar_od30_sust` | `log1p_a_in3` | 0.469 ± 0.024 | 0.504 | +1 | 1.000 | 1.000 | +1.000 |
| `y5_ar_od30_sust` | `c_n_days_with_tx` | 0.533 ± 0.080 | 0.544 | +1 | 1.000 | 0.740 | +0.452 |
| `y5_ar_od30_sust` | `a_io_ratio` | 0.486 ± 0.091 | 0.524 | -1 | 1.000 | 0.556 | +0.136 |
| `y5_ar_od30_sust` | `c_gap_sd` | 0.533 ± 0.109 | 0.552 | -1 | 0.991 | 0.747 | -0.461 |
| `y5_ar_od30_sust` | `d_tx_cp_share` | 0.576 ± 0.079 | 0.611 | -1 | 0.998 | 0.535 | -0.063 |
| `y5_ar_od30_sust` | `h_group_size` | 0.458 ± 0.070 | 0.529 | +1 | 1.000 | 0.506 | +0.080 |

### Column letters

| Y | feature | decision | reason |
|---|---|---|---|
| `y5_ap_od30_ownp80` | `d_supp_hhi` | **PARK** | size AUROC 0.663 ≥ 0.6 |
| `y5_ap_od30_ownp80` | `d_cust_hhi` | **CLOSE** | near chance (CV 0.439) |
| `y5_ap_od30_ownp80` | `a_out6` | **PARK** | size AUROC 0.903 ≥ 0.6 |
| `y5_ap_od30_ownp80` | `log1p_a_in3` | **PARK** | size AUROC 1.000 ≥ 0.6 |
| `y5_ap_od30_ownp80` | `c_n_days_with_tx` | **PARK** | size AUROC 0.763 ≥ 0.6 |
| `y5_ap_od30_ownp80` | `a_io_ratio` | **CLOSE** | no clear Q5 (CV 0.525 vs size 0.556, gap -0.031, shape=False) |
| `y5_ap_od30_ownp80` | `c_gap_sd` | **PARK** | size AUROC 0.765 ≥ 0.6 |
| `y5_ap_od30_ownp80` | `d_tx_cp_share` | **CLOSE** | no clear Q5 (CV 0.532 vs size 0.556, gap -0.025, shape=False) |
| `y5_ap_od30_ownp80` | `h_group_size` | **CLOSE** | beats size by +0.025 but no monotone/tail quintile (CV 0.581) |
| `y5_ar_od30_sust` | `d_supp_hhi` | **PARK** | size AUROC 0.655 ≥ 0.6 |
| `y5_ar_od30_sust` | `d_cust_hhi` | **CLOSE** | near chance (CV 0.464) |
| `y5_ar_od30_sust` | `a_out6` | **PARK** | size AUROC 0.905 ≥ 0.6 |
| `y5_ar_od30_sust` | `log1p_a_in3` | **PARK** | size AUROC 1.000 ≥ 0.6 |
| `y5_ar_od30_sust` | `c_n_days_with_tx` | **PARK** | size AUROC 0.740 ≥ 0.6 |
| `y5_ar_od30_sust` | `a_io_ratio` | **CLOSE** | near chance (CV 0.486) |
| `y5_ar_od30_sust` | `c_gap_sd` | **PARK** | size AUROC 0.747 ≥ 0.6 |
| `y5_ar_od30_sust` | `d_tx_cp_share` | **KEEP** | shape+CV 0.576 beats size 0.469 by +0.108 |
| `y5_ar_od30_sust` | `h_group_size` | **CLOSE** | near chance (CV 0.458) |

## Pass 4 — leak vs E (comparators only)

Fail if Spearman |ρ| ≥ 0.8 vs `e_ap_overdue_30` / `e_ar_overdue_30` / delay. That would be the Y. **Never E as X.** A failed leak is a rewrite, not a why.

| feature | vs | n | Spearman | Pearson | fail |
|---|---|---:|---:|---:|---|
| `d_supp_hhi` | `e_ap_overdue_30` | 5027 | +0.121 | +0.133 | False |
| `d_supp_hhi` | `e_ar_overdue_30` | 4398 | +0.062 | +0.047 | False |
| `d_supp_hhi` | `e_delay_paid` | 4609 | -0.038 | +0.056 | False |
| `d_supp_hhi` | `e_delay_coll` | 3878 | +0.027 | +0.039 | False |
| `d_cust_hhi` | `e_ap_overdue_30` | 4591 | +0.042 | +0.032 | False |
| `d_cust_hhi` | `e_ar_overdue_30` | 4289 | +0.117 | +0.096 | False |
| `d_cust_hhi` | `e_delay_paid` | 4255 | +0.058 | +0.078 | False |
| `d_cust_hhi` | `e_delay_coll` | 3858 | -0.141 | -0.008 | False |
| `a_out6` | `e_ap_overdue_30` | 5031 | -0.119 | -0.008 | False |
| `a_out6` | `e_ar_overdue_30` | 4406 | -0.076 | +0.012 | False |
| `a_out6` | `e_delay_paid` | 4609 | -0.119 | +0.030 | False |
| `a_out6` | `e_delay_coll` | 3878 | -0.175 | +0.036 | False |
| `log1p_a_in3` | `e_ap_overdue_30` | 5031 | -0.110 | -0.136 | False |
| `log1p_a_in3` | `e_ar_overdue_30` | 4406 | -0.097 | -0.101 | False |
| `log1p_a_in3` | `e_delay_paid` | 4609 | -0.061 | -0.038 | False |
| `log1p_a_in3` | `e_delay_coll` | 3878 | -0.113 | -0.064 | False |
| `c_n_days_with_tx` | `e_ap_overdue_30` | 5031 | -0.091 | -0.113 | False |
| `c_n_days_with_tx` | `e_ar_overdue_30` | 4406 | -0.076 | -0.070 | False |
| `c_n_days_with_tx` | `e_delay_paid` | 4609 | -0.054 | -0.137 | False |
| `c_n_days_with_tx` | `e_delay_coll` | 3878 | -0.072 | -0.119 | False |
| `a_io_ratio` | `e_ap_overdue_30` | 5031 | +0.019 | -0.018 | False |
| `a_io_ratio` | `e_ar_overdue_30` | 4406 | +0.005 | -0.017 | False |
| `a_io_ratio` | `e_delay_paid` | 4609 | +0.104 | +0.009 | False |
| `a_io_ratio` | `e_delay_coll` | 3878 | +0.111 | -0.052 | False |
| `c_gap_sd` | `e_ap_overdue_30` | 4988 | +0.055 | +0.145 | False |
| `c_gap_sd` | `e_ar_overdue_30` | 4375 | +0.055 | +0.105 | False |
| `c_gap_sd` | `e_delay_paid` | 4580 | +0.043 | +0.113 | False |
| `c_gap_sd` | `e_delay_coll` | 3852 | +0.060 | +0.097 | False |
| `d_tx_cp_share` | `e_ap_overdue_30` | 5020 | -0.113 | -0.067 | False |
| `d_tx_cp_share` | `e_ar_overdue_30` | 4399 | -0.074 | -0.056 | False |
| `d_tx_cp_share` | `e_delay_paid` | 4602 | -0.048 | -0.006 | False |
| `d_tx_cp_share` | `e_delay_coll` | 3871 | +0.107 | +0.012 | False |
| `h_group_size` | `e_ap_overdue_30` | 5031 | -0.081 | -0.059 | False |
| `h_group_size` | `e_ar_overdue_30` | 4406 | -0.037 | -0.041 | False |
| `h_group_size` | `e_delay_paid` | 4609 | -0.011 | +0.008 | False |
| `h_group_size` | `e_delay_coll` | 3878 | +0.024 | -0.024 | False |

FAIL count: **0**.

## Pass 5 — holdout coverage (LOW_POWER)

Do not quote holdout AUROC as a keep. Expect LOW_POWER (14 / 8 pos in the power table).

| Y | n | n_pos | companies | pos companies | rate |
|---|---:|---:|---:|---:|---:|
| `y5_ap_od30_ownp80` | 195 | 14 | 28 | 8 | 0.072 |
| `y5_ar_od30_sust` | 165 | 8 | 23 | 5 | 0.048 |
| `y5_ap_delay_up15` | 171 | 8 | 27 | 4 | 0.047 |

## Pass 6 — honest 1-month Q6

Only lag-1 is transferable (see `q6_quoted.md`). Do not claim a quarter lead.

| Y | feature | lag | CV AUROC ± sd | train | sign | coverage |
|---|---|---:|---:|---:|---:|---:|
| `y5_ap_od30_ownp80` | `d_supp_hhi` | 0 | 0.539 ± 0.044 | 0.526 | -1 | 0.999 |
| `y5_ap_od30_ownp80` | `d_supp_hhi_lag1` | 1 | 0.549 ± 0.055 | 0.533 | -1 | 0.999 |
| `y5_ap_od30_ownp80` | `a_io_ratio` | 0 | 0.525 ± 0.039 | 0.518 | -1 | 1.000 |
| `y5_ap_od30_ownp80` | `a_io_ratio_lag1` | 1 | 0.528 ± 0.044 | 0.524 | -1 | 1.000 |
| `y5_ap_od30_ownp80` | `h_group_size` | 0 | 0.581 ± 0.086 | 0.599 | -1 | 1.000 |
| `y5_ap_od30_ownp80` | `h_group_size_lag1` | 1 | 0.581 ± 0.086 | 0.599 | -1 | 1.000 |
| `y5_ar_od30_sust` | `d_cust_hhi` | 0 | 0.464 ± 0.065 | 0.521 | -1 | 0.999 |
| `y5_ar_od30_sust` | `d_cust_hhi_lag1` | 1 | 0.460 ± 0.070 | 0.524 | -1 | 0.998 |
| `y5_ar_od30_sust` | `a_io_ratio` | 0 | 0.486 ± 0.091 | 0.524 | -1 | 1.000 |
| `y5_ar_od30_sust` | `a_io_ratio_lag1` | 1 | 0.483 ± 0.099 | 0.524 | -1 | 1.000 |
| `y5_ar_od30_sust` | `d_tx_cp_share` | 0 | 0.576 ± 0.079 | 0.611 | -1 | 0.998 |
| `y5_ar_od30_sust` | `d_tx_cp_share_lag1` | 1 | 0.563 ± 0.075 | 0.602 | -1 | 0.998 |

## Mapping (Q5 / Q6)

Y5 is who is *turning* on invoice overdue>30d (Q5 payment behaviour; Hirshleifer PastDue% / Banque de France >30d). AP train positives are **unexplained_leftover** (low `a_io_ratio` 24% of pos; high `d_supp_hhi` 14%; 2×2 neither 65%). AR train positives are **unexplained_leftover** (low `a_io_ratio` 22%; high `d_cust_hhi` 19%; neither 65%). Best non-E AP `h_group_size` CV 0.581 vs night `h_group_size` 0.599 and chance 0.50. Best non-E AR `d_tx_cp_share` CV 0.576 vs night `d_tx_cp_share` 0.611. Lag-1 `d_supp_hhi` 0.549; lag-1 `a_io_ratio` 0.528; lag-1 `d_tx_cp_share` 0.563 (short so-far<12 0.431 — Q6 CLOSE). AR Q5 sentence (no E): months with a thick invoice book but almost no named bank counterparties run 17.4% sustained AR-od30 vs 6.3% when the bank trail names them. Cannot use E to show last-month overdue persistence (non-E |ρ| vs e_* stays <0.18). Trees stay PARK. Verdict **KEEP**. Not a 0–100.

## Extra cuts

### 1b — 2×2 reminder

AP neither cell 222/341 (65.1%). AR neither cell 126/193 (65.3%). Largest cell is leftover, not cash and not counterpart HHI.

### 7 — HHI monopoly tail (Y4 shape, unused here)

Y4 `d_cust_hhi` is a >0.975 monopoly tail. Supplier `d_supp_hhi` is the unused AP counterpart. If Y5 were the same story, the tail rate would jump.

| Y | feature | n tail / pos | P(Y=1) tail | P(Y=1) rest | tail AUROC |
|---|---|---:|---:|---:|---:|
| `y5_ap_od30_ownp80` | `d_supp_hhi` | 73 / 2 | 0.027 | 0.086 | 0.494 |
| `y5_ar_od30_sust` | `d_cust_hhi` | 363 / 10 | 0.028 | 0.077 | 0.464 |
| `y5_ap_od30_ownp80` | `d_cust_hhi` | 780 / 61 | 0.078 | 0.087 | 0.492 |

### 8 — neither cell

`y5_ap_od30_ownp80` neither n=222 of 341 2×2 positives.

| flag inside neither | n_hi / n | share |
|---|---:|---:|
| `y2_neg_2of3` | 17 / 222 | 0.077 |
| `y4_ds_r_double` | 8 / 44 | 0.182 |
| `y9_fee_r_ownp80` | 26 / 222 | 0.117 |
| `c_gap_sd_hi` | 34 / 221 | 0.154 |
| `a_out6_hi` | 64 / 194 | 0.330 |

`y5_ar_od30_sust` neither n=126 of 193 2×2 positives.

| flag inside neither | n_hi / n | share |
|---|---:|---:|
| `y2_neg_2of3` | 12 / 126 | 0.095 |
| `y4_ds_r_double` | 1 / 15 | 0.067 |
| `y9_fee_r_ownp80` | 8 / 126 | 0.063 |
| `c_gap_sd_hi` | 25 / 124 | 0.202 |
| `a_out6_hi` | 46 / 118 | 0.390 |

### 9 — fold AUCs (night singles)

Wide fold sd means the night train quote is not a stable Q5.

| Y | feature | folds | CV ± sd |
|---|---|---|---:|
| `y5_ap_od30_ownp80` | `h_group_size` | 0.517 0.647 0.488 0.563 0.693 | 0.581 ± 0.086 |
| `y5_ap_od30_ownp80` | `c_n_days_with_tx` | 0.641 0.449 0.473 0.530 0.607 | 0.540 ± 0.083 |
| `y5_ap_od30_ownp80` | `d_supp_hhi` | 0.536 0.575 0.533 0.470 0.579 | 0.539 ± 0.044 |
| `y5_ap_od30_ownp80` | `a_io_ratio` | 0.553 0.477 0.564 0.543 0.489 | 0.525 ± 0.039 |
| `y5_ap_od30_ownp80` | `d_tx_cp_share` | 0.581 0.384 0.489 0.604 0.600 | 0.532 ± 0.095 |
| `y5_ar_od30_sust` | `d_tx_cp_share` | 0.541 0.575 0.485 0.699 0.582 | 0.576 ± 0.079 |
| `y5_ar_od30_sust` | `c_n_days_with_tx` | 0.568 0.438 0.465 0.559 0.634 | 0.533 ± 0.080 |
| `y5_ar_od30_sust` | `d_cust_hhi` | 0.526 0.439 0.487 0.506 0.364 | 0.464 ± 0.065 |
| `y5_ar_od30_sust` | `a_io_ratio` | 0.503 0.621 0.372 0.490 0.443 | 0.486 ± 0.091 |
| `y5_ar_od30_sust` | `h_group_size` | 0.519 0.437 0.532 0.359 0.441 | 0.458 ± 0.070 |

### 10 — residual inside size terciles

| Y | slice | feature | n | n_pos | P(Y=1) | CV AUROC |
|---|---|---|---:|---:|---:|---:|
| `y5_ap_od30_ownp80` | size_T1 | `h_group_size` | 1635 | 118 | 0.072 | 0.585 |
| `y5_ap_od30_ownp80` | size_T1 | `c_n_days_with_tx` | 1635 | 118 | 0.072 | 0.443 |
| `y5_ap_od30_ownp80` | size_T1 | `d_tx_cp_share` | 1635 | 118 | 0.072 | 0.536 |
| `y5_ap_od30_ownp80` | size_T2 | `h_group_size` | 1635 | 132 | 0.081 | 0.568 |
| `y5_ap_od30_ownp80` | size_T2 | `c_n_days_with_tx` | 1635 | 132 | 0.081 | 0.521 |
| `y5_ap_od30_ownp80` | size_T2 | `d_tx_cp_share` | 1635 | 132 | 0.081 | 0.395 |
| `y5_ap_od30_ownp80` | size_T3 | `h_group_size` | 1635 | 168 | 0.103 | 0.584 |
| `y5_ap_od30_ownp80` | size_T3 | `c_n_days_with_tx` | 1635 | 168 | 0.103 | 0.530 |
| `y5_ap_od30_ownp80` | size_T3 | `d_tx_cp_share` | 1635 | 168 | 0.103 | 0.551 |
| `y5_ar_od30_sust` | size_T1 | `d_tx_cp_share` | 1105 | 80 | 0.072 | 0.583 |
| `y5_ar_od30_sust` | size_T1 | `c_n_days_with_tx` | 1105 | 80 | 0.072 | 0.575 |
| `y5_ar_od30_sust` | size_T1 | `h_group_size` | 1105 | 80 | 0.072 | 0.509 |
| `y5_ar_od30_sust` | size_T2 | `d_tx_cp_share` | 1105 | 74 | 0.067 | 0.609 |
| `y5_ar_od30_sust` | size_T2 | `c_n_days_with_tx` | 1105 | 74 | 0.067 | 0.461 |
| `y5_ar_od30_sust` | size_T2 | `h_group_size` | 1105 | 74 | 0.067 | 0.591 |
| `y5_ar_od30_sust` | size_T3 | `d_tx_cp_share` | 1105 | 83 | 0.075 | 0.621 |
| `y5_ar_od30_sust` | size_T3 | `c_n_days_with_tx` | 1105 | 83 | 0.075 | 0.536 |
| `y5_ar_od30_sust` | size_T3 | `h_group_size` | 1105 | 83 | 0.075 | 0.492 |

### 11 — short-book lag1 (honest Q6)

Only 1-month leads transfer (`q6_quoted.md`). Short = months-so-far < 12. `d_tx_cp_share` on short is **0.445 / lag1 0.431** (75 pos) vs long **0.727**. **Q6 CLOSE** — do not transfer the AR KEEP to short books or the hidden 72 as a lead.

| Y | slice | feature | lag | coverage | CV AUROC ± sd | n_pos |
|---|---|---|---:|---:|---:|---:|
| `y5_ap_od30_ownp80` | short_<12 | `d_supp_hhi` | 0 | 1.000 | 0.519 ± 0.054 | 142 |
| `y5_ap_od30_ownp80` | short_<12 | `d_supp_hhi_lag1` | 1 | 1.000 | 0.526 ± 0.059 | 142 |
| `y5_ap_od30_ownp80` | short_<12 | `a_io_ratio` | 0 | 1.000 | 0.492 ± 0.030 | 142 |
| `y5_ap_od30_ownp80` | short_<12 | `a_io_ratio_lag1` | 1 | 1.000 | 0.464 ± 0.036 | 142 |
| `y5_ap_od30_ownp80` | short_<12 | `h_group_size` | 0 | 1.000 | 0.620 ± 0.091 | 142 |
| `y5_ap_od30_ownp80` | short_<12 | `h_group_size_lag1` | 1 | 1.000 | 0.620 ± 0.091 | 142 |
| `y5_ap_od30_ownp80` | long_>=18 | `d_supp_hhi` | 0 | 0.999 | 0.601 ± 0.103 | 114 |
| `y5_ap_od30_ownp80` | long_>=18 | `d_supp_hhi_lag1` | 1 | 1.000 | 0.601 ± 0.101 | 114 |
| `y5_ap_od30_ownp80` | long_>=18 | `a_io_ratio` | 0 | 1.000 | 0.568 ± 0.103 | 114 |
| `y5_ap_od30_ownp80` | long_>=18 | `a_io_ratio_lag1` | 1 | 1.000 | 0.567 ± 0.116 | 114 |
| `y5_ap_od30_ownp80` | long_>=18 | `h_group_size` | 0 | 1.000 | 0.522 ± 0.186 | 114 |
| `y5_ap_od30_ownp80` | long_>=18 | `h_group_size_lag1` | 1 | 1.000 | 0.522 ± 0.186 | 114 |
| `y5_ap_od30_ownp80` | all | `d_supp_hhi` | 0 | 0.999 | 0.539 ± 0.044 | 418 |
| `y5_ap_od30_ownp80` | all | `d_supp_hhi_lag1` | 1 | 0.999 | 0.549 ± 0.055 | 418 |
| `y5_ap_od30_ownp80` | all | `a_io_ratio` | 0 | 1.000 | 0.525 ± 0.039 | 418 |
| `y5_ap_od30_ownp80` | all | `a_io_ratio_lag1` | 1 | 1.000 | 0.528 ± 0.044 | 418 |
| `y5_ap_od30_ownp80` | all | `h_group_size` | 0 | 1.000 | 0.581 ± 0.086 | 418 |
| `y5_ap_od30_ownp80` | all | `h_group_size_lag1` | 1 | 1.000 | 0.581 ± 0.086 | 418 |
| `y5_ar_od30_sust` | short_<12 | `d_cust_hhi` | 0 | 1.000 | 0.427 ± 0.041 | 75 |
| `y5_ar_od30_sust` | short_<12 | `d_cust_hhi_lag1` | 1 | 1.000 | 0.466 ± 0.093 | 75 |
| `y5_ar_od30_sust` | short_<12 | `a_io_ratio` | 0 | 1.000 | 0.553 ± 0.172 | 75 |
| `y5_ar_od30_sust` | short_<12 | `a_io_ratio_lag1` | 1 | 1.000 | 0.440 ± 0.170 | 75 |
| `y5_ar_od30_sust` | short_<12 | `d_tx_cp_share` | 0 | 0.996 | 0.445 ± 0.110 | 75 |
| `y5_ar_od30_sust` | short_<12 | `d_tx_cp_share_lag1` | 1 | 0.997 | 0.431 ± 0.094 | 75 |
| `y5_ar_od30_sust` | long_>=18 | `d_cust_hhi` | 0 | 0.999 | 0.408 ± 0.060 | 67 |
| `y5_ar_od30_sust` | long_>=18 | `d_cust_hhi_lag1` | 1 | 0.999 | 0.442 ± 0.119 | 67 |
| `y5_ar_od30_sust` | long_>=18 | `a_io_ratio` | 0 | 1.000 | 0.589 ± 0.099 | 67 |
| `y5_ar_od30_sust` | long_>=18 | `a_io_ratio_lag1` | 1 | 1.000 | 0.570 ± 0.141 | 67 |
| `y5_ar_od30_sust` | long_>=18 | `d_tx_cp_share` | 0 | 1.000 | 0.727 ± 0.142 | 67 |
| `y5_ar_od30_sust` | long_>=18 | `d_tx_cp_share_lag1` | 1 | 1.000 | 0.711 ± 0.134 | 67 |
| `y5_ar_od30_sust` | all | `d_cust_hhi` | 0 | 0.999 | 0.464 ± 0.065 | 237 |
| `y5_ar_od30_sust` | all | `d_cust_hhi_lag1` | 1 | 0.998 | 0.460 ± 0.070 | 237 |
| `y5_ar_od30_sust` | all | `a_io_ratio` | 0 | 1.000 | 0.486 ± 0.091 | 237 |
| `y5_ar_od30_sust` | all | `a_io_ratio_lag1` | 1 | 1.000 | 0.483 ± 0.099 | 237 |
| `y5_ar_od30_sust` | all | `d_tx_cp_share` | 0 | 0.998 | 0.576 ± 0.079 | 237 |
| `y5_ar_od30_sust` | all | `d_tx_cp_share_lag1` | 1 | 0.998 | 0.563 ± 0.075 | 237 |

### 12 — company-level trait vs turning

Spearman of each train company's mean X vs its Y5 month-share (≥3 labeled months).

| Y | X | companies | ever pos | ρ | med X pos / neg cos |
|---|---|---:|---:|---:|---:|
| `y5_ap_od30_ownp80` | `h_group_size` | 446 | 171 | -0.175 | 11 / 12 |
| `y5_ar_od30_sust` | `d_tx_cp_share` | 331 | 115 | -0.194 | 0.173 / 0.267 |

### 13 — AP ∩ AR

Train both-labeled 3161: both-pos 34, AP-only 243, AR-only 189, Spearman +0.063. Two labels, not a rewrite.

### 14 — `d_tx_cp_share` honesty

Any-named-cp (comparator) AR CV **0.463**. Continuous all-row **0.576** (gap vs binary +0.114). On cp>0 support CV **0.580**. presence_rewrite=False; intensity_dead=False.

| Y | slice | n | n_pos | P(Y=1) |
|---|---|---:|---:|---:|
| `y5_ap_od30_ownp80` | cp_eq_0 | 907 | 79 | 0.087 |
| `y5_ap_od30_ownp80` | cp_gt_0 | 3987 | 339 | 0.085 |
| `y5_ar_od30_sust` | cp_eq_0 | 585 | 72 | 0.123 |
| `y5_ar_od30_sust` | cp_gt_0 | 2723 | 164 | 0.060 |

| Y | feature | slice | CV AUROC ± sd | n |
|---|---|---|---:|---:|
| `y5_ap_od30_ownp80` | `_any_cp` | all_labeled | 0.468 ± 0.037 | 4894 |
| `y5_ap_od30_ownp80` | `d_tx_cp_share` | all_labeled | 0.532 ± 0.095 | 4894 |
| `y5_ap_od30_ownp80` | `d_tx_cp_share` | cp_gt_0 | 0.573 ± 0.084 | 3987 |
| `y5_ar_od30_sust` | `_any_cp` | all_labeled | 0.463 ± 0.074 | 3308 |
| `y5_ar_od30_sust` | `d_tx_cp_share` | all_labeled | 0.576 ± 0.079 | 3308 |
| `y5_ar_od30_sust` | `d_tx_cp_share` | cp_gt_0 | 0.580 ± 0.038 | 2723 |

Honesty: share intensity still ranks after zeros are removed.

### 15 — large-group protective tail

| Y | n ≥18 / pos | P(Y=1) ≥18 | P(Y=1) rest | small-group flag AUROC | companies ≥18 |
|---|---:|---:|---:|---:|---:|
| `y5_ap_od30_ownp80` | 1236 / 36 | 0.029 | 0.104 | 0.591 | 118 |
| `y5_ar_od30_sust` | 616 / 53 | 0.086 | 0.068 | 0.480 | 75 |

AP Q5 (group 18–22) is 2.9%. Protective, not a why for the positives. Not monotone. CLOSE as Q5 X.

### 16 — night singles on the neither leftover

| Y | feature | n | n_pos | CV AUROC ± sd |
|---|---|---:|---:|---:|
| `y5_ap_od30_ownp80` | `h_group_size` | 2379 | 222 | 0.544 ± 0.107 |
| `y5_ap_od30_ownp80` | `d_tx_cp_share` | 2379 | 222 | 0.543 ± 0.053 |
| `y5_ap_od30_ownp80` | `c_n_days_with_tx` | 2379 | 222 | 0.537 ± 0.086 |
| `y5_ar_od30_sust` | `h_group_size` | 1636 | 126 | 0.564 ± 0.065 |
| `y5_ar_od30_sust` | `d_tx_cp_share` | 1636 | 126 | 0.581 ± 0.088 |
| `y5_ar_od30_sust` | `c_n_days_with_tx` | 1636 | 126 | 0.554 ± 0.142 |

### 17 — leftover still has cash

Join QA already: Y5 labels are 100% ever-ERP and ~99% have a tx that month. This cut is the neither cell. CLOSE “died because cash was missing”.

| Y | slice | n | share days>0 | share a_in3 defined | median days |
|---|---|---:|---:|---:|---:|
| `y5_ap_od30_ownp80` | all_pos | 418 | 0.990 | 1.000 | 17.0 |
| `y5_ap_od30_ownp80` | neither_pos | 222 | 0.995 | 1.000 | 18.0 |
| `y5_ar_od30_sust` | all_pos | 237 | 0.996 | 1.000 | 18.0 |
| `y5_ar_od30_sust` | neither_pos | 126 | 1.000 | 1.000 | 19.0 |

### 18 — tagging hole vs invoice thinness

Train AR complete; median `d_n_cust`=18. Low share = `d_tx_cp_share` ≤ 0.01. tagging_hole=True: low-share+many-invoice-cust P(Y=1)=**17.4%** vs named+many-cust 6.3% vs low-share+few-cust 6.6%.

| slice | n | n_pos | P(Y=1) |
|---|---:|---:|---:|
| low_share_hi_cust | 368 | 64 | 0.174 |
| low_share_lo_cust | 333 | 22 | 0.066 |
| hi_share_hi_cust | 1246 | 79 | 0.063 |
| hi_share_lo_cust | 1361 | 71 | 0.052 |

AP analogue (median `d_n_supp`=30): tagging_hole_ap=False low-share+many-supp 10.9% vs named 10.1%.

| slice | n | n_pos | P(Y=1) |
|---|---:|---:|---:|
| low_share_hi_supp | 384 | 42 | 0.109 |
| low_share_lo_supp | 662 | 50 | 0.076 |
| hi_share_hi_supp | 2023 | 204 | 0.101 |
| hi_share_lo_supp | 1825 | 122 | 0.067 |

### 19 — tagging hole × own-low cash

Among above-median `d_n_cust` months with a cash flag. cash_independent=True: hole+ok-cash **16.2%** vs named+ok-cash 7.3%. If the hole only fired when cash was low, it would be a cash rewrite.

| slice | n | n_pos | P(Y=1) |
|---|---:|---:|---:|
| hole_cash | 72 | 18 | 0.250 |
| hole_ok | 260 | 42 | 0.162 |
| named_cash | 313 | 12 | 0.038 |
| named_ok | 880 | 64 | 0.073 |

### 20 — hole as company trait + holdout n

Hole-flag group-fold CV on hi-cust train: **0.569** ± 0.096 (n=1614 pos=143). Companies with a hi-cust AR month: 191; ever-hole 59 (mean company-rate 0.144 vs never 0.090). Holdout hole coverage only: n=54 pos=7 (LOW_POWER — not an AUROC keep).

### 21 — hole_lag1 honest 1-month Q6

Lag-1 of the tagging-hole flag. Short so-far<12 CV **0.480**; long ≥18 0.560; all-train 0.542. CLOSE a quarter lead. Only the 1-month clock is eligible.

| slice | col | n | n_pos | CV AUROC ± sd | coverage |
|---|---|---:|---:|---:|---:|
| short_<12 | `_hole` | 1157 | 75 | 0.480 ± 0.072 | 0.996 |
| short_<12 | `_hole_lag1` | 1157 | 75 | 0.480 ± 0.070 | 0.997 |
| long_>=18 | `_hole` | 718 | 67 | 0.574 ± 0.139 | 1.000 |
| long_>=18 | `_hole_lag1` | 718 | 67 | 0.560 ± 0.106 | 1.000 |
| all | `_hole` | 3315 | 237 | 0.546 ± 0.081 | 0.998 |
| all | `_hole_lag1` | 3315 | 237 | 0.542 ± 0.073 | 0.998 |

### 22 — hole-positive concentration

Hole months hold **27.0%** of AR positives (64/237) across 28 companies. top1=0.078 top3=0.234 HHI=0.046. fragile=False (PARK the sentence if top3≥0.50).

### 23 — intensity on thick vs thin invoice books

`d_tx_cp_share` CV on hi-cust **0.600**; lo-cust 0.466. KEEP lives on the thick book, not the thin one.

| slice | n | n_pos | CV AUROC ± sd |
|---|---:|---:|---:|
| hi_cust | 1614 | 143 | 0.600 ± 0.105 |
| lo_cust | 1701 | 94 | 0.466 ± 0.114 |

### 24 — turning-month Δ `d_tx_cp_share`

First difference (now − lag1). All-train CV 0.558. A drop in named-cp share is not a stronger why than the level.

| slice | n | n_pos | CV AUROC ± sd | sign | coverage |
|---|---:|---:|---:|---:|---:|
| all | 3315 | 237 | 0.558 ± 0.099 | -1 | 0.998 |
| short_<12 | 1157 | 75 | 0.442 ± 0.085 | -1 | 0.996 |

### 25 — short-book contemporaneous hole rate

Q5 can be same-month; Q6 cannot. Short hi-cust hole **11.5%** vs named 5.5%. Long hole 30.9% vs named 7.3%.

| slice | n | n_pos | P(Y=1) |
|---|---:|---:|---:|
| short_<12_hi_hole | 165 | 19 | 0.115 |
| short_<12_hi_named | 380 | 21 | 0.055 |
| long_>=18_hi_hole | 68 | 21 | 0.309 |
| long_>=18_hi_named | 273 | 20 | 0.073 |

### 26 — AP leftover extra D singles

Unused supplier columns. SIZE_PARK if the column ranks large vs small firms.

| feature | CV AUROC ± sd | sign | size AUROC | SIZE_PARK | vs size CV |
|---|---:|---:|---:|---|---:|
| `d_n_supp` | 0.584 ± 0.064 | 1 | 0.799 | True | +0.028 |
| `d_supp_top1` | 0.530 ± 0.043 | -1 | 0.642 | True | -0.027 |
| `d_supp_hhi` | 0.539 ± 0.044 | -1 | 0.663 | True | -0.018 |

### 27 — `d_tx_cp_share` vs size (not a rewrite)

| vs | Spearman ρ |
|---|---:|
| `log1p_a_in3` | -0.063 |
| `h_group_size` | -0.081 |
| `d_n_cust` | +0.014 |
| `c_n_days_with_tx` | -0.050 |

### 28 — hole flag vs E / Y2 / Y9

max |ρ| vs E = 0.105. leak_fail=False (fail ≥0.8). Cannot use E to show persistence; this is the non-E hole vs those E columns.

| kind | col | ρ | fail |
|---|---|---:|---|
| E | `e_ap_overdue_30` | +0.021 | False |
| E | `e_ar_overdue_30` | -0.014 | False |
| E | `e_delay_paid` | +0.105 | False |
| E | `e_delay_coll` | +0.038 | False |
| Y | `y2_neg_2of3` | +0.039 | False |
| Y | `y9_fee_r_ownp80` | -0.038 | False |
| Y | `y5_ap_od30_ownp80` | +0.029 | False |

### 29 — `d_tx_cp_share` quintiles on hi-cust AR

n=1614 head_only=True monotone_down=False tail_only=False.

| Q | interval | n | n_pos | P(Y=1) | median X |
|---:|---|---:|---:|---:|---:|
| 1 | (-0.001, 0.00599] | 323 | 63 | 0.195 | 0 |
| 2 | (0.00599, 0.15] | 323 | 24 | 0.074 | 0.05919 |
| 3 | (0.15, 0.35] | 322 | 27 | 0.084 | 0.2307 |
| 4 | (0.35, 0.504] | 323 | 18 | 0.056 | 0.4189 |
| 5 | (0.504, 0.933] | 323 | 11 | 0.034 | 0.6423 |

### 30 — hole rate by group fold

Hole-rate range 0.041–0.333. If one fold owns the 17.4%, PARK the sentence.

| fold | n hole | pos | P(Y=1) hole | P(Y=1) named |
|---:|---:|---:|---:|---:|
| 0 | 23 | 2 | 0.087 | 0.051 |
| 1 | 6 | 2 | 0.333 | 0.032 |
| 2 | 74 | 3 | 0.041 | 0.093 |
| 3 | 253 | 55 | 0.217 | 0.044 |
| 4 | 12 | 2 | 0.167 | 0.074 |

### 31 — drop the fold that owns hole positives

Fold 3 holds **86%** of hole positives. Without it: hole 7.8% vs named 7.0%; `d_tx_cp_share` CV 0.546. survives=False. Pooled 17.4% is not a leave-one-group fact.

### 32 — who is the heavy fold

111 train AR companies. median `h_group_size`=13, median `d_tx_cp_share`=0.0658, median `d_n_cust`=19, share of months that are the hole cell=23.9%. Not a sibling_h edit.

### 33 — median `d_tx_cp_share` by fold

If one fold is the unnamed cluster, dropping it removes the treatment group — not a proof the pooled Q5 is fake.

| fold | n | companies | median share | share=0 | P(Y=1) |
|---:|---:|---:|---:|---:|---:|
| 0 | 589 | 81 | 0.274 | 0.073 | 0.066 |
| 1 | 326 | 47 | 0.470 | 0.104 | 0.064 |
| 2 | 680 | 73 | 0.198 | 0.196 | 0.071 |
| 3 | 1059 | 111 | 0.066 | 0.332 | 0.091 |
| 4 | 654 | 81 | 0.361 | 0.035 | 0.049 |

### 34 — share=0 × invoice thickness by fold

Fold 2 zeros should be thin-book if the hole is fold-3-only.

| fold | book | n zero | pos | P(Y=1) |
|---:|---|---:|---:|---:|
| 0 | hi | 17 | 0 | 0.000 |
| 0 | lo | 26 | 2 | 0.077 |
| 1 | hi | 6 | 2 | 0.333 |
| 1 | lo | 28 | 1 | 0.036 |
| 2 | hi | 60 | 2 | 0.033 |
| 2 | lo | 73 | 3 | 0.041 |
| 3 | hi | 191 | 48 | 0.251 |
| 3 | lo | 161 | 12 | 0.075 |
| 4 | hi | 6 | 1 | 0.167 |
| 4 | lo | 17 | 1 | 0.059 |

### 35 — AP rate by fold

Range 0.044–0.123. A one-fold spike would be another cluster leftover.

| fold | n | pos | companies | P(Y=1) |
|---:|---:|---:|---:|---:|
| 0 | 807 | 99 | 87 | 0.123 |
| 1 | 665 | 78 | 76 | 0.117 |
| 2 | 882 | 42 | 85 | 0.048 |
| 3 | 1308 | 144 | 134 | 0.110 |
| 4 | 1243 | 55 | 142 | 0.044 |

### 36 — AP leftover share of positives by fold

Neither = not own-low cash and not own-high `d_supp_hhi`.

| fold | n pos | neither / defined | share leftover |
|---:|---:|---:|---:|
| 0 | 99 | 54 / 93 | 0.581 |
| 1 | 78 | 40 / 65 | 0.615 |
| 2 | 42 | 22 / 33 | 0.667 |
| 3 | 144 | 77 / 109 | 0.706 |
| 4 | 55 | 29 / 41 | 0.707 |

### 37 — AR leftover share of positives by fold

Neither = not own-low cash and not own-high `d_cust_hhi`. Fold 3 can still be leftover on cash/HHI while being the unnamed cluster.

| fold | n pos | neither / defined | share leftover |
|---:|---:|---:|---:|
| 0 | 40 | 23 / 34 | 0.676 |
| 1 | 21 | 8 / 17 | 0.471 |
| 2 | 48 | 29 / 38 | 0.763 |
| 3 | 96 | 49 / 73 | 0.671 |
| 4 | 32 | 17 / 31 | 0.548 |

### 38 — hole positives in the leftover cell

28/64 hole positives (43.8%) are leftover (not own-low cash, not own-high customer HHI). The unnamed cluster is a slice of leftover, not a cash/HHI rewrite.

### 39 — 2×2 among hole positives

n=52: cash_only=11 hhi_only=8 both=5 neither=28.

### 40 — months-so-far by fold

If fold 3 is just longer books, the unnamed cluster is a tenure artifact.

| fold | n | median so-far | share <12 | share ≥18 |
|---:|---:|---:|---:|---:|
| 0 | 596 | 14.0 | 0.314 | 0.268 |
| 1 | 326 | 13.0 | 0.393 | 0.230 |
| 2 | 680 | 14.0 | 0.347 | 0.196 |
| 3 | 1059 | 14.0 | 0.346 | 0.229 |
| 4 | 654 | 13.0 | 0.367 | 0.164 |

### 41 — AP months-so-far by fold

High-rate AP folds (0/1/3) vs low (2/4) — tenure confounder?

| fold | n | P(Y=1) | median so-far | share <12 |
|---:|---:|---:|---:|---:|
| 0 | 807 | 0.123 | 14.0 | 0.297 |
| 1 | 665 | 0.117 | 14.0 | 0.343 |
| 2 | 882 | 0.048 | 13.0 | 0.372 |
| 3 | 1308 | 0.110 | 13.0 | 0.346 |
| 4 | 1243 | 0.044 | 12.0 | 0.424 |

### 42 — tenure (so-far) as a single

| Y | CV AUROC ± sd | sign | size AUROC |
|---|---:|---:|---:|
| `y5_ap_od30_ownp80` | 0.546 ± 0.078 | 1 | 0.521 |
| `y5_ar_od30_sust` | 0.560 ± 0.073 | 1 | 0.520 |

### 43 — AP `h_group_size`≥18 by fold

Protective tail should show in every fold if it is a why.

| fold | n ≥18 | pos | P(Y=1) ≥18 | P(Y=1) rest |
|---:|---:|---:|---:|---:|
| 0 | 227 | 16 | 0.070 | 0.143 |
| 1 | 267 | 13 | 0.049 | 0.163 |
| 2 | 1 | 0 | 0.000 | 0.048 |
| 3 | 199 | 1 | 0.005 | 0.129 |
| 4 | 542 | 6 | 0.011 | 0.070 |

## Return (this owner)

Y5 is **unexplained leftover** on cash and counterpart HHI (65% neither). Not cash-stress (days>0 99%+). Not a supplier-tail (`d_supp_hhi` monopoly is protective). AR has a company-group unnamed-cp cluster — quote `d_tx_cp_share`, do not treat 17.4% as a panel law.

Best non-E vs night: AP `h_group_size` CV 0.581 vs night train 0.599 (CLOSE). AR `d_tx_cp_share` CV 0.576 vs night train 0.611 (KEEP quote, PARK as X). Thin-F PARK. Trees PARK. Q6 CLOSE.

Q5 sentence (no family E): sustained AR-od30 is higher when the bank trail names no counterparties on a thick invoice book — in one unnamed company-group (fold 3: 25.1% on 191 months), not as a leave-one-group law. AP leftover: not low cash, not supplier monopoly.


