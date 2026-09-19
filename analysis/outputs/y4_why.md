# Y4 why — crash vs spike, HHI clock, 2-col honesty

- **When:** 2026-09-19T01:31
- **Agent:** `79044b5e`
- **Python:** `/home/walterjtv/.pyenv/versions/base/bin/python3`
- **Re-run:** `python -m analysis.evaluate.y4_why`
- **Holdout:** 72 companies, seed 20260918. Out of every rate, AUROC, and cut.
- **Y:** `y4_ds_r_double` only (train 2370 / 329 / **13.88%**). Other Y4 columns not revived.
- **X:** family D lags. **Never F.** in3/ds3 rebuilt here from transactions + CAT_MAP to split the label — not used as X.
- **Quote:** train group-fold CV (5 folds). Holdout is LOW_POWER.
- **Brief:** Q3 turning / Q5 why / Q6 lead. Not bankruptcy. Not a 0–100.

## Decision

**CLOSE** the 2-col z-avg card. zavg CV 0.634 clears 0.605+0.02 and is not HHI-on-CC (HHI-CC 0.605, gap +0.029) but is n_supp on those same rows (n_supp-CC 0.631, gap +0.003 < 0.02)

Mean-CV HHI peak is **lag 6** CV **0.605** (lag3 is peak: False). Stable quote remains **lag 3 CV 0.605** (sd 0.044; lag 6 sd 0.175 / train 0.533).

## Pass 1 — decompose the double

Thresholds (documented): inflow crash `in3[t+3]/in3[t] < 0.8`; repayment spike `ds3[t+3]/ds3[t] > 1.5`.
Positives median in-ratio **0.36**, ds-ratio **1.15** (neg 1.10 / 0.98).

| slice | n | n_pos | share labeled | share of positives | P(Y=1) | size AUROC |
|---|---:|---:|---:|---:|---:|---:|
| crash_only | 665 | 212 | 0.281 | 0.644 | 0.319 | 0.397 |
| spike_only | 239 | 63 | 0.101 | 0.191 | 0.264 | 0.530 |
| both | 51 | 50 | 0.022 | 0.152 | 0.980 | 0.440 |
| neither | 1346 | 0 | 0.568 | 0.000 | 0.000 | nan |
| undefined_ratio | 69 | 4 | 0.029 | 0.012 | 0.058 | 0.500 |
| crash_any | 716 | 262 | 0.302 | 0.796 | 0.366 | 0.386 |
| spike_any | 290 | 113 | 0.122 | 0.343 | 0.390 | 0.485 |
| all_labeled | 2370 | 329 | 1.000 | 1.000 | 0.139 | 0.493 |

Size AUROC is `log1p(|in3[t]|)` vs the label *inside* the slice. Crash/spike flags use future in3/ds3 only to name the label — they are not X. Holdout 16 positives flip the mix (crash 0.375 / spike 0.875 / med in-ratio 0.88) — LOW_POWER.

## Pass 2 — Q6 clock

| feature | lag | CV AUROC ± sd | train AUROC | sign | coverage (train labeled) |
|---|---:|---:|---:|---:|---:|
| `d_cust_hhi` | 0 | 0.583 ± 0.074 | 0.574 | +1 | 0.442 (1047/2370) |
| `d_cust_hhi_lag1` | 1 | 0.603 ± 0.081 | 0.586 | +1 | 0.416 (986/2370) |
| `d_cust_hhi_lag3` | 3 | 0.605 ± 0.044 | 0.592 | +1 | 0.358 (848/2370) |
| `d_cust_hhi_lag6` | 6 | 0.605 ± 0.175 | 0.533 | +1 | 0.257 (609/2370) |
| `d_cust_top1` | 0 | 0.583 ± 0.072 | 0.573 | +1 | 0.442 (1047/2370) |
| `d_cust_top1_lag1` | 1 | 0.602 ± 0.079 | 0.583 | +1 | 0.416 (986/2370) |
| `d_cust_top1_lag3` | 3 | 0.604 ± 0.043 | 0.590 | +1 | 0.358 (848/2370) |
| `d_cust_top1_lag6` | 6 | 0.605 ± 0.173 | 0.531 | +1 | 0.257 (609/2370) |

HHI is invoice-gated. Lag 3 coverage is not the full ~53% store fill — it needs HHI defined three months earlier on a Y4-labeled month.
**Lag 3 is not the mean-CV peak** (lag 6 wins by <0.001). Do not replace the 0.605 quote with lag 6: fold sd is 0.17, train AUROC collapses to 0.53, coverage 26%. Stable peak stays lag 3.

## Pass 3 — quintiles (train labeled cuts)

| Q | interval | n | n_pos | P(Y=1) | median HHI |
|---:|---|---:|---:|---:|---:|
| 1 | (0.00531, 0.242] | 170 | 14 | 0.082 | 0.112 |
| 2 | (0.242, 0.412] | 169 | 26 | 0.154 | 0.331 |
| 3 | (0.412, 0.645] | 170 | 18 | 0.106 | 0.513 |
| 4 | (0.645, 0.975] | 169 | 20 | 0.118 | 0.782 |
| 5 | (0.975, 1.0] | 170 | 38 | 0.224 | 1.000 |

Monotone non-decreasing: **False**. Cuts from train labeled complete rows only.
Plot: `y4_hhi_quintiles.png`.
Holdout rates with *train* cuts (LOW_POWER, not a claim):
- Q1 n=8 pos=0 rate=0.000
- Q2 n=1 pos=0 rate=0.000
- Q3 n=6 pos=2 rate=0.333
- Q4 n=3 pos=0 rate=0.000
- Q5 n=3 pos=0 rate=0.000

## Pass 4 — 2-col card honesty

- z-avg HHI_lag3 + n_supp_lag3: CV **0.634 ± 0.076** (complete 848 / 116 pos)
- HHI_lag3 all defined rows: CV **0.605**
- HHI_lag3 on the z-avg complete-case: CV **0.605**
- n_supp_lag3 on the same complete-case: CV **0.631**
- gap vs 0.605 bar: +0.029; gap vs HHI-CC: +0.029
- just HHI on complete-case: **False**

**CLOSE.** zavg CV 0.634 clears 0.605+0.02 and is not HHI-on-CC (HHI-CC 0.605, gap +0.029) but is n_supp on those same rows (n_supp-CC 0.631, gap +0.003 < 0.02)

## Pass 5 — leak

| vs | n | Spearman | Pearson |
|---|---:|---:|---:|
| `f_ds_r` | 848 | +0.161 | +0.234 |
| `a_in3` | 848 | -0.293 | -0.134 |
| `d_cust_top1` | 820 | +0.798 | +0.794 |
| `d_cust_top1_lag3` | 848 | +0.991 | +0.979 |

HHI_lag3 **is** a top-1 rewrite (same D stem). Y7 forbids D for that reason; Y4 allowed D — do not stack HHI and top1.
f_ds_r / a_in3 are comparators only. They are not X. Family F never enters a score.

## Mapping (Q3 / Q5 / Q6)

Y4 `ds_r` doubling is who is *turning* (Q3): debt-service / inflow at t+3 is at least twice t, with a 0.05 floor. On train labeled rows the double is mostly a **denominator crash** (Q5) — 80% of positives drop inflow >20% (`in3` ratio < 0.8), 34% raise `ds3` by >50%, 15% do both. Size AUROC inside those slices stays below 0.60, so this is not a big-firm label. `d_cust_hhi` is the leading why (Q6): mean-CV peak is lag **6** CV **0.605** — lag 3 is not the mean-CV peak; lag 6 wins by <0.001 with sd 0.17, so the stable quote stays lag3 0.605. Train quintiles of HHI_lag3 (cuts from train labeled) read 8.2% / 15.4% / 10.6% / 11.8% / 22.4% — not monotone; the 0.605 is the HHI>0.975 bin (body CV 0.445). The 2-col z-avg of HHI_lag3 + n_supp_lag3 is **CLOSE** (CV 0.634 vs HHI-CC 0.605). HHI is a top-1 rewrite; Y4 may use D, but do not stack the two. Holdout's 16 positives flip the mix (mostly spike, not crash) — HHI should invert there. Not bankruptcy. Not a 0–100.

## Extra cuts

### 1b — residual split at spike 1.2 (20% ds rise)

The documented 0.8 / 1.5 cut leaves **no residual positives**: 1.5/0.8 = 1.875 < 2, so a `ds_r` double cannot land in neither without floor effects. The 0.8 / 1.2 pair (already quoted as 80% / 49%) is the split that can have a leftover.

| slice | n | n_pos | share of positives | P(Y=1) | size AUROC |
|---|---:|---:|---:|---:|---:|
| crash_only_1.2 | 611 | 167 | 0.508 | 0.273 | 0.388 |
| spike_only_1.2 | 395 | 63 | 0.191 | 0.159 | 0.528 |
| both_1.2 | 105 | 95 | 0.289 | 0.905 | 0.286 |
| neither_1.2 | 1190 | 0 | 0.000 | 0.000 | nan |
| spike_any_1.2 | 500 | 158 | 0.480 | 0.316 | 0.493 |

### 1c — HHI_lag3 inside the slices

| slice | n | n_pos | HHI n | HHI AUROC | med HHI pos / neg |
|---|---:|---:|---:|---:|---:|
| crash_only | 665 | 212 | 234 | 0.647 | 0.822 / 0.438 |
| spike_only | 239 | 63 | 93 | 0.605 | 0.602 / 0.520 |
| both | 51 | 50 | 23 | nan | 0.400 / nan |
| crash_any | 716 | 262 | 257 | 0.614 | 0.647 / 0.438 |
| spike_any | 290 | 113 | 116 | 0.530 | 0.504 / 0.520 |
| spike_only_1.2 | 395 | 63 | 147 | 0.642 | 0.602 / 0.500 |
| all_labeled | 2370 | 329 | 848 | 0.592 | 0.646 / 0.503 |

### 2b — lag-6 is not a usable peak

mean-CV peak is lag 6 (0.6052) but lag6 sd=0.175 train_auc=0.533 cov=0.257. Stable quote stays lag3 CV=0.6047 sd=0.044 train=0.592 cov=0.358.

### 2c — `d_n_supp` clock

| lag | CV AUROC ± sd | train AUROC | sign | coverage |
|---:|---:|---:|---:|---:|
| 0 | 0.558 ± 0.099 | 0.594 | -1 | 0.514 (1218) |
| 1 | 0.576 ± 0.087 | 0.599 | -1 | 0.485 (1150) |
| 3 | 0.600 ± 0.082 | 0.611 | -1 | 0.416 (986) |
| 6 | 0.609 ± 0.112 | 0.610 | -1 | 0.303 (719) |

n_supp best lag **6** CV **0.609**.

### 2d — invoice-gate

Train company-months HHI defined **42.2%** (8928/21157). Labeled months 44.2%; lag3 35.8% (848/2370).

### 3b — monopoly tail

`d_cust_hhi_lag3` > 0.975: P(Y=1) = **22.1%** (n=172, 38 pos) vs rest **11.5%**. Tail-indicator AUROC 0.572. Quintiles are not monotone (Q2 15.4% > Q3/Q4). Cut 3d: body HHI≤0.975 CV is below dummy — the 0.605 *is* the near-monopoly bin.

### 3c — quintiles on crash_any only

Cuts from the 257 crash+HHI train rows. Not the all-labeled quintiles.

| Q | interval | n | n_pos | P(Y=1) | median HHI |
|---:|---|---:|---:|---:|---:|
| 1 | (0.00559, 0.263] | 52 | 13 | 0.250 | 0.108 |
| 2 | (0.263, 0.391] | 51 | 13 | 0.255 | 0.323 |
| 3 | (0.391, 0.63] | 51 | 16 | 0.314 | 0.482 |
| 4 | (0.63, 0.991] | 51 | 16 | 0.314 | 0.844 |
| 5 | (0.991, 1.0] | 52 | 28 | 0.538 | 1.000 |

### 3d — without the monopoly tail

HHI_lag3 CV on rows with HHI≤0.975: **0.445 ± 0.132** (n=676, 78 pos). **The 0.605 is the monopoly tail.** Body loses to dummy; do not tell a smooth concentration-gradient story. Tail is 43 companies / 172 months. Those same companies on non-tail months: P(Y=1)=15.3% (n=242) vs tail months 22.1%. Binary HHI>0.975 OOF CV **0.578 ± 0.071** (weaker than continuous 0.605; still the honest card shape).

### 1e — HHI vs crash flag

HHI_lag3 vs crash-any on all labeled AUROC **0.494**; among positives **0.478** (n=116). Among positives ρ(HHI, in_ratio)=-0.123, ρ(HHI, ds_ratio)=-0.154. HHI does **not** classify crash vs spike among positives (coin flip). It ranks Y *inside* crash months (OOF 0.64), especially the monopoly tail.

### 1d — OOF inside crash vs spike

| slice | feature | n | n_pos | CV AUROC ± sd | coverage |
|---|---|---:|---:|---:|---:|
| crash_any | `d_cust_hhi_lag3` | 716 | 262 | 0.639 ± 0.070 | 0.359 |
| crash_any | `d_n_supp_lag3` | 716 | 262 | 0.555 ± 0.159 | 0.444 |
| crash_only | `d_cust_hhi_lag3` | 665 | 212 | 0.648 ± 0.077 | 0.352 |
| crash_only | `d_n_supp_lag3` | 665 | 212 | 0.579 ± 0.184 | 0.441 |
| spike_any | `d_cust_hhi_lag3` | 290 | 113 | 0.524 ± 0.046 | 0.400 |
| spike_any | `d_n_supp_lag3` | 290 | 113 | 0.563 ± 0.166 | 0.438 |
| spike_only | `d_cust_hhi_lag3` | 239 | 63 | 0.561 ± 0.081 | 0.389 |
| spike_only | `d_n_supp_lag3` | 239 | 63 | 0.566 ± 0.222 | 0.427 |

HHI clock on crash_any rows only:

| lag | CV AUROC ± sd | train AUROC | coverage |
|---:|---:|---:|---:|
| 0 | 0.622 ± 0.080 | 0.611 | 0.420 |
| 1 | 0.663 ± 0.139 | 0.621 | 0.395 |
| 3 | 0.639 ± 0.070 | 0.614 | 0.359 |
| 6 | 0.643 ± 0.226 | 0.576 | 0.263 |

On crash rows the mean-CV HHI peak is lag **1** CV **0.663** (sd 0.139). Lag 3 is the stable crash clock (0.639 ± 0.070).

### Holdout check (LOW_POWER, not a claim)

Train-only signs. Holdout 16 positives flip the mechanism (mostly spike, not crash).

| feature | sign | hold AUROC | n defined | n_pos |
|---|---:|---:|---:|---:|
| `d_cust_hhi_lag3` | +1 | 0.526 | 21 | 2 |
| `d_n_supp_lag3` | -1 | 0.087 | 25 | 2 |
| `d_cust_hhi` | +1 | 0.550 | 32 | 2 |
| `d_cust_top1_lag3` | +1 | 0.579 | 21 | 2 |


