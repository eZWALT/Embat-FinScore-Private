# Family B leftover on the 44 — DROP evidence

- **When:** 2026-09-19T04:53:31+02:00
- **Agent:** `c8b1440a`
- **Python:** `/home/walterjtv/.pyenv/versions/base/bin/python3`
- **Re-run:** `python -m analysis.evaluate.b_on_44_qa`
- **Holdout:** 72 companies, seed 20260918. Coverage only. Rates / AUROC on train.
- **Never B as Y2/Y3 X.** Singles and leftovers are diagnostics for DROP, not KEEP.
- Walk identity / still-leak not reopened except persist confirm.
- Not a 0–100. Not a parquet rewrite. Not a 15-col retrain. Do not invent `y_bal_vol`.

## Headline

- Train 21,157 CM / 1214 companies. `b_liq` all-null=19 = 13 tx-no-balance + 6 no-cash-walk (CONFIRM 13+6=19). tx-no-balance train=13 (CONFIRM HOLE13). Santander UK on 6 of the 13; last-tx 2026-07-20 n=6. Zombie cash 538/4,408=12.2% (CONFIRM 12.2%) |zombie|/|cash|=0.4% (quote 0.36–0.4%). PARK as a flag.
- Twins ≥0.80 among FOCUS: 1. `b_below_0`↔`b_neg_liq_3` ρ=0.844 (feature-report dropped neg_liq_3). `b_below_0`↔`b_neg_episodes` ρ=0.435 not a twin. `b_bal_vol`↔`a_vol` ρ=0.354 (CONFIRM DRIFT 0.354).
- Y3 days=0.711 (CONFIRM 0.711) size=0.617 (CONFIRM 0.617). Y2 `b_below_0`=0.896 `b_runway`=0.924 — LOCK leak (not KEEP). Singles are diagnostics. Never B as Y2/Y3 X.
- Y3 leftover after days: b_bal_vol=0.734, b_below_0=0.649, b_d_runway=0.748, b_neg_episodes=0.687, b_runway=0.711. Fake-days any=True. Honest leftover still lives. Leftover is a DROP evidence, not a KEEP-as-X.
- Rank leftover kills `b_runway` (0.537); `b_d_runway` leftover is the Q1 crash cell (dummy≈continuous; body leftover 0.527 DIES).
- Y2 leftover after `b_below_0`: b_bal_vol=0.864, b_below_0=self, b_d_runway=0.697, b_neg_episodes=0.596, b_runway=0.618. a leftover lives — inspect.
- ρ(b_bal_vol, a_vol)=0.354 n=14,968 (CONFIRM 0.354 DRIFT). Y3 leftover of store vol after Javier=0.621 after days=0.734 after both=0.732.
- DROP all five from the 44 as Y2/Y3 X: **True**. KEEP as Q1 description: ['b_runway']. Night quotes unchanged 0.762/0.752.
- Night quotes unchanged: Y3 **0.762 / 0.752**. Days **0.711**. Size **0.617**. Y7 TURNOVER **0.72 / 0.712**.

## Brief questions

1. **Who is healthy?** — last-value `b_runway` stays KEEP as Q1 *description* of the extract still. It is not a Y3 X.
2. **Who is improving?** — `b_d_runway` is a 3-month Δ of that still-walked path. Empty-on-short. DROP as X.
3. **Who is turning?** — do not invent `y_bal_vol`. Y2 already *is* the B path.
4. **Dip vs fall?** — `b_below_0` / episodes are the Y2 definition, not a leftover why.
5. **Why did it change?** — store `b_bal_vol` is not Javier cashflow vol (DRIFT 0.354).
6. **Months earlier?** — vol / d_runway need history. Q6 KEEP=False. Short persist lower (0.73), not higher.

## PARK / CLOSE / KEEP / DROP-from-44

| col | Y3 single | Y2 single | honest leftover days | DROP from 44 as X | KEEP as Q1 description | CLOSE as Y2/Y3 X | PARK as Y | why |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b_bal_vol | 0.692 | 0.586 | DIES (fake) | YES | no | YES | YES | DRIFT vs Javier vol ρ=0.354. Y3 leftover after Javier=0.621 after days=0.734. Not the a_out_vol trait. DROP from the 44. Do not invent y_bal_vol. |
| b_below_0 | 0.509 | 0.896 | DIES (fake) | YES | no | YES | YES | Y2 leak (already-neg). Leftover after sign dies. DROP from the 44. PARK as a flag, not a Y. |
| b_d_runway | 0.659 | 0.532 | 0.748 | YES | no | YES | YES | 3-month Δ of the still-walked runway. Empty-on-short. OLS leftover looks high; Q1 dummy≈continuous; body leftover after days=0.527 DIES. Q1 crash cell is the Y3 path (47% of recoveries). DROP from the 44 as X. |
| b_neg_episodes | 0.481 | 0.659 | DIES (fake) | YES | no | YES | YES | not a below_0 twin; still leftover-dies. DROP from the 44. Do not invent a cousin of Y2. |
| b_runway | 0.596 | 0.924 | 0.711 | YES | YES | YES | YES | KEEP last-value as Q1 description (still; p50=1.079). OLS leftover after days=0.711 is days-shaped; rank leftover 0.537 DIES. LOOK-AHEAD still Y3=0.823. DROP from the 44 as a Y3 X. PARK as forecast Y. |


## 1. Coverage; 13 Santander UK nulls; zombie cash

Train 21,157 CM / 1214 companies. `b_liq` all-null=19 = 13 tx-no-balance + 6 no-cash-walk (CONFIRM 13+6=19). tx-no-balance train=13 (CONFIRM HOLE13). Santander UK on 6 of the 13; last-tx 2026-07-20 n=6. Zombie cash 538/4,408=12.2% (CONFIRM 12.2%) |zombie|/|cash|=0.4% (quote 0.36–0.4%). PARK as a flag.

| col | cov_cm | n_finite | n_null | cos_any | cos_all_null |
| --- | --- | --- | --- | --- | --- |
| b_bal_vol | 70.7% | 14,968 | 6,189 | 1192 | 22 |
| b_below_0 | 99.0% | 20,941 | 216 | 1195 | 19 |
| b_d_runway | 70.7% | 14,968 | 6,189 | 1192 | 22 |
| b_neg_episodes | 70.7% | 14,968 | 6,189 | 1192 | 22 |
| b_runway | 87.7% | 18,551 | 2,606 | 1195 | 19 |
| b_liq | 99.0% | 20,941 | 216 | 1195 | 19 |
| b_neg_liq_3 | 87.7% | 18,551 | 2,606 | 1195 | 19 |


| col | holdout_cov | holdout_cos_any |
| --- | --- | --- |
| b_bal_vol | 66.5% | 71 |
| b_below_0 | 100.0% | 72 |
| b_d_runway | 66.5% | 71 |
| b_neg_episodes | 66.5% | 71 |
| b_runway | 86.6% | 72 |


Hole-13 list match=False; tx-no-balance CONFIRM=True; 13+6=19 CONFIRM=True; Santander UK n=6; last-tx 2026-07-20 n=6. Zombie 538/4,408=12.2% |share|=0.4%. PARK as a flag. Do not rewrite `liquidity.py`.

## 2. Spearman twins

Twins ≥0.80 among FOCUS: 1. `b_below_0`↔`b_neg_liq_3` ρ=0.844 (feature-report dropped neg_liq_3). `b_below_0`↔`b_neg_episodes` ρ=0.435 not a twin. `b_bal_vol`↔`a_vol` ρ=0.354 (CONFIRM DRIFT 0.354).

Within-FOCUS matrix:

| col | b_bal_vol | b_below_0 | b_d_runway | b_neg_episodes | b_runway |
| --- | --- | --- | --- | --- | --- |
| b_bal_vol | 1 | -0.080 | 0.011 | -0.038 | 0.623 |
| b_below_0 | -0.080 | 1 | -0.067 | 0.435 | -0.458 |
| b_d_runway | 0.011 | -0.067 | 1 | 0.001 | 0.289 |
| b_neg_episodes | -0.038 | 0.435 | 0.001 | 1 | -0.238 |
| b_runway | 0.623 | -0.458 | 0.289 | -0.238 | 1 |


Twin ≥0.80 → DROP weaker:

| pair | ρ | n | DROP weaker | keep | why |
| --- | --- | --- | --- | --- | --- |
| b_below_0 ↔ b_neg_liq_3 | 0.844 | 18,551 | b_neg_liq_3 | b_below_0 | cov b_neg_liq_3=87.7% ≤ b_below_0=99.0% |


Selected pairs (FOCUS vs size / days / Javier vol / last-value):

| a | b | ρ | n | twin≥0.80 | SIZE |
| --- | --- | --- | --- | --- | --- |
| b_bal_vol | b_runway | 0.623 | 14,968 | no | no |
| b_bal_vol | log_in3 | -0.282 | 14,968 | no | no |
| b_bal_vol | c_n_days_with_tx | -0.282 | 14,968 | no | no |
| b_bal_vol | a_vol | 0.354 | 14,968 | no | no |
| b_bal_vol | a_out_vol | 0.331 | 14,968 | no | no |
| b_bal_vol | b_neg_liq_3 | -0.071 | 14,968 | no | no |
| b_bal_vol | b_runway_last | 0.542 | 14,968 | no | no |
| b_below_0 | b_runway | -0.458 | 18,551 | no | no |
| b_below_0 | log_in3 | 0.041 | 18,551 | no | no |
| b_below_0 | c_n_days_with_tx | 0.079 | 20,941 | no | no |
| b_below_0 | a_vol | 0.004 | 14,968 | no | no |
| b_below_0 | a_out_vol | -0.066 | 14,968 | no | no |
| b_below_0 | b_neg_liq_3 | 0.844 | 18,551 | YES | no |
| b_below_0 | b_runway_last | -0.239 | 20,941 | no | no |
| b_d_runway | b_runway | 0.289 | 14,968 | no | no |
| b_d_runway | log_in3 | -0.045 | 14,968 | no | no |
| b_d_runway | c_n_days_with_tx | -0.054 | 14,968 | no | no |
| b_d_runway | a_vol | -0.024 | 14,968 | no | no |
| b_d_runway | a_out_vol | 0.016 | 14,968 | no | no |
| b_d_runway | b_neg_liq_3 | 0.005 | 14,968 | no | no |
| b_d_runway | b_runway_last | 0.108 | 14,968 | no | no |
| b_neg_episodes | b_runway | -0.238 | 14,968 | no | no |
| b_neg_episodes | log_in3 | 0.066 | 14,968 | no | no |
| b_neg_episodes | c_n_days_with_tx | 0.115 | 14,968 | no | no |
| b_neg_episodes | a_vol | -0.034 | 14,968 | no | no |
| b_neg_episodes | a_out_vol | -0.058 | 14,968 | no | no |
| b_neg_episodes | b_neg_liq_3 | 0.594 | 14,968 | no | no |
| b_neg_episodes | b_runway_last | -0.156 | 14,968 | no | no |
| b_runway | log_in3 | -0.337 | 18,551 | no | no |
| b_runway | c_n_days_with_tx | -0.323 | 18,551 | no | no |
| b_runway | a_vol | 0.195 | 14,968 | no | no |
| b_runway | a_out_vol | 0.258 | 14,968 | no | no |
| b_runway | b_neg_liq_3 | -0.431 | 18,551 | no | no |
| b_runway | b_runway_last | 0.683 | 18,551 | no | no |


## 3. Single-feature group-fold (diagnostic leak)

Y3 days=0.711 (CONFIRM 0.711) size=0.617 (CONFIRM 0.617). Y2 `b_below_0`=0.896 `b_runway`=0.924 — LOCK leak (not KEEP). Singles are diagnostics. Never B as Y2/Y3 X.

Sign from the train side of each fold. Seed 20260918. Holdout never entered a fold.
Y2 looking strong is the lock, not a KEEP.

| Y | feature | CV | ±sd | sign | n | n_pos | folds | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Y3 | b_bal_vol | 0.692 | 0.075 | 1 | 4,212 | 313 | 0.779 0.718 0.733 0.594 0.637 | illegal X |
| Y3 | b_below_0 | 0.509 | 0.017 | -1 | 5,648 | 402 | 0.528 0.499 0.497 0.494 0.528 | illegal X |
| Y3 | b_d_runway | 0.659 | 0.053 | -1 | 4,212 | 313 | 0.639 0.718 0.713 0.618 0.604 | illegal X |
| Y3 | b_neg_episodes | 0.481 | 0.016 | 1 | 4,212 | 313 | 0.458 0.494 0.469 0.487 0.495 | illegal X |
| Y3 | b_runway | 0.596 | 0.049 | 1 | 5,528 | 391 | 0.662 0.580 0.601 0.527 0.609 | illegal X |
| Y3 | log_in3 | 0.617 | 0.061 | -1 | 5,528 | 391 | 0.565 0.632 0.683 0.543 0.661 | bench |
| Y3 | c_n_days_with_tx | 0.711 | 0.031 | -1 | 5,648 | 402 | 0.665 0.738 0.700 0.715 0.740 | bench |
| Y3 | b_runway | 0.596 | 0.049 | 1 | 5,528 | 391 | 0.662 0.580 0.601 0.527 0.609 | illegal X |
| Y3 | b_runway_last | 0.823 | 0.072 | 1 | 5,648 | 402 | 0.938 0.759 0.774 0.849 0.796 | LOOK-AHEAD still |
| Y2 | b_bal_vol | 0.586 | 0.108 | -1 | 11,477 | 748 | 0.737 0.666 0.520 0.500 0.509 | illegal X |
| Y2 | b_below_0 | 0.896 | 0.018 | 1 | 17,356 | 1271 | 0.895 0.919 0.871 0.907 0.888 | illegal X |
| Y2 | b_d_runway | 0.532 | 0.032 | -1 | 11,477 | 748 | 0.515 0.562 0.498 0.514 0.570 | illegal X |
| Y2 | b_neg_episodes | 0.659 | 0.026 | 1 | 11,477 | 748 | 0.700 0.653 0.649 0.631 0.660 | illegal X |
| Y2 | b_runway | 0.924 | 0.016 | -1 | 14,968 | 1044 | 0.948 0.927 0.902 0.924 0.921 | illegal X |
| Y2 | log_in3 | 0.552 | 0.046 | 1 | 14,968 | 1044 | 0.523 0.609 0.595 0.520 0.513 | bench |
| Y2 | c_n_days_with_tx | 0.571 | 0.046 | 1 | 17,356 | 1271 | 0.623 0.539 0.612 0.565 0.517 | bench |
| Y2 | b_runway | 0.924 | 0.016 | -1 | 14,968 | 1044 | 0.948 0.927 0.902 0.924 0.921 | illegal X |
| Y2 | b_runway_last | 0.747 | 0.074 | -1 | 17,356 | 1271 | 0.809 0.682 0.656 0.818 0.769 | LOOK-AHEAD still |


## 4. Honest leftover after days / last-value runway (Y3)

Y3 leftover after days: b_bal_vol=0.734, b_below_0=0.649, b_d_runway=0.748, b_neg_episodes=0.687, b_runway=0.711. Fake-days any=True. Honest leftover still lives. Leftover is a DROP evidence, not a KEEP-as-X.

OLS residual of the column on days (or runway), then oriented-signed group-fold AUROC vs Y3.
Leftover <0.55 dies. ρ(resid, days) ≥0.30 is a fake days leak (same as g_has).

| col | after days | honest days | ρ(resid,days) | Pearson | rank leftover | near-fake | after size | after b_runway | after last-value | after days+size | dies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b_bal_vol | 0.734 | DIES (fake days) | 0.978 | -0.000 | 0.631 | YES | 0.610 | 0.594 | 0.838 | 0.615 | YES |
| b_below_0 | 0.649 | DIES (fake days) | -0.819 | 0.000 | 0.649 | YES | 0.566 | 0.571 | 0.717 | 0.651 | YES |
| b_d_runway | 0.748 | 0.748 | 0.287 | -0.000 | 0.671 | YES | 0.690 | 0.656 | 0.811 | 0.729 | no |
| b_neg_episodes | 0.687 | DIES (fake days) | -0.811 | 0.000 | 0.687 | YES | 0.576 | 0.597 | 0.768 | 0.688 | YES |
| b_runway | 0.711 | 0.711 | 0.224 | 0.000 | 0.537 | no | 0.605 | — | 0.788 | 0.684 | no |


## 5. Y2 leftover after `b_below_0` / last-value sign

Y2 leftover after `b_below_0`: b_bal_vol=0.864, b_below_0=self, b_d_runway=0.697, b_neg_episodes=0.596, b_runway=0.618. a leftover lives — inspect.

| col | Y2 raw | after b_below_0 | honest after sign | after last-value sign | after days | dies after sign |
| --- | --- | --- | --- | --- | --- | --- |
| b_bal_vol | 0.586 | 0.864 | DIES (fake) | 0.601 | 0.616 | YES |
| b_below_0 | 0.896 | self | self | 0.869 | 0.876 | self / Y def |
| b_d_runway | 0.532 | 0.697 | 0.697 | 0.538 | 0.516 | no |
| b_neg_episodes | 0.659 | 0.596 | 0.596 | 0.562 | 0.614 | no |
| b_runway | 0.924 | 0.618 | 0.618 | 0.773 | 0.730 | no |


## 6. `b_bal_vol` vs Javier vol

ρ(b_bal_vol, a_vol)=0.354 n=14,968 (CONFIRM 0.354 DRIFT). Y3 leftover of store vol after Javier=0.621 after days=0.734 after both=0.732.

| object | value | n | verdict |
| --- | --- | --- | --- |
| b_bal_vol vs a_vol ρ | 0.354 | 14,968 | DRIFT CONFIRM |
| Y3 b_bal_vol raw | 0.692 | 4,212 | illegal X |
| Y3 a_vol raw (CLOSE as X) | 0.626 | 4,212 | already CLOSE |
| Y3 leftover vol after Javier | 0.621 | 14,968 | DIES |
| Y3 leftover vol after days | 0.734 | 14,968 | DIES |
| Y3 leftover vol after Javier+days | 0.732 | 14,968 | DIES |
| Y3 leftover vol after a_out_vol | 0.719 | 14,968 | DIES |
| Y3 a_out_vol raw (trait dummy) | 0.722 | 4,212 | CLOSE / not merge |


## 7. SIZE terciles

T1 survive-vs-days vol=False d_runway=True. T1 vol CV=0.634 vs days 0.603 Δdays=0.032 leftover 0.613.

| col | slice | CV | size | days | Δsize | Δdays | leftover days | honest | n_pos | survive T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b_bal_vol | T1_small | 0.634 | 0.427 | 0.603 | 0.207 | 0.032 | 0.613 | DIES | 214 | no |
| b_bal_vol | T2_mid | 0.757 | 0.634 | 0.673 | 0.123 | 0.083 | 0.636 | DIES | 53 | no |
| b_bal_vol | T3_large | — | 0.620 | 0.733 | — | — | — | DIES | 46 | no |
| b_bal_vol | all | 0.692 | 0.617 | 0.711 | 0.076 | -0.019 | 0.734 | DIES | 313 | no |
| b_d_runway | T1_small | 0.641 | 0.427 | 0.603 | 0.213 | 0.038 | 0.668 | 0.668 | 214 | YES |
| b_d_runway | T2_mid | 0.635 | 0.634 | 0.673 | 0.001 | -0.038 | 0.612 | 0.612 | 53 | no |
| b_d_runway | T3_large | — | 0.620 | 0.733 | — | — | — | — | 46 | no |
| b_d_runway | all | 0.659 | 0.617 | 0.711 | 0.042 | -0.053 | 0.748 | 0.748 | 313 | no |
| b_runway | T1_small | 0.610 | 0.427 | 0.603 | 0.183 | 0.007 | 0.602 | 0.602 | 261 | no |
| b_runway | T2_mid | 0.504 | 0.634 | 0.673 | -0.130 | -0.169 | 0.648 | 0.648 | 72 | no |
| b_runway | T3_large | 0.443 | 0.620 | 0.733 | -0.177 | -0.290 | 0.749 | DIES | 58 | no |
| b_runway | all | 0.596 | 0.617 | 0.711 | -0.021 | -0.116 | 0.711 | 0.711 | 391 | no |
| b_below_0 | T1_small | 0.514 | 0.427 | 0.603 | 0.087 | -0.088 | 0.577 | DIES | 269 | no |
| b_below_0 | T2_mid | 0.436 | 0.634 | 0.673 | -0.198 | -0.237 | 0.648 | DIES | 73 | no |
| b_below_0 | T3_large | 0.516 | 0.620 | 0.733 | -0.104 | -0.216 | 0.697 | DIES | 60 | no |
| b_below_0 | all | 0.509 | 0.617 | 0.711 | -0.107 | -0.202 | 0.649 | DIES | 402 | no |


## 8. Q6 lags + short persist

b_liq persist short=0.730 (CONFIRM 0.730) long24=0.862 (CONFIRM 0.862). vol/d_runway need history — empty on so_far<6. Q6 KEEP=False (B forbidden).

| col | lag0 | lag1 | lag3 | lift1 | Q6 KEEP |
| --- | --- | --- | --- | --- | --- |
| b_bal_vol | 0.692 | 0.701 | 0.723 | 0.009 | False (B forbidden) |
| b_d_runway | 0.659 | 0.617 | 0.538 | -0.042 | False (B forbidden) |
| b_runway | 0.596 | 0.637 | 0.677 | 0.041 | False (B forbidden) |


Empty-on-short (needs history):

| col | slice | n_cm | finite | empty |
| --- | --- | --- | --- | --- |
| b_bal_vol | short_<12 | 2,970 | 39.2% | 60.8% |
| b_bal_vol | long_24 | 10,440 | 78.8% | 21.2% |
| b_bal_vol | so_far<6 | 6,068 | 0.0% | 100.0% |
| b_bal_vol | so_far≥6 | 15,089 | 99.2% | 0.8% |
| b_d_runway | short_<12 | 2,970 | 39.2% | 60.8% |
| b_d_runway | long_24 | 10,440 | 78.8% | 21.2% |
| b_d_runway | so_far<6 | 6,068 | 0.0% | 100.0% |
| b_d_runway | so_far≥6 | 15,089 | 99.2% | 0.8% |


Persist t vs t+3 (CONFIRM Family B short 0.730 / 24m 0.862 on `b_liq`):

| col | slice | ρ t,t+3 | n pairs | n cos |
| --- | --- | --- | --- | --- |
| b_liq | all | 0.848 | 17,356 | 1195 |
| b_liq | short_<12 | 0.730 | 1,833 | 336 |
| b_liq | long_24 | 0.862 | 9,093 | 433 |
| b_runway | all | 0.757 | 14,968 | 1192 |
| b_runway | short_<12 | 0.669 | 1,163 | 333 |
| b_runway | long_24 | 0.763 | 8,227 | 433 |
| b_bal_vol | all | 0.803 | 11,477 | 990 |
| b_bal_vol | short_<12 | 0.857 | 249 | 131 |
| b_bal_vol | long_24 | 0.806 | 6,928 | 433 |
| b_d_runway | all | -0.276 | 11,477 | 990 |
| b_d_runway | short_<12 | -0.286 | 249 | 131 |
| b_d_runway | long_24 | -0.310 | 6,928 | 433 |


## 9. ICC / company-demean

`b_bal_vol` ICC=0.123 η²=0.129 demean 0.692→0.789 trait=False. `a_out_vol` Y3-lab η²=0.741 (CONFIRM ~0.741).

| col | ICC | η² | Y3-lab η² | raw | demean | drop | kind |
| --- | --- | --- | --- | --- | --- | --- | --- |
| b_bal_vol | 0.123 | 0.129 | 0.563 | 0.692 | 0.789 | -0.096 | shock |
| b_below_0 | 0.570 | 0.583 | 0.744 | 0.509 | 0.535 | -0.025 | neither |
| b_d_runway | 0.164 | 0.087 | 0.608 | 0.659 | 0.798 | -0.140 | shock |
| b_neg_episodes | 0.590 | 0.557 | 0.681 | 0.481 | 0.557 | -0.076 | neither |
| b_runway | 0.677 | 0.668 | 0.838 | 0.596 | 0.940 | -0.344 | shock |
| a_vol | 0.720 | 0.672 | 0.798 | 0.626 | 0.465 | 0.162 | trait |
| a_out_vol | 0.624 | 0.583 | 0.741 | 0.722 | 0.549 | 0.172 | trait |
| c_n_days_with_tx | 0.768 | 0.800 | 0.902 | 0.711 | 0.502 | 0.209 | trait |


## 10. `b_below_0` vs `b_neg_episodes` / `b_neg_liq_3`

neg_liq_3 twin with below_0 ρ=0.844 (CONFIRM drop). episodes vs below_0 ρ=0.435 twin=False. Last-month below_0 share=4.6%; ever-below=242; ever-episode=173.

| pair | ρ | n | verdict |
| --- | --- | --- | --- |
| b_below_0 ↔ b_neg_liq_3 | 0.844 | 18,551 | TWIN (report already dropped neg_liq_3) |
| b_below_0 ↔ b_neg_episodes | 0.435 | 14,968 | not a twin |
| b_neg_episodes ↔ b_neg_liq_3 | 0.594 | 14,968 | not twin |
| Y3 leftover episodes after below_0 | 0.518 | 14,968 | DIES |
| Y2 leftover episodes after below_0 | 0.596 | 14,968 | lives |


## 11. Holdout coverage (LOW_POWER)

Holdout companies=72 (CONFIRM 72). Y2 pos=23 Y3 pos=14. Coverage only — no holdout AUROC trophy. LOW_POWER.

| col | holdout cov | cos any | n_cm |
| --- | --- | --- | --- |
| b_bal_vol | 66.5% | 71 | 714 |
| b_below_0 | 100.0% | 72 | 1,073 |
| b_d_runway | 66.5% | 71 | 714 |
| b_neg_episodes | 66.5% | 71 | 714 |
| b_runway | 86.6% | 72 | 929 |
| a_vol | 66.5% | 71 | 714 |
| b_liq | 100.0% | 72 | 1,073 |


## 12. Decision table

DROP all five from the 44 as Y2/Y3 X: **True**. KEEP as Q1 description: ['b_runway']. Night quotes unchanged 0.762/0.752.

| col | Y3 single | Y2 single | honest leftover days | DROP from 44 as X | KEEP as Q1 description | CLOSE as Y2/Y3 X | PARK as Y | why |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b_bal_vol | 0.692 | 0.586 | DIES (fake) | YES | no | YES | YES | DRIFT vs Javier vol ρ=0.354. Y3 leftover after Javier=0.621 after days=0.734. Not the a_out_vol trait. DROP from the 44. Do not invent y_bal_vol. |
| b_below_0 | 0.509 | 0.896 | DIES (fake) | YES | no | YES | YES | Y2 leak (already-neg). Leftover after sign dies. DROP from the 44. PARK as a flag, not a Y. |
| b_d_runway | 0.659 | 0.532 | 0.748 | YES | no | YES | YES | 3-month Δ of the still-walked runway. Empty-on-short. OLS leftover looks high; Q1 dummy≈continuous; body leftover after days=0.527 DIES. Q1 crash cell is the Y3 path (47% of recoveries). DROP from the 44 as X. |
| b_neg_episodes | 0.481 | 0.659 | DIES (fake) | YES | no | YES | YES | not a below_0 twin; still leftover-dies. DROP from the 44. Do not invent a cousin of Y2. |
| b_runway | 0.596 | 0.924 | 0.711 | YES | YES | YES | YES | KEEP last-value as Q1 description (still; p50=1.079). OLS leftover after days=0.711 is days-shaped; rank leftover 0.537 DIES. LOOK-AHEAD still Y3=0.823. DROP from the 44 as a Y3 X. PARK as forecast Y. |


## Extras

### vs `a_out_vol` trait

ρ(b_bal_vol, a_out_vol)=0.331 n=14,968 twin=False. bal_vol η²=0.129 vs a_out_vol η²=0.583. same company-vol dummy=False. Do not invent y_bal_vol.

| pair | ρ | n | twin | same dummy |
| --- | --- | --- | --- | --- |
| b_bal_vol ↔ a_out_vol | 0.331 | 14,968 | no | no |


### leakage_check

leakage_check Y3 ok=False issues=["forbidden prefix 'b_': ['b_bal_vol', 'b_below_0', 'b_d_runway', 'b_neg_episodes', 'b_runway']"]. Y2 ok=False. Forbidden prefix fires — lock holds. Singles above are diagnostics, not a model.

### persist confirm (walk not rewritten)

Store first-vs-last `b_liq` ρ=0.613 n=1195 (Family B first-vs-last ~0.61). first-vs-last `b_runway` ρ=0.529. Walk identity / still-leak not reopened.

| pair | ρ | n |
| --- | --- | --- |
| first vs last b_liq | 0.613 | 1,195 |
| first vs last b_runway | 0.529 | 1,195 |


### Y3 quintiles

Y3 quintile shapes: b_bal_vol=tail, b_d_runway=flat, b_runway=flat

| col | Q | n | P(Y3=1) | median X |
| --- | --- | --- | --- | --- |
| b_bal_vol | 1 | 843 | 3.6% | 0.001 |
| b_bal_vol | 2 | 842 | 3.6% | 0.063 |
| b_bal_vol | 3 | 842 | 6.1% | 0.191 |
| b_bal_vol | 4 | 842 | 5.9% | 0.410 |
| b_bal_vol | 5 | 843 | 18.0% | 1.300 |
| b_d_runway | 1 | 843 | 17.6% | -2.338 |
| b_d_runway | 2 | 842 | 5.9% | -0.302 |
| b_d_runway | 3 | 842 | 3.2% | -0.022 |
| b_d_runway | 4 | 842 | 3.7% | 0.007 |
| b_d_runway | 5 | 843 | 6.8% | 0.282 |
| b_runway | 1 | 1,106 | 5.2% | -0.313 |
| b_runway | 2 | 1,105 | 4.3% | 0.018 |
| b_runway | 3 | 1,106 | 4.9% | 0.140 |
| b_runway | 4 | 1,105 | 9.9% | 0.370 |
| b_runway | 5 | 1,106 | 11.2% | 0.748 |


### leftover inside days terciles

Inside days terciles, leftover live=True (expect False).

| col | days tercile | CV | size | leftover | honest | n_pos |
| --- | --- | --- | --- | --- | --- | --- |
| b_bal_vol | D1_quiet | 0.639 | 0.563 | 0.649 | DIES | 173 |
| b_bal_vol | D2 | 0.666 | 0.524 | 0.582 | DIES | 106 |
| b_bal_vol | D3_busy | — | 0.371 | — | DIES | 34 |
| b_below_0 | D1_quiet | 0.523 | 0.563 | 0.600 | DIES | 219 |
| b_below_0 | D2 | 0.511 | 0.524 | 0.537 | DIES | 128 |
| b_below_0 | D3_busy | 0.476 | 0.371 | 0.429 | DIES | 55 |
| b_d_runway | D1_quiet | 0.624 | 0.563 | 0.680 | 0.680 | 173 |
| b_d_runway | D2 | 0.654 | 0.524 | 0.657 | 0.657 | 106 |
| b_d_runway | D3_busy | — | 0.371 | — | — | 34 |
| b_neg_episodes | D1_quiet | 0.484 | 0.563 | 0.650 | DIES | 173 |
| b_neg_episodes | D2 | 0.483 | 0.524 | 0.571 | DIES | 106 |
| b_neg_episodes | D3_busy | — | 0.371 | — | DIES | 34 |
| b_runway | D1_quiet | 0.608 | 0.563 | 0.625 | 0.625 | 214 |
| b_runway | D2 | 0.604 | 0.524 | 0.527 | DIES | 126 |
| b_runway | D3_busy | 0.623 | 0.371 | 0.393 | DIES | 51 |


### last-value runway on short books

Last-value runway short n=350 p50=0.467 long24 n=435. Description only — not a forecast Y.

| slice | n cos | runway p50 | share<1 | null |
| --- | --- | --- | --- | --- |
| short_<12 | 350 | 0.467 | 58.3% | 4.0% |
| long_24 | 435 | 1.346 | 46.2% | 0.5% |
| all | 1214 | 1.079 | 48.4% | 1.6% |


### Y2 already-neg vs clean-now

Y2 on already-neg vs clean-now. High rate on already-neg is the Y definition; leftover on clean-now should die.

| slice | n | P(Y2) | b_runway CV | b_bal_vol CV | episodes CV | n_pos |
| --- | --- | --- | --- | --- | --- | --- |
| already_neg | 1,465 | 70.9% | 0.597 | 0.512 | 0.684 | 849 |
| clean_now | 15,891 | 1.5% | 0.654 | 0.618 | 0.731 | 195 |
| all_labeled | 17,356 | 7.3% | 0.924 | 0.586 | 0.659 | 1044 |


### leftover after runway (Y3)

Y3 leftover after contemporaneous `b_runway` live=True. If they die, the leftover 44-cols are runway shadows.

| col | after b_runway | honest | after last-value still | dies |
| --- | --- | --- | --- | --- |
| b_bal_vol | 0.594 | DIES (fake) | 0.838 | YES |
| b_below_0 | 0.571 | DIES (fake) | 0.717 | YES |
| b_d_runway | 0.656 | 0.656 | 0.811 | no |
| b_neg_episodes | 0.597 | DIES (fake) | 0.768 | YES |


### rank leftover + B-path leak

Rank leftover `b_d_runway`=0.671 ρ(resid,b_liq)=0.133 path_leak=False. `b_runway` rank leftover=0.537.

| col | OLS leftover | rank leftover | ρ(resid,days) | Pearson(resid,days) | ρ(resid,b_liq) | ρ(resid,b_runway) | B-path leak | near-fake days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b_bal_vol | 0.734 | 0.631 | 0.978 | -0.000 | 0.205 | -0.294 | no | YES |
| b_below_0 | 0.649 | 0.649 | -0.819 | 0.000 | -0.422 | 0.055 | YES | YES |
| b_d_runway | 0.748 | 0.671 | 0.287 | -0.000 | 0.133 | 0.141 | no | YES |
| b_neg_episodes | 0.687 | 0.687 | -0.811 | 0.000 | -0.313 | 0.164 | YES | YES |
| b_runway | 0.711 | 0.537 | 0.224 | 0.000 | 0.402 | 0.712 | YES | no |


### `b_d_runway` Q1 crash cell / T1 vs days

d_runway Q1 Y3=17.6% vs body 4.9% crash_cell=True. T1 Δdays=0.038 survive_vs_days=True. after days+runway=0.560. High leftover is the B recovery path (illegal X), not a leftover why.

| item | value | n |
| --- | --- | --- |
| Y3 rate Q1 (most neg Δ) | 17.6% | 843 |
| Y3 rate Q2–Q5 | 4.9% | 3,369 |
| T1 Δ vs days (not size) | 0.038 | pos=214 |
| T1 leftover after days | 0.668 | ρ=0.237 |
| leftover after days+runway | 0.560 | fake=False |
| leftover so_far≥6 after days | 0.748 | rank=0.671 |


### last-value look-ahead (why Q1 ≠ X)

LOOK-AHEAD last-value Y3=0.823 vs month-t 0.596. Last-month Y3 labels=0. last p50=1.079 (CONFIRM 1.079). KEEP as Q1 description, never as X.

| item | value | note |
| --- | --- | --- |
| Y3 last-value still (LOOK-AHEAD) | 0.823 | broadcast extract still onto labeled months |
| Y3 contemporaneous runway | 0.596 | illegal X; still the walk |
| Y2 last-value still (LOOK-AHEAD) | 0.747 | weaker than month-t runway 0.924 |
| Y2 contemporaneous runway | 0.924 | LOCK leak |
| Y3 labels on last month | 0 | horizon 6 — last month has no Y3 |
| last-value runway p50 / share<1 | 1.079 / 48.4% | CONFIRM Family B 1.079 / 49.1% |


### demean raises B AUROC (path shock)

Demean *raises* B AUROC (runway 0.596→0.940, vol 0.692→0.789). That is the month-t cash path — the Y — not a company dummy. a_out_vol demean falls 0.722→0.549 (trait). Different object. No y_bal_vol.

| col | raw | demean | drop | kind |
| --- | --- | --- | --- | --- |
| b_bal_vol | 0.692 | 0.789 | -0.096 | shock |
| b_below_0 | 0.509 | 0.535 | -0.025 | neither |
| b_d_runway | 0.659 | 0.798 | -0.140 | shock |
| b_neg_episodes | 0.481 | 0.557 | -0.076 | neither |
| b_runway | 0.596 | 0.940 | -0.344 | shock |
| a_vol | 0.626 | 0.465 | 0.162 | trait |
| a_out_vol | 0.722 | 0.549 | 0.172 | trait |
| c_n_days_with_tx | 0.711 | 0.502 | 0.209 | trait |


### Y2 leftover on clean-now

Clean-now Y2 base=1.5%. Onset leftover is still the B path (runway 0.654, episodes 0.731). Dies as Y definition, not as a leftover X.

| col | clean-now CV | n_pos | after days | honest | ρ(resid,days) |
| --- | --- | --- | --- | --- | --- |
| b_runway | 0.654 | 195 | 0.420 | 0.420 | 0.212 |
| b_bal_vol | 0.618 | 143 | 0.679 | DIES (fake) | 0.982 |
| b_d_runway | 0.574 | 143 | 0.639 | 0.639 | 0.279 |
| b_neg_episodes | 0.731 | 143 | 0.644 | DIES (fake) | -0.897 |


### leftover vol after Javier honesty

Store vol leftover after Javier=0.621 ρ(resid,a_vol)=-0.981 fake=True. Not a KEEP — different object, still B path / fake residual.

| item | value | note |
| --- | --- | --- |
| OLS leftover after a_vol | 0.621 | fake=True rank=0.677 |
| ρ(resid, a_vol) | -0.981 | Pearson=0.000 |
| leftover after days | 0.734 | ρ(resid,days)=0.978 |
| leftover after a_out_vol | 0.719 | ρ=0.965 |


### Y2 leftover ρ(resid, below_0)

Y2 leftover after the sign bit. Runway leftover is remaining thinness (Y2=2-of-3 future neg). Lock, not KEEP.

| col | after sign | ρ(resid,below_0) | rank leftover | note |
| --- | --- | --- | --- | --- |
| b_bal_vol | 0.864 | 0.433 | 0.460 | DIES fake |
| b_below_0 | self | — | — | Y definition |
| b_d_runway | 0.697 | 0.196 | 0.521 | lives — still B path |
| b_neg_episodes | 0.596 | -0.054 | 0.597 | lives — still B path |
| b_runway | 0.618 | 0.181 | 0.541 | lives — still B path |


### `b_d_runway` leftover kill

d_runway after days+runway=0.560 rank=0.653 dies_adjacent=False. T1 fold-min=0.561 invert=False. T1 leftover after days+runway=0.601.

| item | value | note |
| --- | --- | --- |
| after days+runway OLS | 0.560 | rank=0.653 fake=False |
| after b_liq | 0.657 | rank=0.660 |
| T1 raw / fold-min | 0.641 / 0.561 | folds=0.659 0.681 0.700 0.601 0.561 invert=False |
| T1 leftover after days | 0.668 | rank=0.655 ρ=0.237 |
| T1 leftover after days+runway | 0.601 | rank=0.645 |


| slice | leftover days | rank | ρ(resid,days) | n_pos | near-fake |
| --- | --- | --- | --- | --- | --- |
| so_far 6-11 | 0.763 | 0.705 | 0.252 | 163 | YES |
| so_far ≥12 | 0.727 | 0.655 | 0.309 | 150 | YES |
| long_24 | 0.746 | 0.651 | 0.275 | 201 | YES |


### `b_d_runway` Q1 dummy / drop-Q1

Q1 dummy=0.649 vs continuous 0.659 almost_flag=True. Body leftover after days=0.527 (if this dies, leftover WAS the crash cell). Q1 owns 47.3% of recoveries.

| item | value | note |
| --- | --- | --- |
| continuous d_runway Y3 | 0.659 | 0.639 0.718 0.713 0.618 0.604 |
| Q1 dummy Y3 | 0.649 | almost_flag=True |
| body Q2–Q5 leftover after days | 0.527 | raw=0.487 rank=0.452 |
| Q1 share of Y3 recoveries | 47.3% | 148/313 across 370 companies |


## What this is not

- Not a 0–100. Not pillars. Not `product/`.
- Not `python -m analysis.targets.build_targets`.
- Not a parquet merge. Not Family I/M/J merge.
- Not B on the 15-col Y3 card.
- Not B as Y2/Y3 X (lock).
- Not a holdout AUROC trophy.
- Did not rewrite `liquidity.py` / the walk.

Elapsed 12.5s. PNG: `b_on_44_leftover.png` (True).
