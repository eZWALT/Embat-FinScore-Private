# Wave 4 — a_vol (in-memory)

- **When:** 2026-09-19T04:17:23+02:00
- **Agent:** `6b456387`
- **Files:** `analysis/evaluate/a_vol_qa.py`, `analysis/outputs/a_vol_qa.md`, `analysis/outputs/a_vol_vs_b_bal_vol.png`
- **Columns (in memory only):** `a_vol`, `a_in_vol`, `a_io_vol`, `a_io_sd6`, `a_vol_sumdenom`, `a_out_vol`, `a_vol_w3`, `a_vol_w12`. Not merged.
- **Train coverage:** a_vol 71.3% (1214 train companies).
- **Identity:** vs Javier `volatility` ρ=1.000 **SAME**.
- **DRIFT:** vs `b_bal_vol` ρ=0.354 **DRIFT**.
- **Y3 / Y2 singles:** 0.626 / 0.439 vs size 0.617 / 0.551; vs days 0.711 / night 0.540.
- **Decision:** X **CLOSE** (`a_vol` Javier twin), Y **PARK**. Merge `a_vol`: NO — Y3 loses to size (Δ=+0.009); Y2 loses to size (Δ=-0.112). a_out_vol later-store KEEP=False (trait dummy, not month shock).
- **Small-book tail:** T2+T3 a_vol 0.532 vs size 0.559 (Δ -0.027); T1 0.636 vs size 0.664 (Δ -0.028).
- **In-memory footnote:** `a_out_vol` Y3 0.722 Δ+0.104 monotone_up; T2+T3 0.726 Δ+0.167 vs days 0.725. Dark quiet-days lift +11.5pp; ERP +8.3pp. Recovery pile 211 hits / 96 cos, top8=17.5% (spread). Within-tercile hi-lo T1/T2/T3 +9.9/+5.0/+8.4pp. Company-mean common vs size +0.071; vs days ρ=-0.453 (not COPY). Demean 0.722→0.549 (trait). Company 2×2 quiet/busy +13.2%/+13.1%. Q5 overlap vs a_vol 51.1%; out-only T2+T3 58.0%. Drop a_vol Q5: a_out_vol 0.720 vs size 0.453 (Δ +0.266). Holdout 72 a_out_vol 66.5%, Y3 pos=14. Y2 a_out_vol 0.601 fold-min 0.443 invert (Y3-only). Not a 15-col lag card. Parent decides; not tonight's card.
- **Confirm 0.722:** reproduced=True CV=0.722 Δsize=+0.104. Leak twin=False SIZE=False drop12 Δ=+0.002 move=False. short=0.742 long=0.724 Q6=False. Month quiet +0.098 (12=False); company quiet +0.104 (12=False). Demean 0.722→0.549 η²=0.741 trait=True shock=False later_keep=False. a_out_vol X **CLOSE** Y **PARK** merge **NO** — company-style dummy (demean kills; high ICC); like uncat ICC — PARK as health Y
- **+13pp / ICC:** company 2×2 lift +0.132 (quote+13=True); η² company=0.741 group=0.282.
- **Stability / transfer:** leave-one-fold min mean=0.706 still≥0.70=True; fold signs all_same=True; company-OOF 0.682 transfers=True; night quote 0.762/0.752 files ok=True. company-OOF days=0.666 size=0.631; Pearson twin=False. 12 names Y3 pos=0. drop Q5-cos Δ=+0.066 move=True. max leak -0.411 vs a_n_tx. Q5 leftover vs size Δ=+0.070; Q5 groups=71 ∩12=0. Q1–Q4 company-OOF=0.604 step=False; drop Q4+Q5 CV=0.675. body demean 0.655→0.648 η²=0.590 trait=False. Q5-only CV=0.540 inside=False. Q5 dummy=0.647 vs cont 0.722 dummy-shaped=False. quote-file ages ok=True; Spearman/Pearson twin=False. holdout ≥train-Q5=21/71.
- **Failed / next:** KEEP gate missed — do not merge a_vol tonight; a_io_vol Y2 +0.02 is not the 12 names (footnote only; Y3 loses); a_in_vol Y3 Δ=+0.017 misses KEEP by 0.003; a_out_vol X=CLOSE merge=NO — company-style dummy (demean kills; high ICC); like uncat ICC — PARK as health Y. Do not merge `a_vol`. Do not put `a_out_vol` on the 15-col card tonight.
