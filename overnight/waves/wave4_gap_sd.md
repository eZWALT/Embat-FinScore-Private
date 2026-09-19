# Wave 4 — c_gap_sd

- agent `87e59905` files: `analysis/evaluate/gap_sd_qa.py`, `analysis/outputs/gap_sd_qa.md`, `gap_sd_vs_days.png`
- columns: store `c_gap_sd` (Family C). No new column. No ops.py edit.
- train coverage: 95.1% (CONFIRM 95.1%).
- ρ vs days -0.905 / a_n_tx -0.866 / c_n_tx -0.866; vs log1p(a_in3) -0.532.
- Y3 CV gap 0.686 vs size 0.617 vs days 0.711. Leftover after days 0.535. ICC 0.963.
- verdict: **DROP-from-44** as Y3 X (a). PARK as Y. Q6 CLOSE. Q5 CLOSE.
- what failed: leftover after days 0.535 dies; rank leftover 0.523; Q6 lag leftover 0.483; days leftover after gap 0.658 lives — Y3 engine keeps days. Feature report picked the weaker twin as representative.; unique-day σ vs 1/c_n_tx still 0.867 — collapse cut raw 0.916 but did not escape the twin. Do not rewrite ops.py.
- next: parent decides the 44; do not put gap_sd on the 15-col card.
- elapsed 8s.
