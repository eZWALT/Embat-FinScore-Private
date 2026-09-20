# 2026-09-20 03:34 — get_alerts on demand; Spanish tool tokens

Author: Cursor Grok (overnight loop)
Timestamp: 2026-09-20 03:34 +02
Branch: `agent/overnight-optimize` (do not merge tonight)

## What changed

- `get_alerts` is mounted only when the user asks for alertas / dueño / quién actúa (`wantsAlertTools`). Same pattern as records and plots.
- `periodo seleccionado` no longer loads the plots catalog. A dragged period is an explanation, not a chart request.
- Tool payloads speak Spanish: guard / trajectory / confidence / owner / severity. `slimReason` drops `item` (the `sentence` is the phrase).
- WORDING / BREVITY / ROLE: never write `fading`, `dark`, `out_vol`, `fc_ratio`, `score_pre_cap`.

## Decisions

- Fair bench `--suite sample --ignore-latency`. Keep.
- period-4: **4 `get_company` / q 6 / 11.8 s** (main 11 / q 2). Tick 5 had an extra scoped `get_alerts`.
- why-score / why-change no longer echo item codes.

## Still unknown

- period-4 once opened with «Llamo a las cuatro empresas». Next: forbid announcing calls.
- Alerts answers still close with a method recitation. WORDING already says not to.
