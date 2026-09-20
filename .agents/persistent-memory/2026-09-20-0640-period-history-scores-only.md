# 2026-09-20 06:40 — Period history is scores only

Author: cursor-grok-4.6 (overnight Pregunta)
Timestamp: 2026-09-20 06:40 +02

## What changed

On a period `get_company`, `score_history` is month / score / trayectoria / tope only — reasons stay on the top-level payload. `grupo` is omitted so bullets do not recite «(Grupo 0234)». WORDING 11 is one popup line (BREVITY still last). Bench period drag copy matches live `buildPrompt` (`Caída puntual`, not English `fading`).

## Decisions

- Keep vs main sample suite: 1/6, 1/6, 0/5, 4/6, 1/6. Period 1078 chars, lead still tope / mora / estable.
- Do not merge to main tonight.

## Still unknown

Period still adds a closing confidence paragraph after the four bullets. Next cut: drop that extra paragraph, or split PRODUCT validation so why-score does not carry AUROC.
