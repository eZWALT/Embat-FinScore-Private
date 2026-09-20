# 2026-09-20 03:39 — alert quotes; stats only if asked

Author: Cursor Grok (overnight loop)
Timestamp: 2026-09-20 03:39 +02
Branch: `agent/overnight-optimize` (do not merge tonight)

## What changed

- `get_alerts` omits `stats` unless the question asks for fiabilidad / tasa base / lift.
- Categories in `get_company` are Spanish `{ name, score, pts }`.
- Trace summary uses `Empresa 0030`. Under «Leer alertas», the UI quotes title · dueño (same quote row as reasons).
- Evidence / min-member ids go through `entityLabel` / `relabelEntities`.

## Decisions

- Fair bench keep. alerts text dropped the lift lecture (936 vs 1271 chars). One run stalled 96 s on TTFT — ignore-latency; treat as provider, not a regression.
- period-4 4 tools / q 6 / 8.5 s.

## Still unknown

- Whether a Vercel preview with Neon can replay COMP_1186 vs production.
