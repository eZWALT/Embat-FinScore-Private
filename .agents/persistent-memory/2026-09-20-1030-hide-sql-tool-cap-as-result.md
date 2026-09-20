# Pregunta: hide `query_clean_db` when core is empty; caps answered by the tool

Author: agent (Walter's session) · 2026-09-20 10:30

## What was wrong

Two error chips showed up in production after the `plot_from` commit; neither came from it.

- `records not mounted` ×2: Neon `core` was never loaded (`infra/neon/README.md`), but the tool was still offered, so the model spent two calls and showed two errors.
- `Explicar el cambio ×1 Error`: per-tool cap stripped the tool from `activeTools`; the model called it anyway and the SDK reported an unavailable tool.

## Changes

- `tools.ts`: `chatTools({ records })` omits `query_clean_db` when `coreMounted()` is false (cached 5 min per process). `withCap` wrapper: over the cap the tool returns `{error: "límite de N llamadas a <tool> en esta respuesta; responde con lo que ya tienes"}` instead of executing. Sits inside memoize, so identical repeats do not count.
- `runtime.ts`: `prepareStep` only forces the text step on the overall budget; no per-tool stripping.
- `tools_catalog.md`: if `query_clean_db` is not in the tool list, say records are not loaded and do not call it.

Verified on production: same question, no SQL calls, no error chips, model states records are not loaded and writes with the two `explain_change` results it got.

## Still unknown

- Whether to load `core` into Neon at all before the deadline (storage limit on Free); the app now degrades cleanly without it.
