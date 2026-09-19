# QA pass: sample bundle locally + leftover chrome

- Author: agent
- Timestamp: 2026-09-19 20:25 +02:00
- Still-binding for local web without Neon.

## What changed

`createScoreRepository()` uses `LocalBundleRepository` when `DATABASE_URL` is missing, except on Vercel production. Watcher feed no longer 503s without Neon. Walked Rápido / Profundo / Grupos / Pregunta on the 12-company sample.

Fixes from that walk: principal señal uses Javi money + company currency; Rápido → Profundo keeps the picked companies; Vigilancia shows a pulse (no empty lie); `COUNTERPARTY_*` → `cliente N`; watcher hides `+0`; sidebar rail is not a second “ocultar el menú”.

## Still unknown

Pregunta live replies still need `HELMCODE_API_KEY`. Production still needs `DATABASE_URL`.
