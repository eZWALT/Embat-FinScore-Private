# Ask uses the score repository, not a Neon-only 503

- Author: agent
- Timestamp: 2026-09-19 21:08 +02:00
- Still-binding for Pregunta locally.

## What changed

`streamAgentResponse` only requires `HELMCODE_API_KEY`. Score tools go through `createScoreRepository()`: Neon when `DATABASE_URL` is set, sample bundle otherwise (except Vercel production).

## Decisions

Do not 503 Pregunta just because this process has no Neon. Production already has `DATABASE_URL`.

## Still unknown

Record-level `query_clean_db` still needs Neon.
