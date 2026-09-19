# 2026-09-19 15:35 — Health Score dumb chat

Author: Ruben (Cursor)

## What changed

- Chat tonto en la pestaña Índice de salud: UI shadcn `Message` + `Bubble` + `Input`.
- `POST /api/chat` hace `streamText` con modelo `openai/gpt-5.4` (AI Gateway, OIDC). Sin tools, sin Helmcode.
- Ask / Watcher siguen en Helmcode (`/api/ask`, `/api/watcher/reply`).

## Decisions

- No AI Elements ni `MessageScroller`.
- Auth: OIDC (`vercel env pull`), no `AI_GATEWAY_API_KEY`.
- El chat no recibe empresas seleccionadas ni datos de Neon.

## Still unknown / blocker

- Local 2026-09-19: Gateway 403 `customer_verification_required` — el equipo Vercel necesita tarjeta para desbloquear créditos. El código llega al Gateway; sin eso no hay inferencia.
