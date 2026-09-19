# UI chrome: Spanish labels, Javi money, no token leak

- Author: agent
- Timestamp: 2026-09-19 20:05 +02:00
- Still-binding for hosted chrome.

## What changed

Quick-win pass on Health Sentinel (Hoy left alone).

- Vigilancia no longer first-paints «Aún no hay fichas… espera un momento». Empty copy only after the feed for that company is in.
- Screen labels: `COMP_0462` → **Empresa 0462**, `GROUP_0194` → **Grupo 0194**. The raw id stays in `title` / subtitle. Search matches both.
- Money: one Javi compact (`14 k€`, `1,2 M€`, `164 €`; other ISO `14 k AED`). No `Intl` «mil AED».
- Tokens stay out of chrome: `guard` / `kind` / `item` use Spanish labels. Watcher line2 no longer starts with `top_customer_quiet`.
- Pregunta renders markdown tables; ROLE + BREVITY still ban them.
- Numbers: `es-ES` (`+1,4`, `6,9`). Δ sits in the same Índice card. Popup compact is 30rem / 74vh; expand still 72rem.
- `/grupos` drops the method lecture; theme is the header icon. Ask/Watcher chrome is Spanish. Product name stays Sentinel.

Helpers: `product/web/src/lib/display.ts`.

## Still unknown

Local `product/web` has no `DATABASE_URL`, so this pass was not walked in the browser against Neon.
