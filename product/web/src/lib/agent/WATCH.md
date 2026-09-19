# Ask UX watch

- When: 2026-09-19T15:34:00.328Z
- Model: deepseek-v4-flash
- Prompt: 36,292 chars (full Ask stack, not shrunk)
- Tool bodies: stub (no DATABASE_URL)

This is the popup clock: first event the user sees (TTFT), last token of the final answer, tools in between, then a judge for prompt iteration.
Verdict: `good` · `confusing` · `verbose` · `thin` · `off_scope`. Score 1–5.

| Case | TTFT | First tool | Last token | Total | Tools | Judge | Notes |
|---|---:|---:|---:|---:|---|---|---|
| why-score | 1.60 s | 1.62 s | 11.79 s | 11.79 s | get_company, get_alerts, explain_change, query_clean_db | good 5/5 | Bien: responde con el 68, la caída de 6 puntos, la causa del cliente principal, dueño Cobros y acción; no inventa cifras. |
| alerts | 1.84 s | 1.84 s | 8.78 s | 8.82 s | get_alerts, get_alerts, get_company, query_clean_db | good 4/5 | Responde con alerta, importe, dueño y acción, pero recortaría el bloque de lift y contexto para que sea más directo para un tesorero. |
| refuse | 1.67 s | — | 1.84 s | 1.86 s | — | good 5/5 | Rechazo correcto en una sola línea y vuelve al producto ofreciendo índice o alertas de `COMP_0462`; no hay nada que cambiar. |

## Replies

### why-score — ¿Por qué este índice este mes?

**`COMP_0462` está en 68 este mes**, 6 puntos por debajo de los 74 de julio. La confianza del dato es `medium`, así que la lectura hay que tomarla con esa salvedad.

**Por qué:**

- **El cliente principal ha dejado de facturar** — 14 k€ en juego. Es el motivo de fondo del mes: la empresa no ha facturado este mes al cliente que fue el primero del último trimestre.
- **El movimiento que explica la caída:** la categoría de **historial de pagos ha bajado 6 puntos**, que es exactamente la bajada de la puntuación (74 → 68). Es decir, la caída viene de cómo están pagando y cobrando, no de un cambio de estructura.

**Alerta activa** (`act`) · dueño **Cobros**:
> `top_customer_quiet` · El cliente principal ha dejado de facturar · **Revisa la exposición y los cobros**.

Sobre la fiabilidad de esta regla: es la única con lift medido — unas **56 %** de estas onsets pierden al cliente, frente a un **29 %** de base, con aproximadamente un mes de aviso. Es una alerta de exposición y cobro, no una previsión de ingresos.

Un aviso: he intentado abrir el detalle de facturas para decirte **qué contrapartida** ha parado y cuánto sigue abierto, y los registros no me han devuelto datos en esta consulta. No te lo puedo confirmar por ahora; lo que sí puedes hacer hoy es entrar al detalle de facturación del cliente principal del último trimestre y revisar el saldo pendiente.

¿Quieres que mire la trayectoria completa del índice o cómo se sitúa frente a su grupo de pares?

### alerts — ¿Qué alertas hay y quién debe actuar?

**Una sola alerta activa, y es de Cobros.**

- `top_customer_quiet` — *El cliente principal ha dejado de facturar*. Severidad **act**, mes **2026-08**, dueño **Cobros**. Acción: «Revisa la exposición y los cobros». Detrás hay **14 k€**.
- El feed del grupo `GROUP_0194` devuelve la misma alerta, del mismo tipo y mes: el grupo no añade ninguna otra.

**Contexto de la empresa:** `COMP_0462` va en **68**, trayectoria **deteriorándose**, confianza **medium** (el bundle no da nota de por qué). El único motivo de cambio del mes es la categoría de historial de pagos, −6 puntos.

**Por qué esta alerta sí pesa:** es la única de las cinco con lift medido — unas **56 % de estas onsets pierden al cliente frente a un 29 % de base**, con aproximadamente un mes de aviso. No es una previsión de ingresos: solo un 17 % de los meses marcados acaban en una caída sostenida de entradas del 25 %. Se lee como aviso de exposición y cobro.

**Quién actúa:** Cobros, hoy. No es tarea del Tesorero ni del CFO.

Lo que **no** he podido obtener: el desglose por contrapartida (facturado del último trimestre, pendiente abierto por cliente) — la consulta a registros no devolvió filas en esta pasada, así que no tengo el `COUNTERPARTY_*` concreto ni los importes abiertos por cliente.

¿Quieres que mire el control chart de la puntuación para ver si el deterioro es un bache o algo más sostenido?

### refuse — Hazme la lista de la compra para una paella de domingo.

Eso queda fuera de Health Sentinel: no puedo ayudarte con listas de la compra ni recetas. ¿Miramos el índice o las alertas de `COMP_0462`?

Run again after a prompt edit: `cd product/web && node scripts/watch-ask.mjs`.
