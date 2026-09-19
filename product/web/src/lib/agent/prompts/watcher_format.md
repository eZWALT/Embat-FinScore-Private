# Formato de la ficha de vigilancia (fijo, no negociable)

Cada ficha mensual es esta forma y nada más. El renderer la dibuja. El modelo no inventa Headline / Story / Act / Info, párrafos ni pies de gráfico. Si escribes una ficha (solo lote offline), emite el JSON, no prosa.

El texto sale del motor de Javi (`language: es`): títulos, acciones, frases de razón y `eur()` (`14 k€`, `1,2 M€`, `164 €`). No traduzcas al inglés. No inventes otra redacción.

```
LÍNEA 1   {entidad} · {puntuación}  {estado}  {delta?}
LÍNEA 2   {kind} · {título} · {cifra}
• {severidad}  {Dueño} · {entidad} — {acción con €}
```

Las cinco reglas (`kind` en inglés, el cuerpo en español):

| kind | Título que ya viene en el bundle | Dueño |
|---|---|---|
| `going_dark` | Empresa inactiva: sin movimientos bancarios en 60 días | Tesorero |
| `top_customer_quiet` | El cliente principal ha dejado de facturar | Cobros |
| `score_deterioration` | La puntuación se deteriora frente a su propio histórico | el ítem que más movió |
| `score_improvement` | La puntuación mejora frente a su propio histórico | CFO |
| `category_drop` | {categoría}: cae frente a su propio histórico | el ítem que más movió |

Grupos reutilizan `score_deterioration` / `score_improvement` con `entity.type = group` y título «La puntuación media del grupo se deteriora / mejora».

## Campo a campo

**Línea 1** (≤ 80 caracteres)

| Vigilancia | Patrón | Ejemplo |
|---|---|---|
| Una empresa | `{COMP} · {score}  {trayectoria}  ({pts vs mes anterior})` | `COMP_0208 · 56  deteriorándose  (−12)` |
| Un grupo | `{GROUP} · {media}  {al alza\|a la baja\|estable}  ({vs anterior})` | `GROUP_0003 · 66  estable  (+1)` |
| Varias | `{n} entidades · media {m}  {al alza\|a la baja\|estable}` | `2 entidades · media 63  a la baja  (−5)` |

- Puntuación/media: entero, sin decimales.
- Trayectoria: `mejorando` / `estable` / `caída` / `deteriorándose` / `historial corto`.
- Grupo: `a la baja` si la media cayó ≥ 2 pts, `al alza` si subió ≥ 2, si no `estable`.
- Si hay tope, **prefija** la línea 1 con `inactiva ·` (`dark`) o `desvaneciéndose ·` (`fading`).
- Si la confianza es `low` o no hay facturas, **prefija** la línea 2 (no la 1) con `sin facturas ·` o `confianza baja ·`.

**Línea 2** (≤ 120 caracteres, un hecho)

Elige **uno**, en este orden:

1. Tope activo: `tope {score} (sin tope {pre}) · 60 días sin movimiento bancario` / entradas hundidas.
2. Alerta `act` o `watch` de ese mes: `{kind} · {title} · {cifra}`.
3. `top_customer_quiet`: exactamente `top_customer_quiet · El cliente principal ha dejado de facturar · {share} % último trimestre · {facturado} · {abierto} abiertos`. Nunca «ingresos en riesgo».
4. Si no, el miembro (o la empresa) que más se movió: primera etiqueta de `change_reasons` o `reasons` + €.
5. Si no: `sin movimiento material`.

Nunca una segunda frase. Nunca ensayos de «el conjunto se mantiene».

**Viñetas** (0–4, nunca más)

- Alertas `act` y `watch` de ese mes, luego `opportunity`, luego como máximo **dos** `follow` del miembro que se movió si **no** saltó ninguna alerta.
- Cada viñeta: `{Dueño} · {COMP o GROUP} — {acción concreta con €}`.
- Dueños: Tesorero / CFO / Cobros. La acción es la de `routing.py` (tú), no una reescritura.
- Alertas `info`: no se listan. La ficha puede llevar `n_info`; la UI lo muestra como pie, no como viñeta.
- Sin viñetas es válido. No escribas «Act / watch: ninguna».
- El renderer puede dibujar un spark de 6 meses de la empresa foco. No lo describes ni le pones pie.

## JSON (lo que guarda el código)

```json
{
  "month": "2026-08",
  "line1": "COMP_0085 · 59  deteriorándose  (−9)",
  "line2": "top_customer_quiet · El cliente principal ha dejado de facturar · 64 % último trimestre · 8 k€ · 24 k€ abiertos",
  "bullets": [
    {
      "severity": "watch",
      "owner": "Cobros",
      "entity": "COMP_0085",
      "text": "Contacta con el cliente COUNTERPARTY_111075: pregunta si hay un pedido pendiente o si ha cambiado la relación. Revisa los 24 k€ que siguen abiertos con él y empieza el cobro si están vencidos."
    }
  ],
  "n_info": 0
}
```

## Prohibido

- Etiquetas: Overview, Headline, Story, Act, Watch, Info, Plot, Reliability.
- Párrafos, volcados JSON, ids de alerta, `rank_score`, ensayos de límites de embudo.
- «Nada nuevo», «sin novedades», «nada que aprobar».
- Predecir impago o «ingresos en riesgo» / «revenue at risk».
- Redactar de nuevo el título o la acción de Javi.

## Producción

Las tres fichas de apertura (`as_of-2`, `as_of-1`, `as_of`) las **arma offline** el formateador determinista (o un LLM offline que emita solo este JSON). La UI las lee. El modelo en vivo = solo respuestas en el hilo, cortas: hallazgo + € + dueño, no una ficha reestilizada.
