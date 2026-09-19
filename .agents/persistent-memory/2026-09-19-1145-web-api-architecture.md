# 2026-09-19 11:45 — arquitectura de datos para la v1 web

- **Author:** Codex
- **When:** 2026-09-19 11:45 CEST

## Decision

La v1 de `product/web/` usara Next.js App Router y leera el bundle local existente de `product/score/` desde Server Components. La frontera queda:

`Server Component -> dashboard service -> repository interface -> local bundle adapter`

No se creara un Route Handler para que el propio servidor se llame por HTTP. Los Route Handlers quedan reservados para consumidores de navegador o integraciones externas que realmente necesiten una API. Cuando llegue Supabase, se sustituira el adaptador local por uno de Supabase sin cambiar el servicio ni los componentes visuales.

## V1 scope

- Interfaz limpia basada en shadcn/ui.
- Dos visualizaciones: linea temporal del score y barras por categoria para el ultimo mes.
- Datos derivados del dataset local mediante el contrato `product/score/DATA_CONTRACT.md`.
- Estados vacio/error y tipos compartidos en la frontera de datos.

## Still unknown

- Si el primer despliegue llevara el bundle completo dentro del build o usara `BUNDLE_URL`.
- Que consultas interactivas necesitaran Route Handlers cuando se conecte Supabase.
