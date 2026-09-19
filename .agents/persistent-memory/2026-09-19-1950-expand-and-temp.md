# Plot/chat expand + temperature sweep

- Author: agent
- Timestamp: 2026-09-19 19:50 +02:00

Plots (chat, Rápido, Índice, Resumen, group panel) have a fullscreen control (`PlotExpand`, z-80). Pregunta can grow (`Maximize2`) or dock (`ChevronDown`); the thread stays mounted.

`helmcodeTemperature()` (default 0.2, override `HELMCODE_TEMPERATURE`) is shared by Ask, Watcher, follow-ups, `/api/chat`. Sweep on `deepseek-v4-flash`, same facts, 3 draws: T=0 / 0.1 / 0.2 / 0.3 were exact copies *within* T. Style differs *across* T (0 repeats the tesorero line on every bullet; 0.2 is hallazgo/€/dueño once). Not still-binding until Walter picks.
