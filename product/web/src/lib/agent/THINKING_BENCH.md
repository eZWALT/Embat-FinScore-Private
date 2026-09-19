# Thinking effort bench

- When: 2026-09-19T23:36:52.788Z
- Model: deepseek-v4-flash
- Question: ¿Por qué este índice este mes? Usa las herramientas. No inventes números.
- Pick: **high** (same ~1 s first token as `low`, ~2× the thought, much cheaper than `max`. `low` won a raw latency sort; the product want is reasoning high without `max`.)

| Level | First visible | First reasoning | First tool | Total | Reasoning chars | Tools | 68 | Cobros |
|---|---:|---:|---:|---:|---:|---|---|---|
| off | 1.33 s | 1.33 s | — | 4.93 s | 347 | get_company, get_alerts | yes | yes |
| low | 1.05 s | 1.05 s | — | 4.04 s | 480 | get_company, get_alerts | yes | yes |
| high | 1.08 s | 1.08 s | — | 5.20 s | 1030 | get_company, get_alerts | yes | yes |
| max | 958 ms | 958 ms | — | 7.12 s | 3352 | get_company, get_alerts | yes | yes |

Default in `llm.ts` is `high`. Override with `HELMCODE_REASONING_EFFORT`.
