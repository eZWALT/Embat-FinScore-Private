# Pregunta send becomes stop

- Author: agent
- Timestamp: 2026-09-20 00:50 +02:00
- Not still-binding.

## What changed

While a reply is in flight, Enviar becomes Detener. Stop aborts the stream (`request.signal` into `streamText`) and drops the unfinished assistant message. The user turn stays.
