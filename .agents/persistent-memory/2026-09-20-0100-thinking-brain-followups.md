# Brain toggle + follow-ups after tools

- Author: agent
- Timestamp: 2026-09-20 01:00 +02:00
- Still-binding: Pregunta chips start only after the answer finishes; transcript includes tool in/out. Brain on `/api/ask` sends DeepSeek `thinking: { type: enabled }` + `thinking.md`.

## What changed

Follow-ups used to fire when the assistant bubble first appeared and only saw text. They now wait until the stream is done and include the tools already run. The brain left of Enviar toggles DeepSeek thinking (`thinking` + `reasoning_effort: high`); off sends `disabled` (V4 defaults on). `thinking.md` replaces `brevity.md` for that turn.

## Still unknown

Whether Helmcode forwards `thinking` unchanged from the official DeepSeek body.
