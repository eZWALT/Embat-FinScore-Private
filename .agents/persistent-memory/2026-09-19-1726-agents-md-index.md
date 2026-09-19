# 2026-09-19-1726 — AGENTS.md reorder + journal index

- **Author:** agent (Cursor / Walter)
- **When:** 2026-09-19 17:26 CEST

## What changed

- `AGENTS.md` reordered: Read first → Pointers (current product first) → Plan rules → Neon storage → Agents layers → Memory. Plan table kept as rules, not a kanban. Dummy-score / “read every dated file” pointers removed.
- New `.agents/README.md`: still-binding list (~18 files) vs ~180 night notes. New still-binding entries should add a line there.
- Root `README.md`: no longer says the repo is empty / scaffold.

## Git (reconciled, not rewritten)

- Only branch: `main` → `origin/main`.
- Already on remote before this commit: `b4d1104` (prompt map + tool-call bench). Design B and the full Ask stack are in `51e7bc8`…`b4d1104`.
- GitHub prints that the repo moved to `rubengpr/Embat-FinScore-Private`; fetch/push still work via the `eZWALT` remote (redirect). Remotes were not edited.
- No rebase, no force-push. `d58168c` (“checkpoint before checking out main”) stays in history.

## Decisions

- Do not collapse the 180 night files. Index them.
- Do not put “step 1–5 done” in `AGENTS.md`.

## Still unknown

- Whether Vercel is pointed at this `main` (Root Directory `product/web`).
- Licence / CSV-in-history before the public flip (`2026-09-19-1410-single-repo.md`).
