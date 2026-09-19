# Health Sentinel web

Next.js App Router dashboard. Local and production read the score from Neon Postgres on the server (`DATABASE_URL`). The browser never gets the connection string.

```bash
pnpm install
cp .env.example .env.local   # then paste DATABASE_URL
pnpm dev
```

On Vercel, set the same `DATABASE_URL` for Production (and Preview if you want). Do not use `NEXT_PUBLIC_DATABASE_URL`.

## Data boundary

```text
Server Component (dynamic) -> dashboard service -> ScoreRepository -> NeonScoreRepository
```

The browser receives a small serializable view model. `LocalBundleRepository` remains for tests or an explicit constructor override; it is not the default. Route Handlers are omitted until a browser or external consumer needs an HTTP API.
