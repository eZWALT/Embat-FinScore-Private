# Health Sentinel web

Next.js App Router dashboard for the local Health Score bundle. The first visible version uses shadcn/ui and two charts: score history and the latest score by category.

```bash
pnpm install
pnpm dev
```

The server reads `../score/sample_bundle` by default. Point it at another generated bundle with `SCORE_BUNDLE_DIR=/absolute/path/to/bundle`.

## Data boundary

```text
Server Component -> dashboard service -> ScoreRepository -> LocalBundleRepository
```

The browser receives a small serializable view model. A future Supabase integration should implement `ScoreRepository`; it does not require changing the dashboard components. Route Handlers are intentionally omitted until a browser or external consumer needs an HTTP API.
