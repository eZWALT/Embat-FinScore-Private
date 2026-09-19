import { LocalBundleRepository } from "./local-bundle-repository";
import { NeonScoreRepository } from "./neon-repository";
import type { ScoreRepository } from "./types";

export function createScoreRepository(): ScoreRepository {
  if (process.env.DATABASE_URL) return new NeonScoreRepository();
  if (process.env.VERCEL_ENV === "production") {
    throw new Error(
      "DATABASE_URL is required. Local: copy product/web/.env.example to .env.local. Production: set DATABASE_URL on the Vercel project (server-only, not NEXT_PUBLIC_).",
    );
  }
  return new LocalBundleRepository();
}
