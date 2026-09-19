import { NeonScoreRepository } from "./neon-repository";
import type { ScoreRepository } from "./types";

export function createScoreRepository(): ScoreRepository {
  if (!process.env.DATABASE_URL) {
    throw new Error(
      "DATABASE_URL is required. Local: copy product/web/.env.example to .env.local. Production: set DATABASE_URL on the Vercel project (server-only, not NEXT_PUBLIC_).",
    );
  }

  return new NeonScoreRepository();
}
