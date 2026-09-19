import { helmcodeApiKey, warmupHelmcode } from "@/lib/agent/llm";

export const runtime = "nodejs";
export const maxDuration = 15;
export const dynamic = "force-dynamic";

/**
 * Fire-and-forget ping for the Pregunta popup. Helmcode only — no Neon, no prompts.
 */
export async function POST() {
  if (!helmcodeApiKey()) {
    return new Response(null, { status: 204 });
  }

  try {
    await warmupHelmcode();
  } catch (error) {
    const text = error instanceof Error ? error.message : String(error);
    console.warn(`[agent] warmup failed: ${text}`);
  }

  return new Response(null, { status: 204 });
}
