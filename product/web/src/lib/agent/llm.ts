import { createOpenAI } from "@ai-sdk/openai";
import { streamText } from "ai";

const DEFAULT_BASE_URL = "https://api.helmcode.com/v1";
const DEFAULT_MODEL = "deepseek-v4-flash";

export function helmcodeApiKey(): string | undefined {
  const key = process.env.HELMCODE_API_KEY?.trim();
  return key || undefined;
}

export function helmcodeModelId(): string {
  return process.env.POC_LLM_MODEL?.trim() || DEFAULT_MODEL;
}

/** Sampling for Ask / Watcher / follow-ups. Override with HELMCODE_TEMPERATURE while we QA. */
export function helmcodeTemperature(): number {
  const raw = process.env.HELMCODE_TEMPERATURE?.trim();
  const value = raw === undefined || raw === "" ? 0.2 : Number(raw);
  if (!Number.isFinite(value)) return 0.2;
  return Math.min(1, Math.max(0, value));
}

/**
 * Helmcode is OpenAI-compatible chat completions (`POST /v1/chat/completions`, SSE).
 * In `@ai-sdk/openai` v4, `createOpenAI()(modelId)` uses the Responses API (`/responses`).
 * There is no provider `stream` flag: `streamText` is what sends `stream: true`.
 */
export function createHelmcodeModel() {
  const apiKey = helmcodeApiKey();
  if (!apiKey) {
    throw new Error("HELMCODE_API_KEY is not set");
  }

  const helmcode = createOpenAI({
    name: "helmcode",
    baseURL: process.env.HELMCODE_BASE_URL?.trim() || DEFAULT_BASE_URL,
    apiKey,
  });

  return helmcode.chat(helmcodeModelId());
}

/** One-token streamed ping so the first Pregunta turn is not a cold Helmcode start. */
export async function warmupHelmcode(): Promise<void> {
  const result = streamText({
    model: createHelmcodeModel(),
    prompt: "ok",
    maxOutputTokens: 1,
  });
  await result.consumeStream();
}
