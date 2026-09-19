import { createOpenAI } from "@ai-sdk/openai";

const DEFAULT_BASE_URL = "https://api.helmcode.com/v1";
const DEFAULT_MODEL = "deepseek-v4-flash";

export function helmcodeApiKey(): string | undefined {
  const key = process.env.HELMCODE_API_KEY?.trim();
  return key || undefined;
}

export function helmcodeModelId(): string {
  return process.env.POC_LLM_MODEL?.trim() || DEFAULT_MODEL;
}

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

  return helmcode(helmcodeModelId());
}
