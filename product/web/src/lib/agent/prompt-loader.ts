import { readFile } from "node:fs/promises";
import path from "node:path";

export type AgentRole = "sentinel" | "chat";

const FILES = {
  sentinel: "sentinel_system.md",
  chat: "chat_system.md",
  scope: "scope.md",
  product: "product_context.md",
  wording: "wording_rules.md",
  format: "watcher_format.md",
  tools: "tools_catalog.md",
  schema: "clean_schema.md",
} as const;

function promptsDir() {
  return path.join(process.cwd(), "src/lib/agent/prompts");
}

async function readPrompt(name: string) {
  return readFile(path.join(promptsDir(), name), "utf8");
}

export async function loadSystemPrompt(role: AgentRole, extra?: string) {
  const parts = [
    await readPrompt(FILES[role]),
    await readPrompt(FILES.scope),
    await readPrompt(FILES.product),
    await readPrompt(FILES.wording),
    await readPrompt(FILES.format),
    await readPrompt(FILES.tools),
  ];
  if (role === "chat") parts.push(await readPrompt(FILES.schema));
  if (extra) parts.push(extra);
  return parts.join("\n\n---\n\n");
}
