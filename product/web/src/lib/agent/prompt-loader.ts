import { readFile } from "node:fs/promises";
import path from "node:path";

export type AgentRole = "sentinel" | "chat" | "quick";

/** Ask stack (full context, do not drop a layer): MAP → ROLE → SCOPE → PRODUCT → WORDING → FORMAT → TOOLS → RECORDS → SESSION. */
const FILES = {
  map: "prompt_map.md",
  sentinel: "sentinel_system.md",
  chat: "chat_system.md",
  scope: "scope.md",
  quick: "quick_system.md",
  product: "product_context.md",
  wording: "wording_rules.md",
  format: "watcher_format.md",
  tools: "tools_catalog.md",
  plots: "plots_catalog.md",
  schema: "clean_schema.md",
  brevity: "brevity.md",
} as const;

function promptsDir() {
  return path.join(process.cwd(), "src/lib/agent/prompts");
}

async function readPrompt(name: string) {
  return readFile(path.join(promptsDir(), name), "utf8");
}

export async function loadSystemPrompt(role: AgentRole, extra?: string, options?: { thinking?: boolean }) {
  const parts = [
    await readPrompt(FILES.map),
    await readPrompt(FILES[role]),
    await readPrompt(FILES.scope),
    await readPrompt(FILES.product),
    await readPrompt(FILES.wording),
    await readPrompt(FILES.format),
    await readPrompt(FILES.tools),
  ];
  if (role === "chat" || role === "sentinel") parts.push(await readPrompt(FILES.plots));
  if (role === "chat") parts.push(await readPrompt(FILES.schema));
  if (extra) parts.push(extra);
  // Last on purpose (needle): length rules after the long stack so they are not forgotten.
  if (role === "chat") {
    parts.push(await readPrompt(options?.thinking ? "thinking.md" : FILES.brevity));
  }
  return parts.join("\n\n---\n\n");
}
