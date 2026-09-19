import { getToolName, isTextUIPart, isToolUIPart, type UIMessage } from "ai";

export type AgentContext = {
  companyId?: string;
  groupId?: string;
  asOf?: string;
};

/** Fields the Ask / Watcher reply routes expect besides `messages`. */
export function contextBody(context: AgentContext): Record<string, string> {
  const body: Record<string, string> = {};
  if (context.companyId) body.companyId = context.companyId;
  if (context.groupId) body.groupId = context.groupId;
  if (context.asOf) body.asOf = context.asOf;
  return body;
}

export function textFromParts(parts: UIMessage["parts"]): string {
  return parts
    .filter(isTextUIPart)
    .map((part) => part.text)
    .join("");
}

/** One chip per tool call, in stream order. Same tool twice → two chips. */
export function toolChipsFromParts(parts: UIMessage["parts"]): { id: string; name: string }[] {
  const chips: { id: string; name: string }[] = [];
  const seen = new Set<string>();
  for (const part of parts) {
    if (!isToolUIPart(part)) continue;
    const name = getToolName(part);
    const id = "toolCallId" in part && part.toolCallId ? String(part.toolCallId) : name;
    if (seen.has(id)) continue;
    seen.add(id);
    chips.push({ id, name });
  }
  return chips;
}
