import type { ModelMessage, UIMessage } from "ai";

import { formatDashboardView, type DashboardView } from "./view-context";

type LooseMessage = {
  id?: string;
  role?: string;
  content?: unknown;
  parts?: UIMessage["parts"];
};

function textFromContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((block) =>
        typeof block === "object" && block && "text" in block ? String((block as { text: unknown }).text) : "",
      )
      .join("");
  }
  return "";
}

/** Accept useChat UIMessage[] or a curl-friendly `{ role, content }[]`. */
export function coerceUiMessages(input: unknown): UIMessage[] {
  if (!Array.isArray(input)) return [];
  return input.map((raw, index) => {
    const message = (raw ?? {}) as LooseMessage;
    const role = message.role === "assistant" ? "assistant" : message.role === "system" ? "system" : "user";
    if (Array.isArray(message.parts)) {
      return {
        id: message.id ?? `msg-${index}`,
        role,
        parts: message.parts,
      };
    }
    return {
      id: message.id ?? `msg-${index}`,
      role,
      parts: [{ type: "text", text: textFromContent(message.content) }],
    };
  });
}

export function sessionExtra(input: {
  companyId?: string;
  groupId?: string;
  asOf?: string;
  view?: DashboardView;
}): string | undefined {
  const lines: string[] = [];
  if (input.companyId) lines.push(`company_id=${input.companyId}`);
  if (input.groupId) lines.push(`group_id=${input.groupId}`);
  if (input.asOf) lines.push(`as_of=${input.asOf}`);
  const ids = lines.length ? `session\n${lines.join("\n")}` : undefined;
  const screen = input.view ? formatDashboardView(input.view) : undefined;
  if (ids && screen) return `${ids}\n\n${screen}`;
  return screen ?? ids;
}

export type { ModelMessage, UIMessage };
