import type { ModelMessage, UIMessage } from "ai";

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

export function sessionExtra(companyId?: string, groupId?: string, asOf?: string): string | undefined {
  const lines: string[] = [];
  if (companyId) lines.push(`company_id=${companyId}`);
  if (groupId) lines.push(`group_id=${groupId}`);
  if (asOf) lines.push(`as_of=${asOf}`);
  return lines.length ? `session\n${lines.join("\n")}` : undefined;
}

export type { ModelMessage, UIMessage };
