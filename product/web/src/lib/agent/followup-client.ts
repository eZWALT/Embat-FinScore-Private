import type { UIMessage } from "ai";

import { contextBody, type AgentContext } from "./chat-parts";
import { extractFollowups } from "./suggestions";

function transcript(messages: UIMessage[]): { role: string; content: string }[] {
  return messages.slice(-6).map((message) => ({
    role: message.role,
    content: message.parts
      .map((part) => ("text" in part && typeof part.text === "string" ? part.text : ""))
      .join("")
      .trim(),
  }));
}

/** Start after the main answer so the chips sit above the composer. */
export async function streamFollowupChips(input: {
  messages: UIMessage[];
  context: AgentContext;
  signal?: AbortSignal;
  onChip?: (chips: string[]) => void;
}): Promise<string[]> {
  const response = await fetch("/api/ask/followups", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: transcript(input.messages),
      ...contextBody(input.context),
    }),
    signal: input.signal,
  });

  if (response.status === 204 || !response.body) return [];
  if (!response.ok) return [];

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let full = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    full += decoder.decode(value, { stream: true });
    const chips = extractFollowups(full);
    if (chips.length) input.onChip?.(chips);
    if (chips.length >= 2) break;
  }

  try {
    await reader.cancel();
  } catch {
    /* already closed */
  }

  return extractFollowups(full).slice(0, 2);
}
