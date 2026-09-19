import {
  convertToModelMessages,
  createUIMessageStreamResponse,
  isStepCount,
  streamText,
  toUIMessageStream,
} from "ai";

import { createHelmcodeModel, helmcodeApiKey } from "./llm";
import { coerceUiMessages, sessionExtra } from "./messages";
import { loadSystemPrompt, type AgentRole } from "./prompt-loader";
import { chatTools, sentinelTools } from "./tools";

export async function streamAgentResponse({
  role,
  messages,
  companyId,
  groupId,
  asOf,
}: {
  role: AgentRole;
  messages: unknown;
  companyId?: string;
  groupId?: string;
  asOf?: string;
}): Promise<Response> {
  if (!helmcodeApiKey()) {
    return Response.json({ error: "HELMCODE_API_KEY is not set" }, { status: 503 });
  }
  if (!process.env.DATABASE_URL) {
    return Response.json({ error: "DATABASE_URL is not set" }, { status: 503 });
  }

  const uiMessages = coerceUiMessages(messages);
  if (!uiMessages.length) {
    return Response.json({ error: "messages is required" }, { status: 400 });
  }

  const system = await loadSystemPrompt(role, sessionExtra(companyId, groupId, asOf));
  const modelMessages = await convertToModelMessages(uiMessages);
  const tools = role === "sentinel" ? sentinelTools() : chatTools();

  const result = streamText({
    model: createHelmcodeModel(),
    system,
    messages: modelMessages,
    tools,
    stopWhen: isStepCount(8),
    temperature: 0.2,
  });

  return createUIMessageStreamResponse({
    stream: toUIMessageStream({ stream: result.stream, tools }),
  });
}
