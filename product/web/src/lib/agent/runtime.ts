import { convertToModelMessages, isStepCount, smoothStream, streamText } from "ai";

import { createHelmcodeModel, helmcodeApiKey, helmcodeTemperature } from "./llm";
import { coerceUiMessages, sessionExtra } from "./messages";
import { loadSystemPrompt, type AgentRole } from "./prompt-loader";
import { chatTools, coreMounted, quickTools, sentinelTools, shouldForceTextStep } from "./tools";
import type { DashboardView } from "./view-context";

export async function streamAgentResponse({
  role,
  messages,
  companyId,
  groupId,
  asOf,
  view,
  abortSignal,
  thinking = false,
}: {
  role: AgentRole;
  messages: unknown;
  companyId?: string;
  groupId?: string;
  asOf?: string;
  view?: DashboardView;
  abortSignal?: AbortSignal;
  thinking?: boolean;
}): Promise<Response> {
  if (!helmcodeApiKey()) {
    return Response.json({ error: "HELMCODE_API_KEY is not set" }, { status: 503 });
  }

  const uiMessages = coerceUiMessages(messages);
  if (!uiMessages.length) {
    return Response.json({ error: "messages is required" }, { status: 400 });
  }

  const system = await loadSystemPrompt(role, sessionExtra({ companyId, groupId, asOf, view }), { thinking });
  const modelMessages = await convertToModelMessages(uiMessages);
  const tools =
    role === "sentinel"
      ? sentinelTools()
      : role === "quick"
        ? quickTools()
        : chatTools({ records: await coreMounted() });

  const started = Date.now();
  const result = streamText({
    model: createHelmcodeModel({ thinking }),
    system,
    messages: modelMessages,
    tools,
    // Never stop on tool count: that killed the text step after a parallel burst.
    // Per-tool caps are answered by the tool itself (withCap); only the overall budget forces the text step.
    stopWhen: [isStepCount(8)],
    prepareStep({ steps }) {
      if (shouldForceTextStep(steps)) {
        return { activeTools: [], toolChoice: "none" };
      }
      return {};
    },
    abortSignal,
    temperature: helmcodeTemperature(),
    experimental_transform: smoothStream({
      delayInMs: 16,
      chunking: "word",
    }),
    onStepFinish({ toolCalls }) {
      const names = toolCalls.map((call) => call.toolName);
      if (names.length) {
        console.info(`[agent] ${role} tools=${names.join(",")} +${Date.now() - started}ms`);
      }
    },
    onFinish({ steps }) {
      console.info(`[agent] ${role} steps=${steps.length} total_ms=${Date.now() - started}`);
    },
  });

  // Exists in ai@7 (deprecated alias). Same SSE as createUIMessageStreamResponse + toUIMessageStream.
  return result.toUIMessageStreamResponse({
    sendReasoning: thinking,
    headers: {
      "Content-Encoding": "identity",
      "X-Accel-Buffering": "no",
    },
    onError: (error) => {
      const text = error instanceof Error ? error.message : String(error);
      return text || "No se pudo completar la respuesta.";
    },
  });
}
