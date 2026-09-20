import { convertToModelMessages, isStepCount, smoothStream, streamText } from "ai";

import { createHelmcodeModel, helmcodeApiKey, helmcodeTemperature } from "./llm";
import {
  coerceUiMessages,
  entitiesFromText,
  lastUserText,
  sessionExtra,
  wantsAlertStats,
  wantsAlertTools,
  wantsPlotCatalog,
  wantsRecordTools,
} from "./messages";
import { loadSystemPrompt, type AgentRole } from "./prompt-loader";
import { activeToolsUnderCap, chatTools, quickTools, sentinelTools, shouldForceTextStep } from "./tools";
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

  const question = lastUserText(uiMessages);
  const records = wantsRecordTools(question);
  const plots = wantsPlotCatalog(question);
  const alerts = wantsAlertTools(question);
  const alertsOnly = alerts && !records && !plots && !/índice|por qu[eé]|cambi[oó]|periodo|gr[aá]fico/i.test(question);
  const system = await loadSystemPrompt(role, sessionExtra({ companyId, groupId, asOf, view }), {
    thinking,
    records,
    plots,
  });
  const modelMessages = await convertToModelMessages(uiMessages);
  const session = {
    companyId,
    groupId,
    named: entitiesFromText(question),
    stats: wantsAlertStats(question),
  };
  const tools = role === "sentinel" ? sentinelTools(session) : role === "quick" ? quickTools(session) : chatTools(session);

  const started = Date.now();
  const result = streamText({
    model: createHelmcodeModel({ thinking }),
    system,
    messages: modelMessages,
    tools,
    // Never stop on tool count: that killed the text step after a parallel burst.
    // Caps strip tools in prepareStep so the model still writes.
    stopWhen: [isStepCount(8)],
    prepareStep({ steps }) {
      const names = Object.keys(tools) as (keyof typeof tools)[];
      if (shouldForceTextStep(steps, { alertsOnly })) {
        return { activeTools: [], toolChoice: "none" };
      }
      return { activeTools: activeToolsUnderCap(names as string[], steps, { records, plots, alerts }) as typeof names };
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
