import { readFile } from "node:fs/promises";
import path from "node:path";
import { getToolName, isToolUIPart, smoothStream, streamText, type UIMessage } from "ai";

import { textFromParts } from "./chat-parts";
import { createHelmcodeModel, helmcodeApiKey, helmcodeTemperature } from "./llm";
import { coerceUiMessages } from "./messages";

const MAX_TURNS = 6;

function clip(value: string, max: number) {
  return value.length > max ? `${value.slice(0, max)}…` : value;
}

function toolLines(parts: UIMessage["parts"]): string {
  return parts
    .filter(isToolUIPart)
    .map((part) => {
      const name = getToolName(part);
      const input = "input" in part && part.input != null ? clip(JSON.stringify(part.input), 400) : "";
      const output = "output" in part && part.output != null ? clip(JSON.stringify(part.output), 1200) : "";
      return `  herramienta ${name}${input ? ` entrada=${input}` : ""}${output ? ` salida=${output}` : ""}`;
    })
    .join("\n");
}

function transcriptOf(messages: unknown): string {
  return coerceUiMessages(messages)
    .slice(-MAX_TURNS)
    .map((message) => {
      const who = message.role === "assistant" ? "Sentinel" : message.role === "user" ? "Usuario" : message.role;
      const text = textFromParts(message.parts).trim();
      const tools = toolLines(message.parts);
      return [text ? `${who}: ${text}` : "", tools].filter(Boolean).join("\n");
    })
    .filter(Boolean)
    .join("\n\n");
}

async function loadFollowupPrompt() {
  return readFile(path.join(process.cwd(), "src/lib/agent/prompts/followups.md"), "utf8");
}

/** Two clickable next questions. Real SSE text deltas — not a buffered blob. */
export async function streamFollowupChips(messages: unknown): Promise<Response> {
  if (!helmcodeApiKey()) {
    return new Response(null, { status: 204 });
  }

  const transcript = transcriptOf(messages);
  if (!transcript) {
    return Response.json({ error: "messages is required" }, { status: 400 });
  }

  const result = streamText({
    model: createHelmcodeModel(),
    system: await loadFollowupPrompt(),
    prompt: transcript,
    temperature: helmcodeTemperature(),
    maxOutputTokens: 120,
    experimental_transform: smoothStream({
      delayInMs: 12,
      chunking: "line",
    }),
  });

  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      try {
        for await (const delta of result.textStream) {
          if (delta) controller.enqueue(encoder.encode(delta));
        }
        controller.close();
      } catch (error) {
        controller.error(error);
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "Cache-Control": "no-store, no-transform",
      "Content-Encoding": "identity",
      "X-Accel-Buffering": "no",
    },
  });
}
