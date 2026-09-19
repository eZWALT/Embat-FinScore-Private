import { convertToModelMessages, streamText, type UIMessage } from "ai";

import { createHelmcodeModel, helmcodeApiKey } from "@/lib/agent/llm";

export const maxDuration = 30;
export const dynamic = "force-dynamic";

const SYSTEM = `Eres el asistente de Health Sentinel (Embat).
Responde en el idioma del usuario, en 1–4 frases.
El índice 0–100 es explicable y monitorable: nunca digas que predice quiebra, default ni una probabilidad.
No inventes cifras, empresas ni resultados del índice. Si no tienes el dato, dilo.
Nunca digas "revenue at risk"; si hablas del cliente principal silencioso, di que dejó de facturar y hay que revisar exposición y cobros.`;

export async function POST(request: Request) {
  if (!helmcodeApiKey()) {
    return Response.json({ error: "HELMCODE_API_KEY is not set" }, { status: 503 });
  }

  const { messages }: { messages?: UIMessage[] } = await request.json();
  if (!messages?.length) {
    return Response.json({ error: "messages is required" }, { status: 400 });
  }

  const result = streamText({
    model: createHelmcodeModel(),
    system: SYSTEM,
    messages: await convertToModelMessages(messages),
  });

  return result.toUIMessageStreamResponse({
    onError: (error) => {
      const text = error instanceof Error ? error.message : String(error);
      return text || "No se pudo completar la respuesta.";
    },
  });
}
