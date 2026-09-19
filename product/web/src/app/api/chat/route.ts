import { convertToModelMessages, streamText, type UIMessage } from "ai";

export const maxDuration = 30;
export const dynamic = "force-dynamic";

const SYSTEM = `Eres el asistente de Health Sentinel (Embat).
Responde en el idioma del usuario, en 1–4 frases.
El índice 0–100 es explicable y monitorable: nunca digas que predice quiebra, default ni una probabilidad.
No inventes cifras, empresas ni resultados del índice. Si no tienes el dato, dilo.
Nunca digas "revenue at risk"; si hablas del cliente principal silencioso, di que dejó de facturar y hay que revisar exposición y cobros.`;

export async function POST(request: Request) {
  const { messages }: { messages?: UIMessage[] } = await request.json();
  if (!messages?.length) {
    return Response.json({ error: "messages is required" }, { status: 400 });
  }

  const result = streamText({
    model: "openai/gpt-5.4",
    system: SYSTEM,
    messages: await convertToModelMessages(messages),
  });

  return result.toUIMessageStreamResponse();
}
