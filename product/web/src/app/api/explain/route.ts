import { streamAgentResponse } from "@/lib/agent/runtime";

export const runtime = "nodejs";
export const maxDuration = 60;
export const dynamic = "force-dynamic";

/** Quick mode: one short explanation of the companies and period selected on the chart. */
export async function POST(request: Request) {
  const body = (await request.json()) as { messages?: unknown; asOf?: string };
  return streamAgentResponse({
    role: "quick",
    messages: body.messages,
    asOf: body.asOf,
    abortSignal: request.signal,
  });
}
