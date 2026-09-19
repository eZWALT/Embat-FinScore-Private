import { streamFollowupChips } from "@/lib/agent/followups";

export const runtime = "nodejs";
export const maxDuration = 20;
export const dynamic = "force-dynamic";

/** Agentic next-two chips. Starts while the main answer is still streaming. */
export async function POST(request: Request) {
  const body = (await request.json()) as { messages?: unknown };
  return streamFollowupChips(body.messages);
}
