import { streamFollowupChips } from "@/lib/agent/followups";

export const runtime = "nodejs";
export const maxDuration = 20;
export const dynamic = "force-dynamic";

/** Next-two chips. Client starts this only after the main answer finished, with tools in the transcript. */
export async function POST(request: Request) {
  const body = (await request.json()) as { messages?: unknown };
  return streamFollowupChips(body.messages);
}
