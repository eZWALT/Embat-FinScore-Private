import { streamAgentResponse } from "@/lib/agent/runtime";

export const runtime = "nodejs";
export const maxDuration = 120;
export const dynamic = "force-dynamic";

/** Live thread replies only. Opening month posts are built offline / by watcher-post.ts. */
export async function POST(request: Request) {
  const body = (await request.json()) as {
    messages?: unknown;
    companyId?: string;
    groupId?: string;
    asOf?: string;
  };
  return streamAgentResponse({
    role: "sentinel",
    messages: body.messages,
    companyId: body.companyId,
    groupId: body.groupId,
    asOf: body.asOf,
  });
}
