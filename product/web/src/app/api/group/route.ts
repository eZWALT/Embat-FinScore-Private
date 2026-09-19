import { getGroupOverview } from "@/lib/data/group-service";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  if (!process.env.DATABASE_URL) {
    return Response.json({ error: "DATABASE_URL is not set" }, { status: 503 });
  }
  const id = new URL(request.url).searchParams.get("id") ?? "";
  try {
    return Response.json(await getGroupOverview(id));
  } catch (error) {
    const message = error instanceof Error ? error.message : "group query failed";
    const status = message.startsWith("Invalid") ? 400 : message.startsWith("Group not found") ? 404 : 500;
    return Response.json({ error: message }, { status });
  }
}
