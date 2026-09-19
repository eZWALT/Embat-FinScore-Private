import { getWatcherFeed } from "@/lib/agent/watcher-service";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  if (!process.env.DATABASE_URL) {
    return Response.json({ error: "DATABASE_URL is not set" }, { status: 503 });
  }
  const url = new URL(request.url);
  const company = url.searchParams.get("company") ?? "";
  const group = url.searchParams.get("group") ?? "";
  const feed = await getWatcherFeed(company ? [company] : [], group ? [group] : []);
  return Response.json(feed);
}
