import { getWatcherFeed } from "@/lib/agent/watcher-service";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const company = url.searchParams.get("company") ?? "";
  const group = url.searchParams.get("group") ?? "";
  const feed = await getWatcherFeed(company ? [company] : [], group ? [group] : []);
  return Response.json(feed);
}
