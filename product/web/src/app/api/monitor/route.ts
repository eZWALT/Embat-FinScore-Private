import { getCompanyMonitor } from "@/lib/data/monitor-service";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  if (!process.env.DATABASE_URL) {
    return Response.json({ error: "DATABASE_URL is not set" }, { status: 503 });
  }
  const company = new URL(request.url).searchParams.get("company") ?? "";
  try {
    return Response.json(await getCompanyMonitor(company));
  } catch (error) {
    const message = error instanceof Error ? error.message : "monitor query failed";
    return Response.json({ error: message }, { status: message.startsWith("Invalid") ? 400 : 500 });
  }
}
