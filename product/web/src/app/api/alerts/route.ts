import { getEntityAlerts } from "@/lib/data/alerts-service";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  if (!process.env.DATABASE_URL) {
    return Response.json({ error: "DATABASE_URL is not set" }, { status: 503 });
  }
  const params = new URL(request.url).searchParams;
  const company = params.get("company");
  const group = params.get("group");
  if (!company && !group) return Response.json({ error: "Invalid request: company or group" }, { status: 400 });
  try {
    return Response.json(await getEntityAlerts(company ? { company } : { group: group! }));
  } catch (error) {
    const message = error instanceof Error ? error.message : "alerts query failed";
    return Response.json({ error: message }, { status: message.startsWith("Invalid") ? 400 : 500 });
  }
}
