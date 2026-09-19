import { getCompanySize } from "@/lib/data/size";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  if (!process.env.DATABASE_URL) {
    return Response.json({ error: "DATABASE_URL is not set" }, { status: 503 });
  }
  const params = new URL(request.url).searchParams;
  try {
    return Response.json(await getCompanySize(params.get("company") ?? "", params.get("month") ?? ""));
  } catch (error) {
    const message = error instanceof Error ? error.message : "size query failed";
    const status = message.startsWith("Invalid") ? 400 : message.startsWith("Unavailable") ? 503 : 500;
    return Response.json({ error: message }, { status });
  }
}
