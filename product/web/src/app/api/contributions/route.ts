import { getCompanyContributions } from "@/lib/data/contributions";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const company = new URL(request.url).searchParams.get("company") ?? "";
  try {
    return Response.json(await getCompanyContributions(company));
  } catch (error) {
    const message = error instanceof Error ? error.message : "contributions query failed";
    return Response.json({ error: message }, { status: message.startsWith("Invalid") ? 400 : 500 });
  }
}
