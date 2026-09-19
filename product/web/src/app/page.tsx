import { HealthDashboard } from "@/components/health-dashboard";
import { getDashboardData } from "@/lib/data/dashboard-service";

export const dynamic = "force-dynamic";

type SearchParams = Promise<Record<string, string | string[] | undefined>>;

function first(value: string | string[] | undefined) {
  return Array.isArray(value) ? value[0] : value;
}

export default async function Home({ searchParams }: { searchParams: SearchParams }) {
  const params = await searchParams;
  const data = await getDashboardData();
  const openChat = first(params.chat) === "1";
  const company = first(params.company);
  // Quick is the landing view; a link that names a company, the chat or the deep mode opens deep.
  const deep = first(params.modo) === "profundo" || Boolean(company) || openChat;

  return (
    <HealthDashboard
      data={data}
      openChat={openChat}
      initialCompanyId={company}
      initialMode={deep ? "deep" : "quick"}
    />
  );
}
