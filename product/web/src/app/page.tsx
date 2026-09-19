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

  return <HealthDashboard data={data} openChat={first(params.chat) === "1"} />;
}
