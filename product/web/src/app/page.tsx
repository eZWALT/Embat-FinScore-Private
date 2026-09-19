import { HealthDashboard } from "@/components/health-dashboard";
import { getDashboardData } from "@/lib/data/dashboard-service";

export default async function Home() {
  const data = await getDashboardData();

  return <HealthDashboard data={data} />;
}
