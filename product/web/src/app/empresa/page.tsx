import { ViewTransition } from "react";

import { HealthDashboard } from "@/components/health-dashboard";
import { getDashboardData } from "@/lib/data/dashboard-service";

export const dynamic = "force-dynamic";

type SearchParams = Promise<Record<string, string | string[] | undefined>>;

function first(value: string | string[] | undefined) {
  return Array.isArray(value) ? value[0] : value;
}

export default async function EmpresaPage({ searchParams }: { searchParams: SearchParams }) {
  const params = await searchParams;
  const data = await getDashboardData();
  const openChat = first(params.chat) === "1";
  const company = first(params.company);
  const group = first(params.group);
  const deep = first(params.modo) === "profundo" || Boolean(company) || Boolean(group) || openChat;

  return (
    <ViewTransition enter="route-in" default="none">
      <HealthDashboard
        data={data}
        openChat={openChat}
        initialCompanyId={company}
        initialGroupId={group}
        initialMode={deep ? "deep" : "quick"}
      />
    </ViewTransition>
  );
}
