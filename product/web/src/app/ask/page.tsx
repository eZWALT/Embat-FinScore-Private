import type { Metadata } from "next";

import { AskChat } from "@/components/ask-chat";
import { createScoreRepository } from "@/lib/data/repository";

export const metadata: Metadata = {
  title: "Ask · Health Sentinel",
  description: "Questions about a company or group, from the bundle and clean records.",
};

export const dynamic = "force-dynamic";

type SearchParams = Promise<Record<string, string | string[] | undefined>>;

function first(value: string | string[] | undefined) {
  return Array.isArray(value) ? value[0] : value;
}

export default async function AskPage({ searchParams }: { searchParams: SearchParams }) {
  const params = await searchParams;
  const companyId = first(params.company) ?? "";
  const groupId = first(params.group) ?? "";

  const repo = createScoreRepository();
  const [companies, groups] = await Promise.all([repo.listCompanies(), repo.listGroups()]);

  return (
    <AskChat
      companyId={companyId}
      groupId={groupId}
      companies={companies
        .slice()
        .sort((a, b) => a.score - b.score)
        .map((c) => ({ id: c.company_id, score: c.score, trajectory: c.trajectory }))}
      groups={groups
        .slice()
        .sort((a, b) => (a.latest_mean_score ?? 99) - (b.latest_mean_score ?? 99))
        .map((g) => ({ id: g.group_id, n: g.n_companies, mean: g.latest_mean_score }))}
    />
  );
}
