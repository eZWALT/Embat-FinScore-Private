import type { Metadata } from "next";

import { WatcherChannel } from "@/components/watcher-channel";
import { getWatcherFeed } from "@/lib/agent/watcher-service";
import { LocalBundleRepository } from "@/lib/data/local-bundle-repository";

export const metadata: Metadata = {
  title: "Watcher · Health Sentinel",
  description: "Last three months on a watch set, one fixed format.",
};

type SearchParams = Promise<Record<string, string | string[] | undefined>>;

function first(value: string | string[] | undefined) {
  return Array.isArray(value) ? value[0] : value;
}

export default async function WatcherPage({ searchParams }: { searchParams: SearchParams }) {
  const params = await searchParams;
  const companyId = first(params.company) ?? "";
  const groupId = first(params.group) ?? "";

  const repo = new LocalBundleRepository();
  const [companies, groups, feed] = await Promise.all([
    repo.listCompanies(),
    repo.listGroups(),
    getWatcherFeed(companyId ? [companyId] : [], groupId ? [groupId] : []),
  ]);

  return (
    <WatcherChannel
      asOf={feed.asOf}
      disclaimer={feed.disclaimer}
      posts={feed.posts}
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
