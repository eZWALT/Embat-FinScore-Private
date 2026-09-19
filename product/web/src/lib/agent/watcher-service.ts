import { LocalBundleRepository } from "@/lib/data/local-bundle-repository";
import type { CompanyDetail } from "@/lib/data/types";

import { buildWatcherPosts, type WatcherPost } from "./watcher-post";

const COMPANY_ID = /^COMP_[0-9]{4}$/;
const GROUP_ID = /^GROUP_[0-9]{4}$/;

export async function getWatcherFeed(
  companyIds: string[],
  groupIds: string[],
): Promise<{ asOf: string; posts: WatcherPost[]; disclaimer: string }> {
  const cids = [...new Set(companyIds.filter((id) => COMPANY_ID.test(id)))];
  const gids = [...new Set(groupIds.filter((id) => GROUP_ID.test(id)))];
  if (!cids.length && !gids.length) {
    return { asOf: "", posts: [], disclaimer: "" };
  }

  const repo = new LocalBundleRepository();
  const [manifest, companies, groups, feed] = await Promise.all([
    repo.getManifest(),
    repo.listCompanies(),
    repo.listGroups(),
    repo.getAlerts(),
  ]);

  const watchedGroups = groups.filter((g) => gids.includes(g.group_id));
  const detailIds = new Set(cids);
  for (const g of watchedGroups) {
    for (const id of g.company_ids) detailIds.add(id);
  }

  const details = new Map<string, CompanyDetail>();
  await Promise.all(
    [...detailIds].map(async (id) => {
      try {
        details.set(id, await repo.getCompany(id));
      } catch {
        /* sample bundle may omit a group member */
      }
    }),
  );

  return {
    asOf: manifest.as_of_month,
    disclaimer: manifest.disclaimer,
    posts: buildWatcherPosts({
      asOf: manifest.as_of_month,
      asOfMonths: manifest.months,
      companyIds: cids,
      groupIds: gids,
      companies,
      details,
      groups: watchedGroups,
      alerts: feed.alerts,
    }),
  };
}
