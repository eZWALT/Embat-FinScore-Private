"use client";

import { useRouter } from "next/navigation";

import { ProductNav } from "@/components/product-nav";
import { WatcherPostView } from "@/components/watcher-post-view";
import { WatcherReply } from "@/components/watcher-reply";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { WatcherPost } from "@/lib/agent/watcher-post";

export function WatcherChannel({
  asOf,
  posts,
  companies,
  groups,
  companyId,
  groupId,
}: {
  asOf: string;
  posts: WatcherPost[];
  companies: { id: string; score: number; trajectory: string }[];
  groups: { id: string; n: number; mean: number | null }[];
  companyId: string;
  groupId: string;
}) {
  const router = useRouter();

  function go(nextCompany: string, nextGroup: string) {
    const q = new URLSearchParams();
    if (nextCompany) q.set("company", nextCompany);
    if (nextGroup) q.set("group", nextGroup);
    router.push(`/watcher${q.size ? `?${q}` : ""}`);
  }

  return (
    <div id="sentinel" className="mx-auto flex min-h-svh max-w-2xl flex-col gap-6 px-4 py-6">
      <header className="space-y-3">
        <ProductNav current="/watcher" />
        <div>
          <h1 className="text-xl font-semibold tracking-tight">#sentinel</h1>
          <p className="text-sm text-muted-foreground">
            Last three months{asOf ? ` · as-of ${asOf}` : ""}. Same shape every time.
          </p>
        </div>
        <div className="grid gap-2 sm:grid-cols-2">
          <Select value={companyId || "none"} onValueChange={(v) => go(v === "none" ? "" : v, groupId)}>
            <SelectTrigger aria-label="Company">
              <SelectValue placeholder="Company" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="none">No company</SelectItem>
              {companies.map((c) => (
                <SelectItem key={c.id} value={c.id}>
                  {c.id} · {c.score.toFixed(0)} · {c.trajectory}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select value={groupId || "none"} onValueChange={(v) => go(companyId, v === "none" ? "" : v)}>
            <SelectTrigger aria-label="Group">
              <SelectValue placeholder="Group" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="none">No group</SelectItem>
              {groups.map((g) => (
                <SelectItem key={g.id} value={g.id}>
                  {g.id} · {g.n} · {g.mean == null ? "—" : `mean ${g.mean.toFixed(0)}`}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </header>

      {!companyId && !groupId ? (
        <p className="text-sm text-muted-foreground">Pick a company or a group.</p>
      ) : (
        <ol className="space-y-3">
          {posts.map((post) => (
            <li key={post.month}>
              <WatcherPostView post={post} />
            </li>
          ))}
        </ol>
      )}

      {companyId || groupId ? (
        <WatcherReply companyId={companyId} groupId={groupId} asOf={asOf} />
      ) : null}

    </div>
  );
}
