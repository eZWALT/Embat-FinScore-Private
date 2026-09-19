"use client";

import { useEffect, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { WatcherPostView } from "@/components/watcher-post-view";
import type { WatcherPost } from "@/lib/agent/watcher-post";

/** The company's alerts of the last three months, as a card of the page like the plot and the indicators. */
export function CompanyAlerts({ companyId }: { companyId: string }) {
  const [result, setResult] = useState<{ companyId: string; posts: WatcherPost[]; asOf: string; error: string | null } | null>(null);

  useEffect(() => {
    let cancelled = false;
    const q = new URLSearchParams();
    if (companyId) q.set("company", companyId);
    fetch(`/api/watcher/feed?${q}`)
      .then(async (res) => {
        const body = (await res.json()) as { posts?: WatcherPost[]; asOf?: string; error?: string };
        if (!res.ok) throw new Error(body.error || res.statusText);
        if (!cancelled) setResult({ companyId, posts: body.posts ?? [], asOf: body.asOf ?? "", error: null });
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setResult({ companyId, posts: [], asOf: "", error: err instanceof Error ? err.message : "No se pudieron cargar las alertas" });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [companyId]);

  const current = result?.companyId === companyId ? result : null;
  const loaded = current !== null;
  const posts = current?.posts ?? [];
  const asOf = current?.asOf ?? "";
  const error = current?.error ?? null;

  return (
    <Card id="alerts" className="scroll-mt-20">
      <CardHeader className="gap-1">
        <div className="flex items-center justify-between gap-4">
          <CardTitle className="text-base">Alerts</CardTitle>
          {loaded ? (
            <Badge variant="outline" className="font-normal text-muted-foreground">
              {posts.length} {posts.length === 1 ? "mes" : "meses"}
            </Badge>
          ) : null}
        </div>
        <p className="text-sm text-muted-foreground">
          Últimos tres meses{asOf ? ` · ${asOf}` : ""}. Misma ficha siempre: una línea, un hecho, dueño.
        </p>
      </CardHeader>
      <CardContent>
        {error ? (
          <p className="text-sm text-destructive">{error}</p>
        ) : !loaded ? (
          <p className="text-sm text-muted-foreground">Cargando…</p>
        ) : posts.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            Aún no hay alertas de estos tres meses. Si acabas de cambiar de empresa, espera un momento.
          </p>
        ) : (
          <ol className="divide-y">
            {posts.map((post) => (
              <li key={post.month} className="py-3 first:pt-0 last:pb-0">
                <WatcherPostView post={post} flat />
              </li>
            ))}
          </ol>
        )}
      </CardContent>
    </Card>
  );
}
