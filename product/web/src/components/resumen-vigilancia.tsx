"use client";

import { useEffect, useState } from "react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { WatcherPostView } from "@/components/watcher-post-view";
import type { WatcherPost } from "@/lib/agent/watcher-post";

export function ResumenVigilancia({ companyId }: { companyId: string }) {
  const [posts, setPosts] = useState<WatcherPost[]>([]);
  const [asOf, setAsOf] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const q = new URLSearchParams();
    if (companyId) q.set("company", companyId);
    setPosts([]);
    setError(null);
    fetch(`/api/watcher/feed?${q}`)
      .then(async (res) => {
        const body = (await res.json()) as { posts?: WatcherPost[]; asOf?: string; error?: string };
        if (!res.ok) throw new Error(body.error || res.statusText);
        if (!cancelled) {
          setPosts(body.posts ?? []);
          setAsOf(body.asOf ?? "");
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "No se pudo cargar la vigilancia");
      });
    return () => {
      cancelled = true;
    };
  }, [companyId]);

  return (
    <section id="vigilancia" className="mt-8 scroll-mt-20 space-y-3">
      <div>
        <h2 className="text-base font-semibold tracking-tight">Vigilancia</h2>
        <p className="text-sm text-muted-foreground">
          Últimos tres meses{asOf ? ` · ${asOf}` : ""}. Misma ficha siempre: una línea, un hecho, dueño.
        </p>
      </div>
      {error ? <p className="text-sm text-destructive">{error}</p> : null}
      {!error && posts.length === 0 ? (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Sin fichas aún</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Aún no hay fichas de estos tres meses. Si acabas de cambiar de empresa, espera un momento.
            </p>
          </CardContent>
        </Card>
      ) : (
        <ol className="space-y-3">
          {posts.map((post) => (
            <li key={post.month}>
              <WatcherPostView post={post} />
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
