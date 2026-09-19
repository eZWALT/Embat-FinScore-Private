import { Wrench } from "lucide-react";

import type { WatcherBullet, WatcherPost } from "@/lib/agent/watcher-post";

const SEVERITY: Record<WatcherBullet["severity"], string> = {
  act: "ACT",
  watch: "WATCH",
  opportunity: "OPP",
  follow: "FOLLOW",
};

export function WatcherPostView({ post }: { post: WatcherPost }) {
  return (
    <article className="space-y-2 rounded-xl border bg-background px-4 py-3">
      <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">{post.month}</p>
      <p className="text-sm font-semibold leading-snug">{post.line1}</p>
      <p className="text-sm leading-snug text-muted-foreground">{post.line2}</p>
      {post.bullets.length > 0 ? (
        <ul className="space-y-1.5 pt-1">
          {post.bullets.map((b, i) => (
            <li key={`${b.entity}-${i}`} className="flex gap-2 text-sm leading-snug">
              <span className="mt-0.5 shrink-0 font-mono text-[10px] font-semibold text-muted-foreground">
                {SEVERITY[b.severity]}
              </span>
              <span>
                <span className="font-medium">{b.owner}</span>
                {" · "}
                <span className="font-mono text-xs">{b.entity}</span>
                {" — "}
                {b.text}
              </span>
            </li>
          ))}
        </ul>
      ) : null}
      {post.n_info > 0 ? (
        <p className="text-xs text-muted-foreground">{post.n_info} quieter info flag{post.n_info === 1 ? "" : "s"}</p>
      ) : null}
    </article>
  );
}

export function ToolCallChip({ name }: { name: string }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-md border bg-muted/50 px-1.5 py-0.5 font-mono text-[11px] text-muted-foreground">
      <Wrench className="size-3" aria-hidden="true" />
      <span>tools</span>
      <span>({name})</span>
    </span>
  );
}
