import { Wrench } from "lucide-react";

import { WatcherSpark } from "@/components/agent-plot";
import { toolLabel } from "@/lib/agent/tool-catalog";
import { formatMonth } from "@/lib/format-month";
import type { WatcherBullet, WatcherPost } from "@/lib/agent/watcher-post";

const SEVERITY: Record<WatcherBullet["severity"], { label: string; className: string }> = {
  act: { label: "ACTUAR", className: "bg-destructive/10 text-destructive" },
  watch: { label: "VIGILAR", className: "bg-amber-500/15 text-amber-700 dark:text-amber-400" },
  opportunity: { label: "OPORTUNIDAD", className: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400" },
  follow: { label: "SEGUIR", className: "bg-muted text-muted-foreground" },
};

export function WatcherPostView({ post }: { post: WatcherPost }) {
  return (
    <article className="space-y-2 rounded-xl border bg-background px-4 py-3">
      <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">{formatMonth(post.month)}</p>
      <p className="text-sm font-semibold leading-snug">{post.line1}</p>
      <p className="text-sm leading-snug text-muted-foreground">{post.line2}</p>
      {post.bullets.length > 0 ? (
        <ul className="space-y-1.5 pt-1">
          {post.bullets.map((b, i) => (
            <li key={`${b.entity}-${i}`} className="flex items-start gap-2 text-sm leading-snug">
              <span
                className={`mt-px shrink-0 rounded px-1.5 py-0.5 font-mono text-[10px] leading-none font-semibold ${SEVERITY[b.severity].className}`}
              >
                {SEVERITY[b.severity].label}
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
      {post.spark ? <WatcherSpark id={post.spark.id} points={post.spark.points} /> : null}
      {post.n_info > 0 ? (
        <p className="text-xs text-muted-foreground">
          {post.n_info} aviso{post.n_info === 1 ? "" : "s"} informativo{post.n_info === 1 ? "" : "s"}
        </p>
      ) : null}
    </article>
  );
}

export function ToolCallChip({ name }: { name: string }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] text-muted-foreground">
      <Wrench className="size-3" aria-hidden="true" />
      <span>{toolLabel(name)}</span>
    </span>
  );
}
