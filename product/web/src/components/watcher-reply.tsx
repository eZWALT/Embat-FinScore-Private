"use client";

import { AgentChat } from "@/components/agent-chat";

export function WatcherReply({
  companyId,
  groupId,
  asOf,
}: {
  companyId: string;
  groupId: string;
  asOf: string;
}) {
  return (
    <section className="space-y-2" aria-label="Pregunta por estas fichas">
      <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
        Pregunta
      </p>
      <AgentChat
        key={`${companyId}:${groupId}:${asOf}`}
        api="/api/watcher/reply"
        companyId={companyId || undefined}
        groupId={groupId || undefined}
        asOf={asOf || undefined}
        placeholder="Pregunta por estas fichas"
      />
    </section>
  );
}
