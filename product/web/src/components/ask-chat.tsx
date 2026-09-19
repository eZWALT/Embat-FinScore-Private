"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";

import { AgentChat } from "@/components/agent-chat";
import { companyLabel, groupLabel, trajectoryLabel } from "@/components/group/labels";
import { ProductNav } from "@/components/product-nav";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

const COMPANY_QUESTIONS = [
  "¿Por qué este índice este mes?",
  "¿Qué cambió en tres meses?",
] as const;

const GROUP_QUESTIONS = [
  "¿Qué empresas hay que revisar?",
  "¿Qué alertas debe ver Cobros?",
] as const;

export function AskChat({
  companies,
  groups,
  companyId,
  groupId,
}: {
  companies: { id: string; score: number; trajectory: string }[];
  groups: { id: string; n: number; mean: number | null }[];
  companyId: string;
  groupId: string;
}) {
  const router = useRouter();

  useEffect(() => {
    void fetch("/api/ask/warmup", { method: "POST", keepalive: true }).catch(() => {});
  }, []);

  function go(nextCompany: string, nextGroup: string) {
    const q = new URLSearchParams();
    if (nextCompany) q.set("company", nextCompany);
    if (nextGroup) q.set("group", nextGroup);
    router.push(`/ask${q.size ? `?${q}` : ""}`);
  }

  const suggestions = [
    ...(companyId ? COMPANY_QUESTIONS : []),
    ...(groupId ? GROUP_QUESTIONS : []),
  ];

  return (
    <div className="mx-auto flex min-h-svh max-w-2xl flex-col gap-6 px-4 py-6">
      <header className="space-y-3">
        <ProductNav current="/ask" />
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Pregunta</h1>
          <p className="text-sm text-muted-foreground">
            Por qué un índice es el que es, qué ha cambiado y qué hay que revisar.
          </p>
        </div>
        <div className="grid gap-2 sm:grid-cols-2">
          <Select value={companyId || "none"} onValueChange={(v) => go(v === "none" ? "" : v, groupId)}>
            <SelectTrigger aria-label="Empresa">
              <SelectValue placeholder="Empresa" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="none">Ninguna empresa</SelectItem>
              {companies.map((c) => (
                <SelectItem key={c.id} value={c.id}>
                  {companyLabel(c.id)} · {c.score.toFixed(0)} · {trajectoryLabel(c.trajectory)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select value={groupId || "none"} onValueChange={(v) => go(companyId, v === "none" ? "" : v)}>
            <SelectTrigger aria-label="Grupo">
              <SelectValue placeholder="Grupo" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="none">Ningún grupo</SelectItem>
              {groups.map((g) => (
                <SelectItem key={g.id} value={g.id}>
                  {groupLabel(g.id)} · {g.n} · {g.mean == null ? "—" : `media ${g.mean.toFixed(0)}`}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </header>

      <AgentChat
        key={`${companyId}:${groupId}`}
        api="/api/ask"
        companyId={companyId || undefined}
        groupId={groupId || undefined}
        placeholder="Pregunta por esta empresa o este grupo"
        emptyHint={
          companyId || groupId
            ? undefined
            : "Elige una empresa o un grupo, o nómbralos en la pregunta."
        }
        suggestions={suggestions}
      />
    </div>
  );
}
