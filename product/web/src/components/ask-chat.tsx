"use client";

import { useRouter } from "next/navigation";

import { AgentChat } from "@/components/agent-chat";
import { ProductNav } from "@/components/product-nav";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

const COMPANY_QUESTIONS = [
  "Why is this company's score what it is this month?",
  "What changed in the last three months, and who owns the action?",
] as const;

const GROUP_QUESTIONS = [
  "Which companies in this group need attention this month?",
  "What group alerts should collections or the treasurer review?",
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
          <h1 className="text-xl font-semibold tracking-tight">Ask</h1>
          <p className="text-sm text-muted-foreground">
            Questions about a company or group. Scores and alerts from Neon; records if core is
            mounted.
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

      <AgentChat
        key={`${companyId}:${groupId}`}
        api="/api/ask"
        companyId={companyId || undefined}
        groupId={groupId || undefined}
        placeholder="Ask about this company or group"
        emptyHint={
          companyId || groupId
            ? undefined
            : "Pick a company or a group, or name one in the question."
        }
        suggestions={suggestions}
      />
    </div>
  );
}
