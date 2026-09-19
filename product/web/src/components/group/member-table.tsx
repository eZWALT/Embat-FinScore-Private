"use client";

import { cn } from "cn";
import { ArrowDownRight, ArrowUpRight, Minus } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { SeverityBadge } from "@/components/group/alert-list";
import {
  confidenceLabels,
  formatPoints,
  guardLabels,
  scoreColor,
  trajectoryLabels,
} from "@/components/group/labels";
import type { GroupMemberRow } from "@/lib/data/group-service";

function Delta({ value }: { value: number | null }) {
  if (value === null) return <span className="text-muted-foreground">—</span>;
  const Icon = value > 0 ? ArrowUpRight : value < 0 ? ArrowDownRight : Minus;
  return (
    <span className="inline-flex items-center gap-0.5 font-mono tabular-nums">
      <Icon className="size-3.5" />
      {formatPoints(value)}
    </span>
  );
}

export function GuardBadge({ guard }: { guard: GroupMemberRow["guard"] }) {
  if (!guard) return null;
  return (
    <Badge variant={guard === "dark" ? "destructive" : "outline"} title={guard}>
      {guardLabels[guard]}
    </Badge>
  );
}

export function MemberTable({
  members,
  selectedCompanyId,
  onSelectCompany,
}: {
  members: GroupMemberRow[];
  selectedCompanyId: string | null;
  onSelectCompany: (companyId: string) => void;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="text-left text-xs text-muted-foreground">
          <tr className="[&>th]:pr-3 [&>th]:pb-2 [&>th]:font-medium [&>th]:whitespace-nowrap">
            <th>Empresa</th>
            <th className="text-right">Puntuación</th>
            <th className="text-right">Δ 3m</th>
            <th>Trayectoria</th>
            <th>Confianza</th>
            <th>Límite</th>
            <th className="text-right">Alertas</th>
          </tr>
        </thead>
        <tbody className="divide-y">
          {members.map((member) => {
            const selected = member.companyId === selectedCompanyId;
            return (
              <tr
                key={member.companyId}
                onClick={() => onSelectCompany(member.companyId)}
                aria-selected={selected}
                className={cn(
                  "cursor-pointer transition-colors hover:bg-muted/60 [&>td]:py-1.5 [&>td]:pr-3",
                  selected && "bg-muted",
                )}
              >
                <td>
                  <button
                    type="button"
                    onClick={(event) => {
                      event.stopPropagation();
                      onSelectCompany(member.companyId);
                    }}
                    className="font-mono text-xs outline-none focus-visible:underline"
                  >
                    {member.companyId}
                  </button>
                </td>
                <td className="text-right">
                  <span className="inline-flex items-center justify-end gap-1.5 font-mono tabular-nums">
                    <span
                      className="inline-block size-2 rounded-full"
                      style={{ backgroundColor: scoreColor(member.score) }}
                    />
                    {member.score.toFixed(0)}
                  </span>
                </td>
                <td className="text-right">
                  <Delta value={member.delta3m} />
                </td>
                <td>
                  <Badge variant="secondary">{trajectoryLabels[member.trajectory]}</Badge>
                </td>
                <td className="text-xs">{confidenceLabels[member.confidence]}</td>
                <td>
                  <GuardBadge guard={member.guard} />
                </td>
                <td className="text-right">
                  <span className="inline-flex items-center justify-end gap-1.5">
                    <span className="font-mono tabular-nums">{member.nAlerts}</span>
                    {member.maxAlertSeverity && <SeverityBadge severity={member.maxAlertSeverity} />}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
