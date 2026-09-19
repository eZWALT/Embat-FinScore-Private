"use client";

import { cn } from "cn";

import { SeverityBadge } from "@/components/alerts-table";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { companyLabel, formatDecimal, formatMonth, scoreColor, splitMonth } from "@/components/group/labels";
import type { GroupMemberRow } from "@/lib/data/group-service";

function Cell({
  label,
  month,
  score,
  onClick,
}: {
  label: string;
  month: string;
  score: number | null;
  onClick?: () => void;
}) {
  const color = scoreColor(score);
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          onClick={onClick}
          aria-label={`${label}, ${formatMonth(month)}: ${score === null ? "sin puntuación" : score.toFixed(0)}`}
          className={cn(
            "h-5 w-full rounded-[3px] outline-none transition-opacity hover:opacity-80 focus-visible:ring-2 focus-visible:ring-ring",
            color ? "" : "bg-muted",
          )}
          style={color ? { backgroundColor: color } : undefined}
        />
      </TooltipTrigger>
      <TooltipContent side="top">
        <span>{label}</span>
        <span className="text-background/70">·</span>
        <span>{formatMonth(month)}</span>
        <span className="text-background/70">·</span>
        <span className="font-mono tabular-nums">{score === null ? "sin puntuación" : formatDecimal(score, 1)}</span>
      </TooltipContent>
    </Tooltip>
  );
}

export function ScoreHeatmap({
  months,
  meanScores,
  members,
  selectedCompanyId,
  onSelectCompany,
}: {
  months: string[];
  meanScores: (number | null)[];
  members: GroupMemberRow[];
  selectedCompanyId: string | null;
  onSelectCompany: (companyId: string) => void;
}) {
  const gridTemplateColumns = `minmax(7.5rem, auto) repeat(${months.length}, minmax(1.5rem, 1fr)) auto 2.5rem`;

  return (
    <div className="overflow-x-auto">
      <div className="grid min-w-[46rem] gap-x-1 gap-y-1" style={{ gridTemplateColumns }}>
        <div />
        {months.map((month) => {
          const [name, year] = splitMonth(month);
          return (
            <div
              key={month}
              className="flex flex-col items-center text-center text-[10px] leading-3 text-muted-foreground"
            >
              <span>{name}</span>
              <span>{year}</span>
            </div>
          );
        })}
        <div className="col-span-2 flex items-end justify-center pb-0.5 pl-3 text-[10px] font-medium leading-3 text-muted-foreground">Alertas</div>

        <div className="flex items-center pr-2 text-xs font-medium">Media del grupo</div>
        {months.map((month, index) => (
          <Cell key={month} label="Media del grupo" month={month} score={meanScores[index] ?? null} />
        ))}
        <div className="col-span-2" />

        <div className="col-span-full my-1 h-px bg-border" />

        {members.map((member) => {
          const selected = member.companyId === selectedCompanyId;
          return (
            <MemberRow
              key={member.companyId}
              member={member}
              months={months}
              selected={selected}
              onSelect={() => onSelectCompany(member.companyId)}
            />
          );
        })}
      </div>
    </div>
  );
}

function MemberRow({
  member,
  months,
  selected,
  onSelect,
}: {
  member: GroupMemberRow;
  months: string[];
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <>
      <button
        type="button"
        onClick={onSelect}
        aria-pressed={selected}
        className={cn(
          "group flex h-5 items-center justify-between gap-2 rounded-sm pr-2 pl-1 text-left text-xs tabular-nums outline-none hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring",
          selected && "bg-muted font-medium",
        )}
        title={member.companyId}
      >
        <span className="underline decoration-muted-foreground/30 underline-offset-2 group-hover:decoration-foreground">{companyLabel(member.companyId)}</span>
        <span className="text-muted-foreground">{member.score.toFixed(0)}</span>
      </button>
      {months.map((month, index) => (
        <Cell
          key={month}
          label={companyLabel(member.companyId)}
          month={month}
          score={member.scores[index] ?? null}
          onClick={onSelect}
        />
      ))}
      <div className="flex h-5 items-center justify-center pl-3">
        {member.maxAlertSeverity ? <SeverityBadge severity={member.maxAlertSeverity} /> : null}
      </div>
      <div className={cn("flex h-5 items-center justify-end font-mono text-xs tabular-nums", member.nAlerts === 0 && "text-muted-foreground")}>
        {member.nAlerts}
      </div>
    </>
  );
}

export function HeatmapLegend() {
  const stops = [0, 25, 50, 75, 100].map((s) => scoreColor(s)).join(", ");
  return (
    <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
      <span>0</span>
      <span className="h-2 w-28 rounded-full" style={{ background: `linear-gradient(to right, ${stops})` }} />
      <span>100</span>
      <span className="ml-2 inline-block size-2.5 rounded-[2px] bg-muted" />
      <span>sin puntuación</span>
    </div>
  );
}
