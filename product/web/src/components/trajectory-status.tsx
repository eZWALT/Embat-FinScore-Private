import { ArrowDownRight, ArrowRight, Hourglass, TrendingDown, TrendingUp, type LucideIcon } from "lucide-react";

import { cn } from "cn";

import { trajectoryLabels } from "@/components/group/labels";
import type { Trajectory } from "@/lib/data/types";

const STATUS: Record<Trajectory, { icon: LucideIcon; tone: string; text: string }> = {
  improving: {
    icon: TrendingUp,
    tone: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400",
    text: "Tendencia al alza sostenida",
  },
  stable: {
    icon: ArrowRight,
    tone: "bg-muted text-foreground",
    text: "Sin cambios relevantes",
  },
  dip: {
    icon: ArrowDownRight,
    tone: "bg-amber-500/15 text-amber-700 dark:text-amber-400",
    text: "Caída reciente, aún sin tendencia",
  },
  deteriorating: {
    icon: TrendingDown,
    tone: "bg-destructive/10 text-destructive",
    text: "Tendencia a la baja sostenida",
  },
  "insufficient history": {
    icon: Hourglass,
    tone: "bg-muted text-muted-foreground",
    text: "Hacen falta 4 meses puntuados",
  },
};

/** The company's trajectory as a status: a toned icon, the label at reading size and one line saying what it means. */
export function TrajectoryStatus({ trajectory, className }: { trajectory: Trajectory; className?: string }) {
  const { icon: Icon, tone, text } = STATUS[trajectory];
  return (
    <div className={cn("flex items-center gap-3", className)}>
      <span className={cn("grid size-10 shrink-0 place-items-center rounded-full", tone)} aria-hidden="true">
        <Icon className="size-5" />
      </span>
      <div className="min-w-0">
        <p className="text-xl font-semibold leading-tight tracking-tight">{trajectoryLabels[trajectory]}</p>
        <p className="mt-0.5 text-xs text-muted-foreground">{text}</p>
      </div>
    </div>
  );
}
