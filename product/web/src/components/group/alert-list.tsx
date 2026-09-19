import { Badge } from "@/components/ui/badge";
import {
  formatEur,
  formatMonth,
  kindLabels,
  ownerLabels,
  severityBadgeVariant,
  severityLabels,
} from "@/components/group/labels";
import type { GroupAlertView } from "@/lib/data/group-service";

export function SeverityBadge({ severity }: { severity: GroupAlertView["severity"] }) {
  return <Badge variant={severityBadgeVariant[severity]}>{severityLabels[severity]}</Badge>;
}

export function AlertList({
  alerts,
  emptyText,
  currency = "EUR",
  showEntity = false,
}: {
  alerts: GroupAlertView[];
  emptyText: string;
  currency?: string | null;
  showEntity?: boolean;
}) {
  if (alerts.length === 0) {
    return <p className="text-sm text-muted-foreground">{emptyText}</p>;
  }

  return (
    <ul className="divide-y">
      {alerts.map((alert) => (
        <li key={alert.alertId} className="space-y-1.5 py-3 first:pt-0 last:pb-0">
          <div className="flex flex-wrap items-center gap-2">
            <SeverityBadge severity={alert.severity} />
            <span className="text-sm font-medium">{kindLabels[alert.kind]}</span>
            <span className="text-xs text-muted-foreground">{formatMonth(alert.month)}</span>
            {showEntity && <span className="font-mono text-xs text-muted-foreground">{alert.entityId}</span>}
            <span className="ml-auto text-xs text-muted-foreground">
              Responsable: <span className="text-foreground">{ownerLabels[alert.owner]}</span>
            </span>
          </div>
          <p className="text-sm text-muted-foreground">{alert.summary}</p>
          {alert.reasons.length > 0 && (
            <ul className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground">
              {alert.reasons.map((reason) => (
                <li key={`${alert.alertId}-${reason.item}`} className="font-mono tabular-nums">
                  {reason.item}
                  {reason.eur !== null ? ` · ${formatEur(reason.eur, currency)}` : ""}
                </li>
              ))}
            </ul>
          )}
          <p className="text-sm">
            <span className="text-muted-foreground">Acción: </span>
            {alert.action}
          </p>
        </li>
      ))}
    </ul>
  );
}
