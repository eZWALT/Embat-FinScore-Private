import type {
  Alert,
  CompanyDetail,
  CompanyIndexRow,
  GroupRow,
  MonthRecord,
  Owner,
} from "@/lib/data/types";

export type BulletSeverity = "act" | "watch" | "opportunity" | "follow";

export interface WatcherBullet {
  severity: BulletSeverity;
  owner: "Treasurer" | "CFO" | "Collections";
  entity: string;
  text: string;
}

export interface WatcherSparkPoint {
  month: string;
  score: number;
}

export interface WatcherPost {
  month: string;
  line1: string;
  line2: string;
  bullets: WatcherBullet[];
  n_info: number;
  /** Focus company, last 6 scored months up to this post. Renderer-only; not LLM prose. */
  spark?: { id: string; points: WatcherSparkPoint[] };
}

const OWNER: Record<Owner, WatcherBullet["owner"]> = {
  treasurer: "Treasurer",
  cfo: "CFO",
  collections: "Collections",
};

export function fmtEur(value: number | null | undefined, currency = "€"): string {
  if (value == null || Number.isNaN(value)) return "";
  const abs = Math.abs(value);
  const sign = value < 0 ? "-" : "";
  if (abs >= 1e6) return `${sign}${currency}${(abs / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `${sign}${currency}${Math.round(abs / 1e3)}k`;
  return `${sign}${currency}${Math.round(abs)}`;
}

export function monthAdd(month: string, k: number): string {
  const year = Number(month.slice(0, 4));
  const m = Number(month.slice(5, 7));
  const idx = year * 12 + (m - 1) + k;
  return `${String(Math.floor(idx / 12)).padStart(4, "0")}-${String((idx % 12) + 1).padStart(2, "0")}`;
}

export function lastMonths(asOf: string, n = 3): string[] {
  return Array.from({ length: n }, (_, i) => monthAdd(asOf, -(n - 1 - i)));
}

function signed(n: number | null | undefined, digits = 0): string | null {
  if (n == null || Number.isNaN(n)) return null;
  const v = Number(n.toFixed(digits));
  return `${v > 0 ? "+" : ""}${v}`;
}

function groupState(delta: number | null): "up" | "down" | "held" {
  if (delta == null) return "held";
  if (delta >= 2) return "up";
  if (delta <= -2) return "down";
  return "held";
}

function firstMoney(reasons: { label: string; eur: number | null; sentence: string }[] | undefined): string | null {
  if (!reasons?.length) return null;
  const withEur = reasons.find((r) => r.eur != null);
  if (withEur) {
    const money = fmtEur(withEur.eur);
    return money ? `${withEur.label} · ${money}` : withEur.label;
  }
  return reasons[0].label;
}

function ownerOf(alert: Alert): WatcherBullet["owner"] {
  return OWNER[alert.owner] ?? "CFO";
}

function evNum(evidence: Alert["evidence"], key: string): number | null {
  const value = evidence[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function clip(text: string, max: number): string {
  if (text.length <= max) return text;
  return `${text.slice(0, max - 1).trimEnd()}…`;
}

/** One line per Javi monitor rule. Same string for line 2 and for bullets. */
export function formatRuleAlert(alert: Alert): string {
  const ev = alert.evidence;
  const money = firstMoney(alert.reasons);

  if (alert.kind === "top_customer_quiet") {
    const open = evNum(ev, "open_receivable_eur");
    const billed = evNum(ev, "last_quarter_amount");
    const share = evNum(ev, "share_last_quarter");
    const parts = ["top_customer_quiet · top customer stopped billing, review exposure and collections"];
    if (share != null && billed != null) parts.push(`${Math.round(share * 100)}% last quarter · ${fmtEur(billed)}`);
    else if (billed != null) parts.push(fmtEur(billed));
    if (open != null) parts.push(`${fmtEur(open)} open`);
    return parts.join(" · ");
  }

  if (alert.kind === "going_dark") {
    const pre = evNum(ev, "score_pre_cap");
    const score = evNum(ev, "score");
    const cap =
      pre != null && score != null
        ? `capped ${Math.round(score)} (uncapped ${Math.round(pre)}) · `
        : pre != null
          ? `capped 30 (uncapped ${Math.round(pre)}) · `
          : "";
    return `going_dark · ${cap}no bank booking 60d`;
  }

  if (alert.kind === "category_drop") {
    const cat = typeof ev.category === "string" ? ev.category : "category";
    const score = evNum(ev, "category_score");
    const base = evNum(ev, "baseline");
    const gap = evNum(ev, "gap_points");
    const head =
      score != null && base != null
        ? `${cat} ${Math.round(score)} vs usual ${Math.round(base)}${gap != null ? ` (${signed(gap)})` : ""}`
        : cat;
    return money ? `category_drop · ${head} · ${money}` : `category_drop · ${head}`;
  }

  const score = evNum(ev, "score") ?? evNum(ev, "mean_score");
  const base = evNum(ev, "baseline");
  const gap = evNum(ev, "gap_points");
  const members = typeof ev.members_moving_most === "string" && ev.members_moving_most ? ev.members_moving_most : "";
  const head =
    score != null && base != null
      ? `${Math.round(score)} vs usual ${Math.round(base)}${gap != null ? ` (${signed(gap)} pts)` : ""}`
      : money || alert.action || alert.title;
  const bits = [`${alert.kind} · ${head}`];
  if (money && head !== money) bits.push(money);
  if (members) bits.push(members);
  return bits.join(" · ");
}

function monthOf(detail: CompanyDetail, month: string): MonthRecord | undefined {
  return detail.months.find((m) => m.month === month);
}

function scoreAt(row: CompanyIndexRow | undefined, months: string[], month: string): number | null {
  if (!row) return null;
  const i = months.indexOf(month);
  if (i < 0) return null;
  return row.scores[i] ?? null;
}

export function buildWatcherPost(input: {
  month: string;
  priorMonth: string | null;
  asOfMonths: string[];
  companyIds: string[];
  groupIds: string[];
  companies: CompanyIndexRow[];
  details: Map<string, CompanyDetail>;
  groups: GroupRow[];
  alerts: Alert[];
}): WatcherPost {
  const { month, priorMonth, asOfMonths, companyIds, groupIds, companies, details, groups, alerts } = input;

  const watchedGroups = groups.filter((g) => groupIds.includes(g.group_id));
  const memberIds = new Set(companyIds);
  for (const g of watchedGroups) {
    for (const id of g.company_ids) memberIds.add(id);
  }

  const monthAlerts = alerts.filter(
    (a) => a.month === month && (memberIds.has(a.entity.id) || groupIds.includes(a.entity.id)),
  );
  const actWatch = monthAlerts.filter((a) => a.severity === "act" || a.severity === "watch");
  const opportunities = monthAlerts.filter((a) => a.direction === "opportunity" && a.severity !== "info");
  const nInfo = monthAlerts.filter((a) => a.severity === "info").length;

  const companySnaps = [...memberIds]
    .map((id) => {
      const detail = details.get(id);
      const rec = detail ? monthOf(detail, month) : undefined;
      const prior = detail && priorMonth ? monthOf(detail, priorMonth) : undefined;
      const index = companies.find((c) => c.company_id === id);
      const score = rec?.score ?? scoreAt(index, asOfMonths, month);
      const priorScore = prior?.score ?? (priorMonth ? scoreAt(index, asOfMonths, priorMonth) : null);
      return {
        id,
        rec,
        score,
        priorScore,
        delta: score != null && priorScore != null ? score - priorScore : null,
        trajectory: rec?.trajectory ?? index?.trajectory ?? "stable",
        guard: rec?.guard ?? index?.guard ?? null,
        pre: rec?.score_pre_cap,
        confidence: rec?.confidence ?? index?.confidence ?? "medium",
        note: rec?.confidence_note,
      };
    })
    .filter((c) => c.score != null);

  const focus = [...companySnaps].sort((a, b) => {
    const da = a.delta ?? 0;
    const db = b.delta ?? 0;
    const rank = (t: string) => (t === "deteriorating" ? 0 : t === "dip" ? 1 : 2);
    return rank(a.trajectory) - rank(b.trajectory) || da - db;
  })[0];

  let line1: string;
  if (companyIds.length === 1 && groupIds.length === 0) {
    const c = companySnaps.find((x) => x.id === companyIds[0]) ?? focus;
    const guard = c?.guard ? `${c.guard} · ` : "";
    const d = signed(c?.delta);
    line1 = `${guard}${c?.id ?? companyIds[0]} · ${Math.round(c?.score ?? 0)}  ${c?.trajectory ?? "stable"}${d ? `  (${d})` : ""}`;
  } else if (groupIds.length === 1 && companyIds.length === 0) {
    const g = watchedGroups[0];
    const i = asOfMonths.indexOf(month);
    const pi = priorMonth ? asOfMonths.indexOf(priorMonth) : -1;
    const mean = i >= 0 ? g?.mean_scores[i] ?? null : null;
    const prior = pi >= 0 ? g?.mean_scores[pi] ?? null : null;
    const delta = mean != null && prior != null ? mean - prior : null;
    line1 = `${g?.group_id ?? groupIds[0]} · ${Math.round(mean ?? 0)}  ${groupState(delta)}${signed(delta) ? `  (${signed(delta)})` : ""}`;
  } else {
    const scores = companySnaps.map((c) => c.score!).filter((n) => n != null);
    const mean = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;
    const deltas = companySnaps.map((c) => c.delta).filter((n): n is number => n != null);
    const dMean = deltas.length ? deltas.reduce((a, b) => a + b, 0) / deltas.length : null;
    const n = companyIds.length + groupIds.length;
    line1 = `${n} entities · mean ${Math.round(mean)}  ${groupState(dMean)}${signed(dMean) ? `  (${signed(dMean)})` : ""}`;
  }

  let line2 = "no material move";
  const topAlert = actWatch[0] ?? opportunities[0];
  if (focus?.guard && focus.pre != null) {
    line2 = `capped ${Math.round(focus.score ?? 0)} (uncapped ${Math.round(focus.pre)}) · ${
      focus.guard === "dark" ? "no bank booking 60d" : "inflows faded vs own history"
    }`;
  } else if (topAlert) {
    line2 = clip(formatRuleAlert(topAlert), 120);
  } else if (focus?.rec) {
    const fact = firstMoney(focus.rec.change_reasons) ?? firstMoney(focus.rec.reasons);
    const noInv = /no invoice/i.test(focus.note ?? "");
    const prefix = noInv ? "no invoices · " : focus.confidence === "low" ? "low confidence · " : "";
    line2 = fact
      ? `${prefix}${focus.id} ${focus.trajectory} · ${fact}`
      : `${prefix}${focus.id} ${focus.trajectory}`;
  }

  const bullets: WatcherBullet[] = [];
  for (const a of [...actWatch, ...opportunities]) {
    if (bullets.length >= 4) break;
    const severity: BulletSeverity =
      a.direction === "opportunity" ? "opportunity" : a.severity === "act" ? "act" : "watch";
    bullets.push({
      severity,
      owner: ownerOf(a),
      entity: a.entity.id,
      text: formatRuleAlert(a),
    });
  }
  if (bullets.length === 0 && focus) {
    const fact = firstMoney(focus.rec?.change_reasons) ?? firstMoney(focus.rec?.reasons);
    const owner: WatcherBullet["owner"] =
      /receivable|customer|collect/i.test(fact ?? "") ? "Collections" : "Treasurer";
    bullets.push({
      severity: "follow",
      owner,
      entity: focus.id,
      text: fact ?? `${focus.trajectory} · ${Math.round(focus.score ?? 0)}`,
    });
    const second = companySnaps.find((c) => c.id !== focus.id && (c.trajectory === "deteriorating" || c.trajectory === "dip"));
    if (second && bullets.length < 4) {
      bullets.push({
        severity: "follow",
        owner: "CFO",
        entity: second.id,
        text: `${second.trajectory} · ${Math.round(second.score ?? 0)}`,
      });
    }
  }

  const sparkPoints = focus
    ? (details.get(focus.id)?.months ?? [])
        .filter((m) => m.month <= month)
        .slice(-6)
        .map((m) => ({ month: m.month, score: m.score }))
    : [];

  return {
    month,
    line1,
    line2,
    bullets: bullets.slice(0, 4),
    n_info: nInfo,
    spark: focus && sparkPoints.length >= 2 ? { id: focus.id, points: sparkPoints } : undefined,
  };
}

export function buildWatcherPosts(input: {
  asOf: string;
  asOfMonths: string[];
  companyIds: string[];
  groupIds: string[];
  companies: CompanyIndexRow[];
  details: Map<string, CompanyDetail>;
  groups: GroupRow[];
  alerts: Alert[];
}): WatcherPost[] {
  const months = lastMonths(input.asOf, 3);
  return months.map((month, i) =>
    buildWatcherPost({
      ...input,
      month,
      priorMonth: i > 0 ? months[i - 1] : monthAdd(month, -1),
    }),
  );
}
