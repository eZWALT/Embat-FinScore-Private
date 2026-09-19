import type { Metadata } from "next";

import { GroupHealthMap } from "@/components/group-health-map";
import { getGroupMapData } from "@/lib/data/group-service";

export const metadata: Metadata = {
  title: "Mapa de grupos · Health Sentinel",
  description: "Score de cada empresa del grupo, mes a mes, con sus razones y alertas.",
};

type SearchParams = Promise<Record<string, string | string[] | undefined>>;

function first(value: string | string[] | undefined) {
  return Array.isArray(value) ? value[0] : value;
}

export default async function GruposPage({ searchParams }: { searchParams: SearchParams }) {
  const params = await searchParams;
  const data = await getGroupMapData(first(params.group), first(params.company));

  return <GroupHealthMap data={data} />;
}
