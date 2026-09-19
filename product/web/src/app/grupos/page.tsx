import { redirect } from "next/navigation";

type SearchParams = Promise<Record<string, string | string[] | undefined>>;

function first(value: string | string[] | undefined) {
  return Array.isArray(value) ? value[0] : value;
}

/** Groups live in the deep view now. Old links keep working. */
export default async function GruposRedirect({ searchParams }: { searchParams: SearchParams }) {
  const params = await searchParams;
  const query = new URLSearchParams({ modo: "profundo" });
  const group = first(params.group);
  const company = first(params.company);
  if (group) query.set("group", group);
  else if (company) query.set("company", company);
  redirect(`/?${query}`);
}
