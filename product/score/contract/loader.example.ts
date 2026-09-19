/**
 * Example loader for the Next.js app (App Router, server side). Not part of the contract, just the intended usage.
 *
 * BUNDLE_DIR=./data/bundle        -> read from disk at build time (bundle copied into the repo / build step)
 * BUNDLE_URL=https://.../bundle   -> fetch from static hosting (Vercel Blob, S3...) when the bundle is too big for the deploy
 * Copy types.ts next to this file (it ships inside every bundle).
 */
import { promises as fs } from "fs";
import path from "path";
import type { CompanyDetail, CompanyIndex, GroupIndex, Manifest } from "./types";

async function read<T>(rel: string): Promise<T> {
  const url = process.env.BUNDLE_URL;
  if (url) {
    const res = await fetch(`${url}/${rel}`, { next: { revalidate: false } }); // bundle is immutable: cache forever
    if (!res.ok) throw new Error(`${rel}: ${res.status}`);
    return (await res.json()) as T;
  }
  const dir = process.env.BUNDLE_DIR ?? "./data/bundle";
  return JSON.parse(await fs.readFile(path.join(process.cwd(), dir, rel), "utf8")) as T;
}

export const getManifest = () => read<Manifest>("manifest.json");
export const getCompanyIndex = () => read<CompanyIndex>("companies.json");
export const getGroupIndex = () => read<GroupIndex>("groups.json");
export const getCompany = (id: string) => read<CompanyDetail>(`companies/${id}.json`);

/** Refuse a bundle from a different major version instead of rendering garbage. */
export async function assertSchema(): Promise<Manifest> {
  const m = await getManifest();
  if (!m.schema_version.startsWith("1.")) throw new Error(`unsupported schema ${m.schema_version}`);
  return m;
}

// app/company/[id]/page.tsx:
//   export async function generateStaticParams() { return (await getCompanyIndex()).companies.map(c => ({ id: c.company_id })); }
//   export default async function Page({ params }: { params: { id: string } }) { const c = await getCompany(params.id); ... }
