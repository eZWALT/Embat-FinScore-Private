import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";

import type {
  AlertFeed,
  CompanyDetail,
  CompanyIndex,
  GroupIndex,
  Manifest,
  ScoreRepository,
} from "./types";

const COMPANY_ID = /^COMP_[0-9]{4}$/;

function defaultBundleDirectory(): string {
  if (process.env.SCORE_BUNDLE_DIR) return process.env.SCORE_BUNDLE_DIR;
  const candidates = [
    path.resolve(process.cwd(), "../score/sample_bundle"),
    path.resolve(process.cwd(), "../../product/score/sample_bundle"),
    path.resolve(process.cwd(), "sample_bundle"),
  ];
  for (const dir of candidates) {
    if (existsSync(path.join(dir, "manifest.json"))) return dir;
  }
  return candidates[0];
}

export class LocalBundleRepository implements ScoreRepository {
  constructor(private readonly bundleDirectory = defaultBundleDirectory()) {}

  getManifest(): Promise<Manifest> {
    return this.readJson<Manifest>("manifest.json");
  }

  async listCompanies() {
    const index = await this.readJson<CompanyIndex>("companies.json");
    return index.companies;
  }

  getCompany(companyId: string): Promise<CompanyDetail> {
    if (!COMPANY_ID.test(companyId)) {
      throw new Error(`Invalid company id: ${companyId}`);
    }

    return this.readJson<CompanyDetail>(path.join("companies", `${companyId}.json`));
  }

  async listGroups() {
    const index = await this.readJson<GroupIndex>("groups.json");
    return index.groups;
  }

  getAlerts(): Promise<AlertFeed> {
    return this.readJson<AlertFeed>("alerts.json");
  }

  private async readJson<T>(relativePath: string): Promise<T> {
    const contents = await readFile(path.join(this.bundleDirectory, relativePath), "utf8");
    return JSON.parse(contents) as T;
  }
}
