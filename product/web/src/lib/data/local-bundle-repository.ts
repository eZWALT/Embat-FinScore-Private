import { readFile } from "node:fs/promises";
import path from "node:path";

import type {
  CompanyDetail,
  CompanyIndex,
  Manifest,
  ScoreRepository,
} from "./types";

const COMPANY_ID = /^COMP_[0-9]{4}$/;

export class LocalBundleRepository implements ScoreRepository {
  constructor(
    private readonly bundleDirectory =
      process.env.SCORE_BUNDLE_DIR ??
      path.resolve(process.cwd(), "../score/sample_bundle"),
  ) {}

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

  private async readJson<T>(relativePath: string): Promise<T> {
    const contents = await readFile(path.join(this.bundleDirectory, relativePath), "utf8");
    return JSON.parse(contents) as T;
  }
}
