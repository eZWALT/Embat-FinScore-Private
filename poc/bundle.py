"""Read-only access to an export bundle (schema 1.x, product/score/DATA_CONTRACT.md).

Bundle directory resolution: POC_BUNDLE_DIR, then data/bundle (full, generated),
then product/score/sample_bundle (12 companies, committed).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
CANDIDATES = [
    os.environ.get("POC_BUNDLE_DIR"),
    REPO / "data" / "bundle",
    REPO / "product" / "score" / "sample_bundle",
]


def bundle_dir() -> Path:
    for c in CANDIDATES:
        if c and (Path(c) / "manifest.json").exists():
            return Path(c)
    raise FileNotFoundError(
        "No bundle found. Run: PYTHONUTF8=1 PYTHONPATH=. python -m product.score.export --csv-folder data --out data/bundle"
    )


class Bundle:
    def __init__(self, root: Path | None = None):
        self.root = root or bundle_dir()

    def _read(self, rel: str) -> Any:
        with open(self.root / rel, encoding="utf-8") as f:
            return json.load(f)

    @property
    def manifest(self) -> dict:
        return self._read("manifest.json")

    @property
    def companies(self) -> list[dict]:
        return self._read("companies.json")["companies"]

    @property
    def months(self) -> list[str]:
        return self._read("companies.json")["months"]

    @property
    def groups(self) -> list[dict]:
        return self._read("groups.json")["groups"]

    @property
    def alerts_feed(self) -> dict:
        return self._read("alerts.json")

    @property
    def alerts(self) -> list[dict]:
        return self.alerts_feed["alerts"]

    @property
    def clusters(self) -> dict:
        return self._read("clusters.json")

    def company(self, company_id: str) -> dict:
        p = self.root / "companies" / f"{company_id}.json"
        if not p.exists():
            raise KeyError(f"{company_id} is not in the bundle")
        return self._read(f"companies/{company_id}.json")

    # ---- indexes --------------------------------------------------------
    def company_row(self, company_id: str) -> dict | None:
        return next((c for c in self.companies if c["company_id"] == company_id), None)

    def group(self, group_id: str) -> dict | None:
        return next((g for g in self.groups if g["group_id"] == group_id), None)

    def alerts_by_id(self) -> dict[str, dict]:
        return {a["alert_id"]: a for a in self.alerts}

    def alerts_for(self, entity_id: str) -> list[dict]:
        return [a for a in self.alerts if a["entity"]["id"] == entity_id]

    def company_ids(self) -> list[str]:
        return [c["company_id"] for c in self.companies]

    def group_ids(self) -> list[str]:
        return [g["group_id"] for g in self.groups]


@lru_cache(maxsize=1)
def load() -> Bundle:
    return Bundle()
