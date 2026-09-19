"""LLM provider for the POC: Helmcode (OpenAI-compatible) through LangChain.

Configuration, all by environment variable:
  HELMCODE_API_KEY        the key (required), or
  HELMCODE_API_KEY_FILE   path to a file whose first line is the key
  HELMCODE_BASE_URL       default https://api.helmcode.com/v1
  POC_LLM_MODEL           default deepseek-v4-flash (tool calling supported)
  POC_LLM_TEMPERATURE     default 0.2
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_openai import ChatOpenAI

DEFAULT_BASE_URL = "https://api.helmcode.com/v1"
DEFAULT_MODEL = "deepseek-v4-flash"


def api_key() -> str | None:
    key = os.environ.get("HELMCODE_API_KEY")
    if key:
        return key.strip()
    path = os.environ.get("HELMCODE_API_KEY_FILE")
    if path and Path(path).expanduser().exists():
        return Path(path).expanduser().read_text().splitlines()[0].strip()
    return None


def model_name() -> str:
    return os.environ.get("POC_LLM_MODEL", DEFAULT_MODEL)


def get_llm(temperature: float | None = None) -> ChatOpenAI:
    key = api_key()
    if not key:
        raise RuntimeError("No LLM key: set HELMCODE_API_KEY or HELMCODE_API_KEY_FILE.")
    return ChatOpenAI(
        model=model_name(),
        base_url=os.environ.get("HELMCODE_BASE_URL", DEFAULT_BASE_URL),
        api_key=key,
        temperature=float(os.environ.get("POC_LLM_TEMPERATURE", "0.2")) if temperature is None else temperature,
        timeout=120,
        max_retries=2,
    )
