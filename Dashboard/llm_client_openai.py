
from __future__ import annotations

import json
import os
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))

_CLIENT: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def is_llm_configured() -> bool:
    return bool(_CLIENT)


def ask_openai_text(prompt: str, fallback: str = "") -> str:
    if not _CLIENT:
        return fallback
    try:
        response = _CLIENT.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            temperature=OPENAI_TEMPERATURE,
        )
        text = (response.output_text or "").strip()
        return text or fallback
    except Exception:
        return fallback


def ask_openai_json(prompt: str) -> Any:
    if not _CLIENT:
        return None
    try:
        response = _CLIENT.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            temperature=0,
            text={"format": {"type": "json_object"}},
        )
        text = (response.output_text or "").strip()
        return json.loads(text) if text else None
    except Exception:
        return None
