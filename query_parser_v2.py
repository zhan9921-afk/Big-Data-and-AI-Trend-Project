"""
query_parser.py

Production-friendly query parsing layer for a Streamlit query box.

What this file does:
- Parse common manager-style text queries with rules
- Normalize output into one consistent dict schema
- Merge parsed query results with structured UI filters
- Stay compatible with a future LLM parser

What this file does NOT do:
- No database access
- No LLM API calls
- No ranking logic

Recommended architecture:
query box text -> parser -> normalized filters dict -> recommendation_engine -> results
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

DEFAULT_OBJECTIVE = "balanced"
SAFE_MIN_GAMES = 12

PARSED_QUERY_TEMPLATE = {
    "position": None,
    "max_salary": None,
    "objective": DEFAULT_OBJECTIVE,
    "min_games": None,
    "season": None,
    "compare_mode": False,
    "raw_query": None,
    "parse_method": None,
    "parse_confidence": None,
}


# ------------------------------------------------------------
# Pattern libraries
# ------------------------------------------------------------

POSITION_PATTERNS = {
    "QB": [
        r"\bqb\b",
        r"\bquarterback\b",
        r"\bquarterbacks\b",
    ],
    "RB": [
        r"\brb\b",
        r"\brunning\s*back\b",
        r"\brunning\s*backs\b",
    ],
    "WR": [
        r"\bwr\b",
        r"\bwide\s*receiver\b",
        r"\bwide\s*receivers\b",
        r"\bwideout\b",
        r"\bwideouts\b",
    ],
    "TE": [
        r"\bte\b",
        r"\btight\s*end\b",
        r"\btight\s*ends\b",
        r"\btightend\b",
        r"\btightends\b",
        r"\btes\b",
    ],
}

UNDERVALUE_PATTERNS = [
    r"\bundervalued\b",
    r"\bvalue\b",
    r"\bcheap\b",
    r"\baffordable\b",
    r"\bbudget[- ]friendly\b",
    r"\bbudget friendly\b",
    r"\bvalue[- ]for[- ]money\b",
]

OVERVALUE_PATTERNS = [
    r"\bovervalued\b",
    r"\boverpriced\b",
    r"\btoo expensive\b",
]

LOW_RISK_PATTERNS = [
    r"\blow risk\b",
    r"\bsafe\b",
    r"\bsafer\b",
    r"\bsafest\b",
    r"\breliable\b",
]

COMPARE_PATTERNS = [
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\boptions\b",
    r"\btargets\b",
    r"\bchoices\b",
]

SEASON_PATTERNS = [
    r"\b(?:season\s*)?(20\d{2})\b",
]

# Supports:
# under $6M
# below 8M
# <= 6M
# under 6000000
# below $5000000
SALARY_PATTERNS_M = [
    r"(?:under|below|less than)\s*\$?\s*(\d+(?:\.\d+)?)\s*m\b",
    r"(?:<=|=<)\s*\$?\s*(\d+(?:\.\d+)?)\s*m\b",
    r"\$?\s*(\d+(?:\.\d+)?)\s*m\s*(?:or less|max|maximum)?\b",
]

SALARY_PATTERNS_RAW = [
    r"(?:under|below|less than)\s*\$?\s*(\d{1,9})(?!\s*m)\b",
    r"(?:<=|=<)\s*\$?\s*(\d{1,9})(?!\s*m)\b",
]


# ------------------------------------------------------------
# Normalization helpers
# ------------------------------------------------------------

def normalize_text(query: str) -> str:
    return " ".join((query or "").strip().lower().split())


def empty_parse(raw_query: str = "") -> Dict[str, Any]:
    out = PARSED_QUERY_TEMPLATE.copy()
    out["raw_query"] = raw_query
    return out


# ------------------------------------------------------------
# Extractors
# ------------------------------------------------------------

def extract_position(query: str) -> Optional[str]:
    for position, patterns in POSITION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, query, flags=re.IGNORECASE):
                return position
    return None


def extract_objective(query: str) -> str:
    for pattern in OVERVALUE_PATTERNS:
        if re.search(pattern, query, flags=re.IGNORECASE):
            return "overvalued"

    for pattern in UNDERVALUE_PATTERNS:
        if re.search(pattern, query, flags=re.IGNORECASE):
            return "undervalued"

    return DEFAULT_OBJECTIVE


def extract_max_salary(query: str) -> Optional[float]:
    """
    Returns salary in whole dollars.

    Examples:
    under $6M  -> 6000000
    below 8M   -> 8000000
    <= 5000000 -> 5000000
    """
    for pattern in SALARY_PATTERNS_M:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return float(match.group(1)) * 1_000_000

    for pattern in SALARY_PATTERNS_RAW:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))

    return None


def extract_min_games(query: str) -> Optional[int]:
    for pattern in LOW_RISK_PATTERNS:
        if re.search(pattern, query, flags=re.IGNORECASE):
            return SAFE_MIN_GAMES
    return None


def extract_compare_mode(query: str) -> bool:
    for pattern in COMPARE_PATTERNS:
        if re.search(pattern, query, flags=re.IGNORECASE):
            return True
    return False


def extract_season(query: str) -> Optional[int]:
    for pattern in SEASON_PATTERNS:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            year = int(match.group(1))
            if 2000 <= year <= 2099:
                return year
    return None


# ------------------------------------------------------------
# Confidence logic
# ------------------------------------------------------------

def estimate_parse_confidence(parsed: Dict[str, Any]) -> str:
    hits = 0
    for key in ["position", "max_salary", "min_games", "season"]:
        if parsed.get(key) is not None:
            hits += 1

    if parsed.get("objective") != DEFAULT_OBJECTIVE:
        hits += 1

    if hits >= 3:
        return "high"
    if hits >= 1:
        return "medium"
    return "low"


# ------------------------------------------------------------
# Main parser
# ------------------------------------------------------------

def parse_query_rule_based(query: str) -> Dict[str, Any]:
    """
    Parse a manager-style text query into a normalized dict.

    Example output:
    {
        "position": "RB",
        "max_salary": 6000000,
        "objective": "undervalued",
        "min_games": 12,
        "season": None,
        "compare_mode": False,
        "raw_query": "Find undervalued RB under $6M",
        "parse_method": "rule_based",
        "parse_confidence": "high"
    }
    """
    q = normalize_text(query)

    parsed = empty_parse(raw_query=query)
    parsed["position"] = extract_position(q)
    parsed["max_salary"] = extract_max_salary(q)
    parsed["objective"] = extract_objective(q)
    parsed["min_games"] = extract_min_games(q)
    parsed["season"] = extract_season(q)
    parsed["compare_mode"] = extract_compare_mode(q)
    parsed["parse_method"] = "rule_based"
    parsed["parse_confidence"] = estimate_parse_confidence(parsed)

    return parsed


# ------------------------------------------------------------
# UI / downstream helpers
# ------------------------------------------------------------

def merge_query_with_ui_filters(
    parsed_query: Dict[str, Any],
    ui_filters: Optional[Dict[str, Any]] = None,
    query_priority: bool = True,
) -> Dict[str, Any]:
    """
    Merge parsed query output with structured UI filters.

    If query_priority=True:
        parsed query values override UI values when present

    If query_priority=False:
        UI values override parsed query values when present
    """
    ui_filters = ui_filters or {}

    merged = {
        "position": None,
        "max_salary": None,
        "objective": DEFAULT_OBJECTIVE,
        "min_games": None,
        "season": None,
    }

    keys = list(merged.keys())

    for key in keys:
        query_value = parsed_query.get(key)
        ui_value = ui_filters.get(key)

        if query_priority:
            merged[key] = query_value if query_value is not None else ui_value
        else:
            merged[key] = ui_value if ui_value is not None else query_value

    return merged


def should_fallback_to_llm(parsed_query: Dict[str, Any]) -> bool:
    """
    Simple routing helper.

    Recommended use:
    - If confidence is low, ask an LLM to produce the same output schema
    - Otherwise keep the fast rule-based result
    """
    return parsed_query.get("parse_confidence") == "low"


def build_llm_parser_prompt(query: str) -> str:
    """
    Optional helper for your future LLM parser.

    This returns a prompt that asks an LLM to emit the same schema
    used by the rule-based parser, so your downstream pipeline stays unchanged.
    """
    return f"""
You are a parser for an NFL contract recommendation app.

Convert the user query into a JSON object with exactly these keys:
- position: one of "QB", "RB", "WR", "TE", or null
- max_salary: integer dollar amount or null
- objective: one of "undervalued", "overvalued", "balanced"
- min_games: integer or null
- season: integer or null
- compare_mode: boolean
- raw_query: original query string
- parse_method: "llm"
- parse_confidence: one of "high", "medium", "low"

Rules:
- phrases like cheap / affordable / budget-friendly / value mean "undervalued"
- phrases like safe / low risk / reliable imply min_games = 12
- return max_salary in whole dollars, so 6M becomes 6000000
- if a field is not clearly stated, use null
- output JSON only

User query:
{query}
""".strip()


# ------------------------------------------------------------
# Recommended public entry point
# ------------------------------------------------------------

def parse_query(query: str) -> Dict[str, Any]:
    """
    Current default parser entry point.
    Today: rule-based
    Future: can become hybrid without changing app.py usage
    """
    return parse_query_rule_based(query)


if __name__ == "__main__":
    examples = [
        "Find undervalued RB under $6M",
        "Show cheap WR options",
        "Find safe QB under 10M",
        "Compare budget-friendly TEs",
        "Find affordable tight ends below 8M",
        "Show overvalued WR under 12000000",
        "Need some good players who are not too expensive",
    ]

    for example in examples:
        parsed = parse_query(example)
        print(example)
        print(parsed)
        print("fallback_to_llm:", should_fallback_to_llm(parsed))
        print("-" * 80)
