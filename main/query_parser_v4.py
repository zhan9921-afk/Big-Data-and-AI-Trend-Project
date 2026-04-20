from __future__ import annotations

import re
from difflib import get_close_matches
from typing import Any, Dict, Iterable, Optional

DEFAULT_OBJECTIVE = "balanced"
SAFE_MIN_GAMES = 12

TEAM_ALIASES = {
    "49ers": "SF 49ers",
    "niners": "SF 49ers",
    "sf": "SF 49ers",
    "san francisco": "SF 49ers",
    "chiefs": "KC Chiefs",
    "kc": "KC Chiefs",
    "kansas city": "KC Chiefs",
    "eagles": "PHI Eagles",
    "philly": "PHI Eagles",
    "philadelphia": "PHI Eagles",
    "cowboys": "DAL Cowboys",
    "dallas": "DAL Cowboys",
    "bills": "BUF Bills",
    "buffalo": "BUF Bills",
    "bengals": "CIN Bengals",
    "cincinnati": "CIN Bengals",
    "rams": "LAR Rams",
    "la rams": "LAR Rams",
    "packers": "GB Packers",
    "green bay": "GB Packers",
    "vikings": "MIN Vikings",
    "minnesota": "MIN Vikings",
    "ravens": "BAL Ravens",
    "baltimore": "BAL Ravens",
}

PARSED_QUERY_TEMPLATE = {
    "position": None,
    "team": None,
    "player_name": None,
    "max_salary": None,
    "objective": DEFAULT_OBJECTIVE,
    "min_games": None,
    "season": None,
    "compare_mode": False,
    "intent": "shortlist",
    "target_tab": "Shortlist",
    "target_chart": "value_scatter",
    "needs_explanation": True,
    "raw_query": None,
    "parse_method": None,
    "parse_confidence": None,
}

POSITION_PATTERNS = {
    "QB": [r"\bqb\b", r"\bquarterback\b", r"\bquarterbacks\b"],
    "RB": [r"\brb\b", r"\brunning\s*back\b", r"\brunning\s*backs\b", r"\bbackfield\b"],
    "WR": [r"\bwr\b", r"\bwide\s*receiver\b", r"\bwide\s*receivers\b", r"\bwideout\b", r"\breceiver room\b"],
    "TE": [r"\bte\b", r"\btight\s*end\b", r"\btightends?\b", r"\btes\b"],
}

UNDERVALUE_PATTERNS = [
    r"\bundervalued\b", r"\bcheap\b", r"\bvalue\b", r"\bvaluable\b", r"\baffordable\b",
    r"\bbudget\b", r"\bbargain\b", r"\bunder market\b", r"\bgood deal\b"
]
OVERVALUE_PATTERNS = [
    r"\bovervalued\b", r"\boverpriced\b", r"\btoo expensive\b", r"\boverpaid\b", r"\bbad contract\b"
]
LOW_RISK_PATTERNS = [r"\blow risk\b", r"\bsafe\b", r"\breliable\b", r"\bstable\b"]
COMPARE_PATTERNS = [r"\bcompare\b", r"\bcomparison\b", r"\bversus\b", r"\bvs\b", r"\bchoices\b", r"\bhead to head\b"]
TEAM_PATTERNS = [r"\bteam\b", r"\broster\b", r"\bmy team\b", r"\broom\b"]
PLAYER_PATTERNS = [r"\bprofile\b", r"\bplayer\b", r"\bheadshot\b", r"\bradar\b", r"\btell me about\b", r"\banalyze\b"]
PLOT_PATTERNS = [r"\bplot\b", r"\bgraph\b", r"\bchart\b", r"\bvisualize\b", r"\bshow me\b", r"\boverview\b"]
SEASON_PATTERNS = [r"\b(?:season\s*)?(20\d{2})\b"]
TEAM_GAP_PATTERNS = [r"\bteam value gap\b", r"\bvalue gaps by team\b", r"\bteam gaps\b", r"\bleague team overview\b"]
POSITION_GAP_PATTERNS = [r"\bposition gap\b", r"\bvalue by position\b", r"\bposition chart\b", r"\bposition overview\b"]
ROSTER_PATTERNS = [r"\broster\b", r"\bteam roster\b", r"\bmy roster\b"]
PLAYER_PROFILE_PATTERNS = [r"\bplayer profile\b", r"\bprofile for\b", r"\bshow profile\b", r"\bheadshot\b", r"\bradar\b"]

SALARY_PATTERNS_M = [
    r"(?:under|below|less than|cheaper than)\s*\$?\s*(\d+(?:\.\d+)?)\s*m\b",
    r"(?:<=|=<)\s*\$?\s*(\d+(?:\.\d+)?)\s*m\b",
    r"\$?\s*(\d+(?:\.\d+)?)\s*m\s*(?:or less|max|maximum)?\b",
]
SALARY_PATTERNS_RAW = [
    r"(?:under|below|less than)\s*\$?\s*(\d{1,9})(?!\s*m)\b",
    r"(?:<=|=<)\s*\$?\s*(\d{1,9})(?!\s*m)\b",
]


def normalize_text(query: str) -> str:
    return " ".join((query or "").strip().lower().split())


def empty_parse(raw_query: str = "") -> Dict[str, Any]:
    out = PARSED_QUERY_TEMPLATE.copy()
    out["raw_query"] = raw_query
    return out


def _contains_any(query: str, patterns: Iterable[str]) -> bool:
    return any(re.search(p, query, flags=re.IGNORECASE) for p in patterns)


def extract_position(query: str) -> Optional[str]:
    for position, patterns in POSITION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, query, flags=re.IGNORECASE):
                return position
    return None


def extract_objective(query: str) -> str:
    if _contains_any(query, OVERVALUE_PATTERNS):
        return "overvalued"
    if _contains_any(query, UNDERVALUE_PATTERNS):
        return "undervalued"
    return DEFAULT_OBJECTIVE


def extract_max_salary(query: str) -> Optional[float]:
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
    return SAFE_MIN_GAMES if _contains_any(query, LOW_RISK_PATTERNS) else None


def extract_compare_mode(query: str) -> bool:
    return _contains_any(query, COMPARE_PATTERNS)


def extract_season(query: str) -> Optional[int]:
    for pattern in SEASON_PATTERNS:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            year = int(match.group(1))
            if 2000 <= year <= 2099:
                return year
    return None


def extract_team(query: str, team_aliases: Optional[Dict[str, str]] = None) -> Optional[str]:
    team_aliases = team_aliases or TEAM_ALIASES
    q = normalize_text(query)
    for alias, team_name in team_aliases.items():
        if re.search(rf"\b{re.escape(alias.lower())}\b", q, flags=re.IGNORECASE):
            return team_name
    return None


def detect_player_from_query(query: str, available_names: Iterable[str]) -> Optional[str]:
    query_norm = normalize_text(query)
    names = [str(n) for n in available_names if str(n).strip()]
    if not names:
        return None

    exact_hits = [n for n in names if normalize_text(n) in query_norm]
    if exact_hits:
        return exact_hits[0]

    token_hits = []
    query_words = set(query_norm.split())
    for name in names:
        name_words = set(normalize_text(name).split())
        if len(name_words) >= 2 and len(name_words & query_words) >= 2:
            token_hits.append(name)
    if token_hits:
        return token_hits[0]

    close = get_close_matches(query.strip(), names, n=1, cutoff=0.80)
    return close[0] if close else None


def infer_intent_and_routing(query: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
    q = normalize_text(query)
    intent = "shortlist"
    tab = "Shortlist"
    chart = "value_scatter"

    if parsed.get("compare_mode"):
        intent, tab, chart = "compare", "Compare", "value_scatter"
    elif parsed.get("player_name") or _contains_any(q, PLAYER_PROFILE_PATTERNS):
        intent, tab, chart = "player_analysis", "Player Analysis", "player_radar"
    elif parsed.get("team") or _contains_any(q, TEAM_PATTERNS):
        intent, tab, chart = "team_analysis", "Team Analysis", "roster"
    elif _contains_any(q, TEAM_GAP_PATTERNS):
        intent, tab, chart = "smart_query", "Smart Query", "team_gap"
    elif _contains_any(q, POSITION_GAP_PATTERNS):
        intent, tab, chart = "smart_query", "Smart Query", "position_gap"
    elif _contains_any(q, ROSTER_PATTERNS) and parsed.get("team"):
        intent, tab, chart = "team_analysis", "Team Analysis", "roster"
    elif _contains_any(q, PLOT_PATTERNS):
        intent, tab = "smart_query", "Smart Query"
        if parsed.get("team"):
            chart = "roster"
        elif parsed.get("position") or re.search(r"\bposition\b", q):
            chart = "position_gap"
        else:
            chart = "value_scatter"

    return {
        "intent": intent,
        "target_tab": tab,
        "target_chart": chart,
        "needs_explanation": True,
    }


def estimate_parse_confidence(parsed: Dict[str, Any]) -> str:
    hits = 0
    for key in ["position", "team", "player_name", "max_salary", "min_games", "season"]:
        if parsed.get(key) is not None:
            hits += 1
    if parsed.get("objective") != DEFAULT_OBJECTIVE:
        hits += 1
    if parsed.get("intent") != "shortlist":
        hits += 1
    if hits >= 4:
        return "high"
    if hits >= 2:
        return "medium"
    return "low"


def parse_query(
    query: str,
    team_aliases: Optional[Dict[str, str]] = None,
    available_player_names: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    q = normalize_text(query)
    parsed = empty_parse(raw_query=query)
    parsed["position"] = extract_position(q)
    parsed["team"] = extract_team(q, team_aliases)
    parsed["max_salary"] = extract_max_salary(q)
    parsed["objective"] = extract_objective(q)
    parsed["min_games"] = extract_min_games(q)
    parsed["season"] = extract_season(q)
    parsed["compare_mode"] = extract_compare_mode(q)

    if available_player_names is not None:
        parsed["player_name"] = detect_player_from_query(query, available_player_names)

    parsed.update(infer_intent_and_routing(q, parsed))

    # Free-text fallback: always return a safe, renderable route.
    if not parsed.get("target_tab"):
        parsed["target_tab"] = "Smart Query"
    if not parsed.get("target_chart"):
        parsed["target_chart"] = "value_scatter"
    if not parsed.get("intent"):
        parsed["intent"] = "shortlist"

    parsed["parse_method"] = "rule_based_v4_free_text"
    parsed["parse_confidence"] = estimate_parse_confidence(parsed)
    return parsed


def merge_query_with_ui_filters(
    parsed_query: Dict[str, Any],
    ui_filters: Optional[Dict[str, Any]] = None,
    query_priority: bool = True,
) -> Dict[str, Any]:
    ui_filters = ui_filters or {}
    merged = {
        "position": None,
        "team": None,
        "max_salary": None,
        "objective": DEFAULT_OBJECTIVE,
        "min_games": None,
        "season": None,
    }
    for key in merged:
        q_val = parsed_query.get(key)
        ui_val = ui_filters.get(key)
        if query_priority:
            merged[key] = q_val if q_val is not None else ui_val
        else:
            merged[key] = ui_val if ui_val is not None else q_val
    return merged


if __name__ == "__main__":
    examples = [
        "show me undervalued wrs under 12m",
        "compare two cheap rbs",
        "player profile for joe burrow",
        "show chiefs roster",
        "which qbs are overpaid",
        "team value gap chart",
        "position overview for wr",
    ]
    for q in examples:
        print(q)
        print(parse_query(q))
        print("-" * 80)
