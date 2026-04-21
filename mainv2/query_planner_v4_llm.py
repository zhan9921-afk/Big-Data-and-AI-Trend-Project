from __future__ import annotations

import json
import re
from difflib import get_close_matches
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from llm_client_openai import ask_openai_json

DEFAULT_OBJECTIVE = "balanced"
SAFE_MIN_GAMES = 12

TEAM_ALIASES = {
    "49ers": "San Francisco 49ers",
    "niners": "San Francisco 49ers",
    "sf": "San Francisco 49ers",
    "san francisco": "San Francisco 49ers",
    "chiefs": "Kansas City Chiefs",
    "kc": "Kansas City Chiefs",
    "kansas city": "Kansas City Chiefs",
    "eagles": "Philadelphia Eagles",
    "philly": "Philadelphia Eagles",
    "philadelphia": "Philadelphia Eagles",
    "cowboys": "Dallas Cowboys",
    "dallas": "Dallas Cowboys",
    "bills": "Buffalo Bills",
    "buffalo": "Buffalo Bills",
    "bengals": "Cincinnati Bengals",
    "cincinnati": "Cincinnati Bengals",
    "rams": "Los Angeles Rams",
    "la rams": "Los Angeles Rams",
    "packers": "Green Bay Packers",
    "green bay": "Green Bay Packers",
    "vikings": "Minnesota Vikings",
    "minnesota": "Minnesota Vikings",
    "ravens": "Baltimore Ravens",
    "baltimore": "Baltimore Ravens",
    "cardinals": "Arizona Cardinals",
    "arizona": "Arizona Cardinals",
}

POSITION_PATTERNS = {
    "QB": [r"\bqb\b", r"\bquarterback\b", r"\bquarterbacks\b"],
    "RB": [r"\brb\b", r"\brunning\s*back\b", r"\brunning\s*backs\b", r"\bbackfield\b"],
    "WR": [r"\bwr\b", r"\bwide\s*receiver\b", r"\bwide\s*receivers\b", r"\bwideout\b", r"\breceiver room\b"],
    "TE": [r"\bte\b", r"\btight\s*end\b", r"\btightends?\b", r"\btes\b"],
}

UNDERVALUE_PATTERNS = [r"\bundervalued\b", r"\bcheap\b", r"\bvalue\b", r"\bbudget\b", r"\bbargain\b", r"\bunder market\b"]
OVERVALUE_PATTERNS = [r"\bovervalued\b", r"\boverpriced\b", r"\btoo expensive\b", r"\boverpaid\b", r"\bbad contract\b"]
LOW_RISK_PATTERNS = [r"\blow risk\b", r"\bsafe\b", r"\breliable\b", r"\bstable\b"]
COMPARE_PATTERNS = [r"\bcompare\b", r"\bcomparison\b", r"\bversus\b", r"\bvs\b", r"\bchoices\b", r"\bhead to head\b"]
TEAM_PATTERNS = [r"\bteam\b", r"\broster\b", r"\bmy team\b", r"\broom\b"]
PLAYER_PATTERNS = [r"\bprofile\b", r"\bplayer\b", r"\bheadshot\b", r"\bradar\b", r"\btell me about\b", r"\banalyze\b", r"\bstats\b"]
SIMILAR_PATTERNS = [r"\bsimilar\b", r"\bpeer\b", r"\bpeers\b", r"\blike\b", r"\bcomparables?\b"]
TEAM_GAP_PATTERNS = [r"\bteam value gap\b", r"\bvalue gaps by team\b", r"\bteam gaps\b", r"\bleague team overview\b"]
POSITION_GAP_PATTERNS = [r"\bposition gap\b", r"\bvalue by position\b", r"\bposition chart\b", r"\bposition overview\b"]
EXPIRING_PATTERNS = [r"\bexpiring\b", r"\bexpires\b", r"\bfree agents?\b", r"\bnext contract\b", r"\bre-sign\b"]

SALARY_PATTERNS_M = [
    r"(?:under|below|less than|cheaper than)\s*\$?\s*(\d+(?:\.\d+)?)\s*m\b",
    r"(?:<=|=<)\s*\$?\s*(\d+(?:\.\d+)?)\s*m\b",
    r"\$?\s*(\d+(?:\.\d+)?)\s*m\s*(?:or less|max|maximum)?\b",
]
SEASON_PATTERNS = [r"\b(?:season\s*)?(20\d{2})\b"]


def normalize_text(query: str) -> str:
    return " ".join((query or "").strip().lower().split())


def _contains_any(query: str, patterns: Iterable[str]) -> bool:
    return any(re.search(p, query, flags=re.IGNORECASE) for p in patterns)


def _player_alias_variants(name: str) -> set[str]:
    name = str(name).strip()
    if not name:
        return set()
    tokens = name.replace(".", " ").split()
    variants = {name.lower(), name.lower().replace(".", ""), name.lower().replace(" ", ""), normalize_text(name)}
    if len(tokens) >= 2:
        first, last = tokens[0], tokens[-1]
        variants |= {f"{first[0]}.{last}".lower(), f"{first[0]} {last}".lower(), f"{first}.{last}".lower(), f"{first[0]}{last}".lower(), last.lower()}
    return {v.strip() for v in variants if v.strip()}


def detect_player_from_query(query: str, available_names: Iterable[str]) -> Optional[str]:
    query_norm = normalize_text(query)
    names = [str(n) for n in available_names if str(n).strip()]
    if not names:
        return None
    for name in names:
        for variant in _player_alias_variants(name):
            if re.search(rf"\b{re.escape(variant)}\b", query_norm, flags=re.IGNORECASE):
                return name
    exact_hits = [n for n in names if normalize_text(n) in query_norm]
    if exact_hits:
        return exact_hits[0]
    query_words = set(re.sub(r"[^a-z0-9 ]", " ", query_norm).split())
    token_hits = []
    for name in names:
        name_words = set(re.sub(r"[^a-z0-9 ]", " ", normalize_text(name)).split())
        if len(name_words) >= 2 and len(name_words & query_words) >= 2:
            token_hits.append(name)
    if token_hits:
        return token_hits[0]
    compact_query = query_norm.replace(" ", "")
    for name in names:
        for variant in _player_alias_variants(name):
            if variant.replace(" ", "") in compact_query:
                return name
    close = get_close_matches(query.strip(), names, n=1, cutoff=0.80)
    return close[0] if close else None


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
    return None


def extract_min_games(query: str) -> Optional[int]:
    return SAFE_MIN_GAMES if _contains_any(query, LOW_RISK_PATTERNS) else None


def extract_season(query: str) -> Optional[int]:
    for pattern in SEASON_PATTERNS:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            year = int(match.group(1))
            if 2000 <= year <= 2099:
                return year
    return None


def extract_team(query: str, team_aliases: Optional[Dict[str, str]] = None, available_teams: Optional[Iterable[str]] = None) -> Optional[str]:
    q = normalize_text(query)
    team_aliases = team_aliases or TEAM_ALIASES
    for alias, team_name in team_aliases.items():
        if re.search(rf"\b{re.escape(alias.lower())}\b", q, flags=re.IGNORECASE):
            return team_name
    if available_teams:
        for team in available_teams:
            if normalize_text(team) in q:
                return team
    return None


def infer_rule_plan(query: str, df: pd.DataFrame, team_aliases: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    q = normalize_text(query)
    available_names = df["player_name"].dropna().astype(str).tolist()
    available_teams = df["team"].dropna().astype(str).tolist() if "team" in df.columns else []
    player_name = detect_player_from_query(query, available_names)
    team_name = extract_team(q, team_aliases, available_teams)
    position = extract_position(q)
    objective = extract_objective(q)
    max_salary = extract_max_salary(q)
    min_games = extract_min_games(q)
    season = extract_season(q)

    if player_name:
        player_row = df[df["player_name"] == player_name].head(1)
        if not player_row.empty:
            if not position:
                position = player_row.iloc[0].get("position")
            if not team_name and pd.notna(player_row.iloc[0].get("team")):
                team_name = player_row.iloc[0].get("team")

    compare_mode = _contains_any(q, COMPARE_PATTERNS)
    asks_player = player_name is not None and (_contains_any(q, PLAYER_PATTERNS) or "profile" in q)
    asks_similar = _contains_any(q, SIMILAR_PATTERNS) or compare_mode
    chart = "value_scatter"
    shortlist_mode = "default"
    intent = "shortlist"

    if player_name and compare_mode:
        intent = "compare"
        chart = "compare_contracts"
        shortlist_mode = "player_plus_similar"
    elif player_name and asks_player:
        intent = "player_profile"
        chart = "player_contract_bar"
        shortlist_mode = "player_plus_similar" if asks_similar else "single_player"
    elif team_name and _contains_any(q, EXPIRING_PATTERNS):
        intent = "team_expiring"
        chart = "team_expiring"
    elif team_name or _contains_any(q, TEAM_PATTERNS):
        intent = "team_analysis"
        chart = "team_gap"
    elif _contains_any(q, POSITION_GAP_PATTERNS):
        chart = "position_gap"
        intent = "smart_query"
    elif _contains_any(q, TEAM_GAP_PATTERNS):
        chart = "team_gap"
        intent = "smart_query"

    if player_name and any(w in q for w in ["only", "just", "single"]):
        shortlist_mode = "single_player"

    return {
        "raw_query": query,
        "focus_player": player_name,
        "focus_team": team_name,
        "focus_position": position,
        "objective": objective,
        "max_salary": max_salary,
        "min_games": min_games,
        "season": season,
        "shortlist_mode": shortlist_mode,
        "smart_chart": chart,
        "intent": intent,
        "compare_mode": compare_mode,
        "compare_player": None,
        "llm_used": False,
        "parse_method": "rule_fallback_v4",
    }


def infer_llm_plan(query: str, df: pd.DataFrame, team_aliases: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    sample_players = df["player_name"].dropna().astype(str).head(80).tolist()
    sample_teams = sorted(df["team"].dropna().astype(str).unique().tolist()) if "team" in df.columns else []
    schema_hint = {
        "focus_player": "string|null",
        "focus_team": "string|null",
        "focus_position": "QB|RB|WR|TE|null",
        "objective": "balanced|undervalued|overvalued",
        "max_salary": "number|null",
        "min_games": "number|null",
        "season": "number|null",
        "shortlist_mode": "default|single_player|player_plus_similar",
        "smart_chart": "value_scatter|position_gap|team_gap|player_contract_bar|team_expiring|compare_contracts",
        "intent": "shortlist|player_profile|team_analysis|team_expiring|compare|smart_query",
        "compare_player": "string|null",
    }
    prompt = f"""
You are planning a Streamlit NFL contract dashboard.
Read the user query and return only valid JSON.

User query: {query}

Available player examples: {sample_players}
Available teams: {sample_teams}
Team aliases: {team_aliases or TEAM_ALIASES}

Rules:
- If the query asks for one player's profile or stats, use that player as focus_player and set shortlist_mode to "single_player" unless the query also explicitly asks for similar players or comparison.
- If the query is like "show undervalued WRs under $12M", focus_position should be WR, objective undervalued, and max_salary 12000000.
- If a player name is in shorthand like K.Murray, map it to the closest player in the dataset.
- focus_team should be the player's latest team when obvious from the dataset.
- Choose one smart_chart that best answers the user.
- Return only keys from this schema: {json.dumps(schema_hint)}

Output JSON only.
"""
    llm_plan = ask_openai_json(prompt)
    if not isinstance(llm_plan, dict):
        return None
    llm_plan["llm_used"] = True
    llm_plan["parse_method"] = "openai_json_plan_v4"
    llm_plan["raw_query"] = query
    return llm_plan


def build_dashboard_plan(user_query: str, df: pd.DataFrame, team_aliases: Optional[Dict[str, str]] = None, use_llm: bool = True) -> Dict[str, Any]:
    rule_plan = infer_rule_plan(user_query, df, team_aliases)
    merged = rule_plan
    if use_llm:
        llm_plan = infer_llm_plan(user_query, df, team_aliases)
        if llm_plan:
            merged = {**rule_plan, **{k: v for k, v in llm_plan.items() if v not in [None, "", []]}}

    q = normalize_text(user_query)
    if merged.get("focus_player") and merged.get("intent") == "player_profile":
        if _contains_any(q, SIMILAR_PATTERNS) or merged.get("compare_mode"):
            merged["shortlist_mode"] = "player_plus_similar"
        else:
            merged["shortlist_mode"] = "single_player"
        merged["smart_chart"] = "player_contract_bar"
    if merged.get("focus_player") and merged.get("intent") == "compare":
        merged["shortlist_mode"] = "player_plus_similar"
        merged["smart_chart"] = "compare_contracts"
    return merged


def merge_plan_with_ui_filters(plan: Dict[str, Any], ui_filters: Optional[Dict[str, Any]] = None, query_priority: bool = True) -> Dict[str, Any]:
    ui_filters = ui_filters or {}
    merged = dict(ui_filters)
    key_map = {
        "position": "focus_position",
        "team": "focus_team",
        "objective": "objective",
        "max_salary": "max_salary",
        "min_games": "min_games",
        "season": "season",
    }
    for key, plan_key in key_map.items():
        plan_val = plan.get(plan_key)
        if query_priority:
            if plan_val not in [None, ""]:
                merged[key] = plan_val
        else:
            if merged.get(key) in [None, ""] and plan_val not in [None, ""]:
                merged[key] = plan_val
    return merged
