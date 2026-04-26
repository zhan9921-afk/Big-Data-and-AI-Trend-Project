
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
    "49ers": "San Francisco 49ers", "niners": "San Francisco 49ers", "sf": "San Francisco 49ers", "san francisco": "San Francisco 49ers",
    "chiefs": "Kansas City Chiefs", "kc": "Kansas City Chiefs", "kansas city": "Kansas City Chiefs",
    "eagles": "Philadelphia Eagles", "philly": "Philadelphia Eagles", "philadelphia": "Philadelphia Eagles",
    "cowboys": "Dallas Cowboys", "dallas": "Dallas Cowboys",
    "bills": "Buffalo Bills", "buffalo": "Buffalo Bills",
    "bengals": "Cincinnati Bengals", "cincinnati": "Cincinnati Bengals",
    "rams": "Los Angeles Rams", "la rams": "Los Angeles Rams",
    "packers": "Green Bay Packers", "green bay": "Green Bay Packers",
    "vikings": "Minnesota Vikings", "minnesota": "Minnesota Vikings",
    "ravens": "Baltimore Ravens", "baltimore": "Baltimore Ravens",
    "cardinals": "Arizona Cardinals", "arizona": "Arizona Cardinals",
    "giants": "New York Giants", "jets": "New York Jets", "patriots": "New England Patriots",
    "steelers": "Pittsburgh Steelers", "saints": "New Orleans Saints", "falcons": "Atlanta Falcons",
    "panthers": "Carolina Panthers", "bears": "Chicago Bears", "lions": "Detroit Lions",
    "broncos": "Denver Broncos", "texans": "Houston Texans", "colts": "Indianapolis Colts",
    "jaguars": "Jacksonville Jaguars", "chargers": "Los Angeles Chargers", "raiders": "Las Vegas Raiders",
    "dolphins": "Miami Dolphins", "buccaneers": "Tampa Bay Buccaneers", "titans": "Tennessee Titans",
    "commanders": "Washington Commanders", "seahawks": "Seattle Seahawks", "browns": "Cleveland Browns",
}

POSITION_PATTERNS = {
    "QB": [r"\bqb\b", r"\bqbs\b", r"\bquarterback\b", r"\bquarterbacks\b"],
    "RB": [r"\brb\b", r"\brbs\b", r"\brunning\s*back\b", r"\brunning\s*backs\b", r"\bbackfield\b"],
    "WR": [r"\bwr\b", r"\bwrs\b", r"\bwide\s*receiver\b", r"\bwide\s*receivers\b", r"\bwideout\b", r"\breceiver room\b"],
    "TE": [r"\bte\b", r"\btes\b", r"\btight\s*end\b", r"\btight\s*ends\b", r"\btightends?\b"],
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
SUPERSTAR_PATTERNS = [r"\bsuperstar\b", r"\belite\b", r"\bstar\b", r"\btop\s*tier\b"]

MAX_SALARY_PATTERNS_M = [
    r"(?:under|below|less than|cheaper than)\s*\$?\s*(\d+(?:\.\d+)?)\s*m\b",
    r"(?:<=|=<)\s*\$?\s*(\d+(?:\.\d+)?)\s*m\b",
    r"\$?\s*(\d+(?:\.\d+)?)\s*m\s*(?:or less|max|maximum)\b",
]
MIN_SALARY_PATTERNS_M = [
    r"(?:making\s+over|making\s+at\s+least|over|above|more than|greater than|at least)\s*\$?\s*(\d+(?:\.\d+)?)\s*m\b",
    r"(?:>=|=>)\s*\$?\s*(\d+(?:\.\d+)?)\s*m\b",
    r"\$?\s*(\d+(?:\.\d+)?)\s*m\s*(?:or more|min|minimum|plus)\b",
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
        # Use initial+last variants, but do NOT match last name alone.
        # Last-name-only matching created false positives, e.g. "making" matched Tavarres King.
        variants |= {f"{first[0]}.{last}".lower(), f"{first[0]} {last}".lower(), f"{first}.{last}".lower(), f"{first[0]}{last}".lower()}
    return {v.strip() for v in variants if v.strip()}

def detect_player_from_query(query: str, available_names: Iterable[str]) -> Optional[str]:
    query_norm = normalize_text(query)
    names = [str(n) for n in available_names if str(n).strip()]
    for name in names:
        for variant in _player_alias_variants(name):
            if re.search(rf"\b{re.escape(variant)}\b", query_norm, flags=re.IGNORECASE):
                return name
    query_words = set(re.sub(r"[^a-z0-9 ]", " ", query_norm).split())
    for name in names:
        name_words = set(re.sub(r"[^a-z0-9 ]", " ", normalize_text(name)).split())
        if len(name_words) >= 2 and len(name_words & query_words) >= 2:
            return name
    compact_q = query_norm.replace(" ", "").replace(".", "")
    for name in names:
        for variant in _player_alias_variants(name):
            if variant.replace(" ", "").replace(".", "") in compact_q:
                return name
    close = get_close_matches(query.strip(), names, n=1, cutoff=0.82)
    return close[0] if close else None

def extract_position(query: str) -> Optional[str]:
    for position, patterns in POSITION_PATTERNS.items():
        if _contains_any(query, patterns):
            return position
    return None

def extract_objective(query: str) -> str:
    if _contains_any(query, OVERVALUE_PATTERNS):
        return "overvalued"
    if _contains_any(query, UNDERVALUE_PATTERNS):
        return "undervalued"
    return DEFAULT_OBJECTIVE

def extract_max_salary(query: str) -> Optional[float]:
    for pattern in MAX_SALARY_PATTERNS_M:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return float(match.group(1)) * 1_000_000
    return None

def extract_min_salary(query: str) -> Optional[float]:
    for pattern in MIN_SALARY_PATTERNS_M:
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
            if normalize_text(str(team)) in q:
                return str(team)
    return None

def infer_rule_plan(query: str, df: pd.DataFrame, team_aliases: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    q = normalize_text(query)
    available_names = df["player_name"].dropna().astype(str).unique().tolist()
    available_teams = df["team"].dropna().astype(str).unique().tolist() if "team" in df.columns else []

    player_name = detect_player_from_query(query, available_names)
    team_name = extract_team(q, team_aliases, available_teams)
    position = extract_position(q)
    objective = extract_objective(q)
    max_salary = extract_max_salary(q)
    min_salary = extract_min_salary(q)
    # "budget WRs/RBs/etc." is vague but should still narrow to affordable current contracts.
    if max_salary is None and re.search(r"\bbudget\b|\bcheap\b|\baffordable\b", q):
        max_salary = 12_000_000.0
    min_games = extract_min_games(q)
    season = extract_season(q)
    superstar_only = _contains_any(q, SUPERSTAR_PATTERNS)

    if player_name:
        player_rows = df[df["player_name"] == player_name].sort_values("season", ascending=False)
        if not player_rows.empty:
            latest = player_rows.iloc[0]
            if not position:
                position = latest.get("position")
            if not team_name and pd.notna(latest.get("team")):
                team_name = latest.get("team")

    compare_mode = _contains_any(q, COMPARE_PATTERNS)
    asks_player = player_name is not None and (_contains_any(q, PLAYER_PATTERNS) or "profile" in q)
    asks_similar = _contains_any(q, SIMILAR_PATTERNS) or compare_mode

    chart = "value_scatter"
    shortlist_mode = "default"
    intent = "shortlist"

    if player_name and compare_mode:
        intent, chart, shortlist_mode = "compare", "compare_contracts", "player_plus_similar"
    elif player_name and asks_player:
        intent, chart, shortlist_mode = "player_profile", "player_contract_bar", ("player_plus_similar" if asks_similar else "single_player")
    elif team_name and _contains_any(q, EXPIRING_PATTERNS):
        intent, chart = "team_expiring", "team_expiring"
    elif team_name or _contains_any(q, TEAM_PATTERNS):
        intent, chart = "team_analysis", "team_gap"
    elif _contains_any(q, POSITION_GAP_PATTERNS):
        intent, chart = "smart_query", "position_gap"
    elif _contains_any(q, TEAM_GAP_PATTERNS):
        intent, chart = "smart_query", "team_gap"

    # Salary wording like "making over $30M" should filter Current APY,
    # not actual next APY. This is the key fix for 2025 current-player queries.
    salary_basis = "current_apy" if re.search(r"\b(making|salary|paid|earning|earns|cap|current|budget|cheap|affordable)\b", q) else "actual_apy"
    if min_salary is not None or max_salary is not None:
        if "making" in q or "at least" in q or "over" in q:
            salary_basis = "current_apy"

    # Vague "superstar WRs" means high-end WRs; let dashboard apply a heuristic.
    if superstar_only and not position:
        position = "WR" if re.search(r"\bwrs?\b|\bwide receivers?\b", q) else position

    return {
        "raw_query": query,
        "focus_player": player_name,
        "focus_team": team_name,
        "focus_position": position,
        "objective": objective,
        "max_salary": max_salary,
        "min_salary": min_salary,
        "salary_basis": salary_basis,
        "min_games": min_games,
        "season": season,
        "superstar_only": superstar_only,
        "shortlist_mode": shortlist_mode,
        "smart_chart": chart,
        "intent": intent,
        "compare_mode": compare_mode,
        "compare_player": None,
        "llm_used": False,
        "parse_method": "rule_fallback_v6",
    }

def infer_llm_plan(query: str, df: pd.DataFrame, team_aliases: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    # Send compact choices to the LLM. The rule plan remains the guardrail after this returns.
    sample_players = df["player_name"].dropna().astype(str).drop_duplicates().head(160).tolist()
    sample_teams = sorted(df["team"].dropna().astype(str).unique().tolist()) if "team" in df.columns else []
    schema_hint = {
        "focus_player": "string|null",
        "focus_team": "string|null",
        "focus_position": "QB|RB|WR|TE|null",
        "objective": "balanced|undervalued|overvalued",
        "max_salary": "number|null in dollars",
        "min_salary": "number|null in dollars",
        "salary_basis": "current_apy|actual_apy",
        "season": "number|null",
        "superstar_only": "boolean",
        "shortlist_mode": "default|single_player|player_plus_similar",
        "smart_chart": "value_scatter|position_gap|team_gap|player_contract_bar|team_expiring|compare_contracts",
        "intent": "shortlist|player_profile|team_analysis|team_expiring|compare|smart_query",
        "compare_player": "string|null",
    }
    prompt = f"""
You are planning a Streamlit NFL contract dashboard. Return only valid JSON.

User query: {query}

Available player examples: {sample_players}
Available teams: {sample_teams}
Team aliases: {team_aliases or TEAM_ALIASES}

Rules:
- Plural abbreviations matter: QBs => QB, WRs => WR, RBs => RB, TEs => TE.
- "making over $30M", "making at least $10M", "paid over", or "current salary" means salary_basis = "current_apy".
- Under/over salary filters should be dollars, e.g. 30M => 30000000.
- If query says "superstar WRs", focus_position = WR and superstar_only = true.
- Do not replace a clear rule-derived position with a different position.
- If query asks for a player's profile, use shortlist_mode = "single_player" unless it explicitly asks for similar/comparison.
- Return only keys in this schema: {json.dumps(schema_hint)}.
"""
    plan = ask_openai_json(prompt)
    if not isinstance(plan, dict):
        return None
    plan["llm_used"] = True
    plan["parse_method"] = "openai_json_plan_v6"
    plan["raw_query"] = query
    return plan

def _clean_plan_values(plan: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    out = dict(plan)
    # Normalize player/team names back to exact dataset values.
    if out.get("focus_player"):
        matched = detect_player_from_query(str(out["focus_player"]), df["player_name"].dropna().astype(str).unique())
        if matched:
            out["focus_player"] = matched
    if out.get("compare_player"):
        matched = detect_player_from_query(str(out["compare_player"]), df["player_name"].dropna().astype(str).unique())
        if matched:
            out["compare_player"] = matched
    if out.get("focus_position"):
        pos = str(out["focus_position"]).upper().strip()
        out["focus_position"] = pos if pos in {"QB", "RB", "WR", "TE"} else None
    for k in ["max_salary", "min_salary"]:
        if out.get(k) is not None:
            try:
                out[k] = float(out[k])
            except Exception:
                out[k] = None
    if out.get("season") is not None:
        try:
            out["season"] = int(float(out["season"]))
        except Exception:
            out["season"] = None
    if out.get("salary_basis") not in {"current_apy", "actual_apy"}:
        out["salary_basis"] = "actual_apy"
    out["superstar_only"] = bool(out.get("superstar_only", False))
    return out

def build_dashboard_plan(user_query: str, df: pd.DataFrame, team_aliases: Optional[Dict[str, str]] = None, use_llm: bool = True) -> Dict[str, Any]:
    rule_plan = infer_rule_plan(user_query, df, team_aliases)
    merged = dict(rule_plan)
    if use_llm:
        llm_plan = infer_llm_plan(user_query, df, team_aliases)
        if llm_plan:
            # Guardrails: never let LLM remove clear rule-derived filters.
            for k, v in llm_plan.items():
                if v not in [None, "", []]:
                    merged[k] = v
            for key in ["focus_position", "min_salary", "max_salary", "season", "focus_player", "focus_team", "superstar_only", "salary_basis"]:
                if rule_plan.get(key) not in [None, "", [], False]:
                    merged[key] = rule_plan[key]
            merged["llm_used"] = True
            merged["parse_method"] = f"{rule_plan.get('parse_method')}+openai_guarded_v6"
    merged = _clean_plan_values(merged, df)

    q = normalize_text(user_query)
    if merged.get("focus_player") and merged.get("intent") == "player_profile":
        merged["shortlist_mode"] = "player_plus_similar" if (_contains_any(q, SIMILAR_PATTERNS) or merged.get("compare_mode")) else "single_player"
        merged["smart_chart"] = "player_contract_bar"
    if merged.get("focus_player") and merged.get("intent") == "compare":
        merged["shortlist_mode"] = "player_plus_similar"
        merged["smart_chart"] = "compare_contracts"
    return merged

def merge_plan_with_ui_filters(plan: Dict[str, Any], ui_filters: Optional[Dict[str, Any]] = None, query_priority: bool = True) -> Dict[str, Any]:
    ui_filters = ui_filters or {}
    keys = ["position", "team", "max_salary", "min_salary", "salary_basis", "objective", "min_games", "season", "superstar_only"]
    mapping = {
        "position": "focus_position",
        "team": "focus_team",
        "max_salary": "max_salary",
        "min_salary": "min_salary",
        "salary_basis": "salary_basis",
        "objective": "objective",
        "min_games": "min_games",
        "season": "season",
        "superstar_only": "superstar_only",
    }
    merged: Dict[str, Any] = {}
    for key in keys:
        p_val = plan.get(mapping[key])
        u_val = ui_filters.get(key)
        if query_priority:
            merged[key] = p_val if p_val not in [None, "", []] else u_val
        else:
            merged[key] = u_val if u_val not in [None, "", []] else p_val
    if merged.get("objective") is None:
        merged["objective"] = DEFAULT_OBJECTIVE
    if merged.get("salary_basis") not in {"current_apy", "actual_apy"}:
        merged["salary_basis"] = "actual_apy"
    return merged
