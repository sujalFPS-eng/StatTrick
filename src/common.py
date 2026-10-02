"""Shared helpers: one place for name / club / league / season normalisation.

Every stage (valuation, similarity, system fit, app) imports from here so the
same club or player can never end up under two different labels again.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


# --------------------------------------------------------------------------- dates
def today() -> date:
    """Pipeline 'today'. Override with STATTRICK_TODAY=YYYY-MM-DD for reproducible runs."""
    override = os.getenv("STATTRICK_TODAY")
    return datetime.strptime(override, "%Y-%m-%d").date() if override else date.today()


def current_season_start() -> int:
    """Start year of the season in progress (Jul-Dec 2026 -> 2026, Jan-Jun 2027 -> 2026)."""
    t = today()
    return t.year if t.month >= 7 else t.year - 1


def infer_live_season_start(max_minutes) -> int:
    """Which season does FBref's un-dated 'current' table belong to?

    The calendar says a new season starts on 1 July, but FBref keeps showing the finished season
    until the new one kicks off. In July/August, a table where someone already has more than ten
    matches of minutes is therefore still LAST season - labelling it as the new one would file a
    whole finished season under the wrong year."""
    cur = current_season_start()
    try:
        played = float(max_minutes)
    except (TypeError, ValueError):
        return cur
    if today().month in (7, 8) and played > 900:
        return cur - 1
    return cur


def season_label(start: int) -> str:
    return f"{start}-{start + 1}"


def canon_season(value) -> str | None:
    """'2023-24' / '2023-2024' / 2023 -> '2023-2024'."""
    m = re.match(r"\s*(\d{4})", str(value))
    if not m:
        return None
    y = int(m.group(1))
    return f"{y}-{y + 1}"


def season_start(season: str) -> int:
    return int(str(season)[:4])


# Recency weights by "seasons ago" (0 = season in progress). Same shape as the
# original hand-written dict, but derived from the calendar so it never goes stale.
_WEIGHTS = {0: 1.00, 1: 0.85, 2: 0.65, 3: 0.45, 4: 0.25, 5: 0.10}


def season_weight(season: str) -> float:
    ago = current_season_start() - season_start(season)
    return _WEIGHTS.get(ago, 0.05 if ago > 5 else 1.0)


# --------------------------------------------------------------------------- names
def strip_accents(text) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    return "".join(
        c for c in unicodedata.normalize("NFKD", str(text)) if unicodedata.category(c) != "Mn"
    )


def name_key(text) -> str:
    """Accent-stripped, lower-case, punctuation-free key used for matching people."""
    s = strip_accents(text).lower().replace("-", " ").replace("'", "")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def make_pkey(name: pd.Series, born: pd.Series) -> pd.Series:
    """Player identity = accent-free name + birth year (so two 'Aaron Ramsey's stay two people)."""
    b = pd.to_numeric(born, errors="coerce").astype("Int64").astype(str).replace("<NA>", "na")
    return name.map(name_key) + "|" + b


# --------------------------------------------------------------------------- clubs
# Canonical label = the FBref spelling. Keys are name_key() of the variants.
_CLUB_ALIASES = {
    "newcastle": "Newcastle Utd", "newcastle united": "Newcastle Utd",
    "manchester united": "Manchester Utd", "man utd": "Manchester Utd",
    "man city": "Manchester City",
    "paris sg": "Paris S-G", "paris s g": "Paris S-G", "paris saint germain": "Paris S-G", "psg": "Paris S-G",
    "west ham united": "West Ham", "tottenham hotspur": "Tottenham", "spurs": "Tottenham",
    "nottingham": "Nott'ham Forest", "nottingham forest": "Nott'ham Forest", "nottham forest": "Nott'ham Forest",
    "eintracht frankfurt": "Eint Frankfurt", "frankfurt": "Eint Frankfurt",
    "stade rennais fc": "Rennes", "stade rennais": "Rennes",
    # Understat spellings
    "alaves": "Alavés", "borussia m gladbach": "Gladbach", "fc cologne": "Köln",
    "fc heidenheim": "Heidenheim", "parma calcio 1913": "Parma", "rasenballsport leipzig": "RB Leipzig",
    "real oviedo": "Oviedo", "verona": "Hellas Verona", "vfb stuttgart": "Stuttgart",
    "coventry": "Coventry City", "deportivo la coruna": "Dep. A Coruña", "hull": "Hull City",
    "ipswich": "Ipswich Town", "malaga": "Málaga", "paderborn": "Paderborn 07",
    "racing santander": "Racing Sant",
    "real betis": "Betis", "sheffield united": "Sheffield Utd",
    "st pauli": "St Pauli", "fc st pauli": "St Pauli",
    "brighton and hove albion": "Brighton", "brighton hove albion": "Brighton",
    "wolverhampton wanderers": "Wolves", "leeds": "Leeds United",
    "ac milan": "Milan", "inter milan": "Inter", "internazionale": "Inter",
    "borussia monchengladbach": "Gladbach", "monchengladbach": "Gladbach",
    "borussia dortmund": "Dortmund", "bayer leverkusen": "Leverkusen",
    "atletico de madrid": "Atlético Madrid", "atletico madrid": "Atlético Madrid",
    "athletic bilbao": "Athletic Club", "celta de vigo": "Celta Vigo",
    "1 fc koln": "Köln", "fc koln": "Köln", "koln": "Köln",
    "olympique lyonnais": "Lyon", "olympique de marseille": "Marseille", "as monaco": "Monaco",
}
_CANON_BY_KEY = {name_key(v): v for v in _CLUB_ALIASES.values()}


def canon_club(name) -> str:
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    raw = str(name).strip()
    k = name_key(raw)
    return _CLUB_ALIASES.get(k) or _CANON_BY_KEY.get(k) or raw


# --------------------------------------------------------------------------- leagues
_LEAGUES = {
    "premier league": "Premier League", "la liga": "La Liga", "serie a": "Serie A",
    "bundesliga": "Bundesliga", "ligue 1": "Ligue 1",
}


def canon_league(value) -> str:
    """'eng Premier League' / 'Premier League' / 'de Bundesliga' -> 'Premier League' / 'Bundesliga'."""
    s = re.sub(r"^[a-z]{2,3}\s+", "", str(value).strip())
    return _LEAGUES.get(s.lower(), s)


# --------------------------------------------------------------------------- positions
POS_GROUP = {
    "GK": "GK", "CB": "DEF", "FULLBACK": "DEF",
    "CDM": "MID", "CM": "MID", "CAM": "MID", "WINGER": "ATT", "ST": "ATT",
}


def fbref_pos_group(pos) -> str:
    first = str(pos).upper().replace(" ", "").split(",")[0][:2]
    return {"GK": "GK", "DF": "DEF", "MF": "MID", "FW": "ATT"}.get(first, "MID")


# --------------------------------------------------------------------------- misc
def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default
