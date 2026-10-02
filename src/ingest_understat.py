"""Understat xG ingestion (player-season totals for the Big-5 leagues).

Why: FBref lost its Opta licence in January 2026, so xG / xA stop at 2024-25 in the FBref
files. Understat still publishes xG, xA, npxG, shots and key passes for the same five leagues.
It does NOT publish progressive passes / carries, tackles or clearances.

Politeness: one request per league-season (10 per run by default) with a pause between them.
Understat's robots.txt asks automated clients not to crawl; running this is the operator's
decision. Keep the volume this low.

Output: data/understat_YYZZ.csv, one file per season, one row per player. A player who moved
mid-season appears ONCE, with both clubs in `team_title` ("Aston Villa,Chelsea").

This script only downloads and saves. It is not yet wired into valuation_model.py.

Usage:
    python src/ingest_understat.py            # season in progress + the one before
    python src/ingest_understat.py 2025 2026  # explicit season start years
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests

from src.common import DATA_DIR, current_season_start, write_json

LEAGUES = {                      # Understat code -> canonical league label used by the pipeline
    "EPL": "Premier League", "La_liga": "La Liga", "Serie_A": "Serie A",
    "Bundesliga": "Bundesliga", "Ligue_1": "Ligue 1",
}
BASE = "https://understat.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
PAUSE_S = 4
NUMERIC = ["games", "time", "goals", "xG", "assists", "xA", "shots", "key_passes",
           "yellow_cards", "red_cards", "npg", "npxG", "xGChain", "xGBuildup"]
REQUIRED = {"player_name", "team_title", "time", "xG", "xA"}
MIN_PLAYERS_PER_LEAGUE = 150     # a finished league season has ~500; early season ~300


def _from_api(session: requests.Session, league: str, season: int):
    """Newer site: the league page loads its data from a JSON endpoint."""
    r = session.get(f"{BASE}/getLeagueData/{league}/{season}", timeout=40,
                    headers={**HEADERS, "X-Requested-With": "XMLHttpRequest",
                             "Referer": f"{BASE}/league/{league}/{season}"})
    r.raise_for_status()
    payload = r.json()
    players = payload.get("players") if isinstance(payload, dict) else None
    return players if isinstance(players, list) and players else None


def parse_embedded(html: str):
    """Older site: `var playersData = JSON.parse('\\x5B\\x7B...')` inside a <script> tag."""
    m = re.search(r"playersData\s*=\s*JSON\.parse\('(.*?)'\)", html, flags=re.S)
    if not m:
        return None
    raw = m.group(1).encode("utf-8").decode("unicode_escape")
    players = json.loads(raw)
    return players if isinstance(players, list) and players else None


def _from_page(session: requests.Session, league: str, season: int):
    r = session.get(f"{BASE}/league/{league}/{season}", timeout=40, headers=HEADERS)
    r.raise_for_status()
    return parse_embedded(r.text)


def to_frame(players: list, league_label: str, season: int) -> pd.DataFrame:
    df = pd.DataFrame(players)
    for c in NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["league"] = league_label
    df["season"] = f"{season}-{season + 1}"
    return df


def fetch_league_season(session: requests.Session, league: str, season: int) -> tuple[pd.DataFrame | None, str]:
    for method, fn in (("api", _from_api), ("page", _from_page)):
        try:
            players = fn(session, league, season)
        except Exception as exc:                                  # noqa: BLE001
            print(f"    {method}: {type(exc).__name__}: {str(exc)[:120]}")
            continue
        if players:
            return to_frame(players, LEAGUES[league], season), method
        print(f"    {method}: no player data in the response")
    return None, "none"


def ingest(seasons: list[int]) -> dict:
    report = {}
    with requests.Session() as session:
        for season in seasons:
            frames, methods = [], {}
            for league in LEAGUES:
                print(f"  {league} {season}-{season + 1} ...")
                df, method = fetch_league_season(session, league, season)
                methods[league] = method
                if df is not None:
                    missing = REQUIRED - set(df.columns)
                    if missing:
                        print(f"    WARNING: unexpected columns (missing {missing}); got {list(df.columns)}")
                    elif len(df) < MIN_PLAYERS_PER_LEAGUE:
                        print(f"    WARNING: only {len(df)} players - skipped as incomplete")
                    else:
                        frames.append(df)
                        print(f"    OK: {len(df)} players via {method}")
                time.sleep(PAUSE_S)

            label = f"{season % 100:02d}{(season + 1) % 100:02d}"
            if len(frames) == len(LEAGUES):
                out = pd.concat(frames, ignore_index=True)
                path = DATA_DIR / f"understat_{label}.csv"
                out.to_csv(path, index=False)
                print(f"  saved {len(out):,} players -> {path.name}")
                report[str(season)] = {"players": int(len(out)), "methods": methods, "saved": True,
                                       "columns": list(out.columns)}
            else:
                # never overwrite a complete file with a partial one
                print(f"  NOT saved: only {len(frames)}/{len(LEAGUES)} leagues came back for {season}")
                report[str(season)] = {"players": 0, "methods": methods, "saved": False}
    write_json(DATA_DIR / "understat_health.json", report)
    return report


if __name__ == "__main__":
    cur = current_season_start()
    wanted = [int(a) for a in sys.argv[1:]] or [cur - 1, cur]
    print(f"Understat ingestion for seasons starting {wanted}")
    result = ingest(wanted)
    if not any(v["saved"] for v in result.values()):
        sys.exit("FATAL: no season could be downloaded (site layout changed, or requests are blocked).")
