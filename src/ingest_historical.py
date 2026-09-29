"""Historical multi-season scraper (patched). Same fail-loud philosophy as ingest_fbref.py (#13)."""
from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from bs4 import BeautifulSoup, Comment
from seleniumbase import SB

from src.common import DATA_DIR, current_season_start, write_json

MIN_PLAYERS_PER_SEASON = 300
MIN_SEASONS_OK = 4            # if fewer than this many seasons come back, fail the whole run


def find_stats_table(html: str):
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id=lambda x: x and x.startswith("stats_standard"))
    if table is not None:
        return table
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "stats_standard" in comment:
            inner = BeautifulSoup(comment, "lxml")
            table = inner.find("table", id=lambda x: x and x.startswith("stats_standard"))
            if table is not None:
                return table
    return None


def _seasons(n_back: int = 6) -> list[tuple[str, str]]:
    cur = current_season_start()
    out = []
    for i in range(n_back):
        y = cur - i
        label = f"{y}-{y + 1}"
        if i == 0:
            url = "https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats"
        else:
            url = (f"https://fbref.com/en/comps/Big5/{label}/stats/players/"
                  f"{label}-Big-5-European-Leagues-Stats")
        out.append((label, url))
    return out


def ingest_historical_stats() -> Path:
    seasons = _seasons()
    master_df = pd.DataFrame()
    ok_seasons, failed_seasons = [], []

    print("Initializing SeleniumBase (Undetected ChromeDriver)...")
    with SB(uc=True, headless=False) as sb:
        for season, url in seasons:
            print(f"Scraping {season}...")
            sb.open(url)

            cleared = False
            for _ in range(30):
                if sb.is_element_present("table[id^='stats_standard']"):
                    cleared = True
                    break
                sb.sleep(1)
            if not cleared:
                print(f"  WARNING: {season} - table never appeared (Cloudflare block/timeout).")
                failed_seasons.append(season)
                continue

            sb.sleep(2)
            table = find_stats_table(sb.get_page_source())
            if table is None:
                print(f"  WARNING: {season} - table not found in page source.")
                failed_seasons.append(season)
                continue

            df = pd.read_html(StringIO(str(table)))[0]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(-1)
            df["season"] = season

            min_col = next((c for c in df.columns if str(c).lower() == "min"), None)
            if min_col:
                df[min_col] = pd.to_numeric(df[min_col].astype(str).str.replace(",", "", regex=False),
                                            errors="coerce")
                df = df[df[min_col] >= 500].copy()

            if len(df) < MIN_PLAYERS_PER_SEASON:
                print(f"  WARNING: {season} - only {len(df)} players after filtering (suspiciously low).")
                failed_seasons.append(season)
                continue

            master_df = pd.concat([master_df, df], ignore_index=True)
            ok_seasons.append(season)
            print(f"  OK: {season} ({len(df)} qualified players)")
            sb.sleep(4)

    if len(ok_seasons) < MIN_SEASONS_OK:
        sys.exit(f"FATAL: only {len(ok_seasons)}/{len(seasons)} seasons scraped successfully "
                 f"({ok_seasons}); refusing to overwrite the existing historical file.")

    master_df.columns = [str(c).lower() for c in master_df.columns]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "fbref_historical_stats.csv"

    if failed_seasons and out_path.exists():
        prev = pd.read_csv(out_path, low_memory=False)
        prev.columns = [str(c).lower() for c in prev.columns]
        keep_old = prev[prev["season"].isin(failed_seasons)]
        if not keep_old.empty:
            master_df = pd.concat([master_df, keep_old], ignore_index=True)
            print(f"  kept {len(keep_old)} rows from the previous file for un-refreshed "
                 f"season(s) {failed_seasons}")

    master_df.to_csv(out_path, index=False)

    write_json(DATA_DIR / "historical_health.json", {
        "seasons_ok": ok_seasons, "seasons_failed": failed_seasons, "rows": int(len(master_df)),
    })
    print(f"Compiled {len(master_df):,} player-seasons across {len(ok_seasons)} seasons -> {out_path}")
    if failed_seasons:
        print(f"NOTE: {failed_seasons} were not refreshed this run; previous values for those "
             "seasons are gone from this file - re-run before merging if that matters.")
    return out_path


if __name__ == "__main__":
    try:
        ingest_historical_stats()
    except SystemExit:
        raise
    except Exception as exc:                      # noqa: BLE001
        sys.exit(f"FATAL: unhandled scraping error: {exc}")
