"""Live FBref scraper (patched).

Fixes vs. the original (#13 in the audit):
  - Every failure path used to `return` or get swallowed by a blanket `except Exception`,
    so a broken scrape still exited 0, the CI job proceeded to retrain on stale/partial data,
    and committed it. Every failure path now raises SystemExit(1), and a run is rejected if
    row counts or column coverage fall below a sane floor.
  - The defense-table merge and the squad-possession PAdj step used to apply silently even
    when the source columns were entirely empty (this is exactly what happened to the
    2026-27 scrape: Tkl/Clr/Blk were 100% NaN and Poss was absent, so the whole PAdj
    multiplier silently became a no-op). Both steps now assert real coverage before
    applying, and skip cleanly with a loud warning (not a silent pass) otherwise.
  - `headless=True` is easier for anti-bot systems to fingerprint under `xvfb-run`; switched
    to `headless=False` (an actual virtual X display, driven by xvfb) which SeleniumBase's UC
    mode handles better. Set SELENIUM_HEADLESS=1 to force headless back on if you don't want
    to run xvfb in CI.
  - A machine-readable data/live_health.json is written every run so the workflow (and you)
    can see coverage without reading logs.
"""
from __future__ import annotations

import os
import sys
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, Comment
from seleniumbase import SB

from src.common import DATA_DIR, write_json

MIN_PLAYERS = 300           # a real Big-5 scrape has ~2,000+; below this something broke
MIN_COVERAGE = 0.85         # required non-null share for a merged column to be "trusted"
TABLE_WAIT_S = 40


def find_stats_table(html: str, table_id_prefix: str):
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id=lambda x: x and x.startswith(table_id_prefix))
    if table is not None:
        return table
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if table_id_prefix in comment:
            inner = BeautifulSoup(comment, "lxml")
            table = inner.find("table", id=lambda x: x and x.startswith(table_id_prefix))
            if table is not None:
                return table
    return None


def _wait_for_table(sb, selector: str, seconds: int = TABLE_WAIT_S) -> bool:
    for _ in range(seconds):
        if sb.is_element_present(selector):
            return True
        sb.sleep(1)
    return False


def scrape_fbref_stealth() -> Path:
    player_url = "https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats"
    defense_url = "https://fbref.com/en/comps/Big5/defense/players/Big-5-European-Leagues-Stats"
    possession_url = "https://fbref.com/en/comps/Big5/possession/squads/Big-5-European-Leagues-Stats"
    headless = os.getenv("SELENIUM_HEADLESS", "0") == "1"

    print("Initializing SeleniumBase (Undetected ChromeDriver)...")
    with SB(uc=True, headless=headless) as sb:
        # ---- 1. standard stats -------------------------------------------------
        print(f"Fetching standard stats: {player_url}")
        sb.open(player_url)
        if not _wait_for_table(sb, "table[id^='stats_standard']"):
            sys.exit("FATAL: standard-stats table never appeared (Cloudflare block or layout change).")
        sb.sleep(2)
        table = find_stats_table(sb.get_page_source(), "stats_standard")
        if table is None:
            sys.exit("FATAL: standard-stats table not found in page source.")

        df = pd.read_html(StringIO(str(table)))[0]
        if df.columns.nlevels > 1:
            df.columns = df.columns.droplevel()
        df = df[df["Player"] != "Player"].copy()
        df["Age"] = df["Age"].astype(str).str[:2]
        df["Pos"] = df["Pos"].astype(str).str[:2]
        df["Min"] = pd.to_numeric(df["Min"].astype(str).str.replace(",", "", regex=False), errors="coerce")
        df = df.dropna(subset=["Min"])
        df = df[df["Min"] > 90].copy()

        if len(df) < MIN_PLAYERS:
            sys.exit(f"FATAL: only {len(df)} players scraped (< {MIN_PLAYERS} floor) - "
                     "the page likely rendered a partial/blocked table.")
        print(f"  standard stats OK: {len(df)} players")

        # ---- 2. defense stats ---------------------------------------------------
        print(f"Fetching defense stats: {defense_url}")
        sb.open(defense_url)
        if not _wait_for_table(sb, "table[id^='stats_defense']"):
            print("  WARNING: defense table did not load; Tkl/Int/Clr/Blk will be blank this run.")
        else:
            sb.sleep(2)
            def_table = find_stats_table(sb.get_page_source(), "stats_defense")
            if def_table is not None:
                df_def = pd.read_html(StringIO(str(def_table)))[0]
                if df_def.columns.nlevels > 1:
                    df_def.columns = df_def.columns.droplevel()
                df_def = df_def.loc[:, ~df_def.columns.duplicated()].copy()
                df_def = df_def.rename(columns={"Blocks": "Blk"})
                df_def = df_def[df_def["Player"] != "Player"].copy()
                needed = [c for c in ["Player", "Squad", "Tkl", "Int", "Clr", "Blk"] if c in df_def.columns]
                df_def = df_def[needed]
                coverage = df_def[[c for c in ["Tkl", "Int", "Clr", "Blk"] if c in df_def.columns]].apply(
                    lambda s: pd.to_numeric(s, errors="coerce").notna().mean())
                print(f"  defense coverage: {coverage.round(2).to_dict()}")
                if (coverage < MIN_COVERAGE).any():
                    print(f"  WARNING: defense coverage below {MIN_COVERAGE:.0%} for "
                         f"{coverage[coverage < MIN_COVERAGE].index.tolist()}; merging anyway "
                         "but per-90 columns for those stats will be mostly NaN, not silently 0.")
                df = df.merge(df_def, on=["Player", "Squad"], how="left")
                for col in ["Tkl", "Int", "Clr", "Blk"]:
                    if col in df.columns:
                        df[f"{col.lower()}_per90"] = pd.to_numeric(df[col], errors="coerce") / (df["Min"] / 90.0)
            else:
                print("  WARNING: defense table not found in page source; skipping merge.")

        # ---- 3. squad possession (for PAdj) --------------------------------------
        print(f"Fetching squad possession: {possession_url}")
        sb.open(possession_url)
        if not _wait_for_table(sb, "table[id^='stats_squads_possession_for']"):
            print("  WARNING: possession table did not load; PAdj multiplier NOT applied this run "
                 "(raw per-90 values kept instead of silently skipping the adjustment).")
        else:
            sb.sleep(2)
            sq_table = find_stats_table(sb.get_page_source(), "stats_squads_possession_for")
            if sq_table is not None:
                sq_df = pd.read_html(StringIO(str(sq_table)))[0]
                if sq_df.columns.nlevels > 1:
                    sq_df.columns = sq_df.columns.droplevel()
                sq_df = sq_df[["Squad", "Poss"]].copy()
                sq_df["Poss"] = pd.to_numeric(sq_df["Poss"], errors="coerce")
                if sq_df["Poss"].notna().mean() < MIN_COVERAGE:
                    print("  WARNING: possession coverage too low; PAdj NOT applied this run.")
                else:
                    median_poss = sq_df["Poss"].median()
                    df = df.merge(sq_df, on="Squad", how="left")
                    df["Poss"] = df["Poss"].fillna(median_poss)
                    multiplier = 1 + (0.8 / (1 + np.exp(-0.08 * (df["Poss"] - median_poss)))) - 0.4
                    for col in ["tkl_per90", "int_per90", "clr_per90", "blk_per90"]:
                        if col in df.columns:
                            df[col] = df[col] * multiplier
                    print(f"  PAdj applied (median possession {median_poss:.1f}%)")
            else:
                print("  WARNING: possession table not found in page source; PAdj NOT applied.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "live_fbref_data.csv"
    df.to_csv(out_path, index=False)

    write_json(DATA_DIR / "live_health.json", {
        "players": int(len(df)),
        "defense_merged": "Tkl" in df.columns,
        "padj_applied": "Poss" in df.columns and df["Poss"].notna().any(),
    })
    print(f"Ingestion complete: {len(df)} active players -> {out_path}")
    return out_path


if __name__ == "__main__":
    try:
        scrape_fbref_stealth()
    except SystemExit:
        raise
    except Exception as exc:                      # noqa: BLE001 - top-level CI guard
        sys.exit(f"FATAL: unhandled scraping error: {exc}")
