"""Transfermarkt valuations refresh (patched).

Fixes vs. the original (#14 in the audit):
  - The old script wrote data/transfermarkt_values.csv, a file nothing else reads.
    valuation_model.py has always loaded the static data/players.csv, so every weekly
    retrain compared fresh FBref stats to a frozen market value. This script now writes
    data/players.csv itself (after validating it has the columns valuation_model.py needs),
    so it belongs in the pipeline as a real stage - add it to the workflow before
    valuation_model.py.
  - Fails loudly (sys.exit) instead of printing an error and returning 0, and keeps the
    previous players.csv untouched if the download or the schema check fails.
"""
from __future__ import annotations

import gzip
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests

from src.common import DATA_DIR

URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/players.csv.gz"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
REQUIRED_COLUMNS = {"name", "market_value_in_eur", "last_season", "date_of_birth",
                    "current_club_name", "current_club_domestic_competition_id",
                    "contract_expiration_date"}
MIN_ROWS = 20_000
MIN_VALUED_ROWS = 5_000


def fetch_real_valuations() -> Path:
    print(f"Downloading Transfermarkt snapshot from {URL} ...")
    try:
        resp = requests.get(URL, headers=HEADERS, timeout=60)
        resp.raise_for_status()
    except Exception as exc:
        sys.exit(f"FATAL: could not download valuations snapshot: {exc}")

    try:
        with gzip.open(io.BytesIO(resp.content), "rt", encoding="utf-8") as f:
            df = pd.read_csv(f, low_memory=False)
    except Exception as exc:
        sys.exit(f"FATAL: snapshot did not decompress/parse as CSV: {exc}")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        sys.exit(f"FATAL: downloaded file is missing expected columns {missing}; "
                 "not overwriting data/players.csv. Check whether the upstream schema changed.")

    n_valued = df["market_value_in_eur"].notna().sum()
    if len(df) < MIN_ROWS or n_valued < MIN_VALUED_ROWS:
        sys.exit(f"FATAL: snapshot looks truncated ({len(df):,} rows, {n_valued:,} valued); "
                 "refusing to overwrite data/players.csv.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "players.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows ({n_valued:,} with a market value) to {out_path}")
    return out_path


if __name__ == "__main__":
    fetch_real_valuations()
