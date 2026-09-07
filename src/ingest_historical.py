import pandas as pd
import requests
import time
import os
from io import StringIO

def ingest_historical_stats():
    seasons = [
        ("2026-2027", "https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats"),
        ("2025-2026", "https://fbref.com/en/comps/Big5/2025-2026/stats/players/2025-2026-Big-5-European-Leagues-Stats"),
        ("2024-2025", "https://fbref.com/en/comps/Big5/2024-2025/stats/players/2024-2025-Big-5-European-Leagues-Stats"),
        ("2023-2024", "https://fbref.com/en/comps/Big5/2023-2024/stats/players/2023-2024-Big-5-European-Leagues-Stats"),
        ("2022-2023", "https://fbref.com/en/comps/Big5/2022-2023/stats/players/2022-2023-Big-5-European-Leagues-Stats"),
        ("2021-2022", "https://fbref.com/en/comps/Big5/2021-2022/stats/players/2021-2022-Big-5-European-Leagues-Stats")
    ]

    master_df = pd.DataFrame()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    for season, url in seasons:
        print(f"Scraping {season}...")
        try:
            res = requests.get(url, headers=headers)
            if res.status_code != 200:
                print(f"Skipping {season}: HTTP Status {res.status_code}")
                continue
                
            html = res.text.replace("<!--", "").replace("-->", "")
            
            # Read all tables and find the standard player stats table
            tables = pd.read_html(StringIO(html))
            target_df = None
            for t in tables:
                cols = [str(c).lower() for c in t.columns.get_level_values(-1)] if isinstance(t.columns, pd.MultiIndex) else [str(c).lower() for c in t.columns]
                if 'player' in cols and 'min' in cols:
                    target_df = t
                    break
            
            if target_df is None:
                print(f"Could not locate player stats table for {season}")
                continue

            # Flatten multi-level columns if necessary
            if isinstance(target_df.columns, pd.MultiIndex):
                target_df.columns = target_df.columns.get_level_values(-1)

            target_df['season'] = season
            
            # Apply the 500-minute guardrail safely
            min_col = next((c for c in target_df.columns if c.lower() == 'min'), None)
            if min_col:
                target_df[min_col] = pd.to_numeric(target_df[min_col], errors='coerce')
                target_df = target_df[target_df[min_col] >= 500].copy()
            
            master_df = pd.concat([master_df, target_df], ignore_index=True)
            print(f"Successfully pulled {season} ({len(target_df)} qualified players)")
            time.sleep(5) # Delay to respect rate limits
            
        except Exception as e:
            print(f"Failed parsing {season}: {e}")

    if not master_df.empty:
        master_df.columns = [str(c).lower() for c in master_df.columns]
        os.makedirs("data", exist_ok=True)
        master_df.to_csv("data/fbref_historical_stats.csv", index=False)
        print(f"\nSuccessfully compiled historical dataset with {len(master_df)} total player-seasons.")
    else:
        print("\nNo data collected. Check network or rate limits.")

if __name__ == "__main__":
    ingest_historical_stats()