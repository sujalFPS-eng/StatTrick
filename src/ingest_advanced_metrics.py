import pandas as pd
from curl_cffi import requests
import time
import os
from io import StringIO

def scrape_fbref():
    # Target FBref's specialized metric endpoints
    urls = {
        "standard": "https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats",
        "passing": "https://fbref.com/en/comps/Big5/passing/players/Big-5-European-Leagues-Passing",
        "defense": "https://fbref.com/en/comps/Big5/defense/players/Big-5-European-Leagues-Defensive-Actions",
        "possession": "https://fbref.com/en/comps/Big5/possession/players/Big-5-European-Leagues-Possession"
    }

    dataframes = {}

    for stat_type, url in urls.items():
        print(f"Scraping {stat_type} metrics...")
        try:
            # impersonate="chrome" handles the TLS fingerprinting to bypass the 403 Cloudflare block
            res = requests.get(url, impersonate="chrome")
            if res.status_code != 200:
                print(f"Failed to fetch {stat_type}. HTTP {res.status_code}")
                continue
                
            html = res.text.replace("<!--", "").replace("-->", "")
            tables = pd.read_html(StringIO(html))
            
            target_df = None
            for t in tables:
                # Flatten multi-level columns
                cols = [str(c).lower() for c in t.columns.get_level_values(-1)] if isinstance(t.columns, pd.MultiIndex) else [str(c).lower() for c in t.columns]
                if 'player' in cols and 'squad' in cols:
                    target_df = t
                    break
            
            if target_df is None:
                continue

            if isinstance(target_df.columns, pd.MultiIndex):
                target_df.columns = target_df.columns.get_level_values(-1)
                
            target_df.columns = [str(c).lower() for c in target_df.columns]
            
            # Remove the repeating header rows FBref injects into tables
            target_df = target_df[target_df['player'] != 'Player'].copy()
            
            # Remove duplicated biographical columns (age, nation, position) that appear in every table
            if stat_type != "standard":
                keep_cols = ['player', 'squad'] + [c for c in target_df.columns if c not in dataframes['standard'].columns]
                target_df = target_df[keep_cols]
                
            dataframes[stat_type] = target_df
            time.sleep(5) # Respect rate limits
            
        except Exception as e:
            print(f"Error scraping {stat_type}: {e}")

    if 'standard' not in dataframes:
        print("Failed to pull base data. Exiting.")
        return

    print("\nMerging tactical dataframes...")
    df_final = dataframes['standard']
    for stat_type in ['passing', 'defense', 'possession']:
        if stat_type in dataframes:
            # Merge on player AND squad to prevent mismatches for players with identical names
            df_final = df_final.merge(dataframes[stat_type], on=['player', 'squad'], how='left')

    os.makedirs("data", exist_ok=True)
    df_final.to_csv("data/fbref_clean_stats.csv", index=False)
    print(f"Successfully compiled advanced metrics for {len(df_final)} active players.")

if __name__ == "__main__":
    scrape_fbref()