import pandas as pd
import requests
import os
import io
import gzip

def fetch_real_valuations():
    print("Connecting to Transfermarkt data lake...")
    
    url = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/players.csv.gz"
    
    # Add a User-Agent header to bypass the 403 Forbidden block
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Check for any new errors
        
        # Decompress the gzip file in memory and read into pandas
        with gzip.open(io.BytesIO(response.content), 'rt', encoding='utf-8') as f:
            df_tm = pd.read_csv(f, low_memory=False)
            
    except Exception as e:
        print(f"Failed to fetch data lake: {e}")
        return
        
    print("Successfully downloaded! Filtering records...")
    
    # Filter for players who actually have a market value
    df_tm = df_tm.dropna(subset=['market_value_in_eur'])
    
    # Standardize names for the merger
    df_tm['player'] = df_tm['name']
    df_tm['market_value_m'] = df_tm['market_value_in_eur'] / 1000000.0
    
    # Export a clean subset
    export_df = df_tm[['player', 'market_value_m', 'position', 'current_club_id']]
    
    os.makedirs("data", exist_ok=True)
    export_path = "data/transfermarkt_values.csv"
    export_df.to_csv(export_path, index=False)
    print(f"Saved {len(export_df)} real Transfermarkt valuations to {export_path}")

if __name__ == "__main__":
    fetch_real_valuations()