import pandas as pd
import os
from io import StringIO

def parse_local_fbref_html(filepath):
    print(f"Parsing local file: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        html_content = f.read().replace('<!--', '').replace('-->', '')
    
    tables = pd.read_html(StringIO(html_content))
    
    df = None
    # Loop through all tables on the page to find the main player dataset
    for tbl in tables:
        # Safely flatten multi-level headers by taking the lowest level
        if isinstance(tbl.columns, pd.MultiIndex):
            tbl.columns = tbl.columns.get_level_values(-1)
        
        # Identify the correct table by verifying core columns exist
        if 'Player' in tbl.columns and 'Squad' in tbl.columns:
            df = tbl
            break
            
    if df is None:
        raise ValueError(f"Could not find the main Player stats table in {filepath}")
        
    # Remove the repeating header rows
    df = df[df['Player'] != 'Player'].copy()
    return df

def run_local_pipeline():
    df_std = parse_local_fbref_html("data/fbref_standard.html")
    df_sht = parse_local_fbref_html("data/fbref_shooting.html")
    
    print(f"Loaded {len(df_std)} standard rows and {len(df_sht)} shooting rows.")

    print("Merging standard and shooting stats...")
    # Join on core identifiers
    merge_cols = ['Player', 'Squad', 'Comp']
    master_df = pd.merge(df_std, df_sht, on=merge_cols, how='inner', suffixes=('', '_sht'))
    
    # 500-minute filter
    if 'Min' in master_df.columns:
        master_df['Min'] = pd.to_numeric(master_df['Min'], errors='coerce').fillna(0)
        original_count = len(master_df)
        master_df = master_df[master_df['Min'] >= 500]
        print(f"Filtered out {original_count - len(master_df)} players with < 500 minutes played.")
    
    # Standardize column names
    master_df.rename(columns={'Comp': 'league', 'Player': 'player'}, inplace=True)
    
    os.makedirs("data", exist_ok=True)
    export_path = "data/fbref_clean_stats.csv"
    master_df.to_csv(export_path, index=False)
    print(f"Success! {len(master_df)} full player profiles saved to {export_path}")

if __name__ == "__main__":
    run_local_pipeline()