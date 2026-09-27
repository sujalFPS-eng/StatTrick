import pandas as pd
import numpy as np
from seleniumbase import SB
from bs4 import BeautifulSoup, Comment
import os
from io import StringIO

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

def scrape_fbref_stealth():
    print("🤖 Initializing SeleniumBase (Undetected ChromeDriver)...")
    player_url = "https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats"
    defense_url = "https://fbref.com/en/comps/Big5/defense/players/Big-5-European-Leagues-Stats"
    possession_url = "https://fbref.com/en/comps/Big5/possession/squads/Big-5-European-Leagues-Stats"

    try:
        with SB(uc=True, headless=True) as sb:
            # --- 1. FETCH STANDARD STATS ---
            print(f"📡 Navigating to Standard Stats: {player_url}")
            sb.open(player_url)
            table_selector = "table[id^='stats_standard']"
            
            cleared = False
            for elapsed in range(30):
                if sb.is_element_present(table_selector):
                    cleared = True
                    break
                sb.sleep(1)

            if not cleared:
                print("❌ Standard table never appeared.")
                return

            sb.sleep(2)
            html = sb.get_page_source()
            player_table = find_stats_table(html, "stats_standard")
            
            df = pd.read_html(StringIO(str(player_table)))[0]
            if df.columns.nlevels > 1: df.columns = df.columns.droplevel()

            df = df[df["Player"] != "Player"].copy()
            df["Age"] = df["Age"].astype(str).str[:2]
            df["Pos"] = df["Pos"].astype(str).str[:2]
            df["Min"] = pd.to_numeric(df["Min"].astype(str).str.replace(",", ""), errors="coerce")
            df = df.dropna(subset=["Min"])
            df = df[df["Min"] > 90]

            # --- 2. FETCH DEFENSE STATS ---
            print(f"📡 Navigating to Defense Stats: {defense_url}")
            sb.open(defense_url)
            sb.sleep(2)
            
            def_html = sb.get_page_source()
            def_table = find_stats_table(def_html, "stats_defense")
            
            if def_table:
                print("✅ Defense stats found. Merging...")
                df_def = pd.read_html(StringIO(str(def_table)))[0]
                
                # 1. Flatten multi-level columns
                if df_def.columns.nlevels > 1: 
                    df_def.columns = df_def.columns.droplevel()
                
                # 2. Fix the duplicate "Tkl" column issue (keeps the first 'Total Tackles')
                df_def = df_def.loc[:, ~df_def.columns.duplicated()].copy()
                
                # 3. Rename "Blocks" to "Blk" to match the pipeline architecture
                if "Blocks" in df_def.columns:
                    df_def = df_def.rename(columns={"Blocks": "Blk"})
                
                df_def = df_def[df_def["Player"] != "Player"].copy()
                
                # 4. Safely extract
                df_def = df_def[["Player", "Squad", "Tkl", "Int", "Clr", "Blk"]]
                df = df.merge(df_def, on=["Player", "Squad"], how="left")

                # Calculate base per-90 metrics before possession adjustment
                for col in ["Tkl", "Int", "Clr", "Blk"]:
                    per90_name = f"{col.lower()}_per90"
                    df[per90_name] = pd.to_numeric(df[col], errors="coerce") / (df["Min"] / 90.0)

            # --- 3. FETCH SQUAD POSSESSION ---
            print(f"📡 Navigating to Squad Possession: {possession_url}")
            sb.open(possession_url)
            sb.sleep(2) 
            
            squad_html = sb.get_page_source()
            squad_table = find_stats_table(squad_html, "stats_squads_possession_for")
            
            if squad_table:
                print("✅ Squad possession found. Applying PAdj math...")
                sq_df = pd.read_html(StringIO(str(squad_table)))[0]
                if sq_df.columns.nlevels > 1: sq_df.columns = sq_df.columns.droplevel()
                
                sq_df = sq_df[["Squad", "Poss"]].copy()
                sq_df["Poss"] = pd.to_numeric(sq_df["Poss"], errors="coerce")
                
                df = df.merge(sq_df, on="Squad", how="left")
                df["Poss"] = df["Poss"].fillna(50.0) 
                
                # Apply Sigmoid PAdj
                multiplier = 1 + (0.8 / (1 + np.exp(-0.08 * (df["Poss"] - 50)))) - 0.4
                padj_cols = ["tkl_per90", "int_per90", "clr_per90", "blk_per90"]
                for col in padj_cols:
                    if col in df.columns:
                        df[col] = df[col] * multiplier

            os.makedirs("data", exist_ok=True)
            output_path = "data/live_fbref_data.csv"
            df.to_csv(output_path, index=False)

            print(f"🎯 Ingestion complete! Scraped & Adjusted {len(df)} active players.")

    except Exception as e:
        print(f"❌ Scraping error occurred: {e}")

if __name__ == "__main__":
    scrape_fbref_stealth()