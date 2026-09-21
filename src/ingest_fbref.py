import pandas as pd
from seleniumbase import SB
from bs4 import BeautifulSoup, Comment
import os
from io import StringIO

def find_stats_table(html: str, table_id_prefix: str = "stats_standard"):
    """Find the target table, checking both the live DOM and HTML comments
    (FBref hides several of its tables inside comments)."""
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
    url = "https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats"

    try:
        with SB(uc=True, headless=True) as sb:
            print(f"📡 Navigating to FBref: {url}")
            sb.open(url)

            print("⏳ Waiting for the stats table to actually appear...")
            table_selector = "table[id^='stats_standard']"
            cleared = False
            for elapsed in range(30):
                title = sb.get_title()
                has_table = sb.is_element_present(table_selector)
                print(f"  [{elapsed}s] title={title!r} table_present={has_table}")
                if has_table:
                    cleared = True
                    print(f"✅ Stats table found after {elapsed}s. Title: {title!r}")
                    break
                sb.sleep(1)

            if not cleared:
                os.makedirs("data", exist_ok=True)
                with open("data/debug_page.html", "w", encoding="utf-8") as f:
                    f.write(sb.get_page_source())
                print("❌ Table never appeared. Saved page to data/debug_page.html for inspection.")
                return

            sb.sleep(2)
            html = sb.get_page_source()
            print("📊 Locating the stats table (DOM + commented-out tables)...")

            table = find_stats_table(html, "stats_standard")
            if table is None:
                os.makedirs("data", exist_ok=True)
                with open("data/debug_page.html", "w", encoding="utf-8") as f:
                    f.write(html)
                print("❌ Could not locate stats_standard table anywhere on the page. Saved data/debug_page.html.")
                return

            tables = pd.read_html(StringIO(str(table)))
            df = tables[0]
            print(f"Columns before cleanup: {df.columns.nlevels} level(s), {len(df)} rows")

            if df.columns.nlevels > 1:
                df.columns = df.columns.droplevel()

            df = df[df["Player"] != "Player"].copy()
            df["Age"] = df["Age"].astype(str).str[:2]
            df["Pos"] = df["Pos"].astype(str).str[:2]
            df["Min"] = pd.to_numeric(df["Min"].astype(str).str.replace(",", ""), errors="coerce")
            df = df.dropna(subset=["Min"])
            df = df[df["Min"] > 90]

            os.makedirs("data", exist_ok=True)
            output_path = "data/live_fbref_data.csv"
            df.to_csv(output_path, index=False)

            print(f"🎯 Ingestion complete! Scraped {len(df)} active players.")
            print(f"💾 Saved safely to {output_path}")

    except Exception as e:
        print(f"❌ Scraping error occurred: {e}")

if __name__ == "__main__":
    scrape_fbref_stealth()