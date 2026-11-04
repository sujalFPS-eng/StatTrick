from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page()
    pg.on("response", lambda r: print("RESP", r.status, r.url[:120]) if "streamlit.app" in r.url and r.status >= 400 else None)
    pg.goto("https://stattrick.streamlit.app/", wait_until="domcontentloaded", timeout=120000)
    for i in range(6):
        pg.wait_for_timeout(30000)
        print(f"--- t={30*(i+1)}s main page text ---"); print(pg.inner_text("body")[:800])
        for f in pg.frames[1:]:
            try: print("--- frame", f.url[:100], "---"); print(f.inner_text("body")[:1500])
            except Exception as ex: print("frame err", ex)
    b.close()
