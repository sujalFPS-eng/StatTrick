"""Open the Streamlit app in a real browser so Community Cloud counts it as traffic.

A plain HTTP request is not enough: the app only counts as visited when a browser opens the
page and its websocket session starts. If the app is already asleep, click the wake-up button.
Exits non-zero if the app does not render, so a failed run shows up in the Actions tab.

It also prints how long each stage took, so the Actions log doubles as a load-time record.
"""
import os
import sys
import time

from playwright.sync_api import TimeoutError as PlaywrightTimeout, sync_playwright

URL = os.environ.get("APP_URL", "https://stattrick.streamlit.app/")
WAKE_BUTTON = "Yes, get this app back up"
READY_TEXT = "StatTrick"          # the hero title, rendered once the script has run
RUNNING = "[data-testid='stStatusWidget']"   # Streamlit's "Running..." indicator


def main() -> int:
    t0 = time.monotonic()
    lap = lambda: f"{time.monotonic() - t0:6.1f}s"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        sizes = {"http": 0, "ws": 0}
        files = []

        def on_response(r):
            try:
                n = len(r.body())
                sizes["http"] += n
                files.append((n, r.headers.get("content-encoding", "-"), r.url[:110]))
            except Exception:
                pass

        def on_ws(ws):
            ws.on("framereceived", lambda f: sizes.__setitem__("ws", sizes["ws"] + len(f)))

        page.on("response", on_response)
        page.on("websocket", on_ws)
        page.goto(URL, wait_until="domcontentloaded", timeout=120_000)
        print(f"{lap()}  page shell loaded")

        # Poll for either the wake-up button (asleep) or the app's title (awake and rendered).
        app = page.frame_locator("iframe[title='streamlitApp']")   # the app runs in an iframe
        wake = page.get_by_role("button", name=WAKE_BUTTON)
        title = app.get_by_text(READY_TEXT).first
        woke = False
        deadline = time.monotonic() + 360
        while True:
            if not woke and wake.count() and wake.is_visible():
                wake.click()
                woke = True
                print(f"{lap()}  App was asleep; clicked the wake-up button.")
            try:
                if title.is_visible():
                    break
            except Exception:
                pass                    # iframe not attached yet
            if time.monotonic() > deadline:
                page.screenshot(path="keep_awake_failure.png", full_page=True)
                print(f"{lap()}  App did not render within 6 minutes.")
                return 1
            page.wait_for_timeout(250)
        if not woke:
            print(f"{lap()}  App was awake.")
        print(f"{lap()}  title visible")

        # Wait for the first script run to finish (the "Running..." widget goes away).
        page.wait_for_timeout(500)
        try:
            app.locator(RUNNING).wait_for(state="detached", timeout=180_000)
            print(f"{lap()}  first run finished (page fully loaded)")
        except PlaywrightTimeout:
            print(f"{lap()}  still running after 3 more minutes")

        page.wait_for_timeout(20_000)   # stay a little so the session (and cache warm-up) completes
        print(f"downloaded: {sizes['http'] / 1e6:.1f} MB of files, {sizes['ws'] / 1e6:.1f} MB of app data")
        for n, enc, u in sorted(files, reverse=True)[:5]:
            print(f"  {n / 1e6:5.2f} MB  {enc:5}  {u}")
        print("App rendered.")
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
