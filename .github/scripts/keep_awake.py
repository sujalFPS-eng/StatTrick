"""Open the Streamlit app in a real browser so Community Cloud counts it as traffic.

A plain HTTP request is not enough: the app only counts as visited when a browser opens the
page and its websocket session starts. If the app is already asleep, click the wake-up button.
Exits non-zero if the app does not render, so a failed run shows up in the Actions tab.
"""
import os
import sys

from playwright.sync_api import TimeoutError as PlaywrightTimeout, sync_playwright

URL = os.environ.get("APP_URL", "https://stattrick.streamlit.app/")
WAKE_BUTTON = "Yes, get this app back up"
READY_TEXT = "StatTrick"          # the hero title, rendered once the script has run


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(URL, wait_until="domcontentloaded", timeout=120_000)

        try:
            page.get_by_role("button", name=WAKE_BUTTON).click(timeout=15_000)
            print("App was asleep; clicked the wake-up button.")
        except PlaywrightTimeout:
            print("App was awake.")

        # The app itself runs inside an iframe on *.streamlit.app.
        app = page.frame_locator("iframe[title='streamlitApp']")
        try:
            app.get_by_text(READY_TEXT).first.wait_for(timeout=360_000)
        except PlaywrightTimeout:
            page.screenshot(path="keep_awake_failure.png", full_page=True)
            print("App did not render within 6 minutes.")
            return 1

        page.wait_for_timeout(20_000)   # stay a little so the session (and cache warm-up) completes
        print("App rendered.")
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
