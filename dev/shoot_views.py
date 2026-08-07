"""Screenshot every served view, one image per endpoint.

    python dev/serve_views.py &
    python dev/shoot_views.py

Writes /tmp/lav_<name>.png. One browser session covers every endpoint, and each capture
waits for Panel's loading overlay to clear rather than sleeping a fixed duration, which is
the usual cause of a screenshot showing a spinner.
"""

from __future__ import annotations

import sys

from playwright.sync_api import sync_playwright
from views import VIEWS

PORT = 5007
OUT = "/tmp/lav_{name}.png"


def main(names: list[str]) -> int:
    targets = names or list(VIEWS)
    unknown = [name for name in targets if name not in VIEWS]
    if unknown:
        print(f"unknown view(s): {', '.join(unknown)}")
        return 2

    console: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.on(
            "console",
            lambda msg: console.append(msg.text) if msg.type == "error" else None,
        )
        for name in targets:
            console.clear()
            url = f"http://localhost:{PORT}/{name}"
            page.goto(url, wait_until="networkidle")
            page.wait_for_function(
                "() => !document.querySelector('.pn-loading')", timeout=60000
            )
            page.wait_for_selector("canvas", timeout=60000)
            page.wait_for_timeout(500)
            path = OUT.format(name=name)
            page.screenshot(path=path)
            print(f"wrote {path}")
            for message in console:
                print(f"  console error: {message}")
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
