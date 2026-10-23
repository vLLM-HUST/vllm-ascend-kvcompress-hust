"""Check the maintained page against external originals and read failures."""

import argparse
import functools
import http.server
import json
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--browser", help="Use an existing Chromium executable")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error("browser output must be external")
    output.mkdir(parents=True, exist_ok=True)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=ROOT)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = (
        f"http://127.0.0.1:{server.server_port}/docs/results/benchmark-leaderboard.html"
    )
    report = []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                executable_path=args.browser, args=["--no-sandbox"]
            )
            for width in (1440, 390):
                for scheme in ("light", "dark"):
                    page = browser.new_page(
                        viewport={"width": width, "height": 960}, color_scheme=scheme
                    )
                    errors = []
                    page.on(
                        "pageerror",
                        lambda error, errors=errors: errors.append(str(error)),
                    )
                    assert page.goto(url).status == 200
                    page.wait_for_function(
                        "document.querySelector('#record-count').textContent.trim()"
                        " === '28 / 28'",
                        timeout=60000,
                    )
                    links = page.locator("a").evaluate_all(
                        "nodes => nodes.map(n => n.href)"
                    )
                    evidence = [link for link in links if "/snapshot/" in link]
                    assert evidence and all(
                        "vllm-hust-benchmark/" in link for link in evidence
                    )
                    assert not page.locator('a[href^="evidence/"]').count()
                    assert not page.locator('a[href^="../tests/"]').count()
                    assert "Method only" in page.locator("body").inner_text()
                    page.locator("#view-frontier").click()
                    assert page.locator("#frontier-panel").is_visible()
                    assert not errors, errors
                    page.screenshot(
                        path=str(output / f"leaderboard-{width}-{scheme}.png"),
                        full_page=True,
                    )
                    report.append(
                        {
                            "width": width,
                            "scheme": scheme,
                            "history_records": 28,
                            "evidence_links": len(evidence),
                        }
                    )
                    page.close()
            page = browser.new_page()
            page.route(
                "**/snapshot/docs/results/evidence/*.json", lambda route: route.abort()
            )
            page.goto(url)
            page.locator("#empty").filter(
                has_text="Original data unavailable"
            ).wait_for()
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    (output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
