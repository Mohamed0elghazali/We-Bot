"""
Scraper for te.eg — crawls all pages and outputs markdown files.
Uses crawl4ai for JS rendering and clean markdown extraction.
"""

import asyncio
import re
import json
import uuid
from pathlib import Path
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator


# ── Config ────────────────────────────────────────────────────────────────────

START_URL = "https://te.eg"
ALLOWED_DOMAIN = "te.eg"
OUTPUT_DIR = Path("data/scraped")
MAX_PAGES = 500          # safety cap
MAX_DEPTH = 4            # link-follow depth
CONCURRENCY = 5          # parallel browser tabs


# ── Helpers ───────────────────────────────────────────────────────────────────

def url_to_filename(url: str) -> str:
    """Convert a URL to a safe, short markdown filename."""
    parsed = urlparse(url)
    slug = parsed.path.strip("/").split("/")[-1] or "index"  # just the last path segment
    slug = re.sub(r"[^\w\-]", "_", slug)[:60]               # sanitise + cap at 60 chars
    short_id = uuid.uuid4().hex[:8]
    return f"{slug}__{short_id}.md"


def is_allowed(url: str) -> bool:
    """Only follow links that stay on the allowed domain."""
    try:
        return urlparse(url).netloc.endswith(ALLOWED_DOMAIN)
    except Exception:
        return False


def extract_links(result) -> list[str]:
    """Pull internal links from a crawl result."""
    links = []
    if result.links:
        for link in result.links.get("internal", []):
            href = link.get("href", "")
            if href and is_allowed(href):
                links.append(href)
    return links


# ── Core crawler ──────────────────────────────────────────────────────────────

async def crawl():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    browser_cfg = BrowserConfig(
        headless=True,
        verbose=False,
        # mimic a real browser to bypass antibot detection
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        headers={
            "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )

    run_cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        markdown_generator=DefaultMarkdownGenerator(
            options={"ignore_links": False, "body_width": 0}
        ),
        wait_until="domcontentloaded",  # don't wait for networkidle — te.eg never settles
        page_timeout=60_000,            # 60 s per page
        delay_before_return_html=2.0,   # small pause after DOM load for JS to render
        exclude_external_links=True,
        exclude_social_media_links=True,
    )

    visited: set[str] = set()
    # queue items: (url, depth)
    queue: list[tuple[str, int]] = [(START_URL, 0)]
    saved = 0
    manifest: list[dict] = []

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        while queue and saved < MAX_PAGES:
            batch = []
            next_queue: list[tuple[str, int]] = []

            # take up to CONCURRENCY items at the current depth level
            while queue and len(batch) < CONCURRENCY:
                url, depth = queue.pop(0)
                if url in visited:
                    continue
                visited.add(url)
                batch.append((url, depth))

            if not batch:
                break

            urls = [u for u, _ in batch]
            depths = {u: d for u, d in batch}

            print(f"[crawl] fetching {len(urls)} pages …")
            results = await crawler.arun_many(urls, config=run_cfg)

            for result in results:
                url = result.url
                depth = depths.get(url, 0)

                if not result.success:
                    print(f"  [skip] {url} — {result.error_message}")
                    continue

                # ── save markdown ──────────────────────────────────────────
                md_content = result.markdown or ""
                if not md_content.strip():
                    print(f"  [empty] {url}")
                    continue

                filename = url_to_filename(url)
                out_path = OUTPUT_DIR / filename

                header = f"---\nurl: {url}\ntitle: {result.metadata.get('title', '')}\n---\n\n"
                out_path.write_text(header + md_content, encoding="utf-8")
                saved += 1
                print(f"  [saved] ({saved}) {url} → {filename}")
                manifest.append({
                    "file": filename,
                    "url": url,
                    "title": result.metadata.get("title", ""),
                    "depth": depth,
                })

                # ── enqueue child links ────────────────────────────────────
                if depth < MAX_DEPTH:
                    for link in extract_links(result):
                        if link not in visited:
                            next_queue.append((link, depth + 1))

            queue = next_queue + queue  # depth-first keeps memory bounded

    print(f"\nDone. {saved} pages saved to {OUTPUT_DIR}/")

    manifest_path = Path("data") / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Manifest written → {manifest_path}")


if __name__ == "__main__":
    asyncio.run(crawl())
