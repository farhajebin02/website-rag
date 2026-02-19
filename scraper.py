"""
Web Scraper Module — Recursive URL crawler with content extraction.
Uses BFS to discover linked pages within the same domain,
extracts clean text via trafilatura with BeautifulSoup fallback.
"""

import re
import time
import logging
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass, field
from collections import deque
from typing import List, Optional, Callable

import requests
from bs4 import BeautifulSoup

try:
    import trafilatura
except ImportError:
    trafilatura = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """A single scraped page."""
    url: str
    title: str
    content: str
    depth: int = 0
    links: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

class WebScraper:
    """
    BFS-based recursive web scraper.

    Parameters
    ----------
    max_depth : int
        How many link-hops away from the seed URL to follow (default 2).
    max_pages : int
        Maximum number of pages to scrape (default 50).
    delay : float
        Seconds to wait between requests (politeness, default 0.5).
    timeout : int
        HTTP request timeout in seconds (default 15).
    """

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; WebsiteRAGBot/1.0; "
            "+https://github.com/website-rag)"
        )
    }

    SKIP_EXTENSIONS = {
        ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp",
        ".mp3", ".mp4", ".avi", ".mov", ".zip", ".tar", ".gz",
        ".exe", ".dmg", ".css", ".js", ".woff", ".woff2", ".ttf",
        ".ico", ".xml", ".json", ".rss",
    }

    def __init__(
        self,
        max_depth: int = 2,
        max_pages: int = 50,
        delay: float = 0.5,
        timeout: int = 15,
    ):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.delay = delay
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape(
        self,
        seed_url: str,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[Document]:
        """
        Scrape starting from *seed_url*.

        Parameters
        ----------
        seed_url : str
            The starting URL.
        on_progress : callable, optional
            ``(scraped_count, total_queued, current_url)`` callback for live
            progress updates.

        Returns
        -------
        list[Document]
        """
        seed_url = self._normalise(seed_url)
        base_domain = urlparse(seed_url).netloc

        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque()
        queue.append((seed_url, 0))
        documents: list[Document] = []

        while queue and len(documents) < self.max_pages:
            url, depth = queue.popleft()
            if url in visited:
                continue
            visited.add(url)

            if on_progress:
                on_progress(len(documents), len(queue), url)

            doc = self._fetch_and_parse(url, depth)
            if doc is None:
                continue

            documents.append(doc)
            logger.info(
                "Scraped %d/%d  depth=%d  %s",
                len(documents), self.max_pages, depth, url,
            )

            # Enqueue child links (same domain only)
            if depth < self.max_depth:
                for link in doc.links:
                    if link not in visited and self._same_domain(link, base_domain):
                        queue.append((link, depth + 1))

            time.sleep(self.delay)

        logger.info("Scraping complete — %d documents collected.", len(documents))
        return documents

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_and_parse(self, url: str, depth: int) -> Optional[Document]:
        """Download a page and extract its clean text content."""
        try:
            resp = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                return None

            html = resp.text
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            return None

        # --- extract title ---
        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else url

        # --- extract main text ---
        text = self._extract_text(html, url)
        if not text or len(text.strip()) < 50:
            return None  # skip near-empty pages

        # --- extract same-page links ---
        links = self._extract_links(soup, url)

        return Document(url=url, title=title, content=text, depth=depth, links=links)

    def _extract_text(self, html: str, url: str) -> str:
        """Use trafilatura first, fall back to BeautifulSoup."""
        if trafilatura:
            text = trafilatura.extract(
                html,
                url=url,
                include_tables=True,
                include_links=False,
                include_comments=False,
                favor_recall=True,
            )
            if text and len(text.strip()) > 50:
                return text

        # Fallback: BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        # Remove noise elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        # Extract structured data (tables)
        structured_parts = []
        for table in soup.find_all("table"):
            rows = []
            for tr in table.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                structured_parts.append("\n".join(rows))
            table.decompose()  # remove so we don't double-count

        body = soup.find("body")
        main_text = body.get_text(separator="\n", strip=True) if body else soup.get_text(separator="\n", strip=True)

        # Collapse whitespace
        main_text = re.sub(r"\n{3,}", "\n\n", main_text)

        if structured_parts:
            main_text += "\n\n" + "\n\n".join(structured_parts)

        return main_text

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Return absolute, de-duped links from anchor tags."""
        links: list[str] = []
        seen: set[str] = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            abs_url = self._normalise(urljoin(base_url, href))

            # Skip non-http, anchors-only, and binary files
            if not abs_url.startswith("http"):
                continue
            ext = self._get_extension(abs_url)
            if ext in self.SKIP_EXTENSIONS:
                continue
            if abs_url not in seen:
                seen.add(abs_url)
                links.append(abs_url)
        return links

    @staticmethod
    def _normalise(url: str) -> str:
        """Strip fragment and trailing slash for dedup."""
        parsed = urlparse(url)
        clean = parsed._replace(fragment="")
        result = clean.geturl().rstrip("/")
        return result

    @staticmethod
    def _same_domain(url: str, domain: str) -> bool:
        return urlparse(url).netloc == domain

    @staticmethod
    def _get_extension(url: str) -> str:
        path = urlparse(url).path
        dot = path.rfind(".")
        if dot != -1:
            return path[dot:].lower().split("?")[0]
        return ""
