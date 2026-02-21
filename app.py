"""
Flask Application — Serves the chat UI and exposes scraping/query APIs.
"""

import json
import logging
import os
import threading

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, Response, stream_with_context

from scraper import WebScraper
from rag_engine import RAGEngine

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-18s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global state
rag_engine = RAGEngine(groq_api_key=os.getenv("GROQ_API_KEY"))
scrape_status = {
    "active": False,
    "progress": 0,
    "total": 0,
    "current_url": "",
    "message": "",
    "done": False,
    "error": None,
}
status_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    """Return current index stats and scraping status."""
    stats = rag_engine.stats if rag_engine.is_ready else {"total_chunks": 0, "total_documents": 0}
    with status_lock:
        scraping = dict(scrape_status)
    return jsonify({"index": stats, "scraping": scraping})


@app.route("/api/scrape", methods=["POST"])
def api_scrape():
    """Kick off a scraping job in a background thread."""
    with status_lock:
        if scrape_status["active"]:
            return jsonify({"error": "A scraping job is already running."}), 409

    data = request.get_json(force=True)
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "URL is required."}), 400
    if not url.startswith("http"):
        url = "https://" + url

    max_depth = int(data.get("max_depth", 2))
    max_pages = int(data.get("max_pages", 50))

    # Reset status
    with status_lock:
        scrape_status.update({
            "active": True,
            "progress": 0,
            "total": 0,
            "current_url": url,
            "message": "Starting…",
            "done": False,
            "error": None,
        })

    def _run():
        try:
            scraper = WebScraper(max_depth=max_depth, max_pages=max_pages)

            def on_progress(scraped, queued, cur_url):
                with status_lock:
                    scrape_status["progress"] = scraped
                    scrape_status["total"] = scraped + queued
                    scrape_status["current_url"] = cur_url
                    scrape_status["message"] = f"Scraped {scraped} pages…"

            docs = scraper.scrape(url, on_progress=on_progress)

            # Convert to dicts for ingestion
            doc_dicts = [
                {"url": d.url, "title": d.title, "content": d.content}
                for d in docs
            ]

            with status_lock:
                scrape_status["message"] = "Building vector index…"

            stats = rag_engine.ingest_documents(doc_dicts)

            with status_lock:
                scrape_status.update({
                    "active": False,
                    "done": True,
                    "message": (
                        f"Done! Indexed {stats['total_chunks']} chunks "
                        f"from {stats['total_documents']} pages."
                    ),
                    "progress": stats["total_documents"],
                    "total": stats["total_documents"],
                })

        except Exception as exc:
            logger.exception("Scraping failed")
            with status_lock:
                scrape_status.update({
                    "active": False,
                    "done": True,
                    "error": str(exc),
                    "message": f"Error: {exc}",
                })

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return jsonify({"status": "started", "url": url})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Answer a question using the RAG pipeline."""
    if not rag_engine.is_ready:
        return jsonify({
            "answer": "No data loaded yet. Please scrape a website first!",
            "sources": [],
        })

    data = request.get_json(force=True)
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Question is required."}), 400

    history = data.get("history", [])
    result = rag_engine.query(question, chat_history=history)
    return jsonify({
        "answer": result.answer,
        "sources": result.sources,
    })


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
