# ⚡ WebRAG — AI-Powered Website Q&A

Scrape any website and ask questions about its content using Retrieval-Augmented Generation (RAG).

WebRAG crawls a website recursively, builds a vector index from the scraped content, and uses an LLM to answer your questions with source citations.

## Features

- **Recursive Web Scraping** — BFS crawler that follows same-domain links up to a configurable depth
- **Smart Text Extraction** — Uses [trafilatura](https://github.com/adbar/trafilatura) with BeautifulSoup fallback for clean content extraction
- **Vector Search** — FAISS index with sentence-transformer embeddings (`all-MiniLM-L6-v2`) for fast semantic retrieval
- **LLM-Powered Answers** — Groq API (Llama 3.3 70B) generates answers grounded in retrieved context
- **Source Citations** — Every answer includes `[Source N]` references back to the original pages
- **Conversation History** — Multi-turn chat with follow-up question support (last 10 turns)
- **Live Progress** — Real-time scraping progress bar with page count updates
- **Modern UI** — Glassmorphism design with Inter font, dark theme, and smooth animations

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Flask (Python) |
| Scraping | requests, BeautifulSoup, trafilatura |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS (CPU) |
| LLM | Groq API (Llama 3.3 70B) |
| Frontend | Vanilla HTML/CSS/JS |

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/farhajebin02/website-rag.git
cd website-rag
```

### 2. Create and activate virtual environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key (get one free at [console.groq.com](https://console.groq.com/keys)):

```
GROQ_API_KEY=your-groq-api-key-here
```

### 5. Run the app

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

## Usage

1. **Paste a URL** in the sidebar (e.g., `https://www.python.org`)
2. Configure **Depth** (how many link-hops to follow) and **Max Pages**
3. Click **Scrape & Index** — watch the progress bar as pages are crawled
4. Once indexing is complete, **ask questions** in the chat
5. Ask **follow-up questions** — the chatbot remembers conversation context

## Project Structure

```
website-rag/
├── app.py              # Flask server & API routes
├── rag_engine.py       # RAG pipeline (chunking, FAISS, Groq generation)
├── scraper.py          # BFS web crawler with content extraction
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── templates/
│   └── index.html      # Main HTML page
└── static/
    ├── style.css       # UI styles (glassmorphism dark theme)
    └── app.js          # Frontend logic (chat, scraping, history)
```

## Architecture

```
User Question
     │
     ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Flask API   │────▶│  RAG Engine   │────▶│  Groq LLM   │
│  (app.py)    │     │  - Retrieve   │     │  (Llama 3.3) │
└─────────────┘     │  - FAISS      │     └─────────────┘
                    │  - Embeddings │              │
                    └──────────────┘              ▼
                           ▲                  Answer with
                           │                  [Source N]
                    ┌──────────────┐          citations
                    │  Web Scraper  │
                    │  - BFS crawl  │
                    │  - trafilatura│
                    └──────────────┘
```

## License

MIT
