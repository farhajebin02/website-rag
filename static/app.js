/* ============================================================
   WebRAG — Frontend logic
   ============================================================ */

const $ = (sel) => document.querySelector(sel);

// DOM refs
const urlInput = $("#urlInput");
const maxDepthIn = $("#maxDepth");
const maxPagesIn = $("#maxPages");
const scrapeBtn = $("#scrapeBtn");
const btnText = scrapeBtn.querySelector(".btn-text");
const btnLoader = scrapeBtn.querySelector(".btn-loader");
const progressArea = $("#progressArea");
const progressFill = $("#progressFill");
const progressMsg = $("#progressMsg");
const statsCard = $("#statsCard");
const statDocs = $("#statDocs");
const statChunks = $("#statChunks");
const indexBadge = $("#indexBadge");
const badgeDot = indexBadge.querySelector(".badge-dot");
const badgeText = $("#badgeText");
const chatMessages = $("#chatMessages");
const chatForm = $("#chatForm");
const chatInput = $("#chatInput");
const sendBtn = $("#sendBtn");

let pollTimer = null;

// =========================================================
// Scraping
// =========================================================

scrapeBtn.addEventListener("click", async () => {
    const url = urlInput.value.trim();
    if (!url) { urlInput.focus(); return; }

    scrapeBtn.disabled = true;
    btnText.hidden = true;
    btnLoader.hidden = false;
    progressArea.hidden = false;
    progressFill.style.width = "0%";
    progressMsg.textContent = "Starting…";

    try {
        const res = await fetch("/api/scrape", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                url,
                max_depth: parseInt(maxDepthIn.value, 10),
                max_pages: parseInt(maxPagesIn.value, 10),
            }),
        });
        const data = await res.json();
        if (data.error) {
            progressMsg.textContent = `Error: ${data.error}`;
            resetScrapeBtn();
            return;
        }
        // Start polling for progress
        startPolling();
    } catch (err) {
        progressMsg.textContent = `Network error: ${err.message}`;
        resetScrapeBtn();
    }
});

function startPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(async () => {
        try {
            const res = await fetch("/api/status");
            const data = await res.json();
            const s = data.scraping;
            const idx = data.index;

            // Progress bar
            if (s.total > 0) {
                const pct = Math.min(100, Math.round((s.progress / s.total) * 100));
                progressFill.style.width = pct + "%";
            }
            progressMsg.textContent = s.message || "Working…";

            if (s.done || !s.active) {
                clearInterval(pollTimer);
                pollTimer = null;
                resetScrapeBtn();

                if (s.error) {
                    progressMsg.textContent = `Error: ${s.error}`;
                } else {
                    progressFill.style.width = "100%";
                }

                // Update stats
                updateStats(idx);
            }
        } catch { /* ignore transient fetch errors */ }
    }, 1000);
}

function resetScrapeBtn() {
    scrapeBtn.disabled = false;
    btnText.hidden = false;
    btnLoader.hidden = true;
}

function updateStats(idx) {
    if (!idx) return;
    statDocs.textContent = idx.total_documents;
    statChunks.textContent = idx.total_chunks;

    if (idx.total_chunks > 0) {
        statsCard.hidden = false;
        badgeDot.classList.add("active");
        badgeText.textContent = `${idx.total_chunks} chunks`;
        chatInput.disabled = false;
        sendBtn.disabled = false;
        chatInput.placeholder = "Ask a question about the scraped content…";
    }
}

// =========================================================
// Chat
// =========================================================

chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = chatInput.value.trim();
    if (!question) return;

    // Remove welcome message
    const welcome = chatMessages.querySelector(".welcome-msg");
    if (welcome) welcome.remove();

    appendMessage("user", question);
    chatInput.value = "";
    chatInput.disabled = true;
    sendBtn.disabled = true;

    // Show typing indicator
    const typing = appendTyping();

    try {
        const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question }),
        });
        const data = await res.json();
        typing.remove();

        if (data.error) {
            appendMessage("bot", `Error: ${data.error}`);
        } else {
            appendMessage("bot", data.answer, data.sources);
        }
    } catch (err) {
        typing.remove();
        appendMessage("bot", `Network error: ${err.message}`);
    } finally {
        chatInput.disabled = false;
        sendBtn.disabled = false;
        chatInput.focus();
    }
});

function appendMessage(role, text, sources = []) {
    const msg = document.createElement("div");
    msg.className = `msg ${role}`;

    let html = `<div class="msg-bubble">${escapeHtml(text)}</div>`;

    if (sources && sources.length > 0) {
        let srcHtml = '<details class="msg-sources"><summary>📎 Sources (' + sources.length + ')</summary><ul class="source-list">';
        sources.forEach((s) => {
            const title = s.title || s.url;
            srcHtml += `<li><a href="${escapeAttr(s.url)}" target="_blank" rel="noopener">${escapeHtml(title)}</a></li>`;
        });
        srcHtml += '</ul></details>';
        html += srcHtml;
    }

    msg.innerHTML = html;
    chatMessages.appendChild(msg);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return msg;
}

function appendTyping() {
    const el = document.createElement("div");
    el.className = "msg bot";
    el.innerHTML = `<div class="typing-indicator">
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
    </div>`;
    chatMessages.appendChild(el);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return el;
}

// =========================================================
// Utilities
// =========================================================

function escapeHtml(str) {
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/\n/g, "<br>");
}

function escapeAttr(str) {
    return str.replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

// Initial status check
(async () => {
    try {
        const res = await fetch("/api/status");
        const data = await res.json();
        updateStats(data.index);
        if (data.scraping && data.scraping.active) {
            scrapeBtn.disabled = true;
            btnText.hidden = true;
            btnLoader.hidden = false;
            progressArea.hidden = false;
            startPolling();
        }
    } catch { /* server not ready yet */ }
})();
