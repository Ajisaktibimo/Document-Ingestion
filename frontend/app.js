(() => {
    "use strict";

    // Must cover backend OLLAMA_REQUEST_TIMEOUT_SECONDS + retrieval +
    // pipeline overhead. CPU-bound small models with NUM_PREDICT=2048 can
    // legitimately take 2-3 minutes; 10 min is a generous bound that still
    // fails fast on a truly hung backend.
    const CHAT_TIMEOUT_MS = 600000;
    const ALLOWED_EXTENSIONS = new Set(["pdf", "txt", "md", "jpg", "jpeg", "png"]);
    const SESSIONS_API = "/api/v1/sessions";

    function readUserId() {
        const match = document.cookie
            .split("; ")
            .find((row) => row.startsWith("docai_uid="));
        return match ? match.split("=")[1] : null;
    }

    function userInitials(uid) {
        if (!uid) return "??";
        return uid.replace(/-/g, "").slice(0, 2).toUpperCase();
    }

    const sessionState = {
        sessions: [],          // list from GET /sessions
        activeSessionId: null, // currently-open session UUID
    };

    async function fetchSessions() {
        const r = await fetch(SESSIONS_API);
        if (!r.ok) {
            console.warn("Failed to fetch sessions:", r.status);
            return [];
        }
        return await r.json();
    }

    async function createSession(initialDocIds = []) {
        const r = await fetch(SESSIONS_API, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ doc_ids: initialDocIds }),
        });
        if (!r.ok) {
            console.warn("Failed to create session:", r.status);
            return null;
        }
        return await r.json();
    }

    function escapeHtml(s) {
        return String(s)
            .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;").replaceAll("\"", "&quot;").replaceAll("'", "&#39;");
    }

    async function renameSession(sessionId, newTitle) {
        const trimmed = newTitle.trim();
        if (!trimmed) return null;
        try {
            const r = await fetch(`${SESSIONS_API}/${sessionId}`, {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ title: trimmed }),
            });
            if (!r.ok) {
                console.warn("Failed to rename session:", r.status);
                return null;
            }
            return trimmed;
        } catch (e) {
            console.warn("Failed to rename session:", e);
            return null;
        }
    }

    function activateRename(titleSpan, session, ui) {
        const input = document.createElement("input");
        input.type = "text";
        input.className = "sessions-sidebar__rename-input";
        input.value = session.title;

        titleSpan.replaceWith(input);
        input.select();

        let committed = false;

        async function commit() {
            if (committed) return;
            committed = true;
            const saved = await renameSession(session.session_id, input.value);
            if (saved) {
                const s = sessionState.sessions.find(x => x.session_id === session.session_id);
                if (s) s.title = saved;
            }
            renderSessionsSidebar(ui);
        }

        input.addEventListener("blur", commit);
        input.addEventListener("keydown", (e) => {
            if (e.key === "Enter") { e.preventDefault(); input.blur(); }
            if (e.key === "Escape") {
                committed = true;
                renderSessionsSidebar(ui);
            }
        });
    }

    function renderSessionsSidebar(ui) {
        const list = document.querySelector('[data-role="sessions-list"]');
        const empty = document.querySelector('[data-role="sessions-empty"]');
        if (!list) return;

        list.innerHTML = "";
        if (sessionState.sessions.length === 0) {
            if (empty) empty.hidden = false;
            return;
        }
        if (empty) empty.hidden = true;

        for (const s of sessionState.sessions) {
            const li = document.createElement("li");
            li.className = "sessions-sidebar__row";
            if (s.session_id === sessionState.activeSessionId) {
                li.classList.add("sessions-sidebar__row--active");
            }
            li.dataset.sessionId = s.session_id;

            const titleSpan = document.createElement("span");
            titleSpan.className = "sessions-sidebar__row-title";
            titleSpan.title = s.title;
            titleSpan.textContent = s.title;
            titleSpan.addEventListener("dblclick", (e) => {
                e.stopPropagation();
                activateRename(titleSpan, s, ui);
            });

            const deleteBtn = document.createElement("button");
            deleteBtn.type = "button";
            deleteBtn.className = "sessions-sidebar__row-delete";
            deleteBtn.dataset.action = "delete-session";
            deleteBtn.title = "Delete";
            deleteBtn.textContent = "×";

            li.append(titleSpan, deleteBtn);
            list.appendChild(li);
        }
    }

    async function loadSessionsOnStartup(ui) {
        sessionState.sessions = await fetchSessions();
        renderSessionsSidebar(ui);
    }

    async function syncActiveSessionDocIds() {
        if (!sessionState.activeSessionId) return;
        const docIds = Array.from(state.activeDocIds);
        try {
            await fetch(`${SESSIONS_API}/${sessionState.activeSessionId}`, {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ doc_ids: docIds }),
            });
        } catch (e) {
            console.warn("Failed to sync session doc_ids:", e);
        }
    }

    const state = {
        sessionId: null,
        documents: [],
        activeDocIds: new Set(),
        requestInFlight: false,
        lastRequestAt: null,
    };

    document.addEventListener("DOMContentLoaded", () => {
        const ui = getUi();
        if (!ui) return;

        bindUpload(ui);
        bindChat(ui);
        bindScopeControls(ui);
        bindSessionsSidebar(ui);
        renderCitations(ui, []);
        updateRuntimeDetails(ui);
        updateSendState(ui);
        checkApiHealth(ui);
        loadSessionsOnStartup(ui);
        const uid = readUserId();
        if (uid && ui.sessionIdLabel) {
            ui.sessionIdLabel.textContent = shortId(uid);
        }
    });

    function getUi() {
        const ids = [
            "activeDocInfo",
            "apiStatusDot",
            "apiStatusText",
            "chatEmptyState",
            "chatForm",
            "chatHistory",
            "chatInput",
            "chatStateLabel",
            "citationCount",
            "citationsContainer",
            "clearChat",
            "clearSelection",
            "contextMode",
            "documentList",
            "fileInput",
            "lastRequestDetail",
            "scopeDetail",
            "selectedDocsCount",
            "selectedIdsDetail",
            "sendButton",
            "sessionIdLabel",
            "uploadStatus",
            "uploadStatusText",
            "uploadZone",
        ];
        const elements = {};
        const missing = [];

        ids.forEach((id) => {
            const element = document.getElementById(id);
            if (!element) {
                missing.push(id);
            } else {
                elements[id] = element;
            }
        });

        if (missing.length > 0) {
            document.body.textContent = `UI failed to start. Missing elements: ${missing.join(", ")}`;
            return null;
        }

        return elements;
    }

    function bindUpload(ui) {
        ui.uploadZone.addEventListener("keydown", (event) => {
            if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                ui.fileInput.click();
            }
        });

        ["dragenter", "dragover"].forEach((eventName) => {
            ui.uploadZone.addEventListener(eventName, (event) => {
                event.preventDefault();
                ui.uploadZone.classList.add("dragover");
            });
        });

        ["dragleave", "drop"].forEach((eventName) => {
            ui.uploadZone.addEventListener(eventName, () => {
                ui.uploadZone.classList.remove("dragover");
            });
        });

        ui.uploadZone.addEventListener("drop", (event) => {
            event.preventDefault();
            const file = event.dataTransfer.files[0];
            if (file) {
                handleFileUpload(ui, file);
            }
        });

        ui.fileInput.addEventListener("change", (event) => {
            const file = event.target.files[0];
            if (file) {
                handleFileUpload(ui, file);
            }
            event.target.value = "";
        });
    }

    function bindChat(ui) {
        ui.clearChat.addEventListener("click", () => {
            ui.chatHistory.replaceChildren();
            showEmptyState(ui);
            state.sessionId = null;
            state.lastRequestAt = null;
            renderCitations(ui, []);
            addMessage(ui, "assistant", "Chat history cleared. Ask another question when you are ready.");
            updateRuntimeDetails(ui);
        });

        ui.chatInput.addEventListener("input", () => {
            resizeTextarea(ui.chatInput);
            updateSendState(ui);
        });

        ui.chatInput.addEventListener("keydown", (event) => {
            if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                ui.chatForm.requestSubmit();
            }
        });

        ui.chatForm.addEventListener("submit", async (event) => {
            event.preventDefault();
            await submitChat(ui);
        });
    }

    function bindScopeControls(ui) {
        ui.clearSelection.addEventListener("click", () => {
            state.activeDocIds.clear();
            syncActiveSessionDocIds();
            renderDocuments(ui);
            updateRuntimeDetails(ui);
        });
    }

    function bindSessionsSidebar(ui) {
        const newChatBtn = document.querySelector('[data-action="new-chat"]');
        if (newChatBtn) {
            newChatBtn.addEventListener("click", async () => {
                const created = await createSession([]);
                if (!created) return;
                sessionState.sessions = await fetchSessions();
                sessionState.activeSessionId = created.session_id;
                renderSessionsSidebar(ui);
                ui.chatHistory.replaceChildren();
                showEmptyState(ui);
                // Clear scope for new chat.
                state.activeDocIds.clear();
                renderDocuments(ui);
                updateRuntimeDetails(ui);
            });
        }

        const list = document.querySelector('[data-role="sessions-list"]');
        if (list) {
            list.addEventListener("click", async (event) => {
                const row = event.target.closest(".sessions-sidebar__row");
                if (!row) return;
                const sid = row.dataset.sessionId;

                // Delete clicked?
                if (event.target.dataset.action === "delete-session") {
                    event.stopPropagation();
                    const r = await fetch(`${SESSIONS_API}/${sid}`, { method: "DELETE" });
                    if (r.status === 204) {
                        if (sessionState.activeSessionId === sid) {
                            sessionState.activeSessionId = null;
                            ui.chatHistory.replaceChildren();
                        }
                        sessionState.sessions = await fetchSessions();
                        renderSessionsSidebar(ui);
                    }
                    return;
                }

                // Otherwise — load this session.
                const detail = await fetch(`${SESSIONS_API}/${sid}`);
                if (!detail.ok) return;
                const data = await detail.json();

                sessionState.activeSessionId = sid;
                ui.chatHistory.replaceChildren();
                for (const m of data.messages || []) {
                    addMessage(ui, m.role, m.content || "", {
                        thinking: m.thinking,
                        instant: true,
                    });
                }
                // Restore document scope for this session.
                // NOTE: Do NOT call syncActiveSessionDocIds() here — we are
                // restoring server state, not changing it.
                state.activeDocIds.clear();
                for (const d of data.doc_ids || []) {
                    state.activeDocIds.add(d);
                }
                renderDocuments(ui);
                renderSessionsSidebar(ui);
                updateRuntimeDetails(ui);
            });
        }
    }

    async function handleFileUpload(ui, file) {
        const extension = file.name.split(".").pop().toLowerCase();
        if (!ALLOWED_EXTENSIONS.has(extension)) {
            setUploadState(ui, "error", "Unsupported file type. Use PDF, text, markdown, JPG, or PNG.");
            return;
        }

        setUploadState(ui, "busy", `Ingesting ${file.name}...`);

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("/api/v1/upload", {
                method: "POST",
                body: formData,
            });
            const data = await readResponseJson(response);

            if (!response.ok) {
                throw new Error(data.detail || "Upload failed.");
            }

            addDocument(file.name, data.doc_id);
            state.activeDocIds.add(data.doc_id);
            syncActiveSessionDocIds();
            // 202 = another in-flight ingestion of the same file_hash; the
            // backend short-circuited rather than re-running. Tell the user
            // to wait instead of falsely claiming "Ingested".
            if (response.status === 202) {
                setUploadState(ui, "busy", data.message || `Indexing ${file.name} in progress — please wait.`);
            } else {
                setUploadState(ui, "success", `Ingested ${file.name}.`);
            }
            renderDocuments(ui);
            updateRuntimeDetails(ui);
        } catch (error) {
            setUploadState(ui, "error", error.message || "Upload failed.");
        }
    }

    async function submitChat(ui) {
        const query = ui.chatInput.value.trim();
        if (!query || state.requestInFlight) return;

        addMessage(ui, "user", query);
        ui.chatInput.value = "";
        resizeTextarea(ui.chatInput);
        setChatBusy(ui, true);
        state.lastRequestAt = new Date();
        updateRuntimeDetails(ui);

        const typing = addTypingIndicator(ui);
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), CHAT_TIMEOUT_MS);

        try {
            const payload = {
                query,
                session_id: state.sessionId,
            };
            const selectedIds = Array.from(state.activeDocIds);
            if (selectedIds.length > 0) {
                payload.doc_ids = selectedIds;
            }

            const chatUrl = sessionState.activeSessionId
                ? `${SESSIONS_API}/${sessionState.activeSessionId}/chat`
                : "/api/v1/chat";
            const chatPayload = sessionState.activeSessionId
                ? { query }    // session-aware endpoint reads doc_ids/history from server
                : payload;     // stateless endpoint keeps its current shape
            const response = await fetch(chatUrl, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                signal: controller.signal,
                body: JSON.stringify(chatPayload),
            });
            const data = await readResponseJson(response);

            if (!response.ok) {
                throw new Error(data.detail || "Chat request failed.");
            }

            state.sessionId = data.session_id || state.sessionId;
            typing.remove();

            const hasResponse = data.response && data.response.trim();
            const hasThinking = data.thinking && data.thinking.trim();

            if (!hasResponse && hasThinking) {
                addMessage(ui, "assistant", "Model produced no answer; reasoning shown below.", {
                    thinking: data.thinking,
                    emptyResponse: true,
                });
            } else if (!hasResponse && !hasThinking) {
                addMessage(ui, "assistant", "No response from the model. Please retry.", {
                    emptyResponse: true,
                });
            } else {
                addMessage(ui, "assistant", data.response, {
                    thinking: data.thinking,
                });
            }
            renderCitations(ui, data.citations || []);
            if (sessionState.activeSessionId) {
                sessionState.sessions = await fetchSessions();
                renderSessionsSidebar(ui);
            }
        } catch (error) {
            typing.remove();
            const message = error.name === "AbortError"
                ? "Request timed out. Ollama or retrieval may still be loading."
                : error.message || "Server error.";
            addMessage(ui, "assistant", `Error: ${message}`);
        } finally {
            clearTimeout(timeoutId);
            setChatBusy(ui, false);
            updateRuntimeDetails(ui);
        }
    }

    function addDocument(filename, docId) {
        if (!docId) return;
        const existing = state.documents.find((doc) => doc.id === docId);
        if (existing) {
            existing.name = filename;
            return;
        }
        state.documents.unshift({
            id: docId,
            name: filename,
            addedAt: new Date(),
        });
    }

    function renderDocuments(ui) {
        ui.documentList.replaceChildren();

        if (state.documents.length === 0) {
            ui.documentList.appendChild(emptyState(
                "No documents in this UI session",
                "Upload a file to pin it here. Existing indexed data is still searchable."
            ));
            return;
        }

        state.documents.forEach((doc) => {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "document-item";
            button.classList.toggle("active", state.activeDocIds.has(doc.id));
            button.addEventListener("click", () => {
                if (state.activeDocIds.has(doc.id)) {
                    state.activeDocIds.delete(doc.id);
                } else {
                    state.activeDocIds.add(doc.id);
                }
                syncActiveSessionDocIds();
                renderDocuments(ui);
                updateRuntimeDetails(ui);
            });

            const icon = document.createElement("span");
            icon.className = "doc-icon";
            icon.textContent = "DOC";

            const info = document.createElement("span");
            info.className = "doc-info";

            const name = document.createElement("strong");
            name.textContent = doc.name;

            const meta = document.createElement("span");
            meta.textContent = `${shortId(doc.id)} | ${formatTime(doc.addedAt)}`;

            info.append(name, meta);

            const stateLabel = document.createElement("span");
            stateLabel.className = "doc-state";
            stateLabel.textContent = state.activeDocIds.has(doc.id) ? "Selected" : "Available";

            button.append(icon, info, stateLabel);
            ui.documentList.appendChild(button);
        });
    }

    function hideEmptyState(ui) {
        if (ui.chatEmptyState && !ui.chatEmptyState.hidden) {
            ui.chatEmptyState.hidden = true;
        }
    }

    function showEmptyState(ui) {
        if (ui.chatEmptyState) {
            ui.chatHistory.appendChild(ui.chatEmptyState);
            ui.chatEmptyState.hidden = false;
        }
    }

    function addMessage(ui, sender, text, options = {}) {
        hideEmptyState(ui);
        const row = document.createElement("article");
        row.className = `message ${sender}`;
        if (options.instant) {
            row.style.animation = "none";
        }
        if (options.emptyResponse === true) {
            row.classList.add("message-empty-response");
        }

        const avatar = document.createElement("div");
        avatar.className = "avatar";
        avatar.textContent = sender === "user" ? userInitials(readUserId()) : "AI";

        const content = document.createElement("div");
        content.className = "message-content";

        if (options.thinking) {
            content.appendChild(renderThinking(options.thinking));
        }

        content.appendChild(renderRichText(text));
        row.append(avatar, content);
        ui.chatHistory.appendChild(row);
        scrollToBottom(ui);
    }

    function renderThinking(text) {
        const details = document.createElement("details");
        details.className = "thinking-block";

        const summary = document.createElement("summary");
        summary.textContent = "Reasoning";

        const body = renderRichText(text);
        body.classList.add("thinking-content");

        details.append(summary, body);
        return details;
    }

    function renderRichText(text) {
        const wrapper = document.createElement("div");
        wrapper.className = "text-content";

        const lines = String(text || "").split(/\r?\n/);
        let paragraph = [];
        let list = null;

        const flushParagraph = () => {
            if (paragraph.length === 0) return;
            const p = document.createElement("p");
            appendInlineContent(p, paragraph.join(" "));
            wrapper.appendChild(p);
            paragraph = [];
        };

        const flushList = () => {
            if (!list) return;
            wrapper.appendChild(list);
            list = null;
        };

        lines.forEach((line) => {
            const trimmed = line.trim();
            if (!trimmed) {
                flushParagraph();
                flushList();
                return;
            }

            const heading = /^(#{1,3})\s+(.+)$/.exec(trimmed);
            if (heading) {
                flushParagraph();
                flushList();
                const level = Math.min(heading[1].length + 2, 4);
                const h = document.createElement(`h${level}`);
                appendInlineContent(h, heading[2]);
                wrapper.appendChild(h);
                return;
            }

            const bullet = /^[-*]\s+(.+)$/.exec(trimmed);
            const ordered = /^\d+\.\s+(.+)$/.exec(trimmed);
            if (bullet || ordered) {
                flushParagraph();
                const tagName = ordered ? "OL" : "UL";
                if (!list || list.tagName !== tagName) {
                    flushList();
                    list = document.createElement(ordered ? "ol" : "ul");
                }
                const li = document.createElement("li");
                appendInlineContent(li, (bullet || ordered)[1]);
                list.appendChild(li);
                return;
            }

            paragraph.push(trimmed);
        });

        flushParagraph();
        flushList();

        return wrapper;
    }

    function appendInlineContent(parent, text) {
        const pattern = /(`([^`]+)`|\*\*([^*]+)\*\*|\[([A-Za-z0-9]{4})\])/g;
        let lastIndex = 0;
        let match;

        while ((match = pattern.exec(text)) !== null) {
            if (match.index > lastIndex) {
                parent.appendChild(document.createTextNode(text.slice(lastIndex, match.index)));
            }

            if (match[2]) {
                const code = document.createElement("code");
                code.textContent = match[2];
                parent.appendChild(code);
            } else if (match[3]) {
                const strong = document.createElement("strong");
                strong.textContent = match[3];
                parent.appendChild(strong);
            } else if (match[4]) {
                const button = document.createElement("button");
                button.type = "button";
                button.className = "inline-citation";
                button.textContent = match[4];
                button.addEventListener("click", () => highlightCitation(match[4]));
                parent.appendChild(button);
            }

            lastIndex = pattern.lastIndex;
        }

        if (lastIndex < text.length) {
            parent.appendChild(document.createTextNode(text.slice(lastIndex)));
        }
    }

    function renderCitations(ui, citations) {
        ui.citationsContainer.replaceChildren();
        ui.citationCount.textContent = String(citations.length);

        if (citations.length === 0) {
            ui.citationsContainer.appendChild(emptyState(
                "No citations yet",
                "Ask a grounded question to populate the evidence list."
            ));
            return;
        }

        citations.forEach((citation, index) => {
            const card = document.createElement("article");
            card.className = "citation-card";
            card.id = citationDomId(citation.id);

            const top = document.createElement("div");
            top.className = "citation-topline";

            const badge = document.createElement("button");
            badge.type = "button";
            badge.className = "citation-id";
            badge.textContent = citation.id || `c${index + 1}`;
            badge.addEventListener("click", () => highlightCitation(citation.id));

            const score = document.createElement("span");
            score.className = "citation-score";
            score.textContent = formatScore(citation.score);

            top.append(badge, score);

            const source = document.createElement("strong");
            source.textContent = citation.heading_path || citation.heading || "Retrieved chunk";

            const meta = document.createElement("p");
            meta.className = "citation-meta";
            meta.textContent = `Doc ${shortId(citation.doc_id)} | Page ${citation.page_num || "N/A"} | Rank ${index + 1}`;

            card.append(top, source, meta);
            ui.citationsContainer.appendChild(card);
        });
    }

    function addTypingIndicator(ui) {
        const row = document.createElement("article");
        row.className = "message assistant typing-message";

        const avatar = document.createElement("div");
        avatar.className = "avatar";
        avatar.textContent = "AI";

        const content = document.createElement("div");
        content.className = "message-content";

        const dots = document.createElement("div");
        dots.className = "typing-indicator";
        for (let i = 0; i < 3; i += 1) {
            dots.appendChild(document.createElement("span"));
        }

        content.appendChild(dots);
        row.append(avatar, content);
        ui.chatHistory.appendChild(row);
        scrollToBottom(ui);
        return row;
    }

    async function checkApiHealth(ui) {
        setApiStatus(ui, "checking", "Checking");
        try {
            const response = await fetch("/health", { cache: "no-store" });
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            setApiStatus(ui, "online", "Online");
        } catch (error) {
            setApiStatus(ui, "offline", "Offline");
        }
    }

    async function readResponseJson(response) {
        const text = await response.text();
        if (!text) return {};
        try {
            return JSON.parse(text);
        } catch (error) {
            return { detail: text };
        }
    }

    function setUploadState(ui, stateName, message) {
        ui.uploadStatus.className = `upload-status ${stateName}`;
        ui.uploadStatusText.textContent = message;
    }

    function setChatBusy(ui, busy) {
        state.requestInFlight = busy;
        ui.chatStateLabel.textContent = busy ? "Waiting for response" : "Idle";
        updateSendState(ui);
    }

    function updateSendState(ui) {
        const hasText = ui.chatInput.value.trim().length > 0;
        ui.sendButton.disabled = !hasText || state.requestInFlight;
    }

    function updateRuntimeDetails(ui) {
        const selectedDocs = state.documents.filter((doc) => state.activeDocIds.has(doc.id));
        const selectedCount = selectedDocs.length;

        ui.selectedDocsCount.textContent = `${selectedCount} selected`;
        ui.contextMode.textContent = selectedCount === 0
            ? "All documents"
            : `${selectedCount} selected`;
        ui.scopeDetail.textContent = selectedCount === 0
            ? "All indexed documents"
            : selectedDocs.map((doc) => doc.name).join(", ");
        ui.activeDocInfo.textContent = selectedCount === 0
            ? "Searching across all indexed documents."
            : `Scoped to ${selectedCount} selected document${selectedCount === 1 ? "" : "s"}.`;
        ui.selectedIdsDetail.textContent = selectedCount === 0
            ? "None"
            : Array.from(state.activeDocIds).map(shortId).join(", ");
        ui.sessionIdLabel.textContent = state.sessionId ? shortId(state.sessionId) : "New";
        ui.lastRequestDetail.textContent = state.lastRequestAt ? formatTime(state.lastRequestAt) : "None yet";
    }

    function setApiStatus(ui, stateName, label) {
        ui.apiStatusText.textContent = label;
        ui.apiStatusDot.className = `status-dot ${stateName}`;
    }

    function resizeTextarea(textarea) {
        textarea.style.height = "auto";
        textarea.style.height = `${Math.min(textarea.scrollHeight, 160)}px`;
    }

    function highlightCitation(id) {
        if (!id) return;
        const card = document.getElementById(citationDomId(id));
        if (!card) return;

        card.scrollIntoView({ behavior: "smooth", block: "center" });
        card.classList.add("highlighted");
        setTimeout(() => card.classList.remove("highlighted"), 1800);
    }

    function citationDomId(id) {
        return `cit-${String(id || "").replace(/[^a-zA-Z0-9_-]/g, "")}`;
    }

    function emptyState(title, copy) {
        const block = document.createElement("div");
        block.className = "empty-state compact";

        const strong = document.createElement("strong");
        strong.textContent = title;

        const span = document.createElement("span");
        span.textContent = copy;

        block.append(strong, span);
        return block;
    }

    function shortId(id) {
        return String(id || "unknown").slice(0, 8);
    }

    function formatTime(value) {
        return new Intl.DateTimeFormat(undefined, {
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
        }).format(value);
    }

    function formatScore(score) {
        const numericScore = Number(score);
        if (!Number.isFinite(numericScore)) {
            return "Score N/A";
        }
        if (numericScore >= 0 && numericScore <= 1) {
            return `${Math.round(numericScore * 100)}%`;
        }
        return numericScore.toFixed(3);
    }

    function scrollToBottom(ui) {
        ui.chatHistory.scrollTop = ui.chatHistory.scrollHeight;
    }

    window.highlightCitation = highlightCitation;
})();
