"""
Orchestrator Engine
===================
Integrated RAG engine with:
- Parallel retrieval: Hybrid Vector Search (Dense + Sparse + Rerank) + Graph Search (FalkorDB)
- Over-fetch + Cross-Encoder Reranking for precision
- Citation ID system for deterministic source grounding
- Contradiction resolution (Graph > Vector when they conflict)
"""
import asyncio
import dataclasses
import hashlib
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Tuple
import asyncpg
from falkordb import FalkorDB
from docai.providers import get_llm_client, get_dense_embedder, get_sparse_embedder, get_reranker
from docai.retrieval.qdrant_client import QdrantStore
from docai.orchestrator.base import ChatMessage
from docai.config import settings

logger = logging.getLogger(__name__)
SESSION_HISTORY_MAX_MESSAGES = 20


@dataclasses.dataclass(frozen=True, slots=True)
class DocStatus:
    """Lightweight per-document registry status used by pre-flight checks.

    Mirrors the columns selected from ``doc_registry`` by
    :meth:`OrchestratorEngine._fetch_doc_statuses`. Importable from this
    module so callers (e.g. Task T2's pre-flight gate) can type their
    return values without depending on asyncpg's ``Record`` type.
    """

    doc_id: uuid.UUID
    filename: str
    status: str
    error_message: str | None
    chunk_count: int | None = None


def _doc_status_is_ready(doc_status: DocStatus) -> bool:
    if doc_status.status != "completed":
        return False
    return doc_status.chunk_count is None or doc_status.chunk_count > 0


@dataclasses.dataclass(frozen=True, slots=True)
class SessionContext:
    """Identifies a persistent chat session for the engine.

    Passed to ``handle_query`` to switch from stateless mode (debug
    /chat endpoint) into persistent mode (loads history, persists turns,
    optionally summarizes). When None, the engine is fully stateless.
    """
    session_id: uuid.UUID
    user_id: str


def _generate_citation_id(doc_id: str, index: int) -> str:
    """Generate a short 4-char citation ID from doc_id and index."""
    raw = f"{doc_id}:{index}"
    digest = hashlib.md5(raw.encode()).hexdigest()
    return digest[:4]


class OrchestratorEngine:
    # Class-level default so instances created via __new__ (used by some
    # tests to bypass heavy provider initialization) still expose the
    # attribute. __init__ overwrites this on real instances.
    _schema_cache: "Tuple[float, Dict[str, Any]] | None" = None

    def __init__(self):
        logger.info("Initializing OrchestratorEngine (Loading Models)...")
        self.llm_client = get_llm_client(think=settings.effective_chat_ollama_think)
        logger.info("- LLM Client ready")
        self.dense_embedder = get_dense_embedder()
        logger.info("- Dense Embedder ready")
        self.sparse_embedder = get_sparse_embedder()
        logger.info("- Sparse Embedder ready")
        self.reranker = get_reranker()
        logger.info("- Reranker ready")
        self.qdrant_store = QdrantStore()
        logger.info("- Qdrant Store ready")
        self._session_doc_ids: Dict[str, List[str]] = {}
        self._session_messages: Dict[str, List[ChatMessage]] = {}
        # Cache for the live FalkorDB schema used by schema-aware Cypher
        # generation. Tuple of (monotonic_timestamp, schema_dict). Refreshed
        # per CYPHER_SCHEMA_CACHE_TTL_SECONDS in _get_cached_graph_schema.
        self._schema_cache: Tuple[float, Dict[str, Any]] | None = None
        logger.info("OrchestratorEngine fully initialized.")

    def _get_falkordb_connection(self):
        return FalkorDB(host=settings.FALKORDB_HOST, port=settings.FALKORDB_PORT)

    @staticmethod
    def _normalise_doc_ids(doc_ids: List[str] | None) -> List[str]:
        normalised = []
        for raw_doc_id in doc_ids or []:
            try:
                doc_id = str(uuid.UUID(str(raw_doc_id)))
            except (TypeError, ValueError):
                continue
            if doc_id not in normalised:
                normalised.append(doc_id)
        return normalised

    def _session_scope(self, session_id: str | None) -> List[str]:
        if not session_id:
            return []
        return list(getattr(self, "_session_doc_ids", {}).get(session_id, []))

    def _remember_session_scope(self, session_id: str | None, doc_ids: List[str]) -> None:
        if not session_id or not doc_ids:
            return
        if not hasattr(self, "_session_doc_ids"):
            self._session_doc_ids = {}
        self._session_doc_ids[session_id] = doc_ids

    def _session_history(self, session_id: str | None) -> List[ChatMessage]:
        if not session_id:
            return []
        return list(getattr(self, "_session_messages", {}).get(session_id, []))

    def _remember_session_message(self, session_id: str | None, role: str, content: str) -> None:
        if not session_id:
            return
        if not hasattr(self, "_session_messages"):
            self._session_messages = {}
        history = self._session_messages.setdefault(session_id, [])
        history.append(ChatMessage(role=role, content=content))
        del history[:-SESSION_HISTORY_MAX_MESSAGES]

    def _contextual_retrieval_query(self, session_id: str | None, query: str) -> str:
        history = self._session_history(session_id)[-SESSION_HISTORY_MAX_MESSAGES:]
        if not history:
            return query

        lines = [f"{message.role}: {message.content}" for message in history]
        lines.append(f"user: {query}")
        return "\n".join(lines)

    def _format_recent_history(self, session_id: str | None) -> str:
        history = self._session_history(session_id)[-SESSION_HISTORY_MAX_MESSAGES:]
        if not history:
            return "No previous messages in this chat session."

        return "\n".join(f"{message.role}: {message.content}" for message in history)

    @staticmethod
    def _truncate_for_prompt(text: str, max_chars: int) -> str:
        if max_chars <= 0 or len(text) <= max_chars:
            return text

        marker = "\n[truncated]"
        if max_chars <= len(marker):
            return text[:max_chars]

        return text[: max_chars - len(marker)].rstrip() + marker

    @staticmethod
    def _normalise_generated_cypher(cypher_query: str) -> str | None:
        query = cypher_query.strip("` \n").removeprefix("cypher\n").strip()
        if not query:
            return None

        first_line = query.splitlines()[0].strip().lower()
        if first_line.startswith(("#", "//", "--")):
            return None

        lowered = query.lower()
        first_word = lowered.split(None, 1)[0] if lowered.split() else ""
        read_prefixes = ("match", "optional", "with", "return", "unwind")
        write_tokens = (" create ", " merge ", " delete ", " detach ", " set ", " remove ", " drop ")
        padded = f" {lowered} "

        if first_word not in read_prefixes:
            return None
        if any(token in padded for token in write_tokens):
            return None

        return query

    # ── Vector Retrieval (Over-fetch + Rerank) ──────────────────────────

    async def _get_vector_context(
        self,
        query: str,
        doc_ids: List[str] | None = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Hybrid search with over-fetch + cross-encoder reranking.
        Returns formatted context string and raw citation metadata.
        """
        logger.info("Retrieving context using Hybrid Search (Dense + Sparse)...")
        query_dense = await self.dense_embedder.embed_query(query)
        query_sparse = await self.sparse_embedder.embed_query(query)

        # Over-fetch: retrieve VECTOR_PREFETCH_K candidates
        prefetch_k = settings.VECTOR_PREFETCH_K
        hits = await self.qdrant_store.search_hybrid(
            query_dense,
            query_sparse,
            limit=prefetch_k,
            doc_ids=doc_ids,
        )

        if not hits:
            return "No relevant documents found.", []

        # Rerank: cross-encoder scoring for precision
        logger.info(f"Reranking {len(hits)} candidates → top {settings.RERANKER_OUTPUT_K}...")
        documents = [hit['text'] for hit in hits]
        reranked_pairs = await self.reranker.rerank(query, documents, top_k=settings.RERANKER_OUTPUT_K)

        # Filter by min relevance score
        filtered_pairs = [
            (idx, score) for idx, score in reranked_pairs
            if score >= settings.MIN_RELEVANCE_SCORE
        ]

        # Fallback: if reranker filtered everything, keep top 3
        if not filtered_pairs and reranked_pairs:
            logger.warning(
                f"All {len(reranked_pairs)} candidates below threshold "
                f"{settings.MIN_RELEVANCE_SCORE}, falling back to top 3"
            )
            filtered_pairs = reranked_pairs[:3]

        # Build context with citation IDs
        context_blocks = []
        citations = []
        current_context_chars = 0
        total_context_limit = settings.RAG_CONTEXT_TOTAL_MAX_CHARS
        for rank, (idx, score) in enumerate(filtered_pairs):
            hit = hits[idx]
            citation_id = _generate_citation_id(hit['doc_id'], rank)

            # Extract enriched metadata from payload
            heading = hit.get('heading', '')
            heading_path = hit.get('heading_path', '')
            page_num = hit.get('page_num', 0)

            source_label = f"Doc {hit['doc_id'][:8]}"
            if heading_path:
                source_label += f" | {heading_path}"
            if page_num:
                source_label += f" | Page {page_num}"

            chunk_text = self._truncate_for_prompt(
                str(hit.get('text', '')),
                settings.RAG_CONTEXT_CHUNK_MAX_CHARS,
            )
            block = (
                f"[{citation_id}] Source: {source_label} (Relevance: {score:.3f})\n"
                f"Content: {chunk_text}"
            )

            separator_len = len("\n\n---\n\n") if context_blocks else 0
            next_len = current_context_chars + separator_len + len(block)
            if total_context_limit > 0 and next_len > total_context_limit:
                if context_blocks:
                    break
                block = self._truncate_for_prompt(block, total_context_limit)
                next_len = len(block)

            context_blocks.append(block)
            current_context_chars = next_len
            citations.append({
                "id": citation_id,
                "doc_id": hit['doc_id'],
                "heading": heading,
                "heading_path": heading_path,
                "page_num": page_num,
                "score": score,
            })

        context_str = "\n\n---\n\n".join(context_blocks)
        if filtered_pairs:
            logger.info(
                f"Reranked {len(hits)} → {len(citations)} chunks "
                f"(scores: {filtered_pairs[0][1]:.3f} → {filtered_pairs[-1][1]:.3f})"
            )
        else:
            logger.info("Reranked %d hits → 0 chunks (all below threshold)", len(hits))
        return context_str, citations

    # ── Graph Retrieval ─────────────────────────────────────────────────

    _STATIC_CYPHER_PROMPT = (
        "You are an expert Cypher query generator for FalkorDB. "
        "The graph contains `Entity` nodes with relations like `MENTIONED_IN`, "
        "`ASSOCIATED_WITH`, `PARTICIPATES_IN`, `SUPPORTED_BY`, `BELONGS_TO`, `FOUND_IN`. "
        "Generate a read-only Cypher query to extract relevant information based "
        "on the user's question. "
        "Return ONLY the raw Cypher query, without markdown backticks or explanations."
    )

    _SELF_HEAL_REASON = (
        "did not pass read-only validator (must start with MATCH/OPTIONAL/WITH/"
        "RETURN/UNWIND, no write tokens, no markdown wrappers)"
    )

    async def _fetch_graph_schema(self) -> Dict[str, Any]:
        """Introspect the live FalkorDB graph schema via direct Cypher.

        FalkorDB does not support apoc.meta.schema or db.schema.visualization,
        so we use three small aggregation queries. Returns
        ``{"node_labels": [str], "rel_types": [str], "props_by_label": {str: [str]}}``
        on success, or ``{}`` on any failure (FalkorDB unreachable, empty
        graph, query error). Never raises — the caller should fall back to
        the static prompt when the result is empty.
        """
        def _introspect():
            db = self._get_falkordb_connection()
            graph = db.select_graph("docai_knowledge_graph")

            label_rows = graph.query(
                "MATCH (n) RETURN DISTINCT labels(n) AS labels"
            ).result_set
            labels = sorted({lbl for row in label_rows for lbl in (row[0] or [])})

            rel_rows = graph.query(
                "MATCH ()-[r]->() RETURN DISTINCT type(r) AS rel_type"
            ).result_set
            rel_types = sorted({row[0] for row in rel_rows if row[0]})

            prop_rows = graph.query(
                "MATCH (n) WITH labels(n)[0] AS label, keys(n) AS props "
                "UNWIND props AS prop "
                "RETURN label, collect(DISTINCT prop) AS keys"
            ).result_set
            props_by_label = {
                row[0]: sorted(row[1])
                for row in prop_rows
                if row[0] and row[1]
            }

            return {
                "node_labels": labels,
                "rel_types": rel_types,
                "props_by_label": props_by_label,
            }

        try:
            return await asyncio.to_thread(_introspect)
        except Exception as e:
            logger.warning("Graph schema introspection failed: %s", e)
            return {}

    async def _get_cached_graph_schema(self) -> Dict[str, Any]:
        """Return the schema cached on this engine instance, refreshing if
        the cache is older than ``CYPHER_SCHEMA_CACHE_TTL_SECONDS`` or absent.
        """
        now = time.monotonic()
        ttl = settings.CYPHER_SCHEMA_CACHE_TTL_SECONDS
        if self._schema_cache is not None:
            cached_at, cached = self._schema_cache
            if (now - cached_at) < ttl:
                return cached
        schema = await self._fetch_graph_schema()
        self._schema_cache = (now, schema)
        return schema

    @classmethod
    def _build_cypher_system_prompt(cls, schema: Dict[str, Any]) -> str:
        """Build the Cypher-generation system prompt.

        With a non-empty schema and ``ENABLE_SCHEMA_AWARE_CYPHER`` on, emit
        a structured prompt: schema block + schema-anchored few-shot
        examples + explicit constraints. Otherwise return the historical
        static prompt unchanged (so disabling the flag fully reverts behavior).
        """
        if not settings.ENABLE_SCHEMA_AWARE_CYPHER:
            return cls._STATIC_CYPHER_PROMPT
        if not schema or not schema.get("node_labels"):
            return cls._STATIC_CYPHER_PROMPT

        labels_str = ", ".join(f"`{l}`" for l in schema["node_labels"])
        rels = schema.get("rel_types") or []
        rels_str = ", ".join(f"`{r}`" for r in rels) if rels else "(none)"

        prop_lines = [
            f"  {label}({', '.join(props)})"
            for label, props in sorted(schema.get("props_by_label", {}).items())
            if props
        ]
        props_block = "\n".join(prop_lines) if prop_lines else "  (no property data)"

        # Few-shot examples anchored to the user's actual schema. Prefer
        # `Entity` / `MENTIONED_IN` (the project's ontology) when present;
        # otherwise fall back to whatever's in the graph.
        primary_label = "Entity" if "Entity" in schema["node_labels"] else schema["node_labels"][0]
        primary_rel = "MENTIONED_IN" if "MENTIONED_IN" in rels else (rels[0] if rels else None)

        example_lines = [
            "Examples:",
            f"Q: \"What does the document say about X?\"",
            (
                f"A: MATCH (e:{primary_label})-[:{primary_rel}]->(d) "
                f"WHERE toLower(e.name) CONTAINS toLower('X') RETURN e.name, d LIMIT 10"
            ) if primary_rel else (
                f"A: MATCH (e:{primary_label}) "
                f"WHERE toLower(e.name) CONTAINS toLower('X') RETURN e LIMIT 10"
            ),
            "",
            f"Q: \"How is A related to B?\"",
            (
                f"A: MATCH (a:{primary_label})-[r]->(b:{primary_label}) "
                f"WHERE toLower(a.name) CONTAINS toLower('A') "
                f"AND toLower(b.name) CONTAINS toLower('B') "
                f"RETURN type(r), a.name, b.name LIMIT 10"
            ),
            "",
            f"Q: \"List entities by type\"",
            f"A: MATCH (e:{primary_label}) WHERE e.type = 'Person' RETURN e.name LIMIT 25",
        ]
        examples = "\n".join(example_lines)

        return (
            "You are an expert Cypher query generator for FalkorDB.\n"
            "\n"
            "Live graph schema:\n"
            f"  Node labels: {labels_str}\n"
            f"  Relationship types: {rels_str}\n"
            "  Properties per label:\n"
            f"{props_block}\n"
            "\n"
            f"{examples}\n"
            "\n"
            "Constraints:\n"
            "- Generate a READ-ONLY Cypher query (must start with MATCH, OPTIONAL, "
            "WITH, RETURN, or UNWIND).\n"
            "- No write operations: CREATE, MERGE, DELETE, SET, REMOVE, DROP, "
            "DETACH are forbidden.\n"
            "- Use only the labels, relationships, and properties listed above.\n"
            "- Return ONLY the raw Cypher query — no markdown fences, no commentary."
        )

    async def _generate_cypher_with_retry(self, user_query: str) -> str | None:
        """Generate a validated read-only Cypher query, optionally
        self-healing on validation failure.

        Returns the validated query string or ``None`` if every attempt
        failed validation or the LLM call raised. Self-healing is gated
        by ``settings.ENABLE_CYPHER_SELF_HEALING``: when off, exactly one
        attempt is made; when on, up to ``CYPHER_SELF_HEAL_MAX_ATTEMPTS``,
        each retry receiving the prior failure context.
        """
        schema = await self._get_cached_graph_schema()
        base_prompt = self._build_cypher_system_prompt(schema)

        self_healing = settings.ENABLE_CYPHER_SELF_HEALING
        max_attempts = settings.CYPHER_SELF_HEAL_MAX_ATTEMPTS if self_healing else 1
        max_attempts = max(1, max_attempts)

        last_invalid: str | None = None

        for attempt in range(1, max_attempts + 1):
            if attempt == 1 or last_invalid is None:
                system_prompt = base_prompt
            else:
                system_prompt = (
                    f"{base_prompt}\n\n"
                    f"Previous attempt was invalid: {self._SELF_HEAL_REASON}\n"
                    f"Previous query: {last_invalid}\n"
                    "Fix the issues and emit only the corrected raw Cypher query."
                )

            messages = [ChatMessage(role="user", content=user_query)]
            try:
                llm_res = await self.llm_client.generate_response(
                    messages=messages,
                    system_prompt=system_prompt,
                    temperature=0.0,
                    stream=False,
                )
            except Exception as e:
                logger.error("Cypher LLM call failed on attempt %d: %s", attempt, e)
                return None

            raw = llm_res["content"] if isinstance(llm_res, dict) else llm_res
            normalised = self._normalise_generated_cypher(raw or "")
            if normalised:
                if attempt > 1:
                    logger.info("Cypher self-heal succeeded on attempt %d.", attempt)
                return normalised

            last_invalid = (raw or "").strip()[:500]
            logger.warning(
                "Cypher attempt %d invalid: %s", attempt, self._SELF_HEAL_REASON
            )

        return None

    async def _get_graph_context(self, query: str) -> str:
        """Generate and execute a Cypher query against FalkorDB."""
        logger.info("Generating Cypher query for Graph Search...")

        cypher_query = await self._generate_cypher_with_retry(query)
        if not cypher_query:
            logger.warning("Graph search skipped invalid generated Cypher.")
            return "Graph search skipped: generated query was not valid read-only Cypher."
        logger.info(f"Generated Cypher Query: {cypher_query}")

        try:
            def execute_cypher():
                db = self._get_falkordb_connection()
                graph = db.select_graph("docai_knowledge_graph")
                return graph.query(cypher_query).result_set

            result_set = await asyncio.to_thread(execute_cypher)

            if not result_set:
                return "No relevant graph relationships found."

            graph_blocks = [str(row) for row in result_set]
            return "\n".join(graph_blocks)

        except Exception as e:
            logger.error(f"Graph Search failed: {e}")
            return "Graph search unavailable or failed."

    # ── Unified Context ─────────────────────────────────────────────────

    async def gather_unified_context(
        self,
        query: str,
        doc_ids: List[str] | None = None,
        graph_query: str | None = None,
    ) -> Dict[str, Any]:
        """Runs vector and graph retrievals in parallel and returns both contexts."""
        scoped_doc_ids = self._normalise_doc_ids(doc_ids)
        vector_task = asyncio.create_task(self._get_vector_context(query, doc_ids=scoped_doc_ids))
        graph_task = (
            asyncio.create_task(self._get_graph_context_with_timeout(graph_query or query))
            if settings.ENABLE_GRAPH_SEARCH
            else None
        )

        if graph_task:
            (vector_ctx, citations), graph_ctx = await asyncio.gather(vector_task, graph_task)
        else:
            vector_ctx, citations = await vector_task
            graph_ctx = "Graph search disabled."

        document_inventory = await self._get_document_inventory(
            citations,
            scoped_doc_ids=scoped_doc_ids,
        )

        return {
            "vector": vector_ctx,
            "graph": graph_ctx,
            "citations": citations,
            "document_inventory": document_inventory,
        }

    async def _get_document_inventory(
        self,
        citations: List[Dict[str, Any]],
        scoped_doc_ids: List[str] | None = None,
    ) -> str:
        """Return deterministic document metadata for the documents used as context."""
        doc_ids = [uuid.UUID(doc_id) for doc_id in self._normalise_doc_ids(scoped_doc_ids)]
        if not doc_ids:
            for citation in citations:
                raw_doc_id = citation.get("doc_id")
                if not raw_doc_id:
                    continue
                try:
                    doc_id = uuid.UUID(str(raw_doc_id))
                except ValueError:
                    continue
                if doc_id not in doc_ids:
                    doc_ids.append(doc_id)

        dsn = settings.POSTGRES_DSN.replace("postgresql+asyncpg://", "postgresql://")
        conn = None
        try:
            conn = await asyncpg.connect(dsn)
            if doc_ids:
                rows = await conn.fetch(
                    '''
                    SELECT
                        d.doc_id,
                        d.filename,
                        d.status,
                        d.page_count,
                        COUNT(p.parent_id) AS chunk_count,
                        MIN(p.page_num) AS first_indexed_page,
                        MAX(p.page_num) AS last_indexed_page
                    FROM doc_registry d
                    LEFT JOIN parent_chunks p ON p.doc_id = d.doc_id
                    WHERE d.doc_id = ANY($1::uuid[])
                    GROUP BY d.doc_id, d.filename, d.status, d.page_count
                    ORDER BY d.filename
                    ''',
                    doc_ids,
                )
            else:
                rows = await conn.fetch(
                    '''
                    SELECT
                        d.doc_id,
                        d.filename,
                        d.status,
                        d.page_count,
                        COUNT(p.parent_id) AS chunk_count,
                        MIN(p.page_num) AS first_indexed_page,
                        MAX(p.page_num) AS last_indexed_page
                    FROM doc_registry d
                    LEFT JOIN parent_chunks p ON p.doc_id = d.doc_id
                    WHERE d.status = 'completed'
                    GROUP BY d.doc_id, d.filename, d.status, d.page_count
                    ORDER BY d.created_at DESC
                    LIMIT 10
                    '''
                )
        except Exception as e:
            logger.warning("Document inventory lookup failed: %s", e)
            return "Document inventory unavailable."
        finally:
            if conn:
                await conn.close()

        if not rows:
            return "No completed documents are registered."

        lines = []
        for row in rows:
            page_count = row["page_count"]
            page_count_text = str(page_count) if page_count else "unknown"
            first_page = row["first_indexed_page"]
            last_page = row["last_indexed_page"]
            if first_page and last_page:
                indexed_pages = f"{first_page}-{last_page}" if first_page != last_page else str(first_page)
            else:
                indexed_pages = "none"

            filename_label = row["filename"]
            if row["status"] != "completed":
                filename_label = f"{filename_label} [NOT YET INDEXED]"

            lines.append(
                "- Doc {doc}: filename={filename}; status={status}; "
                "page_count={page_count}; indexed_chunks={chunks}; "
                "indexed_chunk_start_pages={indexed_pages}".format(
                    doc=str(row["doc_id"])[:8],
                    filename=filename_label,
                    status=row["status"],
                    page_count=page_count_text,
                    chunks=row["chunk_count"],
                    indexed_pages=indexed_pages,
                )
            )

        return "\n".join(lines)

    async def _fetch_doc_statuses(self, doc_ids: List[uuid.UUID]) -> List[DocStatus]:
        """Fetch per-document registry status for a list of doc_ids.

        Used by pre-flight checks (Task T2) to verify that requested
        documents exist and are in a usable state before running a query.
        Reuses the same asyncpg-connect pattern as
        :meth:`_get_document_inventory`. Returns an empty list for empty
        input without performing a DB roundtrip.
        """
        if not doc_ids:
            return []

        dsn = settings.POSTGRES_DSN.replace("postgresql+asyncpg://", "postgresql://")
        try:
            conn = await asyncpg.connect(dsn)
            try:
                rows = await conn.fetch(
                    '''
                    SELECT
                        d.doc_id,
                        d.filename,
                        d.status,
                        d.error_message,
                        COUNT(p.parent_id)::int AS chunk_count
                    FROM doc_registry d
                    LEFT JOIN parent_chunks p ON p.doc_id = d.doc_id
                    WHERE d.doc_id = ANY($1::uuid[])
                    GROUP BY d.doc_id, d.filename, d.status, d.error_message
                    ''',
                    doc_ids,
                )
            finally:
                await conn.close()
        except Exception as e:
            logger.warning("Failed to fetch doc statuses: %s", e)
            return []

        statuses = []
        for row in rows:
            try:
                chunk_count = row["chunk_count"]
            except (KeyError, IndexError):
                chunk_count = None
            statuses.append(
                DocStatus(
                    doc_id=row["doc_id"],
                    filename=row["filename"],
                    status=row["status"],
                    error_message=row["error_message"],
                    chunk_count=chunk_count,
                )
            )
        return statuses

    async def _fetch_session(
        self,
        session_id: uuid.UUID,
        user_id: str,
    ) -> Dict[str, Any] | None:
        """Load a session row, scoped to the given user. Returns None if
        the session doesn't exist OR belongs to another user."""
        dsn = settings.POSTGRES_DSN.replace("postgresql+asyncpg://", "postgresql://")
        conn = None
        try:
            conn = await asyncpg.connect(dsn)
            row = await conn.fetchrow(
                """
                SELECT session_id, user_id, title, doc_ids, summary,
                       summary_through_message_id, created_at, updated_at
                FROM sessions
                WHERE session_id = $1 AND user_id = $2
                """,
                session_id, user_id,
            )
            return dict(row) if row else None
        except Exception as e:
            logger.warning("_fetch_session failed: %s", e)
            return None
        finally:
            if conn is not None:
                await conn.close()

    async def _load_recent_messages(
        self,
        session_id: uuid.UUID,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Load the last `limit` messages of a session, oldest-first.

        Caller (the persistent path of handle_query) feeds these into
        the prompt as conversation context. Older messages live in
        sessions.summary if summarization is enabled.
        """
        dsn = settings.POSTGRES_DSN.replace("postgresql+asyncpg://", "postgresql://")
        conn = None
        try:
            conn = await asyncpg.connect(dsn)
            rows = await conn.fetch(
                """
                SELECT id, role, content, thinking, citations, created_at
                FROM session_messages
                WHERE session_id = $1
                ORDER BY id DESC
                LIMIT $2
                """,
                session_id, limit,
            )
            # Sort by id ASC for chronological order — works correctly
            # for both real DB (returned DESC) and test fakes.
            return sorted([dict(r) for r in rows], key=lambda r: r["id"])
        except Exception as e:
            logger.warning("_load_recent_messages failed: %s", e)
            return []
        finally:
            if conn is not None:
                await conn.close()

    async def _persist_user_msg(
        self,
        session_id: uuid.UUID,
        user_message: str,
    ) -> None:
        """Insert a user message + bump sessions.updated_at, atomically.

        Used by the streaming path at stream start, in its own
        short-lived connection.
        """
        dsn = settings.POSTGRES_DSN.replace("postgresql+asyncpg://", "postgresql://")
        conn = None
        try:
            conn = await asyncpg.connect(dsn)
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO session_messages
                        (session_id, role, content, thinking, citations, created_at)
                    VALUES ($1, 'user', $2, NULL, NULL, NOW())
                    """,
                    session_id, user_message,
                )
                await conn.execute(
                    "UPDATE sessions SET updated_at = NOW() WHERE session_id = $1",
                    session_id,
                )
        except Exception as e:
            logger.error(
                "_persist_user_msg failed for session %s: %s",
                session_id, e, exc_info=True,
            )
        finally:
            if conn is not None:
                await conn.close()

    async def _persist_assistant_msg(
        self,
        session_id: uuid.UUID,
        assistant_response: str,
        thinking: str | None,
        citations: Any,
    ) -> None:
        """Insert an assistant message + bump sessions.updated_at, atomically.

        Used by the streaming path at stream end (own connection, own transaction).
        """
        dsn = settings.POSTGRES_DSN.replace("postgresql+asyncpg://", "postgresql://")
        conn = None
        try:
            conn = await asyncpg.connect(dsn)
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO session_messages
                        (session_id, role, content, thinking, citations, created_at)
                    VALUES ($1, 'assistant', $2, $3, $4, NOW())
                    """,
                    session_id, assistant_response, thinking, json.dumps(citations) if citations is not None else None,
                )
                await conn.execute(
                    "UPDATE sessions SET updated_at = NOW() WHERE session_id = $1",
                    session_id,
                )
        except Exception as e:
            logger.error(
                "_persist_assistant_msg failed for session %s: %s",
                session_id, e, exc_info=True,
            )
        finally:
            if conn is not None:
                await conn.close()

    async def _persist_turn(
        self,
        session_id: uuid.UUID,
        user_message: str,
        assistant_response: str,
        thinking: str | None,
        citations: Any,
    ) -> None:
        """Insert two messages (user + assistant) and bump
        sessions.updated_at. Single transaction.

        Logs and swallows DB failures — chat answer should still reach
        the user even if persistence breaks (degradation, not failure).
        """
        dsn = settings.POSTGRES_DSN.replace("postgresql+asyncpg://", "postgresql://")
        conn = None
        try:
            conn = await asyncpg.connect(dsn)
            # Atomic: either both messages and the timestamp bump land,
            # or none do. Prevents orphan user-only rows on partial failure.
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO session_messages
                        (session_id, role, content, thinking, citations, created_at)
                    VALUES ($1, 'user', $2, NULL, NULL, NOW())
                    """,
                    session_id, user_message,
                )
                await conn.execute(
                    """
                    INSERT INTO session_messages
                        (session_id, role, content, thinking, citations, created_at)
                    VALUES ($1, 'assistant', $2, $3, $4, NOW())
                    """,
                    session_id, assistant_response, thinking, json.dumps(citations) if citations is not None else None,
                )
                await conn.execute(
                    "UPDATE sessions SET updated_at = NOW() WHERE session_id = $1",
                    session_id,
                )
        except Exception as e:
            logger.error(
                "_persist_turn failed for session %s: %s",
                session_id, e, exc_info=True,
            )
        finally:
            if conn is not None:
                await conn.close()

    async def _maybe_auto_title(
        self,
        session_id: uuid.UUID,
        first_user_message: str,
    ) -> None:
        """If the session's title is still the default 'New chat',
        replace it with a truncated version of the first user message.
        Uses a WHERE-guarded UPDATE so user renames are never overwritten.
        """
        truncated = first_user_message.strip()
        if len(truncated) > 60:
            truncated = truncated[:60].rsplit(" ", 1)[0]
            if not truncated:  # one giant word
                truncated = first_user_message.strip()[:60]

        dsn = settings.POSTGRES_DSN.replace("postgresql+asyncpg://", "postgresql://")
        conn = None
        try:
            conn = await asyncpg.connect(dsn)
            await conn.execute(
                "UPDATE sessions SET title = $1 WHERE session_id = $2 AND title = 'New chat'",
                truncated, session_id,
            )
        except Exception as e:
            logger.warning("_maybe_auto_title failed: %s", e)
        finally:
            if conn is not None:
                await conn.close()

    async def _should_resummarize(self, session_id: uuid.UUID) -> bool:
        """True when the unsummarized message count past the verbatim
        window has crossed SUMMARIZATION_CADENCE_MESSAGES."""
        dsn = settings.POSTGRES_DSN.replace("postgresql+asyncpg://", "postgresql://")
        conn = None
        try:
            conn = await asyncpg.connect(dsn)
            row = await conn.fetchrow(
                """
                SELECT
                    COALESCE(MAX(id), 0) AS max_id,
                    (SELECT summary_through_message_id FROM sessions WHERE session_id = $1) AS summary_through_message_id
                FROM session_messages
                WHERE session_id = $1
                """,
                session_id,
            )
        except Exception as e:
            logger.warning("_should_resummarize fetch failed: %s", e)
            return False
        finally:
            if conn is not None:
                await conn.close()

        if row is None:
            return False
        max_id = int(row["max_id"] or 0)
        summary_through = int(row["summary_through_message_id"] or 0)
        unsummarized_past_window = (
            max_id - summary_through - settings.SESSION_WINDOW_MESSAGES
        )
        return unsummarized_past_window >= settings.SUMMARIZATION_CADENCE_MESSAGES

    async def _maybe_summarize(self, session_id: uuid.UUID) -> None:
        """Re-summarize the older portion of a session into sessions.summary.

        Loads the existing summary + messages strictly above the
        high-water mark and OLDER than the verbatim window. Asks the
        LLM for a fresh <=SUMMARY_MAX_TOKENS summary. Writes back to
        sessions.summary + sessions.summary_through_message_id. On any
        failure: log + return (keep old summary). Never blocks chat.
        """
        if not settings.ENABLE_SESSION_SUMMARIZATION:
            return

        dsn = settings.POSTGRES_DSN.replace("postgresql+asyncpg://", "postgresql://")
        conn = None
        try:
            conn = await asyncpg.connect(dsn)
            session_meta = await conn.fetchrow(
                "SELECT summary, summary_through_message_id FROM sessions WHERE session_id = $1",
                session_id,
            )
            if session_meta is None:
                return

            existing_summary = session_meta["summary"] or ""
            high_watermark = int(session_meta["summary_through_message_id"] or 0)

            messages = await conn.fetch(
                """
                SELECT id, role, content
                FROM session_messages
                WHERE session_id = $1 AND id > $2
                  AND id <= COALESCE(
                      (SELECT MAX(id) FROM session_messages WHERE session_id = $1)
                      - $3, 0)
                ORDER BY id ASC
                """,
                session_id, high_watermark, settings.SESSION_WINDOW_MESSAGES,
            )

            if not messages:
                return

            transcript_lines: List[str] = []
            for m in messages:
                role_label = "User" if m["role"] == "user" else "Assistant"
                transcript_lines.append(f"{role_label}: {m['content']}")
            transcript = "\n".join(transcript_lines)

            sys_prompt = (
                "You are summarizing the older portion of an ongoing chat between a user "
                "and an AI assistant grounded on documents. Produce a concise summary "
                f"(<= {settings.SUMMARY_MAX_TOKENS} tokens) that preserves: user goals, key facts, "
                "documents discussed, decisions reached, open questions. Omit pleasantries "
                "and false starts. Output plain text only."
            )
            user_prompt = (
                (f"Existing summary:\n{existing_summary}\n\n" if existing_summary else "")
                + f"New transcript to merge:\n{transcript}"
            )
            try:
                llm_res = await self.llm_client.generate_response(
                    messages=[ChatMessage(role="user", content=user_prompt)],
                    system_prompt=sys_prompt,
                    temperature=0.0,
                    stream=False,
                )
            except Exception as e:
                logger.warning("Summarization LLM call failed for session %s: %s", session_id, e)
                return

            new_summary = (llm_res.get("content") if isinstance(llm_res, dict) else llm_res) or ""
            new_summary = new_summary.strip()
            if not new_summary:
                logger.warning("Summarization returned empty content for session %s", session_id)
                return

            new_watermark = int(messages[-1]["id"])
            await conn.execute(
                "UPDATE sessions SET summary = $1, summary_through_message_id = $2 WHERE session_id = $3",
                new_summary, new_watermark, session_id,
            )
        except Exception as e:
            logger.warning("_maybe_summarize unexpected error for session %s: %s", session_id, e)
        finally:
            if conn is not None:
                await conn.close()

    async def handle_query_streaming(
        self,
        query: str,
        session_ctx: "SessionContext",
        is_disconnected=None,
    ):
        """Streaming variant of ``handle_query``.

        Yields ``{"event": str, "data": dict | list}`` dicts. The caller
        (the FastAPI endpoint) translates each into SSE wire format.

        Event sequence on success:
          1. ``citations`` — once, with retrieved citations (possibly empty).
          2. ``thinking_delta`` / ``content_delta`` — streamed per LLM chunk.
          3. (optional) one final ``content_delta`` carrying the
             reason-promoted message if the LLM produced empty content.
          4. ``done`` — once, with ``{guard, accumulated content stats}``.

        On error: a single ``error`` event then end of generator.
        On client disconnect: generator returns silently (no ``done``).
        """
        # ── 1. Load session + recent history (before persisting current message) ──
        session_row = await self._fetch_session(session_ctx.session_id, session_ctx.user_id)
        if session_row is None:
            yield {"event": "error", "data": {
                "detail": "Session not found.", "stage": "retrieval",
            }}
            return

        recent_messages = await self._load_recent_messages(
            session_ctx.session_id, settings.SESSION_WINDOW_MESSAGES,
        )
        session_summary = session_row.get("summary")

        # ── 2. Persist user message + auto-title ──
        await self._persist_user_msg(session_ctx.session_id, query)
        await self._maybe_auto_title(session_ctx.session_id, query)

        explicit_doc_ids = None
        if session_row.get("doc_ids"):
            explicit_doc_ids = [str(d) for d in session_row["doc_ids"]]

        # ── 3. Retrieval (citations + context) ──
        try:
            unified = await self.gather_unified_context(
                query=query, doc_ids=explicit_doc_ids,
            )
        except Exception as e:
            logger.error("Retrieval failed in streaming path: %s", e, exc_info=True)
            yield {"event": "error", "data": {
                "detail": f"Retrieval failed: {e}", "stage": "retrieval",
            }}
            return

        citations = unified.get("citations", [])

        active_doc_ids = explicit_doc_ids or []
        if active_doc_ids:
            active_scope_text = ", ".join(doc_id[:8] for doc_id in active_doc_ids)
        else:
            active_scope_text = "None. Retrieval may include any completed document."

        context_str = (
            f"=== ACTIVE DOCUMENT SCOPE ===\n"
            f"{active_scope_text}\n\n"
            f"=== DOCUMENT INVENTORY (DETERMINISTIC METADATA) ===\n"
            f"{unified.get('document_inventory', '')}\n\n"
            f"=== SEMANTIC TEXT CONTEXT (VECTOR SEARCH) ===\n"
            f"{unified.get('vector', '')}\n\n"
            f"=== DETERMINISTIC RELATIONSHIPS (GRAPH SEARCH) ===\n"
            f"{unified.get('graph', '')}\n"
        )

        yield {"event": "citations", "data": citations}

        # ── 4. Build the system prompt with summary + history blocks ──
        history_block = ""
        if recent_messages:
            lines = ["--- Conversation so far ---"]
            for m in recent_messages:
                role_label = "User" if m["role"] == "user" else "Assistant"
                lines.append(f"{role_label}: {m['content']}")
            lines.append("--- End conversation so far ---")
            history_block = "\n".join(lines) + "\n\n"

        summary_block = ""
        if session_summary:
            summary_block = (
                f"--- Conversation summary so far ---\n{session_summary}\n"
                f"--- End summary ---\n\n"
            )

        system_prompt = (
            summary_block + history_block
            + "Answer the user using the provided context. Cite sources by id.\n\n"
            + context_str
        )

        # ── 5. Stream the LLM, accumulating buffers + checking disconnect ──
        content_buffer: list[str] = []
        thinking_buffer: list[str] = []
        try:
            async for chunk in self.llm_client.generate_response_streaming(
                messages=[ChatMessage(role="user", content=query)],
                system_prompt=system_prompt,
                temperature=0.0,
            ):
                if is_disconnected is not None:
                    if await is_disconnected():
                        logger.info(
                            "Streaming chat cancelled by client for session %s",
                            session_ctx.session_id,
                        )
                        return  # no done, no assistant persist
                kind = chunk.get("kind")
                text = chunk.get("text") or ""
                if kind == "content":
                    content_buffer.append(text)
                    yield {"event": "content_delta", "data": {"text": text}}
                elif kind == "thinking":
                    thinking_buffer.append(text)
                    yield {"event": "thinking_delta", "data": {"text": text}}
        except Exception as e:
            logger.error("LLM stream failed: %s", e, exc_info=True)
            yield {"event": "error", "data": {
                "detail": f"LLM error: {e}", "stage": "llm",
            }}
            return

        # ── 6. Reason promotion at stream end ──
        full_content = "".join(content_buffer)
        full_thinking = "".join(thinking_buffer)
        from docai.orchestrator.base import classify_empty
        reason = classify_empty(full_content, full_thinking)
        promoted_text = None
        if reason == "empty_content_with_thinking":
            promoted_text = (
                "The model returned reasoning but no answer. "
                "Reasoning is shown below."
            )
        elif reason == "empty_response":
            promoted_text = (
                "The model produced no output. "
                "Please retry, or rephrase your question."
            )

        if promoted_text:
            yield {"event": "content_delta", "data": {"text": promoted_text}}
            persisted_content = promoted_text
        else:
            persisted_content = full_content

        # ── 7. Persist assistant message + maybe schedule summarization ──
        try:
            await self._persist_assistant_msg(
                session_id=session_ctx.session_id,
                assistant_response=persisted_content,
                thinking=full_thinking,
                citations=citations,
            )
        except Exception as e:
            logger.error("Assistant persistence failed: %s", e, exc_info=True)
            yield {"event": "error", "data": {
                "detail": f"Persistence failed: {e}", "stage": "persist",
            }}
            return

        if settings.ENABLE_SESSION_SUMMARIZATION:
            if await self._should_resummarize(session_ctx.session_id):
                asyncio.create_task(self._maybe_summarize(session_ctx.session_id))

        # ── 8. Done ──
        yield {"event": "done", "data": {
            "guard": None,
            "content_chars": len(persisted_content),
            "thinking_chars": len(full_thinking),
        }}

    async def _get_graph_context_with_timeout(self, query: str) -> str:
        try:
            return await asyncio.wait_for(
                self._get_graph_context(query),
                timeout=settings.GRAPH_SEARCH_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Graph search timed out after %ss; continuing with vector context only.",
                settings.GRAPH_SEARCH_TIMEOUT_SECONDS,
            )
            return "Graph search timed out; vector context was still used."

    # ── Main Query Handler ──────────────────────────────────────────────

    async def handle_query(
        self,
        query: str,
        session_id: str = None,
        doc_ids: List[str] | None = None,
        *,
        session_ctx: "SessionContext | None" = None,
    ) -> Dict[str, Any]:
        """
        Integrated RAG Engine.
        1. Parallel retrieval from Vector Search (over-fetch + rerank) and Graph Search.
        2. Construct unified prompt with citation instructions and contradiction resolution.
        3. Generate response.
        """
        logger.info(f"Handling query: {query}")

        # ── Persistent-mode setup (sub-project 1, T9) ────────────────
        # When session_ctx is set AND persistence is enabled, load the
        # session row, recent messages, and rolling summary. These feed
        # into both retrieval scope (doc_ids fall back to the session's)
        # and the LLM prompt (history block + summary block).
        session_row: Dict[str, Any] | None = None
        recent_messages: List[Dict[str, Any]] = []
        session_summary: str | None = None
        if session_ctx is not None and settings.ENABLE_SESSION_PERSISTENCE:
            session_row = await self._fetch_session(
                session_ctx.session_id, session_ctx.user_id
            )
            if session_row is None:
                # Concurrently deleted / wrong owner — bail.
                return {
                    "response": "Session not found.",
                    "thinking": "",
                    "citations": [],
                    "guard": "session_not_found",
                }
            session_summary = session_row.get("summary")
            recent_messages = await self._load_recent_messages(
                session_ctx.session_id,
                settings.SESSION_WINDOW_MESSAGES,
            )
            # Fall back to the session's stored doc_ids if the caller
            # didn't override them.
            if doc_ids is None and session_row.get("doc_ids"):
                doc_ids = [str(d) for d in session_row["doc_ids"]]

        explicit_doc_ids = self._normalise_doc_ids(doc_ids)
        preflight_doc_ids = explicit_doc_ids
        if doc_ids is None and not preflight_doc_ids:
            preflight_doc_ids = self._session_scope(session_id)

        # Pre-flight (Task T2): if the caller pinned or remembered a scope, verify
        # those documents are actually ready before doing retrieval/LLM work.
        excluded: List[DocStatus] = []
        missing_ids: List[str] = []
        doc_uuids: list[uuid.UUID] = []
        if preflight_doc_ids:
            try:
                doc_uuids = [uuid.UUID(d) for d in preflight_doc_ids]
            except (ValueError, TypeError) as e:
                logger.warning(
                    "Skipping pre-flight: invalid UUID in active doc scope (%s)", e
                )
                doc_uuids = []
        if preflight_doc_ids and doc_uuids:
            statuses = await self._fetch_doc_statuses(doc_uuids)
            known_ids = {str(s.doc_id) for s in statuses}
            missing_ids = [d for d in preflight_doc_ids if d not in known_ids]
            usable_ids = [str(s.doc_id) for s in statuses if _doc_status_is_ready(s)]
            excluded = [s for s in statuses if not _doc_status_is_ready(s)]

            # All scoped docs unavailable → short-circuit with a human message.
            if not usable_ids and (excluded or missing_ids):
                indexing = [s.filename for s in excluded if s.status != "failed"]
                failed = [
                    f"{s.filename} ({s.error_message or 'unknown error'})"
                    for s in excluded
                    if s.status == "failed"
                ]
                parts = ["None of the requested documents are ready yet."]
                if indexing:
                    parts.append(
                        f"Indexing: {', '.join(indexing)} still being indexed."
                    )
                if failed:
                    parts.append(f"Failed: {', '.join(failed)}.")
                if missing_ids:
                    parts.append(f"Unknown: {', '.join(missing_ids)}.")
                parts.append("Please retry in a few seconds.")
                return {
                    "response": " ".join(parts),
                    "thinking": "",
                    "citations": [],
                    "guard": "all_scoped_docs_unavailable",
                }

            # Partial scope fall-back: use only the completed subset for the
            # rest of the function. The exclusion note is prepended after the
            # LLM returns so we don't bias generation. Use set comparison —
            # ``_fetch_doc_statuses`` SQL has no ORDER BY, so two equal sets
            # in different orders must NOT trigger the subset branch.
            if set(usable_ids) != set(preflight_doc_ids):
                preflight_doc_ids = usable_ids

        active_doc_ids = preflight_doc_ids
        if doc_ids is None and not active_doc_ids:
            active_doc_ids = self._session_scope(session_id)

        if session_ctx is not None and settings.ENABLE_SESSION_PERSISTENCE:
            # Persistent mode: history comes from Postgres via recent_messages
            # (already loaded earlier). Don't double-feed via in-memory dict.
            retrieval_query = query  # use raw query; history block is in system_prompt
            recent_history = ""       # suppress legacy in-memory render
        else:
            retrieval_query = self._contextual_retrieval_query(session_id, query)
            recent_history = self._format_recent_history(session_id)

        # 1. Parallel Context Retrieval
        unified_context = await self.gather_unified_context(
            retrieval_query,
            doc_ids=active_doc_ids,
            graph_query=query,
        )

        cited_doc_ids = self._normalise_doc_ids(
            [citation.get("doc_id") for citation in unified_context["citations"]]
        )
        remembered_doc_ids = active_doc_ids or cited_doc_ids
        self._remember_session_scope(session_id, remembered_doc_ids)

        if active_doc_ids:
            active_scope_text = ", ".join(doc_id[:8] for doc_id in active_doc_ids)
        else:
            active_scope_text = "None. Retrieval may include any completed document."

        history_section = (
            f"=== RECENT CHAT HISTORY (LATEST {SESSION_HISTORY_MAX_MESSAGES} MESSAGES) ===\n"
            f"{recent_history}\n\n"
            if recent_history
            else ""
        )
        context_str = (
            f"=== ACTIVE DOCUMENT SCOPE ===\n"
            f"{active_scope_text}\n\n"
            + history_section
            + f"=== DOCUMENT INVENTORY (DETERMINISTIC METADATA) ===\n"
            f"{unified_context['document_inventory']}\n\n"
            f"=== SEMANTIC TEXT CONTEXT (VECTOR SEARCH) ===\n"
            f"{unified_context['vector']}\n\n"
            f"=== DETERMINISTIC RELATIONSHIPS (GRAPH SEARCH) ===\n"
            f"{unified_context['graph']}\n"
        )

        # 2. Prompt Construction with citation instructions
        system_prompt = (
            "You are a highly accurate helpful assistant grounded in the provided context. "
            "You have access to two types of context: Semantic Text Context (snippets from documents) and "
            "Deterministic Relationships (factual graph triples). "
            "Use BOTH sources to construct a comprehensive answer. "
            "\n\n"
            "CITATION RULES:\n"
            "- Each text source has a 4-character citation ID like [a3z1].\n"
            "- You MUST cite sources inline using these IDs when referencing information.\n"
            "- Use at most 3 citations per claim.\n"
            "- Example: 'Revenue grew 12% year-over-year [a3z1][b2x4].'\n"
            "\n"
            "CONTRADICTION RULES:\n"
            "- If the Semantic Text Context contradicts the Deterministic Relationships, "
            "you MUST favor the Deterministic Relationships (Graph) or explicitly state the discrepancy.\n"
            "- If the answer is not contained in either context, say "
            "'I cannot answer this based on the provided documents.'\n"
            "\n"
            "DOCUMENT FACT RULES:\n"
            "- For filename, ingestion status, page count, and chunk count, use DOCUMENT INVENTORY first.\n"
            "- Do not infer title, author, or page count from prose unless the inventory or source text states it explicitly.\n"
            "- If title or author metadata is not present, say it is not available rather than inventing it.\n"
            "\n"
            "CONVERSATION CONTINUITY RULES:\n"
            f"- Use RECENT CHAT HISTORY, capped to the latest {SESSION_HISTORY_MAX_MESSAGES} messages, "
            "to understand follow-up questions.\n"
            "- If ACTIVE DOCUMENT SCOPE lists one or more docs, keep answers grounded to that active scope.\n"
            "- Do not mix in facts from documents outside ACTIVE DOCUMENT SCOPE.\n"
            "- If there is no active scope and chat history does not identify the intended document, "
            "ask which document the user means.\n"
        )

        user_prompt = f"Context:\n{context_str}\n\nUser Question:\n{query}"
        messages = [ChatMessage(role="user", content=user_prompt)]

        # ── Session prompt blocks (sub-project 1, T9) ────────────────
        history_block = ""
        if recent_messages:
            history_lines = ["--- Conversation so far ---"]
            for m in recent_messages:
                role_label = "User" if m["role"] == "user" else "Assistant"
                history_lines.append(f"{role_label}: {m['content']}")
            history_lines.append("--- End conversation so far ---")
            history_block = "\n".join(history_lines) + "\n\n"

        summary_block = ""
        if session_summary:
            summary_block = (
                f"--- Conversation summary so far ---\n"
                f"{session_summary}\n"
                f"--- End summary ---\n\n"
            )

        # Prepend both blocks to whatever system_prompt was assembled.
        system_prompt = summary_block + history_block + system_prompt

        # 3. Generation
        logger.info("Prompting LLM with Unified Context...")
        llm_res = await self.llm_client.generate_response(
            messages=messages,
            system_prompt=system_prompt,
            temperature=0.3,
            stream=False
        )

        if isinstance(llm_res, dict):
            response = llm_res.get("content", "")
            thinking = llm_res.get("thinking", "")
            reason = llm_res.get("reason")
        else:
            response = llm_res
            thinking = ""
            reason = None

        # Reason promotion (Task T2 step 5): turn the raw classifier code
        # from OllamaClient into a user-facing message.
        if reason == "empty_content_with_thinking":
            response = (
                "The model returned reasoning but no answer. "
                "Reasoning is shown below."
            )
        elif reason == "empty_response":
            response = (
                "The model produced no output. "
                "Please retry, or rephrase your question."
            )

        # Partial-scope exclusion note (Task T2 step 3): if pre-flight
        # dropped some requested docs, surface that to the user up front.
        # By construction, when ``excluded`` or ``missing_ids`` is non-empty
        # we either short-circuited above or kept only ready docs, so a separate
        # guard on the active scope is dead.
        if excluded or missing_ids:
            indexing_names = [s.filename for s in excluded if s.status != "failed"]
            failed_names = [s.filename for s in excluded if s.status == "failed"]
            parts = []
            if indexing_names:
                parts.append(f"still indexing: {', '.join(indexing_names)}")
            if failed_names:
                parts.append(f"failed: {', '.join(failed_names)}")
            if missing_ids:
                parts.append(f"unknown: {', '.join(missing_ids)}")
            if parts:
                note = f"Note: excluded from this answer — {'; '.join(parts)}.\n\n"
                response = note + response

        if session_ctx is None or not settings.ENABLE_SESSION_PERSISTENCE:
            self._remember_session_message(session_id, "user", query)
        if session_ctx is None or not settings.ENABLE_SESSION_PERSISTENCE:
            self._remember_session_message(session_id, "assistant", response)

        result = {
            "response": response,
            "thinking": thinking,
            "citations": unified_context["citations"]
        }

        # ── Persistent-mode write-back (sub-project 1, T10) ────────────
        if (
            session_ctx is not None
            and session_row is not None
            and settings.ENABLE_SESSION_PERSISTENCE
        ):
            # Identify the response/thinking/citations to persist.
            # These come from the result dict assembled by the existing
            # engine logic. If the existing code uses different local
            # names, adapt the keys but keep the values consistent.
            persistable_response = result.get("response", "") if isinstance(result, dict) else str(result)
            persistable_thinking = result.get("thinking", "") if isinstance(result, dict) else ""
            persistable_citations = result.get("citations", []) if isinstance(result, dict) else []

            await self._persist_turn(
                session_id=session_ctx.session_id,
                user_message=query,
                assistant_response=persistable_response,
                thinking=persistable_thinking,
                citations=persistable_citations,
            )
            await self._maybe_auto_title(session_ctx.session_id, query)
            if settings.ENABLE_SESSION_SUMMARIZATION:
                if await self._should_resummarize(session_ctx.session_id):
                    asyncio.create_task(
                        self._maybe_summarize(session_ctx.session_id)
                    )

        return result

