import logging
from typing import Any, Dict, List, Optional

from ..models.base_model import BaseModel

logger = logging.getLogger("JulyEngine.Adapter.RagAdapter")

# Maps task_type → handler method name (no if/elif chains)
_TASK_HANDLERS = {
    "embeddings":    "_embed",
    "rag_add":       "_rag_add",
    "rag_batch_add": "_rag_batch_add",
    "rag_vector_add":"_rag_vector_add",
    "rag_search":    "_rag_search",
    "rag_update":    "_rag_update",
    "rag_delete":    "_rag_delete",
    "rag_list":      "_rag_list",
    "rag_smart_search": "_rag_smart_search",
}


class RagAdapter(BaseModel):
    """
    Handles embedding generation and all vector-store (RAG) operations.

    Mirrors july_engine's Memory domain.  The inner embedding model
    is selected lazily from meta["embedding_engine"] (default: bge_micro).

    task_type is injected into payload["_task_type"] by the orchestrator's
    Runner before calling run(), so this adapter knows which operation to
    perform without any routing in the bridge.
    """

    def __init__(self, task_type: str, backend: str = "cpu", model_meta: Optional[dict] = None):
        super().__init__(backend, model_meta)
        self._emb_model = None
        self.task_type = task_type

    # ------------------------------------------------------------------
    # Lazy embedding model — only imported/created when first needed
    # ------------------------------------------------------------------

    def _get_emb_model(self):
        if self._emb_model is None:
            engine = self.meta.get("embedding_engine", "bge_micro")
            if engine == "multilingual_e5":
                from ..models.multilingual_e5 import MultilingualE5Model
                self._emb_model = MultilingualE5Model(backend=self.backend, model_meta=self.meta)
            else:  # default: bge_micro
                from ..models.bge_micro import BgeMicroModel
                self._emb_model = BgeMicroModel(backend=self.backend, model_meta=self.meta)
        return self._emb_model

    def _get_vector_store(self):
        from ..persistence.vector_store import vector_store
        return vector_store

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        if "_task_type" in payload and not payload["_task_type"].startswith("rag_") and payload["_task_type"] != "embeddings":
            return 0
        return await self._get_emb_model().get_required_vram(payload)

    def load(self, n_ctx: Optional[int] = None, num_layers: Optional[int] = None):
        self._get_emb_model().load()

    def is_loaded(self) -> bool:
        return self._emb_model is not None and self._emb_model.is_loaded()

    def unload(self, model_name: Optional[str] = None):
        if self._emb_model:
            self._emb_model.unload()

    async def run(self, payload: Dict[str, Any]):
        task_type = self.task_type
        handler_name = _TASK_HANDLERS.get(task_type)

        if not handler_name:
            raise ValueError(f"RagAdapter: unknown task_type '{task_type}'")

        handler = getattr(self, handler_name)
        return await handler(payload)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    async def _embed(self, payload: Dict[str, Any]) -> List[float]:
        input_text = payload.get("input") or payload.get("text") or payload.get("query", "")
        emb_type = payload.get("emb_type", "passage")
        emb_payload = {"input": input_text, "emb_type": emb_type}
        return self._get_emb_model().run(emb_payload)

    async def _embed_text(self, text: str, emb_type: str = "passage") -> List[float]:
        result = self._get_emb_model().run({"input": text, "emb_type": emb_type})
        # LLMApi may return [[...]] (matrix), local models return [...]
        if result and isinstance(result[0], list):
            return result[0]
        return result

    async def _rag_add(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        text = payload.get("text", "")
        metadata = payload.get("metadata") or {}
        collection = payload.get("collection", "july_memory")
        doc_id = payload.get("doc_id")
        model_tag = self.meta.get("alias") or self.meta.get("model", "default")

        embedding = await self._embed_text(text, "passage")
        if not embedding:
            return {"success": False, "error": "embedding failed"}

        self._get_vector_store().add(
            text, embedding, metadata,
            collection=collection,
            model_tag=model_tag,
            doc_id=doc_id,
        )
        return {"success": True, "collection": collection}

    async def _rag_batch_add(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        documents = payload.get("documents", [])
        collection = payload.get("collection", "july_memory")
        model_tag = self.meta.get("alias") or self.meta.get("model", "default")
        inserted = failed = 0

        for doc in documents:
            text = doc.get("text", "").strip()
            if not text:
                failed += 1
                continue
            try:
                embedding = await self._embed_text(text, "passage")
                if embedding:
                    self._get_vector_store().add(
                        text, embedding, doc.get("metadata") or {},
                        collection=collection, model_tag=model_tag,
                        doc_id=doc.get("id"),
                    )
                    inserted += 1
                else:
                    failed += 1
            except Exception as e:
                logger.warning(f"RagAdapter: batch insert failed for doc: {e}")
                failed += 1

        return {"inserted": inserted, "failed": failed, "collection": collection}

    async def _rag_vector_add(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        embedding = payload.get("vector", [])
        text = payload.get("text", "")
        metadata = payload.get("metadata") or {}
        collection = payload.get("collection", "july_memory")
        model_tag = self.meta.get("alias") or self.meta.get("model", "default")

        self._get_vector_store().add(
            text, embedding, metadata,
            collection=collection, model_tag=model_tag,
        )
        return {"success": True, "collection": collection}

    async def _rag_search(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        query = payload.get("query", "")
        top_k = payload.get("top_k", 3)
        collection = payload.get("collection", "july_memory")
        filter_ = payload.get("filter")
        model_tag = self.meta.get("alias") or self.meta.get("model", "default")

        # Vector-only search (pre-computed embedding supplied)
        if "vector" in payload:
            return self._get_vector_store().search_with_details(
                payload["vector"], top_k=top_k,
                collection=collection, model_tag=model_tag, filter=filter_,
            )

        embedding = await self._embed_text(query, "query")
        if not embedding:
            return []

        return self._get_vector_store().search_with_details(
            embedding, top_k=top_k,
            collection=collection, model_tag=model_tag, filter=filter_,
        )

    async def _rag_update(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        doc_id = payload.get("id", "")
        new_vector = payload.get("vector", [])
        metadata = payload.get("metadata") or {}
        collection = payload.get("collection", "july_memory")
        model_tag = self.meta.get("alias") or self.meta.get("model", "default")

        self._get_vector_store().update_embedding(
            doc_id, new_vector, collection=collection, model_tag=model_tag,
        )
        return {"success": True, "id": doc_id}

    async def _rag_delete(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        ids = payload.get("ids", [])
        collection = payload.get("collection", "july_memory")
        model_tag = self.meta.get("alias") or self.meta.get("model", "default")

        count = self._get_vector_store().delete(ids, collection=collection, model_tag=model_tag)
        return {"deleted": count, "collection": collection}

    async def _rag_list(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        collection = payload.get("collection", "july_memory")
        filter_ = payload.get("filter")
        model_tag = self.meta.get("alias") or self.meta.get("model", "default")

        return self._get_vector_store().list_metadata(
            collection=collection, model_tag=model_tag, filter=filter_,
        )

    async def _rag_smart_search(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Multi-step intelligent RAG search (mirrors Memory.smart_search):
        1. Use an LLM to decompose the user query into focused sub-queries.
        2. Embed + search each sub-query.
        3. Deduplicate results.
        4. LLM relevance filter.
        5. Optional: structure answer via LLM.
        """
        import asyncio

        prompt = payload.get("prompt", "")
        top_k = payload.get("top_k", 5)
        max_split = payload.get("max_split_questions", 3)
        collection = payload.get("collection", "july_memory")
        llm_model = payload.get("llm_model")
        structured = payload.get("structured_response", False)
        stream_response = payload.get("stream_response", False)
        headers = payload.get("headers", {})

        # -- 1. Decompose query --
        breakdown_payload = {
            "model": llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a technical search expert generating queries for a Vector Database (RAG).\n"
                        f"Break the user task into {max_split} distinct, highly descriptive semantic search queries.\n"
                        "DO NOT invent file extensions or literal class names unless specified.\n"
                        "Focus on concepts, architectures, and technical workflows."
                    ),
                },
                {"role": "user", "content": f"Task: {prompt}\nOutput ONLY the queries, one per line."},
            ],
            "stream": False,
            "headers": {**headers, "x-enable-internal-mcp": "0"},
        }

        queries = await self._llm_call(breakdown_payload, llm_model)
        if queries:
            queries = [q.strip().lstrip("-*123. ") for q in queries.split("\n") if q.strip()]
            queries = queries[:max_split]
        else:
            queries = [prompt]

        # -- 2. Search per sub-query, deduplicate --
        seen_ids: set = set()
        raw_results: List[Dict] = []

        for q in queries:
            try:
                results = await self._rag_search({"query": q, "top_k": top_k, "collection": collection})
                for r in results:
                    rid = r.get("id")
                    if rid not in seen_ids:
                        seen_ids.add(rid)
                        raw_results.append(r)
            except Exception as e:
                logger.warning(f"RagAdapter smart_search: query failed '{q}': {e}")

        # -- 3. Relevance filter --
        final_results: List[Dict] = []

        async def check_relevance(item: Dict) -> bool:
            path = item.get("metadata", {}).get("path") or item.get("metadata", {}).get("file") or "unknown"
            content = item.get("content") or item.get("text") or ""

            rel_payload = {
                "model": llm_model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a technical document relevance analyzer.\n"
                            "Determine if the snippet is strictly useful for the task.\n"
                            "Answer ONLY with YES or NO."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"TASK: {prompt}\nFILE: {path}\nCONTENT: {content[:2000]}\n\nIs this relevant?",
                    },
                ],
                "stream": False,
                "max_tokens": 5,
                "headers": {**headers, "x-enable-internal-mcp": "0"},
            }

            answer = await self._llm_call(rel_payload, llm_model)
            return "YES" in (answer or "").upper()

        for item in raw_results:
            if await check_relevance(item):
                final_results.append({k: v for k, v in item.items() if k != "embedding"})

        # -- 4. Optional structured answer --
        if structured and final_results:
            data = "\n".join(r.get("content", "") for r in final_results)
            struct_payload = {
                "model": llm_model,
                "messages": [
                    {"role": "system", "content": f"Based on this data:\n{data}\nAnswer the user's question."},
                    {"role": "user", "content": prompt},
                ],
                "stream": stream_response,
                "headers": headers,
            }
            from ..bridge import bridge
            result = await bridge.process_openai_chat(struct_payload, headers)
            return {"results": result}

        return {"results": final_results}

    # ------------------------------------------------------------------
    # Internal LLM helper (calls bridge to avoid coupling to orchestrators)
    # ------------------------------------------------------------------

    async def _llm_call(self, chat_payload: Dict[str, Any], model: Optional[str]) -> Optional[str]:
        try:
            from ..bridge import bridge

            headers = chat_payload.get("headers", {})
            response = await bridge.process_openai_chat(chat_payload, headers)

            if isinstance(response, dict) and response.get("choices"):
                return response["choices"][0]["message"].get("content", "")
            return None
        except Exception as e:
            logger.error(f"RagAdapter: LLM call failed: {e}")
            return None
