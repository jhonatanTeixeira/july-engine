from __future__ import annotations
import logging
import inspect
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..engine_models.bge_micro import BgeMicro
    from ..engine_models.multilingual_e5 import MultilingualE5
    from ..engine_models.llm_api import LLMApi

logger = logging.getLogger("JulyEngine.Domain.Memory")

class Memory:
    """
    Handles embeddings for long-term memory.
    Strategies: BgeMicro (cpu), MultilingualE5 (gpu), LLMApi (api).
    """
    def __init__(self, backend: str, model_tag: str):
        self.backend = backend
        self.model_tag = model_tag
        self._strategy = self._get_strategy()

    def _get_strategy(self):
        if self.backend == "api":
            from ..engine_models.llm_api import LLMApi
            return LLMApi(backend=self.backend)
        elif self.model_tag == "bge-micro":
            from ..engine_models.bge_micro import BgeMicro
            return BgeMicro(backend=self.backend)
        elif self.model_tag == "multilingual-e5":
            from ..engine_models.multilingual_e5 import MultilingualE5
            return MultilingualE5(backend=self.backend)
        else:
            raise ValueError(f"Memory: Unsupported backend/model combination: {self.backend}/{self.model_tag}")

    def get_required_vram(self, payload: Dict[str, Any]) -> int:
        """Delega a estimativa de VRAM para a estratégia atual."""
        # Se for uma tarefa de manutenção (sem input de texto), o custo é 0
        if "input" not in payload and "documents" not in payload:
            return 0
            
        if hasattr(self._strategy, "get_required_vram"):
            return self._strategy.get_required_vram(payload)
        return 0

    async def embed(self, payload: Dict[str, Any], emb_type: str = "passage"):
        from ..engine_models.llm_api import LLMApi
        from ..engine_models.bge_micro import BgeMicro
        from ..engine_models.multilingual_e5 import MultilingualE5

        if isinstance(self._strategy, LLMApi):
            model = payload.pop("model", self.model_tag)
            input_text = payload.pop("input", "")
            headers = payload.pop("headers", {})
            res = self._strategy.run_embeddings(model, input_text, headers=headers, **payload)
            
            if inspect.iscoroutine(res):
                res = await res
            
            return res
            
        elif isinstance(self._strategy, (BgeMicro, MultilingualE5)):
            input_text = payload.get("input", "")
            
            if emb_type == 'passage':
                return self._strategy.run_passage(input_text)
            elif emb_type == 'query':
                return self._strategy.run_query(input_text)
            else:
                raise ValueError(f"Memory: Unsupported embedding type: {emb_type}")
            
        return None

    async def add_to_rag(self, text: str, metadata: Dict[str, Any] = None, collection: str = "july_memory", doc_id: str = None):
        """Generates an embedding and adds it to the RAG database."""
        from ..persistence.vector_store import vector_store
        
        # Get embedding
        payload = {"input": text}
        embedding_result = await self.embed(payload, emb_type='passage')
        
        if embedding_result and len(embedding_result) > 0:
            # 🛡️ BLINDAGEM DE DIMENSÃO (Extrai o vetor corretamente)
            if isinstance(embedding_result[0], list):
                embedding = embedding_result[0] # Matriz 2D (API)
            else:
                embedding = embedding_result    # Array 1D (Local)
                
            vector_store.add(text, embedding, metadata, collection=collection, model_tag=self.model_tag, doc_id=doc_id)

            return True
        return False

    async def search(self, query: str, top_k: int = 3, collection: str = "july_memory") -> List[Dict[str, Any]]:
        """Searches the RAG database using the query embedding and returns details."""
        from ..persistence.vector_store import vector_store
        
        payload = {"input": query}
        embedding_result = await self.embed(payload, emb_type='query')
        
        if embedding_result and len(embedding_result) > 0:
            # 🛡️ BLINDAGEM DE DIMENSÃO (Extrai o vetor corretamente)
            if isinstance(embedding_result[0], list):
                embedding = embedding_result[0] # Matriz 2D (API)
            else:
                embedding = embedding_result    # Array 1D (Local)
                
            return vector_store.search_with_details(embedding, top_k=top_k, collection=collection, model_tag=self.model_tag)
            
        return []

    async def add_batch_to_rag(self, documents: List[Dict[str, Any]], collection: str = "july_memory") -> Dict[str, int]:
        """Inserts multiple documents into the RAG database."""
        inserted = 0
        failed = 0
        
        for doc in documents:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            if not text or not text.strip():
                failed += 1
                continue
            
            try:
                success = await self.add_to_rag(text=text, metadata=metadata, collection=collection, doc_id=doc.get("id"))
                if success:
                    inserted += 1
                else:
                    failed += 1
            except Exception as e:
                logger.warning(f"Memory: RAG batch - failed to insert: {e}")
                failed += 1
                
        return {"inserted": inserted, "failed": failed, "collection": collection}

    async def add_vector_to_rag(self, embedding: List[float], text: str = "", metadata: Dict[str, Any] = None, collection: str = "july_memory") -> bool:
        """Adiciona um embedding pré-calculado diretamente no banco (ex: Face Embeddings)."""
        from ..persistence.vector_store import vector_store
        vector_store.add(text, embedding, metadata, collection=collection, model_tag=self.model_tag)
        
        return True

    async def search_with_details_vector(self, query_embedding: List[float], top_k: int = 1, collection: str = "july_memory") -> List[Dict[str, Any]]:
        """Busca RAG pulando o Text-Embedder e pedindo Metadados (PGVector)."""
        from ..persistence.vector_store import vector_store
        return vector_store.search_with_details(query_embedding, top_k=top_k, collection=collection, model_tag=self.model_tag)

    async def update_embedding(self, doc_id: str, new_embedding: List[float], collection: str = "july_memory"):
        """Atualiza a coordenada geométrica de um vetor existente pelo ID."""
        from ..persistence.vector_store import vector_store
        vector_store.update_embedding(doc_id, new_embedding, collection=collection, model_tag=self.model_tag)

    async def delete_from_rag(self, ids: List[str], collection: str = "july_memory") -> int:
        """Remove registros do RAG pelo ID."""
        from ..persistence.vector_store import vector_store
        return vector_store.delete(ids, collection=collection, model_tag=self.model_tag)

    async def list_rag_metadata(self, collection: str = "july_memory") -> List[Dict[str, Any]]:
        """Lista IDs e metadados de uma coleção."""
        from ..persistence.vector_store import vector_store
        return vector_store.list_metadata(collection=collection, model_tag=self.model_tag)

    async def smart_search(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Realiza uma busca 'inteligente' no RAG:
        1. Quebra a query em sub-queries focadas (Processado via Brain).
        2. Busca no RAG para cada sub-query.
        3. Filtra resultados duplicados.
        4. Analisa a relevância de cada resultado (Processado via Brain).
        """
        from ..services.helpers import inference_helper
        import asyncio
        
        prompt = payload.get("prompt")
        top_k = payload.get("top_k", 5)
        max_split_questions = payload.get("max_split_questions", 3)
        collection = payload.get("collection", "july_memory")
        llm_model = payload.get("llm_model")

        # 1. Break down the task into sub-queries
        system_prompt = (
            "You are a technical search expert generating queries for a Vector Database (RAG).\n"
            f"Break the user's task into {max_split_questions} distinct, highly descriptive semantic search queries.\n"
            "DO NOT invent file extensions or literal class names unless the user specifies them.\n"
            "Focus on concepts, architectures, and technical workflows."
        )
        
        user_prompt = f"Task: {prompt}\nOutput ONLY the queries, one per line."

        headers = {"x-enable-internal-mcp": "1"}

        breakdown_payload = {
            "model": llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "headers": headers
        }

        logger.debug(f"Memory: Breakdown payload: {breakdown_payload}")

        breakdown_res = await inference_helper.process("text_chat", breakdown_payload)

        if isinstance(breakdown_res, dict) and "choices" in breakdown_res:
            content = breakdown_res["choices"][0]["message"]["content"]
            queries = [q.strip().lstrip("-*123. ") for q in content.strip().split("\n") if q.strip()]
            queries = queries[:max_split_questions]
        else:
            queries = [prompt]

        logger.debug(f"Memory: Generated queries: {queries}")

        # 2. Search for each query
        all_raw_results = []
        seen_ids = set()
        
        for q in queries:
            try:
                results = await self.search(query=q, top_k=top_k, collection=collection)
                for r in results:
                    rid = r.get("id")
                    if rid not in seen_ids:
                        seen_ids.add(rid)
                        all_raw_results.append(r)
                
                logger.info(f"Memory: Found {len(results)} results for query '{q}'")
            except Exception as e:
                logger.error(f"Memory: Error searching RAG for query '{q}': {e}")

        # 3. Analyze relevance for each result
        final_results = []
        
        async def check_relevance(item):
            path = item.get("metadata", {}).get("path") or item.get("metadata", {}).get("file") or "unknown"
            content = item.get("content") or item.get("text") or ""
            
            system_prompt = (
                "You are a technical document relevance analyzer.\n"
                "Determine if the document snippet is strictly useful for fulfilling the user task.\n"
                "Answer ONLY with YES or NO."
            )
            
            user_prompt = (
                f"TASK: {prompt}\n"
                f"FILE: {path}\n"
                f"CONTENT: {content[:2000]}\n\n"
                "Is this relevant?"
            )
            
            rel_payload = {
                "model": llm_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "max_tokens": 5,
                "headers": headers
            }

            logger.debug(f"Memory: Relevance payload: {rel_payload}")
            logger.info(f"Memory: checking relevance for item")
            
            try:
                rel_res = await inference_helper.process("text_chat", rel_payload)
                logger.debug(f"Memory: Relevance response: {rel_res}")

                if rel_res.get("choices"):
                    ans: str = rel_res.get("choices", [{}])[0].get("message", {}).get("content", "").upper()
                    logger.debug(f"Memory: Relevance answer: {ans}")
                    return "YES" in ans
                return False
            except Exception:
                return False

        # Run relevance checks sequentially
        if all_raw_results:
            for item in all_raw_results:
                is_relevant = await check_relevance(item)
                if is_relevant:
                    # Omit vector/embedding data
                    result_item = {k: v for k, v in item.items() if k != "embedding"}
                    final_results.append(result_item)

        logger.debug(f"Memory: Final results: {final_results}")

        if payload.get('structured_response', False):
            logger.info("Memory: Structuring response")

            data = '\n'.join([r['content'] for r in final_results])
            system_prompt = f"Based on given data: {data}, answer the user's question."
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]

            structured_payload = {
                "model": llm_model,
                "messages": messages,
                "stream": payload.get("stream_response", False),
            }

            from ..bridge import bridge

            return bridge.process_openai_chat(structured_payload, headers)

        return final_results