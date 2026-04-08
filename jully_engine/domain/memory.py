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

    async def search(self, query: str, top_k: int = 3, collection: str = "july_memory") -> str:
        """Searches the RAG database using the query embedding."""
        from ..persistence.vector_store import vector_store
        
        payload = {"input": query}
        embedding_result = await self.embed(payload, emb_type='query')
        
        if embedding_result and len(embedding_result) > 0:
            # 🛡️ BLINDAGEM DE DIMENSÃO (Extrai o vetor corretamente)
            if isinstance(embedding_result[0], list):
                embedding = embedding_result[0] # Matriz 2D (API)
            else:
                embedding = embedding_result    # Array 1D (Local)
                
            results = vector_store.search(embedding, top_k=top_k, collection=collection, model_tag=self.model_tag)
            return "\n---\n".join(results)
            
        return "Nenhuma memória encontrada."

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