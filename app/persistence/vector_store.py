import os
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("JulyEngine.VectorStore")

class VectorStore:
    def __init__(self):
        self.db_type = os.environ.get("RAG_DATABASE", "in-memory").lower()
        self.base_collection = "july_memory"
        
        # Cache de coleções já verificadas/criadas nesta sessão
        self.initialized_collections = set()

        if self.db_type == "chroma":
            try:
                import chromadb
                # Store locally in storage/db/chroma
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                db_path = os.path.join(base_dir, "storage", "db", "chroma")
                os.makedirs(db_path, exist_ok=True)
                self.client = chromadb.PersistentClient(path=db_path)
            except ImportError:
                logger.error("ChromaDB not installed. Fallback to in-memory.")
                self.db_type = "in-memory"
                self._init_in_memory()
        elif self.db_type == "pgvector":
            self.connection_string = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/july_engine")
            self._init_pgvector_extension()
        else:
            self._init_in_memory()

    def _init_in_memory(self):
        self.db_type = "in-memory"
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.json_path = os.path.join(base_dir, "storage", "db", "in_memory_rag.json")
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
        if not os.path.exists(self.json_path):
            with open(self.json_path, 'w') as f:
                json.dump([], f)
        
        with open(self.json_path, 'r') as f:
            self.memory_data = json.load(f)

    def _save_in_memory(self):
        with open(self.json_path, 'w') as f:
            json.dump(self.memory_data, f, indent=2)

    def _get_full_name(self, collection: str, model_tag: str) -> str:
        """Sanitiza e concatena o nome da coleção com a tag do modelo."""
        name = collection or self.base_collection
        if model_tag:
            # Substitui hífen e pontos por underscore para compatibilidade com SQL
            tag = model_tag.lower().replace("-", "_").replace(".", "_")
            name = f"{name}_{tag}"
        return name

    def _init_pgvector_extension(self):
        try:
            from sqlalchemy import create_engine, text
            self.engine = create_engine(self.connection_string)
            with self.engine.begin() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        except Exception as e:
            logger.error(f"Failed to init pgvector: {e}. Fallback to in-memory.")
            self._init_in_memory()

    def _ensure_collection(self, full_name: str, dimension: int):
        """Lazy creation da tabela ou coleção no banco de dados específico."""
        if full_name in self.initialized_collections:
            return

        if self.db_type == "chroma":
            # O Chroma já lida com o 'get_or_create' de forma eficiente
            self.client.get_or_create_collection(
                name=full_name,
                metadata={"hnsw:space": "cosine"}
            )
        elif self.db_type == "pgvector":
            try:
                from sqlalchemy import text
                table_name = f"rag_{full_name}"
                with self.engine.begin() as conn:
                    # PGVector requer uma coluna com dimensão FIXA. 
                    # Por isso criamos uma tabela nova por modelo/coleção.
                    conn.execute(text(f"""
                        CREATE TABLE IF NOT EXISTS {table_name} (
                            id SERIAL PRIMARY KEY,
                            custom_id TEXT UNIQUE,
                            content TEXT,
                            embedding vector({dimension}),
                            metadata JSONB
                        )
                    """))
            except Exception as e:
                logger.error(f"Failed to ensure pgvector table {full_name}: {e}")
        
        self.initialized_collections.add(full_name)

    def add(self, text: str, embedding: List[float], metadata: Dict[str, Any] = None, collection: str = "july_memory", model_tag: str = None, doc_id: str = None):
        import uuid
        if not doc_id:
            doc_id = str(uuid.uuid4())
        
        full_name = self._get_full_name(collection, model_tag)
        dim = len(embedding)
        
        self._ensure_collection(full_name, dim)

        if self.db_type == "chroma":
            coll = self.client.get_collection(name=full_name)
            coll.upsert(
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata] if metadata else None,
                ids=[doc_id]
            )
        elif self.db_type == "pgvector":
            try:
                from sqlalchemy import text as sa_text
                import json
                table_name = f"rag_{full_name}"
                with self.engine.begin() as conn:
                    # Verifica se a tabela tem ID serial ou Text. Se for a primeira vez, criamos como text.
                    # Para simplificar o upsert com ID determinístico, usamos o metadata path ou o id provido.
                    emb_str = f"[{','.join(map(str, embedding))}]"
                    meta_str = json.dumps(metadata or {})
                    
                    # Tenta deletar se já existir (Upsert manual para compatibilidade SQL ampla)
                    conn.execute(sa_text(f"DELETE FROM {table_name} WHERE custom_id = :cid"), {"cid": doc_id})
                    
                    conn.execute(sa_text(f"""
                        INSERT INTO {table_name} (content, embedding, metadata, custom_id)
                        VALUES (:content, :embedding, :metadata, :cid)
                    """), {"content": text, "embedding": emb_str, "metadata": meta_str, "cid": doc_id})
            except Exception as e:
                logger.error(f"Error inserting into pgvector table {table_name}: {e}")
        else:
            # In-memory - Remove antigo se existir
            self.memory_data = [item for item in self.memory_data if not (item.get("id") == doc_id and item.get("collection") == full_name)]
            
            self.memory_data.append({
                "id": doc_id,
                "collection": full_name,
                "content": text,
                "embedding": embedding,
                "metadata": metadata or {}
            })
            self._save_in_memory()

    def search(self, query_embedding: List[float], top_k: int = 3, collection: str = "july_memory", model_tag: str = None, filter: Dict[str, Any] = None) -> List[str]:
        full_name = self._get_full_name(collection, model_tag)
        
        if self.db_type == "chroma":
            try:
                coll = self.client.get_collection(name=full_name)
                results = coll.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=filter
                )
                return results["documents"][0] if results["documents"] else []
            except Exception:
                return []
            
        elif self.db_type == "pgvector":
            try:
                from sqlalchemy import text as sa_text
                table_name = f"rag_{full_name}"
                with self.engine.connect() as conn:
                    emb_str = f"[{','.join(map(str, query_embedding))}]"
                    
                    sql = f"SELECT content FROM {table_name}"
                    params = {"limit": top_k}
                    
                    if filter:
                        where_clauses = []
                        for key, value in filter.items():
                            where_clauses.append(f"metadata->>'{key}' = :val_{key}")
                            params[f"val_{key}"] = str(value)
                        sql += " WHERE " + " AND ".join(where_clauses)
                    
                    sql += f" ORDER BY embedding <=> '{emb_str}' LIMIT :limit"
                    
                    result = conn.execute(sa_text(sql), params)
                    return [row[0] for row in result]
            except Exception as e:
                logger.error(f"Error searching pgvector table {full_name}: {e}")
                return []
                
        else:
            # In-memory filtered by full_name and metadata
            subset = [item for item in self.memory_data if item.get("collection") == full_name]
            
            if filter:
                new_subset = []
                for item in subset:
                    match = True
                    for k, v in filter.items():
                        if item.get("metadata", {}).get(k) != v:
                            match = False
                            break
                    if match:
                        new_subset.append(item)
                subset = new_subset

            if not subset:
                return []
                
            import math
            def cosine_similarity(v1, v2):
                dot = sum(a*b for a, b in zip(v1, v2))
                norm1 = math.sqrt(sum(a*a for a in v1))
                norm2 = math.sqrt(sum(b*b for b in v2))
                if norm1 == 0 or norm2 == 0: return 0
                return dot / (norm1 * norm2)

            scored = []
            for item in subset:
                score = cosine_similarity(query_embedding, item["embedding"])
                scored.append((score, item["content"]))
            
            scored.sort(key=lambda x: x[0], reverse=True)
            return [item[1] for item in scored[:top_k]]
        
    def search_with_details(self, query_embedding: List[float], top_k: int = 1, collection: str = "july_memory", model_tag: str = None, filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        full_name = self._get_full_name(collection, model_tag)
        
        if self.db_type == "chroma":
            try:
                coll = self.client.get_collection(name=full_name)
                results = coll.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=filter,
                    include=["embeddings", "metadatas", "distances", "documents"]
                )
                
                matches = []
                if results["ids"] and len(results["ids"][0]) > 0:
                    for i in range(len(results["ids"][0])):
                        matches.append({
                            "id": results["ids"][0][i],
                            "distance": float(results["distances"][0][i]) if "distances" in results and results["distances"] else 0.0,
                            "metadata": results["metadatas"][0][i] if "metadatas" in results and results["metadatas"] else {},
                            "content": results["documents"][0][i] if "documents" in results and results["documents"] else "",
                            "embedding": results["embeddings"][0][i].tolist() if hasattr(results["embeddings"][0][i], "tolist") else results["embeddings"][0][i]
                        })
                return matches
            except Exception:
                return []
            
        elif self.db_type == "pgvector":
            try:
                from sqlalchemy import text as sa_text
                table_name = f"rag_{full_name}"
                with self.engine.connect() as conn:
                    emb_str = f"[{','.join(map(str, query_embedding))}]"
                    
                    sql = f"SELECT id, embedding <=> '{emb_str}' AS distance, metadata, content, embedding FROM {table_name}"
                    params = {"limit": top_k}
                    
                    if filter:
                        where_clauses = []
                        for key, value in filter.items():
                            where_clauses.append(f"metadata->>'{key}' = :val_{key}")
                            params[f"val_{key}"] = str(value)
                        sql += " WHERE " + " AND ".join(where_clauses)
                        
                    sql += f" ORDER BY distance LIMIT :limit"
                    
                    result = conn.execute(sa_text(sql), params)
                    matches = []
                    for row in result:
                        matches.append({
                            "id": str(row[0]),
                            "distance": float(row[1]),
                            "metadata": row[2] if row[2] else {},
                            "content": row[3],
                            "embedding": list(row[4]) if row[4] else []
                        })
                    return matches
            except Exception as e:
                logger.error(f"Error searching details pgvector table {full_name}: {e}")
                return []
        else:
            # In-memory
            subset = [item for item in self.memory_data if item.get("collection") == full_name]
            
            if filter:
                new_subset = []
                for item in subset:
                    match = True
                    for k, v in filter.items():
                        if item.get("metadata", {}).get(k) != v:
                            match = False
                            break
                    if match:
                        new_subset.append(item)
                subset = new_subset

            matches = []
            if not subset:
                return matches
                
            import math
            def cosine_similarity(v1, v2):
                dot = sum(a*b for a, b in zip(v1, v2))
                norm1 = math.sqrt(sum(a*a for a in v1))
                norm2 = math.sqrt(sum(b*b for b in v2))
                if norm1 == 0 or norm2 == 0: return 0
                return dot / (norm1 * norm2)

            for item in subset:
                score = cosine_similarity(query_embedding, item["embedding"])
                matches.append({
                    "id": item["id"],
                    "distance": 1.0 - score,
                    "metadata": item.get("metadata", {}),
                    "content": item.get("content", ""),
                    "embedding": item["embedding"]
                })
            
            matches.sort(key=lambda x: x["distance"])
            return matches[:top_k]

    def update_embedding(self, doc_id: str, new_embedding: List[float], collection: str = "july_memory", model_tag: str = None, metadata: Dict[str, Any] = None):
        full_name = self._get_full_name(collection, model_tag)
        
        if self.db_type == "chroma":
            try:
                coll = self.client.get_collection(name=full_name)
                update_args = {"ids": [doc_id], "embeddings": [new_embedding]}
                if metadata:
                    update_args["metadatas"] = [metadata]
                coll.update(**update_args)
            except Exception as e:
                logger.error(f"Error updating ChromaDB collection {full_name}: {e}")

        elif self.db_type == "pgvector":
            try:
                from sqlalchemy import text as sa_text
                import json
                table_name = f"rag_{full_name}"
                with self.engine.begin() as conn:
                    emb_str = f"[{','.join(map(str, new_embedding))}]"
                    
                    if metadata:
                        conn.execute(
                            sa_text(f"UPDATE {table_name} SET embedding = :emb, metadata = :meta WHERE custom_id = :id OR id::text = :id"),
                            {"emb": emb_str, "meta": json.dumps(metadata), "id": doc_id}
                        )
                    else:
                        conn.execute(
                            sa_text(f"UPDATE {table_name} SET embedding = :emb WHERE custom_id = :id OR id::text = :id"),
                            {"emb": emb_str, "id": doc_id}
                        )
            except Exception as e:
                logger.error(f"Error updating pgvector table {table_name}: {e}")
        else:
            # In-memory
            for item in self.memory_data:
                if (item.get("id") == doc_id or item.get("custom_id") == doc_id) and item.get("collection") == full_name:
                    item["embedding"] = new_embedding
                    if metadata:
                        item["metadata"] = metadata
                    break
            self._save_in_memory()

    def delete(self, ids: List[str], collection: str = "july_memory", model_tag: str = None) -> int:
        """Deleta registros pelo ID (id ou custom_id). Retorna a contagem de deletados."""
        full_name = self._get_full_name(collection, model_tag)
        deleted_count = 0

        if self.db_type == "chroma":
            try:
                coll = self.client.get_collection(name=full_name)
                # O Chroma não retorna a contagem de deletes facilmente, então assumimos sucesso
                coll.delete(ids=ids)
                deleted_count = len(ids)
            except Exception as e:
                logger.error(f"Error deleting from ChromaDB {full_name}: {e}")

        elif self.db_type == "pgvector":
            try:
                from sqlalchemy import text as sa_text
                table_name = f"rag_{full_name}"
                with self.engine.begin() as conn:
                    result = conn.execute(
                        sa_text(f"DELETE FROM {table_name} WHERE custom_id IN :ids OR id::text IN :ids"),
                        {"ids": tuple(ids)}
                    )
                    deleted_count = result.rowcount
            except Exception as e:
                logger.error(f"Error deleting from pgvector {table_name}: {e}")

        else:
            # In-memory
            orig_len = len(self.memory_data)
            self.memory_data = [item for item in self.memory_data if not (item.get("id") in ids and item.get("collection") == full_name)]
            deleted_count = orig_len - len(self.memory_data)
            self._save_in_memory()

        return deleted_count

    def list_metadata(self, collection: str = "july_memory", model_tag: str = None, filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Lista todos os IDs e Metadados de uma coleção (sem vetores) com suporte a filtro."""
        full_name = self._get_full_name(collection, model_tag)
        
        if self.db_type == "chroma":
            try:
                coll = self.client.get_collection(name=full_name)
                # Buscamos tudo sem incluir o embedding
                results = coll.get(include=["metadatas", "documents"], where=filter)
                
                output = []
                for i in range(len(results["ids"])):
                    output.append({
                        "id": results["ids"][i],
                        "metadata": results["metadatas"][i] if results["metadatas"] else {},
                        "document": results["documents"][i] if results["documents"] and results["documents"][i] else ""
                    })
                return output
            except Exception as e:
                logger.error(f"Error listing ChromaDB metadata {full_name}: {e}")
                return []

        elif self.db_type == "pgvector":
            try:
                from sqlalchemy import text as sa_text
                table_name = f"rag_{full_name}"
                with self.engine.connect() as conn:
                    sql = f"SELECT id, custom_id, metadata, content FROM {table_name}"
                    params = {}
                    
                    if filter:
                        where_clauses = []
                        for key, value in filter.items():
                            where_clauses.append(f"metadata->>'{key}' = :val_{key}")
                            params[f"val_{key}"] = str(value)
                        sql += " WHERE " + " AND ".join(where_clauses)
                        
                    result = conn.execute(sa_text(sql), params)
                    output = []
                    for row in result:
                        output.append({
                            "id": str(row[0]),
                            "custom_id": row[1],
                            "metadata": row[2] if row[2] else {},
                            "document": row[3] if row[3] else ""
                        })
                    return output
            except Exception as e:
                logger.error(f"Error listing pgvector metadata {table_name}: {e}")
                return []
        else:
            # In-memory
            subset = [i for i in self.memory_data if i.get("collection") == full_name]
            if filter:
                new_subset = []
                for item in subset:
                    match = True
                    for k, v in filter.items():
                        if item.get("metadata", {}).get(k) != v:
                            match = False
                            break
                    if match:
                        new_subset.append(item)
                subset = new_subset
                
            return [
                {
                    "id": i.get("id"),
                    "metadata": i.get("metadata", {}),
                    "document": i.get("content", "")
                }
                for i in subset
            ]
vector_store = VectorStore()
