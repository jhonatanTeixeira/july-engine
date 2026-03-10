import os
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("JulyEngine.VectorStore")

class VectorStore:
    def __init__(self):
        self.db_type = os.environ.get("RAG_DATABASE", "in-memory").lower()
        self.collection_name = "july_memory"
        
        if self.db_type == "chroma":
            try:
                import chromadb
                # Store locally in storage/db/chroma
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                db_path = os.path.join(base_dir, "storage", "db", "chroma")
                os.makedirs(db_path, exist_ok=True)
                self.client = chromadb.PersistentClient(path=db_path)
                self.collection = self.client.get_or_create_collection(name=self.collection_name)
            except ImportError:
                logger.error("ChromaDB not installed. Fallback to in-memory.")
                self.db_type = "in-memory"
                self._init_in_memory()
        elif self.db_type == "pgvector":
            # Just placeholder for PGVector connection setup
            # Requires psycopg2, pgvector, sqlalchemy
            self.connection_string = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/july_engine")
            self._init_pgvector()
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

    def _init_pgvector(self):
        try:
            from sqlalchemy import create_engine, text
            self.engine = create_engine(self.connection_string)
            with self.engine.begin() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS july_rag_memory (
                        id SERIAL PRIMARY KEY,
                        content TEXT,
                        embedding vector(1536) 
                    )
                """))
        except Exception as e:
            logger.error(f"Failed to init pgvector: {e}. Fallback to in-memory.")
            self._init_in_memory()

    def add(self, text: str, embedding: List[float], metadata: Dict[str, Any] = None):
        import uuid
        doc_id = str(uuid.uuid4())
        if self.db_type == "chroma":
            self.collection.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata or {}],
                ids=[doc_id]
            )
        elif self.db_type == "pgvector":
            try:
                from sqlalchemy import text
                with self.engine.begin() as conn:
                    # PGVector format [0.1, 0.2, ...]
                    emb_str = f"[{','.join(map(str, embedding))}]"
                    conn.execute(text("INSERT INTO july_rag_memory (content, embedding) VALUES (:content, :embedding)"),
                                 {"content": text, "embedding": emb_str})
            except Exception as e:
                logger.error(f"Error inserting into pgvector: {e}")
        else:
            self.memory_data.append({
                "id": doc_id,
                "content": text,
                "embedding": embedding,
                "metadata": metadata or {}
            })
            self._save_in_memory()

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[str]:
        if self.db_type == "chroma":
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            return results["documents"][0] if results["documents"] else []
            
        elif self.db_type == "pgvector":
            try:
                from sqlalchemy import text
                with self.engine.connect() as conn:
                    emb_str = f"[{','.join(map(str, query_embedding))}]"
                    # Cosine distance <=>
                    result = conn.execute(
                        text(f"SELECT content FROM july_rag_memory ORDER BY embedding <=> '{emb_str}' LIMIT :limit"),
                        {"limit": top_k}
                    )
                    return [row[0] for row in result]
            except Exception as e:
                logger.error(f"Error searching pgvector: {e}")
                return []
                
        else:
            # In-memory cosine similarity
            if not self.memory_data:
                return []
                
            import math
            def cosine_similarity(v1, v2):
                dot = sum(a*b for a, b in zip(v1, v2))
                norm1 = math.sqrt(sum(a*a for a in v1))
                norm2 = math.sqrt(sum(b*b for b in v2))
                if norm1 == 0 or norm2 == 0: return 0
                return dot / (norm1 * norm2)

            scored = []
            for item in self.memory_data:
                score = cosine_similarity(query_embedding, item["embedding"])
                scored.append((score, item["content"]))
            
            scored.sort(key=lambda x: x[0], reverse=True)
            return [item[1] for item in scored[:top_k]]

vector_store = VectorStore()
