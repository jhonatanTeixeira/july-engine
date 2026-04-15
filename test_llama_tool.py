import os
import argparse
from huggingface_hub import hf_hub_download
from gguf import GGUFReader

class ModelMetadata:
    def __init__(self, model_path):
        self.model_path = model_path
        self.file_size_gb = os.path.getsize(model_path) / (1024**3)
        self._reader = GGUFReader(model_path)
        self._raw_metadata = {}
        self._decode_all()

    def _decode_all(self):
        """Traduz todos os metadados binários para um dicionário limpo internamente."""
        for field in self._reader.fields.values():
            parts = field.parts[field.data[0]]
            
            # Tratamento de listas/arrays numéricos
            if hasattr(parts, "__len__") and not isinstance(parts, (str, bytes)):
                val = parts[0] if len(parts) > 0 else 0
            # Tratamento de bytes para strings
            elif isinstance(parts, bytes):
                try:
                    val = parts.decode('utf-8').strip('\x00')
                except:
                    val = parts
            else:
                val = parts
            
            self._raw_metadata[field.name] = val

    def _get_fuzzy(self, suffix, default=0):
        """Busca interna por sufixo (Fuzzy Search)."""
        for key, value in self._raw_metadata.items():
            if key.endswith(f".{suffix}") or key == suffix:
                return value
        return default

    # --- Propriedades do Modelo ---

    @property
    def architecture(self):
        return self._raw_metadata.get("general.architecture", "unknown")

    @property
    def block_count(self):
        return int(self._get_fuzzy("block_count", 0))

    @property
    def embedding_length(self):
        return int(self._get_fuzzy("embedding_length", 0))

    @property
    def context_length(self):
        # Fallback comum de 32k para modelos modernos como Qwen 2.5
        return int(self._get_fuzzy("context_length", 0))

    @property
    def head_count(self):
        return int(self._get_fuzzy("attention.head_count", 0))

    @property
    def head_count_kv(self):
        # Se não houver KV heads, assume-se que é igual ao head_count (MHA)
        return int(self._get_fuzzy("attention.head_count_kv", self.head_count))

    @property
    def gqa_ratio(self):
        if self.head_count > 0:
            return self.head_count_kv / self.head_count
        return 1.0

    # --- Cálculos de Memória ---

    def estimate_vram_gb(self, context_window=None):
        """
        Calcula a estimativa total de VRAM.
        Permite sobrescrever a janela de contexto para testes.
        """
        ctx = context_window if context_window is not None else self.context_length
        
        # KV Cache: 2 (K e V) * camadas * (embed * ratio) * janela * 2 bytes * 2
        kv_cache_bytes = 2 * self.block_count * (self.embedding_length * self.gqa_ratio) * ctx * 2
        kv_cache_gb = kv_cache_bytes / (1024**3)
        
        return {
            "model_weight": self.file_size_gb,
            "kv_cache": kv_cache_gb,
            "total": self.file_size_gb + kv_cache_gb
        }

# --- Exemplo de Uso do Script ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--filename", required=True)
    args = parser.parse_args()

    # 1. Obter o path
    path = hf_hub_download(repo_id=args.repo_id, filename=args.filename)

    # 2. Instanciar a classe (Onde a "mágica" acontece)
    metadata = ModelMetadata(path)

    # 3. Aceder aos dados de forma limpa e elegante
    vram = metadata.estimate_vram_gb()

    print(f"\nANÁLISE OOP DO MODELO:")
    print(f"Arquitetura: {metadata.architecture}")
    print(f"Camadas:     {metadata.block_count}")
    print(f"Embedding:   {metadata.embedding_length}")
    print(f"Contexto:    {metadata.context_length} tokens")
    print("-" * 30)
    print(f"VRAM Modelo: {vram['model_weight']:.2f} GB")
    print(f"VRAM Cache:  {vram['kv_cache']:.2f} GB")
    print(f"TOTAL:       {vram['total']:.2f} GB")