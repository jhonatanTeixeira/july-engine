import os
import json
import hashlib
import logging
import httpx
import io

logger = logging.getLogger("JulyEngine.Services.ResourceCalculator")


class ModelMetadata:
    def __init__(self, model_path, cache_dir="storage/cache", repo_id=None, filename=None, mmproj_path=None, mmproj_repo_id=None, mmproj_filename=None):
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.repo_id = repo_id
        self.filename = filename
        
        # Arquivo principal
        self.file_size_gb = os.path.getsize(model_path) / (1024**3) if os.path.exists(model_path) else 0
        
        # mmproj (Visão)
        self.mmproj_path = mmproj_path
        self.mmproj_repo_id = mmproj_repo_id
        self.mmproj_filename = mmproj_filename
        self.mmproj_size_gb = 0
        if mmproj_path and os.path.exists(mmproj_path):
            self.mmproj_size_gb = os.path.getsize(mmproj_path) / (1024**3)
        
        # Criar pasta de cache se não existir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Gerar um ID único baseado no caminho do arquivo para o nome do cache
        # Usamos MD5 para evitar problemas com caracteres especiais no nome do ficheiro
        self.cache_id = hashlib.md5(model_path.encode()).hexdigest()
        self.cache_file = os.path.join(self.cache_dir, f"{self.cache_id}.json")
        
        self._raw_metadata = {}
        self.is_rough_estimate = False
        self._load_metadata()

    async def _resolve(self):
        """
        Smart Resolver logic for finding metadata without downloading the full file.
        """
        repo_id = self.repo_id
        filename = self.filename

        # 2. Check Hugging Face Cache
        if repo_id and filename:
            url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
            logger.info(f"Metadata Resolver: {filename} not in cache. Performing remote scan (1MB).")
            self._raw_metadata = await self._scan_remote(url)
            
            # Fetch remote size if we don't have it
            if self._raw_metadata and not self.file_size_gb:
                self.file_size_gb = await self._get_remote_size(url)
        
        # Resolva mmproj remotamente se necessário
        if self.mmproj_repo_id and self.mmproj_filename and not self.mmproj_size_gb:
            mmproj_url = f"https://huggingface.co/{self.mmproj_repo_id}/resolve/main/{self.mmproj_filename}"
            self.mmproj_size_gb = await self._get_remote_size(mmproj_url)

    async def _scan_remote(self, url: str) -> dict:
        """
        Baixa apenas o início do arquivo GGUF (Partial Content) para extrair metadados.
        """
        metadata = {}
        # 1MB costuma ser suficiente, mas 2MB é mais seguro para modelos MoE complexos
        range_header = {"Range": "bytes=0-2097152"} 
        
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
                logger.info(f"🌐 Solicitando Range parcial de {url}...")
                response = await client.get(url, headers=range_header)
                
                if response.status_code not in [200, 206]:
                    logger.error(f"❌ Falha ao acessar Hugging Face: Status {response.status_code}")
                    return {}

                # Criamos um stream em memória a partir dos bytes baixados
                file_stream = io.BytesIO(response.content)
                
                # O GGUFReader consegue ler de um stream, mas como não temos o arquivo todo,
                # ele pode lançar um erro ao tentar ler tensores no fim do arquivo.
                # Capturamos apenas os campos (fields).
                reader = GGUFReader(None, mode="r") # Iniciamos sem arquivo
                
                # Gambiarra técnica necessária: alimentamos o reader com o nosso stream parcial
                reader.file_handler = file_stream
                reader._prepare_read() # Lê o cabeçalho e os campos
                
                for field in reader.fields.values():
                    parts = field.parts[field.data[0]]
                    
                    # Reaproveitamos a lógica de decodificação que já tens
                    if hasattr(parts, "__len__") and not isinstance(parts, (str, bytes)):
                        val = parts[0] if len(parts) > 0 else 0
                    elif isinstance(parts, bytes):
                        try: val = parts.decode('utf-8').strip('\x00')
                        except: val = list(parts)
                    else:
                        val = parts
                    
                    # Conversão NumPy/JSON
                    if hasattr(val, "item"): val = val.item()
                    
                    metadata[field.name] = val
                
                logger.info(f"✅ Metadados remotos extraídos com sucesso ({len(metadata)} chaves).")
                return metadata

        except Exception as e:
            logger.error(f"❌ Erro no scan remoto: {str(e)}")
            self.is_rough_estimate = True
            self._estimate_from_filename()
            return {}
    
    def _estimate_from_filename(self):
        """
        Tenta inferir metadados básicos pelo nome do arquivo como fallback.
        """
        import re
        name = (self.filename or self.model_path).lower()
        
        # 1. Tenta pegar a contagem de parâmetros (ex: 8b, 70b, 3.5b)
        params_match = re.search(r'([0-9.]+)b', name)
        params = float(params_match.group(1)) if params_match else 7.0 # Default 7B
        
        # 2. Tenta pegar a quantização
        quant = "q4_k_m" # Default equilibrado
        if "q8_0" in name: quant = "q8_0"
        elif "q4_0" in name: quant = "q4_0"
        elif "q5_k_m" in name: quant = "q5_k_m"
        elif "f16" in name: quant = "f16"
        
        # 3. Estima camadas (Regras de bolso)
        layers = 32
        if params > 50: layers = 80
        elif params > 20: layers = 60
        elif params < 4: layers = 24
        
        # 4. Estima Embedding Length
        embd = 4096
        if params > 50: embd = 8192
        elif params < 4: embd = 2048
        
        logger.info(f"⚠️ Metadata Resolver: Usando estimativa grosseira para {params}B ({layers} layers)")
        self._raw_metadata["general.architecture"] = "fuzzy_estimation"
        self._raw_metadata["general.parameter_count"] = params * 1e9
        self._raw_metadata["llama.block_count"] = layers
        self._raw_metadata["llama.embedding_length"] = embd
        self._raw_metadata["llama.attention.head_count"] = 32 if params < 50 else 64
        self._raw_metadata["llama.attention.head_count_kv"] = 8 # GQA padrão
        self.is_rough_estimate = True
    
    async def _get_remote_size(self, url: str) -> float:
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                res = await client.head(url)
                size = int(res.headers.get("Content-Length", 0))
                return size / (1024**3)
        except Exception as e:
            logger.warning(f"Could not fetch remote size for {url}: {e}")
            return 0

    def _load_metadata(self):
        """Tenta carregar do cache; se falhar, lê o GGUF e grava o cache."""
        if os.path.exists(self.cache_file):
            logger.debug(f"--- Carregando metadados do cache: {self.cache_file} ---")

            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self._raw_metadata = json.load(f)
                
                # Sanity check: if it's empty but parsed as valid JSON (like {}), 
                # we might still want to regenerate if it's missing core keys.
                if not self._raw_metadata:
                    logger.warning(f"⚠️ Cache {self.cache_file} está vazio. Regenerando...")
                    self._read_and_cache_gguf()
            except (json.JSONDecodeError, IOError, ValueError) as e:
                logger.warning(f"⚠️ Erro ao carregar cache {self.cache_file}: {str(e)}. Eliminando cache corrompido.")
                try:
                    os.remove(self.cache_file)
                except:
                    pass
                self._read_and_cache_gguf()
        else:
            logger.debug(f"--- Cache não encontrado. Lendo arquivo GGUF (Processo pesado)... ---")
            self._read_and_cache_gguf()

    def _read_and_cache_gguf(self):
        """Lê o GGUF binário e salva o resultado em JSON."""
        from gguf import GGUFReader

        if not os.path.exists(self.model_path) and self.repo_id and self.filename:
            self._resolve()
            
            return

        try:
            reader = GGUFReader(self.model_path)
        except Exception as e:
            logger.error(f"❌ Erro ao abrir arquivo GGUF: {str(e)}")
            return

        for field in reader.fields.values():
            parts = field.parts[field.data[0]]
            
            # Lógica de decodificação
            if hasattr(parts, "__len__") and not isinstance(parts, (str, bytes)):
                val = parts[0] if len(parts) > 0 else 0
            elif isinstance(parts, bytes):
                try:
                    val = parts.decode('utf-8').strip('\x00')
                except:
                    val = list(parts) # Se falhar, guarda como lista de números
            else:
                val = parts
            
            # --- Conversão para tipos JSON-serializáveis (uint32, float32, numpy, etc) ---
            if hasattr(val, "item"): 
                val = val.item() # Tipos NumPy
            elif isinstance(val, (int, float, str, bool, list, dict)) or val is None:
                pass # Tipos já compatíveis
            else:
                try:
                    # Tenta converter para int ou float se possível
                    if isinstance(val, (int, float)): 
                        val = val
                    else:
                        val = int(val) 
                except:
                    val = str(val) # Fallback seguro
            
            self._raw_metadata[field.name] = val
        
        # Gravar no cache para a próxima vez de forma atómica
        temp_file = self.cache_file + ".tmp"
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self._raw_metadata, f, indent=4)
            
            # No Windows, os.replace substitui o ficheiro se ele já existir (desde que não esteja aberto)
            os.replace(temp_file, self.cache_file)
            logger.debug(f"✅ Metadados guardados em cache para futuras leituras.")
        except Exception as e:
            logger.error(f"❌ Erro ao gravar cache: {str(e)}")
            if os.path.exists(temp_file):
                try: os.remove(temp_file)
                except: pass

    def _get_fuzzy(self, suffix, default=0):
        """Busca interna por sufixo (Fuzzy Search)."""
        for key, value in self._raw_metadata.items():
            if key.endswith(f".{suffix}") or key == suffix:
                return value
        return default

    # --- Propriedades (Getters) ---
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
        return int(self._get_fuzzy("context_length", 32768))

    @property
    def head_count(self): 
        return int(self._get_fuzzy("attention.head_count", 0))

    @property
    def head_count_kv(self):
        return int(self._get_fuzzy("attention.head_count_kv", self.head_count))
    
    @property
    def expert_count(self):
        """Retorna o número de experts (MoE). Retorna 0 se for um modelo denso."""
        return int(self._get_fuzzy("expert_count", 0))

    @property
    def expert_used_count(self):
        """Retorna quantos experts são usados por token."""
        return int(self._get_fuzzy("expert_used_count", 0))

    @property
    def is_moe(self):
        """Verifica se o modelo usa arquitetura Mixture of Experts."""
        return self.expert_count > 1

    def estimate_vram_gb(self, context_window=None, kv_cache_quantization='FP16', gpu_layers=None):
        """
        Calcula a VRAM necessária com base no número de camadas na GPU e quantização de kv cache.
        
        Args:
            gpu_layers (int): Quantidade de camadas a colocar na GPU. 
                              Se None ou > total, usa todas as camadas.
        """
        total_layers = self.block_count

        if gpu_layers is None or gpu_layers > total_layers:
            gpu_layers = total_layers
            
        ctx = parse_context_window(context_window) if isinstance(context_window, str) else (context_window or self.context_length)
        
        # 1. Proporção do Peso do Modelo na GPU
        # Nota: Reservamos cerca de 10% do peso para Embeddings/Header fixos
        fixed_weights_ratio = 0.10
        layer_weights_ratio = 1.0 - fixed_weights_ratio
        
        model_vram = self.file_size_gb * (fixed_weights_ratio + (layer_weights_ratio * (gpu_layers / total_layers)))

        # 2. KV Cache na GPU (Apenas para as camadas que estão na GPU)
        gqa_ratio = self.head_count_kv / self.head_count if self.head_count > 0 else 1.0
        quant_map = {'FP16': 2.0, 'Q8_0': 1.0, 'Q4_0': 0.5}
        bytes_per_element = quant_map.get(kv_cache_quantization.upper(), 2.0)
        
        # O KV Cache só ocupa VRAM para as layers que o motor processar na GPU
        kv_cache_bytes = 2 * gpu_layers * (self.embedding_length * gqa_ratio) * ctx * bytes_per_element * 2
        kv_cache_gb = kv_cache_bytes / (1024**3)

        return {
            "model_vram_gb": model_vram,
            "model_vram_mb": model_vram * 1024,
            "kv_cache_vram_gb": kv_cache_gb,
            "kv_cache_vram_mb": kv_cache_gb * 1024,
            "total_vram_gb": model_vram + kv_cache_gb + self.mmproj_size_gb,
            "total_vram_mb": (model_vram + kv_cache_gb + self.mmproj_size_gb) * 1024,
            "mmproj_size_gb": self.mmproj_size_gb,
            "offloaded_layers": gpu_layers,
            "total_layers": total_layers,
            "percent_on_gpu": (gpu_layers / total_layers) * 100,
            "is_rough_estimate": self.is_rough_estimate
        }


def parse_context_window(ctx_str: str) -> int:
    """Converte strings como '4k' para 4096 e '1m' para 1.048.576."""
    if isinstance(ctx_str, int):
        return ctx_str
        
    ctx_str = ctx_str.lower().strip()
    # Usamos 1024 para respeitar a arquitetura de memória (Base 2)
    multipliers = {
        'k': 1024, 
        'm': 1024 * 1024
    }
    
    if ctx_str[-1] in multipliers:
        unit = ctx_str[-1]
        number_part = ctx_str[:-1]
        try:
            # Aqui 4 * 1024 = 4096
            return int(float(number_part) * multipliers[unit])
        except ValueError:
            return 2048 
            
    return int(ctx_str)


def estimate_vram_ram(
    model_path: str,
    context_window: str | int = "2k",
    kv_cache_quantization: str = "FP16",
    gpu_layers: int = None,
    repo_id: str = None,
    filename: str = None,
    mmproj_path: str = None,
    mmproj_repo_id: str = None,
    mmproj_filename: str = None
):
    metadata = ModelMetadata(
        model_path, 
        repo_id=repo_id, 
        filename=filename,
        mmproj_path=mmproj_path,
        mmproj_repo_id=mmproj_repo_id,
        mmproj_filename=mmproj_filename
    )

    ctx_int = context_window

    if isinstance(context_window, str):
        ctx_int = parse_context_window(context_window)
    
    return metadata.estimate_vram_gb(
        context_window=ctx_int, 
        kv_cache_quantization=kv_cache_quantization,
        gpu_layers=gpu_layers
    )