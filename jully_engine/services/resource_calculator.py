import os
import json
import hashlib
import logging
import httpx

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
        self.cache_id = hashlib.md5(model_path.encode()).hexdigest()
        self.cache_file = os.path.join(self.cache_dir, f"{self.cache_id}.json")

        self._raw_metadata = {}
        self.is_rough_estimate = False
        self._load_metadata()

    async def _resolve(self):
        """
        Lógica para estimar metadados quando o arquivo não está em disco.
        """
        repo_id = self.repo_id
        filename = self.filename

        if repo_id and filename:
            url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
            self._estimate_from_filename()

            if not self.file_size_gb:
                self.file_size_gb = await self._get_remote_size(url)

        if self.mmproj_repo_id and self.mmproj_filename and not self.mmproj_size_gb:
            mmproj_url = f"https://huggingface.co/{self.mmproj_repo_id}/resolve/main/{self.mmproj_filename}"
            self.mmproj_size_gb = await self._get_remote_size(mmproj_url)

    def _estimate_from_filename(self):
        """
        Tenta inferir metadados básicos pelo nome do arquivo como fallback.
        """
        import re
        name = (self.filename or self.model_path).lower()

        params_match = re.search(r'([0-9.]+)b', name)
        params = float(params_match.group(1)) if params_match else 7.0

        layers = 32
        if params > 50: layers = 80
        elif params > 20: layers = 60
        elif params < 4: layers = 24

        embd = 4096
        if params > 50: embd = 8192
        elif params < 4: embd = 2048

        logger.info(f"⚠️ Metadata Resolver: Usando estimativa grosseira para {params}B ({layers} layers)")
        self._raw_metadata["general.architecture"] = "fuzzy_estimation"
        self._raw_metadata["general.parameter_count"] = params * 1e9
        self._raw_metadata["llama.block_count"] = layers
        self._raw_metadata["llama.embedding_length"] = embd
        self._raw_metadata["llama.attention.head_count"] = 32 if params < 50 else 64
        self._raw_metadata["llama.attention.head_count_kv"] = 8
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
            try:
                # 1. Detecta o tamanho dos dados do campo
                data_len = field.data.shape[0] if hasattr(field.data, 'shape') else len(field.data)
                
                # 2. Se for uma lista grande (> 128 elementos), guardamos apenas o tamanho
                if data_len > 128:
                    self._raw_metadata[f"{field.name}.length"] = int(data_len)
                    logger.debug(f"📦 Field {field.name} is a large list ({data_len} items). Stored as length.")
                    continue

                if data_len == 0:
                    continue

                parts = field.parts[field.data[0]]
                
                # Lógica de decodificação de tipos simples
                if hasattr(parts, "__len__") and not isinstance(parts, (str, bytes)):
                    val = parts[0] if len(parts) > 0 else 0
                elif isinstance(parts, bytes):
                    try: val = parts.decode('utf-8').strip('\x00')
                    except: val = list(parts)
                else:
                    val = parts

                if hasattr(val, "item"): val = val.item()
                elif isinstance(val, (int, float, str, bool, list, dict)) or val is None:
                    pass
                else:
                    try: val = int(val)
                    except: val = str(val)
                
                self._raw_metadata[field.name] = val
                logger.debug(f"✅ Processed field: {field.name}")

            except Exception as e:
                logger.warning(f"⚠️ Could not process field {field.name}: {str(e)}")
                continue
        
        logger.info(f"✅ GGUF Metadata Extracted: {len(self._raw_metadata)} fields captured.")

        temp_file = self.cache_file + ".tmp"
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self._raw_metadata, f, indent=4)
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

    # --- Propriedades ---

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
        return int(self._get_fuzzy("expert_count", 0))

    @property
    def expert_used_count(self):
        return int(self._get_fuzzy("expert_used_count", 0))

    @property
    def is_moe(self):
        return self.expert_count > 1

    @property
    def vocab_size(self) -> int:
        """
        Tamanho do vocabulário.
        Prioridade: tokens.length (contagem real) > vocab_size explícito > fallback.
        """
        v = self._raw_metadata.get("tokenizer.ggml.tokens.length", 0)
        if v: return int(v)
        v = self._get_fuzzy("vocab_size", 0)
        if v: return int(v)
        return 32000  # fallback conservador

    @property
    def sliding_window(self) -> int:
        """Tamanho da janela deslizante (0 = sem SWA)."""
        return int(self._get_fuzzy("attention.sliding_window", 0))

    @property
    def shared_kv_layers(self) -> int:
        """Layers que reutilizam o KV Cache de outra layer (Cross-Layer KV Sharing)."""
        return int(self._get_fuzzy("attention.shared_kv_layers", 0))

    def _estimate_kv_cache_gb(self, ctx: int, kv_cache_quantization: str, gpu_layers: int) -> float:
        """
        Calcula o KV Cache respeitando arquiteturas híbridas:
          - key_length / value_length explícitos no GGUF
          - Sliding Window Attention (SWA) vs Global Attention
          - Cross-Layer KV Sharing (shared_kv_layers)
        """
        quant_map = {'FP16': 2.0, 'Q8_0': 1.0, 'Q4_0': 0.5, 'Q5_0': 0.625, 'Q5_K_M': 0.625}
        bytes_per_element = quant_map.get(kv_cache_quantization.upper(), 2.0)
        total_layers = self.block_count

        # --- Dimensões de K/V (lê do GGUF se disponível) ---
        head_dim_fallback = self.embedding_length // self.head_count if self.head_count > 0 else 128

        key_len_global = int(self._get_fuzzy("attention.key_length",     head_dim_fallback))
        val_len_global = int(self._get_fuzzy("attention.value_length",   head_dim_fallback))
        key_len_swa    = int(self._get_fuzzy("attention.key_length_swa", key_len_global))
        val_len_swa    = int(self._get_fuzzy("attention.value_length_swa", val_len_global))

        # --- Sliding Window ---
        swa_ctx   = self.sliding_window          # 0 = sem SWA
        has_swa   = swa_ctx > 0
        swa_pattern = bool(self._get_fuzzy("attention.sliding_window_pattern", False))

        # --- Cross-Layer KV Sharing ---
        # Layers com KV compartilhado NÃO alocam buffer próprio
        shared = self.shared_kv_layers
        layers_with_own_kv = max(total_layers - shared, 1)

        # Proporção das layers com KV próprio que caem na GPU
        gpu_kv_layers = round(layers_with_own_kv * (gpu_layers / max(total_layers, 1)))

        # --- Divisão Global vs SWA ---
        if has_swa and swa_pattern:
            # Heurística genérica conservadora: ~20% global, ~80% SWA
            # (Para Gemma 4 o padrão real é 1 global a cada 6 layers)
            global_ratio = 0.20
        elif has_swa:
            global_ratio = 0.50
        else:
            global_ratio = 1.0

        global_layers_gpu = round(gpu_kv_layers * global_ratio)
        swa_layers_gpu    = gpu_kv_layers - global_layers_gpu

        # --- KV Cache Global (ctx completo) ---
        kv_global = (
            2                                       # K + V
            * self.head_count_kv
            * ((key_len_global + val_len_global) / 2)  # média dim K/V
            * ctx
            * bytes_per_element
            * global_layers_gpu
        )

        # --- KV Cache SWA (limitado pela janela deslizante) ---
        effective_swa_ctx = min(ctx, swa_ctx) if swa_ctx > 0 else ctx
        kv_swa = (
            2
            * self.head_count_kv
            * ((key_len_swa + val_len_swa) / 2)
            * effective_swa_ctx
            * bytes_per_element
            * swa_layers_gpu
        )

        return (kv_global + kv_swa) / (1024**3)

    def estimate_vram_gb(self, context_window=None, kv_cache_quantization='FP16', gpu_layers=None):
        """
        Estima a VRAM necessária para rodar o modelo.
        """
        total_layers = self.block_count
        if total_layers <= 0:
            total_layers = 1 # Proteção contra ZeroDivisionError
            
        if gpu_layers is None or gpu_layers > total_layers:
            gpu_layers = total_layers

        ctx = parse_context_window(context_window) if isinstance(context_window, str) else (context_window or self.context_length)

        # 1. Peso do modelo na GPU
        fixed_weights_ratio = 0.10
        layer_weights_ratio = 1.0 - fixed_weights_ratio
        model_vram = self.file_size_gb * (fixed_weights_ratio + (layer_weights_ratio * (gpu_layers / total_layers)))

        # 2. KV Cache (com suporte a SWA, shared KV e key_length explícito)
        kv_cache_gb = self._estimate_kv_cache_gb(ctx, kv_cache_quantization, gpu_layers)

        # 3. Compute buffer do grafo CUDA
        batch_size = 512  # n_batch default do llama.cpp
        compute_buffer_gb = (4 * batch_size * self.embedding_length * 4) / (1024**3)

        # 4. Logits buffer: vocab_size * batch_size * float32
        # CRÍTICO para modelos com vocabulário grande (ex: Gemma 4 tem 256k tokens)
        vocab_size = self.vocab_size
        logits_buffer_gb = (vocab_size * batch_size * 4) / (1024**3)

        # 5. Overhead fixo CUDA (contexto, streams, alocador CUDA)
        cuda_overhead_gb = 0.12

        total = (
            model_vram
            + kv_cache_gb
            + compute_buffer_gb
            + logits_buffer_gb
            + cuda_overhead_gb
            + self.mmproj_size_gb
        )

        return {
            "model_vram_gb":      model_vram,
            "model_vram_mb":      model_vram * 1024,
            "kv_cache_vram_gb":   kv_cache_gb,
            "kv_cache_vram_mb":   kv_cache_gb * 1024,
            "compute_buffer_gb":  compute_buffer_gb,
            "compute_buffer_mb":  compute_buffer_gb * 1024,
            "logits_buffer_gb":   logits_buffer_gb,
            "logits_buffer_mb":   logits_buffer_gb * 1024,
            "cuda_overhead_gb":   cuda_overhead_gb,
            "cuda_overhead_mb":   cuda_overhead_gb * 1024,
            "mmproj_size_gb":     self.mmproj_size_gb,
            "total_vram_gb":      total,
            "total_vram_mb":      total * 1024,
            "offloaded_layers":   gpu_layers,
            "total_layers":       total_layers,
            "percent_on_gpu":     (gpu_layers / total_layers) * 100,
            "vocab_size":         vocab_size,
            "has_swa":            self.sliding_window > 0,
            "shared_kv_layers":   self.shared_kv_layers,
            "is_rough_estimate":  self.is_rough_estimate,
        }


def parse_context_window(ctx_str: str) -> int:
    """Converte strings como '4k' → 4096 e '1m' → 1.048.576."""
    if isinstance(ctx_str, int):
        return ctx_str

    ctx_str = ctx_str.lower().strip()
    multipliers = {'k': 1024, 'm': 1024 * 1024}

    if ctx_str[-1] in multipliers:
        unit = ctx_str[-1]
        number_part = ctx_str[:-1]
        try:
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