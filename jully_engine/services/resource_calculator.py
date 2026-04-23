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
            # Resolve será chamado explicitamente pelo estimate_vram_ram (é async)
            return

        try:
            logger.info(f"📂 Opening GGUF file: {self.model_path} ({self.file_size_gb:.2f} GB)")
            reader = GGUFReader(self.model_path)
        except Exception as e:
            logger.error(f"❌ Erro ao abrir arquivo GGUF: {str(e)}")
            return

        for field in reader.fields.values():
            try:
                # Detecta o tamanho dos dados do campo
                data_len = field.data.shape[0] if hasattr(field.data, 'shape') else len(field.data)

                # Se for uma lista grande (> 128 elementos), guardamos apenas o tamanho
                if data_len > 128:
                    self._raw_metadata[f"{field.name}.length"] = int(data_len)
                    continue

                if data_len == 0:
                    continue

                parts = field.parts[field.data[0]]

                # Lógica de decodificação de tipos simples
                if hasattr(parts, "__len__") and not isinstance(parts, (str, bytes)):
                    val = parts[0] if len(parts) > 0 else 0
                elif isinstance(parts, bytes):
                    try:
                        val = parts.decode('utf-8').strip('\x00')
                    except:
                        val = list(parts)
                else:
                    val = parts

                if hasattr(val, "item"):
                    val = val.item()

                # Garante que o valor seja serializável em JSON
                if not isinstance(val, (int, float, str, bool, list, dict)) and val is not None:
                    val = str(val)

                self._raw_metadata[field.name] = val

            except Exception as e:
                logger.warning(f"⚠️ Could not process field {field.name}: {str(e)}")
                continue

        logger.info(f"✅ GGUF Metadata Extracted: {len(self._raw_metadata)} fields captured.")

        temp_file = self.cache_file + ".tmp"
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self._raw_metadata, f, indent=4)
            os.replace(temp_file, self.cache_file)
            logger.info(f"✅ Cache generated: {self.cache_file}")
        except Exception as e:
            logger.error(f"❌ Erro ao gravar cache: {str(e)}")
            for k, v in self._raw_metadata.items():
                try:
                    json.dumps({k: v})
                except:
                    logger.error(f"   - Problematic key: {k} (Type: {type(v)})")
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

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
        """Número total de experts no modelo MoE. 0 = modelo denso."""
        return int(self._get_fuzzy("expert_count", 0))

    @property
    def expert_used_count(self):
        """Quantos experts são ativados por token."""
        return int(self._get_fuzzy("expert_used_count", 0))

    @property
    def is_moe(self):
        """True se o modelo usa arquitetura Mixture of Experts."""
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
        """Tamanho da janela deslizante em tokens. 0 = sem SWA."""
        return int(self._get_fuzzy("attention.sliding_window", 0))

    @property
    def shared_kv_layers(self) -> int:
        """Layers que reutilizam KV Cache de outra layer (Cross-Layer KV Sharing)."""
        return int(self._get_fuzzy("attention.shared_kv_layers", 0))

    def _effective_model_size_gb(self) -> float:
        """
        Retorna o tamanho efetivo do modelo para fins de cálculo de VRAM.

        Para modelos DENSOS: retorna file_size_gb sem alteração.
        O peso do arquivo corresponde diretamente ao peso que precisa estar
        na VRAM (proporcional às layers offloadas).

        Para modelos MoE: apenas os experts ATIVOS precisam estar "quentes"
        na VRAM a qualquer momento. Os experts inativos ficam na RAM e são
        carregados sob demanda pelo llama.cpp.

        O modelo é dividido conceitualmente em duas partes:
          - Densa  (~35% do arquivo): attention, embeddings, layer norms.
            Sempre precisa estar na VRAM para as layers offloadas.
          - MoE    (~65% do arquivo): os FFN experts.
            Apenas a fração ativa (expert_used / expert_count) precisa
            estar na VRAM simultaneamente.

        A proporção dense_fraction=0.35 é uma heurística calibrada
        empiricamente para arquiteturas MoE comuns (Mixtral, Qwen MoE,
        DeepSeek MoE). Modelos densos nunca chegam a este método.
        """
        if not self.is_moe or self.expert_count <= 0 or self.expert_used_count <= 0:
            # Modelo denso: sem ajuste — retorna tamanho real do arquivo
            return self.file_size_gb

        active_ratio = self.expert_used_count / self.expert_count

        # Fração do arquivo que corresponde a pesos densos (attention, embed, norms)
        dense_fraction = 0.35
        moe_fraction   = 1.0 - dense_fraction

        effective_size = self.file_size_gb * (dense_fraction + moe_fraction * active_ratio)

        logger.debug(
            f"🧮 MoE effective size: {self.file_size_gb:.2f} GB → {effective_size:.2f} GB "
            f"(experts {self.expert_used_count}/{self.expert_count}, "
            f"active_ratio={active_ratio:.4f})"
        )

        return effective_size

    def _estimate_kv_cache_gb(self, ctx: int, kv_cache_quantization: str, gpu_layers: int) -> float:
        """
        Calcula o KV Cache respeitando arquiteturas híbridas:
          - key_length / value_length explícitos no GGUF
          - Sliding Window Attention (SWA) vs Global Attention
          - Cross-Layer KV Sharing (shared_kv_layers)

        NOTA IMPORTANTE: MoE NÃO afeta o KV Cache.
        A atenção é sempre densa em arquiteturas MoE — apenas o FFN é esparso.
        Por isso este método não faz nenhum ajuste para MoE.
        """
        quant_map = {
            'FP16':   2.0,
            'Q8_0':   1.0,
            'Q4_0':   0.5,
            'Q4_1':   0.5625,
            'Q5_0':   0.625,
            'Q5_1':   0.6875,
            'Q5_K_M': 0.625,
        }
        bytes_per_element = quant_map.get(kv_cache_quantization.upper(), 2.0)
        total_layers = self.block_count

        # --- Dimensões explícitas de K/V do GGUF ---
        # Mais confiáveis que derivar de embedding_length / head_count,
        # especialmente em arquiteturas com GQA ou dimensões não-padrão.
        head_dim_fallback = self.embedding_length // self.head_count if self.head_count > 0 else 128

        key_len_global = int(self._get_fuzzy("attention.key_length",       head_dim_fallback))
        val_len_global = int(self._get_fuzzy("attention.value_length",     head_dim_fallback))
        key_len_swa    = int(self._get_fuzzy("attention.key_length_swa",   key_len_global))
        val_len_swa    = int(self._get_fuzzy("attention.value_length_swa", val_len_global))

        # --- Sliding Window Attention ---
        swa_ctx     = self.sliding_window   # 0 = sem SWA
        has_swa     = swa_ctx > 0
        swa_pattern = bool(self._get_fuzzy("attention.sliding_window_pattern", False))

        # --- Cross-Layer KV Sharing ---
        # Layers com KV compartilhado não alocam buffer próprio.
        shared             = self.shared_kv_layers
        layers_with_own_kv = max(total_layers - shared, 1)

        # Proporção das layers com KV próprio que estão na GPU
        gpu_kv_layers = round(layers_with_own_kv * (gpu_layers / max(total_layers, 1)))

        # --- Proporção de layers Global vs SWA ---
        if has_swa and swa_pattern:
            # Heurística conservadora para padrão alternado:
            # ~20% global, ~80% SWA. Calibrado para Gemma 4 (1 global a cada 6).
            # Para outros modelos com padrão alternado diferente, ainda é
            # uma estimativa razoável (tende a subestimar levemente).
            global_ratio = 0.20
        elif has_swa:
            # SWA presente mas sem padrão explícito: assume 50/50
            global_ratio = 0.50
        else:
            # Modelo sem SWA (LLaMA, Qwen, Mistral denso, etc): 100% global
            global_ratio = 1.0

        global_layers_gpu = round(gpu_kv_layers * global_ratio)
        swa_layers_gpu    = gpu_kv_layers - global_layers_gpu

        # --- KV Cache para layers de atenção global (ctx completo) ---
        kv_global = (
            2                                           # K + V
            * self.head_count_kv                        # cabeças KV (GQA)
            * ((key_len_global + val_len_global) / 2)   # média das dimensões K e V
            * ctx                                       # tokens no contexto
            * bytes_per_element
            * global_layers_gpu
        )

        # --- KV Cache para layers SWA (ctx limitado pela janela deslizante) ---
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
        Estima a VRAM necessária para rodar o modelo no llama.cpp.

        Args:
            context_window (int | str): Janela de contexto. Ex: 4096, "4k", "128k".
            kv_cache_quantization (str): Quantização do KV Cache. Ex: 'FP16', 'Q8_0', 'Q4_0'.
            gpu_layers (int): Layers a colocar na GPU. None = todas as layers.

        Returns:
            dict: Breakdown detalhado do uso de VRAM em GB e MB.
        """
        total_layers = self.block_count
        if total_layers <= 0:
            total_layers = 1  # Proteção contra ZeroDivisionError

        if gpu_layers is None or gpu_layers > total_layers:
            gpu_layers = total_layers

        ctx = (
            parse_context_window(context_window)
            if isinstance(context_window, str)
            else (context_window or self.context_length)
        )

        # ── 1. Peso do modelo na GPU ─────────────────────────────────────────
        # Para MoE: usa tamanho efetivo (só experts ativos).
        # Para densos: usa file_size_gb diretamente.
        # A proporção gpu_layers/total_layers determina quanto vai pra VRAM.
        effective_file_size = self._effective_model_size_gb()
        fixed_weights_ratio = 0.10   # ~10% do peso é fixo (embeddings, norms)
        layer_weights_ratio = 1.0 - fixed_weights_ratio
        model_vram = effective_file_size * (
            fixed_weights_ratio + layer_weights_ratio * (gpu_layers / total_layers)
        )

        # ── 2. KV Cache ──────────────────────────────────────────────────────
        # Suporta SWA, Cross-Layer KV Sharing e dimensões K/V explícitas.
        # MoE não afeta o KV Cache (atenção é sempre densa).
        kv_cache_gb = self._estimate_kv_cache_gb(ctx, kv_cache_quantization, gpu_layers)

        # ── 3. Compute buffer do grafo CUDA ──────────────────────────────────
        # Alocação estática do llama.cpp para buffers intermediários de
        # computação (ativações temporárias, workspace da operação GEMM, etc).
        # Escala com embedding_length e batch_size.
        batch_size = 512  # n_batch default do llama.cpp
        compute_buffer_gb = (4 * batch_size * self.embedding_length * 4) / (1024**3)

        # ── 4. Logits buffer ─────────────────────────────────────────────────
        # Buffer de saída para os logits de todo o vocabulário por token do batch.
        # CRÍTICO para modelos com vocabulário grande:
        #   Gemma 4: 256k tokens → ~0.49 GB
        #   Qwen3:   152k tokens → ~0.29 GB
        #   LLaMA 3:  128k tokens → ~0.25 GB
        #   LLaMA 2:   32k tokens → ~0.06 GB
        vocab_size = self.vocab_size
        logits_buffer_gb = (vocab_size * batch_size * 4) / (1024**3)

        # ── 5. Overhead fixo CUDA ────────────────────────────────────────────
        # Contexto CUDA, streams, alocador de memória.
        # Relativamente constante independente do modelo (~100-150 MB).
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
            # Breakdown por componente
            "model_vram_gb":           model_vram,
            "model_vram_mb":           model_vram * 1024,
            "kv_cache_vram_gb":        kv_cache_gb,
            "kv_cache_vram_mb":        kv_cache_gb * 1024,
            "compute_buffer_gb":       compute_buffer_gb,
            "compute_buffer_mb":       compute_buffer_gb * 1024,
            "logits_buffer_gb":        logits_buffer_gb,
            "logits_buffer_mb":        logits_buffer_gb * 1024,
            "cuda_overhead_gb":        cuda_overhead_gb,
            "cuda_overhead_mb":        cuda_overhead_gb * 1024,
            "mmproj_size_gb":          self.mmproj_size_gb,
            # Total
            "total_vram_gb":           total,
            "total_vram_mb":           total * 1024,
            # Info de layers
            "offloaded_layers":        gpu_layers,
            "total_layers":            total_layers,
            "percent_on_gpu":          (gpu_layers / total_layers) * 100,
            # Info de arquitetura (para debug / UI)
            "vocab_size":              vocab_size,
            "has_swa":                 self.sliding_window > 0,
            "shared_kv_layers":        self.shared_kv_layers,
            "is_moe":                  self.is_moe,
            "expert_count":            self.expert_count,
            "expert_used_count":       self.expert_used_count,
            "effective_model_size_gb": effective_file_size,
            "is_rough_estimate":       self.is_rough_estimate,
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


async def estimate_vram_ram(
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
    logger.info(f"🔍 Checking if model exists: {model_path}")

    exists_locally = os.path.exists(model_path) if model_path and model_path != "model" else False

    if not exists_locally and filename:
        # 1. Busca recursiva em models/
        found_path = None
        models_dir = "models"
        if os.path.exists(models_dir):
            for root, dirs, files in os.walk(models_dir):
                if filename in files:
                    found_path = os.path.join(root, filename)
                    break

        if found_path:
            logger.info(f"📍 Model found in local fallback: {found_path}")
            model_path = found_path
            exists_locally = True

        # 2. Busca no cache do Hugging Face Hub
        if not exists_locally and repo_id:
            try:
                from huggingface_hub import hf_hub_download
                hf_path = hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True)
                if hf_path and os.path.exists(hf_path):
                    logger.info(f"📍 Model found in HF Hub Cache: {hf_path}")
                    model_path = hf_path
                    exists_locally = True
            except Exception as e:
                logger.debug(f"HF Hub Cache check skipped: {e}")

    metadata = ModelMetadata(
        model_path,
        repo_id=repo_id,
        filename=filename,
        mmproj_path=mmproj_path,
        mmproj_repo_id=mmproj_repo_id,
        mmproj_filename=mmproj_filename
    )

    # Resolve remotamente se o modelo não existe localmente
    if not exists_locally and repo_id and filename:
        logger.info(f"🌐 Model not found locally. Proceeding with remote resolution for: {filename}")
        await metadata._resolve()

    ctx_int = context_window
    if isinstance(context_window, str):
        ctx_int = parse_context_window(context_window)

    return metadata.estimate_vram_gb(
        context_window=ctx_int,
        kv_cache_quantization=kv_cache_quantization,
        gpu_layers=gpu_layers
    )