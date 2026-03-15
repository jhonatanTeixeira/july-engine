import os
import csv
import logging
import numpy as np
import onnxruntime as ort
from PIL import Image
from typing import Dict, Any, List
from huggingface_hub import hf_hub_download

logger = logging.getLogger("JulyEngine.Domain.Tagger")

class ONNXTagger:
    """
    ONNX Image Tagger (CPU bound)
    Baixa os modelos WD14 "on the fly" direto do HuggingFace.
    """
    def __init__(self, repo_id: str = "SmilingWolf/wd-swinv2-tagger-v3", threshold: float = 0.35):
        self.repo_id = repo_id
        self.threshold = threshold
        self.cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
        
        self.session = None
        self.tags_names: List[str] = []
        self.target_size = 448 
        
    def load(self):
        if self.is_loaded():
            return
            
        try:
            logger.info(f"ONNXTagger: Baixando/Verificando cache do modelo {self.repo_id}...")
            
            # 1. Download On The Fly (Usa cache se já existir)
            model_path = hf_hub_download(
                repo_id=self.repo_id, 
                filename="model.onnx", 
                cache_dir=self.cache_dir
            )
            
            tags_path = hf_hub_download(
                repo_id=self.repo_id, 
                filename="selected_tags.csv", 
                cache_dir=self.cache_dir
            )

            logger.info("ONNXTagger: Carregando modelo ONNX na CPU...")
            # Força uso da CPU para não roubar VRAM do LLM principal
            providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            
            # 2. Auto-descobre a resolução esperada (WD14 v3 usa 448, V2 usa 448, etc)
            input_shape = self.session.get_inputs()[0].shape
            if len(input_shape) == 4 and input_shape[1] > 0:
                self.target_size = input_shape[1]

            # 3. Carrega o dicionário de tags
            self._load_tags_csv(tags_path)
            
            logger.info("ONNXTagger: Carregamento concluído com sucesso!")
            
        except Exception as e:
            logger.error(f"ONNXTagger: Falha ao carregar modelo: {e}")
            raise e

    def _load_tags_csv(self, tags_path: str):
        """Lê o CSV baixado e extrai apenas a coluna de nomes"""
        self.tags_names = []
        try:
            with open(tags_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                
                # Procura a coluna "name" (geralmente índice 1)
                name_idx = header.index("name") if "name" in header else 1
                
                for row in reader:
                    self.tags_names.append(row[name_idx])
        except Exception as e:
            logger.error(f"ONNXTagger: Failed to read tags CSV: {e}")

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """Prepara a imagem mantendo a proporção (padding) e convertendo para BGR float32"""
        # Converte para RGB para garantir consistência (remove alpha channel)
        image = image.convert("RGB")
        
        # Cria um canvas quadrado com fundo branco (WD14 foi treinado com fundo branco)
        max_dim = max(image.size)
        padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        
        # Cola a imagem centralizada no canvas
        paste_x = (max_dim - image.size[0]) // 2
        paste_y = (max_dim - image.size[1]) // 2
        padded.paste(image, (paste_x, paste_y))
        
        # Redimensiona para o tamanho alvo do ONNX
        resized = padded.resize((self.target_size, self.target_size), Image.Resampling.BICUBIC)

        # Converte para NumPy
        img_array = np.array(resized, dtype=np.float32)
        
        # A maioria dos modelos WD14 espera BGR ao invés de RGB
        img_array = img_array[:, :, ::-1] 
        
        # Adiciona a dimensão do Batch: de (H, W, C) para (1, H, W, C)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array

    def tag(self, image: Image.Image, custom_threshold: float = None) -> Dict[str, Any]:
        """Extrai as tags da imagem"""
        if not self.is_loaded():
            self.load()
            
        threshold_to_use = custom_threshold if custom_threshold is not None else self.threshold

        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name

        input_data = self._preprocess(image)

        # Inferência
        outputs = self.session.run([output_name], {input_name: input_data})[0]
        
        # Output tem formato (1, num_tags)
        probs = outputs[0]

        # Filtra as tags pelo threshold e ignora as tags de "rating" (geralmente as 4 primeiras do WD14)
        result_tags = {}
        for i, prob in enumerate(probs):
            if prob >= threshold_to_use and i < len(self.tags_names):
                tag_name = self.tags_names[i]
                # Opcional: Ignorar tags de classificação etária (rating: general, sensitive, etc)
                if not tag_name.startswith("rating:"):
                    result_tags[tag_name] = float(prob)

        # Ordena da maior probabilidade para a menor
        sorted_tags = dict(sorted(result_tags.items(), key=lambda item: item[1], reverse=True))

        # String formatada pronta para prompt
        tag_string = ", ".join(sorted_tags.keys())
        tag_string = tag_string.replace("_", " ") # Tira underlines padrão de booru

        logger.info(f"Engine Tagger executed successfully on CPU with {self.repo_id}")
        return {
            "tags": sorted_tags,
            "prompt_string": tag_string
        }

    def unload(self):
        """Libera a memória RAM do sistema"""
        self.session = None
        self.tags_names = []
        import gc
        gc.collect()
        logger.info("ONNXTagger: Unloaded from CPU RAM")

    def is_loaded(self) -> bool:
        return self.session is not None