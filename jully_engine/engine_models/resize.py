import base64
import io
import logging
import gc
import os
from PIL import Image
import numpy as np
from typing import Dict, Any, Optional

logger = logging.getLogger("JulyEngine.EngineModels.Resize")

def free_vram():
    """Libera VRAM e memória RAM dinamicamente."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.info("VRAM cleaned up.")
    except ImportError:
        pass

class ResizerBase:
    def __init__(self):
        self._model = None
        self._device = None # Lazy evaluation

    def is_loaded(self):
        return self._model is not None

    @property
    def device(self):
        """Avaliação preguiçosa (Lazy) da GPU para não importar o Torch à toa."""
        if self._device is None:
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"
        return self._device

    def decode_image(self, image_data: Any) -> Image.Image:
        if isinstance(image_data, str):
            if image_data.startswith("data:image"):
                image_data = image_data.split(",")[1]
            img_bytes = base64.b64decode(image_data)
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")
        elif isinstance(image_data, bytes):
            return Image.open(io.BytesIO(image_data)).convert("RGB")
        return image_data

    def encode_image(self, img: Image.Image, format: str = "PNG") -> str:
        buffered = io.BytesIO()
        img.save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode()

    def get_new_size(self, img: Image.Image, scale: float, width: Optional[int], height: Optional[int]) -> tuple:
        orig_w, orig_h = img.size
        if width and height:
            return (int(width), int(height))
        elif width:
            return (int(width), int(orig_h * (int(width) / orig_w)))
        elif height:
            return (int(orig_w * (int(height) / orig_h)), int(height))
        else:
            return (int(orig_w * scale), int(orig_h * scale))

    def unload(self):
        self._model = None
        free_vram()

# ==========================================================
# CPU / CLASSIC RESIZERS
# ==========================================================
class PillowResizer(ResizerBase):
    def resize(self, payload: Dict[str, Any]) -> str:
        img = self.decode_image(payload.get("image"))
        scale = float(payload.get("scale", 1.0))
        width = payload.get("width")
        height = payload.get("height")
        
        new_size = self.get_new_size(img, scale, width, height)
        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
        return self.encode_image(img_resized)

class OpencvResizer(ResizerBase):
    def resize(self, payload: Dict[str, Any]) -> str:
        import cv2
        img_pil = self.decode_image(payload.get("image"))
        scale = float(payload.get("scale", 1.0))
        width = payload.get("width")
        height = payload.get("height")
        
        new_size = self.get_new_size(img_pil, scale, width, height)
        img_np = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        res = cv2.resize(img_bgr, new_size, interpolation=cv2.INTER_LANCZOS4)
        res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        return self.encode_image(Image.fromarray(res_rgb))

# ==========================================================
# GPU / AI UPSCALERS
# ==========================================================
class RealESRGANResizer(ResizerBase):
    def load(self):
        if self._model is not None:
            return
        
        from realesrgan import RealESRGANer
        from basicsr.archs.srvgg_arch import SRVGGNetCompact
        
        # A rede base DEVE bater com a escala nativa do modelo (4x)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4)
        model_path = os.path.join("weights", "RealESRGAN_x4plus.pth")
        
        self._model = RealESRGANer(
            scale=4, # A escala nativa do modelo
            model_path=model_path if os.path.exists(model_path) else None,
            model=model,
            tile=512,
            tile_pad=10,
            pre_pad=0,
            half=True if self.device == "cuda" else False, # Proteção RTX 3050
            device=self.device
        )
        logger.info(f"RealESRGAN loaded on {self.device}")

    def resize(self, payload: Dict[str, Any]) -> str:
        self.load()
        import cv2
        
        scale_requested = float(payload.get("scale", 4.0))
        img_pil = self.decode_image(payload.get("image"))
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # O 'outscale' gerencia redimensionamentos dinâmicos menores ou maiores que 4
        output, _ = self._model.enhance(img_bgr, outscale=scale_requested)
        
        res_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return self.encode_image(Image.fromarray(res_rgb))

class GFPGANResizer(ResizerBase):
    def load(self):
        if self._model is not None:
            return
            
        from gfpgan import GFPGANer
        model_path = os.path.join("weights", "GFPGANv1.4.pth")
        
        self._model = GFPGANer(
            model_path=model_path if os.path.exists(model_path) else None,
            upscale=1, # Mantemos 1 internamente para flexibilidade
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None,
            device=self.device
        )
        logger.info(f"GFPGAN loaded on {self.device}")

    def resize(self, payload: Dict[str, Any]) -> str:
        self.load()
        import cv2
        
        scale_requested = float(payload.get("scale", 2.0))
        img_pil = self.decode_image(payload.get("image"))
        
        # Se usuário pediu upscale, fazemos um redimensionamento base primeiro,
        # depois a rede restaura os detalhes no tamanho novo.
        if scale_requested != 1.0:
            new_size = (int(img_pil.width * scale_requested), int(img_pil.height * scale_requested))
            img_pil = img_pil.resize(new_size, Image.Resampling.LANCZOS)
            
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        _, _, restored_img = self._model.enhance(
            img_bgr,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )
        
        res_rgb = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
        return self.encode_image(Image.fromarray(res_rgb))

class CodeFormerResizer(ResizerBase):
    def load(self):
        if self._model is not None:
            return
            
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper
        # Arquitetura real do CodeFormer importada do módulo (se o projeto tiver baixado)
        try:
            from basicsr.utils.registry import ARCH_REGISTRY
            import torch
            
            # Helper lida com o recorte de faces na imagem inteira
            self.face_helper = FaceRestoreHelper(
                upscale_factor=1,
                face_size=512,
                crop_ratio=(1, 1),
                det_model='retinaface_resnet50',
                save_ext='png',
                device=self.device
            )
            
            # Inicializa a rede CodeFormer nativa
            model_path = os.path.join("weights", "codeformer.pth")
            self._model = ARCH_REGISTRY.get('CodeFormer')(
                dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
                connect_list=['32', '64', '128', '256']
            ).to(self.device)
            
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path)['params_ema']
                self._model.load_state_dict(checkpoint)
            
            self._model.eval()
            logger.info(f"CodeFormer loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Falta a biblioteca ou os módulos do CodeFormer: {e}")
            raise e

    def resize(self, payload: Dict[str, Any]) -> str:
        self.load()
        import cv2
        import torch
        from torchvision.transforms.functional import normalize
        
        scale_requested = float(payload.get("scale", 2.0))
        img_pil = self.decode_image(payload.get("image"))
        
        if scale_requested != 1.0:
            new_size = (int(img_pil.width * scale_requested), int(img_pil.height * scale_requested))
            img_pil = img_pil.resize(new_size, Image.Resampling.LANCZOS)
            
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # Pipeline Real do CodeFormer
        self.face_helper.clean_all()
        self.face_helper.read_image(img_bgr)
        self.face_helper.get_face_landmarks_5(only_center_face=False)
        self.face_helper.align_warp_face()

        for idx, cropped_face in enumerate(self.face_helper.cropped_faces):
            cropped_face_t = cv2.resize(cropped_face, (512, 512), interpolation=cv2.INTER_LINEAR)
            cropped_face_t = torch.tensor(cropped_face_t / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            
            with torch.no_grad():
                output_t = self._model(cropped_face_t, w=0.5, adain=True)[0]
                restored_face = output_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
                restored_face = np.clip((restored_face * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
            
            self.face_helper.add_restored_face(restored_face)

        self.face_helper.get_inverse_affine(None)
        restored_img = self.face_helper.paste_faces_to_input_image()
        
        res_rgb = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
        return self.encode_image(Image.fromarray(res_rgb))