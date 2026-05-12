import base64
import io
import logging
from typing import Any, Dict, Optional

from .base_model import BaseModel

logger = logging.getLogger("JulyEngine.Models.BgRemover")


class BgRemoverModel(BaseModel):
    def __init__(self, backend: str = "cpu", model_meta: Optional[dict] = None):
        super().__init__(backend, model_meta)
        self._session = None

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        return 200 if self.backend == "gpu" else 0

    def load(self, n_ctx=None, num_layers=None):
        if self._session is not None:
            return
        import onnxruntime as ort
        from rembg import new_session

        available = ort.get_available_providers()
        providers = []
        if self.backend == "gpu":
            for p in ("CUDAExecutionProvider", "VulkanExecutionProvider", "ROCMExecutionProvider"):
                if p in available:
                    providers.append(p)
                    break
        providers.append("CPUExecutionProvider")

        logger.info(f"BgRemover: Loading u2net with providers {providers}")
        self._session = new_session(model_name="u2net", providers=providers)
        logger.info("BgRemover loaded.")

    def is_loaded(self) -> bool:
        return self._session is not None

    def unload(self, model_name=None):
        self._session = None

    def run(self, payload: Dict[str, Any], **kwargs) -> str:
        from PIL import Image
        from rembg import remove

        image_data = payload.get("image", "")

        if self._session is None:
            self.load()

        if isinstance(image_data, Image.Image):
            pil_img = image_data.convert("RGBA")
        else:
            img_bytes = base64.b64decode(image_data)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

        output: Image.Image = remove(pil_img, session=self._session)

        buf = io.BytesIO()
        output.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
