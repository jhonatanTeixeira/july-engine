from typing import Optional, Dict, Any

from .base_model import BaseModel

MODEL_ID = "Aero-Ex/Trellis2-SDNQ"


class Trellis2Model(BaseModel):
    """
    Stub only — Trellis2 (3D generation, SDNQ) has too little documented usage
    on its HF model card to safely integrate yet (no confirmed pipeline class,
    input/output contract, or SDNQ application pattern). Revisit once the card
    has concrete loading/inference examples.
    """

    def __init__(self, backend: str = "gpu", model_meta: Optional[dict] = None):
        super().__init__(backend, model_meta)
        self.model_id = self.meta.get("id", MODEL_ID)

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        return 0

    def load(self, n_ctx=None, num_layers=None):
        pass

    def is_loaded(self) -> bool:
        return False

    def unload(self, model_name: Optional[str] = None):
        pass

    def run(self, payload: Dict[str, Any], **kwargs):
        raise NotImplementedError(
            "Trellis2 SDNQ API isn't documented well enough yet to integrate — "
            "revisit once the model card has concrete usage examples."
        )
