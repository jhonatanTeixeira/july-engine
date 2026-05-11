import os
import logging
from typing import Any, Dict, Optional

from .base_model import BaseModel

logger = logging.getLogger("JulyEngine.Models.GoogleSearch")


class GoogleSearchModel(BaseModel):
    def __init__(self, backend: str = "api", model_meta: Optional[dict] = None):
        super().__init__(backend, model_meta)
        self._api_key = os.environ.get("GOOGLE_API_KEY", "")
        self._cx = os.environ.get("GOOGLE_CX", "")

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        return 0

    def load(self, n_ctx=None, num_layers=None):
        pass

    def is_loaded(self) -> bool:
        return True

    def unload(self, model_name=None):
        pass

    def run(self, payload: Dict[str, Any]):
        raise NotImplementedError("GoogleSearchModel.run() — use search() directly")

    async def search(self, query: str, headers: Optional[dict] = None):
        if headers is None:
            headers = {}

        api_key = headers.get("x-api-key") or self._api_key
        if not api_key or not self._cx:
            logger.error("GoogleSearch: API key or CX missing")
            return "Error: Google Search credentials missing."

        try:
            import asyncio
            import requests

            def _fetch():
                resp = requests.get(
                    "https://www.googleapis.com/customsearch/v1",
                    params={"key": api_key, "cx": self._cx, "q": query, "num": 5},
                )
                resp.raise_for_status()
                return resp.json()

            data = await asyncio.to_thread(_fetch)
            results = data.get("items", [])
            if not results:
                return "No relevant information found on Google."

            return "\n".join(
                f"- {r.get('title')}: {r.get('snippet')} ({r.get('link')})"
                for r in results
            )
        except Exception as e:
            logger.error(f"GoogleSearch: failed: {e}")
            return f"Error during Google search: {e}"
