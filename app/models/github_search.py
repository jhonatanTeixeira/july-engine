import asyncio
import os
import logging
from typing import Any, Dict, List, Optional

from .base_model import BaseModel

logger = logging.getLogger("JulyEngine.Models.GithubSearch")


class GithubSearchModel(BaseModel):
    def __init__(self, backend: str = "api", model_meta: Optional[dict] = None):
        super().__init__(backend, model_meta)
        self._api_key = os.environ.get("GITHUB_API_KEY", "")

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        return 0

    def load(self, n_ctx=None, num_layers=None):
        pass

    def is_loaded(self) -> bool:
        return True

    def unload(self, model_name=None):
        pass

    def run(self, payload: Dict[str, Any]):
        raise NotImplementedError("GithubSearchModel.run() — use search() directly")

    async def search(self, query: str) -> List[Dict[str, Any]]:
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"token {self._api_key}"

        try:
            import requests

            def _fetch():
                resp = requests.get(
                    f"https://api.github.com/search/repositories"
                    f"?q={query}&sort=stars&order=desc&per_page=5",
                    headers=headers,
                )
                resp.raise_for_status()
                return resp.json()

            data = await asyncio.to_thread(_fetch)
            results = []
            for item in data.get("items", []):
                results.append({
                    "name":        item.get("full_name"),
                    "url":         item.get("html_url"),
                    "stars":       item.get("stargazers_count"),
                    "forks":       item.get("forks_count"),
                    "description": item.get("description"),
                })
            return results
        except Exception as e:
            logger.error(f"GithubSearch: failed: {e}")
            return []
