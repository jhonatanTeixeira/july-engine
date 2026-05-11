import os
import logging
from typing import Any, Dict, Optional

from .base_model import BaseModel

logger = logging.getLogger("JulyEngine.Models.TavilySearch")


class TavilySearchModel(BaseModel):
    def __init__(self, backend: str = "api", model_meta: Optional[dict] = None):
        super().__init__(backend, model_meta)
        self._env_api_key = os.environ.get("TAVILY_API_KEY", "")

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        return 0

    def load(self, n_ctx=None, num_layers=None):
        pass

    def is_loaded(self) -> bool:
        return True

    def unload(self, model_name=None):
        pass

    def run(self, payload: Dict[str, Any]):
        raise NotImplementedError("TavilySearchModel.run() — use search() directly")

    async def search(
        self,
        query: str,
        headers: Optional[dict] = None,
        search_depth: str = "basic",
        include_answer: bool = True,
        max_results: int = 5,
        include_list: bool = False,
    ):
        if headers is None:
            headers = {}

        api_key = headers.get("x-api-key") or self._env_api_key
        if not api_key:
            logger.error("TavilySearch: API key missing in headers and environment")
            return "Error: TAVILY_API_KEY is missing."

        try:
            from tavily import AsyncTavilyClient
            client = AsyncTavilyClient(api_key=api_key)
            response = await client.search(
                query=query,
                search_depth=search_depth,
                include_answer=include_answer,
                max_results=max_results,
            )
            results = response.get("results", [])

            if include_list:
                return results

            answer = response.get("answer")
            if answer:
                return answer

            if not results:
                return "No relevant results found."

            return "\n".join(f"- {r.get('title')}: {r.get('content')}" for r in results)

        except Exception as e:
            logger.error(f"TavilySearch: search failed: {e}")
            return f"Error during web search: {e}"
