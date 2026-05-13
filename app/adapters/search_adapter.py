import logging
from typing import Any, Dict, List, Optional

from .adapter_base import AdapterBase

logger = logging.getLogger("JulyEngine.Adapter.SearchAdapter")

_TASK_HANDLERS = {
    "web_search":  "_search_web",
    "code_search": "_search_code",
}


class SearchAdapter(AdapterBase):
    """
    Handles web and code search tasks.

    Engine field: "search"

    search_web:  routes to Google (engine="google") or Tavily (default).
    search_code: routes to GitHub search.

    Config is injected from persistence settings:
      WEB_SEARCH        → base_url, api_key, model (default engine)
      REPOSITORY_SEARCH → base_url, api_key
    """

    def __init__(self, task_type: str, backend: str = "cpu", model_meta: Optional[dict] = None):
        super().__init__(task_type, backend, model_meta)
        self._tavily = None
        self._google = None
        self._github = None

    @classmethod
    def get_engine_type(cls, task_type):
        return "WEB_SEARCH"

    # ------------------------------------------------------------------
    # Lazy sub-engine accessors
    # ------------------------------------------------------------------

    def _get_tavily(self):
        if self._tavily is None:
            from ..models.tavily_search import TavilySearchModel
            self._tavily = TavilySearchModel(backend=self.backend, model_meta=self.meta)
        return self._tavily

    def _get_google(self):
        if self._google is None:
            from ..models.google_search import GoogleSearchModel
            self._google = GoogleSearchModel(backend=self.backend, model_meta=self.meta)
        return self._google

    def _get_github(self):
        if self._github is None:
            from ..models.github_search import GithubSearchModel
            self._github = GithubSearchModel(backend=self.backend, model_meta=self.meta)
        return self._github

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        return 0

    def load(self, n_ctx=None, num_layers=None):
        pass

    def is_loaded(self) -> bool:
        return True

    def unload(self, model_name=None):
        pass

    async def run(self, payload: Dict[str, Any]):
        task_type = self.task_type
        handler_name = _TASK_HANDLERS.get(task_type)

        if not handler_name:
            raise ValueError(f"SearchAdapter: unknown task_type '{task_type}'")
        
        return await getattr(self, handler_name)(payload)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    async def _search_web(self, payload: Dict[str, Any]):
        query = payload.get("query", "")
        headers: dict = dict(payload.get("headers", {}))
        engine = payload.get("model", "")

        config = self._load_config("WEB_SEARCH")
        if config:
            headers.setdefault("x-base-url", config.get("base_url"))
            if config.get("api_key"):
                headers.setdefault("x-api-key", config["api_key"])
                headers.setdefault("authorization", f"Bearer {config['api_key']}")

        if not engine:
            if not config:
                raise ValueError("SearchAdapter: no model/engine provided and WEB_SEARCH not configured")
            engine = config.get("model", "tavily")

        if engine.lower() == "google":
            return await self._get_google().search(query, headers=headers)

        return await self._get_tavily().search(
            query,
            headers=headers,
            search_depth=payload.get("search_depth", "basic"),
            include_answer=payload.get("include_answer", True),
            max_results=payload.get("max_results", 5),
            include_list=payload.get("include_list", False),
        )

    async def _search_code(self, payload: Dict[str, Any]):
        query = payload.get("query", "")
        headers: dict = dict(payload.get("headers", {}))

        config = self._load_config("REPOSITORY_SEARCH")
        if config:
            headers.setdefault("x-base-url", config.get("base_url"))
            if config.get("api_key"):
                headers.setdefault("x-api-key", config["api_key"])
                headers.setdefault("authorization", f"Bearer {config['api_key']}")

        return await self._get_github().search(query)

    # ------------------------------------------------------------------
    # Config helper
    # ------------------------------------------------------------------

    def _load_config(self, key: str) -> dict:
        try:
            from ..services.models_service import model_service
            return model_service.backend.get_setting(key) or {}
        except Exception:
            return {}
