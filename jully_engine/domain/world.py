from typing import Any, Dict
import logging
from ..engine_models.github_search import GithubSearch
from ..engine_models.tavily_search import TavilySearch
from ..engine_models.google_search import GoogleSearch
from ..persistence.persistence import get_backend 

logger = logging.getLogger("JulyEngine.Domain.World")

class World:
    def __init__(self, backend: str, model_tag: str):
        self.backend = backend
        self.model_tag = model_tag
        self.tavily = TavilySearch(backend, model_tag)
        self.google = GoogleSearch(backend, model_tag)
        self.github = GithubSearch(backend, model_tag)

    async def search_web(self, payload: Dict[str, Any]):
        query = payload.get("query", "")
        engine = payload.get("model", None)
        headers = payload.get("headers", {})
        
        if not engine:
            config = get_backend().get_setting('WEB_SEARCH')
            
            if not config:
                raise ValueError('no model provided and no configuration set for WEB_SEARCH')
            
            engine = config.get('model')
            headers['x-api-key'] = config.get('api_key')

        if engine.lower() == "google":
            return await self.google.search(query, headers=headers)
        else:
            return await self.tavily.search(query, headers=headers)

    async def search_code(self, payload: Dict[str, Any]):
        query = payload.get("query", "")
        return await self.github.search(query)
