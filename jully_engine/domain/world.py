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
        
        config = get_backend().get_setting('WEB_SEARCH')
        if config:
            if "x-base-url" not in headers and "base_url" in config:
                headers["x-base-url"] = config["base_url"]
            has_auth = "authorization" in headers or "x-api-key" in headers
            if not has_auth and "api_key" in config and config["api_key"]:
                headers["x-api-key"] = config["api_key"]
                headers["authorization"] = f"Bearer {config['api_key']}"

        if not engine:
            if not config:
                raise ValueError('no model provided and no configuration set for WEB_SEARCH')
            
            engine = config.get('model')
            if 'x-api-key' not in headers and config.get('api_key'):
                headers['x-api-key'] = config.get('api_key')

        if engine.lower() == "google":
            return await self.google.search(query, headers=headers)
        else:
            return await self.tavily.search(query, headers=headers)

    async def search_code(self, payload: Dict[str, Any]):
        query = payload.get("query", "")
        headers = payload.get("headers", {})
        
        config = get_backend().get_setting('REPOSITORY_SEARCH')
        if config:
            if "x-base-url" not in headers and "base_url" in config:
                headers["x-base-url"] = config["base_url"]
            has_auth = "authorization" in headers or "x-api-key" in headers
            if not has_auth and "api_key" in config and config["api_key"]:
                headers["x-api-key"] = config["api_key"]
                headers["authorization"] = f"Bearer {config['api_key']}"
                
        return await self.github.search(query)
