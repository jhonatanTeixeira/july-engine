import os
import requests
import logging

logger = logging.getLogger("JulyEngine.Models.GithubSearch")

class GithubSearch:
    def __init__(self, backend="api", model_tag="github"):
        self.backend = backend
        self.api_key = os.environ.get("GITHUB_API_KEY")

    async def search(self, query: str):
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"token {self.api_key}"
        
        # Busca os repositórios mais relevantes (sort=stars, order=desc)
        url = f"https://api.github.com/search/repositories?q={query}&sort=stars&order=desc&per_page=5"
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("items", []):
                results.append({
                    "name": item.get("full_name"),
                    "url": item.get("html_url"),
                    "stars": item.get("stargazers_count"),
                    "forks": item.get("forks_count"),
                    "description": item.get("description")
                })
            return results
        except Exception as e:
            logger.error(f"Github search failed: {e}")
            return []
