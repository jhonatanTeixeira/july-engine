import os
import requests
import logging

logger = logging.getLogger("JulyEngine.Models.TavilySearch")

class TavilySearch:
    def __init__(self, backend="api", model_tag="tavily"):
        self.backend = backend
        self.api_key = os.environ.get("TAVILY_API_KEY")

    async def search(self, query: str):
        if not self.api_key:
            logger.error("TAVILY_API_KEY is not set")
            return "Error: TAVILY_API_KEY is missing."

        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": True,
            "max_results": 5
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            
            # Tavily pode retornar uma "answer" direta se 'include_answer' for True
            answer = data.get("answer", "")
            if answer:
                return answer
                
            # Fallback para os resultados básicos
            results = data.get("results", [])
            content_list = [f"- {r.get('title')}: {r.get('content')}" for r in results]
            return "\n".join(content_list)
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return f"Error during web search: {str(e)}"
