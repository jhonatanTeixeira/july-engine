import os
import requests
import logging

logger = logging.getLogger("JulyEngine.Models.GoogleSearch")

class GoogleSearch:
    def __init__(self, backend="api", model_tag="google"):
        self.backend = backend
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        self.cx = os.environ.get("GOOGLE_CX") # Custom Search Engine ID

    async def search(self, query: str):
        if not self.api_key or not self.cx:
            logger.error("GOOGLE_API_KEY or GOOGLE_CX is not set")
            return "Error: Google Search credentials missing."

        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_key,
            "cx": self.cx,
            "q": query,
            "num": 5
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = data.get("items", [])
            if not results:
                return "No relevant information found on Google."
                
            content_list = [f"- {r.get('title')}: {r.get('snippet')} ({r.get('link')})" for r in results]
            return "\n".join(content_list)
        except Exception as e:
            logger.error(f"Google search failed: {e}")
            return f"Error during Google search: {str(e)}"
