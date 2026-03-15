import os
import logging
from tavily import AsyncTavilyClient

logger = logging.getLogger("JulyEngine.Models.TavilySearch")

class TavilySearch:
    def __init__(self, backend="api", model_tag="tavily"):
        self.backend = backend
        self.env_api_key = os.environ.get("TAVILY_API_KEY")

    def _get_api_key_from_db(self) -> str:
        """Busca a chave no banco da July se não estiver nas variáveis de ambiente."""
        try:
            from ..persistence import get_backend
            db = get_backend()
            config = db.get_setting("WEB_SEARCH")
            if config and "api_key" in config:
                return config["api_key"]
        except Exception as e:
            logger.warning(f"TavilySearch: Não foi possível ler a API Key do banco: {e}")
        return ""

    async def search(self, query: str, headers: dict = None):
        # Correção do anti-pattern de Python (dicionário mutável no parâmetro)
        if headers is None:
            headers = {}

        # Cascata de Autenticação inteligente
        api_key = headers.get("x-api-key") or self._get_api_key_from_db() or self.env_api_key

        if not api_key:
            logger.error("TAVILY_API_KEY is missing in Headers, DB, and ENV")
            return "Error: TAVILY_API_KEY is missing. Configure a chave de pesquisa no painel da July."
        
        try:
            # Instancia o cliente assíncrono oficial
            tavily_client = AsyncTavilyClient(api_key=api_key)
            
            # Chamada limpa e não-bloqueante usando os parâmetros corretos da lib
            response = await tavily_client.search(
                query=query,
                search_depth="basic",
                include_answer=True,
                max_results=5
            )
            
            # Prioriza a resposta mastigada (answer) da própria Tavily
            answer = response.get("answer")
            if answer:
                logger.info(f"Engine TavilySearch executed successfully on {self.backend} (Direct Answer {answer})")
                return answer
                
            # Fallback elegante caso a 'answer' venha vazia
            results = response.get("results", [])
            if not results:
                return "Nenhum resultado relevante encontrado para esta pesquisa."
                
            content_list = [f"- {r.get('title')}: {r.get('content')}" for r in results]
            logger.info(f"Engine TavilySearch executed successfully on {self.backend} ({len(results)} results)")
            
            return "\n".join(content_list)

        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return f"Error during web search: {str(e)}"