from abc import abstractmethod
import html
import json
import logging
import re
import textwrap
import uuid
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Union

from .internal_mcp import InternalMCP

logger = logging.getLogger("JulyEngine.Services.McpEmulator")


class Chunk:
    def __init__(self, content: str, is_reasoning: bool = False):
        self.content = content
        self.is_reasoning = is_reasoning
    
    @property
    def delta(self):
        if self.is_reasoning:
            return {"choices": [{"delta": {"reasoning_content": self.content}}]}
        
        return {"choices": [{"delta": {"content": self.content}}]}
    
    @classmethod
    def from_content(cls, content: str, is_reasoning: bool = False):
        return cls(content, is_reasoning)

@dataclass
class Tag:
    name: str
    arguments: str


class Buffer:
    def __init__(self):
        self.text = ""
        self.state = "NORMAL" # Pode ser "NORMAL" ou "REASONING"
        self.reasoning_tag = None
        
        # Encontra qualquer abertura de tag
        self.open_tag_re = re.compile(r"<([a-zA-Z_]+)>")
        # Prende fragmentos de tag no final do stream (ex: "<sear")
        self.suspect_tag_re = re.compile(r"<[a-zA-Z_]*$")
        
    def append(self, content: str):
        self.text += content
        
    def process(self):
        """A Máquina de Estados que decide o que vira Chunk (UI) e o que vira Tag (Ferramenta)"""
        yields = []
        
        while True:
            if self.state == "NORMAL":
                open_match = self.open_tag_re.search(self.text)
                suspect_match = self.suspect_tag_re.search(self.text)
                
                if open_match:
                    tag_name = open_match.group(1)
                    
                    # Se for tag de pensamento, entramos no MODO CEGO (Reasoning)
                    if tag_name.lower() in ["think", "thought", "reasoning"]:
                        cut_idx = open_match.end()
                        safe_text = self.text[:cut_idx]
                        if safe_text:
                            yields.append(Chunk.from_content(safe_text))
                        
                        self.text = self.text[cut_idx:]
                        self.state = "REASONING"
                        self.reasoning_tag = tag_name
                        continue # Reavalia no novo estado
                        
                    # Se for ferramenta normal, esperamos fechar a tag
                    else:
                        close_tag_re = re.compile(rf"</{re.escape(tag_name)}>")
                        close_match = close_tag_re.search(self.text, open_match.end())
                        
                        if close_match: # Ferramenta completa!
                            safe_text = self.text[:open_match.start()]
                            if safe_text:
                                yields.append(Chunk.from_content(safe_text))
                            
                            tool_args = self.text[open_match.end():close_match.start()].strip()
                            yields.append(Tag(name=tag_name, arguments=tool_args))
                            
                            self.text = self.text[close_match.end():]
                            continue
                        else:
                            # Ferramenta abriu mas não fechou. Prende no buffer.
                            safe_text = self.text[:open_match.start()]
                            if safe_text:
                                yields.append(Chunk.from_content(safe_text))
                                self.text = self.text[open_match.start():]
                            break # Aguarda o próximo pedaço da API
                            
                elif suspect_match: # Fragmento '<' detectado, prende por segurança
                    safe_text = self.text[:suspect_match.start()]
                    if safe_text:
                        yields.append(Chunk.from_content(safe_text))
                        self.text = self.text[suspect_match.start():]
                    break
                    
                else: # Texto 100% limpo, solta na tela!
                    if self.text:
                        yields.append(Chunk.from_content(self.text))
                        self.text = ""
                    break
                    
            elif self.state == "REASONING":
                # No modo cego, SÓ procuramos pelo fechamento do pensamento. 
                # Ferramentas inteiras aqui dentro viram texto puro automaticamente!
                close_str = f"</{self.reasoning_tag}>"
                close_idx = self.text.find(close_str)
                
                if close_idx != -1: # Achou o fim do pensamento
                    end_idx = close_idx + len(close_str)
                    safe_text = self.text[:end_idx]
                    if safe_text:
                        yields.append(Chunk.from_content(safe_text, True))
                        
                    self.text = self.text[end_idx:]
                    self.state = "NORMAL" # Volta a escutar ferramentas!
                    self.reasoning_tag = None
                    continue
                else:
                    # Verifica se o fechamento está pela metade no final do stream (ex: "</thi")
                    last_lt = self.text.rfind('<')
                    if last_lt != -1 and close_str.startswith(self.text[last_lt:]):
                        safe_text = self.text[:last_lt]
                        if safe_text:
                            yields.append(Chunk.from_content(safe_text))
                            self.text = self.text[last_lt:]
                        break
                    else:
                        # Tudo que está no pensamento é liberado pra UI em tempo real!
                        if self.text:
                            yields.append(Chunk.from_content(self.text))
                            self.text = ""
                        break
                        
        return yields


class TagFinder:
    def __init__(self, content: Union[Dict, AsyncGenerator]):
        self.is_stream = isinstance(content, AsyncGenerator)
        self.content = content
        self.buffer = Buffer()
        self.queue = [] # Fila de itens prontos para o stream
        self.clean_text = "" # Guarda o texto limpo no modo síncrono

    # ==========================================
    # MODO NÃO-STREAM (SÍNCRONO)
    # ==========================================
    def __iter__(self):
        if self.is_stream:
            raise Exception('Content is streaming. Use "async for".')
            
        raw_text = self.content.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        raw_text = html.unescape(raw_text)
        self.tags = []
        
        # Regex híbrida: Captura blocos de Raciocínio (Grupo 1) OU Ferramentas (Grupo 2 e 3)
        pattern = re.compile(r"<(think|thought|reasoning)>.*?</\1>|<([a-zA-Z_]+)>(.*?)</\2>", re.DOTALL)
        
        def replacer(match):
            if match.group(1): # É bloco de raciocínio
                return Chunk.from_content(match.group(0), is_reasoning=True) # Mantém no texto!
            else: # É ferramenta
                self.tags.append(Tag(name=match.group(2), arguments=match.group(3).strip()))
                return "" # Apaga do texto limpo
                
        self.clean_text = pattern.sub(replacer, raw_text).strip()
        self.sync_iter = iter(self.tags)

        return self
        
    def __next__(self):
        return next(self.sync_iter)

    def __aiter__(self):
        if not self.is_stream:
            raise Exception('Content is not streaming. Use "for".')
        return self
        
    async def __anext__(self):
        while not self.queue: # Se a fila tá vazia, busca da LLM
            try:
                raw_chunk = await anext(self.content)
                chunk = Chunk(raw_chunk)
                self.buffer.append(chunk.content)
                self.queue.extend(self.buffer.process()) # A máquina de estados enche a fila
            except StopAsyncIteration:
                # O stream acabou. Despeja qualquer sobra do buffer na UI.
                if self.buffer.text:
                    self.queue.append(Chunk.from_content(self.buffer.text))
                    self.buffer.text = ""
                
                if not self.queue:
                    raise StopAsyncIteration
                break
                
        return self.queue.pop(0)


@dataclass
class Tool:
    name: str
    arguments: Dict
    id: str = ""  # Adicionado para manter rastro pro formato OpenAI


@dataclass
class ToolReponse:
    tool_call_id: str
    name: str
    response: Any
    
    def request(self):
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": self.response
        }


@dataclass
class MediaContent:
    type: str
    url: str
    
    def __str__(self):
        return json.dumps({
            "type": self.type,
            self.type: self.url
        })


class McpEmulator:
    def __init__(self, internal_mcp: InternalMCP):
        self.internal_mcp = internal_mcp
        self.indexed_tools = {t['function']['name']: t for t in internal_mcp.get_tools()}
        self.xml_tags = self.json_tools_to_xml_prompt(internal_mcp.get_tools())
    
    def json_tools_to_xml_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """
        Converte uma lista de tools no formato OpenAI JSON para um prompt de 
        instruções em texto puro usando tags XML.
        """
        prompt_lines = []
        
        for tool in tools:
            if tool.get("type") != "function":
                continue
                
            func = tool.get("function", {})
            name = func.get("name", "unknown_tool")
            description = func.get("description", "Sem descrição.")
            
            properties = func.get("parameters", {}).get("properties", {})
            required = func.get("parameters", {}).get("required", [])
            
            if required:
                param_name = required[0].upper()
            elif properties:
                param_name = list(properties.keys())[0].upper()
            else:
                param_name = "VALOR"
                
            line = f"* **<{name}>**{param_name}</{name}> -> {description}"
            prompt_lines.append(line)
            
        return "\n".join(prompt_lines)
    
    def inject_tools(self, payload: Dict):
        system_prompt = textwrap.dedent(f'''
        # TOOLING
        You are an intelligent agent equipped with tools.

        ## ⚠️ CRITICAL SYNTAX AND EXECUTION RULES (MANDATORY)
        1. **STRICT SYNTAX:** Use ONLY less-than/greater-than signs for tags (e.g., `<search>`). It is STRICTLY FORBIDDEN to use square brackets (e.g., `[search]`).
        2. **NO MARKDOWN IN TAGS:** NEVER wrap XML tags in code blocks (```). The tags must be inserted as plain text.
        3. **REASONING ISOLATION:** The system DOES NOT read what you write in your hidden thought block. To execute a tool, the XML tag must be written in your FINAL RESPONSE (outside the thought block).

        ## TOOL AND ACTION GUIDELINES
        {self.xml_tags}

        ## GENERAL RULES
        - If you need more context to answer the user, such as searching for news or memories, respond ONLY with the command that fetches this information. The data will then be provided to you in a new iteration.
        ''').strip() # <-- O .strip() garante que não há quebras de linha fantasmas
        
        system_msg = {
            "role": "system",
            "content": system_prompt
        }
        
        messages = payload.get("messages", [])
        if messages and messages[0].get("role") == 'system':
            messages[0]["content"] += f"\n\n{system_msg['content']}"
        else:
            payload.setdefault("messages", []).insert(0, system_msg)
            
    def parse_xml_tags(self, response: Dict | AsyncGenerator) -> Dict | AsyncGenerator:

        def _build_args(name: str, value: str) -> Dict:
            args = {}
            if name in self.indexed_tools:
                props = self.indexed_tools[name]["function"].get("parameters", {}).get("properties", {})
                if props:
                    first_param = list(props.keys())[0]
                    args[first_param] = value
            else:
                args["VALOR"] = value
            return args

        if isinstance(response, Dict):
            tool_calls = []
            finder = TagFinder(response)
            reasoning = ""
            
            for tag in finder:
                if isinstance(tag, Tag):
                    logger.info(f"🔧 Ferramenta detectada: {tag.name}")
                    
                    args = _build_args(tag.name, tag.arguments)
                    
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {"name": tag.name, "arguments": json.dumps(args)}
                    })
                elif isinstance(tag, Chunk):
                    reasoning += tag.content
            
            response["choices"][0]["message"]["content"] = finder.clean_content
            response["choices"][0]["message"]["reasoning_content"] = reasoning
            
            if tool_calls:
                response["choices"][0]["message"]["tool_calls"] = tool_calls
                response["choices"][0]["finish_reason"] = "tool_calls"
                
            return response
            
        else:
            # Para o Stream: O buffer inteligente
            async def stream_generator():
                buffer = ""
                async for item in TagFinder(response):
                    if isinstance(item, Tag): 
                        logger.info(f"🔧 Ferramenta detectada: {item.name} com args {item.arguments}")
                        yield Tool(item.name, _build_args(item.name, item.arguments))
                    else:
                        yield item.delta

            return stream_generator()
            
    async def orchestrate(self, response: Dict | AsyncGenerator, brain, original_payload: Dict):
        response = self.parse_xml_tags(response)
        
        def clear_system_prompt():
            if original_payload['messages'] and original_payload['messages'][0].get('role') == 'system':
                sys_content = original_payload['messages'][0]['content']
                
                # Corta o texto e joga fora a nossa injeção de ferramentas, 
                # mantendo intacta a personalidade original da July (se houver).
                if "# TOOLING" in sys_content:
                    clean_sys = sys_content.split("# TOOLING")[0].strip()
                    # Se o prompt de sistema era só as ferramentas, podemos deixar um fallback
                    original_payload['messages'][0]['content'] = clean_sys or "Você é uma assistente prestativa. Responda ao usuário com base nos dados fornecidos."
            
        
        if isinstance(response, Dict):
            message = response.get("choices", [{}])[0].get("message", {})
            multimodal_content = []
            tools_called = False
            
            # Adiciona o request do assistente ao histórico para a API não chiar
            if message.get("tool_calls"):
                original_payload['messages'].append({
                    "role": "assistant",
                    "content": message.get('content', None),
                    # "tool_calls": message.get("tool_calls", [])
                })

            for tool_call in message.get("tool_calls", []):
                tools_called = True
                name = tool_call.get("function", {}).get("name")
                args = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
                
                llm, user = await self.internal_mcp.execute_tool(name, args)
                
                if user:
                    multimodal_content.append(user)
                
                if llm:
                    # original_payload['messages'].append({"role": "tool", "content": llm, "tool_call_id": tool_call.get('id'), 'name': tool_call.get('function').get('name')})
                    original_payload['messages'].append({"role": "user", "content": f"[SYSTEM MESSAGE TOOL {name} CALLED]: {llm}"})
                
                original_payload.setdefault("headers", {})["x-enable-internal-mcp"] = "0"
            
            if tools_called:
                clear_system_prompt()
                                        
                response = await brain.chat(original_payload)
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "Unable to fulfill your request")
                
                if len(multimodal_content) > 0:
                    if isinstance(content, str):
                        multimodal_content.append({"type": "text", "text": content})
                    else:
                        multimodal_content.extend(content)
                        
                    response.setdefault("choices", [{}])[0].setdefault("message", {}).setdefault('content', multimodal_content)

            return response
            
        else:
            # FIX: Implementação completa do ReAct Loop no Stream
            async def stream_orchestrator():
                clear_system_prompt()
                original_payload.setdefault("headers", {})["x-enable-internal-mcp"] = "0"
                
                async for chunk in response:
                    if isinstance(chunk, Tool):
                        llm, user = self.internal_mcp.execute_tool(chunk.name, chunk.arguments)
                        
                        if user:
                            yield user
                        
                        if llm:
                            original_payload["messages"].append({"role": "user", "content": f"[SYSTEM MESSAGE TOOL {name} CALLED]: {llm}"})
                            
                            async for second_chunk in await brain.chat(original_payload):
                                yield second_chunk

            return stream_orchestrator()