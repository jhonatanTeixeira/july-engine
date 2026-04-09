import asyncio
import re
import html
import json
from dataclasses import dataclass
import textwrap
from typing import Any, Dict, AsyncGenerator, List, Union
import re

# ==========================================
# 1. CLASSES DE DOMÍNIO (DATA E PARSER)
# ==========================================

@dataclass
class Tool:
    name: str
    arguments: str

class Chunk:
    def __init__(self, raw_chunk: dict):
        self.raw_chunk = raw_chunk
        
        delta = raw_chunk.get('choices', [{}])[0].get("delta", {})
        
        self.content = delta.get("content", "")
        self.is_reasoning = True if 'reasoning_content' in delta else False
        self.reasoning_content = delta.get("reasoning_content", "")
        
    @classmethod
    def from_str(cls, text):
        return cls({"choices": [{"delta": {"content": text}}]})

    @property
    def delta(self):
        """Reconstrói o objeto delta no formato OpenAI-Compatible"""
        return dict(self.raw_chunk)


class XMLStreamParser:
    """
    Atua como um Buffer inteligente. 
    Consome o LLM cru e cospe objetos 'Chunk' ou 'Tool'.
    """
    def __init__(self, stream: AsyncGenerator):
        self.stream = stream
        self.buffer = ""
        self.full_tag = re.compile(r"<([a-zA-Z_]+)>(.*?)</\1>", re.DOTALL)
        self.open_tag = re.compile(r"<([a-zA-Z_]+)>")
        self.suspect_tag = re.compile(r"<[a-zA-Z_]*$")

    async def __aiter__(self):
        buffer: str = ''
        tag_opened: str = None
        arguments: str = None
        
        async for chunk in self.stream:
            delta = Chunk(chunk)
            
            if not delta.is_reasoning:
                yield delta
                continue
            
            buffer += delta.content
            
            if '<' in buffer and re.match(r'(.*?)?<$|<\w+$', buffer) and not tag_opened:
                continue
            
            if (match := re.search(r'<(\w+)>', buffer)) and not tag_opened:
                tag_opened = match.group(1)
                before, after = buffer.split(f'<{tag_opened}>')
                
                yield Chunk.from_str(before)
                
                buffer = after
            
            if not tag_opened:
                yield Chunk.from_str(buffer)
                buffer = ''
            
            if tag_opened and (match := re.search(fr'<\/{tag_opened}>', buffer)):
                split = buffer.split(match.group(0), maxsplit=1)
                arguments = split[0]
                
                if len(split) > 1:
                    buffer = split[1]
                else:
                    buffer = ''
                
                yield Tool(tag_opened, arguments)
            
                tag_opened = None
                arguments = None
            
            await asyncio.sleep(0)
        
        if buffer:
            yield Chunk.from_str(buffer)


# ==========================================
# 2. ORQUESTRADOR PRINCIPAL
# ==========================================

class McpEmulator:
    def __init__(self, internal_mcp):
        self.internal_mcp = internal_mcp
        self.full_tag = re.compile(r"<([a-zA-Z_]+)>(.*?)</\1>", re.DOTALL)
        self.indexed_tools = {t['function']['name']: t for t in internal_mcp.get_tools()}
        self.xml_tags = self.json_tools_to_xml_prompt(internal_mcp.get_tools())
    
    def json_tools_to_xml_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """
        Converte uma lista de tools no formato OpenAI JSON para um prompt de 
        instruções usando blocos XML hierárquicos (Child Nodes).
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
            
            prompt_lines.append(f"### Tool: `{name}`")
            prompt_lines.append(f"Description: {description}")
            prompt_lines.append("USAGE FORMAT:")
            prompt_lines.append(f"<{name}>")
            
            if properties:
                for prop, data in properties.items():
                    prop_type = data.get('type', 'any')
                    prop_desc = data.get('description', '')
                    req_status = "REQUIRED" if prop in required else "OPTIONAL"
                    
                    # Desenha a tag do parâmetro com as instruções de preenchimento dentro
                    prompt_lines.append(f"  <{prop}>[{prop_type}, {req_status}] {prop_desc}</{prop}>")
            else:
                prompt_lines.append("  ")
                
            prompt_lines.append(f"</{name}>\n")
            
        return "\n".join(prompt_lines)
    
    def inject_tools(self, payload: Dict, tools_whitelist: List = []):
        xml_tags = self.xml_tags
        
        if tools_whitelist:
            tools: list = payload.setdefault('tools', [])
            
            for tool in tools_whitelist:
                tools.append(self.indexed_tools[tool])
        
        if 'tools' in payload and payload['tools']:
            xml_tags = self.json_tools_to_xml_prompt(payload['tools'])
            
        system_prompt = textwrap.dedent(f'''
# TOOLING CAPABILITIES
You can call tools to get information or perform actions. The environment will execute the tool and return the results to you.

## 🛑 ANTI-REFUSAL PROTOCOL (CRITICAL)
NEVER say "I cannot create images", "I cannot save memories", or "As an AI, I don't have access to...". 
You DO have access to these capabilities through your tools. Just output the correct XML tag block and the external system will do the rest.

## ⚠️ CRITICAL SYNTAX RULES (MANDATORY)
1. STRICT XML ONLY: You must use properly formatted XML tags with child nodes for parameters.
2. NO JSON/YAML/ATTRIBUTES: Do NOT use HTML-style attributes (e.g., `<tool param="value">`). Do NOT use JSON, YAML, or square brackets.
3. NO MARKDOWN: NEVER wrap your XML blocks in markdown code fences (```xml). Output them as plain raw text.
4. REASONING ISOLATION: The system DOES NOT read your internal `<think>` blocks. Tool calls MUST be in your final visible response.
5. SILENT EXECUTION: Do not announce or explain that you are going to use a tool. Just output the XML block.

## ⚙️ AVAILABLE TOOLS AND GUIDELINES
To execute a tool, you MUST output the EXACT XML block structure shown in the "USAGE FORMAT" for the chosen tool. Replace the bracketed instructions with your actual parameter values.

{xml_tags}
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

    def _build_args(self, name: str, value: str) -> Dict:
        args = {}
        indexed_tools = self.indexed_tools
        
        # 1. Extração à prova de falhas: Captura qualquer padrão <tag>conteudo</tag>
        # O re.DOTALL garante que ele vai capturar mesmo se a IA pular linhas dentro do valor
        matches = re.findall(r'<([a-zA-Z0-9_]+)>(.*?)</\1>', value, re.DOTALL)
        
        if not matches:
            # Se não achou tags, ou a tool não tem parâmetros, ou a IA errou muito.
            if name in indexed_tools:
                props = indexed_tools[name]["function"].get("parameters", {}).get("properties", {})
                if props:
                    first_param = list(props.keys())[0]
                    args[first_param] = value.strip()
            return args

        # Pega a especificação de tipos da ferramenta para recriar o JSON
        props = {}
        if name in indexed_tools:
            props = indexed_tools[name]["function"].get("parameters", {}).get("properties", {})

        # 2. Constrói o dicionário já aplicando o Type Casting correto!
        for key, val in matches:
            val_clean = val.strip()
            expected_type = props.get(key, {}).get("type", "string").lower()
            
            try:
                if expected_type == "number" or expected_type == "float":
                    args[key] = float(val_clean)
                elif expected_type == "integer":
                    args[key] = int(val_clean)
                elif expected_type == "boolean":
                    args[key] = val_clean.lower() in ["true", "1", "yes"]
                else:
                    args[key] = val_clean # Default é string
            except ValueError:
                # Se a IA alucinou texto onde era número, salva como string para não quebrar
                args[key] = val_clean
                
        return args

    async def orchestrate(self, response: Union[Dict, AsyncGenerator], brain, original_payload: Dict):

        if isinstance(response, Dict):
            # ==============================
            # MODO NÃO-STREAM (SÍNCRONO)
            # ==============================
            message = response["choices"][0].setdefault("message", {})
            raw_content = message.get("content") or ""
            
            # Unescape HTML entities (e.g., &lt; -> <)
            import html
            raw_content = html.unescape(raw_content)
            
            tools_to_execute = []
            
            # Substitui as tags por vazio no texto final e guarda os objetos Tool
            def replacer(match):
                tools_to_execute.append(Tool(name=match.group(1), arguments=match.group(2).strip()))
                return ""
                
            clean_content = self.full_tag.sub(replacer, raw_content).strip()
            message["content"] = clean_content if clean_content else None

            # Se não achou nenhuma ferramenta (ou era só reasoning), devolve a resposta original limpa
            if not tools_to_execute:
                return response 

            message['content'] = [{"type": "text", "text": clean_content}]
                
            original_payload['messages'].append(message)
            
            multimodal_content = []
            
            # 2. Executa todas as ferramentas sequencialmente
            requires_second_call = False
            
            for item in tools_to_execute:
                args = self._build_args(item.name, item.arguments)
                
                if item.name == "image_edit":
                    for msg in reversed(original_payload.get("messages", [])):
                        if isinstance(msg.get("content"), list):
                            for sub in msg["content"]:
                                if isinstance(sub, dict) and sub.get("type") == "image_url":
                                    args["image"] = sub.get("image_url", {}).get("url")
                                    break
                        if args.get("image"):
                            break
                            
                llm, user = await self.internal_mcp.execute_tool(item.name, args)
                
                is_faf = self.indexed_tools.get(item.name, {}).get("fire-and-forget", False)

                if "__" in item.name and not llm:
                    is_faf = True
                    
                if not is_faf:
                    requires_second_call = True
                
                # Guarda o visual (UI) para o final
                if user:
                    message['content'].append(user.response)
                
                # Guarda o texto para o LLM ler no segundo turno
                if llm:
                    original_payload['messages'].append({
                        "role": "user", 
                        "content": f"[SYSTEM MESSAGE TOOL {item.name} CALLED]: {llm}"
                    })

            if requires_second_call:
                second_response = await brain.chat(original_payload)
                second_content = second_response.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                if isinstance(second_content, list):
                    message['content'].extend(second_content)
                else:
                    message['content'].append(second_content)

            return response
            
        else:
            # ==============================
            # MODO STREAM (O SEU DESIGN)
            # ==============================
            async def stream_orchestrator():
                first_response = ''
                tools_executed = False
                requires_second_call = False
                
                # O Parser consome a rede neural, nós consumimos o Parser!
                async for item in XMLStreamParser(response):
                    if isinstance(item, Chunk):
                        if not item.is_reasoning:
                            first_response += item.content

                        yield item.delta
                        
                    elif isinstance(item, Tool):
                        tools_executed = True
                        
                        original_payload['messages'].append({
                            'role': 'assistant',
                            'content': first_response
                        })
                        
                        first_response = ''
                        
                        # Executa On-The-Fly!
                        args = self._build_args(item.name, item.arguments)
                        
                        if item.name == "image_edit":
                            for msg in reversed(original_payload.get("messages", [])):
                                if isinstance(msg.get("content"), list):
                                    for sub in msg["content"]:
                                        if isinstance(sub, dict) and sub.get("type") == "image_url":
                                            args["image"] = sub.get("image_url", {}).get("url")
                                            break
                                if args.get("image"):
                                    break
                                    
                        # Status messages mapping
                        status_map = {
                            "search_web": "searching the web",
                            "search_memory": "searching memory",
                            "generate_image": "generating image",
                            "generate_audio": "generating audio",
                            "save_memory": "saving memory",
                            "image_edit": "editing image"
                        }
                        display_name = status_map.get(item.name, f"calling {item.name}")
                        
                        # Yield start status
                        yield {"status_update": display_name}
                        await asyncio.sleep(0)
                        
                        llm, user = await self.internal_mcp.execute_tool(item.name, args)
                        
                        # Yield end status
                        yield {"status_update": ""} # Empty string to clear status
                        await asyncio.sleep(0)
                        
                        is_faf = self.indexed_tools.get(item.name, {}).get("fire-and-forget", False)
                        
                        if "__" in item.name and not llm:
                            is_faf = True
                            
                        if not is_faf:
                            requires_second_call = True
                        
                        if user:
                            yield user.delta
                            
                        if llm:
                            # 2. Injeta o resultado do LLM (Tavily, etc)
                            original_payload["messages"].append({
                                "role": "user", 
                                "content": f"[SYSTEM MESSAGE: TOOL {item.name} CALLED]: {llm}"
                            })
                
                if tools_executed and requires_second_call:
                    # 3. Dispara o segundo turno imediatamente!
                    async for second_chunk in await brain.chat(original_payload):
                        yield dict(second_chunk)

            return stream_orchestrator()

    def merge_usage(self, response1: dict, response2: dict) -> dict:
        """Soma o uso de tokens de duas respostas da API, incluindo tokens de raciocínio."""
        u1 = response1.get("usage", {})
        u2 = response2.get("usage", {})

        merged = {
            "prompt_tokens": u1.get("prompt_tokens", 0) + u2.get("prompt_tokens", 0),
            "completion_tokens": u1.get("completion_tokens", 0) + u2.get("completion_tokens", 0),
            "total_tokens": u1.get("total_tokens", 0) + u2.get("total_tokens", 0),
        }

        # Soma os detalhes de raciocínio (Chain of Thought) se existirem
        r1 = u1.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
        r2 = u2.get("completion_tokens_details", {}).get("reasoning_tokens", 0)

        if r1 > 0 or r2 > 0:
            merged["completion_tokens_details"] = {"reasoning_tokens": r1 + r2}

        return merged