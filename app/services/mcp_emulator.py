import asyncio
import re
import html
import json
import uuid
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
    def __init__(self, raw_chunk: Any):
        # Store as dict for easier manipulation, but keep original if needed
        if hasattr(raw_chunk, "model_dump"): # Pydantic V2
            self.raw_chunk = raw_chunk.model_dump()
        elif hasattr(raw_chunk, "dict"): # Pydantic V1
            self.raw_chunk = raw_chunk.dict()
        else:
            self.raw_chunk = dict(raw_chunk)
        
        self.id = self.raw_chunk.get("id")
        self.model = self.raw_chunk.get("model")
        self.object = self.raw_chunk.get("object")
        self.created = self.raw_chunk.get("created")
        self.usage = self.raw_chunk.get("usage")
        
        choices = self.raw_chunk.get('choices', [{}])
        delta = choices[0].get("delta", {}) if isinstance(choices[0], dict) else getattr(choices[0], "delta", {})
        
        self.content = delta.get("content", "") or "" if isinstance(delta, dict) else getattr(delta, "content", "") or ""
        self.reasoning_content = delta.get("reasoning_content", "") or "" if isinstance(delta, dict) else getattr(delta, "reasoning_content", "") or ""
        self.is_reasoning = True if self.reasoning_content else False
        self.finish_reason = choices[0].get("finish_reason") if isinstance(choices[0], dict) else getattr(choices[0], "finish_reason", None)
        
    @classmethod
    def from_str(cls, text, metadata=None):
        base = {"choices": [{"delta": {"content": text}}]}
        if metadata:
            base.update(metadata)
        return cls(base)

    def clean_delta(self):
        """Returns the chunk without finish_reason and usage."""
        clean = json.loads(json.dumps(self.raw_chunk)) # Deep copy as dict
        if "choices" in clean:
            for choice in clean["choices"]:
                choice.pop("finish_reason", None)
        clean.pop("usage", None)
        return clean


class XMLStreamParser:
    """
    Atua como um Buffer inteligente. 
    Consome o LLM cru e cospe objetos 'Chunk' ou 'Tool'.
    """
    def __init__(self, stream: AsyncGenerator):
        self.stream = stream
        self.buffer = ""
        self.open_tag = re.compile(r"<([a-zA-Z0-9_-]+)>")
        self.last_metadata = {}
        self.last_usage = None

    async def __aiter__(self):
        buffer: str = ''
        tag_opened: str = None
        is_buffering = False
        
        async for chunk_raw in self.stream:
            delta = Chunk(chunk_raw)
            
            # Capture metadata/usage from every single chunk
            if delta.id:
                self.last_metadata = {
                    "id": delta.id, "model": delta.model, 
                    "object": delta.object, "created": delta.created
                }
            if delta.usage:
                self.last_usage = delta.usage

            if delta.is_reasoning:
                yield delta
                continue
            
            if not is_buffering and '<' in delta.content:
                parts = delta.content.split('<', 1)
                before = parts[0]
                if before:
                    yield Chunk.from_str(before, metadata=self.last_metadata)
                
                is_buffering = True
                buffer = '<' + parts[1]
                continue

            if is_buffering:
                buffer += delta.content
            else:
                yield delta
            
            if is_buffering and not tag_opened and (match := self.open_tag.search(buffer)):
                tag_opened = match.group(1)

            if is_buffering and not tag_opened and re.search(r'<\s+', buffer):
                yield Chunk.from_str(buffer, metadata=self.last_metadata)
                is_buffering = False
                buffer = ''

            if is_buffering and tag_opened and re.search(rf'<\/{tag_opened}>', buffer):
                is_buffering = False

                before, after = buffer.split(f'</{tag_opened}>', 1)
                
                match = re.search(rf"<{tag_opened}>([\s\S\n]*?)<\/{tag_opened}>", buffer, re.MULTILINE)
                yield Tool(name=tag_opened, arguments=match.group(1))

                if after:
                    yield Chunk.from_str(after, metadata=self.last_metadata)
                
                tag_opened = None
                buffer = ''
                        
            await asyncio.sleep(0)
        
        if buffer:
            yield Chunk.from_str(buffer, metadata=self.last_metadata)
            await asyncio.sleep(0)


# ==========================================
# 2. ORQUESTRADOR PRINCIPAL
# ==========================================

class McpEmulator:
    def __init__(self, internal_mcp):
        from .internal_mcp import InternalMCP
        
        self.internal_mcp: InternalMCP = internal_mcp
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
        enable_internal_mcp = payload.get("headers", {}).get("x-enable-internal-mcp", "0") == "1"
        
        if tools_whitelist:
            tools: list = payload.setdefault('tools', [])
            
            for tool in tools_whitelist:
                tools.append(self.indexed_tools[tool])
        
        if payload.get('tools'):
            xml_tags = self.json_tools_to_xml_prompt(payload['tools'])
            
        messages = payload.get("messages", [])
        
        # 1. Mapeia IDs de tool_calls para nomes de funções (protocolo OpenAI)
        tool_id_to_name = {}
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    tool_id_to_name[tc.get("id")] = tc.get("function", {}).get("name")
                
                msg.pop("tool_calls", None)

        # 2. Converte role: tool para role: user com prefixo [SYSTEM MESSAGE]
        # Isso é necessário para modelos emulados que não entendem o role 'tool' nativo.
        for i, msg in enumerate(messages):
            if msg.get("role") == "tool":
                tool_id = msg.get("tool_call_id")
                tool_name = tool_id_to_name.get(tool_id, "unknown")
                content = msg.get("content", "")
                
                messages[i] = {
                    "role": "user",
                    "content": f"[SYSTEM MESSAGE: TOOL {tool_name} CALLED]: {content}"
                }

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

        if not original_payload.get('stream'):
            # ==============================
            # MODO NÃO-STREAM (SÍNCRONO)
            # ==============================
            message = response["choices"][0].get("message", {})
            raw_content = message.get("content") or ""
            
            # Unescape HTML entities (e.g., &lt; -> <)
            import html
            raw_content = html.unescape(raw_content)
            
            enable_internal_mcp = original_payload.get("headers", {}).get("x-enable-internal-mcp", "0") == "1"
            
            tools_to_execute = []
            
            # Substitui as tags por vazio no texto final e guarda os objetos Tool
            def replacer(match):
                tag_name = match.group(1)
                if tag_name in ("think", "thought", "thinking"):
                    return "" # Remove pensamento sem tratar como ferramenta
                    
                tools_to_execute.append(Tool(name=tag_name, arguments=match.group(2).strip()))
                return ""
                
            clean_content = self.full_tag.sub(replacer, raw_content).strip()
            message["content"] = clean_content if clean_content else None

            # Se não achou nenhuma ferramenta (ou era só reasoning), devolve a resposta original limpa
            if not tools_to_execute:
                return response 

            if not enable_internal_mcp:
                # CONVERSÃO XML -> OPENAI TOOL CALLS (SEM EXECUÇÃO)
                tool_calls = []
                for item in tools_to_execute:
                    args = self._build_args(item.name, item.arguments)
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:10]}",
                        "type": "function",
                        "function": {
                            "name": item.name,
                            "arguments": json.dumps(args)
                        }
                    })
                
                message["tool_calls"] = tool_calls
                message["content"] = clean_content if clean_content else None
                response["choices"][0]["finish_reason"] = "tool_calls"
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
                    message['content'].append({"type": "text", "text": second_content})

            return response
            
        else:
            # ==============================
            # MODO STREAM (O SEU DESIGN)
            # ==============================
            async def stream_orchestrator():
                first_response = ''
                tools_executed = False
                requires_second_call = False
                enable_internal_mcp = original_payload.get("headers", {}).get("x-enable-internal-mcp", "0") == "1"
                tool_index = 0
                
                # O Parser consome a rede neural, nós consumimos o Parser!
                parser = XMLStreamParser(response)
                async for item in parser:
                    if isinstance(item, Chunk):
                        if not item.is_reasoning:
                            first_response += item.content

                        # Only yield if there's actual content or reasoning to show
                        # This avoids the "null" chunks that are just placeholders for stop reasons
                        clean = item.clean_delta()
                        delta = clean.get("choices", [{}])[0].get("delta", {})
                        
                        has_content = delta.get("content") is not None
                        has_tool_calls = delta.get("tool_calls") is not None
                        
                        if has_content or item.is_reasoning or has_tool_calls:
                            yield clean
                        
                    elif isinstance(item, Tool):
                        tools_executed = True
                        
                        original_payload['messages'].append({
                            'role': 'assistant',
                            'content': first_response
                        })
                        
                        first_response = ''
                        
                        if not enable_internal_mcp:
                            # CONVERSÃO XML -> OPENAI TOOL CALLS (DELTA STREAM)
                            call_id = f"call_{uuid.uuid4().hex[:10]}"
                            args = self._build_args(item.name, item.arguments)
                            
                            # Yield 1: Function name
                            name_chunk = {
                                "choices": [{
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [{
                                            "index": tool_index,
                                            "id": call_id,
                                            "type": "function",
                                            "function": {
                                                "name": item.name,
                                                "arguments": ""
                                            }
                                        }]
                                    }
                                }]
                            }
                            if parser.last_metadata: name_chunk.update(parser.last_metadata)
                            yield name_chunk

                            # Yield 2: Function arguments
                            args_chunk = {
                                "choices": [{
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [{
                                            "index": tool_index,
                                            "function": {
                                                "arguments": json.dumps(args)
                                            }
                                        }]
                                    }
                                }]
                            }
                            if parser.last_metadata: args_chunk.update(parser.last_metadata)
                            yield args_chunk

                            tool_index += 1
                            tools_executed = True
                            continue

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
                        status_chunk = {"status_update": display_name}
                        if parser.last_metadata: status_chunk.update(parser.last_metadata)
                        yield status_chunk
                        
                        llm, user = await self.internal_mcp.execute_tool(item.name, args)
                        
                        # Yield end status
                        clear_status = {"status_update": ""} 
                        if parser.last_metadata: clear_status.update(parser.last_metadata)
                        yield clear_status
                        
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

                    await asyncio.sleep(0)
                
                if tools_executed and not enable_internal_mcp:
                    final_chunk = {
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "tool_calls"
                        }]
                    }
                    if parser.last_metadata: final_chunk.update(parser.last_metadata)
                    if parser.last_usage:
                        final_chunk["usage"] = parser.last_usage
                    yield final_chunk

                if tools_executed and requires_second_call:
                    # 3. Dispara o segundo turno imediatamente!
                    async for second_chunk in await brain.chat(original_payload):
                        yield dict(second_chunk)
                        await asyncio.sleep(0)

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