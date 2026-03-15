import re
import html
import json
from dataclasses import dataclass
import textwrap
from typing import Any, Dict, AsyncGenerator, List, Union

# ==========================================
# 1. CLASSES DE DOMÍNIO (DATA E PARSER)
# ==========================================

@dataclass
class Tool:
    name: str
    arguments: str

class Chunk:
    def __init__(self, raw_chunk: dict, content: str = "", is_reasoning: bool = False):
        self.raw_chunk = raw_chunk
        self.content = content
        self.is_reasoning = is_reasoning

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
        async for chunk in self.stream:
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            
            # 1. Raciocínio (Passa direto, sem ir pro buffer)
            if "reasoning_content" in delta:
                yield Chunk(chunk, content=delta["reasoning_content"], is_reasoning=True)
                continue
                
            # 2. Conteúdo (Vai pro buffer de análise)
            raw_text = delta.get("content", "")
            if not raw_text:
                continue
                
            self.buffer += html.unescape(raw_text)
            
            # 3. Procura ferramentas prontas
            while True:
                match = self.full_tag.search(self.buffer)
                if match:
                    safe_text = self.buffer[:match.start()]
                    if safe_text:
                        yield Chunk(chunk, content=safe_text, is_reasoning=False)
                    
                    yield Tool(name=match.group(1), arguments=match.group(2).strip())
                    self.buffer = self.buffer[match.end():]
                else:
                    break

            # 4. Trava de segurança (Prende fragmentos suspeitos de '<')
            trap_start = -1
            open_match = self.open_tag.search(self.buffer)
            if open_match:
                trap_start = open_match.start()
            else:
                suspect_match = self.suspect_tag.search(self.buffer)
                if suspect_match:
                    trap_start = suspect_match.start()

            # 5. Libera o texto seguro pra tela
            if trap_start != -1:
                safe_text = self.buffer[:trap_start]
                if safe_text:
                    yield Chunk(chunk, content=safe_text, is_reasoning=False)
                self.buffer = self.buffer[trap_start:]
            else:
                if self.buffer:
                    yield Chunk(chunk, content=self.buffer, is_reasoning=False)
                    self.buffer = ""


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
                
            line = f"* <{name}>{param_name}</{name}> -> {description}"
            prompt_lines.append(line)
            
        return "\n".join(prompt_lines)
    
    def inject_tools(self, payload: Dict):
        system_prompt = textwrap.dedent(f'''
# TOOLING
You are an intelligent agent equipped with external tools. 

## 🛑 ANTI-REFUSAL PROTOCOL (CRITICAL)
NEVER say "I cannot create images", "I cannot save memories", or "As an AI, I don't have access to...". 
You DO have access to these capabilities through your tools. The external system will execute the tags you provide. Just output the correct XML tag and the system will do the rest.

## ⚠️ CRITICAL SYNTAX AND EXECUTION RULES (MANDATORY)
1. STRICT SYNTAX: Use ONLY less-than/greater-than signs for tags. It is STRICTLY FORBIDDEN to use square brackets (e.g., `[search]`).
2. NO MARKDOWN IN TAGS: NEVER wrap XML tags in code blocks (```). Insert them as plain text.
3. REASONING ISOLATION: The system DOES NOT read your internal `<think>` block. To execute a tool, the XML tag MUST be written in your FINAL RESPONSE.
4. SILENT EXECUTION: Do not explain that you are going to use a tool. Just output the tag.

## TOOL AND ACTION GUIDELINES
{self.xml_tags}

## FEW-SHOT EXAMPLES (HOW TO RESPOND)
User: "Create a picture of a blue mustang."
Assistant: <generate_image>A beautiful blue Ford Mustang, realistic, 8k</generate_image>

User: "Remember that I love muscle cars."
Assistant: <save_memory>The user loves muscle cars.</save_memory> I will remember that!

User: "What is the weather in Tokyo?"
Assistant: <search_web>current weather in Tokyo</search_web>
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
        
        # Garante que acessa as ferramentas indexadas (ajuste se vier do internal_mcp)
        indexed_tools = self.indexed_tools
        
        if name in indexed_tools:
            props = indexed_tools[name]["function"].get("parameters", {}).get("properties", {})
            if props:
                # Pega dinamicamente o nome do primeiro parâmetro exigido pela ferramenta
                first_param = list(props.keys())[0]
                args[first_param] = value
        else:
            # Fallback genérico caso a ferramenta não seja encontrada no index
            args["VALOR"] = value
            
        return args

    async def orchestrate(self, response: Union[Dict, AsyncGenerator], brain, original_payload: Dict):

        if isinstance(response, Dict):
            # ==============================
            # MODO NÃO-STREAM (SÍNCRONO)
            # ==============================
            message = response["choices"][0].get("message", {})
            raw_content = message.get("content") or ""
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

            # 1. Monta o histórico do Turno 1 (O texto limpo + as tags que a IA gerou)
            reconstructed_assistant = clean_content
            
            for t in tools_to_execute:
                reconstructed_assistant += f"\n<{t.name}>{t.arguments}</{t.name}>"
                
            original_payload['messages'].append({
                "role": "assistant", 
                "content": reconstructed_assistant.strip()
            })
            
            multimodal_content = []
            
            # 2. Executa todas as ferramentas sequencialmente
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
                
                # Guarda o visual (UI) para o final
                if user:
                    multimodal_content.append(user)
                # Guarda o texto para o LLM ler no segundo turno
                if llm:
                    original_payload['messages'].append({
                        "role": "user", 
                        "content": f"[SYSTEM MESSAGE TOOL {item.name} CALLED]: {llm}"
                    })

            # 3. Dispara o Turno 2 (ReAct Loop)
            original_payload.setdefault("headers", {})["x-enable-internal-mcp"] = "0"
            
            second_response = await brain.chat(original_payload)
            second_content = second_response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # 4. Anexa o conteúdo multimodal (ex: Base64 gerado pela tool) na resposta de texto final
            if len(multimodal_content) > 0:
                if isinstance(second_content, str):
                    multimodal_content.append({"type": "text", "text": second_content})
                else:
                    multimodal_content.extend(second_content)
                    
                second_response.setdefault("choices", [{}])[0].setdefault("message", {})['content'] = multimodal_content

            return second_response
            
        else:
            # ==============================
            # MODO STREAM (O SEU DESIGN)
            # ==============================
            async def stream_orchestrator():
                first_response = ''
                tools_executed = False
                
                # O Parser consome a rede neural, nós consumimos o Parser!
                async for item in XMLStreamParser(response):
                    if isinstance(item, Chunk):
                        first_response += item.content
                        # Joga na tela!
                        yield item.delta
                        
                    elif isinstance(item, Tool):
                        tools_executed = True
                        first_response += f"<{item.name}>{item.arguments}</{item.name}>"
                        
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
                                    
                        llm, user = await self.internal_mcp.execute_tool(item.name, args)
                        
                        if user:
                            yield user
                            
                        if llm:
                            # 2. Injeta o resultado do LLM (Tavily, etc)
                            original_payload["messages"].append({
                                "role": "user", 
                                "content": f"[SYSTEM MESSAGE: TOOL {item.name} CALLED]: {llm}"
                            })
                
                if tools_executed:
                    original_payload.setdefault("headers", {})["x-enable-internal-mcp"] = "0"
                                
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