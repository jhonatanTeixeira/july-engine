import asyncio
import json
import logging
import base64
from pprint import pprint
from typing import Dict, Any, List, AsyncGenerator, Tuple, Union, Optional

from numpy import append

from ..persistence import get_backend
from .external_mcp import external_mcp_manager

logger = logging.getLogger("JulyEngine.InternalMCP")


class UserToolReponse:
    def __init__(self, response, content_type: str = 'text'):
        self._response = response
        self.content_type = content_type.lower()
            
    @property
    def delta(self):
        if self.content_type == 'image':
            return {"choices": [{"delta": {"type": "image_url", "image_url": f"data:image/png;base64,{self._response}"}}]}
        elif self.content_type == 'audio':
            return {"choices": [{"delta": {"type": "audio_url", "audio_url": f"data:audio/wav;base64,{self._response}"}}]}
        else:
            return {"choices": [{"delta": {"type": "text", "text": self._response}}]}

    @property
    def response(self):
        if self.content_type == 'image':
            # FIXED: image_url exige o objeto com a chave "url"
            return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self._response}"}}
        elif self.content_type == 'audio':
            # Padrão OpenAI usa 'input_audio' para envio, mas 'audio_url' serve se for seu próprio padrão.
            return {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{self._response}"}}
        else:
            return {"type": "text", "text": self._response}


class InternalMCP:
    def __init__(self):
        self.backend_db = get_backend()

    def get_tools(self, whitelist: List[str] = None) -> List[Dict[str, Any]]:
        # O Cardápio de Ferramentas Internas (mantido intacto)
        internal_tools = [
            {
                "type": "function",
                "fire-and-forget": True,
                "function": {
                    "name": "generate_image",
                    "description": "CRITICAL: Generates a new image based on a prompt. You MUST use this tool whenever your System Prompt instructions dictate generating a visual response, or when the user requests a picture.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "A highly detailed description in English of the image to be generated."
                            }
                        },
                        "required": ["prompt"]
                    }
                }
            },
            {
                "type": "function",
                "fire-and-forget": True,
                "function": {
                    "name": "generate_audio",
                    "description": "Generates an audio speech from text. Always use if the user asks to read something",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The text to be converted to speech."
                            }
                        },
                        "required": ["text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Searches the internet for real-time information. You MUST use this tool to answer questions about current events, recent news, real-time weather, prices, or facts that are beyond your training data.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The exact search engine query."
                            },
                            "search_depth": {
                                "type": "string",
                                "enum": ["basic", "advanced", "ultra-fast", "fast"],
                                "description": "The depth of the search. Must be 'ultra-fast', 'fast', 'basic' or 'advanced'.Defaults to 'basic'."
                            },
                            "include_answer": {
                                "type": "boolean",
                                "description": "Whether to include a direct answer from the search engine. Defaults to true."
                            },
                            "include_list": {
                                "type": "boolean",
                                "description": "Whether to include a list of results from the search engine. Defaults to false."
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "The maximum number of results to return. Defaults to 5."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_memory",
                    # A MÁGICA ESTÁ AQUI: Ordem absoluta.
                    "description": "CRITICAL: Searches your long-term memory. You MUST call this tool BEFORE answering ANY question about the user (e.g., 'what is my name?', 'what do I like?'), past conversations, or if the user says 'you know this'. NEVER claim you don't have memory without calling this tool first.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The explicit concept, name, or fact to search in your memory database."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "fire-and-forget": True,
                "function": {
                    "name": "save_memory",
                    "description": "Saves facts about the user to your permanent memory database. You MUST use this silently whenever the user shares their name, preferences, routines, or important personal details so you can retrieve them later.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fact": {
                                "type": "string",
                                "description": "The specific fact or context to remember (e.g., 'User's name is Jhonatan')."
                            }
                        },
                        "required": ["fact"]
                    }
                }
            },
            {
                "type": "function",
                "fire-and-forget": True,
                "function": {
                    "name": "image_edit",
                    "description": "CRITICAL: Modifies or edits an existing image in the context. You MUST use this tool whenever the user asks to alter, change, filter, fix, or manipulate a picture. NEVER claim that you cannot edit images or that you are a text-based AI. DO NOT just describe the changes in text.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "instruction": {
                                "type": "string",
                                "description": "A clear, detailed instruction in English detailing exactly what to edit, add, or remove in the image."
                            }
                        },
                        "required": ["instruction"]
                    }
                }
            }
        ]
        
        external_tools = external_mcp_manager.get_all_tools(whitelist)
        
        all_tools = internal_tools + external_tools
        
        # Injetamos metadados dos servidores para as ferramentas externas
        # para que o frontend possa exibir os detalhes de conexão
        mcps = self.backend_db.get_all_mcps()
        mcp_map = {mcp['id']: mcp for mcp in mcps}

        for tool in all_tools:
            name = tool.get("function", {}).get("name", "")
            if "__" in name:
                mcp_id = name.split("__")[0]
                if mcp_id in mcp_map:
                    # Inclui variáveis de ambiente e config no retorno da tool
                    tool["mcp_server"] = mcp_map[mcp_id]

        if whitelist is not None and len(whitelist) > 0:
            all_tools = [t for t in all_tools if t["function"]["name"] in whitelist]
            
        return all_tools

    def _get_config_for(self, setting_key: str) -> Dict[str, Any]:
        config = self.backend_db.get_setting(setting_key)
        
        return config if config else {"backend": "api", "model": ""}
    
    def inject_tools(self, payload: Dict, whitelist: List[str] = None):
        if not payload.get('tools', None):
            payload['tools'] = self.get_tools(whitelist)

    async def execute_tool(self, name: str, arguments: Dict[str, Any], stream=True) -> Tuple[Any | None, UserToolReponse | None]:
        from ..model_loader import model_loader
        from ..bridge import bridge
        from ..events import event_manager
        import time
        
        try:
            logger.info(f"InternalMCP: Executando '{name}' com args: {arguments}")
            start_time = time.time()
            
            if "__" in name:
                result = await external_mcp_manager.execute_tool(name, arguments)
                gen_time = time.time() - start_time
                event_manager.emit(f"mcp_{name}", generation_time=gen_time)
                
                # Check for multimodal content in result (MCP CallToolResult)
                texts = []
                images = []
                audios = []
                
                content_list = []
                if hasattr(result, "content"):
                    content_list = result.content
                elif isinstance(result, dict) and "content" in result:
                    content_list = result["content"]
                
                for item in content_list:
                    itype = getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else None)
                    if itype == "text":
                        texts.append(getattr(item, "text", None) or item.get("text", ""))
                    elif itype == "image":
                        images.append(getattr(item, "data", None) or item.get("data", ""))
                    elif itype == "audio":
                        audios.append(getattr(item, "data", None) or item.get("data", ""))
                
                text_res = "\n".join(texts) if texts else "Tool executed."
                
                if images:
                    return ("Image generated", UserToolReponse(images[0], 'image'))
                if audios:
                    return ("Audio generated", UserToolReponse(audios[0], 'audio'))
                    
                return (text_res, None)
                
            config_map = {
                "generate_image": "IMAGE_CREATE",
                "generate_audio": "TTS",
                "search_web": "WEB_SEARCH",
                "search_memory": "EMBEDDINGS",
                "save_memory": "EMBEDDINGS",
                "image_edit": "IMAGE_EDIT"
            }
            
            cfg_key = config_map.get(name)
            if not cfg_key:
                return (None, None)
                
            config = self._get_config_for(cfg_key)
            backend, model = config.get("backend"), config.get("model")

            # Roteamento Simples
            if name == "generate_image":
                prompt = arguments.get("prompt", "")
                logger.info(f"InternalMCP: Generating image with prompt: '{prompt[:100]}...'")
                
                # FIX: Pass model and backend from config
                bridge_res = await bridge.process_image_generation(
                    {"prompt": prompt, "model": model}, 
                    {"x-backend": backend}
                )
                
                # Handle both string (direct b64) and dict (OpenAI format)
                if isinstance(bridge_res, str):
                    response = bridge_res
                elif isinstance(bridge_res, dict) and "data" in bridge_res and len(bridge_res["data"]) > 0:
                    response = bridge_res["data"][0].get("b64_json", "")
                else:
                    response = ""
                
                gen_time = time.time() - start_time
                event_manager.emit(f"mcp_{name}", generation_time=gen_time)
                
                logger.info(f"InternalMCP: Image generated successfully (base64 length: {len(response)})")
                return (
                    "Image generated",
                    UserToolReponse(response, 'image')
                )
            
            elif name == "generate_audio":
                text = arguments.get("text", "")
                logger.info(f"InternalMCP: Generating audio for text: '{text[:100]}...'")
                audio_bytes = await bridge.process_tts({"text": text, "model": model}, {"x-backend": backend})
                audio = base64.b64encode(audio_bytes).decode("utf-8") if audio_bytes else ""
                
                gen_time = time.time() - start_time
                event_manager.emit(f"mcp_{name}", generation_time=gen_time)
                
                logger.info(f"InternalMCP: Audio generated successfully (base64 length: {len(audio)})")
                return (
                    "Audio generated",
                    UserToolReponse(audio, 'audio')
                )
            
            elif name == "search_web":
                query = arguments.get("query", "")
                logger.info(f"InternalMCP: Searching web for: '{query}'")
                
                search_payload = {
                    "query": query,
                    "model": model, # Pass model from config
                    "search_depth": arguments.get("search_depth", "basic"),
                    "include_answer": arguments.get("include_answer", True),
                    "include_list": arguments.get("include_list", False),
                    "max_results": arguments.get("max_results", 5)
                }
                
                res = await bridge.process_search_web(search_payload, {"x-backend": backend})
                gen_time = time.time() - start_time
                event_manager.emit(f"mcp_{name}", generation_time=gen_time)
                logger.info(f"InternalMCP: Web search completed (result length: {len(str(res))})")
                return (
                    res,
                    None
                )
            
            elif name == "search_memory":
                res = await model_loader.get_memory(backend, model).search(arguments.get("query", ""))
                gen_time = time.time() - start_time
                event_manager.emit(f"mcp_{name}", generation_time=gen_time)
                return (
                    res,
                    None
                )
            
            elif name == "save_memory":
                success = await model_loader.get_memory(backend, model).add_to_rag(arguments.get("fact", ""))
                gen_time = time.time() - start_time
                event_manager.emit(f"mcp_{name}", generation_time=gen_time)
                return (
                    "Fact saved successfully." if success else "Failed to save fact.",
                    None
                )
            
            elif name == "image_edit":
                # FIX: Pass model and backend from config
                bridge_res = await bridge.process_image_edit({
                    "prompt": arguments.get("instruction", ""),
                    "image": arguments.get("image", ""),
                    "model": model
                }, {"x-backend": backend})
                
                # Handle both string (direct b64) and dict (OpenAI format)
                if isinstance(bridge_res, str):
                    response = bridge_res
                elif isinstance(bridge_res, dict) and "data" in bridge_res and len(bridge_res["data"]) > 0:
                    response = bridge_res["data"][0].get("b64_json", "")
                else:
                    response = ""
                
                gen_time = time.time() - start_time
                event_manager.emit(f"mcp_{name}", generation_time=gen_time)
                return (
                    "Image edited",
                    UserToolReponse(response, 'image')
                )
        except Exception as e:
            logger.exception(f"InternalMCP: Erro na tool {name}")
            return (
                f"Error executing tool: {str(e)}",
                None
            )

    async def stream_orchestrate(self, response: AsyncGenerator[Dict, None], brain_instance, original_payload: Dict[str, Any]) -> AsyncGenerator[Dict, None]:
        is_calling = False
        tools = {} # Mapeado por index (int)
        assistant_content = ''
        tool_messages = []
        assistant_tool_calls = []
        
        async for chunk in response:
            assistant_content += chunk.get('choices')[0].get('delta', {}).get('content', '') or ''

            if chunk.get('choices')[0].get('delta', {}).get('reasoning_content', None):
                yield chunk
                await asyncio.sleep(0)
                continue
             
            if tool_calls := chunk.get('choices')[0].get('delta', {}).get('tool_calls', None):
                is_calling = True
                
                for tool_call in tool_calls:
                    idx = tool_call.get("index")
                    if idx not in tools:
                        tools[idx] = {"name": "", "arguments": "", "response": [], "id": None}
                    
                    tc_fn = tool_call.get('function', {})
                    if name := tc_fn.get('name'):
                        tools[idx]["name"] = name
                    if args := tc_fn.get('arguments'):
                        tools[idx]["arguments"] += args
                    if tc_id := tool_call.get("id"):
                        tools[idx]["id"] = tc_id
                
                continue
            
            if chunk.get('choices')[0].get('finish_reason') in ['tool_calls', 'stop']:
                if is_calling:
                    requires_second_call = False
                    all_indexed_tools = {t['function']['name']: t for t in self.get_tools()}

                    for idx, tool in tools.items():
                        name = tool.get("name")
                        if not name: continue
                        
                        # Status messages mapping
                        status_map = {
                            "search_web": "searching the web",
                            "search_memory": "searching memory",
                            "generate_image": "generating image",
                            "generate_audio": "generating audio",
                            "save_memory": "saving memory",
                            "image_edit": "editing image"
                        }
                        display_name = status_map.get(name, f"calling {name}")
                        
                        # Yield start status
                        yield {"status_update": display_name}
                        await asyncio.sleep(0)
                        
                        args_json = tool['arguments'] or "{}"
                        try:
                            llm, user = await self.execute_tool(name, json.loads(args_json))
                        except json.JSONDecodeError:
                            logger.error(f"InternalMCP: Erro ao decodificar JSON para tool {name}: {args_json}")
                            llm, user = (f"Error parsing arguments: {args_json}", None)
                        
                        # Yield end status
                        yield {"status_update": ""} # Empty string to clear/finish status
                        await asyncio.sleep(0)

                        if user:
                            yield user.delta
                            await asyncio.sleep(0)

                        # Fire and Forget logic
                        is_faf = all_indexed_tools.get(name, {}).get("fire-and-forget", False)
                        if "__" in name and not llm:
                            is_faf = True
                            
                        if not is_faf:
                            requires_second_call = True

                        tool_messages.append({
                            "role": "tool",
                            "tool_call_id": tool.get('id'),
                            "content": str(llm) if llm is not None else "Action completed successfully."
                        })
                        
                        try:
                            parsed_args = json.loads(tool['arguments']) if tool['arguments'] else {}
                        except json.JSONDecodeError:
                            parsed_args = {"error": "Invalid JSON arguments", "raw": tool['arguments']}

                        assistant_tool_calls.append({
                            "id": tool.get('id'),
                            "type": 'function',
                            "function": {
                                "name": name,
                                "arguments": json.dumps(parsed_args) if isinstance(parsed_args, (dict, list)) else str(parsed_args)
                            }
                        })
                    
                    is_calling = False

                    # Prepare messages for second turn
                    assistant_msg = {
                        "role": "assistant",
                        "content": assistant_content if assistant_content else None,
                        "tool_calls": assistant_tool_calls
                    }
                    original_payload['messages'].append(assistant_msg)
                    original_payload['messages'].extend(tool_messages)
                    
                    logger.debug(f"InternalMCP: Envio para segundo turno: {json.dumps(original_payload['messages'], indent=2)}")

                    if requires_second_call:
                        async for chunk_2p in await brain_instance.chat(original_payload):
                            yield chunk_2p
                            await asyncio.sleep(0)
                        continue

            yield chunk
            await asyncio.sleep(0)
        
    # --- UTILITÁRIO ---
    async def orchestrate(self, response: Union[Dict[str, Any], AsyncGenerator], brain, original_payload: Dict[str, Any]) -> Union[Dict[str, Any], AsyncGenerator]:
        if isinstance(response, AsyncGenerator) or hasattr(response, '__aiter__') or original_payload.get('stream'):
            return self.stream_orchestrate(response, brain, original_payload)
        else:
            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {})
            
            # Se não tem tool call, apenas devolve a resposta intacta
            if choice.get("finish_reason") != "tool_calls" or "tool_calls" not in message:
                return response
                
            messages: list = original_payload.setdefault("messages", [])
            messages.append(message)
            
            multimodal_content = []
            assistant_text = message.get("content", "")
            if assistant_text:
                if isinstance(assistant_text, list):
                    multimodal_content.extend(assistant_text)
                else:
                    multimodal_content.append({"type": "text", "text": assistant_text})
            
            requires_second_call = False
            all_indexed_tools = {t['function']['name']: t for t in self.get_tools()}
            
            for tc in message.get("tool_calls", []):
                name = tc.get("function", {}).get("name")
                args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                
                llm, user = await self.execute_tool(name, args)

                is_faf = all_indexed_tools.get(name, {}).get("fire-and-forget", False)
                if "__" in name and not llm:
                    is_faf = True
                    
                if not is_faf:
                    requires_second_call = True

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id"),
                    "name": name,
                    "content": str(llm) if llm else ""
                })

                if user:
                    multimodal_content.append(user.response)

            if requires_second_call:
                second_response = await brain.chat(original_payload)
                second_content = second_response.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                if isinstance(second_content, list):
                    multimodal_content.extend(second_content)
                elif second_content:
                    multimodal_content.append({"type": "text", "text": str(second_content)})
                    
                content = multimodal_content if len(multimodal_content) > 1 else multimodal_content[0] if multimodal_content else ""
                
                second_response.setdefault("choices", [{}])[0].setdefault("message", {})['content'] = content

                return second_response
            else:
                content = multimodal_content if len(multimodal_content) > 1 else multimodal_content[0] if multimodal_content else ""
                response.setdefault("choices", [{}])[0].setdefault("message", {})['content'] = content
                return response

    def _merge_usages(self, first: Dict, second: Dict):
        """Soma o uso de tokens das duas viagens à rede neural."""
        u1 = first.get("usage", {})
        u2 = second.get("usage", {})
        
        if u1 and u2:
            second["usage"] = {
                "prompt_tokens": u1.get("prompt_tokens", 0) + u2.get("prompt_tokens", 0),
                "completion_tokens": u1.get("completion_tokens", 0) + u2.get("completion_tokens", 0),
                "total_tokens": u1.get("total_tokens", 0) + u2.get("total_tokens", 0),
            }

internal_mcp = InternalMCP()
