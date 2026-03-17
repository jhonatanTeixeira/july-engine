import asyncio
import json
import logging
import base64
from pprint import pprint
from typing import Dict, Any, List, AsyncGenerator, Tuple, Union

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
            return {"type": "text", "text": self._data}


class InternalMCP:
    def __init__(self):
        self.backend_db = get_backend()

    def get_tools(self) -> List[Dict[str, Any]]:
        # O Cardápio de Ferramentas Internas (mantido intacto)
        internal_tools = [
            {
                "type": "function",
                "function": {
                    "name": "generate_image",
                    "description": "Generates a new image based on a prompt.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "A detailed description in English of the image to be generated."
                            }
                        },
                        "required": ["prompt"]
                    }
                }
            },
            {
                "type": "function",
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
                    "description": "Searches the web for current information, news, facts. Always use it in case the user asks for latest news and facts",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query."
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
                    "description": "Searches long-term memory for past facts about the user or conversations. Always use to remember facts about the user when its important to make the conmversation more natural to the user",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The concept or fact to search in memory."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "save_memory",
                    "description": "Saves an important fact or piece of information about the user or the conversation to the long-term memory vector database.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fact": {
                                "type": "string",
                                "description": "The specific fact or context to remember."
                            }
                        },
                        "required": ["fact"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "image_edit",
                    "description": "Edits an existing image based on an instruction.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "instruction": {
                                "type": "string",
                                "description": "Instruction in English detailing what to edit in the image."
                            }
                        },
                        "required": ["instruction"]
                    }
                }
            }
        ]
        
        external_tools = external_mcp_manager.get_all_tools()
        return internal_tools + external_tools

    def _get_config_for(self, setting_key: str) -> Dict[str, Any]:
        config = self.backend_db.get_setting(setting_key)
        return config if config else {"backend": "api", "model": ""}
    
    def inject_tools(self, payload: Dict):
        payload['tools'] = self.get_tools()

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
                
                return (
                    result,
                    None
                )
                
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
                # response = await model_loader.get_presence(backend, model)._strategy.run_image_gen(model, arguments.get("prompt", ""))
                response = await bridge.process_image_generation({"prompt": arguments.get("prompt")}, {})
                
                gen_time = time.time() - start_time
                event_manager.emit(f"mcp_{name}", generation_time=gen_time)
                
                return (
                    "Image generated",
                    UserToolReponse(response, 'image')
                )
            
            elif name == "generate_audio":
                audio_bytes = bridge.process_tts({"text", arguments.get("text")}, {})
                audio = base64.b64encode(audio_bytes).decode("utf-8") if audio_bytes else ""
                
                gen_time = time.time() - start_time
                event_manager.emit(f"mcp_{name}", generation_time=gen_time)
                
                return (
                    "Audio generated",
                    UserToolReponse(audio, 'audio')
                )
            
            elif name == "search_web":
                res = await bridge.process_search_web({"query": arguments.get("query", "")}, {})
                gen_time = time.time() - start_time
                event_manager.emit(f"mcp_{name}", generation_time=gen_time)
                return (
                    res,
                    None
                )
            
            elif name == "search_memory":
                res = await model_loader.get_memory(backend, model).search('query: ' + arguments.get("query", ""))
                gen_time = time.time() - start_time
                event_manager.emit(f"mcp_{name}", generation_time=gen_time)
                return (
                    res,
                    None
                )
            
            elif name == "save_memory":
                success = await model_loader.get_memory(backend, model).add_to_rag('passage: ' + arguments.get("fact", ""))
                gen_time = time.time() - start_time
                event_manager.emit(f"mcp_{name}", generation_time=gen_time)
                return (
                    "Fact saved successfully." if success else "Failed to save fact.",
                    None
                )
            
            elif name == "image_edit":
                response = await bridge.process_image_edit({
                    "prompt": arguments.get("instruction", ""),
                    "image": arguments.get("image", "")
                }, {})
                gen_time = time.time() - start_time
                event_manager.emit(f"mcp_{name}", generation_time=gen_time)
                return (
                    "Image edited",
                    UserToolReponse(response, 'image')
                )

        except Exception as e:
            logger.error(f"InternalMCP: Erro na tool {name}: {e}")
            return (
                f"Error executing tool: {str(e)}",
                None
            )

    async def stream_orchestrate(self, response: AsyncGenerator[Dict, None], brain_instance, original_payload: Dict[str, Any]) -> AsyncGenerator[Dict, None]:
        is_calling = False
        tools = {}
        assistant_content = ''
        tool_messages = []
        assistant_tool_calls = []
        
        async for chunk in response:
            assistant_content += chunk.get('choices')[0].get('delta', {}).get('content', '')
             
            if tool_calls := chunk.get('choices')[0].get('delta', {}).get('tool_calls', None):
                is_calling = True
                
                for tool_call in tool_calls:
                    if name := tool_call.get('function', {}).get('name', None):
                        tools.setdefault(name, {"arguments": "", "response": [], "id": tool_call.get("id", None)})
                    else:
                        tools.get(name)["arguments"] += tool_call.get("function", {}).get('arguments', '')
                
                continue
            
            if is_calling:
                for name, tool in tools.items():
                    tool["response"] = await self.execute_tool(name, json.loads(tool['arguments']))
                
                is_calling = False
            
            if chunk.get('choices')[0].get('finish_reason') == 'tool_calls':
                for name, tool in tools.items():
                    llm, user = tool.get("response")
                    
                    tool_messages.append({
                        "role": "tool",
                        "tool_call_id": tool.get('id'),
                        "name": name,
                        "content": llm
                    })
                    
                    assistant_tool_calls.append({
                        "id": tool.get('id'),
                        "type": 'function',
                        "function": {
                            "name": name,
                            "arguments": tool['arguments']
                        }
                    })
                    
                    if user:
                        yield user.delta
                        await asyncio.sleep(0)
    
                original_payload['messages'].append({
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": assistant_tool_calls
                })
                
                original_payload['messages'].extend(tool_messages)
                
                async for chunk in brain_instance.chat(original_payload):
                    yield chunk
                    await asyncio.sleep(0)

            yield chunk
            await asyncio.sleep(0)
        
    # --- UTILITÁRIO ---
    async def orchestrate(self, response: Union[Dict[str, Any], AsyncGenerator], brain, original_payload: Dict[str, Any]) -> Union[Dict[str, Any], AsyncGenerator]:
        if isinstance(response, AsyncGenerator):
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
            
            for tc in message.get("tool_calls", []):
                name = tc.get("function", {}).get("name")
                args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                
                llm, user = await self.execute_tool(name, args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id"),
                    "name": name,
                    "content": str(llm)
                })

                if user:
                    multimodal_content.append(user._response)

            second_response = await brain.chat(original_payload)
            second_content = second_response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if isinstance(second_content, list):
                multimodal_content.extend(second_content)
            else:
                multimodal_content.append(second_content)
                
            content = multimodal_content if len(multimodal_content) > 1 else multimodal_content[0]
            
            second_response.setdefault("choices", [{}])[0].setdefault("message", {})['content'] = content

            return second_response

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