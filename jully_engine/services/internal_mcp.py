import json
import logging
import base64
from typing import Dict, Any, List, AsyncGenerator
from ..model_loader import model_loader
from ..persistence import get_backend
from .external_mcp import external_mcp_manager

logger = logging.getLogger("JulyEngine.InternalMCP")

class InternalMCP:
    def __init__(self):
        self.backend_db = get_backend()

    def get_tools(self) -> List[Dict[str, Any]]:
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
                    "description": "Generates an audio speech from text.",
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
                    "description": "Searches the web for current information, news, facts.",
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
                    "description": "Searches long-term memory for past facts about the user or conversations.",
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
        if not config:
            return {"backend": "api", "model": ""}
        return config

    async def _execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        try:
            if "__" in name:
                # Delegate to external MCP
                return await external_mcp_manager.execute_tool(name, arguments)
                
            if name == "generate_image":
                config = self._get_config_for("IMAGE_CREATE")
                presence = model_loader.get_presence(config.get("backend", "api"), config.get("model", ""))
                
                # Adapt for presence strategy which might expect a payload
                if hasattr(presence, 'generate_image'):
                    # Local presence might not have generate_image yet, fallback to what's available
                    pass
                
                # Using the domain class interface (from bridge process_image_generation)
                # payload should contain prompt
                return await presence._strategy.run_image_gen(config.get("model", ""), arguments.get("prompt", ""))

            elif name == "generate_audio":
                config = self._get_config_for("TTS")
                mouth = model_loader.get_mouth(config.get("backend", "api"), config.get("model", ""))
                # mouth expects payload dict with 'input'
                audio_bytes = await mouth.speak({"input": arguments.get("text", "")})
                if audio_bytes:
                    return base64.b64encode(audio_bytes).decode("utf-8")
                return ""

            elif name == "search_web":
                config = self._get_config_for("WEB_SEARCH")
                world = model_loader.get_world(config.get("backend", "api"), config.get("model", ""))
                return await world.search_web({"query": arguments.get("query", "")})

            elif name == "search_memory":
                config = self._get_config_for("EMBEDDINGS")
                memory = model_loader.get_memory(config.get("backend", "api"), config.get("model", ""))
                return await memory.search(arguments.get("query", ""))

            elif name == "save_memory":
                config = self._get_config_for("EMBEDDINGS")
                memory = model_loader.get_memory(config.get("backend", "api"), config.get("model", ""))
                success = await memory.add_to_rag(arguments.get("fact", ""))
                return "Fact saved successfully to long-term memory." if success else "Failed to save fact."

            elif name == "image_edit":
                config = self._get_config_for("IMAGE_EDIT")
                presence = model_loader.get_presence(config.get("backend", "api"), config.get("model", ""))
                # Usually requires an image, but let's assume it grabs the last image from context or it's passed via some state
                return "Image edit executed." # Placeholder for full implementation

        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            return f"Error executing tool: {str(e)}"
        
        return "Unknown tool."

    async def orchestrate(self, response: Dict[str, Any], original_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronous flow interceptor. If finish_reason == tool_calls, execute and append results,
        possibly returning a multimodal response.
        """
        choice = response.get("choices", [{}])[0]
        finish_reason = choice.get("finish_reason")
        message = choice.get("message", {})
        
        if finish_reason == "tool_calls" and "tool_calls" in message:
            tool_calls = message.get("tool_calls", [])
            content_array = []
            
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name")
                args_str = func.get("arguments", "{}")
                try:
                    args = json.loads(args_str)
                except:
                    args = {}
                    
                result = await self._execute_tool(name, args)
                
                if name == "generate_image":
                    content_array.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{result}"}})
                elif name == "generate_audio":
                    content_array.append({"type": "audio_url", "audio_url": f"data:audio/wav;base64,{result}"})
                else:
                    content_array.append({"type": "text", "text": f"Result from {name}: {result}"})
            
            # Reconstruct the response with the multimodal array
            response["choices"][0]["message"]["content"] = content_array
            response["choices"][0]["finish_reason"] = "stop"

        return response

    async def stream_orchestrate(self, async_generator: AsyncGenerator[str, None], original_payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """
        Wraps the async generator to intercept tool calls.
        """
        tool_call_buffer = {}
        
        async for chunk_str in async_generator:
            # Parse SSE data
            if not chunk_str.startswith("data: "):
                yield chunk_str
                continue
                
            data_str = chunk_str[6:].strip()
            if data_str == "[DONE]":
                # Process any pending tool calls before finishing
                if tool_call_buffer:
                    for tc_idx, tc_data in tool_call_buffer.items():
                        name = tc_data.get("name")
                        args = tc_data.get("arguments", "")
                        try:
                            args_dict = json.loads(args)
                        except:
                            args_dict = {}
                            
                        result = await self._execute_tool(name, args_dict)
                        
                        # Yield a delta with the multimodal output
                        delta_multimodal = {}
                        if name == "generate_image":
                            delta_multimodal = {"image_url": {"url": f"data:image/png;base64,{result}"}}
                        elif name == "generate_audio":
                            delta_multimodal = {"audio_url": f"data:audio/wav;base64,{result}"}
                        else:
                            delta_multimodal = {"content": f"\n\nTool Result ({name}): {result}\n\n"}
                            
                        result_chunk = {
                            "id": "chatcmpl-mcp",
                            "object": "chat.completion.chunk",
                            "model": "internal-mcp",
                            "choices": [{
                                "index": 0,
                                "delta": delta_multimodal,
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(result_chunk)}\n\n"
                        
                yield "data: [DONE]\n\n"
                return
                
            try:
                chunk = json.loads(data_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                
                # Check for tool_calls in delta
                if "tool_calls" in delta:
                    for tc in delta["tool_calls"]:
                        idx = tc.get("index", 0)
                        if idx not in tool_call_buffer:
                            tool_call_buffer[idx] = {"name": "", "arguments": ""}
                            
                        func = tc.get("function", {})
                        if "name" in func and func["name"]:
                            tool_call_buffer[idx]["name"] = func["name"]
                        if "arguments" in func and func["arguments"]:
                            tool_call_buffer[idx]["arguments"] += func["arguments"]
                    
                    # We might skip yielding this tool_call delta to the client to hide it, 
                    # but if we want to be transparent, we could yield it.
                    # For now, we skip yielding raw tool calls to simplify frontend.
                    continue
                    
                yield chunk_str
                
            except Exception as e:
                logger.error(f"InternalMCP: Error parsing chunk: {e}")
                yield chunk_str
