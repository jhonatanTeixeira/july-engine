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

    async def orchestrate(self, brain_instance, response: Dict[str, Any], original_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronous flow interceptor. If finish_reason == tool_calls, execute and append results.
        If it's a text result (like search), loop back to the LLM to get the final answer,
        accumulating the token usage.
        """
        choice = response.get("choices", [{}])[0]
        finish_reason = choice.get("finish_reason")
        message = choice.get("message", {})
        
        # Accumulators for usage
        total_prompt_tokens = response.get("usage", {}).get("prompt_tokens", 0)
        total_completion_tokens = response.get("usage", {}).get("completion_tokens", 0)
        
        if finish_reason == "tool_calls" and "tool_calls" in message:
            tool_calls = message.get("tool_calls", [])
            
            # Need to figure out if any tool returns multimodal or just text
            # For multimodal (image/audio), we just append and return.
            # For text (search), we append as tool response and call LLM again.
            
            has_text_result = False
            messages = original_payload.get("messages", [])
            messages.append(message) # Append assistant's tool call message
            
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
                    has_text_result = True
                    # Append tool result message
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "name": name,
                        "content": str(result)
                    })
            
            if has_text_result and not content_array:
                # Do a second LLM call with the tool results
                new_payload = dict(original_payload)
                new_payload["messages"] = messages
                
                # Execute second call (ensure internal_mcp recursion is disabled or handled)
                new_payload["headers"] = dict(original_payload.get("headers", {}))
                new_payload["headers"]["x-enable-internal-mcp"] = "0" # Disable for second pass to prevent infinite loop
                
                second_response = await brain_instance.chat(new_payload)
                
                if isinstance(second_response, dict):
                    # Accumulate usage
                    second_usage = second_response.get("usage", {})
                    second_response["usage"] = {
                        "prompt_tokens": total_prompt_tokens + second_usage.get("prompt_tokens", 0),
                        "completion_tokens": total_completion_tokens + second_usage.get("completion_tokens", 0),
                        "total_tokens": total_prompt_tokens + total_completion_tokens + second_usage.get("total_tokens", 0)
                    }
                    return second_response
                return second_response

            # If it was multimodal or mixed
            response["choices"][0]["message"]["content"] = content_array
            response["choices"][0]["finish_reason"] = "stop"

        return response

    async def stream_orchestrate(self, brain_instance, async_generator: AsyncGenerator[str, None], original_payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """
        Wraps the async generator to intercept tool calls.
        If a text tool is called, execute it and restart a stream for the final answer.
        """
        tool_call_buffer = {}
        final_usage = None
        assistant_message = {"role": "assistant", "content": "", "tool_calls": []}
        
        async for chunk_str in async_generator:
            if not chunk_str.startswith("data: "):
                yield chunk_str
                continue
                
            data_str = chunk_str[6:].strip()
            if data_str == "[DONE]":
                if tool_call_buffer:
                    # Reconstruct tool_calls array for the assistant message
                    for idx, tc in tool_call_buffer.items():
                        assistant_message["tool_calls"].append({
                            "id": tc.get("id", f"call_{idx}"),
                            "type": "function",
                            "function": {
                                "name": tc.get("name"),
                                "arguments": tc.get("arguments")
                            }
                        })
                        
                    has_text_result = False
                    multimodal_deltas = []
                    messages = original_payload.get("messages", [])
                    messages.append(assistant_message)
                    
                    for tc_idx, tc_data in tool_call_buffer.items():
                        name = tc_data.get("name")
                        args = tc_data.get("arguments", "")
                        tc_id = tc_data.get("id", f"call_{tc_idx}")
                        try:
                            args_dict = json.loads(args)
                        except:
                            args_dict = {}
                            
                        result = await self._execute_tool(name, args_dict)
                        
                        if name == "generate_image":
                            multimodal_deltas.append({"image_url": {"url": f"data:image/png;base64,{result}"}})
                        elif name == "generate_audio":
                            multimodal_deltas.append({"audio_url": f"data:audio/wav;base64,{result}"})
                        else:
                            has_text_result = True
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "name": name,
                                "content": str(result)
                            })
                            
                    if has_text_result and not multimodal_deltas:
                        # ReAct loop: stream the second response
                        new_payload = dict(original_payload)
                        new_payload["messages"] = messages
                        new_payload["headers"] = dict(original_payload.get("headers", {}))
                        new_payload["headers"]["x-enable-internal-mcp"] = "0"
                        
                        second_stream = await brain_instance.chat(new_payload)
                        
                        # Accumulate tokens from the first stream into the final usage block
                        first_prompt_tokens = final_usage.get("prompt_tokens", 0) if final_usage else 0
                        first_completion_tokens = final_usage.get("completion_tokens", 0) if final_usage else 0
                        
                        async for second_chunk in second_stream:
                            if second_chunk.startswith("data: ") and second_chunk.strip() != "data: [DONE]":
                                try:
                                    s_data = json.loads(second_chunk[6:])
                                    if "usage" in s_data and s_data["usage"]:
                                        s_usage = s_data["usage"]
                                        s_data["usage"] = {
                                            "prompt_tokens": first_prompt_tokens + s_usage.get("prompt_tokens", 0),
                                            "completion_tokens": first_completion_tokens + s_usage.get("completion_tokens", 0),
                                            "total_tokens": first_prompt_tokens + first_completion_tokens + s_usage.get("total_tokens", 0)
                                        }
                                        yield f"data: {json.dumps(s_data)}\n\n"
                                        continue
                                except:
                                    pass
                            yield second_chunk
                        return

                    # If multimodal, yield the results directly
                    for delta_multimodal in multimodal_deltas:
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
                        
                if final_usage:
                    usage_chunk = {
                        "id": "chatcmpl-mcp",
                        "object": "chat.completion.chunk",
                        "model": "internal-mcp",
                        "choices": [],
                        "usage": final_usage
                    }
                    yield f"data: {json.dumps(usage_chunk)}\n\n"

                yield "data: [DONE]\n\n"
                return
                
            try:
                chunk = json.loads(data_str)
                
                if "usage" in chunk and chunk["usage"]:
                    final_usage = chunk["usage"]

                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if "content" in delta and delta["content"]:
                    assistant_message["content"] += delta["content"]
                
                if "tool_calls" in delta:
                    for tc in delta["tool_calls"]:
                        idx = tc.get("index", 0)
                        if idx not in tool_call_buffer:
                            tool_call_buffer[idx] = {"name": "", "arguments": "", "id": tc.get("id", f"call_{idx}")}
                            
                        func = tc.get("function", {})
                        if "name" in func and func["name"]:
                            tool_call_buffer[idx]["name"] = func["name"]
                        if "arguments" in func and func["arguments"]:
                            tool_call_buffer[idx]["arguments"] += func["arguments"]
                    continue
                    
                yield chunk_str
                
            except Exception as e:
                logger.error(f"InternalMCP: Error parsing chunk: {e}")
                yield chunk_str
