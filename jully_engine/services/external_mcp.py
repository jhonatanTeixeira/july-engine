import logging
import json
from typing import AsyncGenerator, Dict, Any, List
from contextlib import AsyncExitStack

# Only import MCP conditionally to avoid breaking if not installed properly
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.sse import sse_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

from ..persistence import get_backend

logger = logging.getLogger("JulyEngine.ExternalMCP")

class ExternalMCPManager:
    def __init__(self):
        self.backend = get_backend()
        self.sessions: Dict[str, Any] = {}
        self.exit_stack = AsyncExitStack()
        self.server_tools: Dict[str, List[Dict[str, Any]]] = {}

    async def start(self):
        if not MCP_AVAILABLE:
            logger.warning("MCP package not installed. External MCPs are disabled.")
            return
            
        mcps = self.backend.get_all_mcps()
        for mcp_conf in mcps:
            if mcp_conf.get("enabled", True):
                await self._connect_server(mcp_conf)

    async def _connect_server(self, conf: Dict[str, Any]):
        mcp_id = conf.get("id")
        mcp_type = conf.get("type", "stdio")
        try:
            if mcp_type == "stdio":
                command = conf.get("command")
                args = conf.get("args", [])
                env = conf.get("env", None)
                if isinstance(args, str):
                    import shlex
                    args = shlex.split(args)
                
                # Merge current env with provided env
                import os
                full_env = os.environ.copy()
                if env and isinstance(env, dict):
                    for k, v in env.items():
                        full_env[k] = str(v)
                        
                server_params = StdioServerParameters(command=command, args=args, env=full_env)
                read, write = await self.exit_stack.enter_async_context(stdio_client(server_params))
                session = await self.exit_stack.enter_async_context(ClientSession(read, write))
                
            elif mcp_type == "sse":
                url = conf.get("url")
                # For SSE, we treat the 'env' dict as HTTP headers
                headers = conf.get("env", {})
                if not isinstance(headers, dict):
                    headers = {}
                
                logger.info(f"Connecting to SSE MCP: {url} with {len(headers)} headers")
                read, write = await self.exit_stack.enter_async_context(sse_client(url, headers=headers))
                session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            else:
                return

            await session.initialize()
            self.sessions[mcp_id] = session
            
            # Fetch tools
            tools_response = await session.list_tools()
            tools = []
            for t in tools_response.tools:
                tools.append({
                    "type": "function",
                    "function": {
                        "name": f"{mcp_id}__{t.name}",
                        "description": f"[{conf.get('name', mcp_id)}] {t.description or ''}",
                        "parameters": t.inputSchema
                    }
                })
            self.server_tools[mcp_id] = tools
            logger.info(f"Connected to MCP {conf.get('name', mcp_id)} with {len(tools)} tools")
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP {conf.get('name', mcp_id)}: {e}")

    def get_all_tools(self, whitelist: List[str] = None) -> List[Dict[str, Any]]:
        all_tools = []
        for tools in self.server_tools.values():
            for t in tools:
                if not whitelist or t["function"]["name"] in whitelist:
                    all_tools.append(t)
        return all_tools

    def inject_tools(self, payload: Dict, whitelist: List[str] = None):
        if 'tools' not in payload:
            tools = self.get_all_tools(whitelist)
            
            if tools:
                payload['tools'] = tools

    async def stream_orchestrate(self, response: AsyncGenerator[Dict, None], brain_instance, original_payload: Dict[str, Any]) -> AsyncGenerator[Dict, None]:
        import asyncio
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
                    res = await self.execute_tool(name, json.loads(tool['arguments']))
                    tool["response"] = (res, None)
                is_calling = False
            
            if chunk.get('choices')[0].get('finish_reason') == 'tool_calls':
                for name, tool in tools.items():
                    llm, user = tool.get("response")
                    
                    tool_messages.append({
                        "role": "tool",
                        "tool_call_id": tool.get('id'),
                        "name": name,
                        "content": str(llm)
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
                
                async for chunk in await brain_instance.chat(original_payload):
                    yield chunk
                    await asyncio.sleep(0)

            yield chunk
            await asyncio.sleep(0)

    async def orchestrate(self, response: Any, brain, original_payload: Dict[str, Any]) -> Any:
        from typing import AsyncGenerator
        if isinstance(response, AsyncGenerator):
            return self.stream_orchestrate(response, brain, original_payload)
        else:
            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {})
            
            if choice.get("finish_reason") != "tool_calls" or "tool_calls" not in message:
                return response
                
            messages: list = original_payload.setdefault("messages", [])
            messages.append(message)
            
            multimodal_content = []
            requires_second_call = False
            
            for tc in message.get("tool_calls", []):
                name = tc.get("function", {}).get("name")
                args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                
                llm = await self.execute_tool(name, args)
                
                is_faf = False
                # If the tool is external and the result is empty, it's fire-and-forget.
                # Actually, any external tool call with empty return is fire-and-forget.
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

            if requires_second_call:
                second_response = await brain.chat(original_payload)
                second_content = second_response.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                if isinstance(second_content, list):
                    multimodal_content.extend(second_content)
                else:
                    multimodal_content.append(second_content)
                    
                content = multimodal_content if len(multimodal_content) > 1 else multimodal_content[0] if multimodal_content else ""
                second_response.setdefault("choices", [{}])[0].setdefault("message", {})['content'] = content

                return second_response
            else:
                content = multimodal_content if len(multimodal_content) > 1 else multimodal_content[0] if multimodal_content else ""
                response.setdefault("choices", [{}])[0].setdefault("message", {})['content'] = content
                return response

    async def execute_tool(self, full_name: str, arguments: Dict[str, Any]) -> Any:
        if "__" not in full_name:
            return "Invalid external tool name format"
        
        mcp_id, tool_name = full_name.split("__", 1)
        session = self.sessions.get(mcp_id)
        if not session:
            logger.warning(f"MCP server {mcp_id} not connected")
            return f"MCP server {mcp_id} not connected"
            
        try:
            logger.info(f"ExternalMCP executing tool: {full_name} with arguments: {arguments}")
            result = await session.call_tool(tool_name, arguments)
            logger.debug(f"ExternalMCP tool result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error executing external tool {full_name}: {e}")
            return f"Error executing tool: {e}"

    async def stop(self):
        try:
            await self.exit_stack.aclose()
        except Exception as e:
            logger.error(f"Error closing ExternalMCP: {e}")
        finally:
            self.sessions.clear()
            self.server_tools.clear()

external_mcp_manager = ExternalMCPManager()
