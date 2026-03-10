import logging
import json
from typing import Dict, Any, List
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
                read, write = await self.exit_stack.enter_async_context(sse_client(url))
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

    def get_all_tools(self) -> List[Dict[str, Any]]:
        all_tools = []
        for tools in self.server_tools.values():
            all_tools.extend(tools)
        return all_tools

    async def execute_tool(self, full_name: str, arguments: Dict[str, Any]) -> Any:
        if "__" not in full_name:
            return "Invalid external tool name format"
        
        mcp_id, tool_name = full_name.split("__", 1)
        session = self.sessions.get(mcp_id)
        if not session:
            return f"MCP server {mcp_id} not connected"
            
        try:
            result = await session.call_tool(tool_name, arguments)
            # Assuming CallToolResult format
            texts = []
            if hasattr(result, "content"):
                for c in result.content:
                    if hasattr(c, "type") and c.type == "text":
                        texts.append(c.text)
                    elif isinstance(c, dict) and c.get("type") == "text":
                        texts.append(c.get("text", ""))
            return "\n".join(texts) if texts else "Tool executed successfully but returned no text."
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
