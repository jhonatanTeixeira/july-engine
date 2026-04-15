from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List
from ..services.mcps_service import McpsService
from ..services.internal_mcp import InternalMCP
from ..services.external_mcp import external_mcp_manager

router = APIRouter(prefix="/v1/mcps", tags=["MCPs"])
service = McpsService()

async def restart_mcp_manager():
    await external_mcp_manager.stop()
    await external_mcp_manager.start()

@router.get("/tools", response_model=List[Dict[str, Any]])
def get_all_tools():
    # Retorna ferramentas internas e externas combinadas
    internal_mcp = InternalMCP()
    tools = internal_mcp.get_tools()
    
    # Adicionamos metadados dos servidores para as ferramentas externas
    # Isso ajuda a UI a agrupar/exibir melhor
    return tools

@router.get("/", response_model=List[Dict[str, Any]])
def get_all_mcps():
    return service.get_all()

@router.post("/")
async def create_mcp(payload: Dict[str, Any], background_tasks: BackgroundTasks):
    mcp_id = service.create(payload)
    background_tasks.add_task(restart_mcp_manager)
    return {"success": True, "id": mcp_id}

@router.put("/{mcp_id}")
async def update_mcp(mcp_id: str, payload: Dict[str, Any], background_tasks: BackgroundTasks):
    service.set(mcp_id, payload)
    background_tasks.add_task(restart_mcp_manager)
    return {"success": True, "id": mcp_id}

@router.get("/{mcp_id}")
def get_mcp(mcp_id: str):
    mcp = service.get(mcp_id)
    if not mcp:
        raise HTTPException(status_code=404, detail="MCP not found")
    return mcp

@router.delete("/{mcp_id}")
async def delete_mcp(mcp_id: str, background_tasks: BackgroundTasks):
    success = service.delete(mcp_id)
    if not success:
        raise HTTPException(status_code=404, detail="MCP not found")
    background_tasks.add_task(restart_mcp_manager)
    return {"success": True}
