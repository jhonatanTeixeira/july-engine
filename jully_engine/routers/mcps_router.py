from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from ..services.mcps_service import McpsService
from ..services.internal_mcp import InternalMCP

router = APIRouter(prefix="/v1/mcps", tags=["MCPs"])
service = McpsService()

@router.get("/tools", response_model=List[Dict[str, Any]])
def get_all_tools():
    internal_mcp = InternalMCP()
    return internal_mcp.get_tools()

@router.get("/", response_model=List[Dict[str, Any]])
def get_all_mcps():
    return service.get_all()

@router.post("/")
def create_mcp(payload: Dict[str, Any]):
    mcp_id = service.create(payload)
    return {"success": True, "id": mcp_id}

@router.put("/{mcp_id}")
def update_mcp(mcp_id: str, payload: Dict[str, Any]):
    service.set(mcp_id, payload)
    return {"success": True, "id": mcp_id}

@router.get("/{mcp_id}")
def get_mcp(mcp_id: str):
    mcp = service.get(mcp_id)
    if not mcp:
        raise HTTPException(status_code=404, detail="MCP not found")
    return mcp

@router.delete("/{mcp_id}")
def delete_mcp(mcp_id: str):
    success = service.delete(mcp_id)
    if not success:
        raise HTTPException(status_code=404, detail="MCP not found")
    return {"success": True}
