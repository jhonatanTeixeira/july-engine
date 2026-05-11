from typing import Dict, Any, List, Optional
from ..persistence import get_backend
import uuid

class McpsService:
    def __init__(self):
        self.backend = get_backend()

    def get_all(self) -> List[Dict[str, Any]]:
        return self.backend.get_all_mcps()

    def get(self, mcp_id: str) -> Optional[Dict[str, Any]]:
        return self.backend.get_mcp(mcp_id)

    def set(self, mcp_id: str, data: Dict[str, Any]) -> None:
        self.backend.set_mcp(mcp_id, data)

    def create(self, data: Dict[str, Any]) -> str:
        mcp_id = data.get("id") or str(uuid.uuid4())
        data["id"] = mcp_id
        self.backend.set_mcp(mcp_id, data)
        return mcp_id

    def delete(self, mcp_id: str) -> bool:
        return self.backend.delete_mcp(mcp_id)
