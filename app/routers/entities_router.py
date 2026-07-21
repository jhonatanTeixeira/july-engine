from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ..bridge import bridge

router = APIRouter(prefix="/july/v1/entities", tags=["Entities"])


class EntityExtractionRequest(BaseModel):
    text: Union[str, List[str]]
    labels: Optional[List[str]] = None
    threshold: Optional[float] = None
    model: Optional[str] = None
    include_confidence: Optional[bool] = None
    include_spans: Optional[bool] = None


@router.post("/extract")
async def extract_entities(payload: EntityExtractionRequest, request: Request):
    headers = dict(request.headers)
    body: Dict[str, Any] = payload.model_dump(exclude_none=True)

    return await bridge.process_entity_extraction(body, headers)
