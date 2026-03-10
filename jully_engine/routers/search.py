from fastapi import APIRouter, HTTPException, Request
from ..bridge import bridge

router = APIRouter(prefix="/search", tags=["Search"])


@router.post("/web", tags=["July"])
async def search_web(request: Request):
    payload = await request.json()
    headers = dict(request.headers)
    
    if "x-backend" not in headers:
        headers["x-backend"] = "api"
    
    result = await bridge.process_search_web(payload, headers)
    
    return {"result": result}


@router.post("/github", tags=["July"])
async def search_github(request: Request):
    payload = await request.json()
    headers = dict(request.headers)
    
    if "x-backend" not in headers:
        headers["x-backend"] = "api"
    
    result = await bridge.process_search_code(payload, headers)
    
    return {"result": result}
