from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uuid
from ..persistence import get_backend

router = APIRouter(prefix="/v1/webhooks", tags=["Webhooks"])

class WebhookCreate(BaseModel):
    url: str
    api_token: Optional[str] = None

class WebhookUpdate(BaseModel):
    url: Optional[str] = None
    api_token: Optional[str] = None

class WebhookResponse(BaseModel):
    id: str
    url: str
    api_token: Optional[str] = None

@router.get("/", response_model=List[WebhookResponse])
def list_webhooks():
    backend = get_backend()
    webhooks = backend.get_setting("EVENTS_WEBHOOKS") or []
    return webhooks

@router.post("/", response_model=WebhookResponse)
def create_webhook(webhook: WebhookCreate):
    backend = get_backend()
    webhooks = backend.get_setting("EVENTS_WEBHOOKS") or []
    
    new_webhook = {
        "id": str(uuid.uuid4()),
        "url": webhook.url,
        "api_token": webhook.api_token
    }
    
    webhooks.append(new_webhook)
    backend.set_setting("EVENTS_WEBHOOKS", webhooks)
    return new_webhook

@router.put("/{webhook_id}", response_model=WebhookResponse)
def update_webhook(webhook_id: str, webhook_update: WebhookUpdate):
    backend = get_backend()
    webhooks = backend.get_setting("EVENTS_WEBHOOKS") or []
    
    for w in webhooks:
        if w.get("id") == webhook_id:
            if webhook_update.url is not None:
                w["url"] = webhook_update.url
            if webhook_update.api_token is not None:
                w["api_token"] = webhook_update.api_token
            
            backend.set_setting("EVENTS_WEBHOOKS", webhooks)
            return w
            
    raise HTTPException(status_code=404, detail="Webhook not found")

@router.delete("/{webhook_id}")
def delete_webhook(webhook_id: str):
    backend = get_backend()
    webhooks = backend.get_setting("EVENTS_WEBHOOKS") or []
    
    filtered_webhooks = [w for w in webhooks if w.get("id") != webhook_id]
    
    if len(filtered_webhooks) == len(webhooks):
        raise HTTPException(status_code=404, detail="Webhook not found")
        
    backend.set_setting("EVENTS_WEBHOOKS", filtered_webhooks)
    return {"success": True}
