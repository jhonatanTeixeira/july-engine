import multiprocessing
import time
import logging
import uuid
import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional

logger = logging.getLogger("JulyEngine.Events")

# Global variables for the process and queue
_event_queue = None
_event_process = None

def _event_worker(queue: multiprocessing.Queue):
    """Background process that listens for events and writes to DB"""
    # Initialize DB inside the worker process so it gets its own connection
    from july_engine.persistence import get_backend, reset_backend
    reset_backend()
    db = get_backend()
    
    logger.info("Event subscriber worker started.")
    while True:
        try:
            event = queue.get()
            if event is None:
                # Stop signal
                logger.info("Event subscriber worker shutting down.")
                break
            
            # Fire and forget write
            db.add_history_event(event)
            
            # Fire webhooks
            try:
                webhooks = db.get_setting("EVENTS_WEBHOOKS") or []
                if webhooks:
                    payload = json.dumps(event).encode('utf-8')
                    for webhook in webhooks:
                        url = webhook.get("url")
                        if not url:
                            continue
                        
                        req = urllib.request.Request(url, data=payload, method="POST")
                        req.add_header('Content-Type', 'application/json')
                        
                        token = webhook.get("api_token")
                        if token:
                            req.add_header('Authorization', f'Bearer {token}')
                            
                        try:
                            urllib.request.urlopen(req, timeout=5) # Short timeout for fire and forget
                        except Exception as e:
                            logger.error(f"Failed to trigger webhook {url}: {e}")
            except Exception as e:
                logger.error(f"Error processing webhooks: {e}")
                
        except Exception as e:
            logger.error(f"Error processing event: {e}")

class EventManager:
    @classmethod
    def start(cls):
        global _event_queue, _event_process
        if _event_process is None:
            _event_queue = multiprocessing.Queue()
            _event_process = multiprocessing.Process(target=_event_worker, args=(_event_queue,), daemon=True)
            _event_process.start()
            logger.info("Event manager started.")

    @classmethod
    def stop(cls):
        global _event_queue, _event_process
        if _event_queue and _event_process:
            _event_queue.put(None)
            _event_process.join(timeout=3)
            if _event_process.is_alive():
                _event_process.terminate()
            _event_queue = None
            _event_process = None
            logger.info("Event manager stopped.")

    @classmethod
    def emit(cls, 
             media_type: str, 
             tokens_spent: int = 0, 
             generation_time: float = 0.0, 
             input_chars: int = 0, 
             audio_duration: float = 0.0,
             interaction_id: Optional[str] = None,
             extra_metadata: Dict[str, Any] = None):
        """
        Emits an event to the background process.
        media_type: e.g., 'text', 'audio_out', 'image', 'vision', 'audio_in'
        """
        if not _event_queue:
            return
            
        event_data = {
            "id": str(uuid.uuid4()),
            "interaction_id": interaction_id or f"evt_{uuid.uuid4().hex[:10]}",
            "timestamp": time.time(),
            "media_type": media_type,
            "tokens_spent": tokens_spent,
            "generation_time": generation_time,
            "input_chars": input_chars,
            "audio_duration": audio_duration,
            "metadata": extra_metadata or {}
        }
        
        try:
            _event_queue.put_nowait(event_data)
        except multiprocessing.queues.Full:
            logger.warning("Event queue is full. Dropping event.")
        except Exception as e:
            logger.error(f"Failed to emit event: {e}")

event_manager = EventManager()
