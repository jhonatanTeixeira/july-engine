from ..models.base_model import BaseModel


class AdapterBase(BaseModel):
    def __init__(self, task_type: str, backend, model_meta=None):
        if not backend:
            raise ValueError("no backend detected for this task")
        
        super().__init__(backend, model_meta)
        self.task_type = task_type