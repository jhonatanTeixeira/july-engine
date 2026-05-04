import os
from typing import Dict, Any, Optional
from .local import LocalDriver
from .s3 import S3Driver
from .base import StorageDriver

class StorageManager:
    _instance: Optional[StorageDriver] = None

    @classmethod
    def get_driver(cls) -> StorageDriver:
        if cls._instance is None:
            driver_type = os.getenv("STORAGE_DRIVER", "local").lower()
            
            if driver_type == "s3":
                cls._instance = S3Driver(
                    bucket=os.getenv("S3_BUCKET", ""),
                    access_key=os.getenv("S3_ACCESS_KEY", ""),
                    secret_key=os.getenv("S3_SECRET_KEY", ""),
                    region=os.getenv("S3_REGION", "us-east-1")
                )
            else:
                # Default to local
                base_path = os.getenv("STORAGE_LOCAL_PATH", "storage")
                cls._instance = LocalDriver(base_path)
                
        return cls._instance

storage = StorageManager.get_driver()
