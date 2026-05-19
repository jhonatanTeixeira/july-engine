from abc import ABC, abstractmethod
from typing import BinaryIO, Optional

class StorageDriver(ABC):
    @abstractmethod
    def put(self, path: str, content: bytes) -> str:
        """Stores content and returns the relative path."""
        pass

    @abstractmethod
    def get(self, path: str) -> bytes:
        """Retrieves content as bytes."""
        pass

    @abstractmethod
    def delete(self, path: str) -> bool:
        """Deletes the file."""
        pass

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Checks if file exists."""
        pass

    @abstractmethod
    def get_local_path(self, path: str) -> str:
        """Returns a local filesystem path (downloads if remote)."""
        pass

    @abstractmethod
    def get_url(self, path: str) -> str:
        """Returns a public URL if applicable."""
        pass
