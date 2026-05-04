import os
import fsspec
import hashlib
import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class CloudPath(os.PathLike):
    """
    A PathLike object that synchronizes between cloud and local storage using fsspec.
    """
    def __init__(self, path: str):
        self.raw_path = str(path)
        # fsspec.core.split_protocol returns (protocol, path)
        self.protocol, self.path_no_proto = fsspec.core.split_protocol(self.raw_path)

    def __fspath__(self) -> str:
        """
        Returns the local filesystem path after ensuring synchronization with cloud.
        """
        # If no protocol or file://, it's local
        if self.protocol in [None, "file"]:
            # If it was file://path, path_no_proto is just path
            return os.path.abspath(self.path_no_proto if self.path_no_proto else self.raw_path)

        # Cloud logic
        local_path = os.path.abspath(self.path_no_proto)
        
        try:
            # Ensure local directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            fs, _ = fsspec.core.url_to_fs(self.raw_path)
            
            cloud_exists = fs.exists(self.raw_path)
            local_exists = os.path.exists(local_path)

            if cloud_exists and local_exists:
                # Both exist: Compare checksums
                cloud_info = fs.info(self.raw_path)
                cloud_checksum = self._get_cloud_checksum(cloud_info)
                local_checksum = self._get_local_checksum(local_path)

                if cloud_checksum != local_checksum:
                    # Sync based on modification time (newer to older)
                    cloud_mtime = self._get_mtime(cloud_info, fs)
                    local_mtime = os.path.getmtime(local_path)

                    if cloud_mtime > local_mtime:
                        logger.info(f"Syncing {self.raw_path} -> {local_path} (Cloud is newer)")
                        fs.get(self.raw_path, local_path)
                    elif local_mtime > cloud_mtime:
                        self._upload_to_cloud(local_path, fs)
                        
            elif cloud_exists and not local_exists:
                # Only Cloud exists: Download
                logger.info(f"Downloading {self.raw_path} to {local_path}")
                fs.get(self.raw_path, local_path)
                
            elif not cloud_exists and local_exists:
                # Only Local exists: Upload
                self._upload_to_cloud(local_path, fs)
                
            else:
                # Neither exists: Create empty file in cloud
                logger.info(f"Neither exists, creating empty placeholder in {self.raw_path}")
                with fs.open(self.raw_path, 'wb') as f:
                    pass
                if not os.path.exists(local_path):
                    with open(local_path, 'wb') as f:
                        pass

        except Exception as e:
            logger.error(f"CloudPath sync error for {self.raw_path}: {e}")
            # Fallback to local path anyway to avoid crashing if possible
            return local_path

        return local_path

    def _get_local_checksum(self, path: str) -> str:
        """Calculates MD5 hash for a local file."""
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _get_cloud_checksum(self, info: dict) -> str:
        """Extracts checksum from cloud provider metadata (ETag or MD5)."""
        # S3 ETag (often MD5)
        if "ETag" in info:
            return info["ETag"].strip('"')
        # GCS MD5
        if "md5Hash" in info:
            return info["md5Hash"]
        # Generic fallback
        return f"{info.get('size', 0)}-{info.get('mtime', 0)}"

    def _get_mtime(self, info: dict, fs: fsspec.AbstractFileSystem) -> float:
        """Heuristically gets the modification timestamp from different clouds."""
        mtime = info.get("mtime")
        if mtime:
            if isinstance(mtime, datetime):
                return mtime.timestamp()
            return float(mtime)
        
        # Some fsspec systems define modified()
        try:
            return fs.modified(self.raw_path).timestamp()
        except:
            return 0.0

    def write_file(self, content: bytes):
        """
        Writes content to local file and immediately synchronizes to cloud.
        """
        local_path = os.path.abspath(self.path_no_proto) if self.protocol not in [None, "file"] else self.raw_path
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        with open(local_path, 'wb') as f:
            f.write(content)
        
        if self.protocol not in [None, "file"]:
            fs, _ = fsspec.core.url_to_fs(self.raw_path)
            self._upload_to_cloud(local_path, fs)

    def _upload_to_cloud(self, local_path: str, fs: fsspec.AbstractFileSystem):
        """
        Private method to upload a local file to cloud storage.
        """
        logger.info(f"Uploading {local_path} -> {self.raw_path}")
        fs.put(local_path, self.raw_path)

    def __str__(self):
        return self.__fspath__()

    def __repr__(self):
        return f"CloudPath('{self.raw_path}')"
