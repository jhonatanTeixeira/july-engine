import httpx
import logging
import io
import os
import tempfile
from typing import Dict, Any, Optional
from gguf import GGUFReader

logger = logging.getLogger("JulyEngine.Services.GGUFMetadataScanner")

class GGUFMetadataScanner:
    """
    Scanner to extract metadata from GGUF files remotely using HTTP Range Requests.
    Reads only the first 1MB of the file to extract the header.
    """
    
    @staticmethod
    async def scan_remote_metadata(url: str) -> Dict[str, Any]:
        """
        Scans a remote GGUF file and returns architectural metadata.
        """
        logger.info(f"Scanning remote GGUF metadata: {url}")
        
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                # Request first 1MB
                headers = {"Range": "bytes=0-1048575"}
                response = await client.get(url, headers=headers)
                
                if response.status_code not in [200, 206]:
                    logger.error(f"Failed to fetch GGUF header. Status: {response.status_code}")
                    return {}
                
                content = response.content
                
                # Write to a temporary file because GGUFReader expects a path
                with tempfile.NamedTemporaryFile(delete=False, suffix=".gguf") as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                
                try:
                    reader = GGUFReader(tmp_path)
                    metadata = GGUFMetadataScanner._parse_reader(reader)
                    logger.info(f"Scanned metadata successfully: {metadata}")
                    return metadata
                    
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                        
        except Exception as e:
            logger.error(f"Error scanning GGUF metadata: {e}")
            return {}

    @staticmethod
    def _parse_reader(reader: GGUFReader) -> Dict[str, Any]:
        """Shared logic to extract metadata from a GGUFReader instance."""
        try:
            # Extract architecture first to use as prefix
            arch = str(reader.get_field("general.architecture"))
            if arch.startswith("["): # Handle potential list/byte return
                 arch = arch.strip("[]' ")
            
            metadata = {
                "architecture": arch,
                "block_count": int(reader.get_field(f"{arch}.block_count") or 0),
                "head_count": int(reader.get_field(f"{arch}.attention.head_count") or 0),
                "head_count_kv": int(reader.get_field(f"{arch}.attention.head_count_kv") or 0),
                "embedding_length": int(reader.get_field(f"{arch}.embedding_length") or 0),
            }
            
            # Diagnostic for zero values
            for k, v in metadata.items():
                if v == 0 and k != "architecture":
                    logger.warning(f"GGUF field '{arch}.{k}' returned 0 or missing.")
            
            # Fallback for head_count_kv if missing (assume MHA instead of GQA/MQA)
            if metadata["head_count_kv"] == 0:
                metadata["head_count_kv"] = metadata["head_count"]
                
            return metadata
        except Exception as e:
            logger.error(f"Error parsing GGUF fields: {e}")
            return {}

    @staticmethod
    def scan_local_file(path: str) -> Dict[str, Any]:
        """Scans a local GGUF file and returns architectural metadata."""
        logger.info(f"Scanning local GGUF metadata: {path}")
        try:
            reader = GGUFReader(path)
            metadata = GGUFMetadataScanner._parse_reader(reader)
            logger.info(f"Local metadata scanned successfully: {metadata}")
            return metadata
        except Exception as e:
            logger.error(f"Error scanning local GGUF: {e}")
            return {}

    @staticmethod
    async def resolve_metadata(repo_id: str, filename: str) -> Dict[str, Any]:
        """
        Smart resolver:
        1. Checks if the file is already in Hugging Face cache.
        2. If cached, reads metadata locally (fast).
        3. If not cached, performs remote Range Request scan (1MB).
        """
        if not repo_id or not filename:
            return {}

        from huggingface_hub import try_to_load_from_cache
        
        # Check if file exists in cache
        cached_path = try_to_load_from_cache(repo_id=repo_id, filename=filename)
        
        if cached_path and os.path.exists(cached_path):
            logger.info(f"Metadata Resolver: Found cached file for {filename}. Reading locally.")
            return GGUFMetadataScanner.scan_local_file(cached_path)
        
        # Fallback to remote scan
        logger.info(f"Metadata Resolver: {filename} not in cache. Performing remote scan.")
        url = GGUFMetadataScanner.get_huggingface_url(repo_id, filename)
        return await GGUFMetadataScanner.scan_remote_metadata(url)

    @staticmethod
    def get_huggingface_url(repo_id: str, filename: str) -> str:
        """Helper to construct HF direct download URL."""
        return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
