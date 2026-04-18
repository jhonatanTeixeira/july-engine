import json
import uuid
import fsspec
import os
import shutil
from typing import List, Dict, Any, Optional
from ..persistence import get_backend
from .storage.cloud_path import CloudPath

class VoiceService:
    def __init__(self):
        # Defina a raiz do storage via variável de ambiente ou default local
        self.base_path = os.environ.get("VOICE_STORAGE_PATH", "storage/voices")
        self.uploaded_dir = f"{self.base_path}/uploaded"
        
        # CloudPath para o arquivo de metadados
        self.voices_json_cp = CloudPath(f"{self.base_path}/voices.json")
        self.voices_json_path = str(self.voices_json_cp)

        # Instancia o sistema de arquivos via fsspec para operações de diretório
        self.fs, _ = fsspec.core.url_to_fs(self.base_path)

        # Garante que os diretórios existam
        self.fs.makedirs(self.uploaded_dir, exist_ok=True)

        self.backend = get_backend()

    def list_voices(self) -> List[Dict[str, Any]]:
        voices = []
        # CloudPath sincroniza automaticamente ao converter para string
        p = str(self.voices_json_cp)
        if os.path.exists(p):
            with open(p, 'r') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        voices.extend(data)
                except json.JSONDecodeError:
                    pass
        
        voices.extend(self.backend.get_uploaded_voices())
        return voices

    def get_voice_info(self, voice_id: str) -> Optional[Dict[str, Any]]:
        all_voices = self.list_voices()
        return next((v for v in all_voices if v["id"] == voice_id), None)

    def get_voice_path(self, voice_id: str) -> Optional[tuple]:
        info = self.get_voice_info(voice_id)
        
        if info:
            if "path" in info:
                # Usa CloudPath para garantir que o arquivo esteja local
                full_remote_path = f"{self.base_path}/{info['path']}"
                local_path = str(CloudPath(full_remote_path))
                return local_path, info.get("language", "en")
        
        if voice_id != 'yuni':
            return self.get_voice_path('yuni')
        return None

    def add_voice(self, name: str, language: str, audio_content: bytes, voice_type: str = "clone") -> Dict[str, Any]:
        from .cleaning_service import cleaning_service
        
        voice_id = str(uuid.uuid4())
        filename = f"{voice_id}.wav"
        rel_path = f"uploaded/{filename}"
        full_path = f"{self.uploaded_dir}/{filename}"
        
        # Escrita via CloudPath (salva local e cloud simultaneamente)
        cp = CloudPath(full_path)
        cp.write_file(audio_content)
        
        # Limpa o áudio passando o CloudPath (PathLike) diretamente
        if cleaning_service.clean_audio(cp, output_path=cp):
            # Garante que as mudanças locais sejam enviadas para a nuvem
            str(cp)
            
        new_voice = {
            "id": voice_id,
            "name": name,
            "language": language,
            "path": rel_path if voice_type != "piper" else None,
            "piper_path": rel_path if voice_type == "piper" else None
        }

        self.backend.add_uploaded_voice(new_voice)
        return new_voice

    def clean_voice(self, voice_id: str) -> bool:
        from .cleaning_service import cleaning_service
        voice_info = self.get_voice_info(voice_id)
        if not voice_info or "path" not in voice_info:
            return False
            
        full_path = f"{self.base_path}/{voice_info['path']}"
        cp = CloudPath(full_path)
            
        if cleaning_service.clean_audio(cp, output_path=cp):
            str(cp) # Trigger upload do arquivo limpo
            return True
        return False

    def delete_voice(self, voice_id: str) -> bool:
        voice_info = self.get_voice_info(voice_id)
        if not voice_info:
            return False
            
        # Delete from backend
        deleted = self.backend.delete_uploaded_voice(voice_id)
        
        # Delete files if they exist
        if "path" in voice_info and voice_info["path"]:
            full_path = f"{self.base_path}/{voice_info['path']}"
            try:
                cp = CloudPath(full_path)
                cp.unlink(missing_ok=True)
            except Exception:
                pass
                
        return deleted

    def update_voice(self, voice_id: str, name: Optional[str] = None, language: Optional[str] = None, metadata: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        voice_info = self.get_voice_info(voice_id)
        if not voice_info:
            return None
            
        if name is not None:
            voice_info["name"] = name
        if language is not None:
            voice_info["language"] = language
        if metadata is not None:
            voice_info["metadata"] = metadata
            
        self.backend.add_uploaded_voice(voice_info)
        return voice_info

voice_service = VoiceService()