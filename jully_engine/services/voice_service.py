import os
import json
import uuid
from typing import List, Dict, Any, Optional

class VoiceService:
    def __init__(self):
        # Localizado em local_models/local_models/voice_service.py
        # Queremos storage na raiz de local_models
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.storage_dir = os.path.join(self.base_dir, "storage", "voices")
        self.uploaded_dir = os.path.join(self.storage_dir, "uploaded")
        
        self.voices_json_path = os.path.join(self.storage_dir, "voices.json")
        self.uploaded_json_path = os.path.join(self.storage_dir, "uploaded_voices.json")
        
        os.makedirs(self.uploaded_dir, exist_ok=True)
        
        if not os.path.exists(self.voices_json_path):
            with open(self.voices_json_path, 'w') as f:
                json.dump([], f)
        
        if not os.path.exists(self.uploaded_json_path):
            with open(self.uploaded_json_path, 'w') as f:
                json.dump([], f)

    def list_voices(self) -> List[Dict[str, Any]]:
        voices = []
        if os.path.exists(self.voices_json_path):
            with open(self.voices_json_path, 'r') as f:
                voices.extend(json.load(f))
        
        if os.path.exists(self.uploaded_json_path):
            with open(self.uploaded_json_path, 'r') as f:
                voices.extend(json.load(f))
        
        return voices

    def get_voice_info(self, voice_id: str) -> Optional[Dict[str, Any]]:
        all_voices = self.list_voices()
        for v in all_voices:
            if v["id"] == voice_id:
                return v
        return None

    def get_voice_path(self, voice_id: str) -> Optional[tuple]:
        info = self.get_voice_info(voice_id)
        if info:
            if "path" in info:
                # Retorna (caminho_absoluto, linguagem)
                # O path no JSON é relativo à pasta storage/voices
                abs_path = os.path.abspath(os.path.join(self.storage_dir, info["path"]))
                return abs_path, info.get("language", "en")
            elif "piper_path" in info:
                # Mouth handles piper_path differently, but for XTTS/standard we return None or fallback
                pass
            
        # Fallback to yuni if requested not found
        if voice_id != 'yuni':
            return self.get_voice_path('yuni')
        return None

    def add_voice(self, name: str, language: str, audio_content: bytes, voice_type: str = "clone") -> Dict[str, Any]:
        voice_id = str(uuid.uuid4())
        filename = f"{voice_id}.wav"
        rel_path = os.path.join("uploaded", filename)
        abs_path = os.path.join(self.uploaded_dir, filename)
        
        with open(abs_path, 'wb') as f:
            f.write(audio_content)
            
        new_voice = {
            "id": voice_id,
            "name": name,
            "language": language
        }

        if voice_type == "piper":
            # If it's piper, we expect the uploaded file to be the .onnx? 
            # Or is it still a wav for reference? 
            # User said: "endpoint de speech/voices... vai determinar qual campo ele vai preencher path ou piper_path"
            # If they upload a wav, it probably fills piper_path with the wav? 
            # Usually Piper needs .onnx. If they upload .wav, it's probably for cloning.
            # But I will follow the instruction: "vai determinar qual campo ele vai preencher path ou piper_path"
            new_voice["piper_path"] = rel_path
        else:
            new_voice["path"] = rel_path
        
        uploaded_voices = []
        if os.path.exists(self.uploaded_json_path):
            with open(self.uploaded_json_path, 'r') as f:
                uploaded_voices = json.load(f)
        
        uploaded_voices.append(new_voice)
        
        with open(self.uploaded_json_path, 'w') as f:
            json.dump(uploaded_voices, f, indent=4)
            
        return new_voice

voice_service = VoiceService()
