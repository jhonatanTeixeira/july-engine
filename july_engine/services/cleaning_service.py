import os
import logging
import tempfile
from typing import Optional

logger = logging.getLogger("JulyEngine.Services.Cleaning")

class AudioCleaningService:
    def __init__(self):
        self.model = None
        self.df_state = None
        self._initialized = False

    def _lazy_init(self):
        if self._initialized:
            return
        
        try:
            from df.enhance import init_df
            logger.info("Initializing DeepFilterNet model...")
            self.model, self.df_state, _ = init_df()
            self._initialized = True
            logger.info("DeepFilterNet initialized successfully.")
        except ImportError:
            logger.warning("DeepFilterNet (df-net) not installed. Audio cleaning will be skipped.")
        except Exception as e:
            logger.error(f"Failed to initialize DeepFilterNet: {e}")

    def clean_audio(self, input_path: str, output_path: Optional[str] = None) -> Optional[str]:
        self._lazy_init()
        
        if not self._initialized:
            return None

        if output_path is None:
            # If no output path, overwrite input or use temp
            output_path = input_path

        try:
            from df.enhance import enhance, load_audio, save_audio
            
            logger.info(f"Cleaning audio: {input_path} -> {output_path}")
            audio, _ = load_audio(input_path, sr=self.df_state.sr())
            enhanced = enhance(self.model, self.df_state, audio)
            save_audio(output_path, enhanced, self.df_state.sr())
            
            return output_path
        except Exception as e:
            logger.error(f"Error cleaning audio with DeepFilterNet: {e}")
            return None

cleaning_service = AudioCleaningService()
