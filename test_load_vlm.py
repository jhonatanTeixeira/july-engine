import os
import logging
from optimum.onnxruntime import ORTModelForVision2Seq
from transformers import AutoProcessor, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LoadTest")

model_id = "onnx-community/FastVLM-0.5B-ONNX"

try:
    logger.info(f"Loading VLM model {model_id}")
    model = ORTModelForVision2Seq.from_pretrained(model_id, provider="CPUExecutionProvider")
    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    logger.info("FastVLM loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load VLM: {e}")
    import traceback
    logger.error(traceback.format_exc())
