"""Axera axengine model utility for Frigate.

Similar to rknn_converter.py, this module handles AX model resolution
and download. When axengine hardware is available and a supported model
type is requested, get_optimized_runner() in detection_runners.py will
use this utility to transparently resolve AX models so that embedding
and detector code remains hardware-agnostic.
"""

import logging
import os
import urllib.request

from frigate.const import MODEL_CACHE_DIR

logger = logging.getLogger(__name__)

# AX model configurations for enrichment models that use dual encoders (e.g., CLIP)
AX_ENRICHMENT_MODEL_CONFIGS = {
    "jina_v2": {
        "model_name": "AXERA-TECH/jina-clip-v2",
        "files": {
            "image_encoder": "image_encoder.axmodel",
            "text_encoder": "text_encoder.axmodel",
        },
        "text_input_name": "inputs_id",
        "text_max_length": 50,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
    },
}


def is_axengine_available() -> bool:
    """Check if axengine library is available."""
    try:
        import axengine  # type: ignore # noqa: F401

        return True
    except ImportError:
        return False


def is_ax_compatible(model_type: str | None) -> bool:
    """Check if a model type can use axengine acceleration.

    Args:
        model_type: The enrichment model type string

    Returns:
        True if axengine is available and the model type is supported
    """
    if not is_axengine_available():
        return False
    return model_type in AX_ENRICHMENT_MODEL_CONFIGS


def get_ax_model_config(model_type: str) -> dict | None:
    """Get AX model configuration for a given enrichment model type.

    Args:
        model_type: The enrichment model type string

    Returns:
        Configuration dict with file mappings and preprocessing params, or None
    """
    return AX_ENRICHMENT_MODEL_CONFIGS.get(model_type)


def auto_resolve_ax_model(model_type: str) -> str | None:
    """Download AX model files if needed and return path to the first model file.

    Similar to rknn_converter.auto_convert_model, this function resolves
    model files for Axera hardware. For AX, this means downloading
    pre-compiled .axmodel files from HuggingFace when they are not present.

    Args:
        model_type: The enrichment model type string (e.g., "jina_v2")

    Returns:
        Path to the first AX model file, or None if unavailable
    """
    config = AX_ENRICHMENT_MODEL_CONFIGS.get(model_type)
    if not config:
        return None

    model_name = config["model_name"]
    model_dir = os.path.join(MODEL_CACHE_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)

    HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")

    for filename in config["files"].values():
        local_path = os.path.join(model_dir, filename)
        if not os.path.isfile(local_path):
            url = f"{HF_ENDPOINT}/{model_name}/resolve/main/{filename}"
            logger.info("Downloading AX model: %s", url)
            try:
                urllib.request.urlretrieve(url, local_path)
            except Exception:
                logger.error("Failed to download AX model: %s", filename)
                return None

    first_file = list(config["files"].values())[0]
    return os.path.join(model_dir, first_file)
