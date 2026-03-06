"""AXEngine model loading utility for Frigate."""

import logging
import os
import time
from pathlib import Path

from frigate.comms.inter_process import InterProcessRequestor
from frigate.const import UPDATE_MODEL_STATE
from frigate.types import ModelStatusTypesEnum
from frigate.util.downloader import ModelDownloader
from frigate.util.file import FileLock

logger = logging.getLogger(__name__)

AXENGINE_JINA_V2_MODEL = "jina_v2"
AXENGINE_JINA_V2_REPO = "AXERA-TECH/jina-clip-v2"


def get_axengine_model_type(model_path: str) -> str | None:
    if "jina-clip-v2" in str(model_path):
        return AXENGINE_JINA_V2_MODEL

    return None


def is_axengine_compatible(
    model_path: str, device: str | None, model_type: str | None = None
) -> bool:
    if (device or "").upper() != "AXENGINE":
        return False

    if not model_type:
        model_type = get_axengine_model_type(model_path)

    return model_type == AXENGINE_JINA_V2_MODEL


def wait_for_download_completion(
    image_model_path: Path,
    text_model_path: Path,
    lock_path: Path,
    timeout: int = 300,
) -> bool:
    start_time = time.time()

    while time.time() - start_time < timeout:
        if image_model_path.exists() and text_model_path.exists():
            return True

        if not lock_path.exists():
            return image_model_path.exists() and text_model_path.exists()

        time.sleep(1)

    logger.warning("Timeout waiting for AXEngine model files: %s", image_model_path)
    return False


def auto_convert_model(model_path: str, model_type: str | None = None) -> str | None:
    """Prepare AXEngine model files and return the image encoder path."""
    if not is_axengine_compatible(model_path, "AXENGINE", model_type):
        return None

    model_dir = Path(model_path).parent
    ui_model_key = f"jinaai/jina-clip-v2-{Path(model_path).name}"
    ui_preprocessor_key = "jinaai/jina-clip-v2-preprocessor_config.json"
    image_model_path = model_dir / "image_encoder.axmodel"
    text_model_path = model_dir / "text_encoder.axmodel"
    model_repo = os.environ.get("AXENGINE_JINA_V2_REPO", AXENGINE_JINA_V2_REPO)
    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    requestor = InterProcessRequestor()

    download_targets = {
        "image_encoder.axmodel": f"{hf_endpoint}/{model_repo}/resolve/main/image_encoder.axmodel",
        "text_encoder.axmodel": f"{hf_endpoint}/{model_repo}/resolve/main/text_encoder.axmodel",
    }

    if image_model_path.exists() and text_model_path.exists():
        requestor.send_data(
            UPDATE_MODEL_STATE,
            {
                "model": ui_preprocessor_key,
                "state": ModelStatusTypesEnum.downloaded,
            },
        )
        requestor.send_data(
            UPDATE_MODEL_STATE,
            {
                "model": ui_model_key,
                "state": ModelStatusTypesEnum.downloaded,
            },
        )
        requestor.stop()
        return str(image_model_path)

    lock_path = model_dir / ".axengine.download.lock"
    lock = FileLock(lock_path, timeout=300, cleanup_stale_on_init=True)

    if lock.acquire():
        try:
            requestor.send_data(
                UPDATE_MODEL_STATE,
                {
                    "model": ui_preprocessor_key,
                    "state": ModelStatusTypesEnum.downloaded,
                },
            )
            requestor.send_data(
                UPDATE_MODEL_STATE,
                {
                    "model": ui_model_key,
                    "state": ModelStatusTypesEnum.downloading,
                },
            )

            for file_name, url in download_targets.items():
                target_path = model_dir / file_name
                if target_path.exists():
                    continue

                target_path.parent.mkdir(parents=True, exist_ok=True)
                ModelDownloader.download_from_url(url, str(target_path))

            requestor.send_data(
                UPDATE_MODEL_STATE,
                {
                    "model": ui_model_key,
                    "state": ModelStatusTypesEnum.downloaded,
                },
            )

            return str(image_model_path)
        except Exception:
            requestor.send_data(
                UPDATE_MODEL_STATE,
                {
                    "model": ui_model_key,
                    "state": ModelStatusTypesEnum.error,
                },
            )
            logger.exception(
                "Failed to prepare AXEngine model files for %s", model_repo
            )
            return None
        finally:
            requestor.stop()
            lock.release()

    logger.info("Another process is preparing AXEngine models, waiting for completion")
    requestor.send_data(
        UPDATE_MODEL_STATE,
        {
            "model": ui_preprocessor_key,
            "state": ModelStatusTypesEnum.downloaded,
        },
    )
    requestor.send_data(
        UPDATE_MODEL_STATE,
        {
            "model": ui_model_key,
            "state": ModelStatusTypesEnum.downloading,
        },
    )
    requestor.stop()

    if wait_for_download_completion(image_model_path, text_model_path, lock_path):
        if image_model_path.exists() and text_model_path.exists():
            requestor = InterProcessRequestor()
            requestor.send_data(
                UPDATE_MODEL_STATE,
                {
                    "model": ui_model_key,
                    "state": ModelStatusTypesEnum.downloaded,
                },
            )
            requestor.stop()
            return str(image_model_path)

    logger.error("Timeout waiting for AXEngine model download lock for %s", model_dir)
    requestor = InterProcessRequestor()
    requestor.send_data(
        UPDATE_MODEL_STATE,
        {
            "model": ui_model_key,
            "state": ModelStatusTypesEnum.error,
        },
    )
    requestor.stop()
    return None
