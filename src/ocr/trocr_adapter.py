from __future__ import annotations

import logging
from urllib.error import URLError
from urllib.request import urlopen

from src.postprocess.plate_rules import normalize_plate_text
from src.utils.types import OcrResult, PlateCrop

try:
    import torch
    from PIL import Image
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]
    TrOCRProcessor = None  # type: ignore[assignment]
    VisionEncoderDecoderModel = None  # type: ignore[assignment]


class TrOcrAdapter:
    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-printed",
        device: str = "cpu",
        cache_dir: str | None = None,
    ) -> None:
        if TrOCRProcessor is None or VisionEncoderDecoderModel is None or torch is None or Image is None:
            raise ImportError(
                "transformers, torch, and pillow are required. "
                "Install with: pip install transformers torch pillow"
            )
        requested_device = device.lower()
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            logging.warning("CUDA requested but unavailable. Falling back to CPU for TrOCR.")
            requested_device = "cpu"

        self.processor, self.model = self._load_model_with_preflight(model_name=model_name, cache_dir=cache_dir)
        self.device = requested_device
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _has_internet(timeout: float = 3.0) -> bool:
        try:
            with urlopen("https://huggingface.co", timeout=timeout):
                return True
        except URLError:
            return False

    def _load_model_with_preflight(self, model_name: str, cache_dir: str | None):
        try:
            processor = TrOCRProcessor.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=True,
            )
            model = VisionEncoderDecoderModel.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=True,
            )
            logging.info("TrOCR model found in local cache: %s", model_name)
            return processor, model
        except OSError:
            if self._has_internet():
                logging.info("TrOCR model not found in local cache, downloading: %s", model_name)
                processor = TrOCRProcessor.from_pretrained(model_name, cache_dir=cache_dir)
                model = VisionEncoderDecoderModel.from_pretrained(model_name, cache_dir=cache_dir)
                logging.info("TrOCR model download complete: %s", model_name)
                return processor, model
            raise RuntimeError(
                "TrOCR model not found in local cache and no internet connection is available. "
                "Connect to internet once to download the model, or set --model-cache-dir to a folder "
                "that already contains the model."
            )

    def recognize(self, plate_crop: PlateCrop, preprocessed) -> OcrResult:  # noqa: ANN001
        pil_image = Image.fromarray(preprocessed).convert("RGB")
        pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
        text_raw = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return OcrResult(
            image_id=plate_crop.image_id,
            text_raw=text_raw,
            text_norm=normalize_plate_text(text_raw),
            ocr_score=1.0,
        )

