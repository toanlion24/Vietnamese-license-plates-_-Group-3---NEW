---
name: vn-license-plate-delivery
description: Delivers and refactors Vietnam license plate recognition pipelines in Python. Use when implementing detector/OCR/pipeline/evaluation modules, dataset manifests, or demo flows for image, video, and webcam inference.
---

# VN License Plate Delivery

## Quick Start
When editing this project:
1. Keep boundaries clear: `io -> detector -> preprocess -> ocr -> postprocess -> eval -> app`.
2. Add or update configs before changing model logic.
3. Run inference on a small validation slice and report CER/WER/plate accuracy changes.

## Directory Responsibilities
- `src/io/`: frame readers and input adapters.
- `src/detector/`: plate localization models and wrappers.
- `src/preprocess/`: crop, rectify, denoise, threshold.
- `src/ocr/`: text recognizers and decoding adapters.
- `src/postprocess/`: normalization, regex repair, voting.
- `src/pipeline/`: end-to-end orchestration.
- `src/eval/`: metrics and error analysis export.
- `src/app/`: CLI/webcam/demo interfaces.

## Implementation Rules
- Prefer pure functions for preprocessing/postprocessing modules.
- Use dataclasses or TypedDict for inter-stage payloads.
- Log per-stage latency and confidence for every prediction path.
- Keep fallback OCR strategy explicit (for example: TrOCR fail -> EasyOCR).

## Evaluation Contract
- Always evaluate on a fixed held-out set and output:
  - `cer`
  - `wer`
  - `plate_accuracy`
  - `mean_latency_ms`
- Save predictions in CSV/JSON with `image_id`, `gt`, `pred`, and `error_type`.

## Done Criteria
- Pipeline runs on image folder and webcam mode.
- Metrics script reproduces the reported numbers.
- README includes setup, inference commands, and known limitations.
