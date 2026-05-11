# Kiểm tra đề tài và câu hỏi hội đồng — nhận dạng biển số Việt Nam

Tài liệu này giúp tự đánh giá và chuẩn bị bảo vệ. Phần **trạng thái** mang tính tham chiếu codebase tại thời điểm bảo trì; dữ liệu lớn và checkpoint thường nằm ngoài Git nên nhiều mục chỉ “đủ” khi bạn điền đủ ảnh, nhãn và weight.

---

## Phần A — Câu hỏi hội đồng thường gặp (theo lĩnh vực)

### A1. Bài toán và đóng góp

| Câu hỏi | Gợi ý trả lời gắn với repo |
|--------|-----------------------------|
| Bài toán là gì, đầu vào/đầu ra? | Ảnh (hoặc frame) xe → phát hiện vùng biển → tiền xử lý → OCR chuỗi biển → chuẩn hóa theo format VN → (tùy chọn) đánh giá CER/WER/plate accuracy. |
| Vì sao tách module? | Luồng rõ: `detector` → `preprocess` → `ocr` → `postprocess` → `eval` (`src/pipeline/infer_plate_pipeline.py`), dễ thay YOLO / EasyOCR / TrOCR và đo lỗi từng bước. |
| Khác gì OCR tài liệu thông thường? | Biển số ngắn, cấu trúc cố định, dễ nhầm ký tự (0/O), cần normalize và regex VN (`src/postprocess/plate_rules.py`). |

### A2. Dữ liệu và chia tập

| Câu hỏi | Gợi ý |
|--------|--------|
| Train/val/test tách thế nào? | Không trộn cùng cảnh; có seed cố định (ví dụ `scripts/split_dataset.py`). Test dùng cho báo cáo cuối phải **khóa** và không dùng để chỉnh mô hình. |
| Ground truth lưu ở đâu? | Manifest CSV (`data/test_manifest.csv` hoặc manifest Buổi 4 có bbox/polygon cho DeepSolo — xem `data/manifests/README.md`). |
| Có cân bằng lớp / biển màu không? | Ghi rõ trong báo cáo nếu dataset lệch; có script gợi ý `scripts/clean_balance_dataset.py`. |

### A3. Mô hình và baseline

| Câu hỏi | Gợi ý |
|--------|--------|
| Detector dùng gì? | Mặc định YOLOv8 qua `src/detector/yolov8_detector.py`, trọng số `weights/yolov8_license_plate.pt`. |
| OCR baseline? | EasyOCR (`src/ocr/easyocr_adapter.py`) và/hoặc TrOCR Hugging Face (`src/ocr/trocr_adapter.py`). |
| Buổi 4 yêu cầu DeepSolo — trong repo có gì? | **Không nhúng code DeepSolo**; dùng repo ngoài train/infer, sau đó **import CSV** prediction đúng schema hoặc tạo nhãn polygon qua `scripts/prepare_buoi4_deepsolo_data.py`. Pipeline thay thế minh họa: YOLOv8 + EasyOCR vs YOLOv8 + TrOCR (`scripts/run_buoi4_manifest_inference.py`) khi chưa có DeepSolo. |

### A4. Thực nghiệm và metric

| Câu hỏi | Gợi ý |
|--------|--------|
| Metric OCR là gì? | **CER**, **WER**, **plate-level accuracy**; có thể thêm **latency** (Buổi 4). Code: `src/eval/metrics_plate.py`, tổng hợp A/B: `scripts/run_buoi4_experiments.py`. |
| So sánh công bằng A/B? | Cùng `image_id`, cùng `gt`, cùng hàm `normalize_plate_text`. Script kiểm tra tập ID (`fairness` trong JSON metrics). |
| Fine-tune TrOCR? | `scripts/train_trocr.py`, LR mặc định `configs/trocr/finetune_defaults.json`. |

### A5. Lỗi và giới hạn

| Câu hỏi | Gợi ý |
|--------|--------|
| Sai chủ yếu do đâu? | Phân loại: `detect_miss`, `bad_crop`, `ocr_error`, `postprocess_helped`, `ambiguous_gt` (`src/eval/error_labels.py`). |
| Giới hạn đề tài? | Domain gap (TrOCR pretrained không phải biển VN), ảnh ngoài trời, ảnh mờ, biển bẩn; không cam kết realtime nếu chưa đo trên thiết bị mục tiêu. |

### A6. Đạo đức và quyền riêng tư

| Cần trình bày |
|----------------|
| Ảnh biển có thể liên hệ phương tiện; nếu thu thập từ camera thực địa cần xin phép / ẩn danh hóa khi công bố. Không đưa biển số thật của người khác vào báo cáo công khai nếu chính sách trường/cơ quan không cho phép. |

---

## Phần B — Ma trận yêu cầu (đối chiếu code + việc bạn phải làm)

| Yêu cầu | Trạng thái trong repo | Bạn cần có thêm (thường ngoài Git) |
|---------|----------------------|-------------------------------------|
| Pipeline end-to-end: detect → OCR → postprocess | Có: `PlateInferencePipeline` | Ảnh + weight detector |
| CLI inference | Có: `scripts/run_infer.py` | Đường dẫn data |
| Đánh giá CER / WER / plate accuracy | Có: `eval_pipeline.py`, `run_buoi4_experiments.py` | CSV `gt` + `pred` |
| Test set cố định có GT | Có chỗ đặt: `data/test_manifest.csv` | Điền đủ dòng ảnh + nhãn |
| Buổi 4: hai cấu hình + metric A/B | Có script; **DeepSolo** cần train/export ngoài repo hoặc import CSV | CSV prediction A/B cùng test set |
| TrOCR fine-tune | Có: `scripts/train_trocr.py` | CSV crop + nhãn |
| Báo cáo / notebook trình bày | Có trong `docs/` | Cập nhật số liệu thật, strip output trước commit |
| DeepSolo annotation (polygon/bbox) | Có: `prepare_buoi4_deepsolo_data.py` | Manifest có `bbox_xyxy` hoặc polygon |

**Ghi chú quan trọng:** báo cáo markdown tự sinh từ `run_buoi4_experiments.py` có thể **ghi đè** file `--report-md` nếu trùng với tài liệu phụ lục dài; dùng `reports/` cho bản auto hoặc chỉ định file riêng.

---

## Phần C — Checklist tự chấm trước nộp / trước bảo vệ

- [ ] Đã có **test split cố định** và **không** dùng để train/tune trong quá trình chọn model cuối.
- [ ] Đã có **bảng số liệu thật** (không chỉ demo 8 mẫu) với **cùng pipeline đánh giá**.
- [ ] Buổi 4: đã giải thích rõ **DeepSolo** chạy ở đâu (repo ngoài) và CSV import vào `run_buoi4_experiments.py`.
- [ ] Đã có **phân tích lỗi** (vài case sai kèm ảnh/crop).
- [ ] Đã ghi **môi trường**: Python, CUDA, phiên bản thư viện chính (`requirements.txt`).
- [ ] Notebook repo: đã chạy `scripts/strip_ipynb_outputs.py` nếu commit.

---

## Phần D — Xác minh nhanh trong repo

```powershell
# Từ thư mục gốc project
.\.venv311\Scripts\python.exe scripts\check_de_tai_readiness.py
.\.venv311\Scripts\python.exe -m compileall src scripts
```

---

## Phần E — Nếu hội đồng hỏi “thiếu DeepSolo trong code”

Trả lời mẫu (điều chỉnh cho đúng thực tế nhóm): *DeepSolo là repo training nặng và tách khỏi codebase pipeline; nhóm huấn luyện / infer trong môi trường riêng, chỉ nối vào hệ thống đánh giá qua schema CSV thống nhất (`run_buoi4_experiments.py`). Phần pipeline trong repo tập trung detector YOLO + OCR để tái lập thực nghiệm và triển khai nhẹ.*
