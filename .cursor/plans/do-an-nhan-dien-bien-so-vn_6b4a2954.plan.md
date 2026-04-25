---
name: do-an-nhan-dien-bien-so-vn
overview: "Kế hoạch chi tiết 7 buổi cho đồ án thị giác máy tính: phát hiện và OCR biển số xe Việt Nam; baseline YOLO + OpenCV; Buổi 4 thực nghiệm DeepSolo + TrOCR (2 cấu hình), có demo và báo cáo hoàn chỉnh."
todos:
  - id: buoi-1-planning
    content: Hoàn thành mô tả bài toán, khảo sát công nghệ, vẽ pipeline, phân công và timeline 7 buổi
    status: pending
  - id: buoi-2-data
    content: Thu thập và gán nhãn dữ liệu biển số VN, viết script EDA và tiền xử lý/augment cơ bản
    status: pending
  - id: buoi-3-baseline-detector
    content: Cấu hình project YOLO và train mô hình detection baseline, chạy inference thử trên ảnh test
    status: pending
  - id: buoi-4-experiments
    content: Thực nghiệm 2 cấu hình DeepSolo (end-to-end) và DeepSolo+TrOCR (2-stage); train/infer/eval CER WER plate-level; báo cáo A/B và chốt pipeline cho Buổi 5
    status: pending
  - id: buoi-5-ocr-eval
    content: Tích hợp pipeline OCR với YOLO, đánh giá OCR trên ≥200 ảnh biển số thực tế và phân tích lỗi
    status: pending
  - id: buoi-6-demo
    content: Xây dựng demo (CLI/web/GUI) chạy được trên ảnh/video/webcam, kiểm thử với dữ liệu thực tế
    status: pending
  - id: buoi-7-report
    content: Hoàn thiện báo cáo, slide, diễn tập bảo vệ và đóng gói mã nguồn + hướng dẫn chạy
    status: pending
isProject: false
---

# Kế hoạch đồ án: Nhận diện biển số xe Việt Nam

## Mục tiêu tổng thể

- Xây dựng hệ thống nhận diện biển số xe Việt Nam gồm 2 phần: **phát hiện vùng biển số (detection)** và **nhận dạng ký tự (OCR)**.
- Đạt **độ chính xác OCR ≥ 85%** trên **≥ 200 ảnh biển số thực tế** (không dùng để train).
- Có **demo** chạy được trên ảnh/video/webcam, kèm **báo cáo và slide** đầy đủ.

## Công nghệ & tổ chức mã nguồn

- **Ngôn ngữ**: Python.
- **Thư viện chính**:
  - Detection: YOLOv8 (Ultralytics) hoặc YOLOv5 (chọn 1, ưu tiên YOLOv8n/s).
  - OCR: EasyOCR (ưu tiên) hoặc Tesseract + `pytesseract`.
  - Nhánh thực nghiệm Buổi 4: DeepSolo (text spotting) + TrOCR (so sánh A/B với baseline).
  - Xử lý ảnh: OpenCV, NumPy.
  - Phân tích & demo: Matplotlib/Seaborn, Streamlit/Gradio hoặc OpenCV GUI.
- **Cấu trúc thư mục gợi ý**:
  - `data/raw/` – ảnh gốc thu thập.
  - `data/labels/` – file nhãn YOLO cho biển số.
  - `data/splits/` – danh sách train/val/test.
  - `experiments/` – log huấn luyện, checkpoint mô hình.
  - `src/detector/` – mã train/inference YOLO.
  - `src/ocr/` – mã pipeline OCR.
  - `src/preprocess/` – tiền xử lý ảnh & biển số.
  - `src/app/` – mã demo (CLI/GUI).
  - `notebooks/` – EDA, thử nghiệm nhanh.

## Kiến trúc đề xuất (production-friendly cho đồ án)

### 1) Kiến trúc phân lớp

- **Ingestion layer** (`src/io/`):
  - Nhận input từ ảnh, video, webcam.
  - Chuẩn hóa frame về định dạng chung (`BGR np.ndarray` + metadata).
- **Perception layer**:
  - **Plate localization** (`src/detector/`): YOLOv8 hoặc DeepSolo (tuong ung buoi 4).
  - **Plate normalization** (`src/preprocess/`): crop, rectify perspective, denoise, threshold.
  - **Text recognition** (`src/ocr/`): EasyOCR/Tesseract/TrOCR.
- **Post-processing layer** (`src/postprocess/`):
  - Chuẩn hóa text (uppercase, bo ky tu la).
  - Regex + logic sua loi theo format bien so VN.
  - Multi-frame voting (neu video/webcam) de tang do on dinh.
- **Evaluation layer** (`src/eval/`):
  - Detection metrics (mAP).
  - OCR metrics (CER, WER, plate-level accuracy).
  - Error analysis va luu hard cases.
- **Application layer** (`src/app/`):
  - CLI cho batch inference.
  - API/GUI demo (Streamlit/Gradio/OpenCV GUI).
  - Logging, latency report.

### 2) Data contract giua cac module

- Detector output:
  - `image_id`, `bbox_xyxy`, `score`, `class_name`.
- OCR input:
  - `plate_crop`, `image_id`, `det_score`.
- OCR output:
  - `text_raw`, `text_norm`, `ocr_score`.
- Final output:
  - `plate_text`, `bbox`, `confidence`, `source_frame`, `timestamp`.

### 3) Cau truc thu muc de xay dung lau dai

- `configs/`
  - `detector/` (yolo/deepsolo yaml)
  - `ocr/` (easyocr/trocr config)
  - `pipeline/` (nguong score, regex, voting)
- `data/`
  - `raw/`, `interim/`, `processed/`, `labels/`, `splits/`
- `src/`
  - `io/`, `detector/`, `preprocess/`, `ocr/`, `postprocess/`, `pipeline/`, `eval/`, `app/`, `utils/`
- `scripts/`
  - `train_detector.py`, `run_infer.py`, `eval_pipeline.py`, `build_manifest.py`
- `experiments/`
  - `<date>_<model>_<setting>/metrics.json`, `predictions.csv`, `artifacts/`
- `reports/`
  - `ablation_deepsolo_trocr.md`, `error-analysis.md`

### 4) Luong chay chinh

- **Train**:
  - `scripts/build_manifest.py` -> `scripts/train_detector.py` -> `scripts/eval_pipeline.py`.
- **Inference batch**:
  - Input folder -> detect -> preprocess -> OCR -> postprocess -> JSON/CSV output.
- **Realtime**:
  - Webcam stream -> detect moi N frame -> track/vote -> overlay bbox + text.

### 5) Non-functional requirements

- Reproducibility: moi lan train luu `config + seed + git_sha`.
- Observability: log latency tung stage (detect/preprocess/ocr/postprocess).
- Maintainability: moi module co unit test cho ham core.
- Safety: khong commit du lieu nhay cam, khong luu bien so that o public report.

## Buổi 1 – Khởi động & Lập kế hoạch chi tiết

**Mục tiêu**: Hiểu rõ bài toán, chốt stack công nghệ, pipeline, kế hoạch 7 buổi và phân công.

**Việc cần làm**

- Viết mô tả bài toán ngắn gọn (1–2 trang):
  - Bối cảnh ứng dụng (bãi gửi xe, camera đường phố…).
  - Input: ảnh/video/webcam chứa xe.
  - Output: chuỗi biển số + bounding box, có thể nhiều biển số/ảnh.
  - Chỉ số đánh giá: mAP cho detection; character accuracy, plate accuracy cho OCR.
- Khảo sát nhanh các hướng kỹ thuật:
  - So sánh YOLOv5 vs YOLOv8 (tài liệu, code ví dụ) và chốt 1 mô hình.
  - Thử code ví dụ YOLO pre-trained trên COCO để chắc chắn môi trường chạy được.
  - Thử EasyOCR/Tesseract trên vài ảnh biển số mẫu (có thể là biển số nước ngoài) để kiểm tra cài đặt.
- Phân tích **format biển số Việt Nam**:
  - Biển 1 dòng/2 dòng, nền trắng/vàng/xanh.
  - Viết ra vài regex/luật cơ bản cho biển số (ví dụ cho xe máy, ô tô).
- Vẽ sơ đồ **pipeline tổng thể** (trong slide/notebook):
  - Ảnh/video → YOLO detect biển số → crop → tiền xử lý (gray, threshold, resize, deskew) → OCR → hậu xử lý regex/đa khung hình.
- Lập **timeline 7 buổi** (gần giống plan này) và phân công người phụ trách:
  - Thành viên A: dữ liệu + gán nhãn.
  - Thành viên B: mô hình YOLO.
  - Thành viên C: OCR + pipeline.
  - Cả nhóm: demo + báo cáo.

**Deliverables**

- Tài liệu mô tả bài toán & yêu cầu.
- Sơ đồ pipeline (ảnh vẽ hoặc slide).
- Ghi chú phân công nhiệm vụ và lịch làm việc.
- Tai lieu trien khai chi tiet de hoc lai tung buoc: `reports/buoi-1-khoi-dong-va-lap-ke-hoach-chi-tiet.md`.

## Buổi 2 – Thu thập & Tiền xử lý dữ liệu

**Mục tiêu**: Có tập dữ liệu bước đầu, đã gán nhãn một phần và có script EDA + tiền xử lý cơ bản.

**Việc cần làm**

- **Thu thập dữ liệu**:
  - Tự chụp ảnh/đoạn video xe máy, ô tô trên đường hoặc trong bãi giữ xe (đa dạng ánh sáng, góc, khoảng cách).
  - Trích frame từ video nếu cần để tăng số lượng ảnh.
  - Bổ sung thêm ảnh từ internet (nếu được) cho phong phú; lưu vào `data/raw/`.
  - Mục tiêu: tối thiểu 300–500 ảnh có chứa biển số VN rõ ràng.
- **Gán nhãn detection**:
  - Dùng LabelImg/Roboflow để vẽ bounding box quanh biển số, class duy nhất `license_plate`.
  - Xuất nhãn theo định dạng YOLO (file `.txt` cùng tên ảnh).
  - Hoàn thành ít nhất ~200 ảnh gán nhãn trong buổi 2.
- **Tách tập train/val/test sơ bộ**:
  - Viết script Python để random split, lưu danh sách file vào `data/splits/{train,val,test}.txt`.
- **EDA & tiền xử lý**:
  - Notebook nhỏ đọc danh sách ảnh + nhãn, hiển thị vài sample kèm box để kiểm tra.
  - Thống kê: số ảnh, số biển/ảnh, kích thước box trung bình, phân bố điều kiện ánh sáng (tự gắn tag đơn giản nếu có).
  - Viết hàm tiền xử lý ảnh chung: resize, normalize, augment (brightness, contrast, small rotation) → sử dụng sau này khi train YOLO.

**Deliverables**

- Thư mục dữ liệu đã tổ chức rõ ràng, có nhãn cho ≥ 200 ảnh.
- Notebook/Script EDA hiển thị thống kê + vài ảnh mẫu.
- Script tiền xử lý/augment cơ bản chạy được.
- Tai lieu trien khai chi tiet + checklist + script khung: `reports/buoi-2-thu-thap-va-tien-xu-ly-du-lieu-chi-tiet.md`.

## Buổi 3 – Xây dựng mô hình baseline (Detection)

**Mục tiêu**: Train được mô hình YOLO baseline với dữ liệu hiện có và chạy inference được.

**Việc cần làm**

- Chuẩn bị **file cấu hình YOLO**:
  - `data.yaml`: đường dẫn tới image/label train/val, khai báo class `license_plate`.
  - Kiểm tra lại cấu trúc thư mục đúng như YOLO yêu cầu.
- Cài đặt & cấu hình môi trường YOLO (Ultralytics) trong Python env.
- Viết/điều chỉnh script train baseline:
  - Chọn model nhỏ (YOLOv8n/yolov5s) để train nhanh.
  - Thiết lập hyperparameter mặc định: epochs ~50, batch size tùy GPU, image size 640.
  - Dùng augment cơ bản đã chuẩn bị.
- Chạy huấn luyện trên tập train/val:
  - Ghi lại log loss, precision, recall, mAP trong vài epoch đầu; lưu checkpoint tốt nhất vào `experiments/`.
- Thử **inference** trên một số ảnh trong tập test hoặc ảnh mới:
  - Viết script nhận đường dẫn ảnh → vẽ bounding box biển số ra file/hiển thị.
  - Quan sát lỗi: biển nhỏ không detect được, detect nhầm background, nhầm biển nền vàng/trắng, v.v.

**Deliverables**

- Checkpoint mô hình YOLO baseline.
- Script hoặc notebook train + inference.
- Ảnh minh họa kết quả detect (đúng/sai) để dùng sau này trong báo cáo.

## Buổi 4 – Cải tiến mô hình & Thực nghiệm (DeepSolo + TrOCR)

**Mục tiêu**: Thử **đúng 2 cấu hình** so sánh công bằng trên cùng tập test, đánh giá bằng **CER / WER / plate-level accuracy** (và tùy chọn latency), rồi chốt pipeline cho Buổi 5–6.

**Hai cấu hình bắt buộc**

1. **Cấu hình A – DeepSolo end-to-end**  
   Text spotting một mô hình: đầu vào ảnh/frame, đầu ra polygon/bbox + chuỗi biển số. Tham khảo repo [DeepSolo](https://github.com/ViTAE-Transformer/DeepSolo).

2. **Cấu hình B – DeepSolo + TrOCR (2-stage)**  
   Giai đoạn 1: DeepSolo chỉ để localize vùng biển (hoặc spotting rồi lấy crop). Giai đoạn 2: crop → tiền xử lý → **TrOCR** đọc ký tự. Tham khảo [TrOCR (Transformers)](https://huggingface.co/docs/transformers/model_doc/trocr).

**Việc cần làm**

- **Dữ liệu thống nhất**
  - Chuẩn hóa transcript GT (uppercase, bỏ khoảng trắng thừa hoặc giữ một format cố định).
  - Giữ nguyên split `train/val/test` (không trộn cùng cảnh giữa train và test).
  - Manifest tối thiểu: `image_id`, `polygon hoặc bbox`, `text_gt` (JSONL hoặc CSV).

- **Cấu hình A**
  - Convert nhãn sang format text spotting mà DeepSolo yêu cầu (polygon + text).
  - Train 1 run, lưu `best` / `last`.
  - Inference: xuất `text_pred`, `score`, polygon.

- **Cấu hình B**
  - Dùng vùng từ DeepSolo → crop/rectify (perspective nếu nghiêng).
  - Fine-tune hoặc inference TrOCR trên crop; hậu xử lý regex biển số VN.
  - Lưu kết quả trung gian (ảnh crop, raw OCR) để debug.

- **Đánh giá (cùng test set cho A và B)**
  - **CER**: tổng khoảng cách chỉnh sửa ký tự / tổng số ký tự GT.
  - **WER**: khoảng cách chỉnh sửa theo từ (token tách bằng khoảng trắng) / số từ GT; với biển số 1 dòng có thể coi cả chuỗi là 1 “từ”.
  - **Plate-level accuracy**: tỉ lệ mẫu có `normalize(pred) == normalize(gt)` (khớp hoàn toàn sau chuẩn hóa).
  - (Tùy chọn) thời gian suy luận trung bình / frame trên CPU hoặc GPU.

- **Phân tích lỗi**
  - Phân loại: detect/spotting sai; đúng vùng nhưng OCR sai; đúng sau hậu xử lý.
  - Lưu 10–20 ảnh lỗi tiêu biểu kèm GT vs pred.

- **Baseline YOLO (tùy chọn, không thay thế 2 cấu hình trên)**  
  Nếu cần so sánh nhanh với Buổi 3: có thể chạy thêm `python -m src.detector.run_buoi4_experiments` — chỉ là tham chiếu, không tính là một trong hai cấu hình chính của Buổi 4.

**Deliverables**

- Hai run thực nghiệm hoàn chỉnh (A/B) + log và checkpoint theo từng repo/convention.
- Bảng so sánh: CER, WER, plate-level accuracy, (latency).
- Tài liệu thao tác chi tiết: [`reports/buoi-4-cai-tien-mo-hinh-va-thuc-nghiem-chi-tiet.md`](reports/buoi-4-cai-tien-mo-hinh-va-thuc-nghiem-chi-tiet.md).
- Notebook hoặc [`reports/ablation_deepsolo_trocr.md`](reports/ablation_deepsolo_trocr.md) tóm tắt A/B và quyết định pipeline cho Buổi 5.
- Code hỗ trợ trong repo: [`src/eval/metrics_plate.py`](src/eval/metrics_plate.py), script so sánh A/B [`scripts/run_buoi4_experiments.py`](scripts/run_buoi4_experiments.py), khung pipeline [`src/pipeline/infer_plate_pipeline.py`](src/pipeline/infer_plate_pipeline.py), hướng dẫn cấu hình [`configs/deepsolo/README.md`](configs/deepsolo/README.md), [`configs/trocr/README.md`](configs/trocr/README.md).

## Buổi 5 – Tích hợp OCR & Đánh giá toàn hệ thống

**Mục tiêu**: Hoàn thiện pipeline detect + OCR, đánh giá định lượng trên ≥ 200 ảnh thực tế.

**Việc cần làm**

- Xây dựng **hàm tiền xử lý biển số** (sau khi crop từ YOLO):
  - Convert sang grayscale.
  - Resize về kích thước chuẩn (ví dụ 120×320 hoặc tương tự, giữ tỉ lệ).
  - Thử các cách threshold (Otsu, adaptive) để làm rõ ký tự.
  - (Tuỳ thời gian) Deskew nhẹ nếu biển bị nghiêng.
- Tích hợp **EasyOCR/Tesseract**:
  - Viết hàm `recognize_plate(cropped_plate) -> text`.
  - Hậu xử lý chuỗi: chuyển uppercase, loại bỏ ký tự lạ, xóa khoảng trắng không cần thiết.
  - Áp dụng regex/luật format biển số VN để sửa lỗi cơ bản (ví dụ thay `O` thành `0` ở những vị trí chỉ có digit).
- Chuẩn bị **bộ test OCR**:
  - Chọn ≥ 200 ảnh (hoặc patch biển số) thực tế, tự gán ground truth text.
  - Viết script chạy pipeline full: YOLO detect → crop → preprocess → OCR.
- Tính **chỉ số đánh giá**:
  - Character accuracy: số ký tự đúng / tổng ký tự.
  - Plate accuracy: số biển số đọc đúng hoàn toàn / tổng.
- Phân tích lỗi OCR:
  - Thống kê các loại lỗi: do detect sai, do ảnh mờ, do font lạ, do ánh sáng, do chữ quá bé.
  - Lưu lại ví dụ điển hình (ảnh + text đúng + text dự đoán) để đưa vào báo cáo.
- (Tùy chọn, nếu bám sát phiếu): thử Grad-CAM hoặc trực quan hoá feature YOLO cho vài case khó để minh họa mô hình học gì.

**Deliverables**

- Pipeline code end-to-end chạy được trên tập test.
- Kết quả đánh giá OCR với bảng số liệu rõ ràng.
- Bộ ảnh/lỗi tiêu biểu để dùng trong phần “Error analysis”.

## Buổi 6 – Demo & Giao diện người dùng

**Mục tiêu**: Có demo trực quan, chạy được với dữ liệu thực, tương tác đơn giản.

**Việc cần làm**

- Thiết kế demo:
  - Chọn 1 trong 3 kiểu: script CLI, web nhẹ (Streamlit/Gradio), hoặc app OpenCV GUI.
  - Chức năng cơ bản:
    - Chọn ảnh/video file → hiển thị kết quả detect + text biển số.
    - Hoặc chế độ webcam: hiển thị realtime với bbox và text.
- Tối ưu tối thiểu để demo mượt:
  - Có thể xử lý 1/2 hoặc 1/3 số frame cho video.
  - Cache kết quả khi biển số cùng ID tracking (nếu có time làm tracking đơn giản).
- Kiểm thử demo với các video/ảnh thực tế:
  - Quay 1–2 video ngắn ngoài thực tế (bãi xe/đường) để trình chiếu.
  - Ghi nhận crash, độ trễ, trường hợp đọc sai nghiêm trọng và sửa nếu kịp.
- Chuẩn bị **bộ case minh hoạ** cho buổi bảo vệ:
  - Vài ảnh “đẹp” (đọc đúng hoàn toàn).
  - Vài ảnh “khó” (đọc gần đúng hoặc sai) để minh hoạ giới hạn hệ thống.

**Deliverables**

- Demo chạy ổn định (script hoặc app).
- Bộ video/ảnh demo kèm theo hướng dẫn chạy.

## Buổi 7 – Báo cáo & Bảo vệ

**Mục tiêu**: Hoàn thiện tài liệu, slide và diễn tập bảo vệ.

**Việc cần làm**

- Viết **báo cáo chính thức** (theo mẫu môn học, ≥ 15 trang):
  - Giới thiệu, bài toán & ứng dụng.
  - Cơ sở lý thuyết: YOLO, OCR, tiền xử lý ảnh, format biển số VN.
  - Dữ liệu & phương pháp: cách thu thập/gán nhãn, kiến trúc pipeline, chi tiết mô hình.
  - Thực nghiệm & kết quả: mAP, character/plate accuracy, bảng so sánh, error analysis.
  - Demo & ứng dụng thực tế, hạn chế, hướng phát triển.
- Chuẩn bị **slide trình bày**:
  - 15–20 slide tóm tắt các ý chính, nhiều hình minh họa (pipeline, ví dụ input/output, biểu đồ kết quả).
- Diễn tập bảo vệ:
  - 1–2 lần chạy thử: trình bày slide + chạy demo live.
  - Liệt kê các câu hỏi giảng viên có thể hỏi (về dữ liệu, độ chính xác, vì sao chọn YOLO, giới hạn hệ thống, cách cải thiện) và chuẩn bị câu trả lời.
- Đóng gói **mã nguồn + dữ liệu mẫu + hướng dẫn chạy**:
  - README.md mô tả cách cài đặt, chạy train/inference/demo.
  - Nén project theo yêu cầu môn học.

**Deliverables**

- Báo cáo viết hoàn chỉnh.
- Slide trình bày.
- Bộ mã nguồn + demo đã kiểm tra chạy được trên máy đích (hoặc theo yêu cầu nộp).
