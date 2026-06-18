---
name: do-an-nhan-dien-bien-so-vn
overview: "Kế hoạch chi tiết 7 buổi cho đồ án thị giác máy tính: phát hiện và OCR biển số xe Việt Nam; baseline YOLO + OpenCV; Buổi 4 fine-tune Qwen2-VL-2B-Instruct với Unsloth + QLoRA, có demo và báo cáo hoàn chỉnh."
todos:
  - id: buoi-1-planning
    content: Hoàn thành mô tả bài toán, khảo sát công nghệ, vẽ pipeline, phân công và timeline 7 buổi
    status: completed
  - id: buoi-2-data
    content: Thu thập và gán nhãn dữ liệu biển số VN, viết script EDA và tiền xử lý/augment cơ bản
    status: completed
  - id: buoi-3-baseline-detector
    content: Cấu hình project YOLO và train mô hình detection baseline, chạy inference thử trên ảnh test
    status: completed
  - id: buoi-4-finetune-qwen
    content: Fine-tune Qwen2-VL-2B-Instruct với Unsloth + QLoRA trên Google Colab, lưu checkpoint lên Hugging Face
    status: completed
  - id: buoi-5-pipeline-integration
    content: Tích hợp pipeline YOLOv8n + Qwen2-VL, đánh giá CER WER plate-level latency trên ≥200 ảnh biển số thực tế và phân tích lỗi
    status: completed
  - id: buoi-6-demo
    content: Xây dựng demo (CLI/web/GUI) chạy được trên ảnh/video/webcam, kiểm thử với dữ liệu thực tế
    status: completed
  - id: buoi-7-report
    content: Hoàn thiện báo cáo, slide, diễn tập bảo vệ và đóng gói mã nguồn + hướng dẫn chạy
    status: pending
isProject: false
---

# Kế hoạch đồ án: Nhận diện biển số xe Việt Nam

## Mục tiêu tổng thể

- Xây dựng hệ thống nhận diện biển số xe Việt Nam gồm 2 phần: **phát hiện vùng biển số (detection)** và **trích xuất ký tự (OCR/VLM)**.
- Đạt **độ chính xác OCR ≥ 85%** trên **≥ 200 ảnh biển số thực tế** (không dùng để train).
- Có **demo** chạy được trên ảnh/video/webcam, kèm **báo cáo và slide** đầy đủ.

## Công nghệ & tổ chức mã nguồn

### Stack công nghệ chính

| Thành phần | Công nghệ / Công cụ |
|------------|---------------------|
| Nhận diện (Detection) | YOLOv8 Nano (YOLOv8n) |
| Trích xuất (OCR/VLM) | Qwen2-VL-2B-Instruct |
| Fine-tuning | Unsloth (Tối ưu hóa VRAM/Tốc độ) |
| Kỹ thuật huấn luyện | QLoRA (4-bit quantization) |
| Giao diện (UI) | Streamlit |
| Môi trường | Google Colab, Hugging Face |

### Ngôn ngữ & Thư viện

- **Ngôn ngữ**: Python.
- **Thư viện chính**:
  - Detection: YOLOv8 (Ultralytics)
  - OCR/VLM: Qwen2-VL-2B-Instruct (fine-tuned với Unsloth)
  - Fine-tuning: Unsloth, bitsandbytes, peft, trl
  - Xử lý ảnh: OpenCV, NumPy, PIL
  - Phân tích & demo: Matplotlib/Seaborn, Streamlit

### Cấu trúc thư mục

```
data/
  raw/                    # Ảnh gốc thu thập
  labels/                 # File nhãn YOLO cho biển số
  splits/                 # Danh sách train/val/test
  manifests/              # Manifest cho OCR training
  crops/                  # Crop biển số cho VLM training
experiments/
  yolo/                   # Log huấn luyện, checkpoint YOLO
  qwen_vl/                # Checkpoint Qwen2-VL fine-tuned
src/
  io/                     # Đọc ảnh, video, webcam
  detector/               # YOLO detector adapter
  ocr/                    # Qwen2-VL adapter
  preprocess/             # Tiền xử lý ảnh & biển số
  postprocess/            # Hậu xử lý regex/luật biển số VN
  pipeline/               # Pipeline tổng hợp
  eval/                   # Metrics, error analysis
  app/                    # Demo CLI/GUI
  utils/                  # Utilities
scripts/
  train_yolo.py           # Train YOLO detector
  train_qwen.py           # Fine-tune Qwen2-VL (chạy trên Colab)
  run_inference.py        # Inference batch
  eval_pipeline.py        # Đánh giá pipeline
configs/
  yolo/                   # Cấu hình YOLO
  qwen_vl/                # Cấu hình QLoRA, prompt
  pipeline/               # Ngưỡng, regex, voting
reports/                  # Báo cáo từng buổi
notebooks/                # EDA, thử nghiệm
```

## Kiến trúc Pipeline

### 1) Pipeline xử lý 4 giai đoạn

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  DETECTION  │───▶│  CROPPING   │───▶│   OCR/VLM   │───▶│ POST-PROCESS│
│  YOLOv8n     │    │  Auto-crop  │    │ Qwen2-VL-2B │    │ Regex+Rules │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

1. **Detection (Nhận diện)**: YOLOv8n quét khung hình, xác định tọa độ bounding box của biển số.
2. **Cropping (Cắt ảnh)**: Tự động cắt vùng ảnh chứa biển số, loại bỏ nhiễu từ môi trường xung quanh.
3. **OCR/VLM (Trích xuất)**: Qwen2-VL-2B-Instruct (fine-tuned) đọc và chuyển đổi hình ảnh biển số thành văn bản.
4. **Post-processing (Xử lý hậu kỳ)**: Lọc chuỗi văn bản, loại ký tự rác, chuẩn hóa format theo quy chuẩn biển số VN.

### 2) Data contract giữa các module

- **Detector output**: `image_id`, `bbox_xyxy`, `score`, `class_name`
- **OCR input**: `plate_crop` (PIL Image), `image_id`, `det_score`
- **OCR output**: `text_raw`, `text_norm`, `ocr_score`
- **Final output**: `plate_text`, `bbox`, `confidence`, `source_frame`, `timestamp`

### 3) Non-functional requirements

- **Reproducibility**: Mỗi lần train lưu `config + seed + git_sha`
- **Observability**: Log latency từng stage (detect/crop/ocr/postprocess)
- **Maintainability**: Mỗi module có unit test cho hàm core
- **Safety**: Không commit dữ liệu nhạy cảm, không lưu biển số thật trong public report

## Chuẩn bị Dataset cho Fine-tuning Qwen2-VL

### Định dạng training data

Qwen2-VL yêu cầu định dạng conversation:

```json
{
  "messages": [
    {"role": "user", "content": "<image>Đọc biển số xe trong ảnh"}, 
    {"role": "assistant", "content": "30G1-12345"}
  ]
}
```

### Cấu trúc manifest cho OCR training

```csv
image_id,image_path,text_gt,split
plate_0001,/path/to/crops/plate_0001.jpg,30G112345,train
plate_0002,/path/to/crops/plate_0002.jpg,51K123456,train
...
```

## Buổi 1 – Khởi động & Lập kế hoạch chi tiết

**Mục tiêu**: Hiểu rõ bài toán, chốt stack công nghệ (YOLOv8n + Qwen2-VL-2B-Instruct + Unsloth), pipeline, kế hoạch 7 buổi và phân công.

**Việc cần làm**

- Viết mô tả bài toán ngắn gọn (1–2 trang):
  - Bối cảnh ứng dụng (bãi gửi xe, camera đường phố…)
  - Input: ảnh/video/webcam chứa xe
  - Output: chuỗi biển số + bounding box
  - Chỉ số đánh giá: mAP cho detection; CER, WER, plate accuracy cho OCR/VLM

- Khảo sát nhanh các hướng kỹ thuật:
  - YOLOv8n cho detection (đã chọn)
  - Qwen2-VL-2B-Instruct cho OCR (thay TrOCR)
  - Unsloth để fine-tune nhanh hơn (tiết kiệm VRAM)
  - Thử chạy mô hình pre-trained trên vài ảnh mẫu để kiểm tra môi trường

- Phân tích **format biển số Việt Nam**:
  - Biển 1 dòng/2 dòng, nền trắng/vàng/xanh
  - Viết ra regex/luật cơ bản cho biển số (xe máy, ô tô)

- Vẽ sơ đồ **pipeline tổng thể**:
  - Ảnh/video → YOLOv8n detect → crop → Qwen2-VL OCR → postprocess regex

- Lập **timeline 7 buổi** và phân công:
  - Thành viên A: dữ liệu + gán nhãn
  - Thành viên B: mô hình YOLO
  - Thành viên C: Fine-tune Qwen2-VL + pipeline OCR
  - Cả nhóm: demo + báo cáo

**Deliverables**

- Tài liệu mô tả bài toán & yêu cầu
- Sơ đồ pipeline (YOLOv8n → Qwen2-VL-2B-Instruct)
- Ghi chú phân công nhiệm vụ và lịch làm việc
- Tài liệu triển khai chi tiết: `reports/buoi-1-khoi-dong-va-lap-ke-hoach-chi-tiet.md`

## Buổi 2 – Thu thập & Tiền xử lý dữ liệu

**Mục tiêu**: Có tập dữ liệu bước đầu, đã gán nhãn và có script EDA + tiền xử lý cơ bản.

**Việc cần làm**

- **Thu thập dữ liệu**:
  - Tự chụp ảnh/đoạn video xe máy, ô tô trên đường hoặc trong bãi giữ xe
  - Trích frame từ video nếu cần để tăng số lượng ảnh
  - Mục tiêu: tối thiểu 300–500 ảnh có chứa biển số VN rõ ràng

- **Gán nhãn detection (YOLO)**:
  - Dùng LabelImg/Roboflow để vẽ bounding box quanh biển số
  - Xuất nhãn theo định dạng YOLO (file `.txt` cùng tên ảnh)
  - Hoàn thành ít nhất ~200 ảnh gán nhãn trong buổi 2

- **Gán nhãn OCR cho Qwen2-VL**:
  - Chuẩn bị manifest CSV: `image_id`, `image_path`, `text_gt`
  - Tạo thư mục `data/crops/` chứa ảnh crop biển số đã detect
  - Mục tiêu: ít nhất 100-200 cặp crop-GT cho fine-tuning

- **Tách tập train/val/test**:
  - Viết script split, lưu vào `data/splits/{train,val,test}.txt`

- **EDA & tiền xử lý**:
  - Notebook đọc danh sách ảnh + nhãn, hiển thị sample kèm box
  - Thống kê: số ảnh, kích thước box trung bình, phân bố điều kiện ánh sáng
  - Hàm tiền xử lý: resize, normalize, augment (brightness, contrast, small rotation)

**Deliverables**

- Thư mục dữ liệu đã tổ chức, có nhãn cho ≥ 200 ảnh (detection)
- Manifest OCR: ≥ 100 cặp crop-GT trong `data/manifests/ocr_training.csv`
- Notebook/Script EDA
- Script tiền xử lý/augment cơ bản
- Tài liệu: `reports/buoi-2-thu-thap-va-tien-xu-ly-du-lieu-chi-tiet.md`

## Buổi 3 – Xây dựng mô hình baseline (Detection)

**Mục tiêu**: Train được mô hình YOLO baseline và chạy inference được.

**Việc cần làm**

- Chuẩn bị **file cấu hình YOLO**:
  - `data.yaml`: đường dẫn tới image/label train/val, class `license_plate`
  - Kiểm tra cấu trúc thư mục đúng YOLO yêu cầu

- Cài đặt & cấu hình môi trường YOLO (Ultralytics)

- Train baseline:
  - Model nhỏ: YOLOv8n (nano) để train nhanh
  - Hyperparameter mặc định: epochs ~50, batch size tùy GPU, image size 640

- Inference trên tập test:
  - Script nhận đường dẫn ảnh → vẽ bounding box ra file/hiển thị
  - Quan sát lỗi: biển nhỏ không detect, detect nhầm background

**Deliverables**

- Checkpoint YOLO baseline: `runs/detect/experiments/detector/yolov8n_augmented/weights/best.pt`
  - mAP50: **0.9471** (epoch 22), mAP50-95: **0.6382**
  - Dataset: 279 train + 31 val augmented images
- Script train + inference
- Ảnh minh họa kết quả detect (đúng/sai)

## Buổi 4 – Fine-tune Qwen2-VL-2B-Instruct với Unsloth + QLoRA

**Mục tiêu**: Fine-tune Qwen2-VL-2B-Instruct sử dụng Unsloth để tối ưu VRAM/tốc độ, lưu checkpoint lên Hugging Face.

**Công nghệ sử dụng**

| Thành phần | Tool |
|------------|------|
| Base Model | Qwen2-VL-2B-Instruct |
| Fine-tuning Framework | Unsloth |
| Quantization | QLoRA (4-bit) |
| Platform | Google Colab (GPU T4/A100) |
| Storage | Hugging Face Hub |

**Việc cần làm**

- **Chuẩn bị dữ liệu fine-tuning**:
  - Format conversation cho Qwen2-VL:
    ```
    messages = [
      {"role": "user", "content": [{"type": "image"}, "Đọc biển số xe trong ảnh"]},
      {"role": "assistant", "content": "30G1-12345"}
    ]
    ```
  - Tạo JSONL file cho training
  - Tối thiểu 100-200 cặp crop-GT

- **Cấu hình Unsloth + QLoRA**:
  - 4-bit quantization với NF4
  - LoRA config: rank=16, alpha=32, dropout=0.1
  - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  - Learning rate: 2e-4, epochs: 3-5, batch size: 1-2 (tùy VRAM)

- **Huấn luyện trên Google Colab**:
  - Cài đặt Unsloth: `pip install unsloth unsloth_granite`
  - Load model với Unsloth:
    ```python
    from unsloth import FastVisionModel
    model, tokenizer = FastVisionModel.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth"
    )
    ```
  - Apply LoRA:
    ```python
    model = FastVisionModel.get_peft_model(
        model,
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", ...],
        use_gradient_checkpointing="unsloth"
    )
    ```
  - Train và theo dõi loss
  - Lưu checkpoint tốt nhất

- **Đẩy lên Hugging Face**:
  - Push adapter đã fine-tune:
    ```python
    model.push_to_hub_LoRA("username/vn-plate-qwen2-vl-2b")
    ```
  - Hoặc push full model nếu cần

- **Test inference cơ bản**:
  - Chạy inference trên vài ảnh test bằng model đã fine-tune
  - So sánh với baseline (model chưa fine-tune)

**Deliverables**

- Checkpoint Qwen2-VL-2B fine-tuned trên Hugging Face: `username/vn-plate-qwen2-vl-2b`
- Script fine-tuning (chạy được trên Colab): `scripts/train_qwen_colab.ipynb`
- Log training: loss curve, sample outputs
- So sánh baseline vs fine-tuned: vài ví dụ input/output
- Tài liệu: `reports/buoi-4-finetune-qwen2vl-voi-unsloth-chi-tiet.md`

## Buổi 5 – Tích hợp Pipeline & Đánh giá toàn hệ thống

**Mục tiêu**: Hoàn thiện pipeline YOLOv8n + Qwen2-VL-2B, đánh giá trên ≥ 200 ảnh thực tế.

**Việc cần làm**

- **Chốt data contract cho đánh giá**:
  - Tạo/kiểm tra manifest test cố định: `image_id`, `image_path`, `text_gt`
  - Test set ≥ 200 ảnh thực tế, không trùng cảnh với train/val
  - Lưu manifest ở `data/manifests/buoi5_test.csv`

- **Tích hợp pipeline YOLOv8n + Qwen2-VL**:
  - Load YOLO detector từ checkpoint Buổi 3
  - Load Qwen2-VL fine-tuned từ Hugging Face (Buổi 4)
  - Pipeline flow: frame → detect → crop → Qwen2-VL OCR → postprocess

- **Tiền xử lý crop trước OCR**:
  - Chuẩn hóa kích thước crop (resize về 224x224 hoặc giữ tỉ lệ)
  - Tăng tương phản nhẹ, denoise nếu cần

- **Hậu xử lý biển số VN**:
  - Regex chuẩn hóa: uppercase, bỏ ký tự lạ
  - Sửa lỗi OCR thường gặp: O↔0, I/L↔1
  - Luật format biển số VN: xe máy 2 dòng, ô tô 1 dòng

- **Chạy inference batch**:
  - Đầu ra CSV: `image_id`, `gt`, `pred_raw`, `pred_norm`, `bbox`, `latency_ms`, `error_type`
  - Lưu vào `outputs/buoi5/predictions.csv`

- **Đánh giá định lượng**:
  - **CER**: tổng edit distance ký tự / tổng ký tự GT
  - **WER**: edit distance token / tổng token GT
  - **Plate-level accuracy**: tỉ lệ pred_norm == gt sau chuẩn hóa
  - **Mean latency ms**: thời gian trung bình cho detect + OCR + postprocess

- **Phân tích lỗi có hệ thống**:
  - Phân loại lỗi: detect_miss, bad_crop, ocr_error, postprocess_helped, ambiguous_gt
  - Lưu 10-20 hard cases tiêu biểu vào `outputs/buoi5/hard_cases/`

**Deliverables**

- Pipeline end-to-end chạy được trên ≥ 200 ảnh test
- File prediction: `outputs/buoi5/predictions.csv`
- Báo cáo metric: `reports/buoi5_metrics.json` (CER, WER, plate accuracy, latency)
- File lỗi chi tiết: `reports/buoi5_error_records.csv`
- Thư mục hard cases: `outputs/buoi5/hard_cases/`
- Checklist quyết định:
  - [ ] Test set ≥ 200 ảnh đã cố định
  - [ ] Pipeline YOLOv8n + Qwen2-VL chạy hết test set
  - [ ] Có CER/WER/plate accuracy/latency
  - [ ] Có 10-20 hard cases đã phân loại lỗi

## Buổi 6 – Demo & Giao diện người dùng

**Mục tiêu**: Demo trực quan chạy được với dữ liệu thực, giao diện đơn giản với Streamlit.

**Việc cần làm**

- Thiết kế demo Streamlit:
  - Upload ảnh/video → hiển thị kết quả detect + text biển số
  - Hoặc chế độ webcam realtime
  - Hiển thị bbox + plate text overlay

- Tối ưu cho demo mượt:
  - Xử lý 1/2 hoặc 1/3 số frame cho video
  - Cache kết quả khi biển số cùng ID tracking

- Kiểm thử demo với video/ảnh thực tế:
  - Quay 1-2 video ngoài thực tế để trình chiếu
  - Ghi nhận crash, độ trễ, trường hợp đọc sai

- Chuẩn bị bộ case minh hoạ cho buổi bảo vệ:
  - Vài ảnh "đẹp" (đọc đúng hoàn toàn)
  - Vài ảnh "khó" (đọc gần đúng hoặc sai)

**Deliverables**

- Demo Streamlit chạy ổn định
- Bộ video/ảnh demo kèm hướng dẫn chạy
- Tài liệu: `reports/buoi-6-demo-va-giao-dien-nguoi-dung.md`

## Buổi 7 – Báo cáo & Bảo vệ

**Mục tiêu**: Hoàn thiện tài liệu, slide và diễn tập bảo vệ.

**Việc cần làm**

- Viết **báo cáo chính thức** (≥ 15 trang):
  - Giới thiệu, bài toán & ứng dụng
  - Cơ sở lý thuyết: YOLOv8, Qwen2-VL, Unsloth, QLoRA
  - Dữ liệu & phương pháp: thu thập/gán nhãn, kiến trúc pipeline, chi tiết fine-tuning
  - Thực nghiệm & kết quả: mAP, CER/WER/plate accuracy, bảng so sánh, error analysis
  - Demo & ứng dụng thực tế, hạn chế, hướng phát triển

- Chuẩn bị **slide trình bày**:
  - 15-20 slide tóm tắt ý chính
  - Nhiều hình minh họa (pipeline, ví dụ input/output, biểu đồ kết quả)

- Diễn tập bảo vệ:
  - Chạy thử: trình bày slide + demo live
  - Chuẩn bị câu trả lời cho câu hỏi giảng viên

- Đóng gói **mã nguồn + dữ liệu mẫu + hướng dẫn chạy**:
  - README.md mô tả cách cài đặt, chạy train/inference/demo
  - Nén project theo yêu cầu nộp

**Deliverables**

- Báo cáo viết hoàn chỉnh
- Slide trình bày
- Bộ mã nguồn + demo đã kiểm tra chạy được

## So sánh công nghệ (Trước vs Sau cập nhật)

| Thành phần | Plan cũ | Plan mới |
|------------|---------|----------|
| Detection | YOLOv8 / DeepSolo | YOLOv8n |
| OCR/VLM | TrOCR / DeepSolo | Qwen2-VL-2B-Instruct |
| Fine-tuning | TrOCR fine-tune | Qwen2-VL fine-tune với Unsloth |
| Optimization | - | QLoRA (4-bit), Unsloth |
| Môi trường | Local | Google Colab + Hugging Face |
| Buổi 4 | DeepSolo vs DeepSolo+TrOCR | Fine-tune Qwen2-VL với Unsloth |

## Checklist triển khai

### Trước Buổi 4 (Fine-tuning)

- [ ] Có ≥ 100 cặp crop-GT cho Qwen2-VL training
- [ ] Có tài khoản Hugging Face (để push model)
- [ ] Có quyền truy cập Google Colab (GPU)
- [ ] Đã cài đặt Unsloth: `pip install unsloth unsloth_granite`

### Sau Buổi 4

- [ ] Model đã push lên Hugging Face
- [ ] Có script inference với model fine-tuned
- [ ] Có log training và sample outputs

### Trước Buổi 5

- [ ] Test set ≥ 200 ảnh đã chuẩn bị
- [ ] Pipeline YOLOv8n + Qwen2-VL tích hợp xong
- [ ] Chạy được inference trên batch ảnh
