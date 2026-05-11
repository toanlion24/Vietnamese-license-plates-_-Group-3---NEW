# Báo cáo A/B Buổi 4 - DeepSolo end-to-end vs DeepSolo + TrOCR

Tự động cập nhật từ `scripts/run_buoi4_experiments.py`.

Ghi chú: Đây là dữ liệu demo/smoke-test để kiểm tra code metric, chưa phải kết quả mô hình thật.

**Bảo trì nội dung:** Các mục từ **mục 7** trở xuống là phần giải thích cố định (TrOCR / OCR Transformer, DeepSolo, RoBERTa). Nếu chạy `scripts/run_buoi4_experiments.py` với `--report-md` trỏ đúng vào file này, script sẽ **ghi đè toàn bộ file** — trước khi generate lại, hãy sao chép phụ lục hoặc dùng đường dẫn báo cáo khác (ví dụ `reports/ablation_deepsolo_trocr.md`).

## 1) Thiết lập

- Thời điểm tạo báo cáo: `2026-04-25T15:16:19.612004+00:00`
- Cấu hình A: DeepSolo end-to-end
- File A: `outputs\buoi4\demo\deepsolo_e2e_predictions.csv`
- Cấu hình B: DeepSolo + TrOCR
- File B: `outputs\buoi4\demo\deepsolo_trocr_predictions.csv`

## 2) Kiểm tra công bằng

- Không có cảnh báo.

## 3) Kết quả định lượng

Cấu hình A - DeepSolo end-to-end:

- Số mẫu: 8
- CER: 0.0469
- WER: 0.3750
- Plate accuracy: 0.6250
- Mean latency ms: 83.7500

Cấu hình B - DeepSolo + TrOCR:

- Số mẫu: 8
- CER: 0.0156
- WER: 0.1250
- Plate accuracy: 0.8750
- Mean latency ms: 145.9250

## 4) Phân bố lỗi

Cấu hình A:

- ok: 5
- ocr_or_spotting: 3

Cấu hình B:

- ok: 7
- ocr_or_spotting: 1

## 5) Quyết định tạm thời

Tạm chọn cấu hình B (DeepSolo + TrOCR) vì plate accuracy cao hơn.

## 6) Việc cần làm tiếp

- Mở các case sai và gán lại loại lỗi: `detect_miss`, `bad_crop`, `ocr_error`, `postprocess_helped`, `ambiguous_gt`.
- Chọn 10-20 hard cases để đưa vào báo cáo cuối.
- Nếu chọn cấu hình B, ưu tiên cải thiện crop/rectify trước khi fine-tune TrOCR.
- Nếu chọn cấu hình A, kiểm tra riêng lỗi spotting sai vùng và lỗi nhận dạng sai text.

## 7) Phụ lục — OCR Transformer, các kiểu kiến trúc, và `microsoft/trocr-base-printed`

### 7.1 OCR cổ điển so với OCR Transformer

- **Cách truyền thống** (ví dụ CRNN + CTC): CNN trích đặc trưng theo chiều ngang → chuỗi ẩn → **CTC** hoặc attention để căn chỉnh đầu ra với nhãn; thường thiết kế cho chuỗi ký tự cố định kiểu “một dòng”.
- **OCR kiểu Transformer (TrOCR)**: xem bài toán gần như **ảnh dòng chữ → sinh chuỗi token** (giống dịch máy/caption):
  - **Encoder ảnh** “nhìn” toàn bộ crop, biểu diễn bằng chuỗi token ảnh (patch).
  - **Decoder văn bản** sinh **tuần tự, tự hồi quy (autoregressive)** — mỗi bước dự đoán token tiếp theo từ biểu diễn ảnh và các token đã sinh.

Trong pipeline Buổi 4, **cấu hình B** (DeepSolo crop → TrOCR) khớp giả định của TrOCR: **một crop ~ một dòng chữ in**. Latency B thường cao hơn A vì thêm giai đoạn OCR và các bước `generate` của decoder.

### 7.2 “Các loại Transformer” trong bối cảnh OCR — nên nhóm theo vai trò

Không có một danh sách duy nhất; thực tế người ta phân theo **encoder / decoder / encoder–decoder**:

| Nhóm | Vai trò trong OCR | Ghi chú |
|------|-------------------|---------|
| Encoder-only (vision) | Trích đặc trưng ảnh; có thể gắn head CTC hoặc attention kiểu cũ | ViT, BEiT (TrOCR dùng tinh thần pretrain vision) |
| Encoder–decoder | Encoder hiểu ảnh, decoder **sinh chuỗi ký tự** từng bước | **TrOCR** (`VisionEncoderDecoder` trong Hugging Face `transformers`) |
| Kiến trúc khác (PARSeq, ABINet, CRNN, …) | Mỗi paper một thiết kế (autoregressive / non-autoregressive / tách nhận dạng ký tự) | So sánh khi viết related work, không trộn với “một loại Transformer” duy nhất |

Trên Hub, họ **TrOCR** còn tách theo:

- **Cỡ mô hình**: `small` / `base` / `large` (quy mô tham số, độ mạnh).
- **printed vs handwritten**: cùng họ kiến trúc nhưng **dữ liệu fine-tune** khác → phù hợp chữ in hay chữ viết tay.
- **Checkpoint cụ thể**: tên repo gắn với **bộ dữ liệu** đã fine-tune (ví dụ bản printed phổ biến dưới đây).

Repo này gọi TrOCR qua `src/ocr/trocr_adapter.py`: `TrOCRProcessor` + `VisionEncoderDecoderModel`, `model.generate()` rồi `batch_decode`.

### 7.3 Checkpoint Hugging Face `microsoft/trocr-base-printed`

Tham chiếu: [microsoft/trocr-base-printed](https://huggingface.co/microsoft/trocr-base-printed) và bài báo [TrOCR (arXiv:2109.10282)](https://arxiv.org/abs/2109.10282).

- **Ý nghĩa tên**: TrOCR **cỡ base**, checkpoint **fine-tune cho chữ in (printed)**; trên thẻ mô hình ghi là huấn luyện thêm trên dữ liệu **SROIE** (bối cảnh receipt / chữ in quét), không phải biển số Việt Nam.
- **Kiến trúc (theo paper / Hub)**:
  - **Encoder ảnh**: Transformer trên ảnh; khởi tạo từ **BEiT**.
  - **Decoder văn bản**: Transformer; khởi tạo từ **RoBERTa** (xem **mục 9**) — tokenizer/subword thiên về ngữ cảnh Latin/Anh; ký tự số và chữ Latin trên biển số vẫn chạy được, nhưng **quy tắc định dạng biển VN** do hậu xử lý (`normalize_plate_text` trong repo), không do TrOCR “học thuộc”.
- **Đầu vào ảnh**: ảnh được cắt **patch 16×16**, nhúng (embed) và cộng **positional encoding**, qua các lớp encoder.
- **Đầu ra**: decoder sinh token **tuần tự** (đúng với lời gọi `generate` trong pipeline).
- **Quy mô**: khoảng **~0,33B tham số** (metadata Hub ~333M tham số float32, hay làm tròn **~0,3B**).
- **Định dạng trên Hub**: PyTorch, thường **safetensors**; pipeline tag **image-to-text** / kiến trúc **vision-encoder-decoder**.

### 7.4 Liên hệ với thực nghiệm Buổi 4 trong file này

- Kết quả bảng ở **mục 3** là **demo/smoke-test** (ít mẫu), dùng để kiểm tra metric và báo cáo — **không** khẳng định TrOCR luôn vượt end-to-end trên mọi dữ liệu thật.
- **Domain gap**: SROIE / chữ in receipt ≠ ảnh biển số ngoài đường; nếu baseline inference chưa đủ, lộ trình khuyến nghị trong **mục 6** vẫn hợp lý: **cải thiện crop, rectify, tiền xử lý**, rồi mới xét **fine-tune TrOCR** khi có đủ cặp (ảnh crop, nhãn).

## 8) Phụ lục — DeepSolo (text spotting)

### 8.1 DeepSolo là gì?

**DeepSolo** là mô hình **text spotting** trên ảnh cảnh: từ ảnh đầu vào, mô hình trả về **vùng chữ** (bbox hoặc polygon) và, ở chế độ end-to-end, **chuỗi ký tự** tương ứng. Bài báo gốc: *DeepSolo: Let Transformer Decoder with Explicit Points Solo for Text Spotting* (CVPR 2023). Repo chính thức: [ViTAE-Transformer/DeepSolo](https://github.com/ViTAE-Transformer/DeepSolo). Có hướng mở rộng **DeepSolo++** (đa ngôn ngữ, cùng tổ chức trên GitHub).

Ý tưởng tổng quát: dùng **Transformer** theo phong cách **DETR** — giải mã từng instance chữ bằng **queries**, với **điểm biên rõ ràng (explicit points)** cho hình dạng text (chữ cong, dài, nhiều hướng) thay vì chỉ một bbox chữ nhật đơn giản.

### 8.2 Vai trò trong Buổi 4 và trong repo này

Theo `configs/deepsolo/README.md`, repo **không nhúng full code DeepSolo** (tránh nặng dependency); luồng khuyến nghị là **train/infer ở repo DeepSolo riêng**, rồi **export CSV** đúng schema (`image_id`, `gt`, `pred`, …) để `scripts/run_buoi4_experiments.py` đánh giá.

Hai cách dùng cần **phân biệt khi báo cáo**:

| Cấu hình | Vai trò DeepSolo | Ghi chú |
|----------|------------------|---------|
| **A — end-to-end** | Text **spotting** đầy đủ: vùng + chuỗi trong **một** pipeline | Khó tách lỗi do sai **định vị** hay sai **nhận dạng** trong cùng mô hình |
| **B — + TrOCR** | **Localize / crop** (polygon hoặc bbox) → crop đưa cho **TrOCR** | Dễ cải thiện crop/rectify và OCR tách bước; latency thường cao hơn (hai giai đoạn + `generate`) |

### 8.3 So sánh nhanh với detector thuần (YOLO) và TrOCR

- **Detector (YOLO, …)** chủ yếu trả lời: *vật thể / biển số ở đâu?* (bbox class plate).
- **DeepSolo end-to-end** trả lời: *chữ ở đâu **và** đọc được gì?* (spotting).
- **TrOCR** trên crop trả lời: *ảnh **một dòng** này là chuỗi gì?* — không thay thế bước tìm vùng trên ảnh lớn.

Kết quả số trong **mục 3** (nếu là demo) phản ánh **smoke-test**; khi có checkpoint thật, thay file prediction CSV và chạy lại metric.

## 9) Phụ lục — RoBERTa và vai trò trong TrOCR

### 9.1 RoBERTa là gì?

**RoBERTa** (*Robustly optimized BERT approach*) là mô hình **ngôn ngữ dạng Transformer encoder-only**, cùng họ **BERT**: đầu vào là văn bản được token hóa (thường subword/BPE), đầu ra là **biểu diễn ngữ cảnh** cho từng token (hai chiều). RoBERTa **không** sinh văn kiểu GPT; mục tiêu pre-train chủ yếu là **Masked Language Modeling (MLM)**: che bớt token và học dự đoán token bị che.

So với BERT gốc, RoBERTa nhấn mạnh **cách train**: ví dụ **bỏ** objective **Next Sentence Prediction (NSP)**, dùng **dynamic masking**, train **lâu hơn / batch lớn hơn / dữ liệu rộng hơn** — nên thường coi là **cải tiến huấn luyện BERT**, không phải kiến trúc hoàn toàn mới.

### 9.2 RoBERTa xuất hiện ở đâu trong TrOCR của bạn?

Trong TrOCR (và checkpoint **`microsoft/trocr-base-printed`**), phần **decoder văn bản** được mô tả là **khởi tạo trọng số từ RoBERTa** trước khi fine-tune OCR.

Ý nghĩa thực tế khi dùng `src/ocr/trocr_adapter.py`:

- Bạn **không** gọi `RobertaModel` trực tiếp; bạn load **`VisionEncoderDecoderModel`** đã chứa kiến trúc đầy đủ.
- RoBERTa mang lại **điểm khởi đầu tốt** cho nhánh xử lý **chuỗi token văn bản Latin**; sau đó mô hình được học nhiệm vụ **sinh chuỗi từ đặc trưng ảnh** do encoder ảnh (BEiT-class) cung cấp.
- Quy tắc format biển số Việt Nam vẫn do **hậu xử lý** trong repo (`normalize_plate_text`), không do RoBERTa/TrOCR “biết luật” sẵn.

### 9.3 Tóm tắt một dòng

**RoBERTa** là encoder ngôn ngữ (BERT-class, MLM, tối ưu train); trong TrOCR nó là **nguồn khởi tạo** cho decoder văn bản trước khi học đọc chữ từ ảnh.
