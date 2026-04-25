# Buổi 3 - Xây dựng mô hình baseline Detection (YOLO) chi tiết

Tài liệu này giúp bạn triển khai trọn Buổi 3 theo hướng thực hành: từ chuẩn bị dữ liệu YOLO, huấn luyện baseline, chạy inference, đến chốt kết quả để chuyển sang Buổi 4.

---

## 1) Mục tiêu Buổi 3

- Huấn luyện được một mô hình YOLO baseline để phát hiện biển số.
- Lưu được checkpoint tốt nhất và log huấn luyện trong `experiments/`.
- Chạy được inference trên ảnh test/ảnh mới và xuất kết quả để kiểm tra.
- Ghi nhận lỗi điển hình (miss, false positive, biển nhỏ, biển nghiêng, nền khó).

---

## 2) Đầu vào, đầu ra kỳ vọng

### Đầu vào

- Dữ liệu ảnh: `data/raw/`
- Nhãn YOLO: `data/labels/` (mỗi ảnh có file `.txt` cùng tên)
- Split từ Buổi 2: `data/splits/train.txt`, `data/splits/val.txt`, `data/splits/test.txt`

### Đầu ra

- File cấu hình YOLO: `data/data.yaml`
- Thư mục run train: `experiments/yolo_buoi3_*`
- Checkpoint tốt nhất: `experiments/.../weights/best.pt`
- Kết quả inference JSON: `outputs/predictions_buoi3.json`
- Ảnh minh họa đúng/sai: `reports/buoi3_samples/` (khuyến nghị tự tạo)

---

## 3) Checklist thao tác theo thứ tự

## Bước 0 - Cài môi trường và thư viện

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install ultralytics
```

Ghi chú:
- Nếu đã có môi trường từ Buổi 2, chỉ cần kích hoạt lại và cài bổ sung `ultralytics` nếu thiếu.

## Bước 1 - Kiểm tra dữ liệu trước khi train

- Đảm bảo tổng số dòng của `train.txt`, `val.txt`, `test.txt` khớp số ảnh đã split.
- Đảm bảo mỗi ảnh trong split có nhãn `.txt` cùng tên trong `data/labels/`.
- Nhãn chỉ có 1 class duy nhất: `license_plate` (class id `0`).

Khuyến nghị kiểm tra nhanh:

```bash
python scripts/eda_dataset.py --images-dir data/raw --labels-dir data/labels --output-dir reports/eda_buoi3 --num-preview 20
```

## Bước 2 - Tạo file `data/data.yaml`

Tạo file `data/data.yaml` với nội dung:

```yaml
path: .
train: data/splits/train.txt
val: data/splits/val.txt
test: data/splits/test.txt

names:
  0: license_plate
```

Lưu ý:
- Các file split nên chứa đường dẫn ảnh theo chuẩn dùng được trực tiếp (tương đối hoặc tuyệt đối).
- Nếu split hiện tại dùng kiểu đường dẫn cũ, chuẩn hóa lại bằng script split trước khi train.

## Bước 3 - Huấn luyện baseline YOLO

### Cách A (khuyến nghị, chạy trực tiếp bằng Ultralytics CLI)

```bash
yolo task=detect mode=train model=yolov8n.pt data=data/data.yaml epochs=50 imgsz=640 batch=16 project=experiments name=yolo_buoi3_baseline
```

Nếu GPU yếu hoặc thiếu VRAM:
- Giảm `batch` xuống `8` hoặc `4`.
- Giữ `imgsz=640` để cân bằng tốc độ và chất lượng.

### Cách B (dùng script Python của repo)

Script `scripts/train_detector.py` hiện là khung placeholder; nếu bạn muốn thống nhất entrypoint theo repo, cập nhật script này để gọi Ultralytics API và lưu run vào `experiments/`.

## Bước 4 - Theo dõi kết quả train

Cần ghi lại các chỉ số:
- `precision`, `recall`, `mAP50`, `mAP50-95`
- Loss chính qua các epoch đầu và cuối

Checkpoint cần lấy:
- `best.pt` để suy luận
- `last.pt` để resume khi cần

## Bước 5 - Chạy inference sau huấn luyện

Copy checkpoint tốt nhất:

```bash
mkdir weights
copy experiments\yolo_buoi3_baseline\weights\best.pt weights\yolov8_license_plate.pt
```

Chạy suy luận batch bằng script hiện có của repo:

```bash
python scripts/run_infer.py --input-dir data/raw --output-json outputs/predictions_buoi3.json --detector-backend yolov8 --detector-model weights/yolov8_license_plate.pt --ocr-backend dummy
```

Giải thích nhanh:
- Buổi 3 tập trung detection baseline nên có thể để `--ocr-backend dummy` để kiểm tra riêng nhánh detect.
- Sang Buổi 5 mới đánh giá OCR đầy đủ (CER/WER/plate accuracy).

## Bước 6 - Lưu ảnh minh họa đúng/sai để báo cáo

Chuẩn bị tối thiểu:
- 10 ảnh detect tốt (góc chụp đa dạng).
- 10 ảnh lỗi điển hình (biển nhỏ, mờ, ngược sáng, nền nhiễu).

Với mỗi ảnh lỗi, ghi chú:
- Loại lỗi: miss / false positive / box lệch.
- Nguyên nhân khả dĩ.
- Hướng cải tiến ở Buổi 4 (thêm dữ liệu khó, tuning augment, đổi backbone).

---

## 4) Gói lệnh đầy đủ Buổi 3 (copy chạy nhanh)

```bash
pip install ultralytics
python scripts/eda_dataset.py --images-dir data/raw --labels-dir data/labels --output-dir reports/eda_buoi3 --num-preview 20
yolo task=detect mode=train model=yolov8n.pt data=data/data.yaml epochs=50 imgsz=640 batch=16 project=experiments name=yolo_buoi3_baseline
copy experiments\yolo_buoi3_baseline\weights\best.pt weights\yolov8_license_plate.pt
python scripts/run_infer.py --input-dir data/raw --output-json outputs/predictions_buoi3.json --detector-backend yolov8 --detector-model weights/yolov8_license_plate.pt --ocr-backend dummy
```

---

## 5) Tiêu chí hoàn thành Buổi 3

- Có `data/data.yaml` hợp lệ và train chạy ổn định hết epoch.
- Có checkpoint `best.pt` trong thư mục `experiments/`.
- Có file kết quả inference `outputs/predictions_buoi3.json`.
- Có bộ ảnh minh họa đúng/sai để đưa vào báo cáo.
- Có ghi chú chỉ số baseline để so sánh với các cấu hình Buổi 4.

---

## 6) Lỗi thường gặp và cách xử lý nhanh

- `FileNotFoundError` khi train:
  - Kiểm tra lại đường dẫn trong `data/data.yaml` và nội dung `train.txt/val.txt`.
- `CUDA out of memory`:
  - Giảm `batch`, có thể giảm nhẹ `imgsz`.
- Model không học (mAP rất thấp):
  - Kiểm tra lại nhãn sai format hoặc bbox lệch.
  - Tăng chất lượng dữ liệu/nhãn trước khi tăng độ phức tạp model.
- Inference không ra box:
  - Giảm `--detector-conf` (ví dụ `0.25` xuống `0.15`) để kiểm tra độ nhạy.

---

## 7) Bàn giao sang Buổi 4

Khi kết thúc Buổi 3, bạn nên chốt:
- Baseline checkpoint tốt nhất.
- Bộ lỗi khó đại diện.
- Bảng chỉ số baseline detection.

Đây là mốc để so sánh công bằng với thực nghiệm Buổi 4 (DeepSolo và DeepSolo + TrOCR).
