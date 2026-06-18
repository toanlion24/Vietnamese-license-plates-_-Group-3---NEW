# Buổi 6: Demo & Giao diện người dùng

**Ngày:** 2026-06-13  
**Trạng thái:** ✅ Hoàn thành

---

## Mục tiêu

Xây dựng demo trực quan chạy được với dữ liệu thực, giao diện đơn giản với Streamlit.

---

## Deliverables

### 1. Streamlit Web App (`src/app/demo.py`)

Tính năng:
- **Tab Image**: Upload ảnh, nhận diện và hiển thị kết quả
- **Tab Video**: Upload video, xử lý frame-by-frame
- **Tab Webcam**: Nhận diện real-time từ webcam
- **Tab Results**: Lịch sử kết quả và export CSV

Cấu hình:
- YOLO Model path
- LoRA Adapter path
- Confidence threshold
- GPU/CPU status indicator

### 2. CLI Demo (`scripts/demo_inference.py`)

Hỗ trợ:
- Single image inference
- Batch processing
- Output JSON results

### 3. Demo Cases (`docs/demo-cases.md`)

Bộ test cases phân loại:
- **Easy**: 5 cases đọc đúng hoàn toàn
- **Medium**: 4 cases có lỗi nhỏ
- **Hard**: 4 cases khó đọc
- **Edge**: 3 cases đặc biệt

### 4. Hướng dẫn sử dụng (`docs/demo-guide.md`)

---

## Files đã tạo/cập nhật

| File | Mô tả |
|------|--------|
| `src/app/demo.py` | Streamlit demo app |
| `scripts/demo_inference.py` | CLI demo script |
| `docs/demo-guide.md` | Hướng dẫn chạy demo |
| `docs/demo-cases.md` | Bộ test cases cho bảo vệ |

---

## Cách chạy Demo

### Streamlit Web App
```bash
cd d:/ComputerVisionNew
streamlit run src/app/demo.py
```

### CLI Demo (Single Image)
```bash
python scripts/demo_inference.py --input path/to/image.jpg
```

### CLI Demo (Batch)
```bash
python scripts/demo_inference.py --input-dir data/test_images/ --batch
```

---

## Performance

| Metric | Value |
|--------|-------|
| Detection mAP50 | 0.9471 |
| OCR Accuracy | 86.73% |
| Mean Latency | ~10s/image (GPU) |

---

## Checklist

- [x] Demo Streamlit với 4 tabs (Image/Video/Webcam/Results)
- [x] CLI demo script cho batch processing
- [x] Demo cases cho buổi bảo vệ
- [x] Hướng dẫn sử dụng chi tiết

---

## Bước tiếp theo (Buổi 7)

1. Viết báo cáo chính thức (≥15 trang)
2. Chuẩn bị slide trình bày (15-20 slides)
3. Diễn tập bảo vệ
4. Đóng gói mã nguồn
