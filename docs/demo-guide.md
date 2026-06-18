# Hướng dẫn chạy Demo

## Cài đặt

### 1. Cài đặt dependencies

```bash
pip install streamlit opencv-python-headless pillow numpy torch
pip install ultralytics transformers peft qwen-vl-utils
```

### 2. Download Models

- **YOLOv8 Detector**: Đã có tại `runs/detect/experiments/detector/yolov8n_augmented/weights/best.pt`
- **Qwen2-VL LoRA Adapter**: Đã có tại `experiments/qwen2vl_crops_lora`

## Chạy Demo

### Option 1: Streamlit Web App

```bash
cd d:/ComputerVisionNew
streamlit run src/app/demo.py
```

Sau đó mở browser tại `http://localhost:8501`

### Option 2: CLI Demo (Single Image)

```bash
cd d:/ComputerVisionNew
python scripts/demo_inference.py --input path/to/image.jpg
```

### Option 3: CLI Demo (Batch)

```bash
cd d:/ComputerVisionNew
python scripts/demo_inference.py --input-dir data/test_images/ --batch
```

## Tính năng

### Streamlit Demo
- 📷 **Upload Ảnh**: Nhận diện biển số từ ảnh tải lên
- 🎬 **Video**: Xử lý video và hiển thị kết quả theo frame
- 📹 **Webcam**: Nhận diện real-time từ webcam
- 📊 **Lịch sử**: Xem lại các kết quả đã xử lý

### CLI Demo
- Xử lý ảnh đơn lẻ
- Xử lý batch nhiều ảnh
- Xuất kết quả ra JSON

## Performance

| Metric | Value |
|--------|-------|
| Detection mAP50 | 0.9471 |
| OCR Accuracy | 86.73% |
| OCR CER | 0.033 |
| Mean Latency | ~10s/img (GPU) |

## Troubleshooting

### Lỗi CUDA out of memory
- Giảm batch size
- Sử dụng CPU thay vì GPU

### Lỗi Cannot find module
- Đảm bảo đang ở thư mục project
- Thử: `cd d:/ComputerVisionNew` trước

### Lỗi Webcam không hoạt động
- Kiểm tra webcam được kết nối
- Thử dùng 0, 1, 2 thay vì default
