# Cấu hình TrOCR cho Buổi 4

TrOCR được dùng trong cấu hình B: DeepSolo localize/crop vùng biển số, TrOCR nhận dạng text từ crop.

## 1) Model mặc định

Model khởi đầu:

```text
microsoft/trocr-base-printed
```

Lý do chọn:

- Dễ dùng qua Hugging Face `transformers`.
- Phù hợp thử nhanh với text in/biển số hơn model handwriting.
- Đã được repo hỗ trợ qua `src/ocr/trocr_adapter.py`.

## 2) Lệnh chạy thử adapter

Chạy nhanh bằng dummy detector để kiểm tra TrOCR tải model và pipeline không lỗi:

```bash
python scripts/run_infer.py --input-dir data/images/raw --output-json outputs/buoi4/trocr_smoke.json --detector-backend dummy --ocr-backend trocr --trocr-model microsoft/trocr-base-printed --device cpu --max-images 5
```

Nếu có GPU:

```bash
python scripts/run_infer.py --input-dir data/images/raw --output-json outputs/buoi4/trocr_smoke_cuda.json --detector-backend dummy --ocr-backend trocr --trocr-model microsoft/trocr-base-printed --device cuda --max-images 5
```

## 3) Cache model

TrOCR sẽ tải model từ Hugging Face ở lần chạy đầu tiên. Nếu máy không có mạng trong buổi demo, hãy tải trước và truyền cache:

```bash
python scripts/run_infer.py --input-dir data/images/raw --output-json outputs/buoi4/trocr_cached.json --detector-backend dummy --ocr-backend trocr --model-cache-dir weights/hf_cache --max-images 5
```

## 4) Input crop tốt cho TrOCR

Crop nên đạt các điều kiện:

- Chứa đủ toàn bộ ký tự, không cắt mép trái/phải.
- Ít nền thừa hơn ảnh gốc.
- Độ cao ký tự đủ lớn sau resize.
- Không bị đảo kênh màu.
- Với biển nghiêng, nên rectify perspective trước khi OCR.

## 5) Output cần lưu để debug

Khi chạy cấu hình B, nên lưu:

- ảnh crop trước preprocess,
- ảnh crop sau preprocess,
- text raw của TrOCR,
- text sau normalize/postprocess,
- score hoặc confidence nếu có,
- latency từng stage nếu đo được.

## 6) Prediction CSV cho đánh giá

Export kết quả cuối cùng theo schema:

```csv
image_id,gt,pred,score,latency_ms,bbox_xyxy,error_type
IMG_001,51H12345,51H12345,0.88,142.4,"120,80,310,140",ok
```

Sau đó chạy:

```bash
python scripts/run_buoi4_experiments.py --config-a-csv outputs/buoi4/deepsolo_e2e_predictions.csv --config-b-csv outputs/buoi4/deepsolo_trocr_predictions.csv
```

## 8) Fine-tune TrOCR và learning rate

File mặc định: [`finetune_defaults.json`](finetune_defaults.json). `learning_rate` đang đặt **1.5e-4**, cao hơn mức thường dùng làm điểm xuất phát (**5e-5**) cho fine-tune TrOCR. Có thể **tăng thêm** (vd. `2e-4`…`3e-4`) bằng `--learning-rate` nếu loss giảm ổn và không phát tán; nếu loss nhảy hoặc CER val xấu đi thì **giảm** về khoảng `5e-5`–`1e-4`.

Chuẩn bị CSV train (cột ảnh crop + text nhãn), ví dụ `image_path,gt` hoặc `path,text`:

```bash
pip install -r requirements.txt
python scripts/train_trocr.py --train-csv data/trocr_train.csv --output-dir experiments/trocr_ft_run1
```

Ghi đè learning rate:

```bash
python scripts/train_trocr.py --train-csv data/trocr_train.csv --learning-rate 2.5e-4
```

Inference sau khi train: truyền thư mục output vào `--trocr-model` (local path) trong `run_infer.py`.

## 7) Khi nào cần fine-tune?

Chỉ fine-tune TrOCR nếu baseline inference trên crop thật còn kém và nhóm có đủ dữ liệu crop + text. Nếu chưa đủ dữ liệu, ưu tiên:

- cải thiện crop,
- resize/contrast ổn định hơn,
- hậu xử lý theo format biển số VN,
- phân tích lỗi để biết sai do ảnh hay do OCR.
