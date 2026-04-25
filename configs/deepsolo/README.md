# Cấu hình DeepSolo cho Buổi 4

DeepSolo là nhánh thực nghiệm text spotting. Repo chính này không vendor toàn bộ DeepSolo để tránh làm nặng project; hãy clone DeepSolo riêng hoặc đặt như dependency ngoài, sau đó export prediction về schema chung của repo.

## 1) Vai trò trong Buổi 4

DeepSolo được dùng cho 2 cấu hình:

- Cấu hình A: DeepSolo end-to-end trả về vùng biển số + text.
- Cấu hình B: DeepSolo localize vùng biển số, sau đó crop cho TrOCR.

## 2) Dữ liệu đầu vào khuyến nghị

Từ manifest Buổi 4:

```csv
image_id,image_path,bbox_xyxy,text_gt,split
IMG_001,data/images/raw/IMG_001.jpg,"120,80,310,140",51H12345,test
```

Nếu DeepSolo yêu cầu polygon, convert bbox thành 4 điểm:

```text
x1,y1,x2,y1,x2,y2,x1,y2,text_gt
```

Với ảnh nghiêng mạnh, nên gán polygon thật để crop/spotting công bằng hơn.

## 3) Thư mục thí nghiệm gợi ý

```text
experiments/
  buoi4_deepsolo_e2e/
    config.yaml
    train.log
    best_checkpoint.pth
    predictions.csv
  buoi4_deepsolo_localizer/
    config.yaml
    train.log
    best_checkpoint.pth
    detections.csv
```

## 4) Prediction CSV bắt buộc

Để dùng script đánh giá trong repo, export kết quả DeepSolo end-to-end thành:

```csv
image_id,gt,pred,score,latency_ms,bbox_xyxy,error_type
IMG_001,51H12345,51H12345,0.91,84.2,"120,80,310,140",ok
```

Cột bắt buộc:

- `image_id`
- `gt`
- `pred`

Cột nên có:

- `score`
- `latency_ms`
- `bbox_xyxy`
- `error_type`

## 5) Checklist train/infer

- [ ] Cố định test split trước khi train.
- [ ] Lưu config train cùng checkpoint.
- [ ] Ghi seed và thông tin GPU/CPU.
- [ ] Không dùng ảnh test để fine-tune.
- [ ] Export prediction CSV theo schema chung.
- [ ] Chạy `scripts/run_buoi4_experiments.py` để so sánh với cấu hình B.

## 6) Lưu ý khi báo cáo

Khi bảo vệ, cần nói rõ DeepSolo đang được dùng theo vai trò nào:

- Nếu dùng end-to-end: đây là text spotting đầy đủ.
- Nếu dùng localizer: đây là detector/cropper cho OCR downstream.

Không trộn hai vai trò khi so sánh metric.
