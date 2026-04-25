# Báo cáo A/B Buổi 4 - DeepSolo end-to-end vs DeepSolo + TrOCR

Tài liệu này là template báo cáo kết quả Buổi 4. Sau khi có prediction CSV, chạy `scripts/run_buoi4_experiments.py` để tự động cập nhật phần số liệu chính.

## 1) Mục tiêu thí nghiệm

So sánh hai cấu hình trên cùng tập test:

- Cấu hình A: DeepSolo end-to-end, một mô hình trả về vùng biển số và text.
- Cấu hình B: DeepSolo dùng để localize/crop, TrOCR dùng để nhận dạng text.

Mục tiêu là chọn pipeline ổn định nhất cho Buổi 5, không chỉ chọn mô hình có metric cao nhất.

## 2) Thiết lập dữ liệu

- Test split: `data/splits/test.txt`
- Manifest: `data/manifests/buoi4_test.csv`
- Số mẫu test: cập nhật sau khi chạy script.
- Quy tắc chuẩn hóa: dùng `normalize_plate_text()` trong `src/postprocess/plate_rules.py`.

Ghi chú kiểm soát công bằng:

- Không dùng ảnh test để train/fine-tune.
- Hai cấu hình dùng cùng ground truth.
- Hai cấu hình xuất prediction theo cùng schema CSV.

## 3) Kết quả định lượng

Phần này sẽ được script cập nhật tự động.

```text
Chưa có kết quả. Hãy chạy:
python scripts/run_buoi4_experiments.py --config-a-csv outputs/buoi4/deepsolo_e2e_predictions.csv --config-b-csv outputs/buoi4/deepsolo_trocr_predictions.csv
```

## 4) Nhận xét ban đầu

Cần điền sau khi có số liệu:

- Cấu hình nào có `plate_accuracy` tốt hơn?
- Lỗi chính nằm ở localization, OCR, hay hậu xử lý?
- Latency của cấu hình B có chấp nhận được cho demo không?

## 5) Hard cases cần đưa vào báo cáo

Điền 10-20 case tiêu biểu:

- Ảnh mờ hoặc thiếu sáng.
- Biển số bị nghiêng mạnh.
- Biển hai dòng.
- Ký tự dễ nhầm: `O/0`, `I/1`, `S/5`, `B/8`.
- Crop bị mất mép ký tự.

## 6) Quyết định cho Buổi 5

Quyết định tạm thời:

- Pipeline được chọn: chưa chốt.
- Lý do: chờ kết quả A/B.
- Việc cần cải thiện tiếp: crop/rectify, hậu xử lý regex, hoặc fine-tune OCR.
