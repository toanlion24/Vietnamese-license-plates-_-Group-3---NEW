# Báo cáo A/B Buổi 4 - DeepSolo end-to-end vs DeepSolo + TrOCR

Tự động cập nhật từ `scripts/run_buoi4_experiments.py`.

Ghi chú: Dữ liệu: ảnh synthetic trong data/synthetic_plates/. Detector dummy. A=EasyOCR thật, B=TrOCR thật (CPU).


## 1) Thiết lập

- Thời điểm tạo báo cáo: `2026-05-10T13:43:43.559195+00:00`
- Cấu hình A: DeepSolo end-to-end
- File A: `outputs\buoi4\deepsolo_e2e_predictions.csv`
- Cấu hình B: DeepSolo + TrOCR
- File B: `outputs\buoi4\deepsolo_trocr_predictions.csv`

## 2) Kiểm tra công bằng

- Không có cảnh báo.

## 3) Kết quả định lượng

Cấu hình A - DeepSolo end-to-end:

- Số mẫu: 15
- CER: 0.4000
- WER: 0.9333
- Plate accuracy: 0.0667
- Mean latency ms: 590.6205

Cấu hình B - DeepSolo + TrOCR:

- Số mẫu: 15
- CER: 0.0000
- WER: 0.0000
- Plate accuracy: 1.0000
- Mean latency ms: 3450.1064

## 4) Phân bố lỗi

Cấu hình A:

- ocr_error: 14
- ok: 1

Cấu hình B:

- ok: 15

## 5) Quyết định tạm thời

Tạm chọn cấu hình B (DeepSolo + TrOCR) vì plate accuracy cao hơn.

## 6) Việc cần làm tiếp

- Mở các case sai và gán lại loại lỗi: `detect_miss`, `bad_crop`, `ocr_error`, `postprocess_helped`, `ambiguous_gt`.
- Chọn 10-20 hard cases để đưa vào báo cáo cuối.
- Nếu chọn cấu hình B, ưu tiên cải thiện crop/rectify trước khi fine-tune TrOCR.
- Nếu chọn cấu hình A, kiểm tra riêng lỗi spotting sai vùng và lỗi nhận dạng sai text.
