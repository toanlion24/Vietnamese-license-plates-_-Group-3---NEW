# Báo cáo A/B Buổi 4 - DeepSolo end-to-end vs DeepSolo + TrOCR

Tự động cập nhật từ `scripts/run_buoi4_experiments.py`.

Ghi chú: Đây là dữ liệu demo/smoke-test để kiểm tra code metric, chưa phải kết quả mô hình thật.


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
