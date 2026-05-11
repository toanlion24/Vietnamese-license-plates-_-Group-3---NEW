# Báo cáo A/B Buổi 4 - DeepSolo end-to-end vs DeepSolo + TrOCR

Tự động cập nhật từ `scripts/run_buoi4_experiments.py`.

Ghi chú: Cấu hình A: YOLOv8 + EasyOCR. Cấu hình B: YOLOv8 + TrOCR (hoặc ensemble nếu --ensemble-b). Có thể thay bằng DeepSolo qua flags --config-*-from-csv. GT lấy từ manifest; pred từ inference hoặc CSV import. Crop margin=0.08, CLAHE=False, aggressive_post=False.


## 1) Thiết lập

- Thời điểm tạo báo cáo: `2026-05-11T04:27:16.236086+00:00`
- Cấu hình A: DeepSolo end-to-end
- File A: `outputs\buoi4\deepsolo_e2e_predictions.csv`
- Cấu hình B: DeepSolo + TrOCR
- File B: `outputs\buoi4\deepsolo_trocr_predictions.csv`

## 2) Kiểm tra công bằng

- Không có cảnh báo.

## 3) Kết quả định lượng

Cấu hình A - DeepSolo end-to-end:

- Số mẫu: 10
- CER: 0.6977
- WER: 1.0000
- Plate accuracy: 0.0000
- Mean latency ms: 608.6203

Cấu hình B - DeepSolo + TrOCR:

- Số mẫu: 10
- CER: 0.8721
- WER: 1.0000
- Plate accuracy: 0.0000
- Mean latency ms: 2605.2676

## 4) Phân bố lỗi

Cấu hình A:

- ocr_error: 10

Cấu hình B:

- ocr_error: 10

## 5) Quyết định tạm thời

Hai cấu hình có plate accuracy bằng nhau; tạm chọn A vì CER thấp hơn.

## 6) Việc cần làm tiếp

- Mở các case sai và gán lại loại lỗi: `detect_miss`, `bad_crop`, `ocr_error`, `postprocess_helped`, `ambiguous_gt`.
- Chọn 10-20 hard cases để đưa vào báo cáo cuối.
- Nếu chọn cấu hình B, ưu tiên cải thiện crop/rectify trước khi fine-tune TrOCR.
- Nếu chọn cấu hình A, kiểm tra riêng lỗi spotting sai vùng và lỗi nhận dạng sai text.
