# Buổi 4 - Cải tiến mô hình và thực nghiệm DeepSolo + TrOCR

Tài liệu này triển khai chi tiết Buổi 4 theo mục tiêu: thử đúng 2 cấu hình, đánh giá công bằng trên cùng tập test, rồi chọn pipeline để tiếp tục Buổi 5-6.

---

## 1) Mục tiêu cuối buổi

Sau Buổi 4, nhóm cần có:

- Một bộ test cố định có ground truth text biển số.
- Kết quả thực nghiệm cho 2 cấu hình:
  - Cấu hình A: DeepSolo end-to-end.
  - Cấu hình B: DeepSolo localize + TrOCR OCR.
- File metric cho từng cấu hình gồm `cer`, `wer`, `plate_accuracy`, `mean_latency_ms`, `num_samples`.
- Báo cáo so sánh A/B tại `reports/ablation_deepsolo_trocr.md`.
- Quyết định rõ: dùng pipeline nào làm nền cho Buổi 5.

Nguyên tắc quan trọng: cùng một ảnh test, cùng một ground truth, cùng một hàm normalize text. Không đổi test set giữa hai cấu hình.

---

## 2) Bản chất thí nghiệm

### 2.1 DeepSolo end-to-end là gì?

DeepSolo là hướng text spotting: mô hình vừa tìm vùng chữ vừa dự đoán chuỗi chữ. Với biển số xe, đầu vào là ảnh xe hoặc frame video, đầu ra mong muốn là vùng biển số và text biển số.

Ưu điểm:

- Pipeline gọn: một mô hình làm cả localization và recognition.
- Có thể xử lý text nghiêng/cong tốt hơn detector bbox đơn giản nếu dữ liệu phù hợp.

Rủi ro:

- Cần convert nhãn sang format text spotting.
- Khó fine-tune nhanh nếu môi trường hoặc format dữ liệu chưa ổn.
- Với biển số VN ít dữ liệu, OCR end-to-end có thể chưa tốt bằng OCR chuyên dụng trên crop.

### 2.2 DeepSolo + TrOCR là gì?

Cấu hình B tách bài toán thành 2 stage:

1. DeepSolo chỉ dùng để tìm vùng biển số hoặc polygon text.
2. Crop vùng biển số, tiền xử lý, rồi dùng TrOCR để đọc ký tự.

Ưu điểm:

- Dễ debug: sai do localization hay do OCR.
- Có thể cải thiện OCR riêng bằng crop tốt hơn, normalize tốt hơn, hoặc fine-tune TrOCR sau.
- Phù hợp kiến trúc module của repo: `detector -> preprocess -> ocr -> postprocess -> eval`.

Rủi ro:

- Latency cao hơn do chạy 2 mô hình.
- Nếu crop sai hoặc quá sát ký tự, TrOCR đọc kém.

---

## 3) Input bắt buộc trước khi chạy

### 3.1 Dữ liệu

Cần chuẩn bị các file tối thiểu:

- `data/splits/test.txt`: danh sách ảnh test cố định.
- `data/manifests/buoi4_test.csv`: manifest test cho OCR/eval.
- Ảnh gốc nằm trong thư mục đã thống nhất, ví dụ `data/images/raw/`.

Manifest nên có các cột:

- `image_id`: tên ảnh hoặc ID duy nhất.
- `image_path`: đường dẫn ảnh.
- `bbox_xyxy` hoặc `polygon`: vùng biển số ground truth nếu có.
- `text_gt`: text biển số đúng.
- `split`: giá trị `test`.

Ví dụ CSV:

```csv
image_id,image_path,bbox_xyxy,text_gt,split
IMG_001,data/images/raw/IMG_001.jpg,"120,80,310,140",51H12345,test
IMG_002,data/images/raw/IMG_002.jpg,"90,60,260,118",30A56789,test
```

### 3.2 Quy tắc chuẩn hóa text

Khi đánh giá, mọi text cần đi qua cùng một chuẩn:

1. Chuyển uppercase.
2. Bỏ ký tự không phải chữ/số.
3. Bỏ khoảng trắng, dấu gạch ngang, dấu chấm nếu mục tiêu là so khớp biển số cuối cùng.

Ví dụ:

- `51H-123.45` -> `51H12345`
- `51 h 12345` -> `51H12345`

Repo đã có hàm `normalize_plate_text()` trong `src/postprocess/plate_rules.py`.

---

## 4) Cấu hình A - DeepSolo end-to-end

### 4.1 Chuẩn bị nhãn cho DeepSolo

DeepSolo thường cần dữ liệu kiểu text spotting: ảnh + polygon vùng chữ + transcript. Nếu nhãn hiện tại chỉ là YOLO bbox, có thể chuyển bbox thành polygon 4 điểm:

```text
x1,y1,x2,y1,x2,y2,x1,y2,text
```

Nếu biển số bị nghiêng nhiều, nên gán polygon thật thay vì lấy bbox chữ nhật.

### 4.2 Train hoặc fine-tune

Checklist train:

1. Clone hoặc đặt DeepSolo ngoài repo chính nếu repo quá nặng.
2. Tạo dataset theo format DeepSolo.
3. Sửa config dataset, số class/text vocabulary nếu cần.
4. Chạy train với seed cố định.
5. Lưu checkpoint vào `experiments/buoi4_deepsolo_e2e/`.

Thông tin cần ghi lại:

- Commit/version của DeepSolo.
- Config train.
- Seed.
- Số epoch/iteration.
- GPU/CPU dùng train.
- Checkpoint tốt nhất.

### 4.3 Inference và export prediction

Sau inference, xuất CSV theo format thống nhất:

```csv
image_id,gt,pred,score,latency_ms,bbox_xyxy,error_type
IMG_001,51H12345,51H12345,0.91,84.2,"120,80,310,140",ok
IMG_002,30A56789,30A5678,0.77,88.5,"90,60,260,118",ocr_or_spotting
```

File gợi ý:

- `outputs/buoi4/deepsolo_e2e_predictions.csv`

---

## 5) Cấu hình B - DeepSolo + TrOCR

### 5.1 Localization bằng DeepSolo

Vẫn chạy DeepSolo trên ảnh gốc, nhưng mục tiêu chính là lấy vùng biển số/polygon tốt nhất. Nếu DeepSolo trả nhiều vùng text, chọn vùng phù hợp nhất bằng:

- score cao nhất,
- kích thước vùng hợp lý,
- vị trí giống biển số,
- text raw gần format biển số VN nếu có.

### 5.2 Crop và tiền xử lý

Quy trình crop:

1. Nếu có polygon: rectify perspective để đưa biển số về hình chữ nhật.
2. Nếu chỉ có bbox: crop theo `x1, y1, x2, y2`, thêm padding nhỏ 2-5%.
3. Resize crop về kích thước ổn định.
4. Chuyển RGB/BGR đúng định dạng trước khi đưa vào TrOCR.

Repo hiện có:

- `src/preprocess/ops.py`: `crop_plate()`, `preprocess_plate()`.
- `src/ocr/trocr_adapter.py`: adapter chạy TrOCR từ Hugging Face.
- `src/pipeline/infer_plate_pipeline.py`: pipeline detector + OCR.

### 5.3 Inference TrOCR

Nếu chỉ chạy thử nhanh:

```bash
python scripts/run_infer.py --input-dir data/images/raw --output-json outputs/buoi4/trocr_predictions.json --detector-backend dummy --ocr-backend trocr --trocr-model microsoft/trocr-base-printed --device cpu
```

Khi đã có detector/localizer thật, thay `--detector-backend dummy` bằng detector tương ứng. Nếu dùng DeepSolo ngoài repo, cần export crop/prediction về CSV thống nhất để đánh giá.

Prediction CSV cho cấu hình B:

```csv
image_id,gt,pred,score,latency_ms,bbox_xyxy,error_type
IMG_001,51H12345,51H12345,0.88,142.4,"120,80,310,140",ok
IMG_002,30A56789,30A56789,0.83,151.0,"90,60,260,118",ok
```

File gợi ý:

- `outputs/buoi4/deepsolo_trocr_predictions.csv`

---

## 6) Đánh giá công bằng

### 6.1 Metric cần báo cáo

- `CER`: khoảng cách chỉnh sửa ký tự chia cho tổng số ký tự ground truth.
- `WER`: khoảng cách chỉnh sửa theo token chia cho tổng số token ground truth.
- `plate_accuracy`: tỷ lệ biển số khớp hoàn toàn sau normalize.
- `mean_latency_ms`: latency trung bình nếu CSV có cột `latency_ms`.

Với biển số xe, `plate_accuracy` là chỉ số dễ giải thích nhất khi bảo vệ: đọc đúng hoàn toàn hay không.

### 6.2 Chuẩn bị annotation DeepSolo từ manifest

Nếu đã có `data/manifests/buoi4_test.csv`, convert bbox/polygon + text sang annotation text spotting đơn giản:

```bash
python scripts/prepare_buoi4_deepsolo_data.py --manifest-csv data/manifests/buoi4_test.csv --output-dir data/deepsolo/buoi4 --split test
```

Output:

- `data/deepsolo/buoi4/test_annotations.txt`
- `data/deepsolo/buoi4/test_annotations.jsonl`
- `data/deepsolo/buoi4/test_summary.json`

File `.txt` có dạng:

```text
image_path	x1,y1,x2,y1,x2,y2,x1,y2,text_gt
```

### 6.3 Chạy đánh giá A/B

Sau khi có hai CSV prediction:

```bash
python scripts/run_buoi4_experiments.py ^
  --config-a-csv outputs/buoi4/deepsolo_e2e_predictions.csv ^
  --config-b-csv outputs/buoi4/deepsolo_trocr_predictions.csv ^
  --metrics-json reports/buoi4_ab_metrics.json ^
  --report-md reports/ablation_deepsolo_trocr.md
```

Kết quả:

- `reports/buoi4_ab_metrics.json`: số liệu máy đọc được.
- `reports/ablation_deepsolo_trocr.md`: báo cáo tóm tắt để đưa vào đồ án.

### 6.4 Xem kết quả demo trong `docs`

Khi chưa có checkpoint DeepSolo/TrOCR thật, có thể chạy demo nhỏ để kiểm tra code metric và format báo cáo:

```bash
python scripts/create_buoi4_demo_predictions.py --output-dir outputs/buoi4/demo
python scripts/run_buoi4_experiments.py ^
  --config-a-csv outputs/buoi4/demo/deepsolo_e2e_predictions.csv ^
  --config-b-csv outputs/buoi4/demo/deepsolo_trocr_predictions.csv ^
  --metrics-json reports/buoi4_demo_ab_metrics.json ^
  --report-md docs/buoi-4-ket-qua-thuc-nghiem-deepsolo-trocr.md ^
  --experiment-note "Đây là dữ liệu demo/smoke-test để kiểm tra code metric, chưa phải kết quả mô hình thật."
```

Mở `docs/buoi-4-ket-qua-thuc-nghiem-deepsolo-trocr.md` để xem bảng kết quả demo.

---

## 7) Phân tích lỗi

Sau khi có prediction, chọn 10-20 ảnh lỗi tiêu biểu và phân nhóm:

- `detect_miss`: không tìm thấy biển số.
- `bad_crop`: tìm đúng vùng tổng quát nhưng crop mất ký tự.
- `ocr_error`: crop đúng nhưng đọc sai ký tự.
- `postprocess_helped`: raw OCR sai nhẹ nhưng hậu xử lý sửa đúng.
- `ambiguous_gt`: ground truth hoặc ảnh quá mờ, khó kết luận.

Với mỗi lỗi, ghi:

- `image_id`
- `gt`
- `pred`
- loại lỗi
- ghi chú ngắn: tối, mờ, nghiêng, biển 2 dòng, bị che, quá nhỏ.

Mục tiêu không chỉ là kể sai ở đâu, mà là chốt việc cần làm ở Buổi 5.

---

## 8) Tiêu chí chọn pipeline cho Buổi 5

Ưu tiên theo thứ tự:

1. `plate_accuracy` cao hơn trên cùng test set.
2. `CER` thấp hơn nếu plate accuracy gần nhau.
3. Dễ debug và cải thiện trong thời gian còn lại.
4. Latency đủ dùng cho demo Buổi 6.

Khuyến nghị thực dụng:

- Nếu DeepSolo end-to-end kém OCR nhưng localization ổn: dùng cấu hình B.
- Nếu TrOCR chậm nhưng chính xác hơn rõ rệt: dùng B cho báo cáo, tối ưu demo sau.
- Nếu DeepSolo setup quá rủi ro: giữ YOLO + EasyOCR làm baseline dự phòng, nhưng Buổi 4 vẫn cần báo cáo rõ kết quả thử DeepSolo/TrOCR.

---

## 9) Checklist hoàn thành Buổi 4

- [ ] Có manifest test chứa `image_id`, `image_path`, `text_gt`.
- [ ] Có prediction CSV cho cấu hình A.
- [ ] Có prediction CSV cho cấu hình B.
- [ ] Chạy `scripts/run_buoi4_experiments.py` thành công.
- [ ] Có `reports/buoi4_ab_metrics.json`.
- [ ] Có `reports/ablation_deepsolo_trocr.md`.
- [ ] Có ít nhất 10 ảnh lỗi tiêu biểu hoặc danh sách lỗi để phân tích.
- [ ] Chốt pipeline dùng cho Buổi 5.

---

## 10) Deliverables nộp cuối buổi

- `reports/buoi-4-cai-tien-mo-hinh-va-thuc-nghiem-chi-tiet.md`
- `reports/ablation_deepsolo_trocr.md`
- `docs/buoi-4-ket-qua-thuc-nghiem-deepsolo-trocr.md` nếu cần xem kết quả demo/thực nghiệm trong thư mục `docs`
- `reports/buoi4_ab_metrics.json`
- `outputs/buoi4/deepsolo_e2e_predictions.csv`
- `outputs/buoi4/deepsolo_trocr_predictions.csv`
- `configs/deepsolo/README.md`
- `configs/trocr/README.md`
- checkpoint/log thí nghiệm trong `experiments/`
