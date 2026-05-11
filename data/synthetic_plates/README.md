# Ảnh biển số tổng hợp (synthetic)

Ảnh được render bằng PIL (chữ in, nền sáng, viền xanh). Ground truth nằm trong [`../test_manifest.csv`](../test_manifest.csv).

Dùng để chạy inference và metric trong repo khi không commit ảnh thật. Trong báo cáo đề tài, nên bổ sung thêm thí nghiệm trên ảnh chụp thật ngoài môi trường.

## Tạo lại hoặc chỉnh số mẫu

```bash
python scripts/generate_synthetic_plate_dataset.py
python scripts/generate_synthetic_plate_dataset.py --num-samples 8
```

## Kết quả đánh giá đã chạy trong repo (tham chiếu)

Sau khi chạy `scripts/run_buoi4_manifest_inference.py` (EasyOCR vs TrOCR thật, detector `dummy`), artifact nằm tại:

- `reports/buoi4_ab_metrics.json` — có thể commit; tóm tắt CER/WER/accuracy.
- `reports/buoi4_ab_run_synthetic.md` — báo cáo Markdown tương ứng.

Bản sao CSV prediction (để tiện lưu trong Git khi `outputs/` bị ignore):

- `reports/buoi4_synthetic_pred_easyocr.csv`
- `reports/buoi4_synthetic_pred_trocr.csv`

Số liệu phụ thuộc máy (CPU/GPU); trên bộ synthetic in sạch, TrOCR printed thường khớp GT tốt hơn EasyOCR baseline.
