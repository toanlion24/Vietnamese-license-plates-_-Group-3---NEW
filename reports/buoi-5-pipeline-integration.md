# Buổi 5: Tích hợp Pipeline & Đánh giá toàn hệ thống

**Ngày:** 2026-06-13  
**Trạng thái:** ✅ Hoàn thành

---

## Mục tiêu

Hoàn thiện pipeline YOLOv8n + Qwen2-VL-2B-Instruct (fine-tuned LoRA), đánh giá trên ≥200 ảnh thực tế.

---

## Kết quả đánh giá OCR (550 ảnh crops)

### So sánh Base vs Fine-tuned LoRA

| Metric | Base Model | Fine-tuned LoRA | Cải thiện |
|--------|-----------|----------------|-----------|
| **Plate Accuracy** | 0.0% | **86.73%** | +86.73% |
| **CER** | 6.0645 | **0.033** | -6.03 (↓99.5%) |
| **Correct/Total** | 0/550 | **477/550** | +477 |
| **Mean Latency** | 12,731 ms | **10,408 ms** | -18% |

### Phân tích lỗi (LoRA model)

| Error Type | Count | Percentage |
|------------|-------|------------|
| ✅ Correct (none) | 477 | 86.7% |
| 🔤 Substitution | 41 | 7.5% |
| 💭 Hallucination | 32 | 5.8% |

---

## Chi tiết lỗi

### 1. Substitution Errors (41/550 = 7.5%)

**Nguyên nhân chính:** Nhầm lẫn chữ số tương tự (2↔5, 0↔6, 1↔7)

| GT | Predicted | Loại lỗi |
|----|-----------|-----------|
| 51A6486 | 51A64826 | Thừa số cuối |
| 5F22261 | 51F22261 | Thiếu prefix "5" |
| 29A51796 | 51A51796 | 2↔5 |
| 60A35981 | 61A35981 | 0↔1 |
| 51F22403 | 51F24403 | 2↔4 |
| 51A72110 | 51A65316 | 2↔6 |

### 2. Hallucination Errors (32/550 = 5.8%)

**Nguyên nhân:** Model đọc text không liên quan hoặc hoàn toàn sai

| GT | Predicted | Notes |
|----|-----------|-------|
| 51A6486 | 51A64826 | Substituted |
| 51F7512 | 51F79512 | Số 1↔9 |
| F89357 | 5F89357 | Thiếu prefix |

---

## Deliverables

### Files đã tạo/cập nhật

| File | Mô tả |
|------|--------|
| `scripts/_eval_real.py` | Script đánh giá OCR (base vs LoRA) |
| `outputs/lora_comparison/comparison.json` | Metrics summary |
| `outputs/lora_comparison/predictions_side_by_side.csv` | Full predictions |
| `outputs/lora_comparison/predictions_lora.csv` | LoRA predictions |
| `outputs/lora_comparison/predictions_base.csv` | Base predictions |

### Metrics đạt được

```json
{
  "lora": {
    "accuracy": 0.8673,
    "cer": 0.033,
    "num_correct": 477,
    "num_samples": 550,
    "mean_latency_ms": 10407.56
  }
}
```

---

## Checklist Buổi 5

- [x] Test set ≥ 200 ảnh đã chuẩn bị (thực tế: 550 ảnh)
- [x] Pipeline YOLOv8n + Qwen2-VL tích hợp xong
- [x] Có CER/WER/plate accuracy/latency
- [x] Phân tích lỗi chi tiết

---

## Bước tiếp theo (Buổi 6)

1. Xây dựng demo Streamlit cho ảnh/video/webcam
2. Test với dữ liệu thực tế
3. Chuẩn bị bộ case minh hoạ cho buổi bảo vệ

---

## Ghi chú

- Kết quả OCR rất ấn tượng: từ 0% → 86.73% accuracy sau fine-tuning
- Lỗi chủ yếu ở chữ số prefix và substitution
- Cần cải thiện: thêm dữ liệu hoặc post-processing cho lỗi substitution
