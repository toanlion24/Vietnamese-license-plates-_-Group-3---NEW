# Demo Cases for Defense
## Bộ test case minh hoạ cho buổi bảo vệ

## Test Cases

### 1. Easy Cases (Đọc đúng hoàn toàn)

| ID | Plate GT | Đặc điểm | Kỳ vọng |
|----|---------|-----------|----------|
| easy_001 | 51G10096 | Ảnh rõ nét, biển sáng, không che khuất | ✅ 51G10096 |
| easy_002 | 51A65474 | Nền trắng, chữ đen, ánh sáng tốt | ✅ 51A65474 |
| easy_003 | 29A51796 | Biển nằm giữa ảnh | ✅ 29A51796 |
| easy_004 | 60A35981 | Ô tô đỗ, chụp thẳng | ✅ 60A35981 |
| easy_005 | 50LD04411 | Biển dài (LoGo xe) | ✅ 50LD04411 |

### 2. Medium Cases (Đọc gần đúng, có lỗi nhỏ)

| ID | Plate GT | Đặc điểm | Kỳ vọng | Lỗi |
|----|---------|-----------|----------|------|
| med_001 | 51A6486 | Ảnh hơi mờ | ⚠️ 51A64826 | Thừa số |
| med_002 | 5F22261 | Thiếu prefix 5 | ⚠️ 51F22261 | Thiếu số |
| med_003 | 51F22403 | Số 2 và 4 gần nhau | ⚠️ 51F24403 | 2↔4 |
| med_004 | 60A20880 | Biển có vết bẩn | ⚠️ 51A72880 | 6↔1 |

### 3. Hard Cases (Khó đọc)

| ID | Plate GT | Đặc điểm | Kỳ vọng | Notes |
|----|---------|-----------|----------|-------|
| hard_001 | 29A23950 | Biển nhỏ, xa | ❌ Variable | Rất khó |
| hard_002 | 56N7186 | Chữ N dễ nhầm | ❌ Variable | OCR khó |
| hard_003 | 68C137520 | Biển dài 9 số | ❌ Variable | Bike plate |
| hard_004 | F58250 | Thiếu số đầu | ❌ Variable | Crop có vấn đề |

### 4. Edge Cases

| ID | Plate GT | Đặc điểm | Kỳ vọng |
|----|---------|-----------|----------|
| edge_001 | - | Không có biển số | ❌ No detection |
| edge_002 | 51G10096 | Nhiều xe trong ảnh | ✅ Detected correctly |
| edge_003 | - | Biển bị che một phần | ⚠️ Partial detection |

---

## Metrics Summary

### Overall Performance (550 test images)

| Metric | Value |
|--------|-------|
| **Plate Accuracy** | 86.73% |
| **CER** | 0.033 |
| **Detection Rate** | ~99% |
| **Mean Latency** | 10.4s/image |

### Error Breakdown

| Error Type | Count | % | Fixable |
|------------|-------|---|---------|
| Correct | 477 | 86.7% | - |
| Substitution | 41 | 7.5% | Cần thêm data |
| Hallucination | 32 | 5.8% | Postprocessing |

---

## Demo Scenarios

### Scenario 1: Image Upload
```
Input: 1 ảnh xe ô tô rõ nét
Expected: Hiển thị plate number + bbox + confidence
Time: ~10-15s
```

### Scenario 2: Video Processing
```
Input: Video quay xe ra vào bãi đỗ
Expected: Hiển thị bbox + plate theo từng frame
Note: Xử lý 1/5 frames để demo nhanh
```

### Scenario 3: Multiple Plates
```
Input: Ảnh có nhiều xe
Expected: Detect tất cả biển số, hiển thị nhiều bbox
```

### Scenario 4: Hard Case
```
Input: Ảnh biển số mờ hoặc thiếu
Expected: Demo fallback/error handling
```
