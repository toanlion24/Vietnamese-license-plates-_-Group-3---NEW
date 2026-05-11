# Manifest cho thí nghiệm

## 1) `data/test_manifest.csv` — đánh giá OCR / Buổi 4 (crop hoặc ảnh đã cắt)

Schema tối thiểu: `image_id`, `image_path`, `gt` (có thể thêm `ambiguous_gt`).

- Dùng với `scripts/run_buoi4_manifest_inference.py` để xuất prediction A/B và metric.
- Ảnh có thể là **ảnh xe** (pipeline sẽ detect) hoặc ảnh đã crop nếu detector/luồng được cấu hình phù hợp.

## 2) Manifest có **bbox hoặc polygon** — chuẩn bị DeepSolo / spotting

Script `scripts/prepare_buoi4_deepsolo_data.py` cần **mỗi dòng** có `image_path` + nhãn ký tự + **`bbox_xyxy`** hoặc **`polygon`**.

- Ví dụ cột: `image_id,image_path,bbox_xyxy,gt,split`
- File mẫu: `buoi4_deepsolo.example.csv`

## 3) Ghép manifest từ thư mục ảnh + file nhãn — `scripts/build_test_manifest_from_folder.py`

Ví dụ **một CSV nhãn** (cột file + cột biển), xem [`example_labels.csv`](example_labels.csv).

```bash
python scripts/build_test_manifest_from_folder.py ^
  --images-dir data/raw/my_plates ^
  --labels-csv data/raw/labels.csv ^
  --output data/test_manifest.csv
```

**File `.txt` cạnh từng ảnh** (cùng thư mục, tên `ảnh_stem.txt`, một dòng là biển):

```bash
python scripts/build_test_manifest_from_folder.py --images-dir data/raw/my_plates --sidecar-next-to-image
```

**Thư mục chỉ chứa `{stem}.txt`:**

```bash
python scripts/build_test_manifest_from_folder.py --images-dir data/raw/my_plates --sidecar-dir data/raw/gt_txt
```

Chi tiết tham số: `python scripts/build_test_manifest_from_folder.py --help`.

`gt` / `text_gt` / `label` / `plate` trong CSV được tự nhận; có thể buộc bằng `--gt-column` và `--key-column`.
