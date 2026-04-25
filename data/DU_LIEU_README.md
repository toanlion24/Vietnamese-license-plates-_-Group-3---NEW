# Tổ chức thư mục dữ liệu (buổi 2) — sau migrate từ `datakangle`

## Cấu trúc thư mục

| Thư mục / file | Mục đích |
|----------------|----------|
| `data/images/raw/` | Ảnh gốc (tương đương “raw” trong plan: ảnh chưa đổi tên nguồn đã gom một chỗ). |
| `data/labels/raw/` | File nhãn YOLO `.txt` (bbox `class xc yc w h`), **cùng stem** với ảnh. |
| `data/splits/` | `train.txt`, `val.txt`, `test.txt` — mỗi dòng một đường dẫn ảnh tương đối từ **gốc repo**. |
| `data/data.yaml` | Cấu hình YOLO (Ultralytics), trỏ tới các file split. |
| `data/migration_from_datakangle.json` | Bản đồ tên cũ → `plate_XXXX` (tạo khi chạy script migrate). |
| `data/video_sources.json` | Đăng ký các **bộ video** (ID → thư mục chứa MP4), ví dụ `giao_thong_vn` → `vid/ Giao thông Việt Nam` (chú ý khoảng trắng sau `vid/` nếu đặt tên thư mục như vậy). |
| `vid/…` | **Kho video thực địa** (không commit lên git nếu file quá lớn): clip quay ngoài đường → dùng để **trích frame**, sau đó **gán nhãn** và đưa vào pipeline huấn luyện như các ảnh trong `data/images/raw`. |

**Vì sao dùng `images/raw` thay vì chỉ `raw`?**  
Ultralytics YOLO tự tìm nhãn bằng cách thay đoạn `images` thành `labels` trong đường dẫn ảnh. Nếu đặt ảnh ở `data/raw/...` (không có chữ `images`), huấn luyện mặc định **không** khớp được `data/labels/...`.

## Quy ước đặt tên

- **Ảnh:** `plate_0001.png`, `plate_0002.png`, … (số thứ tự 4 chữ số, giữ phần mở rộng gốc: `.png` / `.jpg` / …).
- **Nhãn:** cùng stem, đuôi `.txt`: `plate_0001.txt`, `plate_0002.txt`, …
- **Một lớp detection:** toàn bộ class gốc (ví dụ BSD/BSV trong datakangle) được gom về class `0` = `license_plate` trong `data.yaml`.

## Lệnh thường dùng

Từ thư mục gốc repo:

```bash
# Migrate / cập nhật từ datakangle (ghi đè ảnh+nhãn trong data/images/raw và data/labels/raw)
python -m src.detector.organize_datakangle_for_buoi2

# Xem trước, không ghi file
python -m src.detector.organize_datakangle_for_buoi2 --dry-run

# Tạo lại train/val/test (70/20/10, seed 42)
python -m src.detector.prepare_splits
```

Nếu trong `datakangle` chỉ có nhãn polygon (thư mục `labels/`), chạy trước:

```bash
python -m src.detector.convert_polygon_to_yolo_bbox --dataset-root datakangle --swap-for-yolo-detect
```

sau đó mới `organize_datakangle_for_buoi2` (mặc định ưu tiên `labels_bbox/`).

## Thu thập thêm (plan mục 3)

### Video thực địa trong `vid/` — ý nghĩa và chuỗi làm việc

Thư mục **`vid/`** (ví dụ `vid/ Giao thông Việt Nam`) là **kho clip quay thực tế**, không phải tập đã có nhãn. Trong đồ án nó phục vụ hai việc:

1. **Mở rộng dữ liệu huấn luyện:** trích frame định kỳ → được các ảnh tiềm năng → **gán nhãn YOLO** (LabelImg) → gom vào `data/images/raw` + `data/labels/raw` (hoặc gán tại `data/images/from_video/...` với nhãn tương ứng dưới `data/labels/...` theo quy tắc thay `images` → `labels` của Ultralytics) → chạy `prepare_splits` → train.  
2. **Demo / báo cáo:** chạy inference trên cả thư mục video (`infer_baseline` + `--from-video-collection`) để xuất clip hoặc ảnh có bbox, minh họa mô hình trên cảnh VN.

Tóm lại: **video ở `vid/` = nguồn gốc;** ảnh dùng để **học tham số** phải qua bước **trích frame + (lọc) + label** rồi mới cùng hệ thống với dữ liệu `datakangle`/plate_xxxx.

Trích frame từ **một** file vào thư mục tùy chọn (mặc định thường dùng `data/images/raw` nếu gom chung với tập cũ, hoặc dùng `--output-dir` riêng rồi chuyển ảnh tốt sang `raw`):

```bash
python -m src.preprocess.extract_video_frames --video path/to/video.mp4 --every 10
python -m src.preprocess.extract_video_frames --video clip.mp4 --every 5 --auto-index
```

Trích hàng loạt theo **bộ video** đã khai báo (`data/video_sources.json`) — mỗi file MP4 một thư mục con dưới `data/images/from_video/<ID>/`:

```bash
python -m src.preprocess.video_sources
python -m src.preprocess.extract_video_frames --collection giao_thong_vn --every 30
```

**Đưa frame vào tập huấn luyện (`plate_XXXX` trong `data/images/raw`)** — script `merge_from_video_to_raw`:

```bash
python -m src.preprocess.merge_from_video_to_raw --collection giao_thong_vn --dry-run
python -m src.preprocess.merge_from_video_to_raw --collection giao_thong_vn
```

- Mặc định **copy** (giữ bản trong `from_video`). Dùng `--move` nếu muốn chuyển hẳn để tiết kiệm chỗ.
- Đánh số **tiếp nối** sau `plate_*` đã có trong `raw` (không ghi đè).
- Ghi `data/migration_from_video_frames.json` (map file nguồn → `plate_XXXX`).
- Sau đó mở LabelImg, rồi `prepare_splits`.

**Chuỗi đầy đủ (video → train):**

```bash
python -m src.preprocess.extract_video_frames --collection giao_thong_vn --every 30
python -m src.preprocess.merge_from_video_to_raw --collection giao_thong_vn
# LabelImg: data/images/raw  ->  data/labels/raw
python -m src.detector.prepare_splits
```

Hoặc chỉ định thư mục chứa nhiều video (đúng đường dẫn trên máy bạn, kể cả khi có khoảng trắng sau `vid/`):

```bash
python -m src.preprocess.extract_video_frames --video-dir "vid/ Giao thông Việt Nam" --every 30
```

Inference YOLO trên cả thư mục video (Ultralytics đọc từng MP4):

```bash
python -m src.detector.infer_baseline --weights experiments/.../best.pt --from-video-collection giao_thong_vn --name infer_gtvn
```

Ảnh quá mờ / không đọc được biển: xoá tay hoặc không copy vào `data/images/raw/`.

## LabelImg (plan mục 4)

- **Open Dir:** `data/images/raw`
- **Change Save Dir:** `data/labels/raw`
- Định dạng **YOLO**, class: `license_plate` (một class, id `0` trong file `.txt`).

## EDA (plan mục 6)

```bash
jupyter notebook notebooks/eda/00_eda_overview.ipynb
```

## Tiền xử lý / augment thử (plan mục 7)

```bash
python -m src.preprocess.debug_augment_demo
python -m src.detector.prepare_splits --verify-only
```

Module: `src/preprocess/image_utils.py` (`load_image_bgr`, `resize_to_max_side`, `adjust_brightness_contrast`, `rotate_degrees`, …).
