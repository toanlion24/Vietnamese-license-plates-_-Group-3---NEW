# Data Directory

## Structure

```
data/
├── raw/                    # Ảnh gốc thu thập (nên .gitignore)
├── labels/                 # File nhãn YOLO cho biển số
├── splits/                 # Danh sách train/val/test
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── manifests/              # Manifest cho OCR training
│   └── README.md
├── yolo_dataset/           # Dataset định dạng YOLO
│   └── data.yaml
└── crops/                  # Crop biển số đã detect (cho Qwen2-VL training)
    ├── train/
    └── val/
```

## Manifest Format

```csv
image_id,image_path,text_gt,split
plate_0001,/path/to/crops/plate_0001.jpg,30G112345,train
plate_0002,/path/to/crops/plate_0002.jpg,51K123456,train
...
```

## Dataset splits

- **train**: Dùng để huấn luyện
- **val**: Dùng để validation (điều chỉnh hyperparameters)
- **test**: Dùng để đánh giá cuối cùng (KHÔNG dùng trong quá trình train)

## Ghi chú

- Không commit ảnh raw lớn vào git
- Sử dụng `.gitignore` để bỏ qua `raw/`, `crops/`, `*.jpg`, `*.png`
