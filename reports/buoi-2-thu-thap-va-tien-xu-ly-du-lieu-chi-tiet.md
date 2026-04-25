# Buoi 2 - Thu thap va tien xu ly du lieu (chi tiet + chay ngay)

Tai lieu nay theo dung format Buoi 1: hoc tung buoc nho, co checklist thao tac, va co script khung de ban chay ngay.

---

## 1) Muc tieu Buoi 2

- Co bo du lieu bien so VN ban dau (anh + nhan detection).
- Co file split `train/val/test` de san sang train YOLO Buoi 3.
- Co bao cao EDA nhanh de biet chat luong du lieu.
- Co bo quy trinh tien xu ly/augment co ban (de dung lai o Buoi 3-5).

---

## 2) Dau vao, dau ra cua Buoi 2 (de khong roi)

### Dau vao

- Anh xe co bien so trong `data/images/raw/`.
- Nhan YOLO trong `data/labels/` (moi anh 1 file `.txt` cung ten).

### Dau ra

- `data/splits/train.txt`
- `data/splits/val.txt`
- `data/splits/test.txt`
- `reports/eda/dataset_report.json`
- `reports/eda/dataset_report.md`
- `reports/eda/preview_samples/*.jpg` (anh ve box de check nhan)

---

## 3) Checklist thao tac tung buoc (lam theo dung thu tu)

## Buoc 0 - Chuan bi thu muc

- Dat anh vao `data/images/raw/`.
- Dat nhan YOLO vao `data/labels/raw/`.
- Kiem tra nhanh: voi `abc.jpg` phai co `abc.txt`.

> Dinh dang nhan YOLO 1 dong:
> `class_id x_center y_center width height` (toa do da normalize 0-1)

## Buoc 1 - Tao split train/val/test

- Chay script split:

```bash
python scripts/split_dataset.py --input-dir data/images/raw --output-dir data/splits --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --seed 42
```

- Kiem tra 3 file txt da tao.
- Kiem tra tong dong train+val+test = tong so anh.

## Buoc 2 - Chay EDA nhanh

- Chay script EDA:

```bash
python scripts/eda_dataset.py --images-dir data/images --labels-dir data/labels/raw --output-dir reports/eda --num-preview 12
```

- Mo `reports/eda/dataset_report.md` de doc tong quan.
- Mo thu muc `reports/eda/preview_samples/` de xem box da dung chua.

## Buoc 3 - Dan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    h gia chat luong du lieu sau EDA

- Neu `images_without_labels` cao: bo sung nhan.
- Neu box qua nho (avg_area_ratio qua thap): uu tien bo sung anh can canh hon.
- Neu qua nhieu anh "dark": bo sung anh sang hon hoac tang preprocess.

## Buoc 4 - Clean du lieu toi thieu + can bang don gian

- Chay clean + balance:

```bash
python scripts/clean_balance_dataset.py --images-dir data/images --labels-dir data/labels/raw --output-dir data/manifests --report-path reports/eda/clean_balance_report.json --seed 42
```

- Kiem tra:
  - `data/manifests/clean_images.txt`
  - `data/manifests/balanced_images.txt`
  - `reports/eda/clean_balance_report.json`

## Buoc 5 - Tao bo preprocess/augment co ban (giu dong bo bbox)

- Chay script augment:

```bash
python scripts/preprocess_augment.py --input-dir data/images --labels-dir data/labels/raw --output-dir data/interim/augmented --size 640 640 --max-images 0 --seed 42
```

- Kiem tra output trong `data/interim/augmented/` da co cac bien the anh va file `.txt` bbox tuong ung:
  - `__resized`
  - `__brightness_contrast`
  - `__rotated`
- Chon 20-30 anh bat ky de xem chat luong augment co hop ly khong.

## Buoc 6 - Chot du lieu cho Buoi 3

- Khoa split (khong random lai neu khong can).
- Luu seed va ratio da dung (`42`, `0.8/0.1/0.1`).
- Ghi lai so lieu cuoi buoi: tong anh, tong box, tinh trang nhan.

---

## 4) Giai thich de hieu sau (khong chi chay lenh)

### 4.1 Vi sao phai split co seed?

- Seed giup split lap lai duoc.
- Mai train lai van cung tap train/val/test -> so sanh cong bang.

### 4.2 Vi sao EDA can thiet truoc khi train?

- Nhieu bai fail khong phai do model, ma do nhan loi hoac du lieu lech.
- EDA giup phat hien som:
  - thieu nhan,
  - box ve sai,
  - phan bo anh qua toi/sang,
  - box qua be (de detect).

### 4.3 Y nghia mot so chi so trong report

- `avg_boxes_per_image`: trung binh moi anh co bao nhieu bien.
- `avg_area_ratio`: dien tich box/tong dien tich anh (cang nho cang kho detect).
- `brightness_distribution`: phan bo anh toi/thuong/sang.

---

## 5) Script khung da co trong repo (ban vua co the dung ngay)

- `scripts/split_dataset.py`
  - Tach du lieu theo ratio + seed.
  - Xuat `train.txt`, `val.txt`, `test.txt`.
  - Co tuy chon `--absolute-paths` neu ban muon duong dan tuyet doi.
- `scripts/eda_dataset.py`
  - Doc anh + nhan YOLO.
  - Tinh thong ke tong quan.
  - Ve box len mot so anh mau.
  - Xuat report JSON + Markdown.
- `scripts/preprocess_augment.py`
  - Resize ve kich thuoc chuan.
  - Tang/giam sang + contrast co kiem soat.
  - Xoay nhe de mo phong goc chup thuc te.
  - Luu bo anh bien the de tham khao va tai su dung.

---

## 6) Lenh mau full flow Buoi 2 (copy la chay)

```bash
python scripts/split_dataset.py --input-dir data/images/raw --output-dir data/splits --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --seed 42
python scripts/eda_dataset.py --images-dir data/images --labels-dir data/labels/raw --output-dir reports/eda --num-preview 12
python scripts/clean_balance_dataset.py --images-dir data/images --labels-dir data/labels/raw --output-dir data/manifests --report-path reports/eda/clean_balance_report.json --seed 42
python scripts/preprocess_augment.py --input-dir data/images --labels-dir data/labels/raw --output-dir data/interim/augmented --size 640 640 --seed 42
```

---

## 7) Tieu chi "hoan thanh Buoi 2"

- Co >= 200 anh da gan nhan de train baseline.
- Co 3 file split train/val/test.
- Co report EDA va preview box.
- Co bo preprocess/augment co ban tao duoc tu script.
- Co ghi chu ro rang cac loi du lieu can sua truoc Buoi 3.

Neu dat 4 dieu tren, ban da san sang vao Buoi 3 (train YOLO baseline).

---

## 8) Goi y hoc lai theo tung khoi nho (60-90 phut)

1. 20 phut: on ly thuyet nhan YOLO + split.
2. 25 phut: chay split va tu check file txt.
3. 25 phut: chay EDA va doc report.
4. 10 phut: ghi ra 3 van de du lieu lon nhat can sua.

Ban lap lai vong nay 2-3 lan voi du lieu cap nhat se thay tien bo rat ro.