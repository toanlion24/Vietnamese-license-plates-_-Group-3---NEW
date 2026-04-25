# Buoi 1 - Khoi dong va lap ke hoach chi tiet (ban hoc lai tu goc)

Tai lieu nay duoc viet theo muc tieu: **hoc lai tung kien thuc nho**, de ban khong chi "lam duoc" ma con "hieu vi sao lam nhu vay".

---

## 1) Ban chat bai toan (hieu dung de khong bi lac huong)

### 1.1 Bai toan can giai quyet la gi?
- Dau vao: anh, video, hoac webcam co xe.
- Dau ra: voi moi bien so tim thay, tra ve:
  - Vi tri bien so tren anh (`bbox`: x_min, y_min, x_max, y_max).
  - Chuoi ky tu bien so (`plate_text`).
  - Muc do tin cay (`confidence`).

Ngan gon: day la bai toan **detect + OCR**.

### 1.2 Vi sao phai tach 2 buoc detect va OCR?
- Detect tra loi cau hoi: "bien so nam o dau?"
- OCR tra loi cau hoi: "bien so viet gi?"
- Tach 2 buoc giup:
  - De debug (sai do detect hay sai do OCR).
  - De thay the mo hinh linh hoat (doi OCR ma khong phai doi detect).
  - De de danh gia theo tung tang (mAP cho detect, CER/WER/plate accuracy cho OCR).

### 1.3 Cac boi canh ung dung thuc te
- Bai giu xe: doc bien de vao/ra cong.
- Giam sat giao thong: truy vet xe vi pham.
- Kiem soat noi bo: xe ra vao cong ty/khu dan cu.

### 1.4 Kieu du lieu dau vao can luu y
- Ngay/ban dem, nguoc sang, do sang thay doi.
- Bien 1 dong/2 dong, goc nghieng, bi mo.
- Nhieu xe trong cung khung hinh.

---

## 2) Chon stack cong nghe (va ly do de de bao ve)

## 2.1 Detection: YOLOv8 hay YOLOv5?

| Tieu chi | YOLOv5 | YOLOv8 |
|---|---|---|
| Do pho bien trong do an | Rat pho bien | Pho bien + moi hon |
| De bat dau nhanh | Tot | Rat tot (Ultralytics API gon) |
| Tai lieu/vi du | Nhieu | Rat nhieu |
| Kha nang baseline nhanh | Tot | Tot hon cho nguoi moi |

**Chot de xai cho Buoi 1->3: YOLOv8n/s** vi setup nhanh, code infer train gon, phu hop moc thoi gian 7 buoi.

> Goi y hoc: bat dau bang `yolov8n` (nhe, train nhanh), khi on thi thu `yolov8s`.

### 2.2 OCR: EasyOCR hay Tesseract?

| Tieu chi | EasyOCR | Tesseract |
|---|---|---|
| De dung ngay | De | Trung binh |
| Chat luong voi anh thuc te | Thuong on hon | Nhay cam voi tien xu ly |
| Toc do setup | Nhanh | Nhanh |
| Kha nang tuy bien | Vua phai | Cao (nhieu tham so) |

**Chot baseline: EasyOCR** (de dat ket qua nhanh trong do an).  
Tesseract de du phong de so sanh.

### 2.3 Nhanh gon luong cong viec ky thuat Buoi 1
1. Tao moi truong Python.
2. Cai `ultralytics`, `easyocr`, `opencv-python`, `numpy`.
3. Chay thu:
   - 1 lenh YOLO pretrained detect object (de test moi truong).
   - 1 script OCR tren 1 crop bien so mau.

Neu 2 thu nay chay duoc -> moi truong da san sang cho Buoi 2-3.

---

## 3) Format bien so Viet Nam (phan quan trong de hau xu ly)

Muc tieu khong phai nho het format phap ly, ma la nam khung de sua loi OCR.

### 3.1 Mau co ban thuong gap
- Dang tong quat: `NNX-XXXXX` (vi du `51F-12345`)
  - `NN`: 2 chu so dau (ma dia phuong)
  - `X`: 1 ky tu chu
  - `XXXXX`: day so cuoi (4-5 chu so)
- Bien 2 dong: OCR co the doc thanh 2 dong, can noi lai thanh 1 chuoi.

### 3.2 Nhung loi OCR rat hay gap
- `O` <-> `0`
- `I` <-> `1`
- `S` <-> `5`
- `B` <-> `8`
- Mat dau gach noi `-`
- Chen dau cham/khoang trang linh tinh

### 3.3 Bo quy tac regex co ban (de bat dau)
Regex "de lot":
- `^[0-9]{2}[A-Z][0-9]{4,5}$` (bo ky tu phu, bo gach noi)
- `^[0-9]{2}[A-Z]-?[0-9]{4,5}$` (chap nhan co/khong co `-`)

Quy tac sua loi theo vi tri:
1. Hai ky tu dau bat buoc la so -> neu OCR ra chu thi doi sang so gan dung (`O->0`, `I->1`...).
2. Vi tri thu 3 uu tien la chu cai.
3. Cac vi tri cuoi uu tien la so.

> Day la quy tac "thuc dung cho do an". Co the chua bao phu tat ca truong hop dac biet, nhung du de tang do chinh xac baseline.

---

## 4) Pipeline tong the (hieu tung khoi nho)

Luong xu ly 1 anh:
1. **Input**: doc anh/frame.
2. **Detect**: YOLO tim bbox bien so.
3. **Crop + preprocess**:
   - crop theo bbox,
   - grayscale,
   - threshold/contrast,
   - resize,
   - (tuy chon) deskew.
4. **OCR**: EasyOCR doc ky tu.
5. **Postprocess**:
   - uppercase,
   - loai ky tu la,
   - sua theo regex/quy tac.
6. **Output**: `plate_text`, `bbox`, `confidence`, latency moi stage.

### 4.1 Vi sao can preprocess truoc OCR?
- OCR rat nhay voi nhieu va mo.
- Bien so thuong nho, can resize de ky tu ro hon.
- Threshold giup tang tuong phan nen/chu.

### 4.2 Vi sao can postprocess sau OCR?
- OCR hien thi "xac suat", khong biet luat bien so VN.
- Regex + rule giup bien ket qua "giong bien that hon".

---

## 5) Chi so danh gia (de biet tien bo that hay ao)

### 5.1 Detection
- mAP@0.5 (hoac mAP50-95 neu co): do detector khoanh dung bien so den muc nao.

### 5.2 OCR
- Character Accuracy = so ky tu dung / tong ky tu.
- CER = edit_distance_ky_tu / tong ky tu GT.
- WER = edit_distance_theo_tu / tong tu GT.
- Plate Accuracy = % bien so khop hoan toan sau normalize.

**Muc tieu do an**: OCR >= 85% tren >= 200 anh thuc te khong dung de train.

---

## 6) Timeline 7 buoi (ban chi can bam theo)

### Buoi 1 (hom nay)
- Chot stack, pipeline, regex can ban, phan cong, tai lieu mo ta bai toan.

### Buoi 2
- Thu thap + gan nhan >= 200 anh.
- Tao split train/val/test + EDA.

### Buoi 3
- Train YOLO baseline + infer + luu checkpoint.

### Buoi 4
- Thu nghiem A/B DeepSolo va DeepSolo+TrOCR.
- Bao cao CER/WER/Plate accuracy.

### Buoi 5
- Tich hop full pipeline detect->OCR->postprocess.
- Danh gia tren >= 200 anh test thuc te.

### Buoi 6
- Lam demo (CLI/webcam/web) + test tinh on dinh.

### Buoi 7
- Chot bao cao, slide, dien tap bao ve.

---

## 7) Phan cong vai tro (mau de dung ngay)

- Thanh vien A: du lieu + gan nhan + EDA.
- Thanh vien B: YOLO train/eval.
- Thanh vien C: OCR + preprocess + regex.
- Ca nhom: demo + bao cao + slide.

Neu ban lam 1 minh:
- Tuan tu theo module, khong nhay coc:
  1) du lieu,
  2) detect,
  3) OCR,
  4) tich hop,
  5) demo.

---

## 8) Checklist hoan thanh Buoi 1

- [ ] Co tai lieu mo ta bai toan (co input/output/metrics ro rang).
- [ ] Co bang so sanh YOLOv5-v8 va chot YOLOv8.
- [ ] Co test moi truong: YOLO pretrained va EasyOCR da chay.
- [ ] Co regex + quy tac sua loi OCR co ban cho bien so VN.
- [ ] Co so do pipeline tong the.
- [ ] Co timeline 7 buoi + phan cong.

Neu tick het checklist tren, ban da hoan thanh dung muc tieu Buoi 1 va san sang cho Buoi 2.

---

## 9) Ke hoach hoc lai "tung mieng nho" (goi y cho ban)

Moi ngay hoc 60-90 phut:
1. 15 phut on ly thuyet 1 khoi (detect hoac OCR).
2. 30-45 phut chay code minh hoa 1 khoi.
3. 15 phut ghi lai "toi hieu gi / chua hieu gi".
4. 10 phut tong ket bang 3 cau:
   - Dau vao la gi?
   - Dau ra la gi?
   - Vi sao khoi nay can thiet?

Cach nay giup ban nho sau, thay vi chi copy code.
