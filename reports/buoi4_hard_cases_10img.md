# Buổi 4 — hard cases (tự động)

- Nguồn A: `outputs\buoi4\deepsolo_e2e_predictions.csv`
- Nguồn B: `outputs\buoi4\deepsolo_trocr_predictions.csv`
- Top 10 mẫu sai theo max(levenshtein(gt,pred_A), levenshtein(gt,pred_B)).

| rank | image_id | GT | pred A | pred B | error A | error B | max dist |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `plate_0003` | `36B563560` | `3685` | `RM` | ocr_error | ocr_error | 9 |
| 2 | `plate_0005` | `29E172301` | `` | `YAMA` | ocr_error | ocr_error | 9 |
| 3 | `plate_0010` | `15B188870` | `1587` | `NET` | ocr_error | ocr_error | 9 |
| 4 | `plate_0001` | `30B34414` | `` | `` | ocr_error | ocr_error | 8 |
| 5 | `plate_0004` | `20D20224` | `224` | `TAX` | ocr_error | ocr_error | 8 |
| 6 | `plate_0006` | `29AF43850` | `38` | `AMT` | ocr_error | ocr_error | 8 |
| 7 | `plate_0008` | `29AE88280` | `282E0` | `CASH` | ocr_error | ocr_error | 8 |
| 8 | `plate_0009` | `30K21969` | `30` | `RM` | ocr_error | ocr_error | 8 |
| 9 | `plate_0002` | `12D107864` | `07864` | `67864` | ocr_error | ocr_error | 5 |
| 10 | `plate_0007` | `30A38056` | `30A` | `38056` | ocr_error | ocr_error | 5 |
