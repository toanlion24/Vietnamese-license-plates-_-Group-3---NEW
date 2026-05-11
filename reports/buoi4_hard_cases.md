# Buổi 4 — hard cases (tự động)

- Nguồn A: `outputs\buoi4\deepsolo_e2e_predictions.csv`
- Nguồn B: `outputs\buoi4\deepsolo_trocr_predictions.csv`
- Top 14 mẫu sai theo max(levenshtein(gt,pred_A), levenshtein(gt,pred_B)).

| rank | image_id | GT | pred A | pred B | error A | error B | max dist |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `synth_001` | `51H12345` | `5444P54L5` | `51H12345` | ocr_error | ok | 6 |
| 2 | `synth_009` | `14K10203` | `A4KK40PR` | `14K10203` | ocr_error | ok | 6 |
| 3 | `synth_011` | `65M44444` | `H5MAAAAL` | `65M44444` | ocr_error | ok | 6 |
| 4 | `synth_015` | `41S13579` | `AS45574` | `41S13579` | ocr_error | ok | 5 |
| 5 | `synth_003` | `59G11234` | `T9G34` | `59G11234` | ocr_error | ok | 4 |
| 6 | `synth_006` | `61D13579` | `64045579` | `61D13579` | ocr_error | ok | 4 |
| 7 | `synth_007` | `50F99881` | `516Q9884` | `50F99881` | ocr_error | ok | 4 |
| 8 | `synth_004` | `43B67890` | `45P67891` | `43B67890` | ocr_error | ok | 3 |
| 9 | `synth_014` | `18R24680` | `ABP24680` | `18R24680` | ocr_error | ok | 3 |
| 10 | `synth_008` | `72A45678` | `724465678` | `72A45678` | ocr_error | ok | 2 |
| 11 | `synth_012` | `92N12345` | `92142345` | `92N12345` | ocr_error | ok | 2 |
| 12 | `synth_002` | `30A56789` | `30456789` | `30A56789` | ocr_error | ok | 1 |
| 13 | `synth_005` | `29C24680` | `29C24681` | `29C24680` | ocr_error | ok | 1 |
| 14 | `synth_010` | `88L99999` | `88199999` | `88L99999` | ocr_error | ok | 1 |
