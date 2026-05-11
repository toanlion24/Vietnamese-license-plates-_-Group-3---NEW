# Tóm tắt so sánh EasyOCR vs TrOCR trên video

- Nguồn: `D:\ComputerVisionNew\reports\video_compare_easyocr_vs_trocr.csv`
- Tổng số frame so sánh: **20**
- Tỉ lệ dự đoán giống nhau: **4/20 = 20.00%**

## Bảng tổng hợp latency (ms)

| Model/Chỉ số | Mean | P50 | P95 |
| --- | ---: | ---: | ---: |
| EasyOCR latency | 357.21 | 384.66 | 593.34 |
| TrOCR latency | 1153.95 | 1346.55 | 1593.53 |
| Gap (TrOCR - EasyOCR) | 796.74 | 905.93 | 1245.97 |

## Top frame lệch latency lớn nhất

| Rank | frame_idx | easyocr_pred | trocr_pred | easy_latency | trocr_latency | gap (trocr-easy) |
| ---: | ---: | --- | --- | ---: | ---: | ---: |
| 1 | 255 | 25 | 09200 | 363.5935 | 1730.9647 | 1367.3712 |
| 2 | 285 | 26 | 09202 | 346.7172 | 1586.2926 | 1239.5754 |
| 3 | 180 | 01153 | 0355 | 394.9043 | 1536.3528 | 1141.4485 |
| 4 | 150 | 24HE | 24AE | 304.8664 | 1425.6321 | 1120.7657 |
| 5 | 165 | 24AE | 03153 | 404.3909 | 1496.7293 | 1092.3384 |
