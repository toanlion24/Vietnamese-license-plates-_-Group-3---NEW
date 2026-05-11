# Checklist trước commit

Copy chạy từ thư mục gốc project:

```powershell
.\.venv311\Scripts\python.exe scripts\strip_ipynb_outputs.py docs\*.ipynb
.\.venv311\Scripts\python.exe -m compileall src scripts
git status
git diff --stat
```

Trước khi commit, kiểm tra nhanh: notebook không còn output nặng, code `.py` compile được, và `git diff` chỉ có đúng file bạn muốn đưa lên.

Tùy chọn — sanity check đề tài (weight, manifest test, tài liệu hội đồng):

```powershell
.\.venv311\Scripts\python.exe scripts\check_de_tai_readiness.py
```
