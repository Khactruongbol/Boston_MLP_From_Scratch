# Boston Housing MLP Regression - Rebuild From Scratch

Project này được làm lại từ đầu theo đúng quy trình:

1. Cào data thô từ StatLib/CMU.
2. Lọc và kiểm tra data sạch.
3. Chia feature `X` và target `MEDV`.
4. Train nhiều model regression.
5. Chọn model tối ưu nhất theo RMSE thấp nhất trên test set.
6. Lưu các ảnh sau khi train.
7. Test các model bằng MSE, RMSE, MAE và R2.
8. Tạo giao diện Python bằng Streamlit sử dụng model tốt nhất.
9. Kiểm tra lại toàn bộ chương trình bằng `pytest`.
10. Tạo Git repo cho toàn bộ chương trình.

## Cấu Trúc

```text
data/raw/                 data thô đã cào
data/processed/           data sạch
models/                   scaler, model tốt nhất, metadata
reports/                  metric, prediction, analysis
reports/figures/          ảnh sau train
src/data.py               cào data, làm sạch, chia feature
src/modeling.py           train, test, chọn model, lưu artifact
src/train.py              chạy toàn bộ pipeline
app.py                    giao diện Python Streamlit
tests/                    kiểm tra chương trình
```

## Cài Đặt

```powershell
pip install -r requirements.txt
```

## Chạy Pipeline

```powershell
python -m src.train
```

Kết quả được lưu vào:

- `data/raw/boston_raw.txt`
- `data/processed/boston_clean.csv`
- `models/`
- `reports/model_metrics.csv`
- `reports/test_predictions.csv`
- `reports/analysis.txt`
- `reports/figures/`

## Chạy Giao Diện

```powershell
streamlit run app.py
```

Giao diện sẽ load scaler và model tốt nhất đã được lưu sau khi chạy training.

## Kiểm Tra

```powershell
pytest -q
python -m src.validate_project
```
