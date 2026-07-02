# Boston Housing MLP Regression - Lab 4

Project này làm lại quy trình dự đoán giá nhà Boston Housing từ đầu và có thêm bước tối ưu hiệu năng model.

## Mục tiêu

1. Cào raw data từ StatLib/CMU.
2. Reconstruct và lọc dữ liệu sạch.
3. Chia feature `X` và target `MEDV`.
4. Train các model regression bắt buộc.
5. Tune model trên train set, không dùng test set để chọn hyperparameter.
6. Chọn best model theo RMSE thấp nhất trên held-out test set.
7. Lưu model, scaler, metric, tuning report và hình ảnh sau khi train.
8. Cung cấp giao diện Python bằng Streamlit để dự đoán bằng best model.
9. Kiểm tra toàn bộ chương trình bằng `pytest` và validation script.

## Cấu trúc chính

```text
data/raw/                 Raw data đã cào
data/processed/           Data sạch
models/                   Best model, scaler, metadata
reports/                  Metrics, tuning results, predictions, analysis
reports/figures/          Hình ảnh model và validation
notebooks/                Notebook báo cáo ảnh-only, không có code cell
src/data.py               Cào data, reconstruct, clean, chia feature
src/modeling.py           Train, tune, evaluate, chọn model, lưu artifact
src/train.py              Chạy toàn bộ pipeline
src/validate_project.py   Kiểm tra yêu cầu project
app.py                    Giao diện Streamlit
tests/                    Test tự động
```

## Cài đặt

```powershell
pip install -r requirements.txt
```

## Chạy training và tối ưu model

```powershell
python -m src.train
```

Kết quả chính:

- `reports/model_metrics.csv`
- `reports/optimized_model_metrics.csv`
- `reports/tuning_results.csv`
- `reports/test_predictions.csv`
- `reports/analysis.txt`
- `models/model_metadata.json`
- `models/best_model.joblib` hoặc `models/best_model.keras`
- `reports/figures/`

## Chạy giao diện

```powershell
streamlit run app.py
```

App sẽ đọc metadata để biết best model có chứa scaler bên trong hay cần dùng scaler riêng.

## Kiểm tra

```powershell
python -m pytest -q
python -m src.validate_project
```

Validation kiểm tra các yêu cầu: không dùng `load_boston`, data sạch đúng shape, có tuned models, metadata có best params/CV RMSE/test RMSE, notebook không có code cell và tất cả ảnh trong notebook đều tồn tại.
