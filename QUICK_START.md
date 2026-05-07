# QUICK START - DenseNet201

## 1. Bạn sẽ chạy file nào?

Notebook cần chạy là:

```text
animals_DenseNet201.ipynb
```

Notebook này chạy toàn bộ pipeline DenseNet201: đọc dataset, chia dữ liệu, train trên GPU, đánh giá và lưu output.

## 2. Chuẩn bị dataset

Tải dataset từ Kaggle:

https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals

Sau khi giải nén, đặt folder `animals_dataset` ở thư mục gốc của project:

```text
90-Animals-Classification/
├─ animals_dataset/
├─ DenseNet201/
│  └─ animals_DenseNet201.ipynb
```

Bên trong `animals_dataset` phải có 90 folder con, ví dụ:

```text
animals_dataset/
├─ antelope/
├─ badger/
├─ bat/
├─ ...
└─ zebra/
```

## 3. Tạo môi trường Python

Mở PowerShell trong thư mục project:

```powershell
cd C:\path\to\90-Animals-Classification
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Nếu PowerShell chặn script, chạy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 4. Cài thư viện

Nếu dùng GPU NVIDIA trên Windows, cài PyTorch CUDA theo bản phù hợp. Với CUDA 12.8:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install notebook ipykernel pandas numpy matplotlib seaborn scikit-learn pillow tqdm
```

Kiểm tra GPU:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
```

Kết quả mong muốn:

```text
True
NVIDIA ...
```

## 5. Cài kernel cho notebook

```powershell
python -m ipykernel install --user --name animals-densenet201 --display-name "Python (animals-densenet201)"
```

## 6. Mở notebook bằng VS Code

1. Mở VS Code.
2. Chọn `File` -> `Open Folder`.
3. Mở thư mục project.
4. Mở file `DenseNet201/animals_DenseNet201.ipynb`.
5. Chọn kernel `Python (animals-densenet201)`.
6. Chạy notebook từ trên xuống.

## 7. Các cell quan trọng

| Phần | Ý nghĩa |
|---|---|
| Thiết lập môi trường | kiểm tra thư viện và CUDA |
| Cấu hình đường dẫn | xác định dataset, output, epoch, batch size |
| Khám phá dataset | kiểm tra tổng ảnh, số class, phân phối class |
| Chia dữ liệu | tạo train, validation và test |
| DataLoader | biến ảnh thành batch Tensor |
| Build model | load DenseNet201 pretrained và thay classifier |
| Train model | train head rồi fine-tune `last_denseblock` |
| Evaluate | tính accuracy, precision, recall, F1 |
| Visualization | vẽ confusion matrix và ảnh dự đoán |
| Export | lưu model, config, labels và report |

## 8. Output sau khi chạy

Notebook sẽ tạo hoặc cập nhật thư mục:

```text
densenet201_outputs/
```

Các file chính:

```text
animals_densenet201_final.pth
animals_densenet201_config.json
animals_densenet201_labels.json
animals_densenet201_history.csv
animals_densenet201_classification_report.csv
animals_densenet201_training_curves.png
animals_densenet201_metric_overview.png
animals_densenet201_confusion_matrix.png
animals_densenet201_confusion_matrix_normalized.png
animals_densenet201_worst_classes.png
animals_densenet201_top_confusions.png
animals_densenet201_correct_predictions.png
animals_densenet201_incorrect_predictions.png
```

## 9. Lỗi thường gặp

### Không tìm thấy `animals_dataset`

Kiểm tra lại vị trí dataset. Folder `animals_dataset` phải nằm ở thư mục gốc project hoặc vị trí mà notebook có thể tìm thấy.

### `CUDA available: False`

Notebook được thiết kế để train bằng GPU. Hãy kiểm tra lại:

```powershell
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

Nếu `torch.cuda.is_available()` là `False`, bạn cần cài lại đúng bản PyTorch CUDA.

### Hết VRAM

DenseNet201 dùng ảnh 224x224 nhưng model vẫn khá sâu. Nếu hết VRAM, giảm `BATCH_SIZE` trong cell cấu hình:

```python
BATCH_SIZE = 64
```

hoặc:

```python
BATCH_SIZE = 32
```

### Train chậm

DenseNet201 có nhiều layer hơn ResNet-50. Nếu muốn thử nhanh pipeline, có thể giảm số epoch hoặc giảm batch size để kiểm tra trước khi train nghiêm túc.

## 10. Cách xem nhanh kết quả mà không train lại

Nếu thư mục `densenet201_outputs/` đã có sẵn, bạn có thể mở trực tiếp:

- `animals_densenet201_classification_report.csv`
- `animals_densenet201_training_curves.png`
- `animals_densenet201_confusion_matrix.png`
- `animals_densenet201_top_confusions.png`
- `animals_densenet201_correct_predictions.png`
- `animals_densenet201_incorrect_predictions.png`

Đây là cách nhanh nhất để đọc kết quả trước khi quyết định có train lại hay không.
