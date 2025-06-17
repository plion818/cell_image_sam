# 使用 Segment Anything Model (SAM) 自動分割與教師標籤產生

本專案可自動批次處理 `AI_MSC_密度照片/train` 目錄下所有圖片，利用 Facebook Research 的 Segment Anything Model（SAM）進行自動分割，並計算每張圖的覆蓋率（mask 占整張圖的比例），作為教師標籤，儲存於 `train_coverage_labels.json`。

## 步驟概要

1. 安裝 SAM 相依套件。
2. 下載 SAM 權重。
3. 執行程式，批次處理所有訓練圖片，計算覆蓋率並儲存教師標籤。

## 安裝相依套件

### 本地安裝
建議直接安裝 requirements.txt 內所有相依套件：

```powershell
pip install -r requirements.txt
```

### Colab 或需最新版 segment-anything
若在 Colab 或需安裝最新版 segment-anything，請使用：

```python
!pip install git+https://github.com/facebookresearch/segment-anything.git torch torchvision opencv-python pillow
```

- Colab 環境每次重啟都需重新安裝。
- 若需特定 CUDA 版本，請依 PyTorch 官網指示安裝 torch/torchvision。

## segment-anything 套件內容
- SAM 模型架構與推論 API
- 權重載入（不自動下載權重，需手動或程式下載）
- 自動遮罩產生器與分割工具
- 需搭配 PyTorch 執行

## 執行方式
安裝完成後，執行 `sam_train_coverage.py`，程式會自動下載 SAM 權重並處理所有訓練圖片。

```powershell
python sam_train_coverage.py
```

## 輸出
- `train_coverage_labels.json`：每張訓練圖片的覆蓋率教師標籤。

MASK_LEFT_BOTTOM = (98, 1505, 470, 45)     # 遮左下角時間戳（黑底黃字）
MASK_RIGHT_BOTTOM = (1255, 1451, 260, 56)  # 遮右下角 200μm 10x 倍率標註

如有任何問題，歡迎提出！
