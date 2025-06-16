# 使用 Segment Anything Model (SAM) 自動分割與教師標籤產生

本專案可自動批次處理 `AI_MSC_密度照片/train` 目錄下所有圖片，利用 Facebook Research 的 Segment Anything Model（SAM）進行自動分割，並計算每張圖的覆蓋率（mask 占整張圖的比例），作為教師標籤，儲存於 `train_coverage_labels.json`。

## 步驟概要

1. 安裝 SAM 相依套件。
2. 下載 SAM 權重。
3. 執行程式，批次處理所有訓練圖片，計算覆蓋率並儲存教師標籤。

## 安裝相依套件
請在 PowerShell 執行下列指令：

```powershell
pip install git+https://github.com/facebookresearch/segment-anything.git torch torchvision opencv-python pillow
```

## 執行方式
安裝完成後，執行 `sam_train_coverage.py`，程式會自動下載 SAM 權重並處理所有訓練圖片。

```powershell
python sam_train_coverage.py
```

## 輸出
- `train_coverage_labels.json`：每張訓練圖片的覆蓋率教師標籤。

如有任何問題，歡迎提出！
