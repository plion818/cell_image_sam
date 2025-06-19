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

---

## 分類模型訓練與推論流程

本專案支援多種分類模型（MLP、Random Forest、LightGBM、XGBoost、Logistic Regression），並提供完整的訓練、推論、交叉驗證、Grid Search、自動標準化等功能。

### 1. 訓練模型

以 Logistic Regression 為例（其他模型資料夾如 MLP、RF、LGBM、XGB 用法一致）：

```powershell
python LR/lr_train.py
```
- 會自動讀取 `output/train_coverage_labels.csv`，訓練後儲存模型與 label encoder。

### 2. 單筆推論

```powershell
python LR/lr_predict_single.py
```
- 互動式輸入 coverage_ratio、mask_count，顯示預測分類與機率。

### 3. 批次測試集推論

```powershell
python LR/lr_predict_testset.py
```
- 讀取 `output/test_coverage_labels.csv`，批次預測所有測試資料，顯示正確率與每筆機率。

### 4. 交叉驗證（K-Fold/LOOCV）

```powershell
python LR/lr_cv.py --method loocv
python LR/lr_cv.py --method kfold --k 5
```
- 支援 Leave-One-Out 及 K-Fold 交叉驗證，顯示每折/每筆分數與平均正確率。

### 5. Grid Search 自動調參

```powershell
python LR/lr_grid_search.py
```
- 自動搜尋最佳參數（含標準化），最佳模型自動儲存，推論時自動標準化。

### 6. 標準化說明
- Logistic Regression 會自動標準化特徵（StandardScaler），推論時無需手動處理。
- 其他模型（如 MLP）如有標準化需求，請參考對應資料夾內說明。

---

## 其他模型
- `MLP/`、`RF/`、`LGBM/`、`XGB/` 皆有對應 train、predict、cv、grid_search script，指令用法與 LR 類似。
- XGBoost、RF、LGBM 皆支援交叉驗證與 Grid Search。
- 所有 script 路徑皆以專案根目錄為基準，跨資料夾執行不會有路徑錯誤。

---

## 常見問題
- 若遇到 Logistic Regression 收斂警告，已自動標準化並提高 max_iter，仍有問題請檢查資料分布。
- 欲新增/修改模型，請參考現有資料夾結構與 script 範例。

如有任何問題，歡迎提出！
