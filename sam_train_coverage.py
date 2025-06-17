import os
import cv2
import numpy as np
from PIL import Image
import json
import sys

print("[INFO] 程式開始執行")
try:
    print("[INFO] 嘗試匯入 segment_anything ...")
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    print("[INFO] 匯入 segment_anything 成功")
except Exception as e:
    print("[錯誤] 匯入 segment_anything 失敗：", e)
    sys.exit(1)

# 選擇模型：'sam' 或 'cellpose'
MODEL_SELECT = os.environ.get('SEG_MODEL', 'cellpose')  # 預設 cellpose
print(f"[INFO] 選擇模型: {MODEL_SELECT}")

mask_generator = None
if MODEL_SELECT == 'sam':
    # SAM 權重下載網址
    SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
    SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

    print("[INFO] 檢查 SAM 權重檔案 ...")
    if not os.path.exists(SAM_CHECKPOINT):
        import urllib.request
        print(f"[INFO] 下載 SAM 權重: {SAM_CHECKPOINT_URL}")
        urllib.request.urlretrieve(SAM_CHECKPOINT_URL, SAM_CHECKPOINT)
        print("[INFO] SAM 權重下載完成")
    else:
        print("[INFO] SAM 權重已存在")
    print("[INFO] 初始化 SAM ... (自動偵測裝置)")
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
    mask_generator = SamAutomaticMaskGenerator(sam)
    print("[INFO] SAM 初始化完成")

# 取得當前 script 目錄
script_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(script_dir, "images_masked", "train")
label_dict = {}

# 新增 cellpose 匯入
try:
    from cellpose import models as cellpose_models
    print("[INFO] 匯入 cellpose 成功")
    HAS_CELLPOSE = True
except Exception as e:
    print("[INFO] 匯入 cellpose 失敗：", e)
    HAS_CELLPOSE = False

print(f"[INFO] 開始處理訓練資料夾: {train_dir}")
for density_folder in os.listdir(train_dir):
    folder_path = os.path.join(train_dir, density_folder)
    if not os.path.isdir(folder_path):
        continue
    print(f"[INFO] 處理子資料夾: {folder_path}")
    for img_name in os.listdir(folder_path):
        if not img_name.lower().endswith(".tif"):
            continue
        img_path = os.path.join(folder_path, img_name)
        print(f"[INFO] 處理圖片: {img_path}")
        try:
            image = np.array(Image.open(img_path).convert("RGB"))
            print(f"[DEBUG] 讀取圖片 shape: {image.shape}")
            if MODEL_SELECT == 'cellpose' and HAS_CELLPOSE:
                # 支援新版與舊版 cellpose
                if hasattr(cellpose_models, 'CellposeModel'):
                    cp_model = cellpose_models.CellposeModel(model_type='cyto')
                else:
                    cp_model = cellpose_models.Cellpose(model_type='cyto')
                cp_result = cp_model.eval(image, diameter=None)
                if len(cp_result) == 4:
                    masks, flows, styles, diams = cp_result
                else:
                    masks, flows, diams = cp_result
                    styles = None
                mask_sum = (masks > 0).astype(np.uint8)
                mask_count = masks.max()
            else:
                if mask_generator is None:
                    raise RuntimeError("[錯誤] 未初始化 SAM，請確認模型選擇與權重下載！")
                print(f"[DEBUG] 開始呼叫 mask_generator.generate ...")
                masks = mask_generator.generate(image)
                print(f"[DEBUG] mask_generator.generate 完成，mask 數量: {len(masks)}")
                mask_sum = np.zeros(image.shape[:2], dtype=np.uint8)
                for m in masks:
                    mask_sum = np.logical_or(mask_sum, m["segmentation"]).astype(np.uint8)
                mask_count = len(masks)
            coverage = mask_sum.sum() / mask_sum.size
            label_dict[img_path] = {
                "coverage": coverage,
                "mask_count": int(mask_count)
            }
            print(f"[INFO] {img_path}: 覆蓋率={coverage:.4f}, mask數={mask_count}")
        except Exception as img_e:
            print(f"[錯誤] 處理圖片 {img_path} 失敗：{img_e}")

# 儲存教師標籤
with open("train_coverage_labels.json", "w", encoding="utf-8") as f:
    json.dump(label_dict, f, ensure_ascii=False, indent=2)

print("[INFO] 已完成所有圖片分割與覆蓋率計算，教師標籤已儲存 train_coverage_labels.json")
