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

# SAM 權重下載網址
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

try:
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
    # 不強制 sam.to('cpu')，讓其自動選擇 GPU 或 CPU
    mask_generator = SamAutomaticMaskGenerator(sam)
    print("[INFO] SAM 初始化完成")

    # 取得當前 script 目錄
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(script_dir, "images_masked", "train")
    label_dict = {}

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
                print(f"[DEBUG] 開始呼叫 mask_generator.generate ...")
                masks = mask_generator.generate(image)
                print(f"[DEBUG] mask_generator.generate 完成，mask 數量: {len(masks)}")
                # 建立黑色遮蔽區域 mask（假設黑色為 [0,0,0]）
                black_mask = np.all(image == [0, 0, 0], axis=-1)
                valid_area = ~black_mask
                # 合併所有 mask
                mask_sum = np.zeros(image.shape[:2], dtype=np.uint8)
                for m in masks:
                    mask_sum = np.logical_or(mask_sum, m["segmentation"]).astype(np.uint8)
                # 只計算非黑色區域的覆蓋率
                valid_pixel_count = np.count_nonzero(valid_area)
                if valid_pixel_count == 0:
                    coverage = 0.0
                else:
                    coverage = (mask_sum & valid_area).sum() / valid_pixel_count
                # 教師標籤同時記錄覆蓋率與 mask 數量
                label_dict[img_path] = {
                    "coverage": coverage,
                    "mask_count": len(masks)
                }
                print(f"[INFO] {img_path}: 覆蓋率={coverage:.4f}, mask數={len(masks)}")
            except Exception as img_e:
                print(f"[錯誤] 處理圖片 {img_path} 失敗：{img_e}")

    # 儲存教師標籤
    with open("train_coverage_labels.json", "w", encoding="utf-8") as f:
        json.dump(label_dict, f, ensure_ascii=False, indent=2)

    print("[INFO] 已完成所有圖片分割與覆蓋率計算，教師標籤已儲存 train_coverage_labels.json")
except Exception as e:
    import traceback
    print("[主程式錯誤]：")
    traceback.print_exc()
