import os
import numpy as np
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# SAM 權重
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"

# 初始化 SAM
sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
mask_generator = SamAutomaticMaskGenerator(sam)

# 輸出可視化圖片的資料夾
output_dir = os.path.join("train_mask_vis")
os.makedirs(output_dir, exist_ok=True)

# 處理所有訓練圖片
train_dir = os.path.join("AI_MSC_密度照片", "train")

for density_folder in os.listdir(train_dir):
    folder_path = os.path.join(train_dir, density_folder)
    if not os.path.isdir(folder_path):
        continue
    for img_name in os.listdir(folder_path):
        if not img_name.lower().endswith(".tif"):
            continue
        img_path = os.path.join(folder_path, img_name)
        image = np.array(Image.open(img_path).convert("RGB"))
        masks = mask_generator.generate(image)
        # 合併所有 mask
        mask_sum = np.zeros(image.shape[:2], dtype=np.uint8)
        for m in masks:
            mask_sum = np.logical_or(mask_sum, m["segmentation"]).astype(np.uint8)
        # 疊加顏色 (紅色半透明)
        vis = image.copy()
        vis[mask_sum == 1] = (vis[mask_sum == 1] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
        # 儲存可視化圖片
        out_subdir = os.path.join(output_dir, density_folder)
        os.makedirs(out_subdir, exist_ok=True)
        out_path = os.path.join(out_subdir, img_name.replace('.tif', '_maskvis.png'))
        Image.fromarray(vis).save(out_path)
        print(f"已儲存: {out_path}")

print("所有分割可視化圖片已完成！")
