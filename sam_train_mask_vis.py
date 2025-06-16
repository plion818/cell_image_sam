import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ======== 設定參數 ========
# SAM 參數
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"  # 確保與主程式相同權重
MODEL_TYPE = "vit_h"
# 允許直接用 Windows 風格路徑（自動補齊絕對路徑）
IMAGE_PATH = os.path.join(os.getcwd(), r"AI_MSC_密度照片\40-60_\2.tif")
DEVICE = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
# ==========================

# ======== 初始化 SAM ========
print("[INFO] 載入模型...")
sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT).to(DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)

# ======== 圖片讀取與遮罩推論 ========
image = np.array(Image.open(IMAGE_PATH).convert("RGB"))
masks = mask_generator.generate(image)

# 合併遮罩為 binary mask
final_mask = np.zeros(image.shape[:2], dtype=np.uint8)
for m in masks:
    final_mask |= m["segmentation"].astype(np.uint8)

# 計算覆蓋率
coverage = final_mask.sum() / final_mask.size
print(f"[INFO] 覆蓋率為：{coverage:.2%}")

# ======== 可視化疊圖 ========
plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.imshow(final_mask, alpha=0.4, cmap='Reds')
plt.title(f"覆蓋率: {coverage:.2%}")
plt.axis("off")
plt.show()
