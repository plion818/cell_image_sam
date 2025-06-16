import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ======== 設定參數 ========
# SAM 參數
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"  # 確保與主程式相同權重
SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

# 若權重不存在則自動下載
if not os.path.exists(SAM_CHECKPOINT):
    import urllib.request
    print(f"[INFO] SAM 權重不存在，自動下載: {SAM_CHECKPOINT_URL}")
    urllib.request.urlretrieve(SAM_CHECKPOINT_URL, SAM_CHECKPOINT)
    print("[INFO] SAM 權重下載完成")

MODEL_TYPE = "vit_h"
# 取得當前 script 目錄，並組合圖片路徑
script_dir = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(script_dir, "images_masked", "train", "0-20_", "3.tif")
DEVICE = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
# ==========================

# ======== 初始化 SAM ========
print("[INFO] 載入模型...")
sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT).to(DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)

# 新增 cellpose 匯入
try:
    from cellpose import models as cellpose_models
    print("[INFO] 匯入 cellpose 成功")
    HAS_CELLPOSE = True
except Exception as e:
    print("[INFO] 匯入 cellpose 失敗：", e)
    HAS_CELLPOSE = False

# 選擇模型：'sam' 或 'cellpose'
MODEL_SELECT = os.environ.get('SEG_MODEL', 'cellpose')  # 預設 cellpose
print(f"[INFO] 選擇模型: {MODEL_SELECT}")

# ======== 圖片讀取與遮罩推論 ========
image = np.array(Image.open(IMAGE_PATH).convert("RGB"))
if MODEL_SELECT == 'cellpose' and HAS_CELLPOSE:
    cp_model = cellpose_models.Cellpose(model_type='cyto')
    masks, flows, styles, diams = cp_model.eval(image, diameter=None, channels=[0,0])
    final_mask = (masks > 0).astype(np.uint8)
    mask_count = masks.max()
else:
    masks = mask_generator.generate(image)
    final_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for m in masks:
        final_mask |= m["segmentation"].astype(np.uint8)
    mask_count = len(masks)

# 覆蓋率直接以全圖像素計算
coverage = final_mask.sum() / final_mask.size
print(f"[INFO] coverage_ratio：{coverage:.2%}，mask_count：{mask_count}")

# ======== 可視化疊圖 ========
plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.imshow(final_mask, alpha=0.4, cmap='Reds')
plt.title(f"覆蓋率: {coverage:.2%}  mask數: {mask_count}")
plt.axis("off")
plt.show()
