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
# 讓路徑在 Windows/Colab/Linux 都能用
if "google.colab" in sys.modules:
    IMAGE_PATH = "images_masked/train/0-20_/3.tif"
else:
    IMAGE_PATH = os.path.join(os.getcwd(), "images_masked", "train", "0-20_", "3.tif")
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

# 建立黑色遮蔽區域 mask（假設黑色為 [0,0,0]）
black_mask = np.all(image == [0, 0, 0], axis=-1)
valid_area = ~black_mask
valid_pixel_count = np.count_nonzero(valid_area)
if valid_pixel_count == 0:
    coverage = 0.0
else:
    coverage = (final_mask & valid_area).sum() / valid_pixel_count
print(f"[INFO] coverage_ratio：{coverage:.2%}，mask_count：{len(masks)}")

# ======== 可視化疊圖 ========
plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.imshow(final_mask, alpha=0.4, cmap='Reds')
plt.title(f"覆蓋率: {coverage:.2%}  mask數: {len(masks)}")
plt.axis("off")
plt.show()
