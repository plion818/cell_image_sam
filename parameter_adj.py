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

# 選擇模型：'sam' 或 'cellpose'
MODEL_SELECT = os.environ.get('SEG_MODEL', 'cellpose')  # 預設 cellpose
print(f"[INFO] 選擇模型: {MODEL_SELECT}")

mask_generator = None
if MODEL_SELECT == 'sam':
    # 若權重不存在則自動下載
    if not os.path.exists(SAM_CHECKPOINT):
        import urllib.request
        print(f"[INFO] SAM 權重不存在，自動下載: {SAM_CHECKPOINT_URL}")
        urllib.request.urlretrieve(SAM_CHECKPOINT_URL, SAM_CHECKPOINT)
        print("[INFO] SAM 權重下載完成")
    MODEL_TYPE = "vit_h"
    DEVICE = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
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

# cellpose 參數設置
CELLP_PROB_THRESHOLD = -4   # 提高微弱細胞區域的檢出率，預設值=0.0
FLOW_THRESHOLD = 0.2        # 提高細胞邊界的檢出率，預設值=0.4

# ======== 圖片路徑基礎設定 ========
script_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(script_dir, "images_masked", "train")

# ======== 參數組合與圖片清單 ========
CELLP_PROB_THRESHOLD_LIST = [-4]
FLOW_THRESHOLD_LIST = [0.8, 0.9]
image_path_list = [
    os.path.join(train_dir, "0-20_", "1.tif"),
    os.path.join(train_dir, "20-40_", "1.tif"),
    os.path.join(train_dir, "40-60_", "1.tif"),
    os.path.join(train_dir, "60-80_", "1.tif"),
    os.path.join(train_dir, "80-100_", "1.tif"),
]

# ======== 批次測試所有組合 ========
for cellprob in CELLP_PROB_THRESHOLD_LIST:
    for flow in FLOW_THRESHOLD_LIST:
        for img_path in image_path_list:
            print(f"[INFO] 處理: cellprob={cellprob}, flow={flow}, img={img_path}")
            image = np.array(Image.open(img_path).convert("RGB"))
            if MODEL_SELECT == 'cellpose' and HAS_CELLPOSE:
                if hasattr(cellpose_models, 'CellposeModel'):
                    cp_model = cellpose_models.CellposeModel(model_type='cyto')
                else:
                    cp_model = cellpose_models.Cellpose(model_type='cyto')
                cp_result = cp_model.eval(
                    image,
                    diameter=None,
                    cellprob_threshold=cellprob,
                    flow_threshold=flow
                )
                if len(cp_result) == 4:
                    masks, flows, styles, diams = cp_result
                else:
                    masks, flows, diams = cp_result
                    styles = None
                final_mask = (masks > 0).astype(np.uint8)
                mask_count = masks.max()
            else:
                if mask_generator is None:
                    raise RuntimeError("[錯誤] 未初始化 SAM，請確認模型選擇與權重下載！")
                masks = mask_generator.generate(image)
                final_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                for m in masks:
                    final_mask |= m["segmentation"].astype(np.uint8)
                mask_count = len(masks)
            coverage = final_mask.sum() / final_mask.size
            print(f"[INFO] coverage_ratio：{coverage:.2%}，mask_count：{mask_count}")

            # 產生自訂圖片名稱
            parent_folder = os.path.basename(os.path.dirname(img_path))
            file_name = os.path.basename(img_path)
            if parent_folder.endswith('_'):
                percent_name = parent_folder.replace('_', '%_')
            else:
                percent_name = parent_folder
            custom_img_name = f"{percent_name}{file_name}"
            plt.figure(figsize=(6, 6))
            plt.imshow(image)
            plt.imshow(final_mask, alpha=0.4, cmap='Reds')
            plt.title(f"cellprob={cellprob}, flow={flow}\n覆蓋率: {coverage:.2%}  mask數: {mask_count}\n{custom_img_name}")
            plt.axis("off")
            plt.show()
