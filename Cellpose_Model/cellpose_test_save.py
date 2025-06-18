import os
import numpy as np
from PIL import Image
import csv
import cv2

try:
    from cellpose import models as cellpose_models
    print("[INFO] 匯入 cellpose 成功")
    HAS_CELLPOSE = True
except Exception as e:
    print("[INFO] 匯入 cellpose 失敗：", e)
    HAS_CELLPOSE = False

CELLP_PROB_THRESHOLD = -4
FLOW_THRESHOLD = 1.0

if not HAS_CELLPOSE:
    raise ImportError("cellpose 未安裝，請先安裝 cellpose 套件！")

# 支援新版與舊版 cellpose
if hasattr(cellpose_models, 'CellposeModel'):
    cp_model = cellpose_models.CellposeModel(model_type='cyto')
else:
    cp_model = cellpose_models.Cellpose(model_type='cyto')


# 以 cell_sam 專案根目錄為基準
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
test_dir = os.path.join(project_root, "images_masked", "test")
output_dir = os.path.join(project_root, "output")
os.makedirs(output_dir, exist_ok=True)

label_dict = {}

print(f"[INFO] 開始處理測試資料夾: {test_dir}")
for folder in os.listdir(test_dir):
    folder_path = os.path.join(test_dir, folder)
    if not os.path.isdir(folder_path):
        continue
    print(f"[INFO] 處理子資料夾: {folder_path}")
    # 不再建立分割圖片資料夾
    for img_name in os.listdir(folder_path):
        if not img_name.lower().endswith(('.tif', '.png', '.jpg', '.jpeg')):
            continue
        img_path = os.path.join(folder_path, img_name)
        print(f"[INFO] 處理圖片: {img_path}")
        image = np.array(Image.open(img_path).convert("RGB"))
        print(f"[DEBUG] 讀取圖片 shape: {image.shape}")
        cp_result = cp_model.eval(
            image,
            diameter=None,
            cellprob_threshold=CELLP_PROB_THRESHOLD,
            flow_threshold=FLOW_THRESHOLD
        )
        if len(cp_result) == 4:
            masks, flows, styles, diams = cp_result
        else:
            masks, flows, diams = cp_result
            styles = None
        mask_img = (masks > 0).astype(np.uint8) * 255
        coverage = (mask_img > 0).sum() / mask_img.size
        mask_count = masks.max()
        print(f"[INFO] {img_path}: 覆蓋率={coverage:.4f}, mask數={mask_count}")
        label_dict[img_path] = {
            "coverage_ratio": coverage,
            "mask_count": mask_count
        }

# 儲存 CSV
csv_path = os.path.join(output_dir, "test_coverage_labels.csv")
with open(csv_path, "w", encoding="utf-8", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "coverage_ratio", "mask_count"])
    for img_path, info in label_dict.items():
        writer.writerow([img_path, info["coverage_ratio"], info["mask_count"]])
print(f"[INFO] 已輸出統計結果: {csv_path}")
