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

script_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.join(script_dir, "images_masked", "test")
out_dir = os.path.join(script_dir, "cellpose_test_mask")
os.makedirs(out_dir, exist_ok=True)

label_dict = {}

print(f"[INFO] 開始處理測試資料夾: {test_dir}")
for folder in os.listdir(test_dir):
    folder_path = os.path.join(test_dir, folder)
    if not os.path.isdir(folder_path):
        continue
    print(f"[INFO] 處理子資料夾: {folder_path}")
    out_folder = os.path.join(out_dir, folder)
    os.makedirs(out_folder, exist_ok=True)
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
        save_path = os.path.join(out_folder, img_name)
        cv2.imwrite(save_path, mask_img)
        coverage = (mask_img > 0).sum() / mask_img.size
        mask_count = masks.max()
        print(f"[INFO] {img_path}: 覆蓋率={coverage:.4f}, mask數={mask_count}")
        label_dict[img_path] = {
            "coverage_ratio": coverage,
            "mask_count": mask_count
        }

# 儲存 CSV
csv_path = os.path.join(out_dir, "test_coverage_labels.csv")
with open(csv_path, "w", encoding="utf-8", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "coverage_ratio", "mask_count"])
    for img_path, info in label_dict.items():
        writer.writerow([img_path, info["coverage_ratio"], info["mask_count"]])
print(f"[INFO] 已輸出統計結果: {csv_path}")
