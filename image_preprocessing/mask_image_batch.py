import os
import cv2
import numpy as np

# 以 cell_sam 專案根目錄為基準組合路徑
# script_dir：永遠是這支 .py 檔案本身所在的資料夾。
# project_root：是 script_dir 的上一層資料夾。
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
INPUT_DIR = os.path.join(project_root, "AI_MSC_密度照片")           # 原始影像資料夾
OUTPUT_DIR = os.path.join(project_root, "images_masked")       # 輸出遮蔽後影像的資料夾
MASK_LEFT_BOTTOM = (92, 1508, 500, 48)     # 遮左下角時間戳（黑底黃字）
MASK_RIGHT_BOTTOM = (1255, 1451, 260, 56)  # 遮右下角 200μm 10x 倍率標註

COLOR = (0, 0, 0)                    # 遮蔽用顏色（黑）

os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_background_average(image):
    mask = np.any(image != [0, 0, 0], axis=-1)
    background_pixels = image[mask]
    if len(background_pixels) > 0:
        return np.mean(background_pixels, axis=0).astype(np.uint8)
    else:
        return np.array([127, 127, 127], dtype=np.uint8)

def fill_mask_area_with_background_average(image, x, y, w, h, bg_color=None):
    if bg_color is None:
        bg_color = compute_background_average(image)
    image[y:y+h, x:x+w] = bg_color
    return image

def apply_mask(image, mask_area, ref_h, ref_w, bg_color=None):
    x_offset, y_offset, w, h = mask_area
    x = x_offset if x_offset >= 0 else ref_w + x_offset
    y = y_offset if y_offset >= 0 else ref_h + y_offset
    return fill_mask_area_with_background_average(image, x, y, w, h, bg_color)

# === 處理所有圖片（遞迴處理所有子資料夾） ===
bg_color = None
for root, dirs, files in os.walk(INPUT_DIR):
    for fname in files:
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
            continue
        in_path = os.path.join(root, fname)
        image = cv2.imread(in_path)
        if image is None:
            print(f"[跳過] 無法讀取：{in_path}")
            continue
        h, w = image.shape[:2]
        masked = image.copy()
        # 計算背景平均色
        bg_color = compute_background_average(masked)
        masked = apply_mask(masked, MASK_LEFT_BOTTOM, h, w, bg_color)
        masked = apply_mask(masked, MASK_RIGHT_BOTTOM, h, w, bg_color)
        # 保留子資料夾結構
        rel_dir = os.path.relpath(root, INPUT_DIR)
        out_dir = os.path.join(OUTPUT_DIR, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, fname)
        cv2.imwrite(out_path, masked)
        print(f"[完成] 儲存遮蔽圖像：{out_path}")
