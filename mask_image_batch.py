import os
import cv2

# === 使用者參數設定 ===
INPUT_DIR = "AI_MSC_密度照片"           # 原始影像資料夾
OUTPUT_DIR = "images_masked/"       # 輸出遮蔽後影像的資料夾
MASK_LEFT_BOTTOM = (98, 1505, 470, 45)     # 遮左下角時間戳（黑底黃字）
MASK_RIGHT_BOTTOM = (1255, 1451, 260, 56)  # 遮右下角 200μm 10x 倍率標註

COLOR = (0, 0, 0)                    # 遮蔽用顏色（黑）

os.makedirs(OUTPUT_DIR, exist_ok=True)

def apply_mask(image, mask_area, ref_h, ref_w):
    x_offset, y_offset, w, h = mask_area
    x = x_offset if x_offset >= 0 else ref_w + x_offset
    y = y_offset if y_offset >= 0 else ref_h + y_offset
    cv2.rectangle(image, (x, y), (x + w, y + h), COLOR, -1)
    return image

# === 處理所有圖片（遞迴處理所有子資料夾） ===
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
        masked = apply_mask(masked, MASK_LEFT_BOTTOM, h, w)
        masked = apply_mask(masked, MASK_RIGHT_BOTTOM, h, w)
        # 保留子資料夾結構
        rel_dir = os.path.relpath(root, INPUT_DIR)
        out_dir = os.path.join(OUTPUT_DIR, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, fname)
        cv2.imwrite(out_path, masked)
        print(f"[完成] 儲存遮蔽圖像：{out_path}")
