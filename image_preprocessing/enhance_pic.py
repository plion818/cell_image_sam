import cv2
import os
from matplotlib import pyplot as plt

# 以 cell_sam 專案根目錄為基準
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# 指定測試圖片路徑
img_path = os.path.join(project_root, "images_masked", "train", "60-80_", "5.tif")

# 讀取灰階影像
img = cv2.imread(img_path, 0)
if img is None:
    raise FileNotFoundError(f"找不到圖片: {img_path}")

# CLAHE 對比增強
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(img)

# 顯示原圖與增強後影像
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title('原圖')
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.subplot(1,2,2)
plt.title('CLAHE 增強')
plt.imshow(enhanced, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()

# 輸出增強後影像到 output 資料夾
output_dir = os.path.join(project_root, "output")
os.makedirs(output_dir, exist_ok=True)
out_path = os.path.join(output_dir, "5_clahe.tif")
cv2.imwrite(out_path, enhanced)
print(f"[INFO] 增強後影像已儲存: {out_path}")
