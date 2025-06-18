import torch
import joblib
import pandas as pd
from MLP import MLPClassifier, scaler, le

# 載入訓練好的模型（以 cell_sam 專案根目錄為基準）
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
# 直接在 MLP 資料夾下尋找模型
model_path = os.path.join(script_dir, "mlp_classifier.pth")
model = MLPClassifier(input_dim=2, num_classes=len(le.classes_))
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# 使用者輸入測試資料

while True:
    print("請輸入 coverage_ratio 與 mask_count：")
    coverage_ratio = float(input("coverage_ratio: "))
    mask_count = float(input("mask_count: "))

    # 標準化
    X_test = scaler.transform([[coverage_ratio, mask_count]])
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # 預測
    with torch.no_grad():
        output = model(X_test_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        pred_label = le.inverse_transform([pred_idx])[0]

    print(f"模型預測分類結果：{pred_label}")

    cont = input("是否有下一筆輸入？(y/n): ").strip().lower()
    if cont != 'y':
        break
