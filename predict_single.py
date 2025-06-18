import torch
import joblib
import pandas as pd
from classifier import MLPClassifier, scaler, le

# 載入訓練好的模型
model = MLPClassifier(input_dim=2, num_classes=len(le.classes_))
model.load_state_dict(torch.load('mlp_classifier.pth', map_location='cpu'))
model.eval()

# 使用者輸入測試資料
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
