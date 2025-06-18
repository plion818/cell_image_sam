import os
import torch
import pandas as pd
from MLP import MLPClassifier, scaler, le

# 設定路徑
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # cell_sam 目錄
csv_path = os.path.join(project_root, "output", "test_coverage_labels.csv")
model_path = os.path.join(script_dir, "mlp_classifier.pth")

# 讀取測試集
# 假設 test_coverage_labels.csv 有三欄：image_path, coverage_ratio, mask_count
# 若有 label 欄位則自動讀取

df = pd.read_csv(csv_path)

# 檢查有無 label 欄位
has_label = 'label' in df.columns

X_test = df[["coverage_ratio", "mask_count"]].values
if has_label:
    y_true = df["label"].values
    y_true_encoded = le.transform(y_true)
else:
    y_true = None
    y_true_encoded = None

# 標準化
X_test_scaled = scaler.transform(X_test)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# 載入模型
model = MLPClassifier(input_dim=2, num_classes=len(le.classes_))
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# 預測
with torch.no_grad():
    outputs = model(X_test_tensor)
    pred_indices = torch.argmax(outputs, dim=1).cpu().numpy()
    pred_labels = le.inverse_transform(pred_indices)

# 計算正確率
if has_label:
    accuracy = (pred_indices == y_true_encoded).mean()
    print(f"測試集正確率: {accuracy:.4f}")
else:
    print("[警告] 測試集無 label 欄位，無法計算正確率")

# 輸出表格
print("\n測試集預測結果：")
header = ["coverage_ratio", "mask_count"]
if has_label:
    header += ["label"]
header += ["pred_label"]
print("\t".join(header))
for i in range(len(X_test)):
    row = [f"{X_test[i,0]:.4f}", f"{X_test[i,1]:.0f}"]
    if has_label:
        row.append(str(y_true[i]))
    row.append(str(pred_labels[i]))
    print("\t".join(row))
