import os
import pandas as pd
import joblib
import numpy as np

# 載入模型、label encoder
script_dir = os.path.dirname(os.path.abspath(__file__))
rf = joblib.load(os.path.join(script_dir, 'rf_model.pkl'))
le = joblib.load(os.path.join(script_dir, 'le.pkl'))

# 讀取測試集
project_root = os.path.dirname(script_dir)
csv_path = os.path.join(project_root, "output", "test_coverage_labels.csv")
df = pd.read_csv(csv_path)

has_label = 'label' in df.columns
X_test = df[["coverage_ratio", "mask_count"]].values
if has_label:
    y_true = df["label"].values
    y_true_encoded = le.transform(y_true)
else:
    y_true = None
    y_true_encoded = None

probs = rf.predict_proba(X_test)
pred_indices = np.argmax(probs, axis=1)
pred_labels = le.inverse_transform(pred_indices)

if has_label:
    accuracy = (pred_indices == y_true_encoded).mean()
    print(f"測試集正確率: {accuracy:.4f}")
else:
    print("[警告] 測試集無 label 欄位，無法計算正確率")

print("\n測試集預測結果：")
header = ["coverage_ratio", "mask_count"]
if has_label:
    header += ["label"]
header += ["pred_label"] + [f"prob_{label}" for label in le.classes_]
print("\t".join(header))
for i in range(len(X_test)):
    row = [f"{X_test[i,0]:.4f}", f"{X_test[i,1]:.0f}"]
    if has_label:
        row.append(str(y_true[i]))
    row.append(str(pred_labels[i]))
    row += [f"{p:.4f}" for p in probs[i]]
    print("\t".join(row))
