import os
import joblib
import numpy as np

# 載入 pipeline 模型、label encoder
script_dir = os.path.dirname(os.path.abspath(__file__))
lr = joblib.load(os.path.join(script_dir, 'lr_model.pkl'))  # pipeline
le = joblib.load(os.path.join(script_dir, 'le.pkl'))

while True:
    print("請輸入 coverage_ratio 與 mask_count：")
    coverage_ratio = float(input("coverage_ratio: "))
    mask_count = float(input("mask_count: "))
    X_test = np.array([[coverage_ratio, mask_count]])
    probs = lr.predict_proba(X_test)[0]
    pred_idx = np.argmax(probs)
    pred_label = le.inverse_transform([pred_idx])[0]
    print(f"模型預測分類結果：{pred_label}")
    print("各類別機率：")
    for label, prob in zip(le.classes_, probs):
        print(f"{label}: {prob:.4f}")
    cont = input("是否有下一筆輸入？(y/n): ").strip().lower()
    if cont != 'y':
        break
