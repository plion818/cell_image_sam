import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import LabelEncoder
import argparse

# 讀取資料
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(project_root, "output", "train_coverage_labels.csv")
df = pd.read_csv(csv_path)

X = df[["coverage_ratio", "mask_count"]].values
y = df["label"].values
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 參數選擇
parser = argparse.ArgumentParser(description='Random Forest Cross-Validation')
parser.add_argument('--method', type=str, default='loocv', choices=['loocv', 'kfold'], help='交叉驗證方法 (loocv 或 kfold)')
parser.add_argument('--k', type=int, default=5, help='K-Fold 的 K 值 (method=kfold 時有效)')
args = parser.parse_args()

# loocv : 直接執行
# kfold : python RF/rf_cv.py --method kfold --k 5
if args.method == 'loocv':
    cv = LeaveOneOut()
    print('使用 Leave-One-Out Cross-Validation')
else:
    cv = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)
    print(f'使用 {args.k}-Fold Cross-Validation')

rf = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(rf, X, y_encoded, cv=cv)
print(f'平均正確率: {scores.mean():.4f}')
print(f'每折/每筆預測結果: {scores}')
