import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
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
parser = argparse.ArgumentParser(description='XGBoost 交叉驗證')
parser.add_argument('--method', type=str, default='loocv', choices=['loocv', 'kfold'], help='交叉驗證方法 (loocv 或 kfold)')
parser.add_argument('--k', type=int, default=5, help='K-Fold 的 K 值 (method=kfold 時有效)')
args = parser.parse_args()

if args.method == 'loocv':
    cv = LeaveOneOut()
    print('使用 Leave-One-Out Cross-Validation')
else:
    cv = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)
    print(f'使用 {args.k}-Fold Cross-Validation')

 # 移除 use_label_encoder 參數，避免警告
xgb = XGBClassifier(n_estimators=100, eval_metric='mlogloss', random_state=42)
scores = cross_val_score(xgb, X, y_encoded, cv=cv)
print(f'平均正確率: {scores.mean():.4f}')
print(f'每折/每筆預測結果: {scores}')
