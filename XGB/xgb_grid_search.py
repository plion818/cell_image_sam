import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.preprocessing import LabelEncoder
import joblib

# 讀取資料
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(project_root, "output", "train_coverage_labels.csv")
df = pd.read_csv(csv_path)

X = df[["coverage_ratio", "mask_count"]].values
y = df["label"].values
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 參數搜尋範圍
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

# 交叉驗證方式（LOOCV，適合小樣本）
cv = LeaveOneOut()

# 移除 use_label_encoder 參數，避免警告
xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)
gs = GridSearchCV(xgb, param_grid, cv=cv, n_jobs=-1)
gs.fit(X, y_encoded)

print("最佳參數：", gs.best_params_)
print(f"最佳交叉驗證分數：{gs.best_score_:.4f}")

# 用最佳參數在全部資料上訓練最終模型
best_xgb = XGBClassifier(**gs.best_params_, eval_metric='mlogloss', random_state=42)
best_xgb.fit(X, y_encoded)

# 儲存最終模型與 label encoder
joblib.dump(best_xgb, os.path.join(os.path.dirname(__file__), 'xgb_model.pkl'))
joblib.dump(le, os.path.join(os.path.dirname(__file__), 'le.pkl'))
print('最佳模型與 label encoder 已儲存，可直接用於推論測試集！')
