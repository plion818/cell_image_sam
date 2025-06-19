import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# 讀取資料
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(project_root, "output", "train_coverage_labels.csv")
df = pd.read_csv(csv_path)

X = df[["coverage_ratio", "mask_count"]].values
y = df["label"].values
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Pipeline: 標準化 + LR
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000, random_state=42))
])

# 參數搜尋範圍（注意 param name 要加 lr__ 前綴）
param_grid = {
    'lr__C': [0.01, 0.1, 1, 10, 100],
    'lr__penalty': ['l2'],
    'lr__solver': ['lbfgs', 'liblinear']
}


# 交叉驗證方式（LOOCV，適合小樣本）
cv = LeaveOneOut()

gs = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1)
gs.fit(X, y_encoded)


print("最佳參數：", gs.best_params_)
print(f"最佳交叉驗證分數：{gs.best_score_:.4f}")


# 用最佳參數在全部資料上訓練最終 pipeline
final_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000, random_state=42,
        C=gs.best_params_['lr__C'],
        penalty=gs.best_params_['lr__penalty'],
        solver=gs.best_params_['lr__solver']))
])
final_pipe.fit(X, y_encoded)

# 儲存最終 pipeline 與 label encoder
joblib.dump(final_pipe, os.path.join(os.path.dirname(__file__), 'lr_model.pkl'))
joblib.dump(le, os.path.join(os.path.dirname(__file__), 'le.pkl'))
print('最佳模型與 label encoder 已儲存，可直接用於推論測試集！')
