import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# 讀取資料
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(project_root, "output", "train_coverage_labels.csv")
df = pd.read_csv(csv_path)

X = df[["coverage_ratio", "mask_count"]].values
y = df["label"].values

# 標籤編碼
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 直接用全部訓練集訓練
lr = LogisticRegression(max_iter=200, random_state=42)
lr.fit(X, y_encoded)

# 儲存模型、label encoder
joblib.dump(lr, os.path.join(os.path.dirname(__file__), 'lr_model.pkl'))
joblib.dump(le, os.path.join(os.path.dirname(__file__), 'le.pkl'))
print('模型、label encoder 已儲存')
