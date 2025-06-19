import os
import pandas as pd
import joblib
from xgboost import XGBClassifier
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
xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb.fit(X, y_encoded)

# 儲存模型、label encoder
joblib.dump(xgb, os.path.join(os.path.dirname(__file__), 'xgb_model.pkl'))
joblib.dump(le, os.path.join(os.path.dirname(__file__), 'le.pkl'))
print('模型、label encoder 已儲存')
