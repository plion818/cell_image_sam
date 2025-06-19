import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import numpy as np

# 設定隨機種子以確保可重現性
import random
import numpy as np
import torch
random_seed = 1234
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# 讀取 CSV，路徑以 cell_sam 專案根目錄為基準
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # cell_sam 目錄
csv_path = os.path.join(project_root, "output", "train_coverage_labels.csv")
df = pd.read_csv(csv_path)

# 特徵與標籤
X = df[["coverage_ratio", "mask_count"]].values
y = df["label"].values

# 標籤轉為整數編碼
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 確保訓練集每個label至少3個樣本，並以8:2比例分割。
label_counts = Counter(y_encoded)
min_train_per_class = 3

# 先將每個類別分開
X_train_list, y_train_list, X_val_list, y_val_list = [], [], [], []
for label in np.unique(y_encoded):
    idx = np.where(y_encoded == label)[0]
    n_total = len(idx)
    n_train = max(min_train_per_class, int(round(n_total * 0.8)))
    n_train = min(n_train, n_total-1) if n_total > min_train_per_class else n_total  # 至少留1個給val
    np.random.shuffle(idx)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]
    X_train_list.append(X[train_idx])
    y_train_list.append(y_encoded[train_idx])
    X_val_list.append(X[val_idx])
    y_val_list.append(y_encoded[val_idx])
X_train = np.concatenate(X_train_list)
y_train = np.concatenate(y_train_list)
X_val = np.concatenate(X_val_list)
y_val = np.concatenate(y_val_list)

# 打亂訓練集與驗證集
perm_train = np.random.permutation(len(X_train))
X_train, y_train = X_train[perm_train], y_train[perm_train]
perm_val = np.random.permutation(len(X_val))
X_val, y_val = X_val[perm_val], y_val[perm_val]


# 特徵標準化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

class CoverageDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = CoverageDataset(X_train, y_train)
val_dataset = CoverageDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# 顯示類別對應
# print("label 對應：", dict(zip(le.classes_, le.transform(le.classes_))))
# print(f"訓練集數量: {len(train_dataset)}，驗證集數量: {len(val_dataset)}")

# 顯示訓練集與驗證集每個label的樣本數
# train_label_counts = Counter(y_train)
# val_label_counts = Counter(y_val)
# print("訓練集各類別樣本數：")
# for label, idx in zip(le.classes_, range(len(le.classes_))):
#     print(f"  {label}: {train_label_counts[idx]}")
# print("驗證集各類別樣本數：")
# for label, idx in zip(le.classes_, range(len(le.classes_))):
#     print(f"  {label}: {val_label_counts[idx]}")

# 定義簡單 MLP 分類模型
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )
    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    # 取得類別數
    num_classes = len(set(y_train))
    model = MLPClassifier(input_dim=2, num_classes=num_classes)

    # 設定 loss 與 optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Early stopping 參數
    patience = 20
    best_val_acc = 0
    patience_counter = 0
    best_model_state = None

    # 訓練模型
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        # 驗證
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
        val_acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs} - train_loss: {avg_loss:.4f} - val_acc: {val_acc:.4f}")
        # Early stopping 檢查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[EarlyStopping] val_acc 已 {patience} 次未提升，提前停止訓練。")
                break
    # 回復最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"[INFO] 已回復最佳模型 (val_acc={best_val_acc:.4f})")

    # 訓練結束後儲存模型（儲存在 MLP 資料夾下）
    model_save_path = os.path.join(script_dir, 'mlp_classifier.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f'模型已儲存為 {model_save_path}')

    # 儲存 scaler
    scaler_save_path = os.path.join(script_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_save_path)
    print(f'scaler 已儲存為 {scaler_save_path}')
