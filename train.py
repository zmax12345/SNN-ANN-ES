import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os

from dataset import SpeckleVoxelDataset
from model import SnnRegressor

# ================= 配置：清洗后的数据 =================
FULL_CONFIG = {
    'files': {
        0.2: [r'/data/zm/12.23data/0.2mm_clip.csv'],
        0.5: [r'/data/zm/12.23data/0.5mm_clip.csv'],
        0.8: [r'/data/zm/12.23data/0.8mm_clip.csv'],
        1.0: [r'/data/zm/12.23data/1.0mm_clip.csv'],
        1.2: [r'/data/zm/12.23data/1.2mm_clip.csv'],
        1.5: [r'/data/zm/12.23data/1.5mm_clip.csv'],
        1.8: [r'/data/zm/12.23data/1.8mm_clip.csv'],
        2.0: [r'/data/zm/12.23data/2.0mm_clip.csv'],
        2.2: [r'/data/zm/12.23data/2.2mm_clip.csv'],
        # ❌ 剔除 2.5mm (振动污染)
    },
    'roi': {'row_start': 400, 'row_end': 499, 'col_start': 0, 'col_end': 1280},
    'window_size_ms': 25,
    'stride_ms': 5,  # 步长小一点，增加样本数
    'crop_size': 64
}

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "/data/zm/12.22/best_snn_dropout.pth"


# ================= 简单的 MSE Loss (Physics Implicit) =================
# 既然加了 Dropout，我们不需要显式的 Density Loss (因为 Density 在变)
# 我们相信 SNN 会自己学到抗干扰的特征
class SimpleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, v_pred, v_label):
        return self.mse(v_pred, v_label)


# ... (train_one_epoch 和 evaluate 函数保持标准写法即可) ...
# 注意：evaluate 时，dataset 会自动关闭 dropout，
# 这样我们评估的是"全量数据"下的准确性，这是符合逻辑的。

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc="Training", unit="batch")

    # Dataset 返回 3 个值，但我们只用前两个 (Input, Label)
    # 那个 density 只是备用，这里暂不进入 Loss
    for inputs, labels, _ in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return running_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels, _ in tqdm(loader, desc="Eval", unit="batch"):
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy()
            all_preds.append(outputs)
            all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
    return rmse


def main():
    if not os.path.exists(os.path.dirname(SAVE_PATH)): os.makedirs(os.path.dirname(SAVE_PATH))
    print(f"Using device: {DEVICE}")

    # 加载数据集 (自动进行时间切分 80/20)
    train_dataset = SpeckleVoxelDataset(FULL_CONFIG, is_train=True)
    test_dataset = SpeckleVoxelDataset(FULL_CONFIG, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = SnnRegressor(crop_size=FULL_CONFIG['crop_size']).to(DEVICE)
    criterion = SimpleLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    best_rmse = float('inf')

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_rmse = evaluate(model, test_loader, DEVICE)
        scheduler.step()

        print(f"Train Loss: {train_loss:.5f} | Test RMSE: {val_rmse:.4f}")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"✅ Model Saved! (RMSE: {best_rmse:.4f})")


if __name__ == "__main__":
    main()