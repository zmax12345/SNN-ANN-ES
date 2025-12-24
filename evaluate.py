import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from sklearn.metrics import r2_score

# 引入你的模块
from dataset import SpeckleVoxelDataset
from model import SnnRegressor

# ================= 配置区域 (需与 train.py 严格一致) =================
# 只保留 0.2 - 2.2 mm/s 的清洗数据
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
    },
    'roi': {'row_start': 400, 'row_end': 499, 'col_start': 0, 'col_end': 1280},
    'window_size_ms': 25,
    'stride_ms': 25,  # 评估时步长设大一点，避免重复计算，看整体趋势
    'crop_size': 64
}

BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 指向你刚才跑出 0.3201 的那个最佳模型
MODEL_PATH = "/data/zm/12.22/03_snn_dropout_0.3091.pth"


def plot_results(preds, labels):
    """
    绘制回归分析图和误差分布图
    """
    plt.figure(figsize=(15, 6))

    # --- 子图 1: 回归分析 ---
    plt.subplot(1, 2, 1)

    # 散点图
    plt.scatter(labels, preds, alpha=0.4, s=15, color='#4169E1', label='Test Samples')

    # 理想线 (y=x)
    min_val = min(labels.min(), preds.min())
    max_val = max(labels.max(), preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')

    # 计算指标
    rmse = np.sqrt(np.mean((preds - labels) ** 2))
    r2 = r2_score(labels, preds)

    plt.title(f'Regression Analysis\nRMSE={rmse:.4f}, R²={r2:.4f}')
    plt.xlabel('Ground Truth Velocity (mm/s)')
    plt.ylabel('Predicted Velocity (mm/s)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # --- 子图 2: 误差箱线图 ---
    plt.subplot(1, 2, 2)

    # 按真实流速分组计算误差
    unique_labels = np.unique(labels)
    errors_by_label = []
    labels_str = []

    for label in unique_labels:
        mask = (labels == label)
        # 误差 = 预测 - 真实
        errors = preds[mask] - labels[mask]
        errors_by_label.append(errors)
        labels_str.append(f"{label:.2f}")

    plt.boxplot(errors_by_label, labels=labels_str, patch_artist=False, showfliers=False)
    plt.axhline(0, color='r', linestyle='--', linewidth=1)

    plt.title('Error Distribution by Velocity')
    plt.xlabel('Velocity Group (mm/s)')
    plt.ylabel('Prediction Error (mm/s)')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('evaluation_result_mlp.png', dpi=300)
    print("✅ 评估图表已保存为 evaluation_result_mlp.png")
    plt.show()


def main():
    print(f"Using device: {DEVICE}")

    # 1. 加载测试集
    # is_train=False 会自动选择每个文件后 20% 的时间段
    # 并且 dataset 内部会自动关闭 Dropout
    test_dataset = SpeckleVoxelDataset(FULL_CONFIG, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"Test samples (Last 20% of time): {len(test_dataset)}")

    # 2. 加载模型
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        return

    model = SnnRegressor(crop_size=FULL_CONFIG['crop_size']).to(DEVICE)

    # 加载权重
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        print("✅ 模型权重加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    model.eval()

    # 3. 推理
    all_preds = []
    all_labels = []

    print("Running evaluation...")
    with torch.no_grad():
        # dataset 现在返回 3 个值: (voxel, label, density)
        # 我们这里只需要前两个
        for inputs, labels, _ in tqdm(test_loader):
            inputs = inputs.to(DEVICE)

            # 预测
            outputs = model(inputs)

            # 收集结果
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())

    # 转换格式
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 4. 计算指标
    rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
    r2 = r2_score(all_labels, all_preds)

    print("\n" + "=" * 30)
    print(f"📊 Evaluation Results:")
    print(f"   RMSE: {rmse:.4f} mm/s")
    print(f"   R²  : {r2:.4f}")
    print("=" * 30)

    # 5. 绘图
    plot_results(all_preds, all_labels)


if __name__ == "__main__":
    main()