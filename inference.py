import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from model import SnnRegressor

# ================= 配置区域 =================
# 模型路径
MODEL_PATH = "/data/zm/12.22/02_snn_dropout_0.3203.pth"
# 待测试的 CSV 文件路径 (可以是任何一个未见过的新文件)
TEST_FILE = "/data/zm/12.24_data/0.5mm_clip.csv"

# 这里的 ROI 必须和训练时完全一致！
ROI = {'row_start': 400, 'row_end': 499, 'col_start': 0, 'col_end': 1280}
CROP_SIZE = 64
WINDOW_SIZE_MS = 25
STRIDE_MS = 25  # 推理步长，越小曲线越密

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_and_voxelize(df_chunk, t_start):
    """
    将一小段 DataFrame 数据实时转换为 SNN 输入 Tensor
    """
    # 提取数据 (假设 pandas 读进来列名是 col, row, i, p, t)
    # 根据你最新的 check 结果，你的 csv 是 5 列
    x = df_chunk.iloc[:, 0].values
    y = df_chunk.iloc[:, 1].values
    # i = df_chunk.iloc[:, 2].values # intensity 暂时不用
    p = df_chunk.iloc[:, 3].values  # p
    t = df_chunk.iloc[:, 4].values  # t

    # 归一化 Y
    y_norm = y - ROI['row_start']

    # 只有在 ROI 内的事件才有效
    mask = (y_norm >= 0) & (y_norm < (ROI['row_end'] - ROI['row_start']))

    if not mask.any(): return None

    x = x[mask]
    y_norm = y_norm[mask]
    p = p[mask]
    t = t[mask]

    # 中心裁剪 (Center Crop) - 推理时我们通常看中心
    roi_h = ROI['row_end'] - ROI['row_start']
    roi_w = ROI['col_end'] - ROI['col_start']

    x_start = (roi_w - CROP_SIZE) // 2
    y_start = (roi_h - CROP_SIZE) // 2

    # 二次筛选 (Crop 内)
    crop_mask = (x >= x_start) & (x < x_start + CROP_SIZE) & \
                (y_norm >= y_start) & (y_norm < y_start + CROP_SIZE)

    if not crop_mask.any(): return None

    x = x[crop_mask] - x_start
    y_norm = y_norm[crop_mask] - y_start
    p = p[crop_mask]
    t = t[crop_mask]

    # 构建 Voxel Grid
    T_bins = int(WINDOW_SIZE_MS / 1)  # 1ms per bin -> T=25
    grid = torch.zeros((T_bins, 2, CROP_SIZE, CROP_SIZE), dtype=torch.float32)

    # 极性钳位
    ps = np.clip(p, 0, 1).astype(int)
    # 时间索引
    t_idx = ((t - t_start) / 1000).astype(int)
    t_idx = np.clip(t_idx, 0, T_bins - 1)
    # 坐标索引
    xs = np.clip(x, 0, CROP_SIZE - 1).astype(int)
    ys = np.clip(y_norm, 0, CROP_SIZE - 1).astype(int)

    # 填充 Tensor
    grid[t_idx, ps, ys, xs] = 1.0

    return grid


def predict_single_file(file_path):
    print(f"🚀 Loading model from {MODEL_PATH}...")
    model = SnnRegressor(crop_size=CROP_SIZE).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    print(f"📂 Reading file: {file_path}")

    try:
        # 先读第一行获取起始时间 (假设第5列是时间)
        df_head = pd.read_csv(file_path, header=None, nrows=1)
        t_global_start = df_head.iloc[0, 4]

        # 读数据 (为了演示，只读前 300万行，约 1-2秒)
        # names 参数确保列对齐
        df = pd.read_csv(file_path, header=None, nrows=30000000,
                         names=['col', 'row', 'i', 'p', 't'])
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 滑动窗口预测
    preds = []
    timestamps = []

    window_us = WINDOW_SIZE_MS * 1000
    stride_us = STRIDE_MS * 1000

    t_min = df['t'].min()
    t_max = df['t'].max()
    curr_t = t_min

    print("Running inference...")

    while curr_t + window_us < t_max:
        # 获取窗口内数据
        mask = (df['t'] >= curr_t) & (df['t'] < curr_t + window_us)
        df_chunk = df[mask]

        # 只有当窗口内有足够事件才预测 (比如 >100)
        if len(df_chunk) > 10:
            grid = preprocess_and_voxelize(df_chunk, curr_t)

            if grid is not None:
                # 增加 Batch 维度 [1, T, 2, H, W]
                input_tensor = grid.unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    pred_v = model(input_tensor).item()
                    preds.append(pred_v)
                    timestamps.append((curr_t - t_global_start) / 1e6)  # 秒

        curr_t += stride_us

    # 绘图
    if len(preds) == 0:
        print("❌ 未生成任何预测结果，可能是数据不足或ROI设置错误")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, preds, label='Predicted Velocity', alpha=0.7)

    # 画平均线
    mean_val = np.mean(preds)
    plt.axhline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.4f}')

    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (mm/s)')
    plt.title(f'Inference: {os.path.basename(file_path)}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_name = 'inference_result.png'
    plt.savefig(out_name)
    print(f"✅ 推理完成! 平均预测流速: {mean_val:.4f} mm/s")
    print(f"   结果已保存为 {out_name}")


if __name__ == "__main__":
    predict_single_file(TEST_FILE)