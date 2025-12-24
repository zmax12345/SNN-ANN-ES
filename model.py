import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, layer, surrogate


class SnnRegressor(nn.Module):
    def __init__(self, crop_size=64):
        """
        轻量级 SNN-ANN 混合回归网络
        输入: [Batch, T, 2, crop_size, crop_size]
        输出: [Batch, 1] (流速)
        """
        super().__init__()

        # --- 1. SNN Encoder (特征提取器) ---
        # 这一部分负责从含噪的时空数据中提取“运动特征”

        self.encoder = nn.Sequential(
            # Layer 1: 基础特征提取
            # 输入通道 2 (ON/OFF), 输出 32
            layer.Conv2d(2, 32, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(32),  # BN层有助于抵抗不同光照带来的整体偏移
            neuron.LIFNode(surrogate_function=surrogate.ATan()),

            # Pooling 1: 降噪关键步骤
            # 64x64 -> 32x32。这会融合 2x2 区域内的信息，平滑掉单个坏点
            layer.MaxPool2d(2, 2),

            # Layer 2: 提取更复杂的时空纹理
            layer.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(64),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),

            # Pooling 2: 再次降噪与压缩
            # 32x32 -> 16x16
            layer.MaxPool2d(2, 2),

            # Layer 3: 高层语义
            layer.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(128),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),

            # 最终尺寸: 128通道, 16x16 图像
        )

        # --- 2. 升级版 ANN Decoder (MLP) ---
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # 展平后的维度
        self.flat_dim = 128

        # 修改：使用更深的 MLP 来拟合非线性关系
        self.decoder = nn.Sequential(
            nn.Linear(self.flat_dim, 256),  # 升维，增加特征组合能力
            nn.GELU(),  # 使用 GELU 激活函数，比 ReLU 更平滑
            nn.Dropout(0.1),  # 防止过拟合

            nn.Linear(256, 128),  # 隐层
            nn.GELU(),

            nn.Linear(128, 1)  # 输出层
        )

    def forward(self, x_seq):
        x_seq = x_seq.transpose(0, 1)  # [T, B, C, H, W]

        # SNN 提取时空特征
        y_seq = functional.multi_step_forward(x_seq, self.encoder)

        # 时域聚合
        y_mean = y_seq.mean(0)

        # 空域聚合
        y_gap = self.gap(y_mean)
        y_flat = y_gap.view(y_gap.shape[0], -1)

        # 解码
        velocity = self.decoder(y_flat)

        functional.reset_net(self.encoder)
        return velocity