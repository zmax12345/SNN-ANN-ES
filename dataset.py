import torch
import numpy as np
from torch.utils.data import Dataset
import os


class SpeckleVoxelDataset(Dataset):
    def __init__(self, config, is_train=True):
        self.roi = config['roi']
        self.window_size_us = config['window_size_ms'] * 1000
        self.stride_us = config['stride_ms'] * 1000
        self.crop_size = config['crop_size']
        self.is_train = is_train

        # 保持 Dropout 在 0.1 比较温和
        self.dropout_range = (0.0, 0.1)

        self.samples = []
        self.mmap_files = []

        print(f"🚀 初始化数据集 ({'训练' if is_train else '测试'})...")

        for velocity, file_list in config['files'].items():
            for csv_path in file_list:
                npy_path = csv_path.replace('.csv', '.npy')
                if not os.path.exists(npy_path):
                    raise FileNotFoundError(f"找不到 {npy_path}")

                events_mmap = np.load(npy_path, mmap_mode='r')
                self.mmap_files.append(events_mmap)
                cache_idx = len(self.mmap_files) - 1

                # 时间切分逻辑 (80% / 20%)
                times = events_mmap[:, 3]
                total_duration = times[-1] - times[0]
                t_start_global = times[0]
                split_time = t_start_global + total_duration * 0.8

                if is_train:
                    valid_range = (t_start_global, split_time)
                else:
                    valid_range = (split_time, times[-1])

                self.create_sliding_windows(cache_idx, events_mmap, velocity, valid_range)

        print(f"✅ 加载完成！共 {len(self.samples)} 个样本。")

    def create_sliding_windows(self, cache_idx, events_mmap, label, valid_time_range):
        times = events_mmap[:, 3]
        t_min, t_max = valid_time_range

        if t_max - t_min < self.window_size_us: return

        start_times = np.arange(t_min, t_max - self.window_size_us, self.stride_us)

        start_indices = np.searchsorted(times, start_times)
        end_indices = np.searchsorted(times, start_times + self.window_size_us)

        for i in range(len(start_times)):
            idx_start = start_indices[i]
            idx_end = end_indices[i]
            if idx_end > idx_start + 10:
                self.samples.append({
                    'cache_idx': cache_idx,
                    'idx_start': idx_start,
                    'idx_end': idx_end,
                    't_start': start_times[i],
                    'label': label
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        file_idx = info['cache_idx']

        slice_events = self.mmap_files[file_idx][info['idx_start']:info['idx_end']]
        events_data = np.array(slice_events)

        # --- Dropout (训练时) ---
        if self.is_train:
            p = np.random.uniform(*self.dropout_range)
            if p > 0:
                mask = np.random.rand(len(events_data)) > p
                events_data = events_data[mask]
                if len(events_data) == 0:
                    events_data = np.array(slice_events)

        events_tensor = torch.from_numpy(events_data).float()
        voxel_grid = self.spatial_crop_and_voxelize(events_tensor, info['t_start'])

        return voxel_grid, torch.tensor([info['label']], dtype=torch.float32), torch.tensor(0.0)

    def spatial_crop_and_voxelize(self, events, t_start):
        if torch.isnan(events).any(): events = torch.nan_to_num(events, nan=0.0)

        # --- 🌟 修正1: 极性清洗 (Training Phase) ---
        # 假设第3列 (index 2) 是 p
        p_raw = events[:, 2]
        valid_mask = (p_raw != 0)

        if not valid_mask.any():
            T = int(self.window_size_us / 1000)
            return torch.zeros((T, 2, self.crop_size, self.crop_size), dtype=torch.float32)

        events = events[valid_mask]

        # --- 裁剪逻辑 ---
        x_raw, y_raw = events[:, 0], events[:, 1]
        y_norm = y_raw - self.roi['row_start']

        roi_h, roi_w = self.roi['row_end'] - self.roi['row_start'], self.roi['col_end'] - self.roi['col_start']
        if self.is_train:
            x_start = np.random.randint(0, max(1, roi_w - self.crop_size))
            y_start = np.random.randint(0, max(1, roi_h - self.crop_size))
        else:
            x_start = (roi_w - self.crop_size) // 2
            y_start = (roi_h - self.crop_size) // 2

        mask = (x_raw >= x_start) & (x_raw < x_start + self.crop_size) & \
               (y_norm >= y_start) & (y_norm < y_start + self.crop_size)
        valid = events[mask]

        T = int(self.window_size_us / 1000)
        grid = torch.zeros((T, 2, self.crop_size, self.crop_size), dtype=torch.float32)

        if len(valid) > 0:
            xs = (valid[:, 0] - x_start).long()
            ys = (valid[:, 1] - self.roi['row_start'] - y_start).long()

            # --- 🌟 修正2: 极性映射 ---
            # p 只有 -1 和 1
            p_val = valid[:, 2]
            ps = torch.zeros_like(p_val).long()
            ps[p_val == 1] = 1
            ps[p_val == -1] = 0
            # 注意：这里不再需要 clamp，因为我们已经清洗了其他值

            ts = torch.clamp(((valid[:, 3] - t_start) / 1000).long(), 0, T - 1)

            grid[ts, ps, ys, xs] = 1.0

        return grid