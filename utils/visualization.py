"""
可视化工具
用于事件数据、体素网格和重建结果的可视化
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from typing import Optional, List, Tuple, Union, Dict
import cv2
import os
from pathlib import Path


def visualize_events(
    events: Union[torch.Tensor, np.ndarray],
    height: int,
    width: int,
    time_window: Optional[Tuple[float, float]] = None,
    accumulate_time: Optional[float] = None,
    color_scheme: str = 'red_blue',
    background: str = 'black',
    dot_size: float = 1.0,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    可视化事件流
    
    Args:
        events: (N, 4) 事件数组 [x, y, t, p]
        height: 图像高度
        width: 图像宽度
        time_window: 时间窗口 (t_start, t_end)
        accumulate_time: 累积时间窗口大小
        color_scheme: 颜色方案 ('red_blue', 'green_red', 'grayscale')
        background: 背景颜色 ('black', 'white')
        dot_size: 事件点大小
        save_path: 保存路径
    
    Returns:
        img: 可视化图像
    """
    # 转换为numpy
    if isinstance(events, torch.Tensor):
        events = events.cpu().numpy()
    
    # 背景颜色
    if background == 'black':
        img = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # 过滤时间窗口
    if time_window is not None:
        mask = (events[:, 2] >= time_window[0]) & (events[:, 2] <= time_window[1])
        events = events[mask]
    
    if len(events) == 0:
        return img
    
    # 颜色映射
    if color_scheme == 'red_blue':
        pos_color = np.array([255, 0, 0])    # 红色表示正极性
        neg_color = np.array([0, 0, 255])    # 蓝色表示负极性
    elif color_scheme == 'green_red':
        pos_color = np.array([0, 255, 0])    # 绿色
        neg_color = np.array([255, 0, 0])    # 红色
    else:  # grayscale
        pos_color = np.array([255, 255, 255])  # 白色
        neg_color = np.array([128, 128, 128])  # 灰色
    
    # 时间归一化（用于淡出效果）
    if accumulate_time is not None:
        t_max = events[:, 2].max()
        t_min = t_max - accumulate_time
        time_weights = (events[:, 2] - t_min) / accumulate_time
        time_weights = np.clip(time_weights, 0, 1)
    else:
        time_weights = np.ones(len(events))
    
    # 绘制事件
    for i, event in enumerate(events):
        x, y, t, p = event
        x, y = int(x), int(y)
        
        if 0 <= x < width and 0 <= y < height:
            # 选择颜色
            color = pos_color if p > 0 else neg_color
            
            # 应用时间权重（淡出效果）
            alpha = time_weights[i]
            
            # 混合颜色
            if background == 'black':
                img[y, x] = (img[y, x] * (1 - alpha) + color * alpha).astype(np.uint8)
            else:
                img[y, x] = (img[y, x] * (1 - alpha) + color * alpha).astype(np.uint8)
            
            # 如果点大小大于1，绘制更大的点
            if dot_size > 1:
                cv2.circle(img, (x, y), int(dot_size), color.tolist(), -1)
    
    # 保存图像
    if save_path is not None:
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    return img


def visualize_voxel_grid(
    voxel_grid: Union[torch.Tensor, np.ndarray],
    method: str = 'sum',
    colormap: str = 'hot',
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    可视化体素网格
    
    Args:
        voxel_grid: (T, H, W) 或 (B, T, H, W) 体素网格
        method: 可视化方法 ('sum', 'mean', 'max', 'slice')
        colormap: 颜色映射
        save_path: 保存路径
    
    Returns:
        img: 可视化图像
    """
    # 转换为numpy
    if isinstance(voxel_grid, torch.Tensor):
        voxel_grid = voxel_grid.cpu().numpy()
    
    # 处理批次维度
    if voxel_grid.ndim == 4:
        voxel_grid = voxel_grid[0]  # 取第一个样本
    
    T, H, W = voxel_grid.shape
    
    # 聚合时间维度
    if method == 'sum':
        img = np.sum(voxel_grid, axis=0)
    elif method == 'mean':
        img = np.mean(voxel_grid, axis=0)
    elif method == 'max':
        img = np.max(voxel_grid, axis=0)
    elif method == 'slice':
        # 创建时间切片的网格
        n_slices = min(T, 9)  # 最多显示9个切片
        grid_size = int(np.ceil(np.sqrt(n_slices)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        axes = axes.flatten()
        
        for i in range(n_slices):
            t_idx = i * T // n_slices
            axes[i].imshow(voxel_grid[t_idx], cmap=colormap)
            axes[i].set_title(f'Time bin {t_idx}')
            axes[i].axis('off')
        
        # 隐藏多余的子图
        for i in range(n_slices, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        # 转换为图像数组
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return img
    
    # 归一化
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())
    
    # 应用颜色映射
    cmap = plt.get_cmap(colormap)
    img_colored = cmap(img)
    img_colored = (img_colored[:, :, :3] * 255).astype(np.uint8)
    
    # 保存图像
    if save_path is not None:
        cv2.imwrite(save_path, cv2.cvtColor(img_colored, cv2.COLOR_RGB2BGR))
    
    return img_colored


def visualize_reconstruction(
    pred: Union[torch.Tensor, np.ndarray],
    target: Optional[Union[torch.Tensor, np.ndarray]] = None,
    events: Optional[Union[torch.Tensor, np.ndarray]] = None,
    title: str = 'Reconstruction',
    save_path: Optional[str] = None
) -> None:
    """
    可视化重建结果
    
    Args:
        pred: 预测图像
        target: 目标图像（可选）
        events: 事件数据（可选）
        title: 图像标题
        save_path: 保存路径
    """
    # 转换为numpy
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if target is not None and isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    # 处理维度
    if pred.ndim == 4:  # (B, C, H, W)
        pred = pred[0]
    if pred.shape[0] == 1:  # (1, H, W)
        pred = pred[0]
    elif pred.shape[0] == 3:  # (3, H, W)
        pred = np.transpose(pred, (1, 2, 0))
    
    # 创建图形
    n_cols = 1
    if target is not None:
        n_cols += 1
    if events is not None:
        n_cols += 1
    
    fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
    if n_cols == 1:
        axes = [axes]
    
    # 显示预测
    axes[0].imshow(pred, cmap='gray' if pred.ndim == 2 else None)
    axes[0].set_title('Prediction')
    axes[0].axis('off')
    
    # 显示目标
    col_idx = 1
    if target is not None:
        if target.ndim == 4:
            target = target[0]
        if target.shape[0] == 1:
            target = target[0]
        elif target.shape[0] == 3:
            target = np.transpose(target, (1, 2, 0))
        
        axes[col_idx].imshow(target, cmap='gray' if target.ndim == 2 else None)
        axes[col_idx].set_title('Ground Truth')
        axes[col_idx].axis('off')
        col_idx += 1
    
    # 显示事件
    if events is not None:
        height, width = pred.shape[:2]
        event_img = visualize_events(events, height, width)
        axes[col_idx].imshow(event_img)
        axes[col_idx].set_title('Events')
        axes[col_idx].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def create_comparison_grid(
    predictions: List[np.ndarray],
    labels: List[str],
    target: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    创建多个预测结果的对比网格
    
    Args:
        predictions: 预测图像列表
        labels: 标签列表
        target: 目标图像
        save_path: 保存路径
    
    Returns:
        grid: 对比网格图像
    """
    n_predictions = len(predictions)
    n_cols = n_predictions
    if target is not None:
        n_cols += 1
    
    # 创建图形
    fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 4))
    if n_cols == 1:
        axes = [axes]
    
    # 显示目标
    col_idx = 0
    if target is not None:
        axes[0].imshow(target, cmap='gray' if target.ndim == 2 else None)
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')
        col_idx = 1
    
    # 显示预测
    for i, (pred, label) in enumerate(zip(predictions, labels)):
        axes[col_idx + i].imshow(pred, cmap='gray' if pred.ndim == 2 else None)
        axes[col_idx + i].set_title(label)
        axes[col_idx + i].axis('off')
    
    plt.tight_layout()
    
    # 转换为图像数组
    fig.canvas.draw()
    grid = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    grid = grid.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    
    plt.close(fig)
    
    return grid


def save_video(
    frames: Union[List[np.ndarray], np.ndarray],
    save_path: str,
    fps: int = 30,
    codec: str = 'mp4v'
) -> None:
    """
    保存视频
    
    Args:
        frames: 帧列表或帧数组 (T, H, W, C)
        save_path: 保存路径
        fps: 帧率
        codec: 视频编码器
    """
    if isinstance(frames, list):
        frames = np.array(frames)
    
    if frames.ndim == 3:  # (T, H, W) - 灰度图
        frames = np.expand_dims(frames, -1)
        frames = np.repeat(frames, 3, axis=-1)
    
    T, H, W, C = frames.shape
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(save_path, fourcc, fps, (W, H))
    
    # 写入帧
    for frame in frames:
        # 确保是uint8类型
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        # BGR转换
        if C == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        out.write(frame)
    
    out.release()


class EventVisualizer:
    """
    事件可视化器类
    提供更高级的可视化功能
    """
    
    def __init__(
        self,
        height: int,
        width: int,
        color_scheme: str = 'red_blue',
        background: str = 'black'
    ):
        self.height = height
        self.width = width
        self.color_scheme = color_scheme
        self.background = background
        
        # 事件累积器
        self.event_buffer = []
        self.max_buffer_size = 100000
    
    def add_events(self, events: Union[torch.Tensor, np.ndarray]):
        """添加事件到缓冲区"""
        if isinstance(events, torch.Tensor):
            events = events.cpu().numpy()
        
        self.event_buffer.append(events)
        
        # 限制缓冲区大小
        total_events = sum(len(e) for e in self.event_buffer)
        while total_events > self.max_buffer_size and len(self.event_buffer) > 1:
            self.event_buffer.pop(0)
            total_events = sum(len(e) for e in self.event_buffer)
    
    def clear_buffer(self):
        """清空事件缓冲区"""
        self.event_buffer = []
    
    def get_accumulation_image(
        self,
        time_window: Optional[float] = None,
        decay_rate: float = 0.95
    ) -> np.ndarray:
        """
        获取事件累积图像
        
        Args:
            time_window: 时间窗口大小
            decay_rate: 衰减率
        
        Returns:
            img: 累积图像
        """
        # 合并所有事件
        if not self.event_buffer:
            if self.background == 'black':
                return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            else:
                return np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        
        all_events = np.concatenate(self.event_buffer, axis=0)
        
        # 时间窗口过滤
        if time_window is not None:
            t_max = all_events[:, 2].max()
            t_min = t_max - time_window
            mask = all_events[:, 2] >= t_min
            all_events = all_events[mask]
        
        # 创建累积图像
        return visualize_events(
            all_events,
            self.height,
            self.width,
            color_scheme=self.color_scheme,
            background=self.background
        )
    
    def create_animation(
        self,
        events: Union[torch.Tensor, np.ndarray],
        window_size: float,
        step_size: float,
        save_path: str,
        fps: int = 30
    ) -> None:
        """
        创建事件动画
        
        Args:
            events: 所有事件
            window_size: 时间窗口大小
            step_size: 时间步长
            save_path: 保存路径
            fps: 帧率
        """
        if isinstance(events, torch.Tensor):
            events = events.cpu().numpy()
        
        # 计算时间范围
        t_min = events[:, 2].min()
        t_max = events[:, 2].max()
        
        # 生成帧
        frames = []
        t = t_min
        
        while t < t_max:
            # 时间窗口
            window = (t, t + window_size)
            
            # 可视化该窗口的事件
            frame = visualize_events(
                events,
                self.height,
                self.width,
                time_window=window,
                color_scheme=self.color_scheme,
                background=self.background
            )
            
            frames.append(frame)
            t += step_size
        
        # 保存视频
        save_video(frames, save_path, fps=fps)
    
    def plot_event_rate(
        self,
        events: Union[torch.Tensor, np.ndarray],
        bin_size: float = 0.001,
        save_path: Optional[str] = None
    ) -> None:
        """
        绘制事件率随时间的变化
        
        Args:
            events: 事件数据
            bin_size: 时间bin大小（秒）
            save_path: 保存路径
        """
        if isinstance(events, torch.Tensor):
            events = events.cpu().numpy()
        
        # 计算事件率
        t_min = events[:, 2].min()
        t_max = events[:, 2].max()
        
        bins = np.arange(t_min, t_max + bin_size, bin_size)
        hist, _ = np.histogram(events[:, 2], bins=bins)
        
        # 转换为事件率（事件/秒）
        event_rate = hist / bin_size
        time_centers = (bins[:-1] + bins[1:]) / 2
        
        # 绘图
        plt.figure(figsize=(10, 4))
        plt.plot(time_centers, event_rate)
        plt.xlabel('Time (s)')
        plt.ylabel('Event Rate (events/s)')
        plt.title('Event Rate over Time')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
