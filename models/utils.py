"""
EventMamba工具函数
包含事件处理、网络构建块和其他实用函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Union, List
import math


def event_to_voxel_grid(
    events: torch.Tensor,
    num_bins: int,
    height: int,
    width: int,
    normalize: bool = True
) -> torch.Tensor:
    """
    将事件流转换为体素网格表示
    
    Args:
        events: (N, 4) 事件张量 [x, y, t, p]
            - x, y: 空间坐标
            - t: 时间戳（已归一化到[0, 1]或原始时间戳）
            - p: 极性（+1或-1）
        num_bins: 时间维度的bin数量
        height: 输出高度
        width: 输出宽度
        normalize: 是否归一化时间戳
    
    Returns:
        voxel_grid: (num_bins, H, W) 体素网格
    """
    if events.shape[0] == 0:
        return torch.zeros((num_bins, height, width), dtype=torch.float32, device=events.device)
    
    # 确保事件在正确的设备上
    device = events.device
    
    # 提取坐标和极性
    x = events[:, 0].long()
    y = events[:, 1].long()
    t = events[:, 2]
    p = events[:, 3]
    
    # 时间归一化
    if normalize:
        t_min = t.min()
        t_max = t.max()
        if t_max > t_min:
            t = (t - t_min) / (t_max - t_min)
        else:
            t = torch.zeros_like(t)
    
    # 计算时间bin索引
    t_indices = (t * (num_bins - 1)).long()
    t_indices = torch.clamp(t_indices, 0, num_bins - 1)
    
    # 过滤掉超出边界的事件
    valid_mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x = x[valid_mask]
    y = y[valid_mask]
    t_indices = t_indices[valid_mask]
    p = p[valid_mask]
    
    # 创建体素网格
    voxel_grid = torch.zeros((num_bins, height, width), dtype=torch.float32, device=device)
    
    # 使用双线性插值进行亚像素精度
    # 计算相邻bin的权重
    t_continuous = t[valid_mask] * (num_bins - 1)
    t_floor = t_indices.float()
    t_ceil = torch.clamp(t_floor + 1, 0, num_bins - 1)
    
    # 时间维度的权重
    weight_floor = 1.0 - (t_continuous - t_floor)
    weight_ceil = 1.0 - weight_floor
    
    # 累积事件到体素网格
    # Floor bin
    voxel_grid.index_put_(
        (t_indices, y, x),
        p * weight_floor,
        accumulate=True
    )
    
    # Ceil bin（如果不同于floor）
    mask_diff = t_ceil != t_floor
    if mask_diff.any():
        voxel_grid.index_put_(
            (t_ceil[mask_diff].long(), y[mask_diff], x[mask_diff]),
            p[mask_diff] * weight_ceil[mask_diff],
            accumulate=True
        )
    
    return voxel_grid


def normalize_voxel_grid(voxel_grid: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """
    归一化体素网格
    
    Args:
        voxel_grid: 输入体素网格
        eps: 防止除零的小值
    
    Returns:
        normalized_grid: 归一化后的体素网格
    """
    # 计算每个bin的统计信息
    if voxel_grid.dim() == 3:  # (T, H, W)
        mean = voxel_grid.mean(dim=(1, 2), keepdim=True)
        std = voxel_grid.std(dim=(1, 2), keepdim=True)
    elif voxel_grid.dim() == 4:  # (B, T, H, W)
        mean = voxel_grid.mean(dim=(2, 3), keepdim=True)
        std = voxel_grid.std(dim=(2, 3), keepdim=True)
    else:
        raise ValueError(f"Unexpected voxel grid dimensions: {voxel_grid.dim()}")
    
    # 归一化
    normalized_grid = (voxel_grid - mean) / (std + eps)
    
    return normalized_grid


def create_conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    bias: bool = True,
    norm_type: str = "batch",
    activation: str = "gelu",
    dimension: int = 3
) -> nn.Sequential:
    """
    创建卷积块
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        stride: 步幅
        padding: 填充
        bias: 是否使用偏置
        norm_type: 归一化类型 ("batch", "layer", "instance", "none")
        activation: 激活函数类型 ("relu", "gelu", "silu", "none")
        dimension: 卷积维度 (2或3)
    
    Returns:
        conv_block: 卷积块
    """
    layers = []
    
    # 卷积层
    if dimension == 2:
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
    elif dimension == 3:
        conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
    else:
        raise ValueError(f"Unsupported dimension: {dimension}")
    
    layers.append(conv)
    
    # 归一化层
    if norm_type == "batch":
        norm = nn.BatchNorm3d(out_channels) if dimension == 3 else nn.BatchNorm2d(out_channels)
        layers.append(norm)
    elif norm_type == "layer":
        norm = nn.LayerNorm(out_channels)
        layers.append(norm)
    elif norm_type == "instance":
        norm = nn.InstanceNorm3d(out_channels) if dimension == 3 else nn.InstanceNorm2d(out_channels)
        layers.append(norm)
    
    # 激活函数
    if activation == "relu":
        layers.append(nn.ReLU(inplace=True))
    elif activation == "gelu":
        layers.append(nn.GELU())
    elif activation == "silu":
        layers.append(nn.SiLU(inplace=True))
    
    return nn.Sequential(*layers)


def create_downsample_block(
    in_channels: int,
    out_channels: int,
    downsample_type: str = "conv",
    kernel_size: int = 3,
    stride: int = 2,
    padding: int = 1
) -> nn.Module:
    """
    创建下采样块
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        downsample_type: 下采样类型 ("conv", "maxpool", "avgpool")
        kernel_size: 核大小
        stride: 步幅
        padding: 填充
    
    Returns:
        downsample_block: 下采样块
    """
    if downsample_type == "conv":
        return nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
    elif downsample_type == "maxpool":
        return nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
        )
    elif downsample_type == "avgpool":
        return nn.Sequential(
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
        )
    else:
        raise ValueError(f"Unknown downsample type: {downsample_type}")


def create_upsample_block(
    in_channels: int,
    out_channels: int,
    upsample_type: str = "transpose",
    kernel_size: int = 4,
    stride: int = 2,
    padding: int = 1
) -> nn.Module:
    """
    创建上采样块
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        upsample_type: 上采样类型 ("transpose", "bilinear+conv", "nearest+conv")
        kernel_size: 核大小
        stride: 步幅
        padding: 填充
    
    Returns:
        upsample_block: 上采样块
    """
    if upsample_type == "transpose":
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding)
    elif upsample_type == "bilinear+conv":
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        )
    elif upsample_type == "nearest+conv":
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        )
    else:
        raise ValueError(f"Unknown upsample type: {upsample_type}")


def compute_model_stats(model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, Union[int, float]]:
    """
    计算模型统计信息
    
    Args:
        model: PyTorch模型
        input_shape: 输入形状
    
    Returns:
        stats: 包含参数数量、FLOPs等的字典
    """
    # 参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 简化的FLOPs估算（更精确的计算需要使用专门的库）
    # 这里只是一个粗略的估计
    def estimate_conv_flops(in_ch, out_ch, kernel_size, output_size):
        # FLOPs = 2 * kernel_size^d * in_ch * out_ch * output_size
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * len(output_size)
        kernel_ops = np.prod(kernel_size)
        output_ops = np.prod(output_size)
        return 2 * kernel_ops * in_ch * out_ch * output_ops
    
    # 这是一个非常简化的估计
    estimated_flops = total_params * np.prod(input_shape[1:])  # 粗略估计
    
    stats = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params,
        'flops': estimated_flops,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # 假设float32
    }
    
    return stats


def initialize_weights(module: nn.Module):
    """
    初始化模型权重
    
    Args:
        module: nn.Module
    """
    if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        # Kaiming初始化（适用于ReLU/GELU）
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    
    elif isinstance(module, nn.Linear):
        # Xavier初始化
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


class EventAugmentation:
    """
    事件数据增强类
    """
    
    def __init__(
        self,
        random_flip_p: float = 0.5,
        random_time_flip_p: float = 0.0,
        noise_std: float = 0.0,
        event_drop_p: float = 0.0,
        contrast_threshold_range: Optional[Tuple[float, float]] = None
    ):
        self.random_flip_p = random_flip_p
        self.random_time_flip_p = random_time_flip_p
        self.noise_std = noise_std
        self.event_drop_p = event_drop_p
        self.contrast_threshold_range = contrast_threshold_range
    
    def __call__(self, events: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        应用数据增强
        
        Args:
            events: (N, 4) 事件张量 [x, y, t, p]
            height: 图像高度
            width: 图像宽度
        
        Returns:
            augmented_events: 增强后的事件
        """
        events = events.clone()
        
        # 随机水平翻转
        if torch.rand(1).item() < self.random_flip_p:
            events[:, 0] = width - 1 - events[:, 0]
        
        # 随机时间翻转
        if torch.rand(1).item() < self.random_time_flip_p:
            t_max = events[:, 2].max()
            events[:, 2] = t_max - events[:, 2]
            events[:, 3] = -events[:, 3]  # 翻转极性
        
        # 添加噪声
        if self.noise_std > 0:
            noise = torch.randn_like(events[:, :2]) * self.noise_std
            events[:, :2] += noise
            # 裁剪到有效范围
            events[:, 0] = torch.clamp(events[:, 0], 0, width - 1)
            events[:, 1] = torch.clamp(events[:, 1], 0, height - 1)
        
        # 随机丢弃事件
        if self.event_drop_p > 0:
            keep_mask = torch.rand(len(events)) >= self.event_drop_p
            events = events[keep_mask]
        
        # 对比度阈值扰动
        if self.contrast_threshold_range is not None:
            # 这里可以实现基于对比度阈值的事件过滤
            pass
        
        return events


def create_event_batch(
    events_list: List[torch.Tensor],
    batch_size: int,
    max_events: Optional[int] = None
) -> torch.Tensor:
    """
    创建事件批次，处理不同长度的事件序列
    
    Args:
        events_list: 事件张量列表
        batch_size: 批次大小
        max_events: 最大事件数（用于截断或填充）
    
    Returns:
        batch: (B, N, 4) 批次张量
    """
    if max_events is None:
        max_events = max(len(events) for events in events_list)
    
    batch = torch.zeros((batch_size, max_events, 4), dtype=torch.float32)
    
    for i, events in enumerate(events_list[:batch_size]):
        num_events = min(len(events), max_events)
        batch[i, :num_events] = events[:num_events]
        # 用-1填充坐标以标记无效事件
        if num_events < max_events:
            batch[i, num_events:, :2] = -1
    
    return batch
