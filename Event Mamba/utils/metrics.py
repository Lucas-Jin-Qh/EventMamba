"""
评估指标计算
包含MSE、PSNR、SSIM、LPIPS等常用指标
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import math

# 尝试导入额外的库
try:
    from skimage.metrics import structural_similarity as compare_ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def compute_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    计算均方误差(MSE)
    
    Args:
        pred: 预测图像
        target: 目标图像
        reduction: 'mean', 'sum', 'none'
    
    Returns:
        mse: MSE值
    """
    mse = (pred - target) ** 2
    
    if reduction == 'mean':
        return mse.mean()
    elif reduction == 'sum':
        return mse.sum()
    else:
        return mse


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    计算峰值信噪比(PSNR)
    
    Args:
        pred: 预测图像
        target: 目标图像
        max_val: 像素最大值（通常为1.0或255）
        reduction: 如何聚合批次维度
    
    Returns:
        psnr: PSNR值（dB）
    """
    mse = compute_mse(pred, target, reduction='none')
    
    # 计算每个样本的PSNR
    if pred.dim() == 4:  # (B, C, H, W)
        mse = mse.view(pred.shape[0], -1).mean(dim=1)
    elif pred.dim() == 3:  # (C, H, W)
        mse = mse.mean()
    
    # 避免log(0)
    mse = torch.clamp(mse, min=1e-10)
    
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    
    if reduction == 'mean':
        return psnr.mean()
    elif reduction == 'sum':
        return psnr.sum()
    else:
        return psnr


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    K1: float = 0.01,
    K2: float = 0.03,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    计算结构相似性指数(SSIM)
    
    Args:
        pred: 预测图像 (B, C, H, W)
        target: 目标图像
        window_size: 高斯窗口大小
        sigma: 高斯标准差
        data_range: 数据范围
        K1, K2: SSIM常数
        reduction: 聚合方式
    
    Returns:
        ssim: SSIM值
    """
    # 创建高斯窗口
    def gaussian_window(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g
    
    # 1D高斯核
    gaussian_1d = gaussian_window(window_size, sigma)
    
    # 2D高斯核
    gaussian_2d = gaussian_1d.unsqueeze(1) @ gaussian_1d.unsqueeze(0)
    gaussian_2d = gaussian_2d.unsqueeze(0).unsqueeze(0)
    
    # 扩展到所有通道
    channels = pred.shape[1]
    gaussian_kernel = gaussian_2d.repeat(channels, 1, 1, 1)
    gaussian_kernel = gaussian_kernel.to(pred.device)
    
    # 常数
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    
    # 计算局部均值
    mu1 = F.conv2d(pred, gaussian_kernel, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(target, gaussian_kernel, padding=window_size//2, groups=channels)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # 计算局部方差和协方差
    sigma1_sq = F.conv2d(pred * pred, gaussian_kernel, padding=window_size//2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, gaussian_kernel, padding=window_size//2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, gaussian_kernel, padding=window_size//2, groups=channels) - mu1_mu2
    
    # SSIM公式
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_map = numerator / denominator
    
    # 聚合
    if reduction == 'mean':
        return ssim_map.mean()
    elif reduction == 'sum':
        return ssim_map.sum()
    else:
        return ssim_map


def compute_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    net: str = 'vgg',
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    计算LPIPS感知距离
    
    Args:
        pred: 预测图像
        target: 目标图像
        net: 网络类型
        device: 设备
    
    Returns:
        lpips: LPIPS值
    """
    # 导入LPIPS模块
    try:
        from ..losses.lpips_loss import LPIPS
        
        if device is None:
            device = pred.device
        
        # 创建LPIPS模型
        lpips_model = LPIPS(net=net).to(device)
        lpips_model.eval()
        
        with torch.no_grad():
            lpips_value = lpips_model(pred, target)
        
        return lpips_value
    
    except ImportError:
        # 如果无法导入，返回0
        print("Warning: LPIPS module not available")
        return torch.tensor(0.0, device=pred.device)


@dataclass
class EventMetrics:
    """事件重建评估指标数据类"""
    mse: float
    psnr: float
    ssim: float
    lpips: float
    temporal_consistency: Optional[float] = None
    event_rate_error: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'mse': self.mse,
            'psnr': self.psnr,
            'ssim': self.ssim,
            'lpips': self.lpips,
            'temporal_consistency': self.temporal_consistency,
            'event_rate_error': self.event_rate_error
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        s = f"MSE: {self.mse:.4f}, PSNR: {self.psnr:.2f}dB, SSIM: {self.ssim:.4f}, LPIPS: {self.lpips:.4f}"
        if self.temporal_consistency is not None:
            s += f", TC: {self.temporal_consistency:.4f}"
        if self.event_rate_error is not None:
            s += f", ERE: {self.event_rate_error:.4f}"
        return s


class MetricsCalculator:
    """
    评估指标计算器
    用于批量计算和跟踪各种指标
    """
    
    def __init__(
        self,
        metrics: List[str] = ['mse', 'psnr', 'ssim', 'lpips'],
        lpips_net: str = 'vgg',
        device: Optional[torch.device] = None
    ):
        self.metrics = metrics
        self.device = device
        
        # 初始化LPIPS模型（如果需要）
        if 'lpips' in metrics:
            try:
                from ..losses.lpips_loss import LPIPS
                self.lpips_model = LPIPS(net=lpips_net).to(device)
                self.lpips_model.eval()
            except:
                self.lpips_model = None
                print("Warning: LPIPS model not available")
        
        # 重置累积器
        self.reset()
    
    def reset(self):
        """重置累积器"""
        self.accumulated = {metric: 0.0 for metric in self.metrics}
        self.count = 0
    
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        flows: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        更新指标
        
        Args:
            pred: 预测图像
            target: 目标图像
            flows: 光流（用于时间一致性）
        
        Returns:
            metrics: 当前批次的指标值
        """
        batch_metrics = {}
        
        with torch.no_grad():
            # MSE
            if 'mse' in self.metrics:
                mse = compute_mse(pred, target).item()
                batch_metrics['mse'] = mse
                self.accumulated['mse'] += mse
            
            # PSNR
            if 'psnr' in self.metrics:
                psnr = compute_psnr(pred, target).item()
                batch_metrics['psnr'] = psnr
                self.accumulated['psnr'] += psnr
            
            # SSIM
            if 'ssim' in self.metrics:
                ssim = compute_ssim(pred, target).item()
                batch_metrics['ssim'] = ssim
                self.accumulated['ssim'] += ssim
            
            # LPIPS
            if 'lpips' in self.metrics and self.lpips_model is not None:
                lpips = self.lpips_model(pred, target).item()
                batch_metrics['lpips'] = lpips
                self.accumulated['lpips'] += lpips
            
            # 时间一致性（如果是视频）
            if 'temporal_consistency' in self.metrics and flows is not None:
                tc = self.compute_temporal_consistency(pred, flows).item()
                batch_metrics['temporal_consistency'] = tc
                self.accumulated['temporal_consistency'] += tc
        
        self.count += 1
        return batch_metrics
    
    def compute_temporal_consistency(
        self,
        frames: torch.Tensor,
        flows: torch.Tensor
    ) -> torch.Tensor:
        """计算时间一致性误差"""
        # 简化实现：计算帧差
        if frames.dim() == 5 and frames.shape[1] > 1:
            frame_diff = torch.abs(frames[:, 1:] - frames[:, :-1]).mean()
            return frame_diff
        else:
            return torch.tensor(0.0)
    
    def get_average(self) -> Dict[str, float]:
        """获取平均指标"""
        if self.count == 0:
            return {metric: 0.0 for metric in self.metrics}
        
        return {
            metric: self.accumulated[metric] / self.count
            for metric in self.metrics
        }
    
    def compute_all(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> EventMetrics:
        """
        计算所有指标
        
        Args:
            pred: 预测图像
            target: 目标图像
        
        Returns:
            metrics: EventMetrics对象
        """
        with torch.no_grad():
            mse = compute_mse(pred, target).item()
            psnr = compute_psnr(pred, target).item()
            ssim = compute_ssim(pred, target).item()
            
            if self.lpips_model is not None:
                lpips = self.lpips_model(pred, target).item()
            else:
                lpips = 0.0
        
        return EventMetrics(
            mse=mse,
            psnr=psnr,
            ssim=ssim,
            lpips=lpips
        )


def compute_event_generation_error(
    pred_image: torch.Tensor,
    target_image: torch.Tensor,
    threshold: float = 0.1
) -> torch.Tensor:
    """
    计算事件生成误差
    基于图像差异应该产生的事件数量
    
    Args:
        pred_image: 预测图像
        target_image: 目标图像
        threshold: 事件生成阈值
    
    Returns:
        error: 事件生成误差
    """
    # 计算对数强度变化
    log_pred = torch.log(pred_image + 1e-3)
    log_target = torch.log(target_image + 1e-3)
    
    # 强度变化
    intensity_change = torch.abs(log_pred - log_target)
    
    # 预期事件数量
    expected_events = (intensity_change > threshold).float()
    
    # 事件生成误差
    error = expected_events.mean()
    
    return error


def calculate_video_metrics(
    pred_video: torch.Tensor,
    target_video: torch.Tensor,
    metrics_calculator: Optional[MetricsCalculator] = None
) -> Dict[str, float]:
    """
    计算视频序列的指标
    
    Args:
        pred_video: (B, T, C, H, W) 预测视频
        target_video: 目标视频
        metrics_calculator: 指标计算器
    
    Returns:
        metrics: 平均指标字典
    """
    if metrics_calculator is None:
        metrics_calculator = MetricsCalculator()
    
    metrics_calculator.reset()
    
    B, T, C, H, W = pred_video.shape
    
    # 对每帧计算指标
    for t in range(T):
        metrics_calculator.update(
            pred_video[:, t],
            target_video[:, t]
        )
    
    return metrics_calculator.get_average()
