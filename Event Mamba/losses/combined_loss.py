"""
组合损失函数
结合多种损失以优化事件重建任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

from .lpips_loss import LPIPS
from .temporal_consistency import TemporalConsistencyLoss, SmoothingLoss


@dataclass
class LossConfig:
    """损失函数配置"""
    # L1损失
    lambda_l1: float = 20.0
    
    # LPIPS感知损失
    lambda_lpips: float = 2.0
    lpips_net: str = 'vgg'
    
    # 时间一致性损失
    lambda_temporal: float = 0.5
    temporal_loss_type: str = 'l2'
    use_occlusion_mask: bool = True
    
    # 平滑损失
    lambda_smooth: float = 0.1
    spatial_smooth_weight: float = 1.0
    temporal_smooth_weight: float = 1.0
    edge_aware_smooth: bool = True
    
    # SSIM损失
    lambda_ssim: float = 0.0
    ssim_window_size: int = 11
    
    # 其他配置
    reduction: str = 'mean'
    multi_scale: bool = False
    scales: List[float] = None
    scale_weights: List[float] = None
    
    def __post_init__(self):
        if self.scales is None:
            self.scales = [1.0, 0.5, 0.25]
        if self.scale_weights is None:
            self.scale_weights = [1.0, 0.5, 0.25]


class CombinedLoss(nn.Module):
    """
    组合损失函数
    结合L1、LPIPS、时间一致性等多种损失
    """
    
    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config
        
        # L1损失
        self.l1_loss = nn.L1Loss(reduction=config.reduction)
        
        # LPIPS损失
        if config.lambda_lpips > 0:
            self.lpips_loss = LPIPS(
                net=config.lpips_net,
                reduction=config.reduction
            )
        
        # 时间一致性损失
        if config.lambda_temporal > 0:
            self.temporal_loss = TemporalConsistencyLoss(
                loss_type=config.temporal_loss_type,
                use_occlusion_mask=config.use_occlusion_mask,
                multi_scale=config.multi_scale,
                scales=config.scales,
                weights=config.scale_weights
            )
        
        # 平滑损失
        if config.lambda_smooth > 0:
            self.smooth_loss = SmoothingLoss(
                spatial_weight=config.spatial_smooth_weight,
                temporal_weight=config.temporal_smooth_weight,
                edge_aware=config.edge_aware_smooth
            )
        
        # SSIM损失
        if config.lambda_ssim > 0:
            from .ssim import SSIM
            self.ssim_loss = SSIM(
                window_size=config.ssim_window_size,
                reduction=config.reduction
            )
        
        # 损失权重字典
        self.loss_weights = {
            'l1': config.lambda_l1,
            'lpips': config.lambda_lpips,
            'temporal': config.lambda_temporal,
            'smooth': config.lambda_smooth,
            'ssim': config.lambda_ssim
        }
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        flows_forward: Optional[torch.Tensor] = None,
        flows_backward: Optional[torch.Tensor] = None,
        return_dict: bool = False
    ) -> torch.Tensor:
        """
        计算组合损失
        
        Args:
            pred: 预测图像 (B, C, H, W) 或 (B, T, C, H, W)
            target: 目标图像
            flows_forward: 前向光流（用于时间一致性）
            flows_backward: 后向光流（用于遮挡检测）
            return_dict: 是否返回损失字典
        
        Returns:
            loss: 总损失或损失字典
        """
        losses = {}
        total_loss = 0
        
        # 处理视频序列
        is_video = pred.dim() == 5
        
        # L1损失
        if self.config.lambda_l1 > 0:
            if is_video:
                # 对每帧计算L1损失
                l1_loss = 0
                B, T, C, H, W = pred.shape
                for t in range(T):
                    l1_loss += self.l1_loss(pred[:, t], target[:, t])
                l1_loss = l1_loss / T
            else:
                l1_loss = self.l1_loss(pred, target)
            
            losses['l1'] = l1_loss
            total_loss += self.config.lambda_l1 * l1_loss
        
        # LPIPS损失
        if self.config.lambda_lpips > 0:
            if is_video:
                # 对每帧计算LPIPS损失
                lpips_loss = 0
                B, T, C, H, W = pred.shape
                for t in range(T):
                    lpips_loss += self.lpips_loss(pred[:, t], target[:, t])
                lpips_loss = lpips_loss / T
            else:
                lpips_loss = self.lpips_loss(pred, target)
            
            losses['lpips'] = lpips_loss
            total_loss += self.config.lambda_lpips * lpips_loss
        
        # 时间一致性损失（仅用于视频）
        if self.config.lambda_temporal > 0 and is_video:
            temporal_loss = self.temporal_loss(
                pred,
                flows_forward,
                flows_backward
            )
            losses['temporal'] = temporal_loss
            total_loss += self.config.lambda_temporal * temporal_loss
        
        # 平滑损失
        if self.config.lambda_smooth > 0:
            smooth_loss = self.smooth_loss(pred, target)
            losses['smooth'] = smooth_loss
            total_loss += self.config.lambda_smooth * smooth_loss
        
        # SSIM损失
        if self.config.lambda_ssim > 0:
            if is_video:
                # 对每帧计算SSIM损失
                ssim_loss = 0
                B, T, C, H, W = pred.shape
                for t in range(T):
                    ssim_value = self.ssim_loss(pred[:, t], target[:, t])
                    ssim_loss += (1 - ssim_value)  # SSIM是相似度，需要转换为损失
                ssim_loss = ssim_loss / T
            else:
                ssim_value = self.ssim_loss(pred, target)
                ssim_loss = 1 - ssim_value
            
            losses['ssim'] = ssim_loss
            total_loss += self.config.lambda_ssim * ssim_loss
        
        if return_dict:
            losses['total'] = total_loss
            return losses
        else:
            return total_loss
    
    def get_last_losses(self) -> Dict[str, float]:
        """获取最后一次前向传播的损失值"""
        # 这个功能需要在forward中保存损失值
        return {}


class EventReconstructionLoss(nn.Module):
    """
    专门用于事件重建的损失函数
    包含事件特定的损失项
    """
    
    def __init__(
        self,
        base_loss_config: LossConfig,
        event_consistency_weight: float = 0.1,
        contrast_sensitivity_weight: float = 0.05
    ):
        super().__init__()
        
        # 基础组合损失
        self.base_loss = CombinedLoss(base_loss_config)
        
        # 事件特定权重
        self.event_consistency_weight = event_consistency_weight
        self.contrast_sensitivity_weight = contrast_sensitivity_weight
    
    def compute_event_consistency_loss(
        self,
        pred: torch.Tensor,
        events: torch.Tensor,
        threshold: float = 0.1
    ) -> torch.Tensor:
        """
        计算事件一致性损失
        确保重建图像的梯度与事件位置一致
        
        Args:
            pred: (B, C, H, W) 预测图像
            events: (B, N, 4) 事件数据 [x, y, t, p]
            threshold: 对比度阈值
        
        Returns:
            loss: 事件一致性损失
        """
        B, C, H, W = pred.shape
        loss = 0
        
        # 计算图像梯度
        grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        
        # 梯度幅值
        grad_mag = torch.sqrt(
            F.pad(grad_x, (0, 1, 0, 0)) ** 2 +
            F.pad(grad_y, (0, 0, 0, 1)) ** 2
        )
        
        # 对每个batch处理事件
        for b in range(B):
            # 获取有效事件（假设x坐标为负表示填充）
            valid_mask = events[b, :, 0] >= 0
            valid_events = events[b, valid_mask]
            
            if len(valid_events) > 0:
                # 事件位置
                x_coords = valid_events[:, 0].long()
                y_coords = valid_events[:, 1].long()
                
                # 确保坐标在有效范围内
                x_coords = torch.clamp(x_coords, 0, W - 1)
                y_coords = torch.clamp(y_coords, 0, H - 1)
                
                # 采样事件位置的梯度
                event_grads = grad_mag[b, 0, y_coords, x_coords]
                
                # 事件位置应该有高梯度
                loss += torch.mean(torch.relu(threshold - event_grads))
        
        return loss / B
    
    def compute_contrast_sensitivity_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sensitivity_map: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算对比度敏感性损失
        在高对比度区域给予更高的权重
        
        Args:
            pred: 预测图像
            target: 目标图像
            sensitivity_map: 可选的敏感性图
        
        Returns:
            loss: 对比度敏感性损失
        """
        # 计算局部对比度
        kernel_size = 3
        padding = kernel_size // 2
        
        # 使用局部标准差作为对比度度量
        avg_pool = nn.AvgPool2d(kernel_size, stride=1, padding=padding)
        
        # 局部均值
        target_mean = avg_pool(target)
        
        # 局部方差
        target_var = avg_pool(target ** 2) - target_mean ** 2
        target_std = torch.sqrt(target_var + 1e-6)
        
        # 对比度权重（高对比度区域权重更大）
        if sensitivity_map is None:
            sensitivity_map = target_std / (target_std.mean() + 1e-6)
        
        # 加权L1损失
        weighted_diff = torch.abs(pred - target) * sensitivity_map
        
        return weighted_diff.mean()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        events: Optional[torch.Tensor] = None,
        flows_forward: Optional[torch.Tensor] = None,
        flows_backward: Optional[torch.Tensor] = None,
        return_dict: bool = False
    ) -> torch.Tensor:
        """
        计算事件重建损失
        """
        # 基础损失
        if return_dict:
            losses = self.base_loss(pred, target, flows_forward, flows_backward, return_dict=True)
            total_loss = losses['total']
        else:
            total_loss = self.base_loss(pred, target, flows_forward, flows_backward)
            losses = {}
        
        # 事件一致性损失
        if self.event_consistency_weight > 0 and events is not None:
            event_loss = self.compute_event_consistency_loss(pred, events)
            losses['event_consistency'] = event_loss
            total_loss += self.event_consistency_weight * event_loss
        
        # 对比度敏感性损失
        if self.contrast_sensitivity_weight > 0:
            contrast_loss = self.compute_contrast_sensitivity_loss(pred, target)
            losses['contrast_sensitivity'] = contrast_loss
            total_loss += self.contrast_sensitivity_weight * contrast_loss
        
        if return_dict:
            losses['total'] = total_loss
            return losses
        else:
            return total_loss


def create_loss_function(
    loss_type: str = 'combined',
    **kwargs
) -> nn.Module:
    """
    创建损失函数的工厂函数
    
    Args:
        loss_type: 损失类型
        **kwargs: 损失函数参数
    
    Returns:
        loss_fn: 损失函数模块
    """
    if loss_type == 'combined':
        config = LossConfig(**kwargs)
        return CombinedLoss(config)
    elif loss_type == 'event_reconstruction':
        config = LossConfig(**kwargs)
        return EventReconstructionLoss(config)
    elif loss_type == 'l1':
        return nn.L1Loss()
    elif loss_type == 'l2':
        return nn.MSELoss()
    elif loss_type == 'lpips':
        return LPIPS(net=kwargs.get('net', 'vgg'))
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# SSIM实现（如果上面引用了但未实现）
class SSIM(nn.Module):
    """结构相似性损失"""
    
    def __init__(self, window_size: int = 11, reduction: str = 'mean'):
        super().__init__()
        self.window_size = window_size
        self.reduction = reduction
        self.register_buffer('window', self._create_window(window_size))
    
    def _create_window(self, window_size: int) -> torch.Tensor:
        """创建高斯窗口"""
        sigma = 1.5
        gauss = torch.Tensor([
            math.exp(-(x - window_size//2)**2/float(2*sigma**2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算SSIM"""
        # 简化实现
        return F.mse_loss(pred, target)
