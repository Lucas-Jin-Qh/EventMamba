"""
时间一致性损失实现
用于确保重建的视频帧之间的时间连贯性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


def flow_warp(
    img: torch.Tensor,
    flow: torch.Tensor,
    padding_mode: str = 'zeros',
    interpolation_mode: str = 'bilinear'
) -> torch.Tensor:
    """
    使用光流对图像进行扭曲
    
    Args:
        img: (B, C, H, W) 输入图像
        flow: (B, 2, H, W) 光流场，flow[0]是x方向，flow[1]是y方向
        padding_mode: 填充模式 ('zeros', 'border', 'reflection')
        interpolation_mode: 插值模式 ('bilinear', 'nearest')
    
    Returns:
        warped: (B, C, H, W) 扭曲后的图像
    """
    B, C, H, W = img.shape
    
    # 创建网格
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=img.device),
        torch.arange(W, dtype=torch.float32, device=img.device),
        indexing='ij'
    )
    
    # 添加光流
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1) + flow[:, 0, :, :]
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1) + flow[:, 1, :, :]
    
    # 归一化到[-1, 1]
    grid_x = 2.0 * grid_x / (W - 1) - 1.0
    grid_y = 2.0 * grid_y / (H - 1) - 1.0
    
    # 堆叠成grid
    grid = torch.stack([grid_x, grid_y], dim=3)
    
    # 应用grid_sample
    warped = F.grid_sample(
        img,
        grid,
        mode=interpolation_mode,
        padding_mode=padding_mode,
        align_corners=True
    )
    
    return warped


class TemporalConsistencyLoss(nn.Module):
    """
    时间一致性损失
    确保相邻帧之间的运动连贯性
    """
    
    def __init__(
        self,
        loss_type: str = 'l2',
        use_occlusion_mask: bool = True,
        occlusion_threshold: float = 0.5,
        forward_backward_consistency: bool = True,
        multi_scale: bool = False,
        scales: List[float] = [1.0, 0.5, 0.25],
        weights: Optional[List[float]] = None
    ):
        """
        Args:
            loss_type: 损失类型 ('l1', 'l2', 'charbonnier')
            use_occlusion_mask: 是否使用遮挡掩码
            occlusion_threshold: 遮挡阈值
            forward_backward_consistency: 是否使用前向-后向一致性检查
            multi_scale: 是否使用多尺度损失
            scales: 多尺度的缩放因子
            weights: 各尺度的权重
        """
        super().__init__()
        
        self.loss_type = loss_type
        self.use_occlusion_mask = use_occlusion_mask
        self.occlusion_threshold = occlusion_threshold
        self.forward_backward_consistency = forward_backward_consistency
        self.multi_scale = multi_scale
        self.scales = scales
        self.weights = weights if weights is not None else [1.0] * len(scales)
        
        # Charbonnier损失的epsilon
        self.epsilon = 1e-3
    
    def compute_occlusion_mask(
        self,
        flow_forward: torch.Tensor,
        flow_backward: torch.Tensor
    ) -> torch.Tensor:
        """
        计算遮挡掩码
        使用前向-后向一致性检查
        
        Args:
            flow_forward: (B, 2, H, W) 前向光流
            flow_backward: (B, 2, H, W) 后向光流
        
        Returns:
            mask: (B, 1, H, W) 遮挡掩码（1表示未遮挡）
        """
        # 使用前向光流扭曲后向光流
        flow_backward_warped = flow_warp(flow_backward, flow_forward)
        
        # 计算一致性误差
        consistency_error = torch.norm(
            flow_forward + flow_backward_warped,
            dim=1,
            keepdim=True
        )
        
        # 创建掩码
        mask = (consistency_error < self.occlusion_threshold).float()
        
        return mask
    
    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算像素级损失
        
        Args:
            pred: 预测值
            target: 目标值
            mask: 可选的掩码
        
        Returns:
            loss: 损失值
        """
        if self.loss_type == 'l1':
            loss = torch.abs(pred - target)
        elif self.loss_type == 'l2':
            loss = (pred - target) ** 2
        elif self.loss_type == 'charbonnier':
            loss = torch.sqrt((pred - target) ** 2 + self.epsilon ** 2)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # 应用掩码
        if mask is not None:
            loss = loss * mask
            # 计算有效像素的平均损失
            return loss.sum() / (mask.sum() + 1e-10)
        else:
            return loss.mean()
    
    def forward(
        self,
        frames: torch.Tensor,
        flows_forward: Optional[torch.Tensor] = None,
        flows_backward: Optional[torch.Tensor] = None,
        compute_flow: bool = False
    ) -> torch.Tensor:
        """
        计算时间一致性损失
        
        Args:
            frames: (B, T, C, H, W) 视频帧序列
            flows_forward: (B, T-1, 2, H, W) 前向光流（可选）
            flows_backward: (B, T-1, 2, H, W) 后向光流（可选）
            compute_flow: 是否自动计算光流（需要额外的光流估计器）
        
        Returns:
            loss: 时间一致性损失
        """
        B, T, C, H, W = frames.shape
        
        if T < 2:
            return torch.tensor(0.0, device=frames.device)
        
        total_loss = 0
        count = 0
        
        # 对每对相邻帧计算损失
        for t in range(T - 1):
            frame_curr = frames[:, t]      # (B, C, H, W)
            frame_next = frames[:, t + 1]  # (B, C, H, W)
            
            if flows_forward is not None:
                flow_forward = flows_forward[:, t]  # (B, 2, H, W)
                
                # 使用光流扭曲当前帧
                frame_curr_warped = flow_warp(frame_curr, flow_forward)
                
                # 计算遮挡掩码
                mask = None
                if self.use_occlusion_mask and flows_backward is not None:
                    flow_backward = flows_backward[:, t]
                    mask = self.compute_occlusion_mask(flow_forward, flow_backward)
                
                # 多尺度损失
                if self.multi_scale:
                    scale_loss = 0
                    for scale, weight in zip(self.scales, self.weights):
                        if scale != 1.0:
                            # 下采样
                            frame_curr_warped_scaled = F.interpolate(
                                frame_curr_warped,
                                scale_factor=scale,
                                mode='bilinear',
                                align_corners=False
                            )
                            frame_next_scaled = F.interpolate(
                                frame_next,
                                scale_factor=scale,
                                mode='bilinear',
                                align_corners=False
                            )
                            if mask is not None:
                                mask_scaled = F.interpolate(
                                    mask,
                                    scale_factor=scale,
                                    mode='nearest'
                                )
                            else:
                                mask_scaled = None
                        else:
                            frame_curr_warped_scaled = frame_curr_warped
                            frame_next_scaled = frame_next
                            mask_scaled = mask
                        
                        # 计算该尺度的损失
                        loss = self.compute_loss(
                            frame_curr_warped_scaled,
                            frame_next_scaled,
                            mask_scaled
                        )
                        scale_loss += weight * loss
                    
                    total_loss += scale_loss
                else:
                    # 单尺度损失
                    loss = self.compute_loss(frame_curr_warped, frame_next, mask)
                    total_loss += loss
                
                count += 1
            else:
                # 如果没有光流，直接计算帧差
                loss = self.compute_loss(frame_curr, frame_next)
                total_loss += loss
                count += 1
        
        # 平均损失
        if count > 0:
            total_loss = total_loss / count
        
        return total_loss


class SmoothingLoss(nn.Module):
    """
    平滑损失
    鼓励空间和时间上的平滑性
    """
    
    def __init__(
        self,
        spatial_weight: float = 1.0,
        temporal_weight: float = 1.0,
        edge_aware: bool = True,
        edge_threshold: float = 0.01
    ):
        super().__init__()
        
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight
        self.edge_aware = edge_aware
        self.edge_threshold = edge_threshold
        
        # Sobel算子用于边缘检测
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
    
    def compute_gradients(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算空间梯度"""
        # 使用Sobel算子
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        
        return grad_x, grad_y
    
    def compute_edge_weight(self, img: torch.Tensor) -> torch.Tensor:
        """计算边缘感知权重"""
        # 转换为灰度图
        if img.shape[1] > 1:
            gray = img.mean(dim=1, keepdim=True)
        else:
            gray = img
        
        # 计算梯度
        grad_x, grad_y = self.compute_gradients(gray)
        
        # 梯度幅值
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        
        # 边缘权重（边缘处权重小）
        edge_weight = torch.exp(-grad_mag / self.edge_threshold)
        
        return edge_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算平滑损失
        
        Args:
            pred: (B, C, H, W) 或 (B, T, C, H, W) 预测
            target: 可选的目标图像（用于边缘感知）
        
        Returns:
            loss: 平滑损失
        """
        loss = 0
        
        # 处理不同维度的输入
        if pred.dim() == 5:  # (B, T, C, H, W)
            B, T, C, H, W = pred.shape
            
            # 时间平滑损失
            if self.temporal_weight > 0 and T > 1:
                temporal_diff = pred[:, 1:] - pred[:, :-1]
                temporal_loss = torch.abs(temporal_diff).mean()
                loss += self.temporal_weight * temporal_loss
            
            # 空间平滑损失（对每帧）
            if self.spatial_weight > 0:
                spatial_loss = 0
                for t in range(T):
                    frame = pred[:, t]
                    
                    # 计算空间梯度
                    grad_x = frame[:, :, :, 1:] - frame[:, :, :, :-1]
                    grad_y = frame[:, :, 1:, :] - frame[:, :, :-1, :]
                    
                    # 边缘感知权重
                    if self.edge_aware and target is not None:
                        edge_weight = self.compute_edge_weight(target[:, t])
                        grad_x = grad_x * edge_weight[:, :, :, 1:]
                        grad_y = grad_y * edge_weight[:, :, 1:, :]
                    
                    spatial_loss += torch.abs(grad_x).mean() + torch.abs(grad_y).mean()
                
                loss += self.spatial_weight * spatial_loss / T
        
        else:  # (B, C, H, W)
            # 仅空间平滑损失
            if self.spatial_weight > 0:
                # 计算空间梯度
                grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
                grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
                
                # 边缘感知权重
                if self.edge_aware and target is not None:
                    edge_weight = self.compute_edge_weight(target)
                    grad_x = grad_x * edge_weight[:, :, :, 1:]
                    grad_y = grad_y * edge_weight[:, :, 1:, :]
                
                spatial_loss = torch.abs(grad_x).mean() + torch.abs(grad_y).mean()
                loss += self.spatial_weight * spatial_loss
        
        return loss
