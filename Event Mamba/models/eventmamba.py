"""
EventMamba主模型实现
基于U-Net架构，结合RWOMamba和HSFCMamba模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

from ..modules import RWOMamba, HSFCMamba
from .encoder import EventMambaEncoder
from .decoder import EventMambaDecoder
from .utils import event_to_voxel_grid, normalize_voxel_grid, initialize_weights


@dataclass
class EventMambaConfig:
    """EventMamba配置类"""
    # 输入配置
    num_bins: int = 5  # 体素网格的时间bin数量
    height: int = 256  # 输入高度
    width: int = 256   # 输入宽度
    
    # 架构配置
    base_channel: int = 32  # 基础通道数
    num_stages: int = 4     # U-Net的阶段数
    channel_multiplier: List[int] = None  # 每个阶段的通道倍数
    
    # RWOMamba配置
    window_size: int = 8
    monte_carlo_test: bool = True
    num_monte_carlo_samples: int = 8
    
    # HSFCMamba配置
    bidirectional: bool = True
    merge_mode: str = "concat"  # "concat", "add", "avg"
    
    # Mamba通用配置
    d_state: int = 16
    ssm_ratio: float = 2.0
    mamba_depth: int = 1  # 每个块中的Mamba层数
    
    # 训练配置
    dropout: float = 0.0
    drop_path_rate: float = 0.1
    norm_layer: str = "LayerNorm"  # "LayerNorm" or "RMSNorm"
    
    # 输出配置
    output_channels: int = 1  # 输出图像的通道数（灰度图为1）
    
    def __post_init__(self):
        if self.channel_multiplier is None:
            # 默认通道倍数：[1, 2, 4, 8]
            self.channel_multiplier = [2**i for i in range(self.num_stages)]


class EventMamba(nn.Module):
    """
    EventMamba主模型
    将事件流重建为强度图像
    """
    
    def __init__(self, config: EventMambaConfig):
        super().__init__()
        self.config = config
        
        # 归一化层选择
        norm_layer = nn.LayerNorm if config.norm_layer == "LayerNorm" else nn.RMSNorm
        
        # 初始特征提取
        self.stem = nn.Sequential(
            nn.Conv3d(1, config.base_channel, kernel_size=3, padding=1),
            norm_layer(config.base_channel),
            nn.GELU()
        )
        
        # 计算每个阶段的通道数
        enc_channels = [config.base_channel]
        for i in range(config.num_stages):
            enc_channels.append(config.base_channel * config.channel_multiplier[i])
        
        # 创建编码器
        self.encoder = EventMambaEncoder(
            in_channels=config.base_channel,
            base_channels=config.base_channel,
            num_stages=config.num_stages,
            channel_multiplier=config.channel_multiplier,
            window_size=config.window_size,
            d_state=config.d_state,
            ssm_ratio=config.ssm_ratio,
            mamba_depth=config.mamba_depth,
            bidirectional=config.bidirectional,
            merge_mode=config.merge_mode,
            dropout=config.dropout,
            drop_path_rate=config.drop_path_rate,
            norm_layer=norm_layer
        )
        
        # 瓶颈层
        bottleneck_channels = enc_channels[-1]
        self.bottleneck = nn.Sequential(
            RWOMamba(
                dim=bottleneck_channels,
                depth=config.mamba_depth,
                window_size=config.window_size,
                d_state=config.d_state,
                ssm_ratio=config.ssm_ratio,
                monte_carlo_test=config.monte_carlo_test,
                num_monte_carlo_samples=config.num_monte_carlo_samples,
                drop_path=config.drop_path_rate,
                norm_layer=norm_layer
            ),
            HSFCMamba(
                dim=bottleneck_channels,
                depth=config.mamba_depth,
                d_state=config.d_state,
                ssm_ratio=config.ssm_ratio,
                bidirectional=config.bidirectional,
                merge_mode=config.merge_mode,
                drop_path=config.drop_path_rate,
                norm_layer=norm_layer
            )
        )
        
        # 创建解码器
        self.decoder = EventMambaDecoder(
            enc_channels=enc_channels[1:],  # 跳过第一个（stem输出）
            base_channels=config.base_channel,
            num_stages=config.num_stages,
            channel_multiplier=config.channel_multiplier[::-1],  # 反向
            window_size=config.window_size,
            d_state=config.d_state,
            ssm_ratio=config.ssm_ratio,
            mamba_depth=config.mamba_depth,
            bidirectional=config.bidirectional,
            merge_mode=config.merge_mode,
            monte_carlo_test=config.monte_carlo_test,
            num_monte_carlo_samples=config.num_monte_carlo_samples,
            dropout=config.dropout,
            drop_path_rate=config.drop_path_rate,
            norm_layer=norm_layer
        )
        
        # 输出头
        self.output_head = nn.Sequential(
            nn.Conv2d(config.base_channel, config.base_channel // 2, kernel_size=3, padding=1),
            norm_layer(config.base_channel // 2),
            nn.GELU(),
            nn.Conv2d(config.base_channel // 2, config.output_channels, kernel_size=1),
            nn.Sigmoid()  # 将输出限制在[0, 1]范围内
        )
        
        # 初始化权重
        self.apply(initialize_weights)
        
    def forward(
        self,
        events: Optional[torch.Tensor] = None,
        voxel_grid: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        前向传播
        Args:
            events: 事件张量 (B, N, 4) - [x, y, t, p]
            voxel_grid: 预计算的体素网格 (B, num_bins, H, W)
            return_features: 是否返回中间特征
        Returns:
            output: 重建的图像 (B, C, H, W)
            features (可选): 中间特征字典
        """
        # 事件转换为体素网格
        if voxel_grid is None:
            if events is None:
                raise ValueError("Either events or voxel_grid must be provided")
            voxel_grid = self.events_to_voxel(events)
        
        # 添加通道维度
        if voxel_grid.dim() == 4:  # (B, T, H, W)
            voxel_grid = voxel_grid.unsqueeze(1)  # (B, 1, T, H, W)
        
        # 初始特征提取
        x = self.stem(voxel_grid)  # (B, C, T, H, W)
        
        # 编码器路径
        encoder_features = self.encoder(x)
        
        # 瓶颈处理
        x_bottleneck = encoder_features[-1]
        x_bottleneck = self.bottleneck(x_bottleneck)
        
        # 解码器路径
        x_decoded = self.decoder(x_bottleneck, encoder_features[:-1])
        
        # 时间维度聚合
        # 从 (B, C, T, H, W) 到 (B, C, H, W)
        if x_decoded.dim() == 5:
            # 可以使用平均池化或选择特定时间步
            x_decoded = x_decoded.mean(dim=2)
        
        # 输出头
        output = self.output_head(x_decoded)
        
        if return_features:
            features = {
                'encoder_features': encoder_features,
                'bottleneck_features': x_bottleneck,
                'decoder_features': x_decoded
            }
            return output, features
        
        return output
    
    def events_to_voxel(self, events: torch.Tensor) -> torch.Tensor:
        """
        将事件转换为体素网格
        Args:
            events: (B, N, 4) 事件张量 [x, y, t, p]
        Returns:
            voxel_grid: (B, num_bins, H, W) 体素网格
        """
        B = events.shape[0]
        voxel_grids = []
        
        for b in range(B):
            # 获取单个样本的事件
            event_batch = events[b]
            # 过滤掉填充的事件（假设填充事件的坐标为负）
            mask = event_batch[:, 0] >= 0
            event_batch = event_batch[mask]
            
            # 转换为体素网格
            voxel_grid = event_to_voxel_grid(
                event_batch,
                num_bins=self.config.num_bins,
                height=self.config.height,
                width=self.config.width
            )
            voxel_grids.append(voxel_grid)
        
        # 堆叠所有体素网格
        voxel_grid = torch.stack(voxel_grids, dim=0)
        
        # 归一化
        voxel_grid = normalize_voxel_grid(voxel_grid)
        
        return voxel_grid
    
    def compute_flops(self, input_shape: Tuple[int, ...]) -> int:
        """
        计算模型的FLOPs
        Args:
            input_shape: 输入形状 (B, T, H, W) 或 (B, N, 4)
        Returns:
            flops: 浮点运算次数
        """
        # 简化的FLOPs计算
        # 实际实现应该更精确
        from .utils import compute_model_stats
        return compute_model_stats(self, input_shape)['flops']
    
    @torch.no_grad()
    def monte_carlo_inference(self, events: torch.Tensor, K: int = 8) -> torch.Tensor:
        """
        使用蒙特卡洛采样进行推理
        Args:
            events: 事件张量
            K: 采样次数
        Returns:
            output: 平均后的输出
        """
        # 临时设置蒙特卡洛采样数
        original_K = self.config.num_monte_carlo_samples
        self.config.num_monte_carlo_samples = K
        
        # 设置RWOMamba模块的采样数
        for module in self.modules():
            if isinstance(module, RWOMamba):
                module.num_monte_carlo_samples = K
        
        # 推理
        output = self.forward(events)
        
        # 恢复原始设置
        self.config.num_monte_carlo_samples = original_K
        for module in self.modules():
            if isinstance(module, RWOMamba):
                module.num_monte_carlo_samples = original_K
        
        return output


def create_eventmamba(
    num_bins: int = 5,
    height: int = 256,
    width: int = 256,
    base_channel: int = 32,
    num_stages: int = 4,
    window_size: int = 8,
    pretrained: Optional[str] = None,
    **kwargs
) -> EventMamba:
    """
    创建EventMamba模型的便捷函数
    Args:
        num_bins: 时间bin数量
        height: 输入高度
        width: 输入宽度
        base_channel: 基础通道数
        num_stages: U-Net阶段数
        window_size: 窗口大小
        pretrained: 预训练权重路径
        **kwargs: 其他配置参数
    Returns:
        model: EventMamba模型
    """
    config = EventMambaConfig(
        num_bins=num_bins,
        height=height,
        width=width,
        base_channel=base_channel,
        num_stages=num_stages,
        window_size=window_size,
        **kwargs
    )
    
    model = EventMamba(config)
    
    if pretrained is not None:
        # 加载预训练权重
        state_dict = torch.load(pretrained, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from {pretrained}")
    
    return model
