"""
EventMamba编码器实现
包含下采样路径和特征提取
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from einops import rearrange

from ..modules import RWOMamba, HSFCMamba
from .utils import create_downsample_block, create_conv_block


class EncoderBlock(nn.Module):
    """
    编码器块
    包含RWOMamba和HSFCMamba，以及下采样
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        window_size: int = 8,
        d_state: int = 16,
        ssm_ratio: float = 2.0,
        mamba_depth: int = 1,
        bidirectional: bool = True,
        merge_mode: str = "concat",
        downsample: bool = True,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        
        self.downsample = downsample
        
        # 通道调整
        if in_channels != out_channels:
            self.channel_proj = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.channel_proj = nn.Identity()
        
        # RWOMamba处理空间特征
        self.rwo_mamba = RWOMamba(
            dim=out_channels,
            depth=mamba_depth,
            window_size=window_size,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dropout=dropout,
            drop_path=drop_path,
            norm_layer=norm_layer
        )
        
        # HSFCMamba处理时空特征
        self.hsfc_mamba = HSFCMamba(
            dim=out_channels,
            depth=mamba_depth,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            bidirectional=bidirectional,
            merge_mode=merge_mode,
            dropout=dropout,
            drop_path=drop_path,
            norm_layer=norm_layer
        )
        
        # 下采样层
        if downsample:
            self.downsample_layer = create_downsample_block(
                in_channels=out_channels,
                out_channels=out_channels,
                downsample_type="conv",  # 可选: "conv", "maxpool", "avgpool"
                kernel_size=3,
                stride=2,
                padding=1
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: (B, C, T, H, W) 输入特征
        Returns:
            x: (B, C', T, H', W') 输出特征
        """
        # 通道投影
        x = self.channel_proj(x)
        
        B, C, T, H, W = x.shape
        
        # 对每个时间步应用RWOMamba
        x_list = []
        for t in range(T):
            x_t = self.rwo_mamba(x[:, :, t, :, :])
            x_list.append(x_t.unsqueeze(2))
        x = torch.cat(x_list, dim=2)
        
        # 应用HSFCMamba处理时空特征
        x = self.hsfc_mamba(x)
        
        # 下采样
        if self.downsample:
            x = self.downsample_layer(x)
        
        return x


class EventMambaEncoder(nn.Module):
    """
    EventMamba编码器
    多阶段下采样路径
    """
    
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        num_stages: int = 4,
        channel_multiplier: Optional[List[int]] = None,
        window_size: int = 8,
        d_state: int = 16,
        ssm_ratio: float = 2.0,
        mamba_depth: int = 1,
        bidirectional: bool = True,
        merge_mode: str = "concat",
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        
        self.num_stages = num_stages
        
        # 默认通道倍数
        if channel_multiplier is None:
            channel_multiplier = [2**i for i in range(num_stages)]
        
        # 构建编码器阶段
        self.stages = nn.ModuleList()
        
        # Drop path率线性增加
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_stages)]
        
        # 第一阶段
        self.stages.append(
            EncoderBlock(
                in_channels=in_channels,
                out_channels=base_channels * channel_multiplier[0],
                window_size=window_size,
                d_state=d_state,
                ssm_ratio=ssm_ratio,
                mamba_depth=mamba_depth,
                bidirectional=bidirectional,
                merge_mode=merge_mode,
                downsample=True,
                dropout=dropout,
                drop_path=dpr[0],
                norm_layer=norm_layer
            )
        )
        
        # 后续阶段
        for i in range(1, num_stages):
            in_ch = base_channels * channel_multiplier[i-1]
            out_ch = base_channels * channel_multiplier[i]
            
            self.stages.append(
                EncoderBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    window_size=window_size,
                    d_state=d_state,
                    ssm_ratio=ssm_ratio,
                    mamba_depth=mamba_depth,
                    bidirectional=bidirectional,
                    merge_mode=merge_mode,
                    downsample=True,
                    dropout=dropout,
                    drop_path=dpr[i],
                    norm_layer=norm_layer
                )
            )
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        前向传播
        Args:
            x: (B, C, T, H, W) 输入特征
        Returns:
            features: 每个阶段的特征列表
        """
        features = []
        
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        
        return features
    
    def get_feature_info(self) -> List[dict]:
        """
        获取每个阶段的特征信息
        Returns:
            info: 包含通道数和步幅的字典列表
        """
        info = []
        stride = 1
        
        for i, stage in enumerate(self.stages):
            if hasattr(stage, 'downsample') and stage.downsample:
                stride *= 2
            
            # 获取输出通道数
            if hasattr(stage, 'channel_proj') and isinstance(stage.channel_proj, nn.Conv3d):
                out_channels = stage.channel_proj.out_channels
            else:
                # 从rwo_mamba获取
                out_channels = stage.rwo_mamba.dim
            
            info.append({
                'stage': i,
                'channels': out_channels,
                'stride': stride
            })
        
        return info


class MultiScaleEncoder(nn.Module):
    """
    多尺度编码器
    在不同尺度上处理事件数据
    """
    
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        num_scales: int = 3,
        num_stages_per_scale: int = 2,
        window_size: int = 8,
        d_state: int = 16,
        ssm_ratio: float = 2.0,
        mamba_depth: int = 1,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        
        self.num_scales = num_scales
        
        # 为每个尺度创建编码器
        self.scale_encoders = nn.ModuleList()
        
        for scale in range(num_scales):
            scale_factor = 2 ** scale
            scale_channels = base_channels * scale_factor
            
            encoder = EventMambaEncoder(
                in_channels=in_channels,
                base_channels=scale_channels,
                num_stages=num_stages_per_scale,
                window_size=window_size // scale_factor,  # 调整窗口大小
                d_state=d_state,
                ssm_ratio=ssm_ratio,
                mamba_depth=mamba_depth,
                norm_layer=norm_layer
            )
            
            self.scale_encoders.append(encoder)
        
        # 特征融合
        total_channels = sum(base_channels * (2**scale) * (2**num_stages_per_scale) 
                           for scale in range(num_scales))
        self.fusion = nn.Sequential(
            nn.Conv3d(total_channels, base_channels * 4, kernel_size=1),
            norm_layer(base_channels * 4),
            nn.GELU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        多尺度前向传播
        """
        B, C, T, H, W = x.shape
        scale_features = []
        
        for scale, encoder in enumerate(self.scale_encoders):
            # 下采样输入到相应尺度
            if scale > 0:
                scale_factor = 2 ** scale
                x_scaled = F.interpolate(
                    x,
                    size=(T, H // scale_factor, W // scale_factor),
                    mode='trilinear',
                    align_corners=False
                )
            else:
                x_scaled = x
            
            # 编码
            features = encoder(x_scaled)
            
            # 上采样最后一个特征到原始分辨率
            final_feature = features[-1]
            if scale > 0:
                final_feature = F.interpolate(
                    final_feature,
                    size=(T, H // (2**encoder.num_stages), W // (2**encoder.num_stages)),
                    mode='trilinear',
                    align_corners=False
                )
            
            scale_features.append(final_feature)
        
        # 融合多尺度特征
        fused_features = torch.cat(scale_features, dim=1)
        output = self.fusion(fused_features)
        
        return output
