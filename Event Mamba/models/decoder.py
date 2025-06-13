"""
EventMamba解码器实现
包含上采样路径和skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from einops import rearrange

from ..modules import RWOMamba, HSFCMamba
from .utils import create_upsample_block, create_conv_block


class DecoderBlock(nn.Module):
    """
    解码器块
    包含上采样、skip connection融合、RWOMamba和HSFCMamba
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        window_size: int = 8,
        d_state: int = 16,
        ssm_ratio: float = 2.0,
        mamba_depth: int = 1,
        bidirectional: bool = True,
        merge_mode: str = "concat",
        monte_carlo_test: bool = True,
        num_monte_carlo_samples: int = 8,
        upsample: bool = True,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        
        self.upsample = upsample
        
        # 上采样层
        if upsample:
            self.upsample_layer = create_upsample_block(
                in_channels=in_channels,
                out_channels=in_channels,
                upsample_type="transpose",  # 可选: "transpose", "bilinear+conv"
                kernel_size=4,
                stride=2,
                padding=1
            )
        
        # Skip connection融合
        fusion_channels = in_channels + skip_channels if skip_channels > 0 else in_channels
        self.fusion = nn.Sequential(
            nn.Conv3d(fusion_channels, out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.GELU()
        )
        
        # RWOMamba处理空间特征
        self.rwo_mamba = RWOMamba(
            dim=out_channels,
            depth=mamba_depth,
            window_size=window_size,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            monte_carlo_test=monte_carlo_test,
            num_monte_carlo_samples=num_monte_carlo_samples,
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
        
        # 残差连接的投影（如果需要）
        self.residual_proj = nn.Conv3d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        Args:
            x: (B, C, T, H, W) 输入特征
            skip: (B, C_skip, T, H', W') 来自编码器的skip特征
        Returns:
            x: (B, C_out, T, H', W') 输出特征
        """
        # 保存用于残差连接
        residual = x
        
        # 上采样
        if self.upsample:
            x = self.upsample_layer(x)
            residual = F.interpolate(
                residual,
                size=x.shape[2:],
                mode='trilinear',
                align_corners=False
            )
        
        # Skip connection融合
        if skip is not None:
            # 确保尺寸匹配
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(
                    skip,
                    size=x.shape[2:],
                    mode='trilinear',
                    align_corners=False
                )
            x = torch.cat([x, skip], dim=1)
        
        # 融合特征
        x = self.fusion(x)
        
        B, C, T, H, W = x.shape
        
        # 对每个时间步应用RWOMamba
        x_list = []
        for t in range(T):
            x_t = self.rwo_mamba(x[:, :, t, :, :])
            x_list.append(x_t.unsqueeze(2))
        x = torch.cat(x_list, dim=2)
        
        # 应用HSFCMamba处理时空特征
        x = self.hsfc_mamba(x)
        
        # 残差连接
        x = x + self.residual_proj(residual)
        
        return x


class EventMambaDecoder(nn.Module):
    """
    EventMamba解码器
    多阶段上采样路径，包含skip connections
    """
    
    def __init__(
        self,
        enc_channels: List[int],  # 编码器每个阶段的通道数
        base_channels: int = 32,
        num_stages: int = 4,
        channel_multiplier: Optional[List[int]] = None,
        window_size: int = 8,
        d_state: int = 16,
        ssm_ratio: float = 2.0,
        mamba_depth: int = 1,
        bidirectional: bool = True,
        merge_mode: str = "concat",
        monte_carlo_test: bool = True,
        num_monte_carlo_samples: int = 8,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        
        self.num_stages = num_stages
        
        # 默认通道倍数（与编码器相反）
        if channel_multiplier is None:
            channel_multiplier = [2**(num_stages-1-i) for i in range(num_stages)]
        
        # 构建解码器阶段
        self.stages = nn.ModuleList()
        
        # Drop path率线性减少
        dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, num_stages)]
        
        # 反向遍历编码器通道
        enc_channels = enc_channels[::-1]
        
        for i in range(num_stages):
            # 输入通道数
            if i == 0:
                in_ch = enc_channels[0]  # 瓶颈层输出
            else:
                in_ch = base_channels * channel_multiplier[i-1]
            
            # Skip通道数
            skip_ch = enc_channels[i+1] if i+1 < len(enc_channels) else 0
            
            # 输出通道数
            out_ch = base_channels * channel_multiplier[i]
            
            self.stages.append(
                DecoderBlock(
                    in_channels=in_ch,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    window_size=window_size,
                    d_state=d_state,
                    ssm_ratio=ssm_ratio,
                    mamba_depth=mamba_depth,
                    bidirectional=bidirectional,
                    merge_mode=merge_mode,
                    monte_carlo_test=monte_carlo_test,
                    num_monte_carlo_samples=num_monte_carlo_samples,
                    upsample=True,
                    dropout=dropout,
                    drop_path=dpr[i],
                    norm_layer=norm_layer
                )
            )
        
        # 最终投影到base_channels
        self.final_proj = nn.Sequential(
            nn.Conv3d(base_channels * channel_multiplier[-1], base_channels, kernel_size=3, padding=1),
            norm_layer(base_channels),
            nn.GELU()
        )
    
    def forward(self, x: torch.Tensor, encoder_features: List[torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        Args:
            x: (B, C, T, H, W) 瓶颈层输出
            encoder_features: 编码器特征列表（不包含瓶颈层）
        Returns:
            x: (B, C_out, T, H', W') 解码器输出
        """
        # 反向遍历编码器特征
        encoder_features = encoder_features[::-1]
        
        for i, stage in enumerate(self.stages):
            # 获取对应的skip特征
            skip = encoder_features[i] if i < len(encoder_features) else None
            x = stage(x, skip)
        
        # 最终投影
        x = self.final_proj(x)
        
        return x


class AttentionDecoderBlock(nn.Module):
    """
    带注意力机制的解码器块
    在skip connection融合时使用注意力
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        window_size: int = 8,
        d_state: int = 16,
        ssm_ratio: float = 2.0,
        num_heads: int = 8,
        mamba_depth: int = 1,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        
        # 上采样
        self.upsample = create_upsample_block(
            in_channels=in_channels,
            out_channels=in_channels,
            upsample_type="transpose"
        )
        
        # 特征投影
        self.query_proj = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.key_proj = nn.Conv3d(skip_channels, out_channels, kernel_size=1)
        self.value_proj = nn.Conv3d(skip_channels, out_channels, kernel_size=1)
        
        # 多头注意力参数
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 输出投影
        self.out_proj = nn.Conv3d(out_channels, out_channels, kernel_size=1)
        
        # Mamba块
        self.mamba_block = nn.Sequential(
            RWOMamba(
                dim=out_channels,
                depth=mamba_depth,
                window_size=window_size,
                d_state=d_state,
                ssm_ratio=ssm_ratio,
                norm_layer=norm_layer
            ),
            HSFCMamba(
                dim=out_channels,
                depth=mamba_depth,
                d_state=d_state,
                ssm_ratio=ssm_ratio,
                norm_layer=norm_layer
            )
        )
        
        # 归一化
        self.norm1 = norm_layer(out_channels)
        self.norm2 = norm_layer(out_channels)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        使用注意力机制融合skip特征
        """
        # 上采样
        x = self.upsample(x)
        
        B, C, T, H, W = x.shape
        
        # 投影
        q = self.query_proj(x)  # (B, C, T, H, W)
        k = self.key_proj(skip)
        v = self.value_proj(skip)
        
        # 重塑为多头格式
        q = rearrange(q, 'b (h d) t x y -> b h (t x y) d', h=self.num_heads)
        k = rearrange(k, 'b (h d) t x y -> b h (t x y) d', h=self.num_heads)
        v = rearrange(v, 'b (h d) t x y -> b h (t x y) d', h=self.num_heads)
        
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力
        out = attn @ v
        out = rearrange(out, 'b h (t x y) d -> b (h d) t x y', t=T, x=H, y=W)
        
        # 输出投影和残差
        out = self.out_proj(out)
        x = x + out
        x = self.norm1(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        
        # Mamba处理
        identity = x
        x_processed = x
        for t in range(T):
            x_t = self.mamba_block[0](x_processed[:, :, t, :, :])
            x_processed[:, :, t, :, :] = x_t
        x_processed = self.mamba_block[1](x_processed)
        
        x = identity + x_processed
        x = self.norm2(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        
        return x
