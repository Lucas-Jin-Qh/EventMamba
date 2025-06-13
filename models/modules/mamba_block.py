"""
Mamba Block基础实现
提供标准的Mamba块和相关组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple
import math
from functools import partial
from mamba_ssm import Mamba


class MambaBlock(nn.Module):
    """
    标准Mamba块实现
    包含SSM层、归一化、MLP等组件
    """
    
    def __init__(
        self,
        dim: int,
        depth: int = 1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        ssm_ratio: float = 2.0,
        conv_bias: bool = True,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        rms_norm: bool = True,
        residual_in_fp32: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        
        # 构建多个Mamba层
        self.layers = nn.ModuleList([
            create_block(
                dim=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dt_rank=dt_rank,
                ssm_ratio=ssm_ratio,
                conv_bias=conv_bias,
                dropout=dropout,
                drop_path=drop_path_rate * i / (depth - 1) if depth > 1 else 0.0,
                norm_layer=norm_layer,
                rms_norm=rms_norm,
                device=device,
                dtype=dtype
            )
            for i in range(depth)
        ])
        
    def forward(self, x: torch.Tensor, inference_params=None) -> torch.Tensor:
        """
        前向传播
        Args:
            x: (B, L, D) 输入序列
            inference_params: 推理时的缓存参数
        Returns:
            output: (B, L, D) 输出序列
        """
        for layer in self.layers:
            x = layer(x, inference_params=inference_params)
        return x


def create_block(
    dim: int,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    dt_rank: str = "auto",
    ssm_ratio: float = 2.0,
    conv_bias: bool = True,
    dropout: float = 0.0,
    drop_path: float = 0.0,
    norm_layer: nn.Module = nn.LayerNorm,
    rms_norm: bool = True,
    residual_in_fp32: bool = True,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> nn.Module:
    """
    创建单个Mamba层
    """
    if rms_norm:
        norm_layer = partial(RMSNorm, eps=1e-5)
    
    # 使用官方Mamba实现
    mixer_cls = partial(
        Mamba,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        dt_rank=dt_rank,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=conv_bias,
        bias=False,
        use_fast_path=True,
    )
    
    block = ResidualBlock(
        dim=dim,
        mixer_cls=mixer_cls,
        norm_cls=norm_layer,
        dropout=dropout,
        drop_path=drop_path,
        residual_in_fp32=residual_in_fp32,
        device=device,
        dtype=dtype
    )
    return block


class ResidualBlock(nn.Module):
    """
    带残差连接的块
    """
    
    def __init__(
        self,
        dim: int,
        mixer_cls=None,
        norm_cls=nn.LayerNorm,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        residual_in_fp32: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        
        self.norm = norm_cls(dim, device=device, dtype=dtype)
        self.mixer = mixer_cls(d_model=dim, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
    def forward(self, x: torch.Tensor, inference_params=None) -> torch.Tensor:
        """
        前向传播
        """
        residual = x
        x = self.norm(x)
        
        if self.residual_in_fp32:
            x = x.to(dtype=torch.float32)
        
        x = self.mixer(x)
        x = self.dropout(x)
        x = self.drop_path(x)
        
        if self.residual_in_fp32:
            residual = residual.to(dtype=torch.float32)
        
        output = residual + x
        return output.to(dtype=residual.dtype)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    """
    
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return self.weight * x


class DropPath(nn.Module):
    """
    Stochastic Depth路径丢弃
    """
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
            
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class BiMambaBlock(nn.Module):
    """
    双向Mamba块
    支持前向和后向扫描
    """
    
    def __init__(
        self,
        dim: int,
        depth: int = 1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        ssm_ratio: float = 2.0,
        conv_bias: bool = True,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        rms_norm: bool = True,
        residual_in_fp32: bool = True,
        merge_mode: str = "add",  # "add", "concat", "avg"
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.merge_mode = merge_mode
        
        # 调整维度
        if merge_mode == "concat":
            block_dim = dim // 2
        else:
            block_dim = dim
            
        # 前向Mamba块
        self.forward_block = MambaBlock(
            dim=block_dim,
            depth=depth,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            ssm_ratio=ssm_ratio,
            conv_bias=conv_bias,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            device=device,
            dtype=dtype
        )
        
        # 后向Mamba块
        self.backward_block = MambaBlock(
            dim=block_dim,
            depth=depth,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            ssm_ratio=ssm_ratio,
            conv_bias=conv_bias,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            device=device,
            dtype=dtype
        )
        
        # 合并投影
        if merge_mode == "concat":
            self.merge_proj = nn.Linear(dim, dim)
            
    def forward(self, x: torch.Tensor, inference_params=None) -> torch.Tensor:
        """
        双向前向传播
        """
        B, L, D = x.shape
        
        # 前向扫描
        x_forward = self.forward_block(x, inference_params=inference_params)
        
        # 后向扫描 - 翻转序列
        x_backward = torch.flip(x, dims=[1])
        x_backward = self.backward_block(x_backward, inference_params=inference_params)
        x_backward = torch.flip(x_backward, dims=[1])
        
        # 合并结果
        if self.merge_mode == "add":
            output = x_forward + x_backward
        elif self.merge_mode == "avg":
            output = (x_forward + x_backward) / 2
        elif self.merge_mode == "concat":
            output = torch.cat([x_forward, x_backward], dim=-1)
            output = self.merge_proj(output)
        else:
            raise ValueError(f"Unknown merge mode: {self.merge_mode}")
            
        return output


class CrossMambaBlock(nn.Module):
    """
    交叉Mamba块
    支持两个输入序列之间的交互
    """
    
    def __init__(
        self,
        dim: int,
        depth: int = 1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        ssm_ratio: float = 2.0,
        conv_bias: bool = True,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        rms_norm: bool = True,
        residual_in_fp32: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        
        # Self-Mamba块
        self.self_block = MambaBlock(
            dim=dim,
            depth=depth,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            ssm_ratio=ssm_ratio,
            conv_bias=conv_bias,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            device=device,
            dtype=dtype
        )
        
        # Cross-Mamba块
        self.cross_block = MambaBlock(
            dim=dim,
            depth=depth,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            ssm_ratio=ssm_ratio,
            conv_bias=conv_bias,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            device=device,
            dtype=dtype
        )
        
        # 交叉注意力的投影
        self.cross_proj = nn.Linear(dim * 2, dim)
        
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        inference_params=None
    ) -> torch.Tensor:
        """
        交叉前向传播
        Args:
            x: (B, L1, D) 主输入序列
            context: (B, L2, D) 上下文序列
            inference_params: 推理参数
        Returns:
            output: (B, L1, D) 输出序列
        """
        # Self处理
        x_self = self.self_block(x, inference_params=inference_params)
        
        # 准备交叉输入
        B, L1, D = x.shape
        _, L2, _ = context.shape
        
        # 将x和context拼接用于交叉处理
        x_expanded = x.unsqueeze(2).expand(B, L1, L2, D)
        context_expanded = context.unsqueeze(1).expand(B, L1, L2, D)
        cross_input = torch.cat([x_expanded, context_expanded], dim=-1)
        cross_input = cross_input.view(B * L1, L2, D * 2)
        
        # 投影到原始维度
        cross_input = self.cross_proj(cross_input)
        
        # 交叉Mamba处理
        cross_output = self.cross_block(cross_input, inference_params=inference_params)
        
        # 聚合交叉信息
        cross_output = cross_output.view(B, L1, L2, D)
        cross_output = cross_output.mean(dim=2)  # 平均池化
        
        # 组合self和cross
        output = x_self + cross_output
        
        return output
