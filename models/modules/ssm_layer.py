"""
Selective State Space Model (SSM) Layer
使用官方Mamba库的标准实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional
import math

try:
    from mamba_ssm import Mamba
except ImportError:
    raise ImportError("Please install mamba-ssm: pip install mamba-ssm")


class SSMLayer(nn.Module):
    """
    选择性状态空间模型层
    使用官方Mamba实现
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: float = 2.0,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        # 使用官方Mamba实现
        self.mamba = Mamba(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            conv_bias=conv_bias,
            bias=bias,
            **factory_kwargs
        )
        
    def forward(self, x: torch.Tensor, inference_params=None) -> torch.Tensor:
        """
        前向传播
        Args:
            x: (B, L, D) 输入序列
            inference_params: 推理时的缓存参数
        Returns:
            output: (B, L, D) 输出序列
        """
        # 直接使用Mamba的forward方法
        return self.mamba(x)


class OptimizedSSMLayer(nn.Module):
    """
    优化的SSM层实现
    直接使用官方Mamba，已经包含了优化
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: float = 2.0,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        conv_bias: bool = True,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.ssm = SSMLayer(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            conv_bias=conv_bias,
            bias=bias,
            device=device,
            dtype=dtype
        )
        
    def forward(self, x: torch.Tensor, inference_params=None) -> torch.Tensor:
        return self.ssm(x, inference_params)


class CausalConv1d(nn.Module):
    """
    因果1D卷积
    用于SSM中的时序建模
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            device=device,
            dtype=dtype
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: (B, C, L) 输入
        Returns:
            y: (B, C, L) 输出
        """
        # 左侧填充以保持因果性
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class ParallelSSM(nn.Module):
    """
    并行SSM实现
    支持多头SSM
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: float = 2.0,
        dt_rank: str = "auto",
        conv_bias: bool = True,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        # 创建多个SSM头，每个使用官方Mamba
        self.ssm_heads = nn.ModuleList([
            Mamba(
                d_model=self.d_head,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dt_rank=dt_rank,
                conv_bias=conv_bias,
                bias=bias,
                device=device,
                dtype=dtype
            )
            for _ in range(n_heads)
        ])
        
    def forward(self, x: torch.Tensor, inference_params=None) -> torch.Tensor:
        """
        多头并行前向传播
        """
        B, L, D = x.shape
        
        # 分割输入到多个头
        x = x.view(B, L, self.n_heads, self.d_head)
        x = x.transpose(1, 2)  # (B, n_heads, L, d_head)
        
        # 并行处理每个头
        outputs = []
        for i, ssm_head in enumerate(self.ssm_heads):
            head_output = ssm_head(x[:, i])
            outputs.append(head_output)
        
        # 合并输出
        output = torch.stack(outputs, dim=1)  # (B, n_heads, L, d_head)
        output = output.transpose(1, 2).contiguous()  # (B, L, n_heads, d_head)
        output = output.view(B, L, D)
        
        return output
