"""
Selective State Space Model (SSM) Layer
Mamba的核心SSM层实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple
import math


class SSMLayer(nn.Module):
    """
    选择性状态空间模型层
    实现了Mamba的核心SSM机制
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
        ssm_ratio: float = 2.0,
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
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        
        # 输入投影
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        # 卷积层
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs
        )
        
        # SSM参数投影
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        
        # SSM参数初始化
        # A参数 - 负对数初始化确保稳定性
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            'n -> d n',
            d=self.d_inner
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        
        # D参数 - 跳跃连接
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        
        # dt初始化
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
            
    def forward(self, x: torch.Tensor, inference_params=None) -> torch.Tensor:
        """
        前向传播
        Args:
            x: (B, L, D) 输入序列
            inference_params: 推理时的缓存参数
        Returns:
            output: (B, L, D) 输出序列
        """
        batch, seqlen, _ = x.shape
        
        # 输入投影
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # 各 (B, L, d_inner)
        
        # 卷积
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :seqlen]
        x = rearrange(x, 'b d l -> b l d')
        
        # 激活
        x = F.silu(x)
        
        # SSM计算
        y = self.ssm(x)
        
        # 门控
        y = y * F.silu(z)
        
        # 输出投影
        output = self.out_proj(y)
        
        return output
    
    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        """
        选择性扫描机制
        Args:
            x: (B, L, D) 输入
        Returns:
            y: (B, L, D) 输出
        """
        batch, seqlen, _ = x.shape
        
        # 投影得到dt, B, C
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        dt, BC = torch.split(x_dbl, [self.dt_rank, 2 * self.d_state], dim=-1)
        
        # 计算dt
        dt = self.dt_proj(dt)  # (B, L, d_inner)
        dt = F.softplus(dt)
        
        # 分离B和C
        B, C = BC.chunk(2, dim=-1)  # 各 (B, L, d_state)
        
        # 计算SSM
        y = self.selective_scan(x, dt, B, C)
        
        return y
    
    def selective_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor
    ) -> torch.Tensor:
        """
        选择性扫描算法的实现
        这是一个简化版本，实际实现应该使用CUDA优化
        Args:
            x: (B, L, D) 输入
            dt: (B, L, D) 时间步长
            B: (B, L, N) 输入矩阵
            C: (B, L, N) 输出矩阵
        Returns:
            y: (B, L, D) 输出
        """
        batch, seqlen, d_inner = x.shape
        d_state = B.shape[-1]
        
        # 获取A矩阵
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        
        # 离散化A和B
        deltaA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, d_inner, d_state)
        deltaB = dt.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, d_inner, d_state)
        
        # 初始化状态
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        y = torch.zeros_like(x)
        
        # 扫描（简化版本，实际应该使用并行扫描）
        for t in range(seqlen):
            # 更新状态
            h = deltaA[:, t] * h + deltaB[:, t] * x[:, t].unsqueeze(-1)
            
            # 计算输出
            y[:, t] = torch.einsum('bdn,bn->bd', h, C[:, t])
        
        # 添加跳跃连接
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        
        return y


class OptimizedSSMLayer(nn.Module):
    """
    优化的SSM层实现
    使用更高效的扫描算法
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
    
    def optimized_selective_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor
    ) -> torch.Tensor:
        """
        优化的选择性扫描实现
        使用并行前缀和算法
        """
        # 这里应该实现更高效的并行扫描算法
        # 可以使用associative scan或者其他并行算法
        # 由于实现复杂，这里仅提供接口
        raise NotImplementedError("Optimized scan not implemented in this version")


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


def build_ssm_init(
    d_model: int,
    d_state: int = 16,
    dt_rank: str = "auto",
    dt_scale: float = 1.0,
    dt_init: str = "random",
    dt_min: float = 0.001,
    dt_max: float = 0.1,
    dt_init_floor: float = 1e-4,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> dict:
    """
    构建SSM初始化参数
    """
    factory_kwargs = {"device": device, "dtype": dtype}
    
    dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
    
    # 初始化dt
    if dt_init == "constant":
        dt = torch.ones(d_model, **factory_kwargs) * dt_scale
    elif dt_init == "random":
        dt = torch.exp(
            torch.rand(d_model, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor) * dt_scale
    else:
        raise NotImplementedError(f"dt_init {dt_init} not implemented")
    
    # 初始化A
    A = repeat(
        torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
        'n -> d n',
        d=d_model
    ).contiguous()
    A_log = torch.log(A)
    
    return {
        "dt": dt,
        "A_log": A_log,
        "D": torch.ones(d_model, **factory_kwargs)
    }


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
        
        # 创建多个SSM头
        self.ssm_heads = nn.ModuleList([
            SSMLayer(
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
            head_output = ssm_head(x[:, i], inference_params)
            outputs.append(head_output)
        
        # 合并输出
        output = torch.stack(outputs, dim=1)  # (B, n_heads, L, d_head)
        output = output.transpose(1, 2).contiguous()  # (B, L, n_heads, d_head)
        output = output.view(B, L, D)
        
        return output
