"""
Random Window Offset Mamba (RWOMamba) Module
用于EventMamba的随机窗口偏移策略实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple
import math


class RWOMamba(nn.Module):
    """
    Random Window Offset Mamba模块
    实现了随机窗口偏移策略以保持平移不变性
    """
    
    def __init__(
        self,
        dim: int,
        dim_inner: Optional[int] = None,
        depth: int = 1,
        window_size: int = 8,
        dt_rank: str = "auto",
        d_state: int = 16,
        ssm_ratio: float = 2.0,
        attn_drop: float = 0.0,
        mlp_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        use_checkpoint: bool = False,
        monte_carlo_test: bool = True,
        num_monte_carlo_samples: int = 8
    ):
        super().__init__()
        
        self.dim = dim
        self.window_size = window_size
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.monte_carlo_test = monte_carlo_test
        self.num_monte_carlo_samples = num_monte_carlo_samples
        
        # 构建Vision Mamba Blocks
        self.blocks = nn.ModuleList([
            VisionMambaBlock(
                dim=dim,
                dim_inner=dim_inner,
                dt_rank=dt_rank,
                d_state=d_state,
                ssm_ratio=ssm_ratio,
                attn_drop=attn_drop,
                mlp_drop=mlp_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        
        # 用于测试时的偏移量缓存
        self.register_buffer('test_offsets', None)
        
    def generate_random_offset(self, B: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成随机窗口偏移量"""
        offset_h = torch.randint(0, self.window_size, (B, 1), device=device)
        offset_w = torch.randint(0, self.window_size, (B, 1), device=device)
        return offset_h, offset_w
    
    def window_partition(self, x: torch.Tensor, window_size: int) -> torch.Tensor:
        """
        将特征图划分为不重叠的窗口
        Args:
            x: (B, H, W, C)
            window_size: 窗口大小
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows
    
    def window_reverse(self, windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
        """
        将窗口合并回特征图
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            window_size: 窗口大小
            H, W: 原始特征图的高度和宽度
        Returns:
            x: (B, H, W, C)
        """
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x
    
    def apply_window_with_offset(self, x: torch.Tensor, offset_h: torch.Tensor, offset_w: torch.Tensor) -> torch.Tensor:
        """应用窗口偏移并处理"""
        B, C, H, W = x.shape
        
        # 转换为 (B, H, W, C) 以便于窗口操作
        x = x.permute(0, 2, 3, 1)
        
        # 对每个batch元素应用不同的偏移
        x_shifted_list = []
        for b in range(B):
            x_b = x[b:b+1]  # (1, H, W, C)
            # 应用循环偏移
            x_shifted = torch.roll(x_b, shifts=(offset_h[b].item(), offset_w[b].item()), dims=(1, 2))
            x_shifted_list.append(x_shifted)
        
        x_shifted = torch.cat(x_shifted_list, dim=0)  # (B, H, W, C)
        
        # 窗口分区
        x_windows = self.window_partition(x_shifted, self.window_size)  # (num_windows*B, window_size, window_size, C)
        
        # 将窗口展平为序列
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # (num_windows*B, window_size^2, C)
        
        # 通过Vision Mamba Blocks处理
        for blk in self.blocks:
            x_windows = blk(x_windows)
        
        # 恢复窗口形状
        x_windows = x_windows.view(-1, self.window_size, self.window_size, C)
        
        # 合并窗口
        x_reversed = self.window_reverse(x_windows, self.window_size, H, W)  # (B, H, W, C)
        
        # 反向偏移恢复原始位置
        x_output_list = []
        for b in range(B):
            x_b = x_reversed[b:b+1]
            x_restored = torch.roll(x_b, shifts=(-offset_h[b].item(), -offset_w[b].item()), dims=(1, 2))
            x_output_list.append(x_restored)
        
        x_output = torch.cat(x_output_list, dim=0)  # (B, H, W, C)
        
        # 转换回 (B, C, H, W)
        x_output = x_output.permute(0, 3, 1, 2)
        
        return x_output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: (B, C, H, W) 输入特征
        Returns:
            x: (B, C, H, W) 输出特征
        """
        B, C, H, W = x.shape
        assert H % self.window_size == 0 and W % self.window_size == 0, \
            f"Feature dimensions ({H}, {W}) must be divisible by window_size {self.window_size}"
        
        if self.training:
            # 训练时：使用随机偏移
            offset_h, offset_w = self.generate_random_offset(B, x.device)
            x = self.apply_window_with_offset(x, offset_h, offset_w)
        else:
            if self.monte_carlo_test:
                # 测试时：蒙特卡洛采样
                x_samples = []
                for _ in range(self.num_monte_carlo_samples):
                    offset_h, offset_w = self.generate_random_offset(B, x.device)
                    x_sample = self.apply_window_with_offset(x.clone(), offset_h, offset_w)
                    x_samples.append(x_sample)
                # 平均所有采样结果
                x = torch.stack(x_samples, dim=0).mean(dim=0)
            else:
                # 不使用蒙特卡洛采样，使用零偏移
                offset_h = torch.zeros(B, 1, dtype=torch.long, device=x.device)
                offset_w = torch.zeros(B, 1, dtype=torch.long, device=x.device)
                x = self.apply_window_with_offset(x, offset_h, offset_w)
        
        return x


class VisionMambaBlock(nn.Module):
    """
    Vision Mamba Block (VMB)
    包含Mamba层、深度卷积和MLP
    """
    
    def __init__(
        self,
        dim: int,
        dim_inner: Optional[int] = None,
        dt_rank: str = "auto",
        d_state: int = 16,
        ssm_ratio: float = 2.0,
        attn_drop: float = 0.0,
        mlp_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        
        self.dim = dim
        dim_inner = dim_inner or int(dim * ssm_ratio)
        
        # 归一化层
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        
        # Mamba层
        self.mamba = MambaLayer(
            d_model=dim,
            d_inner=dim_inner,
            dt_rank=dt_rank,
            d_state=d_state
        )
        
        # 深度卷积
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        
        # MLP
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * 4),
            act_layer=nn.GELU,
            drop=mlp_drop
        )
        
        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: (B, L, C) 输入序列
        Returns:
            x: (B, L, C) 输出序列
        """
        # Mamba分支
        shortcut = x
        x = self.norm1(x)
        
        # 通过Mamba层
        x = self.mamba(x)
        
        # 深度卷积
        B, L, C = x.shape
        x_conv = x.transpose(1, 2)  # (B, C, L)
        x_conv = self.dwconv(x_conv)
        x = x_conv.transpose(1, 2)  # (B, L, C)
        
        x = shortcut + self.drop_path(x)
        
        # MLP分支
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class MambaLayer(nn.Module):
    """
    简化的Mamba层实现
    实际实现中应该使用官方的Mamba实现以获得最佳性能
    """
    
    def __init__(
        self,
        d_model: int,
        d_inner: int,
        dt_rank: str = "auto",
        d_state: int = 16
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state
        
        # 计算dt_rank
        if dt_rank == "auto":
            self.dt_rank = math.ceil(d_model / 16)
        else:
            self.dt_rank = dt_rank
        
        # 输入投影
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        
        # SSM参数
        self.A_log = nn.Parameter(torch.randn(d_inner, d_state))
        self.D = nn.Parameter(torch.randn(d_inner))
        
        # dt, B, C的投影
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=False)
        self.x_proj = nn.Linear(d_inner, self.dt_rank + d_state * 2, bias=False)
        
        # 输出投影
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        简化的前向传播
        注意：这是一个简化实现，实际使用时应替换为官方Mamba实现
        """
        B, L, _ = x.shape
        
        # 输入投影
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # 各 (B, L, d_inner)
        
        # 应用激活函数
        x = F.silu(x)
        
        # SSM计算（简化版本）
        # 实际实现需要使用选择性扫描算法
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        
        # 投影得到dt, B, C
        x_proj = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        dt, BC = torch.split(x_proj, [self.dt_rank, 2*self.d_state], dim=-1)
        dt = self.dt_proj(dt)  # (B, L, d_inner)
        dt = F.softplus(dt)
        
        B_param, C_param = BC.chunk(2, dim=-1)  # 各 (B, L, d_state)
        
        # 简化的SSM步骤（实际需要扫描算法）
        # 这里仅作为占位符
        y = x * F.silu(z)
        
        # 输出投影
        output = self.out_proj(y)
        
        return output


class Mlp(nn.Module):
    """MLP模块"""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output
