"""
Hilbert Space-Filling Curve Mamba (HSFCMamba) Module
用于EventMamba的希尔伯特空间填充曲线策略实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple, List
import numpy as np
import math


class HSFCMamba(nn.Module):
    """
    Hilbert Space-Filling Curve Mamba模块
    使用Hilbert曲线保持时空局部性
    """
    
    def __init__(
        self,
        dim: int,
        dim_inner: Optional[int] = None,
        depth: int = 1,
        dt_rank: str = "auto",
        d_state: int = 16,
        ssm_ratio: float = 2.0,
        attn_drop: float = 0.0,
        mlp_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        use_checkpoint: bool = False,
        bidirectional: bool = True,
        merge_mode: str = "concat"  # "concat" or "add"
    ):
        super().__init__()
        
        self.dim = dim
        self.depth = depth
        self.bidirectional = bidirectional
        self.merge_mode = merge_mode
        self.use_checkpoint = use_checkpoint
        
        # 如果使用双向扫描且concat模式，需要调整维度
        if self.bidirectional and self.merge_mode == "concat":
            mamba_dim = dim // 2
        else:
            mamba_dim = dim
        
        # 构建Vision Mamba Blocks
        self.blocks = nn.ModuleList([
            VisionMambaBlock3D(
                dim=mamba_dim,
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
        
        # 如果使用双向扫描，创建反向的blocks
        if self.bidirectional:
            self.blocks_reverse = nn.ModuleList([
                VisionMambaBlock3D(
                    dim=mamba_dim,
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
        
        # 用于融合双向特征的投影层
        if self.bidirectional and self.merge_mode == "concat":
            self.merge_proj = nn.Linear(dim, dim)
        
        # 缓存Hilbert曲线索引
        self._hilbert_indices_cache = {}
        self._trans_hilbert_indices_cache = {}
    
    def generate_hilbert_curve(self, n: int) -> List[Tuple[int, int]]:
        """
        生成2D Hilbert曲线坐标
        Args:
            n: 曲线阶数（边长为2^n）
        Returns:
            坐标列表
        """
        def rot(n: int, x: int, y: int, rx: int, ry: int) -> Tuple[int, int]:
            if ry == 0:
                if rx == 1:
                    x = n - 1 - x
                    y = n - 1 - y
                x, y = y, x
            return x, y
        
        def d2xy(n: int, d: int) -> Tuple[int, int]:
            x = y = 0
            s = 1
            while s < n:
                rx = 1 & (d // 2)
                ry = 1 & (d ^ rx)
                x, y = rot(s, x, y, rx, ry)
                x += s * rx
                y += s * ry
                d //= 4
                s *= 2
            return x, y
        
        coords = []
        size = n * n
        for i in range(size):
            coords.append(d2xy(n, i))
        return coords
    
    def generate_3d_hilbert_indices(self, T: int, H: int, W: int) -> torch.Tensor:
        """
        生成3D Hilbert曲线索引
        将时间维度和空间维度结合
        """
        key = (T, H, W)
        if key in self._hilbert_indices_cache:
            return self._hilbert_indices_cache[key]
        
        # 确保H和W是2的幂
        n_h = int(np.ceil(np.log2(H)))
        n_w = int(np.ceil(np.log2(W)))
        n = max(n_h, n_w)
        size = 2 ** n
        
        # 生成2D Hilbert曲线
        hilbert_2d = self.generate_hilbert_curve(size)
        
        # 创建3D索引
        indices = []
        for t in range(T):
            for x, y in hilbert_2d:
                if x < H and y < W:
                    # 将3D坐标转换为1D索引
                    idx = t * H * W + x * W + y
                    indices.append(idx)
        
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        self._hilbert_indices_cache[key] = indices_tensor
        return indices_tensor
    
    def generate_trans_hilbert_indices(self, T: int, H: int, W: int) -> torch.Tensor:
        """
        生成Trans-Hilbert曲线索引（反向扫描）
        """
        key = (T, H, W)
        if key in self._trans_hilbert_indices_cache:
            return self._trans_hilbert_indices_cache[key]
        
        # Trans-Hilbert是Hilbert的转置版本
        # 交换x和y坐标
        n_h = int(np.ceil(np.log2(H)))
        n_w = int(np.ceil(np.log2(W)))
        n = max(n_h, n_w)
        size = 2 ** n
        
        hilbert_2d = self.generate_hilbert_curve(size)
        
        # 创建3D索引，但交换x和y
        indices = []
        for t in range(T):
            for y, x in hilbert_2d:  # 注意这里交换了x和y
                if x < H and y < W:
                    idx = t * H * W + x * W + y
                    indices.append(idx)
        
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        self._trans_hilbert_indices_cache[key] = indices_tensor
        return indices_tensor
    
    def hilbert_scan(self, x: torch.Tensor, use_trans: bool = False) -> torch.Tensor:
        """
        使用Hilbert曲线扫描3D特征
        Args:
            x: (B, C, T, H, W) 输入特征
            use_trans: 是否使用Trans-Hilbert曲线
        Returns:
            x_scanned: (B, L, C) 扫描后的序列
        """
        B, C, T, H, W = x.shape
        
        # 获取Hilbert索引
        if use_trans:
            indices = self.generate_trans_hilbert_indices(T, H, W).to(x.device)
        else:
            indices = self.generate_3d_hilbert_indices(T, H, W).to(x.device)
        
        # 展平特征
        x_flat = x.view(B, C, -1)  # (B, C, T*H*W)
        
        # 按Hilbert顺序重排
        x_scanned = x_flat[:, :, indices]  # (B, C, L)
        x_scanned = x_scanned.transpose(1, 2)  # (B, L, C)
        
        return x_scanned
    
    def inverse_hilbert_scan(self, x_scanned: torch.Tensor, shape: Tuple[int, int, int], use_trans: bool = False) -> torch.Tensor:
        """
        Hilbert扫描的逆操作
        Args:
            x_scanned: (B, L, C) 扫描后的序列
            shape: (T, H, W) 原始形状
            use_trans: 是否使用Trans-Hilbert曲线
        Returns:
            x: (B, C, T, H, W) 恢复的3D特征
        """
        B, L, C = x_scanned.shape
        T, H, W = shape
        
        # 获取Hilbert索引
        if use_trans:
            indices = self.generate_trans_hilbert_indices(T, H, W).to(x_scanned.device)
        else:
            indices = self.generate_3d_hilbert_indices(T, H, W).to(x_scanned.device)
        
        # 转置
        x_scanned = x_scanned.transpose(1, 2)  # (B, C, L)
        
        # 创建输出张量
        x_flat = torch.zeros(B, C, T * H * W, device=x_scanned.device, dtype=x_scanned.dtype)
        
        # 反向映射
        valid_len = min(L, len(indices))
        x_flat[:, :, indices[:valid_len]] = x_scanned[:, :, :valid_len]
        
        # 恢复形状
        x = x_flat.view(B, C, T, H, W)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: (B, C, T, H, W) 输入特征
        Returns:
            x: (B, C, T, H, W) 输出特征
        """
        B, C, T, H, W = x.shape
        shape = (T, H, W)
        
        if self.bidirectional:
            # 双向扫描
            # Hilbert扫描
            x_hilbert = self.hilbert_scan(x, use_trans=False)
            for blk in self.blocks:
                x_hilbert = blk(x_hilbert)
            
            # Trans-Hilbert扫描
            x_trans = self.hilbert_scan(x, use_trans=True)
            for blk in self.blocks_reverse:
                x_trans = blk(x_trans)
            
            # 融合双向特征
            if self.merge_mode == "concat":
                # 恢复到3D
                x_hilbert_3d = self.inverse_hilbert_scan(x_hilbert, shape, use_trans=False)
                x_trans_3d = self.inverse_hilbert_scan(x_trans, shape, use_trans=True)
                
                # 拼接
                x_merged = torch.cat([x_hilbert_3d, x_trans_3d], dim=1)  # (B, 2C, T, H, W)
                
                # 投影回原始维度
                x_merged = rearrange(x_merged, 'b c t h w -> b t h w c')
                x_merged = self.merge_proj(x_merged)
                x = rearrange(x_merged, 'b t h w c -> b c t h w')
            else:  # add
                # 恢复到3D并相加
                x_hilbert_3d = self.inverse_hilbert_scan(x_hilbert, shape, use_trans=False)
                x_trans_3d = self.inverse_hilbert_scan(x_trans, shape, use_trans=True)
                x = x_hilbert_3d + x_trans_3d
        else:
            # 单向扫描
            x_scanned = self.hilbert_scan(x, use_trans=False)
            
            # 通过Mamba blocks
            for blk in self.blocks:
                x_scanned = blk(x_scanned)
            
            # 恢复到3D
            x = self.inverse_hilbert_scan(x_scanned, shape, use_trans=False)
        
        return x


class VisionMambaBlock3D(nn.Module):
    """
    用于3D特征的Vision Mamba Block
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
        self.mamba = MambaLayer3D(
            d_model=dim,
            d_inner=dim_inner,
            dt_rank=dt_rank,
            d_state=d_state
        )
        
        # 1D卷积用于序列
        self.conv1d = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        
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
        
        # 1D卷积
        x_conv = x.transpose(1, 2)  # (B, C, L)
        x_conv = self.conv1d(x_conv)
        x = x_conv.transpose(1, 2)  # (B, L, C)
        
        x = shortcut + self.drop_path(x)
        
        # MLP分支
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class MambaLayer3D(nn.Module):
    """
    用于3D特征的Mamba层
    简化实现，实际使用时应替换为官方Mamba实现
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
        """
        B, L, _ = x.shape
        
        # 输入投影
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        # 应用激活函数
        x = F.silu(x)
        
        # SSM计算（简化版本）
        A = -torch.exp(self.A_log)
        
        # 投影得到dt, B, C
        x_proj = self.x_proj(x)
        dt, BC = torch.split(x_proj, [self.dt_rank, 2*self.d_state], dim=-1)
        dt = self.dt_proj(dt)
        dt = F.softplus(dt)
        
        B_param, C_param = BC.chunk(2, dim=-1)
        
        # 简化的SSM步骤
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
