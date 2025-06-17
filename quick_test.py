import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import os

# 尝试导入mamba_ssm，如果失败则使用简化版本
try:
    from mamba_ssm import Mamba # type: ignore
    MAMBA_AVAILABLE = True
    print("✓ Mamba-SSM 可用")
except ImportError:
    MAMBA_AVAILABLE = False
    print("✗ Mamba-SSM 不可用，使用简化版本")
    
    # 简化的Mamba实现用于测试
    class Mamba(nn.Module):
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            super().__init__()
            self.d_model = d_model
            self.expand = expand
            d_inner = int(expand * d_model)
            
            self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
            self.conv1d = nn.Conv1d(
                d_inner, d_inner, 
                kernel_size=d_conv, 
                bias=True,
                groups=d_inner,
                padding=d_conv - 1
            )
            self.out_proj = nn.Linear(d_inner, d_model, bias=False)
            
        def forward(self, x):
            """x: (B, L, D)"""
            b, l, d = x.shape
            
            x_and_res = self.in_proj(x)  # (b, l, 2 * d_inner)
            x, res = x_and_res.split(x_and_res.shape[-1] // 2, dim=-1)
            
            x = x.transpose(1, 2)  # (b, d_inner, l)
            x = self.conv1d(x)[:, :, :l]  # ensure same length
            x = x.transpose(1, 2)  # (b, l, d_inner)
            
            x = x * torch.sigmoid(res)
            x = self.out_proj(x)
            
            return x


class EventVoxelGrid:
    """事件到体素网格的转换器"""
    
    @staticmethod
    def events_to_voxel(events: np.ndarray, 
                       num_bins: int, 
                       height: int, 
                       width: int) -> torch.Tensor:
        """
        将事件转换为体素网格
        Args:
            events: (N, 4) array of [x, y, t, p]
            num_bins: 时间维度的bin数量
            height, width: 空间维度
        Returns:
            voxel_grid: (num_bins, height, width) tensor
        """
        voxel_grid = np.zeros((num_bins, height, width), dtype=np.float32)
        
        if len(events) == 0:
            return torch.from_numpy(voxel_grid)
        
        # 归一化时间戳
        t_min, t_max = events[:, 2].min(), events[:, 2].max()
        if t_max > t_min:
            t_norm = (events[:, 2] - t_min) / (t_max - t_min)
            t_idx = np.clip((t_norm * num_bins).astype(int), 0, num_bins - 1)
        else:
            t_idx = np.zeros(len(events), dtype=int)
        
        # 填充体素
        for i in range(len(events)):
            x, y = int(events[i, 0]), int(events[i, 1])
            if 0 <= x < width and 0 <= y < height:
                pol = 2.0 * events[i, 3] - 1.0  # 转换到 [-1, 1]
                voxel_grid[t_idx[i], y, x] += pol
        
        return torch.from_numpy(voxel_grid)


class MinimalEventMamba(nn.Module):
    """最小化的EventMamba实现用于测试"""
    
    def __init__(self, 
                 num_bins: int = 5,
                 hidden_dim: int = 64,
                 num_layers: int = 4,
                 num_frames: int = 10):
        super().__init__()
        
        self.num_bins = num_bins
        self.hidden_dim = hidden_dim
        self.num_frames = num_frames
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(num_bins, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Mamba层
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=hidden_dim,
                d_state=16,
                d_conv=4,
                expand=2
            ) for _ in range(num_layers)
        ])
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_frames, 3, padding=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_bins, H, W) 输入体素网格
        Returns:
            output: (B, num_frames, H, W) 重建的帧
        """
        B, C, H, W = x.shape
        
        # 编码
        x = self.encoder(x)  # (B, hidden_dim, H, W)
        
        # 重塑为序列
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, hidden_dim)
        
        # 通过Mamba层
        for mamba in self.mamba_layers:
            x = x + mamba(x)  # 残差连接
        
        # 重塑回空间维度
        x = x.transpose(1, 2).reshape(B, self.hidden_dim, H, W)
        
        # 解码
        output = self.decoder(x)  # (B, num_frames, H, W)
        
        return output


def generate_synthetic_events(num_events: int = 10000,
                            height: int = 128,
                            width: int = 128,
                            duration: float = 1.0) -> np.ndarray:
    """生成合成的事件数据（模拟移动的边缘）"""
    
    # 时间戳
    t = np.sort(np.random.uniform(0, duration, num_events))
    
    # 创建移动的圆形轨迹
    angle = 2 * np.pi * t / duration
    radius = min(height, width) * 0.3
    cx, cy = width / 2, height / 2
    
    # 基础位置
    x = cx + radius * np.cos(angle) + np.random.normal(0, 5, num_events)
    y = cy + radius * np.sin(angle) + np.random.normal(0, 5, num_events)
    
    # 限制在图像范围内
    x = np.clip(x, 0, width - 1).astype(int)
    y = np.clip(y, 0, height - 1).astype(int)
    
    # 极性（随机）
    p = np.random.choice([0, 1], num_events)
    
    events = np.column_stack([x, y, t, p])
    return events


def visualize_test_results(events: np.ndarray,
                          voxel_grid: torch.Tensor,
                          output: torch.Tensor,
                          save_path: str = "test_results.png"):
    """可视化测试结果"""
    
    fig = plt.figure(figsize=(15, 5))
    
    # 1. 事件的2D直方图
    ax1 = plt.subplot(131)
    plt.hist2d(events[:, 0], events[:, 1], bins=50, cmap='hot')
    plt.colorbar()
    plt.title("Event Distribution")
    plt.xlabel("X")
    plt.ylabel("Y")
    
    # 2. 输入体素网格（显示中间时间bin）
    ax2 = plt.subplot(132)
    mid_bin = voxel_grid.shape[0] // 2
    plt.imshow(voxel_grid[mid_bin].numpy(), cmap='RdBu_r', interpolation='nearest')
    plt.colorbar()
    plt.title(f"Voxel Grid (bin {mid_bin})")
    
    # 3. 输出帧（显示中间帧）
    ax3 = plt.subplot(133)
    if output is not None:
        mid_frame = output.shape[0] // 2
        plt.imshow(output[mid_frame].detach().cpu().numpy(), cmap='gray', interpolation='nearest')
        plt.colorbar()
        plt.title(f"Reconstructed Frame {mid_frame}")
    else:
        plt.text(0.5, 0.5, "No output", ha='center', va='center')
        plt.title("Output")
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"结果已保存到: {save_path}")
    plt.close()


def run_minimal_test():
    """运行最小化测试，全程使用GPU（如果可用）"""
    
    print("\n=== 开始最小化EventMamba测试 ===\n")
    
    # 参数设置
    height, width = 128, 128
    num_bins = 5
    num_frames = 10
    batch_size = 2
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 生成合成事件
    print("1. 生成合成事件数据...")
    events = generate_synthetic_events(
        num_events=5000,
        height=height,
        width=width,
        duration=1.0
    )
    print(f"   生成 {len(events)} 个事件")
    
    # 2. 转换为体素网格
    print("\n2. 转换事件到体素网格...")
    voxel_grid = EventVoxelGrid.events_to_voxel(
        events, num_bins, height, width
    ).to(device)  # 移动到指定设备
    print(f"   体素网格形状: {voxel_grid.shape}")
    
    # 3. 创建批次
    batch_input = voxel_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    print(f"   批次输入形状: {batch_input.shape}")
    
    # 4. 创建模型
    print("\n3. 创建EventMamba模型...")
    model = MinimalEventMamba(
        num_bins=num_bins,
        hidden_dim=32,
        num_layers=2,
        num_frames=num_frames
    ).to(device)  # 移动模型到指定设备
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   模型参数量: {total_params:,}")
    
    # 5. 前向传播测试
    print("\n4. 测试前向传播...")
    model.eval()
    output = None
    with torch.no_grad():
        try:
            output = model(batch_input)
            print(f"   ✓ 前向传播成功!")
            print(f"   输出形状: {output.shape}")
            print(f"   输出统计: min={output.min():.3f}, max={output.max():.3f}, mean={output.mean():.3f}")
        except Exception as e:
            print(f"   ✗ 前向传播失败: {e}")
    
    # 6. 梯度测试
    print("\n5. 测试反向传播...")
    has_grad = False
    try:
        target = torch.randn_like(output).to(device) if output is not None else torch.randn(batch_size, num_frames, height, width).to(device)
        loss_fn = nn.MSELoss()
        output = model(batch_input)
        loss = loss_fn(output, target)
        loss.backward()
        
        has_grad = any(p.grad is not None for p in model.parameters())
        if has_grad:
            print(f"   ✓ 反向传播成功!")
            print(f"   损失值: {loss.item():.4f}")
        else:
            print(f"   ✗ 没有梯度产生")
    except Exception as e:
        print(f"   ✗ 反向传播失败: {e}")
    
    # 7. 内存和速度测试
    print("\n6. 性能测试...")
    if device.type == 'cuda':
        # 预热
        for _ in range(5):
            with torch.no_grad():
                _ = model(batch_input)
        
        # 计时
        import time
        torch.cuda.synchronize()
        start = time.time()
        
        num_iterations = 20
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(batch_input)
                
        torch.cuda.synchronize()
        end = time.time()
        
        avg_time = (end - start) / num_iterations * 1000
        print(f"   GPU推理时间: {avg_time:.2f} ms/batch")
        print(f"   吞吐量: {1000/avg_time * batch_size:.1f} samples/s")
    else:
        print("   CUDA不可用，跳过GPU性能测试")
    
    # 8. 可视化
    print("\n7. 生成可视化结果...")
    if output is not None:
        visualize_test_results(
            events,
            voxel_grid.cpu(),  # 移动到CPU以进行可视化
            output[0].cpu(),   # 移动到CPU以进行可视化
            "minimal_test_results.png"
        )
    
    print("\n=== 测试完成 ===")
    
    # 总结
    print("\n测试总结:")
    print(f"- Mamba-SSM 可用: {'是' if MAMBA_AVAILABLE else '否'}")
    print(f"- 模型创建: 成功")
    print(f"- 前向传播: {'成功' if output is not None else '失败'}")
    print(f"- 反向传播: {'成功' if has_grad else '失败'}")
    print(f"- 可视化: {'已保存到 minimal_test_results.png' if output is not None else '无输出，无法可视化'}")
    
    return {
        'model': model,
        'events': events,
        'voxel_grid': voxel_grid,
        'output': output
    }


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行测试
    results = run_minimal_test()
    
    # 额外：测试不同输入尺寸
    print("\n\n=== 测试不同输入尺寸 ===")
    test_sizes = [(64, 64), (128, 128), (256, 256)]
    
    for h, w in test_sizes:
        print(f"\n测试尺寸: {h}x{w}")
        try:
            # 生成输入
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            test_input = torch.randn(1, 5, h, w).to(device)
            
            # 创建模型
            model = MinimalEventMamba(num_bins=5, hidden_dim=32, num_layers=2).to(device)
            
            # 测试
            with torch.no_grad():
                output = model(test_input)
            
            print(f"  ✓ 成功 - 输出形状: {output.shape}")
            
        except Exception as e:
            print(f"  ✗ 失败 - {e}")