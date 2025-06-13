"""
EventMamba模块使用示例
展示如何使用RWOMamba和HSFCMamba模块
"""

import torch
import torch.nn as nn
from modules import RWOMamba, HSFCMamba


def test_rwo_mamba():
    """测试RWOMamba模块"""
    print("Testing RWOMamba...")
    
    # 参数设置
    batch_size = 2
    channels = 64
    height = 64
    width = 64
    window_size = 8
    
    # 创建模块
    rwo_mamba = RWOMamba(
        dim=channels,
        depth=2,
        window_size=window_size,
        d_state=16,
        ssm_ratio=2.0,
        monte_carlo_test=True,
        num_monte_carlo_samples=8
    )
    
    # 创建输入
    x = torch.randn(batch_size, channels, height, width)
    
    # 训练模式
    rwo_mamba.train()
    output_train = rwo_mamba(x)
    print(f"Training mode - Input shape: {x.shape}, Output shape: {output_train.shape}")
    
    # 评估模式
    rwo_mamba.eval()
    with torch.no_grad():
        output_eval = rwo_mamba(x)
    print(f"Evaluation mode - Input shape: {x.shape}, Output shape: {output_eval.shape}")
    
    print("RWOMamba test passed!\n")


def test_hsfc_mamba():
    """测试HSFCMamba模块"""
    print("Testing HSFCMamba...")
    
    # 参数设置
    batch_size = 2
    channels = 64
    time_steps = 5
    height = 32
    width = 32
    
    # 创建模块
    hsfc_mamba = HSFCMamba(
        dim=channels,
        depth=2,
        d_state=16,
        ssm_ratio=2.0,
        bidirectional=True,
        merge_mode="concat"
    )
    
    # 创建输入
    x = torch.randn(batch_size, channels, time_steps, height, width)
    
    # 前向传播
    output = hsfc_mamba(x)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")
    
    print("HSFCMamba test passed!\n")


def test_combined_usage():
    """测试组合使用RWOMamba和HSFCMamba"""
    print("Testing combined usage of RWOMamba and HSFCMamba...")
    
    class EventMambaBlock(nn.Module):
        """组合RWOMamba和HSFCMamba的示例块"""
        
        def __init__(self, dim, window_size=8):
            super().__init__()
            self.rwo_mamba = RWOMamba(
                dim=dim,
                window_size=window_size,
                depth=1,
                d_state=16,
                ssm_ratio=2.0
            )
            self.hsfc_mamba = HSFCMamba(
                dim=dim,
                depth=1,
                d_state=16,
                ssm_ratio=2.0,
                bidirectional=True,
                merge_mode="add"
            )
            
        def forward(self, x):
            # x: (B, C, T, H, W)
            B, C, T, H, W = x.shape
            
            # 对每个时间步应用RWOMamba
            x_list = []
            for t in range(T):
                x_t = self.rwo_mamba(x[:, :, t, :, :])
                x_list.append(x_t.unsqueeze(2))
            x = torch.cat(x_list, dim=2)
            
            # 应用HSFCMamba处理时空特征
            x = self.hsfc_mamba(x)
            
            return x
    
    # 测试组合块
    batch_size = 2
    channels = 32
    time_steps = 5
    height = 64
    width = 64
    
    block = EventMambaBlock(dim=channels, window_size=8)
    x = torch.randn(batch_size, channels, time_steps, height, width)
    
    output = block(x)
    print(f"Combined block - Input shape: {x.shape}, Output shape: {output.shape}")
    
    print("Combined usage test passed!\n")


def test_memory_efficiency():
    """测试内存效率"""
    print("Testing memory efficiency...")
    
    # 参数设置
    batch_size = 1
    channels = 128
    height = 256
    width = 256
    window_size = 16
    
    # 创建大尺寸输入
    x = torch.randn(batch_size, channels, height, width).cuda()
    
    # 创建模块
    rwo_mamba = RWOMamba(
        dim=channels,
        depth=4,
        window_size=window_size,
        d_state=16,
        ssm_ratio=2.0,
        monte_carlo_test=False  # 关闭蒙特卡洛以减少内存使用
    ).cuda()
    
    # 测试前向传播
    torch.cuda.synchronize()
    start_mem = torch.cuda.memory_allocated()
    
    with torch.no_grad():
        output = rwo_mamba(x)
    
    torch.cuda.synchronize()
    end_mem = torch.cuda.memory_allocated()
    
    print(f"Input size: {x.shape}")
    print(f"Output size: {output.shape}")
    print(f"Memory used: {(end_mem - start_mem) / 1024 / 1024:.2f} MB")
    
    print("Memory efficiency test passed!\n")


def test_gradient_flow():
    """测试梯度流"""
    print("Testing gradient flow...")
    
    # 创建简单模型
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        RWOMamba(dim=64, depth=2, window_size=8),
        nn.Conv2d(64, 3, 3, padding=1)
    )
    
    # 创建输入和目标
    x = torch.randn(2, 3, 64, 64, requires_grad=True)
    target = torch.randn(2, 3, 64, 64)
    
    # 前向传播
    output = model(x)
    loss = nn.MSELoss()(output, target)
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    print(f"Input gradient norm: {x.grad.norm().item():.4f}")
    
    # 检查模型参数梯度
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} gradient norm: {param.grad.norm().item():.4f}")
    
    print("Gradient flow test passed!\n")


if __name__ == "__main__":
    # 运行所有测试
    test_rwo_mamba()
    test_hsfc_mamba()
    test_combined_usage()
    
    # 如果有GPU，测试内存效率
    if torch.cuda.is_available():
        test_memory_efficiency()
    
    test_gradient_flow()
    
    print("All tests completed successfully!")
