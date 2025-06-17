import torch
import numpy as np
import sys
import os

# 添加EventMamba路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 检查 GPU 可用性
print("GPU 可用性:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("当前设备:", torch.cuda.current_device())
    print("GPU 名称:", torch.cuda.get_device_name(0))
else:
    print("警告: 未检测到 GPU，将使用 CPU。")

def test_imports():
    """测试关键模块导入"""
    try:
        # 测试mamba导入
        from mamba_ssm import Mamba
        print("✓ Mamba-SSM 导入成功")
        
        # 测试EventMamba模块（根据实际代码结构调整）
        # from models.eventmamba import EventMamba
        # print("✓ EventMamba 模型导入成功")
        
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_mamba_layer():
    """测试单个Mamba层"""
    try:
        from mamba_ssm import Mamba
        
        # 创建一个Mamba层
        batch, length, dim = 2, 64, 128
        mamba = Mamba(
            d_model=dim,
            d_state=16,
            d_conv=4,
            expand=2,
        ).to("cuda")  # 将模型移动到 GPU
        
        # 测试前向传播
        x = torch.randn(batch, length, dim).to("cuda")  # 将输入移动到 GPU
        y = mamba(x)
        
        print(f"✓ Mamba层测试成功")
        print(f"  输入: {x.shape}")
        print(f"  输出: {y.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Mamba层测试失败: {e}")
        return False

def test_event_processing():
    """测试事件数据处理"""
    # 生成模拟事件数据
    num_events = 1000
    events = np.random.rand(num_events, 4)
    events[:, 0] = np.random.randint(0, 256, num_events)  # x
    events[:, 1] = np.random.randint(0, 256, num_events)  # y
    events[:, 2] = np.sort(events[:, 2])  # t
    events[:, 3] = np.random.choice([-1, 1], num_events)  # p
    
    print(f"✓ 生成事件数据: {events.shape}")
    return events

if __name__ == "__main__":
    print("=== EventMamba 快速测试 ===\n")
    
    # 运行测试
    test_imports()
    test_mamba_layer()
    test_event_processing()
    
    print("\n测试完成！")