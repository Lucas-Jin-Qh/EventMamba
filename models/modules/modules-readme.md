# EventMamba 核心模块文档

本目录包含EventMamba的核心模块实现，包括Random Window Offset (RWO) Mamba和Hilbert Space-Filling Curve (HSFC) Mamba等关键组件。

## 模块概览

### 1. RWOMamba (`rwo_mamba.py`)
随机窗口偏移Mamba模块，解决了固定窗口分区导致的平移不变性损失问题。

**主要特性：**
- 训练时随机窗口偏移
- 测试时蒙特卡洛采样（默认K=8）
- 保持空间平移不变性
- 支持多层堆叠

**使用示例：**
```python
from modules import RWOMamba

# 创建RWOMamba模块
rwo_mamba = RWOMamba(
    dim=64,                    # 特征维度
    depth=2,                   # Mamba块的数量
    window_size=8,             # 窗口大小
    d_state=16,                # 状态维度
    ssm_ratio=2.0,             # SSM扩展比率
    monte_carlo_test=True,     # 测试时使用蒙特卡洛采样
    num_monte_carlo_samples=8  # 蒙特卡洛采样数
)

# 输入: (B, C, H, W)
x = torch.randn(2, 64, 64, 64)
output = rwo_mamba(x)  # 输出: (B, C, H, W)
```

### 2. HSFCMamba (`hsfc_mamba.py`)
希尔伯特空间填充曲线Mamba模块，保持事件数据的时空局部性。

**主要特性：**
- Hilbert和Trans-Hilbert双向扫描
- 3D到1D序列化保持局部性
- 支持单向和双向处理
- 多种特征融合模式

**使用示例：**
```python
from modules import HSFCMamba

# 创建HSFCMamba模块
hsfc_mamba = HSFCMamba(
    dim=64,                # 特征维度
    depth=2,               # Mamba块的数量
    d_state=16,            # 状态维度
    ssm_ratio=2.0,         # SSM扩展比率
    bidirectional=True,    # 使用双向扫描
    merge_mode="concat"    # 融合模式: "concat", "add", "avg"
)

# 输入: (B, C, T, H, W)
x = torch.randn(2, 64, 5, 32, 32)
output = hsfc_mamba(x)  # 输出: (B, C, T, H, W)
```

### 3. MambaBlock (`mamba_block.py`)
基础Mamba块实现，提供标准和变体Mamba块。

**包含的类：**
- `MambaBlock`: 标准Mamba块
- `BiMambaBlock`: 双向Mamba块
- `CrossMambaBlock`: 交叉Mamba块（支持两个输入序列的交互）
- `ResidualBlock`: 带残差连接的块
- `RMSNorm`: Root Mean Square归一化

**使用示例：**
```python
from modules import MambaBlock, BiMambaBlock

# 标准Mamba块
mamba_block = MambaBlock(
    dim=128,
    depth=4,
    d_state=16,
    ssm_ratio=2.0
)

# 双向Mamba块
bi_mamba = BiMambaBlock(
    dim=128,
    depth=2,
    merge_mode="add"  # 或 "concat", "avg"
)
```

### 4. SSMLayer (`ssm_layer.py`)
选择性状态空间模型（SSM）的核心层实现。

**主要组件：**
- `SSMLayer`: 标准SSM层
- `OptimizedSSMLayer`: 优化的SSM层（接口）
- `ParallelSSM`: 多头并行SSM
- `CausalConv1d`: 因果1D卷积

**核心算法：**
- 选择性扫描机制
- 输入依赖的参数（Δ, B, C）
- 硬件感知的实现

## 关键设计决策

### 1. 内存效率
- 使用分组卷积减少参数量
- 支持梯度检查点（gradient checkpointing）
- 优化的张量操作顺序

### 2. 计算效率
- 线性复杂度O(L)而非Transformer的O(L²)
- 支持并行处理多个窗口
- 缓存Hilbert曲线索引

### 3. 灵活性
- 模块化设计，易于组合
- 支持不同的归一化方式（LayerNorm, RMSNorm）
- 可配置的drop path和dropout

## 注意事项

1. **Mamba官方实现**：当前SSM层是简化实现，生产环境建议使用官方Mamba实现以获得最佳性能。

2. **GPU内存**：RWOMamba的蒙特卡洛采样会增加内存使用，可以通过调整`num_monte_carlo_samples`来平衡精度和内存。

3. **输入尺寸**：
   - RWOMamba要求H和W能被window_size整除
   - HSFCMamba对2的幂次尺寸有更好的性能

4. **混合精度训练**：所有模块都支持混合精度训练（AMP）。

## 扩展和定制

### 自定义窗口策略
```python
class CustomRWOMamba(RWOMamba):
    def generate_random_offset(self, B, device):
        # 实现自定义偏移策略
        # 例如：基于输入内容的自适应偏移
        pass
```

### 自定义扫描曲线
```python
class CustomHSFCMamba(HSFCMamba):
    def generate_custom_curve(self, T, H, W):
        # 实现自定义空间填充曲线
        # 例如：Z字形扫描、螺旋扫描等
        pass
```

## 性能优化建议

1. **批处理大小**：较大的批处理可以更好地利用GPU并行性
2. **窗口大小**：8或16通常是良好的选择，平衡局部性和计算效率
3. **深度**：2-4层通常足够，更深的网络收益递减
4. **状态维度**：16是默认值，可以根据任务复杂度调整

## 故障排除

### 常见问题

1. **维度不匹配**：确保输入维度符合要求，特别是窗口大小的整除性
2. **内存溢出**：减少批处理大小或关闭蒙特卡洛采样
3. **梯度消失**：检查学习率和归一化设置

### 调试工具
```python
# 打印模块信息
print(rwo_mamba)

# 检查参数数量
total_params = sum(p.numel() for p in rwo_mamba.parameters())
print(f"Total parameters: {total_params:,}")

# 可视化梯度流
for name, param in rwo_mamba.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm().item():.4f}")
```
